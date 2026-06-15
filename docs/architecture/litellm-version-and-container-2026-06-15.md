# LiteLLM in RouteIQ — Version Setup, Bump Delta, and Container Optimization

**Date:** 2026-06-15
**Status:** Architecture reference (read-only research synthesis; no code changed).
**Sources:** `research/litellm-version-container/discover-docker.md` (Docker/image baseline),
`research/litellm-version-container/discover-litellm-abi.md` (ABI surface map),
`research/notes/final_report_litellm-bump-docker-slim-12cb7b.md` (version delta + container research),
`research/litellm-version-container/synth-digest.json` (key findings).
**Verified against repo:** `pyproject.toml:21`, `docker/Dockerfile:26,48,50,144`,
`reference/litellm/pyproject.toml:3`, seed `RouteIQ-af75`.

---

## TL;DR

1. **RouteIQ does NOT run the upstream `berriai/litellm` prebuilt image.** It builds its
   own multi-stage image from the Astral `ghcr.io/astral-sh/uv` Python base and installs
   `litellm==1.82.3` as a *pip dependency* (via the RouteIQ `[prod]` extra — deliberately
   **not** `litellm[proxy]`). LiteLLM runs **in-process** under RouteIQ's own FastAPI app,
   mounted at `/v1` (ADR-0012), and the only Router-internal touch point is RouteIQ's
   `CustomRoutingStrategyBase` plugin (ADR-0002).
2. **Version drift is real but cosmetic at the install layer.** `pyproject.toml` pins
   `litellm==1.82.3` (what actually runs); the `Dockerfile` ARG/label and the
   `reference/litellm` submodule both say `1.81.3` (stale). Tracked as seed `RouteIQ-af75`.
3. **Bump verdict: 1.82.3 → 1.89.0 (or latest 1.88.x). ABI risk LOW, security upside HIGH.**
   Eight 2026 GHSA advisories (3 Critical, 5 High) all postdate 1.82.3.
4. **Top container lever: torch-CPU enforcement** — removes ~2 GB of CUDA wheels from amd64
   full images. The current `UV_EXTRA_INDEX_URL` env var does *not* guarantee it.

---

# PART 1 — How LiteLLM Is Set Up in RouteIQ

This part is definitive, drawn from the two read-only discovery dives. The headline:
**RouteIQ is not a LiteLLM distribution — it is its own gateway that embeds the LiteLLM
library.** Three facts make this concrete.

## 1.1 RouteIQ builds its own image — it does not pull `berriai/litellm`

Every stage of `docker/Dockerfile` (302 lines, BuildKit `# syntax=docker/dockerfile:1.4`)
starts `FROM ghcr.io/astral-sh/uv:...` — the Astral `uv` Python image — **not**
`FROM berriai/litellm` (`discover-docker.md` §a; `Dockerfile:74,108,125`):

| Stage | Base image | Job |
|-------|-----------|-----|
| `builder` | `ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm` (sha256-pinned `BUILDER_DIGEST`) | `git clone` LLMRouter at a pinned commit, `uv build --wheel` → `/wheels` |
| `ui-builder` | `oven/bun:1-alpine` | `bun install --frozen-lockfile` + `bun run build` of the React admin UI → `/ui/dist` |
| `runtime` | `ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm-slim` (sha256-pinned `RUNTIME_DIGEST`) | Final image: install wheels + project, copy app source + UI dist, non-root user, Prisma, healthcheck, tini |

LiteLLM enters as a **pip dependency**: `pyproject.toml:21` pins `litellm==1.82.3` in the
core `dependencies` list, and the runtime install (`Dockerfile:185-199`) runs
`uv pip install --system -e ".[${ROUTEIQ_EXTRAS}]"` (default extra `prod`). There is no
`FROM berriai/litellm` anywhere in the build.

### Why RouteIQ builds-not-pulls

The upstream `berriai/litellm` image is *just the LiteLLM proxy* — a packaged
`litellm.proxy.proxy_server` app with its own entrypoint. RouteIQ cannot use it as a base
because RouteIQ needs to ship four things the upstream image does not contain and cannot
host:

1. **Its own FastAPI app** (ADR-0012). RouteIQ creates `create_gateway_app()` and *mounts*
   LiteLLM's proxy app as a sub-app at `/v1` — the inverse of "be the LiteLLM proxy." The
   upstream image hands you the LiteLLM app as the root; RouteIQ needs the LiteLLM app to be
   a guest.
2. **The `CustomRoutingStrategyBase` routing plugin** (ADR-0002), installed onto LiteLLM's
   Router at startup via `Router.set_custom_routing_strategy()`. This is RouteIQ source that
   has to be present in the image and wired in-process.
3. **The React admin UI** (built by the `ui-builder` Bun stage into `/ui/dist`) — RouteIQ's
   own control-plane surface, not LiteLLM's.
4. **The LLMRouter wheel**, built from source. LLMRouter is **not on PyPI / not in `uv.lock`**;
   the builder stage clones it at `ARG LLMROUTER_COMMIT=7890cd9d36951a3c73fec83619321f3704a7aaa8`
   (`Dockerfile:50`), `uv build --wheel`s it, and installs the wheel only when
   `INSTALL_LLMROUTER=true`. There is no upstream artifact to pull for this; it must be built.

In short: the upstream image is a *product* (the LiteLLM proxy); RouteIQ needs LiteLLM as a
*library* inside a larger product. Building its own image is the only way to compose
own-app + plugin + UI + LLMRouter wheel around the embedded library.

## 1.2 How LiteLLM runs: in-process, mounted at `/v1`, one plugin touch point

RouteIQ runs LiteLLM **in-process** (never `os.execvp` to a separate proxy binary) so the
RouteIQ routing-strategy plugin persists in the same interpreter as the Router it patches
(`discover-litellm-abi.md` §2):

- **Init** (`gateway/app.py:562-571`): `import litellm.proxy.proxy_server as _proxy_server`
  then `await _proxy_server.initialize(config=..., telemetry=False)` loads the config and
  constructs the `llm_router`.
- **Mount** (`app.py:820-822`): `from litellm.proxy.proxy_server import app as litellm_app`
  then `app.mount("/v1", litellm_app)` — LiteLLM's proxy becomes a sub-app at `/v1`. This is
  the ADR-0012 inversion: RouteIQ's app is the root, LiteLLM is the guest.
- **Strategy install** (`app.py:582-586` → `startup.install_plugin_routing_strategy`): after
  init, RouteIQ reads the live `llm_router` and calls
  `router.set_custom_routing_strategy(RouteIQRoutingStrategy(...))`, which rebinds the
  Router's `get_available_deployment` / `async_get_available_deployment` to RouteIQ's bound
  methods (`discover-litellm-abi.md` §1c).

The single most load-bearing coupling is `litellm.types.router.CustomRoutingStrategyBase`
(ADR-0002): it is the *only* place RouteIQ touches Router internals, and its failure mode is
**silent** — the import is guarded by `except ImportError` only (a local stub fallback), so a
signature change upstream would let `RouteIQRoutingStrategy` construct against the stub while
ML routing quietly degrades to LiteLLM default (`discover-litellm-abi.md` §1, the
"silent-failure trap"). Beyond that, RouteIQ reaches into a handful of LiteLLM internals:
`litellm._logging.verbose_proxy_logger` (14 module-top-level imports — the widest coupling by
file count), `litellm.proxy.proxy_server.{app, llm_router, initialize}`, and
`litellm.proxy.auth.user_api_key_auth.user_api_key_auth` (the data-plane auth dependency).
RouteIQ's own `/llmrouter/mcp/*` REST surface has **zero** `from litellm` imports
(`discover-litellm-abi.md` §4); the only LiteLLM MCP coupling is the opaque native `/mcp`
served under the `/v1` sub-mount.

### Why no `litellm[proxy]` extra

RouteIQ installs `litellm` *without* the `[proxy]` extra and **hand-vendors** the proxy's
transitive deps (`pyproject.toml:19-33`): `apscheduler`, `email-validator`, `fastapi-sso`,
`websockets`, `backoff`, `redis`. The historical justification (a `pyproject.toml` comment)
is "strict pinned dependencies in proxy extra conflict with project requirements." As shown
in Part 2, **that premise is now stale** — the modern `[proxy]` extra uses `>=` ranges, not
`==` pins — but the architectural consequence stands today: because RouteIQ bypasses the
extra, the dependency resolver will **not** flag a newly-required proxy import. A new minor
that `import`s a module LiteLLM still does not declare as core surfaces only as a
`ModuleNotFoundError` at `initialize()`, never at build/resolve time.

## 1.3 Version drift (seed RouteIQ-af75)

There are three places a LiteLLM version appears, and they disagree:

| Location | Value | Drives the install? |
|----------|-------|---------------------|
| `pyproject.toml:21` `litellm==1.82.3` | **1.82.3** | **Yes** — this is what actually installs and runs (confirmed via `uv pip show litellm`). |
| `docker/Dockerfile:26,48` `ARG LITELLM_VERSION=1.81.3` + `:144` `LABEL litellm.version="${LITELLM_VERSION}"` | **1.81.3** (stale) | No — the ARG only feeds the OCI label and is otherwise a no-op in the builder stage. |
| `reference/litellm/pyproject.toml:3` `version = "1.81.3"` | **1.81.3** (stale) | No — read-only git submodule, lags the runtime pin by one minor. |

**Net effect:** the image is *labeled* `1.81.3` but *ships* `1.82.3`, and the read-only
submodule used to verify ABI signatures is **not an exact mirror** of what runs (signatures
were verified at 1.81.3; they are stable across 1.81→1.82, but this is a verification hazard).
Both are cosmetic/metadata issues, not runtime bugs, and both are tracked as seed
**`RouteIQ-af75`** ("LiteLLM version drift: Dockerfile ARG/label + reference submodule say
1.81.3 but pyproject pins 1.82.3"). Fix the label and re-sync the submodule whenever the
Dockerfile is next touched (e.g., alongside the bump in Part 2).

---

# PART 2 — Version Delta 1.82.3 → Latest, and Bump Recommendation

Source: `final_report_litellm-bump-docker-slim-12cb7b.md` §§1-7; `synth-digest.json`.

## 2.1 The delta

**Latest stable as of 2026-06-14 is `v1.89.0`** (PyPI current, GitHub "Latest"). RouteIQ's
`1.82.3` baseline dates to ~March 2026, so the delta is **7 minor series (1.83 → 1.89) over
~3 months**. Cadence is ~1-2 weeks/minor, each `X.Y.0` auto-spawning a `stable/X.Y.x` backport
branch; the `1.84.x`-`1.88.x` lines were all still receiving backports on 2026-06-14.

| Version | Date | Headline |
|---------|------|----------|
| v1.83.10/.14 | Apr 27 | Claude Opus 4.7, GPT-5.5, Prompt Compression & Memory API, multi-window budgets |
| v1.84.0 | May 14 | Reliability hardening + multi-pod budget accuracy (cross-pod spend fix) |
| v1.85.0 | May 16 | **Realtime API GA**, MCP Gateway expansion, hardened multi-tenancy |
| v1.86.0 | May 16 | **Weighted-Routing Failover**, native Anthropic web-search citations, OTel-standard server spans |
| v1.87.0 | May 23 | OCI Generative AI provider, Gemini 3.5 Flash day-0, MCP UI for OAuth servers |
| v1.88.0 | Jun 4 | Claude Opus 4.8, MCP access-group authorization, typed OpenTelemetry |
| v1.89.0 | Jun 14 | Claude Fable 5 (1M ctx), A2A agent providers (watsonx Orchestrate, LangFlow), MCP per-server controls |

New providers across the arc: OCI Generative AI, Claude Fable 5 (1M ctx on
Anthropic/Bedrock/Azure/Vertex), Gemini 3.5 Flash day-0, `fal_ai` image gen, APISerpent/You.com
(search), Soniox (transcription), Mistral `ministral-8b`. Bedrock/AgentCore: Fable 5 on
Bedrock, Mantle Responses SigV4 + CrowdStrike AIDR backported to 1.87-1.88.

## 2.2 Breaking changes to RouteIQ's coupled surfaces — NONE

Checked against RouteIQ's own 9-point bump checklist (`discover-litellm-abi.md`), verified via
Context7 (`/berriai/litellm` @ `v1.83.3-stable`) and DeepWiki (current head):

- **`CustomRoutingStrategyBase` — UNCHANGED.** Still at `litellm.types.router`, same name, same
  5-arg async/sync signatures (`model, messages, input, specific_deployment, request_kwargs`).
  RouteIQ's overrides match exactly. (Checklist 1-3 hold.)
- **`Router.set_custom_routing_strategy()` — UNCHANGED.** Still rebinds both
  `get_available_deployment` / `async_get_available_deployment` to the strategy's bound methods.
- **Cooldown model — UNCHANGED.** `allowed_fails` (default 3) + `cooldown_time` (default 5s)
  with deployment-level override identical. Only additive change: **Weighted-Routing Failover**
  (v1.86.0), an opt-in failover that does not touch the cooldown contract.
- **Proxy globals — intact.** `proxy_server.app`, `proxy_server.llm_router`,
  `_logging.verbose_proxy_logger` (14 imports), `proxy.auth.user_api_key_auth.user_api_key_auth`,
  and the shipped `proxy/schema.prisma` all persist at the current head. (Checklist 6-8 hold.)
- **Deployment-dict shape — unchanged** (`model_name` + `litellm_params.model`), the one
  un-guarded assumption RouteIQ carries; spot-check after bump.

**One item to verify hands-on, not breaking:** `proxy_server.initialize(config=, debug=,
telemetry=)`. The signature is not contradicted by any release note, but DeepWiki flags that
init logic is migrating toward a `ProxyConfig` class — a refactor to watch. Because a broken
`initialize()` is *caught and logged* (not raised) in RouteIQ's own-app mode, it degrades
silently — so it must be smoke-tested explicitly.

## 2.3 Dependency / `[proxy]` extra delta (the real residual risk)

**The premise behind RouteIQ avoiding `litellm[proxy]` is now stale.** At v1.89.0 the `[proxy]`
extra uses `>=` *ranges*, not the `==` pins RouteIQ's comment assumes ("so SDK consumers can
coexist"; reproducibility via `uv.lock`). All four hand-vendored deps are compatible:

| Dep | RouteIQ pin | 1.89.0 `[proxy]` range | Compatible? |
|-----|-------------|------------------------|-------------|
| apscheduler | `>=3.10.0` | `>=3.11.2,<4.0` | Yes (proxy floor higher) |
| fastapi-sso | `>=0.16.0` | `>=0.19.0,<1.0` | Yes (proxy floor higher) |
| websockets | `>=15.0.0` | `>=15.0.1,<16.0` | Yes |
| backoff | `>=2.0.0` | `>=2.2.1,<3.0` | Yes |

**Two caveats the resolver will NOT catch:**

1. The `[proxy]` extra now declares deps RouteIQ does **not** vendor — notably
   **`mcp>=1.26.0,<2.0`** and **`cryptography>=46.0.7,<47.0`** (plus `orjson`, `rq`,
   `python-multipart`, `uvicorn`, `gunicorn`, `boto3`). A newly-required proxy import surfaces
   only as `ModuleNotFoundError` at `initialize()`. **Re-diffing the proxy's startup imports
   against the vendored list is the single highest-value pre-bump action.**
2. Core `[project]` deps are also ranges (`pydantic>=2.10.0,<3.0`, `openai>=2.20.0,<3.0`,
   `httpx>=0.28.0,<1.0`), so the bump may pull newer transitive majors — verify via `uv.lock`.

**Practical implication:** either (a) keep hand-vendoring and add `mcp` (+ any other newly
imported proxy module), or (b) switch to `litellm[proxy]` outright now that pins are ranges and
delete the hand-maintained list. Option (b) is the cleaner long-term move but is a *behavior
change* (brings in `gunicorn`, `rq`, etc.) — do it as a separate, separately-tested step, not
folded into the version bump.

## 2.4 Security — the decisive argument to leave 1.82.3

**Eight GHSA advisories published in 2026 — 3 Critical, 5 High — all postdate the 1.82.3
(March) baseline, landing Apr-May 2026 and fixed in the 1.83-1.89 line.**

| GHSA | Date | Severity | Description |
|------|------|----------|-------------|
| GHSA-jjhc-v7c2-5hh6 | Apr 3 | **Critical** | Auth bypass via OIDC userinfo cache-key collision |
| GHSA-r75f-5x8p-qvmc | Apr 20 | **Critical** | SQL injection in proxy API-key verification |
| GHSA-4xpc-pv4p-pm3w | May 28 | **Critical** | Auth bypass via host-header injection |
| GHSA-53mr-6c8q-9789 | Apr 3 | High | Privilege escalation via unrestricted proxy config endpoint |
| GHSA-69x8-hrgq-fjj8 | Apr 4 | High | Password-hash exposure enabling pass-the-hash |
| GHSA-xqmj-j6mv-4862 | Apr 20 | High | Server-side template injection in `/prompts/test` |
| GHSA-wxxx-gvqv-xp7p | May 7 | High | Sandbox escape in custom-code guardrail |
| GHSA-v4p8-mg3p-g94g | Apr 21 | High | Authenticated command execution via MCP stdio test endpoints |

Three are **directly material** to RouteIQ (it runs the proxy in-process with OIDC and
management endpoints enabled): the OIDC userinfo cache-key collision (RouteIQ uses OIDC/SSO),
the proxy API-key SQL injection, and the unrestricted proxy-config privilege escalation. All
touch paths RouteIQ exposes. 1.82.3 predates this entire cluster of fixes.

## 2.5 Bump verdict

**BUMP `litellm==1.82.3` → `litellm==1.89.0`** (or the latest `1.88.x` patch for a more settled
line with the same security backports and one fewer minor of feature churn).

- **ABI risk: LOW.** Every coupled surface is unchanged across the 7-minor delta. No release in
  1.83-1.89 declares a breaking change.
- **Security upside: HIGH.** Closes 3 Critical + 5 High advisories, ≥3 directly material.

**Gate the merge on (do not skip):**

1. Re-sync the `reference/litellm` submodule to the bump target (currently stale at 1.81.3 — see
   seed RouteIQ-af75).
2. **Re-diff the proxy's startup imports against the hand-vendored dep list** — add `mcp`/
   `cryptography` if newly imported, or switch to `litellm[proxy]` as a separate step. The
   resolver will not catch this gap.
3. In-process smoke test confirming a real routing decision flows through
   `RouteIQRoutingStrategy` (not the swallowed-TypeError silent fallback to LiteLLM default).
4. Confirm `initialize()` succeeds and the deployment-dict shape (`model_name` +
   `litellm_params.model`) is unchanged.

Then run the unit + integration suite and ship.

---

# PART 3 — Container Optimization Plan

Source: `discover-docker.md` (baseline, **VERIFIED** against the repo) and
`final_report_litellm-bump-docker-slim-12cb7b.md` §6 (2026 best-practice research,
**SPECULATIVE** size deltas until a real `linux/amd64` build + `dive`/`docker history`
measurement confirms them).

## 3.1 What makes the image heavy

The weight is the **`knn` extra** (`pyproject.toml:84`): `sentence-transformers>=5.2.0` +
`scikit-learn>=1.3.0`. The default extra `prod` =
`routeiq[db,otel,cloud,callbacks,knn,a2a,hotreload,oidc]` (`pyproject.toml:86`) force-includes
`knn`, so the default Docker build pulls the entire ML/CUDA stack. The `uv.lock` chain:
`sentence-transformers → torch==2.10.0 (PyPI registry) → 15 nvidia-*-cu12 wheels + triton`
(gated `x86_64 and linux`). On amd64 that is ~2 GB before any CPU-only optimization
(`nvidia-cudnn-cu12` ~707 MB, `nvidia-cublas-cu12` ~594 MB, the torch wheel itself ~916 MB
uncompressed). Documented sizes: ADR-0009 says slim ~500 MB / full ~2 GB / GPU ~4 GB;
`docs/operations/docker.md` says `latest` ~1.2 GB / `slim` ~500 MB — the gap is exactly the
CPU-vs-CUDA torch question below.

### Optimizations ALREADY present (VERIFIED, `discover-docker.md` §c)

Multi-stage build; uv BuildKit cache mounts (`--mount=type=cache,target=/root/.cache/uv`);
`build-essential` purge in the same RUN; non-root `litellm` user (UID/GID 1000);
sha256-pinned base images; cache-friendly layer ordering (deps before source);
`.dockerignore` (excludes `.git`, `docs/`, `tests/`, `reference/`, `models/*.pt|*.pkl`);
conditional `libatomic1`/Prisma; `tini` + `PYTHONDONTWRITEBYTECODE`; and parameterized
tiering via `ROUTEIQ_EXTRAS`/`BUILD_UI`/`INSTALL_LLMROUTER` + a dedicated `Dockerfile.slim`
(ADR-0007, ADR-0009). `UV_EXTRA_INDEX_URL=.../whl/cpu` is set on the install RUNs — but see #1.

## 3.2 Ranked optimization moves

Ordered by expected size leverage. Effort is S/M/L. Each marked **VERIFIED** (the baseline
state is confirmed in the repo by the discovery dive) vs **SPECULATIVE** (the *size delta* is a
research hypothesis to validate with a real amd64 build + `dive`/`docker history`).

### High leverage (hundreds of MB to GB)

**1. Enforce torch-CPU deterministically via `[tool.uv]` index, not `UV_EXTRA_INDEX_URL`.**
*Effort: M. Expected delta: −~2 GB on amd64 full images (8-10×). VERIFIED problem / SPECULATIVE delta.*
The current `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu` does **not** guarantee
CPU-only torch: (a) `uv pip install -e` is the pip-compat layer and does **not** consult
`uv.lock`; (b) `uv.lock` pins `torch==2.10.0` from the **PyPI** registry with the full
15-wheel CUDA dep set, and there is **no** `[tool.uv.sources]`/`[[tool.uv.index]]` torch pin in
`pyproject.toml`. `UV_EXTRA_INDEX_URL` only *adds* an index; PyPI still satisfies the pin. Fix:
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```
then re-lock so `uv.lock` carries `+cpu` torch (no nvidia deps). Validate the resolver drops
all 15 `nvidia-*-cu12` wheels + `triton`. This is the single biggest lever; it also makes the
documented ~1.2 GB "CPU-only" claim actually true.

**2. Switch the Python install to a true lockfile path (`uv sync --frozen` / `uv pip sync uv.lock`).**
*Effort: M. Delta: enables #1's determinism (no standalone size win). VERIFIED.*
Today the install uses `uv pip install --system -e .[prod]`, which resolves fresh against
indexes and sidesteps `uv.lock` entirely — so the "lockfile-driven reproducibility" the header
docs tout does not apply to the actual Python install (only the UI's `bun --frozen-lockfile` is
truly frozen). Closing this is the prerequisite that makes the torch-CPU pin from #1
deterministic.

**3. Split `knn` out of `prod` into its own image tier.**
*Effort: M. Delta: default image −~1 GB+ (torch/sklearn/transformers gone). VERIFIED design / SPECULATIVE delta.*
`prod` currently force-includes `knn`, but **centroid routing needs no torch** (ADR-0010) and
most deployments use it. Introduce a `prod-ml` extra; keep `prod` torch-free; build two targets
(lean gateway + `gateway-ml`). Aligns with ADR-0009's slim/full/GPU intent but pushes the
*default* lighter. Pairs naturally with the multi-arch buildx work in #9.

### Medium leverage (tens to low-hundreds of MB)

**4. Distroless / Wolfi (Chainguard) runtime base.**
*Effort: L. Delta: −OS userland + most OS-level CVEs. SPECULATIVE.*
Replace the Debian `bookworm-slim` runtime with `cgr.dev/chainguard/python` or a Wolfi base.
**Choose Wolfi (glibc) over Alpine (musl)** — musl forces source builds of the manylinux wheels
torch/scipy ship. Trade-offs to validate: tini vs built-in init; the `curl` healthcheck
(`Dockerfile:266-267`) must move to a Python-based check (distroless has no curl); Prisma/Node
engine glibc compatibility (Wolfi OK, pure distroless may not be). The builder/runtime Python
minor + libc must match exactly — the copied `.venv` is not relocatable across them.

**5. Move uvloop/native compilation fully into the builder stage.**
*Effort: M. Delta: prerequisite for a clean #4 (removes runtime compilers). VERIFIED.*
Today `build-essential` is apt-installed then purged in the *runtime* stage. Compile wheels in
the builder and copy only installed site-packages so the runtime base stays compiler-free.

**6. Strip torch/sklearn/transformers test data, `*.pyi`, bundled test dirs, `*.so` debug symbols.**
*Effort: S. Delta: tens of MB. SPECULATIVE.*
`torch`/`transformers` ship large `include/`, test, and example trees. Extend the existing
`*.pyc` prune to a `find`-based strip; validate nothing imported at runtime is removed.

**7. Drop Prisma/Node engine from slim and non-`db` builds.**
*Effort: S. Delta: Prisma query-engine binary (tens of MB). VERIFIED gate / SPECULATIVE delta.*
The main Dockerfile already gates `prisma generate` on `prod|db`. Confirm `Dockerfile.slim`
ships zero Prisma artifacts, and that the `db` tier downloads only the one platform engine
variant needed (`prisma generate` can fetch multiple).

### Lower leverage / hygiene

**8. `.dockerignore` tightening.**
*Effort: S. Delta: build-context only. VERIFIED.*
Already strong. Add `ui/node_modules`, `.mulch/`, `.seeds/`, `research/`, `plans/`,
`*.coverage`, `ha-gate-report.json` (currently untracked at repo root and would enter the build
context), and any `.overstory/`/`.canopy/` os-eco dirs.

**9. Multi-arch build (`docker buildx --platform linux/amd64,linux/arm64`).**
*Effort: M. Delta: none (correctness/portability). SPECULATIVE.*
Prefer native arm64 runners over QEMU; confirm CPU torch wheels exist for both arches (they do
on the cpu index). Pairs with #1 and #3. Note: macOS arm64 torch is only ~79 MB with no nvidia
deps — which is why the amd64 CUDA bloat is easy to miss when resolving on a Mac.

**10. Fix the `LITELLM_VERSION` label drift + re-sync the submodule.**
*Effort: S. Delta: none (correctness/supply-chain metadata). VERIFIED — seed RouteIQ-af75.*
`Dockerfile:48` ARG `1.81.3` (label only) vs `pyproject.toml:21` `1.82.3` (actual). Update the
ARG to match the installed version and re-sync `reference/litellm`. Do this while touching the
Dockerfile for the bump in Part 2.

**11. Justify / pin Python 3.14.**
*Effort: S. Delta: avoids silent source-build bloat. VERIFIED divergence.*
`Dockerfile:46` `ARG PYTHON_VERSION=3.14` (hardcoded in Prisma `chown` paths) vs
`pyproject.toml requires-python = ">=3.12"`. Confirm 3.14 has cp-wheels for every heavy dep; a
missing 3.14 wheel forces a source build (slower, pulls compilers into runtime).

**12. GPU tier is documented but not implemented.**
*Effort: L (if pursued). Delta: +~4 GB (its own target). SPECULATIVE.*
ADR-0009 lists a ~4 GB GPU tier but no CUDA-base Dockerfile exists. If ever wanted, make it an
explicit separate build target — the flip side of #1's CPU-by-default default.

## 3.3 Measurement gate

Treat **image size** and **CVE count** as CI gates. Measure with `dive` (per-layer waste +
efficiency score), `docker history --no-trunc`, `skopeo inspect` / `crane manifest` (remote
size without pulling), and `syft` (SBOM) + `grype`/`trivy` (CVE count). Every SPECULATIVE delta
above must be confirmed with a real `linux/amd64` build before it is treated as fact — the
torch-CPU question (#1) in particular determines whether the documented ~1.2 GB full image is
real or aspirational.

---

## Appendix — Key file references

- `pyproject.toml:21` — `litellm==1.82.3` (actual install); `:84` `knn` extra; `:86` `prod`;
  `:19-33` hand-vendored proxy deps. **No `[tool.uv]` index/sources section.**
- `docker/Dockerfile` — `:26,48` `LITELLM_VERSION=1.81.3` (label only); `:50` `LLMROUTER_COMMIT`;
  `:57` `ROUTEIQ_EXTRAS="prod"`; `:74,108,125` stage bases; `:144` OCI label; `:185-199` install
  + torch-cpu env + purge; `:266-267` curl healthcheck.
- `docker/Dockerfile.slim` — standalone single-stage slim (no torch/Prisma/UI).
- `reference/litellm/pyproject.toml:3` — `version = "1.81.3"` (stale submodule).
- `gateway/app.py:562-571` init, `:582-586` strategy install, `:820-822` `/v1` mount.
- `custom_routing_strategy.py:32-57` — `CustomRoutingStrategyBase` import + silent-stub fallback.
- ADR-0002 (plugin routing), ADR-0007 (dependency tiering), ADR-0009 (multi-tier images),
  ADR-0010 (centroid routing, torch-free), ADR-0012 (own FastAPI app).
- Seed `RouteIQ-af75` — version drift tracking.
