# VSR vs RouteIQ Decision — v3 (Reality-Synced)

> **Supersedes:** `vsr-vs-routeiq-decision-v2.md` (2026-04-27, 3,439 words).
> v2 was written against a 0.2.0-era mental model that predated the 22-commit,
> 25-ADR rearchitecture shipped in v1.0.0rc1 on 2026-04-02. Reconciliation
> audits (two teams, four auditors) found v2 to be ~55% obsolete — the
> load-bearing premises (Evidence-greenfield, M3-dispatcher-as-future,
> control-plane-lock-in, monkey-patch-as-current-state) no longer hold. v3
> replaces, not amends.
>
> **Status:** Decision doc, post-reconciliation.
> **Date:** 2026-04-27.

---

## TL;DR

The live strategic question has changed.

**v2 asked:** Should we stack VSR upstream of LiteLLM, migrate off LiteLLM, or
ship Evidence Console first?

**v3 asks:** Given that v1.0.0rc1 already shipped the full trained-routing
dataplane (18 strategies + centroid + personalized + router-r1 + context
optimizer + LLM-as-judge eval pipeline + native control plane), **is VSR
integration still worth any engineering weeks at all?**

**Verdict:** Mostly no. Demote VSR from "substrate candidate" to "optional
modality-routing enrichment, gated on judge-verdict CPTC uplift ≥15% vs
*our shipped trained arm* — not vs LiteLLM's auto_router."

The remaining work is not a substrate decision. It is filling a narrow
~20% gap around the evaluation pipeline: per-request CPTC attribution,
a tenant-level routing-backend experiment harness (distinct from the
shipped prompt A/B), a `routing_backend` OTel span attribute convention,
a judge-verdict → LoRA nightly-retrain loop, and Experiments/Evidence/Judge
UI pages to complement the shipped Governance/Guardrails/Prompts/Observability
quartet.

VSR is not load-bearing for any of those.

---

## What shipped between v2 and v3

We deliberated v2 against RouteIQ 0.2.0. While we were deliberating,
origin/main advanced to v1.0.0rc1 via commit `ba89b9e` — a 22-commit
squash, 263 files changed, +46,521/-13,857 LOC, 3,222 tests passing, 25
ADRs. The CHANGELOG shows most of the product surface actually landed in
**v0.2.0** (2026-03-28); rc1 primarily added persistence (file-backed state
for governance/policies/guardrails/prompts), CORS fail-closed default, SSRF
fail-closed DNS, Python floor widening (3.14 → 3.12), and a second round of
post-release bugfix commits.

### Architectural inversions

- **Monkey-patch → plugin API** (ADR-0002). `routing_strategy_patch.py`
  deleted. `custom_routing_strategy.py` uses LiteLLM's official
  `CustomRoutingStrategyBase`. Multi-worker safe.
- **Embedded-in-LiteLLM → own FastAPI app** (ADR-0012). RouteIQ owns the
  ASGI composition root; LiteLLM Router is installed as a strategy plugin
  rather than wrapping LiteLLM's app.
- **Custom MCP surfaces deleted** (ADR-0003, ADR-0017). `mcp_jsonrpc.py`,
  `mcp_parity.py`, `mcp_sse_transport.py` removed. JSON-RPC and SSE now
  served by upstream LiteLLM. RouteIQ keeps only the REST MCP gateway.
- **Scattered `os.environ.get()` → Pydantic `BaseSettings`** (ADR-0013).
  124+ env lookups collapsed into `settings.py`.
- **Redis-lease HA → K8s Lease API primary, Redis fallback** (ADR-0015).

### New top-level subsystems

| Module | Purpose | ADR |
|---|---|---|
| `governance.py` (+718 LOC) | Workspaces, API keys, org hierarchy | 0020 |
| `usage_policies.py` (+781 LOC) | Dynamic rate limits + budgets + condition matching | 0022 |
| `guardrail_policies.py` (+1,285 LOC) | Input/output guardrails, 14 check types | 0023 |
| `oidc.py` (+1,475 LOC) | Keycloak / Auth0 / Okta / Azure AD SSO | 0008 |
| `personalized_routing.py` (+782 LOC) | Per-user EMA preference vectors + feedback | 0025 |
| `router_r1.py` (+353 LOC) | NeurIPS 2025 iterative reasoning router, native over LiteLLM | — |
| `centroid_routing.py` (expanded) | Zero-config centroid routing | 0010 |
| `eval_pipeline.py` (+617 LOC) | COLLECT / EVALUATE / AGGREGATE / FEEDBACK with LLM-as-judge | — |
| `service_discovery.py` (+652 LOC) | Startup probe of optional deps with graceful degradation | 0011 |
| `settings.py` (+1,406 LOC) | Pydantic `BaseSettings` centralized config | 0013 |
| `prompt_management.py` (+703 LOC) | Prompt templates + A/B + rollback | — |
| `cli.py` (+181 LOC) | `routeiq start|validate-config|version|probe-services` | — |

### New plugins

14 total (was 13): added `agentic_pipeline.py` and `context_optimizer.py`
(ADR-0024: 6 lossless token-reduction transforms, 30–70% reduction).

### New admin UI pages

- **Governance.tsx** (+781 LOC) — workspaces, API keys, usage policies CRUD
- **Guardrails.tsx** (+407 LOC) — guardrail policy CRUD
- **Prompts.tsx** (+467 LOC) — template CRUD, versioning, A/B, rollback, import/export
- **Observability.tsx** (+288 LOC) — service health, model quality ranking, eval pipeline stats + Run-Batch trigger

Pre-existing: Dashboard, Routing Config.

### New API endpoints

- **Governance**: `/api/v1/routeiq/governance/{workspaces,keys,policies,guardrails}` + `/status` + `/status/summary`
- **Prompts**: `/api/v1/routeiq/prompts/*` CRUD, `{name}/rollback`, `{name}/ab-test` start/stop, `prompts-export`, `prompts-import`
- **Eval**: `/api/v1/routeiq/eval/{stats,samples,run-batch,model-quality,push-feedback}`
- **Routing feedback**: `/api/v1/routeiq/routing/{feedback,preferences/{user_id},personalized/stats,r1}`

### Deployment & ops

- Helm templates: `grafana-dashboard.yaml`, `leader-election-rbac.yaml`,
  `prometheusrule.yaml`, `servicemonitor.yaml`, `pdb.yaml`,
  `networkpolicy.yaml`, `externalsecret.yaml`
- Docker: multi-stage slim build; UI toolchain migrated from npm to Bun
- Docker compose: 10 scenario examples in `examples/docker/` (basic,
  batteries-included, disaggregated-ui, full-stack, ha, local-dev,
  observability, plug-in, slim, testing)
- 25 ADRs (0001–0025) in `docs/adr/`

### Absent from v1.0.0rc1

- `/api/v1/experiments`, `/cptc`, `/judge`, `/evidence` endpoints (evaluation
  is per-model and per-prompt; not per-request trace-joined CPTC)
- Task classifier module
- VSR integration
- Kubernetes Operator / CRDs (Helm only)
- Request-level cost/trace dashboard UI
- Breaking-changes migration guide for v0.2.0 → v1.0.0rc1
- End-to-end integration tests for the new governance / OIDC / plugin-strategy paths

### Release quality signal

Two consecutive post-release commits (`7844419`, `59f80e9`) fixed the same
silent-routing bug: `install_plugin_routing_strategy(app)` called without
the `app` arg, TypeError swallowed, ML routing fell back to LiteLLM default
in own-app mode. The release squash was not fully review-gated before the
tag. Treat as a process finding, not an architectural one.

---

## What v2 got right

1. **Stacking is viable.** Still true. VSR's Envoy ExtProc architecture
   remains technically coherent as an optional backend.
2. **LiteLLM's breadth matters.** Still true. `litellm.model_cost`,
   provider-adapter matrix, streaming normalization, and function-calling
   shims remain load-bearing.
3. **The moat is evidence, not substrate.** Even more true now — v1.0.0rc1
   shipped the first 80% of the evidence pipeline exactly as v2 argued.
4. **CPTC is the right measurement frame.** Still unbuilt; still the right
   gate.
5. **Python plugin coherence kills Topology A.** Still applies. Go-classifier
   upstream of Python guardrails remains a governance smell. VSR-as-upstream
   is still rejected.
6. **Dispatcher-as-A/B-harness discipline** (v2 §5.4). Still the right
   framing once a second backend exists. The numeric M4 rubric (≥15%
   promote / 5–15% judgment / <5% sunset) still applies — only the denominator
   changes (now vs our shipped trained arm, not vs `auto_router`).

---

## What v2 got wrong

### 1. "Evidence Console is greenfield — zero code" (v2 §2.1, v2 §3)

**Wrong.** `eval_pipeline.py:1-23` shipped the COLLECT / EVALUATE /
AGGREGATE / FEEDBACK loop. LLM-as-judge plugin (`gateway/plugins/evaluator.py`)
is wired. The `/api/v1/routeiq/eval/*` endpoints are live. The Observability
admin UI page renders model-quality rankings + eval stats + Run-Batch
trigger. Per `CHANGELOG.md:52`, this landed in v0.2.0 (2026-03-28), **a month
before v2 was written.**

Remaining 20% gap: per-request CPTC (trace-joined cost/quality/latency),
tenant-level routing experiment harness, judge-verdict → LoRA retraining
loop (we have online feedback; offline dataset export is missing),
Experiments/Evidence/Judge UI pages.

### 2. "M3 CustomRoutingStrategyBase dispatcher is future work" (v2 §5.2 killshot, v2 M3 phased plan)

**Wrong.** `custom_routing_strategy.py:1-58` already uses the official
`CustomRoutingStrategyBase` API. `routing_strategy_patch.py` is deleted per
ADR-0002. The "sharpest killshot" v2 named — "Evidence-without-active-RouteIQ-arm
trap" — was **already mitigated** before v2 was authored. v2 cites the file
at `v2:29` while still prescribing its construction as M1 work later in the
same document. That is the clearest internal contradiction in v2.

### 3. "70–110 eng-week control plane rebuild if we migrate" (v2 §2.2)

**Wrong.** The multi-tenant control plane is **RouteIQ-native**, not
LiteLLM-inherited. `governance.py`, `usage_policies.py`,
`guardrail_policies.py`, `oidc.py` import only RouteIQ's own modules. They
are already portable to any substrate. This collapses v2's "Path A preserves
LiteLLM lock-in value" argument into a much weaker "Path A preserves
LiteLLM's provider SDK access."

### 4. LiteLLM `auto_router` commoditization threat is ~50% weaker than v2 claimed

**Wrong sizing.** v2 §2.5 treated LiteLLM's shipped `auto_router` as the
existential threat. Verified reality: `auto_router` is embedding-similarity
based with **no online feedback loop** (`reference/litellm/litellm/router_strategy/auto_router/auto_router.py`).
RouteIQ ships:

- `personalized_routing.py` — per-user EMA preferences with online feedback via `/routing/feedback`
- `router_r1.py` — NeurIPS 2025 iterative reasoning router
- `centroid_routing.py` — zero-config centroid (ADR-0010)
- `context_optimizer.py` — 30–70% token reduction (ADR-0024)
- 18 strategies via `CustomRoutingStrategyBase` plugin

LiteLLM's `auto_router` does none of these. The commoditization threat
narrows to "LiteLLM can commoditize embedding-similarity routing only."
That is not a moat-level threat; it is a feature-parity threat on one
strategy family.

### 5. "CLAUDE.md reflects current architecture" (implicit inherited assumption)

**Wrong.** CLAUDE.md at v2-authoring time still described the monkey-patch,
13 plugins, 5 MCP surfaces, Python 3.14. All four were stale. v2 inherited
the stale framing. CLAUDE.md has been updated in this session to reflect
v1.0.0rc1 reality.

### 6. "Topology A / B / C are distinct architectures" (v2 §3)

**Partially wrong.** Stackability-Investigator's "B" (subprocess-spawn) and
Integration-Architect-v2's "B.2" (same-process `CustomRoutingStrategyBase`
backend) evaluated different architectures and produced opposite verdicts —
v2 acknowledged this collapse. What v2 did not acknowledge is that "Topology
C" **is itself** the same architecture: a tenant/flag selector in front of
the `CustomRoutingStrategyBase` plugin is Topology C by any other name.
Since the plugin already ships, **Topology C's backend-1 arm is already
built**. The only remaining work is the selector.

---

## The live question, precisely

Given v1.0.0rc1's shipped surface, what remains between here and the
"OpenRouter-clone with trust/evidence layer" north star?

**Not:** substrate decisions. Those are done.
**Not:** build Evidence Console. It's 80% built.
**Not:** migrate off LiteLLM. Lock-in reframed; not load-bearing.
**Not:** stack VSR as a default backend. Premature.

**Yes:** close the 20% evidence gap around request-level granularity.
**Yes:** build the routing-backend A/B harness that makes the M4 rubric
operational.
**Yes:** ship the Experiments / Evidence / Judge UI pages that expose what's
in the eval pipeline today.
**Yes:** decide whether VSR's incremental capabilities (BERT+LoRA classifier
batteries, mmBERT-32K modality routing, <10ms Go ExtProc latency) are worth
an optional opt-in backend *after* the evidence pipeline can measure the
delta.

---

## The remaining 20% — eight concrete gaps

| # | Gap | Rough eng-weeks | Notes |
|---|---|---|---|
| 1 | **Per-request CPTC attribution** — OTel span schema versioned in `telemetry_contracts.py`; trace-joined (prompt_tokens × input_price) + (completion_tokens × output_price) + routing_overhead + judge_quality + latency | 2–3 | Dimensions: tenant × team × key × model × provider × region × hour. Real-time via Redis; durable via Postgres + ClickHouse materialized views (Cloud-Native-Architect §7). |
| 2 | **`routing_backend` OTel span attribute convention** | 0.5 | Required for M4 rubric; enforced in `router_decision_callback.py`. Emit one value per decision. Any future backend arm (incl. VSR) conforms. |
| 3 | **Tenant-level routing-backend experiment harness** | 2–3 | Distinct from the shipped prompt A/B at `/prompts/{name}/ab-test`. Feature-flag at tenant level; hash-based traffic split keyed on `tenant_id + request_id`. Assignment stamped as `routing_arm` span attribute. |
| 4 | **Experiments / Evidence / Judge UI pages** | 3–4 | Today: Governance, Guardrails, Prompts, Observability. Add: Experiments (list, create, assign cohort), Evidence (CPTC time-series per backend), Judge (verdict review + correction). React + Vite + React Query consistent with the shipped quartet. |
| 5 | **Judge-verdict → LoRA nightly retrain loop** | 3–4 | Eval pipeline currently feeds personalized routing online via EMA. Missing: offline dataset export from `eval_pipeline.py` samples → Parquet to S3 → K8s GPU Job (HuggingFace trainer / PEFT) → LoRA safetensors → hot-reload into strategy. Signed artifacts via `model_artifacts.py`. |
| 6 | **Task classifier for cold-start requests** | 2 | Narrow scope: pre-personalization feature extraction for users with no history. Modality gating for image/audio/embedding routing. Not a standalone product — a feature of `personalized_routing.py`'s `_cold_start_centroid_fallback` path. |
| 7 | **Dispatcher-sprawl guardrail codification** | 0.5 | Enforce: pure function of tenant_id + flags + headers; no body parsing; single `routing_backend` span attribute; no multi-step orchestration. Linter rule or `arch-unit` style test. Blocks any future drift toward dispatcher-as-router. |
| 8 | **CLAUDE.md / release hygiene** | 0.5 | CLAUDE.md reality-synced this session. Missing: breaking-changes migration guide for v0.2.0 → v1.0.0rc1 (MCP surfaces deleted silently, plugin-strategy signature changed). `MIGRATION.md` before v1.0.0 GA. |

**Total: 13.5 – 17.0 eng-weeks.** Not a replatform. A finishing kick.

---

## VSR — the actual decision

v2 reserved an M3 backend slot for VSR and forced the decision at M4 against
`auto_router`. v3 changes both.

### What VSR uniquely offers, today

- BERT+LoRA Jailbreak / PII / Prompt-Guard classifiers with HuggingFace
  training scripts
- mmBERT-32K modality router (DIFFUSION / AR / BOTH)
- Go ExtProc with <10ms classification latency
- Milvus-backed vector semantic cache

### What RouteIQ already has that makes VSR redundant

- `gateway/plugins/pii_guard.py`, `prompt_injection_guard.py`,
  `content_filter.py`, `llamaguard_plugin.py`, `bedrock_guardrails.py`
  (Python classifiers / regex / external guardrail integrations)
- `guardrail_policies.py` — 14 check-type policy engine (ADR-0023)
- `centroid_routing.py` (ADR-0010), `personalized_routing.py` (ADR-0025),
  `router_r1.py`, 18 ML strategies — classifier-adjacent routing coverage
- `semantic_cache.py` — Python semantic cache (pre-existing)

### The new gate for VSR

**Previous gate (v2):** ship at M3 if SSE benchmark passes.
**New gate (v3):** ship as an optional `routeiq-smart-vsr` backend **only**
if and when the evidence pipeline (gap #3 above) measures a judge-verdict
CPTC uplift of ≥15% on real traffic vs our shipped trained arm — *not* vs
LiteLLM's auto_router.

**If uplift ≥15%:** integrate as opt-in backend, with Python plugin
coherence constraint enforced (guardrails fire at LiteLLM-front; VSR
classifies behind it; v2 §5.1 killshot stands).

**If uplift 5–15%:** product judgment call (enterprise K8s tier only?).

**If uplift <5%:** formally close the VSR path. Pick up BERT+LoRA training
scripts from the VSR repo and port selected classifiers into RouteIQ's
plugin system — the ML weights are Apache-licensed; the Envoy runtime is not
required.

Rationale: The sunk-cost trap to avoid is reserving architectural space
for a substrate whose unique capability shrinks as RouteIQ ships its own
classifier-adjacent stack. Measure the delta before committing integration
effort.

---

## Phased plan M1–M6, reality-synced

**M1 (weeks 1–4) — Evidence gap closure, part 1**
- Gap #1: per-request CPTC schema + emitter + Postgres schema
- Gap #2: `routing_backend` span attribute convention
- Gap #7: dispatcher-sprawl guardrail (codified before new backends land)

**M2 (weeks 5–8) — Evidence gap closure, part 2 + UI**
- Gap #3: tenant-level experiment harness
- Gap #4: Experiments / Evidence / Judge UI pages (first shipped)

**M3 (weeks 9–12) — MLOps loop**
- Gap #5: judge-verdict → offline dataset → K8s GPU training → LoRA artifact → hot-reload
- Gap #6: task classifier (cold-start + modality gating only)

**M4 (weeks 13–16) — Measure and decide**
- Collect ≥6 weeks of tenant-experiment data
- Run M4 rubric on VSR integration (if wanted as opt-in backend)
- Decision: integrate / port-subset / close
- Gap #8: breaking-changes migration guide; v1.0.0 GA candidate

**M5 (weeks 17–20) — HA/DR hardening**
- K8s Lease leader election proven under failure injection
- Aurora Multi-AZ / ElastiCache cluster mode validated
- Cross-region S3 CRR for config and model artifacts
- DR runbook + quarterly drill cadence

**M6 (weeks 21–24) — Federation + v1.0.0 GA**
- Route53 latency + geolocation pinning
- Regional ClickHouse with global rollup
- Multi-region identity via OIDC JWKS cache
- v1.0.0 GA tag

**Total: ~24 eng-weeks to v1.0.0 GA** from v1.0.0rc1, on a 2-person team.

---

## Killshots that still apply

### §5.1 — Python plugin coherence (kills VSR-as-upstream)

Still active. A Go classifier upstream of Python guardrails would route
PII-tagged content across a process boundary before `pii_guard.py`,
`llamaguard_plugin.py`, `bedrock_guardrails.py` fire. Unchanged from v2.

### §5.4 — The dispatcher is an A/B harness, not load balancing

Still active. Any routing selector must emit `routing_backend` as a single
span attribute and remain a pure function of tenant + flags + headers.
Codified as gap #7 in M1.

### M4 promotion rubric (still valid, denominator changed)

Still active. Thresholds: ≥15% CPTC uplift → promote / 5–15% → judgment /
<5% → sunset. The denominator moves from `auto_router` (v2) to "RouteIQ's
shipped trained arm" (v3) because auto_router is no longer the realistic
baseline — RouteIQ's own trained stack is.

## Killshots that are now moot

### §5.2 — Evidence-without-active-RouteIQ-arm trap (moot)

`custom_routing_strategy.py` is the active arm. Shipped. The trap cannot
fire.

### §5.3 — Retrain-cadence as substrate-forcing function (partially moot)

v2 argued LiteLLM in-process hot-reload (seconds) vs VSR pod-rotation
(5–15 min) would force the substrate choice. With the v3 plan, retrain
cadence is set by the nightly MLOps loop (gap #5), not by infra mechanics.
The substrate becomes a consequence of measured uplift, not of reload
timing.

---

## What the user was right about (preserved from v2)

1. **Stacking is viable** — still true, remains an option at M4 gate.
2. **LiteLLM's breadth matters** — still true, unchanged.
3. **VSR's routing intelligence is real** — still true; the BERT+LoRA
   classifiers are genuine ML assets. What v3 reframes is the *cost of
   buying them* (integrating Go ExtProc) vs. *building the Python analog*
   (porting weights into our existing plugin system). Gap #6 and the
   redirected §5 "port-subset" option cover this cheaper.
4. **OpenRouter-clone north star** — still the product target; v3 closes the
   gap to it without buying VSR's substrate.

The reordering that v3 makes: **measure before integrating.** Don't
reserve architectural slots for substrates whose unique value is shrinking.

---

## Appendix A — Where agents disagreed

| Claim | Investigators (Team 1) | Reconciliation Lead | Reconciliation Adversary | v3 |
|---|---|---|---|---|
| Evidence Console is "zero code" (v2 claim) | Disagree — eval pipeline shipped | Disagree | Disagree (55% obsolete claim) | Disagree — eval is 80% |
| Monkey-patch is current state (v2 assumption) | Disagree — deleted | Disagree | Disagree | Disagree |
| 13 plugins (v2 inherited claim) | Disagree — 15 | Disagree — 15 | Disagree — 15 | **14** (ground-truth: `ls gateway/plugins/`) |
| Control-plane lock-in to LiteLLM | Silent | Disagree — RouteIQ-native | Disagree (strongly) | Disagree |
| LiteLLM auto_router is heuristic | Disagree — embedding-similarity | Disagree | Disagree (threat overstated) | Disagree |
| Topology C backend-1 is future work | Disagree — shipped | Disagree | Disagree | Disagree |
| VSR worth M3 backend slot | Silent | "Demote to M6" | "No" | "Measure first" |
| v2 should be replaced by v3 | Implicit | Explicit | Explicit | Confirmed |

Note on plugin count: Team 1 said 15. Direct filesystem verification
(`ls src/litellm_llmrouter/gateway/plugins/`) shows 14 `.py` files
excluding `__init__.py` and `__pycache__`. Team 1's likely error: counting
a neighbor or a variant. v3 anchors to ground truth.

## Appendix B — Reconciliation evidence trail

- **Team 1 outputs** (three unbiased auditors, v1.0.0rc1 ground truth):
  - `v1-code-auditor` — code delta: `routing_strategy_patch.py` deleted, 13 new modules, custom MCP surfaces removed, new UI pages wired
  - `v1-feature-auditor` — feature delta: 6 admin UI pages, `/eval/*` endpoints live, governance + guardrails + OIDC shipped, prompt A/B live; absent: `/experiments`, `/cptc`, `/evidence`, `/judge`, task_classifier, VSR
  - `v1-history-narrator` — commit history: 22-commit squash, 25 ADRs, rearchitecture verified not marketing; post-release quality signal (two consecutive fixes for the same silent-routing bug)

- **Team 2 outputs** (reconciliation lead + adversary):
  - `reconciliation-lead` — structured stale/shipped/remaining/new categorization; verdict: write v3 from scratch
  - `reconciliation-adversary` — v2 ~55% obsolete; load-bearing premises all wrong; write v3 from scratch

- **Direct filesystem verification** (this document):
  - `ls src/litellm_llmrouter/gateway/plugins/` → 14 plugins
  - `pyproject.toml: requires-python = ">=3.12"` → 3.12 floor
  - 13 new top-level modules present
  - 4 deleted files confirmed absent (`routing_strategy_patch`, `mcp_jsonrpc`, `mcp_sse_transport`, `mcp_parity`)

- **ADR anchors cited:**
  - ADR-0002 plugin-routing-strategy
  - ADR-0003 delete-redundant-mcp
  - ADR-0010 centroid-zero-config-routing
  - ADR-0011 pluggable-external-services
  - ADR-0012 own-fastapi-app
  - ADR-0013 pydantic-settings
  - ADR-0015 k8s-native-leader-election
  - ADR-0017 leverage-litellm-upstream
  - ADR-0020 governance-layer
  - ADR-0022 governance-middleware-integration
  - ADR-0023 guardrail-policy-pipeline
  - ADR-0024 context-optimizer
  - ADR-0025 personalized-routing

## Appendix C — How v3 was produced

1. Committed v1 + v2 decision docs (scoped, not `-am`).
2. Discovered origin/main advanced 10 commits during deliberation.
3. Stashed pre-existing mods (CLAUDE.md, docs/index.md, .gitignore,
   technical-roadmap, submodules).
4. Fast-forward pulled origin/main.
5. Observed `ba89b9e` release commit: 263 files, 25 ADRs.
6. Dispatched **Team 1** (three unbiased code/feature/history auditors)
   explicitly blocked from reading v2 — produced ground-truth snapshot.
7. Dispatched **Team 2** (reconciliation lead + adversary) — compared v2
   against Team 1's snapshot; both concluded v3-from-scratch.
8. Updated CLAUDE.md's stale Architecture Overview, Key Patterns,
   Non-Obvious Behaviors sections to match v1.0.0rc1 reality.
9. Wrote v3 against verified ground truth with direct filesystem checks
   for ambiguous claims (plugin count, deleted files, Python floor).
10. Hold push until v3 lands and user reviews.

**Durable artifact:** the `strategic-replatform-deliberation` skill
(`~/.claude/skills/strategic-replatform-deliberation/SKILL.md`) now has
an implicit refinement — *always verify against current origin/main before
dispatching deliberation agents*. Will codify in a skill update if this
happens again.
