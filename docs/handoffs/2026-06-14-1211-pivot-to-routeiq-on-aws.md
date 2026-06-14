# Handoff — Pivot to RouteIQ-on-AWS (drop vllm-sr, build on LiteLLM)

> **Created:** 2026-06-14 12:11 PDT
> **From:** the `vllm-sr-on-aws` work stream (now wound down; infra torn down, repo kept as port reference)
> **To:** active development in **this repo (`RouteIQ`)** — `/Users/baladita/Documents/DevBox/RouteIQ`
> **Repo HEAD at handoff:** `f09c937` — *docs: AWS-native re-architecture — deployment-pattern ADRs + Kumaraswamy-Thompson router + pluggable MLOps adapters* (the ADRs 0026–0030 + `aws-rearchitecture/` doc set landed here). This handoff doc itself was committed one step later as `c48b39f`.

This is the **canonical doc to open every time you start RouteIQ-on-AWS work.** It is self-contained: it carries the pivot rationale, the repo map with exact file:line seams, the landed AWS design, the preserved data, the P0–P4 work queue, the dev loop, the hard rules, and the first concrete actions. You should not need to chase pointers to start.

---

## 1. ▶ STARTER PROMPT (paste this into a fresh RouteIQ session)

```
We are working in the RouteIQ repo (/Users/baladita/Documents/DevBox/RouteIQ) and
DROPPING the vllm-sr-on-aws effort. RouteIQ is ALREADY built on LiteLLM + LLMRouter —
it IS our intelligently-routing AI gateway; we do NOT build a new one. The goal: make
RouteIQ a production, HA, cloud-native intelligently-routing AI gateway ON AWS, using
LiteLLM's provider breadth + format translation to plug in extra features (Responses-API
bridge, MLOps, governance, multi-account capacity).

The vllm-sr AWS infra was torn down (greenfield). Its deployment patterns are ALREADY
ported here as ADRs 0026-0030 + docs/architecture/aws-rearchitecture/. The vllm-sr repo
and its CDK stay intact as the PORT REFERENCE (../vllm-sr-on-aws/cdk/). The MLOps corpus
was preserved (RDS snapshot + pg_dump/CSV; 241 rows / 237 SFT tuples / 31 models).

READ FIRST, in this order:
  1. docs/handoffs/2026-06-14-1211-pivot-to-routeiq-on-aws.md  (this handoff — everything)
  2. docs/architecture/aws-rearchitecture/30-migration-roadmap.md  (the P0-P4 work queue)
  3. skim docs/adr/0026-...md through 0030-...md  (one substrate decision each)

FIRST PHASE = P0 (CDK foundation): create deploy/cdk/ in THIS repo with a RouteIqStack,
porting the vllm-sr substrate constructs cited in ADRs 0026-0030 (VPC + EKS Auto Mode +
IRSA-via-CfnJson + ECR/GHCR-pull-through + cdk-nag), parameterized for RouteIQ's single
stateless pod + the chart's IRSA ServiceAccount annotation. PROPOSE the P0 plan before
writing any CDK.

STANDING CONSTRAINTS (non-negotiable):
- AWS profile `baladita+Bedrock-Admin`, account 386931836011, region us-west-2 — ALWAYS
  pass --region us-west-2 explicitly (the shell default is us-east-1).
- COMMIT ONLY, NEVER push. Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Deploy/cutover is OPERATOR-GATED — you produce committed, flag-gated IaC + runbooks +
  CDK tests; the operator runs cdk bootstrap / cdk deploy / kubectl / helm. Do NOT run
  aws/kubectl/cdk yourself.
- NEVER log bearer/token values. NEVER delete the `-v2` S3 buckets.
- Fable 5 is GOV-BANNED as a routable arm — never add it to live routing config.
- RouteIQ naming rules (CLAUDE.md): always call it "RouteIQ"; upstream is "LiteLLM"/
  "LLMRouter". Do NOT rename LITELLM_* env vars. Do NOT claim RouteIQ implements features
  that are inherited from upstream LiteLLM (e.g. the chat/messages<->Responses translation).
  reference/litellm/ is a READ-ONLY submodule — never modify it.
```

---

## 2. The pivot thesis

`vllm-sr-on-aws` (VSR) was the effort to take **vllm-project/semantic-router** (a Go routing app) and run it on an AWS-native substrate. We proved the *substrate* is excellent and the *routing intelligence* is real. But **RouteIQ already is** the "LiteLLM-in-front-of-an-ML-router" synthesis we were converging toward: LiteLLM for provider breadth + format translation, UIUC-LLMRouter + ~9 native strategies for ML routing, plus a governance / OIDC / React-UI control plane.

**The cleanest path to the goal — an intelligently-routing, HA, cloud-native AI gateway — is to keep RouteIQ's layer and give it the AWS substrate VSR proved out, rather than keep wrestling a Go app onto EKS.** This is NOT abandoning work; it is **promoting the better layer.** The three projects are different layers of one stack:

- **LiteLLM** = translation + provider breadth (the base).
- **VSR / vllm-sr** = routing-substrate intelligence + the AWS deployment lessons (the substrate).
- **RouteIQ** = the synthesis (the gateway).

**Why LiteLLM is the base specifically:** LiteLLM is the mechanism that lets us "plug in extra features." It normalizes ~100+ provider adapters to the OpenAI schema and translates `chat`/`messages` ↔ `Responses` — which is exactly what reaches a model like Bedrock's gpt-5.5 (Responses-API-only) that VSR's Envoy-AI-Gateway passthrough architecturally 404s. The Responses bridge, MLOps loop, governance, and multi-account capacity all hang off LiteLLM's breadth + RouteIQ's control plane on top of it.

---

## 3. Repo orientation — RouteIQ's real layout + the routing seam

Repo root: `/Users/baladita/Documents/DevBox/RouteIQ`. **Dist/project name is `routeiq` but the import package is `litellm_llmrouter`** (src layout: `src/litellm_llmrouter/`). Console script `routeiq` → `litellm_llmrouter.cli:main` (`pyproject.toml:91-92`).

### 3.1 The routing seam — the single place new algorithms plug in

RouteIQ does **not** route inside LiteLLM's native math. It binds to LiteLLM via the official `CustomRoutingStrategyBase` plugin API at ONE mount point, then runs its own strategy stack behind it. Two layers:

- **Strategy contract layer** — an ABC + a thread-safe registry + a pipeline, all in `strategy_registry.py`. **This is the seam.**
  - `class RoutingStrategy(ABC)` — `src/litellm_llmrouter/strategy_registry.py:313`.
  - **THE abstract method:** `select_deployment(self, context: RoutingContext) -> Optional[Dict]` — `strategy_registry.py:321-335`. Returns a LiteLLM deployment dict (a row of `router.model_list`/`healthy_deployments`) or None. Optional overrides: `name` (`:337`), `version` (`:342`), `validate()` readiness hook for the staged-rollout gate (`:347`).
  - `@dataclass RoutingContext` — `:220`; per-request input (`router, model, messages, input, request_id, user_id, tenant_id, metadata, ...`). `get_ab_hash_key()` (`:261`) gives sticky A/B keys (priority `tenant+user > user > request_id > random`).
  - `RoutingStrategyRegistry` — `:493` (thread-safe `RLock`). Registration `register(...)` (`:564`), single-active `set_active(...)` (`:645`), weighted A/B `set_weights(...)` (`:676`), staged/canary `stage_strategy`/`promote_staged`/`rollback_staged` (`:777`/`:830`/`:868`), selection `select_strategy(...)` (`:897`), hot-reload `reload_from_config(...)` (`:1063`). Singleton accessor `get_routing_registry()` (`:1389`).
  - `RoutingPipeline.route(context) -> RoutingResult` — `:1150`: hash → `select_strategy` → `strategy.select_deployment`; **on any exception it silently falls back to `DefaultStrategy`** (`:1199-1210`) — a broken strategy degrades quietly, it does NOT error loudly. Emits the `routeiq.router_decision` OTEL span event (`_emit_routing_telemetry`, `:1221`) — **this is the MLOps capture point.**

- **Algorithm layer** — the concrete strategies:
  - `strategies.py` (123 KB): `LLMROUTER_STRATEGIES` list of 19 names (`:1820`); inference-only adapters `InferenceKNN/SVM/MLP/MF/ELORouter`; `class LLMRouterStrategyFamily` (`:1925`, the dispatch seam, lazy-loads UIUC models); `class CostAwareRoutingStrategy(RoutingStrategy)` (`:2669`) — **the reference example of an ABC subclass living here, the exact shape the new Kumaraswamy-Thompson strategy copies.**
  - `centroid_routing.py`: `CentroidRoutingStrategy` (`:1135`, ~2ms zero-config classifier) — duck-typed to the ABC surface, not a literal subclass. `register_centroid_strategy()` (in `custom_routing_strategy.py:941`) is the call-site shape new strategies copy.
  - `personalized_routing.py`: `PreferenceStore` (`:266`) EMA per-user vector + `γ^days` decay — the closest existing online-feedback analog to the new bandit (carries a point estimate, no uncertainty — the gap Thompson fills). `record_feedback(user_id, model, score)` (`:584`).
  - `router_r1.py`: Router-R1 iterative reasoning, exposed at `POST /api/v1/routeiq/routing/r1`.

- **The LiteLLM mount** — `custom_routing_strategy.py`:
  - `class RouteIQRoutingStrategy(CustomRoutingStrategyBase)` — `:140`. The ONE place RouteIQ touches LiteLLM internals (ADR-0002). Imports fail gracefully to a stub base class if litellm is absent (test env).
  - `async_get_available_deployment(...)` — `:190`. A HARD-CODED progressive-enhancement chain: governance (`:318`) → amplification guard (`:389`, caps 3 routings/request, defuses LiteLLM bug #17329) → pipeline/A-B (`_route_via_pipeline`, `:429` — **this is how registry strategies get invoked**) → direct ML family (`:485`) → personalized (`:594`) → centroid (`:534`) → first-healthy fallback (`:850`).
  - Arm set = `healthy_deployments` filtered by `model_name == requested group`; arm key = `litellm_params.model` (`_get_model_list`, `:738`). **Arm set is DYNAMIC** (providers + circuit breakers add/remove arms) — any bandit must tolerate appearing/vanishing arms.
  - `install_routeiq_strategy(router, strategy_name)` — `:886` → `router.set_custom_routing_strategy(strategy)`.

**Where the new P3 work plugs in (no hot-path edit needed):** add `src/litellm_llmrouter/kumaraswamy_thompson.py` with `class KumaraswamyThompsonStrategy(RoutingStrategy)` implementing `select_deployment`; register it exactly like centroid via `get_routing_registry().register("routeiq-kumaraswamy-thompson", ...)`; activate/A-B/canary via the registry methods above. It rides the existing pipeline → `select_strategy` path with ZERO changes to the LiteLLM mount. The `thompson`/`kumaraswamy` symbols do **not** exist in `src/` yet (confirmed) — P3 is genuinely net-new code.

### 3.2 The LiteLLM integration + Responses bridge

- **Pin:** `litellm==1.82.3` (hard `==`, `pyproject.toml:21`), installed **without** the `[proxy]` extra — proxy runtime deps (`apscheduler`, `email-validator`, `fastapi-sso`, `websockets`, `backoff`, `redis`) are added individually (`pyproject.toml:19-33`). Bumping the pin risks the `CustomRoutingStrategyBase` ABI + the missing-proxy-dep set.
- **Boot:** `startup.py:main()` (`:708`) runs the LiteLLM proxy **in-process under uvicorn, never `os.execvp`** (`:494-515`) — that preservation is the whole reason RouteIQ controls boot. Default own-app mode (`ROUTEIQ_OWN_APP=true`) → `create_gateway_app()` (`gateway/app.py:689`) owns the FastAPI app and **mounts the LiteLLM proxy as a sub-app at `/v1/`** (`gateway/app.py:780-790`). The legacy `litellm.Router` monkey-patch is **gone** — `_apply_patch_safely()` is a no-op stub; the only routing integration is `set_custom_routing_strategy()`.
- **Responses-API "bridge" — the load-bearing correction:** `LiteLLMAnthropicToResponsesAPIAdapter` / `responses_api_bridge_check` are **upstream LiteLLM symbols that exist ONLY under `reference/litellm/`** — they are NOT RouteIQ code. There is **no RouteIQ-authored Responses adapter in `src/`**. The chat/messages ↔ Responses translation is **inherited from LiteLLM** (forbidden to claim as a RouteIQ feature, per CLAUDE.md). `/v1/responses` works because the whole proxy is mounted at `/v1/`. What RouteIQ *does* add is **Responses-API awareness** in its own control plane: path→api_type registry `LLM_API_PATHS` maps `/v1/responses`, `/responses`, `/openai/v1/responses` → `"responses"` (`telemetry_contracts.py:709`); policy can allow chat but deny Responses (`PolicyContext.api_type`, `policy_engine.py:209`); the routing strategy accepts Responses-shaped bodies (`input` arg, not just `messages`). The gpt-5.5 framing ("the bridge closes the gap EAIG can't") is the *strategic motivation* — LiteLLM's inherited `/responses` translation is the mechanism that *can* reach it; it is a planned capability, not shipped RouteIQ code.

### 3.3 deploy/ + Helm (the only shipped deploy artifact)

- **There is NO `cdk/` anywhere in RouteIQ** (verified). The AWS story today is the 641-line manual runbook `docs/deployment/aws.md` (ADR-0030 replaces it).
- **Helm chart:** `deploy/charts/routeiq-gateway/` (`Chart.yaml` v1.0.0-rc1, `kubeVersion >=1.23`, 14 K8s template manifests + `_helpers.tpl` + `NOTES.txt`, self-contained). It **emits K8s objects only and presupposes an AWS-native cluster it does not create** ("a chart with no cluster under it" — ADR-0030). What it already expects from AWS:
  - **IRSA ServiceAccount** (`templates/serviceaccount.yaml:8-11` + `values.yaml:205-212`): `eks.amazonaws.com/role-arn` annotation (commented example at `values.yaml:206-207`).
  - **External Secrets Operator** (`templates/externalsecret.yaml`, `values.yaml:393-409`): a `ClusterSecretStore` named `aws-secrets-manager`.
  - **Prometheus Operator** `servicemonitor.yaml` / `prometheusrule.yaml` / `grafana-dashboard.yaml` (need a Prometheus + Grafana nobody provisions yet).
  - **PDB / HPA / Ingress / NetworkPolicy** (NetworkPolicy default-OFF; egress already excludes private ranges 10/8,172.16/12,192.168/16 — so LLM egress is allowed, internal blocked; P4 must add explicit egress to Aurora:5432 / cache:6379).
  - **Service** is `ClusterIP` with NO `loadBalancerSourceRanges` field yet (must be added for the Auto-Mode CIDR-lock lesson — §9).
  - Pod spec (`templates/deployment.yaml`, 187 lines): **stateless** — `emptyDir` volumes only, NO PVC (`:154-163`); db-migrate initContainer (`:44-64`); graceful drain `preStop sleep 5` + `terminationGracePeriod 30` (`:128-131`); `RollingUpdate maxSurge 1/maxUnavailable 0`; security context `runAsNonRoot`/`readOnlyRootFilesystem: true`/drop-ALL-caps. Env contract rendered by `_helpers.tpl:159-329` (`envVars`) — all `ROUTEIQ_*`/`LITELLM_*`/`REDIS_*`/`DATABASE_URL`/`OTEL_*`. `DATABASE_URL` is string-interpolated `postgresql://user:$(POSTGRES_PASSWORD)@...` (`_helpers.tpl:142,276`) — carries the VSR "no env-expansion → boot-render" lesson.
- **Docker:** `docker/Dockerfile` (301 lines, multi-stage, uv-driven, lockfile-frozen). Slim (~500MB, no ML deps) vs full (~2GB) via `ROUTEIQ_EXTRAS`/`BUILD_UI`/`INSTALL_LLMROUTER` build-args. Non-root UID 1000, tini, HEALTHCHECK → `/_health/live`. **Publishes to `ghcr.io/baladithyab/routeiq`** (`values.yaml:33`) — this is why P0 ports the ECR construct as a **GHCR pull-through cache** (zero image-pipeline rework). NOTE a discrepancy: `Dockerfile` default build-arg pins LiteLLM `1.81.3` and Python 3.14, while `pyproject.toml` pins `1.82.3` / `>=3.12` — pyproject is authoritative for the app.
- **Tests layout:** `tests/unit/` (82 files, fast, no external deps), `tests/integration/` (15 files, need the Docker stack on host :4010 else auto-skip), `tests/property/` (8 hypothesis files), `tests/perf/`. Root `tests/conftest.py` auto-skips integration when :4010 is closed.

---

## 4. What already landed in THIS repo (commit `f09c937`)

The VSR deployment patterns are already ported here as design artifacts — **start here, don't re-derive.** All dated 2026-06-14, **Status: Proposed**, adversarially reviewed PASS (0 CRITICAL).

| File | One-line summary |
|---|---|
| `docs/adr/0026-appconfig-gitops-config-delivery.md` | AppConfig as the validated/staged/audited GitOps config plane (Lambda-ARN validator for cross-field rules, linear 20%/12min + 5min bake) over RouteIQ's S3+ETag hot-reload; also delivers `strategy_registry.py` A/B + staged-rollout. |
| `docs/adr/0027-otel-amp-amg-observability-on-aws.md` | OTel → AMP (ADOT remote_write) + AMG (flag-gated OFF) + CloudWatch metric filters + 9 alarms; single `routeiq-{env}-oncall` SNS topic; per-model dimensioned filter keyed on `$.gen_ai.response.model`. |
| `docs/adr/0028-aurora-postgres-serverless-v2-state.md` | Aurora Serverless v2 (PG 16, scale-to-zero, IAM auth, KMS, 30-day secret rotation, schema-bootstrap Lambda + custom resource) for RouteIQ's `asyncpg`/`DATABASE_URL` state. Pin CDK enum AND in-region engine version. |
| `docs/adr/0029-elasticache-serverless-valkey-cache.md` | ElastiCache Serverless (Valkey v8, redis-py wire-compat) for cache + rate-limit + EMA state; IAM-auth user group (`user_id==user_name`), KMS, TLS always required (`REDIS_SSL=true`). |
| `docs/adr/0030-eks-auto-mode-irsa-substrate.md` | EKS Auto Mode (L1 `CfnCluster`, K8s 1.33, 3 Auto-Mode blocks) + OIDC + IRSA factory with `CfnJson` trust-key. CDK owns cluster+IAM+OIDC; Helm/kubectl owns the app. |
| `docs/architecture/aws-rearchitecture/10-aws-native-target-architecture.md` | The full target: RouteIQ stateless control plane on EKS Auto Mode → Aurora / ElastiCache / AppConfig / AMP-AMG / data-lake; the per-module → CDK-construct map; the 3-layer HA story; what RouteIQ KEEPS as its edge (LiteLLM Responses translation + strategies + control plane). |
| `docs/architecture/aws-rearchitecture/20-kumaraswamy-thompson-router.md` | The NEW Thompson-sampling bandit with Kumaraswamy closed-form quantile sampling (math sympy-verified in the 99-doc); plugs into the `RoutingStrategy` ABC. P3 net-new. |
| `docs/architecture/aws-rearchitecture/30-migration-roadmap.md` | **The P0–P4 phased plan (~15–20 eng-weeks) — this is the work queue (§8 below).** |
| `docs/architecture/aws-rearchitecture/40-pluggable-routing-and-mlops.md` | The pluggable routing-adapter framework (`RoutingAdapter` Protocol superset of the ABC, entry-point discovery, `AdapterManifest`) + the telemetry→train→sign→hot-reload MLOps loop. |
| `docs/architecture/aws-rearchitecture/99-review-findings.md` | The adversarial review (3 lenses PASS, 0 CRITICAL, 2 LOW residuals resolved/accepted). |
| `docs/architecture/aws-rearchitecture/vllmsr-patterns.md` | **The canonical construct→pattern source map** — which VSR `cdk/lib/*.py` taught each pattern (the "what to port" list, §8). |

Also for orientation: `docs/architecture/four-way-comparison.md` + `vsr-vs-routeiq-decision-v3.md` (LiteLLM vs LLMRouter vs RouteIQ vs VSR positioning); cross-repo `../vllm-sr-on-aws/docs/architecture/routeiq-vs-vllmsr-aws-gap.md` (the gap analysis that motivated the pivot). Earlier `plans/v1.0-rearchitecture-plan.md` + `plans/technical-roadmap.md` (2026-04-02) are the **framework** rearchitecture (own-the-app, plugin extraction, K8s leader election) — they PREDATE the AWS pivot and are about the Python framework, not the substrate. The substrate plan is exclusively the `aws-rearchitecture/` set + ADRs 0026–0030.

---

## 5. The state / MLOps / governance / observability layers → AWS service map

Each RouteIQ subsystem the substrate must back, the service it maps onto, and its ADR. (All paths under `src/litellm_llmrouter/`.) **Two caveats that bite:** (a) some subsystems read NON-`ROUTEIQ_`-prefixed env vars that take precedence over typed settings (`REDIS_*`, `CONFIG_*`, `LLMROUTER_*`, `CONVERSATION_AFFINITY_*`) — the substrate's secret/config delivery must use those exact names; (b) governance + quota are **fail-open on Redis loss**, so ElastiCache scale-to-zero cold starts open a spend/rate window unless `*_FAIL_MODE=closed`.

| Layer | RouteIQ module (file:line) | AWS service | ADR / construct |
|---|---|---|---|
| **Durable state** | `database.py` — `get_db_pool()` `:63` (single shared `asyncpg.Pool` from `DATABASE_URL`), inline-DDL `run_migrations()` `:472`. **GOTCHA: TWO migration paths** — asyncpg DDL (a2a/mcp/activity/audit) AND LiteLLM Prisma `prisma db push` via `migrations.py` (leader-gated, `ROUTEIQ_LEADER_MIGRATIONS`). Aurora must satisfy both. | **Aurora PG Serverless v2** | ADR-0028 / `replay_store_construct.py` |
| **Hot state / cache** | `redis_pool.py` — `get_async_redis_client()` `:133`. **GOTCHA: mixed client usage** — `quota.py:408` uses the deprecated per-call factory, `governance.py` uses the singleton → size connections accordingly. | **ElastiCache Serverless (Valkey)** | ADR-0029 / `cache_construct.py` |
| **Rate-limit / quota** | `quota.py` — `quota_guard()` `:1065` (all checks PRE-request, streaming-safe), atomic Lua scripts `:299/:325/:357`. **Fail-open default** `:521`. | ElastiCache | ADR-0029 |
| **MLOps — artifact registry** | `model_artifacts.py` — `ModelArtifactVerifier.verify_artifact()` `:475` (Ed25519/HMAC), `ManifestSigner` `:1319`, `ModelActivationManager` stage/promote/rollback `:954/:1043/:1105`. **LIVE** (hot-path callers in `strategies.py`). Env: `LLMROUTER_MODEL_*` (not `ROUTEIQ_`). **GOTCHA: activation-state JSON is a local file** → needs EFS/shared storage on multi-replica EKS or staged/promote diverges per pod. | **S3 artifacts + KMS/Secrets keys** | doc 40 MLOps loop / `security_construct.py` |
| **MLOps — eval loop** | `eval_pipeline.py` — `EvalPipeline` `:240`, `push_feedback()` `:359`. **GOTCHA: the loop runs but is FED NOTHING** — zero callers of `pipeline.collect(...)` in `src/`. The real corpus comes from the **telemetry→data-lake** path, not this in-process pipeline. | **Data Lake (Firehose→S3 Parquet→Glue→Athena)** | doc 10/40 / `data_lake_construct.py` |
| **Config delivery** | `config_sync.py` (S3+ETag, leader-elected) + `hot_reload.py` (A/B + staged control, `:108`). **GOTCHA: reload = `os.kill(SIGHUP)` to self** — AppConfig replaces the S3 source + adds a validation Lambda gate; the SIGHUP mechanism stays. | **AWS AppConfig** | ADR-0026 / `config_state_construct.py` |
| **Governance / tenancy** | `governance.py` — `GovernanceEngine.enforce(key,model)` `:426`, Org→Workspace→Key most-restrictive-wins. **GOTCHA: workspace/key metadata is in-memory + optional JSON file, NOT in Postgres today** → porting to Aurora is ADDITIVE (no governance table exists). Budget checks fail-open `:404`. | Aurora (metadata + durable spend) + ElastiCache (counters) | ADR-0028 + ADR-0029 |
| **Identity / SSO** | `oidc.py` (1475 LOC) — `resolve_identity()`, JWKS validation, OIDC token → RouteIQ API key. Config all `ROUTEIQ_OIDC_*`; needs `authlib` (optional extra). | ALB/ACM ingress + External Secrets → Secrets Manager + KMS | doc 10 §141 / `security_construct.py` |
| **Observability** | `telemetry_contracts.py` — `routeiq.router_decision.v1` contract, `gen_ai.*` attrs `:650`, PII-safe (logs `query_length`, never prompt content). Emission in `router_decision_callback.py` / `observability.py`. | **AMP + AMG + CloudWatch** | ADR-0027 / `observability_construct.py` |
| **Dormant: conversation affinity** | `conversation_affinity.py` — `ConversationAffinityTracker` `:105` maps `response_id`→deployment for Responses provider-affinity. **CONFIRMED DORMANT** — zero hot-path callers; finished feature with no socket. doc 40 proposes wiring it as a built-in adapter consulted only when a request carries `previous_response_id`. | ElastiCache (its Redis backend) | doc 40 §1.5/§5.1 |

**Two of the four MLOps/state features are scaffolds, not live** (`eval_pipeline` COLLECT, `conversation_affinity`). The real training corpus is the telemetry→data-lake path; the affinity feature needs adapter-wiring before its ElastiCache usage matters.

---

## 6. Preserved MLOps corpus (the seed for the loop)

The VSR AWS infra was torn down 2026-06-14 (greenfield — 5 CFN stacks across 5 accounts, VPC clean). **The routing/MLOps corpus was preserved 3 ways** and is the seed for RouteIQ's MLOps loop (doc 40 / P3):

- **RDS snapshot:** `vllmsr-replaystore-final-pre-routeiq-greenfield` (account `386931836011`, us-west-2, `available`) — full Aurora restore point.
- **Logical export (portable, VERIFIED on disk):**
  - Local: `/Users/baladita/Documents/DevBox/vllm-sr-on-aws/mlops-corpus-export/router_replay_records.sql` (1.65 MB) + `sft_corpus.csv` (767 KB).
  - Durable S3 (protected `-v2` bucket — **never delete**): `s3://vllm-sr-dev-386931836011-us-west-2-v2/mlops-corpus-export/pre-routeiq-greenfield-2026-06-14/`.
- **Row stats:** **241 rows; 237 complete `prompt`+`completion`+`selected_model` SFT tuples; 31 distinct models routed.** This is what `BUFFERED` response capture got us — it's the de-risk corpus for building P3's bandit + MLOps loop offline before wiring live.
- The **`vllm-sr-on-aws` repo/CDK is intact** as the PORT REFERENCE (`/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/`).

---

## 7. The vllm-sr CDK constructs to PORT (the source map)

Cross-repo path: `/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/` — all confirmed present. Port priority by roadmap phase (from `vllmsr-patterns.md`). **Cite VSR constructs by SYMBOL NAME and grep for the class/method — the `:NNN` offsets in the ADRs are accurate-as-of-2026-06-14 but the VSR repo will drift on its next commit.**

**P0 (foundation — port first):**
- `eks_cluster_construct.py` — EKS Auto Mode L1 `CfnCluster` + 3 Auto-Mode blocks + IRSA-via-`CfnJson` + OIDC provider + container-insights + per-model metric filter. `class EksClusterConstruct`, `irsa_role()` factory, `enable_container_insights()`. [ADR-0030, ADR-0027]
- `vllm_sr_eks_stack.py` — own-VPC pattern + per-workload IRSA factory + CfnOutputs. [ADR-0030, doc 10]
- `network_construct.py` — VPC, 7 SGs, 11 interface endpoints incl. `BEDROCK_RUNTIME`, S3 gateway endpoint, Cloud Map. [doc 10]
- `ecr_construct.py` — immutable + scan-on-push repos + **GHCR/DockerHub pull-through cache**. [doc 30]
- `nag_suppressions.py` — evidenced cdk-nag suppression pattern. [reference]

**P1 (state):**
- `replay_store_construct.py` — Aurora PG Serverless v2 (scale-to-zero, IAM auth, KMS, 30d rotation, schema-bootstrap Lambda+CR). [ADR-0028]
- `cache_construct.py` — ElastiCache Serverless Valkey (IAM-auth user group, KMS, always-on TLS). [ADR-0029]
- `security_construct.py` — KMS CMKs, task/exec roles (Bedrock invoke + bearer-mint), Secrets. [ADR-0030]

**P2 (config/obs/lake):**
- `config_state_construct.py` — AppConfig GitOps (App/Env/Profile/Strategy/Version/Deployment + validator Lambda + S3 + SNS). [ADR-0026]
- `observability_construct.py` — AMP `CfnWorkspace` + AMG (flag-gated) + CW dashboard + 9 alarms + 3 metric filters + 2 log groups (PII DataProtectionPolicy) + SNS. [ADR-0027]
- `data_lake_construct.py` — routing_decision CW Logs → Firehose (JSON→Parquet) → S3 → Glue (partition projection, no crawler) → Athena. [doc 10, doc 40]

**P3/P4 (optional/later):**
- `build_pipeline_construct.py` (CodeBuild/CodePipeline model-promotion + deploy), `waf_construct.py` (WAFv2), `native_guardrail_construct.py` (`bedrock.CfnGuardrail`, flag-gated), `bedrock_capacity_member_stack.py` (cross-account capacity — a SEPARATE app, DEFER), `jailbreak_canary_construct.py` / `opus_share_monitor_construct.py` / `policy_version_construct.py`, `_rl_baselines.py` (doc 20 contrast).

**Skip (VSR-specific):** `compute_construct.py` (ECS Fargate — RouteIQ is EKS), `vllm_sr_stack.py` (old ECS stack), `cosign_signer_construct.py`, `dns_construct.py`, `replay_redactor_construct.py`. `_lambda_bundling.py` is a helper likely needed transitively by the Aurora bootstrap.

---

## 8. The P0–P4 work queue (from `30-migration-roadmap.md`)

Critical path: **P0 → {P1 ∥ P2} → P3 → P4.** Total **~15–20 eng-weeks.** A deployable substrate running the existing framework (P0+P1+P2+P4-wiring) lands well before the P3 net-new work.

| Phase | Scope | Est. | Type |
|---|---|---|---|
| **P0 — CDK Foundation** | VPC/SGs/endpoints + EKS Auto Mode + IRSA factory + ECR/GHCR-pull-through + cdk-nag. BUILD-NEW (small): a `RouteIqStack` CDK app wiring the constructs + the chart SA IRSA annotation. | ~3–4w | **PORT** (Low risk) |
| **P1 — Externalized State** | Aurora Serverless v2 (`replay_store_construct.py`) + ElastiCache Serverless (`cache_construct.py`); wire chart `externalPostgresql`/`externalRedis` (`values.yaml:316-338`). BUILD-NEW (small): ESO `ClusterSecretStore` + boot-render DB-URL. **Deploy Aurora SEPARATELY from app CI (~30min rollback).** | ~2–3w | **PORT** (Low–Med) |
| **P2 — Config + Obs + Data Lake** | AppConfig (`config_state_construct.py`) + AMP/AMG/CW (`observability_construct.py`) + Firehose data lake (`data_lake_construct.py`); wire app (AppConfig poll adapter, OTel emit, structured `routing_decision` log line). Can overlap P1. | ~3w | **PORT** (Low–Med) |
| **P3 — Routing + MLOps** | **NET-NEW**: Kumaraswamy-Thompson as a `CustomRoutingStrategyBase` plugin (posteriors→Aurora, hot EMA→ElastiCache, arm-config→AppConfig); the adapter+MLOps framework; wire `eval_pipeline.py` to the data-lake corpus. PORT (optional): VSR `rl_driven` Thompson as one backend strategy. | ~4–6w | **BUILD-NEW (RouteIQ's differentiation)** |
| **P4 — Governance / Tenancy / UI hardening** | WIRE `governance.py`/`usage_policies.py`→Aurora; `oidc.py` + React UI behind ALB/ACM + ESO; optional native Guardrail + WAF; HARDEN NetworkPolicy (egress to Aurora:5432/cache:6379) + readOnlyRootFS. | ~3–4w | **HARDEN** (Med) |

**Biggest risk:** **P3 is the only net-new code** (the rest is a tested CDK port — VSR's constructs carry 446 tests, live across two substrates). De-risk P3 by building the bandit + MLOps loop against the **preserved corpus (§6)** offline before wiring it live.

---

## 9. Dev loop cheat-sheet + conventions + standing constraints

### 9.1 Dev loop (package manager is **uv**; everything runs through `uv run`; a Makefile wraps the same)

```bash
# Install
uv sync --extra dev                 # + pytest, ruff, mypy, hypothesis
uv sync --extra prod                # all production extras (db,otel,cloud,callbacks,knn,a2a,hotreload,oidc)
make setup                          # uv sync --extra dev + install_lefthook.sh

# Run the gateway (two equivalent entry points)
uv run python -m litellm_llmrouter.startup --config config/config.yaml --port 4000
uv run routeiq start --config config/config.yaml --port 4000 --workers 1
uv run routeiq validate-config --config config/config.local-test.yaml
uv run routeiq probe-services       # probes Postgres/Redis/OTel/OIDC

# Tests
uv run pytest tests/unit/ -x                       # fast, no external deps (82 files)
uv run pytest tests/integration/                   # needs Docker stack on :4010 (else AUTO-SKIPS)
uv run pytest tests/property/                       # hypothesis (max_examples=100)
make test          # uv run pytest tests/unit/ -x -v --tb=short
make test-all      # uv run pytest tests/ -x -v

# Lint / format / types
uv run ruff format src/ tests/ && uv run ruff check --fix src/ tests/
uv run mypy src/litellm_llmrouter/ --ignore-missing-imports
make lint / make fix / make typecheck

# Git hooks (lefthook)
./scripts/install_lefthook.sh
lefthook run pre-commit             # ruff-format + ruff-check + yamllint + detect-secrets + detect-private-keys + whitespace (auto-fixes & RE-STAGES)
lefthook run pre-push               # unit-tests (pytest -x) + mypy (best-effort) + full secret-scan
LEFTHOOK=0 git commit ...           # bypass

# Local full HA stack (gateway on host :4010 → container :4000)
docker compose -f docker-compose.local-test.yml up -d   # postgres/redis/jaeger/minio/mlflow/mcp stubs
# Bedrock creds first: eval $(aws --profile baladita+Bedrock-Admin configure export-credentials --format env)
```

### 9.2 CLAUDE.md hard rules (do not violate)
- **Always call it "RouteIQ."** Upstream is "LiteLLM" / "LLMRouter".
- **Do NOT rename `LITELLM_*` env vars.**
- **Do NOT claim RouteIQ implements features only inherited from upstream LiteLLM** (esp. the chat/messages↔Responses translation — that's LiteLLM's).
- **`reference/litellm/` is a READ-ONLY submodule — never modify.**
- **No real secrets in code/tests** — use `test-api-key`/`test-key` placeholders (secret-scan hook enforces).
- **New config goes through `settings.py` Pydantic `BaseSettings` via `get_settings()`, not `os.environ.get()`** (ADR-0013) — EXCEPT the four legacy non-`ROUTEIQ_`-prefixed namespaces (`REDIS_*`, `CONFIG_*`, `LLMROUTER_*`, `CONVERSATION_AFFINITY_*`) which take precedence and whose names the substrate must use verbatim.
- **Test discipline:** singletons everywhere → every new subsystem MUST add its `reset_*()` to the autouse `_reset_all_singletons()` in `tests/unit/conftest.py` or you get cross-test contamination; OTel tests use the `shared_span_exporter` fixture (never call `trace.set_tracer_provider()` in a test); mock all external services.
- **Docs/architecture commits land direct-to-main**; feature work uses `tg<id>-<desc>` branch → `git merge --squash` → single `feat: complete TG<id> ...` commit. Conventional Commits with em-dash subject separators.
- **Issue tracker is Beads (`bd`, Dolt backend, `.beads/`/`BEADS.md`)** — the Seeds/Mulch blocks in CLAUDE.md are os-eco onboarding boilerplate; confirm with the operator before filing.
- **Python: treat 3.12 as the floor (pyproject authoritative); CI runs 3.14.** Don't trust AGENTS.md's stale counts ("~30 unit", "3.14+ required").

### 9.3 Standing operational constraints (carry from the vllm-sr stream)
- AWS profile `baladita+Bedrock-Admin`, account `386931836011`, **us-west-2** — pass `--region us-west-2` explicitly (env default is us-east-1).
- **Commit only, never push.** Trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Never log token/bearer values. Never delete the `-v2` buckets.**
- Deploy/cutover is **operator-gated** — produce committed, flag-gated IaC + CDK tests + runbooks; the operator runs `cdk bootstrap`/`cdk deploy`/`kubectl`/`helm`. Do NOT run `aws`/`kubectl`/`cdk` yourself (no creds; they hang).
- **Fable 5 is gov-banned as a routable arm** — never add it to live routing config.

### 9.4 Hard-won live-ops lessons (must be encoded or the deploy silently breaks)
1. **Auto-Mode CIDR-lock = `spec.loadBalancerSourceRanges`, NOT the LB annotation** (the annotation is silently ignored under Auto Mode). RouteIQ's `service.yaml` has no source-ranges field yet — add it. Allowlist the NAT-pool **/24, not a /32**.
2. **IRSA trust-key MUST be the ARN-derived OIDC issuer wrapped in `CfnJson`** — `issuer_url.replace("https://","")` is a silent no-op against an unresolved CFN token → `AssumeRoleWithWebIdentity` AccessDenied (root-caused live). The L1 `CfnCluster` forces this manual path.
3. **RWO-EBS + RollingUpdate = Multi-Attach deadlock** → keep RouteIQ pods stateless (chart volumes are emptyDir only); if any PVC is ever added use `strategy: Recreate`.
4. **APM/observability webhook is PSA-restricted** → explicit opt-out label on every injected pod.
5. **`kubectl apply -k` clobbers the IRSA SA annotation** (writes the literal `${...}`) → use `helm upgrade` or `envsubst` in a `deploy-*.sh` wrapper, NEVER bare `apply -k`.
6. **Aurora rollback ~30 min** → deploy the DB in a SEPARATE CI stage from the app.
7. **CW MetricFilter needs a CDK-created log group that pre-exists** — a runtime-created (Fluent Bit `auto_create_group`) group makes the CFN MetricFilter fail/roll back on first deploy.
8. **Aurora engine-version retirement** → pin a CDK enum AND verify the in-region version is still offered (VSR was bitten by "Cannot find version 15.4"; synth/nag tests cannot catch this).
9. **ElastiCache IAM auth `user_id == user_name`** must match exactly or auth fails opaquely; TLS is non-optional → `REDIS_SSL=true`.
10. **`DATABASE_URL` `$(POSTGRES_PASSWORD)` string-interp** needs runtime shell env-expansion or a boot-render step (VSR's `render_replay_config.py` lesson) — K8s `value:` does NOT expand `$(VAR)` unless it references a prior env entry in the same container.

---

## 10. First three concrete actions for the next session

1. **Read** this handoff → `docs/architecture/aws-rearchitecture/30-migration-roadmap.md` → skim `docs/adr/0026-...md` through `0030-...md`.
2. **Decide the CDK home.** Recommendation: **create `deploy/cdk/` in THIS repo** (co-locates with the existing `deploy/charts/routeiq-gateway/`, keeps RouteIQ self-contained, satisfies the P0 deliverable "a `cdk/` tree with `RouteIqStack`"). Mirror the VSR scaffold from `../vllm-sr-on-aws/cdk/`: `deploy/cdk/app.py` + `cdk.json` (`"app": "python3 app.py"`) + `deploy/cdk/lib/<construct>.py` + `deploy/cdk/{requirements.txt or pyproject}` (pin `aws-cdk-lib>=2.150.0,<3`, `cdk-nag>=2.27.0,<3`, `cdk-monitoring-constructs>=10,<11`, `constructs>=10,<11`, `pytest`) + `deploy/cdk/tests/{unit,snapshot}/` with a per-construct unit test + a `test_template_snapshot.py`. One `RouteIqStack` likely suffices initially (single pod, one VPC) but split Aurora into a separate stack/CI-stage (the ~30min-rollback rule); defer the VSR two-app cross-account split.
3. **Propose the P0 plan BEFORE writing CDK** — VPC + EKS Auto Mode + ECR/GHCR-pull-through + IRSA factory (one pod role: `aps:RemoteWrite` + `rds-db:connect` + `elasticache:Connect` + AppConfig read + Bedrock invoke) + cdk-nag, parameterized for RouteIQ's stateless pod and wired to the chart's IRSA SA annotation (`values.yaml:206-207`). Carry the synth-time guards: ASCII-only IAM descriptions; VPC-quota / retain-resources on teardown. Then build.

---

## 11. Useful skills + cross-repo references

**Skills (auto-surface, but worth invoking deliberately):**
- `cdk-gotchas` — the CDK/CloudFormation gotcha catalog (IRSA `CfnJson`, `Arn.format` slash-vs-colon, token `.replace` silent no-op, EKS `fromClusterAttributes`, custom-resource AccessDenied). **Directly relevant to P0–P2.**
- `eks-container-insights-fluentbit-wraps-stdout`, `kms-pending-deletion-blocks-ecr-pull`, `elasticache-rg-to-serverless-name-collision`, `cfn-stack-teardown-rollback-recovery` — substrate-specific failure modes seen live.
- `preserve-state-before-greenfield-teardown` — the safe teardown/preserve sequence used for the §6 corpus.
- `litellm-master-key-db-seeding`, `routeiq-config-reload-oom` — RouteIQ-specific.
- `strategic-replatform-deliberation` — produced the gap analysis that grounds this pivot.

**Cross-repo references:**
- **PORT REFERENCE (VSR, intact):** `/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/` — `cdk/app.py`, `cdk/cdk.json`, `cdk/lib/*_construct.py` (§7), `cdk/requirements.txt`, `cdk/tests/{unit,snapshot}/`.
- **Gap analysis:** `/Users/baladita/Documents/DevBox/vllm-sr-on-aws/docs/architecture/routeiq-vs-vllmsr-aws-gap.md`.
- **Preserved corpus:** local `/Users/baladita/Documents/DevBox/vllm-sr-on-aws/mlops-corpus-export/` + S3 `s3://vllm-sr-dev-386931836011-us-west-2-v2/mlops-corpus-export/pre-routeiq-greenfield-2026-06-14/` + RDS snapshot `vllmsr-replaystore-final-pre-routeiq-greenfield`.

---

*Handoff authored at the close of the vllm-sr-on-aws stream. The substrate is proven, the patterns are documented here, the corpus is preserved. RouteIQ is the gateway; LiteLLM is the base that connects the extra features. P0–P2 are a tested CDK port; P3 is the net-new differentiation. Build forward.*
