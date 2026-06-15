# RouteIQ — Canonical Architecture

> **As of v1.0.0rc1, 2026-06-15.** This is the single canonical architecture map for
> RouteIQ. It is derived from a read-only frame-by-frame trace of the live code under
> `src/litellm_llmrouter/` and `deploy/cdk/`. Every node and edge in every diagram below
> reflects a real module, class, or call path — there are no aspirational boxes.
>
> Diagrams are **inline GitHub-flavored Mermaid** (v10+). GitHub renders them natively;
> do not export PNGs.

---

## 1. The two-layer model (the inversion)

RouteIQ is a **host application that owns its own FastAPI app and treats LiteLLM as a
mounted routing plugin** — the inverse of the pre-v1.0 layout where RouteIQ mounted itself
on top of `litellm.proxy.proxy_server.app`. Two ownership facts define the whole system
(see `CLAUDE.md` "Non-Obvious Behaviors" and the `aws-rearchitecture/` set, especially
`10-aws-native-target-architecture.md`, `40-pluggable-routing-and-mlops.md`, and
`50-litellm-universal-surface.md`):

1. **App ownership.** `create_gateway_app()` (`gateway/app.py`, ADR-0012) constructs a fresh
   `FastAPI(lifespan=_routeiq_lifespan)`, controls the entire middleware stack and route
   table, and calls `app.mount("/v1", litellm_app)` — LiteLLM is a **sub-application**, not
   the host.

2. **Routing ownership.** RouteIQ does **not** monkey-patch LiteLLM's `Router`. It installs
   its routing brain via LiteLLM's official `CustomRoutingStrategyBase` plugin API
   (ADR-0002): `install_plugin_routing_strategy(app)` → `router.set_custom_routing_strategy(
   RouteIQRoutingStrategy(...))`, which swaps the Router's `get_available_deployment` /
   `async_get_available_deployment` methods. This is multi-worker safe (registers
   per-worker, no patch to preserve).

Above the request path sits a **control plane** (governance / usage-policies / quota / RBAC
/ OIDC / policy-engine / audit) that admits-or-rejects a request *before* it reaches the
router. Out of the dataplane runs the **only true cycle** in the system: a
COLLECT → EVALUATE → AGGREGATE → FEEDBACK loop (`eval_pipeline.py`) that pushes quality
signal back into the routing arms via a strategy-agnostic MLOps adapter ABI. The whole stack
runs on an **AWS substrate** of three independent CDK stacks (P0 EKS/ECR foundation, P1
Aurora/ElastiCache state, P2 AppConfig/observability/data-lake), with a single EKS Pod
Identity role threading the running app to the cloud.

**The reader should walk away understanding:** RouteIQ is the host; LiteLLM is a routing
plugin inside it; the ML routing intelligence is a hot-swappable arm registry fed by a closed
eval loop; governance gates the front door; and the whole thing runs on a 3-stack
EKS/Aurora/observability AWS substrate.

### Naming convention (used across all diagrams)

- **Solid bold `==>`** — the primary request hot path.
- **Solid `-->`** — a real synchronous call or registration edge.
- **Dashed `-.->`** — a secondary, failure, fallback, or observability/telemetry path.
- **Invisible `~~~`** — layout-only ordering between independent subgraphs.
- Subgraphs are the architectural lanes; cross-lane edges connect to **subgraph boundaries**
  wherever possible (not individual interior nodes) to keep the layout legible.

---

## 2. Eagle-eye view

This single diagram is the spine of the whole document: client → RouteIQ's **own** FastAPI
app (which mounts LiteLLM at `/v1`) → the RouteIQ routing plugin → strategy registry/pipeline
→ LiteLLM Router → the dynamic arm set of providers. The control plane gates the front door;
the eval loop is the curved feedback cycle; the AWS substrate sits beneath everything.

```mermaid
graph TD
    CLIENT["Client / Admin<br/>(OpenAI-compatible + control-plane)"]

    subgraph CONTROL["Control plane (gates the front door)"]
        direction TB
        GATE["Policy / Mgmt / Auth / OIDC / RBAC<br/>governance.enforce + usage_policies + quota"]
        AUDIT["audit.py (denials + mutations sink)"]
        GATE -.-> AUDIT
    end

    subgraph HOST["RouteIQ own FastAPI app (gateway/app.py, ADR-0012)"]
        direction TB
        MW["Middleware stack<br/>Backpressure -> CORS/RequestID/Policy/Mgmt/Plugin/RouterDecision"]
        PLUGIN["RouteIQRoutingStrategy (CustomRoutingStrategyBase)<br/>governance -> pipeline -> ML -> centroid -> fallback"]
        PIPE["RoutingPipeline + RoutingStrategyRegistry<br/>(active arm / A/B weighting / hot-swap)"]
        LLVR["LiteLLM Router (mounted at /v1)<br/>set_custom_routing_strategy"]
        MW ==> PLUGIN ==> PIPE ==> LLVR
    end

    subgraph ARMS["Dynamic arm set (pluggable ML strategies)"]
        direction TB
        STRATS["18+ LLMRouter arms / Kumaraswamy-Thompson /<br/>centroid / personalized / router-R1"]
        PROVIDERS["Providers: Bedrock / OpenAI / self-hosted<br/>(Fable 5 gov-banned as routable arm)"]
        STRATS --> PROVIDERS
    end

    subgraph LOOP["Eval / feedback loop (the only cycle)"]
        direction TB
        EVAL["eval_pipeline: COLLECT -> EVALUATE -> AGGREGATE -> FEEDBACK<br/>+ MLOps adapter ABI"]
    end

    subgraph SUB["AWS substrate (3 CDK stacks + in-process infra)"]
        direction TB
        INFRA["leader-election-gated config-sync/migrations<br/>Redis/Postgres/HTTP pools + OTel"]
        AWS["P0 EKS+ECR / P1 Aurora+ElastiCache / P2 AppConfig+AMP+lake<br/>(single Pod Identity role)"]
        INFRA --> AWS
    end

    CLIENT ==> CONTROL
    CONTROL ==>|"admitted"| HOST
    PIPE ==> ARMS
    LLVR ==>|"deployment dict"| ARMS
    PIPE -.->|"router-decision telemetry"| LOOP
    LOOP -.->|"updated weights -> registry / personalized"| PIPE
    HOST -.->|"OTel spans / metrics / cost / audit"| SUB
    CONTROL -.-> SUB

    CONTROL ~~~ HOST
    ARMS ~~~ LOOP
```

**Five cross-cutting signals** fan out of the hot path (drawn as dashed edges above and
detailed per-section below): routing-decision telemetry, audit events, OTel spans/metrics,
cost tracking, and guardrail verdicts.

---

## 3. Startup & boot order — own-FastAPI + LiteLLM-as-plugin

`cli.py` (`routeiq start`) hands off to `startup.main()`, which runs a sequence of
synchronous, pre-uvicorn `*_if_enabled()` helpers (env validation, observability init,
strategy registration, MLOps feedback wiring) and then branches on `_use_own_app()` (default
`True`) into `_run_gateway_app()`. `create_gateway_app()` is the **composition root**: it
loads `Settings`, loads plugins *before* routes (deterministic order), adds the
`BackpressureMiddleware` **first** (innermost, wrapping the raw ASGI app directly because
`BaseHTTPMiddleware` breaks streaming), configures the outer middleware LIFO, registers
routes, and mounts LiteLLM at `/v1`.

The heavy Router build is **deferred to the async `_routeiq_lifespan`** that runs inside
uvicorn: step 0 calls `_proxy_server.initialize()` to build the LiteLLM Router from config;
step 0b calls `install_plugin_routing_strategy(app)` to swap in `RouteIQRoutingStrategy`. The
**fragile contract**: that install call *must* receive `app` (it reads
`app.state.use_plugin_strategy`); if `app` is omitted the resulting `TypeError` is swallowed
and ML routing **silently falls back to the LiteLLM default** with no error surfaced
(historical fix `59f80e9` + `7844419`). `env_validation` and `service_discovery` are advisory
— neither blocks boot.

```mermaid
graph TD
    subgraph ENTRY["Entrypoints (pre-uvicorn, synchronous)"]
        direction TB
        CLI["cli.py main() — routeiq start"]
        SM["startup.main()<br/>resolve_worker_count, validate_environment,<br/>init_observability, register_strategies, wire_mlops"]
        RUN["run_litellm_proxy_inprocess()<br/>_use_own_app()? -> _run_gateway_app()"]
        CLI --> SM --> RUN
    end

    subgraph BUILD["create_gateway_app() (composition root)"]
        direction TB
        B1["1. load Settings (env + YAML)<br/>set app.state.use_plugin_strategy = True"]
        B2["2. _load_plugins_before_routes()"]
        B3["3. add_backpressure_middleware(app) FIRST<br/>(innermost, wraps app.app ASGI directly)"]
        B4["4. _configure_middleware()<br/>CORS, RequestID, Policy, Mgmt, Plugin, RouterDecision"]
        B5["5. _register_routes() + mount /v1 = LiteLLM"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph LIFE["_routeiq_lifespan() (async, at uvicorn start)"]
        direction TB
        L0["0. _proxy_server.initialize() -> builds LiteLLM Router"]
        L0B["0b. install_plugin_routing_strategy(app)<br/>set_custom_routing_strategy(RouteIQRoutingStrategy)"]
        L1["1-4. HTTP pool, plugin startup, DB migrate +<br/>hydrate governance, probe services, eval loop"]
        L0 --> L0B --> L1
    end

    FALLBACK["LiteLLM default routing<br/>(no error surfaced)"]

    ENTRY ==> BUILD ==> LIFE
    L0B -.->|"app arg MISSING -> TypeError swallowed<br/>ML routing SILENTLY falls back"| FALLBACK
```

---

## 4. Routing dataplane — pipeline, registry & ML strategy arms

An inbound chat/completion request reaches LiteLLM's Router, which invokes RouteIQ's
`RouteIQRoutingStrategy` hook. That hook runs a strict **progressive-enhancement chain**:
`_enforce_governance` (budget/rate-limit, short-circuits with `HTTPException` on violation,
fail-open otherwise) → `_check_amplification_guard` (caps a `litellm_call_id` at 3 attempts)
→ `_route_via_pipeline` (the **primary** path) → and on `None`, falls back through direct
LLMRouter ML → personalized re-rank → centroid (~2ms) → first-healthy deployment.

`RoutingPipeline.route()` computes an A/B hash key (priority `tenant+user` > `user` >
`request_id` > random UUID for sticky variant assignment), asks the thread-safe
`RoutingStrategyRegistry` to `select_strategy` (active arm, or an A/B-weighted variant via
`sha256(hash_key) % total_weight`), runs `strategy.select_deployment(context)`, falls back to
`DefaultStrategy` on exception, and emits routing-decision telemetry out of the hot path. The
registry holds the strategy family as **pluggable arms** and supports hot-swap
(stage → validate → promote/rollback).

```mermaid
graph TD
    LITELLM["LiteLLM Router (.get_available_deployment)"]

    subgraph HOOK["RouteIQRoutingStrategy (custom_routing_strategy.py)"]
        direction TB
        GOV["_enforce_governance (budget / rate-limit)"]
        GUARD["_check_amplification_guard (max 3)"]
        PIPECALL["_route_via_pipeline (builds RoutingContext)"]
        FALLBACKS["fallback chain:<br/>llmrouter -> personalized -> centroid -> first-healthy"]
        GOV --> GUARD --> PIPECALL --> FALLBACKS
    end

    subgraph PIPE["RoutingPipeline.route() (strategy_registry.py)"]
        direction TB
        HASH["get_ab_hash_key (user > request > random)"]
        SELECT["registry.select_strategy -> ABSelectionResult"]
        RUN["strategy.select_deployment(context)"]
        DEFFB["DefaultStrategy fallback on exception"]
        HASH --> SELECT --> RUN --> DEFFB
    end

    subgraph REG["RoutingStrategyRegistry (pluggable arms)"]
        direction TB
        ACTIVE["active strategy (VersionedStrategyEntry)"]
        AB["A/B weighted (ExperimentConfig, sha256 % total_weight)"]
        SWAP["hot-swap: stage / promote / rollback"]
    end

    TELE["router_decision_callback + pipeline telemetry<br/>(X-RouteIQ-* headers, OTel span event)"]
    EVAL(["Section 5: eval / feedback + Section 9: infra"])

    LITELLM ==> HOOK ==> PIPE ==> REG
    REG -.->|"reason: active_strategy / ab_test"| PIPE
    PIPE ==>|"selected deployment"| HOOK
    HOOK ==>|"deployment dict"| LITELLM
    PIPE -.->|"routing-decision event"| TELE
    TELE -.-> EVAL
```

The arms themselves — all implementing `RoutingStrategy.select_deployment(context)` and
registered at startup by `register_strategies()` — form a fan feeding the registry:

```mermaid
graph TD
    REG["RoutingStrategyRegistry (select_strategy -> RoutingStrategy)"]

    subgraph ARMS["Registered RoutingStrategy arms (select_deployment)"]
        direction TB
        FAMILY["LLMRouterStrategyFamily — 18+ ML routers<br/>(KNN/SVM/MLP/MF/ELO/RouterDC/hybrid/causallm/graph/automix)"]
        KTS["KumaraswamyThompsonStrategy — online Beta-bandit<br/>arm = litellm_params.model"]
        CENTROID["CentroidRoutingStrategy — zero-config ~2ms tier match<br/>(all-MiniLM-L6-v2)"]
        R1["RouterR1 — iterative reasoning (think/route/result/answer)"]
    end

    subgraph SIDE["Re-rank + affinity sidecars"]
        direction TB
        PERS["PersonalizedRouter.rank_models — per-user EMA + quality bias"]
        AFF["ConversationAffinityTracker — response_id -> provider"]
    end

    subgraph STATE["State + warm-start + safety"]
        direction TB
        QBIAS["_DEFAULT_QUALITY_BIASES (cold-start prior + quality bias)"]
        REDIS[("Redis / in-memory backends")]
        BREAKER["circuit-breaker manager (drops vanishing arms)"]
        FABLE["Fable 5: GOV-BANNED arm (never added)"]
    end

    REG ==> ARMS
    KTS -.->|"warm-start alpha/beta"| QBIAS
    KTS -.->|"exclude open breakers"| BREAKER
    KTS -.-x FABLE
    PERS -.->|"prefs + quality bias"| QBIAS
    PERS -.-> REDIS
    AFF -.-> REDIS
    ARMS ==>|"deployment dict"| REG
    SIDE -.->|"re-rank / sticky route"| REG
```

---

## 5. Adapter ABI + MLOps framework + eval/feedback loop

This is the **only true cycle** in the system, made of two halves bolted together. **(A) The
adapter ABI** (`adapters/contract.py`) defines a strategy-agnostic `RoutingAdapter` Protocol
that is a superset of the in-tree `RoutingStrategy` ABC; `attach_route_alias()` binds
`route = select_deployment` so every in-tree strategy satisfies the Protocol with zero edits.
`_abi_compatible` does SemVer negotiation; `loader.py` (`AdapterLoaderPlugin`) discovers
*out-of-tree* adapters via the `routeiq.routing_adapters` entry-point group through a
5-gate validate-then-promote pipeline that never raises.

**(B) The eval/feedback loop** (`eval_pipeline.py`) runs COLLECT → EVALUATE (LLM-as-judge via
`litellm.acompletion`) → AGGREGATE (per-model quality) → FEEDBACK. The FEEDBACK arm closes the
loop two ways: directly into `personalized_routing.update_quality_bias` (EMA), and via the
`MLOpsCoordinator.on_aggregate_feedback()` fan-out into every registered continuous-learning
adapter (the Kumaraswamy-Thompson bandit's posterior `update`). `model_artifacts.py` is the
**trust gate**: any artifact reloaded into an adapter (`apply_artifact`) is SHA256 + optional
Ed25519/HMAC verified first; pickle loading is signature-gated and off by default.

> **Honest gap (not aspirational):** the COLLECT producer is not yet auto-wired per request —
> `router_decision_callback.py` emits telemetry/metrics/spend but does not construct
> `EvalSample`s. The live driver today is the admin `POST .../eval/run-batch` endpoint over
> already-queued samples (drawn dashed). FEEDBACK fan-out is flag-gated
> (`adapter_framework.mlops_feedback_loop`) and off by default.

```mermaid
flowchart TB
    subgraph ABI["(A) Adapter ABI seam — adapters/"]
        direction TB
        CONTRACT["contract.py: RoutingAdapter Protocol + AdapterManifest<br/>+ _abi_compatible + ArtifactRef + attach_route_alias"]
        LOADER["loader.py: AdapterLoaderPlugin<br/>(entry-points -> 5-gate stage/validate/promote)"]
        REGISTRY["strategy_registry.py: RoutingStrategyRegistry.stage_strategy"]
        LOADER --> CONTRACT
        LOADER --> REGISTRY
    end

    subgraph LOOP["(B) Eval / feedback loop — eval_pipeline.py"]
        direction TB
        COLLECT["COLLECT: should_sample() + collect() -> deque[EvalSample]"]
        EVALUATE["EVALUATE: EvalJudge.evaluate_batch -> litellm.acompletion (judge)"]
        AGGREGATE["AGGREGATE: ModelQualityTracker -> {model: quality}"]
        FEEDBACK["FEEDBACK: push_feedback() -> callbacks"]
        COLLECT --> EVALUATE --> AGGREGATE --> FEEDBACK
    end

    subgraph MLOPS["MLOps coordinator — adapters/mlops.py"]
        direction TB
        COORD["MLOpsCoordinator: register_learning_adapter /<br/>on_aggregate_feedback / apply_artifact"]
        WIRE["wire_mlops_feedback_loop(): discover continuous adapters + subscribe"]
        WIRE --> COORD
    end

    subgraph DATAPLANE["Dataplane learning adapters"]
        direction TB
        PERS["personalized_routing.py: update_quality_bias (EMA)"]
        KT["kumaraswamy_thompson.py: update_from_feedback -> Beta posterior;<br/>reload/load_artifact warm-start"]
    end

    TRUST["model_artifacts.py: ModelArtifactVerifier.verify_artifact<br/>(SHA256 + Ed25519/HMAC; pickle off by default)"]

    REGISTRY -. "discover learning adapters (learns=True)" .-> MLOPS
    LOADER -. "register_learning_adapter (if manifest.learns)" .-> MLOPS
    FEEDBACK -- "update_quality_bias({model: quality})" --> PERS
    FEEDBACK -- "on_aggregate_feedback({model: quality})" --> MLOPS
    MLOPS -- "RoutingFeedback -> update_from_feedback" --> KT
    MLOPS -- "RoutingFeedback -> update" --> PERS
    MLOPS -- "apply_artifact(ref)" --> TRUST
    TRUST -- "verified -> reload(ArtifactRef)" --> KT
    DATAPLANE -. "routed responses sampled (admin run-batch drives COLLECT today)" .-> COLLECT

    ABI ~~~ LOOP
    LOOP ~~~ MLOPS
```

---

## 6. Control plane — multi-tenant governance, policy & identity

The control plane sits **beside** the dataplane and admits-or-rejects a request *before* it
reaches the router, through an ordered funnel layered at the ASGI/FastAPI seam (outermost
first, fixed in `_configure_middleware()`): `RequestIDMiddleware` (raw ASGI, stamps an
`X-Request-ID` ContextVar) → `PolicyMiddleware` (OPA-style, runs **before** routing AND
FastAPI auth; fail-open default, fail-closed optional; 403 short-circuits) →
`ManagementMiddleware` (classifies management ops, optional RBAC + audit) → two-tier auth
(admin `X-Admin-API-Key`, **fail-closed**; vs. LiteLLM `user_api_key_auth`) → OIDC SSO
(token-exchange of an IdP JWT for a RouteIQ key) → RBAC (`requires_permission`) → per-tenant
limits (`governance.enforce` + `usage_policies` + `quota`, most-restrictive-wins).

Two cross-cutting facts complete the picture: a **single shared state layer** — in-memory
engine singletons backed by either `*_STATE_PATH` JSON files *or* the Aurora `GovernanceStore`
— is read on every request and written after every CRUD mutation; and **`audit.py` is the
single sink** every denial and management mutation funnels into.

```mermaid
graph TD
    REQ["Incoming request (ASGI)"]

    subgraph FUNNEL["Admission funnel (ASGI -> FastAPI, outermost first)"]
        direction TB
        RID["RequestIDMiddleware (auth.py) — X-Request-ID ContextVar"]
        POL["PolicyMiddleware (policy_engine.py) — OPA eval (fail-open/closed)"]
        MGMT["ManagementMiddleware + classifier — RBAC/audit/OTel on mgmt"]
        AUTH["Two-tier auth (auth.py)<br/>admin X-Admin-API-Key (fail-closed) | user_api_key_auth"]
        OIDC["OIDC SSO (oidc.py) — /auth/token-exchange — JWT -> API key"]
        RBAC["RBAC (rbac.py) — requires_permission"]
        LIM["Per-tenant limits<br/>governance.enforce + usage_policies + quota"]
        RID --> POL --> MGMT --> AUTH --> OIDC --> RBAC --> LIM
    end

    ROUTER["LiteLLM Router (dataplane)"]

    subgraph STATE["Shared state layer (read every request / write every CRUD)"]
        direction TB
        ENG["Engine singletons (in-memory)<br/>governance | usage_policies | guardrails | prompts"]
        JSON["JSON file backing (ROUTEIQ_*_STATE_PATH)"]
        STORE["GovernanceStore (governance_store.py)<br/>Aurora — durable budget system-of-record"]
        REDIS["Redis hot counters (spend / rpm + usage/quota)"]
        ENG --- JSON
        ENG --- STORE
        ENG --- REDIS
    end

    AUDIT["audit.py — AuditLogRepository<br/>Postgres audit_logs (cross-cutting sink)"]

    REQ ==> FUNNEL
    FUNNEL ==>|"admitted"| ROUTER
    FUNNEL <-->|"resolve ctx / enforce limits"| STATE
    FUNNEL -.->|"denials + mgmt mutations"| AUDIT
    POL -.->|"403 deny"| REQ

    STATE ~~~ AUDIT
```

---

## 7. Plugin system & guardrails — GatewayPlugin lifecycle + 14 built-ins

Plugins are loaded **once at startup, ordered deterministically, then fan out into three
attach points** on the live request/response path. `PluginManager` reads `LLMROUTER_PLUGINS`,
validates against an allowlist + capability policy *before* import, topologically sorts by
`depends_on` + `priority` (Kahn's algorithm), caches the sorted order, and then partitions
that one list into three subsets: `get_middleware_plugins()` → `PluginMiddleware` (pure ASGI,
request path); `get_callback_plugins()` → `PluginCallbackBridge` (LiteLLM pre/post-call); and
all plugins carry a typed `PluginContext` whose subsystem accessors come from
`plugin_adapters.create_all_adapters()` typed by `plugin_protocols`.

```mermaid
graph TD
    CFG["LLMROUTER_PLUGINS env (allowlist + capability policy)"]

    subgraph MGR["PluginManager (plugin_manager.py)"]
        direction TB
        LOAD["load_from_config() — import + validate"]
        SORT["_topological_sort() — Kahn + priority -> _sorted_plugins"]
        START["startup(app) — per-plugin startup() w/ timeout"]
        PART["partition the one sorted list"]
        LOAD --> SORT --> START --> PART
    end

    subgraph SEAMS["3 Attach points"]
        direction TB
        MW["PluginMiddleware (pure ASGI, request path)"]
        BR["PluginCallbackBridge (litellm.callbacks pre/post)"]
        CTX["PluginContext + subsystem adapters"]
    end

    subgraph TYPED["Typed accessor seam"]
        direction TB
        ADP["plugin_adapters — create_all_adapters()"]
        PROTO["plugin_protocols (Protocols: MCP/A2A/Routing/...)"]
        ADP -.implements.-> PROTO
    end

    CFG --> MGR
    PART -->|"get_middleware_plugins()"| MW
    PART -->|"get_callback_plugins()"| BR
    PART -->|"_create_context()"| CTX
    CTX --> ADP

    MW ~~~ BR ~~~ CTX
```

**Guardrails are not a separate subsystem** — they ride the `PluginCallbackBridge`
LLM-lifecycle seam. Pre-call (`async_log_pre_api_call`): `context_optimizer` (6 lossless
transforms, -30..70% tokens) → `semantic_cache` (L1→L2→embedding, hit short-circuits) →
guardrail plugins (`pii_guard` / `prompt_injection_guard` / `content_filter` / `llamaguard` /
`bedrock_guardrails`, raising `GuardrailBlockError`) → `GuardrailPolicyEngine.evaluate_input`
(14 check types, DENY → HTTP 446). Post-call (`async_log_success_event`): output guardrails
(log/alert only), `cost_tracker` (actual cost + OTel + quota reconcile), and cache store.

```mermaid
graph TD
    REQ["HTTP request /v1/chat/completions"]

    subgraph ASGI["Request path — PluginMiddleware (ASGI)"]
        ONREQ["on_request hooks (short-circuit -> PluginResponse)"]
    end

    subgraph PRE["Pre-call — async_log_pre_api_call"]
        direction TB
        OPT["context_optimizer — 6 lossless transforms (-30..70% tokens)"]
        CACHE["semantic_cache (cache_plugin) — L1 -> L2 -> embedding lookup"]
        GUARDS["Guardrail plugins (guardrails_base)<br/>pii / prompt_injection / content_filter /<br/>llamaguard / bedrock -> GuardrailBlockError"]
        POLICY["GuardrailPolicyEngine.evaluate_input<br/>14 checks · DENY/LOG/ALERT"]
        OPT --> CACHE --> GUARDS --> POLICY
    end

    MODEL["LiteLLM Router -> model call"]

    subgraph POST["Post-call — async_log_success_event"]
        direction TB
        OUTG["output guardrails (log/alert only)"]
        COST["cost_tracker — actual cost + OTel + quota reconcile"]
        CACHESET["cache store on_llm_success"]
    end

    EXT["External seams: AWS Bedrock · LlamaGuard HTTP · Redis"]
    RESP["Response"]
    DENY["DENY — HTTP 446 (policy-engine) / 400 (plugin guard block)"]

    REQ ==> ASGI ==> PRE
    CACHE -.cache HIT short-circuit.-> RESP
    GUARDS -.block.-> DENY
    POLICY -.DENY.-> DENY
    PRE ==> MODEL ==> POST ==> RESP
    GUARDS -.invoke.-> EXT
    CACHE -.L2.-> EXT
```

---

## 8. MCP / A2A / agentic tool surface

RouteIQ keeps a **thin REST control surface** for MCP/A2A and **delegates the heavy protocol
transports to upstream LiteLLM** (ADR-0003/0017, after `mcp_jsonrpc.py` /
`mcp_sse_transport.py` / `mcp_parity.py` were deleted). RouteIQ owns a REST MCP gateway
(`/llmrouter/mcp/*`, `/v1/llmrouter/mcp/*` — server registry + tool discovery) and an
in-memory `A2AGateway`, while JSON-RPC + SSE transports and the DB-backed
`global_agent_registry` are served by LiteLLM. Two gates a reader must see: tool invocation is
**off by default** even when the gateway is enabled (needs both `MCP_GATEWAY_ENABLED` and
`LLMROUTER_ENABLE_MCP_TOOL_INVOCATION`; otherwise 501), and every outbound URL passes through
`url_security` SSRF validation **twice** — once at registration (`resolve_dns=False`) and
again at invocation (DNS resolved, **fail-closed**) to defeat DNS rebinding.

> **AWS Agent Registry relationship** (see `agentcore-registry/registry-integration.md`):
> the registry is a *discovery/governance catalog*, not a runtime tool plane. It complements
> rather than duplicates this surface. The recommended integration is **R1 (publish)** —
> register RouteIQ's MCP endpoint as a `descriptorType: MCP` record with `fromUrl` sync;
> **R2 (consume)** is blocked on an auth gap (LiteLLM 1.82.3's MCP client has no SigV4
> auth_type, so an IAM-authorizer registry needs CUSTOM_JWT or the `mcp-proxy-for-aws` shim).

```mermaid
graph TD
    CLIENT["Client / Admin (X-Admin-API-Key)"]

    subgraph REST["RouteIQ REST surface (auth + RBAC + audit)"]
        direction TB
        MCPR["routes/mcp.py — /llmrouter/mcp/* + /v1/llmrouter/mcp/*"]
        A2AR["routes/a2a.py — /a2a/agents (convenience)"]
    end

    subgraph REG["RouteIQ-native registries / engines"]
        direction TB
        MCPG["mcp_gateway.MCPGateway (singleton: servers + tool map)"]
        A2AG["a2a_gateway.A2AGateway (agents + TaskStore + JSON-RPC/SSE)"]
    end

    subgraph UPSTREAM["Upstream LiteLLM (delegated, ADR-0003/0017)"]
        direction TB
        LLJSONRPC["MCP JSON-RPC + SSE transports"]
        LLAGREG["global_agent_registry (DB-backed /v1/agents)"]
    end

    subgraph SEC["Security + observability seams"]
        direction TB
        SSRF["url_security.py — SSRF guard (fail-closed, checked TWICE)"]
        OBS["mcp_tracing + a2a_tracing — OTel spans -> infra"]
    end

    EXT["External MCP servers / A2A agents / Redis HA sync"]

    CLIENT ==> REST ==> REG
    REG --> SEC
    REG --> EXT
    REG -.-> OBS
    CLIENT -.->|"JSON-RPC / SSE NOT served by RouteIQ"| UPSTREAM
    A2AR -.->|"wraps in-memory registry"| LLAGREG

    REST ~~~ REG ~~~ UPSTREAM
```

The agentic consumer plugins ride the same `GatewayPlugin` lifecycle but consume the surface
rather than serving it: `bedrock_agentcore_mcp` registers AgentCore agents as MCP servers
(producer into the registry, SSRF-validated); `agentic_pipeline` does
DETECT→DECOMPOSE→ROUTE→EXECUTE→AGGREGATE over the routing dataplane; `skills_discovery` serves
`/.well-known/skills/*` from the filesystem with path-traversal guards.

```mermaid
graph TD
    subgraph CONSUMERS["Agentic consumer plugins (GatewayPlugin)"]
        direction TB
        BAC["bedrock_agentcore_mcp — registers AgentCore as MCP servers"]
        AGP["agentic_pipeline — on_llm_pre_call: decompose->route->aggregate"]
        SKD["skills_discovery — /.well-known/skills/*"]
    end

    MCPG["mcp_gateway.MCPGateway (server registry)"]
    SSRF["url_security.validate_outbound_url"]
    DATAPLANE["litellm.acompletion (RouteIQ ML routing dataplane)"]
    OBS["OTel spans + metrics -> infra"]
    FS["ROUTEIQ_SKILLS_DIR (filesystem)"]

    BAC -->|"context.mcp.register_server"| MCPG
    BAC -->|"context.validate_outbound_url"| SSRF
    AGP -->|"sub-query fan-out"| DATAPLANE
    AGP -.-> OBS
    BAC -.-> OBS
    SKD -->|"scan + path-traversal guard"| FS

    CONSUMERS ~~~ MCPG
```

---

## 9. Infra, state, config & observability — HA substrate (in-process)

**Leader election emits a single `is-leader` signal that gates all singleton background
work.** `leader_election.py` auto-selects a backend (K8s Lease API → Redis SETNX → Postgres
lease → None) and exposes one boolean (`MultiBackendLeaderElection.is_leader`); a daemon
renewal thread auto-demotes after 2 consecutive failures. Three things hang off that gate:
**config_sync** (the S3 ETag + AppConfig poll loop runs only on the leader; non-leaders
no-op), **migrations** (`run_migrations_if_leader` lets the leader run `prisma db push` while
followers block on an `asyncio.Event`), and any other singleton job. Around the gate sit the
shared state pools (`database.py` asyncpg with optional RDS IAM auth, `redis_pool.py` with
optional ElastiCache IAM auth, `http_client_pool.py`) and the resilience primitives wrapping
them.

```mermaid
graph TD
    subgraph DETECT["Backend detection"]
        ENV["detect_leader_election_backend()<br/>K8s host > Redis > Postgres > None"]
    end

    subgraph GATE["leader_election.py — the is-leader gate"]
        direction TB
        MBLE["MultiBackendLeaderElection<br/>is_leader + renewal thread (2 fails -> auto-demote)"]
        K8S["K8sLeaseLeaderElection (coordination.k8s.io/v1)"]
        RED["RedisLeaderElection (SET NX EX + Lua release)"]
        PG["LeaderElection (Postgres lease + fencing gen)"]
        MBLE --> K8S
        MBLE --> RED
        MBLE --> PG
    end

    subgraph GATED["Leader-gated singleton work"]
        direction TB
        SYNC["config_sync.ConfigSyncManager<br/>_sync_loop: leader-only, ETag/AppConfig diff"]
        MIG["migrations.run_migrations_if_leader<br/>leader: prisma db push / follower: wait Event"]
    end

    subgraph CFG["Config flow"]
        direction TB
        LOADER["config_loader — S3/GCS download"]
        HOT["hot_reload.HotReloadManager — force_sync -> SIGHUP"]
    end

    subgraph POOLS["Shared state pools"]
        direction TB
        DB["database.get_db_pool — asyncpg (+RDS IAM token)"]
        RPOOL["redis_pool.get_async_redis_client (+ElastiCache IAM token)"]
    end

    AWS["AWS seams: S3 / AppConfig / K8s Lease API"]

    ENV --> MBLE
    MBLE -->|"is_leader"| GATED
    SYNC -.->|"reads ETag, on change SIGHUP"| HOT
    LOADER --> SYNC
    PG --> DB
    RED --> RPOOL
    GATED --> POOLS
    AWS -.-> CFG
    AWS -.-> GATE
```

The observability column is the **telemetry sink**: `ObservabilityManager` reuses an existing
LiteLLM `TracerProvider`/`MeterProvider` if present, attaches an OTLP `BatchSpanProcessor`, and
emits structured `routing_decision` + error JSON lines (the dual/triple-key model field
satisfies both the CloudWatch per-model MetricFilter and the Glue/Athena lake column
contract). `/_health/ready` deliberately returns **200 even when circuit breakers are open**
(degraded ≠ unready).

```mermaid
graph TD
    subgraph EMIT["Telemetry emitters (other sections)"]
        direction TB
        RDC["router_decision_callback"]
        MCPA["mcp_tracing / a2a_tracing"]
        AUD["audit"]
    end

    subgraph RESIL["resilience.py"]
        direction TB
        BPM["BackpressureMiddleware — ASGI, 503 over-capacity, streaming-safe"]
        DRAIN["DrainManager — active reqs + is_draining"]
        CBM["CircuitBreakerManager — db/redis/leader/provider"]
        SCBS["SharedCircuitBreakerState (cross-worker via Redis)"]
        CBM --> SCBS
    end

    subgraph OBS["Observability column"]
        direction TB
        OBSMGR["observability.ObservabilityManager<br/>REUSES LiteLLM TracerProvider"]
        MET["metrics.GatewayMetrics — gen_ai.* + gateway.* instruments"]
        TC["telemetry_contracts — RouterDecisionEvent v1 + GenAIAttributes"]
        OBSMGR --> MET
        OBSMGR -.->|"schema"| TC
    end

    subgraph SINKS["AWS observability substrate"]
        direction TB
        OTLP["OTLP collector / ADOT"]
        CW["CloudWatch metric filters + RouterErrorFilter"]
        LAKE["Firehose -> Glue/Athena (selected_model column)"]
    end

    HEALTH["/_health/ready — 200 even when breakers OPEN"]

    EMIT --> OBS
    RESIL -.->|"CB state-change OTel events"| OBS
    CBM -.->|"degraded status (advisory)"| HEALTH
    OBSMGR -->|"BatchSpanProcessor"| OTLP
    OBSMGR -->|"routing_decision + error JSON lines"| SINKS
    OTLP ~~~ CW ~~~ LAKE
```

---

## 10. AWS substrate — 3 CDK stacks (P0 foundation / P1 state / P2 observability)

Three CDK `Stack`s deploy as three independent blast radii, **wired together by reference,
never by `from_lookup`.** `app.py:main()` builds them in dependency order and threads the
*Python object* of the P0 stack (`foundation=foundation`) into the other two; because all
three live in one `cdk.App`, CDK resolves those cross-stack reads at **synth** time into
auto-generated `Export`/`Fn::ImportValue` pairs — cred-free, enabling the offline cdk-nag
gate. Two things thread through all three stacks: the **single pod IAM role**
(`RouteIqStack.pod_role`, bound via `CfnPodIdentityAssociation` — Pod Identity, not IRSA) onto
which P1 and P2 each `attach_to_role` a stack-local `iam.Policy` (cycle-safe — they never
mutate the *imported* role via `add_to_principal_policy`; P0 populates its own role with
`add_to_policy`, which is correct in-stack); and the **KMS boundary** — two customer CMKs, one per state-bearing
stack so each rolls back with its own key.

```mermaid
graph TD
    APP["app.py main()<br/>cdk.App + AwsSolutionsChecks"]

    subgraph P0["P0 RouteIqStack (foundation)"]
        direction TB
        NET["NetworkConstruct — VPC / 3 SGs / 5 VPCE + S3 GW"]
        EKS["EksClusterConstruct — EKS Auto Mode + routing_log_group<br/>+ CfnPodIdentityAssociation"]
        ECR["EcrConstruct — GHCR pull-through cache"]
        POD["pod_role (single IAM identity) — Bedrock / Secrets / S3 / Logs"]
        NAG0["nag_suppressions (cdk-nag evidence)"]
    end

    subgraph P1["P1 RouteIqStateStack (durable state)"]
        direction TB
        KMS1["StateCmk (KMS)"]
        AUR["ReplayStoreConstruct — Aurora PG Serverless v2 (backs database.py)"]
        CACHE["CacheConstruct — ElastiCache Valkey serverless (backs redis_pool)"]
        PSG["PodStateGrants iam.Policy — rds-db:connect + elasticache:Connect"]
        NAG1["state_nag_suppressions"]
    end

    subgraph P2["P2 RouteIqObservabilityStack (obs + lake)"]
        direction TB
        CFG["ConfigStateConstruct — AppConfig (backs config_sync)"]
        OBS["ObservabilityConstruct — AMP / AMG / CW filters+alarms<br/>(sink for observability.py)"]
        LAKE["DataLakeConstruct + Athena WG — LakeKey CMK / Firehose / Glue"]
        POG["PodObsGrants iam.Policy — AppConfig poll + aps:RemoteWrite"]
        NAG2["obs_nag_suppressions"]
    end

    APP ==>|"instantiate, dep order"| P0
    APP ==>|"foundation=foundation"| P1
    APP ==>|"foundation=foundation"| P2

    P0 -.->|"by-ref Export / Fn::ImportValue (cred-free)"| P1
    P0 -.->|"by-ref ILogGroup + add_dependency"| P2
    POD -.->|"imported, attach_to_role"| PSG
    POD -.->|"imported, attach_to_role"| POG

    NET ~~~ EKS ~~~ ECR
    KMS1 ~~~ AUR ~~~ CACHE
    CFG ~~~ OBS ~~~ LAKE
```

---

## 11. Where to read the code

| Section | Key modules (under `src/litellm_llmrouter/` unless noted) |
|---------|-----------------------------------------------------------|
| **Boot / composition root** | `cli.py`, `startup.py`, `gateway/app.py` (`create_gateway_app`, `_routeiq_lifespan`, `_configure_middleware`), `settings.py`, `service_discovery.py`, `env_validation.py` |
| **Routing dataplane** | `custom_routing_strategy.py` (`RouteIQRoutingStrategy`), `strategy_registry.py` (`RoutingPipeline`, `RoutingStrategyRegistry`), `strategies.py` (`LLMRouterStrategyFamily`), `kumaraswamy_thompson.py`, `centroid_routing.py`, `personalized_routing.py`, `router_r1.py`, `conversation_affinity.py`, `router_decision_callback.py` |
| **Adapter ABI + MLOps + eval loop** | `adapters/contract.py` (`RoutingAdapter`, `AdapterManifest`, `attach_route_alias`, `_abi_compatible`), `adapters/loader.py` (`AdapterLoaderPlugin`), `adapters/mlops.py` (`MLOpsCoordinator`, `wire_mlops_feedback_loop`), `eval_pipeline.py` (`EvalPipeline`, `EvalJudge`, `ModelQualityTracker`), `model_artifacts.py` (`ModelArtifactVerifier`) |
| **Control plane** | `policy_engine.py` (`PolicyMiddleware`), `management_middleware.py` / `management_classifier.py`, `auth.py` (`RequestIDMiddleware`, `admin_api_key_auth`), `oidc.py`, `rbac.py`, `governance.py` (`GovernanceEngine`), `governance_store.py`, `usage_policies.py`, `quota.py`, `prompt_management.py`, `audit.py` |
| **Plugin system & guardrails** | `gateway/plugin_manager.py` (`PluginManager`, `GatewayPlugin`), `gateway/plugin_middleware.py`, `gateway/plugin_callback_bridge.py`, `gateway/plugin_adapters.py`, `gateway/plugin_protocols.py`, `guardrail_policies.py` (`GuardrailPolicyEngine`), `semantic_cache.py`, `gateway/plugins/{guardrails_base,pii_guard,prompt_injection_guard,content_filter,llamaguard_plugin,bedrock_guardrails,context_optimizer,cache_plugin,cost_tracker}.py` |
| **MCP / A2A / agentic** | `mcp_gateway.py` (`MCPGateway`), `a2a_gateway.py` (`A2AGateway`), `routes/mcp.py`, `routes/a2a.py`, `url_security.py`, `mcp_tracing.py`, `a2a_tracing.py`, `gateway/plugins/{bedrock_agentcore_mcp,agentic_pipeline,skills_discovery}.py` |
| **Infra / state / config / observability** | `leader_election.py` (`MultiBackendLeaderElection`), `config_loader.py`, `config_sync.py` (`ConfigSyncManager`), `hot_reload.py`, `migrations.py`, `database.py`, `redis_pool.py`, `http_client_pool.py`, `resilience.py` (`BackpressureMiddleware`, `DrainManager`, `CircuitBreakerManager`), `observability.py`, `metrics.py`, `telemetry_contracts.py` |
| **AWS substrate** | `deploy/cdk/app.py`, `deploy/cdk/lib/{routeiq_stack,network_construct,eks_cluster_construct,ecr_construct}.py` (P0); `deploy/cdk/lib/{routeiq_state_stack,replay_store_construct,cache_construct}.py` (P1); `deploy/cdk/lib/{routeiq_observability_stack,config_state_construct,observability_construct,data_lake_construct}.py` (P2); `deploy/cdk/lib/{naming,nag_suppressions}.py` |

---

## Related docs

- **Project guide & non-obvious behaviors:** `CLAUDE.md` (repo root)
- **AWS re-architecture set:** `docs/architecture/aws-rearchitecture/` —
  `10-aws-native-target-architecture.md`, `20-kumaraswamy-thompson-router.md`,
  `40-pluggable-routing-and-mlops.md`, `50-litellm-universal-surface.md`,
  `31-p0-cdk-foundation-proposal.md`
- **AgentCore integration:** `docs/architecture/agentcore-integration-and-arch-2026-06-15.md`
  and `research/agentcore-registry/registry-integration.md`
- **ADRs:** `docs/adr/` (25 decisions; ADR-0002 plugin routing, ADR-0012 own-app,
  ADR-0017 MCP delegation, ADR-0028/0029 IAM auth)
