# TG3: Alternative Gateway Architecture Patterns

> **Version**: 1.0  
> **Date**: 2026-02-18  
> **Status**: Proposal — awaiting review  
> **Scope**: Evaluate five architecture options for RouteIQ v1.0, recommend a path forward

---

## Executive Summary

RouteIQ v0.2.0 is a 36,207-line monolithic Python gateway that extends LiteLLM via
class-level monkey-patching of three Router methods. This approach forces a
single-worker constraint (`workers=1` hardcoded in
[`startup.py`](../../src/litellm_llmrouter/startup.py)), creates tight version coupling
with upstream LiteLLM, and duplicates ~7,578 lines of MCP/A2A functionality that
LiteLLM already provides natively.

A critical discovery changes the calculus: LiteLLM ships
[`CustomRoutingStrategyBase`](../../reference/litellm/litellm/types/router.py:671) and
[`Router.set_custom_routing_strategy()`](../../reference/litellm/litellm/router.py:8782)
as its **official** extension point. These replace the exact same `get_available_deployment`
and `async_get_available_deployment` methods that RouteIQ monkey-patches, but act on a
**per-instance** basis instead of class-wide — eliminating the single-worker mandate
entirely. Additionally, LiteLLM's `AutoRouter` provides
[`async_pre_routing_hook`](../../reference/litellm/litellm/router.py:8326) for
pre-classification.

This document evaluates five architecture options — LiteLLM Plugin, Sidecar, Microservices,
Fork & Own, and Hybrid — and recommends **Option A (LiteLLM Plugin/Extension)** as the
primary path, with **Option E (Hybrid)** as a scaling evolution. Option A eliminates the
monkey-patch, enables multi-worker operation, sheds ~7,578 lines of redundant code, and
can be packaged as a standalone `pip install routeiq-router` library in ~5,000 lines.

---

## Current Architecture Problems

These are the blockers identified in TG1 and TG2 that any architecture must resolve:

| # | Problem | Impact |
|---|---------|--------|
| 1 | **Monkey-patching 3 methods** on `litellm.router.Router` ([`routing_strategy_patch.py:476-488`](../../src/litellm_llmrouter/routing_strategy_patch.py:476)) | Mandates `workers=1`; class-level mutation prevents multi-process scaling |
| 2 | **16 module-level singletons**, only 5 of 21 reset in test conftest | 76% leak between tests; non-deterministic test failures |
| 3 | **~7,578 lines redundant** with upstream LiteLLM | MCP gateway (5,958 lines) + A2A gateway (1,620 lines) re-implemented |
| 4 | **No admin UI** despite LiteLLM shipping a full Next.js dashboard | Users get no visual management interface |
| 5 | **124 env vars, only 54 documented** | 70 undocumented env vars create operational risk |
| 6 | **Goal alignment score of 3.05/5** | Rated NOT production-ready by independent assessment |

### What RouteIQ Genuinely Adds (~14,000 lines, 39% of codebase)

- **ML routing**: 18 strategies across KNN, SVM, MLP, MF, ELO, hybrid, etc. + A/B testing pipeline (~5,500 lines across [`strategies.py`](../../src/litellm_llmrouter/strategies.py), [`strategy_registry.py`](../../src/litellm_llmrouter/strategy_registry.py), [`routing_strategy_patch.py`](../../src/litellm_llmrouter/routing_strategy_patch.py), [`router_decision_callback.py`](../../src/litellm_llmrouter/router_decision_callback.py))
- **Security**: SSRF dual-validation, OPA-style policy engine (~3,500 lines)
- **Resilience**: Backpressure middleware, drain manager, circuit breakers (~2,000 lines)
- **Observability**: Versioned telemetry contracts, decision spans (~1,500 lines)
- **Model artifacts**: Hash + signature verification for ML models (~1,484 lines)
- **Plugin system**: 11 capabilities, 12 lifecycle hooks, dependency resolution (~1,242 lines)

---

## Option A: LiteLLM Plugin/Extension ⭐ RECOMMENDED

### Description

Instead of monkey-patching LiteLLM's Router class, RouteIQ becomes a **proper LiteLLM
extension** using the official `CustomRoutingStrategyBase` interface. The core deliverable
is a standalone Python package (`routeiq-router`) of approximately 5,000 lines that wraps
all 18 ML routing strategies behind LiteLLM's `set_custom_routing_strategy()` API.

The implementation creates a `RouteIQRoutingStrategy(CustomRoutingStrategyBase)` class
that delegates to the existing [`LLMRouterStrategyFamily`](../../src/litellm_llmrouter/strategies.py)
and [`RoutingPipeline`](../../src/litellm_llmrouter/strategy_registry.py) for A/B testing.
Because `set_custom_routing_strategy()` works per-instance (replacing instance methods via
`setattr` at [`router.py:8794-8803`](../../reference/litellm/litellm/router.py:8794)),
there is no class-level mutation — enabling multi-worker uvicorn.

The redundant MCP gateway (5,958 lines across [`mcp_gateway.py`](../../src/litellm_llmrouter/mcp_gateway.py),
[`mcp_jsonrpc.py`](../../src/litellm_llmrouter/mcp_jsonrpc.py),
[`mcp_sse_transport.py`](../../src/litellm_llmrouter/mcp_sse_transport.py),
[`mcp_parity.py`](../../src/litellm_llmrouter/mcp_parity.py),
[`mcp_tracing.py`](../../src/litellm_llmrouter/mcp_tracing.py)) and A2A gateway (1,620
lines in [`a2a_gateway.py`](../../src/litellm_llmrouter/a2a_gateway.py),
[`a2a_tracing.py`](../../src/litellm_llmrouter/a2a_tracing.py)) are shed — LiteLLM
handles these natively. RouteIQ keeps only its genuine differentiators: ML routing,
policy engine, resilience, observability, and model artifacts.

### Architecture Diagram

```
                    +----------------------------------------------+
                    |            LiteLLM Proxy (upstream)           |
                    |                                              |
                    |  +----------+  +--------+  +----------+     |
                    |  | Auth/RBAC|  | MCP GW |  | A2A GW   |     |
                    |  +----------+  +--------+  +----------+     |
                    |                                              |
                    |  +------------------------------------------+|
                    |  |             litellm.Router                ||
                    |  |                                          ||
                    |  |  router.set_custom_routing_strategy(     ||
                    |  |      RouteIQRoutingStrategy()            ||
                    |  |  )                                      ||
                    |  |                                          ||
                    |  |  +------------------------------------+ ||
                    |  |  | RouteIQRoutingStrategy              | ||
                    |  |  | (CustomRoutingStrategyBase)         | ||
                    |  |  |                                    | ||
                    |  |  |  - 18 ML strategies (KNN,SVM,etc) | ||
                    |  |  |  - A/B testing pipeline            | ||
                    |  |  |  - Routing telemetry               | ||
                    |  |  |  - Model artifact verification     | ||
                    |  |  +------------------------------------+ ||
                    |  +------------------------------------------+|
                    |                                              |
                    |  +------------------------------------------+|
                    |  | CustomLogger callbacks                   ||
                    |  |  - Policy engine (pre-call)              ||
                    |  |  - Resilience (backpressure, circuit)    ||
                    |  |  - Observability (decision spans)        ||
                    |  +------------------------------------------+|
                    +----------------------------------------------+
                                        |
                        +---------------+---------------+
                        |               |               |
                   OpenAI API     Anthropic API    Bedrock API
```

### Evaluation Matrix

| Criterion | Assessment |
|-----------|-----------|
| **Development effort** | 6-8 person-weeks |
| **Scaling** | Multi-worker uvicorn (N workers); horizontal scaling via LiteLLM HA |
| **Maintenance burden** | Low — consume upstream releases via pip; no fork to maintain |
| **Breaking change risk** | Medium — config migration from `llmrouter-*` strategies to `routing_strategy: custom`; env vars unchanged |
| **Self-hosting ease** | High — `pip install litellm routeiq-router`, single docker-compose, identical to current |
| **Cloud-native fit** | Excellent — K8s HPA on replicas, works with LiteLLM Helm chart |

### Migration Path

1. Implement `RouteIQRoutingStrategy(CustomRoutingStrategyBase)` wrapping existing strategy logic
2. Implement `RouteIQCallbackLogger(CustomLogger)` for policy engine, telemetry, and resilience hooks
3. Wire into LiteLLM via `router.set_custom_routing_strategy()` and `litellm.callbacks`
4. Validate multi-worker support with `workers=4` in integration tests
5. Shed MCP/A2A modules (delete ~7,578 lines)
6. Package as `routeiq-router` on PyPI
7. Update config.yaml to use `routing_strategy: custom` instead of `llmrouter-knn`

### Pros & Cons

**Pros:**
- Eliminates monkey-patching entirely — uses official LiteLLM extension point
- Enables multi-worker uvicorn (4-8x throughput improvement)
- Reduces codebase from ~36,000 to ~5,000 lines (86% reduction)
- Inherits all upstream LiteLLM improvements automatically (MCP, A2A, UI, auth, etc.)
- Minimal deployment complexity — single process, single container
- Community can use RouteIQ routing with any LiteLLM deployment
- Lowest maintenance burden of all options

**Cons:**
- Coupled to LiteLLM's `CustomRoutingStrategyBase` interface (stable but limited surface)
- `set_custom_routing_strategy` replaces `get_available_deployment` entirely — does not compose with LiteLLM's built-in strategies for the same router instance
- No control over LiteLLM's pre-call checks, cooldown logic, or fallback behavior
- `async_pre_routing_hook` is only available via `AutoRouter` — not via `CustomRoutingStrategyBase`
- Some RouteIQ middleware (backpressure, drain) must be implemented as ASGI middleware, not via `CustomLogger`

---

## Option B: Sidecar Architecture

### Description

RouteIQ runs as a lightweight **sidecar process** alongside LiteLLM. The client sends
requests to RouteIQ, which makes a routing decision (model selection, A/B test assignment,
policy evaluation) and then forwards the request to LiteLLM for actual provider proxying.
RouteIQ becomes a "routing brain" that never touches LLM provider APIs directly.

This cleanly separates concerns: RouteIQ handles only what it is genuinely good at
(ML-based routing, policy enforcement, A/B testing), while LiteLLM handles everything
else (provider proxy, auth, rate limiting, MCP, A2A, caching, fallbacks). There is no
monkey-patching, no code sharing, and no version coupling beyond the OpenAI-compatible
API contract between the two services.

The sidecar pattern is well-established in service meshes (Envoy, Linkerd) and maps
naturally to container orchestration — RouteIQ runs in the same pod as LiteLLM in K8s,
with localhost communication eliminating network latency.

### Architecture Diagram

```
    Client Request
          |
          v
    +------------------+
    |   RouteIQ Sidecar |    (Port 4001)
    |                  |
    |  - ML routing    |
    |  - A/B testing   |
    |  - Policy engine |
    |  - Observability |
    |                  |
    |  Decision:       |
    |  model=claude-3  |
    |  deployment=az-2 |
    +--------+---------+
             |
             | X-RouteIQ-Model: claude-3-opus
             | X-RouteIQ-Deployment: az-east-2
             v
    +------------------+
    |   LiteLLM Proxy  |    (Port 4000)
    |                  |
    |  - Provider proxy|
    |  - Auth/RBAC     |
    |  - Rate limiting |
    |  - Caching       |
    |  - MCP/A2A       |
    |  - Fallbacks     |
    |  - Admin UI      |
    +--------+---------+
             |
        +----+----+
        |    |    |
      OpenAI  Anthropic  Bedrock
```

### Evaluation Matrix

| Criterion | Assessment |
|-----------|-----------|
| **Development effort** | 8-10 person-weeks |
| **Scaling** | Independent scaling — RouteIQ N replicas, LiteLLM M replicas |
| **Maintenance burden** | Low — only API contract coupling; no code-level dependency |
| **Breaking change risk** | High — clients must point to new endpoint; proxy chain adds latency |
| **Self-hosting ease** | Medium — two containers instead of one; more docker-compose complexity |
| **Cloud-native fit** | Good — sidecar pattern native to K8s; but adds pod resource overhead |

### Migration Path

1. Extract routing logic into standalone FastAPI service
2. Implement routing decision API (`POST /route` → returns model + deployment)
3. Configure LiteLLM to accept routing hints via headers or metadata
4. Deploy both services in same pod (K8s) or docker-compose network
5. Update client SDKs to point to RouteIQ endpoint

### Pros & Cons

**Pros:**
- Complete decoupling from LiteLLM internals — zero version coupling
- Independent scaling of routing and proxy layers
- Can be replaced with any routing service (not LiteLLM-specific)
- Multi-language support — RouteIQ sidecar could be rewritten in Go/Rust for performance
- Clean failure isolation — RouteIQ crash does not take down LiteLLM

**Cons:**
- Additional network hop adds 1-5ms latency per request
- Two containers to manage, monitor, and debug
- Routing decisions cannot access LiteLLM's internal state (cooldowns, deployment health)
- Requires LiteLLM to accept and trust routing hints from the sidecar
- Docker-compose setup more complex for self-hosters
- Harder to implement tight integration features (e.g., post-call routing feedback)

---

## Option C: Microservices Split

### Description

RouteIQ decomposes into four independent services, each owning a specific domain.
`routeiq-proxy` is a thin deployment wrapper around LiteLLM that handles all provider
communication. `routeiq-router` is a stateless ML routing service exposing a gRPC/REST
API for model selection. `routeiq-control-plane` manages configuration, RBAC, policies,
quotas, and serves an admin API. `routeiq-ui` is a web dashboard (potentially extending
LiteLLM's existing Next.js dashboard).

This is the "textbook" cloud-native approach: each service scales independently, can be
deployed and versioned separately, and failures are isolated. However, it introduces
significant operational complexity — four deployment targets, inter-service communication,
distributed tracing, and eventual consistency challenges.

### Architecture Diagram

```
                          +-------------------+
                          |   routeiq-ui      |
                          |   (React/Next.js) |
                          +--------+----------+
                                   |
                          +--------v----------+
                          | routeiq-control   |
    +------------+        |   -plane          |        +---------------+
    |            |        |                   |        |               |
    |  Client    +------->+ - Admin API       +------->+ routeiq-router|
    |  Request   |        | - Config mgmt     |        |               |
    |            |        | - RBAC / Policies  |        | - 18 ML strats|
    +-----+------+        | - Quota tracking  |        | - A/B testing |
          |               +-------------------+        | - gRPC + REST |
          |                                            | - Stateless   |
          |               +-------------------+        +-------+-------+
          +-------------->+ routeiq-proxy     |                |
                          |                   |<---------------+
                          | - LiteLLM wrapper |  routing decision
                          | - Provider proxy  |
                          | - MCP / A2A       |
                          | - Caching         |
                          +-------------------+
```

### Evaluation Matrix

| Criterion | Assessment |
|-----------|-----------|
| **Development effort** | 16-24 person-weeks |
| **Scaling** | Excellent — each service scales independently; router is stateless |
| **Maintenance burden** | High — 4 repos/services; inter-service API contracts to maintain |
| **Breaking change risk** | Very High — complete architecture overhaul; all users affected |
| **Self-hosting ease** | Low — 4+ containers, service discovery, inter-service auth |
| **Cloud-native fit** | Excellent — textbook microservices; native K8s, ECS, Fargate fit |

### Migration Path

1. Define service boundaries and API contracts (gRPC protobufs, OpenAPI specs)
2. Extract routing engine as standalone gRPC service
3. Extract control plane (admin API, config, RBAC) as standalone service
4. Wrap LiteLLM as thin proxy service with routing client
5. Build or extend admin UI
6. Implement service mesh (Envoy/Istio) for inter-service communication
7. Migrate existing users with compatibility shim

### Pros & Cons

**Pros:**
- Best possible scaling characteristics — each service sized independently
- Clean domain boundaries enable focused team ownership
- Router service can serve multiple proxy backends (not just LiteLLM)
- Individual service deployments — update router without touching proxy
- Technology diversity possible (e.g., Rust router for p99 latency)

**Cons:**
- Extremely high development and operational overhead for the current team size
- Distributed systems complexity: consensus, partial failures, eventual consistency
- Self-hosting difficulty increases dramatically
- 4x the CI/CD pipelines, monitoring dashboards, and on-call surfaces
- Premature for current scale — complexity cost exceeds scaling benefit
- Docker-compose for self-hosters becomes unwieldy (4+ services + Redis + Postgres)

---

## Option D: Fork & Own

### Description

Fork LiteLLM entirely, integrate RouteIQ's ML routing natively into the LiteLLM codebase,
and maintain the combined project as a single product. This gives full control over the
proxy, routing, and all extension points — no monkey-patching needed because RouteIQ IS
the proxy.

This is the highest-control option but also the highest-maintenance one. LiteLLM's
upstream repository averages 50-100 commits per week. Tracking upstream changes,
resolving merge conflicts, and validating that LiteLLM's fast-moving features (new
providers, API endpoints, UI updates) do not break RouteIQ's modifications would become
the dominant engineering cost. LiteLLM is 300,000+ lines of Python — the maintenance
burden is proportional.

### Architecture Diagram

```
    +----------------------------------------------------+
    |         RouteIQ (Forked LiteLLM)                   |
    |                                                    |
    |  +--------------------+  +---------------------+  |
    |  | LiteLLM Core       |  | RouteIQ Extensions  |  |
    |  | (300K+ lines)      |  | (14K lines)         |  |
    |  |                    |  |                     |  |
    |  | - Provider proxy   |  | - ML routing (18)   |  |
    |  | - Auth / RBAC      |  | - A/B testing       |  |
    |  | - Rate limiting    |  | - Policy engine     |  |
    |  | - MCP / A2A        |  | - Resilience        |  |
    |  | - Admin UI         |  | - Observability     |  |
    |  | - Caching          |  | - Model artifacts   |  |
    |  | - Fallbacks        |  | - Plugin system     |  |
    |  +--------------------+  +---------------------+  |
    |                                                    |
    |  router.routing_strategy_init() supports           |
    |  llmrouter-* strategies natively (no patch)        |
    +----------------------------------------------------+
```

### Evaluation Matrix

| Criterion | Assessment |
|-----------|-----------|
| **Development effort** | 4-6 person-weeks (initial); 8-12 person-weeks/quarter (ongoing merge) |
| **Scaling** | Multi-worker possible (native integration, no monkey-patch) |
| **Maintenance burden** | Very High — must track 50-100 upstream commits/week |
| **Breaking change risk** | Medium-High — fork divergence creates upgrade friction |
| **Self-hosting ease** | High — single binary, identical to current LiteLLM deployment |
| **Cloud-native fit** | Good — same as upstream LiteLLM |

### Migration Path

1. Fork LiteLLM repository
2. Integrate RouteIQ routing strategies into `routing_strategy_init()`
3. Add RouteIQ config sections to LiteLLM's config schema
4. Extend `RoutingStrategy` enum with `llmrouter-*` values
5. Run full LiteLLM test suite + RouteIQ tests
6. Establish weekly upstream sync cadence
7. Publish as `routeiq-gateway` (replacing `litellm` package)

### Pros & Cons

**Pros:**
- Full control — can modify anything in the stack
- No monkey-patching — native integration
- Single deployment artifact — simplest operational model
- Can contribute improvements upstream
- Multi-worker support trivially enabled

**Cons:**
- Crushing maintenance burden — LiteLLM moves fast (50-100 commits/week)
- Fork divergence accelerates over time — merge conflicts become exponential
- Must re-validate all LiteLLM features on every sync
- Community perception: "yet another LiteLLM fork"
- Cannot benefit from LiteLLM's hosted/managed offering
- Team becomes responsible for LiteLLM bugs in their fork

---

## Option E: Hybrid (Plugin + External Service)

### Description

This option combines the simplicity of Option A with the scaling flexibility of Option B.
LiteLLM runs unmodified with a thin `CustomRoutingStrategyBase` implementation that calls
out to an external RouteIQ routing service for complex decisions. For simple routing
(random, round-robin), the in-process strategy handles it directly. For ML-based routing
(KNN, SVM, MLP, etc.), the strategy makes an async HTTP call to the RouteIQ service.

The external RouteIQ service is a focused, stateless ML inference service running the 18
routing strategies, A/B testing pipeline, and model artifact management. It can be
deployed as a sidecar (same pod) for low latency, or as a shared service for
multi-tenant routing. The MLOps pipeline pushes trained models directly to this service.

This is the natural evolution of Option A when scaling demands exceed what a single
process can handle for routing computation (e.g., large embedding models for KNN routing).

### Architecture Diagram

```
    Client Request
          |
          v
    +---------------------------------------------------+
    |            LiteLLM Proxy (unmodified)              |
    |                                                    |
    |  +----------------------------------------------+ |
    |  |  litellm.Router                              | |
    |  |                                              | |
    |  |  CustomRoutingStrategy:                      | |
    |  |  +----------------------------------------+  | |
    |  |  | RouteIQHybridStrategy                  |  | |
    |  |  |                                        |  | |
    |  |  |  if simple_strategy:                   |  | |
    |  |  |    return local_route()                |  | |
    |  |  |  else:                                 |  | |
    |  |  |    resp = http.post(routeiq_svc/route) |  | |
    |  |  |    return resp.deployment              |  | |
    |  |  +----------------------------------------+  | |
    |  +----------------------------------------------+ |
    +------------------------+---------------------------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+         +---------v---------+
    | RouteIQ Router Svc|         |   LLM Providers   |
    | (optional sidecar)|         |                   |
    |                   |         | OpenAI, Anthropic, |
    | - 18 ML strategies|         | Bedrock, etc.     |
    | - A/B testing     |         +-------------------+
    | - Model artifacts |
    | - MLOps pipeline  |
    +-------------------+
```

### Evaluation Matrix

| Criterion | Assessment |
|-----------|-----------|
| **Development effort** | 10-14 person-weeks |
| **Scaling** | Excellent — routing service scales independently; LiteLLM scales independently |
| **Maintenance burden** | Low-Medium — thin plugin + focused routing service |
| **Breaking change risk** | Medium — same migration as Option A, plus optional external service |
| **Self-hosting ease** | Medium-High — works as single process (Option A mode) OR with external service |
| **Cloud-native fit** | Excellent — routing service is stateless, horizontally scalable, Lambda-compatible |

### Migration Path

Phase 1 (identical to Option A):
1. Implement `RouteIQRoutingStrategy(CustomRoutingStrategyBase)` with local routing
2. Shed MCP/A2A code, validate multi-worker
3. Ship as `routeiq-router` v1.0

Phase 2 (when scaling demands it):
4. Extract ML inference into standalone routing service
5. Add HTTP client to `RouteIQRoutingStrategy` for remote routing calls
6. Implement routing service with gRPC/REST API
7. Deploy routing service as sidecar or shared service

### Pros & Cons

**Pros:**
- Preserves all Option A benefits as the starting point
- Graceful scaling path — start in-process, extract when needed
- Routing service can serve multiple LiteLLM clusters
- ML model loading isolated from proxy process (memory, CPU)
- Routing service is Lambda/Fargate-deployable for serverless
- MLOps pipeline deploys directly to routing service without proxy restart

**Cons:**
- Full hybrid requires more development effort than Option A alone
- External service adds operational complexity when deployed
- Routing latency increases with network hop (mitigated by sidecar)
- Two deployment configurations to document and support
- More complex debugging when routing fault is in external service

---

## Comparative Analysis

| Criterion | A: Plugin ⭐ | B: Sidecar | C: Microservices | D: Fork | E: Hybrid |
|-----------|:---:|:---:|:---:|:---:|:---:|
| **Development effort** | 6-8 pw | 8-10 pw | 16-24 pw | 4-6 pw + ongoing | 10-14 pw |
| **Multi-worker support** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Lines of code** | ~5,000 | ~6,000 | ~12,000 (4 svcs) | ~314,000 (fork) | ~7,000 |
| **Version coupling** | Low (API) | None | None | Very High (fork) | Low (API) |
| **Maintenance burden** | Low | Low | High | Very High | Low-Medium |
| **Breaking change risk** | Medium | High | Very High | Medium-High | Medium |
| **Self-hosting ease** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★★★ | ★★★★☆ |
| **Cloud-native fit** | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★★★ |
| **Horizontal scaling** | Per-worker | Per-service | Per-service | Per-worker | Per-service |
| **Upstream LiteLLM compat** | Full | Full | Full | Diverging | Full |
| **Admin UI** | LiteLLM native | LiteLLM native | Custom required | Fork of LiteLLM | LiteLLM native |
| **ML model isolation** | In-process | Separate process | Separate service | In-process | Optional |
| **Existing test reuse** | ~80% | ~40% | ~30% | ~90% | ~75% |

### Scoring (weighted 1-5 scale, higher is better)

| Criterion (weight) | A: Plugin | B: Sidecar | C: Micro | D: Fork | E: Hybrid |
|---------------------|:---------:|:----------:|:--------:|:-------:|:---------:|
| Development speed (25%) | 5 | 3 | 1 | 4 | 3 |
| Maintenance burden (25%) | 5 | 4 | 2 | 1 | 4 |
| Scaling capability (15%) | 3 | 4 | 5 | 3 | 5 |
| Self-hosting ease (15%) | 5 | 3 | 1 | 5 | 4 |
| Breaking change risk (10%) | 4 | 2 | 1 | 3 | 4 |
| Cloud-native fit (10%) | 4 | 4 | 5 | 3 | 5 |
| **Weighted Total** | **4.45** | **3.35** | **2.10** | **2.95** | **3.95** |

---

## Recommendation

### Primary: Option A — LiteLLM Plugin/Extension

Option A scores highest (4.45/5) because it delivers the most value with the least effort
and ongoing cost. The core arguments:

1. **Eliminates the #1 blocker**: Monkey-patching goes away entirely. The
   `CustomRoutingStrategyBase` at [`router.py:671`](../../reference/litellm/litellm/types/router.py:671)
   is LiteLLM's official, stable extension point — it has existed since at least litellm
   1.50.0 and maps exactly to RouteIQ's needs.

2. **86% code reduction**: From ~36,000 lines to ~5,000 lines. The shed code (MCP, A2A,
   redundant auth) is already implemented and maintained by LiteLLM upstream.

3. **Multi-worker unlocked**: With per-instance method replacement instead of class-level
   patching, `workers=4` (or more) becomes trivially possible, providing 4-8x throughput.

4. **Lowest TCO**: A pip-installable library has near-zero operational overhead. No
   additional containers, no inter-service communication, no distributed debugging.

5. **Community leverage**: Users of vanilla LiteLLM can add ML routing via
   `pip install routeiq-router` without switching their entire gateway.

### Evolution Path: Option A → Option E

When scaling demands exceed in-process capacity (e.g., KNN embedding models consuming
excessive memory, or routing decision latency exceeding SLOs), the team should evolve
to Option E by extracting the ML inference into a standalone service. This is a natural,
non-breaking evolution — the `CustomRoutingStrategyBase` interface remains identical; only
the internal implementation changes from local computation to an HTTP call.

### Phased Migration Plan

**Phase 1: Plugin Foundation (weeks 1-4)**
- Implement `RouteIQRoutingStrategy(CustomRoutingStrategyBase)`
- Implement `RouteIQPolicyCallback(CustomLogger)` for policy engine hooks
- Wire A/B testing through `strategy_registry` behind the new interface
- Validate with `workers=4` integration tests
- Package as `routeiq-router` v0.1.0

**Phase 2: Code Shed (weeks 5-6)**
- Delete MCP gateway modules (~5,958 lines)
- Delete A2A gateway modules (~1,620 lines)
- Remove [`routing_strategy_patch.py`](../../src/litellm_llmrouter/routing_strategy_patch.py) (546 lines)
- Remove [`startup.py`](../../src/litellm_llmrouter/startup.py) in-process uvicorn launcher
- Update all tests to use `CustomRoutingStrategyBase` path

**Phase 3: Polish & Release (weeks 7-8)**
- Migration guide for existing users
- Updated docker-compose examples
- Helm chart updates
- PyPI release as `routeiq-router` v1.0.0
- Updated documentation

---

## Appendix: LiteLLM Extension Points Reference

### CustomRoutingStrategyBase

**Location**: [`reference/litellm/litellm/types/router.py:671-718`](../../reference/litellm/litellm/types/router.py:671)

```python
class CustomRoutingStrategyBase:
    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """Returns an element from litellm.router.model_list"""
        pass

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """Sync version of async_get_available_deployment"""
        pass
```

### Router.set_custom_routing_strategy()

**Location**: [`reference/litellm/litellm/router.py:8782-8803`](../../reference/litellm/litellm/router.py:8782)

```python
def set_custom_routing_strategy(
    self, CustomRoutingStrategy: CustomRoutingStrategyBase
):
    """
    Sets get_available_deployment and async_get_available_deployment
    on an instance of litellm.Router.

    Uses setattr() for per-instance method replacement (not class-level).
    """
    setattr(
        self,
        "get_available_deployment",
        CustomRoutingStrategy.get_available_deployment,
    )
    setattr(
        self,
        "async_get_available_deployment",
        CustomRoutingStrategy.async_get_available_deployment,
    )
```

**Key insight**: This uses `setattr` on the **instance**, not the class. This means
multiple Router instances can have different routing strategies, and the replacement
survives `os.execvp()` / multi-worker forking because each worker creates its own
Router instance.

### async_pre_routing_hook

**Location**: [`reference/litellm/litellm/router.py:8326-8351`](../../reference/litellm/litellm/router.py:8326)

```python
async def async_pre_routing_hook(
    self,
    model: str,
    request_kwargs: Dict,
    messages: Optional[List[Dict[str, str]]] = None,
    input: Optional[Union[str, List]] = None,
    specific_deployment: Optional[bool] = False,
) -> Optional[PreRoutingHookResponse]:
    """
    Called before the routing decision.
    Used by AutoRouter to modify model/messages before routing.
    """
    if model in self.auto_routers:
        return await self.auto_routers[model].async_pre_routing_hook(...)
    return None
```

**Note**: This hook is only invoked for models registered as `AutoRouter` deployments
(model name starts with `auto_router/`). It is NOT called for `CustomRoutingStrategyBase`
strategies. If RouteIQ needs pre-classification, it must implement this within the
`async_get_available_deployment` method of its custom strategy.

### CustomLogger (Callback Hooks)

**Location**: [`reference/litellm/litellm/integrations/custom_logger.py`](../../reference/litellm/litellm/integrations/custom_logger.py)

Relevant hooks for RouteIQ:
- `pre_call_check(deployment)` — sync pre-call validation (e.g., policy check)
- `async_pre_call_check(deployment, parent_otel_span)` — async pre-call validation
- `async_filter_deployments(model, healthy_deployments, messages, request_kwargs)` — filter deployments before routing selection
- `log_success_event(kwargs, response_obj, start_time, end_time)` — post-call telemetry
- `async_log_success_event(...)` — async post-call telemetry

These hooks enable RouteIQ to inject policy evaluation, telemetry, and resilience logic
without modifying LiteLLM's Router class.

### RouteIQ's Current Monkey-Patch (to be replaced)

**Location**: [`src/litellm_llmrouter/routing_strategy_patch.py:476-488`](../../src/litellm_llmrouter/routing_strategy_patch.py:476)

```python
# Current approach (to be eliminated):
Router.routing_strategy_init = create_patched_routing_strategy_init(...)
Router.get_available_deployment = create_patched_get_available_deployment(...)
Router.async_get_available_deployment = create_patched_async_get_available_deployment(...)
```

This patches three methods on the **class** (not instance), which means:
- Every Router instance is affected (global side effect)
- Multi-worker uvicorn requires patches to be re-applied after `os.fork()`
- The current workaround is `workers=1` hardcoded in startup.py

The migration to `CustomRoutingStrategyBase` replaces all three patches with a single
`router.set_custom_routing_strategy()` call that operates per-instance.
