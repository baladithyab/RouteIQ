# ADR-0001: Adopt Three-Layer Architecture

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem Statement

RouteIQ v0.1.0 was architecturally monolithic — a single Python application that
extended LiteLLM by directly borrowing its FastAPI app instance, monkey-patching
three methods on its `Router` class, and reimplementing several subsystems (MCP,
A2A, caching) that LiteLLM already provides natively.

This created several critical problems:

1. **Tight coupling to LiteLLM internals**: Monkey-patching `Router.routing_strategy_init()`,
   `Router.get_available_deployment()`, and `Router.async_get_available_deployment()` at
   the class level meant any upstream LiteLLM change to these method signatures could
   silently break RouteIQ. The patch in `routing_strategy_patch.py` was ~592 lines of
   fragile interception code.

2. **Single-worker constraint**: Class-level monkey-patching mutates the shared class
   object. Under `os.fork()` (multi-worker uvicorn), the patched methods were lost because
   the patch was applied in the parent process before `fork()`, and `os.fork()` copies
   memory but the class-level mutations aren't guaranteed to survive across all fork
   implementations. This forced `workers=1` in `startup.py`.

3. **Redundant code**: ~7,578 lines duplicated functionality LiteLLM already provides:
   - MCP gateway: 5,958 lines across `mcp_parity.py` (1,072), `mcp_jsonrpc.py` (868),
     `mcp_sse_transport.py` (1,439), plus portions of `mcp_gateway.py` and `routes/mcp.py`
   - A2A gateway: 1,620 lines wrapping what LiteLLM's `global_agent_registry` already does

4. **No clear separation of concerns**: RouteIQ's genuine value (ML routing, policy engine,
   resilience, observability) was interleaved with upstream functionality, making it
   difficult to reason about what RouteIQ actually adds.

5. **Difficult to test in isolation**: With no layer boundaries, unit tests required
   mocking LiteLLM internals extensively, and 16 module-level singletons leaked between
   tests (only 5 of 21 had proper reset functions in conftest).

### Assessment

An independent architecture review scored RouteIQ v0.2.0 at 3.05/5 for goal alignment —
"NOT production-ready". The review identified that RouteIQ's genuine value was ~14,000
lines (39% of the 36,207-line codebase), with the remaining 61% being either redundant
with upstream or infrastructure scaffolding.

The [TG3 Alternative Patterns Analysis](../architecture/tg3-alternative-patterns.md)
evaluated five architecture options to resolve these problems.

## Decision

Adopt a three-layer architecture that cleanly separates concerns:

### Layer 1: LiteLLM (Unmodified Upstream)

LiteLLM provides the OpenAI-compatible proxy, auth/RBAC, team/org management,
virtual keys, MCP gateway, A2A protocol, caching (11 backends), 60+ callback
integrations, and the `litellm.Router` class with its official extension points.

RouteIQ treats LiteLLM as a **dependency**, not a codebase to fork or patch.
No modifications to LiteLLM source code. No monkey-patching of LiteLLM classes
(except as a deprecated backward-compatibility fallback). LiteLLM environment
variables (`LITELLM_*`) are passed through unchanged.

### Layer 2: RouteIQ Plugins (Independently Installable)

RouteIQ's genuine value is packaged as plugins that integrate with LiteLLM via
official extension points:

- **Routing Plugin** (`custom_routing_strategy.py`): Implements
  `CustomRoutingStrategyBase` from `litellm.types.router`. Provides 18 ML routing
  strategies, A/B testing pipeline, centroid-based zero-config routing, and routing
  profiles. Installed via `router.set_custom_routing_strategy()`.

- **Security Plugins** (`policy_engine.py`, `url_security.py`, `auth.py`):
  OPA-style policy evaluation, SSRF protection, admin auth hardening.

- **Observability Plugins** (`observability.py`, `telemetry_contracts.py`,
  `router_decision_callback.py`): OpenTelemetry integration, versioned telemetry
  contracts, routing decision span attributes.

- **Resilience Plugins** (`resilience.py`, `http_client_pool.py`):
  Backpressure middleware, drain manager, circuit breakers.

- **Gateway Plugins** (13 built-in in `gateway/plugins/`): Evaluator, skills
  discovery, guardrails, PII detection, cost tracking, etc.

Each plugin is designed to work with vanilla LiteLLM. The long-term goal (see
[ADR-0014](0014-plugin-extraction.md)) is to extract these into independently
pip-installable packages.

### Layer 3: Gateway (Composition Root)

The `gateway/app.py` module serves as the composition root that wires everything
together:

1. Creates or borrows the FastAPI application
2. Installs the routing strategy (plugin or legacy monkey-patch)
3. Configures middleware stack in deterministic order:
   - Backpressure (innermost, registered first)
   - RequestID correlation
   - CORS
   - Policy engine
   - Management middleware
   - Plugin middleware
   - Router decision callback
4. Loads and starts plugins via `PluginManager`
5. Registers routes
6. Sets up lifecycle hooks (HTTP client pool, Redis, database, drain manager)

The composition root is the only place that knows about all three layers. Plugins
don't reference the composition root, and LiteLLM doesn't know about RouteIQ.

### Load Order

```
startup.py
  -> create_app()
      -> patch_litellm_router() [if legacy mode]
         OR install_routeiq_strategy() [if plugin mode]
      -> get/create FastAPI app
      -> add_backpressure_middleware() [innermost]
      -> _configure_middleware() [RequestID, CORS, Policy, etc.]
      -> _load_plugins()
      -> _register_routes()
      -> setup lifecycle hooks on app.state
  -> uvicorn.run(app=app)
```

## Consequences

### Positive

- **Multi-worker support**: Plugin routing strategy uses per-instance `setattr()`,
  not class-level mutation. `os.fork()` preserves the app state. Configurable via
  `ROUTEIQ_WORKERS`.

- **Upstream resilience**: No class-level patches means LiteLLM can be upgraded
  without risk of silent breakage in routing. The `CustomRoutingStrategyBase` API
  is LiteLLM's official contract.

- **Reduced codebase**: Shedding redundant MCP/A2A reimplementations removes
  ~3,379 lines (see [ADR-0003](0003-delete-redundant-mcp.md)) with more
  reductions planned.

- **Clear mental model**: Developers can reason about "what does RouteIQ add?"
  by looking at Layer 2 plugins only. Layer 1 is upstream. Layer 3 is glue.

- **Testability**: Plugins can be unit-tested against mock Router instances
  without starting the full LiteLLM proxy. Integration tests only need the
  composition root.

- **Independent deployability**: Users who only want ML routing can (eventually)
  `pip install routeiq-routing` and use it with their own LiteLLM setup.

### Negative

- **API surface dependency**: RouteIQ depends on LiteLLM's `CustomRoutingStrategyBase`
  and `Router.set_custom_routing_strategy()` APIs remaining stable. If LiteLLM changes
  these, RouteIQ must adapt.

- **Boot ordering complexity**: Plugin startup/shutdown must be coordinated with
  LiteLLM's own lifespan. Currently uses `app.state` lambdas because LiteLLM manages
  its own lifespan (see [ADR-0012](0012-own-fastapi-app.md) for proposed resolution).

- **Backward compatibility burden**: The deprecated monkey-patch path
  (`ROUTEIQ_USE_PLUGIN_STRATEGY=false`) must be maintained until all deployments
  migrate, adding code that will eventually be removed.

- **Feature lag**: RouteIQ cannot add capabilities that require modifying LiteLLM's
  request pipeline unless LiteLLM provides appropriate hooks (callbacks, middleware,
  or extension points).

## Alternatives Considered

### Alternative A: Fork LiteLLM

Fork the LiteLLM repository and maintain RouteIQ as a direct modification.

- **Pros**: Full control over all internals; no API surface dependency; can modify
  any LiteLLM behavior.
- **Cons**: LiteLLM receives 50-100 commits per week. Tracking upstream would
  consume 2-4 person-weeks per month in merge conflict resolution alone. The fork
  would inevitably diverge, losing access to new LiteLLM features (provider
  integrations, security patches, etc.).
- **Score**: 2.95/5 in TG3 evaluation. Rejected due to unsustainable maintenance
  cost.

### Alternative B: Sidecar Pattern

Deploy RouteIQ as a separate service that LiteLLM calls for routing decisions
via HTTP/gRPC.

- **Pros**: Clean separation; independent scaling; language-agnostic interface;
  no dependency on LiteLLM internals.
- **Cons**: Adds 1-5ms network latency per routing decision; requires service
  discovery, health checking, and circuit breaking between sidecar and proxy;
  doubles operational complexity (two services to deploy, monitor, scale).
- **Score**: 3.35/5 in TG3 evaluation. Rejected as premature for current scale,
  but noted as a valid evolution path for Option E (Hybrid) when ML inference
  outgrows in-process capacity.

### Alternative C: Microservices

Decompose RouteIQ into independent microservices (routing service, policy service,
observability service, etc.).

- **Pros**: Best independent scaling; fault isolation; polyglot capability.
- **Cons**: Crushing complexity for a team of this size. Distributed transactions,
  service mesh, API versioning, eventual consistency. Estimated 16-24 person-weeks
  just for the initial decomposition, plus ongoing coordination cost.
- **Score**: 2.10/5 in TG3 evaluation. Rejected as massively over-engineered for
  current requirements.

### Alternative D: Hybrid Plugin + External Services

Start with Plugin (Option A) but design the `CustomRoutingStrategyBase`
implementation to delegate to an external service when ML inference demands exceed
in-process capacity.

- **Pros**: Best of both worlds; starts simple, scales when needed; the routing
  strategy interface is identical whether computing locally or calling a service.
- **Cons**: Higher initial cost (10-14 person-weeks vs 6-8 for pure Plugin);
  service extraction complexity when the time comes.
- **Score**: 3.95/5 in TG3 evaluation. Accepted as the evolution path but not
  the starting point. The current Plugin architecture is designed to enable this
  transition.

## References

- [TG3 Alternative Patterns Analysis](../architecture/tg3-alternative-patterns.md)
- [TG3 Rearchitecture Proposal](../architecture/tg3-rearchitecture-proposal.md)
- [Deep Architecture Review v0.2.0](../architecture/deep-review-v0.2.0.md)
- LiteLLM `CustomRoutingStrategyBase`: `litellm.types.router`
- LiteLLM `Router.set_custom_routing_strategy()`: `litellm.router`
- Gateway composition root: `src/litellm_llmrouter/gateway/app.py`
