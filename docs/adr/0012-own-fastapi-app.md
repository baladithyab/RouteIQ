# ADR-0012: RouteIQ Owns Its FastAPI Application

**Status**: Proposed
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: Borrowed FastAPI App

RouteIQ currently borrows LiteLLM's FastAPI app instance. The `create_app()`
factory in `gateway/app.py` calls `litellm.proxy.proxy_server.app` to get
LiteLLM's existing FastAPI application, then adds RouteIQ's middleware,
routes, and plugins on top.

This creates several architectural problems:

1. **No lifespan control**: LiteLLM manages its own `@asynccontextmanager`
   lifespan. RouteIQ cannot define its own lifespan function because FastAPI
   only supports one. Plugin startup/shutdown hooks are stored as lambdas on
   `app.state` and called manually from `startup.py`, which is fragile.

2. **Middleware stack ordering**: LiteLLM may add its own middleware during
   its startup. RouteIQ has no guarantee about the relative ordering of
   its middleware vs LiteLLM's. The `BackpressureMiddleware` being "innermost"
   relies on being registered before LiteLLM adds any middleware, which is a
   temporal coupling.

3. **Exception handler conflicts**: Both LiteLLM and RouteIQ want to define
   custom exception handlers. Only one can win for a given exception type.

4. **OpenAPI schema pollution**: LiteLLM's routes appear in RouteIQ's OpenAPI
   schema and vice versa, creating a confusing API documentation experience.

5. **Testing isolation**: To test RouteIQ's routes in isolation, you must
   either mock LiteLLM's entire app or accept that LiteLLM's middleware runs
   during RouteIQ route tests.

## Decision

Create RouteIQ's own FastAPI application with a custom lifespan, and mount
LiteLLM's proxy as a sub-application.

### Architecture

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def routeiq_lifespan(app: FastAPI):
    # Startup
    await startup_http_client_pool()
    await plugin_manager.startup_all(app)
    warmup_centroid_classifier()
    yield
    # Shutdown
    await plugin_manager.shutdown_all(app)
    await graceful_shutdown()
    await close_db_pool()
    await close_async_redis_client()
    await shutdown_http_client_pool()

app = FastAPI(
    title="RouteIQ Gateway",
    lifespan=routeiq_lifespan,
)

# Mount LiteLLM proxy at /v1/
from litellm.proxy.proxy_server import app as litellm_app
app.mount("/v1", litellm_app)
```

### Benefits of Ownership

1. **Proper lifespan**: `routeiq_lifespan` runs startup/shutdown hooks in
   deterministic order. No more `app.state` lambda hacks.

2. **Middleware control**: RouteIQ's middleware stack is fully defined before
   LiteLLM's sub-app is mounted. The sub-app has its own middleware.

3. **Clean exception handling**: RouteIQ's exception handlers apply to
   RouteIQ routes. LiteLLM's handlers apply within the sub-app.

4. **Separate OpenAPI**: RouteIQ and LiteLLM can have independent OpenAPI
   schemas, or RouteIQ can merge them with custom logic.

### URL Mapping

| Before (Shared App) | After (Owned App) |
|--------------------|-------------------|
| `/chat/completions` | `/v1/chat/completions` (LiteLLM sub-app) |
| `/v1/chat/completions` | `/v1/chat/completions` (LiteLLM sub-app) |
| `/_health/ready` | `/_health/ready` (RouteIQ) |
| `/llmrouter/mcp/*` | `/llmrouter/mcp/*` (RouteIQ) |
| `/ui/*` | `/ui/*` (RouteIQ admin UI) |

## Consequences

### Positive

- **Proper lifecycle management**: No temporal boot coupling.
- **Clean middleware ordering**: Fully deterministic.
- **Independent exception handling**: No conflicts.
- **Better testability**: RouteIQ app can be tested without LiteLLM.

### Negative

- **URL changes**: Existing clients hitting `/chat/completions` directly
  would need to use `/v1/chat/completions`. Mitigated by a redirect
  middleware or by mounting LiteLLM at `/` (with prefix stripping).
- **Sub-app limitations**: FastAPI sub-apps don't share middleware with
  the parent app. Cross-cutting concerns (request ID, auth) need to be
  applied at both levels or via ASGI wrapping.
- **LiteLLM coupling**: The sub-app mounting approach still requires
  importing LiteLLM's app object, which may have import-time side effects.

## Alternatives Considered

### Alternative A: Keep Borrowing LiteLLM's App

- **Pros**: No migration; works today.
- **Cons**: All the problems listed in Context persist.
- **Status quo**: Currently implemented. This ADR proposes changing it.

### Alternative B: Reverse Proxy

Run LiteLLM and RouteIQ as separate processes behind Nginx.

- **Pros**: Complete isolation; no Python-level coupling.
- **Cons**: Doubles operational complexity; loses in-process routing
  strategy integration; adds network latency.
- **Rejected**: The in-process integration is a core architectural
  requirement for routing strategies.

## References

- `src/litellm_llmrouter/gateway/app.py` — Current app factory
- `src/litellm_llmrouter/startup.py` — Current startup with manual hooks
- [ADR-0001: Three-Layer Architecture](0001-three-layer-architecture.md)
- FastAPI Sub Applications: https://fastapi.tiangolo.com/advanced/sub-applications/
