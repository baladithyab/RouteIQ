# ADR-0005: Redis Singleton Client with Lifecycle Management

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: Per-Call Client Creation

RouteIQ uses Redis for multiple subsystems: health check probes, quota
enforcement, MCP gateway HA synchronization, cache plugin, conversation
affinity, and leader election. The original implementation used a factory
function `create_async_redis_client()` that created a **new** `redis.asyncio.Redis`
instance on every call.

Each consumer called the factory independently:

```python
# health.py
client = create_async_redis_client()
await client.ping()

# quota.py
client = create_async_redis_client()
await client.get(quota_key)

# mcp_gateway.py
client = create_async_redis_client()
await client.hgetall("mcp_servers")
```

### Problems

1. **Resource waste**: Each `redis.asyncio.Redis()` instance creates its own
   internal connection pool. With 6+ consumers each creating clients
   independently, the process maintained 6+ separate connection pools to the
   same Redis server.

2. **No lifecycle management**: Clients were created but never explicitly
   closed. Python's garbage collector would eventually clean them up, but
   during shutdown, connections could be left in a half-open state, causing
   `ConnectionResetError` noise in logs.

3. **No health checking**: The factory created clients without verifying
   Redis was actually reachable. Consumers discovered connection failures
   at query time, often with confusing error messages.

4. **Configuration drift**: Each factory call read environment variables
   independently (`REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_SSL`,
   `REDIS_DB`). If environment variables changed mid-process (e.g., via
   hot-reload), different consumers could be connected to different Redis
   configurations.

5. **Password logging risk**: The original factory didn't mask the Redis
   password in connection logging, risking credential exposure in debug logs.

## Decision

Replace per-call client creation with a process-level singleton that provides
a single, shared Redis client with proper lifecycle management.

### Singleton Implementation

```python
_async_client: Optional[Any] = None
_async_client_lock = asyncio.Lock()

async def get_async_redis_client() -> Optional[redis.asyncio.Redis]:
    global _async_client
    if _async_client is not None:
        return _async_client
    async with _async_client_lock:
        if _async_client is not None:
            return _async_client
        settings = _redis_settings()
        if not settings.get("host"):
            return None
        client = redis.asyncio.Redis(**settings)
        # Health check: verify connectivity
        await client.ping()
        _async_client = client
        return _async_client
```

Key design decisions:

- **Double-checked locking** with `asyncio.Lock()` prevents duplicate creation
  under concurrent startup.
- **Health check ping** at creation time ensures the client is actually
  connected. If Redis is unreachable, `get_async_redis_client()` raises
  immediately rather than returning a broken client.
- **Returns `None`** when Redis is not configured (no `REDIS_HOST`), allowing
  callers to gracefully degrade.

### Password Security

The `_redis_settings()` function masks passwords in log output:

```python
def _redis_settings() -> dict[str, Any]:
    password = os.getenv("REDIS_PASSWORD")
    if password:
        logger.warning(
            "Redis password configured via REDIS_PASSWORD env var. "
            "Consider using Redis ACLs or IAM auth for production."
        )
    return {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "password": password,
        "ssl": os.getenv("REDIS_SSL", "false").lower() == "true",
        "db": int(os.getenv("REDIS_DB", "0")),
    }
```

### Lifecycle Management

The singleton is closed during gateway shutdown:

```python
async def close_async_redis_client():
    global _async_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None
```

This is registered as a shutdown hook in `gateway/app.py`.

### Backward Compatibility

The old `create_async_redis_client()` function is retained with a deprecation
warning:

```python
def create_async_redis_client() -> redis.asyncio.Redis:
    warnings.warn(
        "create_async_redis_client() is deprecated. "
        "Use await get_async_redis_client() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return redis.asyncio.Redis(**_redis_settings())
```

All internal consumers have been migrated to `get_async_redis_client()`.

## Consequences

### Positive

- **Single connection pool**: All 6+ Redis consumers share one connection
  pool, reducing memory and file descriptor usage.

- **Consistent configuration**: All consumers use the same Redis settings,
  read once at singleton creation time.

- **Clean shutdown**: The `close_async_redis_client()` hook ensures all
  connections are properly closed before process exit, eliminating
  `ConnectionResetError` noise.

- **Early failure detection**: The health check ping at creation time
  catches Redis connectivity issues immediately, not at first query time.

- **Password awareness**: The warning about `REDIS_PASSWORD` encourages
  operators to use stronger authentication mechanisms in production.

### Negative

- **Singleton testing burden**: Tests must call `close_async_redis_client()`
  between tests to prevent cross-contamination. An `autouse=True` fixture
  handles this.

- **Hot-reload limitation**: Once created, the singleton uses the settings
  from creation time. If `REDIS_HOST` changes via hot-reload, the client
  won't reconnect to the new host until restart. This is acceptable because
  Redis connection changes typically require a restart anyway.

- **Startup ordering**: Consumers that call `get_async_redis_client()` before
  Redis is available will get `None` (Redis not configured) or a connection
  error. The composition root handles this by not requiring Redis at startup.

## Alternatives Considered

### Alternative A: Per-Call Creation (Status Quo)

Keep creating new clients for each call.

- **Pros**: Simplest code; no singleton lifecycle.
- **Cons**: Resource waste, no lifecycle management, configuration drift.
- **Rejected**: Unacceptable resource waste in production.

### Alternative B: Dependency Injection

Pass a Redis client instance through function parameters or a DI container.

- **Pros**: Explicit dependencies; easy to test with mocks; no global state.
- **Cons**: Requires threading a Redis client through every call chain, which
  is invasive across 6+ modules. RouteIQ doesn't use a DI framework.
- **Rejected**: Too invasive for the benefit. The singleton pattern with
  explicit `reset_*()` for testing is the established pattern in the codebase.

### Alternative C: Redis Connection Pool (External)

Use Redis Sentinel or a Redis proxy (like Twemproxy) for connection management.

- **Pros**: Infrastructure-level connection management; failover support.
- **Cons**: Adds operational complexity; doesn't solve the in-process resource
  waste problem.
- **Rejected**: Orthogonal to the in-process singleton problem. Can be added
  later for HA Redis deployments.

## References

- `src/litellm_llmrouter/redis_pool.py` — Singleton implementation
- `src/litellm_llmrouter/gateway/app.py` — Shutdown hook registration
- `tests/unit/conftest.py` — Reset fixture
- redis-py documentation: https://redis.readthedocs.io/
