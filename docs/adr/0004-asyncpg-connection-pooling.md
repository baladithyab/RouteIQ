# ADR-0004: Implement asyncpg Connection Pooling

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: Per-Query TCP Connections

RouteIQ's database operations (A2A agent persistence, MCP server registry, audit
logging, leader election, health checks) originally created a new TCP connection
to PostgreSQL for every operation. Each database call followed this pattern:

```python
conn = await asyncpg.connect(db_url)
try:
    result = await conn.fetch(query)
finally:
    await conn.close()
```

This approach had several performance and reliability problems:

1. **Connection overhead**: Each TCP connection establishment requires a 3-way
   handshake (~1-3ms on localhost, 5-20ms cross-AZ). With SSL negotiation, this
   adds another 5-15ms. For high-frequency operations like health checks
   (every 10s) and audit logging (every request), this overhead was substantial.

2. **Connection storms**: Under load, concurrent requests each opening their own
   connection could exhaust PostgreSQL's `max_connections` (default: 100). A
   burst of 50 concurrent audit log writes would open 50 simultaneous connections.

3. **No connection reuse**: Prepared statements, session-level caches, and
   connection-level optimizations were lost on every call since connections were
   immediately closed.

4. **Resource leaks**: If an exception occurred between `connect()` and `close()`,
   or if the `finally` block failed, connections could leak. Over time, this
   would exhaust both client and server resources.

5. **No backpressure**: There was no mechanism to queue or reject requests when
   all connections were in use. Every caller would attempt to open a new
   connection, exacerbating connection storms.

### Impact Assessment

In production profiling, database connection establishment accounted for ~15-25%
of total request latency for audit-logged control-plane operations. Leader election
(polling every 5s per replica) was particularly wasteful, creating and destroying
a connection every cycle.

## Decision

Implement a shared `asyncpg.Pool` singleton that all database consumers share.

### Pool Configuration

```python
pool = await asyncpg.create_pool(
    dsn=db_url,
    min_size=2,           # Pre-warm 2 connections at startup
    max_size=10,          # Maximum 10 concurrent connections
    command_timeout=30,   # 30s query timeout
    max_inactive_connection_lifetime=300,  # Close idle connections after 5min
    # SSL prefer mode for cloud deployments
)
```

The pool parameters are chosen for a typical deployment:

- **min_size=2**: Ensures at least 2 connections are always ready, avoiding
  cold-start latency for the first queries after idle periods.
- **max_size=10**: Limits total connections to prevent PostgreSQL exhaustion.
  With 2-4 replicas in HA mode, total connections stay under 40-80.
- **command_timeout=30**: Prevents runaway queries from holding connections.
- **max_inactive_connection_lifetime=300**: Recycles idle connections to
  accommodate PostgreSQL server restarts and cloud provider connection limits.

### Singleton Pattern

```python
_pool = None
_pool_lock = asyncio.Lock()

async def get_db_pool(db_url: Optional[str] = None):
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is not None:  # Double-checked locking
            return _pool
        url = db_url or get_database_url()
        if not url:
            return None
        _pool = await asyncpg.create_pool(dsn=url, ...)
        return _pool
```

Double-checked locking with `asyncio.Lock()` ensures only one pool is created
even under concurrent startup. The pool is lazily initialized on first use,
not at module import time.

### Consumer Migration

All database consumers were migrated to use `pool.acquire()` instead of
`asyncpg.connect()`:

```python
# Before:
conn = await asyncpg.connect(db_url)
try:
    result = await conn.fetch(query)
finally:
    await conn.close()

# After:
pool = await get_db_pool()
async with pool.acquire() as conn:
    result = await conn.fetch(query)
```

Affected consumers:
- A2A agent repository (`database.py`)
- MCP server repository (`database.py`)
- Audit log writer (`database.py`, `audit.py`)
- Leader election (`leader_election.py`)
- Health check probes (`routes/health.py`)
- Database migration runner (`migrations.py`)

### Lifecycle Management

The pool is closed during gateway shutdown via `close_db_pool()`, which is
registered as a lifecycle hook in the composition root:

```python
async def close_db_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
```

This is called from `gateway/app.py`'s shutdown sequence, ensuring all
connections are properly released before the process exits.

### Testing

Unit tests use the `reset_db_pool()` function (registered in conftest) to
prevent pool singleton leakage between tests:

```python
@pytest.fixture(autouse=True)
async def reset_database():
    yield
    await close_db_pool()
```

## Consequences

### Positive

- **Dramatic latency reduction**: Database operations skip TCP/SSL handshake
  on every call. Audit log writes dropped from ~15ms to ~1-2ms.

- **Connection stability**: The pool manages connection lifecycle, handling
  reconnection after server restarts and connection pruning after idle periods.

- **Backpressure**: When all 10 connections are in use, `pool.acquire()` blocks
  (up to timeout) rather than opening a new connection. This prevents
  connection storms under load.

- **Resource safety**: `async with pool.acquire()` guarantees connection return
  to the pool even if an exception occurs. No more connection leaks.

- **Monitoring**: asyncpg pools expose `get_size()`, `get_min_size()`,
  `get_max_size()`, `get_idle_size()` for operational visibility.

### Negative

- **Startup dependency**: The pool requires a working PostgreSQL connection at
  first use. If the database is temporarily unavailable, the first caller gets
  an error. Mitigated by lazy initialization (only connects when needed) and
  by the health check probing database connectivity.

- **Connection limit coordination**: In HA deployments with N replicas, total
  connections = N * max_size. With 4 replicas * 10 max = 40 connections.
  Operators must ensure PostgreSQL's `max_connections` accommodates this.

- **Singleton testing complexity**: Module-level singleton requires explicit
  reset in tests. Forgetting `reset_db_pool()` in conftest causes test
  cross-contamination.

## Alternatives Considered

### Alternative A: Connection-per-Request (Status Quo)

Keep creating new connections for each database operation.

- **Pros**: Simplest code; no pool lifecycle management.
- **Cons**: Unacceptable performance overhead; connection storms under load;
  resource leaks.
- **Rejected**: The performance and reliability problems are too severe for
  production use.

### Alternative B: SQLAlchemy Async Engine

Use SQLAlchemy's async engine with its built-in connection pooling.

- **Pros**: Mature ORM; migration tooling (Alembic); connection pool built-in.
- **Cons**: Adds a heavy dependency (~20MB); RouteIQ's queries are simple
  enough that an ORM adds complexity without benefit; SQLAlchemy's async
  support is relatively new and less battle-tested than asyncpg's native pool.
- **Rejected**: Over-engineered for RouteIQ's simple query patterns.

### Alternative C: PgBouncer External Pooler

Deploy PgBouncer as a connection pooler between RouteIQ and PostgreSQL.

- **Pros**: Language-agnostic; handles pooling at the infrastructure level;
  supports connection multiplexing.
- **Cons**: Adds operational complexity (another service to deploy, monitor,
  configure); doesn't eliminate the per-request connection overhead within
  the application; overkill when asyncpg's built-in pool suffices.
- **Rejected**: Valid for very large deployments but unnecessary at current
  scale. Can be added later as an infrastructure optimization.

## References

- `src/litellm_llmrouter/database.py` — Pool implementation and repositories
- `src/litellm_llmrouter/leader_election.py` — Pool consumer (leader election)
- `src/litellm_llmrouter/gateway/app.py` — Pool lifecycle hooks
- `tests/unit/conftest.py` — Pool reset fixture
- asyncpg documentation: https://magicstack.github.io/asyncpg/
