# ADR-0021: Externalize In-Process State to Redis for Multi-Worker Safety

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

RouteIQ supports multi-worker deployments via `ROUTEIQ_WORKERS` (using the plugin
routing strategy, which preserves app state across `os.fork()`). However, two
subsystems relied on in-process state that could not be shared across workers:

1. **SessionCache** in `centroid_routing.py` — an in-memory `OrderedDict` that
   maintains conversation routing affinity (same conversation → same model).
   Different workers had independent caches, so the same conversation could be
   routed to different models depending on which worker handled each request.

2. **CircuitBreaker** in `resilience.py` — per-process failure tracking via
   `deque[float]` timestamps. Workers could not share failure observations, so
   a provider experiencing 50% error rate might only be detected by one worker
   while others continued sending traffic.

Both issues are invisible in single-worker deployments but cause incorrect
behavior under multi-worker configurations, which are the recommended production
setup.

## Decision

Externalize both state stores to Redis when available, falling back to in-memory
when Redis is not configured. This preserves backward compatibility for
single-worker and development setups while enabling correct multi-worker behavior.

### SessionCache (centroid_routing.py)

- Uses the **sync** Redis client (`get_sync_redis_client()`) because
  `select_deployment()` is a synchronous method (part of the `RoutingStrategy`
  protocol).
- Redis key format: `routeiq:session:{session_key}` with value `{model}|{tier}`.
- TTL is set via Redis `EX` parameter, replacing the manual expiry tracking.
- LRU eviction is handled implicitly by Redis's memory policies; the in-memory
  `max_size` cap only applies to the fallback backend.
- Lazy initialization: the first `get()` or `put()` call checks Redis
  availability and caches the result.

### CircuitBreaker (resilience.py)

- New `SharedCircuitBreakerState` class coordinates state across workers via
  the **async** Redis client (`get_async_redis_client()`).
- Redis key format:
  - `routeiq:cb:{name}:state` — circuit state string (`closed`/`open`/`half_open`)
  - `routeiq:cb:{name}:failures` — atomic failure counter via `INCR` with TTL
- State transitions (`_transition_to`) broadcast to Redis.
- `record_failure` increments both local and shared counters; the circuit opens
  when *either* local or shared count exceeds the threshold.
- `allow_request` checks shared state so a circuit opened by worker A is
  respected by worker B.
- All Redis operations are best-effort — failures are logged at DEBUG level and
  never propagate to callers.

### Fallback Behavior

When Redis is unavailable (not configured, unreachable, or erroring):

- `SessionCache` uses the existing in-memory `OrderedDict` with LRU eviction.
- `CircuitBreaker` uses the existing per-process `deque`-based failure tracking.
- Behavior is identical to pre-change single-worker mode.

## Consequences

### Positive

- **Multi-worker correctness**: Conversation routing affinity and circuit breaker
  state are now consistent across all workers.
- **Zero breaking changes**: All public APIs retain their signatures. The sync
  `get()`/`put()` interface for `SessionCache` is unchanged.
- **Graceful degradation**: Redis failure never breaks request processing —
  the system falls back to in-memory automatically.
- **No new dependencies**: Uses the existing `redis` package and the
  `redis_pool.py` singleton infrastructure (ADR-0005).

### Negative

- **Redis latency on hot path**: `SessionCache.get()` adds one Redis `GET`
  (~0.1ms on localhost) to every centroid routing decision. This is acceptable
  relative to the ~2ms embedding classification.
- **Eventual consistency**: There is a small window where workers may disagree
  on circuit state (Redis propagation delay). This is acceptable because the
  circuit breaker already tolerates race conditions between the timeout check
  and state transition.
- **Redis memory usage**: Session cache entries consume ~100 bytes each in
  Redis. At `max_size=10000` this is ~1MB — negligible.

## Alternatives Considered

### Alternative A: Shared Memory via `multiprocessing.shared_memory`

- Pros: No external dependency, very fast.
- Cons: Complex serialization, not portable to Kubernetes multi-pod deployments,
  requires manual locking, does not survive process restarts.

### Alternative B: Make `select_deployment` async and use async Redis

- Pros: Consistent with the rest of the async codebase.
- Cons: Breaking change to the `RoutingStrategy` protocol interface which is
  synchronous by design (inherited from LiteLLM's `CustomRoutingStrategyBase`).
  Would require changes across all strategy implementations.

### Alternative C: PostgreSQL-backed state

- Pros: Already available in most deployments.
- Cons: Too slow for hot-path session lookups (~5ms per query vs ~0.1ms Redis).
  Circuit breaker INCR operations would cause write amplification.

### Alternative D: Do nothing (document single-worker limitation)

- Pros: No code changes.
- Cons: Multi-worker is the recommended production setup; users would hit
  silent routing inconsistencies and incomplete circuit breaking.
