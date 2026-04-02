# ADR-0015: K8s-Native Leader Election via Lease API

**Status**: Proposed
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: PostgreSQL-Based Leader Election

RouteIQ's current leader election (`leader_election.py`) uses PostgreSQL
for distributed coordination. Each replica:

1. Polls the database every 5 seconds
2. Attempts to acquire/renew a lease row
3. Non-leaders skip config sync and other leader-only tasks

This works but has problems:

1. **Unnecessary DB dependency**: Leader election requires PostgreSQL even
   when the gateway doesn't otherwise need database persistence.
2. **Connection overhead**: Each poll opens a database query (now pooled via
   [ADR-0004](0004-asyncpg-connection-pooling.md), but still adds load).
3. **Not Kubernetes-native**: In K8s, the preferred mechanism for leader
   election is the `coordination.k8s.io/v1` Lease API, which is built into
   the API server and requires no external database.
4. **Clock skew sensitivity**: Database-based leases depend on synchronized
   clocks. In cloud environments, clock skew between replicas can cause
   split-brain scenarios.

### Kubernetes Context

Most production RouteIQ deployments run on Kubernetes. The K8s API server
provides a Lease resource specifically designed for leader election:

```yaml
apiVersion: coordination.k8s.io/v1
kind: Lease
metadata:
  name: routeiq-leader
  namespace: default
spec:
  holderIdentity: routeiq-pod-abc123
  leaseDurationSeconds: 15
  renewTime: "2026-04-02T09:00:00Z"
```

## Decision

Use the Kubernetes Lease API for leader election in K8s environments. Fall
back to Redis SETNX for non-K8s deployments.

### Strategy Selection

```python
def get_leader_elector():
    if is_kubernetes_environment():
        return KubernetesLeaseElector(
            lease_name="routeiq-leader",
            namespace=os.getenv("POD_NAMESPACE", "default"),
            identity=os.getenv("POD_NAME", socket.gethostname()),
            lease_duration=15,
            renew_interval=5,
        )
    elif is_redis_configured():
        return RedisLeaseElector(
            key="routeiq:leader",
            ttl=15,
            renew_interval=5,
        )
    else:
        return SingletonLeader()  # Always leader (single replica)
```

### Kubernetes Lease Elector

Uses the `kubernetes` Python client to acquire and renew leases:

- Requires `ServiceAccount` with `coordination.k8s.io` Lease permissions
- Uses optimistic concurrency (resourceVersion-based)
- Graceful handover on pod shutdown

### Redis Lease Elector

For non-K8s deployments (Docker Compose, bare metal):

- Uses `SETNX` with TTL for atomic lease acquisition
- Renewal via `SET` with `XX` (only if exists) and new TTL
- No PostgreSQL dependency

## Consequences

### Positive

- **No DB dependency for leader election**: Removes PostgreSQL requirement
  for HA mode.
- **Kubernetes-native**: Uses the platform's built-in coordination primitive.
- **Better failure detection**: K8s Lease API integrates with pod lifecycle;
  deleted pods automatically lose leadership.
- **Redis fallback**: Non-K8s deployments get Redis-based election without
  PostgreSQL.

### Negative

- **K8s client dependency**: Adds `kubernetes` Python client for K8s mode.
- **RBAC requirements**: Pods need Lease permissions in their ServiceAccount.
- **Three election backends**: K8s Lease, Redis SETNX, and singleton. More
  code paths to test.

## Alternatives Considered

### Alternative A: Keep PostgreSQL Election

- **Pros**: Works today; no new dependencies.
- **Cons**: Unnecessary DB dependency; not K8s-native.
- **Rejected**: Adds unnecessary coupling to PostgreSQL.

### Alternative B: etcd-Based Election

- **Pros**: Purpose-built for distributed coordination.
- **Cons**: Requires running etcd (or using K8s's etcd, which isn't directly
  accessible). Additional infrastructure.
- **Rejected**: K8s Lease API provides etcd-backed election without direct
  etcd access.

## References

- `src/litellm_llmrouter/leader_election.py` — Current implementation
- K8s Lease API: https://kubernetes.io/docs/concepts/architecture/leases/
- [ADR-0004: asyncpg Connection Pooling](0004-asyncpg-connection-pooling.md)
- [ADR-0011: Pluggable External Services](0011-pluggable-external-services.md)
