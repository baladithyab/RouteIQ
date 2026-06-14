# ADR-0029: ElastiCache Serverless (Valkey) for RouteIQ Cache + Rate-Limit State

**Status**: Proposed
**Date**: 2026-06-14
**Decision Makers**: RouteIQ Core Team

## Context

Redis is load-bearing in RouteIQ. The singleton client
([ADR-0005](0005-redis-singleton.md), `redis_pool.py`) backs at least six
subsystems: health probes, quota enforcement, MCP gateway HA sync, the cache
plugin, conversation affinity, and (optionally) leader election. The externalized
multi-worker state ([ADR-0021](0021-externalized-state.md)) — `SessionCache`
routing affinity and the shared `CircuitBreaker` — also rides Redis. Rate limiting
runs atomic counters via a Redis Lua `INCRBY` + `EXPIRE` script
(`quota.py:299-319`), and the semantic cache uses Redis as its L2 vector store
(`semantic_cache.py`, `settings.py:790`).

But RouteIQ provisions no managed Redis. `docs/deployment/aws.md` hand-runs
`aws elasticache create-*`, and `redis_pool.py` even warns that a
`REDIS_PASSWORD` env var is weaker than IAM auth (`0005-redis-singleton.md`
"Password Security"). There is no IAM-auth user group, no KMS, no TLS-required
endpoint, and no provisioned scale-to-zero cache.

vllm-sr-on-aws provisions exactly this in `cdk/lib/cache_construct.py`: an
**ElastiCache Serverless (Valkey)** cache with an IAM-auth user group, customer
KMS key, and always-on TLS — having migrated *off* a node-based replication group
(~$70-100/mo) to serverless (~$7/mo idle). RouteIQ should adopt the same
construct as the managed home for everything `redis_pool.py` touches.

## Decision

Provision RouteIQ's Redis-backed state on **ElastiCache Serverless (Valkey)**,
following `cache_construct.py`.

### The serverless cache (`cache_construct.py:194-215`)

An L1 `elasticache.CfnServerlessCache` with `engine="valkey"` (Valkey 8.x;
wire-compatible with the `redis` client `redis_pool.py` uses) and
`major_engine_version="8"` — serverless takes the **major version only**
(`:195-204`). Name `routeiq-{env}-cache-sl`. Auto-scaling, multi-AZ, no node
sizing/replica/snapshot management.

### IAM-auth user group (`cache_construct.py:119-178`)

The security upgrade ADR-0005 asked for. A `CfnUser` in `authentication_mode={"Type":"iam"}`
with `access_string="on ~* +@all"` (`:125-133`) — **IAM auth requires
`user_id == user_name`** (`:119-123`). The mandatory `default` user is created
disabled (forced PASSWORD mode with an inert placeholder and `access_string="off ~keys* -@all"`,
because IAM and `no_password_required` are both rejected on the default user,
`:139-166`). A `CfnUserGroup` binds both users (`:168-174`) with explicit
`add_dependency` on each (CDK does not infer ordering from the `user_ids` string
list, `:177-178`).

The app-side contract: RouteIQ sets `SR_CACHE_USER` (or the RouteIQ equivalent)
to **exactly the IAM user name** so SigV4 IAM auth resolves the signing principal
onto this user (`:42-47,119-123`). `redis_pool._redis_settings()` (today reading
`REDIS_PASSWORD`, `0005-redis-singleton.md`) gains an IAM-auth path that mints a
short-lived auth token instead of a static password — closing the ADR-0005
password-logging risk.

### KMS + TLS (`cache_construct.py:210`, docstring `:13-14,24`)

At-rest encryption via `kms_key_id=<customer CMK>` (`:210`). **TLS is always
required** on serverless caches — it is not a toggle, so `redis_pool.py` must set
`REDIS_SSL=true` (`settings.py:90-91`). This makes in-transit encryption
non-optional, another ADR-0005 improvement.

### Networking + discovery (`cache_construct.py:108-117,208-209,230-265`)

Serverless takes a flat `subnet_ids` list (private), **not** a `CfnSubnetGroup`
ref (`:180-183`), and `security_group_ids` from an own SG (`:208-209`). Ingress
tcp/6379 from the app SG is wired late via `attach_dependencies(sr_task_sg=...)`,
idempotency-guarded (`:248-265`). The endpoint is exposed as
`attr_endpoint_address` / `attr_endpoint_port` (`:230-231`) for the app's
`REDIS_HOST`/`REDIS_PORT`; ARNs (`cache_arn`, `iam_user_arn`) feed the
`elasticache:Connect` IAM grant (`:233-246`).

### Name-collision lesson (`cache_construct.py:185-194`)

If RouteIQ ever migrates an existing node-based replication group to serverless:
ElastiCache enforces name uniqueness **across cache types**, and CFN creates the
new resource before deleting the old — so a same-named serverless cache collides
with the still-`available` RG ("Serverless Cache already exists"). Use a
distinct suffix (e.g. `-sl`) on the serverless name so both coexist during the
create-then-delete window.

## Consequences

### Positive

- **IAM auth replaces `REDIS_PASSWORD`.** Closes the ADR-0005 password-logging
  risk; SigV4 identity instead of a static secret.
- **TLS + KMS by default.** In-transit encryption is mandatory on serverless;
  at-rest via customer CMK.
- **Scale-to-zero cost.** ~$7/mo idle vs ~$70-100/mo node-based, with no node
  sizing or failover management.
- **Drop-in for `redis_pool.py`.** The singleton (ADR-0005) and externalized
  state (ADR-0021) point at the serverless endpoint with one config change plus
  the IAM-token auth path; the Lua rate-limit script (`quota.py:299`) is
  unchanged.

### Negative

- **Client must support IAM-token + TLS.** `redis_pool._redis_settings()` needs
  an IAM-auth path (mint/refresh the token) and `REDIS_SSL=true`. (vllm-sr's own
  upstream Go client was blocked here until patched — RouteIQ's `redis-py`
  supports both, so this is a config change, not a code blocker.)
- **`user_id == user_name` constraint** must be honored in IaC and the app's
  `SR_CACHE_USER`-equivalent must match exactly, or auth fails opaquely.
- **Migration name collision** if replacing an existing RG — requires a suffix.

## Alternatives Considered

### Alternative A: Node-based ElastiCache replication group

- **Pros**: Mature; explicit replica/failover control.
- **Cons**: ~10x idle cost; manual node sizing, replica count, snapshot windows;
  vllm-sr explicitly migrated *away* from it.
- **Rejected**: Serverless removes the ops burden at a fraction of the cost.

### Alternative B: Self-managed Redis/Valkey on the cluster

- **Pros**: No managed-service cost.
- **Cons**: HA, persistence, upgrades, and TLS/auth all become RouteIQ's problem
  — the manual burden this ADR removes.
- **Rejected**: Managed service.

### Alternative C: Keep Redis password auth on a managed cache

- **Pros**: Smallest change to `redis_pool.py`.
- **Cons**: Retains the exact static-password risk ADR-0005 flagged; password in
  env/logs.
- **Rejected**: IAM-auth is the security win that justifies the move.

## References

- `cdk/lib/cache_construct.py` (vllm-sr-on-aws) — `CfnServerlessCache`
  (`:194-215`), IAM-auth user group (`:119-178`), KMS + always-on TLS
  (`:210,13-14`), networking + endpoint discovery (`:108-117,208-209,230-265`),
  RG→serverless name-collision lesson (`:185-194`)
- `../architecture/aws-rearchitecture/vllmsr-patterns.md` — "cache_construct.py: ElastiCache Serverless Valkey
  (IAM-auth user group, KMS)"
- `src/litellm_llmrouter/redis_pool.py` — Redis singleton (`REDIS_PASSWORD` →
  IAM-token path)
- `src/litellm_llmrouter/quota.py` — Lua `INCRBY`/`EXPIRE` rate-limit script
  (`:299-319`)
- `src/litellm_llmrouter/semantic_cache.py` — L2 Redis vector cache
- [ADR-0005: Redis Singleton Client](0005-redis-singleton.md)
- [ADR-0021: Externalize In-Process State to Redis](0021-externalized-state.md)
- [ADR-0030: EKS Auto Mode + IRSA Deployment Substrate](0030-eks-auto-mode-irsa-substrate.md)
