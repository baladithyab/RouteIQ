# ADR-0028: Aurora PostgreSQL Serverless v2 for RouteIQ Persistent State

**Status**: Proposed
**Date**: 2026-06-14
**Decision Makers**: RouteIQ Core Team

## Context

RouteIQ persists its control-plane state — API keys, spend/usage records,
governance (Org/Workspace/Key), audit log — in PostgreSQL. The application layer
is already built for this: `database.py` opens a single shared `asyncpg.Pool`
from `DATABASE_URL` (`database.py:39,86,106`, pooling per
[ADR-0004](0004-asyncpg-connection-pooling.md)), `migrations.py` runs schema
migrations against it (`database.py:476`), and `governance.py` / `audit.py` /
`quota.py` read and write through it.

But RouteIQ provisions **zero** of this database. `docs/deployment/aws.md` tells
the operator to hand-run `aws rds create-*` (or BYO Postgres) and paste the
resulting `DATABASE_URL`. There is no managed cluster, no IAM auth, no KMS
encryption, no rotation, and no schema bootstrap — all of which a production AWS
deployment needs.

vllm-sr-on-aws provisions exactly this in `cdk/lib/replay_store_construct.py`:
an Aurora PostgreSQL Serverless v2 cluster with scale-to-zero, IAM auth, a
customer-managed KMS key, 30-day secret rotation, and a Lambda-backed
schema-bootstrap custom resource. RouteIQ should adopt the same construct shape
as its managed-DB layer.

## Decision

Provision RouteIQ's persistent state on **Aurora PostgreSQL Serverless v2**,
following `replay_store_construct.py`.

### Cluster + Serverless v2 scaling (`replay_store_construct.py:254-290`)

An L2 `rds.DatabaseCluster` with `aurora_postgres(version=VER_16_13)`
(`:254-257`). **Pin a CDK-enum *and* in-region engine version** — vllm-sr was
bitten live when AWS retired 15.4 ("Cannot find version 15.4"), which synth/nag
tests cannot catch (`:238-246`). Serverless v2 scaling via
`serverless_v2_min_capacity` / `serverless_v2_max_capacity` (`:267-268`), with
ACU bounds min `0.5` / max `2.0` (dev) or `8.0` (prod) (`:172-177`).

**Scale-to-zero** is opt-in: set `min_capacity=0` *with*
`serverless_v2_auto_pause_duration=Duration.hours(24)` — Aurora only allows
min-0 paired with an auto-pause window (`:269-279`). Note the cold-resume caveat:
the resume can exceed a short connect timeout, so a workload with a tight DB-init
timeout should keep `min_capacity=0.5`. A writer + one reader, both
`publicly_accessible=False` (`:280-290`).

### IAM database auth (`replay_store_construct.py:260,461-482`)

`iam_authentication=True` (`:260`) so the runtime presents an IAM identity, not a
static password. A `grant_iam_db_connect(role)` helper writes an `rds-db:connect`
statement scoped to
`arn:...:rds-db:...:dbuser:{cluster_resource_identifier}/{db_user}`
(`:473-479`). For RouteIQ the workload's Pod Identity role (ADR-0030) is granted, and
`database.py` builds a `DATABASE_URL` whose auth token is minted from IAM rather
than a static secret. The Secrets Manager credential remains for break-glass DBA
access only.

### KMS + rotation (`replay_store_construct.py:229-231,261-262,305-307`)

`storage_encrypted=True` + `storage_encryption_key=<customer CMK>` (`:261-262`);
the generated credentials secret is also CMK-encrypted (`:229-231`). Secret
rotation via `add_rotation_single_user(automatically_after=Duration.days(30))`
(`:305-307`) — a 30-day schedule with the L2-provisioned rotation Lambda.

### Schema bootstrap (`replay_store_construct.py:336-457`)

A Python 3.13 Lambda (`:381`) bundled with the `psycopg2-binary` manylinux wheel
via a docker-free local bundler (`:350-364`), invoked through a
`custom_resources.Provider` + `CustomResource` (`:426-453`). The handler runs
**idempotent `CREATE TABLE IF NOT EXISTS`** DDL (`:378-379`). For RouteIQ this
runs the **same migrations `migrations.py` runs** (`database.py:476`), so the
schema is created at deploy time inside the VPC rather than on first app boot. A
static `schema_version` custom-resource property is the re-run lever: bump it and
CFN issues an Update that re-runs the idempotent DDL (`:436-452`) — without an
`onUpdate`, the CR framework skips DDL on stack updates. Mind the dependency
cycle: `secret.grant_read` writes a `kms:Decrypt` grant naming the Lambda role,
so put the cluster dependency on the *custom resource*, not the role (`:454-457`).

### Networking + lifecycle (`replay_store_construct.py:183-205,291-300`)

Own SG, ingress tcp/5432 from the app SG only, plus a self-reference so the
bootstrap Lambda (same SG) can connect (`:183-205`); isolated subnets by default
(`:159`), `PRIVATE_WITH_EGRESS` where the VPC has no isolated tier (`:207-213`).
`deletion_protection` on in non-dev (`:296`); `RemovalPolicy.SNAPSHOT` in non-dev
(`:297-299`); 7/14-day backups (`:291-294`); `postgresql` log export (`:300`).

### Operational lesson — deploy DB changes separately

Aurora rollbacks are slow (~30 minutes) per the live-ops record
(`../architecture/aws-rearchitecture/vllmsr-patterns.md` HA/ops; memory `eks-live-stress-test-mlops-proven`).
RouteIQ's deploy pipeline must run **Aurora stack changes in a separate
pipeline/stage from the app deploy**, so a fast app rollback is never blocked
behind a slow database rollback.

## Consequences

### Positive

- **Managed, auto-scaling DB.** Scale-to-zero collapses idle cost; no instance
  sizing or failover management.
- **No static DB password on the hot path.** IAM auth + Pod Identity; the secret is
  break-glass only, and it rotates every 30 days.
- **Schema is deploy-time, in-VPC.** The bootstrap Lambda runs RouteIQ's
  migrations at deploy, removing the first-boot migration race and giving every
  replica a ready schema.
- **Drop-in for `database.py`.** RouteIQ's `asyncpg.Pool` only needs a
  `DATABASE_URL` (IAM-token-augmented); the application code is unchanged.

### Negative

- **Cold-resume latency** if `min_capacity=0` — keep `0.5` for latency-sensitive
  DB-init paths.
- **IAM-token auth wiring.** `database.py` must mint/refresh the IAM auth token
  (15-min lifetime) when building `DATABASE_URL`, vs a static URL today.
- **Slow rollback.** Aurora changes must be a separate deploy stage (~30-min
  rollback) so app rollbacks stay fast.
- **VPC-bound bootstrap Lambda** adds an in-VPC function + custom resource.

## Alternatives Considered

### Alternative A: BYO Postgres / manual `aws rds create-*` (status quo)

- **Pros**: Zero new IaC; works with any `DATABASE_URL`.
- **Cons**: 100% manual; no IAM auth, KMS, rotation, or schema bootstrap — the
  exact gaps this ADR closes.
- **Rejected**: Manual provisioning is the gap.

### Alternative B: RDS provisioned PostgreSQL (fixed instance)

- **Pros**: Simpler mental model; no resume latency.
- **Cons**: Pay for peak 24/7; manual instance sizing and failover config.
- **Rejected**: Serverless v2 scale-to-zero fits RouteIQ's spiky control-plane
  load better; the resume caveat is handled by `min_capacity=0.5`.

### Alternative C: DynamoDB / non-relational store

- **Pros**: Fully serverless, no VPC, no schema.
- **Cons**: RouteIQ's governance/spend/audit model and `migrations.py` are
  relational (asyncpg, SQL); LiteLLM's own key/spend tables are Postgres. A
  rewrite, not a deployment.
- **Rejected**: Out of scope — RouteIQ *is* a Postgres application
  (`database.py:106`).

## References

- `cdk/lib/replay_store_construct.py` (vllm-sr-on-aws) — Aurora Serverless v2
  cluster (`:254-290`), IAM auth (`:260,461-482`), KMS + 30-day rotation
  (`:229-231,261-262,305-307`), schema-bootstrap Lambda + custom resource
  (`:336-457`), engine-retirement lesson (`:238-246`), networking/lifecycle
  (`:183-205,291-300`)
- `../architecture/aws-rearchitecture/vllmsr-patterns.md` — "replay_store_construct.py: Aurora PostgreSQL
  Serverless v2 (scale-to-zero, IAM auth, KMS, 30d rotation, schema-bootstrap
  Lambda+custom resource)"; HA/ops: "Aurora rollback ~30min deploy CI separately"
- `src/litellm_llmrouter/database.py` — `asyncpg.Pool` from `DATABASE_URL`
- `src/litellm_llmrouter/migrations.py` — schema migrations (bootstrap reuse)
- [ADR-0004: asyncpg Connection Pooling](0004-asyncpg-connection-pooling.md)
- [ADR-0020: Governance Layer](0020-governance-layer.md)
- [ADR-0030: EKS Auto Mode + Pod Identity Deployment Substrate](0030-eks-auto-mode-irsa-substrate.md)
