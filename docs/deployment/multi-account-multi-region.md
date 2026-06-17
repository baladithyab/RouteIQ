# Multi-Account and Multi-Region Deployment

RouteIQ ships three default-OFF, operator-gated capabilities for running across
multiple AWS accounts and regions:

1. **Per-request region-aware / data-residency routing** (RouteIQ-60cc) — a
   dataplane candidate filter that keeps a request in-region.
2. **Service-managed StackSet capacity onboarding** (RouteIQ-ea99) — onboard
   capacity accounts across an AWS Organizations OU.
3. **Multi-region active-active / DR topology** (RouteIQ-6fd3) — Aurora Global
   Database, ElastiCache Global Datastore, and a Route53/Global Accelerator edge.

Everything is OFF by default. The CDK constructs synthesize zero resources when
their flags are off (the P0/P1/P2 template snapshots stay byte-stable). The live
AWS enablement steps that cannot be expressed in a member-deploy stack are
called out as **OPERATOR STEP** below.

---

## 1. Region-aware / data-residency routing (RouteIQ-60cc)

Bedrock cross-region inference profiles are already used statically (each
`model_list` arm pins `litellm_params.aws_region_name`). This feature adds a
**per-request** region preference plus a **hard residency constraint** on top, as
a pre-scoring candidate filter on the same `filter_routable_candidates` seam every
RouteIQ routing strategy uses — so it composes with the gov-ban and cooldown
filters and benefits every strategy without a new strategy subclass.

The filter only sees the per-request region signal when the **request context is
threaded** into `filter_routable_candidates(router, candidates, context=...)`.
That context is now passed at **every** RouteIQ candidate source — not just the
deterministic strategies — so a hard-residency request cannot leak out-of-region
on any path:

| Path | Candidate source | Context threaded |
|------|------------------|------------------|
| DEFAULT strategy | `strategy_registry.DefaultStrategy._get_deployments` | yes |
| ML (`LLMRouterStrategyFamily`) | `custom_routing_strategy._get_model_list` (built from `request_kwargs`) | yes |
| Personalized re-ranking | `custom_routing_strategy._get_model_list` | yes |
| Centroid (zero-config) | `centroid_routing._get_healthy_deployments` + `_fallback_deployment` | yes |
| Kumaraswamy-Thompson / LinUCB bandit | `kumaraswamy_thompson.select_deployment` | yes |
| Tag / regex / cost-aware deterministic strategies | `strategies.py` | yes |

A path that omits the context (or a request with no region token) is a
byte-stable no-op — the region filter is inert and the candidate set is returned
unchanged.

### How the region is resolved (per request, first match wins)

1. The request header named by `region_header` (default `X-RouteIQ-Region`,
   case-insensitive) — e.g. `X-RouteIQ-Region: eu`.
2. An explicit `region` / `residency_region` / `aws_region` key in the request
   `request_kwargs` or `metadata` (e.g. a governance-workspace region the auth
   layer stamps onto request metadata).

A request is **hard-constrained** (data residency) when either the matched source
carries a truthy residency flag (`residency` / `data_residency` / `hard_region`)
or `hard_residency_default` is `true`.

### Semantics

| Request | Behaviour |
|---------|-----------|
| Hard residency, in-region arm exists | Routes in-region. |
| Hard residency, **no** in-region arm | Candidate set is **emptied** → the strategy returns `None` and a fallback triggers. **Never** routes out-of-region. |
| Soft preference, in-region arm exists | Prefers the in-region subset. |
| Soft preference, no in-region arm | Falls back to the full set (availability over locality). |
| No region token, or feature disabled | No-op (byte-stable; the input set is returned unchanged). |

A candidate's region is read from `litellm_params.aws_region_name`, then
normalized through `region_map` (a request token like `eu` maps to one or more
concrete AWS regions). A token absent from `region_map` is matched verbatim
against the candidate region, so an exact-region token (`eu-west-1`) needs no map
entry.

### Configuration

```yaml
# settings (env: ROUTEIQ_REGION_ROUTING__*)
region_routing:
  enabled: true
  region_header: "X-RouteIQ-Region"
  region_map:
    eu: ["eu-west-1", "eu-central-1"]
    us: ["us-east-1", "us-west-2"]
  hard_residency_default: false
```

Environment variable form (`__` nested delimiter):

```bash
ROUTEIQ_REGION_ROUTING__ENABLED=true
ROUTEIQ_REGION_ROUTING__REGION_HEADER=X-RouteIQ-Region
ROUTEIQ_REGION_ROUTING__REGION_MAP='{"eu":["eu-west-1","eu-central-1"]}'
ROUTEIQ_REGION_ROUTING__HARD_RESIDENCY_DEFAULT=false
```

Default OFF / identity: with `enabled: false` the filter is a byte-stable no-op.

---

## 2. Service-managed StackSet capacity onboarding (RouteIQ-ea99)

`OrganizationsStackSetStack` (`deploy/cdk/lib/organizations_stackset_stack.py`)
declares one service-managed `AWS::CloudFormation::StackSet` that fans the
cross-account Bedrock capacity role out to every member account in one or more
Organizational Units — and, with auto-deploy on, to any account later moved into
those OUs. It is the org-native alternative to running a per-account
`cdk deploy` of `BedrockCapacityMemberStack`.

The StackSet's embedded member template mints the **same** stable-named
`RouteIqBedrockCapacity-<env>` role the standalone member stack mints: it trusts
the home gateway pod-role ARN via plain `sts:AssumeRole` (no web-identity — the
Pod-Identity home cluster has no OIDC issuer) with the 4-action Bedrock invoke
contract. The stable name keeps the member ARN predictable so the home grant
(RouteIqStack `capacity_account_ids`) and LiteLLM `aws_role_name` reference it.

### When to use which onboarding path

| Path | Reaches | Onboarding |
|------|---------|------------|
| `BedrockCapacityMemberStack` (per-account) | Any account, **including standalone accounts outside the org** | One `cdk deploy` per account with that account's credentials. |
| `OrganizationsStackSetStack` (this) | Only accounts **inside** the home Organization | Enroll an OU once; new accounts auto-onboard. |

### OPERATOR STEP — enable trusted access (one-time, management account)

Service-managed StackSets require AWS Organizations *trusted access* for
CloudFormation StackSets. This is a one-time toggle in the management account
that **cannot** be expressed in this stack:

```bash
aws cloudformation activate-organizations-access
# or: Console → CloudFormation → StackSets → "Activate trusted access"
```

### Deploy (operator opts in)

This stack is **not** instantiated by the default `app.py` synth. Instantiate it
in a deploy app from the org management account:

```python
OrganizationsStackSetStack(
    app, "RouteIqOrgStackSet-prod",
    env=env, env_name="prod",
    home_pod_role_arn="arn:aws:iam::<home-acct>:role/RouteIqStack-prod-PodRole-...",
    organizational_unit_ids=["ou-abcd-12345678"],
    auto_deploy=True,
)
```

Then wire the home grant: set `routeiq:capacity_account_ids` to the onboarded
account ids so the home pod role gains `sts:AssumeRole` on the computed member
ARNs, and add the per-account `model_list` rows with
`litellm_params.aws_role_name` pointing at each member ARN.

---

## 3. Multi-region active-active / DR (RouteIQ-6fd3)

`MultiRegionDrStack` (`deploy/cdk/lib/multi_region_dr_stack.py`) declares the AWS
primitives a multi-region deployment needs, on top of the single-region HA
primitives already shipped (the P1 `RouteIqStateStack` Aurora reader +
ElastiCache Serverless cache). Each primitive is independently flag-gated; an
all-defaults synth emits **zero** DR resources.

| Primitive | Resource | Flag |
|-----------|----------|------|
| Aurora Global Database | `AWS::RDS::GlobalCluster` | `enable_aurora_global` |
| ElastiCache Global Datastore | `AWS::ElastiCache::GlobalReplicationGroup` | `enable_cache_global` |
| Route53 failover DNS | `AWS::Route53::HealthCheck` + 2× `RecordSet` | `edge_mode="route53"` |
| Global Accelerator | `AWS::GlobalAccelerator::Accelerator` + `Listener` | `edge_mode="global_accelerator"` |

### Divergence — ElastiCache Global Datastore is node-based only

Global Datastore is **not supported on serverless caches**. The P1 state cache is
serverless (ADR-0029), so `enable_cache_global` attaches a **node-based** primary
replication group the operator supplies, not the P1 serverless cache. Treat the
node-based cache as a separate, operator-provisioned primary.

### Aurora Global Database

`enable_aurora_global` attaches the primary-region Aurora cluster
(`source_db_cluster_identifier` = the P1 `RouteIqStateStack` cluster id) as the
global cluster's source writer. The global cluster is storage-encrypted and
deletion-protected.

**OPERATOR STEP** — the secondary-region read-replica cluster + instance that
joins the global cluster is a separate per-region deploy in the secondary region
(it cannot be authored in the primary-region stack).

### Edge failover

- **`route53`**: a health check on the primary endpoint (`/_health/ready` over
  HTTPS:443) drives a PRIMARY/SECONDARY failover routing policy. The PRIMARY
  record serves while healthy; the SECONDARY (no health check) is the fallback.
- **`global_accelerator`**: a Global Accelerator + a TCP/443 listener provide the
  static-anycast-IP entry point. **OPERATOR STEP** — the per-region endpoint
  groups (weighted active-active or PRIMARY/SECONDARY) reference per-region
  ALB/NLB ARNs that do not exist at synth, so they are attached per-region by the
  operator.

### Failover runbook (OPERATOR)

1. Confirm cross-region replication lag is within RPO (Aurora Global Database
   metrics; ElastiCache Global Datastore lag).
2. For an unplanned primary-region outage, **fail over the Aurora Global
   Database** (managed promotion of the secondary cluster to a standalone
   writer), then point the gateway's `DATABASE_URL` at the promoted writer in the
   secondary region.
3. Promote the ElastiCache Global Datastore secondary, then point `REDIS_HOST` at
   the promoted endpoint.
4. Edge cutover is automatic: Route53 fails over when the primary health check
   trips; Global Accelerator shifts traffic when the primary endpoint group is
   unhealthy or de-weighted.
5. Pre-warm / scale up the secondary-region EKS data plane before cutover.

### Deploy (operator opts in)

This stack is **not** in the default `app.py` synth. Instantiate it with only the
primitives you need:

```python
MultiRegionDrStack(
    app, "RouteIqDr-prod",
    env=env, env_name="prod",
    enable_aurora_global=True,
    source_db_cluster_identifier="routeiq-prod-aurora",
    edge_mode="global_accelerator",
)
```

---

## Testing

All halves are credential-free:

- Region routing: `uv run pytest tests/unit/test_region_routing.py`
- CDK constructs (from `deploy/cdk/`):
  `uv run --extra cdk pytest tests/unit/test_organizations_stackset.py tests/unit/test_multi_region_dr.py`

The CDK template snapshots (`deploy/cdk/tests/snapshot/`) stay green because
neither new stack is in the default `app.py` synth.
