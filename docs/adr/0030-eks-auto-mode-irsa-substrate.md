# ADR-0030: EKS Auto Mode + IRSA as the Deployment Substrate

**Status**: Proposed
**Date**: 2026-06-14
**Decision Makers**: RouteIQ Core Team

## Context

RouteIQ ships a complete Helm chart (`deploy/charts/routeiq-gateway/`) — 15
templates including `serviceaccount.yaml`, `deployment.yaml`, `hpa.yaml`,
`pdb.yaml`, `networkpolicy.yaml`, `servicemonitor.yaml`, `prometheusrule.yaml`,
`externalsecret.yaml`, `leader-election-rbac.yaml`, and `ingress.yaml`. The chart
**presupposes a Kubernetes cluster it does not create**, and presupposes
cluster-side machinery the chart cannot provision: an IRSA-annotated
ServiceAccount, an external-secrets ClusterSecretStore, a Prometheus operator for
the ServiceMonitor, and a load balancer for the ingress/service. On AWS, the
operator stands all of this up by hand (`docs/deployment/aws.md`).

This is RouteIQ's largest gap: **a chart with no cluster under it.** The peer
ADRs in this set (0026 AppConfig, 0027 AMP/AMG, 0028 Aurora, 0029 ElastiCache)
all assume a provisioned cluster with IRSA so workloads can call AWS services
without static keys. This ADR provisions that cluster.

vllm-sr-on-aws provisions it in `cdk/lib/eks_cluster_construct.py`: an EKS Auto
Mode cluster (L1 `CfnCluster`) with an OIDC provider and an IRSA factory, where
all Kubernetes objects are applied out-of-band via kubectl/Helm. That boundary —
**CDK provisions cluster + IAM + OIDC; Helm deploys the app** — is exactly how
RouteIQ's chart should land.

## Decision

Provision an **EKS Auto Mode** cluster in IaC with an **IRSA factory**, and
deploy RouteIQ's existing Helm chart onto it. CDK owns the cluster, the OIDC
provider, the node/cluster IAM roles, and a per-workload IRSA role; the chart and
all manifests are applied by kubectl/Helm in CI, never by CDK.

### EKS Auto Mode cluster (`eks_cluster_construct.py:151-210`)

Use the **L1 `eks.CfnCluster`** (Auto Mode is not on the stable L2, and the alpha
L2 is breaking-change-prone — `:16-22`). K8s 1.33 (`:62,155`). Three Auto Mode
blocks toggled together (`:174-187`):

- **ComputeConfig** `enabled=True`, `node_pools=["general-purpose","system"]`,
  `node_role_arn=<node role>` (`:175-179`) — Auto Mode manages capacity; the two
  AWS-managed node pools collapse the Karpenter templates RouteIQ would otherwise
  hand-author.
- **StorageConfig.blockStorage** `enabled=True` (`:180-182`).
- **KubernetesNetworkConfig.elasticLoadBalancing** `enabled=True` (`:183-187`).

`bootstrap_self_managed_addons=False` (`:198`) since Auto Mode supplies
CoreDNS/kube-proxy/VPC-CNI. `access_config.authentication_mode="API"` with
`bootstrap_cluster_creator_admin_permissions=True` (`:192-195`); operator/CI
identities each get an explicit `CfnAccessEntry` (`:311-344`) — the creator-admin
flag only covers the CFN exec role, not a human's kubectl identity.

### OIDC provider + IRSA factory — the CfnJson gotcha (`eks_cluster_construct.py:219-309`)

An `iam.OpenIdConnectProvider` from the cluster's issuer attribute with
`client_ids=["sts.amazonaws.com"]` (`:219-224`). The construct exposes
`oidc_provider_issuer` — the **scheme-stripped, ARN-derived** issuer (`:226-240`).

**The load-bearing gotcha** (`:275-300`): an IRSA trust policy uses the OIDC
issuer as a **map key** (`<oidc>:sub` / `<oidc>:aud`), and CloudFormation map
keys must be literal strings at synth — a `Fn::GetAtt` token cannot be a key. The
fix wraps the inner condition map in **`CfnJson`** so the keys resolve at deploy
time:

```python
string_equals = CfnJson(self, f"{cid}TrustKeys", value={
    f"{oidc}:aud": "sts.amazonaws.com",
    f"{oidc}:sub": f"system:serviceaccount:{ns}:{sa}",
})
role = iam.Role(..., assumed_by=iam.WebIdentityPrincipal(
    self.oidc_provider_arn, conditions={"StringEquals": string_equals}))
```

Two silent-failure traps the construct documents (`:231-237,281-286`): (a)
`issuer_url.replace("https://","")` is a **silent no-op on an unresolved CFN
token** — it produced `https://...:sub` keys and `AssumeRoleWithWebIdentity`
AccessDenied (root-caused live 2026-06-08); always use the ARN-derived
`oidc_provider_issuer`. (b) The L2 `cluster.addServiceAccount` would hide all of
this, but the L1 `CfnCluster` required for Auto Mode forces the manual CfnJson
path.

For RouteIQ, an `irsa_role(namespace, service_account, ...)` factory mints one
role per workload SA. The chart's `serviceaccount.yaml` is annotated with that
role ARN (`eks.amazonaws.com/role-arn`), and the role is granted exactly what the
peer ADRs need: `aps:RemoteWrite` (ADR-0027), `rds-db:connect` (ADR-0028),
`elasticache:Connect` (ADR-0029), AppConfig read (ADR-0026), Bedrock invoke.

### CDK/Helm boundary (`eks_cluster_construct.py:24-28,242-259`)

This construct **does not** apply Kubernetes objects — no CDK `KubernetesManifest`
or `HelmChart`. It emits CfnOutputs (`ClusterName`, `ClusterEndpoint`,
`OidcProviderArn`, `OidcProviderIssuerUrl`, the per-SA IRSA role ARN) (`:242-259`)
that RouteIQ's Helm chart consumes. The chart is applied by `helm upgrade` /
`kubectl` in CI — the boundary RouteIQ already implicitly assumes.

### Observability already wired (`eks_cluster_construct.py:346-967`)

The construct's `enable_container_insights` provisions the
`amazon-cloudwatch-observability` addon, a CDK-created routing log group, the
**per-model dimensioned `routing_latency_ms_by_model` metric filter**
(`:757-767`, see ADR-0027), 7 alarms, a dashboard, and an SNS topic. RouteIQ's
chart `servicemonitor.yaml` / `prometheusrule.yaml` complement this.

### Three HA/ops lessons for the chart (app-layer, from `../architecture/aws-rearchitecture/vllmsr-patterns.md`)

These live in the manifests/deploy scripts, not in the CDK construct, but they
gate a working RouteIQ deploy on Auto Mode:

1. **NLB CIDR-lock is `spec.loadBalancerSourceRanges`, NOT the AWS LB
   annotation.** Under Auto Mode's managed load balancing, the source-ranges spec
   field is honored; the legacy `service.beta.kubernetes.io/...` allowlist
   annotation is not. RouteIQ's `service.yaml` (LoadBalancer type) must use
   `loadBalancerSourceRanges`.
2. **RWO EBS volume + `RollingUpdate` = Multi-Attach deadlock → use `Recreate`.**
   A ReadWriteOnce EBS PVC with a RollingUpdate strategy deadlocks (old pod holds
   the volume, new pod cannot attach). Any RouteIQ Deployment that mounts an RWO
   EBS PVC must set `strategy.type: Recreate`.
3. **`kubectl apply -k` (kustomize) clobbers the IRSA SA annotation.** Kustomize
   overwrites the ServiceAccount's `role-arn` annotation, breaking IRSA. Deploy
   scripts must `envsubst` the annotation in (or use `helm upgrade`, which
   templates it), never a bare `apply -k` on the SA.

## Consequences

### Positive

- **The chart finally has a cluster.** RouteIQ's existing 15-template chart
  deploys onto a provisioned EKS Auto Mode cluster with no manual setup.
- **Auto Mode removes node ops.** No Karpenter templates, no node group sizing,
  no CoreDNS/CNI/kube-proxy management.
- **IRSA, no static keys.** Every AWS dependency in ADRs 0026-0029 is reached via
  a per-SA IRSA role; nothing needs a static AWS key.
- **Clean CDK/Helm split.** CDK owns infra + IAM; Helm owns the app — matching
  RouteIQ's chart-first reality and keeping app deploys fast and out-of-band.

### Negative

- **L1 `CfnCluster` ergonomics.** Auto Mode forces the L1 escape hatch and the
  manual CfnJson IRSA trust map — more verbose than the L2 `addServiceAccount`.
- **Two-tool deploy.** CDK for infra + Helm/kubectl for the app means two deploy
  steps and access-entry management for the CI identity.
- **Three app-layer foot-guns.** The CIDR-lock field, RWO+Recreate, and
  apply-k-clobbers-SA lessons must be encoded in the chart/deploy scripts or the
  deploy silently breaks.

## Alternatives Considered

### Alternative A: Manual `eksctl` / console cluster + manual Helm (status quo)

- **Pros**: No CDK.
- **Cons**: 100% manual; no reproducible IRSA, OIDC, or access entries — the gap.
- **Rejected**: Manual provisioning is the gap.

### Alternative B: Standard EKS managed node groups (not Auto Mode)

- **Pros**: Stable L2 `eks.Cluster` with `addServiceAccount` hides the CfnJson
  gotcha; mature.
- **Cons**: RouteIQ must own Karpenter/node-group capacity, CNI/addon lifecycle,
  and scaling — exactly the ops Auto Mode removes.
- **Rejected**: Auto Mode's managed capacity is the point; the CfnJson cost is
  one-time and documented.

### Alternative C: ECS Fargate (no Kubernetes)

- **Pros**: No cluster ops at all; simplest substrate.
- **Cons**: Throws away RouteIQ's entire Helm chart, ServiceMonitor,
  leader-election RBAC, and PDB; ADR-0015 (K8s Lease leader election) presumes
  K8s. A different deployment model, not a home for the chart.
- **Rejected**: RouteIQ is K8s-native by design (ADR-0015); the chart is the
  asset to deploy.

## References

- `cdk/lib/eks_cluster_construct.py` (vllm-sr-on-aws) — L1 `CfnCluster` + 3 Auto
  Mode blocks (`:151-210`), OIDC provider (`:219-224`), IRSA factory + CfnJson
  trust-key gotcha + `.replace("https://")` no-op lesson (`:261-309,231-237`),
  access entries (`:311-344`), CDK/Helm boundary + CfnOutputs (`:24-28,242-259`),
  Container Insights + per-model filter (`:346-967,757-767`)
- `../architecture/aws-rearchitecture/vllmsr-patterns.md` — "eks_cluster_construct.py: EKS Auto Mode (L1
  CfnCluster, 3 Auto-Mode blocks, IRSA via CfnJson trust-key, OIDC provider...)";
  HA/ops lessons: CIDR-lock=`loadBalancerSourceRanges`,
  RWO-EBS+RollingUpdate→`Recreate`, `apply -k` clobbers SA annotation
- `deploy/charts/routeiq-gateway/templates/` — the chart that deploys onto this
  cluster (`serviceaccount.yaml`, `service.yaml`, `servicemonitor.yaml`, …)
- [ADR-0015: K8s-Native Leader Election via Lease API](0015-k8s-native-leader-election.md)
- [ADR-0026: AppConfig GitOps Config Delivery](0026-appconfig-gitops-config-delivery.md)
- [ADR-0027: AWS-Native Observability (AMP/AMG/CloudWatch)](0027-otel-amp-amg-observability-on-aws.md)
- [ADR-0028: Aurora PostgreSQL Serverless v2 for State](0028-aurora-postgres-serverless-v2-state.md)
- [ADR-0029: ElastiCache Serverless (Valkey) for Cache](0029-elasticache-serverless-valkey-cache.md)
