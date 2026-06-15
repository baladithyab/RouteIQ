# ADR-0030: EKS Auto Mode + Pod Identity as the Deployment Substrate

**Status**: Proposed — **AMENDED 2026-06-14: pod IAM mechanism flipped IRSA → EKS Pod Identity** (see Amendment below); body reconciled with the shipped CDK 2026-06-15
**Date**: 2026-06-14
**Decision Makers**: RouteIQ Core Team

> ## ⚠️ Amendment (2026-06-14) — Pod Identity supersedes IRSA for the pod-IAM mechanism
>
> The P0 CDK Foundation proposal
> (`docs/architecture/aws-rearchitecture/31-p0-cdk-foundation-proposal.md`, §3) and its
> research report (`.../p0-discovery/research-report-irsa-vs-podidentity.md`,
> *"very high confidence"*) **supersede this ADR's IRSA decision with EKS Pod Identity**.
> What changes, and what does NOT:
>
> - **CHANGED — pod-IAM mechanism.** The pod→role binding is a single CDK-side
>   `eks.CfnPodIdentityAssociation` over a **static `pods.eks.amazonaws.com` trust** —
>   NOT IRSA. This **deletes** the `OpenIdConnectProvider`, the `oidc_provider_issuer`
>   derivation, and the **`CfnJson` token-keyed trust map** this ADR describes below.
>   The `irsa_role()` factory becomes a `pod_identity_association()` helper. A defensive
>   `eks-pod-identity-agent` `CfnAddon` (`resolve_conflicts="OVERWRITE"`, `DependsOn` the
>   cluster) is emitted so the association resolves regardless of the "built into Auto
>   Mode" claim (the production VSR construct installs it by hand at
>   `eks_cluster_construct.py:391-398` — verified).
> - **CHANGED — chart seam.** With Pod Identity the chart needs **NO
>   `eks.amazonaws.com/role-arn` ServiceAccount annotation**. The binding is keyed on a
>   stable `(namespace, serviceAccount)` pinned by the CDK. This **moots** lesson #2
>   (the `.replace("https://")` token no-op) and the IAM slice of lesson #6
>   (`apply -k` clobbers the SA annotation) below — there is no annotation to clobber.
> - **UNCHANGED.** Everything else in this ADR stands: the L1 `eks.CfnCluster`, the three
>   Auto Mode blocks, `bootstrap_self_managed_addons=False`, `CfnAccessEntry` for
>   cluster access, `enable_container_insights`, the CDK→Helm boundary, and lessons
>   #1/#3/#7/#8. The L1-vs-L2 rationale below should be read as *a choice* (aws-cdk-lib's
>   newer `aws-eks-v2` now offers Auto Mode on an L2), not a hard constraint — but L1 is
>   retained per the VSR port. Why Pod Identity wins on L1: the L1 path strips the L2
>   `addServiceAccount` helper that would hide IRSA's complexity, so on L1 the gap
>   between trivial Pod Identity and hand-rolled IRSA+CfnJson is at its widest.
>
> The filename (`0030-eks-auto-mode-irsa-substrate.md`) is **retained for link
> stability** — ADRs 0026/0027/0028/0029 all cross-link it, so the on-disk name is
> kept and only the display text was corrected to "Pod Identity" (the lower-risk
> path vs. a rename + four cross-link edits). The title and the body below have now
> been reconciled with the shipped CDK (`deploy/cdk/lib/`): Pod Identity, no OIDC
> provider, no `WebIdentityPrincipal`, no `CfnJson` trust map.

## Context

RouteIQ ships a complete Helm chart (`deploy/charts/routeiq-gateway/`) — 15
templates including `serviceaccount.yaml`, `deployment.yaml`, `hpa.yaml`,
`pdb.yaml`, `networkpolicy.yaml`, `servicemonitor.yaml`, `prometheusrule.yaml`,
`externalsecret.yaml`, `leader-election-rbac.yaml`, and `ingress.yaml`. The chart
**presupposes a Kubernetes cluster it does not create**, and presupposes
cluster-side machinery the chart cannot provision: a ServiceAccount bound to an
AWS IAM role, an external-secrets ClusterSecretStore, a Prometheus operator for
the ServiceMonitor, and a load balancer for the ingress/service. On AWS, the
operator stands all of this up by hand (`docs/deployment/aws.md`).

This is RouteIQ's largest gap: **a chart with no cluster under it.** The peer
ADRs in this set (0026 AppConfig, 0027 AMP/AMG, 0028 Aurora, 0029 ElastiCache)
all assume a provisioned cluster whose pods can call AWS services without static
keys. This ADR provisions that cluster.

vllm-sr-on-aws provisions it in `cdk/lib/eks_cluster_construct.py`: an EKS Auto
Mode cluster (L1 `CfnCluster`) with a pod→IAM binding, where all Kubernetes
objects are applied out-of-band via kubectl/Helm. That boundary — **CDK provisions
cluster + IAM; Helm deploys the app** — is exactly how RouteIQ's chart should
land. (VSR's reference used an OIDC provider + IRSA factory; RouteIQ's shipped
port realizes the same pod→IAM binding via EKS Pod Identity — see the Amendment
and the Decision below.)

## Decision

Provision an **EKS Auto Mode** cluster in IaC with an **EKS Pod Identity** pod→IAM
binding, and deploy RouteIQ's existing Helm chart onto it. CDK owns the cluster,
the node/cluster IAM roles, the defensive `eks-pod-identity-agent` add-on, and a
per-workload pod role + `CfnPodIdentityAssociation`; the chart and all manifests
are applied by kubectl/Helm in CI, never by CDK.

### EKS Auto Mode cluster (`deploy/cdk/lib/eks_cluster_construct.py:161-208`)

Use the **L1 `eks.CfnCluster`** (Auto Mode is not on the stable L2, and the alpha
L2 is breaking-change-prone). K8s 1.33 (`:67`). Three Auto Mode blocks toggled
together (`:182-194`):

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

### Pod Identity binding — no OIDC, no CfnJson (`eks_cluster_construct.py:254-325`)

The shipped construct binds pods to IAM via **EKS Pod Identity**, not IRSA. Two
pieces:

1. A defensive **`eks-pod-identity-agent` `CfnAddon`** (`:263-270`,
   `resolve_conflicts="OVERWRITE"`, `DependsOn` the cluster). AWS docs claim the
   agent is built into Auto Mode, but the production VSR construct installs it by
   hand (`:391-398`, verified), so RouteIQ emits it explicitly and idempotently —
   re-applying a built-in add-on is then a harmless no-op.
2. A **`pod_identity_association(namespace, service_account, role)`** helper
   (`:285-325`) that emits one `eks.CfnPodIdentityAssociation` keyed on
   `(namespace, serviceAccount)`. It `DependsOn` the agent add-on so it never
   races the agent (`:324`).

The binding rests on a **static `pods.eks.amazonaws.com` service-principal
trust** — the pod role's trust document grants `sts:AssumeRole` + `sts:TagSession`
to `pods.eks.amazonaws.com` (`routeiq_stack.py:177-191`). There is **no
`OpenIdConnectProvider`, no `oidc_provider_issuer` derivation, no `CfnJson`
token-keyed trust map, and no `WebIdentityPrincipal`** — and because creds come
from the pod-identity agent rather than an STS web-identity exchange,
`sts:AssumeRoleWithWebIdentity` is **not** granted. That entire IRSA surface
(OIDC provider, the `CfnJson(...)` condition-key wrapper, the
`.replace("https://")` issuer-derivation no-op, the L2 `addServiceAccount`
trade-off) is **deleted** on the Pod Identity path; it survives only as the
*rejected* VSR reference (`irsa_role()` factory) the Amendment supersedes.

RouteIQ provisions a **single** pod role (`routeiq_stack.py:177-206`) bound to the
gateway ServiceAccount, granted exactly what the peer ADRs need: `aps:RemoteWrite`
(ADR-0027), `rds-db:connect` (ADR-0028), `elasticache:Connect` (ADR-0029),
AppConfig read (ADR-0026), Bedrock invoke. The chart's `serviceaccount.yaml`
needs **no `eks.amazonaws.com/role-arn` annotation** — the CDK-pinned
`(namespace, serviceAccount)` is the entire key.

### CDK/Helm boundary (`eks_cluster_construct.py:276-283`, `routeiq_stack.py:376-379`)

This construct **does not** apply Kubernetes objects — no CDK `KubernetesManifest`
or `HelmChart`. It emits CfnOutputs (`ClusterName`, `ClusterEndpoint`,
`NodeRoleName`, `KubectlConfigCommand`) (`eks_cluster_construct.py:276-283`) that
RouteIQ's Helm chart / CI consume. The IRSA-only `OidcProviderArn` /
`OidcProviderIssuerUrl` outputs are **dropped** — there is no OIDC provider on the
Pod Identity path (`routeiq_stack.py:376-379`), and the pod→role binding is the
CDK-side association, not a chart-consumed role ARN. The chart is applied by
`helm upgrade` / `kubectl` in CI — the boundary RouteIQ already implicitly
assumes.

### Observability — P0-minimal, the rest deferred to P2 (`eks_cluster_construct.py:327-378`)

The shipped `enable_container_insights` is **P0-minimal**: it installs the
`amazon-cloudwatch-observability` add-on (per-pod/per-container Container Insights
+ Fluent Bit log forwarding) — its CloudWatch agent gets permissions via a Pod
Identity association, reusing the same helper (`:374-378`) — and **CDK-creates the
routing log group** the P2 metric filters attach to. It ships **no alarms, no
dashboard, no SNS topic, and no routing MetricFilter** (`:338`); the per-model
dimensioned filter and the CW-Logs-native alarms are **owned by P2**
(`ObservabilityConstruct`, ADR-0027) — which ships **3** routing alarms, not the
VSR ECS-substrate set. RouteIQ's chart `servicemonitor.yaml` /
`prometheusrule.yaml` complement this.

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
3. **(Mooted under Pod Identity) `kubectl apply -k` clobbering the SA role-arn
   annotation.** Under IRSA, kustomize overwrote the ServiceAccount's
   `eks.amazonaws.com/role-arn` annotation and broke the binding. With Pod
   Identity the binding is keyed on `(namespace, serviceAccount)` by the CDK-side
   association and there is **no SA annotation to clobber**, so this foot-gun no
   longer applies (see the Amendment). Retained here for the IRSA-era record.

## Consequences

### Positive

- **The chart finally has a cluster.** RouteIQ's existing 15-template chart
  deploys onto a provisioned EKS Auto Mode cluster with no manual setup.
- **Auto Mode removes node ops.** No Karpenter templates, no node group sizing,
  no CoreDNS/CNI/kube-proxy management.
- **Pod Identity, no static keys.** Every AWS dependency in ADRs 0026-0029 is
  reached via the pod role bound by a `CfnPodIdentityAssociation`; nothing needs a
  static AWS key, and there is no OIDC provider or SA annotation to manage.
- **Clean CDK/Helm split.** CDK owns infra + IAM; Helm owns the app — matching
  RouteIQ's chart-first reality and keeping app deploys fast and out-of-band.

### Negative

- **L1 `CfnCluster` ergonomics.** Auto Mode is realized via the L1 escape hatch
  (a choice inherited from the VSR port; aws-cdk-lib's newer `aws-eks-v2` now
  offers Auto Mode on an L2). On L1 there is no `addServiceAccount` helper — but
  with Pod Identity the pod→IAM wiring is a single `CfnPodIdentityAssociation`
  plus the defensive agent add-on, far less verbose than the L1 IRSA + CfnJson
  trust map the VSR reference required.
- **Two-tool deploy.** CDK for infra + Helm/kubectl for the app means two deploy
  steps and access-entry management for the CI identity.
- **Three app-layer foot-guns.** The CIDR-lock field, RWO+Recreate, and
  apply-k-clobbers-SA lessons must be encoded in the chart/deploy scripts or the
  deploy silently breaks.

## Alternatives Considered

### Alternative A: Manual `eksctl` / console cluster + manual Helm (status quo)

- **Pros**: No CDK.
- **Cons**: 100% manual; no reproducible pod-IAM binding or access entries — the
  gap.
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
