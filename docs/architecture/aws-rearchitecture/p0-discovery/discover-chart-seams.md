# Discover: RouteIQ Helm Chart Seams the CDK Must Satisfy

Deep-read of `deploy/charts/routeiq-gateway/` (chart `1.0.0-rc1`, `kubeVersion: ">=1.23.0-0"`).
The path prefix in the task ("undefined/") resolves to the repo root
`/Users/baladita/Documents/DevBox/RouteIQ`.

**The governing decision is ADR-0030** (`docs/adr/0030-eks-auto-mode-irsa-substrate.md`,
status: Proposed, 2026-06-14): *CDK provisions cluster + IAM + OIDC; Helm deploys the
app.* The chart **presupposes a cluster it does not create** and presupposes cluster-side
machinery it cannot provision (IRSA-annotated SA, external-secrets ClusterSecretStore,
Prometheus operator, a load balancer). This doc enumerates exactly what crosses that seam.

---

## 1. The CDK→Chart handoff contract (what the CDK must hand the chart)

The CDK construct emits CfnOutputs; the chart consumes them as Helm values. Per ADR-0030
the construct emits `ClusterName`, `ClusterEndpoint`, `OidcProviderArn`,
`OidcProviderIssuerUrl`, and the per-SA IRSA role ARN.

| What CDK produces | Where the chart consumes it | Exact value / wiring |
|---|---|---|
| **Per-workload IRSA role ARN** | `serviceAccount.annotations` (`values.yaml:205-209`), rendered by `serviceaccount.yaml:8-11` (`{{- with .Values.serviceAccount.annotations }}`) | `eks.amazonaws.com/role-arn: arn:aws:iam::<acct>:role/<routeiq-irsa-role>` — set via `--set-string serviceAccount.annotations.eks\.amazonaws\.com/role-arn=<ARN>` or a values file. The chart does **not** template this; it only passes through whatever annotations the operator supplies. The CDK must hand the literal ARN. |
| **OIDC provider ARN + issuer** | Not consumed by the chart at all | Used **only** CDK-side to mint the IRSA role's `WebIdentityPrincipal` trust policy (`<oidc>:sub` = `system:serviceaccount:<ns>:<sa>`). The chart side of the trust is the SA name (below). |
| **ServiceAccount name (the `:sub` the trust policy pins)** | `routeiq-gateway.serviceAccountName` (`_helpers.tpl:56-62`) — `serviceAccount.name` or, default, the fullname (`<release>-routeiq-gateway`) | The CDK's IRSA trust condition `<oidc>:sub = system:serviceaccount:<namespace>:<SA-name>` must be computed against the **exact** rendered SA name + the release namespace. If the operator overrides `serviceAccount.name` or `fullnameOverride`, the CDK trust policy must match or `AssumeRoleWithWebIdentity` returns AccessDenied. **This is a coupling the CDK must be told about** (pass the SA name + namespace as CDK parameters). |
| **Cluster name / endpoint** | Not a chart value | Used by CI's `helm upgrade` / `aws eks update-kubeconfig`, and by the CDK `CfnAccessEntry` for the CI identity (`bootstrap_cluster_creator_admin_permissions` only covers the CFN exec role, not the human/CI kubectl identity — ADR-0030 §EKS Auto Mode). |
| **Cluster OIDC `client_ids=["sts.amazonaws.com"]`** | n/a (CDK-internal) | Required for the OIDC provider; chart-invisible. |

**The IRSA seam is one-directional and string-typed.** The chart is a pure consumer:
`serviceaccount.yaml` does nothing but `toYaml` the annotations map. All the IRSA logic
(OIDC provider, CfnJson trust-key map, `WebIdentityPrincipal`) lives CDK-side. The single
artifact crossing the seam is the **role ARN string** — plus the **SA-name/namespace pair
the CDK trust policy must pin** (flowing CDK←chart at design time).

### The CfnJson gotcha the CDK owns (ADR-0030 §OIDC provider, lines 56-83)

The CDK must NOT do `issuer_url.replace("https://","")` on an unresolved CFN token (silent
no-op → `https://...:sub` keys → AssumeRoleWithWebIdentity AccessDenied, root-caused live
2026-06-08). It must use the ARN-derived, scheme-stripped `oidc_provider_issuer`, and wrap
the trust condition map in **`CfnJson`** because CFN map keys must be literal strings at
synth (a `Fn::GetAtt` token cannot be a map key). Auto Mode forces the L1 `CfnCluster`,
which forecloses the L2 `cluster.addServiceAccount` that would hide this. (See skill
`cdk-gotchas`: "IRSA L1 CfnCluster KeyMustResolveToString CfnJson".)

---

## 2. Chart-side gaps P0 must close

These are app-layer foot-guns ADR-0030 §"Three HA/ops lessons" calls out. **Two of the
three are genuine chart gaps today.**

### GAP-1 (P0): `service.yaml` cannot CIDR-lock under Auto Mode — MISSING `loadBalancerSourceRanges`

**Confirmed gap.** `service.yaml` (full file) emits only `type`, `ports`, `selector`, and
optional `nodePort`. There is:
- **No `spec.loadBalancerSourceRanges`** field anywhere (`grep` across the chart finds
  `LoadBalancer` only in `NOTES.txt:20-22` display text).
- **No `service.loadBalancerSourceRanges` value** in `values.yaml`.
- Default `service.type: ClusterIP` (`values.yaml:69`), `port: 80` → `targetPort: 4000`
  (`values.yaml:70-71`; container port is `gateway.port: 4000`, `values.yaml:220` /
  `deployment.yaml:77`).
- The only Service annotation guidance (`values.yaml:75-76`) is the legacy AWS internal-LB
  annotation `service.beta.kubernetes.io/aws-load-balancer-internal`.

Per ADR-0030 lesson #1: under EKS **Auto Mode managed load balancing**, the NLB CIDR
allowlist is honored via **`spec.loadBalancerSourceRanges`**, NOT the legacy
`service.beta.kubernetes.io/...` annotation. To expose RouteIQ via a `LoadBalancer` Service
on Auto Mode with a source-IP allowlist, the chart MUST:
1. Add a `service.loadBalancerSourceRanges: []` value.
2. Render `spec.loadBalancerSourceRanges` in `service.yaml` when `type == LoadBalancer`
   (guard like the existing `nodePort` block at `service.yaml:18-20`).

Without this, an Auto-Mode LoadBalancer Service is **open to 0.0.0.0/0** — the CIDR-lock the
annotation-style allowlist silently fails to provide.

### GAP-2 (P0/process): `helm upgrade` is mandatory — `kubectl apply -k` clobbers the IRSA SA annotation

ADR-0030 lesson #3: kustomize (`apply -k`) overwrites the ServiceAccount's `role-arn`
annotation, breaking IRSA. Because the chart treats the annotation as pure pass-through
(`serviceaccount.yaml:8-11`), the deploy path **must** be `helm upgrade` (which templates
the annotation from values) or `envsubst` it in — never a bare `apply -k` on the SA. This
is a deploy-script/CI guardrail, not a template change, but P0 must encode it (CI runs
`helm upgrade`, not kustomize). The companion deploy lesson: any future RWO-EBS PVC mount
needs `strategy.type: Recreate` (ADR-0030 lesson #2) — the chart currently hardcodes
`RollingUpdate` (`values.yaml:532-536`) and mounts only `emptyDir` volumes
(`deployment.yaml:154-164`), so there is no Multi-Attach deadlock **today**, but the moment
a persistent EBS PVC is added the strategy must flip. Not a P0 blocker now; a tripwire.

### GAP-3 (informational): the SA token-automount / leader-election coupling

`serviceaccount.yaml:12-17` and `deployment.yaml:34` force `automountServiceAccountToken:
true` when leader election is enabled with the `kubernetes` backend (or auto-detect). With
IRSA, the projected SA token is what STS exchanges for AWS creds — automount must be **on**
for IRSA regardless of leader election. Default is `automountServiceAccountToken: false`
(`values.yaml:212`). **P0 must set `serviceAccount.automountServiceAccountToken: true`** (or
enable K8s leader election) so the IRSA token projection works. This is a values setting,
not a template gap, but it is easy to miss because the default is `false`.

---

## 3. Each chart env var → the IAM action the pod (IRSA) role must enable

Env vars are emitted by `routeiq-gateway.envVars` (`_helpers.tpl:162-329`) and
`routeiq-gateway.databaseEnv` (`_helpers.tpl:139-157`). Mapping to the peer ADRs that own
the IAM grant on the IRSA role:

| Chart value → env var | `_helpers.tpl` line | AWS service touched | IAM action on IRSA role | ADR |
|---|---|---|---|---|
| `gateway.configSync.s3.bucket` → `CONFIG_S3_BUCKET`, `CONFIG_S3_KEY` | 188-197 | S3 config object (ETag poll today; AppConfig target) | `s3:GetObject` (+ `s3:GetObjectAttributes`/`HeadObject` for ETag) on `arn:aws:s3:::<bucket>/<key>`; under ADR-0026 the 5 AppConfig read actions (`appconfig:GetLatestConfiguration`, `StartConfigurationSession`, …) scoped to the profile ARN | 0026 |
| `externalPostgresql.host` → `DATABASE_URL` | 142, 276 | Aurora PostgreSQL Serverless v2 | `rds-db:connect` on `arn:...:rds-db:...:dbuser:<cluster-resource-id>/<db_user>` (IAM DB auth replaces the static `POSTGRES_PASSWORD`) | 0028 |
| `externalRedis.host` → `REDIS_HOST`/`REDIS_PORT`/`REDIS_DB`/`REDIS_SSL` | 287-295 | ElastiCache Serverless (Valkey) | `elasticache:Connect` on `<cache_arn>` + `<iam_user_arn>` (IAM auth replaces `REDIS_PASSWORD`; **TLS mandatory → `externalRedis.ssl: true`**) | 0029 |
| `gateway.otel.endpoint` / `externalOtel.endpoint` → `OTEL_EXPORTER_OTLP_ENDPOINT` | 222-224, 306-310 | Amazon Managed Prometheus (AMP) remote-write | `aps:RemoteWrite` on the workspace ARN (if exporting to AMP; for a sidecar/collector OTLP target, no IRSA grant) | 0027 |
| (config-driven, not a Helm env) Bedrock provider models in `config.gateway` | n/a (model_list) | Amazon Bedrock | `bedrock:InvokeModel` / `bedrock:InvokeModelWithResponseStream` on the model ARNs RouteIQ routes to | 0030 §IRSA factory |
| `gateway.configSync.gcs.bucket` → `CONFIG_GCS_BUCKET` | 198-203 | GCS (not AWS) | n/a on AWS — GKE Workload Identity path, mutually exclusive with the S3 path | — |

**IRSA role policy the CDK must attach** (ADR-0030 §IRSA factory, line 88-89): exactly
`aps:RemoteWrite` (0027) + `rds-db:connect` (0028) + `elasticache:Connect` (0029) +
AppConfig read (0026) + Bedrock invoke. The chart never emits an `AWS_REGION` /
`AWS_DEFAULT_REGION` / Bedrock env var (`grep` confirms none in `_helpers.tpl`), so the CDK
must inject region via `extraEnv` (`values.yaml:442`) or rely on the IMDS/region from the
node — **flag this**: a Bedrock or AMP client with no region will fail; P0 should add
`extraEnv: [{name: AWS_REGION, value: <region>}]`.

### Secrets that IRSA *removes* vs. what still needs External Secrets

ADRs 0028/0029 replace `POSTGRES_PASSWORD` and `REDIS_PASSWORD` with IAM auth — so on the
AWS-native path those secret keys (`values.yaml:387,391`) become **unused**. What still
needs a real secret is `LITELLM_MASTER_KEY` / `ADMIN_API_KEYS` and any provider API keys not
on Bedrock (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …). The production path is **External
Secrets Operator**:

- `externalsecret.yaml` renders an `external-secrets.io/v1beta1` `ExternalSecret` when
  `externalSecrets.enabled` (default `false`, `values.yaml:395`).
- `secretStoreRef`: `name: aws-secrets-manager`, `kind: ClusterSecretStore`
  (`values.yaml:397-399`, rendered `externalsecret.yaml:10-12`). **The CDK/cluster must
  pre-provision a `ClusterSecretStore` named `aws-secrets-manager`** wired to AWS Secrets
  Manager — the chart references it by name but does not create it. The ESO controller's
  own IRSA/permissions (`secretsmanager:GetSecretValue`, `kms:Decrypt`) are a **separate**
  IAM role from RouteIQ's pod role and must also be provisioned CDK-side.
- `externalSecrets.data` (`values.yaml:403`) is empty by default → the rendered
  ExternalSecret has **no `data:` block** (`externalsecret.yaml:16-29` is gated on
  `.data`), syncing nothing. P0 must populate `externalSecrets.data` with the
  `secretKey`/`remoteRef.key` pairs for `LITELLM_MASTER_KEY` etc.

---

## 4. NetworkPolicy egress (chart) — what it allows and where it leaks under Auto Mode

`networkpolicy.yaml` (gated on `networkPolicy.enabled`, default `false`, `values.yaml:487`):

- **Ingress** (`networkpolicy.yaml:15-38`): if no `fromNamespaceSelector`/`fromPodSelector`/
  `fromCidrs` are set, it renders `- {}` → **allow-from-all** on `gateway.port` (4000). This
  is a defense-in-depth complement to GAP-1's LoadBalancer CIDR-lock, not a substitute (NLB
  source-ranges filter at the LB; NetworkPolicy filters pod-to-pod). P0 should set ingress
  selectors when enabling it.
- **Egress** (`networkpolicy.yaml:39-67`):
  - `allowDns: true` → UDP/TCP 53 to `k8s-app: kube-dns` (`:40-52`).
  - `allowHttpsExternal: true` → TCP 443 + 80 to `0.0.0.0/0` **except** the three RFC-1918
    ranges 10/8, 172.16/12, 192.168/16 (`:53-67`). This permits LLM provider + Bedrock
    egress while **blocking the in-VPC services** (Aurora, ElastiCache, AMP) — which are
    reached over private IPs. **Therefore the egress `to: []` list (`values.yaml:508`) MUST
    be populated** with explicit rules to Aurora (5432), ElastiCache (6379), and any
    in-cluster OTel collector, or the NetworkPolicy silently severs RouteIQ from its managed
    state plane the moment it is enabled. The chart supports this via
    `networkPolicy.egress.to` (`networkpolicy.yaml:68-95`), but ships it **empty**.
  - Caveat for Auto Mode: EKS Auto Mode's managed CNI must enforce NetworkPolicy; the
    `except` RFC-1918 carve-out assumes provider endpoints are public IPs — Bedrock via a
    **VPC interface endpoint** would be a private IP and thus blocked by the
    `allowHttpsExternal` rule, requiring an explicit `to` entry. Flag for P0 if Bedrock is
    reached via PrivateLink.

---

## 5. Quick file index (load-bearing lines)

- `serviceaccount.yaml:8-11` — IRSA annotation pass-through (the seam); `:12-17` automount
  forced on for K8s leader election.
- `service.yaml` (whole) — **no `loadBalancerSourceRanges`** (GAP-1); `:12` type, `:14-15`
  port→targetPort 80→4000, `:18-20` nodePort guard (pattern to copy).
- `deployment.yaml:34` automount coupling; `:46,73` image via `routeiq-gateway.image`;
  `:114-127` env wiring + envFrom secretRef; `:44-64` db-migrate initContainer (leader/DB
  gated); `:154-164` emptyDir-only volumes (no RWO PVC today).
- `externalsecret.yaml:10-12` `ClusterSecretStore: aws-secrets-manager` (must pre-exist);
  `:16-29` data block gated on empty `externalSecrets.data`.
- `networkpolicy.yaml:53-67` HTTPS-external-except-RFC1918 egress; `:68-95` the empty
  `egress.to` extension point for in-VPC services.
- `_helpers.tpl:139-157` `databaseEnv`; `:162-329` `envVars` (the full env→IAM map);
  `:56-62` SA name; `:68-77` image (digest>tag); `:82-88` secretName.
- `values.yaml:33` `image.repository: ghcr.io/baladithyab/routeiq` (GHCR — image is pulled
  from GHCR, so the cluster needs either public pull or an `imagePullSecrets` regcred;
  unrelated to IRSA but a cluster prereq); `:199-212` SA/IRSA; `:316-338` external PG/Redis;
  `:394-409` External Secrets.
- ADR-0030 — the authoritative CDK/Helm boundary + 3 ops lessons.
