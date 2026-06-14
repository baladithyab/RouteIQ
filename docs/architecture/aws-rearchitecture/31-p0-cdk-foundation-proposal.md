# P0 — CDK Foundation Proposal: `deploy/cdk/` (VPC / EKS Auto Mode / ECR-GHCR / Pod Identity)

> **Status**: Proposal (P0-propose wave). **Date**: 2026-06-14.
> **Scope**: RouteIQ's first provisioning IaC — a `deploy/cdk/` tree with a single
> `RouteIqStack` (private multi-AZ EKS Auto Mode cluster, ECR with a GHCR
> pull-through cache, one least-privilege pod IAM role via **EKS Pod Identity**, and
> a credential-free `pytest` synth+nag gate).
> **Companion docs**: `docs/architecture/aws-rearchitecture/30-migration-roadmap.md`
> (P0 row, `:31-66`), ADR-0030 (`docs/adr/0030-eks-auto-mode-irsa-substrate.md`,
> status *Proposed* — superseding amendment filed by this proposal, §14), the five
> discover findings under `p0-discovery/` (`p0-discovery/discover-network.md`,
> `p0-discovery/discover-ecr-ghcr.md`, `p0-discovery/discover-scaffold-tests.md`,
> `p0-discovery/discover-chart-seams.md`, `p0-discovery/discover-eks-cluster.md`), and
> the research report (`p0-discovery/research-report-irsa-vs-podidentity.md`).
>
> **The single most important framing this proposal carries**: it **OVERRIDES the
> IRSA assumption baked into ADR-0030** (titled `0030-eks-auto-mode-irsa-substrate.md`)
> and every discover findings file. The research verdict is **EKS Pod Identity, not
> IRSA** (*"very high confidence"*). This proposal (1) states the flip explicitly,
> (2) requires a superseding amendment to ADR-0030, and (3) shows the chart-seam
> simplification the flip produces (no ServiceAccount `eks.amazonaws.com/role-arn`
> annotation). The construct API is named the **pod-identity association wiring**,
> not an "IRSA factory."
>
> **The one thing this proposal does NOT do**: it writes **no CDK `.py` files**. The
> `deploy/cdk/` tree, constructs, and tests below are the **specification** the next,
> operator-gated build wave implements. P0-propose is design only.

---

## 1. Summary / TL;DR

RouteIQ gains its first provisioning IaC: a `deploy/cdk/` tree with a single
**`RouteIqStack`** that stands up a private multi-AZ **EKS Auto Mode** cluster (L1
`CfnCluster`), an **ECR** repository fronted by a **GHCR pull-through cache**, one
least-privilege pod IAM role bound via **EKS Pod Identity** (a single
`CfnPodIdentityAssociation`), and a **credential-free `pytest` synth + cdk-nag gate**
(`uv run pytest deploy/cdk/tests/`).

- **Scope boundary.** P0 is the **substrate only**. P1 (Aurora + ElastiCache) is a
  SEPARATE stack/CI-stage per the ~30-minute-rollback rule and is explicitly out of
  P0. Therefore **P0 ships exactly one `RouteIqStack`**.
- **Division of labor.** The CDK provisions cluster + IAM + ECR + network. The
  existing Helm chart (`deploy/charts/routeiq-gateway/`, version `1.0.0-rc1`) deploys
  the app. **The CDK never runs `KubernetesManifest` or `HelmChart`** — the
  CDK→Helm boundary is the same one RouteIQ already implicitly assumes.
- **Risk: LOW.** Almost entirely a PORT from the proven `vllm-sr-on-aws/cdk/`
  scaffold (the P0 roadmap row classifies this Low). The net-new is a thin stack
  wiring layer plus the Pod Identity simplification — which *removes* code relative
  to the IRSA design rather than adding it.
- **This is a proposal.** No `.py` is authored here. The build is the next wave,
  gated on the open questions in §14 (chiefly account topology and the
  public-CIDR-lock-vs-bastion decision).

---

## 2. Context & Provenance

RouteIQ today is **Helm-only on AWS with zero provisioning IaC** — there is no
`cdk.json`, no `cdk/` tree, no `*.tf`. The only Python "app" in the tree is the
FastAPI factory at `src/litellm_llmrouter/gateway/app.py` (not a CDK construct). The
entire AWS story is the 640-line manual `docs/deployment/aws.md` runbook plus a chart
that emits Kubernetes objects onto a cluster it does not create
(discover-ecr-ghcr §TL;DR; discover-chart-seams §1; roadmap `:5-7`). ADR-0030 names
this *"RouteIQ's largest gap: a chart with no cluster under it."*

**The four ported constructs are not yet in the RouteIQ repo, but the VSR source
IS present locally.** They are PORTS from
`/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/` —
`network_construct.py`, `eks_cluster_construct.py` (50,477 bytes),
`ecr_construct.py`, `nag_suppressions.py` (plus the `vllm_sr_eks_stack.py`
composition root that `RouteIqStack` mirrors), alongside a `cdk/tests/` tree. That
directory is **confirmed checked out on this machine** and is the canonical source of
truth — exactly as `discover-network.md:8-9` already cites it (it read
`network_construct.py` verbatim from this same path).

> **Provenance correction.** The "PROVENANCE — READ FIRST" header in
> `discover-eks-cluster.md:3-7` claims the file *"does not exist on this machine"* and
> that a *"full-filesystem `find` returned nothing."* **That claim is FALSE.** It
> directly contradicts `discover-network.md:8-9`, which cites the exact local path and
> extracts from it verbatim. `eks_cluster_construct.py` is present at
> `/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/eks_cluster_construct.py`
> (50,477 bytes), next to `network_construct.py`, `ecr_construct.py`,
> `nag_suppressions.py`, and `vllm_sr_eks_stack.py`. Treat the discover-eks-cluster
> skeletons as a faithful summary index, **not** as a substitute for the real file.

**The build wave MUST re-derive every symbol — class names, kwarg names, block
shapes, logical IDs, prop spellings — directly from the real local source files at
`/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/`, NOT from the discover-doc
distillations.** The discover docs are an index into the source, not a replacement for
it; where a discover doc and the source disagree, the source wins. The `:NNN` line
cites in the discover docs were accurate at the 2026-06-14 snapshot and will drift —
**cite by SYMBOL NAME, read the symbol from the real file, then copy.**

Authoritative inputs for this proposal:

- The P0 roadmap row (`30-migration-roadmap.md:31-66`).
- ADR-0030 (the decision being amended).
- The pivot handoff (`docs/handoffs/2026-06-14-1211-pivot-to-routeiq-on-aws.md`,
  §7 / §10 / §11).
- The research report
  (`research/notes/final_report_eks-automode-cdk-irsa-podidentity-9dc4a2.md`),
  sources [1]–[11].

---

## 3. The Load-Bearing Decision: Pod Identity, not IRSA  *(covers doneMeans (b))*

**Decision: use EKS Pod Identity for RouteIQ's pod IAM. Do NOT use IRSA.**

This flips ADR-0030's IRSA assumption. The verdict is strong — the research report
calls it *"very high confidence"* — and, critically, **the L1 `CfnCluster`
constraint is the CLINCHER, not a complication**.

### 3.1 Evidence chain (research report)

1. **AWS recommends Pod Identity for Auto Mode.** AWS's own Auto Mode security
   best-practices guidance names Pod Identity the *recommended* mechanism for IAM
   permissions in EKS Auto Mode (report §3, src [1]).
2. **The agent: AWS docs say built-in, but VSR's own Auto Mode CDK installs it by
   hand — resolve empirically.** AWS docs state *"You do not need to install the EKS
   Pod Identity Agent on EKS Auto Mode Clusters. This capability is built into EKS Auto
   Mode"* (report §3 / §5, src [6]), which would make `eks-pod-identity-agent` the **one
   exception to the L1 "wire your own addons by hand" rule.** **HOWEVER**, the
   production VSR Auto Mode construct — the very source this proposal ports — installs
   `eks-pod-identity-agent` **explicitly** via `eks.CfnAddon(...,
   resolve_conflicts="OVERWRITE")` with `add_dependency(self.cfn_cluster)`, inside the
   same construct that sets `compute_config`, `node_pools=["general-purpose","system"]`,
   and `bootstrap_self_managed_addons=False`
   (`vllm-sr-on-aws/cdk/lib/eks_cluster_construct.py`, the `eks-pod-identity-agent`
   `CfnAddon` in `enable_container_insights()`; comment: *"The Pod Identity Agent add-on
   must be present for Pod Identity associations to resolve"*). This **directly
   contradicts** the load-bearing "pre-installed, no install step" claim. **Do NOT count
   on zero addon cost without verifying empirically** against a live Auto Mode cluster.
   Because the addon is **cheap and idempotent** (`resolve_conflicts="OVERWRITE"` makes
   re-applying a built-in addon a no-op), the safe default is to **add it defensively**
   so the association is guaranteed to resolve regardless of which behavior is live —
   see §7.4 and the acceptance criteria (§15 / §12.1).
3. **The CDK build is one resource.** Pod Identity is a single
   **`CfnPodIdentityAssociation`** (CloudFormation `AWS::EKS::PodIdentityAssociation`)
   with four required string props — `clusterName`, `namespace`, `serviceAccount`,
   `roleArn` — over a **static** trust policy to the `pods.eks.amazonaws.com` service
   principal (report §5, src [10]).

### 3.2 Why the L1 `CfnCluster` constraint is the CLINCHER

Auto Mode forces the L1 `eks.CfnCluster`. The L1 path **strips the L2
`cluster.addServiceAccount()` helper that normally HIDES IRSA's complexity** (no
KubectlProvider on L1 → no `addManifest`/`addHelmChart`/`addServiceAccount`; report
§2/§4, src [2][7]). On L1, IRSA forces you to **hand-build the IAM OIDC provider
AND wrap the token-keyed trust condition in `CfnJson`** to dodge the
`KeyMustResolveToString` failure — because CloudFormation forbids intrinsic-function
tokens as JSON dictionary *keys*, and an IRSA trust condition uses the OIDC issuer as
a key (`<issuer>:sub` / `<issuer>:aud`) (report §4, src [9]; gotcha
`cdk-irsa-l1-cluster-cfnjson-token-key`).

Pod Identity's static principal has **no token-keyed condition**, therefore:

- **no `CfnJson`**,
- **no `OpenIdConnectProvider`**,
- **zero `.replace`-on-token trap.**

The `.replace`/`.split`-on-an-unresolved-CFN-token silent no-op
(`issuer_url.replace("https://", "")` returns the token unchanged, producing
`https://...:sub` keys and a runtime `AssumeRoleWithWebIdentity` AccessDenied; synth
stays green) was **root-caused live on 2026-06-08** (discover-eks-cluster §8a;
gotcha `cdk-token-string-op-silent-noop`). Choosing Pod Identity **deletes this
entire class of CDK token-resolution failure from the build.** *"The L1 constraint is
the clincher rather than a complication: it strips the L2 `addServiceAccount` helper
that normally hides IRSA's complexity, so on L1 the gap between trivial Pod Identity
and hand-rolled IRSA is at its widest"* (report Verdict).

> **Honest scope of the win.** Pod Identity removes only the *pod-IAM* slice of L1
> friction. The cluster-access and manifest-application slices of L1 remain (you
> still hand-roll `CfnAccessEntry`, the two Auto Mode roles, and deploy manifests
> out-of-band via Helm). "Pod Identity makes L1 easy" overstates it; what is true is
> that Pod Identity makes the *pod-IAM* slice nearly free (report §2, src [2]).

### 3.3 The scaling asymmetry

| Dimension | IRSA | Pod Identity |
|---|---|---|
| AWS recommendation on Auto Mode | supported | **recommended** (src [1]) |
| Agent on Auto Mode | n/a | AWS docs say **built-in** (src [6]), but VSR's production Auto Mode CDK installs `eks-pod-identity-agent` via `CfnAddon` by hand — **resolve empirically; add defensively** (§3.1 #2, §7.4) |
| Trust policy | per-cluster OIDC, token-keyed condition | **static** `pods.eks.amazonaws.com` (src [8]) |
| Account scaling | 100 OIDC providers / account limit | not applicable (src [8]) |
| Role scaling | 2048-byte trust-policy ceiling (~4–8 trusts/role) | not applicable (src [8]) |
| Session tags / ABAC | no | yes (src [8]) |
| Cluster readiness | role must wait for cluster Ready | role can pre-exist (src [8]) |
| Reusability | trust edits per cluster | role reusable across clusters (src [8]) |
| Association ceiling | n/a | 5,000 / cluster — far above RouteIQ's needs (src [11]) |
| CDK L1 effort | high (`CfnJson` + OIDC provider) | low (4 props) |

### 3.4 Where IRSA would still win — and why NONE apply to RouteIQ

IRSA remains the right call under four specific conditions, **all FALSE for RouteIQ**
(report §3 / Verdict, src [8][11]):

- **Fargate pods** — Pod Identity is worker-nodes-only. Auto Mode runs on **EC2
  Managed Instances**, so this is moot.
- **Cross-platform IAM trust** — sharing trust with EKS-Anywhere / ROSA / self-managed
  Kubernetes. RouteIQ is EKS-in-cloud only.
- **A pre-Pod-Identity AWS SDK** — RouteIQ uses current SDKs.
- **A mature existing IRSA estate** — RouteIQ is greenfield (zero IaC today).

Greenfield EKS-Auto-Mode-in-cloud is squarely inside the scope AWS names Pod-Identity-
recommended. AWS *"continues to invest in IRSA"* — it is **supported, not
deprecated**; the recommended-vs-still-supported split is a **SCOPE statement, not a
contradiction** (report §3, src [8]).

### 3.5 Chart-seam impact (the ripple — also covers doneMeans (e))

With Pod Identity the chart needs **NO `eks.amazonaws.com/role-arn` ServiceAccount
annotation at all.** The pod→role binding is a CDK-side `CfnPodIdentityAssociation`
keyed on `(namespace, serviceAccount)`. This **DELETES discover-chart-seams GAP-2**
(the `kubectl apply -k`-clobbers-the-annotation hazard) for the IAM path — there is no
annotation to clobber. The chart's ServiceAccount still must exist with a **STABLE
`(namespace, name)`** that the CDK association pins; the contract becomes
`serviceAccount.create: true` plus an explicit `serviceAccount.name` (or a documented
rendered fullname). `automountServiceAccountToken` is **NOT required for Pod Identity
credentials** — the agent injects creds via the pod-identity webhook, not via a
projected SA token — but it is kept as-is for Kubernetes leader election. **This is a
change from discover-chart-seams GAP-3**, which assumed IRSA token projection. See
§11 for the full chart wiring and the explicit retraction of the IRSA-annotation
guidance.

### 3.6 Open caveats to honor (they do not change the verdict)

- **Associations are eventually consistent.** AWS warns not to create/update them in
  HA-critical code paths → **create them at provision time in CDK, never in a hot
  startup path** (report Verdict, src [11]).
- **Keep IPv6 enabled** — the agent listens on a link-local IPv6 address.
- **IMDSv2 hop-limit restriction** is already applied by Auto Mode.
- **The one open question is account topology** (single vs cross-account):
  single-account makes the verdict lopsided; cross-account narrows the margin but
  **still favors Pod Identity** (it adds exactly one `targetRoleArn` prop and uses
  ~59-minute credential-cache role chaining). See §14.

---

## 4. The Single Pod IAM Role — Exact Policy Statements  *(covers doneMeans (b)/(e))*

**One role, one association.** RouteIQ is a single stateless gateway pod, so P0 mints
**ONE** role bound to **ONE** `(namespace, serviceAccount)`. The vllm-sr
router / EAIG / bearer-minter three-role split and the cross-account capacity loop
are dropped entirely (discover-eks-cluster §9).

### 4.1 Trust policy (static, Pod Identity)

```text
Principal:  Service  pods.eks.amazonaws.com   (iam.ServicePrincipal)
Actions:    sts:AssumeRole
            sts:TagSession
```

This is the whole trust policy — a static service-principal grant of `sts:AssumeRole`
+ `sts:TagSession` to `pods.eks.amazonaws.com` (report §5, src [10]). There is **no
`WebIdentityPrincipal`, no `OpenIdConnectPrincipal`, no `CfnJson` condition map.**

### 4.2 Permission statements (P0 — each its own `PolicyStatement` with an explicit `sid`)

Each statement is least-privilege: explicit resources, never `*` except where the
action itself requires it.

```text
sid = "BedrockInvoke"          # P0, data-plane critical
  actions   = bedrock:InvokeModel
              bedrock:InvokeModelWithResponseStream
              bedrock:Converse
              bedrock:ConverseStream
  resources = the foundation-model / inference-profile ARNs RouteIQ routes to
              (arn:aws:bedrock:<region>::foundation-model/*  scoped per deployment,
               or explicit inference-profile ARNs)
  # discover-chart-seams §3 row; discover-network §5

sid = "SecretsRead"            # P0
  actions   = secretsmanager:GetSecretValue
              secretsmanager:DescribeSecret
  resources = the LITELLM_MASTER_KEY / ADMIN_API_KEYS / provider-key secret ARNs
  # discover-network §5; discover-chart-seams §3

sid = "ConfigS3Read"           # P0
  actions   = s3:GetObject
              s3:GetObjectAttributes      # ETag poll (gateway.configSync.s3 today)
  resources = arn:aws:s3:::<config-bucket>/<key>
  # discover-chart-seams §3 row

sid = "Logs"                   # P0
  actions   = logs:CreateLogStream
              logs:PutLogEvents
  resources = the CDK-created routing log group ARN (the MetricFilters consume these)
  # Scoped to the routing log group, NOT logs:* — Auto Mode + the
  # amazon-cloudwatch-observability addon largely handle pod logs.
  # discover-network §5; discover-eks-cluster §7
```

### 4.3 Statements wired but resource-pending until peer P-tiers land

State these as commented / flag-gated in the eventual build so the role is
forward-compatible while P0 stays least-privilege:

```text
# P2 (ADR-0026):  appconfig:GetLatestConfiguration
#                 appconfigdata:StartConfigurationSession      → on the AppConfig profile ARN
# P2 (ADR-0027):  aps:RemoteWrite                              → on the AMP workspace ARN
# P1 (ADR-0028):  rds-db:connect                               → on the Aurora dbuser ARN
# P1 (ADR-0029):  elasticache:Connect                          → on the cache + IAM-user ARN
```

### 4.4 Statements explicitly DROPPED on the Pod Identity path

- **DROP `sts:AssumeRoleWithWebIdentity`.** Pod Identity does not use it.
  discover-network §5 listed it under the IRSA assumption; on the Pod Identity path
  it is **gone** (Pod Identity creds come from the agent, not an STS web-identity
  exchange).
- **ECR pull permissions live on the NODE role, NOT this pod role.** The Auto Mode
  node role carries `AmazonEC2ContainerRegistryPullOnly` (report §2). The
  pull-through-cache `ecr:BatchImportUpstreamImage` is the **node/pull identity's**
  grant (discover-ecr-ghcr §3.3b), not the application pod role's. **Do not over-grant
  the pod role with `ecr:*`** — see §9.3.

### 4.5 Synth-time guard: ASCII-only descriptions

Every role/statement `Description` must be ASCII / Latin-1. **An em-dash (U+2014)
passes `cdk synth` but FAILS the IAM CREATE API** — a synth-green/deploy-red trap
(discover-scaffold-tests §1.6; roadmap P0 operator note). This is asserted by a unit
test (§12).

---

## 5. `deploy/cdk/` Tree + Dependency Pins  *(covers doneMeans (d))*

**Home:** `deploy/cdk/`, co-located next to `deploy/charts/routeiq-gateway/`
(handoff §10 recommendation; keeps RouteIQ self-contained). Naming convention:
`routeiq:` context keys (mirrors VSR's `vllm_sr:`), stack id `RouteIqStack-<env>`,
construct files `*_construct.py`, package import path `lib.*`.

> **The Pod Identity flip is reflected in this tree.** It diverges from the
> discover-scaffold-tests §2 layout in exactly two places: the cluster construct
> emits a `pod_identity_association()` helper (no OIDC provider, no `CfnJson`), and
> the test suite adds a negative regression guard for the absence of OIDC/CfnJson.

```text
deploy/cdk/
├── app.py            # _ctx / _bool_ctx / _split_csv_or_list helpers — copy the SHAPE
│                     #   verbatim from VSR app.py (_bool_ctx defuses the bool("false")
│                     #   == True CLI-string footgun); build cdk.App, read routeiq:*
│                     #   context, build cdk.Environment from CDK_DEFAULT_ACCOUNT/REGION,
│                     #   RouteIqStack(app, f"RouteIqStack-{env}", env=..., **flags),
│                     #   cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True)),
│                     #   app.synth()
├── cdk.json          # "app": "python3 app.py" + flat "context": { "routeiq:*": ... }
├── requirements.txt  # the pins (below)
├── README.md         # how to synth + the routeiq: context keys + operator deploy steps
├── lib/
│   ├── __init__.py
│   ├── routeiq_stack.py            # composition root (mirrors vllm_sr_eks_stack.py)
│   ├── eks_cluster_construct.py    # L1 CfnCluster + 3 Auto-Mode blocks + node/cluster
│   │                               #   roles + CfnAccessEntry + container insights +
│   │                               #   the pod_identity_association() helper
│   │                               #   (NO OIDC provider, NO CfnJson)
│   ├── ecr_construct.py            # CfnPullThroughCacheRule(ghcr.io) + the ghcr
│   │                               #   CfnRepositoryCreationTemplate (IMMUTABLE,
│   │                               #   AppliedFor=PULL_THROUGH_CACHE) + registry
│   │                               #   scan-on-push for ghcr/* (NO standalone repo)
│   ├── network_construct.py        # VPC + SGs + interface endpoints (incl
│   │                               #   BEDROCK_RUNTIME) + S3 gateway endpoint
│   └── nag_suppressions.py         # apply_nag_suppressions(stack) + _suppress_<c>(stack)
└── tests/
    ├── __init__.py
    ├── unit/
    │   ├── __init__.py
    │   ├── test_eks_cluster.py
    │   ├── test_ecr.py
    │   ├── test_routeiq_stack.py
    │   └── test_cdk_nag.py
    └── snapshot/
        ├── __init__.py
        ├── test_template_snapshot.py
        └── __snapshots__/
            └── dev.json            # committed flag-off baseline (UPDATE_SNAPSHOTS=1)
```

### 5.1 Dependency pins (`requirements.txt`)

From discover-scaffold-tests §1.3, matching handoff §10:

```text
aws-cdk-lib>=2.150.0,<3.0.0     # floor for the BEDROCK_RUNTIME interface-endpoint
                                #   enum (discover-network §6.3) + Auto Mode L1 props
cdk-nag>=2.27.0,<3.0.0          # add_*_by_path RAISES on an absent path at >=2.27
                                #   (drives the flag-gated getattr guards, §13)
constructs>=10.0.0,<11.0.0
pytest>=7.4.0,<9.0.0
```

- **DROP `cdklabs.ecs-codedeploy==0.0.441`** — it is VSR ECS blue/green only; RouteIQ
  is EKS + Helm (discover-scaffold-tests §1.3).
- **`cdk-monitoring-constructs` is OPTIONAL for P0** — omit unless the
  container-insights helper needs it.

### 5.2 uv wiring (the cred-free gate's dependency path)

Add a `cdk` extra to the root `pyproject.toml` carrying the four pins and run
`uv sync --extra cdk` — OR keep `deploy/cdk/requirements.txt` and `uv pip install -r`
into the env. **The root-extra is the uv-native path** (one lockfile, one
`uv run pytest`) and is recommended; either is acceptable
(discover-scaffold-tests §3).

### 5.3 `routeiq:` context keys in `cdk.json` (mirrors VSR's `vllm_sr:`)

| Key | Default | Purpose |
|---|---|---|
| `routeiq:env` | `dev` | environment suffix on the stack id |
| `routeiq:vpc_cidr` | `10.40.0.0/16` | distinct from VSR (10.20 ECS / 10.30 EKS) so a future peering stays overlap-free |
| `routeiq:nat_gateways` | `1` | floor; bump to 2 for prod AZ-resilient egress |
| `routeiq:k8s_version` | `1.33` | Auto Mode cluster version |
| `routeiq:enable_ghcr_ptc` | `true` | GHCR pull-through cache rule |
| `routeiq:sa_namespace` | (chart ns) | the Pod Identity association binding — MUST match the chart's rendered SA |
| `routeiq:sa_name` | (chart SA name) | the Pod Identity association binding |
| `routeiq:image_tag` | (e.g. `1.0.0-rc1`) | chart image override |
| `routeiq:admin_principal_arns` | `[]` | `CfnAccessEntry` for the CI/operator kubectl identity |
| `routeiq:bedrock_model_arns` | (optional) | scopes the `BedrockInvoke` statement |
| `routeiq:config_s3_bucket` | (optional) | scopes the `ConfigS3Read` statement |
| `routeiq:secret_arns` | (optional) | scopes the `SecretsRead` statement |

> **Why `_bool_ctx` is load-bearing.** A `cdk synth --context routeiq:enable_ghcr_ptc=false`
> passes the **string** `"false"`, and `bool("false")` is `True` in Python — a naive
> `bool(_ctx(...))` silently keeps the default. `_bool_ctx` returns native bools as-is
> and parses CLI strings via `str(value).strip().lower() in ("true","1","yes")`. Copy
> this helper's shape verbatim (discover-scaffold-tests §1.1).

---

## 6. One Stack vs Split — the Decision  *(covers doneMeans (d))*

**Decision: ONE `RouteIqStack` for P0.** A single stateless pod, one VPC, one cluster,
one ECR, one pod role — there is no rollback-coupling pressure yet.

The ~30-minute-rollback rule (handoff §11; roadmap P1 operator note; vllmsr-patterns
lesson #7) mandates that **Aurora be a SEPARATE stack / CI-stage** so an app rollback
is not gated on a 30-minute DB revert. **But Aurora is P1**, so it is out of P0 scope.
Therefore:

- **P0 = one stack.**
- The split happens **when P1 lands** — Aurora + ElastiCache become a
  `RouteIqStateStack` deployed in a CI stage separate from app CI.

The VSR two-app cross-account split (per-account Bedrock capacity member stacks) is
**deferred entirely** at P0 (discover-eks-cluster §9.2).

---

## 7. The EKS Cluster Construct  *(covers doneMeans (b))*

Port `eks_cluster_construct.py`. Symbol names attested across ADR-0030,
`vllmsr-patterns.md`, and `99-review-findings.md` (discover-eks-cluster §1):
`EksClusterConstruct`, the three Auto Mode blocks,
`node_pools=["general-purpose","system"]`, `bootstrap_self_managed_addons=False`,
`CfnAccessEntry`, `enable_container_insights`, the
`routing_latency_ms_by_model` metric filter with `dimensions={"model":"$.[\"gen_ai.response.model\"]"}`
(the VSR source uses `$.selected_model`; the RouteIQ port must use the telemetry
contract key per ADR-0027:64-67 — RouteIQ emits `gen_ai.response.model`
(`telemetry_contracts.py:673`); `selected_model` is never emitted).

### 7.1 L1 `eks.CfnCluster`, K8s `1.33`, three Auto Mode blocks (all `enabled`)

```text
compute_config              enabled = True
                            node_pools = ["general-purpose", "system"]
                            node_role_arn = <AmazonEKSAutoNodeRole>
storage_config.block_storage                              enabled = True
kubernetes_network_config.elastic_load_balancing          enabled = True
```

- `bootstrap_self_managed_addons = False` — Auto Mode supplies CoreDNS / kube-proxy /
  VPC-CNI (discover-eks-cluster §2).
- `access_config.authentication_mode = "API"` +
  `bootstrap_cluster_creator_admin_permissions = True`.
- The two AWS-managed node pools collapse the Karpenter templates RouteIQ would
  otherwise hand-author; Auto Mode manages capacity.

### 7.2 Two hand-built Auto Mode IAM roles (report §2, src [5])

> **Provenance note.** The CDK **logical IDs** in the real source
> (`eks_cluster_construct.py`) are `"ClusterRole"` (≈ line 109) and `"NodeRole"`
> (≈ line 136) — they are NOT `AmazonEKSAutoClusterRole` / `AmazonEKSAutoNodeRole`.
> The `AmazonEKSAuto*` names below are the AWS-managed-policy *concept* labels (the
> role of each role), not the construct's `iam.Role` construct ids. When porting,
> re-read the two `iam.Role(...)` calls from the real file and copy the logical IDs
> verbatim; the bracketed `<AmazonEKSAutoNodeRole>` token in §7.1 is likewise a
> concept label for the node role's ARN, not a source symbol.

- **The cluster role (logical id `ClusterRole`)** — the compute / block-storage /
  load-balancing / networking / cluster AWS-managed policies.
- **The node role (logical id `NodeRole`)** — `AmazonEKSWorkerNodeMinimalPolicy` +
  `AmazonEC2ContainerRegistryPullOnly` (this is where ECR *pull* lives; see §4.4).

### 7.3 Access entries (`CfnAccessEntry`)

- **(a) The Auto Mode NODE role** — type **`EC2`**, with **NO access policies** (an
  AWS EKS API constraint forbids access policies on `EC2`-type entries; report §2,
  src [4]). **Provenance:** the `EC2` access-entry type for the node role is an
  AWS-documentation reconstruction (the EKS access-entry API constraint), **not** a
  symbol read out of the VSR `eks_cluster_construct.py`. Confirm the exact
  `CfnAccessEntry` `type` string against the AWS EKS docs at build time and treat the
  VSR source as informative for the surrounding wiring only.
- **(b) The operator / CI kubectl identity** — `bootstrap_cluster_creator_admin_permissions`
  covers only the CFN exec role, **not** a human's or the CI role's kubectl identity,
  so each needs an explicit entry (report §2; discover-eks-cluster §6). Driven by
  `routeiq:admin_principal_arns`.

### 7.4 Pod Identity replaces the IRSA factory (the construct's biggest divergence)

A `pod_identity_association(self, cid, *, namespace, service_account, role)` helper
emitting a single **`eks.CfnPodIdentityAssociation`**:

```text
CfnPodIdentityAssociation
  cluster_name    = <cluster ref>
  namespace       = routeiq:sa_namespace
  service_account = routeiq:sa_name
  role_arn        = <the one pod role>
```

**NO `OpenIdConnectProvider`. NO `oidc_provider_issuer` derivation. NO `CfnJson`
trust map.** Call this divergence out explicitly when porting: the VSR source has all
three; the RouteIQ port deletes them. The two silent-failure traps the VSR construct
documents — (8a) the `.replace("https://")` no-op on an unresolved token, and (8b) the
L2 `addServiceAccount` hidden path that L1 forbids (discover-eks-cluster §8) — **no
longer apply**, because there is no token-keyed trust condition to get wrong.

#### 7.4a Defensive `eks-pod-identity-agent` addon (resolve the §3.1 #2 contradiction)

AWS docs claim the Pod Identity agent is built into Auto Mode (src [6]), **but the
production VSR Auto Mode construct this proposal ports installs it by hand** via
`eks.CfnAddon` (`vllm-sr-on-aws/cdk/lib/eks_cluster_construct.py`, the
`eks-pod-identity-agent` `CfnAddon` in `enable_container_insights()`; comment: *"The
Pod Identity Agent add-on must be present for Pod Identity associations to resolve"*).
That contradiction is unresolved (§3.1 #2). The safe, **cheap, idempotent** default is
to add the addon explicitly so the association is guaranteed to resolve regardless of
which behavior is live on the target cluster. Emit it in `eks_cluster_construct.py`
**before** the association:

```text
eks.CfnAddon
  addon_name        = "eks-pod-identity-agent"
  cluster_name      = <cluster ref>
  resolve_conflicts = "OVERWRITE"        # idempotent: re-applying a built-in addon
                                         #   is a no-op, so this is safe whether or not
                                         #   Auto Mode pre-installs the agent
  add_dependency(<the L1 CfnCluster>)    # the addon must depend on the cluster

# then pin ordering so the association never races the agent:
CfnPodIdentityAssociation.add_dependency(<this addon>)
```

This costs one always-present (free) managed addon and **removes the dependence on the
"pre-installed" claim being true.** The build wave MUST verify empirically (`aws eks
list-addons` on a live Auto Mode cluster) whether the agent is already present; if it
is provably built in, this addon stays a harmless `OVERWRITE` no-op rather than being
deleted — keeping the association guaranteed to resolve either way. This is the exact
shape VSR uses, so it is a faithful port, not net-new design.

### 7.5 `enable_container_insights()` — P0-minimal vs deferred

```text
P0-minimal (ship these):
  • eks.CfnAddon  amazon-cloudwatch-observability
  • logs.LogGroup RoutingLogGroup            (CDK-created — see the trap below)

PREP-ONLY at P0 / data-source-blocked (defer to P2):
  • logs.MetricFilter RoutingLatencyByModel
        metric_name = "routing_latency_ms_by_model"
        dimensions  = {"model": "$.[\"gen_ai.response.model\"]"}

Deferred to P2 (ADR-0027):
  • the 7 alarms, the dashboard, the SNS topic
```

> **Dimension-key fix (ADR-0027:64-67).** The VSR source filter dimensions on
> `$.selected_model`. RouteIQ does **not** emit `selected_model` as a structured field —
> the telemetry contract emits **`gen_ai.response.model`** (`telemetry_contracts.py:673`),
> so the RouteIQ filter must dimension on `$.["gen_ai.response.model"]`. A filter keyed on
> `$.selected_model` matches **zero** RouteIQ events.
>
> **`RoutingLatencyByModel` is PREP-ONLY at P0 (data-source-blocked).** The dimensioned
> filter has no data to match until P2. Its data source is the **structured
> `routing_decision` CW JSON log line**, which is itself a **P2 BUILD-NEW** item
> (roadmap:124-129). The gateway today emits an OTel span/`logger.info` event
> (`observability.py:784` `log_routing_decision`), **not** the CW JSON line the filter
> would scan — so the filter matches nothing until P2 lands the structured log line.
> Therefore the dimensioned `RoutingLatencyByModel` MetricFilter ships **PREP-ONLY /
> deferred** at P0 (the same posture as `alb_sg`, §8.3 / §11.1a). The
> `amazon-cloudwatch-observability` addon **and** the CDK-created `RoutingLogGroup` are
> fine to ship at P0 — only the dimensioned filter is data-source-blocked.
>
> **Live-ops trap (vllmsr-patterns lesson #8; gotcha
> `cdk-custom-resource-runtime-iam-invisible-to-synth` family).** The CW MetricFilter
> needs a **CDK-created log group that pre-exists.** A runtime-created (Fluent Bit)
> log group makes the CFN MetricFilter fail on first deploy. Create the log group in
> CDK, not at runtime (discover-eks-cluster §7).

### 7.6 CfnOutputs

```text
ClusterName        ClusterEndpoint        PodRoleArn
PodAssociationId   NodeRoleName           EcrGhcrPrefix
```

**DROPPED relative to the IRSA design:** `OidcProviderArn` and
`OidcProviderIssuerUrl` — there is no OIDC provider on the Pod Identity path
(discover-eks-cluster §10 lists those as the IRSA-only outputs). `PodAssociationId` is
the new operator-visible binding output.

---

## 8. The Network Construct  *(covers doneMeans (a))*

Port `network_construct.py` (discover-network). RouteIQ runs **one stateless pod**
(`emptyDir` only, no PVC) on Auto Mode in a **private** subnet; with no public egress
(or NAT-restricted), every AWS API the pod touches must reach a VPC endpoint or it
hangs.

### 8.1 VPC

- CIDR `10.40.0.0/16`, `max_azs=2`, `nat_gateways=1` (floor; bump to 2 for prod
  AZ-resilient egress).
- `enable_dns_hostnames=True` + `enable_dns_support=True` — **required for the
  `private_dns_enabled` interface endpoints**, or the Bedrock/Secrets SDK won't
  resolve to the endpoint (discover-network §6.4).

### 8.2 Subnet tiers

| Name | Type | Purpose |
|---|---|---|
| `public` | PUBLIC | ALB / NAT |
| `private-app` | PRIVATE_WITH_EGRESS | pod ENIs + ALL interface endpoints |
| `private-data` | PRIVATE_ISOLATED | reserved for P1 Aurora / cache |

Tag `public` with `kubernetes.io/role/elb=1` and `private` with
`kubernetes.io/role/internal-elb=1` for the Auto Mode managed LB.

### 8.3 Security groups (two-phase: instantiate ALL first, wire ingress after)

This avoids the CFN SG↔SG circular-dependency trap (discover-network §6.2). The
RouteIQ-minimal set:

| SG | Ingress |
|---|---|
| `alb_sg` | 443 from `Peer.any_ipv4()` (narrowable to an org allowlist via a future context flag) |
| `pod_sg` | the gateway pod ENI SG; ingress on the **uvicorn port 4000** from `alb_sg` (NOT VSR's Envoy 8080 / dashboard 8700) |
| `vpce_sg` | 443 from `pod_sg` — **the load-bearing one**, shared by all interface endpoints |

- **`alb_sg` is PREP-ONLY at P0.** Its `443 from Peer.any_ipv4()` ingress has **no
  consumer** until an operator flips the chart `service.type` to `LoadBalancer` — the
  chart default is `ClusterIP` / `ingress.enabled: false`
  (`deploy/charts/routeiq-gateway/values.yaml:69` and `:82`), so the Auto-Mode managed
  LB does not render by default and `alb_sg` backs nothing live. P0 provisions the SG
  as a forward-compatible seam only; the public edge is explicitly deferred. See §11.1a
  for the full edge decision and the matching `loadBalancerSourceRanges` PREP-ONLY chart
  edit.
- **DROP** VSR's `efs_sg` / `milvus_sg` / `backend_vllm_sg` (RouteIQ is stateless, no
  self-hosted backends).
- Add an `aurora_sg` **placeholder only as a P1 seam**, not P0.
- Every SG description stays in the EC2 charset allowlist — **no arrows, em-dashes,
  backticks, pipes, angle-brackets, or question-marks** (mulch
  `ec2-sg-description-charset`; discover-network §6.1).

### 8.4 P0-minimum interface endpoints (6) + S3 gateway

All share `vpce_sg`, land in `private-app`, `private_dns_enabled=True`
(discover-network §5):

| Endpoint | CDK service enum | Why P0 |
|---|---|---|
| ECR api | `ECR` | image auth/metadata to start the pod |
| ECR dkr | `ECR_DOCKER` | pull container layers |
| CloudWatch Logs | `CLOUDWATCH_LOGS` | pod logs; routing-decision lines feed the MetricFilter |
| Secrets Manager | `SECRETS_MANAGER` | master / provider keys |
| **Bedrock runtime** | **`BEDROCK_RUNTIME`** | **the data plane** — needs `aws-cdk-lib>=2.150.0` |
| S3 (gateway) | `S3` | ECR layer blobs + config download (route-table assoc on **both** private tiers; free, no SG) |

- **DROP `STS` from the P0-required set.** STS was load-bearing **ONLY** for IRSA's
  `AssumeRoleWithWebIdentity`. Pod Identity creds come from the pod-identity agent,
  not an STS endpoint call from the pod, so STS is **optional, not P0-required** on
  the Pod Identity path. (This is a direct consequence of the §3 flip; discover-network
  §5 listed STS under the IRSA assumption.)
- AppConfig × 2 + `aps-workspaces` + `XRay` arrive with **P2**. **Drop**
  `SagemakerRuntime` and the `sr.internal` Cloud Map namespace (VSR-specific).
- ElastiCache and RDS get **no** interface endpoints — they are VPC-resident resources
  reached over the SG matrix (P1).

### 8.5 nag note

Enable VPC flow logs (`flow_logs={...ALL...}`) to satisfy `AwsSolutions-VPC7`
(discover-network §6; see §13).

---

## 9. The ECR / GHCR Pull-Through Construct  *(covers doneMeans (c))*

Port `ecr_construct.py` (discover-ecr-ghcr). Parameterize the VSR "7 repos" **down** —
RouteIQ needs **ONE** gateway repo plus the cache rule.

### 9.1 The governance target is the PTC-CACHED repo, NOT a standalone repo

> **Load-bearing correction.** Deliverable (c)'s payoff — *"immutable-tag governance +
> scan-on-push on the cached copy"* (§9.4) — applies to the **repo the pull-through-cache
> rule auto-creates on first pull**, i.e.
> `<acct>.dkr.ecr.<region>.amazonaws.com/ghcr/baladithyab/routeiq` (the §9.4 pull path).
> A standalone `ecr.Repository` is **NOT** that repo and is **NOT** the pull target — the
> chart pulls the `ghcr/*` PTC path, so nobody pulls from a standalone repo and any
> governance on it is unobserved. PTC-created repos take their settings from a
> **`RepositoryCreationTemplate`** (§9.1b), not from any `ecr.Repository` construct, and
> ECR's default for an un-templated PTC repo is **tag-MUTABLE, no scan** (AWS ECR docs,
> *repository-creation-templates*). The governance therefore MUST be expressed as the
> template in §9.1b.

**Decision: DROP the standalone `ecr.Repository`.** It is not the pull target, so an
IMMUTABLE / scan-on-push repo construct buys nothing — the PTC rule never writes to it and
the chart never reads from it. The governance moves entirely to the
`RepositoryCreationTemplate` (§9.1b), which is what actually configures the `ghcr/*` cached
repo. If a future P-tier needs a *first-party* (non-cached) RouteIQ image repo — e.g. an
in-account build pushed directly rather than pulled through GHCR — reintroduce a standalone
repo **then**, scoped to that build's prefix. It has no purpose at P0.

### 9.1a Honest property split — immutable is a template prop, scan-on-push is registry-level

Two corrections to the naive "set IMMUTABLE + scanOnPush on the cached copy" framing:

- **Tag immutability** IS settable on PTC-created repos — via the template's
  `ImageTagMutability=IMMUTABLE` prop (§9.1b).
- **Scan-on-push is NOT a `RepositoryCreationTemplate` property.** The CFN
  `AWS::ECR::RepositoryCreationTemplate` has no `ImageScanningConfiguration` field
  (verified against the CFN TemplateReference — its props are `AppliedFor`, `Prefix`,
  `ImageTagMutability`, `EncryptionConfiguration`, `LifecyclePolicy`, `RepositoryPolicy`,
  `ResourceTags`, `CustomRoleArn`). Scan-on-push for PTC-cached repos is a
  **registry-level** setting — `AWS::ECR::RegistryScanningConfiguration` with a
  `ScanningRule` (`ScanFrequency=SCAN_ON_PUSH`, `RepositoryFilter` matching `ghcr/*`),
  consistent with discover-ecr-ghcr §2 ("scanning is a registry-level setting, not a
  per-repo prop"). So "scan-on-push on the cached copy" is delivered by the registry
  scanning config (§9.1c), not a per-repo prop.

> **AWS recommends MUTABLE for PTC templates** so ECR can refresh a same-tagged cached
> image from upstream. Choosing `IMMUTABLE` means a re-pushed upstream `1.0.0-rc1` will
> NOT refresh the cached copy — **acceptable for RouteIQ** because the chart pins by
> `image.digest` (takes precedence) and uses immutable release tags
> (discover-ecr-ghcr §2; chart `values.yaml:32-39`). State this trade-off explicitly when
> porting.

### 9.1b RepositoryCreationTemplate — the governance that actually lands on the cached repo

Port VSR's `CfnRepositoryCreationTemplate` (the construct emits it at
`ecr_construct.py` ~`:145-160`), applied to the `ghcr` prefix:

```text
ecr.CfnRepositoryCreationTemplate
  prefix               = "ghcr"                  # MUST equal the PTC rule's EcrRepositoryPrefix (§9.2)
  applied_for          = ["PULL_THROUGH_CACHE"]  # the scenario that creates ghcr/* repos
  image_tag_mutability = "IMMUTABLE"             # immutable-tag governance on the cached copy
  # custom_role_arn    = <repo-creation role>    # REQUIRED only if the template sets ResourceTags or KMS;
                                                 #   omit for the minimal IMMUTABLE-only P0 template
```

- **`prefix` MUST match the PTC rule's `EcrRepositoryPrefix` (`ghcr`).** A
  `RepositoryCreationTemplate` is matched to a new repo by namespace prefix; if it does not
  match `ghcr/*`, ECR creates the cached repo with **default (MUTABLE, no scan)** settings
  and the governance silently does not apply.
- **`DependsOn` the PTC rule** so CloudFormation orders the template and the rule
  deterministically within the stack.
- **The template must exist BEFORE the first pull.** Template settings apply only at repo
  *creation* time; **a PTC-cached repo cannot be made immutable retroactively** (AWS ECR
  docs). If `ghcr/baladithyab/routeiq` already exists from a pull predating the template,
  its mutability is fixed and the repo must be re-created. Provision the template in the
  same `cdk deploy` that creates the PTC rule, ahead of any chart deploy that triggers a
  pull.

### 9.1c Registry-level scan-on-push for the cached copy (delivers the scan half of (c))

```text
ecr.CfnRegistryScanningConfiguration
  scan_type = "BASIC"                            # or "ENHANCED" (Inspector) per the deployment
  rules     = [ { scan_frequency     = "SCAN_ON_PUSH",
                  repository_filters = [ { filter = "ghcr/*", filter_type = "WILDCARD" } ] } ]
```

This is what makes *"scan-on-push on the cached copy"* true: the cached `ghcr/*` images are
scanned on import. (Registry scanning config is account/region-wide; scope its filter to
`ghcr/*` so it does not over-broaden to unrelated repos.)

### 9.2 GHCR pull-through cache (the load-bearing part)

A single L1 `ecr.CfnPullThroughCacheRule`:

```text
CfnPullThroughCacheRule
  ecr_repository_prefix = "ghcr"
  upstream_registry     = "github-container-registry"
  upstream_registry_url = "ghcr.io"
  credential_arn        = <ecr-pullthroughcache/* secret ARN>
```

**The credential secret is operator-provisioned out-of-band, NOT created by CDK**
(it holds a real GitHub PAT — no secrets in source; discover-ecr-ghcr §3.2). Hard
constraints:

- Name MUST be prefixed **`ecr-pullthroughcache/`** (the CFN `CredentialArn` pattern
  enforces this).
- Same account + region as the rule.
- **MUST use the AWS-managed `aws/secretsmanager` key** — ECR does **not** support a
  CMK for the PTC credential secret (a documented hard constraint; related failure
  note `kms-pending-deletion-blocks-ecr-pull`). **Provenance:** this credential-key
  posture is an **AWS-current documentation** constraint, **not** the shape of the VSR
  `ecr_construct.py` — that source threads a `kms_key: kms.IKey | None` param and uses
  `AES256` in its repository-creation template; it does NOT itself assert the
  `aws/secretsmanager`-only rule for the PTC credential secret. Verify the
  `CredentialArn` key requirement against current AWS ECR docs at build time.
- Contents `{"username", "accessToken"}`; the PAT needs `read:packages` scope.
- The construct takes the secret **ARN** as a param.

### 9.3 IAM split (discover-ecr-ghcr §3.3) — do NOT collapse onto the pod role

- **(a) The CDK / CFN exec role** needs `ecr:CreatePullThroughCacheRule`
  (identity-based) + `secretsmanager:GetSecretValue` / `DescribeSecret` on the PTC
  secret.
- **(b) The NODE / pull identity** needs `ecr:BatchImportUpstreamImage` + the normal
  pull set (`ecr:GetAuthorizationToken`, `ecr:GetDownloadUrlForLayer`,
  `ecr:BatchGetImage`, `ecr:BatchCheckLayerAvailability`).
- **Neither set goes on the application pod role** (§4.4).

### 9.4 Zero image-pipeline rework (the payoff)

CI keeps pushing to `ghcr.io/baladithyab/routeiq` **unchanged**. Only the chart's
`image.repository` is repointed **at deploy time** to:

```text
<acct>.dkr.ecr.<region>.amazonaws.com/ghcr/baladithyab/routeiq:<tag>
```

On first pull, ECR lazily creates + imports + scans the image into the **PTC-auto-created**
private in-VPC repo `ghcr/baladithyab/routeiq` (via the ECR interface endpoints from §8) —
this is the repo §9.1b's `RepositoryCreationTemplate` governs, NOT the dropped standalone
repo. **No Dockerfile change, no GitHub Actions change, no registry change**
(discover-ecr-ghcr §4). This is preferred over a raw GHCR `imagePullSecret` because it adds
private in-VPC pulls, scan-on-push on the cached copy (registry scanning config, §9.1c),
immutable-tag governance on the cached copy (`RepositoryCreationTemplate`, §9.1b), and
insulation from GHCR rate limits.

> **This repoint is REQUIRED, not optional — and it is a deploy-time override, not a
> default.** The chart's `values.yaml` default still points at `ghcr.io` with
> `global.imagePullSecrets: []`, so an unchanged `helm upgrade` pulls from `ghcr.io`
> directly and the pull-through cache is never populated. The override is specified as
> an explicit operator deploy step in **§11.6 (GAP-4)**, with the `EcrGhcrPrefix`
> CfnOutput (§7.6 / §10) as the source of truth for the prefix.

### 9.5 Per-region caveat

PTC rules are per-Region; `CredentialArn` updates **require replacement**. Flag for
any multi-region build.

---

## 10. `RouteIqStack` Wiring  *(covers doneMeans (d))*

Composition root mirrors `vllm_sr_eks_stack.py` (discover-scaffold-tests §1.4): typed
kwargs (`env_name`, `vpc_cidr`, `nat_gateways`, `admin_principal_arns`,
`sa_namespace`, `sa_name`, flag bools) read from `routeiq:` context in `app.py`.

**Instantiate in order:**

```text
1. NetworkConstruct
2. EksClusterConstruct(vpc=...)
3. EcrConstruct
4. the ONE pod role
   + pod_identity_association(namespace=sa_namespace,
                              service_account=sa_name,
                              role=pod_role)
5. CfnOutputs
6. self._suppress_nag()        # LAST in the ctor
```

- `Tags.of(self).add("routeiq:env", env_name)` + conditional cost tags **only when
  supplied** — this keeps the default synth byte-stable for the snapshot test (§12).
- **CfnOutputs the chart / CI consume:** `ClusterName` / `ClusterEndpoint` (for
  `aws eks update-kubeconfig` + `helm upgrade`), `PodRoleArn` + `PodAssociationId`
  (operator-visible binding), `NodeRoleName`, `EcrGhcrPrefix` (the chart
  `image.repository` override value).
- `_suppress_nag()` is called LAST so every resource exists before suppressions are
  applied by path (§13).

---

## 11. Chart Seam Wiring  *(covers doneMeans (e))*

These are the app-layer edits to `deploy/charts/routeiq-gateway/` that P0 must make so
the chart deploys cleanly onto the Auto Mode + Pod Identity substrate. **Cited by
file** (chart `1.0.0-rc1`).

### 11.1 GAP-1 (P0, genuine chart gap): add `loadBalancerSourceRanges`

**Confirmed gap.** `templates/service.yaml` emits only `type`, `ports`, `selector`,
and an optional `nodePort` (verified — the `nodePort` block is guarded by
`{{- if and (eq .Values.service.type "NodePort") .Values.service.nodePort }}`). It has
**no `spec.loadBalancerSourceRanges`** field and there is no
`service.loadBalancerSourceRanges` value in `values.yaml`
(discover-chart-seams §2 GAP-1).

Under EKS **Auto Mode managed load balancing**, the NLB CIDR allowlist is honored via
`spec.loadBalancerSourceRanges`, **NOT** the legacy
`service.beta.kubernetes.io/aws-load-balancer-internal` annotation (vllmsr-patterns
lesson #1; ADR-0030 lesson #1). **Without this, an Auto-Mode `LoadBalancer` Service is
open to `0.0.0.0/0`.** Two edits:

1. Add `service.loadBalancerSourceRanges: []` to `values.yaml`.
2. Render `spec.loadBalancerSourceRanges` in `templates/service.yaml` when
   `type == LoadBalancer` — guarded exactly like the existing `nodePort` block.

#### 11.1a P0 edge decision — `loadBalancerSourceRanges` ships PREP-ONLY (inert until the operator flips `service.type`)

**Decision: P0 explicitly DEFERS the public edge. The chart stays `service.type:
ClusterIP` / `ingress.enabled: false`, and the `loadBalancerSourceRanges` edit above
is PREP-ONLY — it is inert until an operator flips `service.type` to
`LoadBalancer`.** The reachability path at P0 is in-cluster (port-forward / a P-later
Ingress or ALB); the public internet-facing edge is a deliberate P-later decision,
not a P0 deliverable.

The chart default **today** is `service.type: ClusterIP`
(`deploy/charts/routeiq-gateway/values.yaml:69`) and `ingress.enabled: false`
(`:82`), so **the Auto-Mode managed LB does NOT render by default** — there is no
`LoadBalancer` Service for the NLB to back. This is why the GAP-1 edit above is
**inert-until-LoadBalancer**: the conditional render guard (kept exactly as the
existing `nodePort` block — see edit #2) means `spec.loadBalancerSourceRanges` only
materializes once `service.type == LoadBalancer`. Until then the field is present in
`values.yaml` as an empty default but emits nothing.

The corollary on the CDK side is that **`alb_sg` (§8.3) and its `443 from
Peer.any_ipv4()` ingress are likewise PREP-ONLY at P0** — a network seam that
**attaches a consumer only when the operator flips `service.type` to `LoadBalancer`**
(and supplies the source-range allowlist). Provisioning the SG at P0 is cheap and
keeps the substrate forward-compatible, but it carries no live public edge until the
flip. The narrowability of `alb_sg` ingress (to an org allowlist via a future context
flag) and the chart `loadBalancerSourceRanges` allowlist are the **two halves of the
same future edge-lock**; neither is load-bearing at P0.

> **If P0 instead wanted a live edge** (NOT the chosen path): set `service.type:
> LoadBalancer` in `values.yaml` (internet-facing, or an internal NLB via the
> `aws-load-balancer-internal`-equivalent Auto Mode annotation), which makes the
> Auto-Mode managed LB render and gives `alb_sg` + `loadBalancerSourceRanges` a live
> consumer. **P0 does not take this path** — it defers per the decision above and
> ships the chart edit + `alb_sg` as PREP-ONLY.

### 11.2 ServiceAccount binding under Pod Identity (the flip from IRSA)

The chart's ServiceAccount needs a **STABLE `(namespace, name)`** pinned by the CDK
`CfnPodIdentityAssociation`:

- Set `serviceAccount.create: true` and an explicit `serviceAccount.name` (or document
  the rendered fullname from `_helpers.tpl`'s `routeiq-gateway.serviceAccountName`).
- Pass that `(namespace, name)` pair into the CDK as
  `routeiq:sa_namespace` / `routeiq:sa_name`.
- **NO `eks.amazonaws.com/role-arn` annotation is added** — that annotation is
  IRSA-only.

This **SIMPLIFIES discover-chart-seams GAP-2 and GAP-3**: there is no annotation for
`kubectl apply -k` to clobber (so vllmsr-patterns lesson #6 / GAP-2 no longer bites
the IAM path), and IRSA token automount is not the credential path (so GAP-3's
"`automountServiceAccountToken: true` is mandatory for IRSA" requirement is retracted).

> **Explicit retraction.** The IRSA-annotation guidance in discover-chart-seams §1
> (the `eks.amazonaws.com/role-arn` pass-through seam), GAP-2 (the `apply -k`-clobbers-
> the-annotation hazard), and GAP-3 (IRSA token automount must be on) are **superseded
> by this proposal**. On the Pod Identity path the binding is a CDK-side association
> keyed on `(namespace, serviceAccount)`; `automountServiceAccountToken` stays at its
> chart default and is enabled only for Kubernetes leader election, not for AWS creds.

### 11.3 `AWS_REGION` injection (P0)

The chart emits **no `AWS_REGION` / `AWS_DEFAULT_REGION` env** (confirmed by grep over
`_helpers.tpl`), so a Bedrock or AMP client with no region fails. P0 must add:

```yaml
extraEnv:
  - name: AWS_REGION
    value: <region>
```

(discover-chart-seams §3.)

### 11.4 External Secrets seam (P0 prep; full wiring P1)

`externalsecret.yaml` references a `ClusterSecretStore` named `aws-secrets-manager`
that **it does not create**. The CDK / cluster must pre-provision that store **plus the
ESO controller's OWN IAM role** — a separate identity from the application pod role.
P0 may **stub** the store; populating `externalSecrets.data` is **P1**
(discover-chart-seams §3).

### 11.5 NetworkPolicy (informational; leave OFF at P0)

If enabled, the chart's `allowHttpsExternal` egress rule permits 443/80 to `0.0.0.0/0`
**except** the three RFC-1918 ranges — which **blocks in-VPC services and a PrivateLink
Bedrock endpoint** (a private IP). **P0 leaves `networkPolicy.enabled: false`**;
enabling it later requires explicit `egress.to` entries for Aurora (5432), ElastiCache
(6379), and the in-VPC Bedrock endpoint (vllmsr-patterns NetworkPolicy-egress lesson;
discover-chart-seams §4).

### 11.6 GAP-4 (P0, REQUIRED deploy-time override): repoint `image.repository` at the `EcrGhcrPrefix` output

**Confirmed gap — this is the step that actually exercises the §9 pull-through cache.**
The chart ships `image.repository: ghcr.io/baladithyab/routeiq`
(`deploy/charts/routeiq-gateway/values.yaml:33`) and `global.imagePullSecrets: []`
(`:18`). **An unchanged `helm upgrade` therefore pulls straight from `ghcr.io` and the
ECR GHCR pull-through cache from §9 is NEVER populated** — the construct is provisioned
but dead. §9.4 only *narrates* the repoint as a payoff; this subsection promotes it to a
**required, explicit operator deploy step** (consistent with the rest of §11's seam
edits) so the cache is actually on the pull path.

**The `EcrGhcrPrefix` CfnOutput (§7.6 / §10) is the source of truth** for the override
value. At deploy time the operator MUST repoint `image.repository` from
`ghcr.io/baladithyab/routeiq` to the cache-fronted path:

```text
<EcrGhcrPrefix>/baladithyab/routeiq        # where EcrGhcrPrefix =
<acct>.dkr.ecr.<region>.amazonaws.com/ghcr
```

This is a **deploy-time value override, not a `values.yaml` default flip.** The chart's
`values.yaml` default stays `ghcr.io/baladithyab/routeiq` so the chart remains
account/region-agnostic (usable outside this AWS substrate, and the snapshot baseline
does not bake in an account id). The operator supplies the override at `helm upgrade`
time, sourcing the prefix from the stack output. The documented operator step (lands in
`deploy/cdk/README.md`'s operator-deploy section, alongside the `aws eks
update-kubeconfig` + `helm upgrade` flow that consumes `ClusterName` / `ClusterEndpoint`):

```bash
# 1. Read the prefix from the deployed RouteIqStack (the source of truth)
PREFIX=$(aws cloudformation describe-stacks \
  --stack-name "RouteIqStack-${ENV}" \
  --query "Stacks[0].Outputs[?OutputKey=='EcrGhcrPrefix'].OutputValue" \
  --output text)            # -> <acct>.dkr.ecr.<region>.amazonaws.com/ghcr

# 2. Repoint image.repository at deploy time (override, NOT a values.yaml edit)
helm upgrade --install routeiq-gateway ./deploy/charts/routeiq-gateway \
  --set image.repository="${PREFIX}/baladithyab/routeiq" \
  --set image.tag="${IMAGE_TAG}"        # routeiq:image_tag (e.g. 1.0.0-rc1)
```

> **Why this is the cache trigger.** On the FIRST pull against the
> `${PREFIX}/baladithyab/routeiq` path, ECR lazily imports + scans the upstream GHCR
> image into the private in-VPC repo (via the ECR interface endpoints, §8). Pulling the
> unchanged `ghcr.io` default instead skips ECR entirely — no lazy import, no
> scan-on-push, no immutable-tag governance, no rate-limit insulation — and with
> `global.imagePullSecrets: []` a private upstream image would fail auth outright. **The
> repoint is what makes §9 load-bearing rather than cosmetic.** (discover-ecr-ghcr §4;
> §9.4 narrates the same repoint.)

**Acceptance for this seam:** the operator deploy step is documented (CDK README +
chart deploy docs) with `EcrGhcrPrefix` named as the source-of-truth output, and the
override is exercised at `helm upgrade` time. No `values.yaml` default change is made —
the default stays `ghcr.io/...`. Optionally a `helm template` / chart unit test asserts
that `--set image.repository=<prefix>/...` flows through to the rendered Deployment
image, but the load-bearing artifact is the documented step.

---

## 12. The Cred-Free Test Suite  *(covers doneMeans (f))*

**The gate command:**

```bash
uv run pytest deploy/cdk/tests/
```

Runs **fully offline** because every test calls `Template.from_stack(stack)` after
passing an explicit dummy `env=cdk.Environment(account="123456789012",
region="us-west-2")` — a concrete-but-fake account/region. **No AWS creds, no `cdk`
CLI, no network.** Suitable for the pre-push hook / CI (discover-scaffold-tests §1.6,
§3).

### 12.1 `test_eks_cluster.py`

- `has_resource_properties("AWS::EKS::Cluster", {...})` asserting the three Auto Mode
  blocks `enabled`.
- The node `CfnAccessEntry` is type `EC2`.
- **A `CfnPodIdentityAssociation` exists with the right `namespace` / `serviceAccount`**
  (the Pod Identity analogue of VSR's IRSA trust-key test).
- **An `AWS::EKS::Addon` `eks-pod-identity-agent` exists with
  `ResolveConflicts: OVERWRITE`** (the §7.4a defensive addon), and the
  `CfnPodIdentityAssociation` declares a `DependsOn` the addon so the association is
  guaranteed to resolve regardless of whether Auto Mode pre-installs the agent
  (resolves the §3.1 #2 contradiction).
- The IAM-role-`Description`-is-ASCII guard (an em-dash → IAM CREATE failure; §4.5).
- Flag-on/off byte-stability.
- Use `find_resources` / `Match.object_like` over brittle full counts (mulch
  `cdk-resource-count-test-tripwire`).

### 12.2 NEW negative regression guard (replaces VSR's https-strip test)

Assert the template contains **NO `Custom::AWSCDKCfnJson`** resource **AND NO
`AWS::IAM::OIDCProvider`**. This proves the Pod Identity path took and that an
accidental IRSA regression did not creep in. **This is the regression guard for THIS
proposal's decision** — it replaces VSR's `test_irsa_trust_keys_strip_https_scheme`
(which asserted the `CfnJson` TrustKeys value stripped `https://`), because RouteIQ
has no `CfnJson` and no OIDC provider to test.

### 12.3 `test_ecr.py`

- The GHCR `CfnPullThroughCacheRule` (`upstream_registry_url="ghcr.io"`).
- **The `CfnRepositoryCreationTemplate`** exists with `AppliedFor=["PULL_THROUGH_CACHE"]`,
  `Prefix="ghcr"`, and `ImageTagMutability="IMMUTABLE"` (the immutable-tag governance lands
  on the PTC-cached `ghcr/*` repo — §9.1b). This replaces the old standalone-repo
  `image_tag_mutability`/`image_scan_on_push` prop assertions.
- The registry scan-on-push for `ghcr/*` (`AWS::ECR::RegistryScanningConfiguration`,
  `ScanFrequency=SCAN_ON_PUSH`, §9.1c).
- **No standalone `AWS::ECR::Repository`** is emitted by the construct (the standalone repo
  was dropped — §9.1; it was never the pull target).

### 12.4 `test_routeiq_stack.py`

- The pod role has **exactly** the P0 grant set (Bedrock invoke + Secrets + S3 + Logs).
- The static `pods.eks.amazonaws.com` trust (and, implicitly via §12.2, the absence of
  a web-identity trust).
- The CfnOutputs exist.
- VPC CIDR is `10.40.0.0/16`.

### 12.5 `test_cdk_nag.py`

- `_synth_with_nag(**flags)` adds `AwsSolutionsChecks`, asserts **NO surviving
  `AwsSolutions-*` errors AND no `CdkNagValidationFailure.*`** at both the default AND
  the maximal flag surface.
- Failure messages render the offending `id` + `data[:200]` so they are actionable.

### 12.6 `test_template_snapshot.py`

- `_synth_template → to_json → _canonicalise (json.dumps(sort_keys=True, indent=2)) →
  _assert_snapshot` against `__snapshots__/dev.json`.
- Auto-write **ONLY** under `UPDATE_SNAPSHOTS=1`; **a missing baseline is a LOUD
  failure** (no silent create — a false-green hazard).
- Commit the flag-off `dev.json` baseline once
  (`UPDATE_SNAPSHOTS=1 uv run pytest deploy/cdk/tests/snapshot/`).
- An `autouse` fixture monkeypatches any Docker-bundled-asset path globals to a
  non-existent dir so synth stays hermetic regardless of host Docker
  (discover-scaffold-tests §1.6).

---

## 13. cdk-nag Suppressions  *(covers doneMeans (g))*

Port the **SHAPE** of `nag_suppressions.py`, **NOT** the 2400-line VSR file
(discover-scaffold-tests §1.5). `apply_nag_suppressions(stack)` fans out to
`_suppress_<construct>(stack)` helpers; each calls:

```python
NagSuppressions.add_resource_suppressions_by_path(
    stack,
    "/<stack_id>/<Construct>/<Resource>/Resource",
    [NagPackSuppression(id=..., reason=...)],
)
```

- **Flag-gated families** are guarded with `getattr(stack, "x", None)` because
  `add_*_by_path` **RAISES** on an absent path under cdk-nag `>=2.27`.
- **Evidenced-suppression shape (the only thing to copy):** every suppression carries
  an `id` (e.g. `AwsSolutions-IAM4` / `-IAM5` / `-EKS1` / `-VPC3`), an explicit
  `appliesTo` list (specific managed-policy ARNs or `Resource::*`), and a `reason`
  stating **(a)** why it is safe, **(b)** that it is least-privilege / the only valid
  form, and **(c)** an `Owner:` line. **RouteIQ-specific reasons — do not reuse VSR's.**

### Expected P0 suppressions to enumerate

| nag id | What | Reason shape |
|---|---|---|
| `AwsSolutions-IAM4` | the 5 AWS-managed Auto Mode policies on the cluster/node roles | "Auto Mode REQUIRES these exact AWS-published policies, not replaceable with a narrower custom policy. Owner: EksClusterConstruct" |
| `AwsSolutions-IAM5` | `bedrock:InvokeModel` on `foundation-model/*` | "model set is deploy-time dynamic; scoped to the bedrock service + region; the only valid form for a router. Owner: RouteIqStack" |
| (ALB ingress) | the public 443 ALB ingress, if flagged | least-privilege rationale + the CIDR-lock note from §11.1 |
| (ECR/log wildcards) | as needed | scoped-resource + least-privilege rationale |

Each suppression carries the 3-part reason.

---

## 14. Risks, Open Concerns & Operator-Gated Items

- **ADR-0030 must be amended / superseded.** Its title and body assume IRSA; this
  proposal flips to Pod Identity. **File the ADR amendment as part of P0.** The
  cluster machinery (L1 `CfnCluster`, 3 Auto Mode blocks, access entries, container
  insights, the CDK/Helm boundary) is unchanged; only the pod-IAM mechanism, the
  construct's `pod_identity_association()` helper, and the dropped OIDC/CfnJson
  surface change.
- **The four ported constructs are not yet in this repo, but the VSR source IS
  present locally.** Builders must **re-derive every symbol from the real source at
  `/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/` before copying** (it is
  confirmed checked out — see §2). Use the discover docs only as an index into that
  source, not as a substitute; where a discover doc and the source disagree, the
  source wins (`verify-external-audit-before-acting`).
- **Open question — account topology (single vs cross-account).** Single-account
  makes the Pod Identity verdict lopsided; cross-account adds one `targetRoleArn`
  prop (and ~59-minute credential-cache role chaining) but still favors Pod Identity.
  **Operator must confirm** before the build wave.
- **Operator-gated, NOT part of the cred-free gate:** `cdk bootstrap`, `cdk deploy`,
  `kubectl`, `helm`, provisioning the `ecr-pullthroughcache/` GitHub PAT secret, and
  the EKS Auto Mode CIDR-lock decision (public-CIDR-locked vs private+bastion;
  roadmap P0 operator note).
- **P1 forward note (gotcha `cdk-rds-aurora-engine-version-retired-at-deploy`).** When
  the P1 `RouteIqStateStack` lands, pin the Aurora engine version against a version
  that is **not retired at deploy time** — a retired engine version passes synth but
  yields a "Cannot find version" CREATE failure. Out of scope for P0 (no Aurora here),
  flagged so the P1 wave carries it.
- **Carry the synth-time guards:** ASCII-only IAM descriptions (§4.5);
  VPC-quota / retain-resources on teardown (roadmap P0 operator note).

### The 10 live-ops lessons and where each lands in P0

These are the hard-won operational foot-guns from the `vllm-sr-on-aws` team
(`vllmsr-patterns.md` §"HA / ops lessons" #1–#8) plus the two roadmap P0 operator
notes. The P0-relevant ones are handled as marked; the rest are flagged for later
waves.

| # | Lesson | P0 handling |
|---|---|---|
| 1 | NLB CIDR-lock = `spec.loadBalancerSourceRanges`, NOT the LB annotation | **P0 — handled, §11.1** (GAP-1 chart edit) |
| 2 | IRSA trust-key must be the ARN-derived OIDC issuer, never `.replace("https://")` (silent no-op on a token → runtime AccessDenied) | **Moot on the Pod Identity path** — no token-keyed trust condition exists; the negative guard in §12.2 proves it |
| 3 | RWO-EBS + `RollingUpdate` = Multi-Attach deadlock → `strategy: Recreate` | **Moot — RouteIQ stays stateless** (`emptyDir` only); flagged as a tripwire if a persistent EBS PVC is ever added (discover-eks-cluster §9.4) |
| 4 | APM webhook is PSA-restricted → needs an explicit opt-out label on injected pods | Deferred (P2 observability wave) |
| 5 | Bearer tokens ≤ 12h rotation via a CronJob (cross-account capacity) | **Moot — RouteIQ drops the bearer-minter role** (single account, one pod role, §4.1) |
| 6 | `kubectl apply -k` clobbers SA annotations | **Moot for the IAM path** — Pod Identity adds no SA annotation to clobber (§11.2); CI still uses `helm upgrade`, not kustomize |
| 7 | Aurora rollback ≈ 30 min → deploy the DB in a SEPARATE CI stage from the app | **P0 — honored by scope, §6** (Aurora is P1 / a separate `RouteIqStateStack`; P0 = one stack) |
| 8 | CW MetricFilter needs a CDK-created log group that pre-exists | **P0 — partially, §7.5.** CDK-created `RoutingLogGroup` + the `amazon-cloudwatch-observability` addon **ship at P0**; the dimensioned `RoutingLatencyByModel` MetricFilter ships **PREP-ONLY / deferred to P2** (data-source-blocked — its source, the structured `routing_decision` CW JSON line, is a P2 BUILD-NEW item, roadmap:124-129; today the gateway emits an OTel/`logger.info` event, `observability.py:784`, not the CW JSON line). Its dimension key is corrected to `$.["gen_ai.response.model"]` per ADR-0027:64-67 (`selected_model` is never emitted). |
| 9 | ASCII-only IAM role descriptions at synth time (em-dash passes synth, FAILS the IAM CREATE API) | **P0 — handled, §4.5 + §12.1** (asserted by a unit test) |
| 10 | VPC-quota / retain-resources on teardown | **P0 — operator-gated**, §14 (carried as a teardown guard; the PTC rule + `RepositoryCreationTemplate` persist the `ghcr/*` cache settings, §9.1b — the standalone RETAIN repo is dropped, §9.1) |

---

## 15. Acceptance / Done Criteria (map to doneMeans a–g)

| ID | Criterion | Section |
|---|---|---|
| **(a)** | Network construct ported — VPC / SGs / 6 endpoints incl. `BEDROCK_RUNTIME` + S3 gateway | §8 |
| **(b)** | EKS cluster construct ported with the **Pod Identity decision baked in** (no OIDC, no `CfnJson`); a **defensive `eks-pod-identity-agent` `CfnAddon` (`resolve_conflicts=OVERWRITE`, depends-on cluster)** is emitted so the `CfnPodIdentityAssociation` resolves regardless of the "pre-installed on Auto Mode" claim (§3.1 #2 / §7.4a), with the association declaring `DependsOn` the addon; `enable_container_insights` ships the `amazon-cloudwatch-observability` addon + CDK-created `RoutingLogGroup` at P0, while the dimensioned `RoutingLatencyByModel` MetricFilter ships **PREP-ONLY / deferred to P2** (data-source-blocked; key corrected to `$.["gen_ai.response.model"]` per ADR-0027:64-67) | §3, §7, §7.4a, §7.5 |
| **(c)** | ECR / GHCR construct ported — GHCR pull-through + `RepositoryCreationTemplate` (IMMUTABLE) + registry scan-on-push on the **cached** `ghcr/*` repo (no standalone repo) | §9 |
| **(d)** | `RouteIqStack` wiring + `cdk.json` `routeiq:` context keys (**ONE** stack) | §5, §6, §10 |
| **(e)** | Chart `loadBalancerSourceRanges` + SA `(namespace, name)` binding + `AWS_REGION` + `image.repository`→`EcrGhcrPrefix` deploy-time override | §11 |
| **(f)** | Cred-free `uv run pytest deploy/cdk/tests/` synth + nag suite | §12 |
| **(g)** | cdk-nag evidenced suppressions | §13 |

---

> **Reminder of scope.** This is a proposal. It writes **no CDK `.py`**. The
> `deploy/cdk/` tree, the constructs, the chart edits, and the test suite specified
> above are the contract the next, **operator-gated build wave** implements — gated on
> the §14 open questions (account topology and the CIDR-lock-vs-bastion decision) and
> the ADR-0030 superseding amendment.
