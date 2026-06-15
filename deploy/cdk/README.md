# RouteIQ CDK Foundation (`deploy/cdk/`)

RouteIQ's P0 provisioning IaC: a single **`RouteIqStack`** that stands up a
private multi-AZ **EKS Auto Mode** cluster, an **ECR** repository fronted by a
**GHCR pull-through cache**, one least-privilege pod IAM role bound via **EKS
Pod Identity** (a single `CfnPodIdentityAssociation`, no IRSA / no OIDC
provider), and a credential-free `pytest` synth + cdk-nag gate.

> **Scope.** P0 is the substrate only. The CDK provisions cluster + IAM + ECR +
> network; the existing Helm chart (`deploy/charts/routeiq-gateway/`) deploys
> the app. The CDK never runs `KubernetesManifest` or `HelmChart`. P1 (Aurora +
> ElastiCache) is a separate stack and is out of P0.
>
> See `docs/architecture/aws-rearchitecture/31-p0-cdk-foundation-proposal.md`
> for the full design. This directory is the scaffold; constructs land in a
> later build stage.

## Layout

```
deploy/cdk/
├── app.py            # CDK app: reads routeiq:* context, builds RouteIqStack-<env>,
│                     #   adds the AwsSolutionsChecks aspect, synthesizes
├── cdk.json          # "app": "python3 app.py" + routeiq:* context defaults
├── requirements.txt  # the four pins (also carried by the root pyproject `cdk` extra)
├── README.md         # this file
├── lib/              # constructs + RouteIqStack composition root (next stage)
└── tests/            # credential-free pytest synth + cdk-nag + snapshot gate
```

## Synthesizing

Install the CDK dependencies into the uv environment (the uv-native path — one
lockfile):

```bash
uv sync --extra cdk
```

Or, standalone:

```bash
uv pip install -r deploy/cdk/requirements.txt
```

Then, from `deploy/cdk/`:

```bash
# Synthesize the dev stack (requires the cdk CLI + AWS creds for context lookups)
cdk synth --context routeiq:env=dev

# Override any routeiq: key at synth/deploy time
cdk synth --context routeiq:enable_ghcr_ptc=false
```

> Boolean overrides via `--context` arrive as the string `"false"` / `"true"`;
> the app's `_bool_ctx` helper parses these correctly (a naive `bool("false")`
> is `True` in Python). Always set booleans through `routeiq:*` context keys.

## The cred-free test gate

The test suite calls `Template.from_stack(stack)` after passing an explicit
dummy `env=cdk.Environment(account="123456789012", region="us-west-2")`, so it
synthesizes fully offline — no AWS creds, no `cdk` CLI, no network. Suitable for
the pre-push hook / CI:

```bash
uv run pytest deploy/cdk/tests/
```

The committed snapshot baseline `tests/snapshot/__snapshots__/dev.json` is
generated once and then enforced; a missing baseline is a loud failure (never a
silent create). To (re)generate it intentionally:

```bash
UPDATE_SNAPSHOTS=1 uv run pytest deploy/cdk/tests/snapshot/
```

## `routeiq:` context keys

| Key | Default | Purpose |
|---|---|---|
| `routeiq:env` | `dev` | environment suffix on the stack id (`RouteIqStack-<env>`) |
| `routeiq:vpc_cidr` | `10.40.0.0/16` | VPC CIDR (distinct from VSR 10.20 / 10.30 so a future peering stays overlap-free) |
| `routeiq:nat_gateways` | `1` | NAT gateway count (floor; bump to 2 for prod AZ-resilient egress) |
| `routeiq:k8s_version` | `1.33` | EKS Auto Mode cluster version |
| `routeiq:enable_ghcr_ptc` | `true` | GHCR pull-through cache rule |
| `routeiq:sa_namespace` | `routeiq` | Pod Identity association binding — MUST match the chart's rendered ServiceAccount namespace |
| `routeiq:sa_name` | `routeiq-gateway` | Pod Identity association binding — MUST match the chart's rendered ServiceAccount name |
| `routeiq:image_tag` | `1.0.0-rc1` | chart image tag override (see operator deploy below) |
| `routeiq:admin_principal_arns` | `[]` | `CfnAccessEntry` for the CI/operator kubectl identity |
| `routeiq:bedrock_model_arns` | `[]` | scopes the `BedrockInvoke` statement (foundation-model / inference-profile ARNs) |
| `routeiq:config_s3_bucket` | `null` | scopes the `ConfigS3Read` statement (config bucket) |
| `routeiq:secret_arns` | `[]` | scopes the `SecretsRead` statement (master / provider-key secrets) |

> The Pod Identity binding pins the chart's ServiceAccount `(namespace, name)`.
> There is **NO `eks.amazonaws.com/role-arn` annotation** on the Pod Identity
> path — that annotation is IRSA-only. Set `serviceAccount.create: true` and an
> explicit `serviceAccount.name` in the chart, and pass that pair into the CDK
> via `routeiq:sa_namespace` / `routeiq:sa_name`.

## Operator deploy (post-`cdk deploy`)

`cdk bootstrap`, `cdk deploy`, `kubectl`, `helm`, and provisioning the GHCR PAT
secret are operator-gated steps (NOT part of the cred-free test gate). After
the stack is deployed:

### 1. Provision the GHCR credential secret (out-of-band, NOT created by CDK)

The pull-through cache rule references a Secrets Manager secret holding a real
GitHub PAT. The CDK does **not** create it (no secrets in source). The operator
provisions it once:

- Name MUST be prefixed `ecr-pullthroughcache/` (the ECR `CredentialArn` pattern
  enforces this).
- Same account + region as the rule; AWS-managed `aws/secretsmanager` key (ECR
  does not support a CMK for the PTC credential secret).
- Contents `{"username", "accessToken"}`; the PAT needs `read:packages` scope.

### 2. Connect kubectl to the cluster

```bash
aws eks update-kubeconfig \
  --name "$(aws cloudformation describe-stacks \
    --stack-name "RouteIqStack-${ENV}" \
    --query "Stacks[0].Outputs[?OutputKey=='ClusterName'].OutputValue" \
    --output text)" \
  --region "${AWS_REGION}"
```

### 3. Repoint `image.repository` at the ECR GHCR cache (REQUIRED deploy-time override)

The chart's `values.yaml` default points at `ghcr.io/baladithyab/routeiq` and
ships `global.imagePullSecrets: []`. An unchanged `helm upgrade` therefore pulls
straight from `ghcr.io` and the pull-through cache is **never populated**. The
`EcrGhcrPrefix` CfnOutput is the source of truth for the override value:

```bash
# Read the prefix from the deployed RouteIqStack (the source of truth)
PREFIX=$(aws cloudformation describe-stacks \
  --stack-name "RouteIqStack-${ENV}" \
  --query "Stacks[0].Outputs[?OutputKey=='EcrGhcrPrefix'].OutputValue" \
  --output text)            # -> <acct>.dkr.ecr.<region>.amazonaws.com/ghcr

# Repoint image.repository at deploy time (override, NOT a values.yaml edit)
helm upgrade --install routeiq-gateway ./deploy/charts/routeiq-gateway \
  --set image.repository="${PREFIX}/baladithyab/routeiq" \
  --set image.tag="${IMAGE_TAG}"        # routeiq:image_tag (e.g. 1.0.0-rc1)
```

> This is a **deploy-time value override, not a `values.yaml` default flip** —
> the chart stays account/region-agnostic. On the first pull against the
> `${PREFIX}/baladithyab/routeiq` path, ECR lazily imports + scans the upstream
> GHCR image into the private in-VPC repo (via the ECR interface endpoints). The
> repoint is what makes the pull-through cache load-bearing rather than
> cosmetic. See proposal §9.4 and §11.6.
