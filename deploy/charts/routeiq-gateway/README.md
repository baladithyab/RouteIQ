# RouteIQ Gateway Helm Chart

A Helm chart for deploying the RouteIQ Gateway - an intelligent LLM gateway with ML-powered routing capabilities.

## Prerequisites

- Kubernetes 1.23+
- Helm 3.8+
- (Optional) External secrets operator for production secret management

## Quick Start

### Install with default values

```bash
helm install routeiq-gateway ./deploy/charts/routeiq-gateway
```

### Install with custom values

```bash
helm install routeiq-gateway ./deploy/charts/routeiq-gateway -f myvalues.yaml
```

### Example: Production installation with existing secrets

```bash
helm install routeiq-gateway ./deploy/charts/routeiq-gateway \
  --namespace llm-gateway \
  --create-namespace \
  --set image.tag=1.82.0 \
  --set replicaCount=3 \
  --set secrets.existingSecret=routeiq-credentials \
  --set autoscaling.enabled=true \
  --set podDisruptionBudget.enabled=true \
  -f production-values.yaml
```

## Configuration Reference

See [`values.yaml`](values.yaml) for the full list of configurable parameters.

### Required Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `secrets.existingSecret` OR `secrets.create` | API keys must be provided via Secret | `""` / `false` |
| `config.gateway` | Gateway config YAML (model_list, etc.) | Example config |

### Image Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image repository | `ghcr.io/baladithyab/litellm-llm-router` |
| `image.tag` | Container image tag (defaults to Chart appVersion) | `""` |
| `image.digest` | Container image digest (takes precedence over tag) | `""` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |

### Scaling & Availability

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `2` |
| `autoscaling.enabled` | Enable HPA | `false` |
| `autoscaling.minReplicas` | Minimum replicas for HPA | `2` |
| `autoscaling.maxReplicas` | Maximum replicas for HPA | `10` |
| `podDisruptionBudget.enabled` | Enable PDB | `false` |
| `podDisruptionBudget.minAvailable` | Minimum available pods | `1` |

### Resources

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resources.requests.memory` | Memory request | `512Mi` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.limits.memory` | Memory limit | `2Gi` |
| `resources.limits.cpu` | CPU limit | `2000m` |

### Health Probes

The chart configures health probes using the gateway's internal health endpoints:

| Probe | Endpoint | Purpose |
|-------|----------|---------|
| Liveness | `/_health/live` | Basic health check (no external deps) |
| Readiness | `/_health/ready` | Full health check (includes DB/Redis if configured) |
| Startup | `/_health/live` | Slow-start support for model loading (enabled by default, allows up to 5 min) |

### Routing & Strategy

RouteIQ-specific routing and strategy configuration:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `routeiq.pluginStrategy.enabled` | Use plugin-based routing strategy (recommended) | `true` |
| `routeiq.workers` | Uvicorn workers per pod (NOT K8s replicas) | `2` |
| `routeiq.centroidRouting.enabled` | Enable centroid-based routing (~2ms latency) | `true` |
| `routeiq.centroidRouting.warmup` | Pre-warm centroid classifier at startup | `false` |
| `routeiq.routingProfile` | Default routing profile: auto/eco/premium/free/reasoning | `"auto"` |
| `routeiq.adminUI.enabled` | Enable admin UI dashboard at `/ui/` | `false` |

> **Note**: `routeiq.workers` controls per-pod uvicorn workers and is independent of
> `replicaCount` (K8s pod replicas). Multi-worker mode is safe when
> `routeiq.pluginStrategy.enabled` is `true` (the default). With the legacy monkey-patch
> strategy, only 1 worker is supported.

### Secrets Management

**Option 1: Create secrets via Helm (development only)**
```yaml
secrets:
  create: true
  values:
    LITELLM_MASTER_KEY: "your-master-key"
    OPENAI_API_KEY: "sk-..."
```

**Option 2: Use existing Kubernetes secret (recommended)**
```yaml
secrets:
  existingSecret: "my-gateway-secrets"
```

Required keys in your secret:
- `LITELLM_MASTER_KEY` - Master API key for admin access
- Provider API keys as needed: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_API_KEY`, etc.

**Option 3: External Secrets Operator (production)**
```yaml
externalSecrets:
  enabled: true
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  data:
    - secretKey: LITELLM_MASTER_KEY
      remoteRef:
        key: routeiq/master-key
```

### Gateway Configuration

The gateway configuration is stored in a ConfigMap and mounted at `/app/config/config.yaml`:

```yaml
config:
  gateway: |
    model_list:
      - model_name: gpt-4o
        litellm_params:
          model: openai/gpt-4o
      - model_name: claude-3-5-sonnet
        litellm_params:
          model: anthropic/claude-3-5-sonnet-20241022
    litellm_settings:
      drop_params: true
    general_settings:
      master_key: env/LITELLM_MASTER_KEY
```

### Optional Features

Enable optional Kubernetes resources:

```yaml
# Horizontal Pod Autoscaler
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10

# Pod Disruption Budget
podDisruptionBudget:
  enabled: true
  minAvailable: 1

# Ingress
ingress:
  enabled: true
  className: nginx
  hosts:
    - host: api.example.com
      paths:
        - path: /
          pathType: Prefix

# Network Policy
networkPolicy:
  enabled: true
  egress:
    allowDns: true
    allowHttpsExternal: true
```

## RouteIQ-on-AWS state plane (P1: Aurora + ElastiCache)

The managed state plane (Aurora Postgres Serverless v2 — ADR-0028; ElastiCache
Serverless Valkey — ADR-0029) is provisioned by the **separate** `RouteIqStateStack`
CDK stack (NOT the P0 `RouteIqStack`; the two are separate stacks per the
~30-minute-rollback rule). The chart consumes that stack's `CfnOutput`s as
`helm upgrade --set` flags — the same "CfnOutput string → `--set` value" seam P0
uses for `EcrGhcrPrefix` / `ClusterName`, never a CDK-side `HelmChart` resource.

### Wire the chart host seams from the state-stack CfnOutputs

```bash
STACK=RouteIqStateStack-<env>
q() { aws cloudformation describe-stacks --stack-name "$STACK" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }

helm upgrade routeiq deploy/charts/routeiq-gateway \
  --set externalPostgresql.host="$(q DbClusterEndpoint)" \
  --set externalRedis.host="$(q CacheEndpoint)" \
  --set externalRedis.port="$(q CachePort)" \
  --set-string externalRedis.ssl=true        # serverless Valkey = TLS-mandatory
```

State-stack outputs: `DbClusterEndpoint` (Aurora writer → `externalPostgresql.host`),
`DbClusterReaderEndpoint`, `DbSecretArn` (master/break-glass only — the pod
IAM-auths and does NOT read it), `DbRuntimeUser` (= `externalPostgresql.username`
= `routeiq`), `DbClusterResourceId`, `CacheEndpoint` / `CachePort`
(→ `externalRedis.host`/`.port`), `CacheIamUserName` (the cache IAM user the pod
presents on connect; `user_id == user_name`).

The seam is one-directional and inert until `host` is non-empty: leave `host`
empty and the gateway degrades gracefully (no DB, no Redis); set it and the whole
state-plane wiring turns on.

### IAM-auth boot-render (no static password)

> **✅ Shipped in P1-fix (RouteIQ-d3a4).** The in-process token-minting below is
> now implemented and DEFAULTS OFF. `database.py:get_db_pool()` mints a 15-min
> `rds-db:connect` token per connection and passes it to asyncpg as a callable
> password (refresh-per-reconnect) behind `ROUTEIQ_DB_IAM_AUTH`; `redis_pool.py`
> presents `REDIS_USERNAME` + a short-lived `elasticache:Connect` SigV4 token as
> the Redis AUTH behind `ROUTEIQ_REDIS_IAM_AUTH`. The chart emits both flags
> automatically on the empty-`existingSecret` IAM path (and `REDIS_USERNAME` from
> `externalRedis.iamUserName`). Both flags default OFF, so the static-password /
> static-AUTH interim (Shape B, `existingSecret` set) remains fully supported.

On the ADR-0028/0029 IAM-auth path there is **no static credential**.
Leave `externalPostgresql.existingSecret` / `externalRedis.existingSecret`
**empty**: the chart renders a complete, password-less `DATABASE_URL` at
boot-render time and the app (`database.py` / `redis_pool.py`) mints the
15-min `rds-db:connect` / `elasticache:Connect` token in-process via the Pod
Identity credentials. Do **not** rely on K8s `$(VAR)` env interpolation to
assemble the URL — K8s only expands a `$(VAR)` defined earlier in the env list, so
the legacy `$(POSTGRES_PASSWORD)` splice was left literal (an auth failure). The
static-password interim (Shape B, `existingSecret` set) now emits
`POSTGRES_PASSWORD` **before** `DATABASE_URL` so the expansion fires.

### External Secrets Operator (ClusterSecretStore prerequisite)

The chart references the `ClusterSecretStore` by name (`aws-secrets-manager`) but
**does not create it**, and neither does the P0/P1 CDK. The ESO controller + a
`ClusterSecretStore` of that exact name (with the ESO controller's **own** Pod
Identity association — distinct from RouteIQ's `PodRole`) are a platform
prerequisite. The name is load-bearing and string-matched. On the IAM-auth path,
sync only `LITELLM_MASTER_KEY` / `ADMIN_API_KEYS` / non-Bedrock provider keys
(`routeiq/<name>` convention); `POSTGRES_PASSWORD` / `REDIS_PASSWORD` /
`DATABASE_URL` are retired.

### Pod-role grants (stack-composition note)

The `rds-db:connect` (Aurora dbuser `…/routeiq`) and `elasticache:Connect` (cache
ARN + IAM-user ARN) grants the gateway pod needs are applied by the
`RouteIqStateStack` as a **separate `iam.Policy` attached to the P0 `PodRole`**
(NOT a mutation of the P0 role's default policy — that would close a cross-stack
dependency cycle, since the state stack already imports the P0 VPC). This happens
automatically when the state stack is given `foundation=<RouteIqStack>` in the same
CDK app. When the two stacks are deployed from **separate** apps/pipelines (no
`foundation`), the grants are skipped and must be attached to the pod role
out-of-band (both are ARN-scoped, no wildcard).

## RouteIQ-on-AWS observability plane (P2: AMP + AppConfig + AWS_REGION)

The P2 observability plane (Amazon Managed Prometheus — ADR-0027; AWS AppConfig
runtime config retrieval — ADR-0026) is provisioned by the **separate**
`RouteIqObservabilityStack` CDK stack. The chart consumes that stack's
`CfnOutput`s as `helm upgrade --set` flags (the same "CfnOutput string → `--set`
value" seam P1 uses), and emits `AWS_REGION` (RouteIQ-bf9f).

### AWS_REGION (load-bearing on the AWS substrate)

The chart emits **no** region by default. On the AWS substrate **every** boto3
client (Bedrock invoke, AppConfig poll, AMP remote-write, S3 config sync, the
`rds-db:connect` / `elasticache:Connect` token minting) needs a region or it
fails. Set `aws.region` and the chart emits **both** `AWS_REGION` and
`AWS_DEFAULT_REGION` (boto3 reads the former first, the latter as a fallback):

```bash
helm upgrade routeiq deploy/charts/routeiq-gateway --set aws.region=us-west-2
```

Default empty keeps the render byte-stable / cloud-agnostic.

### Wire the AppConfig + AMP seams from the P2 CfnOutputs

```bash
STACK=RouteIqObservabilityStack-<env>
q() { aws cloudformation describe-stacks --stack-name "$STACK" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }

helm upgrade routeiq deploy/charts/routeiq-gateway \
  --set aws.region=us-west-2 \
  --set aws.appConfig.enabled=true \
  --set aws.appConfig.application="$(q AppConfigApplicationId)" \
  --set aws.appConfig.environment=<env> \
  --set aws.appConfig.profile=<profile> \
  --set aws.amp.remoteWriteUrl="$(q AmpRemoteWriteUrl)"
```

P2-stack outputs: `AppConfigApplicationId` (→ `aws.appConfig.application`),
`AppConfigProfileArn` / `AppConfigArn` (the polled configuration ARN),
`AmpWorkspaceId`, `AmpRemoteWriteUrl` (→ `aws.amp.remoteWriteUrl`),
`AlarmTopicArn`.

`aws.appConfig.enabled=true` wires the gateway's `config_sync` AppConfig poll
adapter (ADR-0026 / RouteIQ-4333) via its `ROUTEIQ_CONFIG_SYNC__APPCONFIG_*`
settings; the pod-role grant (`appconfigdata:StartConfigurationSession` +
`GetLatestConfiguration` scoped to the profile ARN) is **RouteIQ-569f**, applied
CDK-side. `aws.amp.remoteWriteUrl` is emitted as `AMP_REMOTE_WRITE_URL` for an
**ADOT collector sidecar** to read; the sidecar itself is a documented follow-up
(`sidecars: []` is the seam) — this ships the env-var consume seam, not the
collector. The pod-role `aps:RemoteWrite` grant is **RouteIQ-74c0/717b**, applied
CDK-side.

Every `aws.*` value defaults empty, so the env vars are emitted only when set and
a non-AWS / no-observability deploy renders byte-identically.

### Fluent Bit routing-JSON promotion (RouteIQ-27b6 / RouteIQ-547d)

The P2 CloudWatch metric filters and the Glue/Athena data lake scan **top-level**
JSON keys (`$.event`, `$.latency_ms`, `$.["gen_ai.response.model"]`, `$.level`).
But the EKS Container Insights add-on's Fluent Bit **wraps** every container
stdout line as a stringified `log` field, so those top-level keys resolve to
`null` on a real cluster — the metric filters and the lake stay **inert** until a
Fluent Bit JSON-parse-and-**merge** lifts the inner router JSON to the record top
level.

Enable the shipped promotion config:

```bash
helm template routeiq deploy/charts/routeiq-gateway \
  --set fluentBit.routingPromotion.enabled=true \
  --set fluentBit.routingPromotion.clusterName=routeiq-dev \
  --set aws.region=us-west-2 \
  --show-only templates/fluent-bit-config.yaml
```

This renders a `ConfigMap` carrying `parsers.conf` (a JSON `[PARSER]`) and
`routeiq-routing.conf` (a full pipeline: a routing `[INPUT] tail`, a
`kubernetes` filter with `Merge_Log Off` + `Keep_Log On` so it leaves `log`
intact, the load-bearing `[FILTER] parser` (`Format json`, `Reserve_Data On`)
that lifts the wrapped `log` JSON to the record top level, and a
`cloudwatch_logs [OUTPUT]` targeting the dedicated
`/aws/containerinsights/<cluster>/routeiq-routing` group).

> **Why the parser filter and not `Merge_Log`.** The kubernetes filter's own
> `Merge_Log On` would nest the parsed router JSON under a sub-key (`Merge_Log_Key`)
> rather than the record root the metric filters scan, and `Keep_Log Off` would
> delete `log` before the parser filter could read it — a silent no-op. So the
> top-level promotion is owned solely by the `[FILTER] parser`; the kubernetes
> filter keeps `Merge_Log Off` + `Keep_Log On`. The static promotion contract
> (parser is `Format json`; kubernetes does not consume `log`; the chain promotes
> the inner keys) is pinned by `tests/unit/test_fluent_bit_routing_promotion.py`.

> **Deploy contract — read before relying on this.** The
> `amazon-cloudwatch-observability` add-on's `extraFiles["application-log.conf"]`
> **REPLACES** (does **not** append to) the default application-log pipeline.
> Shipping only a `[FILTER]` would wipe the default `[INPUT]`/`[OUTPUT]` and break
> **all** log delivery. So the rendered content carries the **full** pipeline, and
> the operator must merge the `parsers.conf` + `routeiq-routing.conf` keys into the
> add-on's `configurationValues` (`containerLogs.fluentBit.config.customParsers` +
> `.extraFiles["application-log.conf"]`) at deploy time — this ConfigMap is **not**
> auto-consumed by the add-on on its own.
>
> **Field population is live-validation-only.** `helm template` / `helm lint`
> assert the config *shape* (the `[INPUT]`, the JSON parser, the `Merge_Log`, the
> routing log group); they **cannot** prove the top-level keys actually populate.
> Verify on a live cluster by sampling the routing log group after traffic flows
> (skill `eks-container-insights-fluentbit-wraps-stdout`).

Default `fluentBit.routingPromotion.enabled=false` keeps the chart byte-stable.

## Public edge + OIDC SSO on AWS (P4)

The public edge (ALB + ACM cert) and OIDC/SSO ship **inert and operator-flipped** —
there is **no ALB/ACM construct in the CDK** (`deploy/cdk/` has zero
`elbv2`/`acm`/`route53`; verified). The L7 ALB is **AWS-managed by EKS Auto Mode**,
rendered from the chart `Ingress` when an operator turns it on. The app-layer
OIDC/SSO (`oidc.py`) is **pure env-var config** with no CDK footprint — OIDC/SSO
terminates **in the gateway app**, not at the ALB.

> **Note on naming:** the EKS "OIDC provider" (IRSA workload identity) is a
> different concept and is **not used here** — this substrate uses EKS Pod Identity.
> The application OIDC/SSO below is unrelated to it.

### Operator deploy-time steps

1. **Provision the cert + register the IdP callback (out-of-band).** Issue an ACM
   certificate for `routeiq.<domain>` and register
   `https://routeiq.<domain>/sso/callback` as a redirect URI with your IdP
   (Keycloak / Auth0 / Okta / Azure AD). The cert is **not** chart- or CDK-managed;
   you pass its ARN as an Ingress annotation.

2. **Turn on the ALB Ingress** (EKS Auto Mode managed ALB). The chart `Ingress`
   template is provider-agnostic — supply the ALB/ACM annotations at deploy time
   (account-agnostic placeholders below; substitute your own values):

   ```bash
   helm upgrade routeiq deploy/charts/routeiq-gateway \
     --set ingress.enabled=true \
     --set ingress.className=alb \
     --set 'ingress.annotations.alb\.ingress\.kubernetes\.io/scheme=internet-facing' \
     --set 'ingress.annotations.alb\.ingress\.kubernetes\.io/target-type=ip' \
     --set 'ingress.annotations.alb\.ingress\.kubernetes\.io/listen-ports=[{"HTTPS":443}]' \
     --set 'ingress.annotations.alb\.ingress\.kubernetes\.io/certificate-arn=arn:aws:acm:<region>:<acct>:certificate/<id>' \
     --set ingress.hosts[0].host=routeiq.<domain>
   ```

   (The annotation path works under the **self-managed AWS Load Balancer
   Controller**. Under **EKS Auto Mode** the AWS-level config inverts off
   annotations into an `IngressClassParams` CR — `certificate-arn` becomes a
   `spec.certificateARNs` list and `targetType` is explicit; that runbook is
   tracked as **seed D5**. ALB-level OIDC `auth-type: oidc` is **unsupported on
   Auto Mode** — which is fine, since OIDC terminates in the gateway app.)

3. **Sync the OIDC client-secret via ESO and align the OIDC block.** Add the
   `oidc-client-secret` ExternalSecret entry and point the `oidc` block at the ESO
   target secret so the `secretKeyRef` resolves:

   ```bash
   helm upgrade routeiq deploy/charts/routeiq-gateway \
     --set externalSecrets.enabled=true \
     --set externalSecrets.data[0].secretKey=oidc-client-secret \
     --set externalSecrets.data[0].remoteRef.key=routeiq/oidc-client-secret \
     --set oidc.enabled=true \
     --set oidc.issuerUrl=https://your-tenant.auth0.com/ \
     --set oidc.clientId=<client-id> \
     --set oidc.existingSecret=routeiq-routeiq-gateway-secrets \
     --set oidc.existingSecretKey=oidc-client-secret
   ```

   The `externalSecrets.data[].secretKey` and `oidc.existingSecretKey` names are
   **load-bearing** and must be identical (`oidc-client-secret`). If they drift the
   chart still renders, but `ROUTEIQ_OIDC_CLIENT_SECRET` is silently absent and the
   app falls back to OIDC public-client mode. Likewise `oidc.existingSecret` must
   equal the ESO `target.name` (defaults to `<fullname>-secrets`,
   i.e. `<release>-routeiq-gateway-secrets`).

4. **Prerequisite: the `aws-secrets-manager` ClusterSecretStore + ESO's own Pod
   Identity.** As with all ESO usage on this chart, the `ClusterSecretStore` named
   **exactly** `aws-secrets-manager` is **referenced, not created** — it (and the
   ESO controller) must pre-exist out-of-band, wired to Secrets Manager via the
   **ESO controller's own** `CfnPodIdentityAssociation` (a *second*, distinct
   principal from RouteIQ's `PodRole`). The name is string-matched; a drift is a
   silent sync failure. The `routeiq/*` Secrets Manager id convention matches the
   `PodRole` `SecretsRead` wildcard — but that wildcard is the **pod's own** direct
   read, **not** ESO's sync grant.

### Multi-pod caveat (seed D4)

OIDC exchanged keys (`sk-oidc-*` minted by `/auth/token-exchange`) and the identity
cache are **in-process only** today (bounded TTL dicts in `oidc.py`). With
`replicaCount: 2` a key minted on pod A is unknown to pod B, so token-exchange +
the `sk-oidc-` resolution path are effectively sticky-session-dependent. Moving
them to a Redis-backed shared store is deferred as **seed D4**.

## Security Defaults

This chart implements security best practices by default:

### Pod Security

| Setting | Value | Description |
|---------|-------|-------------|
| `runAsNonRoot` | `true` | Prevents container from running as root |
| `runAsUser` | `1000` | Runs as non-privileged user |
| `readOnlyRootFilesystem` | `true` | Immutable container filesystem |
| `allowPrivilegeEscalation` | `false` | Prevents privilege escalation |
| `capabilities.drop` | `["ALL"]` | Drops all Linux capabilities |
| `seccompProfile.type` | `RuntimeDefault` | Uses default seccomp profile |

> **readOnlyRootFilesystem cache redirect (P4 C6).** The container runs as
> `HOME=/app` under a read-only root, so any library writing to `$HOME/.cache`
> (HuggingFace transformers/tokenizers, matplotlib) would `EROFS` because
> `/app/.cache` is **not** a mounted volume. The chart redirects `HF_HOME`,
> `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, and `MPLCONFIGDIR`
> into the writable `/app/data` emptyDir (backed by `tmpVolume.enabled: true`,
> which also mounts `/tmp` and `/app/models`). If you set any RouteIQ state path
> (`ROUTEIQ_GOVERNANCE_STATE_PATH`, `ROUTEIQ_USAGE_POLICIES_STATE_PATH`,
> `ROUTEIQ_GUARDRAIL_POLICIES_STATE_PATH`, `ROUTEIQ_PROMPTS_STATE_PATH`) via
> `extraEnv`, it **must** live under `/app/data`. Optional pre-release check:
> `docker run --read-only --tmpfs /tmp -v rw-data:/app/data -v rw-models:/app/models -u 1000:1000 -e HOME=/app <image>`
> and exercise a transformers/mmBERT strategy to confirm no `EROFS`.

### Service Account

| Setting | Value | Description |
|---------|-------|-------------|
| `serviceAccount.create` | `true` | Creates dedicated service account |
| `serviceAccount.automountServiceAccountToken` | `false` | Disables K8s API token mount |

To enable Kubernetes API access (e.g., for IRSA/Workload Identity), set:
```yaml
serviceAccount:
  automountServiceAccountToken: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/routeiq-gateway
```

### Network Policy

NetworkPolicy is **opt-in** (`enabled: false` by default).

When enabled, the default configuration:
- **Ingress**: Allows traffic from all sources (configure `ingress.fromNamespaceSelector`/`fromPodSelector` to restrict)
- **Egress**: Allows DNS + HTTPS/HTTP to external IPs only (blocks private IP ranges)

For strict production lockdown:
```yaml
networkPolicy:
  enabled: true
  ingress:
    fromNamespaceSelector:
      kubernetes.io/metadata.name: my-app-namespace
    fromPodSelector:
      app.kubernetes.io/name: my-app
  egress:
    allowDns: true
    allowHttpsExternal: true
    to:
      - namespaceSelector:
          matchLabels:
            name: database
        podSelector:
          matchLabels:
            app: postgres
        ports:
          - port: 5432
```

#### AWS substrate: enable `egress.inVpc` or it's an outage (P4 C5)

> **Outage landmine.** The `allowHttpsExternal` rule carves OUT every RFC-1918
> range (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`). On the RouteIQ-on-AWS
> substrate the three things the pod MUST reach are **all inside RFC-1918**
> (the CDK VPC CIDR is `10.40.0.0/16`):
>
> | Target | Port |
> |--------|------|
> | Aurora PostgreSQL Serverless v2 | 5432 |
> | ElastiCache Serverless Valkey (TLS) | 6379 |
> | Bedrock runtime + Secrets Manager + ECR + CloudWatch Logs **interface VPC endpoints** | 443 |
>
> Bedrock is the non-obvious killer: `private_dns_enabled` resolves
> `bedrock-runtime.<region>.amazonaws.com` to an in-VPC ENI, so the rule
> labelled "HTTPS to external LLM providers" **blocks the primary LLM
> provider**. Flipping `networkPolicy.enabled=true` as-shipped on AWS fails boot
> (DB migration / asyncpg connect) and every Bedrock call.

Whenever you enable `networkPolicy.enabled` on AWS, also enable `egress.inVpc`:

```yaml
networkPolicy:
  enabled: true
  egress:
    allowDns: true
    allowHttpsExternal: true   # keep TRUE: covers S3 (gateway-endpoint, non-RFC-1918) + public LLM providers
    inVpc:
      enabled: true            # re-opens the in-VPC targets the carve-out blocks
      cidr: "10.40.0.0/16"     # MUST equal the CDK vpc_cidr (network_construct.py:53)
      ports: [5432, 6379, 443] # Aurora, ElastiCache, interface VPC endpoints
```

The `inVpc.cidr` is coupled to the CDK `vpc_cidr`; if an operator overrides the
VPC CIDR, this value must move with it. `inVpc.enabled` defaults `false` so the
chart stays account-agnostic and the default render is byte-stable.

### SSRF Protection

The gateway has built-in SSRF protection enabled by default:
```yaml
gateway:
  ssrf:
    allowPrivateIps: false  # Blocks requests to private IPs
    allowlistHosts: ""      # Comma-separated allowed hosts
    allowlistCidrs: ""      # Comma-separated allowed CIDRs
```

## Upgrading

```bash
helm upgrade routeiq-gateway ./deploy/charts/routeiq-gateway -f myvalues.yaml
```

## Uninstalling

```bash
helm uninstall routeiq-gateway
```

## Template Validation

Validate the chart templates without installing:

```bash
# Lint the chart
helm lint ./deploy/charts/routeiq-gateway

# Render templates locally
helm template routeiq-gateway ./deploy/charts/routeiq-gateway --debug
```

## Troubleshooting

### Pod not starting

1. Check secrets are properly configured:
   ```bash
   kubectl get secret -n <namespace>
   kubectl describe pod <pod-name> -n <namespace>
   ```

2. Verify config is valid:
   ```bash
   kubectl logs <pod-name> -n <namespace>
   ```

### Health checks failing

1. Check the endpoints directly:
   ```bash
   kubectl port-forward svc/routeiq-gateway 4000:80
   curl http://localhost:4000/_health/live
   curl http://localhost:4000/_health/ready
   ```

### NetworkPolicy blocking traffic

1. Verify NetworkPolicy is applied:
   ```bash
   kubectl get networkpolicy -n <namespace>
   kubectl describe networkpolicy <name> -n <namespace>
   ```

2. Check pod labels match selectors:
   ```bash
   kubectl get pods -n <namespace> --show-labels
   ```

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.
