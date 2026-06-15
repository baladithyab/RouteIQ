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
