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
