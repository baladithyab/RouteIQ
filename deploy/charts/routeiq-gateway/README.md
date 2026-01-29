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

## Configuration

See [`values.yaml`](values.yaml) for the full list of configurable parameters.

### Key Configuration Sections

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image repository | `ghcr.io/baladithyab/litellm-llm-router` |
| `image.tag` | Container image tag (defaults to Chart appVersion) | `""` |
| `image.digest` | Container image digest (takes precedence over tag) | `""` |
| `replicaCount` | Number of replicas | `2` |
| `resources.requests` | Resource requests | `512Mi/500m` |
| `resources.limits` | Resource limits | `2Gi/2000m` |

### Health Probes

The chart configures health probes using the gateway's internal health endpoints:

| Probe | Endpoint | Purpose |
|-------|----------|---------|
| Liveness | `/_health/live` | Basic health check (no external deps) |
| Readiness | `/_health/ready` | Full health check (includes DB/Redis) |

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
helm template routeiq-gateway ./deploy/charts/routeiq-gateway --debug
```

Lint the chart:

```bash
helm lint ./deploy/charts/routeiq-gateway
```

## Security Considerations

- Always use `secrets.existingSecret` or `externalSecrets` in production
- Enable `networkPolicy` to restrict traffic
- The chart sets `readOnlyRootFilesystem: true` by default
- Runs as non-root user (UID 1000)
- SSRF protection is enabled by default (`gateway.ssrf.allowPrivateIps: false`)

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

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.
