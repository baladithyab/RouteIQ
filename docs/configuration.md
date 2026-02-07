# Configuration Guide

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

This guide covers all configuration options for the RouteIQ Gateway.

## Configuration File Location

The gateway looks for configuration in these locations (in order):

1. Path specified via `--config` CLI argument
2. `LITELLM_CONFIG_PATH` environment variable
3. `/app/config/config.yaml` (default in container)

## Configuration Sections

### Model List

Define your LLM providers and their endpoints:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY
```

### Issue

### Router Settings

Configure routing behavior:

```yaml
router_settings:
  # Choose routing strategy
  routing_strategy: llmrouter-knn

  # LLMRouter-specific settings
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
    llm_data_path: /app/config/llm_candidates.json
    hot_reload: true
    reload_interval: 300

  # Retry settings
  num_retries: 2
  timeout: 600
```

### Available Routing Strategies

| Strategy | Description |
|----------|-------------|
| **LiteLLM Built-in** | |
| `simple-shuffle` | Random selection (LiteLLM default) |
| `least-busy` | Route to least busy deployment |
| `latency-based-routing` | Optimize for latency |
| `cost-based-routing` | Optimize for cost |
| `usage-based-routing` | Route based on token usage |
| **LLMRouter Single-Round** | |
| `llmrouter-knn` | K-Nearest Neighbors (embedding similarity) |
| `llmrouter-svm` | Support Vector Machine |
| `llmrouter-mlp` | Multi-Layer Perceptron neural network |
| `llmrouter-mf` | Matrix Factorization |
| `llmrouter-elo` | Elo Rating based |
| `llmrouter-routerdc` | Dual Contrastive learning (BERT-based) |
| `llmrouter-hybrid` | Probabilistic hybrid |
| `llmrouter-causallm` | Transformer-based (GPT-2) |
| `llmrouter-graph` | Graph neural network |
| `llmrouter-automix` | Automatic model mixing |
| **LLMRouter Multi-Round** | |
| `llmrouter-r1` | Pre-trained multi-turn router (requires vLLM) |
| `llmrouter-knn-multiround` | KNN agentic router |
| `llmrouter-llm-multiround` | LLM agentic router |
| **LLMRouter Personalized** | |
| `llmrouter-gmt` | Graph-based personalized router |
| **LLMRouter Baseline** | |
| `llmrouter-smallest` | Always picks smallest model |
| `llmrouter-largest` | Always picks largest model |
| `llmrouter-custom` | User-defined custom router |

### General Settings

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
```

### LiteLLM Settings

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: 6379
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LITELLM_MASTER_KEY` | Admin API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | If using OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | If using Anthropic |
| `DATABASE_URL` | PostgreSQL connection string | For HA |
| `REDIS_HOST` | Redis host for caching | For caching |
| `CONFIG_S3_BUCKET` | S3 bucket for config | For S3 config |
| `CONFIG_S3_KEY` | S3 key for config file | For S3 config |

## Loading Config from S3

Set these environment variables to load config from S3 on startup:

```bash
CONFIG_S3_BUCKET=my-config-bucket
CONFIG_S3_KEY=configs/litellm-config.yaml
```

## LLM Candidates JSON

The `llm_candidates.json` file describes available models for LLMRouter:

```json
{
  "gpt-4": {
    "provider": "openai",
    "capabilities": ["reasoning", "coding"],
    "cost_per_1k_tokens": {"input": 0.03, "output": 0.06},
    "quality_score": 0.95
  }
}
```

## Hot Reloading

This guide explains how to update routing models without restarting the gateway.

### Enabling Hot Reload

Enable in your configuration:

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
    hot_reload: true
    reload_interval: 300  # seconds
```

### How It Works

1. **File Monitoring**: The gateway checks the model file's modification time
2. **Reload Trigger**: If the file changed since last check, reload is triggered
3. **Thread-Safe Loading**: New model is loaded while old model handles requests
4. **Atomic Swap**: Once loaded, requests switch to the new model

### Updating Models

#### Local Volume Mount

If using volume mounts, simply replace the model file:

```bash
cp new_model.pt ./models/knn_router.pt
```

The gateway will detect the change within `reload_interval` seconds.

#### S3-Based Updates

For S3-stored models:

1. Upload new model to S3:
   ```bash
   aws s3 cp new_model.pt s3://my-bucket/models/knn_router.pt
   ```

2. The gateway downloads and loads on next check

#### API-Triggered Reload

Force immediate reload via API:

```bash
curl -X POST http://localhost:4000/router/reload \
  -H "Authorization: Bearer sk-master-key" \
  -H "Content-Type: application/json" \
  -d '{"strategy": "llmrouter-knn"}'
```

### Configuration Reload

Reload entire config without restart:

```bash
curl -X POST http://localhost:4000/config/reload \
  -H "Authorization: Bearer sk-master-key"
```

### Best Practices

1. **Test Before Deploy**: Always test new models in staging.
2. **Monitor After Reload**: Watch metrics after model updates (`curl http://localhost:4000/metrics | grep llmrouter`).
3. **Keep Rollback Ready**: Maintain previous model version.
4. **Use Version Tags**: Tag model versions in S3.

### Troubleshooting

- **Model Not Reloading**: Check file permissions and `hot_reload: true`.
- **Reload Errors**: Check logs for format compatibility or missing dependencies.

## Configuring Anthropic Skills

To use Anthropic Skills (Computer Use, etc.), configure a model with your Anthropic API key.

```yaml
model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```

**Moat Mode Note:** For production, you can back the `litellm_proxy` skills state with a database (PostgreSQL) instead of memory. This is configured via the standard LiteLLM database settings.

## Local Testing Stack

For local development and testing, use the local test docker-compose:

```bash
# Start all services
docker compose -f docker-compose.local-test.yml up -d

# Access services:
# - RouteIQ Gateway: http://localhost:4010
# - Jaeger UI: http://localhost:16686
# - MLflow UI: http://localhost:5050
# - MinIO Console: http://localhost:9001

# Run integration tests
pytest tests/integration/test_local_stack.py -v

# Stop the stack
docker compose -f docker-compose.local-test.yml down
```

The local stack includes:
- **RouteIQ Gateway** with all features enabled (A2A, MCP, hot reload)
- **PostgreSQL** for persistence
- **Redis** for caching
- **Jaeger** for distributed tracing
- **MLflow** for experiment tracking
- **MinIO** for S3-compatible storage
- **MCP Proxy** for MCP server access

## Kubernetes Deployment Notes

This section covers configuration considerations for deploying the gateway in Kubernetes.

### Environment Variables Reference

The following tables list all environment variables relevant for Kubernetes deployments:

#### Core Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `LITELLM_MASTER_KEY` | Admin API key | Yes | - |
| `LITELLM_CONFIG_PATH` | Path to config file | No | `/app/config/config.yaml` |
| `DATABASE_URL` | PostgreSQL connection string | For HA | - |
| `STORE_MODEL_IN_DB` | Store models in database | Recommended for K8s | `false` |

#### Redis Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `REDIS_HOST` | Redis hostname | For caching | - |
| `REDIS_PORT` | Redis port | No | `6379` |
| `REDIS_PASSWORD` | Redis password | If auth required | - |

#### Object Storage Config Sync

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `CONFIG_S3_BUCKET` | S3 bucket for config | For S3 sync | - |
| `CONFIG_S3_KEY` | S3 key for config file | For S3 sync | - |
| `CONFIG_GCS_BUCKET` | GCS bucket for config | For GCS sync | - |
| `CONFIG_GCS_KEY` | GCS key for config file | For GCS sync | - |
| `CONFIG_HOT_RELOAD` | Enable hot reload | No | `false` |
| `CONFIG_SYNC_ENABLED` | Enable config sync | No | `true` |
| `CONFIG_SYNC_INTERVAL` | Sync interval in seconds | No | `60` |

#### Feature Flags

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `MCP_GATEWAY_ENABLED` | Enable MCP gateway | No | `false` |
| `A2A_GATEWAY_ENABLED` | Enable A2A gateway | No | `false` |
| `MCP_HA_SYNC_ENABLED` | MCP registry sync via Redis | For MCP HA | `false` |

#### OpenTelemetry

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTEL Collector endpoint | For observability | - |
| `OTEL_SERVICE_NAME` | Service name for traces | No | `litellm-gateway` |
| `OTEL_TRACES_EXPORTER` | Traces exporter type | No | `none` |
| `OTEL_METRICS_EXPORTER` | Metrics exporter type | No | `none` |
| `OTEL_LOGS_EXPORTER` | Logs exporter type | No | `none` |
| `OTEL_ENABLED` | Enable OTEL integration | No | `true` |

### Health Probes

The gateway exposes both LiteLLM's native health endpoints and internal endpoints optimized for Kubernetes:

#### Internal Endpoints (Recommended for K8s)

```yaml
livenessProbe:
  httpGet:
    path: /_health/live
    port: 4000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /_health/ready
    port: 4000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 3
```

| Endpoint | Auth Required | Checks | Use Case |
|----------|---------------|--------|----------|
| `/_health/live` | No | Process alive | Liveness probe |
| `/_health/ready` | No | DB, Redis (if configured) | Readiness probe |
| `/health/liveliness` | Depends on config | LiteLLM internal | Alternative liveness |
| `/health/readiness` | Depends on config | LiteLLM internal | Alternative readiness |

**Why use `/_health/*` endpoints?**
- Always unauthenticated (no API key required)
- Fast response times with short timeouts (2s)
- Check only configured dependencies
- Non-fatal for optional dependencies (MCP)

### Database Migration Pattern

**⚠️ Important:** Do NOT run database migrations on every replica. This can cause race conditions and data loss in multi-replica deployments.

**Recommended Pattern: Init Container or Job**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: litellm-db-migrate
spec:
  template:
    spec:
      containers:
      - name: migrate
        image: ghcr.io/baladithyab/litellm-llm-router:latest
        command: ["/bin/bash", "-c"]
        args:
          - |
            SCHEMA_PATH=$(python -c "import litellm; import os; print(os.path.join(os.path.dirname(litellm.__file__), 'proxy', 'schema.prisma'))")
            prisma migrate deploy --schema="$SCHEMA_PATH"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: litellm-secrets
              key: database-url
      restartPolicy: Never
  backoffLimit: 3
```

**Alternative: Set migration flag on single replica**

```yaml
# Set on ONE replica only (e.g., via a separate Deployment)
- name: LITELLM_RUN_DB_MIGRATIONS
  value: "true"
```

### Network Policy Considerations

The gateway requires egress to several external services. Here's a template NetworkPolicy:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: litellm-egress
spec:
  podSelector:
    matchLabels:
      app: litellm-gateway
  policyTypes:
  - Egress
  egress:
  # DNS resolution
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - port: 53
      protocol: UDP
  
  # PostgreSQL
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - port: 5432
  
  # Redis
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - port: 6379
  
  # OTEL Collector
  - to:
    - namespaceSelector:
        matchLabels:
          name: observability
    ports:
    - port: 4317  # gRPC
    - port: 4318  # HTTP
  
  # External LLM providers (OpenAI, Anthropic, etc.)
  # Note: Use CIDR ranges or allow all for simplicity
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
    ports:
    - port: 443
```

**MCP/A2A Egress Considerations:**
- If `MCP_GATEWAY_ENABLED=true`, allow egress to MCP server URLs
- If `A2A_GATEWAY_ENABLED=true`, allow egress to registered agent URLs
- URLs are validated against SSRF attacks at runtime

### ReadOnlyRootFilesystem Support

The container supports `readOnlyRootFilesystem: true` with the following writable mounts:

```yaml
securityContext:
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false

volumeMounts:
- name: tmp
  mountPath: /tmp
- name: data
  mountPath: /app/data
- name: models
  mountPath: /app/models
  readOnly: true  # If not hot-reloading

volumes:
- name: tmp
  emptyDir: {}
- name: data
  emptyDir: {}
- name: models
  emptyDir: {}  # Or PVC for persistent models
```

### Resource Recommendations

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

Consider HPA based on:
- CPU/Memory utilization
- Custom metrics (requests per second, queue depth)
- External metrics from OTEL
