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

---

## Comprehensive Environment Variables Reference

This section provides a complete reference of all environment variables supported by RouteIQ Gateway. Variables are grouped by category and include defaults, descriptions, and notes.

> **Tip**: Use `python -m litellm_llmrouter.startup --validate-env` to validate your environment variables before deployment (CI/CD friendly).

### Required

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_MASTER_KEY` | *(none)* | Master API key for admin access. **Required.** Generate with `openssl rand -hex 32`. |

### Admin / Auth

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_API_KEYS` | *(none)* | Comma-separated admin API keys for control-plane endpoints (hot reload, MCP management, A2A registration). |
| `ADMIN_API_KEY` | *(none)* | Legacy single admin key (prefer `ADMIN_API_KEYS`). |
| `ADMIN_AUTH_ENABLED` | `true` | Set to `false` to disable admin auth. **Not recommended for production.** |
| `ROUTEIQ_KEY_PREFIX` | `sk-riq-` | Custom prefix for RouteIQ-generated API keys. |
| `ROUTEIQ_SKIP_ENV_VALIDATION` | `false` | Skip environment variable validation at startup. Useful for CI/testing. |

### LLM Provider Keys

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(none)* | OpenAI API key. |
| `ANTHROPIC_API_KEY` | *(none)* | Anthropic API key. |
| `AZURE_API_KEY` | *(none)* | Azure OpenAI API key. |
| `AZURE_API_BASE` | *(none)* | Azure OpenAI endpoint base URL. |
| `GOOGLE_APPLICATION_CREDENTIALS` | *(none)* | Path to Google Cloud service account JSON file (Vertex AI). |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for Bedrock. |
| `AWS_ACCESS_KEY_ID` | *(none)* | AWS access key (prefer IRSA/Pod Identity in K8s). |
| `AWS_SECRET_ACCESS_KEY` | *(none)* | AWS secret key (prefer IRSA/Pod Identity in K8s). |

### Database (PostgreSQL)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | *(none)* | PostgreSQL connection string. Required for HA. Format: `postgresql://USER:PASS@HOST:PORT/DB`. |
| `POSTGRES_USER` | `litellm` | PostgreSQL username (used by Docker Compose). |
| `POSTGRES_PASSWORD` | *(none)* | PostgreSQL password (used by Docker Compose). |
| `POSTGRES_DB` | `litellm` | PostgreSQL database name (used by Docker Compose). |
| `STORE_MODEL_IN_DB` | `false` | Store LiteLLM models/config in database. Set to `true` for K8s. |

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | *(none)* | Redis hostname. Optional — Redis is not required for basic operation. |
| `REDIS_PORT` | `6379` | Redis port. |
| `REDIS_PASSWORD` | *(none)* | Redis password (if authentication is required). |
| `REDIS_SSL` | `false` | Enable TLS/SSL for Redis connections. |

### Config Sync / Hot Reload

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_CONFIG_PATH` | `/app/config/config.yaml` | Path to the LiteLLM config file. |
| `CONFIG_S3_BUCKET` | *(none)* | S3 bucket for config sync. |
| `CONFIG_S3_KEY` | `configs/config.yaml` | S3 key for config file. |
| `CONFIG_GCS_BUCKET` | *(none)* | GCS bucket for config sync (alternative to S3). |
| `CONFIG_GCS_KEY` | `configs/config.yaml` | GCS key for config file. |
| `CONFIG_HOT_RELOAD` | `false` | Enable filesystem-watching config hot reload. |
| `CONFIG_SYNC_ENABLED` | `true` | Enable background config sync from S3/GCS. |
| `CONFIG_SYNC_INTERVAL` | `60` | Config sync polling interval in seconds. |
| `LLMROUTER_MODEL_S3_BUCKET` | *(none)* | S3 bucket for ML model artifacts. |
| `LLMROUTER_MODEL_S3_KEY` | `models/router.pt` | S3 key for ML model file. |
| `LLMROUTER_HOT_RELOAD` | `true` | Enable hot-reloading of LLMRouter ML models. |
| `LLMROUTER_RELOAD_INTERVAL` | `300` | ML model hot reload polling interval in seconds. |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_GATEWAY_ENABLED` | `false` | Enable MCP gateway (JSON-RPC, SSE, REST surfaces). |
| `MCP_SSE_TRANSPORT_ENABLED` | `false` | Enable MCP SSE transport at `/mcp/sse`. Requires `MCP_GATEWAY_ENABLED`. |
| `MCP_SSE_LEGACY_MODE` | `false` | Enable MCP SSE legacy mode for older clients. |
| `MCP_PROTOCOL_PROXY_ENABLED` | `false` | Enable MCP protocol-level proxy at `/mcp-proxy/*` (admin-only). |
| `MCP_OAUTH_ENABLED` | `false` | Enable MCP OAuth support. |
| `LLMROUTER_ENABLE_MCP_TOOL_INVOCATION` | `false` | Enable MCP remote tool invocation. **Security sensitive** — only enable in trusted environments. |
| `MCP_HA_SYNC_ENABLED` | `false` | MCP registry sync via Redis for multi-replica deployments. |
| `A2A_GATEWAY_ENABLED` | `false` | Enable A2A (Agent-to-Agent) gateway. |
| `POLICY_ENGINE_ENABLED` | `false` | Enable OPA-style policy evaluation middleware. |
| `POLICY_CONFIG_PATH` | *(none)* | Path to policy YAML config file (required when `POLICY_ENGINE_ENABLED=true`). |
| `LLMROUTER_ALLOW_PICKLE_MODELS` | `false` | Allow loading pickle-serialized ML models. **Security risk** — only enable in trusted environments. |
| `LLMROUTER_ENFORCE_SIGNED_MODELS` | `false` | Require manifest verification for ML model artifacts. |
| `LLMROUTER_ROUTER_CALLBACK_ENABLED` | `true` | Enable router decision callback for routing telemetry (TG4.1). |

### Plugin Strategy & Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTEIQ_USE_PLUGIN_STRATEGY` | `true` | Use LiteLLM's official `CustomRoutingStrategyBase` plugin API instead of legacy monkey-patch. |
| `ROUTEIQ_ROUTING_STRATEGY` | *(auto-detected)* | Override routing strategy name (e.g., `llmrouter-knn`, `llmrouter-mlp`). Auto-detected from config if not set. |
| `ROUTEIQ_WORKERS` | `1` | Number of uvicorn workers. Multi-worker requires plugin strategy mode. Legacy monkey-patch forces 1. |

### Centroid Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTEIQ_CENTROID_ROUTING` | `true` | Enable centroid routing fallback (~2ms, zero-config). |
| `ROUTEIQ_ROUTING_PROFILE` | `auto` | Default routing profile. Values: `auto`, `eco`, `premium`, `free`, `reasoning`. |
| `ROUTEIQ_CENTROID_WARMUP` | `false` | Pre-warm centroid classifier at startup (loads embeddings into memory). |
| `ROUTEIQ_CENTROID_DIR` | `models/centroids` | Directory containing centroid `.npy` files. |
| `ROUTEIQ_CONFIDENCE_THRESHOLD` | `0.06` | Centroid classification confidence threshold (cosine distance). |

### Admin UI

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTEIQ_ADMIN_UI_ENABLED` | `false` | Enable Admin UI (serves React SPA at `/ui/`). Requires `ui/dist/` to be built. |

### SSRF Protection

| Variable | Default | Description |
|----------|---------|-------------|
| `LLMROUTER_ALLOW_PRIVATE_IPS` | `false` | Allow all private IP ranges. Only for development/testing. |
| `LLMROUTER_SSRF_ALLOWLIST_HOSTS` | *(none)* | Allowlist specific hosts/domains (comma-separated). Supports exact and suffix match. |
| `LLMROUTER_SSRF_ALLOWLIST_CIDRS` | *(none)* | Allowlist specific IP ranges in CIDR notation (comma-separated). |
| `LLMROUTER_OUTBOUND_ALLOW_PRIVATE` | `false` | Canonical name for `LLMROUTER_ALLOW_PRIVATE_IPS`. |
| `LLMROUTER_OUTBOUND_HOST_ALLOWLIST` | *(none)* | Canonical name for `LLMROUTER_SSRF_ALLOWLIST_HOSTS`. |
| `LLMROUTER_OUTBOUND_CIDR_ALLOWLIST` | *(none)* | Canonical name for `LLMROUTER_SSRF_ALLOWLIST_CIDRS`. |
| `LLMROUTER_OUTBOUND_URL_ALLOWLIST` | *(none)* | Allowlist full URL prefixes that bypass SSRF checks (comma-separated). |
| `LLMROUTER_SSRF_USE_SYNC_DNS` | `false` | Use synchronous DNS resolution (rollback flag). |
| `LLMROUTER_SSRF_DNS_TIMEOUT` | `5.0` | DNS resolution timeout in seconds. |
| `LLMROUTER_SSRF_DNS_CACHE_TTL` | `60` | DNS cache TTL in seconds. |
| `LLMROUTER_SSRF_DNS_CACHE_SIZE` | `1000` | Maximum entries in the DNS resolution cache. |

### Semantic Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_ENABLED` | `false` | Enable the semantic cache plugin. |
| `CACHE_SEMANTIC_ENABLED` | `false` | Enable embedding-based semantic matching. |
| `CACHE_TTL_SECONDS` | `3600` | Default TTL for cached responses in seconds. |
| `CACHE_L1_MAX_SIZE` | `1000` | Maximum entries in the L1 in-memory LRU cache. |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Cosine similarity threshold for semantic cache hits (0.0–1.0). |
| `CACHE_REDIS_URL` | *(none)* | Redis URL for L2 shared cache tier. Format: `redis://[:password@]host:port/db`. |
| `CACHE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for semantic embeddings. |
| `CACHE_MAX_TEMPERATURE` | `0.1` | Maximum temperature for cacheable requests. |

### Resilience / Backpressure

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTEIQ_MAX_CONCURRENT_REQUESTS` | `0` | Maximum concurrent requests before 503 load shedding. `0` = disabled. |
| `ROUTEIQ_DRAIN_TIMEOUT_SECONDS` | `30` | Graceful shutdown drain timeout in seconds. |
| `ROUTEIQ_BACKPRESSURE_EXCLUDED_PATHS` | *(none)* | Additional paths excluded from backpressure (comma-separated). |
| `ROUTEIQ_CB_FAILURE_THRESHOLD` | `5` | Global circuit breaker: failures before circuit opens. |
| `ROUTEIQ_CB_SUCCESS_THRESHOLD` | `2` | Global circuit breaker: successes in half-open before closing. |
| `ROUTEIQ_CB_TIMEOUT_SECONDS` | `30` | Global circuit breaker: seconds before recovery attempt. |
| `ROUTEIQ_CB_WINDOW_SECONDS` | `60` | Global circuit breaker: sliding window for failure tracking. |

### Observability (OTel)

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `true` | Enable OpenTelemetry integration. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(none)* | OTel Collector endpoint (gRPC). E.g., `http://localhost:4317`. |
| `OTEL_SERVICE_NAME` | `litellm-gateway` | Service name for traces/metrics. |
| `OTEL_TRACES_EXPORTER` | `none` | Traces exporter type (`none`, `otlp`, `console`). |
| `OTEL_METRICS_EXPORTER` | `none` | Metrics exporter type (`none`, `otlp`, `console`). |
| `OTEL_LOGS_EXPORTER` | `none` | Logs exporter type (`none`, `otlp`, `console`). |
| `OTEL_TRACES_SAMPLER` | *(none)* | Sampler type: `always_on`, `always_off`, `traceidratio`, `parentbased_traceidratio`, etc. |
| `OTEL_TRACES_SAMPLER_ARG` | *(none)* | Ratio for ratio-based samplers (0.0–1.0). |
| `LLMROUTER_OTEL_SAMPLE_RATE` | *(none)* | Convenience: sampling rate 0.0–1.0 (uses `parentbased_traceidratio`). |
| `PROMETHEUS_MULTIPROC_DIR` | *(none)* | Shared directory for Prometheus multiprocess metrics. |

### Leader Election / HA

| Variable | Default | Description |
|----------|---------|-------------|
| `LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED` | `true` (if `DATABASE_URL` set) | Enable leader election for config sync. |
| `LLMROUTER_CONFIG_SYNC_LEASE_SECONDS` | `30` | Lease duration in seconds. |
| `LLMROUTER_CONFIG_SYNC_RENEW_INTERVAL_SECONDS` | `10` | Lease renewal interval (should be < `lease_seconds / 3`). |
| `LLMROUTER_CONFIG_SYNC_LOCK_NAME` | `config_sync` | Lock name for leader election. |
| `ROUTEIQ_LEADER_MIGRATIONS` | `false` | Enable leader-election-based DB migrations on startup. |

### MLOps (Reference Only)

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow tracking server URI. |
| `MLFLOW_ARTIFACT_BUCKET` | `llmrouter-artifacts` | MLflow artifact storage bucket. |
| `WANDB_API_KEY` | *(none)* | Weights & Biases API key. |
| `HF_TOKEN` | *(none)* | Hugging Face token. |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO root username (local dev). |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO root password (local dev). |
| `JUPYTER_TOKEN` | `llmrouter` | Jupyter notebook token. |
