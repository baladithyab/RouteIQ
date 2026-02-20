# Deployment Guide

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

RouteIQ Gateway is designed to be cloud-native and deployment-agnostic. This guide covers deployment using Docker, Docker Compose, and Kubernetes.

## Reproducible Builds

RouteIQ uses lockfile-driven dependency management for deterministic, reproducible builds. This reduces supply-chain risk and ensures consistent behavior across environments.

### Key Features

1. **Lockfile-Driven Installs**: All dependencies are installed via `uv sync --frozen` using `uv.lock`, ensuring exact version pinning.
2. **Pinned Base Images**: Docker base images are pinned to SHA256 digests for supply-chain integrity.
3. **CI Verification**: The CI pipeline verifies lockfile integrity on every PR using `uv lock --check`.
4. **SBOM/Provenance**: Docker builds automatically generate SBOM (Software Bill of Materials) and provenance attestations.

### Updating Dependencies

To update dependencies while maintaining reproducibility:

```bash
# Update all dependencies to latest compatible versions
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package <package-name>

# Verify lockfile is in sync (CI runs this automatically)
uv lock --check
```

### Verifying Build Reproducibility

```bash
# Verify lockfile is up to date before building
uv lock --check

# Build with frozen dependencies (fails if lockfile is stale)
docker build -f docker/Dockerfile -t routeiq-gateway .

# Inspect image SBOMs (available for pushed images)
docker buildx imagetools inspect ghcr.io/baladithyab/litellm-llm-router:latest --format '{{json .SBOM}}'
```

### Base Image Digests

The Dockerfiles pin base images by digest for reproducibility. To update the base image digests:

```bash
# Pull and get the digest of the new image
docker pull ghcr.io/astral-sh/uv:0.9-python3.12-bookworm
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/astral-sh/uv:0.9-python3.12-bookworm

# Update the digest in docker/Dockerfile ARG BUILDER_DIGEST and RUNTIME_DIGEST
```

## Docker

The easiest way to run RouteIQ Gateway is using the official Docker image.

```bash
docker run -p 4000:4000 \
  -e OPENAI_API_KEY="sk-..." \
  ghcr.io/baladithyab/litellm-llm-router:latest
```

## Docker Compose

We provide several Docker Compose configurations for different deployment scenarios, organized under `examples/docker/`:

| Scenario | Path | Description |
|----------|------|-------------|
| **Basic** | `examples/docker/basic/` | Minimal single-container setup |
| **High Availability** | `examples/docker/ha/` | Multi-replica with Redis + Postgres + Nginx |
| **Observability** | `examples/docker/observability/` | OTel Collector + Jaeger for tracing/metrics |
| **Full Stack** | `examples/docker/full-stack/` | HA + Observability combined |
| **Local Dev** | `examples/docker/local-dev/` | Local development with hot-reload |

Each scenario includes its own `docker-compose.yml`, `.env.example`, and `README.md`.

### Quick Start Examples

```bash
# Basic single-container
cd examples/docker/basic && docker compose up -d

# High Availability (Redis + Postgres + Nginx)
cd examples/docker/ha && docker compose up -d

# Observability (OTel Collector + Jaeger)
cd examples/docker/observability && docker compose up -d

# Full Stack (HA + Observability)
cd examples/docker/full-stack && docker compose up -d

# Local Development (hot-reload)
cd examples/docker/local-dev && docker compose up -d
```

> **Legacy**: The root-level `docker-compose.yml`, `docker-compose.ha.yml`, and `docker-compose.otel.yml` files are still available but may be removed in a future release. Prefer the `examples/docker/` layouts for new deployments.

## Worker Configuration

The `ROUTEIQ_WORKERS` environment variable controls the number of uvicorn workers (default: `1`).

### Plugin Strategy (Default — Multi-Worker Safe)

When `ROUTEIQ_USE_PLUGIN_STRATEGY=true` (the default), RouteIQ registers routing logic via a `CustomRoutingStrategyBase` adapter (`custom_routing_strategy.py`). This approach is compatible with multiple workers because `os.fork()` preserves the installed strategy state.

```bash
# Production: 4 workers with the plugin strategy (default)
ROUTEIQ_USE_PLUGIN_STRATEGY=true  # default, can be omitted
ROUTEIQ_WORKERS=4
```

### Legacy Monkey-Patch Mode (Single Worker Only)

When `ROUTEIQ_USE_PLUGIN_STRATEGY=false`, RouteIQ monkey-patches `litellm.Router` at runtime. This **requires exactly 1 worker** because `os.execvp()` would replace the process and lose the patches.

```bash
# Legacy mode: must use 1 worker
ROUTEIQ_USE_PLUGIN_STRATEGY=false
ROUTEIQ_WORKERS=1  # only valid value in legacy mode
```

### Recommendations

| Environment | Workers | Strategy |
|-------------|---------|----------|
| Development | 1 | Either |
| Production (small) | 2–4 | Plugin (default) |
| Production (large) | 4–8+ | Plugin (default) |

## Admin UI

Set `ROUTEIQ_ADMIN_UI_ENABLED=true` to enable a built-in admin UI accessible at `/ui/`. This provides a web-based interface for monitoring and managing the gateway.

```yaml
environment:
  - ROUTEIQ_ADMIN_UI_ENABLED=true
```

## Centroid Routing Deployment

Centroid routing provides intelligent, zero-config model routing with ~2ms latency. It is enabled by default (`ROUTEIQ_CENTROID_ROUTING=true`) and requires `numpy` (included in default dependencies).

### Routing Profiles

Set `ROUTEIQ_ROUTING_PROFILE` to control default routing behavior:

| Profile | Description |
|---------|-------------|
| `auto` (default) | Balanced cost/quality routing |
| `eco` | Prefer lower-cost models |
| `premium` | Prefer highest-quality models |
| `free` | Prefer free-tier models |
| `reasoning` | Prefer models with reasoning capabilities |

### Custom Centroid Vectors

Pre-computed centroid `.npy` files can optionally be placed in `models/centroids/` for custom centroid vectors. If no custom vectors are present, built-in defaults are used.

```yaml
environment:
  - ROUTEIQ_CENTROID_ROUTING=true       # default
  - ROUTEIQ_ROUTING_PROFILE=auto        # default
  - ROUTEIQ_CENTROID_WARMUP=false       # set true to pre-warm at startup
```

## Kubernetes

RouteIQ Gateway is Kubernetes-ready. Below is a blueprint for a production deployment.

### Deployment Blueprint

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: routeiq-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: routeiq-gateway
  template:
    metadata:
      labels:
        app: routeiq-gateway
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
        - name: gateway
          image: ghcr.io/baladithyab/litellm-llm-router:latest
          ports:
            - containerPort: 4000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: routeiq-secrets
                  key: database-url
            - name: REDIS_HOST
              value: "redis-master"
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
          readinessProbe:
            httpGet:
              path: /_health/ready
              port: 4000
            initialDelaySeconds: 5
            periodSeconds: 10
      volumes:
        - name: config-volume
          configMap:
            name: routeiq-config
```

### Horizontal Pod Autoscaler (HPA)

For production, use HPA to scale based on CPU/Memory or custom metrics.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: routeiq-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: routeiq-gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Migration Job

Run migrations separately, not on every replica, to avoid race conditions.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: routeiq-db-migrate
  annotations:
    helm.sh/hook: pre-install,pre-upgrade
    helm.sh/hook-weight: "-1"
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
            if [ -f "$SCHEMA_PATH" ]; then
              echo "Running migrations from $SCHEMA_PATH"
              prisma migrate deploy --schema="$SCHEMA_PATH"
            else
              echo "Schema not found, skipping migrations"
            fi
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: routeiq-secrets
              key: database-url
      restartPolicy: Never
  backoffLimit: 3
```

## High Availability & Scaling

### Config Sync Leader Election

When running multiple replicas, config sync from S3/GCS can cause issues like "thundering herd" (all replicas downloading at once) or conflicting updates.

RouteIQ includes an optional **leader election** mechanism that ensures only one replica performs config sync at a time.

#### HA Mode Setting

RouteIQ supports two HA modes controlled by the `LLMROUTER_HA_MODE` environment variable:

| Mode | Value | Description |
|------|-------|-------------|
| **Single** | `single` (default) | All instances sync independently. Use for single-instance deployments. |
| **Leader Election** | `leader_election` | Only the elected leader performs config sync. Use for multi-replica HA deployments. |

#### How It Works

1. **Database-backed lease lock**: Uses PostgreSQL to coordinate across replicas
2. **Lease-based with renewal**: Leader holds a lease that expires automatically if not renewed
3. **Crash recovery**: If a leader crashes, the lease expires and another replica takes over
4. **Non-blocking**: Non-leaders skip sync quietly without blocking

#### Configuration

Leader election is **automatically enabled** when `DATABASE_URL` is configured (HA mode).

Set `LLMROUTER_HA_MODE=leader_election` to enable coordinated config sync:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLMROUTER_HA_MODE` | `single` | HA mode: `single` or `leader_election` |
| `LLMROUTER_CONFIG_SYNC_LEASE_SECONDS` | `30` | How long a leader holds the lock |
| `LLMROUTER_CONFIG_SYNC_RENEW_INTERVAL_SECONDS` | `10` | How often to renew the lease |
| `LLMROUTER_CONFIG_SYNC_LOCK_NAME` | `config_sync` | Lock name (for multiple independent locks) |

### Monitoring Leader Election

Check the config sync status endpoint to see leader election status:

```bash
curl http://localhost:4000/llmrouter/config/sync/status
```

Response includes:
```json
{
  "enabled": true,
  "running": true,
  "leader_election": {
    "enabled": true,
    "is_leader": true,
    "holder_id": "gateway-1-abc123",
    "lease_expires_at": "2024-01-15T10:30:00Z",
    "skipped_sync_count": 0
  }
}
```

## Configuration Management

RouteIQ Gateway supports multiple configuration sources:

1.  **Local Files**: Mount `config.yaml` to `/app/config/config.yaml`.
2.  **Environment Variables**: Override settings using `LITELLM_*` env vars.
3.  **S3**: Load configuration and models from an S3 bucket.

```bash
# Enable S3 config loading
export CONFIG_SOURCE="s3"
export S3_BUCKET_NAME="my-config-bucket"
```

## Cloud Deployment

For specific cloud provider guides, see:

- [AWS Deployment Guide](deployment/aws.md)

## Observability Configuration

RouteIQ Gateway supports OpenTelemetry for distributed tracing, metrics, and structured logging.

### Quick Setup

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
  - OTEL_SERVICE_NAME=litellm-gateway
  - OTEL_TRACES_EXPORTER=otlp
  - OTEL_METRICS_EXPORTER=otlp
```

### Production Recommendations

- **Trace Sampling**: In high-traffic environments, enable sampling to control costs:
  ```yaml
  environment:
    - LLMROUTER_OTEL_SAMPLE_RATE=0.1  # Sample 10% of traces
  ```

- **Multiprocess Metrics**: When running with multiple workers (gunicorn/uvicorn), use an OTEL Collector for metric aggregation. See the full guide for configuration details.

For complete observability configuration including sampling options, multiprocess setup, and backend-specific guides, see the [Observability Guide](observability.md).
