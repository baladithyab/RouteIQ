# Deployment

RouteIQ Gateway is cloud-native and deployment-agnostic. This guide covers
deployment using Docker, Docker Compose, and Kubernetes.

## Docker

```bash
docker run -p 4000:4000 \
  -e LITELLM_MASTER_KEY=sk-your-key \
  -e OPENAI_API_KEY="sk-..." \
  ghcr.io/baladithyab/litellm-llm-router:latest
```

## Docker Compose

| Scenario | Path | Description |
|----------|------|-------------|
| Basic | `examples/docker/basic/` | Minimal single-container |
| HA | `examples/docker/ha/` | Multi-replica with Redis + Postgres + Nginx |
| Observability | `examples/docker/observability/` | OTel Collector + Jaeger |
| Full Stack | `examples/docker/full-stack/` | HA + Observability |
| Local Dev | `examples/docker/local-dev/` | Local development with hot-reload |

```bash
# Basic single-container
cd examples/docker/basic && docker compose up -d

# High Availability (Redis + Postgres + Nginx)
cd examples/docker/ha && docker compose up -d

# Full Stack (HA + Observability)
cd examples/docker/full-stack && docker compose up -d
```

## Reproducible Builds

RouteIQ uses lockfile-driven dependency management:

- **Lockfile-driven installs**: `uv sync --frozen` with `uv.lock`
- **Pinned base images**: Docker base images pinned to SHA256 digests
- **CI verification**: `uv lock --check` on every PR
- **SBOM/Provenance**: Docker builds generate SBOM and provenance attestations

```bash
# Update dependencies
uv lock --upgrade

# Verify lockfile
uv lock --check

# Build production image
docker build -f docker/Dockerfile -t routeiq-gateway .
```

## Worker Configuration

The `ROUTEIQ_WORKERS` environment variable controls uvicorn worker count (default: `1`).

The plugin strategy (default) uses `os.fork()` to preserve in-process state,
enabling multi-worker deployments:

```bash
ROUTEIQ_WORKERS=4  # Production: 4 workers
```

| Environment | Workers | Notes |
|-------------|---------|-------|
| Development | 1 | Either strategy |
| Production (small) | 2-4 | Plugin strategy (default) |
| Production (large) | 4-8+ | Plugin strategy (default) |

## Centroid Routing Deployment

Centroid routing provides zero-config intelligent routing (~2ms latency).
Enabled by default (`ROUTEIQ_CENTROID_ROUTING=true`).

```yaml
environment:
  - ROUTEIQ_CENTROID_ROUTING=true
  - ROUTEIQ_ROUTING_PROFILE=auto   # auto | eco | premium | free | reasoning
  - ROUTEIQ_CENTROID_WARMUP=false  # Set true to pre-warm at startup
```

## Admin UI

```bash
ROUTEIQ_ADMIN_UI_ENABLED=true  # Accessible at /ui/
```

## Kubernetes

Helm charts in `deploy/charts/`:

```bash
helm install routeiq deploy/charts/routeiq \
  --namespace routeiq --create-namespace
```

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
          readinessProbe:
            httpGet:
              path: /_health/ready
              port: 4000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /_health/live
              port: 4000
            initialDelaySeconds: 30
            periodSeconds: 10
```

### HPA

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
```

### Database Migration Job

Run migrations separately to avoid race conditions:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: routeiq-db-migrate
  annotations:
    helm.sh/hook: pre-install,pre-upgrade
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
              name: routeiq-secrets
              key: database-url
      restartPolicy: Never
  backoffLimit: 3
```

## High Availability

HA mode requires Redis and optionally PostgreSQL:

```bash
docker compose -f examples/docker/ha/docker-compose.ha.yml up -d
```

Features:

- Redis-based leader election
- Config sync across replicas (leader only)
- Session affinity for conversations
- Graceful shutdown with request draining

### Leader Election

Database-backed lease lock for coordinated config sync:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLMROUTER_HA_MODE` | `single` | `single` or `leader_election` |
| `LLMROUTER_CONFIG_SYNC_LEASE_SECONDS` | `30` | Leader lease duration |
| `LLMROUTER_CONFIG_SYNC_RENEW_INTERVAL_SECONDS` | `10` | Lease renewal interval |

## Configuration Management

Multiple configuration sources:

1. **Local Files**: Mount `config.yaml` to `/app/config/config.yaml`
2. **Environment Variables**: Override with `LITELLM_*` env vars
3. **S3/GCS**: Load configuration from cloud storage

```bash
CONFIG_SOURCE=s3
S3_BUCKET_NAME=my-config-bucket
```

## AWS Production

See the [AWS Production Guide](../aws-production-guide.md) for ECS, ALB, and
CloudWatch deployment patterns.

For detailed technical steps, see the [AWS Deployment Guide](../deployment/aws.md).
