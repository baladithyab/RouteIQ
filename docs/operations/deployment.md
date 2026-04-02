# Deployment

RouteIQ Gateway is cloud-native and deployment-agnostic.

## Docker

```bash
docker run -p 4000:4000 \
  -e LITELLM_MASTER_KEY=sk-your-key \
  -e OPENAI_API_KEY=sk-... \
  routeiq:latest
```

## Docker Compose

| Scenario | Path | Description |
|----------|------|-------------|
| Basic | `examples/docker/basic/` | Minimal single-container |
| HA | `examples/docker/ha/` | Multi-replica with Redis + Postgres + Nginx |
| Observability | `examples/docker/observability/` | OTel Collector + Jaeger |
| Full Stack | `examples/docker/full-stack/` | HA + Observability |

## Kubernetes

Helm charts in `deploy/charts/`:

```bash
helm install routeiq deploy/charts/routeiq \
  --namespace routeiq --create-namespace
```

## AWS Production

See the [AWS Production Guide](../operations/docker.md) for ECS, ALB, and CloudWatch deployment.

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

## Multi-Worker

```bash
ROUTEIQ_WORKERS=4  # Number of uvicorn workers
```

Multi-worker mode uses `os.fork()` to preserve in-process state.
