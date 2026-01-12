# High Availability Setup

This guide covers deploying the LiteLLM + LLMRouter gateway in a high-availability configuration.

## Architecture Overview

```
                    ┌─────────────┐
                    │   Nginx     │
                    │   (LB)      │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │  Gateway 1  │          │  Gateway 2  │
       └──────┬──────┘          └──────┬──────┘
              │                         │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
  │  PostgreSQL │   │    Redis    │   │     S3      │
  │  (State)    │   │   (Cache)   │   │  (Config)   │
  └─────────────┘   └─────────────┘   └─────────────┘
```

## Quick Start

```bash
docker compose -f docker-compose.ha.yml up -d
```

## Components

### PostgreSQL
Persistent storage for:
- User/team configurations
- API keys and budgets
- Request logs

### Redis
Distributed caching for:
- Response caching
- Rate limiting state
- Session data

### S3 (Optional)
Configuration sync:
- Config files
- Routing models
- Custom routers

## Configuration

### Environment Variables

```bash
# PostgreSQL
POSTGRES_USER=litellm
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=litellm

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# S3 Config Sync
CONFIG_S3_BUCKET=my-config-bucket
CONFIG_S3_KEY=configs/config.yaml
LLMROUTER_MODEL_S3_BUCKET=my-models-bucket
LLMROUTER_MODEL_S3_KEY=models/router.pt
```

### Nginx Configuration

Load balancing configuration in `config/nginx.conf`:

```nginx
upstream litellm_backend {
    least_conn;
    server litellm-gateway-1:4000;
    server litellm-gateway-2:4000 backup;
}
```

## Scaling

### Horizontal Scaling

Add more gateway instances in `docker-compose.ha.yml`:

```yaml
litellm-gateway-3:
  image: ghcr.io/baladithyab/litellm-llm-router:latest
  # ... same config as gateway-1
```

Update Nginx upstream:

```nginx
upstream litellm_backend {
    server litellm-gateway-1:4000;
    server litellm-gateway-2:4000;
    server litellm-gateway-3:4000;
}
```

### Kubernetes Deployment

For Kubernetes, use a Deployment with HorizontalPodAutoscaler:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-llmrouter
spec:
  replicas: 3
  # ...
```

## Health Checks

All gateways expose health endpoints:

- `GET /health` - Basic health check
- `GET /health/liveliness` - Kubernetes liveness probe
- `GET /health/readiness` - Kubernetes readiness probe

## Monitoring

Enable Prometheus metrics:

```yaml
litellm_settings:
  success_callback: ["prometheus"]
```

Access metrics at `GET /metrics`.
