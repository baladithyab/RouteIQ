# RouteIQ Gateway - High Availability Setup

Production-grade HA deployment with PostgreSQL, Redis, two gateway replicas with leader election, and an Nginx load balancer.

## What This Provides

- **Two gateway replicas** with automatic leader election
- **PostgreSQL** for persistent API key and spend tracking storage
- **Redis** for response caching, rate limiting, and distributed state
- **Nginx load balancer** with least-connections routing and streaming support
- **Config hot-reload** with leader-only sync to prevent conflicts

## Architecture

```
                    ┌──────────┐
                    │  Nginx   │ :8080
                    │    LB    │
                    └────┬─────┘
                    ┌────┴─────┐
              ┌─────┴──┐  ┌───┴────┐
              │Gateway 1│  │Gateway 2│
              │  :4000  │  │  :4001  │
              └────┬────┘  └───┬────┘
                   │           │
            ┌──────┴───────────┴──────┐
            │                         │
       ┌────┴─────┐           ┌──────┴───┐
       │PostgreSQL │           │  Redis   │
       │   :5432   │           │          │
       └──────────┘           └──────────┘
```

## Prerequisites

- Docker and Docker Compose v2+
- An OpenRouter API key ([get one here](https://openrouter.ai/keys))
- At least 4GB RAM available for Docker

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env to set your API key and change default passwords

# 2. Start the full HA stack
docker compose up -d

# 3. Verify all services are healthy
docker compose ps
```

## Verify It's Working

```bash
# Check via load balancer
curl http://localhost:8080/health

# Check individual gateways
curl http://localhost:4000/_health/live
curl http://localhost:4001/_health/live

# Send a request via load balancer
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-test-master-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Key Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_MASTER_KEY` | `sk-test-master-key-change-me` | Admin API key |
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key |
| `POSTGRES_PASSWORD` | `litellm_password` | PostgreSQL password |
| `REDIS_PASSWORD` | `changeme` | Redis password |
| `LLMROUTER_HA_MODE` | `leader_election` | HA mode |
| `STORE_MODEL_IN_DB` | `true` | Store models in DB for cross-replica sync |

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Load Balancer | http://localhost:8080 | Primary access point |
| Gateway 1 | http://localhost:4000 | Direct access to replica 1 |
| Gateway 2 | http://localhost:4001 | Direct access to replica 2 |
| PostgreSQL | localhost:5432 | Database (for debugging) |

## Production Notes

- **Change all default passwords** before deploying to production
- **Generate a strong master key**: `openssl rand -hex 32`
- Only the **leader replica** performs config sync (prevents conflicts)
- Nginx uses **least-connections** routing with streaming support

## Next Steps

- **Add observability?** See [../full-stack/](../full-stack/)
- **Need simpler setup?** See [../basic/](../basic/)
- **Development environment?** See [../local-dev/](../local-dev/)

## Related Documentation

- [HA Quickstart Tutorial](../../../docs/tutorials/ha-quickstart.md)
- [Deployment Guide](../../../docs/deployment.md)
- [Configuration Guide](../../../docs/configuration.md)
