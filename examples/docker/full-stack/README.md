# RouteIQ Gateway - Full Stack Setup (HA + Observability)

The complete production-grade deployment combining high availability with distributed tracing, plus A2A and MCP gateway features enabled.

## What This Provides

- **Two gateway replicas** with automatic leader election
- **PostgreSQL** for persistent API key and spend tracking storage
- **Redis** for response caching, rate limiting, and distributed state
- **Nginx load balancer** with least-connections routing and streaming support
- **Jaeger** for distributed tracing with a web UI
- **A2A Gateway** for agent-to-agent communication
- **MCP Gateway** for Model Context Protocol tools
- **Admin UI** at `/ui/` for gateway management

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
              └──┬──┬───┘  └──┬──┬──┘
                 │  │         │  │
      ┌──────────┘  │         │  └──────────┐
      │             └────┬────┘             │
 ┌────┴─────┐     ┌─────┴────┐     ┌──────┴───┐
 │PostgreSQL │     │  Redis   │     │  Jaeger  │
 │   :5432   │     │          │     │  :16686  │
 └──────────┘     └──────────┘     └──────────┘
```

## Prerequisites

- Docker and Docker Compose v2+
- An OpenRouter API key ([get one here](https://openrouter.ai/keys))
- At least 6GB RAM available for Docker

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env to set your API key and change default passwords

# 2. Start the full stack
docker compose up -d

# 3. Verify all services are healthy
docker compose ps
```

## Verify It's Working

```bash
# Check via load balancer
curl http://localhost:8080/health

# Send a request
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-test-master-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# View traces in Jaeger
open http://localhost:16686
# Select "litellm-gateway-1" or "litellm-gateway-2" service
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Load Balancer | http://localhost:8080 | Primary access point |
| Gateway 1 | http://localhost:4000 | Direct access to replica 1 |
| Gateway 2 | http://localhost:4001 | Direct access to replica 2 |
| Jaeger UI | http://localhost:16686 | Trace visualization |
| Admin UI | http://localhost:8080/ui/ | Gateway management (via LB) |
| PostgreSQL | localhost:5432 | Database (for debugging) |
| OTLP gRPC | localhost:4317 | Trace collector endpoint |

## Key Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_MASTER_KEY` | `sk-test-master-key-change-me` | Admin API key |
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key |
| `POSTGRES_PASSWORD` | `litellm_password` | PostgreSQL password |
| `REDIS_PASSWORD` | `changeme` | Redis password |
| `A2A_GATEWAY_ENABLED` | `true` | Agent-to-Agent gateway |
| `MCP_GATEWAY_ENABLED` | `true` | Model Context Protocol gateway |
| `ROUTEIQ_ADMIN_UI_ENABLED` | `true` | Admin UI at /ui/ |

## Production Notes

- **Change all default passwords** before deploying to production
- **Generate a strong master key**: `openssl rand -hex 32`
- Only the **leader replica** performs config sync
- Consider using **AWS ADOT Collector** instead of Jaeger for production

## Next Steps

- **Need simpler setup?** See [../basic/](../basic/)
- **HA without observability?** See [../ha/](../ha/)
- **Development environment?** See [../local-dev/](../local-dev/)

## Related Documentation

- [Deployment Guide](../../../docs/deployment.md)
- [Observability Guide](../../../docs/observability.md)
- [MCP Gateway Guide](../../../docs/mcp-gateway.md)
- [A2A Gateway Guide](../../../docs/a2a-gateway.md)
- [AWS Production Guide](../../../docs/aws-production-guide.md)
