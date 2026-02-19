# RouteIQ Gateway - Local Development Setup

Full development environment simulating production AWS services locally. Includes PostgreSQL, Redis, Jaeger, MinIO (S3), and MLflow for a complete local development experience.

## What This Provides

- **Gateway** on port 4010 with all features enabled
- **PostgreSQL** simulating Amazon RDS/Aurora (API keys, spend tracking)
- **Redis** simulating ElastiCache (caching, rate limiting)
- **Jaeger** simulating AWS X-Ray (distributed tracing)
- **MinIO** simulating Amazon S3 (config storage, ML models)
- **MLflow** simulating SageMaker (experiment tracking, model versioning)
- **A2A + MCP gateways** enabled for agent and tool development
- **Admin UI** at `/ui/` for gateway management

## Prerequisites

- Docker and Docker Compose v2+
- An OpenRouter API key ([get one here](https://openrouter.ai/keys))
- At least 6GB RAM available for Docker

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env to set your API key (pre-filled with a test key)

# 2. Start the full dev stack
docker compose up -d

# 3. Or start just the gateway (minimal)
docker compose up litellm-gateway
```

## Verify It's Working

```bash
# Check gateway health
curl http://localhost:4010/_health/live

# List available models
curl http://localhost:4010/v1/models \
  -H "Authorization: Bearer sk-test-master-key-change-me"

# Send a chat completion
curl http://localhost:4010/v1/chat/completions \
  -H "Authorization: Bearer sk-test-master-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# View traces
open http://localhost:16686
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Gateway | http://localhost:4010 | LLM API endpoint |
| Admin UI | http://localhost:4010/ui/ | Gateway management |
| Jaeger UI | http://localhost:16686 | Trace visualization |
| MinIO Console | http://localhost:9001 | S3 object browser (minioadmin/minioadmin) |
| MLflow UI | http://localhost:5050 | Experiment tracking |
| PostgreSQL | localhost:5432 | Database (litellm/testpassword) |
| Redis | localhost:6379 | Cache |
| MinIO API | localhost:9000 | S3-compatible API |

## AWS Service Mapping

| Local Service | AWS Equivalent | Purpose |
|--------------|----------------|---------|
| PostgreSQL | RDS / Aurora PostgreSQL | API key & spend storage |
| Redis | ElastiCache Redis | Response caching, rate limiting |
| Jaeger | AWS X-Ray / CloudWatch | Distributed tracing |
| MinIO | Amazon S3 | Config files, ML models |
| MLflow | SageMaker Experiments | ML model tracking |

## Key Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_MASTER_KEY` | `sk-test-master-key-change-me` | Admin API key |
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key |
| `ADMIN_API_KEY` | `local-dev-admin-key` | Control-plane API key |
| `ROUTEIQ_ADMIN_UI_ENABLED` | `true` | Enable admin UI |

## Development Workflow

1. Start the dev stack: `docker compose up -d`
2. Make code changes in `src/litellm_llmrouter/`
3. Rebuild and restart gateway: `docker compose up -d --build litellm-gateway`
4. Check traces in Jaeger: http://localhost:16686
5. Run tests against the stack: `uv run pytest tests/integration/ -v`

## Next Steps

- **Need simpler setup?** See [../basic/](../basic/)
- **Production deployment?** See [../full-stack/](../full-stack/)
- **MLOps training pipeline?** See [../../mlops/](../../mlops/)

## Related Documentation

- [Configuration Guide](../../../docs/configuration.md)
- [MLOps Training Guide](../../../docs/mlops-training.md)
- [MCP Gateway Guide](../../../docs/mcp-gateway.md)
- [A2A Gateway Guide](../../../docs/a2a-gateway.md)
- [Quickstart Docker Compose](../../../docs/quickstart-docker-compose.md)
