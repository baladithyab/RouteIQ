# Quick Start

Get RouteIQ Gateway running in 5 minutes.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- An API key for at least one LLM provider (OpenAI, Anthropic, etc.)

## 1. Clone and Configure

```bash
git clone https://github.com/baladithyab/RouteIQ.git
cd routeiq
cp .env.example .env
```

Edit `.env` and set your keys:

```bash
# Generate a secure master key
export LITELLM_MASTER_KEY=$(openssl rand -hex 32)

# Add your LLM provider keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## 2. Start the Gateway

```bash
docker compose up -d
```

This starts RouteIQ on port **4000**.

## 3. Verify

```bash
curl http://localhost:4000/_health/ready
# {"status": "healthy", ...}
```

## 4. Make a Request

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello from RouteIQ!"}]
  }'
```

## 5. Enable Intelligent Routing

RouteIQ ships with **centroid-based zero-config routing** out of the box. To use it,
set a routing profile in your config:

```yaml
# config/config.yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
```

Or use a routing profile without training any models:

```bash
export ROUTEIQ_ROUTING_PROFILE=auto  # auto | eco | premium | free | reasoning
```

## What's Next?

- [Installation](installation.md) — Full installation options (pip, Docker, Helm)
- [Configuration](configuration.md) — Detailed configuration reference
- [Routing Strategies](../features/routing.md) — All 18+ routing algorithms
- [Deployment](../operations/deployment.md) — Production deployment guides
