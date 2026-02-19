# RouteIQ Gateway - Basic Setup

The simplest way to run RouteIQ Gateway with a single instance using OpenRouter as the LLM provider.

## What This Provides

- **Single gateway instance** on port 4000
- **OpenRouter integration** with access to GPT-4o-mini, Claude 3 Haiku, and Gemini Flash
- **Centroid-based routing** for intelligent model selection (~2ms classification)
- **OpenAI-compatible API** - use any OpenAI SDK client

## Prerequisites

- Docker and Docker Compose v2+
- An OpenRouter API key ([get one here](https://openrouter.ai/keys))

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env to set your API key (pre-filled with a test key)

# 2. Start the gateway
docker compose up -d

# 3. Verify it's running
curl http://localhost:4000/_health/live
```

## Verify It's Working

```bash
# Check health
curl http://localhost:4000/_health/live

# List available models
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer sk-test-master-key-change-me"

# Send a chat completion request
curl http://localhost:4000/v1/chat/completions \
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
| `ROUTEIQ_ROUTING_PROFILE` | `auto` | Routing profile: auto/eco/premium/free/reasoning |
| `ROUTEIQ_WORKERS` | `2` | Number of uvicorn workers |

## Available Models

| Model Name | Provider | Via OpenRouter |
|-----------|----------|----------------|
| `gpt-4o-mini` | OpenAI | `openrouter/openai/gpt-4o-mini` |
| `claude-3-haiku` | Anthropic | `openrouter/anthropic/claude-3-haiku` |
| `gemini-flash` | Google | `openrouter/google/gemini-2.0-flash-001` |

## Next Steps

- **Need high availability?** See [../ha/](../ha/)
- **Want observability?** See [../observability/](../observability/)
- **Full production stack?** See [../full-stack/](../full-stack/)
- **Development environment?** See [../local-dev/](../local-dev/)

## Related Documentation

- [Configuration Guide](../../../docs/configuration.md)
- [Routing Strategies](../../../docs/routing-strategies.md)
- [Deployment Guide](../../../docs/deployment.md)
