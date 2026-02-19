# RouteIQ Gateway - Observability Setup

Single gateway instance with OpenTelemetry tracing via Jaeger. Perfect for development, debugging, and understanding request flows through the gateway.

## What This Provides

- **Single gateway instance** on port 4001
- **Jaeger** for distributed tracing with a web UI
- **Full trace sampling** - every request is traced
- **Router decision telemetry** - see which routing strategy was used

## Prerequisites

- Docker and Docker Compose v2+
- An OpenRouter API key ([get one here](https://openrouter.ai/keys))

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env to set your API key (pre-filled with a test key)

# 2. Start the stack
docker compose up -d

# 3. Send a request to generate traces
curl http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-test-master-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# 4. View traces in Jaeger
open http://localhost:16686
```

## Verify It's Working

```bash
# Check gateway health
curl http://localhost:4001/_health/live

# Send a request and then check Jaeger UI
# Navigate to http://localhost:16686, select "litellm-gateway" service
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Gateway | http://localhost:4001 | LLM API endpoint |
| Jaeger UI | http://localhost:16686 | Trace visualization |
| OTLP gRPC | localhost:4317 | For external trace export |
| OTLP HTTP | localhost:4318 | For external trace export |

## Key Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_MASTER_KEY` | `sk-test-master-key-change-me` | Admin API key |
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key |
| `OTEL_TRACES_SAMPLER` | `always_on` | Trace sampling strategy |
| `OTEL_SERVICE_NAME` | `litellm-gateway` | Service name in traces |

## What to Look For in Traces

- **Request duration** - total time from request to response
- **Router decision** - which routing strategy selected the model
- **Model selection** - which LLM model was chosen
- **Provider latency** - time spent waiting for the LLM provider

## Next Steps

- **Need HA + observability?** See [../full-stack/](../full-stack/)
- **Just need basic setup?** See [../basic/](../basic/)
- **Full dev environment?** See [../local-dev/](../local-dev/)

## Related Documentation

- [Observability Guide](../../../docs/observability.md)
- [Observability Quickstart](../../../docs/tutorials/observability-quickstart.md)
