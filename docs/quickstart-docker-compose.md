# Quickstart: Docker Compose

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

This guide helps you get RouteIQ Gateway up and running quickly using Docker Compose. This setup is ideal for local development and testing.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.
- `openssl` (for generating keys).

## 1. Clone the Repository

```bash
git clone https://github.com/codeseys/RouteIQ.git
cd RouteIQ
```

## 2. Configure Environment

Start by copying the example environment file:

```bash
cp .env.example .env
```

Then edit `.env` to set your keys.

**Required Environment Variables:**

| Variable | Description |
|----------|-------------|
| `LITELLM_MASTER_KEY` | **Critical**. The admin master key for the gateway. Generate a strong random string. |
| `OPENAI_API_KEY` | (Optional) If using OpenAI models. |
| `ANTHROPIC_API_KEY` | (Optional) If using Anthropic models. |

**Security Note:** You **must** generate a secure `LITELLM_MASTER_KEY`.

```bash
# Generate a secure master key
export LITELLM_MASTER_KEY=$(openssl rand -hex 32)
# Update your .env file with this key
```

## 3. Start the Gateway

Run the default Docker Compose configuration:

```bash
docker-compose up -d
```

This starts the `litellm-llmrouter` service on port **4000**.

## 4. Verify Installation

Check the health endpoint:

```bash
curl http://localhost:4000/health
# Expected: {"status":"healthy", ...}
```

## 5. Make a Test Request

Send a request to the OpenAI-compatible endpoint:

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello from RouteIQ!"}
    ]
  }'
```

## Configuration

The default setup mounts `./config/config.yaml` to the container. You can modify this file to add models or change routing strategies.

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LITELLM_MASTER_KEY` | **Required**. Admin master key. | (none) |
| `LLMROUTER_HOT_RELOAD` | Enable hot reloading of routing models. | `true` |
| `A2A_GATEWAY_ENABLED` | Enable Agent-to-Agent gateway. | `false` |
| `MCP_GATEWAY_ENABLED` | Enable MCP gateway. | `false` |

### Security Defaults

- **SSRF Protection**: Deny-by-default for private IPs.
- **Admin Auth**: Control plane endpoints require `X-Admin-API-Key` or `LITELLM_MASTER_KEY`.
- **MCP Tool Invocation**: Disabled by default. Enable with `LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true`.

## Next Steps

- **Need reliability?** Switch to the [High Availability Setup](tutorials/ha-quickstart.md).
- **Need visibility?** Add the [Observability Stack](tutorials/observability-quickstart.md).
- **Configuration:** See the [Configuration Guide](configuration.md).
