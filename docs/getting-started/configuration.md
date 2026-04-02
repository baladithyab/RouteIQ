# Configuration

RouteIQ Gateway is configured via YAML files and environment variables.

## Configuration File Location

The gateway looks for configuration in this order:

1. `--config` CLI argument
2. `LITELLM_CONFIG_PATH` environment variable
3. `/app/config/config.yaml` (default in container)

## Configuration Sections

### Model List

Define your LLM providers:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gemini-pro
    litellm_params:
      model: vertex_ai/gemini-pro
      vertex_project: my-project
      vertex_location: us-central1
```

### Router Settings

Configure routing behavior:

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
    llm_data_path: /app/config/llm_candidates.json
    hot_reload: true
    reload_interval: 300
  num_retries: 2
  timeout: 600
```

### General Settings

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
```

## Available Routing Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| `simple-shuffle` | LiteLLM | Random selection (default) |
| `least-busy` | LiteLLM | Route to least busy deployment |
| `latency-based-routing` | LiteLLM | Optimize for latency |
| `cost-based-routing` | LiteLLM | Optimize for cost |
| `llmrouter-knn` | ML | K-Nearest Neighbors routing |
| `llmrouter-svm` | ML | Support Vector Machine routing |
| `llmrouter-mlp` | ML | Neural network routing |
| `llmrouter-mf` | ML | Matrix factorization routing |
| `llmrouter-elo` | ML | Elo rating routing |
| `llmrouter-hybrid` | ML | Probabilistic hybrid routing |

See [Routing Strategies](../features/routing.md) for the full list and configuration details.

## Environment Variables

### Core

| Variable | Required | Description |
|----------|----------|-------------|
| `LITELLM_MASTER_KEY` | Yes | Master API key for admin access |
| `LITELLM_CONFIG_PATH` | No | Config file path |
| `DATABASE_URL` | No | PostgreSQL connection string |
| `REDIS_HOST` / `REDIS_PORT` | No | Redis for caching/state |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `true` | Enable OpenTelemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OTLP collector endpoint |
| `OTEL_SERVICE_NAME` | `litellm-gateway` | Service name |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_GATEWAY_ENABLED` | `false` | Enable MCP gateway |
| `A2A_GATEWAY_ENABLED` | `false` | Enable A2A gateway |
| `POLICY_ENGINE_ENABLED` | `false` | Enable policy engine |
| `CONFIG_HOT_RELOAD` | `false` | Enable config hot-reload |
| `ROUTEIQ_CENTROID_ROUTING` | `true` | Enable centroid routing fallback |
| `ROUTEIQ_ROUTING_PROFILE` | `auto` | Default routing profile |
| `ROUTEIQ_ADMIN_UI_ENABLED` | `false` | Enable admin UI |
| `ROUTEIQ_OIDC_ENABLED` | `false` | Enable OIDC/SSO |

## Hot Reload

Enable config hot-reload to pick up changes without restarting:

```bash
CONFIG_HOT_RELOAD=true
```

Requires the `hotreload` extra (`watchdog`).

## Pydantic Settings

All settings are defined in `src/litellm_llmrouter/settings.py` using Pydantic.
Access settings at runtime via:

```python
from litellm_llmrouter.settings import get_settings

settings = get_settings()
print(settings.routing.default_profile)
```
