# Configuration Guide

This guide covers all configuration options for the LiteLLM + LLMRouter gateway.

## Configuration File Location

The gateway looks for configuration in these locations (in order):

1. Path specified via `--config` CLI argument
2. `LITELLM_CONFIG_PATH` environment variable
3. `/app/config/config.yaml` (default in container)

## Configuration Sections

### Model List

Define your LLM providers and their endpoints:

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
```

### Router Settings

Configure routing behavior:

```yaml
router_settings:
  # Choose routing strategy
  routing_strategy: llmrouter-knn

  # LLMRouter-specific settings
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
    llm_data_path: /app/config/llm_candidates.json
    hot_reload: true
    reload_interval: 300

  # Retry settings
  num_retries: 2
  timeout: 600
```

### Available Routing Strategies

| Strategy | Description |
|----------|-------------|
| `simple-shuffle` | Random selection (LiteLLM default) |
| `least-busy` | Route to least busy deployment |
| `latency-based-routing` | Optimize for latency |
| `cost-based-routing` | Optimize for cost |
| `llmrouter-knn` | K-Nearest Neighbors routing |
| `llmrouter-svm` | Support Vector Machine routing |
| `llmrouter-mlp` | Multi-Layer Perceptron routing |
| `llmrouter-mf` | Matrix Factorization routing |
| `llmrouter-hybrid` | Hybrid probabilistic routing |
| `llmrouter-custom` | Your custom trained model |

### General Settings

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
```

### LiteLLM Settings

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: 6379
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LITELLM_MASTER_KEY` | Admin API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | If using OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | If using Anthropic |
| `DATABASE_URL` | PostgreSQL connection string | For HA |
| `REDIS_HOST` | Redis host for caching | For caching |
| `CONFIG_S3_BUCKET` | S3 bucket for config | For S3 config |
| `CONFIG_S3_KEY` | S3 key for config file | For S3 config |

## Loading Config from S3

Set these environment variables to load config from S3 on startup:

```bash
CONFIG_S3_BUCKET=my-config-bucket
CONFIG_S3_KEY=configs/litellm-config.yaml
```

## LLM Candidates JSON

The `llm_candidates.json` file describes available models for LLMRouter:

```json
{
  "gpt-4": {
    "provider": "openai",
    "capabilities": ["reasoning", "coding"],
    "cost_per_1k_tokens": {"input": 0.03, "output": 0.06},
    "quality_score": 0.95
  }
}
```
