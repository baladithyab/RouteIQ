# Hot Reloading Guide

This guide explains how to update routing models without restarting the gateway.

## Overview

LLMRouter strategies support hot reloading, allowing you to:
- Update routing models in production
- A/B test new models
- Rollback quickly if issues occur

## Enabling Hot Reload

Enable in your configuration:

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
    hot_reload: true
    reload_interval: 300  # seconds
```

## How It Works

1. **File Monitoring**: The gateway checks the model file's modification time
2. **Reload Trigger**: If the file changed since last check, reload is triggered
3. **Thread-Safe Loading**: New model is loaded while old model handles requests
4. **Atomic Swap**: Once loaded, requests switch to the new model

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Request 1  │────►│  Model v1   │     │             │
└─────────────┘     └─────────────┘     │             │
                                        │   Gateway   │
┌─────────────┐     ┌─────────────┐     │             │
│ Model v2    │────►│   Loading   │     │             │
│ uploaded    │     └──────┬──────┘     └─────────────┘
└─────────────┘            │
                           ▼
┌─────────────┐     ┌─────────────┐
│  Request 2  │────►│  Model v2   │  ◄── Atomic swap
└─────────────┘     └─────────────┘
```

## Updating Models

### Local Volume Mount

If using volume mounts, simply replace the model file:

```bash
cp new_model.pt ./models/knn_router.pt
```

The gateway will detect the change within `reload_interval` seconds.

### S3-Based Updates

For S3-stored models:

1. Upload new model to S3:
   ```bash
   aws s3 cp new_model.pt s3://my-bucket/models/knn_router.pt
   ```

2. The gateway downloads and loads on next check

### API-Triggered Reload

Force immediate reload via API:

```bash
curl -X POST http://localhost:4000/router/reload \
  -H "Authorization: Bearer sk-master-key" \
  -H "Content-Type: application/json" \
  -d '{"strategy": "llmrouter-knn"}'
```

## Configuration Reload

Reload entire config without restart:

```bash
curl -X POST http://localhost:4000/config/reload \
  -H "Authorization: Bearer sk-master-key"
```

## Best Practices

### 1. Test Before Deploy
Always test new models in staging:

```bash
# Deploy to staging
docker compose -f docker-compose.staging.yml exec gateway \
  python -c "from litellm_llmrouter.strategies import LLMRouterStrategyFamily; ..."
```

### 2. Monitor After Reload
Watch metrics after model updates:

```bash
# Check routing decisions
curl http://localhost:4000/metrics | grep llmrouter
```

### 3. Keep Rollback Ready
Maintain previous model version:

```bash
cp models/knn_router.pt models/knn_router.pt.backup
```

### 4. Use Version Tags
Tag model versions in S3:

```bash
aws s3 cp new_model.pt s3://bucket/models/knn_router_v2.pt
# Update config to point to new version
```

## Troubleshooting

### Model Not Reloading

1. Check file permissions
2. Verify `hot_reload: true` in config
3. Check logs: `docker compose logs -f litellm-llmrouter`

### Reload Errors

Check for:
- Model format compatibility
- Missing dependencies
- Memory constraints

View detailed logs:
```bash
export LITELLM_LOG=DEBUG
docker compose up
```
