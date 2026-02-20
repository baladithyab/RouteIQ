# RouteIQ Examples

Ready-to-run deployment scenarios and tooling examples for **RouteIQ Gateway** — a production-grade, cloud-native General AI Gateway with intelligent ML-based routing.

---

## Quick Start with OpenRouter

All Docker examples are pre-configured to use [OpenRouter](https://openrouter.ai/) so you can test with **100+ models** using a single API key.

1. **Get an API key** at <https://openrouter.ai/>
2. **Set it in your `.env`**:
   ```bash
   OPENROUTER_API_KEY=your-key-here
   ```
3. **Pick a deployment scenario** from the table below

The shared config [`config/config.openrouter.yaml`](../config/config.openrouter.yaml) ships test-ready with three models:

| Model | Alias |
|-------|-------|
| `openrouter/openai/gpt-4o-mini` | gpt-4o-mini |
| `openrouter/anthropic/claude-3-haiku` | claude-3-haiku |
| `openrouter/google/gemini-flash-1.5` | gemini-flash |

---

## Docker Deployment Scenarios

| Scenario | Directory | Description | Best For |
|----------|-----------|-------------|----------|
| **Basic** | `docker/basic/` | Single gateway instance | Quick start, evaluation |
| **High Availability** | `docker/ha/` | 2 replicas + PostgreSQL + Redis + Nginx | Production staging |
| **Observability** | `docker/observability/` | Gateway + OTel Collector + Jaeger | Debugging, tracing |
| **Full Stack** | `docker/full-stack/` | HA + Observability + A2A + MCP + Admin UI | Production deployment |
| **Local Development** | `docker/local-dev/` | Dev gateway + PostgreSQL + Redis + Jaeger + MinIO + MLflow | Development, testing |

### Quick-Run Commands

Every scenario follows the same pattern:

#### Basic

```bash
cd examples/docker/basic
cp .env.example .env
docker compose up -d
```

#### High Availability

```bash
cd examples/docker/ha
cp .env.example .env
docker compose up -d
```

#### Observability

```bash
cd examples/docker/observability
cp .env.example .env
docker compose up -d
```

#### Full Stack

```bash
cd examples/docker/full-stack
cp .env.example .env
docker compose up -d
```

#### Local Development

```bash
cd examples/docker/local-dev
cp .env.example .env
docker compose up -d
```

---

## MLOps Training Pipeline

The [`mlops/`](mlops/) directory contains an end-to-end ML pipeline for training routing models used by RouteIQ's intelligent routing engine.

**Capabilities:**

- **Synthetic data generation** — create training datasets without production traffic
- **Jaeger trace extraction** — harvest real routing decisions from OpenTelemetry traces
- **Model training** — train KNN, MLP, SVM, and MF (matrix factorization) routing models
- **Model deployment** — deploy trained models back to the gateway

👉 See [`mlops/README.md`](mlops/README.md) for full setup and usage instructions.
📖 See [`docs/mlops-training.md`](../docs/mlops-training.md) for detailed documentation.

---

## What's Included in Each Docker Example

| Service | Basic | HA | Observability | Full Stack | Local Dev |
|---------|:-----:|:--:|:-------------:|:----------:|:---------:|
| RouteIQ Gateway | ✅ | ✅ (×2) | ✅ | ✅ (×2) | ✅ |
| PostgreSQL | | ✅ | | ✅ | ✅ |
| Redis | | ✅ | | ✅ | ✅ |
| Nginx LB | | ✅ | | ✅ | |
| OTel Collector | | | ✅ | ✅ | |
| Jaeger | | | ✅ | ✅ | ✅ |
| MinIO (S3) | | | | | ✅ |
| MLflow | | | | | ✅ |

---

## Key Features by Default

All deployment scenarios include the following features out of the box:

- **Centroid Routing** — Zero-config intelligent routing with ~2ms classification latency. Routes requests to the best model without any ML model training required.
- **Plugin Strategy** — Multi-worker support enabled via `ROUTEIQ_USE_PLUGIN_STRATEGY=true`. Allows horizontal scaling with multiple uvicorn workers.
- **SSRF Protection** — All external requests are validated against SSRF attacks, including DNS rebinding protection.
- **Health Probes** — Kubernetes-compatible liveness and readiness endpoints at `/_health/live` and `/_health/ready`.

---

## Customization

### Use Your Own LLM Provider

1. Edit the config YAML in your scenario's directory (or override with a custom config):
   ```yaml
   model_list:
     - model_name: my-model
       litellm_params:
         model: openai/gpt-4o
         api_key: os.environ/OPENAI_API_KEY
   ```
2. Set the corresponding API keys in your `.env` file:
   ```bash
   OPENAI_API_KEY=sk-...
   ```

### Enable Optional Features

Add these environment variables to your `.env` to enable additional capabilities:

| Feature | Environment Variable | Default |
|---------|---------------------|---------|
| Admin UI | `ROUTEIQ_ADMIN_UI_ENABLED=true` | `false` |
| MCP Gateway | `MCP_GATEWAY_ENABLED=true` | `false` |
| A2A Gateway | `A2A_GATEWAY_ENABLED=true` | `false` |
| Policy Engine | `POLICY_ENGINE_ENABLED=true` | `false` |
| Config Hot-Reload | `CONFIG_HOT_RELOAD=true` | `false` |

📖 See [`docs/configuration.md`](../docs/configuration.md) for the full configuration reference.

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [Quickstart](../docs/quickstart-docker-compose.md) | Get up and running in under 5 minutes |
| [Deployment Guide](../docs/deployment.md) | Docker, Kubernetes, and cloud deployment |
| [Configuration](../docs/configuration.md) | Full configuration reference |
| [Routing Strategies](../docs/routing-strategies.md) | ML routing algorithms and tuning |
| [MLOps Training](../docs/mlops-training.md) | End-to-end model training pipeline |
| [Security](../docs/security.md) | Security considerations and hardening |
