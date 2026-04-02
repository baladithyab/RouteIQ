# Installation

RouteIQ Gateway can be installed via pip, Docker, or Helm.

## Python Package

### Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install with uv

```bash
# Install core package
uv add routeiq

# Install with optional extras
uv add routeiq[knn]       # KNN routing (sentence-transformers)
uv add routeiq[oidc]      # OIDC/SSO authentication
uv add routeiq[prod]      # All production dependencies
uv add routeiq[all]       # Everything including dev tools
```

### Install with pip

```bash
pip install routeiq
pip install routeiq[prod]  # Production extras
```

### Start the Gateway

```bash
routeiq start --config config/config.yaml --port 4000
```

Or directly with Python:

```bash
python -m litellm_llmrouter.startup --config config/config.yaml --port 4000
```

## Docker

RouteIQ provides multi-tier Docker images:

| Image | Size | Use Case |
|-------|------|----------|
| `routeiq:latest` | ~1.2GB | Full gateway with ML routing dependencies |
| `routeiq:slim` | ~500MB | Proxy-only, no ML dependencies |

### Pull and Run

```bash
# Full image
docker run -p 4000:4000 \
  -e LITELLM_MASTER_KEY=sk-your-key \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/config:/app/config \
  routeiq:latest

# Slim image (no ML routing)
docker run -p 4000:4000 \
  -e LITELLM_MASTER_KEY=sk-your-key \
  routeiq:slim
```

### Build Locally

```bash
# Full image
docker build -f docker/Dockerfile -t routeiq:latest .

# Slim image
docker build -f docker/Dockerfile.slim -t routeiq:slim .
```

## Docker Compose

Several compose configurations are provided under `examples/docker/`:

| Scenario | Path | Description |
|----------|------|-------------|
| Basic | `examples/docker/basic/` | Minimal single-container setup |
| HA | `examples/docker/ha/` | Multi-replica with Redis + Postgres + Nginx |
| Observability | `examples/docker/observability/` | OTel Collector + Jaeger |
| Full Stack | `examples/docker/full-stack/` | HA + Observability combined |

```bash
# Basic setup
docker compose up -d

# HA setup
docker compose -f examples/docker/ha/docker-compose.ha.yml up -d
```

## Kubernetes (Helm)

Helm charts are provided in `deploy/charts/`:

```bash
helm install routeiq deploy/charts/routeiq \
  --namespace routeiq \
  --create-namespace \
  --set config.masterKey=sk-your-key
```

See [Helm Chart](../operations/helm.md) for detailed configuration.

## Optional Dependencies

RouteIQ uses tiered dependencies to keep the base install lightweight:

| Extra | Packages | Purpose |
|-------|----------|--------|
| `dev` | pytest, ruff, mypy | Development and testing |
| `db` | asyncpg, prisma | PostgreSQL support |
| `otel` | opentelemetry-* | Full observability instrumentation |
| `cloud` | boto3, google-cloud, azure-identity | Cloud provider SDKs |
| `knn` | sentence-transformers, scikit-learn | KNN routing inference |
| `a2a` | a2a-sdk | Agent-to-Agent protocol |
| `oidc` | authlib | OIDC/SSO authentication |
| `hotreload` | watchdog | Filesystem config hot-reload |
| `callbacks` | langfuse | Callback integrations |
| `prod` | All of the above (except dev) | Production deployment |
| `all` | Everything | Full install |
