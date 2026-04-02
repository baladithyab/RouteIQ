# Docker

RouteIQ provides multi-tier Docker images for different deployment scenarios.

## Image Variants

| Image | Size | ML Routing | Use Case |
|-------|------|------------|----------|
| `routeiq:latest` | ~1.2GB | Yes | Full gateway |
| `routeiq:slim` | ~500MB | No | Proxy-only |

## Building Images

```bash
# Full image
docker build -f docker/Dockerfile -t routeiq:latest .

# Slim image
docker build -f docker/Dockerfile.slim -t routeiq:slim .
```

## Docker Compose Examples

All examples under `examples/docker/`:

### Basic

```bash
docker compose up -d
```

### High Availability

```bash
docker compose -f examples/docker/ha/docker-compose.ha.yml up -d
```

Includes:
- 2 RouteIQ replicas
- Redis for state
- PostgreSQL for persistence
- Nginx load balancer

### Observability

```bash
docker compose -f examples/docker/observability/docker-compose.otel.yml up -d
```

Includes:
- RouteIQ gateway
- OTel Collector
- Jaeger for traces

## Reproducible Builds

RouteIQ uses lockfile-driven builds:

- Dependencies installed via `uv sync --frozen`
- Base images pinned to SHA256 digests
- SBOM and provenance attestations

```bash
# Verify lockfile before building
uv lock --check

# Build with frozen dependencies
docker build -f docker/Dockerfile -t routeiq:latest .
```
