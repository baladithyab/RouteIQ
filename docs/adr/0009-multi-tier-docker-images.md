# ADR-0009: Multi-Tier Docker Images (Slim/Full/GPU)

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: Monolithic 2-4GB Docker Image

RouteIQ's production Docker image included all dependencies regardless of
which features were used. The primary size contributors:

| Component | Size | Required For |
|-----------|-----:|-------------|
| PyTorch | ~800MB-1.5GB | KNN routing (sentence-transformers) |
| sentence-transformers | ~200MB | KNN routing embeddings |
| boto3 + botocore | ~60MB | S3 config sync |
| scikit-learn | ~30MB | KNN/SVM/MLP model inference |
| Base Python + system libs | ~200MB | Everything |
| LiteLLM + deps | ~150MB | Core proxy |
| RouteIQ + other deps | ~50MB | Gateway |

For deployments using only centroid routing (zero-config, no ML models),
~1.7GB of the image was unused dependencies.

### Impact

1. **Cold start time**: Larger images take longer to pull from registries.
   In Kubernetes, a 3GB image can add 30-60s to pod startup on cold nodes.

2. **Registry costs**: Storing multiple versions of 3GB images consumes
   significant registry storage (ECR, GCR, ACR).

3. **Attack surface**: PyTorch and scikit-learn include C extensions with
   potential vulnerabilities. Installing them when unused is risk without
   benefit.

4. **Resource consumption**: Even when not actively used, loaded Python
   packages consume memory via imported modules and cached objects.

## Decision

Implement a parameterized Dockerfile that produces multiple image tiers
via build ARGs.

### Dockerfile Parameterization

```dockerfile
# Build arguments for image customization
ARG PYTHON_VERSION=3.14
ARG ROUTEIQ_EXTRAS="prod"
ARG BUILD_UI=true
ARG INSTALL_LLMROUTER=true

# Stage 1: Build
FROM python:${PYTHON_VERSION}-slim AS builder
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN pip install uv && uv sync --extra ${ROUTEIQ_EXTRAS} --no-dev

# Stage 2: Runtime
FROM python:${PYTHON_VERSION}-slim AS runtime
COPY --from=builder /app/.venv /app/.venv
...
```

### Image Tiers

| Tier | Build Command | Size | Features |
|------|--------------|-----:|----------|
| **Slim** | `ROUTEIQ_EXTRAS=db,otel` | ~500MB | Centroid routing, DB, OTel. No ML models. |
| **Full** | `ROUTEIQ_EXTRAS=prod` | ~2GB | All features including KNN routing. |
| **GPU** | `ROUTEIQ_EXTRAS=prod` + CUDA base | ~4GB | Full + GPU-accelerated inference. |

### Slim Image (~500MB)

The slim image is the default for most deployments:

- Core gateway with LiteLLM proxy
- Centroid-based zero-config routing (~2ms, no ML deps needed)
- PostgreSQL persistence (`[db]`)
- OpenTelemetry observability (`[otel]`)
- OIDC authentication (`[oidc]`)
- No PyTorch, no sentence-transformers, no scikit-learn
- No boto3 (S3 config sync unavailable)

Centroid routing (see [ADR-0010](0010-centroid-zero-config-routing.md))
provides intelligent prompt-based model selection using pre-computed
embedding vectors (~100KB), making ML dependencies unnecessary for most
use cases.

### Full Image (~2GB)

For deployments that need ML routing strategies:

- Everything in slim
- KNN routing with sentence-transformers (`[knn]`)
- S3/GCS config sync (`[cloud]`)
- A2A protocol (`[a2a]`)
- Hot-reload (`[hotreload]`)
- All callback integrations (`[callbacks]`)

### GPU Image (~4GB)

For high-throughput deployments with GPU-accelerated inference:

- Everything in full
- NVIDIA CUDA base image
- PyTorch with CUDA support
- GPU-accelerated sentence-transformers for KNN routing

### Build Examples

```bash
# Slim (default for most deployments)
docker build -f docker/Dockerfile \
  --build-arg ROUTEIQ_EXTRAS="db,otel,oidc" \
  -t routeiq:slim .

# Full (all features)
docker build -f docker/Dockerfile \
  --build-arg ROUTEIQ_EXTRAS="prod" \
  -t routeiq:full .

# GPU (CUDA-accelerated)
docker build -f docker/Dockerfile \
  --build-arg ROUTEIQ_EXTRAS="prod" \
  --build-arg PYTHON_VERSION=3.14-cuda12.1 \
  -t routeiq:gpu .
```

## Consequences

### Positive

- **Faster cold starts**: Slim image pulls in 10-15s vs 30-60s for full.
  Critical for Kubernetes autoscaling and serverless deployments.

- **Reduced attack surface**: Slim image has ~70% fewer installed packages,
  proportionally fewer potential CVEs.

- **Lower registry costs**: Storing slim images uses 75% less storage.

- **Appropriate resource allocation**: Deployments not using ML routing
  don't consume memory for PyTorch and scikit-learn.

- **Single Dockerfile**: One parameterized Dockerfile instead of multiple
  Dockerfiles to maintain.

### Negative

- **Feature discovery**: Users must know which extras they need. The default
  slim image doesn't include ML routing, which might surprise users expecting
  all features. Mitigated by documentation and docker-compose defaults.

- **Upgrade complexity**: Switching from slim to full requires rebuilding
  the image with different ARGs. Can't add ML routing to a running slim
  deployment without redeployment.

- **CI matrix**: Building and testing all image tiers adds CI time and
  complexity.

## Alternatives Considered

### Alternative A: Single Monolithic Image

Keep one image with all dependencies.

- **Pros**: Simplest; all features always available.
- **Cons**: 2-4GB image for every deployment; wasted resources.
- **Rejected**: Unacceptable for production Kubernetes deployments.

### Alternative B: Separate Dockerfiles

Maintain separate `Dockerfile.slim`, `Dockerfile.full`, `Dockerfile.gpu`.

- **Pros**: Clear separation; each Dockerfile is self-contained.
- **Cons**: Triple maintenance burden; common layers must be kept in sync;
  changes to base setup must be applied to all three.
- **Rejected**: Build ARG parameterization achieves the same result with
  a single Dockerfile.

### Alternative C: Multi-Stage with Feature Layers

Use Docker multi-stage builds with feature-specific layers that can be
included or excluded.

- **Pros**: More granular control; shared base layers.
- **Cons**: Docker doesn't support conditional COPY between stages based
  on build ARGs cleanly. The Dockerfile becomes complex with many stages.
- **Partially adopted**: The multi-stage build (builder + runtime) is used,
  but feature selection is done via `ROUTEIQ_EXTRAS` rather than separate
  stages.

## References

- `docker/Dockerfile` — Parameterized production Dockerfile
- `docker/Dockerfile.local` — Local development Dockerfile
- `pyproject.toml` — Optional extras definitions
- [ADR-0007: Dependency Tiering](0007-dependency-tiering.md)
- [ADR-0010: Centroid-Based Zero-Config Routing](0010-centroid-zero-config-routing.md)
