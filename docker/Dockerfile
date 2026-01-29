# syntax=docker/dockerfile:1.4
# =============================================================================
# LiteLLM + LLMRouter Production Dockerfile
# =============================================================================
# Multi-stage build for minimal, secure production images
# Uses uv for fast, reproducible Python package management with lockfile
# Supports: linux/amd64, linux/arm64
#
# Build optimization:
# - Lockfile-driven installs via `uv sync --frozen` for reproducibility
# - BuildKit cache mounts for uv to avoid baking caches into layers
# - All cleanup in same RUN statements to minimize layer size
# - Dependencies installed before app source for better layer reuse
#
# Security features:
# - Pinned base images via sha256 digest for supply-chain integrity
# - Lockfile verification (--frozen flag fails on mismatch)
# - Non-root user execution
# - Minimal base image (slim variant)
# - Read-only filesystem compatible
# - Health checks for orchestrators
#
# Build args:
#   PYTHON_VERSION  - Python version (default: 3.12)
#   UV_VERSION      - uv package manager version (default: 0.9)
#   LITELLM_VERSION - LiteLLM version (default: 1.80.15)
#   LLMROUTER_COMMIT - LLMRouter git commit hash
#
# Reproducible Builds:
#   This Dockerfile uses uv.lock for deterministic dependency resolution.
#   Run `uv lock --check` locally to verify lockfile is up-to-date before building.
# =============================================================================

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.9
ARG LITELLM_VERSION=1.81.3
# LLMRouter pinned to specific commit for reproducibility
ARG LLMROUTER_COMMIT=7890cd9d36951a3c73fec83619321f3704a7aaa8

# ==============================================================================
# Base image digests for reproducibility
# To update: docker pull <image> && docker inspect --format='{{index .RepoDigests 0}}' <image>
# ==============================================================================
ARG BUILDER_DIGEST=sha256:d8c02dc966030abc9b0184685a451b593fe5ff339a26cfd6d2fbff2d256c960b
ARG RUNTIME_DIGEST=sha256:a0cb65a2dada255d392fe0426d04ed50911f0b78b3295de0c3286c1c86148b15

# ==============================================================================
# Stage 1: Builder - Clone repos and build wheels with uv
# ==============================================================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm@${BUILDER_DIGEST} AS builder

ARG LLMROUTER_COMMIT
ARG LITELLM_VERSION

WORKDIR /build

# Install git for cloning LLMRouter (apt cleanup in same layer)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Clone and build LLMRouter wheel from GitHub at pinned commit
# All git operations and cleanup in single RUN to prevent .git in layer
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth 1 https://github.com/ulab-uiuc/LLMRouter.git /tmp/LLMRouter \
    && cd /tmp/LLMRouter \
    && git fetch --depth 1 origin ${LLMROUTER_COMMIT} \
    && git checkout ${LLMROUTER_COMMIT} \
    && uv build --wheel --out-dir /wheels \
    && rm -rf /tmp/LLMRouter

# ==============================================================================
# Stage 2: Runtime - Production image with uv and OTEL support
# ==============================================================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm-slim@${RUNTIME_DIGEST} AS runtime

ARG LITELLM_VERSION
ARG LLMROUTER_COMMIT
ARG BUILD_DATE
ARG VCS_REF

# OCI Image Labels (https://github.com/opencontainers/image-spec)
LABEL org.opencontainers.image.source="https://github.com/baladithyab/litellm-llm-router"
LABEL org.opencontainers.image.description="LiteLLM + LLMRouter: Intelligent LLM Gateway with ML-Powered Routing"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="${LITELLM_VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="LiteLLM + LLMRouter"
LABEL org.opencontainers.image.title="LiteLLM Gateway"
LABEL llmrouter.commit="${LLMROUTER_COMMIT}"
LABEL build.reproducible="true"
LABEL build.lockfile-driven="uv.lock"

# Security: Create non-root user with specific UID/GID for consistency
RUN groupadd -r -g 1000 litellm && \
    useradd -r -u 1000 -g litellm -d /app -s /sbin/nologin litellm

WORKDIR /app

# Install minimal runtime dependencies in single layer (cleanup in same RUN)
# Note: build-essential needed for uvloop compilation
# Note: libatomic1 needed for Prisma/Node.js on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    tini \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /root/.cache

# Copy wheels from builder (can be cleaned after install)
COPY --from=builder /wheels /wheels

# =============================================================================
# Lockfile-driven dependency installation for reproducibility
# =============================================================================
# Copy lockfile and project metadata first for layer caching
COPY pyproject.toml uv.lock /app/
COPY src/ /app/src/

# Install all Python dependencies using uv sync --frozen
# --frozen: Fail if lockfile is out of sync with pyproject.toml (ensures reproducibility)
# Order: wheels → uv sync with all production extras
RUN --mount=type=cache,target=/root/.cache/uv \
    # Install LLMRouter wheel first (not in lockfile, built from source)
    uv pip install --system /wheels/*.whl || true \
    # Install project dependencies from lockfile with production extras
    # --frozen ensures builds fail if lockfile is out of date
    && cd /app && uv sync --frozen --no-dev --extra prod --extra db --extra otel --extra cloud --extra callbacks --extra knn \
    # Install OpenTelemetry instrumentation packages (may not be in all lockfiles)
    && uv pip install --system \
    opentelemetry-instrumentation-requests \
    # Cleanup wheels and temp files
    && rm -rf /wheels

# Pre-generate Prisma client and fix permissions on LiteLLM directories
# This is done as root to avoid runtime permission errors
# Security: Use least-privilege permissions instead of 777
RUN SCHEMA_PATH=$(python -c "import litellm; import os; print(os.path.join(os.path.dirname(litellm.__file__), 'proxy', 'schema.prisma'))") && \
    if [ -f "$SCHEMA_PATH" ]; then \
    echo "Generating Prisma client from $SCHEMA_PATH..." && \
    python -m prisma generate --schema="$SCHEMA_PATH" || echo "Warning: prisma generate failed"; \
    fi && \
    # Fix permissions on LiteLLM directories that need write access at runtime
    # Using least-privilege: litellm user owns directories with 755 (rwxr-xr-x)
    chown -R litellm:litellm /usr/local/lib/python3.12/site-packages/prisma/ 2>/dev/null || true && \
    chmod -R 755 /usr/local/lib/python3.12/site-packages/prisma/ 2>/dev/null || true && \
    chown -R litellm:litellm /usr/local/lib/python3.12/site-packages/litellm/proxy/_experimental/out/ 2>/dev/null || true && \
    chmod -R 755 /usr/local/lib/python3.12/site-packages/litellm/proxy/_experimental/out/ 2>/dev/null || true

# Copy application files (after dependencies for better layer caching)
COPY --chown=litellm:litellm src/litellm_llmrouter /app/litellm_llmrouter
COPY --chown=litellm:litellm docker/entrypoint.sh /app/entrypoint.sh
COPY --chown=litellm:litellm config/ /app/config/

# Create directories with proper permissions and cleanup
RUN mkdir -p /app/models /app/data /app/custom_routers && \
    chown -R litellm:litellm /app && \
    chmod +x /app/entrypoint.sh && \
    # Security: Remove unnecessary files
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -delete

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONFAULTHANDLER=1 \
    # Application config
    LITELLM_CONFIG_PATH=/app/config/config.yaml \
    LLMROUTER_MODELS_PATH=/app/models \
    LLMROUTER_CUSTOM_ROUTERS_PATH=/app/custom_routers \
    # OTEL defaults (override at runtime)
    OTEL_SERVICE_NAME=litellm-gateway \
    OTEL_TRACES_EXPORTER=none \
    OTEL_METRICS_EXPORTER=none \
    OTEL_LOGS_EXPORTER=none \
    # Gateway features
    A2A_GATEWAY_ENABLED=false \
    MCP_GATEWAY_ENABLED=false \
    # Config sync and hot reload
    CONFIG_HOT_RELOAD=false \
    CONFIG_SYNC_ENABLED=true \
    CONFIG_SYNC_INTERVAL=60 \
    # Store models/MCPs in database for persistence
    STORE_MODEL_IN_DB=false

# Health check for orchestrators (ECS, K8s, etc.)
# Uses the internal health endpoint which is unauthenticated
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:4000/_health/live || exit 1

EXPOSE 4000

# Security: Ensure proper signal handling for graceful shutdown
STOPSIGNAL SIGTERM

# Security: Run as non-root user
USER litellm

# =============================================================================
# ReadOnlyRootFilesystem Support
# =============================================================================
# This image supports Kubernetes readOnlyRootFilesystem: true with the following
# writable volume mounts required:
#   - /app/data      - Runtime data (if using local file storage)
#   - /app/models    - Model files (if hot-reloading models)
#   - /tmp           - Temporary files (Python, Prisma)
#   - /app/.cache    - Application cache (optional)
#
# Example K8s securityContext:
#   securityContext:
#     readOnlyRootFilesystem: true
#     runAsNonRoot: true
#     runAsUser: 1000
#   volumeMounts:
#     - name: tmp-vol
#       mountPath: /tmp
#     - name: data-vol
#       mountPath: /app/data
# =============================================================================

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["--config", "/app/config/config.yaml", "--port", "4000"]
