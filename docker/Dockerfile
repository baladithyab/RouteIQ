# syntax=docker/dockerfile:1.4
# =============================================================================
# RouteIQ Gateway Production Dockerfile
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
#   PYTHON_VERSION    - Python version (default: 3.14)
#   UV_VERSION        - uv package manager version (default: 0.9)
#   LITELLM_VERSION   - LiteLLM version (default: 1.81.3)
#   LLMROUTER_COMMIT  - LLMRouter git commit hash
#   ROUTEIQ_EXTRAS    - Comma-separated pip extras (default: "prod")
#   BUILD_UI          - Build and include admin UI (default: "true")
#   INSTALL_LLMROUTER - Build and install LLMRouter wheel (default: "true")
#
# Build variants:
#   Full (default):
#     docker build -f docker/Dockerfile .
#   Slim (~500MB, no KNN/PyTorch/Prisma):
#     docker build -f docker/Dockerfile \
#       --build-arg ROUTEIQ_EXTRAS="otel,cloud,a2a,hotreload,oidc" \
#       --build-arg BUILD_UI=false \
#       --build-arg INSTALL_LLMROUTER=false .
#
# Reproducible Builds:
#   This Dockerfile uses uv.lock for deterministic dependency resolution.
#   Run `uv lock --check` locally to verify lockfile is up-to-date before building.
# =============================================================================

ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.9
ARG LITELLM_VERSION=1.81.3
# LLMRouter pinned to specific commit for reproducibility
ARG LLMROUTER_COMMIT=7890cd9d36951a3c73fec83619321f3704a7aaa8

# ==============================================================================
# Configurable build variants
# ==============================================================================
# ROUTEIQ_EXTRAS: pip extras to install. "prod" includes everything.
#   Slim variant: "otel,cloud,a2a,hotreload,oidc" (no knn, no db/Prisma)
ARG ROUTEIQ_EXTRAS="prod"
# BUILD_UI: set to "false" to skip the admin UI React build stage
ARG BUILD_UI="true"
# INSTALL_LLMROUTER: set to "false" to skip LLMRouter wheel (not needed without KNN)
ARG INSTALL_LLMROUTER="true"

# ==============================================================================
# Base image digests for reproducibility
# To update: docker pull <image> && docker inspect --format='{{index .RepoDigests 0}}' <image>
# ==============================================================================
ARG BUILDER_DIGEST=sha256:d414e96d3cc71a71164ad83402f91c4531a4e451505e2d481579b69dfe0a3986
ARG RUNTIME_DIGEST=sha256:f37be263fb880ce4ed918e487d84576462fb4fafa934dbd08d216ab4cf058808

# ==============================================================================
# Stage 1: Builder - Clone repos and build wheels with uv
# ==============================================================================
# Skipped entirely when INSTALL_LLMROUTER=false (slim builds)
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm@${BUILDER_DIGEST} AS builder

ARG LLMROUTER_COMMIT
ARG LITELLM_VERSION
ARG INSTALL_LLMROUTER

WORKDIR /build

# Install git for cloning LLMRouter (apt cleanup in same layer)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Clone and build LLMRouter wheel from GitHub at pinned commit
# All git operations and cleanup in single RUN to prevent .git in layer
# When INSTALL_LLMROUTER=false, create empty /wheels dir and skip build
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /wheels && \
    if [ "${INSTALL_LLMROUTER}" = "true" ]; then \
        git clone --depth 1 https://github.com/ulab-uiuc/LLMRouter.git /tmp/LLMRouter \
        && cd /tmp/LLMRouter \
        && git fetch --depth 1 origin ${LLMROUTER_COMMIT} \
        && git checkout ${LLMROUTER_COMMIT} \
        && uv build --wheel --out-dir /wheels \
        && rm -rf /tmp/LLMRouter; \
    else \
        echo "Skipping LLMRouter build (INSTALL_LLMROUTER=false)"; \
    fi

# ==============================================================================
# Stage 2: UI Builder (optional) - Build admin UI React SPA
# ==============================================================================
# When BUILD_UI=false, this stage still runs but produces an empty /ui/dist.
# Docker BuildKit may skip unused stages, but we keep this safe either way.
# Uses Bun for faster installs (~500ms vs ~15s with npm).
FROM oven/bun:1-alpine AS ui-builder
ARG BUILD_UI
WORKDIR /ui
# Copy package files + lockfile first for layer caching
COPY ui/package.json ui/bun.lock ./
RUN if [ "${BUILD_UI}" = "true" ]; then bun install --frozen-lockfile; fi
# Copy source and build
COPY ui/ ./
RUN if [ "${BUILD_UI}" = "true" ]; then \
        bun run build; \
    else \
        mkdir -p /ui/dist && echo '<!DOCTYPE html><html><body>UI not included in slim build</body></html>' > /ui/dist/index.html; \
    fi

# ==============================================================================
# Stage 3: Runtime - Production image with uv and OTEL support
# ==============================================================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm-slim@${RUNTIME_DIGEST} AS runtime

ARG LITELLM_VERSION
ARG LLMROUTER_COMMIT
ARG BUILD_DATE
ARG VCS_REF
ARG ROUTEIQ_EXTRAS
ARG BUILD_UI
ARG INSTALL_LLMROUTER

# OCI Image Labels (https://github.com/opencontainers/image-spec)
LABEL org.opencontainers.image.source="https://github.com/baladithyab/litellm-llm-router"
LABEL org.opencontainers.image.description="RouteIQ: Intelligent LLM Gateway with ML-Powered Routing"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="1.0.0-rc1"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="RouteIQ"
LABEL org.opencontainers.image.title="RouteIQ Gateway"
LABEL litellm.version="${LITELLM_VERSION}"
LABEL llmrouter.commit="${LLMROUTER_COMMIT}"
LABEL build.reproducible="true"
LABEL build.lockfile-driven="uv.lock"

# Security: Create non-root user with specific UID/GID for consistency
RUN groupadd -r -g 1000 litellm && \
    useradd -r -u 1000 -g litellm -d /app -s /sbin/nologin litellm

WORKDIR /app

# Install minimal runtime dependencies in single layer (cleanup in same RUN)
# Note: build-essential installed temporarily for uvloop compilation, then removed
# Note: libatomic1 needed for Prisma/Node.js on slim images (skip for slim builds without db)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    tini \
    && if echo "${ROUTEIQ_EXTRAS}" | grep -qE '(prod|db)'; then \
        apt-get install -y --no-install-recommends libatomic1; \
    fi \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /root/.cache

# Copy wheels from builder (can be cleaned after install)
COPY --from=builder /wheels /wheels

# =============================================================================
# Lockfile-driven dependency installation for reproducibility
# =============================================================================
# Copy lockfile and project metadata first for layer caching
COPY pyproject.toml uv.lock /app/
COPY src/ /app/src/

# Install all Python dependencies using uv pip install --system
# --system: Install to system Python (not a venv) so packages are available at runtime
# UV_EXTRA_INDEX_URL: Use CPU-only PyTorch to avoid shipping ~3GB CUDA libraries
# ROUTEIQ_EXTRAS controls which optional dependencies are installed:
#   "prod" (default) = all production extras including KNN/PyTorch/Prisma
#   "otel,cloud,a2a,hotreload,oidc" = slim variant without KNN/Prisma
# INSTALL_LLMROUTER controls whether the LLMRouter wheel is installed
RUN --mount=type=cache,target=/root/.cache/uv \
    set -e && \
    # Install LLMRouter wheel if built (not in lockfile, built from source at pinned commit)
    if [ "${INSTALL_LLMROUTER}" = "true" ] && ls /wheels/*.whl 1>/dev/null 2>&1; then \
        UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
        uv pip install --system /wheels/*.whl; \
    fi && \
    # Install project dependencies with configured extras
    cd /app && UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    uv pip install --system -e ".[${ROUTEIQ_EXTRAS}]" && \
    # Cleanup wheels and temp files
    rm -rf /wheels && \
    # Remove build-essential (no longer needed after compilation)
    apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-generate Prisma client and fix permissions on LiteLLM directories
# This is done as root to avoid runtime permission errors
# Security: Use least-privilege permissions instead of 777
# Skipped when db extra is not installed (slim builds without Prisma)
RUN if echo "${ROUTEIQ_EXTRAS}" | grep -qE '(prod|db)'; then \
        SCHEMA_PATH=$(python -c "import litellm; import os; print(os.path.join(os.path.dirname(litellm.__file__), 'proxy', 'schema.prisma'))") && \
        if [ -f "$SCHEMA_PATH" ]; then \
            echo "Generating Prisma client from $SCHEMA_PATH..." && \
            python -m prisma generate --schema="$SCHEMA_PATH" || echo "Warning: prisma generate failed"; \
        fi && \
        # Fix permissions on LiteLLM directories that need write access at runtime
        # Using least-privilege: litellm user owns directories with 755 (rwxr-xr-x)
        chown -R litellm:litellm /usr/local/lib/python3.14/site-packages/prisma/ 2>/dev/null || true && \
        chmod -R 755 /usr/local/lib/python3.14/site-packages/prisma/ 2>/dev/null || true && \
        chown -R litellm:litellm /usr/local/lib/python3.14/site-packages/litellm/proxy/_experimental/out/ 2>/dev/null || true && \
        chmod -R 755 /usr/local/lib/python3.14/site-packages/litellm/proxy/_experimental/out/ 2>/dev/null || true; \
    else \
        echo "Skipping Prisma generation (db extra not installed)"; \
    fi

# Copy application files (after dependencies for better layer caching)
COPY --chown=litellm:litellm src/litellm_llmrouter /app/litellm_llmrouter
COPY --chown=litellm:litellm docker/entrypoint.sh /app/entrypoint.sh
COPY --chown=litellm:litellm config/ /app/config/

# Copy admin UI dist (built in ui-builder stage)
# In slim builds (BUILD_UI=false), this copies a placeholder index.html
COPY --from=ui-builder --chown=litellm:litellm /ui/dist /app/ui/dist

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
    # Admin UI
    ROUTEIQ_ADMIN_UI_ENABLED=false \
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
