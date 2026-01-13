# syntax=docker/dockerfile:1.4
# =============================================================================
# LiteLLM + LLMRouter Production Dockerfile
# =============================================================================
# Multi-stage build for minimal, secure production images
# Uses uv for fast, reproducible Python package management
# Supports: linux/amd64, linux/arm64
#
# Security features:
# - Non-root user execution
# - Minimal base image (slim variant)
# - No shell in final image (optional)
# - Read-only filesystem compatible
# - Health checks for orchestrators
#
# Build args:
#   PYTHON_VERSION  - Python version (default: 3.12)
#   UV_VERSION      - uv package manager version (default: 0.9)
#   LITELLM_VERSION - LiteLLM version (default: 1.80.15)
#   LLMROUTER_COMMIT - LLMRouter git commit hash
# =============================================================================

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.9
ARG LITELLM_VERSION=1.80.15
# LLMRouter pinned to specific commit for reproducibility
ARG LLMROUTER_COMMIT=7890cd9d36951a3c73fec83619321f3704a7aaa8

# ==============================================================================
# Stage 1: Builder - Clone repos and build wheels with uv
# ==============================================================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm AS builder

ARG LLMROUTER_COMMIT
ARG LITELLM_VERSION

WORKDIR /build

# Install git for cloning LLMRouter
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/*

# Clone and build LLMRouter wheel from GitHub at pinned commit
RUN git clone --depth 1 https://github.com/ulab-uiuc/LLMRouter.git /tmp/LLMRouter && \
    cd /tmp/LLMRouter && \
    git fetch --depth 1 origin ${LLMROUTER_COMMIT} && \
    git checkout ${LLMROUTER_COMMIT} && \
    uv build --wheel --out-dir /wheels && \
    rm -rf /tmp/LLMRouter/.git

# ==============================================================================
# Stage 2: Runtime - Production image with uv and OTEL support
# ==============================================================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm-slim AS runtime

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

# Security: Create non-root user with specific UID/GID for consistency
RUN groupadd -r -g 1000 litellm && \
    useradd -r -u 1000 -g litellm -d /app -s /sbin/nologin litellm

WORKDIR /app

# Install minimal runtime dependencies in single layer
# Note: build-essential needed for uvloop compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/* \
    && rm -rf /root/.cache

# Copy and install wheels from builder
COPY --from=builder /wheels /wheels
RUN uv pip install --system /wheels/*.whl || true

# Install LiteLLM with proxy extras
RUN uv pip install --system "litellm[proxy]==${LITELLM_VERSION}"

# Install OpenTelemetry for distributed tracing (single layer)
RUN uv pip install --system \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-exporter-otlp-proto-http \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-httpx \
    opentelemetry-instrumentation-logging \
    opentelemetry-instrumentation-requests \
    && rm -rf /root/.cache/uv

# Install additional runtime dependencies
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt || true && \
    rm -rf /wheels /tmp/* /root/.cache

# Copy application files
COPY --chown=litellm:litellm src/litellm_llmrouter /app/litellm_llmrouter
COPY --chown=litellm:litellm docker/entrypoint.sh /app/entrypoint.sh
COPY --chown=litellm:litellm config/ /app/config/

# Create directories with proper permissions
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
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:4000/health/liveliness || exit 1

EXPOSE 4000

# Security: Run as non-root user
USER litellm

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["--config", "/app/config/config.yaml", "--port", "4000"]
