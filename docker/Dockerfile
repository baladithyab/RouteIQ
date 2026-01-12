# syntax=docker/dockerfile:1.4
# LiteLLM + LLMRouter Production Dockerfile
# Supports: linux/amd64, linux/arm64
# Uses pinned GitHub versions for reproducible builds

ARG PYTHON_VERSION=3.11
ARG LITELLM_VERSION=1.80.15.rc2
# LLMRouter pinned to specific commit
ARG LLMROUTER_COMMIT=7890cd9d36951a3c73fec83619321f3704a7aaa8

# ==============================================================================
# Stage 1: Builder - Clone repos and build wheels
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG LLMROUTER_COMMIT
ARG LITELLM_VERSION

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Clone and install LLMRouter from GitHub at pinned commit
RUN git clone https://github.com/MasterGodzworng/LLMRouter.git /tmp/LLMRouter && \
    cd /tmp/LLMRouter && \
    git checkout ${LLMROUTER_COMMIT} && \
    pip wheel --no-deps --wheel-dir /wheels .

# Build LiteLLM wheel
RUN pip wheel --no-deps --wheel-dir /wheels "litellm[proxy]==${LITELLM_VERSION}"

# ==============================================================================
# Stage 2: Runtime - Production image with OTEL support
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG LITELLM_VERSION
ARG LLMROUTER_COMMIT

LABEL org.opencontainers.image.source="https://github.com/baladithyab/litellm-llm-router"
LABEL org.opencontainers.image.description="LiteLLM + LLMRouter: Intelligent LLM Gateway with ML-Powered Routing"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="${LITELLM_VERSION}"
LABEL llmrouter.commit="${LLMROUTER_COMMIT}"

# Create non-root user
RUN groupadd -r litellm && useradd -r -g litellm litellm

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy and install wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl || true

# Install LiteLLM (ensure it's installed even if wheel failed)
RUN pip install --no-cache-dir "litellm[proxy]==${LITELLM_VERSION}"

# Install OpenTelemetry for distributed tracing
RUN pip install --no-cache-dir \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-exporter-otlp-proto-http \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-httpx \
    opentelemetry-instrumentation-logging \
    opentelemetry-instrumentation-requests

# Install additional runtime dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true && \
    rm -rf /wheels /tmp/*

# Copy custom routing integration
COPY src/litellm_llmrouter /app/litellm_llmrouter

# Copy entrypoint and config
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY config/ /app/config/

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/custom_routers && \
    chown -R litellm:litellm /app && \
    chmod +x /app/entrypoint.sh

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LITELLM_CONFIG_PATH=/app/config/config.yaml \
    LLMROUTER_MODELS_PATH=/app/models \
    LLMROUTER_CUSTOM_ROUTERS_PATH=/app/custom_routers \
    # OTEL defaults (override at runtime)
    OTEL_SERVICE_NAME=litellm-gateway \
    OTEL_TRACES_EXPORTER=none \
    OTEL_METRICS_EXPORTER=none \
    OTEL_LOGS_EXPORTER=none

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:4000/health/liveliness || exit 1

EXPOSE 4000

USER litellm

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--config", "/app/config/config.yaml", "--port", "4000"]

