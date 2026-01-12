# syntax=docker/dockerfile:1.4
# LiteLLM + LLMRouter Multi-Architecture Dockerfile
# Supports: linux/amd64, linux/arm64

ARG PYTHON_VERSION=3.11
ARG LITELLM_VERSION=1.56.5

# ==============================================================================
# Stage 1: Builder - Install dependencies and build packages
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy LLMRouter source and install
COPY reference/LLMRouter /tmp/LLMRouter
RUN cd /tmp/LLMRouter && pip install --no-cache-dir -e . && \
    pip wheel --no-deps --wheel-dir /wheels .

# Install LiteLLM and dependencies
ARG LITELLM_VERSION
RUN pip install --no-cache-dir litellm[proxy]==${LITELLM_VERSION} && \
    pip wheel --no-deps --wheel-dir /wheels litellm[proxy]==${LITELLM_VERSION}

# Install additional dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip wheel --no-deps --wheel-dir /wheels -r /tmp/requirements.txt 2>/dev/null || true

# ==============================================================================
# Stage 2: Runtime - Minimal production image
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

LABEL org.opencontainers.image.source="https://github.com/baladithyab/litellm-llm-router"
LABEL org.opencontainers.image.description="LiteLLM + LLMRouter: Intelligent LLM Gateway with ML-Powered Routing"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user
RUN groupadd -r litellm && useradd -r -g litellm litellm

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/*.whl 2>/dev/null || true

# Install LiteLLM and LLMRouter directly (fallback)
ARG LITELLM_VERSION
RUN pip install --no-cache-dir litellm[proxy]==${LITELLM_VERSION}

# Copy LLMRouter source and install
COPY reference/LLMRouter /tmp/LLMRouter
RUN cd /tmp/LLMRouter && pip install --no-cache-dir -e .

# Copy custom routing integration
COPY src/litellm_llmrouter /app/litellm_llmrouter

# Copy entrypoint and config
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY config/ /app/config/

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/custom_routers && \
    chown -R litellm:litellm /app

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LITELLM_CONFIG_PATH=/app/config/config.yaml \
    LLMROUTER_MODELS_PATH=/app/models \
    LLMROUTER_CUSTOM_ROUTERS_PATH=/app/custom_routers

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

# Expose LiteLLM proxy port
EXPOSE 4000

# Switch to non-root user
USER litellm

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--config", "/app/config/config.yaml", "--port", "4000"]

