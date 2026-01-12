# syntax=docker/dockerfile:1.4
# LiteLLM + LLMRouter Multi-Architecture Dockerfile
# Supports: linux/amd64, linux/arm64

ARG PYTHON_VERSION=3.11
ARG LITELLM_VERSION=1.72.6

# ==============================================================================
# Single Stage Build (simpler for local development)
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/baladithyab/litellm-llm-router"
LABEL org.opencontainers.image.description="LiteLLM + LLMRouter: Intelligent LLM Gateway with ML-Powered Routing"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install LiteLLM with proxy extras
ARG LITELLM_VERSION
RUN pip install --no-cache-dir "litellm[proxy]==${LITELLM_VERSION}"

# Install additional dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

# Copy and install LLMRouter
COPY reference/LLMRouter /tmp/LLMRouter
RUN cd /tmp/LLMRouter && pip install --no-cache-dir . || echo "LLMRouter install skipped"

# Copy custom routing integration
COPY src/litellm_llmrouter /app/litellm_llmrouter

# Copy entrypoint and config
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY config/ /app/config/

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/custom_routers

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LITELLM_CONFIG_PATH=/app/config/config.yaml \
    LLMROUTER_MODELS_PATH=/app/models \
    LLMROUTER_CUSTOM_ROUTERS_PATH=/app/custom_routers \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:4000/health/liveliness || exit 1

# Expose LiteLLM proxy port
EXPOSE 4000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--config", "/app/config/config.yaml", "--port", "4000"]

