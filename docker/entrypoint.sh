#!/bin/bash
set -e

# LiteLLM + LLMRouter Production Entrypoint
# Handles config loading, OTEL setup, A2A/MCP gateway, and startup

echo "🚀 Starting LiteLLM + LLMRouter Gateway..."
echo "   Config: ${LITELLM_CONFIG_PATH:-/app/config/config.yaml}"

# =============================================================================
# Feature Detection
# =============================================================================

# A2A Gateway (Agent-to-Agent protocol)
if [ "${A2A_GATEWAY_ENABLED:-false}" = "true" ]; then
    echo "🤖 A2A Gateway: ENABLED"
    echo "   Agents can be added via /v1/agents API or UI"
fi

# MCP Gateway (Model Context Protocol)
if [ "${MCP_GATEWAY_ENABLED:-false}" = "true" ]; then
    echo "🔧 MCP Gateway: ENABLED"
    echo "   MCP servers can be added via API or config"
fi

# Hot Reload
if [ "${CONFIG_HOT_RELOAD:-false}" = "true" ]; then
    echo "🔄 Hot Reload: ENABLED (sync interval: ${CONFIG_SYNC_INTERVAL:-60}s)"
fi

# =============================================================================
# OpenTelemetry Configuration
# =============================================================================

if [ -n "$OTEL_EXPORTER_OTLP_ENDPOINT" ]; then
    echo "📡 OpenTelemetry enabled"
    echo "   Endpoint: $OTEL_EXPORTER_OTLP_ENDPOINT"
    echo "   Service:  ${OTEL_SERVICE_NAME:-litellm-gateway}"

    # Set OTEL exporters to otlp if endpoint is configured
    export OTEL_TRACES_EXPORTER="${OTEL_TRACES_EXPORTER:-otlp}"
    export OTEL_METRICS_EXPORTER="${OTEL_METRICS_EXPORTER:-otlp}"
    export OTEL_LOGS_EXPORTER="${OTEL_LOGS_EXPORTER:-otlp}"

    # Configure resource attributes
    export OTEL_RESOURCE_ATTRIBUTES="${OTEL_RESOURCE_ATTRIBUTES:-service.name=${OTEL_SERVICE_NAME:-litellm-gateway}}"
fi

# =============================================================================
# Configuration Loading from S3/GCS
# =============================================================================

if [ -n "$CONFIG_S3_BUCKET" ] && [ -n "$CONFIG_S3_KEY" ]; then
    echo "📥 Loading config from S3: s3://${CONFIG_S3_BUCKET}/${CONFIG_S3_KEY}"
    python3 -c "
from litellm_llmrouter.config_loader import download_config_from_s3
download_config_from_s3('${CONFIG_S3_BUCKET}', '${CONFIG_S3_KEY}', '/app/config/config.yaml')
" || echo "⚠️ Failed to load config from S3, using local config"
fi

if [ -n "$CONFIG_GCS_BUCKET" ] && [ -n "$CONFIG_GCS_KEY" ]; then
    echo "📥 Loading config from GCS: gs://${CONFIG_GCS_BUCKET}/${CONFIG_GCS_KEY}"
    python3 -c "
import asyncio
from litellm_llmrouter.config_loader import download_config_from_gcs
asyncio.run(download_config_from_gcs('${CONFIG_GCS_BUCKET}', '${CONFIG_GCS_KEY}', '/app/config/config.yaml'))
" || echo "⚠️ Failed to load config from GCS, using local config"
fi

# =============================================================================
# Model Pre-loading from S3
# =============================================================================

if [ -n "$LLMROUTER_MODEL_S3_BUCKET" ] && [ -n "$LLMROUTER_MODEL_S3_KEY" ]; then
    echo "📥 Downloading LLMRouter model from S3..."
    python3 -c "
from litellm_llmrouter.config_loader import download_model_from_s3
download_model_from_s3('${LLMROUTER_MODEL_S3_BUCKET}', '${LLMROUTER_MODEL_S3_KEY}', '/app/models/')
" || echo "⚠️ Failed to download model from S3"
fi

# =============================================================================
# Start Background Config Sync (if enabled)
# =============================================================================

if [ "${CONFIG_HOT_RELOAD:-false}" = "true" ] && [ -n "$CONFIG_S3_BUCKET" ]; then
    echo "🔄 Starting background config sync..."
    python3 -c "
from litellm_llmrouter.config_sync import start_config_sync
start_config_sync()
" &
fi

# =============================================================================
# Start LiteLLM Proxy
# =============================================================================

echo "🌐 Starting LiteLLM Proxy Server..."

# Use opentelemetry-instrument if OTEL is configured
if [ -n "$OTEL_EXPORTER_OTLP_ENDPOINT" ] && command -v opentelemetry-instrument &> /dev/null; then
    echo "   With OpenTelemetry instrumentation"
    exec opentelemetry-instrument litellm "$@"
else
    exec litellm "$@"
fi
