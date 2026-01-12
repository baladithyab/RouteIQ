#!/bin/bash
set -e

# LiteLLM + LLMRouter Entrypoint Script
# Handles initialization, model downloading, and startup

echo "🚀 Starting LiteLLM + LLMRouter Gateway..."

# =============================================================================
# Configuration Loading
# =============================================================================

# Check for S3/GCS config if specified
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
# Model Pre-loading (if S3/GCS paths specified)
# =============================================================================

if [ -n "$LLMROUTER_MODEL_S3_BUCKET" ] && [ -n "$LLMROUTER_MODEL_S3_KEY" ]; then
    echo "📥 Downloading LLMRouter model from S3..."
    python3 -c "
from litellm_llmrouter.config_loader import download_model_from_s3
download_model_from_s3('${LLMROUTER_MODEL_S3_BUCKET}', '${LLMROUTER_MODEL_S3_KEY}', '/app/models/')
" || echo "⚠️ Failed to download model from S3"
fi

# =============================================================================
# Custom Router Loading
# =============================================================================

if [ -n "$CUSTOM_ROUTER_S3_BUCKET" ] && [ -n "$CUSTOM_ROUTER_S3_KEY" ]; then
    echo "📥 Downloading custom router from S3..."
    python3 -c "
from litellm_llmrouter.config_loader import download_custom_router_from_s3
download_custom_router_from_s3('${CUSTOM_ROUTER_S3_BUCKET}', '${CUSTOM_ROUTER_S3_KEY}', '/app/custom_routers/')
" || echo "⚠️ Failed to download custom router from S3"
fi

# =============================================================================
# Initialize LLMRouter Integration
# =============================================================================

echo "🔧 Initializing LLMRouter routing strategies..."
python3 -c "
from litellm_llmrouter import register_llmrouter_strategies
register_llmrouter_strategies()
print('✅ LLMRouter strategies registered successfully')
" || echo "⚠️ Failed to register LLMRouter strategies (will use defaults)"

# =============================================================================
# Start LiteLLM Proxy
# =============================================================================

echo "🌐 Starting LiteLLM Proxy Server..."
echo "   Config: ${LITELLM_CONFIG_PATH:-/app/config/config.yaml}"
echo "   Models: ${LLMROUTER_MODELS_PATH:-/app/models}"

# Handle different startup modes
if [ "$SEPARATE_HEALTH_APP" = "1" ]; then
    echo "📊 Starting with separate health app (supervisord)"
    export LITELLM_ARGS="$@"
    exec supervisord -c /etc/supervisord.conf
elif [ "$USE_DDTRACE" = "true" ]; then
    echo "🔍 Starting with Datadog tracing"
    export DD_TRACE_OPENAI_ENABLED="False"
    exec ddtrace-run litellm "$@"
else
    exec litellm "$@"
fi

