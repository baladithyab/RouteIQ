#!/bin/bash
set -e

# LiteLLM + LLMRouter Entrypoint Script

echo "🚀 Starting LiteLLM + LLMRouter Gateway..."
echo "   Config: ${LITELLM_CONFIG_PATH:-/app/config/config.yaml}"

# =============================================================================
# Optional: Configuration Loading from S3/GCS
# =============================================================================

if [ -n "$CONFIG_S3_BUCKET" ] && [ -n "$CONFIG_S3_KEY" ]; then
    echo "📥 Loading config from S3: s3://${CONFIG_S3_BUCKET}/${CONFIG_S3_KEY}"
    python3 -c "
from litellm_llmrouter.config_loader import download_config_from_s3
download_config_from_s3('${CONFIG_S3_BUCKET}', '${CONFIG_S3_KEY}', '/app/config/config.yaml')
" || echo "⚠️ Failed to load config from S3, using local config"
fi

# =============================================================================
# Start LiteLLM Proxy
# =============================================================================

echo "🌐 Starting LiteLLM Proxy Server..."

exec litellm "$@"

