#!/bin/bash
set -e

# LiteLLM + LLMRouter Local Dev Entrypoint

echo "🚀 Starting LiteLLM + LLMRouter Gateway (Local Dev)..."
echo "   Config: ${LITELLM_CONFIG_PATH:-/app/config/config.yaml}"

# =============================================================================
# OpenTelemetry Configuration
# =============================================================================

if [ -n "$OTEL_EXPORTER_OTLP_ENDPOINT" ]; then
    echo "📡 OpenTelemetry enabled"
    echo "   Endpoint: $OTEL_EXPORTER_OTLP_ENDPOINT"
    echo "   Service:  ${OTEL_SERVICE_NAME:-litellm-gateway}"

    export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-litellm-gateway}"
    export OTEL_TRACES_EXPORTER="${OTEL_TRACES_EXPORTER:-otlp}"
    export OTEL_METRICS_EXPORTER="${OTEL_METRICS_EXPORTER:-none}"
    export OTEL_LOGS_EXPORTER="${OTEL_LOGS_EXPORTER:-none}"
    export OTEL_EXPORTER_OTLP_INSECURE="${OTEL_EXPORTER_OTLP_INSECURE:-true}"
fi

# =============================================================================
# Start LiteLLM Proxy
# =============================================================================

echo "🌐 Starting LiteLLM Proxy Server..."

# Use opentelemetry-instrument if OTEL endpoint is configured
if [ -n "$OTEL_EXPORTER_OTLP_ENDPOINT" ] && command -v opentelemetry-instrument &> /dev/null; then
    echo "   With OpenTelemetry auto-instrumentation"
    exec opentelemetry-instrument litellm "$@"
else
    exec litellm "$@"
fi
