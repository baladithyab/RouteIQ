#!/bin/bash
set -e

# LiteLLM + LLMRouter Local Dev Entrypoint

echo "🚀 Starting LiteLLM + LLMRouter Gateway (Local Dev)..."
echo "   Config: ${LITELLM_CONFIG_PATH:-/app/config/config.yaml}"

# =============================================================================
# Database & Prisma Setup
# =============================================================================

if [ -n "$DATABASE_URL" ]; then
    echo "🗄️  Database configured, generating Prisma client..."

    # Find litellm's schema.prisma location
    SCHEMA_PATH=$(python -c "import litellm; import os; print(os.path.join(os.path.dirname(litellm.__file__), 'proxy', 'schema.prisma'))" 2>/dev/null || echo "")

    if [ -n "$SCHEMA_PATH" ] && [ -f "$SCHEMA_PATH" ]; then
        echo "   Schema: $SCHEMA_PATH"
        prisma generate --schema="$SCHEMA_PATH" 2>&1 || echo "   Warning: prisma generate failed, continuing..."
        prisma db push --schema="$SCHEMA_PATH" --accept-data-loss 2>&1 || echo "   Warning: prisma db push failed, continuing..."
    else
        echo "   Warning: Could not find Prisma schema, skipping..."
    fi
fi

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
    # Set protocol based on port (4318 = HTTP, 4317 = gRPC)
    if echo "$OTEL_EXPORTER_OTLP_ENDPOINT" | grep -q ":4318"; then
        export OTEL_EXPORTER_OTLP_PROTOCOL="${OTEL_EXPORTER_OTLP_PROTOCOL:-http/protobuf}"
    else
        export OTEL_EXPORTER_OTLP_PROTOCOL="${OTEL_EXPORTER_OTLP_PROTOCOL:-grpc}"
    fi
    echo "   Protocol: $OTEL_EXPORTER_OTLP_PROTOCOL"
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
