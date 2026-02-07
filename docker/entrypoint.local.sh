#!/bin/bash
set -e

# LiteLLM + LLMRouter Local Dev Entrypoint

echo "🚀 Starting LiteLLM + LLMRouter Gateway (Local Dev)..."
echo "   Config: ${LITELLM_CONFIG_PATH:-/app/config/config.yaml}"

# =============================================================================
# Database & Prisma Setup (HA-Safe)
# =============================================================================
# ⚠️  IMPORTANT: Database migrations are DISABLED by default for HA safety.
#     Running `prisma db push --accept-data-loss` per-replica in a multi-node
#     setup can cause data loss and race conditions.
#
#     To run migrations, set LITELLM_RUN_DB_MIGRATIONS=true on ONE replica ONLY
#     (e.g., via a separate init job, or on a single designated leader).
#
#     For local dev, you can safely enable: LITELLM_RUN_DB_MIGRATIONS=true
# =============================================================================

if [ -n "$DATABASE_URL" ]; then
    echo "🗄️  Database configured"

    # Find litellm's schema.prisma location
    SCHEMA_PATH=$(python -c "import litellm; import os; print(os.path.join(os.path.dirname(litellm.__file__), 'proxy', 'schema.prisma'))" 2>/dev/null || echo "")

    if [ -n "$SCHEMA_PATH" ] && [ -f "$SCHEMA_PATH" ]; then
        echo "   Schema: $SCHEMA_PATH"

        # Always generate Prisma client (safe, no DB changes)
        prisma generate --schema="$SCHEMA_PATH" 2>&1 || echo "   Warning: prisma generate failed, continuing..."

        # Only run migrations if explicitly enabled
        if [ "${LITELLM_RUN_DB_MIGRATIONS:-false}" = "true" ]; then
            echo "   ⚠️  LITELLM_RUN_DB_MIGRATIONS=true - running migrations"
            # Use 'prisma db push' for local dev (fast, schema-sync style)
            # Note: 'prisma migrate deploy' hangs in some container runtimes (e.g., Finch)
            # due to migration engine binary download issues. db push is more reliable.
            prisma db push --schema="$SCHEMA_PATH" --accept-data-loss 2>&1 || \
                echo "   Warning: prisma migration failed, continuing..."
        else
            echo "   ℹ️  Skipping migrations (LITELLM_RUN_DB_MIGRATIONS not set)"
            echo "      Set LITELLM_RUN_DB_MIGRATIONS=true for local dev migrations"
        fi
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
# Start LiteLLM Proxy via LLMRouter Startup Module
# =============================================================================
# We use our startup module instead of `litellm` CLI directly because:
# 1. The routing_strategy_patch MUST be imported BEFORE any Router is created
# 2. Using `exec litellm` would spawn a new process without our patches
# 3. The startup module runs uvicorn in-process, preserving monkey-patches
# =============================================================================

echo "🌐 Starting LiteLLM Proxy Server via LLMRouter startup module..."
echo "   ✅ llmrouter-* routing strategies will be available"

# Use opentelemetry-instrument if OTEL endpoint is configured for auto-instrumentation
if [ -n "$OTEL_EXPORTER_OTLP_ENDPOINT" ] && command -v opentelemetry-instrument &> /dev/null; then
    echo "   With OpenTelemetry auto-instrumentation"
    exec opentelemetry-instrument python -m litellm_llmrouter.startup "$@"
else
    exec python -m litellm_llmrouter.startup "$@"
fi
