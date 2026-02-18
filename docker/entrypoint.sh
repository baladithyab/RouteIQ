#!/bin/bash
set -e

# RouteIQ Gateway Production Entrypoint
# Handles config loading, OTEL setup, A2A/MCP gateway, and startup

echo "🚀 Starting RouteIQ Gateway..."
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
# Database & Prisma Setup (HA-Safe)
# =============================================================================
# ⚠️  IMPORTANT: Database migrations are DISABLED by default for HA safety.
#     Running `prisma db push --accept-data-loss` per-replica in a multi-node
#     setup can cause data loss and race conditions.
#
#     To run migrations, set LITELLM_RUN_DB_MIGRATIONS=true on ONE replica ONLY
#     (e.g., via a separate init job, or on a single designated leader).
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
            if [ "${DB_MIGRATION_SKIP:-false}" = "true" ]; then
                echo "   ℹ️  DB_MIGRATION_SKIP=true - skipping migrations (replica mode)"
            else
                echo "   ⚠️  LITELLM_RUN_DB_MIGRATIONS=true - running migrations (use with caution in HA!)"

                MAX_RETRIES=${DB_MIGRATION_MAX_RETRIES:-10}
                BASE_DELAY=${DB_MIGRATION_RETRY_DELAY:-2}
                MAX_DELAY=${DB_MIGRATION_MAX_DELAY:-30}
                MIGRATION_SUCCESS=false

                # Exponential backoff with cap: 2, 4, 8, 16, 30, 30, ...
                # Optimized for serverless databases (Aurora Serverless v2)
                # that may need 15-30s to scale from zero.
                for i in $(seq 1 $MAX_RETRIES); do
                    echo "   Attempting database migration (attempt $i/$MAX_RETRIES)..."
                    if prisma db push --schema="$SCHEMA_PATH" --accept-data-loss 2>&1; then
                        echo "   ✅ Database migration successful."
                        MIGRATION_SUCCESS=true
                        break
                    fi
                    if [ "$i" -eq "$MAX_RETRIES" ]; then
                        echo "   ❌ Database migration failed after $MAX_RETRIES attempts."
                        exit 1
                    fi
                    # Exponential backoff: base * 2^(attempt-1), capped at MAX_DELAY
                    SLEEP_TIME=$((BASE_DELAY * (1 << (i - 1))))
                    if [ "$SLEEP_TIME" -gt "$MAX_DELAY" ]; then
                        SLEEP_TIME=$MAX_DELAY
                    fi
                    echo "   Migration failed. Retrying in ${SLEEP_TIME}s..."
                    sleep $SLEEP_TIME
                done
            fi
        else
            echo "   ℹ️  Skipping migrations (LITELLM_RUN_DB_MIGRATIONS not set)"
            echo "      For HA deployments, run migrations via a separate init job or leader election"
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
# Start LiteLLM Proxy via LLMRouter Startup Module
# =============================================================================
# We use our startup module instead of `litellm` CLI directly because:
# 1. The routing_strategy_patch MUST be imported BEFORE any Router is created
# 2. Using `exec litellm` would spawn a new process without our patches
# 3. The startup module runs uvicorn in-process, preserving monkey-patches
# =============================================================================

echo "🌐 Starting RouteIQ Gateway via startup module..."
echo "   ✅ llmrouter-* routing strategies will be available"

# Use opentelemetry-instrument if OTEL is configured for additional auto-instrumentation
if [ -n "$OTEL_EXPORTER_OTLP_ENDPOINT" ] && command -v opentelemetry-instrument &> /dev/null; then
    echo "   With OpenTelemetry instrumentation"
    exec opentelemetry-instrument python -m litellm_llmrouter.startup "$@"
else
    exec python -m litellm_llmrouter.startup "$@"
fi
