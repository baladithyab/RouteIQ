"""
LiteLLM + LLMRouter Startup Script
===================================

This script starts LiteLLM proxy with LLMRouter extensions:
- A2A Gateway routes
- MCP Gateway routes
- Hot reload routes
- Custom routing strategies

Usage:
    python -m litellm_llmrouter.startup --config config.yaml --port 4000
"""

import os
import sys

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def register_routes_with_litellm():
    """Register LLMRouter routes with LiteLLM's FastAPI app."""
    try:
        from litellm.proxy.proxy_server import app
        from litellm_llmrouter.routes import router

        # Add our routes to LiteLLM's app
        app.include_router(router, prefix="")

        print("✅ LLMRouter routes registered with LiteLLM")
        return True
    except ImportError as e:
        print(f"⚠️ Could not register routes: {e}")
        return False


def register_strategies():
    """Register LLMRouter strategies with LiteLLM."""
    try:
        from litellm_llmrouter.strategies import register_llmrouter_strategies

        strategies = register_llmrouter_strategies()
        print(f"✅ Registered {len(strategies)} LLMRouter strategies")
        return strategies
    except ImportError as e:
        print(f"⚠️ Could not register strategies: {e}")
        return []


def start_config_sync_if_enabled():
    """Start background config sync if enabled."""
    if os.getenv("CONFIG_HOT_RELOAD", "false").lower() == "true":
        try:
            from litellm_llmrouter.config_sync import start_config_sync

            start_config_sync()
            print("✅ Config sync started")
        except ImportError as e:
            print(f"⚠️ Could not start config sync: {e}")


def init_observability_if_enabled():
    """Initialize OpenTelemetry observability if enabled."""
    if os.getenv("OTEL_ENABLED", "true").lower() == "true":
        try:
            from litellm_llmrouter.observability import init_observability

            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            service_name = os.getenv("OTEL_SERVICE_NAME", "litellm-gateway")
            
            init_observability(
                service_name=service_name,
                otlp_endpoint=otlp_endpoint,
                enable_traces=True,
                enable_logs=True,
                enable_metrics=True,
            )
            print(f"✅ OpenTelemetry observability initialized (service: {service_name})")
        except ImportError as e:
            print(f"⚠️ Could not initialize observability: {e}")
        except Exception as e:
            print(f"⚠️ Observability initialization failed: {e}")


def main():
    """Main entry point for LiteLLM + LLMRouter."""
    import argparse

    parser = argparse.ArgumentParser(description="LiteLLM + LLMRouter Gateway")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--port", type=int, default=4000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args, unknown = parser.parse_known_args()

    print("🚀 Starting LiteLLM + LLMRouter Gateway...")

    # Initialize observability first (so it's available for other components)
    init_observability_if_enabled()

    # Register strategies
    register_strategies()

    # Start config sync if enabled
    start_config_sync_if_enabled()

    # Build litellm command
    litellm_args = ["litellm"]

    if args.config:
        litellm_args.extend(["--config", args.config])

    litellm_args.extend(["--port", str(args.port)])
    litellm_args.extend(["--host", args.host])

    # Add any additional args
    litellm_args.extend(unknown)

    # Register routes after LiteLLM starts (via callback)
    os.environ["LITELLM_LLMROUTER_REGISTER_ROUTES"] = "true"

    # Execute litellm
    print(f"   Running: {' '.join(litellm_args)}")
    os.execvp("litellm", litellm_args)


if __name__ == "__main__":
    main()
