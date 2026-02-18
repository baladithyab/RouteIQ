"""
RouteIQ Gateway Startup Script
===============================

This script starts the RouteIQ Gateway (LiteLLM proxy with LLMRouter extensions):
- A2A Gateway convenience routes (/a2a/agents - wraps LiteLLM's global_agent_registry)
- MCP Gateway routes
- Hot reload routes
- Custom routing strategies (llmrouter-*)

Note: The main A2A functionality is provided by LiteLLM's built-in endpoints:
- POST /v1/agents - Create agent (DB-backed)
- GET /v1/agents - List agents (DB-backed)
- DELETE /v1/agents/{agent_id} - Delete agent (DB-backed)
- POST /a2a/{agent_id} - Invoke agent (A2A JSON-RPC protocol)

The key difference from the standard LiteLLM startup is that we run
the proxy IN-PROCESS using uvicorn, not via os.execvp. This ensures
our monkey-patches to LiteLLM's Router class persist.

Usage:
    python -m litellm_llmrouter.startup --config config.yaml --port 4000

Docker Usage:
    # These are the same args that litellm CLI accepts
    python -m litellm_llmrouter.startup --config /app/config/config.yaml --port 4000
"""

import asyncio as _asyncio
import logging
import os
import signal
import sys

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Configure module logger
logger = logging.getLogger(__name__)


def register_router_decision_callback():
    """
    Register the router decision callback for TG4.1 telemetry.

    This ensures that router decision span attributes (router.strategy, etc.)
    are emitted for all routing decisions, regardless of which LiteLLM
    routing strategy is used.
    """
    if os.getenv("LLMROUTER_ROUTER_CALLBACK_ENABLED", "true").lower() != "true":
        return None

    try:
        from litellm_llmrouter.router_decision_callback import (
            register_router_decision_callback as do_register,
        )

        callback = do_register()
        if callback:
            print("✅ Router decision callback registered (TG4.1 telemetry)")
        return callback
    except ImportError as e:
        logger.debug(f"Could not register router decision callback: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to register router decision callback: {e}")
        return None


def run_leader_migrations_if_enabled():
    """Run DB migrations via leader election if ROUTEIQ_LEADER_MIGRATIONS=true.

    Initialises leader election, checks leadership, and delegates to
    :func:`litellm_llmrouter.migrations.run_migrations_if_leader`.
    This is a no-op when the feature is disabled.
    """
    if os.getenv("ROUTEIQ_LEADER_MIGRATIONS", "false").lower() not in (
        "true",
        "1",
        "yes",
    ):
        return

    try:
        from litellm_llmrouter.leader_election import initialize_leader_election
        from litellm_llmrouter.migrations import run_migrations_if_leader

        async def _do_migrations():
            election = await initialize_leader_election()
            is_leader = election.is_leader if election else True
            success = await run_migrations_if_leader(is_leader)
            if success:
                print("✅ Leader migrations completed")
            else:
                print("⚠️ Leader migrations failed (continuing startup)")

        try:
            _asyncio.get_event_loop().run_until_complete(_do_migrations())
        except RuntimeError:
            _asyncio.run(_do_migrations())
    except ImportError as e:
        logger.debug(f"Could not run leader migrations: {e}")
    except Exception as e:
        logger.warning(f"Leader migrations failed: {e}")
        print(f"⚠️ Leader migrations failed: {e}")


def register_strategies():
    """Log available LLMRouter strategies.

    Strategies are activated at request time via the monkey-patch in
    ``routing_strategy_patch.py``, not through a runtime registry.
    This helper calls :func:`register_llmrouter_strategies` to enumerate
    and log the available strategy names at startup.
    """
    try:
        from litellm_llmrouter.strategies import register_llmrouter_strategies

        strategies = register_llmrouter_strategies()
        print(
            f"✅ {len(strategies)} LLMRouter strategies available (activated via routing_strategy_patch)"
        )
        return strategies
    except ImportError as e:
        print(f"⚠️ Could not load strategies: {e}")
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
            print(
                f"✅ OpenTelemetry observability initialized (service: {service_name})"
            )
        except ImportError as e:
            print(f"⚠️ Could not initialize observability: {e}")
        except Exception as e:
            # Use logger.exception to capture full stack trace
            logger.exception(f"Observability initialization failed: {e}")
            print(f"⚠️ Observability initialization failed: {e}")


def init_mcp_tracing_if_enabled():
    """Initialize MCP tracing with OTel if MCP gateway is enabled."""
    if os.getenv("MCP_GATEWAY_ENABLED", "false").lower() == "true":
        try:
            from litellm_llmrouter.mcp_tracing import instrument_mcp_gateway

            if instrument_mcp_gateway():
                print("✅ MCP gateway tracing initialized")
            else:
                print("⚠️ MCP tracing not enabled (OTel not available or disabled)")
        except ImportError as e:
            print(f"⚠️ Could not initialize MCP tracing: {e}")
        except Exception as e:
            # Use logger.exception to capture full stack trace
            logger.exception(f"MCP tracing initialization failed: {e}")
            print(f"⚠️ MCP tracing initialization failed: {e}")


def init_a2a_tracing_if_enabled(app):
    """
    Initialize A2A tracing with OTel if A2A gateway is enabled.

    This function:
    1. Instruments the A2A gateway (wraps gateway methods) if A2A_GATEWAY_ENABLED=true
    2. Registers A2A tracing middleware with LiteLLM's FastAPI app (always, for /a2a/* routes)

    The middleware is separate from gateway instrumentation because:
    - The A2A gateway is an optional component we provide
    - LiteLLM's built-in /a2a/* routes are always present when A2A is configured
    - We want to capture spans for LiteLLM's native A2A routes regardless of our gateway
    """
    # Always try to register the A2A middleware for LiteLLM's /a2a/* routes
    # This is independent of our A2A gateway - it instruments LiteLLM's built-in routes
    try:
        from litellm_llmrouter.a2a_tracing import register_a2a_middleware

        if register_a2a_middleware(app):
            print("✅ A2A HTTP tracing middleware registered for /a2a/* routes")
        else:
            print(
                "⚠️ A2A HTTP tracing middleware not registered (OTel not available or disabled)"
            )
    except ImportError as e:
        logger.warning(f"Could not register A2A middleware: {e}")
        print(f"⚠️ Could not register A2A middleware: {e}")
    except Exception as e:
        # Use logger.exception to capture full stack trace for debugging
        logger.exception(f"A2A tracing middleware registration failed: {e}")
        print(f"⚠️ A2A tracing middleware registration failed: {e}")

    # Also instrument our A2A gateway if it's enabled
    if os.getenv("A2A_GATEWAY_ENABLED", "false").lower() == "true":
        try:
            from litellm_llmrouter.a2a_tracing import instrument_a2a_gateway

            if instrument_a2a_gateway():
                print("✅ A2A gateway tracing initialized")
            else:
                print(
                    "⚠️ A2A gateway tracing not enabled (OTel not available or disabled)"
                )
        except ImportError as e:
            print(f"⚠️ Could not initialize A2A gateway tracing: {e}")
        except Exception as e:
            # Use logger.exception to capture full stack trace
            logger.exception(f"A2A gateway tracing initialization failed: {e}")
            print(f"⚠️ A2A gateway tracing initialization failed: {e}")


def run_litellm_proxy_inprocess(config_path: str, host: str, port: int, **kwargs):
    """
    Run LiteLLM proxy in-process using uvicorn.

    This is the preferred method because it preserves our monkey-patches
    to LiteLLM's Router class. Using os.execvp() would replace the process
    and lose all patches.

    Args:
        config_path: Path to the LiteLLM config file
        host: Host to bind to
        port: Port to listen on
        **kwargs: Additional arguments passed to uvicorn
    """
    import uvicorn

    # Set environment variables that LiteLLM expects
    if config_path:
        os.environ["LITELLM_CONFIG_PATH"] = config_path
        # Also set the path that LiteLLM's proxy_server reads
        os.environ["CONFIG_FILE_PATH"] = config_path

    # Import gateway factory AFTER setting env vars
    # The factory applies the patch explicitly before importing litellm
    from litellm_llmrouter.gateway import create_app

    # Create and configure the app using the composition root
    # This applies the patch, adds middleware, and registers routes
    app = create_app(
        apply_patch=True,
        include_admin_routes=True,
        enable_plugins=True,
    )

    # Get LiteLLM's initialize function
    from litellm.proxy.proxy_server import initialize

    # Initialize LiteLLM with the config
    # This is what litellm --config does internally
    import asyncio

    async def init_litellm():
        """Initialize LiteLLM proxy with config."""
        try:
            await initialize(
                config=config_path,
                debug=kwargs.get("debug", False),
            )
            print(f"✅ LiteLLM initialized with config: {config_path}")
        except Exception as e:
            print(f"⚠️ LiteLLM initialization warning: {e}")
            # Continue anyway - some initialization errors are non-fatal

    # Run initialization
    try:
        asyncio.get_event_loop().run_until_complete(init_litellm())
    except RuntimeError:
        # No event loop in current thread
        asyncio.run(init_litellm())

    # Initialize HTTP client pool BEFORE plugins (they may use it)
    async def run_http_pool_startup():
        if hasattr(app.state, "llmrouter_http_pool_startup"):
            await app.state.llmrouter_http_pool_startup()
            print("✅ HTTP client pool initialized")

    try:
        asyncio.get_event_loop().run_until_complete(run_http_pool_startup())
    except RuntimeError:
        asyncio.run(run_http_pool_startup())

    # Initialize A2A tracing AFTER LiteLLM initialization
    init_a2a_tracing_if_enabled(app)

    # Run plugin startup if enabled
    async def run_plugin_startup():
        if hasattr(app.state, "llmrouter_plugin_startup"):
            await app.state.llmrouter_plugin_startup()

    try:
        asyncio.get_event_loop().run_until_complete(run_plugin_startup())
    except RuntimeError:
        asyncio.run(run_plugin_startup())

    print("✅ LLMRouter routes registered with LiteLLM")
    print("   ├── /_health/* (K8s probes, unauthenticated)")
    print("   ├── /a2a/agents (convenience wrapper, auth-protected)")
    print("   ├── /llmrouter/mcp/* (MCP gateway, auth-protected)")
    print("   └── /router/*, /config/* (hot reload, auth-protected)")
    print(
        "   Note: LiteLLM provides /v1/agents (DB-backed) and /a2a/{agent_id} (A2A protocol)"
    )

    # Register SIGTERM handler to trigger graceful drain before uvicorn shuts down.
    # Without this, uvicorn's default SIGTERM handler stops accepting connections
    # but doesn't notify the DrainManager, so readiness probes don't go unhealthy
    # and in-flight requests may be dropped without waiting.
    _original_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm_handler(signum, frame):
        """Trigger DrainManager drain on SIGTERM, then delegate to uvicorn."""
        logger.info("SIGTERM received, starting graceful drain...")
        if hasattr(app.state, "graceful_shutdown"):
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(app.state.graceful_shutdown())
            except RuntimeError:
                asyncio.run(app.state.graceful_shutdown())
        # Delegate to uvicorn's original signal handler
        if callable(_original_sigterm):
            _original_sigterm(signum, frame)
        elif _original_sigterm == signal.SIG_DFL:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Configure uvicorn
    uvicorn_config = {
        "app": app,
        "host": host,
        "port": port,
        "log_level": kwargs.get("log_level", "info"),
        "access_log": kwargs.get("access_log", True),
        "workers": kwargs.get("workers", 1),  # Use 1 worker to preserve patches
    }

    # Add SSL config if provided
    if kwargs.get("ssl_keyfile"):
        uvicorn_config["ssl_keyfile"] = kwargs["ssl_keyfile"]
    if kwargs.get("ssl_certfile"):
        uvicorn_config["ssl_certfile"] = kwargs["ssl_certfile"]

    print(f"🚀 Starting LiteLLM proxy on {host}:{port}")
    uvicorn.run(**uvicorn_config)


def main():
    """
    Main entry point for RouteIQ Gateway.

    CLI Args Supported (compatible with litellm CLI):
        --config PATH    Path to config file (also: -c, --config_file)
        --port PORT      Port to listen on (default: 4000)
        --host HOST      Host to bind to (default: 0.0.0.0)
        --debug          Enable debug mode
        --workers N      Number of uvicorn workers (1 recommended for patches)
        --ssl-keyfile    SSL key file path
        --ssl-certfile   SSL certificate file path

    Environment Variables:
        LITELLM_CONFIG_PATH  Default config path if --config not provided
    """
    import argparse

    # Import patch status check - does NOT apply patch
    from litellm_llmrouter import is_patch_applied

    parser = argparse.ArgumentParser(
        description="RouteIQ Gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m litellm_llmrouter.startup --config config.yaml --port 4000
    python -m litellm_llmrouter.startup --config /app/config/config.yaml --port 4000 --host 0.0.0.0
        """,
    )
    # Config file - support multiple aliases for compat with litellm CLI
    parser.add_argument(
        "--config",
        "-c",
        "--config_file",
        type=str,
        dest="config",
        default=os.getenv("LITELLM_CONFIG_PATH"),
        help="Path to config file (default: $LITELLM_CONFIG_PATH)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=int(os.getenv("LITELLM_PORT", "4000")),
        help="Port to listen on (default: 4000)",
    )
    parser.add_argument(
        "--host",
        "-H",
        type=str,
        default=os.getenv("LITELLM_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=os.getenv("LITELLM_DEBUG", "false").lower() == "true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of workers (1 recommended to preserve patches)",
    )
    parser.add_argument("--ssl-keyfile", type=str, help="SSL key file path")
    parser.add_argument("--ssl-certfile", type=str, help="SSL certificate file path")

    # Parse known args and pass through unknown args (for forward compatibility)
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"   Note: Ignoring unknown args: {unknown}")

    print("🚀 Starting RouteIQ Gateway...")
    print(
        f"   Patch status: {'✅ applied' if is_patch_applied() else '⏳ pending (will be applied at startup)'}"
    )
    print(f"   Config: {args.config or '(none)'}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")

    # Validate environment variables early (advisory only — never prevents startup)
    from litellm_llmrouter.env_validation import validate_environment

    env_result = validate_environment()
    if env_result.errors:
        logger.error("Environment validation found %d error(s)", len(env_result.errors))
    if env_result.warnings:
        logger.warning(
            "Environment validation found %d warning(s)", len(env_result.warnings)
        )

    # Initialize observability first (so it's available for other components)
    init_observability_if_enabled()

    # Run leader-election-based migrations if opted in
    run_leader_migrations_if_enabled()

    # Register router decision callback for TG4.1 telemetry
    register_router_decision_callback()

    init_mcp_tracing_if_enabled()

    # Register strategies
    register_strategies()

    # Start config sync if enabled
    start_config_sync_if_enabled()

    # Run LiteLLM proxy IN-PROCESS using the gateway factory
    # This is critical - using os.execvp would lose our patches!
    run_litellm_proxy_inprocess(
        config_path=args.config,
        host=args.host,
        port=args.port,
        debug=args.debug,
        workers=args.workers,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
