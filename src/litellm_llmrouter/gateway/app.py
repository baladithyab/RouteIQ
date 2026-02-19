"""
Gateway Application Factory (Composition Root)
==============================================

This module provides the FastAPI application factory for the LLMRouter gateway.
It explicitly configures all middleware, routers, and patches in a single place.

Load Order:
1. Apply LiteLLM router patch (if enabled, and not using plugin strategy)
2. Get/create FastAPI app
3. Add backpressure middleware (INNERMOST – registered first so outer
   middleware like RequestID and CORS can decorate its 503 responses)
4. Add remaining middleware (RequestID, CORS, Policy, Plugins, etc.)
5. Load and register plugins (deterministically before routes)
6. Register built-in routes
7. Set up plugin lifecycle hooks
8. Set up HTTP client pool lifecycle hooks
9. Set up graceful shutdown hooks

Usage with LiteLLM proxy (in-process):
    from litellm_llmrouter.gateway import create_app

    # This configures the LiteLLM proxy's FastAPI app
    app = create_app()

Usage standalone (without LiteLLM):
    from litellm_llmrouter.gateway import create_standalone_app

    # This creates a standalone FastAPI app with just LLMRouter routes
    app = create_standalone_app()
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI

from ..routing_strategy_patch import is_patch_applied, patch_litellm_router
from ..resilience import (
    add_backpressure_middleware,
    get_drain_manager,
    graceful_shutdown,
)
from ..http_client_pool import (
    startup_http_client_pool,
    shutdown_http_client_pool,
)
from ..policy_engine import add_policy_middleware

logger = logging.getLogger(__name__)


def _use_plugin_strategy() -> bool:
    """
    Check whether the plugin-based routing strategy should be used.

    Reads ``ROUTEIQ_USE_PLUGIN_STRATEGY`` environment variable.
    Defaults to ``True`` (new behaviour).  Set to ``"false"`` to fall back
    to the legacy monkey-patch approach.
    """
    return os.environ.get("ROUTEIQ_USE_PLUGIN_STRATEGY", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def _parse_cors_origins() -> list[str]:
    """Parse ROUTEIQ_CORS_ORIGINS into a list of allowed origins."""
    raw = os.getenv("ROUTEIQ_CORS_ORIGINS", "*")
    return [o.strip() for o in raw.split(",")]


def _parse_cors_credentials() -> bool:
    """Parse ROUTEIQ_CORS_CREDENTIALS flag."""
    return os.getenv("ROUTEIQ_CORS_CREDENTIALS", "false").lower() == "true"


def _apply_patch_safely() -> bool:
    """
    Apply the LiteLLM router patch idempotently.

    Returns:
        True if patch is applied (either now or was already applied)
    """
    if is_patch_applied():
        logger.debug("LiteLLM router patch already applied")
        return True

    result = patch_litellm_router()
    if result:
        logger.info("LiteLLM router patch applied successfully")
    else:
        logger.warning("Failed to apply LiteLLM router patch")

    return result


def _install_plugin_strategy(
    router_instance: Any,
    strategy_name: Optional[str] = None,
) -> bool:
    """
    Install the RouteIQ custom routing strategy on a Router instance.

    Uses ``install_routeiq_strategy`` from ``custom_routing_strategy`` to wire
    the strategy into the Router via ``set_custom_routing_strategy()``.

    On failure, falls back to the legacy monkey-patch approach via
    ``_apply_patch_safely()`` and logs a warning.

    Args:
        router_instance: The LiteLLM Router instance
        strategy_name: Optional ML strategy name (e.g., "llmrouter-knn")

    Returns:
        True if the strategy was installed (either plugin or fallback patch)
    """
    try:
        from ..custom_routing_strategy import install_routeiq_strategy

        install_routeiq_strategy(router_instance, strategy_name)
        logger.info(
            "Plugin routing strategy installed successfully "
            f"(strategy={strategy_name or 'default'})"
        )
        return True
    except Exception as e:
        logger.warning(
            f"Failed to install plugin routing strategy: {e}. "
            f"Falling back to legacy monkey-patch approach."
        )
        return _apply_patch_safely()


def _configure_middleware(app: FastAPI) -> None:
    """
    Configure all middleware for the application.

    Middleware is added in order (first added = outermost).

    Load order:
    1. RequestIDMiddleware - Request correlation (outermost)
    2. PolicyMiddleware - OPA-style policy enforcement (ASGI level)
    3. PluginMiddleware - Plugin on_request/on_response hooks
    4. RouterDecisionMiddleware - Telemetry for routing decisions

    Args:
        app: The FastAPI application instance
    """
    from starlette.middleware.cors import CORSMiddleware

    from ..auth import RequestIDMiddleware
    from ..router_decision_callback import register_router_decision_middleware
    from .plugin_middleware import PluginMiddleware

    # CORS middleware - configurable via ROUTEIQ_CORS_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(),
        allow_credentials=_parse_cors_credentials(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.debug("Added CORSMiddleware")

    # Request ID middleware - raw ASGI (streaming-safe, outermost for correlation)
    # Starlette's add_middleware works with any class accepting (app, **kwargs) --
    # our raw ASGI class qualifies since __init__(self, app: ASGIApp).
    app.add_middleware(RequestIDMiddleware)
    logger.debug("Added RequestIDMiddleware (raw ASGI)")

    # Policy middleware - OPA-style enforcement at ASGI layer
    # This runs BEFORE routing and FastAPI authentication
    # Enables denial before streaming begins, no response buffering
    if add_policy_middleware(app):
        logger.info("Added PolicyMiddleware (policy enforcement enabled)")

    # Management middleware - classifies LiteLLM management endpoints and layers
    # RBAC, audit, OTel attributes, and plugin hooks on top
    from ..management_middleware import add_management_middleware

    add_management_middleware(app)

    # Plugin middleware - hooks for on_request/on_response
    # Starlette creates the instance lazily; __init__ self-registers
    # as the module singleton so plugin startup can call set_plugins().
    app.add_middleware(PluginMiddleware)
    logger.debug("Added PluginMiddleware (plugins wired during startup)")

    # Router decision telemetry middleware - emits TG4.1 router.* span attributes
    if register_router_decision_middleware(app):
        logger.debug("Added RouterDecisionMiddleware")


def _register_routes(app: FastAPI, include_admin: bool = True) -> None:
    """
    Register all LLMRouter routes with the application.

    Args:
        app: The FastAPI application instance
        include_admin: Whether to include admin routes (default: True)
    """
    from ..routes import (
        admin_router,
        health_router,
        llmrouter_router,
        mcp_parity_router,
        mcp_parity_admin_router,
        mcp_rest_router,
        mcp_proxy_router,
        oauth_callback_router,
        mcp_jsonrpc_router,
        mcp_sse_router,
        MCP_OAUTH_ENABLED,
        MCP_PROTOCOL_PROXY_ENABLED,
        MCP_SSE_TRANSPORT_ENABLED,
        MCP_SSE_LEGACY_MODE,
    )

    # Health router - unauthenticated K8s probes
    app.include_router(health_router, prefix="")
    logger.debug("Registered health_router")

    # LLMRouter routes - user auth protected
    app.include_router(llmrouter_router, prefix="")
    logger.debug("Registered llmrouter_router")

    # Admin routes - admin auth protected
    if include_admin:
        app.include_router(admin_router, prefix="")
        logger.debug("Registered admin_router")

    # MCP Parity Layer - upstream-compatible aliases
    # User-accessible parity endpoints (read operations)
    app.include_router(mcp_parity_router, prefix="")
    logger.debug("Registered mcp_parity_router (upstream-compatible /v1/mcp/*)")

    # Admin parity endpoints (write operations)
    if include_admin:
        app.include_router(mcp_parity_admin_router, prefix="")
        logger.debug(
            "Registered mcp_parity_admin_router (upstream-compatible /v1/mcp/* admin)"
        )

    # MCP REST API (/mcp-rest/*) - upstream-compatible
    app.include_router(mcp_rest_router, prefix="")
    logger.debug("Registered mcp_rest_router (upstream-compatible /mcp-rest/*)")

    # MCP Native JSON-RPC surface (/mcp) - for Claude Desktop / IDE MCP clients
    # This provides native MCP protocol (JSON-RPC 2.0 over HTTP)
    app.include_router(mcp_jsonrpc_router, prefix="")
    logger.debug("Registered mcp_jsonrpc_router (native MCP JSON-RPC at /mcp)")

    # MCP SSE Transport (/mcp/sse) - for real-time streaming events
    # Conditionally enabled based on feature flags
    if MCP_SSE_TRANSPORT_ENABLED and not MCP_SSE_LEGACY_MODE:
        app.include_router(mcp_sse_router, prefix="")
        logger.info(
            "Registered mcp_sse_router (SSE transport at /mcp/sse, "
            "MCP_SSE_TRANSPORT_ENABLED=true, MCP_SSE_LEGACY_MODE=false)"
        )
    else:
        logger.debug(
            f"Skipped mcp_sse_router (MCP_SSE_TRANSPORT_ENABLED={MCP_SSE_TRANSPORT_ENABLED}, "
            f"MCP_SSE_LEGACY_MODE={MCP_SSE_LEGACY_MODE})"
        )

    # Feature-flagged routers
    if MCP_PROTOCOL_PROXY_ENABLED and include_admin:
        app.include_router(mcp_proxy_router, prefix="")
        logger.info("Registered mcp_proxy_router (MCP_PROTOCOL_PROXY_ENABLED=true)")

    if MCP_OAUTH_ENABLED:
        app.include_router(oauth_callback_router, prefix="")
        logger.info("Registered oauth_callback_router (MCP_OAUTH_ENABLED=true)")


async def _run_plugin_startup(app: FastAPI) -> None:
    """
    Run plugin startup hooks, then wire middleware and callback bridges.

    After plugins are started, this function:
    1. Identifies plugins with on_request/on_response hooks → PluginMiddleware
    2. Identifies plugins with on_llm_* hooks → PluginCallbackBridge

    Args:
        app: The FastAPI application instance

    Raises:
        PluginDependencyError: If plugin dependencies cannot be resolved
        Exception: If any plugin with failure_mode=abort fails during startup
    """
    from .plugin_manager import get_plugin_manager, PluginDependencyError
    from .plugin_middleware import get_plugin_middleware
    from .plugin_callback_bridge import register_callback_bridge

    manager = get_plugin_manager()

    # Load plugins from config if not already loaded
    if not manager.plugins:
        loaded = manager.load_from_config()
        if loaded:
            logger.info(f"Loaded {loaded} plugins from configuration")

    try:
        await manager.startup(app)
    except PluginDependencyError as e:
        logger.error(f"Plugin dependency error: {e}")
        raise
    except Exception as e:
        # Re-raise if it's a startup abort
        logger.error(f"Plugin startup error: {e}")
        raise

    # Wire middleware-capable plugins into PluginMiddleware
    middleware_plugins = manager.get_middleware_plugins()
    plugin_mw = get_plugin_middleware()
    if plugin_mw and middleware_plugins:
        plugin_mw.set_plugins(middleware_plugins)
        logger.info(
            f"Wired {len(middleware_plugins)} middleware plugins into PluginMiddleware"
        )

    # Wire callback-capable plugins into LiteLLM via PluginCallbackBridge
    callback_plugins = manager.get_callback_plugins()
    if callback_plugins:
        bridge = register_callback_bridge(callback_plugins)
        if bridge:
            logger.info(f"Wired {len(callback_plugins)} callback plugins into LiteLLM")


async def _run_plugin_shutdown(app: FastAPI) -> None:
    """
    Run plugin shutdown hooks and clean up bridges.

    Args:
        app: The FastAPI application instance
    """
    from .plugin_manager import get_plugin_manager
    from .plugin_middleware import get_plugin_middleware
    from .plugin_callback_bridge import reset_callback_bridge

    # Clear middleware plugins first (stop intercepting new requests)
    plugin_mw = get_plugin_middleware()
    if plugin_mw:
        plugin_mw.set_plugins([])

    # Clear callback bridge
    reset_callback_bridge()

    # Then run plugin shutdown hooks
    manager = get_plugin_manager()
    await manager.shutdown(app)


async def _startup_http_client_pool() -> None:
    """
    Initialize the shared HTTP client pool.

    This is called during application startup to create the shared
    httpx.AsyncClient for outbound requests.
    """
    await startup_http_client_pool()


async def _shutdown_http_client_pool() -> None:
    """
    Shutdown the shared HTTP client pool.

    This is called during application shutdown to properly close
    all connections and cleanup resources.
    """
    await shutdown_http_client_pool()


def _load_plugins_before_routes() -> int:
    """
    Load plugins synchronously before routes are registered.

    This ensures plugins are discovered and validated BEFORE routes
    are finalized, allowing plugin route registration to work correctly.

    Returns:
        Number of plugins loaded
    """
    from .plugin_manager import get_plugin_manager

    manager = get_plugin_manager()

    # Only load if not already loaded
    if manager.plugins:
        logger.debug(f"Plugins already loaded: {len(manager.plugins)}")
        return len(manager.plugins)

    loaded = manager.load_from_config()
    if loaded:
        logger.info(f"Pre-loaded {loaded} plugins (startup hooks will run later)")

        # Log plugin order for debugging
        for i, plugin in enumerate(manager.plugins, 1):
            meta = plugin.metadata
            logger.debug(
                f"  [{i}] {plugin.name} "
                f"(priority={meta.priority}, capabilities={[c.value for c in meta.capabilities]})"
            )

    return loaded


def _mount_admin_ui(app: FastAPI) -> None:
    """Mount the admin UI static files if enabled and available."""
    if os.environ.get("ROUTEIQ_ADMIN_UI_ENABLED", "false").lower() != "true":
        return

    from pathlib import Path

    # Check multiple possible locations for UI dist
    possible_paths = [
        Path("/app/ui/dist"),  # Docker production
        Path("ui/dist"),  # Local development
        Path(__file__).parent.parent.parent.parent / "ui" / "dist",  # Relative to source
    ]

    ui_dist = None
    for path in possible_paths:
        if path.is_dir() and (path / "index.html").exists():
            ui_dist = path
            break

    if ui_dist is None:
        logger.warning(
            "ROUTEIQ_ADMIN_UI_ENABLED=true but no UI dist directory found. "
            "Build the UI with: cd ui && npm run build"
        )
        return

    from starlette.staticfiles import StaticFiles

    # Mount static files at /ui/
    app.mount(
        "/ui", StaticFiles(directory=str(ui_dist), html=True), name="routeiq-admin-ui"
    )

    logger.info(f"Admin UI mounted at /ui/ from {ui_dist}")


def create_app(
    *,
    apply_patch: bool = True,
    include_admin_routes: bool = True,
    enable_plugins: bool = True,
    enable_resilience: bool = True,
) -> FastAPI:
    """
    Configure the LiteLLM proxy's FastAPI app with LLMRouter extensions.

    This function:
    1. Applies the LiteLLM router patch (explicit, idempotent) — OR skips it
       when ``ROUTEIQ_USE_PLUGIN_STRATEGY=true`` (default), deferring to the
       plugin-based strategy that is installed after ``initialize()`` completes.
    2. Gets the LiteLLM proxy's FastAPI app
    3. Adds backpressure middleware (innermost, so 503s get CORS/RequestID headers)
    4. Adds remaining middleware (RequestID, CORS, Policy, Plugins, etc.)
    5. Loads plugins (discovery + validation, before routes)
    6. Registers LLMRouter routes (health, llmrouter, admin)
    7. Sets up plugin lifecycle hooks (startup runs later)
    8. Sets up HTTP client pool lifecycle hooks
    9. Configures graceful shutdown via drain manager

    This is the preferred method for in-process LiteLLM proxy usage.

    Args:
        apply_patch: Whether to apply the LiteLLM router patch (default: True)
        include_admin_routes: Whether to include admin routes (default: True)
        enable_plugins: Whether to enable plugin lifecycle (default: True)
        enable_resilience: Whether to enable backpressure/drain middleware (default: True)

    Returns:
        The configured FastAPI application instance
    """
    # Step 1: Determine routing approach — plugin strategy vs legacy monkey-patch.
    #
    # When ROUTEIQ_USE_PLUGIN_STRATEGY=true (the default), we skip the
    # monkey-patch entirely.  The plugin strategy will be installed later
    # by startup.py after LiteLLM's initialize() creates the Router instance.
    #
    # When ROUTEIQ_USE_PLUGIN_STRATEGY=false, we apply the legacy monkey-patch
    # BEFORE importing litellm.proxy (preserving the original behaviour).
    use_plugin = _use_plugin_strategy()

    if apply_patch:
        if use_plugin:
            logger.info(
                "ROUTEIQ_USE_PLUGIN_STRATEGY=true — skipping legacy monkey-patch. "
                "Plugin routing strategy will be installed after Router initialisation."
            )
        else:
            _apply_patch_safely()

    # Step 2: Get LiteLLM's FastAPI app
    from litellm.proxy.proxy_server import app

    # Store the routing approach decision so startup.py can read it
    app.state.use_plugin_strategy = use_plugin

    # Step 3: Add backpressure middleware FIRST (wraps app.app directly).
    # ─────────────────────────────────────────────────────────────────
    # ASGI middleware is LIFO: last-registered = outermost.  By registering
    # backpressure BEFORE RequestID/CORS, it becomes the INNERMOST layer.
    # This ensures 503 load-shed responses still pass through CORS and
    # RequestID middleware on the way out, so clients always receive
    # X-Request-ID and proper CORS headers — even under back-pressure.
    # ─────────────────────────────────────────────────────────────────
    if enable_resilience:
        add_backpressure_middleware(app)
        # Store graceful shutdown function for external use
        app.state.graceful_shutdown = lambda timeout=None: graceful_shutdown(
            app, timeout
        )
        logger.debug("Resilience middleware and drain manager attached (innermost)")

    # Step 4: Add remaining middleware (outermost layers)
    _configure_middleware(app)

    # Step 5: Load plugins BEFORE routes (for deterministic ordering)
    if enable_plugins:
        try:
            _load_plugins_before_routes()
        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")
            # Continue without plugins if loading fails

    # Step 6: Register routes
    _register_routes(app, include_admin=include_admin_routes)

    # Step 6b: Mount admin UI static files (AFTER routes, so API routes take priority)
    _mount_admin_ui(app)

    # Step 7: Set up plugin lifecycle if enabled
    if enable_plugins:
        # Store original lifespan if any
        original_lifespan = getattr(app.router, "lifespan_context", None)

        @asynccontextmanager
        async def lifespan_with_plugins(app: FastAPI) -> AsyncGenerator[None, None]:
            """Lifespan context manager that includes plugin lifecycle."""
            # Run original lifespan startup if exists
            if original_lifespan:
                async with original_lifespan(app):
                    await _run_plugin_startup(app)
                    try:
                        yield
                    finally:
                        await _run_plugin_shutdown(app)
            else:
                await _run_plugin_startup(app)
                try:
                    yield
                finally:
                    await _run_plugin_shutdown(app)

        # Note: We don't replace the lifespan here since LiteLLM manages its own.
        # Instead, plugins are started explicitly by startup.py after initialization.
        app.state.llmrouter_plugin_startup = lambda: _run_plugin_startup(app)
        app.state.llmrouter_plugin_shutdown = lambda: _run_plugin_shutdown(app)

    # Step 8: Set up HTTP client pool lifecycle hooks
    # These are called explicitly by startup.py for proper ordering
    app.state.llmrouter_http_pool_startup = _startup_http_client_pool
    app.state.llmrouter_http_pool_shutdown = _shutdown_http_client_pool
    logger.debug("HTTP client pool lifecycle hooks attached")

    logger.info("Gateway app created and configured")
    return app


def create_standalone_app(
    *,
    title: str = "LLMRouter Gateway",
    version: str = "0.0.3",
    include_admin_routes: bool = True,
    enable_plugins: bool = True,
    enable_resilience: bool = True,
) -> FastAPI:
    """
    Create a standalone FastAPI app with just LLMRouter routes.

    This does NOT include LiteLLM proxy - use this for:
    - Testing LLMRouter routes in isolation
    - Running LLMRouter as a separate service

    Note: LiteLLM router patch is not applied since there's no LiteLLM proxy.

    Args:
        title: FastAPI app title
        version: FastAPI app version
        include_admin_routes: Whether to include admin routes (default: True)
        enable_plugins: Whether to enable plugin lifecycle (default: True)
        enable_resilience: Whether to enable backpressure/drain middleware (default: True)

    Returns:
        A new standalone FastAPI application instance
    """
    # Load plugins BEFORE creating app (for deterministic ordering)
    if enable_plugins:
        try:
            _load_plugins_before_routes()
        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan context manager for standalone app."""
        # Initialize HTTP client pool
        await _startup_http_client_pool()

        if enable_plugins:
            await _run_plugin_startup(app)
        try:
            yield
        finally:
            # Graceful shutdown with drain
            if enable_resilience:
                drain_manager = get_drain_manager()
                await drain_manager.start_drain()
                await drain_manager.wait_for_drain()
            if enable_plugins:
                await _run_plugin_shutdown(app)
            # Shutdown HTTP client pool
            await _shutdown_http_client_pool()

    app = FastAPI(
        title=title,
        version=version,
        lifespan=lifespan,
    )

    # Add backpressure middleware FIRST (wraps app.app directly).
    # See create_app() for the full rationale — registering backpressure
    # before CORS/RequestID makes it the INNERMOST layer so that 503
    # load-shed responses still receive X-Request-ID and CORS headers.
    if enable_resilience:
        add_backpressure_middleware(app)
        app.state.graceful_shutdown = lambda timeout=None: graceful_shutdown(
            app, timeout
        )
        logger.debug("Resilience middleware and drain manager attached (innermost)")

    # Add remaining middleware (outermost layers)
    _configure_middleware(app)

    # Register routes
    _register_routes(app, include_admin=include_admin_routes)

    # Mount admin UI static files (AFTER routes, so API routes take priority)
    _mount_admin_ui(app)

    logger.info("Standalone gateway app created")
    return app
