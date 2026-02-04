"""
Gateway Application Factory (Composition Root)
==============================================

This module provides the FastAPI application factory for the LLMRouter gateway.
It explicitly configures all middleware, routers, and patches in a single place.

Load Order:
1. Apply LiteLLM router patch (if enabled)
2. Get/create FastAPI app
3. Add middleware (including backpressure/resilience)
4. Load and register plugins (deterministically before routes)
5. Register built-in routes
6. Set up plugin lifecycle hooks
7. Set up HTTP client pool lifecycle hooks
8. Set up graceful shutdown hooks

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
from contextlib import asynccontextmanager
from typing import AsyncGenerator

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

logger = logging.getLogger(__name__)


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


def _configure_middleware(app: FastAPI) -> None:
    """
    Configure all middleware for the application.

    Middleware is added in order (first added = outermost).

    Args:
        app: The FastAPI application instance
    """
    from ..auth import RequestIDMiddleware
    from ..router_decision_callback import register_router_decision_middleware

    # Request ID middleware - should be outermost for correlation
    app.add_middleware(RequestIDMiddleware)
    logger.debug("Added RequestIDMiddleware")
    
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
    Run plugin startup hooks.

    Args:
        app: The FastAPI application instance

    Raises:
        PluginDependencyError: If plugin dependencies cannot be resolved
        Exception: If any plugin with failure_mode=abort fails during startup
    """
    from .plugin_manager import get_plugin_manager, PluginDependencyError

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


async def _run_plugin_shutdown(app: FastAPI) -> None:
    """
    Run plugin shutdown hooks.

    Args:
        app: The FastAPI application instance
    """
    from .plugin_manager import get_plugin_manager

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
    1. Applies the LiteLLM router patch (explicit, idempotent)
    2. Gets the LiteLLM proxy's FastAPI app
    3. Adds RequestID middleware
    4. Loads plugins (discovery + validation, before routes)
    5. Registers LLMRouter routes (health, llmrouter, admin)
    6. Sets up plugin lifecycle hooks (startup runs later)
    7. Sets up HTTP client pool lifecycle hooks
    8. Adds backpressure middleware and drain manager (if enabled)

    This is the preferred method for in-process LiteLLM proxy usage.

    Args:
        apply_patch: Whether to apply the LiteLLM router patch (default: True)
        include_admin_routes: Whether to include admin routes (default: True)
        enable_plugins: Whether to enable plugin lifecycle (default: True)
        enable_resilience: Whether to enable backpressure/drain middleware (default: True)

    Returns:
        The configured FastAPI application instance
    """
    # Step 1: Apply patch BEFORE importing litellm.proxy
    if apply_patch:
        _apply_patch_safely()

    # Step 2: Get LiteLLM's FastAPI app
    from litellm.proxy.proxy_server import app

    # Step 3: Add middleware
    _configure_middleware(app)

    # Step 4: Load plugins BEFORE routes (for deterministic ordering)
    if enable_plugins:
        try:
            _load_plugins_before_routes()
        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")
            # Continue without plugins if loading fails

    # Step 5: Register routes
    _register_routes(app, include_admin=include_admin_routes)

    # Step 6: Set up plugin lifecycle if enabled
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

    # Step 7: Set up HTTP client pool lifecycle hooks
    # These are called explicitly by startup.py for proper ordering
    def http_pool_setup(app: FastAPI) -> None:
        app.state.llmrouter_http_pool_startup = _startup_http_client_pool
        app.state.llmrouter_http_pool_shutdown = _shutdown_http_client_pool
        logger.debug("HTTP client pool lifecycle hooks attached")

    app.state.http_pool_setup = http_pool_setup

    # Step 8: Add backpressure middleware (wraps ASGI app)
    if enable_resilience:
        add_backpressure_middleware(app)
        # Store graceful shutdown function for external use
        app.state.graceful_shutdown = lambda timeout=None: graceful_shutdown(app, timeout)
        logger.debug("Resilience middleware and drain manager attached")

    logger.info("Gateway app created and configured")
    return app


def create_standalone_app(
    *,
    title: str = "LLMRouter Gateway",
    version: str = "0.1.1",
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

    # Add middleware
    _configure_middleware(app)

    # Register routes
    _register_routes(app, include_admin=include_admin_routes)

    # Add backpressure middleware (wraps ASGI app)
    if enable_resilience:
        add_backpressure_middleware(app)
        app.state.graceful_shutdown = lambda timeout=None: graceful_shutdown(app, timeout)
        logger.debug("Resilience middleware and drain manager attached")

    logger.info("Standalone gateway app created")
    return app
