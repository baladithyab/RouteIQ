"""
Gateway Application Factory (Composition Root)
==============================================

This module provides the FastAPI application factory for the LLMRouter gateway.
It explicitly configures all middleware, routers, and patches in a single place.

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

    # Request ID middleware - should be outermost for correlation
    app.add_middleware(RequestIDMiddleware)
    logger.debug("Added RequestIDMiddleware")


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
        MCP_OAUTH_ENABLED,
        MCP_PROTOCOL_PROXY_ENABLED,
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
    """
    from .plugin_manager import get_plugin_manager

    manager = get_plugin_manager()

    # Load plugins from config if not already loaded
    if not manager.plugins:
        loaded = manager.load_from_config()
        if loaded:
            logger.info(f"Loaded {loaded} plugins from configuration")

    await manager.startup(app)


async def _run_plugin_shutdown(app: FastAPI) -> None:
    """
    Run plugin shutdown hooks.

    Args:
        app: The FastAPI application instance
    """
    from .plugin_manager import get_plugin_manager

    manager = get_plugin_manager()
    await manager.shutdown(app)


def create_app(
    *,
    apply_patch: bool = True,
    include_admin_routes: bool = True,
    enable_plugins: bool = True,
) -> FastAPI:
    """
    Configure the LiteLLM proxy's FastAPI app with LLMRouter extensions.

    This function:
    1. Applies the LiteLLM router patch (explicit, idempotent)
    2. Gets the LiteLLM proxy's FastAPI app
    3. Adds RequestID middleware
    4. Registers LLMRouter routes (health, llmrouter, admin)
    5. Optionally runs plugin startup hooks

    This is the preferred method for in-process LiteLLM proxy usage.

    Args:
        apply_patch: Whether to apply the LiteLLM router patch (default: True)
        include_admin_routes: Whether to include admin routes (default: True)
        enable_plugins: Whether to enable plugin lifecycle (default: True)

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

    # Step 4: Register routes
    _register_routes(app, include_admin=include_admin_routes)

    # Step 5: Set up plugin lifecycle if enabled
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

    logger.info("Gateway app created and configured")
    return app


def create_standalone_app(
    *,
    title: str = "LLMRouter Gateway",
    version: str = "0.1.1",
    include_admin_routes: bool = True,
    enable_plugins: bool = True,
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

    Returns:
        A new standalone FastAPI application instance
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan context manager for standalone app."""
        if enable_plugins:
            await _run_plugin_startup(app)
        try:
            yield
        finally:
            if enable_plugins:
                await _run_plugin_shutdown(app)

    app = FastAPI(
        title=title,
        version=version,
        lifespan=lifespan,
    )

    # Add middleware
    _configure_middleware(app)

    # Register routes
    _register_routes(app, include_admin=include_admin_routes)

    logger.info("Standalone gateway app created")
    return app
