"""
Gateway Application Factory (Composition Root)
==============================================

This module provides the FastAPI application factory for the LLMRouter gateway.
It explicitly configures all middleware, routers, and patches in a single place.

Three factory functions are provided:

- ``create_gateway_app()`` — **ADR-0012** (recommended): RouteIQ owns the FastAPI
  application and its full lifecycle.  LiteLLM proxy is optionally mounted as a
  sub-application at ``/v1/``.  Activated via ``ROUTEIQ_OWN_APP=true``.

- ``create_app()`` — **Legacy default**: borrows LiteLLM's ``FastAPI`` instance and
  layers RouteIQ middleware/routes on top.  Lifecycle hooks are stored as lambdas
  on ``app.state`` and invoked by ``startup.py``.  Emits a deprecation notice
  pointing to ``create_gateway_app()``.

- ``create_standalone_app()`` — Testing / standalone mode: creates a standalone
  FastAPI app with only RouteIQ routes (no LiteLLM proxy).

Load Order (create_gateway_app):
1. Create RouteIQ-owned FastAPI app with proper lifespan
2. Add backpressure middleware (INNERMOST)
3. Add remaining middleware (RequestID, CORS, Policy, Plugins, etc.)
4. Load and register plugins (deterministically before routes)
5. Register built-in routes
6. Mount LiteLLM proxy at /v1/ (optional)
7. Lifespan manages ALL startup/shutdown — no more app.state lambdas

Load Order (create_app — legacy):
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

Usage (new — ADR-0012):
    from litellm_llmrouter.gateway import create_gateway_app

    # RouteIQ owns the app; LiteLLM mounted at /v1/
    app = create_gateway_app()

Usage with LiteLLM proxy (legacy):
    from litellm_llmrouter.gateway import create_app

    # This configures the LiteLLM proxy's FastAPI app
    app = create_app()

Usage standalone (without LiteLLM):
    from litellm_llmrouter.gateway import create_standalone_app

    # This creates a standalone FastAPI app with just LLMRouter routes
    app = create_standalone_app()
"""

import importlib.metadata
import logging
import os
import warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI

from ..settings import get_settings

# Legacy monkey-patch module has been removed.
# is_patch_applied() and patch_litellm_router() are now no-op stubs in __init__.py.
from ..resilience import (
    add_backpressure_middleware,
    get_drain_manager,
    graceful_shutdown,
)
from ..http_client_pool import (
    startup_http_client_pool,
    shutdown_http_client_pool,
)
from ..redis_pool import close_async_redis_client
from ..database import close_db_pool
from ..policy_engine import add_policy_middleware

logger = logging.getLogger(__name__)


def _get_version() -> str:
    """Return the RouteIQ package version, falling back to ``0.0.0-dev``."""
    try:
        return importlib.metadata.version("routeiq")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0-dev"


def _use_plugin_strategy() -> bool:
    """
    Always returns True — the plugin-based routing strategy is the only
    supported path.  The legacy monkey-patch has been removed.
    """
    return True


def _api_surface_bare_paths_enabled() -> bool:
    """Whether to register the first-class bare-path API surfaces (RouteIQ-e48a).

    Honours the legacy flat env var first, then the typed setting
    (``settings.api_surfaces.bare_paths_enabled``).  Defaults ON so the surfaces
    are documented + middleware-wrapped out of the box; a settings failure also
    defaults ON (the registration is itself a graceful no-op when LiteLLM's
    handlers are unavailable).
    """
    raw = os.getenv("ROUTEIQ_API_SURFACE_BARE_PATHS")
    if raw is not None:
        return raw.strip().lower() not in ("false", "0", "no", "off")
    try:
        return get_settings().api_surfaces.bare_paths_enabled
    except Exception:
        return True


def _parse_cors_origins() -> list[str]:
    """Parse CORS origins into a list of allowed origins.

    Checks ``ROUTEIQ_CORS_ORIGINS`` env var first (legacy), then typed
    settings, then defaults to ``["*"]``.
    """
    # Legacy env var takes precedence (it may differ from the settings
    # model's env var name which is ROUTEIQ_SECURITY__CORS_ORIGINS).
    raw = os.getenv("ROUTEIQ_CORS_ORIGINS")
    if raw is not None:
        return [o.strip() for o in raw.split(",")]
    try:
        settings = get_settings()
        return settings.cors_origins_list
    except Exception:
        return ["*"]


def _parse_cors_credentials() -> bool:
    """Parse CORS credentials flag.

    Checks ``ROUTEIQ_CORS_CREDENTIALS`` env var first (legacy), then
    typed settings.
    """
    raw = os.getenv("ROUTEIQ_CORS_CREDENTIALS")
    if raw is not None:
        return raw.lower() == "true"
    try:
        settings = get_settings()
        return settings.security.cors_credentials
    except Exception:
        return False


def _apply_patch_safely() -> bool:
    """
    Legacy stub — the monkey-patch module has been removed.

    Always returns False and logs a warning. The plugin-based routing
    strategy (``RouteIQRoutingStrategy``) is the only supported path.
    """
    logger.warning(
        "_apply_patch_safely() called but the legacy monkey-patch module "
        "has been removed. The plugin-based routing strategy is the only "
        "supported path. Returning False."
    )
    return False


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
        logger.error(
            f"Failed to install plugin routing strategy: {e}. "
            f"No fallback available (legacy monkey-patch has been removed)."
        )
        return False


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
    from ..router_decision_callback import (
        register_router_decision_middleware,
        ROUTEIQ_RESPONSE_HEADERS,
    )
    from .plugin_middleware import PluginMiddleware

    # CORS middleware - configurable via ROUTEIQ_CORS_ORIGINS
    # When the UI is deployed separately (disaggregated mode), its origin
    # must be in the allowed list.  ROUTEIQ_ADMIN_UI_EXTERNAL_URL is
    # automatically added to avoid manual CORS configuration.
    cors_origins = _parse_cors_origins()
    try:
        settings = get_settings()
        ext_url = settings.admin_ui_external_url
    except Exception:
        ext_url = os.getenv("ROUTEIQ_ADMIN_UI_EXTERNAL_URL")
    if ext_url:
        # Normalise: strip trailing slash to match Origin header format
        ext_origin = ext_url.rstrip("/")
        if ext_origin not in cors_origins and "*" not in cors_origins:
            cors_origins.append(ext_origin)
            logger.info("Added external UI origin to CORS: %s", ext_origin)

    cors_credentials = _parse_cors_credentials()
    if "*" in cors_origins:
        # Wildcard + credentials is a browser security vulnerability
        cors_credentials = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=ROUTEIQ_RESPONSE_HEADERS,
    )
    logger.debug("Added CORSMiddleware (origins=%s)", cors_origins)

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

    # Model-alias rewrite middleware (RouteIQ-0dcb) - rewrites the request
    # ``model`` field PRE-routing so an unmodified Anthropic client pinning a
    # concrete id (e.g. claude-sonnet-4-...) is transparently routed through a
    # target group (e.g. claude-auto + the bandit). No-op / not added when the
    # alias layer is disabled or has no rules (default identity).
    from ..model_alias import add_model_alias_middleware

    if add_model_alias_middleware(app):
        logger.info("Added ModelAliasMiddleware (pre-routing model rewrite)")


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
    )

    # Health router - unauthenticated K8s probes.
    # Also carries the Prometheus scrape endpoint GET /metrics (RouteIQ-f60a):
    # network/admin-scoped (scrapers carry no user key), same scope as /_health/*.
    # Registered via routes/metrics_endpoint.py importing onto health_router.
    app.include_router(health_router, prefix="")
    logger.debug("Registered health_router")

    # LLMRouter routes - user auth protected
    # Includes MCP gateway REST endpoints (/llmrouter/mcp/*)
    app.include_router(llmrouter_router, prefix="")
    logger.debug("Registered llmrouter_router")

    # Admin routes - admin auth protected
    if include_admin:
        app.include_router(admin_router, prefix="")
        logger.debug("Registered admin_router")

    # Register OIDC routes if configured
    try:
        from litellm_llmrouter.oidc import create_oidc_router, get_oidc_config

        oidc_config = get_oidc_config()
        if oidc_config.enabled:
            oidc_router = create_oidc_router(oidc_config)
            app.include_router(oidc_router)
            logger.info(
                "OIDC/SSO routes registered: /sso/login, /sso/callback, "
                "/auth/token-exchange, /auth/userinfo"
            )
    except ImportError:
        logger.debug("OIDC module not available")
    except Exception as e:
        logger.warning(f"OIDC route registration failed: {e}")

    # Note: MCP parity layer, JSON-RPC, and SSE transport routers have been
    # removed. These are now provided natively by LiteLLM.


def _input_guardrail_policies_configured() -> bool:
    """True if any input-phase guardrail policy is configured.

    Used to decide whether the ``PluginCallbackBridge`` must be registered even
    when there are no callback-capable plugins — the bridge hosts the input
    guardrail deny seam, so skipping registration would silently bypass input
    guardrails (fail-open). Fail-safe: any error resolving the engine returns
    ``False`` (we only force registration when policies are positively present;
    plugins still register on their own path).
    """
    try:
        from litellm_llmrouter.guardrail_policies import (
            GuardrailPhase,
            get_guardrail_policy_engine,
        )

        engine = get_guardrail_policy_engine()
        return bool(engine.list_policies(phase=GuardrailPhase.INPUT))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not determine input guardrail policy presence: %s", exc)
        return False


async def _run_plugin_startup(app: FastAPI) -> None:
    """
    Run plugin startup hooks, then wire middleware and callback bridges.

    After plugins are started, this function:
    1. Identifies plugins with on_request/on_response hooks → PluginMiddleware
    2. Identifies plugins with on_llm_* hooks → PluginCallbackBridge (a
       litellm CustomLogger; request mutations ride its
       async_pre_call_deployment_hook, not the logging hook -- RouteIQ-60e3)

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

    # Wire callback-capable plugins into LiteLLM via PluginCallbackBridge.
    #
    # SECURITY: the bridge ALSO hosts the input-guardrail deny seam, which must
    # run whenever guardrail policies are configured — even with zero
    # callback-capable plugins. Registering only when callback_plugins is
    # non-empty would silently bypass input guardrails for an operator whose
    # only control-plane surface is guardrail policies (fail-open). So we force
    # registration when guardrail policies may exist. Byte-stable when neither
    # plugins nor guardrail policies are present (register_callback_bridge
    # returns None without touching litellm.callbacks).
    callback_plugins = manager.get_callback_plugins()
    guardrails_require_bridge = _input_guardrail_policies_configured()
    if callback_plugins or guardrails_require_bridge:
        bridge = register_callback_bridge(
            callback_plugins, force=guardrails_require_bridge
        )
        if bridge:
            if callback_plugins:
                logger.info(
                    f"Wired {len(callback_plugins)} callback plugins into LiteLLM"
                )
            if guardrails_require_bridge and not callback_plugins:
                logger.info(
                    "Wired PluginCallbackBridge into LiteLLM for input guardrail "
                    "enforcement (no callback plugins; guardrail policies present)"
                )


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


def _register_adapter_loader_plugin(manager: Any) -> bool:
    """Register the out-of-tree adapter loader behind a feature flag.

    ``AdapterLoaderPlugin`` discovers ``routeiq.routing_adapters`` entry-point
    adapters at startup (RouteIQ-a089).  It is gated by
    ``settings.adapter_framework.entrypoint_discovery`` (default OFF) so the
    default boot stays byte-stable — when the flag is off this is a no-op and
    no entry-point enumeration happens.

    Args:
        manager: The plugin manager to register the loader into.

    Returns:
        True if the loader plugin was registered, False otherwise.
    """
    try:
        settings = get_settings()
        adapter_fw = settings.adapter_framework
        discovery_on = adapter_fw.entrypoint_discovery
    except Exception as exc:
        logger.debug("Adapter-framework settings unavailable: %s", exc)
        return False

    if not discovery_on:
        return False

    try:
        from ..adapters.loader import AdapterLoaderPlugin
        from .plugin_manager import PluginCapability

        # Capability negotiation, when enabled, restricts staged adapters to
        # the routing-strategy capability (the allowlist the loader enforces
        # per-adapter).  When negotiation is off, the loader allows all.
        allowed = (
            {PluginCapability.ROUTING_STRATEGY}
            if adapter_fw.capability_negotiation
            else None
        )
        manager.register(AdapterLoaderPlugin(allowed_capabilities=allowed))
        logger.info(
            "Registered adapter-loader plugin (entrypoint_discovery=on, "
            "capability_negotiation=%s)",
            adapter_fw.capability_negotiation,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to register adapter-loader plugin: %s", exc)
        return False


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

    # Register the built-in out-of-tree adapter loader (RouteIQ-a089) behind
    # the entrypoint_discovery flag.  Default OFF => no-op => byte-stable boot.
    if _register_adapter_loader_plugin(manager):
        loaded += 1

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
    raw = os.environ.get("ROUTEIQ_ADMIN_UI_ENABLED")
    if raw is not None:
        admin_ui_enabled = raw.lower() == "true"
    else:
        try:
            settings = get_settings()
            admin_ui_enabled = settings.admin_ui_enabled
        except Exception:
            admin_ui_enabled = False
    if not admin_ui_enabled:
        return

    from pathlib import Path

    # Check multiple possible locations for UI dist
    possible_paths = [
        Path("/app/ui/dist"),  # Docker production
        Path("ui/dist"),  # Local development
        Path(__file__).parent.parent.parent.parent
        / "ui"
        / "dist",  # Relative to source
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


# ---------------------------------------------------------------------------
# ADR-0012: RouteIQ-owned application lifecycle
# ---------------------------------------------------------------------------


async def _run_http_pool_startup() -> None:
    """Initialize the shared HTTP client pool during lifespan startup."""
    await startup_http_client_pool()


async def _probe_services() -> None:
    """Run service discovery probes and log the results table."""
    try:
        from ..service_discovery import (
            probe_all_services,
            get_feature_availability,
            format_service_status_table,
        )

        services = await probe_all_services()
        features = get_feature_availability(services)
        table = format_service_status_table(services, features)
        logger.info("Service discovery complete:\n%s", table)
    except ImportError as exc:
        logger.debug("Service discovery not available: %s", exc)
    except Exception as exc:
        logger.warning("Service discovery failed: %s", exc)


@asynccontextmanager
async def _routeiq_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """RouteIQ application lifespan manager (ADR-0012).

    Handles startup and shutdown of **all** subsystems in the correct order,
    replacing the ``app.state.*`` lambda-hook pattern used by
    :func:`create_app`.

    Startup order:
        1. HTTP client pool (outbound connections — plugins may use this)
        2. Plugin startup (middleware wiring, callback bridges)
        3. Service discovery probes (informational logging)

    Shutdown order (reverse):
        1. Graceful drain (stop accepting, wait for in-flight)
        2. Plugin shutdown (middleware + callbacks cleared)
        3. Database connection pool close
        4. Redis client close
        5. HTTP client pool close
    """
    # === STARTUP ===
    logger.info("RouteIQ Gateway starting (ADR-0012 own-app mode)...")

    # 0. Initialize LiteLLM proxy (loads config, creates Router instance)
    #    This MUST happen before strategy installation and plugin startup.
    try:
        import litellm.proxy.proxy_server as _proxy_server

        config_path = os.environ.get("LITELLM_CONFIG_PATH") or os.environ.get(
            "CONFIG_FILE_PATH"
        )
        if config_path:
            await _proxy_server.initialize(
                config=config_path,
                telemetry=False,
            )
            logger.info("LiteLLM proxy initialized (config=%s)", config_path)
        else:
            logger.warning(
                "No LITELLM_CONFIG_PATH set — LiteLLM proxy not initialized. "
                "Model routing will not work until config is loaded."
            )
    except Exception as exc:
        logger.error("LiteLLM initialization failed: %s", exc)

    # 0a2. Bedrock auto-discovery -> merge synthesized arms into the live
    #      model_list BEFORE the routing strategy is installed (RouteIQ-c417).
    #      Leader- and flag-gated: default-off => byte-stable no-op (the
    #      operator-authored model_list is untouched). Must run BEFORE the
    #      strategy install so the discovered group is routable on first request.
    try:
        from ..startup import merge_bedrock_discovered_models

        merged = merge_bedrock_discovered_models()
        if merged:
            logger.info("Bedrock discovery merged %d model_list entries", merged)
    except Exception as exc:
        logger.warning("Bedrock discovery merge skipped/failed (non-fatal): %s", exc)

    # 0b. Install RouteIQ plugin routing strategy on the LiteLLM Router
    if getattr(app.state, "use_plugin_strategy", True):
        try:
            from ..startup import install_plugin_routing_strategy

            install_plugin_routing_strategy(app)
            logger.info("Plugin routing strategy installed on LiteLLM Router")
        except Exception as exc:
            logger.warning("Plugin routing strategy installation failed: %s", exc)

    # 1. HTTP client pool — must be ready before plugins that may issue requests
    await _run_http_pool_startup()
    logger.info("HTTP client pool initialized")

    # 2. Plugin startup
    await _run_plugin_startup(app)

    # 3a. Run RouteIQ-native DB migrations at boot (P4, additive + idempotent).
    #     Creates A2A/MCP/audit/governance/spend tables via CREATE ... IF NOT
    #     EXISTS.  DB-optional: run_migrations() early-returns when no DATABASE_URL.
    #     This also fixes the pre-existing gap that audit/A2A/MCP tables were only
    #     migrated lazily.  Never blocks boot.
    try:
        from ..database import run_migrations as run_routeiq_migrations

        await run_routeiq_migrations()
    except Exception as exc:
        logger.warning("RouteIQ DB migrations skipped/failed (non-fatal): %s", exc)

    # 3b. Load persisted governance state.  DB-first (Aurora) when the store is
    #     enabled; otherwise the JSON-file path (unchanged).  A DB hiccup degrades
    #     to JSON, never blocks boot.
    try:
        from ..governance import (
            get_governance_engine,
            load_governance_state,
        )
        from ..usage_policies import (
            get_usage_policy_engine,
            load_usage_policies_state,
        )
        from ..guardrail_policies import (
            get_guardrail_policy_engine,
            load_guardrail_policies_state,
        )
        from ..prompt_management import (
            get_prompt_manager,
            load_prompts_state,
        )
        from ..governance_store import get_governance_store

        store = get_governance_store()
        if store.enabled:
            gov = get_governance_engine()
            for org in await store.load_all_orgs():
                gov.register_org(org)
            # Hydration order: orgs -> workspaces -> keys (keys soft-FK workspaces).
            for ws in await store.load_all_workspaces():
                gov.register_workspace(ws)
            for kg in await store.load_all_keys():
                gov.register_key_governance(kg)
            up = get_usage_policy_engine()
            for p in await store.load_all_policies():
                up.add_policy(p)
            # Guardrail policies + prompts share the same durable-backend pattern
            # (RouteIQ-4f30 + RouteIQ-c2af): DB-first when the store is enabled so
            # the 14-check guardrail layer + prompt versioning survive pod churn.
            gp = get_guardrail_policy_engine()
            for guard in await store.load_all_guardrails():
                gp.add_policy(guard)
            pm = get_prompt_manager()
            for storage_key, prompt in await store.load_all_prompts():
                pm._prompts[storage_key] = prompt
            gov_count = len(gov.list_workspaces()) + len(gov.list_orgs())
            up_count = len(up.list_policies())
            gp_count = len(gp.list_policies())
            pm_count = len(pm._prompts)
            logger.info(
                "Hydrated governance state from Aurora: workspaces+orgs=%d, "
                "usage_policies=%d, guardrail_policies=%d, prompts=%d",
                gov_count,
                up_count,
                gp_count,
                pm_count,
            )
        else:
            gov_count = load_governance_state(get_governance_engine())
            up_count = load_usage_policies_state(get_usage_policy_engine())
            gp_count = load_guardrail_policies_state(get_guardrail_policy_engine())
            pm_count = load_prompts_state(get_prompt_manager())
        total = gov_count + up_count + gp_count + pm_count
        if total > 0:
            logger.info(
                "Loaded persisted state: governance=%d, usage_policies=%d, "
                "guardrail_policies=%d, prompts=%d",
                gov_count,
                up_count,
                gp_count,
                pm_count,
            )
    except Exception as exc:
        logger.warning("Failed to load persisted governance state: %s", exc)

    # 4. Service discovery (informational only, never blocks startup)
    await _probe_services()

    # 4. Start eval pipeline background loop if enabled
    eval_pipeline = None
    try:
        from ..eval_pipeline import get_eval_pipeline

        eval_pipeline = get_eval_pipeline()
        if eval_pipeline is not None:
            await eval_pipeline.start_background_loop()
            logger.info("Eval pipeline background loop started")

            # 4b. Wire the MLOps closed loop (Cluster H, RouteIQ-fc5c) on top of
            #     the running eval pipeline: subscribe the drift+promotion feedback
            #     callback so each push_feedback() aggregate reaches the drift
            #     detector and champion/challenger promoter. Gated on any
            #     settings.mlops sub-loop being enabled -> byte-stable no-op when
            #     all three are off (the default).
            try:
                from ..eval_pipeline import wire_mlops_loop

                if wire_mlops_loop(eval_pipeline=eval_pipeline):
                    logger.info("MLOps closed loop wired into eval pipeline")
            except Exception as exc:
                logger.warning("MLOps loop wiring skipped/failed: %s", exc)
    except ImportError:
        logger.debug("Eval pipeline module not available")
    except Exception as exc:
        logger.warning("Eval pipeline startup failed: %s", exc)

    logger.info("RouteIQ Gateway ready")

    yield

    # === SHUTDOWN ===
    logger.info("RouteIQ Gateway shutting down...")

    # 0. Stop eval pipeline background loop
    if eval_pipeline is not None:
        try:
            await eval_pipeline.stop()
            logger.info("Eval pipeline stopped")
        except Exception as exc:
            logger.warning("Eval pipeline shutdown error: %s", exc)

    # 1. Graceful drain — let in-flight requests finish
    try:
        drain_manager = get_drain_manager()
        await drain_manager.start_drain()
        await drain_manager.wait_for_drain()
    except Exception as exc:
        logger.warning("Drain manager error during shutdown: %s", exc)

    # 1b. Flush durable bandit posteriors (RouteIQ-95a8 DEFECT-2). After drain no
    #     more updates arrive, so persist the debounced tail (up to
    #     dirty_threshold-1 updates) the FilePosteriorBackend has not yet flushed
    #     — otherwise a clean shutdown loses them and convergence-across-restarts
    #     regresses. Best-effort / fail-open; a no-op for the memory backend.
    try:
        from ..kumaraswamy_thompson import flush_posteriors_on_shutdown

        flush_posteriors_on_shutdown()
    except Exception as exc:
        logger.warning("KTS posterior flush at shutdown skipped/failed: %s", exc)

    # 2. Plugin shutdown (stops middleware interception first)
    await _run_plugin_shutdown(app)

    # 3. Database connection pool
    await close_db_pool()

    # 4. Redis client
    await close_async_redis_client()

    # 5. HTTP client pool (last — other shutdown steps may issue requests)
    await shutdown_http_client_pool()

    logger.info("RouteIQ Gateway stopped")


def create_gateway_app(
    config_path: Optional[str] = None,
    *,
    mount_litellm: bool = True,
    include_admin_routes: bool = True,
    enable_plugins: bool = True,
    enable_resilience: bool = True,
) -> FastAPI:
    """Create the RouteIQ gateway FastAPI application (ADR-0012).

    This is the **new recommended entry point**.  RouteIQ owns the app
    lifecycle, middleware stack, and exception handlers.  LiteLLM proxy is
    mounted as a sub-application at ``/v1/`` when *mount_litellm* is True.

    Compared to :func:`create_app` (legacy):

    * RouteIQ owns the ``FastAPI`` instance and its lifespan.
    * All startup/shutdown logic lives in :func:`_routeiq_lifespan` — no
      more ``app.state`` lambda hacks.
    * LiteLLM is an optional sub-mount, not the host application.

    This is the default mode (``ROUTEIQ_OWN_APP=true``).  Set
    ``ROUTEIQ_OWN_APP=false`` to fall back to the legacy :func:`create_app` path.

    Args:
        config_path: Path to config YAML file.  If provided, sets
            ``LITELLM_CONFIG_PATH`` / ``CONFIG_FILE_PATH`` for LiteLLM.
        mount_litellm: Whether to mount LiteLLM proxy at ``/v1/``
            (default: True).
        include_admin_routes: Whether to include admin routes (default: True).
        enable_plugins: Whether to enable plugin lifecycle (default: True).
        enable_resilience: Whether to enable backpressure / drain middleware
            (default: True).

    Returns:
        FastAPI application with RouteIQ routes and optional LiteLLM proxy.
    """
    # Propagate config path so LiteLLM can pick it up when mounted
    if config_path:
        os.environ.setdefault("LITELLM_CONFIG_PATH", config_path)
        os.environ.setdefault("CONFIG_FILE_PATH", config_path)

    # Pre-load plugins (discovery + validation) before app creation so plugin
    # route registration can participate in the route table.
    if enable_plugins:
        try:
            _load_plugins_before_routes()
        except Exception as exc:
            logger.error("Failed to load plugins: %s", exc)

    # --- Create RouteIQ's own FastAPI app with proper lifespan ---
    _raw_ui = os.getenv("ROUTEIQ_ADMIN_UI_ENABLED")
    if _raw_ui is not None:
        _admin_ui_on = _raw_ui.lower() == "true"
    else:
        try:
            _settings = get_settings()
            _admin_ui_on = _settings.admin_ui_enabled
        except Exception:
            _admin_ui_on = False

    app = FastAPI(
        title="RouteIQ Gateway",
        description="Cloud-Native AI Gateway with Intelligent Routing",
        version=_get_version(),
        lifespan=_routeiq_lifespan,
        docs_url="/docs" if _admin_ui_on else None,
    )

    # Tag the app so downstream code can detect ADR-0012 mode
    app.state.routeiq_own_app = True
    app.state.use_plugin_strategy = _use_plugin_strategy()

    # --- Resilience (innermost middleware) ---
    if enable_resilience:
        add_backpressure_middleware(app)
        app.state.graceful_shutdown = lambda timeout=None: graceful_shutdown(
            app, timeout
        )
        logger.debug("Resilience middleware and drain manager attached (innermost)")

    # --- Remaining middleware (outermost layers) ---
    _configure_middleware(app)

    # --- Routes ---
    _register_routes(app, include_admin=include_admin_routes)

    # --- Admin UI static files (after routes so API routes take priority) ---
    _mount_admin_ui(app)

    # --- Anthropic Messages bare-path surface (must precede the /v1 mount) ---
    # Upstream LiteLLM registers the Anthropic Messages family only at the
    # /v1-prefixed path, so mounting it under /v1 would expose it at
    # /v1/v1/messages (404 for Anthropic SDK callers POSTing /v1/messages).
    # Registering these explicit routes on the parent app BEFORE the /v1 mount
    # makes them match first at the bare external /v1/messages path.
    if mount_litellm:
        from ..routes.messages import register_messages_routes

        register_messages_routes(app)

    # --- First-class API surfaces at the bare external path (RouteIQ-e48a) ---
    # The OpenAI/Cohere-style families (Responses, rerank, RAG, batches,
    # vector-store, audio, images) register their bare paths upstream, but the
    # /v1 mount only exposes them under /v1/* (the bare /rerank etc. 404s for
    # SDK callers).  Registering thin delegators on the parent app BEFORE the
    # /v1 mount makes the documented bare paths first-class RouteIQ routes that
    # get the full middleware stack.  Mirrors the Messages bare-path fix above.
    if mount_litellm and _api_surface_bare_paths_enabled():
        from ..routes.api_surfaces import register_api_surface_routes

        register_api_surface_routes(app)

    # --- Mount LiteLLM as sub-application ---
    if mount_litellm:
        try:
            from litellm.proxy.proxy_server import app as litellm_app

            app.mount("/v1", litellm_app)
            logger.info("LiteLLM proxy mounted at /v1/")
        except ImportError:
            logger.warning(
                "LiteLLM proxy not available; running RouteIQ standalone "
                "(mount_litellm=True but litellm package not found)"
            )

    logger.info(
        "Gateway app created (ADR-0012 own-app mode, mount_litellm=%s)",
        mount_litellm,
    )
    return app


def create_app(
    *,
    apply_patch: bool = True,
    include_admin_routes: bool = True,
    enable_plugins: bool = True,
    enable_resilience: bool = True,
) -> FastAPI:
    """
    Configure the LiteLLM proxy's FastAPI app with LLMRouter extensions.

    .. deprecated::
        Use :func:`create_gateway_app` instead (ADR-0012).  This function
        borrows LiteLLM's FastAPI instance and stores lifecycle hooks as
        lambdas on ``app.state``.  ``create_gateway_app()`` owns the app
        and manages lifecycle via a proper lifespan context manager.

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
    warnings.warn(
        "create_app() is deprecated — use create_gateway_app() instead (ADR-0012). "
        "Own-app mode is now the default; this legacy path requires ROUTEIQ_OWN_APP=false.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.info(
        "Using legacy create_app() — LiteLLM owns the FastAPI instance. "
        "Set ROUTEIQ_OWN_APP=true to use RouteIQ-owned app (ADR-0012)."
    )

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

    # Step 9: Set up database pool shutdown hook
    app.state.llmrouter_db_pool_shutdown = close_db_pool
    logger.debug("Database connection pool shutdown hook attached")

    # Step 10: Set up Redis client shutdown hook
    app.state.llmrouter_redis_shutdown = close_async_redis_client
    logger.debug("Redis client shutdown hook attached")

    # Step 11: Set up OIDC lifecycle hooks (no-op when OIDC is disabled)
    def _oidc_setup():
        try:
            from litellm_llmrouter.oidc import setup_oidc

            result = setup_oidc()
            if result and result.enabled:
                logger.info("OIDC module initialized")
            return result
        except ImportError:
            logger.debug("OIDC module not available for setup")
            return None
        except Exception as e:
            logger.warning(f"OIDC setup failed: {e}")
            return None

    def _oidc_reset():
        try:
            from litellm_llmrouter.oidc import reset_oidc

            reset_oidc()
            logger.debug("OIDC module state reset")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"OIDC reset failed: {e}")

    app.state.llmrouter_oidc_setup = _oidc_setup
    app.state.llmrouter_oidc_shutdown = _oidc_reset
    logger.debug("OIDC lifecycle hooks attached")

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

        # Initialize OIDC if configured
        try:
            from litellm_llmrouter.oidc import setup_oidc

            setup_oidc()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"OIDC setup failed during standalone startup: {e}")

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
            # Reset OIDC state
            try:
                from litellm_llmrouter.oidc import reset_oidc

                reset_oidc()
            except ImportError:
                pass
            # Close database connection pool
            await close_db_pool()
            # Close Redis client singleton
            await close_async_redis_client()
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
