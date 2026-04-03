# ruff: noqa: E402  — module-level imports after no-op stub functions (intentional)
"""
LiteLLM + LLMRouter Integration
================================

This module provides the integration layer between LiteLLM's routing
infrastructure and LLMRouter's ML-based routing strategies.

Features:
- ML-powered routing strategies (18+ including KNN, SVM, MLP, ELO, etc.)
- A2A (Agent-to-Agent) gateway support
- MCP (Model Context Protocol) gateway support
- S3/GCS config sync with hot reload
- ETag-based change detection for efficient syncing
- Runtime A/B testing via strategy registry and routing pipeline
- Centroid routing for zero-config intelligent routing (~2ms)

Recommended Usage (Plugin Strategy — default):
    from litellm_llmrouter import install_routeiq_strategy

    # After creating a LiteLLM Router instance:
    install_routeiq_strategy(router, strategy_name="llmrouter-knn")

    # Or use the gateway factory which handles everything automatically:
    from litellm_llmrouter.gateway import create_app
    app = create_app()

Legacy Usage (REMOVED — monkey-patch approach):
    # The monkey-patch module has been deleted.
    # patch_litellm_router() is now a no-op that emits a DeprecationWarning.
    from litellm_llmrouter import patch_litellm_router
    patch_litellm_router()  # no-op, emits DeprecationWarning

A/B Testing:
    from litellm_llmrouter import get_routing_registry, get_routing_pipeline

    # Set A/B weights for strategy testing
    registry = get_routing_registry()
    registry.set_weights({"baseline": 90, "candidate": 10})

Note:
    Importing this module does NOT apply any monkey patches.
    The gateway factory (``create_app()``) handles routing strategy installation.
    The plugin strategy is the only supported routing path.
    The legacy monkey-patch module has been removed.

Build: Migrated CI to uv for faster package management (2026-01-26)
"""

# Legacy routing strategy patch — REMOVED.
# The monkey-patch module (routing_strategy_patch.py) has been deleted.
# These no-op stubs are retained for backward compatibility only.
import logging as _logging
import warnings as _warnings

_legacy_logger = _logging.getLogger(__name__)


def patch_litellm_router() -> bool:
    """No-op stub. The legacy monkey-patch module has been removed."""
    _warnings.warn(
        "patch_litellm_router() has been removed. "
        "The plugin-based routing strategy (RouteIQRoutingStrategy) is now "
        "the only supported path. This call is a no-op.",
        DeprecationWarning,
        stacklevel=2,
    )
    _legacy_logger.warning(
        "REMOVED: patch_litellm_router() called — this is a no-op. "
        "The plugin-based routing strategy is the only supported path."
    )
    return True


def unpatch_litellm_router() -> bool:
    """No-op stub. The legacy monkey-patch module has been removed."""
    _warnings.warn(
        "unpatch_litellm_router() has been removed. "
        "The plugin-based routing strategy does not require patching/unpatching. "
        "This call is a no-op.",
        DeprecationWarning,
        stacklevel=2,
    )
    _legacy_logger.warning(
        "REMOVED: unpatch_litellm_router() called — this is a no-op."
    )
    return True


def is_patch_applied() -> bool:
    """No-op stub. Always returns False — the patch system has been removed."""
    return False


def is_pipeline_routing_enabled() -> bool:
    """No-op stub. Always returns True — pipeline routing is always enabled."""
    return True


from .strategies import (
    LLMRouterStrategyFamily,
    register_llmrouter_strategies,
    LLMROUTER_STRATEGIES,
)
from .config_loader import (
    download_config_from_s3,
    download_config_from_gcs,
    download_model_from_s3,
    download_custom_router_from_s3,
)
from .config_sync import (
    ConfigSyncManager,
    get_sync_manager,
    start_config_sync,
    stop_config_sync,
)
from .hot_reload import (
    HotReloadManager,
    get_hot_reload_manager,
)
from .a2a_gateway import (
    A2AAgent,
    A2AGateway,
    get_a2a_gateway,
)
from .mcp_gateway import (
    MCPServer,
    MCPGateway,
    MCPTransport,
    get_mcp_gateway,
)
from .routes import router as api_router

# Strategy registry for A/B testing and runtime hot-swapping
from .strategy_registry import (
    RoutingStrategy,
    RoutingStrategyRegistry,
    RoutingPipeline,
    RoutingContext,
    RoutingResult,
    DefaultStrategy,
    get_routing_registry,
    get_routing_pipeline,
    reset_routing_singletons,
)

# Gateway composition root
from .gateway import create_app, create_standalone_app
from .gateway.plugin_manager import GatewayPlugin, PluginManager, get_plugin_manager

# Custom routing strategy (plugin-based, replaces monkey-patch)
from .custom_routing_strategy import (
    RouteIQRoutingStrategy,
    create_routeiq_strategy,
    install_routeiq_strategy,
    register_centroid_strategy,
    CENTROID_ROUTING_AVAILABLE,
)

# Centroid routing (zero-config intelligent routing)
try:
    from .centroid_routing import (
        CentroidRoutingStrategy,
        CentroidClassifier,
        AgenticDetector,
        ReasoningDetector,
        SessionCache,
        RoutingProfile,
        ClassificationResult,
        get_centroid_strategy,
        reset_centroid_strategy,
        warmup_centroid_classifier,
    )
except ImportError:
    pass  # centroid routing deps not installed

# Personalized routing (per-user preference learning)
try:
    from .personalized_routing import (
        PersonalizedRouter,
        PreferenceStore,
        UserPreference,
        get_personalized_router,
        reset_personalized_router,
    )
except ImportError:
    pass  # personalized routing deps not installed

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("routeiq")
except Exception:
    __version__ = "1.0.0rc1"  # fallback if not installed as package

__all__ = [
    # Router patch stubs — REMOVED (no-op stubs for backward compatibility)
    "patch_litellm_router",
    "unpatch_litellm_router",
    "is_patch_applied",
    "is_pipeline_routing_enabled",
    # Gateway composition root
    "create_app",
    "create_standalone_app",
    # Plugin lifecycle
    "GatewayPlugin",
    "PluginManager",
    "get_plugin_manager",
    # Custom routing strategy (plugin-based)
    "RouteIQRoutingStrategy",
    "create_routeiq_strategy",
    "install_routeiq_strategy",
    "register_centroid_strategy",
    "CENTROID_ROUTING_AVAILABLE",
    # Centroid routing (zero-config intelligent routing)
    "CentroidRoutingStrategy",
    "CentroidClassifier",
    "AgenticDetector",
    "ReasoningDetector",
    "SessionCache",
    "RoutingProfile",
    "ClassificationResult",
    "get_centroid_strategy",
    "reset_centroid_strategy",
    "warmup_centroid_classifier",
    # Personalized routing (per-user preference learning)
    "PersonalizedRouter",
    "PreferenceStore",
    "UserPreference",
    "get_personalized_router",
    "reset_personalized_router",
    # Strategies
    "LLMRouterStrategyFamily",
    "register_llmrouter_strategies",
    "LLMROUTER_STRATEGIES",
    # Strategy registry (A/B testing)
    "RoutingStrategy",
    "RoutingStrategyRegistry",
    "RoutingPipeline",
    "RoutingContext",
    "RoutingResult",
    "DefaultStrategy",
    "get_routing_registry",
    "get_routing_pipeline",
    "reset_routing_singletons",
    # Config loading
    "download_config_from_s3",
    "download_config_from_gcs",
    "download_model_from_s3",
    "download_custom_router_from_s3",
    # Config sync
    "ConfigSyncManager",
    "get_sync_manager",
    "start_config_sync",
    "stop_config_sync",
    # Hot reload
    "HotReloadManager",
    "get_hot_reload_manager",
    # A2A Gateway
    "A2AAgent",
    "A2AGateway",
    "get_a2a_gateway",
    # MCP Gateway
    "MCPServer",
    "MCPGateway",
    "MCPTransport",
    "get_mcp_gateway",
    # API Router
    "api_router",
]
