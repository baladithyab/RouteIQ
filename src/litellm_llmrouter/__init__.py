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

Legacy Usage (DEPRECATED — monkey-patch approach):
    # Only used when ROUTEIQ_USE_PLUGIN_STRATEGY=false
    from litellm_llmrouter import patch_litellm_router
    patch_litellm_router()  # emits DeprecationWarning

A/B Testing:
    from litellm_llmrouter import get_routing_registry, get_routing_pipeline

    # Set A/B weights for strategy testing
    registry = get_routing_registry()
    registry.set_weights({"baseline": 90, "candidate": 10})

Note:
    Importing this module does NOT apply any monkey patches automatically.
    The gateway factory (``create_app()``) handles routing strategy installation.
    The plugin strategy (``ROUTEIQ_USE_PLUGIN_STRATEGY=true``) is the default
    and recommended path. The legacy monkey-patch is deprecated.

Build: Migrated CI to uv for faster package management (2026-01-26)
"""

# Legacy routing strategy patch — DEPRECATED.
# Prefer the plugin-based strategy in custom_routing_strategy.py
# (ROUTEIQ_USE_PLUGIN_STRATEGY=true, the default).
# These exports are retained for backward compatibility only.
from .routing_strategy_patch import (
    patch_litellm_router,  # deprecated — use install_routeiq_strategy() instead
    unpatch_litellm_router,  # deprecated
    is_patch_applied,  # deprecated — plugin strategy doesn't use patches
    is_pipeline_routing_enabled,  # deprecated
)

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
    __version__ = "0.2.0"  # fallback if not installed as package

__all__ = [
    # Router patch — DEPRECATED (retained for backward compatibility)
    # Prefer RouteIQRoutingStrategy / install_routeiq_strategy() instead.
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
