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

Usage:
    from litellm_llmrouter import register_llmrouter_strategies, patch_litellm_router

    # Apply the router patch BEFORE creating any Router instances
    # This is explicit and idempotent (safe to call multiple times)
    patch_litellm_router()

    # Register all LLMRouter strategies with LiteLLM
    register_llmrouter_strategies()

A/B Testing:
    from litellm_llmrouter import get_routing_registry, get_routing_pipeline

    # Set A/B weights for strategy testing
    registry = get_routing_registry()
    registry.set_weights({"baseline": 90, "candidate": 10})

Note:
    Importing this module does NOT apply any monkey patches automatically.
    You must call patch_litellm_router() explicitly from your startup code.
    For convenience, use the gateway.create_app() factory which handles this.

Build: Migrated CI to uv for faster package management (2026-01-26)
"""

# Routing strategy patch - NOT auto-applied on import
# Call patch_litellm_router() explicitly from startup
from .routing_strategy_patch import (
    patch_litellm_router,
    unpatch_litellm_router,
    is_patch_applied,
    is_pipeline_routing_enabled,
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

__version__ = "0.0.4"
__all__ = [
    # Router patch (for llmrouter-* strategies)
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
