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

Usage:
    from litellm_llmrouter import register_llmrouter_strategies

    # Register all LLMRouter strategies with LiteLLM
    register_llmrouter_strategies()
"""

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

__version__ = "0.1.0"
__all__ = [
    # Strategies
    "LLMRouterStrategyFamily",
    "register_llmrouter_strategies",
    "LLMROUTER_STRATEGIES",
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
]
