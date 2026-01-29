"""
Gateway Composition Root
========================

This subpackage provides the gateway application factory and plugin lifecycle management.

Modules:
- app: FastAPI application factory with explicit configuration
- plugin_manager: Plugin lifecycle management for extensibility

Usage:
    from litellm_llmrouter.gateway import create_app

    app = create_app()  # Creates configured FastAPI app
"""

from .app import create_app, create_standalone_app
from .plugin_manager import GatewayPlugin, PluginManager, get_plugin_manager

__all__ = [
    "create_app",
    "create_standalone_app",
    "GatewayPlugin",
    "PluginManager",
    "get_plugin_manager",
]
