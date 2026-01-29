"""
Plugin Manager for Gateway Extensibility
=========================================

This module provides a minimal plugin lifecycle interface for extending the gateway.

Plugins can hook into:
- startup(app): Called after the FastAPI app is created but before server starts
- shutdown(app): Called during application shutdown

Usage:
    from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin, PluginManager

    class MyPlugin(GatewayPlugin):
        async def startup(self, app):
            print("Plugin starting up!")

        async def shutdown(self, app):
            print("Plugin shutting down!")

    manager = PluginManager()
    manager.register(MyPlugin())
    await manager.startup(app)

Configuration:
    Environment variables:
    - LLMROUTER_PLUGINS: Comma-separated list of plugin module paths to load
      Example: LLMROUTER_PLUGINS=mypackage.plugin1,mypackage.plugin2
"""

import importlib
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Singleton instance
_plugin_manager: "PluginManager | None" = None


class GatewayPlugin(ABC):
    """
    Abstract base class for gateway plugins.

    Plugins must implement startup() and shutdown() hooks.
    Both methods receive the FastAPI application instance.
    """

    @property
    def name(self) -> str:
        """Return the plugin name (defaults to class name)."""
        return self.__class__.__name__

    @abstractmethod
    async def startup(self, app: "FastAPI") -> None:
        """
        Called during application startup.

        Use this to register routes, middleware, or initialize resources.

        Args:
            app: The FastAPI application instance
        """
        pass

    @abstractmethod
    async def shutdown(self, app: "FastAPI") -> None:
        """
        Called during application shutdown.

        Use this to clean up resources, close connections, etc.

        Args:
            app: The FastAPI application instance
        """
        pass


class NoOpPlugin(GatewayPlugin):
    """
    A no-op plugin that does nothing.

    Useful as a placeholder or for testing.
    """

    async def startup(self, app: "FastAPI") -> None:
        """No-op startup."""
        pass

    async def shutdown(self, app: "FastAPI") -> None:
        """No-op shutdown."""
        pass


class PluginManager:
    """
    Manages the lifecycle of gateway plugins.

    The plugin manager:
    - Loads plugins from configuration (env var LLMROUTER_PLUGINS)
    - Calls startup hooks in registration order
    - Calls shutdown hooks in reverse registration order
    """

    def __init__(self) -> None:
        self._plugins: list[GatewayPlugin] = []
        self._started = False

    def register(self, plugin: GatewayPlugin) -> None:
        """
        Register a plugin with the manager.

        Args:
            plugin: The plugin instance to register

        Raises:
            RuntimeError: If called after startup() has been invoked
        """
        if self._started:
            raise RuntimeError(
                f"Cannot register plugin {plugin.name} after startup() has been called"
            )
        self._plugins.append(plugin)
        logger.info(f"Registered plugin: {plugin.name}")

    def load_from_config(self) -> int:
        """
        Load plugins from the LLMROUTER_PLUGINS environment variable.

        Returns:
            Number of plugins successfully loaded

        Note:
            Plugin paths should be fully qualified module paths containing
            a class that inherits from GatewayPlugin.
            Example: LLMROUTER_PLUGINS=mypackage.myplugin.MyPlugin
        """
        plugins_str = os.getenv("LLMROUTER_PLUGINS", "").strip()
        if not plugins_str:
            logger.debug("No plugins configured via LLMROUTER_PLUGINS")
            return 0

        loaded = 0
        for plugin_path in plugins_str.split(","):
            plugin_path = plugin_path.strip()
            if not plugin_path:
                continue

            try:
                plugin = self._load_plugin(plugin_path)
                if plugin:
                    self.register(plugin)
                    loaded += 1
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_path}: {e}")

        return loaded

    def _load_plugin(self, plugin_path: str) -> GatewayPlugin | None:
        """
        Load a plugin from a module path.

        Args:
            plugin_path: Fully qualified path to the plugin class
                        (e.g., 'mypackage.myplugin.MyPlugin')

        Returns:
            Plugin instance or None if loading failed
        """
        try:
            # Split module path and class name
            if "." not in plugin_path:
                logger.error(f"Invalid plugin path: {plugin_path} (no class name)")
                return None

            module_path, class_name = plugin_path.rsplit(".", 1)

            # Import the module
            module = importlib.import_module(module_path)

            # Get the class
            plugin_class = getattr(module, class_name)

            # Verify it's a GatewayPlugin subclass
            if not issubclass(plugin_class, GatewayPlugin):
                logger.error(
                    f"Plugin {plugin_path} is not a GatewayPlugin subclass"
                )
                return None

            # Instantiate
            return plugin_class()

        except ImportError as e:
            logger.error(f"Could not import plugin module {plugin_path}: {e}")
            return None
        except AttributeError as e:
            logger.error(f"Plugin class not found in module {plugin_path}: {e}")
            return None

    async def startup(self, app: "FastAPI") -> None:
        """
        Call startup hooks on all registered plugins.

        Plugins are started in registration order.

        Args:
            app: The FastAPI application instance
        """
        if self._started:
            logger.warning("Plugin manager startup() called multiple times")
            return

        self._started = True

        for plugin in self._plugins:
            try:
                logger.debug(f"Starting plugin: {plugin.name}")
                await plugin.startup(app)
                logger.info(f"Plugin started: {plugin.name}")
            except Exception as e:
                logger.error(f"Plugin {plugin.name} startup failed: {e}")
                # Continue with other plugins - don't fail the entire startup

    async def shutdown(self, app: "FastAPI") -> None:
        """
        Call shutdown hooks on all registered plugins.

        Plugins are shut down in reverse registration order.

        Args:
            app: The FastAPI application instance
        """
        if not self._started:
            logger.warning("Plugin manager shutdown() called before startup()")
            return

        # Shutdown in reverse order
        for plugin in reversed(self._plugins):
            try:
                logger.debug(f"Shutting down plugin: {plugin.name}")
                await plugin.shutdown(app)
                logger.info(f"Plugin shut down: {plugin.name}")
            except Exception as e:
                logger.error(f"Plugin {plugin.name} shutdown failed: {e}")
                # Continue with other plugins - don't fail the entire shutdown

        self._started = False

    @property
    def plugins(self) -> list[GatewayPlugin]:
        """Return a copy of the registered plugins list."""
        return list(self._plugins)

    @property
    def is_started(self) -> bool:
        """Return whether startup() has been called."""
        return self._started


def get_plugin_manager() -> PluginManager:
    """
    Get the global plugin manager instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        The global PluginManager instance
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def reset_plugin_manager() -> None:
    """
    Reset the global plugin manager instance.

    Primarily useful for testing.
    """
    global _plugin_manager
    _plugin_manager = None
