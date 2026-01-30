"""
Tests for the Plugin Manager.

These tests verify that:
1. Plugins can be registered and managed
2. Startup/shutdown hooks are called in correct order
3. Plugin loading from config works
4. Error handling is robust
"""

import os
import pytest
from unittest.mock import MagicMock, patch


class TestGatewayPlugin:
    """Test the GatewayPlugin abstract base class."""

    def test_plugin_name_defaults_to_class_name(self):
        """Test that plugin name defaults to class name."""
        from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin

        class MyTestPlugin(GatewayPlugin):
            async def startup(self, app):
                pass

            async def shutdown(self, app):
                pass

        plugin = MyTestPlugin()
        assert plugin.name == "MyTestPlugin"


class TestNoOpPlugin:
    """Test the NoOpPlugin implementation."""

    @pytest.mark.asyncio
    async def test_noop_plugin_startup_does_nothing(self):
        """Test that NoOpPlugin startup does nothing."""
        from litellm_llmrouter.gateway.plugin_manager import NoOpPlugin

        plugin = NoOpPlugin()
        # Should not raise
        await plugin.startup(MagicMock())

    @pytest.mark.asyncio
    async def test_noop_plugin_shutdown_does_nothing(self):
        """Test that NoOpPlugin shutdown does nothing."""
        from litellm_llmrouter.gateway.plugin_manager import NoOpPlugin

        plugin = NoOpPlugin()
        # Should not raise
        await plugin.shutdown(MagicMock())


class TestPluginManager:
    """Test the PluginManager class."""

    def test_register_plugin(self):
        """Test that plugins can be registered."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            NoOpPlugin,
        )

        manager = PluginManager()
        plugin = NoOpPlugin()

        manager.register(plugin)

        assert plugin in manager.plugins
        assert len(manager.plugins) == 1

    def test_register_multiple_plugins(self):
        """Test that multiple plugins can be registered."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            NoOpPlugin,
        )

        manager = PluginManager()
        plugin1 = NoOpPlugin()
        plugin2 = NoOpPlugin()

        manager.register(plugin1)
        manager.register(plugin2)

        assert len(manager.plugins) == 2

    def test_cannot_register_after_startup(self):
        """Test that registration fails after startup is called."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            NoOpPlugin,
        )
        import asyncio

        manager = PluginManager()
        asyncio.run(manager.startup(MagicMock()))

        with pytest.raises(RuntimeError, match="Cannot register plugin"):
            manager.register(NoOpPlugin())

    @pytest.mark.asyncio
    async def test_startup_calls_plugins_in_order(self):
        """Test that startup calls plugins in registration order."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
        )

        call_order = []

        class Plugin1(GatewayPlugin):
            async def startup(self, app):
                call_order.append("Plugin1")

            async def shutdown(self, app):
                pass

        class Plugin2(GatewayPlugin):
            async def startup(self, app):
                call_order.append("Plugin2")

            async def shutdown(self, app):
                pass

        manager = PluginManager()
        manager.register(Plugin1())
        manager.register(Plugin2())

        await manager.startup(MagicMock())

        assert call_order == ["Plugin1", "Plugin2"]

    @pytest.mark.asyncio
    async def test_shutdown_calls_plugins_in_reverse_order(self):
        """Test that shutdown calls plugins in reverse registration order."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
        )

        call_order = []

        class Plugin1(GatewayPlugin):
            async def startup(self, app):
                pass

            async def shutdown(self, app):
                call_order.append("Plugin1")

        class Plugin2(GatewayPlugin):
            async def startup(self, app):
                pass

            async def shutdown(self, app):
                call_order.append("Plugin2")

        manager = PluginManager()
        manager.register(Plugin1())
        manager.register(Plugin2())

        await manager.startup(MagicMock())
        await manager.shutdown(MagicMock())

        # Reverse order
        assert call_order == ["Plugin2", "Plugin1"]

    @pytest.mark.asyncio
    async def test_startup_is_idempotent(self):
        """Test that calling startup multiple times logs warning."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        manager = PluginManager()

        # First call
        await manager.startup(MagicMock())
        assert manager.is_started is True

        # Second call should not raise
        await manager.startup(MagicMock())
        assert manager.is_started is True

    @pytest.mark.asyncio
    async def test_shutdown_without_startup_warns(self):
        """Test that calling shutdown before startup logs warning."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        manager = PluginManager()

        # Should not raise
        await manager.shutdown(MagicMock())

    @pytest.mark.asyncio
    async def test_plugin_startup_failure_does_not_stop_others(self):
        """Test that one plugin's startup failure doesn't stop other plugins."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
        )

        started = []

        class FailingPlugin(GatewayPlugin):
            async def startup(self, app):
                raise RuntimeError("Startup failed!")

            async def shutdown(self, app):
                pass

        class SuccessPlugin(GatewayPlugin):
            async def startup(self, app):
                started.append("SuccessPlugin")

            async def shutdown(self, app):
                pass

        manager = PluginManager()
        manager.register(FailingPlugin())
        manager.register(SuccessPlugin())

        # Should not raise
        await manager.startup(MagicMock())

        # Second plugin should still have started
        assert "SuccessPlugin" in started

    def test_plugins_property_returns_copy(self):
        """Test that plugins property returns a copy of the list."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            NoOpPlugin,
        )

        manager = PluginManager()
        plugin = NoOpPlugin()
        manager.register(plugin)

        plugins = manager.plugins
        plugins.append(NoOpPlugin())

        # Original list should be unchanged
        assert len(manager.plugins) == 1


class TestPluginManagerConfig:
    """Test plugin loading from configuration."""

    def test_load_from_config_no_env_var(self):
        """Test that load_from_config returns 0 when no env var set."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        manager = PluginManager()

        with patch.dict(os.environ, {}, clear=True):
            # Ensure LLMROUTER_PLUGINS is not set
            os.environ.pop("LLMROUTER_PLUGINS", None)
            loaded = manager.load_from_config()

        assert loaded == 0

    def test_load_from_config_empty_env_var(self):
        """Test that load_from_config handles empty env var."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        manager = PluginManager()

        with patch.dict(os.environ, {"LLMROUTER_PLUGINS": ""}):
            loaded = manager.load_from_config()

        assert loaded == 0

    def test_load_from_config_invalid_path(self):
        """Test that load_from_config handles invalid plugin path."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        manager = PluginManager()

        with patch.dict(os.environ, {"LLMROUTER_PLUGINS": "nonexistent.module.Plugin"}):
            loaded = manager.load_from_config()

        # Should not raise, just return 0
        assert loaded == 0


class TestGetPluginManager:
    """Test the get_plugin_manager singleton."""

    def test_get_plugin_manager_returns_same_instance(self):
        """Test that get_plugin_manager returns the same instance."""
        from litellm_llmrouter.gateway.plugin_manager import (
            get_plugin_manager,
            reset_plugin_manager,
        )

        # Reset to ensure clean state
        reset_plugin_manager()

        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()

        assert manager1 is manager2

    def test_reset_plugin_manager(self):
        """Test that reset_plugin_manager creates a new instance."""
        from litellm_llmrouter.gateway.plugin_manager import (
            get_plugin_manager,
            reset_plugin_manager,
        )

        manager1 = get_plugin_manager()
        reset_plugin_manager()
        manager2 = get_plugin_manager()

        assert manager1 is not manager2
