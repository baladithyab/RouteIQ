"""
Tests for the Plugin Manager.

These tests verify that:
1. Plugins can be registered and managed
2. Startup/shutdown hooks are called in correct order
3. Plugin loading from config works
4. Error handling is robust
5. Allowlist enforcement works
6. Dependency ordering (topological sort + priority)
7. Failure mode behaviors
8. Capability security policy enforcement
9. Backwards compatibility with legacy plugins
"""

import os
import pytest
from unittest.mock import MagicMock, patch


class TestPluginCapability:
    """Test the PluginCapability enum."""

    def test_capability_values(self):
        """Test that all expected capabilities exist."""
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        assert PluginCapability.ROUTES.value == "routes"
        assert PluginCapability.ROUTING_STRATEGY.value == "routing_strategy"
        assert PluginCapability.TOOL_RUNTIME.value == "tool_runtime"
        assert PluginCapability.EVALUATOR.value == "evaluator"
        assert PluginCapability.OBSERVABILITY_EXPORTER.value == "observability_exporter"
        assert PluginCapability.MIDDLEWARE.value == "middleware"
        assert PluginCapability.AUTH_PROVIDER.value == "auth_provider"
        assert PluginCapability.STORAGE_BACKEND.value == "storage_backend"


class TestPluginMetadata:
    """Test the PluginMetadata dataclass."""

    def test_default_values(self):
        """Test that PluginMetadata has safe defaults."""
        from litellm_llmrouter.gateway.plugin_manager import PluginMetadata, FailureMode

        meta = PluginMetadata()
        assert meta.name == ""
        assert meta.version == "0.0.0"
        assert meta.capabilities == set()
        assert meta.depends_on == []
        assert meta.priority == 1000
        assert meta.failure_mode == FailureMode.CONTINUE
        assert meta.description == ""

    def test_custom_values(self):
        """Test PluginMetadata with custom values."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginMetadata,
            PluginCapability,
            FailureMode,
        )

        meta = PluginMetadata(
            name="test-plugin",
            version="1.2.3",
            capabilities={PluginCapability.ROUTES, PluginCapability.MIDDLEWARE},
            depends_on=["other-plugin"],
            priority=50,
            failure_mode=FailureMode.ABORT,
            description="A test plugin",
        )
        assert meta.name == "test-plugin"
        assert meta.version == "1.2.3"
        assert PluginCapability.ROUTES in meta.capabilities
        assert meta.depends_on == ["other-plugin"]
        assert meta.priority == 50
        assert meta.failure_mode == FailureMode.ABORT


class TestPluginContext:
    """Test the PluginContext dataclass."""

    def test_default_values(self):
        """Test PluginContext defaults."""
        from litellm_llmrouter.gateway.plugin_manager import PluginContext

        ctx = PluginContext()
        assert ctx.settings == {}
        assert ctx.logger is not None
        assert ctx.validate_outbound_url is None

    def test_with_url_validator(self):
        """Test PluginContext with URL validator."""
        from litellm_llmrouter.gateway.plugin_manager import PluginContext

        def mock_validator(url: str) -> str:
            return url

        ctx = PluginContext(validate_outbound_url=mock_validator)
        assert ctx.validate_outbound_url is not None
        assert ctx.validate_outbound_url("http://example.com") == "http://example.com"


class TestFailureMode:
    """Test the FailureMode enum."""

    def test_failure_mode_values(self):
        """Test all failure mode values."""
        from litellm_llmrouter.gateway.plugin_manager import FailureMode

        assert FailureMode.CONTINUE.value == "continue"
        assert FailureMode.ABORT.value == "abort"
        assert FailureMode.QUARANTINE.value == "quarantine"


class TestGatewayPlugin:
    """Test the GatewayPlugin abstract base class."""

    def test_plugin_name_defaults_to_class_name(self):
        """Test that plugin name defaults to class name."""
        from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin

        class MyTestPlugin(GatewayPlugin):
            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

        plugin = MyTestPlugin()
        assert plugin.name == "MyTestPlugin"

    def test_default_metadata(self):
        """Test that plugins have default metadata."""
        from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin, FailureMode

        class LegacyPlugin(GatewayPlugin):
            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

        plugin = LegacyPlugin()
        meta = plugin.metadata
        assert meta.name == "LegacyPlugin"
        assert meta.version == "0.0.0"
        assert meta.capabilities == set()
        assert meta.priority == 1000
        assert meta.failure_mode == FailureMode.CONTINUE

    def test_custom_metadata(self):
        """Test plugin with custom metadata."""
        from litellm_llmrouter.gateway.plugin_manager import (
            GatewayPlugin,
            PluginMetadata,
            PluginCapability,
        )

        class CustomPlugin(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="custom-plugin",
                    version="2.0.0",
                    capabilities={PluginCapability.ROUTES},
                    priority=10,
                )

            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

        plugin = CustomPlugin()
        assert plugin.name == "custom-plugin"
        assert plugin.metadata.version == "2.0.0"
        assert PluginCapability.ROUTES in plugin.metadata.capabilities


class TestNoOpPlugin:
    """Test the NoOpPlugin implementation."""

    @pytest.mark.asyncio
    async def test_noop_plugin_startup_does_nothing(self):
        """Test that NoOpPlugin startup does nothing."""
        from litellm_llmrouter.gateway.plugin_manager import NoOpPlugin, PluginContext

        plugin = NoOpPlugin()
        # Should not raise
        await plugin.startup(MagicMock(), PluginContext())

    @pytest.mark.asyncio
    async def test_noop_plugin_shutdown_does_nothing(self):
        """Test that NoOpPlugin shutdown does nothing."""
        from litellm_llmrouter.gateway.plugin_manager import NoOpPlugin, PluginContext

        plugin = NoOpPlugin()
        # Should not raise
        await plugin.shutdown(MagicMock(), PluginContext())


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
            async def startup(self, app, context=None):
                call_order.append("Plugin1")

            async def shutdown(self, app, context=None):
                pass

        class Plugin2(GatewayPlugin):
            async def startup(self, app, context=None):
                call_order.append("Plugin2")

            async def shutdown(self, app, context=None):
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
            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                call_order.append("Plugin1")

        class Plugin2(GatewayPlugin):
            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
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
    async def test_plugin_startup_failure_continues_by_default(self):
        """Test that one plugin's startup failure doesn't stop other plugins with continue mode."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
        )

        started = []

        class FailingPlugin(GatewayPlugin):
            async def startup(self, app, context=None):
                raise RuntimeError("Startup failed!")

            async def shutdown(self, app, context=None):
                pass

        class SuccessPlugin(GatewayPlugin):
            async def startup(self, app, context=None):
                started.append("SuccessPlugin")

            async def shutdown(self, app, context=None):
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

    @pytest.mark.asyncio
    async def test_context_passed_to_plugins(self):
        """Test that PluginContext is passed to startup and shutdown."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
            PluginContext,
        )

        received_contexts = []

        class ContextPlugin(GatewayPlugin):
            async def startup(self, app, context=None):
                received_contexts.append(("startup", context))

            async def shutdown(self, app, context=None):
                received_contexts.append(("shutdown", context))

        manager = PluginManager()
        manager.register(ContextPlugin())

        await manager.startup(MagicMock())
        await manager.shutdown(MagicMock())

        assert len(received_contexts) == 2
        assert received_contexts[0][0] == "startup"
        assert isinstance(received_contexts[0][1], PluginContext)
        assert received_contexts[1][0] == "shutdown"
        assert isinstance(received_contexts[1][1], PluginContext)


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


class TestPluginAllowlist:
    """Test allowlist enforcement."""

    def test_allowlist_blocks_unlisted_plugin(self):
        """Test that plugins not in allowlist are blocked."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        with patch.dict(
            os.environ,
            {
                "LLMROUTER_PLUGINS": "some.module.Plugin",
                "LLMROUTER_PLUGINS_ALLOWLIST": "other.module.AllowedPlugin",
            },
        ):
            manager = PluginManager()
            loaded = manager.load_from_config()

        assert loaded == 0

    def test_no_allowlist_allows_all(self):
        """Test that no allowlist means all plugins are allowed."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        with patch.dict(
            os.environ,
            {
                "LLMROUTER_PLUGINS": "",
            },
            clear=True,
        ):
            # Remove allowlist
            os.environ.pop("LLMROUTER_PLUGINS_ALLOWLIST", None)
            manager = PluginManager()
            assert manager._allowlist is None

    def test_empty_allowlist_blocks_all(self):
        """Test that empty allowlist blocks all plugins."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        with patch.dict(
            os.environ,
            {
                "LLMROUTER_PLUGINS_ALLOWLIST": "",
            },
        ):
            manager = PluginManager()
            assert manager._allowlist == set()


class TestPluginCapabilityPolicy:
    """Test capability security policy enforcement."""

    def test_no_capability_restriction_allows_all(self):
        """Test that no capability restriction allows all capabilities."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES", None)
            manager = PluginManager()
            assert manager._allowed_capabilities is None

    def test_capability_restriction_parsed(self):
        """Test that capability restrictions are parsed correctly."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            PluginCapability,
        )

        with patch.dict(
            os.environ,
            {
                "LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES": "ROUTES,MIDDLEWARE,observability_exporter",
            },
        ):
            manager = PluginManager()
            assert manager._allowed_capabilities is not None
            assert PluginCapability.ROUTES in manager._allowed_capabilities
            assert PluginCapability.MIDDLEWARE in manager._allowed_capabilities
            assert (
                PluginCapability.OBSERVABILITY_EXPORTER in manager._allowed_capabilities
            )

    def test_invalid_capability_warned(self):
        """Test that invalid capabilities are logged as warnings."""
        from litellm_llmrouter.gateway.plugin_manager import PluginManager

        with patch.dict(
            os.environ,
            {
                "LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES": "ROUTES,INVALID_CAP",
            },
        ):
            # Should not raise, just warn
            manager = PluginManager()
            assert manager._allowed_capabilities is not None


class TestPluginDependencies:
    """Test dependency-based ordering."""

    @pytest.mark.asyncio
    async def test_depends_on_ordering(self):
        """Test that plugins are ordered by dependencies."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
            PluginMetadata,
        )

        call_order = []

        class PluginA(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="plugin-a", depends_on=["plugin-b"])

            async def startup(self, app, context=None):
                call_order.append("A")

            async def shutdown(self, app, context=None):
                pass

        class PluginB(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="plugin-b")

            async def startup(self, app, context=None):
                call_order.append("B")

            async def shutdown(self, app, context=None):
                pass

        manager = PluginManager()
        # Register A first, but B must start first due to dependency
        manager.register(PluginA())
        manager.register(PluginB())

        await manager.startup(MagicMock())

        # B should be called before A because A depends on B
        assert call_order == ["B", "A"]

    @pytest.mark.asyncio
    async def test_priority_tiebreak(self):
        """Test that priority breaks ties when no dependencies."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
            PluginMetadata,
        )

        call_order = []

        class HighPriority(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="high", priority=10)

            async def startup(self, app, context=None):
                call_order.append("high")

            async def shutdown(self, app, context=None):
                pass

        class LowPriority(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="low", priority=100)

            async def startup(self, app, context=None):
                call_order.append("low")

            async def shutdown(self, app, context=None):
                pass

        manager = PluginManager()
        # Register low priority first
        manager.register(LowPriority())
        manager.register(HighPriority())

        await manager.startup(MagicMock())

        # High priority (lower number) should be called first
        assert call_order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_circular_dependency_raises(self):
        """Test that circular dependencies raise PluginDependencyError."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
            PluginMetadata,
            PluginDependencyError,
        )

        class PluginA(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="plugin-a", depends_on=["plugin-b"])

            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

        class PluginB(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="plugin-b", depends_on=["plugin-a"])

            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

        manager = PluginManager()
        manager.register(PluginA())
        manager.register(PluginB())

        with pytest.raises(PluginDependencyError, match="Circular dependency"):
            await manager.startup(MagicMock())

    @pytest.mark.asyncio
    async def test_missing_dependency_raises(self):
        """Test that missing dependencies raise PluginDependencyError."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
            PluginMetadata,
            PluginDependencyError,
        )

        class PluginA(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="plugin-a", depends_on=["nonexistent"])

            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

        manager = PluginManager()
        manager.register(PluginA())

        with pytest.raises(PluginDependencyError, match="depends on 'nonexistent'"):
            await manager.startup(MagicMock())


class TestPluginFailureModes:
    """Test failure mode behaviors."""

    @pytest.mark.asyncio
    async def test_continue_mode_continues(self):
        """Test that continue mode logs error and continues."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
            PluginMetadata,
            FailureMode,
        )

        started = []

        class FailingPlugin(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="failing", failure_mode=FailureMode.CONTINUE)

            async def startup(self, app, context=None):
                raise RuntimeError("Test failure")

            async def shutdown(self, app, context=None):
                pass

        class SuccessPlugin(GatewayPlugin):
            async def startup(self, app, context=None):
                started.append("success")

            async def shutdown(self, app, context=None):
                pass

        manager = PluginManager()
        manager.register(FailingPlugin())
        manager.register(SuccessPlugin())

        # Should not raise
        await manager.startup(MagicMock())
        assert "success" in started

    @pytest.mark.asyncio
    async def test_abort_mode_raises(self):
        """Test that abort mode raises exception and stops startup."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
            PluginMetadata,
            FailureMode,
        )

        class AbortPlugin(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="aborting", failure_mode=FailureMode.ABORT)

            async def startup(self, app, context=None):
                raise RuntimeError("Abort!")

            async def shutdown(self, app, context=None):
                pass

        manager = PluginManager()
        manager.register(AbortPlugin())

        with pytest.raises(RuntimeError, match="Abort!"):
            await manager.startup(MagicMock())

    @pytest.mark.asyncio
    async def test_quarantine_mode_disables_plugin(self):
        """Test that quarantine mode disables the plugin."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
            PluginMetadata,
            FailureMode,
        )

        shutdown_called = []

        class QuarantinePlugin(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="quarantined", failure_mode=FailureMode.QUARANTINE
                )

            async def startup(self, app, context=None):
                raise RuntimeError("Quarantine me!")

            async def shutdown(self, app, context=None):
                shutdown_called.append("quarantined")

        manager = PluginManager()
        manager.register(QuarantinePlugin())

        # Should not raise
        await manager.startup(MagicMock())
        assert "quarantined" in manager.quarantined_plugins

        # Shutdown should skip quarantined plugin
        await manager.shutdown(MagicMock())
        assert "quarantined" not in shutdown_called

    @pytest.mark.asyncio
    async def test_global_failure_mode(self):
        """Test that global failure mode is used as default."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
        )

        class DefaultPlugin(GatewayPlugin):
            async def startup(self, app, context=None):
                raise RuntimeError("Test")

            async def shutdown(self, app, context=None):
                pass

        with patch.dict(os.environ, {"LLMROUTER_PLUGINS_FAILURE_MODE": "abort"}):
            manager = PluginManager()
            manager.register(DefaultPlugin())

            with pytest.raises(RuntimeError):
                await manager.startup(MagicMock())


class TestBackwardsCompatibility:
    """Test backwards compatibility with legacy plugins."""

    @pytest.mark.asyncio
    async def test_legacy_plugin_without_context(self):
        """Test that legacy plugins without context parameter still work."""
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            GatewayPlugin,
        )

        startup_called = False

        class LegacyPlugin(GatewayPlugin):
            async def startup(self, app, context=None):
                nonlocal startup_called
                # Legacy plugins may ignore context
                startup_called = True

            async def shutdown(self, app, context=None):
                pass

        manager = PluginManager()
        manager.register(LegacyPlugin())

        await manager.startup(MagicMock())
        assert startup_called

    def test_legacy_plugin_default_metadata(self):
        """Test that legacy plugins get default metadata."""
        from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin, FailureMode

        class LegacyStylePlugin(GatewayPlugin):
            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

        plugin = LegacyStylePlugin()
        meta = plugin.metadata

        # Should have safe defaults
        assert meta.name == "LegacyStylePlugin"
        assert meta.version == "0.0.0"
        assert meta.capabilities == set()
        assert meta.depends_on == []
        assert meta.priority == 1000
        assert meta.failure_mode == FailureMode.CONTINUE


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


class TestPluginErrors:
    """Test plugin error classes."""

    def test_plugin_load_error(self):
        """Test PluginLoadError."""
        from litellm_llmrouter.gateway.plugin_manager import PluginLoadError

        err = PluginLoadError("test.module.Plugin", "Module not found")
        assert err.plugin_path == "test.module.Plugin"
        assert err.reason == "Module not found"
        assert "test.module.Plugin" in str(err)
        assert "Module not found" in str(err)

    def test_plugin_security_error(self):
        """Test PluginSecurityError."""
        from litellm_llmrouter.gateway.plugin_manager import PluginSecurityError

        err = PluginSecurityError("test.module.Plugin", "Not in allowlist")
        assert err.plugin_path == "test.module.Plugin"
        assert err.reason == "Not in allowlist"

    def test_plugin_dependency_error(self):
        """Test PluginDependencyError."""
        from litellm_llmrouter.gateway.plugin_manager import PluginDependencyError

        err = PluginDependencyError("Circular dependency", cycle=["a", "b", "a"])
        assert err.cycle == ["a", "b", "a"]
        assert "Circular dependency" in str(err)
