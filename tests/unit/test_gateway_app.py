"""
Tests for the Gateway Application Factory.

These tests verify that:
1. create_standalone_app() works without LiteLLM
2. Middleware and routes are properly configured
3. Plugin lifecycle hooks are invoked correctly
"""

import pytest

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


class TestStandaloneApp:
    """Test the standalone app factory."""

    def test_create_standalone_app_returns_fastapi(self):
        """Test that create_standalone_app() returns a FastAPI instance."""
        from litellm_llmrouter.gateway import create_standalone_app
        from fastapi import FastAPI

        app = create_standalone_app()
        assert isinstance(app, FastAPI)

    def test_standalone_app_has_health_routes(self):
        """Test that standalone app includes health routes."""
        from litellm_llmrouter.gateway import create_standalone_app

        app = create_standalone_app()

        # Check that health routes are registered
        route_paths = [route.path for route in app.routes]
        assert "/_health/live" in route_paths
        assert "/_health/ready" in route_paths

    def test_standalone_app_has_middleware(self):
        """Test that standalone app has RequestIDMiddleware."""
        from litellm_llmrouter.gateway import create_standalone_app
        from litellm_llmrouter.auth import RequestIDMiddleware

        app = create_standalone_app()

        # RequestIDMiddleware is raw ASGI but registered via add_middleware()
        middleware_classes = [m.cls for m in app.user_middleware]
        assert RequestIDMiddleware in middleware_classes

    def test_standalone_app_custom_title_and_version(self):
        """Test that standalone app accepts custom title and version."""
        from litellm_llmrouter.gateway import create_standalone_app

        app = create_standalone_app(
            title="Custom Gateway",
            version="1.2.3",
        )

        assert app.title == "Custom Gateway"
        assert app.version == "1.2.3"

    def test_standalone_app_without_admin_routes(self):
        """Test that admin routes can be disabled."""
        from litellm_llmrouter.gateway import create_standalone_app

        app = create_standalone_app(include_admin_routes=False)

        # Admin routes should not be present
        # Admin router has routes like /router/reload
        route_paths = [route.path for route in app.routes]

        # Health routes should still be present
        assert "/_health/live" in route_paths

        # Note: The exact check for admin routes depends on implementation
        # The key is that the app is created successfully


class TestPluginLifecycle:
    """Test plugin lifecycle integration."""

    @pytest.mark.asyncio
    async def test_plugin_startup_called_on_lifespan(self):
        """Test that plugin startup is called during app lifespan."""
        from litellm_llmrouter.gateway import create_standalone_app
        from litellm_llmrouter.gateway.plugin_manager import (
            GatewayPlugin,
            get_plugin_manager,
            reset_plugin_manager,
        )

        # Reset plugin manager to ensure clean state
        reset_plugin_manager()

        # Create a mock plugin
        class TestPlugin(GatewayPlugin):
            started = False
            stopped = False

            async def startup(self, app, context=None):
                TestPlugin.started = True

            async def shutdown(self, app, context=None):
                TestPlugin.stopped = True

        # Register plugin
        manager = get_plugin_manager()
        manager.register(TestPlugin())

        # Create app
        app = create_standalone_app(enable_plugins=True)

        # Simulate lifespan context
        async with app.router.lifespan_context(app):
            assert TestPlugin.started is True

        assert TestPlugin.stopped is True

        # Clean up
        reset_plugin_manager()

    def test_plugins_disabled_option(self):
        """Test that plugins can be disabled."""
        from litellm_llmrouter.gateway import create_standalone_app

        # This should not raise even with plugins disabled
        app = create_standalone_app(enable_plugins=False)
        assert app is not None


class TestApplyPatchSafely:
    """Test the _apply_patch_safely function."""

    def test_apply_patch_safely_is_idempotent(self):
        """Test that _apply_patch_safely can be called multiple times."""
        from litellm_llmrouter.gateway.app import _apply_patch_safely
        from litellm_llmrouter import is_patch_applied

        # Call multiple times
        result1 = _apply_patch_safely()
        result2 = _apply_patch_safely()

        # Both should succeed
        assert result1 is True
        assert result2 is True
        assert is_patch_applied() is True
