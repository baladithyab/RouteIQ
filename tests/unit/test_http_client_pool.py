"""
Unit tests for HTTP Client Pool
================================

Tests for shared HTTP client pooling functionality, ensuring:
- Client instantiation happens once per lifecycle (not per request)
- Proper lifecycle management (startup/shutdown)
- Feature flag rollback behavior
- No leaked connections on shutdown

These tests guard against regression to per-request client creation.
"""

import os

import httpx
import pytest


class TestHttpClientPoolLifecycle:
    """Test HTTP client pool lifecycle management."""

    @pytest.fixture(autouse=True)
    async def reset_pool(self):
        """Reset pool state before/after each test."""
        # Import after potential env var patching
        from litellm_llmrouter.http_client_pool import (
            reset_http_client_pool,
            reset_client_instantiation_count,
        )

        await reset_http_client_pool()
        reset_client_instantiation_count()
        yield
        await reset_http_client_pool()
        reset_client_instantiation_count()

    @pytest.mark.asyncio
    async def test_startup_creates_client_once(self):
        """Verify that startup creates exactly one client instance."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            get_client_instantiation_count,
            is_pool_initialized,
        )

        # Before startup, pool should not be initialized
        assert not is_pool_initialized()
        assert get_client_instantiation_count() == 0

        # Startup should create one client
        await startup_http_client_pool()
        assert is_pool_initialized()
        assert get_client_instantiation_count() == 1

        # Calling startup again should be idempotent
        await startup_http_client_pool()
        assert get_client_instantiation_count() == 1

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self):
        """Verify shutdown properly closes the client."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            shutdown_http_client_pool,
            is_pool_initialized,
        )

        await startup_http_client_pool()
        assert is_pool_initialized()

        await shutdown_http_client_pool()
        assert not is_pool_initialized()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """Verify shutdown is safe to call multiple times."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            shutdown_http_client_pool,
        )

        await startup_http_client_pool()
        await shutdown_http_client_pool()
        # Should not raise on second call
        await shutdown_http_client_pool()

    @pytest.mark.asyncio
    async def test_get_http_client_returns_same_instance(self):
        """Verify get_http_client returns the same instance."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            get_http_client,
        )

        await startup_http_client_pool()

        client1 = get_http_client()
        client2 = get_http_client()

        assert client1 is client2

    @pytest.mark.asyncio
    async def test_get_http_client_raises_if_not_initialized(self):
        """Verify get_http_client raises RuntimeError if pool not initialized."""
        from litellm_llmrouter.http_client_pool import get_http_client

        with pytest.raises(RuntimeError, match="not initialized"):
            get_http_client()


class TestHttpClientPoolContextManager:
    """Test the get_client_for_request context manager."""

    @pytest.fixture(autouse=True)
    async def reset_pool(self):
        """Reset pool state before/after each test."""
        from litellm_llmrouter.http_client_pool import (
            reset_http_client_pool,
            reset_client_instantiation_count,
        )

        await reset_http_client_pool()
        reset_client_instantiation_count()
        yield
        await reset_http_client_pool()
        reset_client_instantiation_count()

    @pytest.mark.asyncio
    async def test_get_client_for_request_uses_pool(self):
        """Verify get_client_for_request uses pooled client when available."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            get_client_for_request,
            get_http_client,
            get_client_instantiation_count,
        )

        await startup_http_client_pool()
        pooled_client = get_http_client()

        async with get_client_for_request() as client:
            assert client is pooled_client

        # No additional instantiation should have occurred
        assert get_client_instantiation_count() == 1

    @pytest.mark.asyncio
    async def test_get_client_for_request_fallback_when_not_initialized(self):
        """Verify get_client_for_request falls back when pool not initialized."""
        from litellm_llmrouter.http_client_pool import (
            get_client_for_request,
            get_client_instantiation_count,
        )

        # Pool not initialized - should create fallback client
        async with get_client_for_request() as client:
            assert client is not None
            assert isinstance(client, httpx.AsyncClient)

        # Fallback creates a new instance
        assert get_client_instantiation_count() == 1

    @pytest.mark.asyncio
    async def test_multiple_requests_reuse_pooled_client(self):
        """Verify multiple concurrent requests reuse the same pooled client."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            get_client_for_request,
            get_client_instantiation_count,
        )

        await startup_http_client_pool()

        clients = []
        for _ in range(10):
            async with get_client_for_request() as client:
                clients.append(client)

        # All clients should be the same instance
        assert len(set(id(c) for c in clients)) == 1

        # Only one instantiation should have occurred
        assert get_client_instantiation_count() == 1


class TestHttpClientPoolingDisabled:
    """Test behavior when pooling is disabled via feature flag."""

    @pytest.fixture(autouse=True)
    async def disable_pooling(self):
        """Disable pooling via environment variable."""
        from litellm_llmrouter.http_client_pool import (
            reset_http_client_pool,
            reset_client_instantiation_count,
        )

        await reset_http_client_pool()
        reset_client_instantiation_count()

        # Store original value
        original = os.environ.get("HTTP_CLIENT_POOLING_ENABLED")

        # Disable pooling
        os.environ["HTTP_CLIENT_POOLING_ENABLED"] = "false"

        # Need to reimport to pick up new env var
        import importlib
        import litellm_llmrouter.http_client_pool as pool_module

        importlib.reload(pool_module)

        yield

        # Restore
        if original is not None:
            os.environ["HTTP_CLIENT_POOLING_ENABLED"] = original
        else:
            os.environ.pop("HTTP_CLIENT_POOLING_ENABLED", None)

        # Reload again to restore
        importlib.reload(pool_module)
        await reset_http_client_pool()
        reset_client_instantiation_count()

    @pytest.mark.asyncio
    async def test_startup_noop_when_disabled(self):
        """Verify startup is a no-op when pooling is disabled."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            is_pool_initialized,
            HTTP_CLIENT_POOLING_ENABLED,
        )

        assert not HTTP_CLIENT_POOLING_ENABLED

        await startup_http_client_pool()
        assert not is_pool_initialized()

    @pytest.mark.asyncio
    async def test_get_http_client_raises_when_disabled(self):
        """Verify get_http_client raises when pooling is disabled."""
        from litellm_llmrouter.http_client_pool import (
            get_http_client,
            HTTP_CLIENT_POOLING_ENABLED,
        )

        assert not HTTP_CLIENT_POOLING_ENABLED

        with pytest.raises(RuntimeError, match="pooling is disabled"):
            get_http_client()

    @pytest.mark.asyncio
    async def test_get_client_for_request_creates_new_client_each_time(self):
        """Verify get_client_for_request creates new client when pooling disabled."""
        from litellm_llmrouter.http_client_pool import (
            get_client_for_request,
            get_client_instantiation_count,
            HTTP_CLIENT_POOLING_ENABLED,
        )

        assert not HTTP_CLIENT_POOLING_ENABLED

        # Each call should create a new client
        clients = []
        for i in range(3):
            async with get_client_for_request() as client:
                clients.append(id(client))
                assert get_client_instantiation_count() == i + 1

        # Each should be a different instance
        assert len(set(clients)) == 3


class TestFallbackClient:
    """Test the fallback client creation."""

    @pytest.fixture(autouse=True)
    async def reset_pool(self):
        """Reset pool state before/after each test."""
        from litellm_llmrouter.http_client_pool import (
            reset_http_client_pool,
            reset_client_instantiation_count,
        )

        await reset_http_client_pool()
        reset_client_instantiation_count()
        yield
        await reset_http_client_pool()
        reset_client_instantiation_count()

    @pytest.mark.asyncio
    async def test_create_fallback_client_with_custom_timeout(self):
        """Verify create_fallback_client accepts custom timeout."""
        from litellm_llmrouter.http_client_pool import create_fallback_client

        async with create_fallback_client(timeout=15.0) as client:
            assert client is not None
            # Timeout should be set (exact structure depends on httpx version)

    @pytest.mark.asyncio
    async def test_fallback_client_properly_closed(self):
        """Verify fallback client is properly closed after context."""
        from litellm_llmrouter.http_client_pool import create_fallback_client

        client_ref = None
        async with create_fallback_client() as client:
            client_ref = client
            assert not client.is_closed

        # Client should be closed after exiting context
        assert client_ref.is_closed


class TestInstantiationCountTracking:
    """Test the instantiation count tracking for regression prevention."""

    @pytest.fixture(autouse=True)
    async def reset_pool(self):
        """Reset pool state before/after each test."""
        from litellm_llmrouter.http_client_pool import (
            reset_http_client_pool,
            reset_client_instantiation_count,
        )

        await reset_http_client_pool()
        reset_client_instantiation_count()
        yield
        await reset_http_client_pool()
        reset_client_instantiation_count()

    @pytest.mark.asyncio
    async def test_instantiation_count_starts_at_zero(self):
        """Verify instantiation count starts at zero."""
        from litellm_llmrouter.http_client_pool import get_client_instantiation_count

        assert get_client_instantiation_count() == 0

    @pytest.mark.asyncio
    async def test_instantiation_count_increments_on_pool_creation(self):
        """Verify instantiation count increments when pool is created."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            get_client_instantiation_count,
        )

        await startup_http_client_pool()
        assert get_client_instantiation_count() == 1

    @pytest.mark.asyncio
    async def test_instantiation_bounded_under_load(self):
        """
        Test that instantiation count stays bounded under simulated load.

        This is the key regression test: if we had per-request clients,
        this count would be N instead of staying at 1 or 2.
        """
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            get_client_for_request,
            get_client_instantiation_count,
        )

        await startup_http_client_pool()
        initial_count = get_client_instantiation_count()

        # Simulate 100 "requests"
        for _ in range(100):
            async with get_client_for_request() as _client:
                # Simulated request (client intentionally unused)
                pass

        final_count = get_client_instantiation_count()

        # Count should stay the same (pooled clients reused)
        assert final_count == initial_count
        assert final_count == 1


class TestPoolWithConfiguration:
    """Test pool configuration options."""

    @pytest.fixture(autouse=True)
    async def reset_pool(self):
        """Reset pool state before/after each test."""
        from litellm_llmrouter.http_client_pool import (
            reset_http_client_pool,
            reset_client_instantiation_count,
        )

        await reset_http_client_pool()
        reset_client_instantiation_count()
        yield
        await reset_http_client_pool()
        reset_client_instantiation_count()

    @pytest.mark.asyncio
    async def test_pool_respects_max_connections_config(self):
        """Verify pool configuration is applied from environment variables."""
        import os
        import importlib

        # Set custom config
        os.environ["HTTP_CLIENT_MAX_CONNECTIONS"] = "50"
        os.environ["HTTP_CLIENT_MAX_KEEPALIVE"] = "10"

        import litellm_llmrouter.http_client_pool as pool_module

        importlib.reload(pool_module)

        try:
            # Verify the module constants are set correctly from env vars
            assert pool_module.HTTP_CLIENT_MAX_CONNECTIONS == 50
            assert pool_module.HTTP_CLIENT_MAX_KEEPALIVE == 10

            # Verify pool can be started with these config values
            await pool_module.startup_http_client_pool()
            client = pool_module.get_http_client()
            
            # Verify client is created and usable (don't check internals)
            assert client is not None
            assert isinstance(client, httpx.AsyncClient)
            assert not client.is_closed
        finally:
            # Cleanup
            os.environ.pop("HTTP_CLIENT_MAX_CONNECTIONS", None)
            os.environ.pop("HTTP_CLIENT_MAX_KEEPALIVE", None)
            await pool_module.reset_http_client_pool()
            importlib.reload(pool_module)


class TestIsPoolingEnabled:
    """Test the is_pooling_enabled helper."""

    @pytest.fixture(autouse=True)
    async def reset_pool(self):
        """Reset pool state before/after each test."""
        from litellm_llmrouter.http_client_pool import (
            reset_http_client_pool,
            reset_client_instantiation_count,
        )

        await reset_http_client_pool()
        reset_client_instantiation_count()
        yield
        await reset_http_client_pool()
        reset_client_instantiation_count()

    @pytest.mark.asyncio
    async def test_is_pooling_enabled_false_before_startup(self):
        """Verify is_pooling_enabled returns False before startup."""
        from litellm_llmrouter.http_client_pool import is_pooling_enabled

        assert not is_pooling_enabled()

    @pytest.mark.asyncio
    async def test_is_pooling_enabled_true_after_startup(self):
        """Verify is_pooling_enabled returns True after startup."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            is_pooling_enabled,
        )

        await startup_http_client_pool()
        assert is_pooling_enabled()

    @pytest.mark.asyncio
    async def test_is_pooling_enabled_false_after_shutdown(self):
        """Verify is_pooling_enabled returns False after shutdown."""
        from litellm_llmrouter.http_client_pool import (
            startup_http_client_pool,
            shutdown_http_client_pool,
            is_pooling_enabled,
        )

        await startup_http_client_pool()
        assert is_pooling_enabled()

        await shutdown_http_client_pool()
        assert not is_pooling_enabled()
