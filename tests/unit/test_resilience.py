"""
Tests for Resilience Primitives (Backpressure and Drain Mode).

These tests verify:
1. BackpressureMiddleware returns 503 when at capacity
2. DrainManager tracks active requests correctly
3. Readiness endpoint returns non-200 when draining
4. Concurrency slots are held until response is complete
5. Health endpoints are excluded from limiting
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

# asyncio_mode = "auto" in pyproject.toml handles async test detection


class TestDrainManager:
    """Test the DrainManager class."""

    @pytest.fixture
    def drain_manager(self):
        """Create a fresh DrainManager for each test."""
        from litellm_llmrouter.resilience import DrainManager, ResilienceConfig

        config = ResilienceConfig(
            max_concurrent_requests=10,
            drain_timeout_seconds=5.0,
        )
        return DrainManager(config)

    async def test_acquire_and_release(self, drain_manager):
        """Test basic acquire and release of request slots."""
        assert drain_manager.active_requests == 0

        # Acquire
        result = await drain_manager.acquire()
        assert result is True
        assert drain_manager.active_requests == 1

        # Acquire again
        result = await drain_manager.acquire()
        assert result is True
        assert drain_manager.active_requests == 2

        # Release
        await drain_manager.release()
        assert drain_manager.active_requests == 1

        await drain_manager.release()
        assert drain_manager.active_requests == 0

    async def test_acquire_rejected_when_draining(self, drain_manager):
        """Test that new requests are rejected when draining."""
        assert drain_manager.is_draining is False

        await drain_manager.start_drain()
        assert drain_manager.is_draining is True

        # New acquire should fail
        result = await drain_manager.acquire()
        assert result is False

    async def test_drain_waits_for_active_requests(self, drain_manager):
        """Test that drain waits for active requests to complete."""
        # Acquire a slot
        await drain_manager.acquire()
        assert drain_manager.active_requests == 1

        # Start drain
        await drain_manager.start_drain()

        # Create a task to release after a short delay
        async def delayed_release():
            await asyncio.sleep(0.1)
            await drain_manager.release()

        release_task = asyncio.create_task(delayed_release())

        # Wait for drain should complete
        result = await drain_manager.wait_for_drain()
        assert result is True
        assert drain_manager.active_requests == 0

        await release_task

    async def test_drain_timeout(self, drain_manager):
        """Test that drain times out if requests don't complete."""
        # Override timeout to be very short
        drain_manager._config.drain_timeout_seconds = 0.1

        # Acquire a slot but don't release
        await drain_manager.acquire()

        await drain_manager.start_drain()

        # Wait should timeout
        result = await drain_manager.wait_for_drain()
        assert result is False
        assert drain_manager.active_requests == 1

    async def test_immediate_drain_when_no_requests(self, drain_manager):
        """Test that drain completes immediately when no active requests."""
        assert drain_manager.active_requests == 0

        await drain_manager.start_drain()
        result = await drain_manager.wait_for_drain()

        assert result is True

    def test_get_status(self, drain_manager):
        """Test get_status returns correct information."""
        status = drain_manager.get_status()

        assert "active_requests" in status
        assert "is_draining" in status
        assert "drain_timeout_seconds" in status
        assert status["active_requests"] == 0
        assert status["is_draining"] is False


class TestResilienceConfig:
    """Test ResilienceConfig loading."""

    def test_default_config(self):
        """Test default configuration values."""
        from litellm_llmrouter.resilience import ResilienceConfig

        config = ResilienceConfig()

        assert config.max_concurrent_requests == 0  # Disabled by default
        assert config.drain_timeout_seconds == 30
        assert config.is_enabled() is False

    def test_config_from_env(self):
        """Test loading config from environment variables."""
        from litellm_llmrouter.resilience import ResilienceConfig
        from litellm_llmrouter.settings import reset_settings

        with patch.dict(
            "os.environ",
            {
                "ROUTEIQ_MAX_CONCURRENT_REQUESTS": "100",
                "ROUTEIQ_DRAIN_TIMEOUT_SECONDS": "60",
                # pydantic-settings nested delimiter format
                "ROUTEIQ_RESILIENCE__MAX_CONCURRENT_REQUESTS": "100",
                "ROUTEIQ_RESILIENCE__DRAIN_TIMEOUT_SECONDS": "60",
            },
        ):
            reset_settings()
            config = ResilienceConfig.from_env()

            assert config.max_concurrent_requests == 100
            assert config.drain_timeout_seconds == 60.0
            assert config.is_enabled() is True

    def test_config_disabled_when_zero(self):
        """Test that middleware is disabled when max_concurrent is 0."""
        from litellm_llmrouter.resilience import ResilienceConfig

        with patch.dict(
            "os.environ",
            {"ROUTEIQ_MAX_CONCURRENT_REQUESTS": "0"},
        ):
            config = ResilienceConfig.from_env()
            assert config.is_enabled() is False

    def test_config_invalid_values_use_defaults(self):
        """Test that invalid env values fall back to defaults."""
        from litellm_llmrouter.resilience import ResilienceConfig

        with patch.dict(
            "os.environ",
            {
                "ROUTEIQ_MAX_CONCURRENT_REQUESTS": "not-a-number",
                "ROUTEIQ_DRAIN_TIMEOUT_SECONDS": "invalid",
            },
        ):
            config = ResilienceConfig.from_env()

            assert config.max_concurrent_requests == 0
            assert config.drain_timeout_seconds == 30


class TestBackpressureMiddleware:
    """Test the BackpressureMiddleware ASGI middleware."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app."""

        async def app(scope, receive, send):
            if scope["type"] == "http":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"OK",
                        "more_body": False,
                    }
                )

        return app

    @pytest.fixture
    def slow_mock_app(self):
        """Create a slow mock ASGI app for concurrency testing."""

        async def app(scope, receive, send):
            if scope["type"] == "http":
                await asyncio.sleep(0.2)  # Simulate slow request
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"OK",
                        "more_body": False,
                    }
                )

        return app

    async def test_503_when_at_capacity(self, slow_mock_app):
        """Test that middleware returns 503 when concurrency limit is reached."""
        from litellm_llmrouter.resilience import (
            BackpressureMiddleware,
            ResilienceConfig,
            DrainManager,
            reset_drain_manager,
        )

        # Reset global state
        reset_drain_manager()

        config = ResilienceConfig(max_concurrent_requests=1)
        drain_manager = DrainManager(config)
        middleware = BackpressureMiddleware(slow_mock_app, config, drain_manager)

        responses = []

        async def capture_send(message):
            responses.append(message)

        async def mock_receive():
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "path": "/api/v1/chat",
            "headers": [],
        }

        # Start first request (will be slow)
        first_request = asyncio.create_task(
            middleware(scope, mock_receive, capture_send)
        )

        # Give first request time to acquire semaphore
        await asyncio.sleep(0.05)

        # Second request should get 503
        second_responses = []

        async def second_capture(message):
            second_responses.append(message)

        await middleware(scope, mock_receive, second_capture)

        # Check second request got 503
        assert len(second_responses) >= 1
        start_msg = second_responses[0]
        assert start_msg["type"] == "http.response.start"
        assert start_msg["status"] == 503

        # Parse body for error
        if len(second_responses) > 1:
            body_msg = second_responses[1]
            body = json.loads(body_msg["body"].decode())
            assert body["error"] == "over_capacity"

        # Wait for first request to complete
        await first_request

        # Clean up
        reset_drain_manager()

    async def test_health_endpoints_excluded(self, mock_app):
        """Test that health endpoints bypass concurrency limiting."""
        from litellm_llmrouter.resilience import (
            BackpressureMiddleware,
            ResilienceConfig,
            DrainManager,
            reset_drain_manager,
        )

        reset_drain_manager()

        # Set limit to 0 to block all requests
        config = ResilienceConfig(max_concurrent_requests=1)
        drain_manager = DrainManager(config)

        # First, fill up the capacity
        drain_manager._active_requests = 1

        middleware = BackpressureMiddleware(mock_app, config, drain_manager)

        responses = []

        async def capture_send(message):
            responses.append(message)

        async def mock_receive():
            return {"type": "http.disconnect"}

        # Health endpoint should still work
        scope = {
            "type": "http",
            "path": "/_health/live",
            "headers": [],
        }

        await middleware(scope, mock_receive, capture_send)

        # Should get 200, not 503
        assert len(responses) >= 1
        assert responses[0]["status"] == 200

        reset_drain_manager()

    async def test_503_when_draining(self, mock_app):
        """Test that middleware returns 503 when draining."""
        from litellm_llmrouter.resilience import (
            BackpressureMiddleware,
            ResilienceConfig,
            DrainManager,
            reset_drain_manager,
        )

        reset_drain_manager()

        config = ResilienceConfig(max_concurrent_requests=10)
        drain_manager = DrainManager(config)

        # Start draining
        await drain_manager.start_drain()

        middleware = BackpressureMiddleware(mock_app, config, drain_manager)

        responses = []

        async def capture_send(message):
            responses.append(message)

        async def mock_receive():
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "path": "/api/v1/chat",
            "headers": [],
        }

        await middleware(scope, mock_receive, capture_send)

        # Should get 503 due to draining
        assert len(responses) >= 1
        assert responses[0]["status"] == 503

        reset_drain_manager()

    async def test_middleware_disabled_when_limit_zero(self, mock_app):
        """Test that requests pass through when limit is 0 (disabled)."""
        from litellm_llmrouter.resilience import (
            BackpressureMiddleware,
            ResilienceConfig,
            DrainManager,
            reset_drain_manager,
        )

        reset_drain_manager()

        config = ResilienceConfig(max_concurrent_requests=0)  # Disabled
        drain_manager = DrainManager(config)
        middleware = BackpressureMiddleware(mock_app, config, drain_manager)

        responses = []

        async def capture_send(message):
            responses.append(message)

        async def mock_receive():
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "path": "/api/v1/chat",
            "headers": [],
        }

        await middleware(scope, mock_receive, capture_send)

        # Should pass through and get 200
        assert len(responses) >= 1
        assert responses[0]["status"] == 200

        reset_drain_manager()


class TestReadinessWithDrain:
    """Test readiness endpoint integration with drain mode."""

    @pytest.fixture
    def client(self):
        """Create a test client for the standalone app."""
        from litellm_llmrouter.gateway import create_standalone_app
        from litellm_llmrouter.resilience import reset_drain_manager
        from starlette.testclient import TestClient

        reset_drain_manager()
        app = create_standalone_app(enable_plugins=False, enable_resilience=True)
        return TestClient(app)

    def test_readiness_returns_200_when_healthy(self, client):
        """Test readiness returns 200 when not draining."""
        response = client.get("/_health/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ready"
        assert "drain_status" in data["checks"]
        assert data["checks"]["drain_status"]["status"] == "accepting"

    def test_readiness_returns_503_when_draining(self, client):
        """Test readiness returns 503 when draining."""
        from litellm_llmrouter.resilience import get_drain_manager

        drain_manager = get_drain_manager()

        # Set draining state directly (bypass async lock for test)
        drain_manager._draining = True

        response = client.get("/_health/ready")
        assert response.status_code == 503

        data = response.json()
        assert data["detail"]["status"] == "not_ready"
        assert data["detail"]["checks"]["drain_status"]["status"] == "draining"


class TestGlobalSingleton:
    """Test global drain manager singleton."""

    def test_get_drain_manager_returns_same_instance(self):
        """Test that get_drain_manager returns the same instance."""
        from litellm_llmrouter.resilience import (
            get_drain_manager,
            reset_drain_manager,
        )

        reset_drain_manager()

        dm1 = get_drain_manager()
        dm2 = get_drain_manager()

        assert dm1 is dm2

    def test_reset_drain_manager_creates_new_instance(self):
        """Test that reset creates a new instance."""
        from litellm_llmrouter.resilience import (
            get_drain_manager,
            reset_drain_manager,
        )

        dm1 = get_drain_manager()
        reset_drain_manager()
        dm2 = get_drain_manager()

        assert dm1 is not dm2


class TestSharedCircuitBreakerState:
    """Test the Redis-backed SharedCircuitBreakerState."""

    @pytest.fixture(autouse=True)
    def _reset_shared_state(self):
        """Reset shared state singleton between tests."""
        from litellm_llmrouter.resilience import reset_shared_circuit_breaker_state

        reset_shared_circuit_breaker_state()
        yield
        reset_shared_circuit_breaker_state()

    async def test_returns_none_when_redis_unavailable(self):
        """get_state should return None when Redis is not available."""
        from litellm_llmrouter.resilience import SharedCircuitBreakerState

        state = SharedCircuitBreakerState()
        state._checked = True
        state._available = False

        result = await state.get_state("test_breaker")
        assert result is None

    async def test_record_failure_returns_negative_when_no_redis(self):
        """record_failure should return -1 when Redis is unavailable."""
        from litellm_llmrouter.resilience import SharedCircuitBreakerState

        state = SharedCircuitBreakerState()
        state._checked = True
        state._available = False

        count = await state.record_failure("test_breaker")
        assert count == -1

    @patch("litellm_llmrouter.redis_pool.get_async_redis_client")
    async def test_get_state_from_redis(self, mock_get_client):
        """get_state should return the state string from Redis."""
        from litellm_llmrouter.resilience import SharedCircuitBreakerState

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.get = AsyncMock(return_value="open")
        mock_get_client.return_value = mock_client

        state = SharedCircuitBreakerState()
        result = await state.get_state("provider:openai")

        assert result == "open"
        mock_client.get.assert_called_once_with("routeiq:cb:provider:openai:state")

    @patch("litellm_llmrouter.redis_pool.get_async_redis_client")
    async def test_set_state_to_redis(self, mock_get_client):
        """set_state should write to Redis with TTL."""
        from litellm_llmrouter.resilience import SharedCircuitBreakerState

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.set = AsyncMock()
        mock_get_client.return_value = mock_client

        state = SharedCircuitBreakerState()
        await state.set_state("database", "open", timeout_seconds=30.0)

        mock_client.set.assert_called_once_with(
            "routeiq:cb:database:state", "open", ex=60
        )

    @patch("litellm_llmrouter.redis_pool.get_async_redis_client")
    async def test_record_failure_increments(self, mock_get_client):
        """record_failure should INCR the failure counter."""
        from litellm_llmrouter.resilience import SharedCircuitBreakerState

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.incr = AsyncMock(return_value=3)
        mock_client.expire = AsyncMock()
        mock_get_client.return_value = mock_client

        state = SharedCircuitBreakerState()
        count = await state.record_failure("provider:openai", window_seconds=60.0)

        assert count == 3
        mock_client.incr.assert_called_once_with("routeiq:cb:provider:openai:failures")

    @patch("litellm_llmrouter.redis_pool.get_async_redis_client")
    async def test_record_failure_sets_ttl_on_first(self, mock_get_client):
        """record_failure should set TTL on the first failure."""
        from litellm_llmrouter.resilience import SharedCircuitBreakerState

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.incr = AsyncMock(return_value=1)  # First failure
        mock_client.expire = AsyncMock()
        mock_get_client.return_value = mock_client

        state = SharedCircuitBreakerState()
        await state.record_failure("test", window_seconds=60.0)

        mock_client.expire.assert_called_once_with("routeiq:cb:test:failures", 60)

    @patch("litellm_llmrouter.redis_pool.get_async_redis_client")
    async def test_reset_failures_deletes_keys(self, mock_get_client):
        """reset_failures should delete both failure and state keys."""
        from litellm_llmrouter.resilience import SharedCircuitBreakerState

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.delete = AsyncMock()
        mock_get_client.return_value = mock_client

        state = SharedCircuitBreakerState()
        await state.reset_failures("provider:openai")

        mock_client.delete.assert_called_once_with(
            "routeiq:cb:provider:openai:failures",
            "routeiq:cb:provider:openai:state",
        )

    async def test_lazy_redis_check_only_once(self):
        """_ensure_redis should only probe Redis once."""
        from litellm_llmrouter.resilience import SharedCircuitBreakerState

        state = SharedCircuitBreakerState()

        with patch(
            "litellm_llmrouter.redis_pool.get_async_redis_client",
            new_callable=AsyncMock,
        ) as mock_get_client:
            mock_get_client.return_value = None

            result1 = await state._ensure_redis()
            result2 = await state._ensure_redis()

        assert result1 is False
        assert result2 is False
        # Called only once due to _checked flag
        mock_get_client.assert_called_once()


class TestCircuitBreakerWithSharedState:
    """Test CircuitBreaker integration with SharedCircuitBreakerState."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        """Reset shared state between tests."""
        from litellm_llmrouter.resilience import reset_shared_circuit_breaker_state

        reset_shared_circuit_breaker_state()
        yield
        reset_shared_circuit_breaker_state()

    async def test_circuit_breaker_broadcasts_open_to_shared_state(self):
        """Opening circuit should broadcast to shared state."""
        from litellm_llmrouter.resilience import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitBreakerState as CBState,
            SharedCircuitBreakerState,
        )

        shared = SharedCircuitBreakerState()
        shared._checked = True
        shared._available = False  # No Redis — just verify calls
        shared.open_circuit = AsyncMock()  # type: ignore[method-assign]
        shared.set_state = AsyncMock()  # type: ignore[method-assign]
        shared.close_circuit = AsyncMock()  # type: ignore[method-assign]
        shared.record_failure = AsyncMock(return_value=-1)  # type: ignore[method-assign]
        shared.get_failure_count = AsyncMock(return_value=-1)  # type: ignore[method-assign]

        config = CircuitBreakerConfig(
            failure_threshold=2, timeout_seconds=10.0, window_seconds=60.0
        )
        breaker = CircuitBreaker("test", config=config, shared_state=shared)

        # Record enough failures to open
        await breaker.record_failure("error1")
        await breaker.record_failure("error2")

        assert breaker._state == CBState.OPEN
        shared.open_circuit.assert_called_once_with("test", 10.0)

    async def test_circuit_breaker_broadcasts_close_to_shared_state(self):
        """Closing circuit should broadcast to shared state."""
        from litellm_llmrouter.resilience import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitBreakerState as CBState,
            SharedCircuitBreakerState,
        )

        shared = SharedCircuitBreakerState()
        shared._checked = True
        shared._available = False
        shared.open_circuit = AsyncMock()  # type: ignore[method-assign]
        shared.set_state = AsyncMock()  # type: ignore[method-assign]
        shared.close_circuit = AsyncMock()  # type: ignore[method-assign]
        shared.record_failure = AsyncMock(return_value=-1)  # type: ignore[method-assign]
        shared.get_failure_count = AsyncMock(return_value=-1)  # type: ignore[method-assign]

        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout_seconds=0.01,
            window_seconds=60.0,
        )
        breaker = CircuitBreaker("test", config=config, shared_state=shared)

        # Open the circuit
        await breaker.record_failure("err1")
        await breaker.record_failure("err2")
        assert breaker._state == CBState.OPEN

        # Wait for timeout to transition to half-open
        import time

        time.sleep(0.02)

        # Record success to close
        await breaker.record_success()
        assert breaker._state == CBState.CLOSED
        shared.close_circuit.assert_called_with("test")

    async def test_allow_request_checks_shared_state(self):
        """allow_request should check shared state for cross-worker opens."""
        from litellm_llmrouter.resilience import (
            CircuitBreaker,
            CircuitBreakerConfig,
            SharedCircuitBreakerState,
        )

        shared = SharedCircuitBreakerState()
        shared._checked = True
        shared._available = True
        shared.get_state = AsyncMock(return_value="open")  # type: ignore[method-assign]
        shared.open_circuit = AsyncMock()  # type: ignore[method-assign]
        shared.set_state = AsyncMock()  # type: ignore[method-assign]
        shared.close_circuit = AsyncMock()  # type: ignore[method-assign]
        shared.record_failure = AsyncMock(return_value=-1)  # type: ignore[method-assign]

        config = CircuitBreakerConfig(failure_threshold=5, timeout_seconds=30.0)
        breaker = CircuitBreaker("test", config=config, shared_state=shared)

        # Local state is CLOSED, but shared state says OPEN
        allowed = await breaker.allow_request()
        assert allowed is False

    async def test_shared_failure_count_opens_circuit(self):
        """Circuit should open when shared failure count exceeds threshold."""
        from litellm_llmrouter.resilience import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitBreakerState as CBState,
            SharedCircuitBreakerState,
        )

        shared = SharedCircuitBreakerState()
        shared._checked = True
        shared._available = True
        # Shared state reports 4 failures already from other workers
        shared.record_failure = AsyncMock(return_value=5)  # type: ignore[method-assign]
        shared.get_failure_count = AsyncMock(return_value=5)  # type: ignore[method-assign]
        shared.open_circuit = AsyncMock()  # type: ignore[method-assign]
        shared.set_state = AsyncMock()  # type: ignore[method-assign]
        shared.close_circuit = AsyncMock()  # type: ignore[method-assign]
        shared.get_state = AsyncMock(return_value=None)  # type: ignore[method-assign]

        config = CircuitBreakerConfig(
            failure_threshold=5, timeout_seconds=30.0, window_seconds=60.0
        )
        breaker = CircuitBreaker("test", config=config, shared_state=shared)

        # Even a single local failure should open the circuit because
        # the shared count (5) >= threshold (5)
        await breaker.record_failure("err")
        assert breaker._state == CBState.OPEN
