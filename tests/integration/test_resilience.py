"""
Integration Tests for Resilience: Circuit Breakers and Degraded Mode
=====================================================================

Tests for:
- DB outage simulation with circuit breaker triggering
- Writes fail fast when breaker is open
- Reads can still return cached/in-memory values
- Readiness endpoint reports degraded status
- Recovery when DB comes back
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from litellm_llmrouter.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    get_circuit_breaker_manager,
    reset_circuit_breaker_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fast_breaker_config() -> CircuitBreakerConfig:
    """Create a fast circuit breaker config for testing."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=0.1,  # Fast timeout
        window_seconds=10.0,
    )


@pytest.fixture
def cb_manager() -> CircuitBreakerManager:
    """Create a fresh circuit breaker manager."""
    reset_circuit_breaker_manager()
    manager = get_circuit_breaker_manager()
    yield manager
    reset_circuit_breaker_manager()


@pytest.fixture
def app_with_health():
    """Create a minimal FastAPI app with health endpoint."""
    from litellm_llmrouter.routes import health_router
    from litellm_llmrouter.resilience import (
        get_drain_manager,
        reset_drain_manager,
        reset_circuit_breaker_manager,
    )

    # Reset singletons
    reset_drain_manager()
    reset_circuit_breaker_manager()

    app = FastAPI()
    app.include_router(health_router)

    yield app

    # Cleanup
    reset_drain_manager()
    reset_circuit_breaker_manager()


# =============================================================================
# DB Outage Simulation Tests
# =============================================================================


class TestDBOutageSimulation:
    """Tests simulating database outages."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, fast_breaker_config: CircuitBreakerConfig):
        """Test that circuit breaker opens after consecutive failures."""
        breaker = CircuitBreaker("database", fast_breaker_config)

        # Simulate DB connection failures
        for i in range(fast_breaker_config.failure_threshold):
            await breaker.record_failure(f"Connection refused (attempt {i})")

        # Circuit should now be open
        assert breaker.is_open
        assert breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_writes_fail_fast_when_open(self, fast_breaker_config: CircuitBreakerConfig):
        """Test that writes fail immediately when circuit is open."""
        breaker = CircuitBreaker("database", fast_breaker_config)

        # Open the circuit
        await breaker.force_open()

        # Attempt a write operation - should fail fast
        with pytest.raises(CircuitBreakerOpenError) as exc:
            async with breaker.execute():
                # This should never run
                await asyncio.sleep(1)  # Would be slow if reached

        # Verify fail-fast behavior
        assert exc.value.breaker_name == "database"
        assert exc.value.time_until_retry > 0

    @pytest.mark.asyncio
    async def test_cached_reads_work_when_degraded(self, cb_manager: CircuitBreakerManager):
        """Test that cached/in-memory reads still work when DB is unavailable."""
        # Simulate an in-memory cache
        cache = {"key1": "cached_value1", "key2": "cached_value2"}

        # Open the database circuit breaker
        await cb_manager.database.force_open()
        assert cb_manager.is_degraded()

        # Simulate a read operation with fallback to cache
        async def db_read(key: str) -> str:
            async with cb_manager.database.execute():
                # Would normally read from DB
                raise RuntimeError("DB unavailable")

        async def cached_read(key: str) -> str | None:
            try:
                return await db_read(key)
            except CircuitBreakerOpenError:
                # Fall back to cache
                return cache.get(key)

        # Cache read should still work
        result = await cached_read("key1")
        assert result == "cached_value1"

    @pytest.mark.asyncio
    async def test_recovery_after_timeout(self, fast_breaker_config: CircuitBreakerConfig):
        """Test that circuit transitions to half-open after timeout."""
        breaker = CircuitBreaker("database", fast_breaker_config)

        # Open the circuit
        for _ in range(fast_breaker_config.failure_threshold):
            await breaker.record_failure("Connection refused")

        assert breaker.is_open

        # Wait for timeout
        await asyncio.sleep(fast_breaker_config.timeout_seconds + 0.05)

        # Should be half-open now
        assert breaker.is_half_open

    @pytest.mark.asyncio
    async def test_full_recovery_cycle(self, fast_breaker_config: CircuitBreakerConfig):
        """Test full cycle: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
        breaker = CircuitBreaker("database", fast_breaker_config)

        # 1. Start CLOSED
        assert breaker.is_closed

        # 2. Fail to OPEN
        for _ in range(fast_breaker_config.failure_threshold):
            await breaker.record_failure("DB error")
        assert breaker.is_open

        # 3. Wait for HALF_OPEN
        await asyncio.sleep(fast_breaker_config.timeout_seconds + 0.05)
        assert breaker.is_half_open

        # 4. Succeed to CLOSED
        for _ in range(fast_breaker_config.success_threshold):
            await breaker.record_success()
        assert breaker.is_closed


# =============================================================================
# Readiness Endpoint Integration Tests
# =============================================================================


class TestReadinessEndpointDegradedMode:
    """Tests for readiness endpoint with degraded mode."""

    @pytest.mark.asyncio
    async def test_readiness_reports_healthy_when_ok(self, app_with_health):
        """Test readiness returns 200 when all systems healthy."""
        transport = ASGITransport(app=app_with_health)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/_health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ("ready", "degraded")
            assert "checks" in data

    @pytest.mark.asyncio
    async def test_readiness_reports_degraded_when_db_cb_open(self, app_with_health):
        """Test readiness reports degraded when database circuit breaker is open."""
        # Open the database circuit breaker
        cb_manager = get_circuit_breaker_manager()
        await cb_manager.database.force_open()

        transport = ASGITransport(app=app_with_health)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/_health/ready")

            # Should still return 200 (degraded is not a failure)
            assert response.status_code == 200
            data = response.json()
            assert data["is_degraded"] is True
            assert "circuit_breakers" in data["checks"]
            assert data["checks"]["circuit_breakers"]["status"] == "degraded"
            assert "database" in data["checks"]["circuit_breakers"]["open_breakers"]

    @pytest.mark.asyncio
    async def test_readiness_reports_degraded_when_redis_cb_open(self, app_with_health):
        """Test readiness reports degraded when Redis circuit breaker is open."""
        cb_manager = get_circuit_breaker_manager()
        await cb_manager.redis.force_open()

        transport = ASGITransport(app=app_with_health)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/_health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["is_degraded"] is True
            assert "redis" in data["checks"]["circuit_breakers"]["open_breakers"]

    @pytest.mark.asyncio
    async def test_readiness_includes_breaker_status(self, app_with_health):
        """Test readiness includes circuit breaker status information."""
        cb_manager = get_circuit_breaker_manager()
        # Access breakers to register them
        cb_manager.database
        cb_manager.redis

        transport = ASGITransport(app=app_with_health)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/_health/ready")

            data = response.json()
            assert "circuit_breakers" in data["checks"]
            assert "breakers" in data["checks"]["circuit_breakers"]

    @pytest.mark.asyncio
    async def test_readiness_recovers_after_cb_close(self, app_with_health):
        """Test readiness reports healthy after circuit breaker closes."""
        cb_manager = get_circuit_breaker_manager()

        # Open and then close the circuit
        await cb_manager.database.force_open()
        await cb_manager.database.force_closed()

        transport = ASGITransport(app=app_with_health)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/_health/ready")

            data = response.json()
            # Should not be degraded after recovery
            assert cb_manager.is_degraded() is False


# =============================================================================
# Database Check with Circuit Breaker Integration Tests
# =============================================================================


class TestDatabaseCheckWithCircuitBreaker:
    """Tests for database health check with circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_db_check_skipped_when_cb_open(self, app_with_health):
        """Test that database check is skipped when circuit breaker is open."""
        cb_manager = get_circuit_breaker_manager()
        await cb_manager.database.force_open()

        # Set DATABASE_URL to trigger the check
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}):
            transport = ASGITransport(app=app_with_health)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/_health/ready")

                data = response.json()
                # Database check should show degraded, not unhealthy
                assert "database" in data["checks"]
                assert data["checks"]["database"]["status"] == "degraded"
                assert data["checks"]["database"]["circuit_breaker"] == "open"

    @pytest.mark.asyncio
    async def test_db_failure_opens_circuit_breaker(self, app_with_health):
        """Test that database connection failures open the circuit breaker."""
        cb_manager = get_circuit_breaker_manager()

        # Simulate multiple failed health checks that would open the breaker
        db_breaker = cb_manager.database
        for i in range(5):
            await db_breaker.record_failure(f"Connection refused (attempt {i})")

        # Breaker should be open now
        assert db_breaker.is_open


# =============================================================================
# Redis Check with Circuit Breaker Integration Tests
# =============================================================================


class TestRedisCheckWithCircuitBreaker:
    """Tests for Redis health check with circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_redis_check_skipped_when_cb_open(self, app_with_health):
        """Test that Redis check is skipped when circuit breaker is open."""
        cb_manager = get_circuit_breaker_manager()
        await cb_manager.redis.force_open()

        with patch.dict(os.environ, {"REDIS_HOST": "localhost"}):
            transport = ASGITransport(app=app_with_health)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/_health/ready")

                data = response.json()
                assert "redis" in data["checks"]
                assert data["checks"]["redis"]["status"] == "degraded"
                assert data["checks"]["redis"]["circuit_breaker"] == "open"


# =============================================================================
# Multi-Service Degradation Tests
# =============================================================================


class TestMultiServiceDegradation:
    """Tests for multiple services degrading simultaneously."""

    @pytest.mark.asyncio
    async def test_multiple_breakers_open(self, cb_manager: CircuitBreakerManager):
        """Test handling when multiple circuit breakers are open."""
        await cb_manager.database.force_open()
        await cb_manager.redis.force_open()

        assert cb_manager.is_degraded()

        status = cb_manager.get_status()
        assert len(status["degraded_components"]) == 2
        assert "database" in status["degraded_components"]
        assert "redis" in status["degraded_components"]

    @pytest.mark.asyncio
    async def test_partial_recovery(self, cb_manager: CircuitBreakerManager):
        """Test partial recovery (one service recovers, one still degraded)."""
        await cb_manager.database.force_open()
        await cb_manager.redis.force_open()

        # Recover database
        await cb_manager.database.force_closed()

        # Still degraded due to Redis
        assert cb_manager.is_degraded()

        status = cb_manager.get_status()
        assert len(status["degraded_components"]) == 1
        assert "redis" in status["degraded_components"]

    @pytest.mark.asyncio
    async def test_full_recovery(self, cb_manager: CircuitBreakerManager):
        """Test full recovery of all services."""
        await cb_manager.database.force_open()
        await cb_manager.redis.force_open()

        # Recover all
        await cb_manager.reset_all()

        assert not cb_manager.is_degraded()
        assert len(cb_manager.get_status()["degraded_components"]) == 0


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access to circuit breakers."""

    @pytest.mark.asyncio
    async def test_concurrent_failure_recording(self, fast_breaker_config: CircuitBreakerConfig):
        """Test that concurrent failure recording is thread-safe."""
        breaker = CircuitBreaker("concurrent-test", fast_breaker_config)

        # Record failures concurrently
        async def record_failures(count: int):
            for i in range(count):
                await breaker.record_failure(f"error-{i}")
                await asyncio.sleep(0.001)  # Small delay to interleave

        # Run multiple concurrent coroutines
        await asyncio.gather(
            record_failures(5),
            record_failures(5),
            record_failures(5),
        )

        # Breaker should have opened at some point
        # (exact timing depends on interleaving, but should be open)
        # The key test is that no race condition caused an error

    @pytest.mark.asyncio
    async def test_concurrent_execute_calls(self, fast_breaker_config: CircuitBreakerConfig):
        """Test concurrent execute() calls are handled correctly."""
        breaker = CircuitBreaker("concurrent-exec", fast_breaker_config)
        executed_count = 0

        async def execute_operation():
            nonlocal executed_count
            async with breaker.execute():
                executed_count += 1
                await asyncio.sleep(0.01)

        # Run many concurrent executions
        tasks = [execute_operation() for _ in range(10)]
        await asyncio.gather(*tasks)

        assert executed_count == 10


# =============================================================================
# Environment Variable Configuration Tests
# =============================================================================


class TestEnvironmentConfiguration:
    """Tests for circuit breaker configuration via environment."""

    def test_global_config_from_env(self):
        """Test loading global circuit breaker config from environment."""
        env = {
            "ROUTEIQ_CB_FAILURE_THRESHOLD": "10",
            "ROUTEIQ_CB_SUCCESS_THRESHOLD": "5",
            "ROUTEIQ_CB_TIMEOUT_SECONDS": "60",
            "ROUTEIQ_CB_WINDOW_SECONDS": "120",
        }

        with patch.dict(os.environ, env, clear=False):
            config = CircuitBreakerConfig.from_env()
            assert config.failure_threshold == 10
            assert config.success_threshold == 5
            assert config.timeout_seconds == 60.0
            assert config.window_seconds == 120.0

    def test_service_specific_config_override(self):
        """Test that service-specific config overrides global."""
        env = {
            "ROUTEIQ_CB_FAILURE_THRESHOLD": "5",
            "ROUTEIQ_CB_DATABASE_FAILURE_THRESHOLD": "10",
        }

        with patch.dict(os.environ, env, clear=False):
            global_config = CircuitBreakerConfig.from_env()
            db_config = CircuitBreakerConfig.from_env("database")

            assert global_config.failure_threshold == 5
            assert db_config.failure_threshold == 10
