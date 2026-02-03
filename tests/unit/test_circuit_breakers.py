"""
Unit Tests for Circuit Breaker Implementation
==============================================

Tests for:
- Circuit breaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure tracking in sliding window
- Success threshold for recovery
- Circuit breaker manager and degraded mode
- Configuration from environment variables
"""

import asyncio
import os
import time
from unittest.mock import patch

import pytest

from litellm_llmrouter.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    execute_with_circuit_breaker,
    get_circuit_breaker_manager,
    reset_circuit_breaker_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def breaker_config() -> CircuitBreakerConfig:
    """Create a test circuit breaker config with fast timeouts."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=0.1,  # Fast timeout for testing
        window_seconds=10.0,
    )


@pytest.fixture
def breaker(breaker_config: CircuitBreakerConfig) -> CircuitBreaker:
    """Create a test circuit breaker."""
    return CircuitBreaker(name="test", config=breaker_config)


@pytest.fixture(autouse=True)
def reset_manager():
    """Reset the global circuit breaker manager before each test."""
    reset_circuit_breaker_manager()
    yield
    reset_circuit_breaker_manager()


# =============================================================================
# CircuitBreakerConfig Tests
# =============================================================================


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30.0
        assert config.window_seconds == 60.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout_seconds=60.0,
            window_seconds=120.0,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.timeout_seconds == 60.0
        assert config.window_seconds == 120.0

    def test_from_env_defaults(self):
        """Test loading config from environment with defaults."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove any existing env vars
            for key in list(os.environ.keys()):
                if key.startswith("ROUTEIQ_CB_"):
                    del os.environ[key]

            config = CircuitBreakerConfig.from_env()
            assert config.failure_threshold == 5
            assert config.success_threshold == 2
            assert config.timeout_seconds == 30.0
            assert config.window_seconds == 60.0

    def test_from_env_custom_values(self):
        """Test loading config from environment with custom values."""
        env = {
            "ROUTEIQ_CB_FAILURE_THRESHOLD": "10",
            "ROUTEIQ_CB_SUCCESS_THRESHOLD": "5",
            "ROUTEIQ_CB_TIMEOUT_SECONDS": "60.0",
            "ROUTEIQ_CB_WINDOW_SECONDS": "120.0",
        }
        with patch.dict(os.environ, env, clear=False):
            config = CircuitBreakerConfig.from_env()
            assert config.failure_threshold == 10
            assert config.success_threshold == 5
            assert config.timeout_seconds == 60.0
            assert config.window_seconds == 120.0

    def test_from_env_with_prefix(self):
        """Test loading config with service-specific prefix."""
        env = {
            "ROUTEIQ_CB_DATABASE_FAILURE_THRESHOLD": "8",
            "ROUTEIQ_CB_FAILURE_THRESHOLD": "5",  # Global default
        }
        with patch.dict(os.environ, env, clear=False):
            config = CircuitBreakerConfig.from_env("database")
            assert config.failure_threshold == 8


# =============================================================================
# CircuitBreaker State Transition Tests
# =============================================================================


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, breaker: CircuitBreaker):
        """Test that new breaker starts in CLOSED state."""
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
        assert not breaker.is_half_open

    @pytest.mark.asyncio
    async def test_closed_to_open_on_failures(self, breaker: CircuitBreaker):
        """Test transition from CLOSED to OPEN after failure threshold."""
        # Record failures up to threshold
        for i in range(breaker._config.failure_threshold):
            await breaker.record_failure(f"error {i}")

        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.is_open
        assert not breaker.is_closed

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_timeout(self, breaker: CircuitBreaker):
        """Test transition from OPEN to HALF_OPEN after timeout."""
        # Trigger open state
        for _ in range(breaker._config.failure_threshold):
            await breaker.record_failure("error")

        assert breaker.is_open

        # Wait for timeout
        await asyncio.sleep(breaker._config.timeout_seconds + 0.05)

        # Should now be half-open
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        assert breaker.is_half_open

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_successes(self, breaker: CircuitBreaker):
        """Test transition from HALF_OPEN to CLOSED after success threshold."""
        # Trigger open state
        for _ in range(breaker._config.failure_threshold):
            await breaker.record_failure("error")

        # Wait for timeout to enter half-open
        await asyncio.sleep(breaker._config.timeout_seconds + 0.05)
        assert breaker.is_half_open

        # Record successes to close the circuit
        for _ in range(breaker._config.success_threshold):
            await breaker.record_success()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self, breaker: CircuitBreaker):
        """Test transition from HALF_OPEN back to OPEN on any failure."""
        # Trigger open state
        for _ in range(breaker._config.failure_threshold):
            await breaker.record_failure("error")

        # Wait for timeout to enter half-open
        await asyncio.sleep(breaker._config.timeout_seconds + 0.05)
        assert breaker.is_half_open

        # A single failure should reopen
        await breaker.record_failure("error in half-open")

        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.is_open


# =============================================================================
# CircuitBreaker Execute Context Manager Tests
# =============================================================================


class TestCircuitBreakerExecute:
    """Tests for the execute() context manager."""

    @pytest.mark.asyncio
    async def test_execute_in_closed_state(self, breaker: CircuitBreaker):
        """Test that execute() allows requests in CLOSED state."""
        executed = False

        async with breaker.execute():
            executed = True

        assert executed
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_execute_raises_when_open(self, breaker: CircuitBreaker):
        """Test that execute() raises CircuitBreakerOpenError when OPEN."""
        # Force the circuit open
        await breaker.force_open()

        with pytest.raises(CircuitBreakerOpenError) as exc:
            async with breaker.execute():
                pass

        assert exc.value.breaker_name == "test"
        assert exc.value.time_until_retry > 0

    @pytest.mark.asyncio
    async def test_execute_records_success(self, breaker: CircuitBreaker):
        """Test that successful execution records success."""
        # Make half-open to track successes
        await breaker.force_open()
        await asyncio.sleep(breaker._config.timeout_seconds + 0.05)

        initial_count = breaker._success_count

        async with breaker.execute():
            pass

        assert breaker._success_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_records_failure_on_exception(self, breaker: CircuitBreaker):
        """Test that exceptions record failure."""
        initial_failures = breaker.failure_count

        with pytest.raises(ValueError):
            async with breaker.execute():
                raise ValueError("test error")

        assert breaker.failure_count == initial_failures + 1
        assert breaker.last_failure_error == "test error"

    @pytest.mark.asyncio
    async def test_execute_propagates_exceptions(self, breaker: CircuitBreaker):
        """Test that exceptions are propagated after recording failure."""
        with pytest.raises(RuntimeError, match="original error"):
            async with breaker.execute():
                raise RuntimeError("original error")


# =============================================================================
# CircuitBreaker Sliding Window Tests
# =============================================================================


class TestCircuitBreakerSlidingWindow:
    """Tests for sliding window failure tracking."""

    @pytest.mark.asyncio
    async def test_failures_in_window_are_counted(self, breaker_config: CircuitBreakerConfig):
        """Test that failures within window are counted."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=0.1,
            window_seconds=10.0,  # 10 second window
        )
        breaker = CircuitBreaker("test", config)

        for i in range(3):
            await breaker.record_failure(f"error {i}")

        assert breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_old_failures_are_cleaned_up(self):
        """Test that failures outside window are removed."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=0.1,
            window_seconds=0.1,  # Very short window for testing
        )
        breaker = CircuitBreaker("test", config)

        # Record failures
        for i in range(3):
            await breaker.record_failure(f"error {i}")

        assert breaker.failure_count == 3

        # Wait for window to expire
        await asyncio.sleep(0.15)

        # Failures should be cleaned up
        assert breaker.failure_count == 0


# =============================================================================
# CircuitBreaker Manual Control Tests
# =============================================================================


class TestCircuitBreakerManualControl:
    """Tests for manual circuit control."""

    @pytest.mark.asyncio
    async def test_force_open(self, breaker: CircuitBreaker):
        """Test manually opening the circuit."""
        assert breaker.is_closed

        await breaker.force_open()

        assert breaker.is_open
        assert breaker._opened_at is not None

    @pytest.mark.asyncio
    async def test_force_closed(self, breaker: CircuitBreaker):
        """Test manually closing the circuit."""
        await breaker.force_open()
        assert breaker.is_open

        await breaker.force_closed()

        assert breaker.is_closed
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_reset(self, breaker: CircuitBreaker):
        """Test resetting the circuit breaker."""
        # Record some failures and open
        for _ in range(breaker._config.failure_threshold):
            await breaker.record_failure("error")

        assert breaker.is_open

        await breaker.reset()

        assert breaker.is_closed
        assert breaker.failure_count == 0
        assert breaker._success_count == 0
        assert breaker._opened_at is None
        assert breaker.last_failure_error is None


# =============================================================================
# CircuitBreaker Status Tests
# =============================================================================


class TestCircuitBreakerStatus:
    """Tests for circuit breaker status reporting."""

    @pytest.mark.asyncio
    async def test_get_status_closed(self, breaker: CircuitBreaker):
        """Test status when closed."""
        status = breaker.get_status()

        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert status["time_until_retry"] == 0
        assert status["last_failure_error"] is None
        assert "config" in status

    @pytest.mark.asyncio
    async def test_get_status_open(self, breaker: CircuitBreaker):
        """Test status when open."""
        for _ in range(breaker._config.failure_threshold):
            await breaker.record_failure("test error")

        status = breaker.get_status()

        assert status["state"] == "open"
        assert status["failure_count"] == breaker._config.failure_threshold
        assert status["time_until_retry"] > 0
        assert status["last_failure_error"] == "test error"


# =============================================================================
# CircuitBreakerManager Tests
# =============================================================================


class TestCircuitBreakerManager:
    """Tests for CircuitBreakerManager."""

    def test_get_breaker_creates_new(self):
        """Test that get_breaker creates a new breaker if not exists."""
        manager = CircuitBreakerManager()
        breaker = manager.get_breaker("test-service")

        assert breaker is not None
        assert breaker.name == "test-service"

    def test_get_breaker_returns_same_instance(self):
        """Test that get_breaker returns the same instance."""
        manager = CircuitBreakerManager()
        breaker1 = manager.get_breaker("test-service")
        breaker2 = manager.get_breaker("test-service")

        assert breaker1 is breaker2

    def test_database_property(self):
        """Test the database property shortcut."""
        manager = CircuitBreakerManager()
        breaker = manager.database

        assert breaker.name == "database"
        assert breaker is manager.get_breaker("database")

    def test_redis_property(self):
        """Test the redis property shortcut."""
        manager = CircuitBreakerManager()
        breaker = manager.redis

        assert breaker.name == "redis"
        assert breaker is manager.get_breaker("redis")

    def test_leader_election_property(self):
        """Test the leader_election property shortcut."""
        manager = CircuitBreakerManager()
        breaker = manager.leader_election

        assert breaker.name == "leader_election"
        assert breaker is manager.get_breaker("leader_election")

    @pytest.mark.asyncio
    async def test_is_degraded_false_when_all_closed(self):
        """Test is_degraded returns False when all breakers are closed."""
        manager = CircuitBreakerManager()
        # Create some breakers but don't fail them
        manager.get_breaker("service1")
        manager.get_breaker("service2")

        assert manager.is_degraded() is False

    @pytest.mark.asyncio
    async def test_is_degraded_true_when_any_open(self):
        """Test is_degraded returns True when any breaker is open."""
        manager = CircuitBreakerManager()
        manager.get_breaker("service1")
        await manager.get_breaker("service2").force_open()

        assert manager.is_degraded() is True

    @pytest.mark.asyncio
    async def test_get_degraded_components(self):
        """Test getting list of degraded components."""
        manager = CircuitBreakerManager()
        await manager.get_breaker("healthy").force_closed()
        await manager.get_breaker("unhealthy").force_open()

        components = manager.get_degraded_components()

        assert len(components) == 1
        assert components[0].name == "unhealthy"
        assert components[0].is_degraded is True

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting aggregated status."""
        manager = CircuitBreakerManager()
        manager.get_breaker("service1")
        await manager.get_breaker("service2").force_open()

        status = manager.get_status()

        assert status["is_degraded"] is True
        assert "service2" in status["degraded_components"]
        assert "breakers" in status
        assert "service1" in status["breakers"]
        assert "service2" in status["breakers"]

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all breakers."""
        manager = CircuitBreakerManager()
        await manager.get_breaker("service1").force_open()
        await manager.get_breaker("service2").force_open()

        await manager.reset_all()

        assert manager.get_breaker("service1").is_closed
        assert manager.get_breaker("service2").is_closed


# =============================================================================
# Global Manager Singleton Tests
# =============================================================================


class TestGlobalManager:
    """Tests for the global circuit breaker manager singleton."""

    def test_get_circuit_breaker_manager_singleton(self):
        """Test that get_circuit_breaker_manager returns a singleton."""
        manager1 = get_circuit_breaker_manager()
        manager2 = get_circuit_breaker_manager()

        assert manager1 is manager2

    def test_reset_circuit_breaker_manager(self):
        """Test that reset creates a new instance."""
        manager1 = get_circuit_breaker_manager()
        reset_circuit_breaker_manager()
        manager2 = get_circuit_breaker_manager()

        assert manager1 is not manager2


# =============================================================================
# execute_with_circuit_breaker Helper Tests
# =============================================================================


class TestExecuteWithCircuitBreaker:
    """Tests for the execute_with_circuit_breaker helper."""

    @pytest.mark.asyncio
    async def test_execute_success(self, breaker: CircuitBreaker):
        """Test successful execution."""
        async def operation():
            return "success"

        result = await execute_with_circuit_breaker(breaker, operation)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_failure(self, breaker: CircuitBreaker):
        """Test execution failure is propagated."""
        async def operation():
            raise ValueError("operation failed")

        with pytest.raises(ValueError, match="operation failed"):
            await execute_with_circuit_breaker(breaker, operation)

    @pytest.mark.asyncio
    async def test_execute_with_fallback_when_open(self, breaker: CircuitBreaker):
        """Test fallback is used when circuit is open."""
        await breaker.force_open()

        async def operation():
            return "should not execute"

        def fallback():
            return "fallback value"

        result = await execute_with_circuit_breaker(
            breaker, operation, fallback=fallback
        )
        assert result == "fallback value"

    @pytest.mark.asyncio
    async def test_execute_raises_without_fallback_when_open(self, breaker: CircuitBreaker):
        """Test CircuitBreakerOpenError is raised without fallback."""
        await breaker.force_open()

        async def operation():
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError):
            await execute_with_circuit_breaker(breaker, operation)

    @pytest.mark.asyncio
    async def test_execute_with_async_fallback(self, breaker: CircuitBreaker):
        """Test async fallback is awaited."""
        await breaker.force_open()

        async def operation():
            return "should not execute"

        async def async_fallback():
            return "async fallback"

        result = await execute_with_circuit_breaker(
            breaker, operation, fallback=async_fallback
        )
        assert result == "async fallback"

    @pytest.mark.asyncio
    async def test_execute_with_args_and_kwargs(self, breaker: CircuitBreaker):
        """Test passing arguments to the operation."""
        async def operation(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await execute_with_circuit_breaker(
            breaker, operation, "x", "y", c="z"
        )
        assert result == "x-y-z"


# =============================================================================
# CircuitBreakerOpenError Tests
# =============================================================================


class TestCircuitBreakerOpenError:
    """Tests for CircuitBreakerOpenError exception."""

    def test_error_attributes(self):
        """Test error has correct attributes."""
        error = CircuitBreakerOpenError("test-breaker", 15.5)

        assert error.breaker_name == "test-breaker"
        assert error.time_until_retry == 15.5
        assert "test-breaker" in str(error)
        assert "15.5s" in str(error)
