"""
Tests for the Routing Strategy Registry and Pipeline.

These tests verify:
1. RoutingStrategyRegistry registration and thread-safety
2. Deterministic weighted A/B selection (same key -> same choice)
3. RoutingPipeline execution with fallback and telemetry
4. Integration with routing_strategy_patch
5. Concurrency safety for registry updates
"""

import os
import threading
from typing import Dict, Optional
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.strategy_registry import (
    RoutingStrategyRegistry,
    RoutingStrategy,
    RoutingPipeline,
    RoutingContext,
    get_routing_registry,
    reset_routing_singletons,
    ENV_ACTIVE_STRATEGY,
    ENV_STRATEGY_WEIGHTS,
)


class MockStrategy(RoutingStrategy):
    """Mock strategy for testing."""

    def __init__(self, name: str = "mock", deployment_to_return: Optional[Dict] = None):
        self._name = name
        self._deployment_to_return = deployment_to_return or {
            "model_name": "test-model"
        }
        self.call_count = 0
        self.last_context: Optional[RoutingContext] = None

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        self.call_count += 1
        self.last_context = context
        return self._deployment_to_return

    @property
    def name(self) -> str:
        return self._name


class FailingStrategy(RoutingStrategy):
    """Strategy that always raises an exception."""

    def __init__(self, error_message: str = "Strategy failed"):
        self._error_message = error_message

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        raise RuntimeError(self._error_message)

    @property
    def name(self) -> str:
        return "failing-strategy"


class TestRoutingStrategyRegistry:
    """Test RoutingStrategyRegistry functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        # Clear env vars
        for key in [ENV_ACTIVE_STRATEGY, ENV_STRATEGY_WEIGHTS]:
            if key in os.environ:
                del os.environ[key]
        yield
        reset_routing_singletons()

    def test_register_strategy(self):
        """Test registering a strategy."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("test-strategy")

        registry.register("test-strategy", strategy)

        assert "test-strategy" in registry.list_strategies()
        assert registry.get("test-strategy") is strategy

    def test_unregister_strategy(self):
        """Test unregistering a strategy."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("test-strategy")

        registry.register("test-strategy", strategy)
        result = registry.unregister("test-strategy")

        assert result is True
        assert "test-strategy" not in registry.list_strategies()

    def test_unregister_nonexistent(self):
        """Test unregistering a strategy that doesn't exist."""
        registry = RoutingStrategyRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_set_active_strategy(self):
        """Test setting the active strategy."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)

        result = registry.set_active("strategy2")

        assert result is True
        assert registry.get_active() == "strategy2"

    def test_set_active_clears_weights(self):
        """Test that setting active strategy clears A/B weights."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)
        registry.set_weights({"strategy1": 50, "strategy2": 50})

        registry.set_active("strategy1")

        assert registry.get_weights() == {}
        assert registry.get_active() == "strategy1"

    def test_set_active_invalid(self):
        """Test setting an invalid active strategy."""
        registry = RoutingStrategyRegistry()

        result = registry.set_active("nonexistent")

        assert result is False

    def test_set_weights(self):
        """Test setting A/B weights."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)

        result = registry.set_weights({"strategy1": 90, "strategy2": 10})

        assert result is True
        assert registry.get_weights() == {"strategy1": 90, "strategy2": 10}
        assert registry.get_active() is None  # Should be None when using weights

    def test_set_weights_invalid_strategy(self):
        """Test setting weights with an invalid strategy."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")

        registry.register("strategy1", strategy1)

        result = registry.set_weights({"strategy1": 50, "nonexistent": 50})

        assert result is False
        assert registry.get_weights() == {}

    def test_clear_weights(self):
        """Test clearing A/B weights."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)
        registry.set_weights({"strategy1": 50, "strategy2": 50})

        registry.clear_weights()

        assert registry.get_weights() == {}
        assert registry.get_active() == "strategy1"  # First weighted becomes active

    def test_select_strategy_single_active(self):
        """Test selecting a strategy when single active is set."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("test-strategy")

        registry.register("test-strategy", strategy)
        registry.set_active("test-strategy")

        selected = registry.select_strategy("any-hash-key")

        assert selected is strategy

    def test_get_status(self):
        """Test getting registry status."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)
        registry.set_weights({"strategy1": 90, "strategy2": 10})

        status = registry.get_status()

        assert set(status["registered_strategies"]) == {"strategy1", "strategy2"}
        assert status["ab_weights"] == {"strategy1": 90, "strategy2": 10}
        assert status["ab_enabled"] is True


class TestDeterministicWeightedSelection:
    """Test deterministic A/B selection based on hash key."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_same_key_same_result(self):
        """Test that the same hash key always produces the same result."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)
        registry.set_weights({"strategy1": 50, "strategy2": 50})

        hash_key = "user:test-user-123"

        # Select multiple times with same key
        results = [registry.select_strategy(hash_key) for _ in range(100)]

        # All results should be the same
        assert all(r == results[0] for r in results)

    def test_different_keys_distribute(self):
        """Test that different keys distribute across strategies."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)
        registry.set_weights({"strategy1": 50, "strategy2": 50})

        # Generate many different keys and track distribution
        selections = {"strategy1": 0, "strategy2": 0}
        num_samples = 1000

        for i in range(num_samples):
            selected = registry.select_strategy(f"user:test-user-{i}")
            selections[selected.name] += 1

        # With 50/50 weights, expect roughly equal distribution (with some variance)
        # Allow 10% tolerance
        assert abs(selections["strategy1"] - 500) < 100
        assert abs(selections["strategy2"] - 500) < 100

    def test_weighted_distribution_90_10(self):
        """Test that 90/10 weights produce correct distribution."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("baseline")
        strategy2 = MockStrategy("candidate")

        registry.register("baseline", strategy1)
        registry.register("candidate", strategy2)
        registry.set_weights({"baseline": 90, "candidate": 10})

        # Generate many different keys
        selections = {"baseline": 0, "candidate": 0}
        num_samples = 1000

        for i in range(num_samples):
            selected = registry.select_strategy(f"request:{i}")
            selections[selected.name] += 1

        # With 90/10 weights, baseline should be ~900, candidate ~100
        # Allow 5% tolerance
        assert abs(selections["baseline"] - 900) < 50
        assert abs(selections["candidate"] - 100) < 50

    def test_user_id_sticky_assignment(self):
        """Test that same user_id always gets same strategy."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)
        registry.set_weights({"strategy1": 50, "strategy2": 50})

        # Simulate multiple requests from same user
        user_key = "user:persistent-user-abc"
        first_selection = registry.select_strategy(user_key)

        # Even with different request IDs, same user should get same strategy
        for _ in range(50):
            selected = registry.select_strategy(user_key)
            assert selected == first_selection


class TestRoutingPipeline:
    """Test RoutingPipeline execution."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def _create_mock_router(self):
        """Create a mock router for testing."""
        router = MagicMock()
        router.model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ]
        router.healthy_deployments = router.model_list
        router._llmrouter_strategy = "llmrouter-knn"
        router._llmrouter_strategy_args = {}
        return router

    def test_pipeline_success(self):
        """Test successful pipeline routing."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("test-strategy", {"model_name": "selected-model"})
        registry.register("test-strategy", strategy)
        registry.set_active("test-strategy")

        pipeline = RoutingPipeline(registry, emit_telemetry=False)

        context = RoutingContext(
            router=self._create_mock_router(),
            model="test-model",
        )

        result = pipeline.route(context)

        assert result.deployment == {"model_name": "selected-model"}
        assert result.strategy_name == "test-strategy"
        assert result.is_fallback is False
        assert result.error is None
        assert result.latency_ms > 0

    def test_pipeline_fallback_on_error(self):
        """Test pipeline falls back to default on strategy error."""
        registry = RoutingStrategyRegistry()
        failing = FailingStrategy("Test error")
        fallback = MockStrategy("default", {"model_name": "fallback-model"})

        registry.register("failing", failing)
        registry.set_active("failing")

        pipeline = RoutingPipeline(
            registry, default_strategy=fallback, emit_telemetry=False
        )

        context = RoutingContext(
            router=self._create_mock_router(),
            model="test-model",
        )

        result = pipeline.route(context)

        assert result.deployment == {"model_name": "fallback-model"}
        assert result.strategy_name == "default"
        assert result.is_fallback is True
        assert "Test error" in result.fallback_reason

    def test_pipeline_no_strategy(self):
        """Test pipeline with no registered strategies uses default."""
        registry = RoutingStrategyRegistry()
        default = MockStrategy("default", {"model_name": "default-model"})

        pipeline = RoutingPipeline(
            registry, default_strategy=default, emit_telemetry=False
        )

        context = RoutingContext(
            router=self._create_mock_router(),
            model="test-model",
        )

        result = pipeline.route(context)

        assert result.deployment == {"model_name": "default-model"}
        assert result.strategy_name == "default"

    def test_pipeline_ab_selection(self):
        """Test pipeline selects strategy based on A/B weights."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1", {"model_name": "model1"})
        strategy2 = MockStrategy("strategy2", {"model_name": "model2"})

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)
        registry.set_weights({"strategy1": 100, "strategy2": 0})  # 100% strategy1

        pipeline = RoutingPipeline(registry, emit_telemetry=False)

        context = RoutingContext(
            router=self._create_mock_router(),
            model="test-model",
            user_id="test-user",
        )

        result = pipeline.route(context)

        assert result.deployment == {"model_name": "model1"}
        assert result.strategy_name == "strategy1"


class TestRoutingContext:
    """Test RoutingContext hash key generation."""

    def test_hash_key_with_user_id(self):
        """Test hash key uses user_id when available."""
        context = RoutingContext(
            router=MagicMock(),
            model="test-model",
            user_id="user-123",
            request_id="request-456",
        )

        key = context.get_ab_hash_key()

        assert key == "user:user-123"

    def test_hash_key_with_request_id(self):
        """Test hash key uses request_id when user_id not available."""
        context = RoutingContext(
            router=MagicMock(),
            model="test-model",
            request_id="request-456",
        )

        key = context.get_ab_hash_key()

        assert key == "request:request-456"

    def test_hash_key_random_fallback(self):
        """Test hash key generates random when no identifiers."""
        context = RoutingContext(
            router=MagicMock(),
            model="test-model",
        )

        key = context.get_ab_hash_key()

        assert key.startswith("random:")


class TestConcurrencySafety:
    """Test thread-safety of registry operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_concurrent_registration(self):
        """Test concurrent strategy registration is thread-safe."""
        registry = RoutingStrategyRegistry()
        num_threads = 10
        num_strategies_per_thread = 100
        errors = []

        def register_strategies(thread_id: int):
            try:
                for i in range(num_strategies_per_thread):
                    strategy = MockStrategy(f"strategy-{thread_id}-{i}")
                    registry.register(f"strategy-{thread_id}-{i}", strategy)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_strategies, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert (
            len(registry.list_strategies()) == num_threads * num_strategies_per_thread
        )

    def test_concurrent_selection(self):
        """Test concurrent strategy selection is thread-safe."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)
        registry.set_weights({"strategy1": 50, "strategy2": 50})

        num_threads = 10
        num_selections_per_thread = 100
        errors = []
        results = []
        results_lock = threading.Lock()

        def select_strategies(thread_id: int):
            local_results = []
            try:
                for _ in range(num_selections_per_thread):
                    context = RoutingContext(MagicMock(), "test-model")
                    selected = registry.select_strategy(context.get_ab_hash_key())
                    if selected:
                        local_results.append(selected.name)
            except Exception as e:
                errors.append(e)

            with results_lock:
                results.extend(local_results)

        threads = [
            threading.Thread(target=select_strategies, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == num_threads * num_selections_per_thread

    def test_concurrent_weight_updates(self):
        """Test concurrent weight updates are thread-safe."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)

        num_threads = 5
        num_updates_per_thread = 50
        errors = []

        def update_weights(thread_id: int):
            try:
                for i in range(num_updates_per_thread):
                    if i % 2 == 0:
                        registry.set_weights({"strategy1": 90, "strategy2": 10})
                    else:
                        registry.set_weights({"strategy1": 10, "strategy2": 90})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_weights, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Weights should be one of the two valid states
        weights = registry.get_weights()
        assert weights in [
            {"strategy1": 90, "strategy2": 10},
            {"strategy1": 10, "strategy2": 90},
        ]


class TestEnvironmentConfiguration:
    """Test configuration loading from environment variables."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons and clear env vars before each test."""
        reset_routing_singletons()
        # Clear env vars
        for key in [ENV_ACTIVE_STRATEGY, ENV_STRATEGY_WEIGHTS]:
            if key in os.environ:
                del os.environ[key]
        yield
        # Clear env vars again
        for key in [ENV_ACTIVE_STRATEGY, ENV_STRATEGY_WEIGHTS]:
            if key in os.environ:
                del os.environ[key]

    def test_load_active_strategy_from_env(self):
        """Test loading active strategy from environment."""
        os.environ[ENV_ACTIVE_STRATEGY] = "test-strategy"

        registry = RoutingStrategyRegistry()

        # Active is set but strategy not registered yet
        # This is expected - will be used when strategy is registered
        assert registry._active_strategy == "test-strategy"

    def test_load_weights_from_env(self):
        """Test loading A/B weights from environment."""
        os.environ[ENV_STRATEGY_WEIGHTS] = '{"baseline": 90, "candidate": 10}'

        registry = RoutingStrategyRegistry()

        assert registry._weights == {"baseline": 90, "candidate": 10}

    def test_invalid_weights_json(self):
        """Test handling of invalid weights JSON."""
        os.environ[ENV_STRATEGY_WEIGHTS] = "not-valid-json"

        # Should not raise, just log error
        registry = RoutingStrategyRegistry()

        assert registry._weights == {}


class TestUpdateCallbacks:
    """Test registry update callbacks."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_callback_on_set_active(self):
        """Test callback is called when active strategy changes."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("test-strategy")
        registry.register("test-strategy", strategy)

        callback_count = [0]

        def callback():
            callback_count[0] += 1

        registry.add_update_callback(callback)
        registry.set_active("test-strategy")

        assert callback_count[0] == 1

    def test_callback_on_set_weights(self):
        """Test callback is called when weights change."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")
        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)

        callback_count = [0]

        def callback():
            callback_count[0] += 1

        registry.add_update_callback(callback)
        registry.set_weights({"strategy1": 50, "strategy2": 50})

        assert callback_count[0] == 1

    def test_callback_error_does_not_propagate(self):
        """Test that callback errors don't break updates."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("test-strategy")
        registry.register("test-strategy", strategy)

        def failing_callback():
            raise RuntimeError("Callback error")

        registry.add_update_callback(failing_callback)

        # Should not raise
        result = registry.set_active("test-strategy")

        assert result is True


class TestSingletonAccess:
    """Test singleton access functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_get_routing_registry_singleton(self):
        """Test that get_routing_registry returns same instance."""
        registry1 = get_routing_registry()
        registry2 = get_routing_registry()

        assert registry1 is registry2

    def test_pipeline_creation_with_custom_default(self):
        """Test creating a pipeline with a custom default strategy.

        This avoids the lazy import issue with get_routing_pipeline().
        """
        from litellm_llmrouter.strategy_registry import RoutingPipeline

        registry = RoutingStrategyRegistry()
        mock_default = MockStrategy("mock-default")

        # Create pipeline with mock default - this should not trigger lazy imports
        pipeline = RoutingPipeline(
            registry, default_strategy=mock_default, emit_telemetry=False
        )

        # Verify pipeline works
        assert pipeline._default_strategy is mock_default
        assert pipeline._registry is registry

    def test_reset_routing_singletons(self):
        """Test that reset creates new instances.

        We use only the registry singleton to avoid lazy import hangs.
        """
        registry1 = get_routing_registry()

        reset_routing_singletons()

        registry2 = get_routing_registry()

        assert registry1 is not registry2
