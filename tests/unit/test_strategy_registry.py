"""
Tests for the Routing Strategy Registry and Pipeline.

These tests verify:
1. RoutingStrategyRegistry registration and thread-safety
2. Deterministic weighted A/B selection (same key -> same choice)
3. RoutingPipeline execution with fallback and telemetry
4. Integration with custom_routing_strategy
5. Concurrency safety for registry updates
6. Staged loading with promotion/rollback
7. Experiment assignment with version tracking
"""

import os
import threading
from typing import Dict, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.strategy_registry import (
    RoutingStrategyRegistry,
    RoutingStrategy,
    RoutingPipeline,
    RoutingContext,
    ExperimentConfig,
    ABSelectionResult,
    StrategyState,
    get_routing_registry,
    reset_routing_singletons,
    ENV_ACTIVE_STRATEGY,
    ENV_STRATEGY_WEIGHTS,
    ENV_EXPERIMENT_ID,
)


class MockStrategy(RoutingStrategy):
    """Mock strategy for testing."""

    def __init__(
        self,
        name: str = "mock",
        deployment_to_return: Optional[Dict] = None,
        version: Optional[str] = None,
        should_fail_validation: bool = False,
    ):
        self._name = name
        self._version = version
        self._deployment_to_return = deployment_to_return or {
            "model_name": "test-model"
        }
        self._should_fail_validation = should_fail_validation
        self.call_count = 0
        self.last_context: Optional[RoutingContext] = None

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        self.call_count += 1
        self.last_context = context
        return self._deployment_to_return

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Optional[str]:
        return self._version

    def validate(self) -> Tuple[bool, Optional[str]]:
        if self._should_fail_validation:
            return False, "Intentional validation failure"
        return True, None


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
        for key in [ENV_ACTIVE_STRATEGY, ENV_STRATEGY_WEIGHTS, ENV_EXPERIMENT_ID]:
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

    def test_register_strategy_with_version(self):
        """Test registering a strategy with version."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("test-strategy", version="v1.0")

        registry.register("test-strategy", strategy, version="v1.0", family="test")

        entry = registry.get_entry("test-strategy")
        assert entry is not None
        assert entry.version == "v1.0"
        assert entry.family == "test"
        assert entry.state == StrategyState.ACTIVE

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

        result = registry.select_strategy("any-hash-key")

        assert isinstance(result, ABSelectionResult)
        assert result.strategy is strategy
        assert result.strategy_name == "test-strategy"

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
        assert all(r.strategy_name == results[0].strategy_name for r in results)

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
            result = registry.select_strategy(f"user:test-user-{i}")
            selections[result.strategy_name] += 1

        # With 50/50 weights, expect roughly equal distribution (with some variance)
        # Allow 10% tolerance
        assert abs(selections["strategy1"] - 500) < 100
        assert abs(selections["strategy2"] - 500) < 100

    def test_weighted_distribution_90_10(self):
        """Test that 90/10 weights produce correct distribution."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("baseline", version="v1.0")
        strategy2 = MockStrategy("candidate", version="v2.0")

        registry.register("baseline", strategy1, version="v1.0")
        registry.register("candidate", strategy2, version="v2.0")
        registry.set_weights(
            {"baseline": 90, "candidate": 10}, experiment_id="test-exp"
        )

        # Generate many different keys
        selections = {"baseline": 0, "candidate": 0}
        num_samples = 1000

        for i in range(num_samples):
            result = registry.select_strategy(f"request:{i}")
            selections[result.strategy_name] += 1

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
        first_result = registry.select_strategy(user_key)

        # Even with different request IDs, same user should get same strategy
        for _ in range(50):
            result = registry.select_strategy(user_key)
            assert result.strategy_name == first_result.strategy_name

    def test_ab_selection_returns_hash_bucket(self):
        """Test that A/B selection includes hash bucket info."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)
        registry.set_weights({"strategy1": 100, "strategy2": 0})  # 100% strategy1

        result = registry.select_strategy("user:test", "user")

        assert result.weight == 100
        assert result.hash_bucket is not None
        assert 0 <= result.hash_bucket < 100  # Valid bucket range
        assert result.total_weight == 100
        assert result.hash_key_type == "user"


class TestStagedLoading:
    """Test staged loading, promotion, and rollback."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_stage_strategy_success(self):
        """Test that staging a valid strategy auto-promotes it."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("new-strategy", version="v2.0")

        success, error = registry.stage_strategy(
            "new-strategy",
            strategy,
            version="v2.0",
        )

        # When validation passes, strategy is auto-promoted
        assert success is True
        assert error is None
        assert "new-strategy" in registry.list_strategies()  # Now active
        assert "new-strategy" not in registry.list_staged()  # No longer staged

    def test_stage_strategy_validation_failure(self):
        """Test staging a strategy that fails validation."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("bad-strategy", should_fail_validation=True)

        success, error = registry.stage_strategy("bad-strategy", strategy)

        # Validation failed - strategy stays staged with error
        assert success is True  # Staging itself succeeded
        assert error is not None
        assert "validation failure" in error.lower()
        assert "bad-strategy" in registry.list_staged()

        staged = registry.get_staged("bad-strategy")
        assert staged is not None
        assert staged.validation_passed is False

    def test_promote_staged_strategy(self):
        """Test promoting a staged strategy that failed initial validation."""
        registry = RoutingStrategyRegistry()
        # First stage a failing strategy
        failing_strategy = MockStrategy(
            "new-strategy", version="v2.0", should_fail_validation=True
        )

        registry.stage_strategy("new-strategy", failing_strategy, version="v2.0")
        assert "new-strategy" in registry.list_staged()

        # Now stage with a valid strategy - this will auto-promote
        valid_strategy = MockStrategy("new-strategy", version="v2.0")
        registry.stage_strategy("new-strategy", valid_strategy, version="v2.0")

        # The valid strategy was auto-promoted, so it's in active strategies now
        assert "new-strategy" in registry.list_strategies()
        assert "new-strategy" not in registry.list_staged()

        # Trying to promote again should fail (nothing staged)
        success, error = registry.promote_staged("new-strategy")

        assert success is False
        assert error is not None

    def test_promote_failed_validation(self):
        """Test that promoting a failed validation strategy fails."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("bad-strategy", should_fail_validation=True)

        registry.stage_strategy("bad-strategy", strategy)
        success, error = registry.promote_staged("bad-strategy")

        assert success is False
        assert "validation" in error.lower()

    def test_rollback_staged_strategy(self):
        """Test rolling back (discarding) a staged strategy."""
        registry = RoutingStrategyRegistry()
        # Stage a failing strategy so it stays in staged state
        strategy = MockStrategy("new-strategy", should_fail_validation=True)

        registry.stage_strategy("new-strategy", strategy)
        assert "new-strategy" in registry.list_staged()

        result = registry.rollback_staged("new-strategy")

        assert result is True
        assert "new-strategy" not in registry.list_staged()
        assert "new-strategy" not in registry.list_strategies()

    def test_auto_promote(self):
        """Test auto-promote flag stages and promotes in one call."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("auto-strategy", version="v1.0")

        success, error = registry.stage_strategy(
            "auto-strategy",
            strategy,
            version="v1.0",
            auto_promote=True,
        )

        assert success is True
        assert "auto-strategy" in registry.list_strategies()
        assert "auto-strategy" not in registry.list_staged()


class TestExperimentConfig:
    """Test experiment configuration and assignment."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        for key in [ENV_ACTIVE_STRATEGY, ENV_STRATEGY_WEIGHTS, ENV_EXPERIMENT_ID]:
            if key in os.environ:
                del os.environ[key]
        yield
        reset_routing_singletons()

    def test_set_experiment(self):
        """Test setting a full experiment configuration."""
        registry = RoutingStrategyRegistry()
        control = MockStrategy("control")
        treatment = MockStrategy("treatment")

        registry.register("baseline", control)
        registry.register("candidate", treatment)

        experiment = ExperimentConfig(
            experiment_id="routing-v2-rollout-2024",
            variants={"control": "baseline", "treatment": "candidate"},
            weights={"control": 90, "treatment": 10},
            description="Testing new routing strategy",
        )

        result = registry.set_experiment(experiment)

        assert result is True
        assert registry.get_experiment() == experiment
        assert registry.get_weights() == {"control": 90, "treatment": 10}

    def test_experiment_selection_includes_variant(self):
        """Test that A/B selection includes experiment/variant info."""
        registry = RoutingStrategyRegistry()
        control = MockStrategy("control")
        treatment = MockStrategy("treatment")

        registry.register("baseline", control)
        registry.register("candidate", treatment)

        experiment = ExperimentConfig(
            experiment_id="test-experiment",
            variants={"control": "baseline", "treatment": "candidate"},
            weights={"control": 50, "treatment": 50},
        )
        registry.set_experiment(experiment)

        result = registry.select_strategy("user:test", "user")

        assert result.experiment_id == "test-experiment"
        assert result.variant in ["control", "treatment"]

    def test_set_weights_with_experiment_id(self):
        """Test setting weights with an experiment ID."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)

        result = registry.set_weights(
            {"strategy1": 80, "strategy2": 20},
            experiment_id="exp-123",
        )

        assert result is True
        experiment = registry.get_experiment()
        assert experiment is not None
        assert experiment.experiment_id == "exp-123"


class TestVersionedStrategies:
    """Test versioned strategy support."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_list_versions(self):
        """Test listing all versions of a strategy family."""
        registry = RoutingStrategyRegistry()
        v1 = MockStrategy("knn-v1", version="1.0")
        v2 = MockStrategy("knn-v2", version="2.0")

        registry.register("knn-v1", v1, version="1.0", family="llmrouter-knn")
        registry.register("knn-v2", v2, version="2.0", family="llmrouter-knn")

        versions = registry.list_versions("llmrouter-knn")

        assert len(versions) == 2
        assert all(v.family == "llmrouter-knn" for v in versions)
        version_nums = [v.version for v in versions]
        assert "1.0" in version_nums
        assert "2.0" in version_nums

    def test_selection_includes_version(self):
        """Test that selection result includes strategy version."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("versioned", version="sha256:abc123")

        registry.register(
            "versioned", strategy, version="sha256:abc123", family="llmrouter-knn"
        )
        registry.set_active("versioned")

        result = registry.select_strategy("user:test")

        assert result.version == "sha256:abc123"

    def test_get_entry_with_metadata(self):
        """Test getting a strategy entry with metadata."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("test")

        registry.register(
            "test",
            strategy,
            version="v1.0",
            family="llmrouter-test",
            metadata={"model_hash": "abc123", "trained_at": "2024-01-01"},
        )

        entry = registry.get_entry("test")

        assert entry is not None
        assert entry.metadata["model_hash"] == "abc123"
        assert entry.metadata["trained_at"] == "2024-01-01"


class TestReloadFromConfig:
    """Test reloading registry from config dict."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons before each test."""
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_reload_weights(self):
        """Test reloading weights from config."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        registry.register("strategy1", strategy1)
        registry.register("strategy2", strategy2)

        success, errors = registry.reload_from_config(
            {"weights": {"strategy1": 70, "strategy2": 30}}
        )

        assert success is True
        assert len(errors) == 0
        assert registry.get_weights() == {"strategy1": 70, "strategy2": 30}

    def test_reload_experiment(self):
        """Test reloading experiment config."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("baseline")
        strategy2 = MockStrategy("candidate")

        registry.register("baseline", strategy1)
        registry.register("candidate", strategy2)

        success, errors = registry.reload_from_config(
            {
                "experiment": {
                    "experiment_id": "new-experiment",
                    "variants": {"control": "baseline"},
                    "weights": {"baseline": 100},
                }
            }
        )

        assert success is True
        experiment = registry.get_experiment()
        assert experiment.experiment_id == "new-experiment"

    def test_reload_invalid_strategy(self):
        """Test reload with invalid strategy name returns error."""
        registry = RoutingStrategyRegistry()

        success, errors = registry.reload_from_config({"weights": {"unknown": 100}})

        assert success is False
        assert len(errors) > 0
        assert "unknown" in errors[0].lower()


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
        strategy1 = MockStrategy("strategy1", {"model_name": "model1"}, version="v1.0")
        strategy2 = MockStrategy("strategy2", {"model_name": "model2"}, version="v2.0")

        registry.register("strategy1", strategy1, version="v1.0")
        registry.register("strategy2", strategy2, version="v2.0")
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

    def test_pipeline_ab_selection_result(self):
        """Test pipeline includes A/B selection result."""
        registry = RoutingStrategyRegistry()
        strategy1 = MockStrategy("baseline", {"model_name": "model1"}, version="1.0")
        strategy2 = MockStrategy("candidate", {"model_name": "model2"}, version="2.0")

        registry.register("baseline", strategy1, version="1.0")
        registry.register("candidate", strategy2, version="2.0")
        registry.set_weights(
            {"baseline": 50, "candidate": 50}, experiment_id="test-exp"
        )

        pipeline = RoutingPipeline(registry, emit_telemetry=False)

        context = RoutingContext(
            router=self._create_mock_router(),
            model="test-model",
            user_id="test-user",
        )

        result = pipeline.route(context)

        assert result.ab_selection is not None
        assert result.ab_selection.experiment_id == "test-exp"
        assert result.ab_selection.variant is not None
        assert result.ab_selection.hash_bucket is not None


class TestRoutingContext:
    """Test RoutingContext hash key generation."""

    def test_hash_key_with_tenant_and_user_id(self):
        """Test hash key uses tenant_id+user_id when both available."""
        context = RoutingContext(
            router=MagicMock(),
            model="test-model",
            tenant_id="tenant-abc",
            user_id="user-123",
            request_id="request-456",
        )

        key, key_type = context.get_ab_hash_key()

        assert key_type == "tenant_user"
        assert "tenant:tenant-abc" in key
        assert "user:user-123" in key

    def test_hash_key_with_user_id(self):
        """Test hash key uses user_id when available."""
        context = RoutingContext(
            router=MagicMock(),
            model="test-model",
            user_id="user-123",
            request_id="request-456",
        )

        key, key_type = context.get_ab_hash_key()

        assert key_type == "user"
        assert "user:user-123" in key

    def test_hash_key_with_request_id(self):
        """Test hash key uses request_id when user_id not available."""
        context = RoutingContext(
            router=MagicMock(),
            model="test-model",
            request_id="request-456",
        )

        key, key_type = context.get_ab_hash_key()

        assert key_type == "request"
        assert "request:" in key

    def test_hash_key_random_fallback(self):
        """Test hash key generates random when no identifiers."""
        context = RoutingContext(
            router=MagicMock(),
            model="test-model",
        )

        key, key_type = context.get_ab_hash_key()

        assert key_type == "random"
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
                    key, _ = context.get_ab_hash_key()
                    selected = registry.select_strategy(key)
                    if selected and selected.strategy:
                        local_results.append(selected.strategy_name)
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
        for key in [ENV_ACTIVE_STRATEGY, ENV_STRATEGY_WEIGHTS, ENV_EXPERIMENT_ID]:
            if key in os.environ:
                del os.environ[key]
        yield
        # Clear env vars again
        for key in [ENV_ACTIVE_STRATEGY, ENV_STRATEGY_WEIGHTS, ENV_EXPERIMENT_ID]:
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

    def test_load_experiment_id_from_env(self):
        """Test loading experiment ID from environment."""
        os.environ[ENV_EXPERIMENT_ID] = "exp-from-env"
        os.environ[ENV_STRATEGY_WEIGHTS] = '{"a": 50, "b": 50}'

        registry = RoutingStrategyRegistry()

        assert registry._experiment is not None
        assert registry._experiment.experiment_id == "exp-from-env"


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

    def test_callback_on_promote(self):
        """Test callback is called when strategy is promoted."""
        registry = RoutingStrategyRegistry()
        strategy = MockStrategy("new-strategy")

        callback_count = [0]

        def callback():
            callback_count[0] += 1

        registry.add_update_callback(callback)
        registry.stage_strategy("new-strategy", strategy)
        registry.promote_staged("new-strategy")

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
