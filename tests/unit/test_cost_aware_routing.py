"""
Tests for the CostAwareRoutingStrategy.

These tests verify:
1. Strategy selects cheapest model above quality threshold
2. Fallback to best quality when no cheap options meet threshold
3. Cost lookup from litellm.model_cost (mocked)
4. Cost lookup fallback when litellm unavailable
5. Combined scoring with different cost_weight values
6. max_cost_per_1k_tokens filtering
7. Empty candidate list handling
8. Single candidate (always selected)
9. Strategy registration in LLMROUTER_STRATEGIES
10. Integration with inner_strategy delegation
11. Pareto frontier computation
12. Cost database refresh from litellm.model_cost
13. Provider fallback when circuit breaker is open
14. Configuration via kwargs (quality_weight, cost_refresh_interval, etc.)
15. Inner strategy lazy resolution by name from registry
"""

import time
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.strategies import (
    CostAwareRoutingStrategy,
    LLMROUTER_STRATEGIES,
    DEFAULT_ROUTER_HPARAMS,
)
from litellm_llmrouter.strategy_registry import (
    RoutingContext,
    RoutingStrategy,
    RoutingStrategyRegistry,
    RoutingPipeline,
    reset_routing_singletons,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deployment(model: str, model_name: str = "test-model") -> Dict:
    """Create a deployment dict matching LiteLLM's format."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": model},
    }


def _make_router(deployments: List[Dict]) -> MagicMock:
    """Create a mock Router with the given deployments."""
    router = MagicMock()
    router.model_list = deployments
    router.healthy_deployments = deployments
    return router


def _make_context(
    deployments: List[Dict],
    model_name: str = "test-model",
) -> RoutingContext:
    """Create a RoutingContext for testing."""
    return RoutingContext(
        router=_make_router(deployments),
        model=model_name,
    )


MOCK_MODEL_COST = {
    "gpt-3.5-turbo": {
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
    },
    "gpt-4": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-turbo": {
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
    },
    "claude-3-opus": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
    },
}


class MockInnerStrategy(RoutingStrategy):
    """Inner strategy that always selects a specific model."""

    def __init__(self, preferred_model: str):
        self._preferred_model = preferred_model

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        router = context.router
        healthy = getattr(router, "healthy_deployments", router.model_list)
        for dep in healthy:
            if dep.get("litellm_params", {}).get("model") == self._preferred_model:
                return dep
        return None

    @property
    def name(self) -> str:
        return "mock-inner"


class FailingInnerStrategy(RoutingStrategy):
    """Inner strategy that always raises."""

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        raise RuntimeError("Inner strategy failed")

    @property
    def name(self) -> str:
        return "failing-inner"


class NoneInnerStrategy(RoutingStrategy):
    """Inner strategy that always returns None."""

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        return None

    @property
    def name(self) -> str:
        return "none-inner"


# ---------------------------------------------------------------------------
# Test: Strategy registration
# ---------------------------------------------------------------------------


class TestCostAwareRegistration:
    """Test that the strategy is properly registered in the catalog."""

    def test_strategy_in_llmrouter_strategies(self):
        """llmrouter-cost-aware should be in LLMROUTER_STRATEGIES."""
        assert "llmrouter-cost-aware" in LLMROUTER_STRATEGIES

    def test_default_hparams_exist(self):
        """Default hyperparameters should exist for cost-aware."""
        assert "cost-aware" in DEFAULT_ROUTER_HPARAMS
        hparams = DEFAULT_ROUTER_HPARAMS["cost-aware"]
        assert hparams["quality_threshold"] == 0.7
        assert hparams["cost_weight"] == 0.7
        assert hparams["inner_strategy"] is None
        assert hparams["inner_strategy_name"] is None
        assert hparams["max_cost_per_1k_tokens"] is None
        assert hparams["cost_refresh_interval"] == 3600
        assert hparams["enable_circuit_breaker_filtering"] is True

    def test_strategy_name_property(self):
        """Strategy name should return llmrouter-cost-aware."""
        strategy = CostAwareRoutingStrategy()
        assert strategy.name == "llmrouter-cost-aware"

    def test_strategy_version_property(self):
        """Strategy version should return 2.0.0."""
        strategy = CostAwareRoutingStrategy()
        assert strategy.version == "2.0.0"

    def test_strategy_validates_successfully(self):
        """Default parameters should pass validation."""
        strategy = CostAwareRoutingStrategy()
        valid, error = strategy.validate()
        assert valid is True
        assert error is None


# ---------------------------------------------------------------------------
# Test: Registry integration
# ---------------------------------------------------------------------------


class TestCostAwareRegistryIntegration:
    """Test registration in the RoutingStrategyRegistry."""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_register_in_registry(self):
        """Strategy can be registered in RoutingStrategyRegistry."""
        registry = RoutingStrategyRegistry()
        strategy = CostAwareRoutingStrategy()

        registry.register("llmrouter-cost-aware", strategy)

        assert "llmrouter-cost-aware" in registry.list_strategies()
        assert registry.get("llmrouter-cost-aware") is strategy

    def test_register_and_select(self):
        """Registered strategy is selected when set as active."""
        registry = RoutingStrategyRegistry()
        strategy = CostAwareRoutingStrategy()

        registry.register("llmrouter-cost-aware", strategy)
        registry.set_active("llmrouter-cost-aware")

        result = registry.select_strategy("hash-key")
        assert result.strategy is strategy
        assert result.strategy_name == "llmrouter-cost-aware"

    def test_pipeline_execution(self):
        """Strategy works through the RoutingPipeline."""
        registry = RoutingStrategyRegistry()
        strategy = CostAwareRoutingStrategy(quality_threshold=0.0)

        registry.register("llmrouter-cost-aware", strategy)
        registry.set_active("llmrouter-cost-aware")

        pipeline = RoutingPipeline(registry, emit_telemetry=False)

        deployments = [_make_deployment("gpt-3.5-turbo")]
        context = _make_context(deployments)

        result = pipeline.route(context)
        assert result.deployment is not None
        assert result.strategy_name == "llmrouter-cost-aware"


# ---------------------------------------------------------------------------
# Test: Cost lookup
# ---------------------------------------------------------------------------


class TestCostLookup:
    """Test _get_model_cost lookups."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_known_model_cost(self):
        """Known model returns correct average cost per 1K tokens."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("gpt-4")
        # input: 0.00003 * 1000 = 0.03, output: 0.00006 * 1000 = 0.06
        # average: (0.03 + 0.06) / 2 = 0.045
        assert abs(cost - 0.045) < 1e-9

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_cheap_model_cost(self):
        """Cheap model returns lower cost."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("gpt-3.5-turbo")
        # input: 0.0000005 * 1000 = 0.0005, output: 0.0000015 * 1000 = 0.0015
        # average: (0.0005 + 0.0015) / 2 = 0.001
        assert abs(cost - 0.001) < 1e-9

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_unknown_model_returns_inf(self):
        """Unknown model returns infinity."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("unknown-model-xyz")
        assert cost == float("inf")

    @patch("litellm.model_cost", {})
    def test_empty_cost_db_returns_inf(self):
        """Empty cost database returns infinity."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("gpt-4")
        assert cost == float("inf")

    def test_litellm_import_failure_returns_inf(self):
        """When litellm import fails, returns infinity."""
        strategy = CostAwareRoutingStrategy()
        with patch.dict("sys.modules", {"litellm": None}):
            cost = strategy._get_model_cost("gpt-4")
            assert cost == float("inf")

    @patch(
        "litellm.model_cost",
        {
            "zero-cost-model": {
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }
        },
    )
    def test_zero_cost_model_returns_inf(self):
        """Model with zero cost returns inf (treated as unknown pricing)."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("zero-cost-model")
        assert cost == float("inf")


# ---------------------------------------------------------------------------
# Test: Cheapest model selection
# ---------------------------------------------------------------------------


class TestCheapestModelSelection:
    """Test selection of cheapest model above quality threshold."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_selects_cheapest_model(self):
        """With no inner strategy, selects cheapest from all candidates."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.5,
            cost_weight=1.0,  # Pure cost optimization
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
            _make_deployment("gpt-4-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_selects_cheapest_above_threshold_with_inner(self):
        """With inner strategy, filters by quality then selects cheapest."""
        # Inner strategy prefers gpt-4; gpt-3.5 gets lower quality
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.8,
            cost_weight=0.9,
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # gpt-4 gets quality=1.0 (preferred), gpt-3.5 gets 0.5 (below 0.8 threshold)
        # Only gpt-4 meets threshold
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_all_candidates_meet_threshold_picks_cheapest(self):
        """When all meet threshold, picks cheapest by combined score."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,  # All pass
            cost_weight=1.0,  # Pure cost
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
            _make_deployment("claude-3-opus"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Test: Fallback to best quality
# ---------------------------------------------------------------------------


class TestFallbackToBestQuality:
    """Test fallback when no cheap options meet threshold."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_fallback_when_none_meet_threshold(self):
        """Falls back to best quality when no candidate meets threshold."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=1.1,  # Impossible threshold
            cost_weight=0.5,
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Falls back to inner strategy's selection
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_fallback_without_inner_returns_first(self):
        """Without inner strategy, quality=1.0 always meets threshold.

        To test fallback, use NoneInnerStrategy (quality 0.5) with
        a threshold above 0.5 so no candidate qualifies.
        """
        inner = NoneInnerStrategy()
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.9,  # Above 0.5 from NoneInner
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # All get quality 0.5 (below 0.9) -> fallback to best quality
        # Inner returns None -> falls to first candidate
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_fallback_with_failing_inner(self):
        """Fallback handles inner strategy failure gracefully."""
        inner = FailingInnerStrategy()
        strategy = CostAwareRoutingStrategy(
            quality_threshold=1.1,  # Impossible
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Inner fails, falls back to first candidate
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Test: Combined scoring with cost_weight
# ---------------------------------------------------------------------------


class TestCombinedScoring:
    """Test combined quality-cost scoring with different cost_weight values."""

    def test_combined_score_cost_only(self):
        """cost_weight=1.0 means only cost matters."""
        strategy = CostAwareRoutingStrategy(cost_weight=1.0)
        # quality irrelevant, normalized_cost=0.0 (cheapest) -> score = 1.0
        score = strategy._compute_combined_score(quality=0.0, normalized_cost=0.0)
        assert abs(score - 1.0) < 1e-9

    def test_combined_score_quality_only(self):
        """cost_weight=0.0 means only quality matters."""
        strategy = CostAwareRoutingStrategy(cost_weight=0.0)
        score = strategy._compute_combined_score(quality=0.8, normalized_cost=1.0)
        assert abs(score - 0.8) < 1e-9

    def test_combined_score_balanced(self):
        """cost_weight=0.5 balances quality and cost."""
        strategy = CostAwareRoutingStrategy(cost_weight=0.5)
        # quality=0.8, normalized_cost=0.4 -> (1-0.5)*0.8 + 0.5*(1-0.4) = 0.4+0.3 = 0.7
        score = strategy._compute_combined_score(quality=0.8, normalized_cost=0.4)
        assert abs(score - 0.7) < 1e-9

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_cost_weight_zero_selects_highest_quality(self):
        """cost_weight=0.0 with inner strategy selects highest quality."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.0,  # Only quality matters
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-3.5-turbo"),
            _make_deployment("gpt-4"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # gpt-4 has quality 1.0, gpt-3.5 has 0.5
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_cost_weight_one_selects_cheapest(self):
        """cost_weight=1.0 selects cheapest regardless of quality."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=1.0,  # Only cost matters
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Test: max_cost_per_1k_tokens filtering
# ---------------------------------------------------------------------------


class TestMaxCostFiltering:
    """Test max_cost_per_1k_tokens hard cap."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_filters_expensive_models(self):
        """Models above max_cost_per_1k_tokens are excluded."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.5,
            max_cost_per_1k_tokens=0.01,  # Only gpt-3.5-turbo is cheap enough
        )

        deployments = [
            _make_deployment("gpt-4"),  # avg cost: 0.045
            _make_deployment("gpt-3.5-turbo"),  # avg cost: 0.001
            _make_deployment("claude-3-opus"),  # avg cost: 0.045
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_all_exceed_cap_falls_back(self):
        """When all models exceed cost cap, falls back to best quality."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.5,
            max_cost_per_1k_tokens=0.0001,  # Nothing is this cheap
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Falls back to first candidate
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_no_max_cost_allows_all(self):
        """Without max_cost_per_1k_tokens, all candidates considered."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=1.0,
            max_cost_per_1k_tokens=None,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Cheapest wins
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases in cost-aware routing."""

    def test_empty_candidate_list(self):
        """Empty candidate list returns None."""
        strategy = CostAwareRoutingStrategy()
        deployments: list = []
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is None

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_single_candidate_always_selected(self):
        """Single candidate is always returned."""
        strategy = CostAwareRoutingStrategy(quality_threshold=0.99)

        deployments = [_make_deployment("gpt-4")]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_no_matching_model_name(self):
        """Deployments with non-matching model_name are filtered out."""
        strategy = CostAwareRoutingStrategy()

        deployments = [
            _make_deployment("gpt-4", model_name="other-model"),
        ]
        context = _make_context(deployments, model_name="test-model")

        result = strategy.select_deployment(context)

        assert result is None

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_all_same_cost_picks_highest_quality(self):
        """When all candidates have same cost, picks highest quality."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.5,
            inner_strategy=inner,
        )

        # Same model, same cost -> quality breaks the tie
        deployments = [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4"},
                "id": "deploy-1",
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4"},
                "id": "deploy-2",
            },
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        # Both have same quality (1.0 since model matches) and same cost
        # First one wins since combined scores are equal and we iterate in order
        assert result["litellm_params"]["model"] == "gpt-4"

    def test_quality_threshold_clamped(self):
        """Quality threshold is clamped to [0, 1]."""
        strategy = CostAwareRoutingStrategy(quality_threshold=2.0)
        assert strategy._quality_threshold == 1.0

        strategy = CostAwareRoutingStrategy(quality_threshold=-0.5)
        assert strategy._quality_threshold == 0.0

    def test_cost_weight_clamped(self):
        """Cost weight is clamped to [0, 1]."""
        strategy = CostAwareRoutingStrategy(cost_weight=1.5)
        assert strategy._cost_weight == 1.0

        strategy = CostAwareRoutingStrategy(cost_weight=-0.2)
        assert strategy._cost_weight == 0.0


# ---------------------------------------------------------------------------
# Test: Inner strategy delegation
# ---------------------------------------------------------------------------


class TestInnerStrategyDelegation:
    """Test delegation to inner strategy for quality prediction."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_inner_strategy_quality_scoring(self):
        """Inner strategy's preferred model gets quality=1.0."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.0,  # Pure quality
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-3.5-turbo"),
            _make_deployment("gpt-4"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_no_inner_strategy_all_quality_equal(self):
        """Without inner strategy, all get quality 1.0 -> cheapest wins."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.5,
            inner_strategy=None,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_inner_strategy_returns_none(self):
        """Inner strategy returning None gives quality 0.5."""
        inner = NoneInnerStrategy()
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.0,  # Pure quality
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Both get 0.5 quality; gpt-3.5-turbo is cheaper with same quality
        # so it dominates gpt-4 on the Pareto frontier, leaving only gpt-3.5
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_inner_strategy_exception_gives_half_quality(self):
        """Inner strategy raising exception gives quality 0.5."""
        inner = FailingInnerStrategy()
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=1.0,  # Pure cost
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Both get 0.5 quality (exception), cost_weight=1.0 -> cheapest wins
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Test: Quality prediction
# ---------------------------------------------------------------------------


class TestQualityPrediction:
    """Test _predict_quality method directly."""

    def test_no_inner_returns_one(self):
        """Without inner strategy, quality is always 1.0."""
        strategy = CostAwareRoutingStrategy(inner_strategy=None)
        dep = _make_deployment("gpt-4")
        context = _make_context([dep])

        quality = strategy._predict_quality(context, dep)

        assert quality == 1.0

    def test_inner_selects_this_deployment(self):
        """When inner selects this deployment, quality is 1.0."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(inner_strategy=inner)

        dep = _make_deployment("gpt-4")
        context = _make_context([dep])

        quality = strategy._predict_quality(context, dep)

        assert quality == 1.0

    def test_inner_selects_different_deployment(self):
        """When inner selects a different deployment, quality is 0.5."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(inner_strategy=inner)

        dep = _make_deployment("gpt-3.5-turbo")
        context = _make_context([dep, _make_deployment("gpt-4")])

        quality = strategy._predict_quality(context, dep)

        assert quality == 0.5

    def test_inner_raises_returns_half(self):
        """When inner raises, quality is 0.5."""
        inner = FailingInnerStrategy()
        strategy = CostAwareRoutingStrategy(inner_strategy=inner)

        dep = _make_deployment("gpt-4")
        context = _make_context([dep])

        quality = strategy._predict_quality(context, dep)

        assert quality == 0.5

    def test_inner_returns_none_gives_half(self):
        """When inner returns None, quality is 0.5."""
        inner = NoneInnerStrategy()
        strategy = CostAwareRoutingStrategy(inner_strategy=inner)

        dep = _make_deployment("gpt-4")
        context = _make_context([dep])

        quality = strategy._predict_quality(context, dep)

        assert quality == 0.5


# ---------------------------------------------------------------------------
# Test: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Test strategy validation."""

    def test_valid_default_params(self):
        """Default parameters pass validation."""
        strategy = CostAwareRoutingStrategy()
        valid, error = strategy.validate()
        assert valid is True
        assert error is None

    def test_valid_custom_params(self):
        """Custom valid parameters pass validation."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.5,
            cost_weight=0.3,
        )
        valid, error = strategy.validate()
        assert valid is True
        assert error is None

    def test_boundary_values_pass(self):
        """Boundary values (0.0 and 1.0) pass validation."""
        for qt in [0.0, 1.0]:
            for cw in [0.0, 1.0]:
                strategy = CostAwareRoutingStrategy(
                    quality_threshold=qt,
                    cost_weight=cw,
                )
                valid, error = strategy.validate()
                assert valid is True, f"Failed for qt={qt}, cw={cw}: {error}"


# ---------------------------------------------------------------------------
# Test: Candidate extraction
# ---------------------------------------------------------------------------


class TestCandidateExtraction:
    """Test _get_candidates method."""

    def test_extracts_matching_candidates(self):
        """Only deployments matching model_name are returned."""
        strategy = CostAwareRoutingStrategy()

        deployments = [
            _make_deployment("gpt-4", model_name="model-a"),
            _make_deployment("gpt-3.5-turbo", model_name="model-b"),
            _make_deployment("claude-3-opus", model_name="model-a"),
        ]
        context = _make_context(deployments, model_name="model-a")

        candidates = strategy._get_candidates(context)

        assert len(candidates) == 2
        models = [c["litellm_params"]["model"] for c in candidates]
        assert "gpt-4" in models
        assert "claude-3-opus" in models

    def test_empty_when_no_match(self):
        """Returns empty list when no deployments match."""
        strategy = CostAwareRoutingStrategy()

        deployments = [
            _make_deployment("gpt-4", model_name="other"),
        ]
        context = _make_context(deployments, model_name="test-model")

        candidates = strategy._get_candidates(context)

        assert candidates == []

    def test_uses_healthy_deployments(self):
        """Uses healthy_deployments attribute when available."""
        strategy = CostAwareRoutingStrategy()

        all_deps = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        healthy_deps = [_make_deployment("gpt-3.5-turbo")]

        router = MagicMock()
        router.model_list = all_deps
        router.healthy_deployments = healthy_deps

        context = RoutingContext(router=router, model="test-model")
        candidates = strategy._get_candidates(context)

        assert len(candidates) == 1
        assert candidates[0]["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Helpers for provider deployments
# ---------------------------------------------------------------------------


def _make_provider_deployment(
    model: str,
    provider: str,
    model_name: str = "test-model",
) -> Dict:
    """Create a deployment dict with custom_llm_provider."""
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": model,
            "custom_llm_provider": provider,
        },
    }


# ---------------------------------------------------------------------------
# Test: Pareto frontier computation
# ---------------------------------------------------------------------------


class TestParetoFrontier:
    """Test _compute_pareto_optimal method."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_single_candidate_is_pareto(self):
        """Single candidate is always on the Pareto frontier."""
        strategy = CostAwareRoutingStrategy()
        dep = _make_deployment("gpt-4")
        result = strategy._compute_pareto_optimal([dep], {"gpt-4": 1.0})
        assert len(result) == 1
        assert result[0] is dep

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_empty_candidates_returns_empty(self):
        """Empty candidate list returns empty."""
        strategy = CostAwareRoutingStrategy()
        result = strategy._compute_pareto_optimal([], {})
        assert result == []

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_dominated_candidate_removed(self):
        """A candidate dominated in both cost and quality is excluded."""
        strategy = CostAwareRoutingStrategy()

        cheap_dep = _make_deployment("gpt-3.5-turbo")  # cost: 0.001
        expensive_dep = _make_deployment("gpt-4")  # cost: 0.045

        # gpt-3.5-turbo is both cheaper AND higher quality -> dominates gpt-4
        quality_scores = {"gpt-3.5-turbo": 1.0, "gpt-4": 0.5}

        result = strategy._compute_pareto_optimal(
            [cheap_dep, expensive_dep], quality_scores
        )

        assert len(result) == 1
        assert result[0]["litellm_params"]["model"] == "gpt-3.5-turbo"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_pareto_tradeoff_both_kept(self):
        """Candidates with a cost-quality trade-off are both on the frontier."""
        strategy = CostAwareRoutingStrategy()

        cheap_dep = _make_deployment("gpt-3.5-turbo")  # cheap, low quality
        expensive_dep = _make_deployment("gpt-4")  # expensive, high quality

        # Trade-off: cheap is cheaper but lower quality
        quality_scores = {"gpt-3.5-turbo": 0.5, "gpt-4": 1.0}

        result = strategy._compute_pareto_optimal(
            [cheap_dep, expensive_dep], quality_scores
        )

        # Both should be on the frontier (neither dominates the other)
        assert len(result) == 2
        models = {r["litellm_params"]["model"] for r in result}
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_three_candidates_one_dominated(self):
        """With three candidates, only the dominated one is removed."""
        strategy = CostAwareRoutingStrategy()

        dep_cheap = _make_deployment("gpt-3.5-turbo")  # cost: 0.001
        dep_mid = _make_deployment("gpt-4-turbo")  # cost: 0.02
        dep_exp = _make_deployment("gpt-4")  # cost: 0.045

        # gpt-4-turbo is both cheaper than gpt-4 AND has equal quality -> dominates
        quality_scores = {"gpt-3.5-turbo": 0.3, "gpt-4-turbo": 0.9, "gpt-4": 0.9}

        result = strategy._compute_pareto_optimal(
            [dep_cheap, dep_mid, dep_exp], quality_scores
        )

        # gpt-4 is dominated by gpt-4-turbo (cheaper, same quality)
        models = {r["litellm_params"]["model"] for r in result}
        assert "gpt-3.5-turbo" in models
        assert "gpt-4-turbo" in models
        assert "gpt-4" not in models

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_pareto_integrated_into_select_deployment(self):
        """Pareto frontier filtering affects final selection."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.5,
            inner_strategy=inner,
        )

        # gpt-3.5-turbo: cheap, quality 0.5 (not preferred)
        # gpt-4-turbo: mid-price, quality 0.5 (not preferred)
        # gpt-4: expensive, quality 1.0 (preferred)
        # gpt-4-turbo is dominated by gpt-3.5-turbo (both quality 0.5, but
        # gpt-3.5 is cheaper)
        deployments = [
            _make_deployment("gpt-3.5-turbo"),
            _make_deployment("gpt-4-turbo"),
            _make_deployment("gpt-4"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        # gpt-4-turbo dominated -> not on frontier -> gpt-4 or gpt-3.5 wins
        assert result["litellm_params"]["model"] in ("gpt-3.5-turbo", "gpt-4")


# ---------------------------------------------------------------------------
# Test: Cost database refresh
# ---------------------------------------------------------------------------


class TestCostDatabaseRefresh:
    """Test _refresh_cost_db method and cached cost lookups."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_refresh_populates_cost_db(self):
        """_refresh_cost_db populates the internal cost database."""
        strategy = CostAwareRoutingStrategy(cost_refresh_interval=0)

        # Initially empty
        assert strategy._cost_db == {}

        strategy._refresh_cost_db()

        assert len(strategy._cost_db) > 0
        assert "gpt-4" in strategy._cost_db
        assert "gpt-3.5-turbo" in strategy._cost_db

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_refresh_respects_interval(self):
        """Cost DB is not refreshed if interval has not elapsed."""
        strategy = CostAwareRoutingStrategy(cost_refresh_interval=3600)
        strategy._last_cost_refresh = time.time()  # Just refreshed

        strategy._refresh_cost_db()

        # Should not have populated because interval not elapsed
        assert strategy._cost_db == {}

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_refresh_after_interval_elapsed(self):
        """Cost DB is refreshed after interval elapses."""
        strategy = CostAwareRoutingStrategy(cost_refresh_interval=1)
        strategy._last_cost_refresh = time.time() - 2  # Expired

        strategy._refresh_cost_db()

        assert len(strategy._cost_db) > 0

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_cost_db_stores_raw_per_token_sum(self):
        """Cost DB stores input_cost + output_cost per token."""
        strategy = CostAwareRoutingStrategy(cost_refresh_interval=0)
        strategy._refresh_cost_db()

        # gpt-4: input=0.00003 + output=0.00006 = 0.00009
        expected = 0.00003 + 0.00006
        assert abs(strategy._cost_db["gpt-4"] - expected) < 1e-12

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_get_model_cost_uses_cached_db(self):
        """_get_model_cost uses cached cost DB when available."""
        strategy = CostAwareRoutingStrategy(cost_refresh_interval=0)
        strategy._refresh_cost_db()

        cost = strategy._get_model_cost("gpt-4")

        # From cache: raw = 0.00003 + 0.00006 = 0.00009
        # per_1k = 0.00009 * 1000 / 2 = 0.045
        assert abs(cost - 0.045) < 1e-9

    def test_refresh_handles_missing_litellm_gracefully(self):
        """_refresh_cost_db does not crash when litellm is unavailable."""
        strategy = CostAwareRoutingStrategy(cost_refresh_interval=0)
        with patch.dict("sys.modules", {"litellm": None}):
            strategy._refresh_cost_db()  # Should not raise
        assert strategy._cost_db == {}

    @patch("litellm.model_cost", {"bad-model": "not-a-dict"})
    def test_refresh_skips_non_dict_entries(self):
        """Non-dict entries in model_cost are skipped."""
        strategy = CostAwareRoutingStrategy(cost_refresh_interval=0)
        strategy._refresh_cost_db()

        assert "bad-model" not in strategy._cost_db


# ---------------------------------------------------------------------------
# Test: Provider fallback with circuit breaker
# ---------------------------------------------------------------------------


class TestProviderFallbackCircuitBreaker:
    """Test _get_available_candidates with circuit breaker filtering."""

    def test_all_providers_healthy(self):
        """All candidates returned when no circuit breaker is open."""
        strategy = CostAwareRoutingStrategy(enable_circuit_breaker_filtering=True)

        deps = [
            _make_provider_deployment("gpt-4", "openai"),
            _make_provider_deployment("claude-3-opus", "anthropic"),
        ]

        # Mock circuit breaker manager where all breakers are closed
        mock_breaker = MagicMock()
        mock_breaker.is_open = False

        mock_cb_manager = MagicMock()
        mock_cb_manager.get_breaker.return_value = mock_breaker

        with (
            patch(
                "litellm_llmrouter.strategies.get_circuit_breaker_manager",
                return_value=mock_cb_manager,
                create=True,
            ),
            patch(
                "litellm_llmrouter.resilience.get_circuit_breaker_manager",
                return_value=mock_cb_manager,
            ),
        ):
            result = strategy._get_available_candidates(deps)

        assert len(result) == 2

    def test_one_provider_circuit_open(self):
        """Provider with open circuit breaker is filtered out."""
        strategy = CostAwareRoutingStrategy(enable_circuit_breaker_filtering=True)

        deps = [
            _make_provider_deployment("gpt-4", "openai"),
            _make_provider_deployment("claude-3-opus", "anthropic"),
        ]

        # openai breaker is open, anthropic is closed
        def make_breaker(name):
            breaker = MagicMock()
            breaker.is_open = name == "openai"
            return breaker

        mock_cb_manager = MagicMock()
        mock_cb_manager.get_breaker.side_effect = make_breaker

        with patch(
            "litellm_llmrouter.resilience.get_circuit_breaker_manager",
            return_value=mock_cb_manager,
        ):
            result = strategy._get_available_candidates(deps)

        assert len(result) == 1
        assert result[0]["litellm_params"]["model"] == "claude-3-opus"

    def test_all_providers_circuit_open_falls_back(self):
        """When all providers have open circuit breakers, returns all."""
        strategy = CostAwareRoutingStrategy(enable_circuit_breaker_filtering=True)

        deps = [
            _make_provider_deployment("gpt-4", "openai"),
            _make_provider_deployment("claude-3-opus", "anthropic"),
        ]

        mock_breaker = MagicMock()
        mock_breaker.is_open = True

        mock_cb_manager = MagicMock()
        mock_cb_manager.get_breaker.return_value = mock_breaker

        with patch(
            "litellm_llmrouter.resilience.get_circuit_breaker_manager",
            return_value=mock_cb_manager,
        ):
            result = strategy._get_available_candidates(deps)

        # Falls back to all candidates
        assert len(result) == 2

    def test_no_provider_info_included_by_default(self):
        """Deployments without custom_llm_provider are always included."""
        strategy = CostAwareRoutingStrategy(enable_circuit_breaker_filtering=True)

        deps = [
            _make_deployment("gpt-4"),  # No provider
            _make_provider_deployment("claude-3-opus", "anthropic"),
        ]

        mock_breaker = MagicMock()
        mock_breaker.is_open = True  # anthropic is open

        mock_cb_manager = MagicMock()
        mock_cb_manager.get_breaker.return_value = mock_breaker

        with patch(
            "litellm_llmrouter.resilience.get_circuit_breaker_manager",
            return_value=mock_cb_manager,
        ):
            result = strategy._get_available_candidates(deps)

        # gpt-4 (no provider) included, anthropic excluded but fallback kicks in
        # since at least one candidate has no provider
        assert any(d["litellm_params"]["model"] == "gpt-4" for d in result)

    def test_circuit_breaker_disabled(self):
        """When CB filtering is disabled, all candidates pass through."""
        strategy = CostAwareRoutingStrategy(enable_circuit_breaker_filtering=False)

        deps = [
            _make_provider_deployment("gpt-4", "openai"),
            _make_provider_deployment("claude-3-opus", "anthropic"),
        ]

        result = strategy._get_available_candidates(deps)
        assert len(result) == 2

    def test_import_error_returns_all(self):
        """When resilience module cannot be imported, returns all candidates."""
        strategy = CostAwareRoutingStrategy(enable_circuit_breaker_filtering=True)

        deps = [
            _make_provider_deployment("gpt-4", "openai"),
        ]

        with patch(
            "litellm_llmrouter.resilience.get_circuit_breaker_manager",
            side_effect=ImportError("not available"),
        ):
            result = strategy._get_available_candidates(deps)

        assert len(result) == 1

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_select_deployment_routes_around_open_cb(self):
        """select_deployment skips providers with open circuit breaker."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=1.0,  # Pure cost
            enable_circuit_breaker_filtering=True,
        )

        # gpt-3.5-turbo (openai) is cheapest but circuit open
        # claude-3-opus (anthropic) is more expensive but available
        deployments = [
            _make_provider_deployment("gpt-3.5-turbo", "openai"),
            _make_provider_deployment("claude-3-opus", "anthropic"),
        ]

        def make_breaker(name):
            breaker = MagicMock()
            breaker.is_open = name == "openai"
            return breaker

        mock_cb_manager = MagicMock()
        mock_cb_manager.get_breaker.side_effect = make_breaker

        context = _make_context(deployments)

        with patch(
            "litellm_llmrouter.resilience.get_circuit_breaker_manager",
            return_value=mock_cb_manager,
        ):
            result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "claude-3-opus"


# ---------------------------------------------------------------------------
# Test: Configuration via kwargs
# ---------------------------------------------------------------------------


class TestKwargsConfiguration:
    """Test configuration through constructor kwargs."""

    def test_quality_weight_default(self):
        """quality_weight defaults to 1 - cost_weight."""
        strategy = CostAwareRoutingStrategy(cost_weight=0.7)
        assert abs(strategy._quality_weight - 0.3) < 1e-9

    def test_quality_weight_explicit(self):
        """Explicit quality_weight overrides the default."""
        strategy = CostAwareRoutingStrategy(cost_weight=0.7, quality_weight=0.8)
        assert abs(strategy._quality_weight - 0.8) < 1e-9

    def test_quality_weight_clamped(self):
        """quality_weight is clamped to [0, 1]."""
        strategy = CostAwareRoutingStrategy(quality_weight=1.5)
        assert strategy._quality_weight == 1.0

        strategy = CostAwareRoutingStrategy(quality_weight=-0.3)
        assert strategy._quality_weight == 0.0

    def test_cost_refresh_interval(self):
        """cost_refresh_interval is stored correctly."""
        strategy = CostAwareRoutingStrategy(cost_refresh_interval=60)
        assert strategy._cost_refresh_interval == 60

    def test_cost_refresh_interval_negative_clamped(self):
        """Negative cost_refresh_interval is clamped to 0."""
        strategy = CostAwareRoutingStrategy(cost_refresh_interval=-100)
        assert strategy._cost_refresh_interval == 0

    def test_enable_circuit_breaker_filtering(self):
        """enable_circuit_breaker_filtering is stored correctly."""
        strategy = CostAwareRoutingStrategy(enable_circuit_breaker_filtering=False)
        assert strategy._enable_circuit_breaker_filtering is False

    def test_inner_strategy_name_stored(self):
        """inner_strategy_name is stored for lazy resolution."""
        strategy = CostAwareRoutingStrategy(inner_strategy_name="llmrouter-knn")
        assert strategy._inner_strategy_name == "llmrouter-knn"
        assert strategy._inner_strategy is None  # Not yet resolved

    def test_extra_kwargs_accepted(self):
        """Extra kwargs are accepted and ignored gracefully."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.5,
            unknown_param="value",
        )
        assert strategy._quality_threshold == 0.5


# ---------------------------------------------------------------------------
# Test: Inner strategy lazy resolution by name
# ---------------------------------------------------------------------------


class TestInnerStrategyLazyResolution:
    """Test _resolve_inner_strategy via inner_strategy_name."""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_resolve_returns_direct_inner_if_set(self):
        """If inner_strategy is set directly, resolution returns it."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(inner_strategy=inner)

        resolved = strategy._resolve_inner_strategy()
        assert resolved is inner

    def test_resolve_by_name_from_registry(self):
        """Inner strategy resolved by name from the global registry."""
        inner = MockInnerStrategy("gpt-4")

        # Register the inner strategy in the global registry
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        registry.register("mock-inner", inner)

        strategy = CostAwareRoutingStrategy(inner_strategy_name="mock-inner")

        resolved = strategy._resolve_inner_strategy()
        assert resolved is inner
        # Should also cache the resolved strategy
        assert strategy._inner_strategy is inner

    def test_resolve_by_name_not_found(self):
        """Returns None if inner_strategy_name is not in registry."""
        strategy = CostAwareRoutingStrategy(inner_strategy_name="nonexistent-strategy")

        resolved = strategy._resolve_inner_strategy()
        assert resolved is None

    def test_resolve_with_no_inner_config(self):
        """Returns None when neither inner_strategy nor name is set."""
        strategy = CostAwareRoutingStrategy()

        resolved = strategy._resolve_inner_strategy()
        assert resolved is None

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_select_deployment_uses_resolved_inner(self):
        """select_deployment uses inner strategy resolved by name."""
        inner = MockInnerStrategy("gpt-4")

        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        registry.register("mock-inner", inner)

        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.8,
            cost_weight=0.5,
            inner_strategy_name="mock-inner",
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # gpt-4 gets quality 1.0 from inner, gpt-3.5 gets 0.5 (below 0.8)
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Test: Version updated
# ---------------------------------------------------------------------------


class TestVersionUpdate:
    """Test that the strategy version reflects the v2 enhancements."""

    def test_version_is_2_0_0(self):
        """Version should be 2.0.0 after enhancements."""
        strategy = CostAwareRoutingStrategy()
        assert strategy.version == "2.0.0"

    def test_cost_db_initialized_empty(self):
        """Cost database should be initialized empty."""
        strategy = CostAwareRoutingStrategy()
        assert strategy._cost_db == {}
        assert strategy._last_cost_refresh == 0.0
