"""Tests for strategy comparison endpoint."""

import pytest

from litellm_llmrouter.strategy_registry import (
    get_strategy_comparison,
    get_routing_registry,
    reset_routing_singletons,
    RoutingStrategy,
    RoutingContext,
    StrategyState,
)
from typing import Dict, Optional


class _DummyStrategy(RoutingStrategy):
    """Minimal strategy for testing."""

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        return None

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def version(self) -> Optional[str]:
        return "1.0.0"


class TestGetStrategyComparison:
    """Test get_strategy_comparison function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_returns_dict_with_required_keys(self):
        """get_strategy_comparison returns a dict with expected keys."""
        result = get_strategy_comparison()
        assert isinstance(result, dict)
        assert "strategies" in result
        assert "total" in result
        assert "active_strategy" in result
        assert "ab_enabled" in result
        assert "ab_weights" in result

    def test_strategies_is_list(self):
        """strategies field is a list."""
        result = get_strategy_comparison()
        assert isinstance(result["strategies"], list)

    def test_total_matches_strategies_length(self):
        """total matches the number of strategies."""
        result = get_strategy_comparison()
        assert result["total"] == len(result["strategies"])

    def test_empty_registry_returns_zero(self):
        """Empty registry returns zero strategies."""
        result = get_strategy_comparison()
        assert result["total"] == 0
        assert result["strategies"] == []

    def test_registered_strategy_appears(self):
        """A registered strategy appears in the comparison."""
        registry = get_routing_registry()
        strategy = _DummyStrategy()
        registry.register("test-strategy", strategy, version="1.0.0")

        result = get_strategy_comparison()

        assert result["total"] == 1
        assert len(result["strategies"]) == 1

        entry = result["strategies"][0]
        assert entry["name"] == "test-strategy"
        assert entry["version"] == "1.0.0"
        assert entry["active"] is True
        assert entry["state"] == StrategyState.ACTIVE.value

    def test_multiple_strategies(self):
        """Multiple strategies are listed."""
        registry = get_routing_registry()
        registry.register("strategy-a", _DummyStrategy(), version="1.0")
        registry.register("strategy-b", _DummyStrategy(), version="2.0")

        result = get_strategy_comparison()

        assert result["total"] == 2
        names = {s["name"] for s in result["strategies"]}
        assert names == {"strategy-a", "strategy-b"}

    def test_active_strategy_reported(self):
        """The active strategy is reported correctly."""
        registry = get_routing_registry()
        registry.register("strat-1", _DummyStrategy())
        registry.register("strat-2", _DummyStrategy())
        registry.set_active("strat-2")

        result = get_strategy_comparison()

        assert result["active_strategy"] == "strat-2"

    def test_ab_weights_reported(self):
        """A/B weights are reported when set."""
        registry = get_routing_registry()
        registry.register("control", _DummyStrategy())
        registry.register("treatment", _DummyStrategy())
        registry.set_weights({"control": 90, "treatment": 10})

        result = get_strategy_comparison()

        assert result["ab_enabled"] is True
        assert result["ab_weights"] == {"control": 90, "treatment": 10}

    def test_no_ab_weights_by_default(self):
        """A/B is disabled by default."""
        result = get_strategy_comparison()
        assert result["ab_enabled"] is False
        assert result["ab_weights"] == {}

    def test_strategy_entry_has_family(self):
        """Strategy entry includes family field."""
        registry = get_routing_registry()
        registry.register("llmrouter-knn:v2", _DummyStrategy(), family="llmrouter-knn")

        result = get_strategy_comparison()

        assert result["strategies"][0]["family"] == "llmrouter-knn"

    def test_strategy_entry_has_registered_at(self):
        """Strategy entry includes registered_at timestamp."""
        registry = get_routing_registry()
        registry.register("test", _DummyStrategy())

        result = get_strategy_comparison()

        assert "registered_at" in result["strategies"][0]
        assert isinstance(result["strategies"][0]["registered_at"], float)
