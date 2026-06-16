"""
Tests for the CostCascadeRoutingStrategy (RouteIQ-90d0).

Cost-aware speculative cascade / escalation. DISTINCT from the heuristic-Pareto
CostAwareRoutingStrategy. Verifies:
1. Cheapest rung is selected first (cheapest-first invariant).
2. The ordered escalation ladder (cheap -> strong) is exposed in metadata.
3. A prior-attempt rung signal escalates to the NEXT rung (mode b).
4. A low prior-attempt confidence signal escalates one rung (mode b).
5. Cost-unknown arms sort LAST and degrade gracefully (all-unknown still routes).
6. Strategy registration: in LLMROUTER_STRATEGIES + selectable by name.
7. The router's shared deployment dict is never mutated (shallow-copy ladder).
8. validate() bounds-checks config.
"""

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.strategies import (
    CostCascadeRoutingStrategy,
    LLMROUTER_STRATEGIES,
    DEFAULT_ROUTER_HPARAMS,
    register_cost_cascade_strategy,
)
from litellm_llmrouter.strategy_registry import (
    RoutingContext,
    get_routing_registry,
    reset_routing_singletons,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deployment(model: str, model_name: str = "test-model") -> Dict:
    return {"model_name": model_name, "litellm_params": {"model": model}}


def _make_router(deployments: List[Dict]) -> MagicMock:
    router = MagicMock()
    router.model_list = deployments
    router.healthy_deployments = deployments
    return router


def _make_context(
    deployments: List[Dict],
    model_name: str = "test-model",
    request_kwargs: Dict = None,
    metadata: Dict = None,
) -> RoutingContext:
    return RoutingContext(
        router=_make_router(deployments),
        model=model_name,
        request_kwargs=request_kwargs,
        metadata=metadata or {},
    )


# gpt-3.5 cheapest, gpt-4-turbo middle, gpt-4 most expensive
MOCK_MODEL_COST = {
    "gpt-3.5-turbo": {
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
    },
    "gpt-4-turbo": {
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
    },
    "gpt-4": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
}


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_routing_singletons()
    yield
    reset_routing_singletons()


def _patch_cost():
    return patch("litellm.model_cost", MOCK_MODEL_COST)


# ---------------------------------------------------------------------------
# Cheapest-first
# ---------------------------------------------------------------------------


def test_selects_cheapest_rung_first():
    deployments = [
        _make_deployment("gpt-4"),
        _make_deployment("gpt-4-turbo"),
        _make_deployment("gpt-3.5-turbo"),
    ]
    ctx = _make_context(deployments)
    strat = CostCascadeRoutingStrategy()
    with _patch_cost():
        result = strat.select_deployment(ctx)
    assert result is not None
    assert result["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Ladder ordering exposed in metadata
# ---------------------------------------------------------------------------


def test_ladder_ordering_exposed_in_metadata():
    deployments = [
        _make_deployment("gpt-4"),
        _make_deployment("gpt-3.5-turbo"),
        _make_deployment("gpt-4-turbo"),
    ]
    ctx = _make_context(deployments)
    strat = CostCascadeRoutingStrategy()
    with _patch_cost():
        result = strat.select_deployment(ctx)
    cascade = result["metadata"]["routeiq_cascade"]
    ladder_models = [rung["model"] for rung in cascade["ladder"]]
    # cheapest -> strongest
    assert ladder_models == ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"]
    assert cascade["selected_rung"] == 0
    assert cascade["escalated"] is False
    # cheapest rung cost descriptor present (not None / inf)
    assert cascade["ladder"][0]["cost_per_1k"] is not None


# ---------------------------------------------------------------------------
# Mode (b): prior rung signal escalates to next rung
# ---------------------------------------------------------------------------


def test_prior_rung_signal_escalates_to_next_rung():
    deployments = [
        _make_deployment("gpt-3.5-turbo"),
        _make_deployment("gpt-4-turbo"),
        _make_deployment("gpt-4"),
    ]
    # caller already tried rung 0 (cheapest) -> escalate to rung 1
    ctx = _make_context(deployments, request_kwargs={"cascade_rung": 0})
    strat = CostCascadeRoutingStrategy()
    with _patch_cost():
        result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4-turbo"
    cascade = result["metadata"]["routeiq_cascade"]
    assert cascade["selected_rung"] == 1
    assert cascade["escalated"] is True


def test_prior_rung_signal_from_metadata():
    deployments = [
        _make_deployment("gpt-3.5-turbo"),
        _make_deployment("gpt-4-turbo"),
        _make_deployment("gpt-4"),
    ]
    ctx = _make_context(deployments, metadata={"cascade_rung": 1})
    strat = CostCascadeRoutingStrategy()
    with _patch_cost():
        result = strat.select_deployment(ctx)
    # tried rung 1 -> escalate to rung 2 (strongest)
    assert result["litellm_params"]["model"] == "gpt-4"


def test_escalation_clamped_to_strongest_rung():
    deployments = [
        _make_deployment("gpt-3.5-turbo"),
        _make_deployment("gpt-4"),
    ]
    # already tried last rung -> cannot escalate past strongest
    ctx = _make_context(deployments, request_kwargs={"cascade_rung": 5})
    strat = CostCascadeRoutingStrategy()
    with _patch_cost():
        result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4"
    assert result["metadata"]["routeiq_cascade"]["selected_rung"] == 1


# ---------------------------------------------------------------------------
# Mode (b): low-confidence signal escalates one rung
# ---------------------------------------------------------------------------


def test_low_confidence_escalates_one_rung():
    deployments = [
        _make_deployment("gpt-3.5-turbo"),
        _make_deployment("gpt-4"),
    ]
    # confidence 0.2 < threshold 0.7 -> escalate to rung 1
    ctx = _make_context(deployments, metadata={"cascade_confidence": 0.2})
    strat = CostCascadeRoutingStrategy(confidence_threshold=0.7)
    with _patch_cost():
        result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4"
    assert result["metadata"]["routeiq_cascade"]["escalated"] is True


def test_high_confidence_stays_on_cheapest():
    deployments = [
        _make_deployment("gpt-3.5-turbo"),
        _make_deployment("gpt-4"),
    ]
    # confidence 0.95 >= threshold 0.7 -> stay cheapest
    ctx = _make_context(deployments, metadata={"cascade_confidence": 0.95})
    strat = CostCascadeRoutingStrategy(confidence_threshold=0.7)
    with _patch_cost():
        result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-3.5-turbo"
    assert result["metadata"]["routeiq_cascade"]["escalated"] is False


# ---------------------------------------------------------------------------
# Cost-unknown graceful fallback
# ---------------------------------------------------------------------------


def test_cost_unknown_sorts_last():
    # gpt-3.5 is known/cheap, mystery-model is unknown -> unknown sorts LAST
    deployments = [
        _make_deployment("mystery-model"),
        _make_deployment("gpt-3.5-turbo"),
    ]
    ctx = _make_context(deployments)
    strat = CostCascadeRoutingStrategy()
    with _patch_cost():
        result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-3.5-turbo"
    ladder = result["metadata"]["routeiq_cascade"]["ladder"]
    # unknown-cost rung carries cost_per_1k=None
    assert ladder[-1]["model"] == "mystery-model"
    assert ladder[-1]["cost_per_1k"] is None


def test_all_unknown_cost_still_routes():
    deployments = [
        _make_deployment("mystery-a"),
        _make_deployment("mystery-b"),
    ]
    ctx = _make_context(deployments)
    strat = CostCascadeRoutingStrategy()
    with _patch_cost():
        result = strat.select_deployment(ctx)
    # graceful: still routes to a deployment (stable order keeps first)
    assert result is not None
    assert result["litellm_params"]["model"] == "mystery-a"


def test_no_candidates_returns_none():
    ctx = _make_context([], model_name="nonexistent")
    strat = CostCascadeRoutingStrategy()
    with _patch_cost():
        assert strat.select_deployment(ctx) is None


# ---------------------------------------------------------------------------
# No mutation of shared router deployment
# ---------------------------------------------------------------------------


def test_does_not_mutate_router_deployment():
    dep = _make_deployment("gpt-3.5-turbo")
    deployments = [dep, _make_deployment("gpt-4")]
    ctx = _make_context(deployments)
    strat = CostCascadeRoutingStrategy()
    with _patch_cost():
        result = strat.select_deployment(ctx)
    # the returned dict carries cascade metadata...
    assert "routeiq_cascade" in result["metadata"]
    # ...but the original shared deployment dict is untouched
    assert "metadata" not in dep or "routeiq_cascade" not in dep.get("metadata", {})


# ---------------------------------------------------------------------------
# Registration / selectability
# ---------------------------------------------------------------------------


def test_registered_in_strategy_list():
    assert "llmrouter-cost-cascade" in LLMROUTER_STRATEGIES
    assert "cost-cascade" in DEFAULT_ROUTER_HPARAMS


def test_register_disabled_by_default():
    # default settings.cost_cascade.enabled is False -> no registration
    assert register_cost_cascade_strategy() is False


def test_register_when_enabled(monkeypatch):
    from litellm_llmrouter import settings as settings_mod

    settings_mod.reset_settings()
    monkeypatch.setenv("ROUTEIQ_COST_CASCADE__ENABLED", "true")
    settings_mod.reset_settings()
    try:
        assert register_cost_cascade_strategy() is True
        registry = get_routing_registry()
        strat = registry.get("llmrouter-cost-cascade")
        assert isinstance(strat, CostCascadeRoutingStrategy)
        assert strat.name == "llmrouter-cost-cascade"
    finally:
        settings_mod.reset_settings()


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_ok():
    ok, err = CostCascadeRoutingStrategy().validate()
    assert ok and err is None


def test_validate_rejects_bad_threshold():
    strat = CostCascadeRoutingStrategy()
    strat._confidence_threshold = 1.5  # bypass clamp to exercise validate
    ok, err = strat.validate()
    assert not ok
    assert "confidence_threshold" in err
