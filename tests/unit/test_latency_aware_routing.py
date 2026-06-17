"""
Tests for the latency-SLA / usage-based / least-busy routers (RouteIQ-904b).

Upstream LiteLLM latency-based / usage-based / least-busy strategies are
bypassed by RouteIQ's custom routing strategy; these are RouteIQ-native
re-implementations reading LIVE signals. Verifies each selection rule with
the signals mocked:

Latency-SLA:
1. Picks a deployment whose percentile latency is at/below the target.
2. When none meet the target, picks the lowest-latency deployment.
3. Cold start (no samples) is optimistic (meets SLA) -> first candidate.

Usage-based:
4. Picks the least-used deployment per the live stats accumulator.
5. Fail-open: no accumulator data -> first-candidate order.

Least-busy:
6. Picks the deployment with the fewest in-flight requests.
7. release() floors at 0 (no negative on double release).
8. select_deployment() acquires the chosen slot.

Plus registration (all three under one flag) + opt-in default-off.
"""

from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.router_decision_callback import (
    get_stats_accumulator,
    reset_stats_accumulator,
)
from litellm_llmrouter.strategies import (
    DEFAULT_ROUTER_HPARAMS,
    LLMROUTER_STRATEGIES,
    LatencySLARoutingStrategy,
    LeastBusyRoutingStrategy,
    UsageBasedRoutingStrategy,
    register_latency_aware_strategies,
)
from litellm_llmrouter.strategy_registry import (
    RoutingContext,
    get_routing_registry,
    reset_routing_singletons,
)


def _make_deployment(model: str, model_name: str = "test-model") -> Dict:
    return {"model_name": model_name, "litellm_params": {"model": model}}


def _make_router(deployments: List[Dict]) -> MagicMock:
    router = MagicMock()
    router.model_list = deployments
    router.healthy_deployments = deployments
    return router


def _make_context(deployments: List[Dict], model_name: str = "test-model"):
    return RoutingContext(router=_make_router(deployments), model=model_name)


@pytest.fixture(autouse=True)
def _reset():
    reset_routing_singletons()
    reset_stats_accumulator()
    yield
    reset_routing_singletons()
    reset_stats_accumulator()


# ---------------------------------------------------------------------------
# Latency-SLA
# ---------------------------------------------------------------------------


def test_latency_sla_picks_under_target():
    deployments = [_make_deployment("slow"), _make_deployment("fast")]
    strat = LatencySLARoutingStrategy(
        p_latency_target_ms=2000.0, percentile=0.95, max_latency_samples=64
    )
    for _ in range(30):
        strat.record_latency("slow", 5000.0)
        strat.record_latency("fast", 400.0)
    ctx = _make_context(deployments)
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "fast"


def test_latency_sla_none_meets_target_picks_lowest():
    deployments = [_make_deployment("slower"), _make_deployment("slow")]
    strat = LatencySLARoutingStrategy(p_latency_target_ms=100.0, percentile=0.95)
    for _ in range(20):
        strat.record_latency("slower", 9000.0)
        strat.record_latency("slow", 3000.0)
    ctx = _make_context(deployments)
    result = strat.select_deployment(ctx)
    # neither below 100ms target -> picks the lowest-latency one
    assert result["litellm_params"]["model"] == "slow"


def test_latency_sla_cold_start_optimistic():
    deployments = [_make_deployment("a"), _make_deployment("b")]
    strat = LatencySLARoutingStrategy(p_latency_target_ms=2000.0)
    # no samples recorded -> optimistic -> first candidate
    ctx = _make_context(deployments)
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "a"


def test_latency_sla_no_candidates_none():
    strat = LatencySLARoutingStrategy()
    ctx = _make_context([], model_name="nope")
    assert strat.select_deployment(ctx) is None


def test_latency_sla_validate():
    ok, err = LatencySLARoutingStrategy().validate()
    assert ok and err is None


# ---------------------------------------------------------------------------
# Usage-based
# ---------------------------------------------------------------------------


def test_usage_based_picks_least_used():
    deployments = [_make_deployment("hot"), _make_deployment("cold")]
    acc = get_stats_accumulator()
    for _ in range(12):
        acc.record_decision(model="hot")
    # 'cold' has 0 decisions -> least-used -> selected
    strat = UsageBasedRoutingStrategy()
    ctx = _make_context(deployments)
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "cold"


def test_usage_based_failopen_no_data():
    deployments = [_make_deployment("x"), _make_deployment("y")]
    # no decisions recorded -> all counts 0 -> first candidate (deterministic)
    strat = UsageBasedRoutingStrategy()
    ctx = _make_context(deployments)
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "x"


def test_usage_based_no_candidates_none():
    strat = UsageBasedRoutingStrategy()
    ctx = _make_context([], model_name="nope")
    assert strat.select_deployment(ctx) is None


# ---------------------------------------------------------------------------
# Least-busy
# ---------------------------------------------------------------------------


def test_least_busy_picks_idle():
    deployments = [_make_deployment("busy"), _make_deployment("idle")]
    strat = LeastBusyRoutingStrategy()
    for _ in range(5):
        strat.acquire("busy")
    ctx = _make_context(deployments)
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "idle"


def test_least_busy_select_acquires_slot():
    deployments = [_make_deployment("a"), _make_deployment("b")]
    strat = LeastBusyRoutingStrategy()
    ctx = _make_context(deployments)
    selected = strat.select_deployment(ctx)["litellm_params"]["model"]
    # selecting it should have incremented its in-flight count
    assert strat.in_flight(selected) == 1


def test_least_busy_release_floors_at_zero():
    strat = LeastBusyRoutingStrategy()
    strat.acquire("m")
    strat.release("m")
    assert strat.in_flight("m") == 0
    # double release / unknown release must not go negative
    strat.release("m")
    strat.release("unknown")
    assert strat.in_flight("m") == 0
    assert strat.in_flight("unknown") == 0


def test_least_busy_round_robins_without_release():
    deployments = [_make_deployment("a"), _make_deployment("b")]
    strat = LeastBusyRoutingStrategy()
    ctx = _make_context(deployments)
    picks = [strat.select_deployment(ctx)["litellm_params"]["model"] for _ in range(4)]
    # each pick increments the chosen arm, so it alternates a, b, a, b
    assert picks == ["a", "b", "a", "b"]


def test_least_busy_no_candidates_none():
    strat = LeastBusyRoutingStrategy()
    ctx = _make_context([], model_name="nope")
    assert strat.select_deployment(ctx) is None


# ---------------------------------------------------------------------------
# Registration / selectability
# ---------------------------------------------------------------------------


def test_all_three_in_strategy_list():
    for name in (
        "llmrouter-latency-sla",
        "llmrouter-usage-based",
        "llmrouter-least-busy",
    ):
        assert name in LLMROUTER_STRATEGIES
    assert "latency-sla" in DEFAULT_ROUTER_HPARAMS
    assert "usage-based" in DEFAULT_ROUTER_HPARAMS
    assert "least-busy" in DEFAULT_ROUTER_HPARAMS


def test_register_disabled_by_default():
    assert register_latency_aware_strategies() is False


def test_register_when_enabled(monkeypatch):
    from litellm_llmrouter import settings as settings_mod

    settings_mod.reset_settings()
    monkeypatch.setenv("ROUTEIQ_LATENCY_AWARE__ENABLED", "true")
    settings_mod.reset_settings()
    try:
        assert register_latency_aware_strategies() is True
        registry = get_routing_registry()
        assert isinstance(
            registry.get("llmrouter-latency-sla"), LatencySLARoutingStrategy
        )
        assert isinstance(
            registry.get("llmrouter-usage-based"), UsageBasedRoutingStrategy
        )
        assert isinstance(
            registry.get("llmrouter-least-busy"), LeastBusyRoutingStrategy
        )
    finally:
        settings_mod.reset_settings()
