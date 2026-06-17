"""
Tests for the LinUCB feature-vector contextual bandit (RouteIQ-6c67).

The Kumaraswamy-Thompson bandit is bucket-contextual; LinUCB is feature-vector
contextual. It is ADDITIVE alongside KT (separate class / registry name / flag).
Verifies:
1. Cold start EXPLORES: untried arms get selected before evidence accumulates.
2. As reward evidence accumulates, the higher-reward arm comes to DOMINATE.
3. The Sherman-Morrison inverse stays correct (matches a brute-force inverse).
4. The feature vector has the documented fixed dimension.
5. Single-candidate / no-candidate edge cases.
6. update() reward shaping ([-1,1] -> [0,1]) and no-op without context.
7. Registration: in LLMROUTER_STRATEGIES + selectable by name + opt-in.
8. Determinism with a fixed seed.
"""

from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.kumaraswamy_thompson import (
    LINUCB_STRATEGY_NAME,
    LinUCBRoutingStrategy,
    _LinUCBArm,
    identity_matrix,
    register_linucb_strategy,
    sherman_morrison_update,
)
from litellm_llmrouter.strategies import (
    DEFAULT_ROUTER_HPARAMS,
    LLMROUTER_STRATEGIES,
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


def _ctx(deployments: List[Dict], request_id: str, text: str = "hello world"):
    return RoutingContext(
        router=_make_router(deployments),
        model="test-model",
        input=text,
        request_id=request_id,
    )


@pytest.fixture(autouse=True)
def _reset():
    reset_routing_singletons()
    yield
    reset_routing_singletons()


# ---------------------------------------------------------------------------
# Cold-start exploration
# ---------------------------------------------------------------------------


def test_cold_start_explores_multiple_arms():
    deployments = [_make_deployment("a"), _make_deployment("b")]
    strat = LinUCBRoutingStrategy(alpha=1.0, seed=11)
    picks = {
        strat.select_deployment(_ctx(deployments, f"r{i}"))["litellm_params"]["model"]
        for i in range(12)
    }
    # cold start must not collapse onto a single arm
    assert len(picks) > 1


# ---------------------------------------------------------------------------
# Convergence: higher-reward arm dominates as evidence accumulates
# ---------------------------------------------------------------------------


def test_higher_reward_arm_dominates_after_learning():
    deployments = [_make_deployment("cheap"), _make_deployment("good")]
    strat = LinUCBRoutingStrategy(alpha=0.3, seed=3)
    # Train: 'good' rewarded +1, 'cheap' rewarded -1, same-ish context.
    for i in range(50):
        ctx = _ctx(deployments, f"t{i}", text="solve this hard reasoning problem")
        sel = strat.select_deployment(ctx)["litellm_params"]["model"]
        strat.update(sel, 1.0 if sel == "good" else -1.0, request_id=f"t{i}")
    # Evaluate post-learning selections.
    final = [
        strat.select_deployment(
            _ctx(deployments, f"f{i}", "solve this hard reasoning problem")
        )["litellm_params"]["model"]
        for i in range(20)
    ]
    assert final.count("good") > final.count("cheap")
    assert final.count("good") >= 15  # strong domination


def test_explicit_features_update_drives_selection():
    deployments = [_make_deployment("x"), _make_deployment("y")]
    strat = LinUCBRoutingStrategy(alpha=0.1, seed=5)
    # Push strong reward to 'y' for the actual request context many times.
    for i in range(40):
        ctx = _ctx(deployments, f"e{i}", "fixed context text")
        # observe the context vector used, reward y high, x low
        strat.update("y", 1.0, request_id=None, features=strat._featurize(ctx))
        strat.update("x", -1.0, request_id=None, features=strat._featurize(ctx))
    sel = strat.select_deployment(_ctx(deployments, "fin", "fixed context text"))
    assert sel["litellm_params"]["model"] == "y"


# ---------------------------------------------------------------------------
# Sherman-Morrison correctness
# ---------------------------------------------------------------------------


def _brute_inverse(m: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan inverse (test oracle only)."""
    n = len(m)
    aug = [
        row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)
    ]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        pivval = aug[col][col]
        aug[col] = [v / pivval for v in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def test_sherman_morrison_matches_brute_inverse():
    d = 4
    a_inv = identity_matrix(d)
    # Apply several rank-1 updates and track A explicitly.
    a = identity_matrix(d)
    vectors = [
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.0, 0.0, 2.0],
    ]
    for x in vectors:
        a_inv = sherman_morrison_update(a_inv, x)
        for i in range(d):
            for j in range(d):
                a[i][j] += x[i] * x[j]
    expected = _brute_inverse(a)
    for i in range(d):
        for j in range(d):
            assert abs(a_inv[i][j] - expected[i][j]) < 1e-9


def test_arm_ucb_decreases_with_evidence():
    arm = _LinUCBArm.fresh(3)
    x = [1.0, 1.0, 0.0]
    cold_width = arm.ucb(x, alpha=1.0) - arm.ucb(x, alpha=0.0)
    for _ in range(20):
        arm.update(x, 1.0)
    warm_width = arm.ucb(x, alpha=1.0) - arm.ucb(x, alpha=0.0)
    # the exploration term shrinks as evidence in direction x accumulates
    assert warm_width < cold_width


# ---------------------------------------------------------------------------
# Feature vector shape
# ---------------------------------------------------------------------------


def test_feature_vector_dimension():
    strat = LinUCBRoutingStrategy(tenant_buckets=8)
    ctx = _ctx([_make_deployment("a")], "r0")
    x = strat._featurize(ctx)
    # bias(1) + prompt_len(1) + profiles(3) + tenant one-hot(8) = 13
    assert len(x) == 13
    assert x[0] == 1.0  # bias


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_candidate_returns_it():
    deployments = [_make_deployment("only")]
    strat = LinUCBRoutingStrategy(seed=1)
    result = strat.select_deployment(_ctx(deployments, "r0"))
    assert result["litellm_params"]["model"] == "only"


def test_no_candidates_returns_none():
    strat = LinUCBRoutingStrategy(seed=1)
    ctx = RoutingContext(router=_make_router([]), model="nonexistent", request_id="r0")
    assert strat.select_deployment(ctx) is None


def test_update_no_context_is_noop():
    strat = LinUCBRoutingStrategy(seed=1)
    # no features, no logged request_id -> no-op, must not raise / create an arm
    strat.update("m", 1.0, request_id="never-seen")
    assert "m" not in strat._arms


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_seeded_runs_are_deterministic():
    deployments = [_make_deployment("a"), _make_deployment("b")]

    def run():
        s = LinUCBRoutingStrategy(alpha=1.0, seed=99)
        return [
            s.select_deployment(_ctx(deployments, f"r{i}"))["litellm_params"]["model"]
            for i in range(15)
        ]

    assert run() == run()


# ---------------------------------------------------------------------------
# Registration / selectability
# ---------------------------------------------------------------------------


def test_registered_in_strategy_list():
    assert "llmrouter-linucb" in LLMROUTER_STRATEGIES
    assert "linucb" in DEFAULT_ROUTER_HPARAMS


def test_register_disabled_by_default():
    assert register_linucb_strategy() is False


def test_register_when_enabled(monkeypatch):
    from litellm_llmrouter import settings as settings_mod

    settings_mod.reset_settings()
    monkeypatch.setenv("ROUTEIQ_LINUCB__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_LINUCB__ALPHA", "0.5")
    settings_mod.reset_settings()
    try:
        assert register_linucb_strategy() is True
        registry = get_routing_registry()
        strat = registry.get(LINUCB_STRATEGY_NAME)
        assert isinstance(strat, LinUCBRoutingStrategy)
        assert strat.name == "llmrouter-linucb"
    finally:
        settings_mod.reset_settings()
