"""Unit tests for the Kumaraswamy-Thompson bandit routing strategy.

Covers: the closed-form quantile/CDF math (range, monotonicity, round-trip, the
log-space stability fix vs the naive form, the Beta special case), the conjugate
posterior update + fractional reward, decay toward the prior, dynamic arms
(circuit-breaker drop, vanish/reappear), cold-start warm-start, the arm key
(``litellm_params.model``), registration through the existing pipeline with no
LiteLLM-mount edit, and the in-memory default backend requiring no external deps.

Every test runs with the in-memory backend and NO Redis/Aurora/env dependency.
"""

from __future__ import annotations

import pytest

from litellm_llmrouter.kumaraswamy_thompson import (
    STRATEGY_NAME,
    STRATEGY_VERSION,
    InMemoryPosteriorBackend,
    KumaraswamyThompsonStrategy,
    Posterior,
    _q_naive,
    kumaraswamy_cdf,
    kumaraswamy_quantile,
    register_kumaraswamy_thompson_strategy,
    sample_kumaraswamy,
)
from litellm_llmrouter.strategy_registry import RoutingContext, get_routing_registry


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeRouter:
    """Minimal LiteLLM-router stand-in exposing healthy_deployments / model_list."""

    def __init__(self, deployments):
        self.healthy_deployments = deployments
        self.model_list = deployments


def _dep(model_name: str, arm: str, provider: str = "") -> dict:
    params = {"model": arm}
    if provider:
        params["custom_llm_provider"] = provider
    return {"model_name": model_name, "litellm_params": params}


def _ctx(router, model="gpt-group", request_id="req-1", messages=None):
    return RoutingContext(
        router=router,
        model=model,
        messages=messages or [{"role": "user", "content": "hello world"}],
        request_id=request_id,
    )


# ===========================================================================
# 1. Sampler math
# ===========================================================================


def test_sampler_determinism_two_instances_same_seed():
    s1 = KumaraswamyThompsonStrategy(seed=42)
    s2 = KumaraswamyThompsonStrategy(seed=42)
    draws1 = [sample_kumaraswamy(2.0, 5.0, s1._rng) for _ in range(50)]
    draws2 = [sample_kumaraswamy(2.0, 5.0, s2._rng) for _ in range(50)]
    assert draws1 == draws2


def test_sampler_uses_threaded_rng_not_global():
    # Two instances with different seeds must diverge (no shared global state).
    s1 = KumaraswamyThompsonStrategy(seed=1)
    s2 = KumaraswamyThompsonStrategy(seed=2)
    d1 = [sample_kumaraswamy(2.0, 5.0, s1._rng) for _ in range(20)]
    d2 = [sample_kumaraswamy(2.0, 5.0, s2._rng) for _ in range(20)]
    assert d1 != d2


@pytest.mark.parametrize("u", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
@pytest.mark.parametrize("a,b", [(1.0, 1.0), (2.0, 5.0), (0.5, 0.5), (10.0, 3.0)])
def test_quantile_range(u, a, b):
    x = kumaraswamy_quantile(u, a, b)
    assert 0.0 < x < 1.0


def test_quantile_monotone_increasing_in_u():
    a, b = 2.0, 3.0
    prev = -1.0
    for i in range(1, 100):
        u = i / 100.0
        x = kumaraswamy_quantile(u, a, b)
        assert x >= prev
        prev = x


def test_quantile_uniform_special_case():
    # Kumaraswamy(1,1) == Uniform => Q(0.5) == 0.5.
    assert kumaraswamy_quantile(0.5, 1.0, 1.0) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("a", [0.3, 1.0, 2.0, 7.5, 30.0])
@pytest.mark.parametrize("b", [0.3, 1.0, 3.0, 12.0, 30.0])
@pytest.mark.parametrize("u", [0.01, 0.25, 0.5, 0.75, 0.99])
def test_round_trip_well_conditioned(a, b, u):
    # Architect verified worst round-trip 2.96e-12 in this band; assert < 1e-9.
    x = kumaraswamy_quantile(u, a, b)
    back = kumaraswamy_cdf(x, a, b)
    assert abs(back - u) < 1e-9


def test_stability_logspace_beats_naive_on_small_u_small_b():
    # The load-bearing fix: naive form underflows to exactly 0.0; log-space stays positive.
    u, a, b = 1e-18, 1.0, 0.05
    assert kumaraswamy_quantile(u, a, b) > 0.0
    assert _q_naive(u, a, b) == 0.0


def test_stability_extreme_small_u():
    # u=1e-300 -> naive 0.0, log-space positive (correct ~1e-150 for a=1,b=2).
    assert kumaraswamy_quantile(1e-300, 1.0, 2.0) > 0.0
    assert _q_naive(1e-300, 1.0, 2.0) == 0.0


@pytest.mark.parametrize("b", [0.5, 1.0, 2.0, 5.0])
def test_cdf_beta_special_case_a_equals_one(b):
    # Kumaraswamy(1, b) == Beta(1, b): both CDFs are 1 - (1-x)^b.
    for x in (0.1, 0.3, 0.5, 0.8):
        assert kumaraswamy_cdf(x, 1.0, b) == pytest.approx(1.0 - (1.0 - x) ** b)


def test_cdf_boundaries():
    assert kumaraswamy_cdf(0.0, 2.0, 3.0) == 0.0
    assert kumaraswamy_cdf(1.0, 2.0, 3.0) == 1.0


# ===========================================================================
# 2. Posterior + update
# ===========================================================================


def test_posterior_update_alpha_beta_exact():
    backend = InMemoryPosteriorBackend()
    backend.update("b", "m", 1.0)
    backend.update("b", "m", 0.0)
    post = backend.get("b", "m")
    # start (1,1); +1 success (alpha+=1), +1 failure (beta+=1) => (2, 2)
    assert post.alpha == pytest.approx(2.0)
    assert post.beta == pytest.approx(2.0)


def test_posterior_mean_converges_to_empirical():
    strat = KumaraswamyThompsonStrategy(w_quality=1.0, w_cost=0.0, w_latency=0.0)
    # 80 successes (score=1 -> r_quality=1), 20 failures (score=-1 -> r_quality=0).
    for _ in range(80):
        strat.update("m", 1.0, bucket="b")
    for _ in range(20):
        strat.update("m", -1.0, bucket="b")
    post = strat._backend.get("b", "m")
    # Laplace-smoothed: (1+80)/(1+80 + 1+20) = 81/102.
    assert post.mean() == pytest.approx(81.0 / 102.0, abs=1e-9)


def test_fractional_reward_score_zero_maps_to_half():
    # score=0.0 in [-1,1] => r_quality=0.5; quality-only weights => alpha,beta both +0.5.
    strat = KumaraswamyThompsonStrategy(w_quality=1.0, w_cost=0.0, w_latency=0.0)
    strat.update("m", 0.0, bucket="b")
    post = strat._backend.get("b", "m")
    assert post.alpha == pytest.approx(1.5)
    assert post.beta == pytest.approx(1.5)


def test_reward_clamped_to_unit_interval():
    backend = InMemoryPosteriorBackend()
    backend.update("b", "m", 5.0)  # over-large reward clamps to 1.0
    post = backend.get("b", "m")
    assert post.alpha == pytest.approx(2.0)
    assert post.beta == pytest.approx(1.0)


def test_reward_shaping_cost_term_blends():
    # quality 1.0 with norm_cost 0.0 (cheapest) and equal weights -> reward 1.0.
    strat = KumaraswamyThompsonStrategy(w_quality=0.5, w_cost=0.5, w_latency=0.0)
    strat.update("m", 1.0, bucket="b", norm_cost=0.0)
    post = strat._backend.get("b", "m")
    assert post.alpha == pytest.approx(2.0, abs=1e-9)


# ===========================================================================
# 3. Decay toward prior
# ===========================================================================


def test_decay_keeps_mean_reinflates_variance():
    # Decay multiplies the *deviation from the prior* by gamma^days. With a
    # MILD decay (gamma close to 1, few days) the mean is approximately
    # preserved while the total strength shrinks toward the prior — i.e. the
    # variance re-inflates, re-opening exploration for a stale arm. (A near-
    # total decay toward a balanced Beta(1,1) prior would instead pull the mean
    # back to 0.5; that is the prior's mean, not a bug.)
    strat = KumaraswamyThompsonStrategy(
        decay_gamma=0.99, cold_start_kappa=0.0, w_quality=1.0, w_cost=0.0, w_latency=0.0
    )
    # Build a concentrated posterior at mean ~0.8 from prior (1,1).
    for _ in range(40):
        strat.update("m", 1.0, bucket="b")  # success
    for _ in range(10):
        strat.update("m", -1.0, bucket="b")  # failure
    post = strat._backend.get("b", "m")
    mean_before = post.mean()
    strength_before = post.strength()
    strat.maybe_decay("b", "m", days_elapsed=5.0)
    post_after = strat._backend.get("b", "m")
    # Mean approximately preserved under mild decay.
    assert post_after.mean() == pytest.approx(mean_before, abs=0.06)
    # Concentration shrinks toward the prior strength (variance re-inflates).
    assert post_after.strength() < strength_before


def test_decay_noop_when_gamma_one():
    backend = InMemoryPosteriorBackend()
    backend.update("b", "m", 1.0)
    a0 = backend.get("b", "m").alpha
    backend.decay("b", "m", gamma=1.0, days=100.0, prior=(1.0, 1.0))
    assert backend.get("b", "m").alpha == a0


# ===========================================================================
# 4. Dynamic arms
# ===========================================================================


def test_select_over_multiple_candidates_returns_one():
    router = _FakeRouter(
        [_dep("g", "bedrock/a"), _dep("g", "bedrock/b"), _dep("g", "bedrock/c")]
    )
    strat = KumaraswamyThompsonStrategy(seed=7)
    chosen = strat.select_deployment(_ctx(router, model="g"))
    assert chosen in router.healthy_deployments


def test_circuit_breaker_drops_arm(monkeypatch):
    router = _FakeRouter(
        [
            _dep("g", "bedrock/a", provider="prov_a"),
            _dep("g", "bedrock/b", provider="prov_b"),
        ]
    )

    class _Breaker:
        def __init__(self, is_open):
            self.is_open = is_open

    class _CBManager:
        def get_breaker(self, provider):
            return _Breaker(is_open=(provider == "prov_a"))

    import litellm_llmrouter.resilience as resilience

    monkeypatch.setattr(resilience, "get_circuit_breaker_manager", lambda: _CBManager())
    strat = KumaraswamyThompsonStrategy(seed=1)
    # prov_a is open -> only prov_b survives -> always chosen.
    for _ in range(20):
        chosen = strat.select_deployment(_ctx(router, model="g"))
        assert chosen["litellm_params"]["model"] == "bedrock/b"


def test_vanished_arm_posterior_preserved():
    strat = KumaraswamyThompsonStrategy(seed=3)
    # Update arm X.
    strat.update("bedrock/x", 1.0, bucket="g")
    mean_before = strat._backend.get("g", "bedrock/x").mean()
    # Route over candidates without X (X vanished from the live set).
    router = _FakeRouter([_dep("g", "bedrock/y")])
    chosen = strat.select_deployment(_ctx(router, model="g"))
    assert chosen["litellm_params"]["model"] == "bedrock/y"
    # X's posterior persists, mean unchanged.
    assert strat._backend.get("g", "bedrock/x").mean() == pytest.approx(mean_before)


# ===========================================================================
# 5. Cold start
# ===========================================================================


def test_cold_start_warm_start_prior_from_quality_table(monkeypatch):
    import litellm_llmrouter.personalized_routing as pr

    monkeypatch.setattr(pr, "_DEFAULT_QUALITY_BIASES", {"gpt-5": 0.9}, raising=False)
    strat = KumaraswamyThompsonStrategy(cold_start_kappa=5.0)
    a0, b0 = strat._cold_start_prior("gpt-5")
    assert a0 == pytest.approx(1.0 + 5.0 * 0.9)
    assert b0 == pytest.approx(1.0 + 5.0 * 0.1)


def test_cold_start_kappa_zero_reduces_to_uniform():
    strat = KumaraswamyThompsonStrategy(cold_start_kappa=0.0)
    a0, b0 = strat._cold_start_prior("unknown-model")
    assert (a0, b0) == (1.0, 1.0)


def test_new_pair_lazily_created_with_prior(monkeypatch):
    import litellm_llmrouter.personalized_routing as pr

    monkeypatch.setattr(
        pr, "_DEFAULT_QUALITY_BIASES", {"bedrock/q": 0.6}, raising=False
    )
    strat = KumaraswamyThompsonStrategy(cold_start_kappa=5.0)
    post = strat._get_posterior("g", "bedrock/q")
    assert post.alpha == pytest.approx(1.0 + 5.0 * 0.6)


# ===========================================================================
# 6. Arm key
# ===========================================================================


def test_arm_key_is_litellm_params_model():
    router = _FakeRouter([_dep("g", "bedrock/global.anthropic.claude-haiku-4-5")])
    strat = KumaraswamyThompsonStrategy(seed=1)
    strat.select_deployment(_ctx(router, model="g", request_id="r"))
    strat.update("bedrock/global.anthropic.claude-haiku-4-5", 1.0, request_id="r")
    # The posterior is keyed on the arm string (litellm_params.model).
    snap = strat._backend.snapshot()
    models = {row["model"] for row in snap["posteriors"]}
    assert "bedrock/global.anthropic.claude-haiku-4-5" in models


# ===========================================================================
# 7. Registration through the existing pipeline (no LiteLLM-mount edit)
# ===========================================================================


def test_registration_via_settings(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_KUMARASWAMY_THOMPSON__ENABLED", "true")
    from litellm_llmrouter.settings import reset_settings

    reset_settings()
    assert register_kumaraswamy_thompson_strategy() is True
    registry = get_routing_registry()
    strat = registry.get(STRATEGY_NAME)
    assert isinstance(strat, KumaraswamyThompsonStrategy)


def test_registration_disabled_when_setting_off(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_KUMARASWAMY_THOMPSON__ENABLED", "false")
    from litellm_llmrouter.settings import reset_settings

    reset_settings()
    assert register_kumaraswamy_thompson_strategy() is False
    assert get_routing_registry().get(STRATEGY_NAME) is None


def test_dispatches_through_pipeline_after_set_active(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_KUMARASWAMY_THOMPSON__ENABLED", "true")
    from litellm_llmrouter.settings import reset_settings

    reset_settings()
    register_kumaraswamy_thompson_strategy()
    registry = get_routing_registry()
    registry.set_active(STRATEGY_NAME)

    # NOTE: we construct RoutingPipeline directly from the registry rather than
    # calling get_routing_pipeline(), which self-deadlocks on a cold process
    # (the non-reentrant _instance_lock is re-acquired via get_routing_registry).
    # That deadlock is an out-of-P3-scope core-registry defect (recorded as an
    # incidental finding); building the pipeline from the registry still proves
    # the bandit rides the existing dispatch with NO LiteLLM-mount edit.
    from litellm_llmrouter.strategy_registry import RoutingPipeline

    pipeline = RoutingPipeline(registry)
    router = _FakeRouter([_dep("g", "bedrock/a"), _dep("g", "bedrock/b")])
    ctx = _ctx(router, model="g", request_id="rp")
    result = pipeline.route(ctx)
    assert result.deployment is not None
    assert result.deployment["litellm_params"]["model"] in {"bedrock/a", "bedrock/b"}
    assert result.strategy_name == STRATEGY_NAME
    assert result.is_fallback is False


# ===========================================================================
# 8. In-memory default (no external deps)
# ===========================================================================


def test_in_memory_default_routes_with_no_external_deps():
    # Constructs and routes with no redis/aurora/env -> proves tests need no deps.
    strat = KumaraswamyThompsonStrategy(seed=99)
    assert isinstance(strat._backend, InMemoryPosteriorBackend)
    router = _FakeRouter([_dep("g", "bedrock/a"), _dep("g", "bedrock/b")])
    assert strat.select_deployment(_ctx(router, model="g")) is not None


def test_single_candidate_short_circuits():
    strat = KumaraswamyThompsonStrategy(seed=1)
    router = _FakeRouter([_dep("g", "bedrock/only")])
    chosen = strat.select_deployment(_ctx(router, model="g"))
    assert chosen["litellm_params"]["model"] == "bedrock/only"


def test_no_candidates_returns_none():
    strat = KumaraswamyThompsonStrategy(seed=1)
    router = _FakeRouter([])
    assert strat.select_deployment(_ctx(router, model="g")) is None


def test_validate_ok_with_backend():
    strat = KumaraswamyThompsonStrategy()
    ok, err = strat.validate()
    assert ok is True
    assert err is None


def test_identity_name_version():
    strat = KumaraswamyThompsonStrategy()
    assert strat.name == STRATEGY_NAME
    assert strat.version == STRATEGY_VERSION
    assert STRATEGY_VERSION == "v1"


# ===========================================================================
# 9. Posterior dataclass + snapshot/hydrate round-trip
# ===========================================================================


def test_posterior_mean_and_shape():
    p = Posterior(alpha=3.0, beta=1.0)
    assert p.mean() == pytest.approx(0.75)
    assert p.shape() == (3.0, 1.0)
    assert p.strength() == pytest.approx(4.0)


def test_snapshot_hydrate_round_trip():
    b1 = InMemoryPosteriorBackend()
    b1.update("g", "bedrock/a", 1.0)
    b1.update("g", "bedrock/b", 0.0)
    snap = b1.snapshot()
    b2 = InMemoryPosteriorBackend()
    b2.hydrate(snap)
    assert b2.get("g", "bedrock/a").alpha == pytest.approx(2.0)
    assert b2.get("g", "bedrock/b").beta == pytest.approx(2.0)


def test_export_load_artifact_round_trip(tmp_path):
    import json

    from litellm_llmrouter.adapters.contract import ArtifactRef

    s1 = KumaraswamyThompsonStrategy()
    s1.update("bedrock/a", 1.0, bucket="g")
    artifact = s1.export_artifact()
    path = tmp_path / "posteriors.json"
    path.write_text(json.dumps(artifact))

    s2 = KumaraswamyThompsonStrategy()
    ref = ArtifactRef(path=str(path), payload=artifact)
    assert s2.load_artifact(ref) is True
    assert s2._backend.get("g", "bedrock/a").alpha == pytest.approx(2.0)
