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
    fit_kumaraswamy_moments,
    kumaraswamy_cdf,
    kumaraswamy_mean_var,
    kumaraswamy_quantile,
    register_kumaraswamy_thompson_strategy,
    sample_kumaraswamy,
    strength_bucket,
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


# ===========================================================================
# 10. RouteIQ-f9e9 — Kumaraswamy Newton moment-fit (doc-20 §3.1 option-2)
# ===========================================================================
#
# The option-1 ``a=alpha, b=beta`` shortcut distorts the posterior mean
# (Beta(51,51) mean 0.5 -> Kumaraswamy 0.9155) and can INVERT the exploit
# decision. The cached 5-iteration Newton moment-fit maps Beta(alpha,beta) ->
# Kumaraswamy(a,b) matching mean + variance, restoring the correct mean while
# keeping the hot path at ~1 uniform draw + the quantile.


@pytest.mark.parametrize(
    "alpha,beta",
    [
        (1, 1),
        (51, 51),
        (40, 40),
        (6, 4),
        (2, 5),
        (10, 3),
        (80, 20),
        (20, 80),
        (0.5, 0.5),
        (3, 3),
        (100, 2),
        (2, 100),
        (7, 7),
    ],
)
def test_moment_fit_mean_matches_beta(alpha, beta):
    # The fitted Kumaraswamy mean approximates the Beta mean across [0,1].
    # Worst real-regime error is ~1e-12; 1e-2 is a comfortable unit tolerance.
    a, b = fit_kumaraswamy_moments(float(alpha), float(beta))
    fit_mean, _ = kumaraswamy_mean_var(a, b)
    beta_mean = alpha / (alpha + beta)
    assert abs(fit_mean - beta_mean) < 1e-2
    assert a > 0.0 and b > 0.0  # always valid Kumaraswamy params


def test_moment_fit_special_cases_exact():
    # The three exact Beta==Kumaraswamy corners short-circuit to (a,b) exactly
    # (byte-stable, no iteration): Kuma(1,b)=Beta(1,b), Kuma(a,1)=Beta(a,1),
    # Kuma(1,1)=Beta(1,1)=Uniform.
    assert fit_kumaraswamy_moments(1.0, 5.0) == (1.0, 5.0)
    assert fit_kumaraswamy_moments(5.0, 1.0) == (5.0, 1.0)
    assert fit_kumaraswamy_moments(1.0, 1.0) == (1.0, 1.0)


def test_moment_fit_beats_shortcut_on_exploit_inversion():
    """Shortcut ranks Beta(40,40) [true 0.5] above Beta(6,4) [true 0.6] (WRONG);
    the moment-fit restores the correct ranking and Thompson over the fitted
    shapes picks the genuinely-better arm the majority of the time."""
    import random

    A1, A2 = (40.0, 40.0), (6.0, 4.0)  # arm2 genuinely better (0.6 > 0.5)

    # Shortcut MEANS invert the ranking (the bug):
    sm1, _ = kumaraswamy_mean_var(*A1)
    sm2, _ = kumaraswamy_mean_var(*A2)
    assert sm1 > sm2  # shortcut WRONGLY prefers the over-concentrated arm1

    # Moment-fit MEANS restore the correct ranking:
    a1, b1 = fit_kumaraswamy_moments(*A1)
    a2, b2 = fit_kumaraswamy_moments(*A2)
    fm1, _ = kumaraswamy_mean_var(a1, b1)
    fm2, _ = kumaraswamy_mean_var(a2, b2)
    assert fm1 == pytest.approx(0.5, abs=1e-2)
    assert fm2 == pytest.approx(0.6, abs=1e-2)
    assert fm2 > fm1  # CORRECT ranking restored

    # Thompson over the fitted shapes picks the better arm the majority:
    rng = random.Random(123)
    fit_wins = sum(
        kumaraswamy_quantile(rng.random(), a2, b2)
        > kumaraswamy_quantile(rng.random(), a1, b1)
        for _ in range(20000)
    )
    # Measured ~0.74 for the moment-fit vs ~0.06 for the shortcut.
    assert fit_wins / 20000 > 0.5

    # And the shortcut shapes pick the WORSE arm the vast majority (regression
    # guard: proves the moment-fit is what flips the decision, not the seed).
    rng = random.Random(123)
    shortcut_wins = sum(
        kumaraswamy_quantile(rng.random(), *A2)
        > kumaraswamy_quantile(rng.random(), *A1)
        for _ in range(20000)
    )
    assert shortcut_wins / 20000 < 0.2


def test_default_path_argmax_picks_better_arm_near_half():
    """RouteIQ-1817: assert the ACTUAL shipped draw/argmax behavior in the ~0.5
    inversion zone.

    The other strong tests assert on ``alpha/(alpha+beta)`` (the Beta MEAN), but
    ``select_deployment`` routes on ``sample_kumaraswamy(a, b)`` over the SHIPPED
    DEFAULT mapping (``Posterior.shape(moment_fit=False) -> (alpha, beta)``,
    i.e. ``Kumaraswamy(alpha, beta)``) — not the mean. This pins that the actual
    sampled argmax favors the genuinely-better arm when the two Beta means
    STRADDLE 0.5 (the zone the 0.79-0.95 backtest never exercises).

    Arms: worse ``Beta(2, 3)`` (mean 0.4 < 0.5) vs better ``Beta(3, 2)``
    (mean 0.6 > 0.5). The loop mirrors ``select_deployment``'s argmax (draw per
    arm, keep the max) over many SEEDED draws.
    """
    import random

    worse = Posterior(alpha=2.0, beta=3.0)
    better = Posterior(alpha=3.0, beta=2.0)
    # The two Beta means genuinely straddle 0.5 (one arm is genuinely better).
    assert worse.mean() < 0.5 < better.mean()

    # Draw via the SHIPPED default mapping the hot path uses: moment_fit=False
    # -> shape() -> (alpha, beta) -> sample_kumaraswamy(alpha, beta, rng).
    aw, bw = worse.shape(moment_fit=False)
    ab, bb = better.shape(moment_fit=False)
    assert (aw, bw) == (2.0, 3.0)
    assert (ab, bb) == (3.0, 2.0)

    rng = random.Random(20240117)
    n = 20000
    better_wins = 0
    for _ in range(n):
        # Same argmax shape as select_deployment: one draw per arm, keep the max.
        x_worse = sample_kumaraswamy(aw, bw, rng)
        x_better = sample_kumaraswamy(ab, bb, rng)
        if x_better > x_worse:
            better_wins += 1
    # Measured ~0.747 on this seed; the better arm wins the argmax majority.
    assert better_wins / n > 0.5


def test_moment_fit_cache_hit_when_counts_unchanged():
    # RouteIQ-f9e9 defect-1 fix: the fit is cached on the EXACT (alpha, beta),
    # so a repeat call with unchanged counts is a cache hit (identical result),
    # but ANY evidence change recomputes (no stale within-bucket fit).
    p = Posterior(alpha=4.0, beta=4.0)
    a1, b1 = p.shape(moment_fit=True)
    a1b, b1b = p.shape(moment_fit=True)  # counts unchanged -> cached, identical
    assert (a1, b1) == (a1b, b1b)


def test_moment_fit_cache_recomputes_on_within_bucket_update():
    # RouteIQ-f9e9 defect-1: the OLD bucket key served a STALE fit for evidence
    # changes WITHIN a strength bucket -> a ~0.24-wrong sampled mean. Beta(8,8)
    # and Beta(23,8) are BOTH in bucket 4 (floor(log2(s)) for s=16 and s=31), so
    # the bucket key would NOT refresh. The exact-(alpha,beta) key must.
    assert strength_bucket(16.0) == strength_bucket(31.0) == 4  # same bucket
    p = Posterior(alpha=8.0, beta=8.0)
    a1, b1 = p.shape(moment_fit=True)
    m1, _ = kumaraswamy_mean_var(a1, b1)
    assert m1 == pytest.approx(0.5, abs=1e-3)  # Beta(8,8) mean

    p.alpha = 23.0  # within-bucket evidence change -> MUST refit
    a2, b2 = p.shape(moment_fit=True)
    m2, _ = kumaraswamy_mean_var(a2, b2)
    assert (a2, b2) != (a1, b1)  # not the stale cached fit
    assert m2 == pytest.approx(23.0 / 31.0, abs=1e-3)  # tracks the new mean


def test_moment_fit_deterministic():
    # Pure deterministic function of (alpha, beta): no RNG, no global state.
    assert fit_kumaraswamy_moments(40.0, 40.0) == fit_kumaraswamy_moments(40.0, 40.0)
    assert fit_kumaraswamy_moments(6.0, 4.0) == fit_kumaraswamy_moments(6.0, 4.0)


@pytest.mark.parametrize(
    "alpha,beta",
    [
        (185.25, 190.0),  # the hypothesis-found infeasible-variance regression
        (190.0, 185.25),
        (200.0, 200.0),
        (150.0, 160.0),
        (1000.0, 1000.0),
        (199.0, 200.0),
    ],
)
def test_moment_fit_mean_holds_on_high_evidence_near_symmetric(alpha, beta):
    # High-evidence near-symmetric posteriors can sit below Kumaraswamy's minimum
    # achievable variance (the floor). The 1-D fit (RouteIQ-f9e9) holds the mean
    # EXACTLY at every step (b is solved for the mean), so even an infeasible
    # target variance yields the right mean -- the doc's degradation contract
    # (right mean, variance slightly above the floor = more exploration, never
    # the shortcut's wrong mean, never under-exploration).
    a, b = fit_kumaraswamy_moments(alpha, beta)
    fit_mean, _ = kumaraswamy_mean_var(a, b)
    assert abs(fit_mean - alpha / (alpha + beta)) < 1e-2


def test_moment_fit_extreme_params_stay_valid():
    # Numerical guards: no overflow / nan, always valid (a>0, b>0) at extremes.
    import math

    for alpha, beta in [
        (0.3, 0.3),
        (0.3, 200.0),
        (200.0, 0.3),
        (1000.0, 1000.0),
        (2000.0, 1.5),
        (0.5, 500.0),
    ]:
        a, b = fit_kumaraswamy_moments(alpha, beta)
        assert a > 0.0 and b > 0.0
        assert math.isfinite(a) and math.isfinite(b)
        q = kumaraswamy_quantile(0.5, a, b)
        assert 0.0 < q < 1.0 and math.isfinite(q)


def test_shortcut_default_unchanged():
    # Default shape() path is byte-stable (no-flag == old option-1 behavior).
    p = Posterior(alpha=40.0, beta=40.0)
    assert p.shape() == (40.0, 40.0)  # option-1 shortcut, unchanged
    assert p.shape(moment_fit=False) == (40.0, 40.0)


def test_moment_fit_flag_off_by_default_in_strategy():
    # The strategy defaults to the shortcut; flag is opt-in.
    strat = KumaraswamyThompsonStrategy()
    assert strat._moment_fit is False
    strat_on = KumaraswamyThompsonStrategy(moment_fit=True)
    assert strat_on._moment_fit is True


def test_moment_fit_selection_seeded_deterministic():
    # Seeded determinism preserved on the moment-fit scoring path.
    router = _FakeRouter([_dep("g", "bedrock/a"), _dep("g", "bedrock/b")])
    s1 = KumaraswamyThompsonStrategy(seed=7, moment_fit=True)
    s2 = KumaraswamyThompsonStrategy(seed=7, moment_fit=True)
    picks1 = [
        s1.select_deployment(_ctx(router, model="g", request_id=f"r{i}"))[
            "litellm_params"
        ]["model"]
        for i in range(10)
    ]
    picks2 = [
        s2.select_deployment(_ctx(router, model="g", request_id=f"r{i}"))[
            "litellm_params"
        ]["model"]
        for i in range(10)
    ]
    assert picks1 == picks2


# ===========================================================================
# 11. RouteIQ-99e8 + RouteIQ-badb — pre-scoring filter THROUGH the bandit seam
# ===========================================================================
#
# These assert the filter runs BEFORE the bandit's scoring loop (not retried
# after a failed selection): a cooled-down / gov-banned arm is excluded from the
# scored candidate set, so it is never returned across many seeded draws.

from litellm_llmrouter.settings import get_settings as _get_settings  # noqa: E402
from litellm_llmrouter.settings import reset_settings as _reset_settings  # noqa: E402

_FABLE5 = "bedrock/global.anthropic.claude-fable-5"


def _dep_id(model_name: str, arm: str, dep_id: str) -> dict:
    """Deployment dict carrying model_info.id (the cooldown match key)."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": arm},
        "model_info": {"id": dep_id},
    }


def test_bandit_excludes_cooled_down_arm(monkeypatch):
    """RouteIQ-99e8 through the bandit: a cooled-down arm is EXCLUDED from the
    scored candidate set -- never returned across many seeded draws."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: {"d2"},  # d2 is cooled down
    )
    router = _FakeRouter(
        [_dep_id("g", "bedrock/a", "d1"), _dep_id("g", "bedrock/b", "d2")]
    )
    strat = KumaraswamyThompsonStrategy(seed=99)
    chosen_arms = set()
    for i in range(200):
        chosen = strat.select_deployment(_ctx(router, model="g", request_id=f"r{i}"))
        assert chosen is not None
        chosen_arms.add(chosen["litellm_params"]["model"])
    assert "bedrock/b" not in chosen_arms  # cooled-down arm never scored/selected
    assert chosen_arms == {"bedrock/a"}


def test_bandit_cooldown_fail_open_when_all_cooled(monkeypatch):
    """If every arm is cooled down, the bandit still routes (fail-open)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: {"d1", "d2"},
    )
    router = _FakeRouter(
        [_dep_id("g", "bedrock/a", "d1"), _dep_id("g", "bedrock/b", "d2")]
    )
    strat = KumaraswamyThompsonStrategy(seed=5)
    assert strat.select_deployment(_ctx(router, model="g")) is not None


def test_bandit_never_selects_gov_banned_arm():
    """RouteIQ-badb through the bandit: a gov-banned arm (Fable 5) is REMOVED
    before scoring and never returned across many seeded draws."""
    _reset_settings()
    _get_settings(governance={"banned_models": [_FABLE5]})
    router = _FakeRouter(
        [
            _dep_id("g", "bedrock/anthropic.claude-3-sonnet", "d1"),
            _dep_id("g", _FABLE5, "d2"),
        ]
    )
    strat = KumaraswamyThompsonStrategy(seed=123)
    chosen_arms = set()
    for i in range(200):
        chosen = strat.select_deployment(_ctx(router, model="g", request_id=f"r{i}"))
        assert chosen is not None
        chosen_arms.add(chosen["litellm_params"]["model"])
    assert _FABLE5 not in chosen_arms
    assert chosen_arms == {"bedrock/anthropic.claude-3-sonnet"}


def test_bandit_gov_banned_sole_arm_returns_none():
    """A gov-banned arm that is the SOLE candidate -> no selection (compliance
    fail-closed-to-removal; never route to a banned model)."""
    _reset_settings()
    _get_settings(governance={"banned_models": [_FABLE5]})
    router = _FakeRouter([_dep_id("g", _FABLE5, "d1")])
    strat = KumaraswamyThompsonStrategy(seed=1)
    assert strat.select_deployment(_ctx(router, model="g")) is None


def test_bandit_default_no_ban_byte_stable():
    """Default-empty ban config -> both arms remain selectable (byte-stable)."""
    _reset_settings()
    _get_settings()  # no banned_models
    router = _FakeRouter(
        [_dep_id("g", "bedrock/a", "d1"), _dep_id("g", "bedrock/b", "d2")]
    )
    strat = KumaraswamyThompsonStrategy(seed=42)
    chosen_arms = set()
    for i in range(200):
        chosen = strat.select_deployment(_ctx(router, model="g", request_id=f"r{i}"))
        assert chosen is not None
        chosen_arms.add(chosen["litellm_params"]["model"])
    assert chosen_arms == {"bedrock/a", "bedrock/b"}
