"""Property-based tests for the Kumaraswamy-Thompson bandit.

Hypothesis (max_examples=100 from pyproject). Validates the universal properties
of the closed-form quantile sampler (range, monotonicity, round-trip in the
well-conditioned band), the conjugate posterior's monotone response to evidence,
and seeded selection determinism. No external deps — in-memory backend only.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from litellm_llmrouter.kumaraswamy_thompson import (
    KumaraswamyThompsonStrategy,
    kumaraswamy_cdf,
    kumaraswamy_quantile,
)
from litellm_llmrouter.strategy_registry import RoutingContext


# Shape-parameter and uniform-draw strategies.
_a = st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
_b = st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
_u = st.floats(min_value=1e-6, max_value=1.0 - 1e-6, allow_nan=False)

# Well-conditioned band for the round-trip property (architect-verified).
_a_wc = st.floats(min_value=0.3, max_value=30.0, allow_nan=False)
_b_wc = st.floats(min_value=0.3, max_value=30.0, allow_nan=False)
_u_wc = st.floats(min_value=0.01, max_value=0.99, allow_nan=False)


@given(u=_u, a=_a, b=_b)
@settings(max_examples=100)
def test_quantile_in_open_unit_interval(u, a, b):
    x = kumaraswamy_quantile(u, a, b)
    assert 0.0 <= x <= 1.0
    assert not math.isnan(x)


@given(a=_a, b=_b, u1=_u, u2=_u)
@settings(max_examples=100)
def test_quantile_monotone_in_u(a, b, u1, u2):
    lo, hi = sorted((u1, u2))
    x_lo = kumaraswamy_quantile(lo, a, b)
    x_hi = kumaraswamy_quantile(hi, a, b)
    # Monotone non-decreasing (allow a tiny fp tolerance at near-equal u).
    assert x_lo <= x_hi + 1e-12


@given(a=_a_wc, b=_b_wc, u=_u_wc)
@settings(max_examples=100)
def test_round_trip_well_conditioned(a, b, u):
    x = kumaraswamy_quantile(u, a, b)
    back = kumaraswamy_cdf(x, a, b)
    assert abs(back - u) < 1e-6


@given(
    n_success=st.integers(min_value=0, max_value=200),
    n_fail=st.integers(min_value=0, max_value=200),
)
@settings(max_examples=100)
def test_posterior_mean_in_unit_interval(n_success, n_fail):
    strat = KumaraswamyThompsonStrategy(w_quality=1.0, w_cost=0.0, w_latency=0.0)
    for _ in range(n_success):
        strat.update("m", 1.0, bucket="b")  # success -> r_quality=1
    for _ in range(n_fail):
        strat.update("m", -1.0, bucket="b")  # failure -> r_quality=0
    mean = strat._backend.get("b", "m").mean()
    assert 0.0 <= mean <= 1.0


@given(
    extra_success=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_more_successes_raise_mean(extra_success):
    # An arm with more 1-rewards must have a higher posterior mean than an arm
    # with the same total observations but fewer 1-rewards (monotone response).
    strat = KumaraswamyThompsonStrategy(w_quality=1.0, w_cost=0.0, w_latency=0.0)
    base = 50
    for _ in range(base):
        strat.update("high", 1.0, bucket="b")
        strat.update("low", 1.0, bucket="b")
    # 'high' gets extra successes; 'low' gets the same count of failures.
    for _ in range(extra_success):
        strat.update("high", 1.0, bucket="b")
        strat.update("low", -1.0, bucket="b")
    high_mean = strat._backend.get("b", "high").mean()
    low_mean = strat._backend.get("b", "low").mean()
    assert high_mean > low_mean


@given(a=_a_wc, b=_b_wc)
@settings(max_examples=50, deadline=None)
def test_sampled_mean_near_analytic_mean(a, b):
    # Goodness-of-fit: the Monte-Carlo mean of Kumaraswamy(a,b) approximates the
    # analytic mean E[X] = b*B(1+1/a, b) within a band sized for the MC error.
    import random as _random

    rng = _random.Random(0xC0FFEE)
    n = 8000
    total = 0.0
    for _ in range(n):
        total += kumaraswamy_quantile(rng.random(), a, b)
    sampled = total / n
    analytic = (
        b * math.gamma(1.0 + 1.0 / a) * math.gamma(b) / math.gamma(1.0 + 1.0 / a + b)
    )
    # MC std-error scales ~ 0.3/sqrt(n); 5e-2 is a robust band across the
    # well-conditioned shape grid at the fixed seed.
    assert abs(sampled - analytic) < 5e-2


@given(a=_a, b=_b)
@settings(max_examples=100)
def test_quantile_boundaries_bracket_unit_interval(a, b):
    # Q(0) -> ~0 (strictly positive via the interior u-clamp), Q(1) -> ~1 (may
    # round to exactly 1.0 in float64 for a=1, b<=1 — the correct limit), and
    # the endpoints bracket the median: Q(0) <= Q(0.5) <= Q(1).
    q0 = kumaraswamy_quantile(0.0, a, b)
    qm = kumaraswamy_quantile(0.5, a, b)
    q1 = kumaraswamy_quantile(1.0, a, b)
    assert 0.0 < q0 < 1.0
    assert 0.0 < q1 <= 1.0
    assert q0 <= qm <= q1


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50)
def test_seeded_selection_deterministic(seed):
    # Same seed + same candidates => identical selection sequence.
    deps = [
        {"model_name": "g", "litellm_params": {"model": f"bedrock/arm{i}"}}
        for i in range(4)
    ]

    class _Router:
        healthy_deployments = deps
        model_list = deps

    def _make_ctx(rid):
        return RoutingContext(
            router=_Router(),
            model="g",
            messages=[{"role": "user", "content": "x"}],
            request_id=rid,
        )

    s1 = KumaraswamyThompsonStrategy(seed=seed)
    s2 = KumaraswamyThompsonStrategy(seed=seed)
    picks1 = [
        s1.select_deployment(_make_ctx(f"r{i}"))["litellm_params"]["model"]
        for i in range(10)
    ]
    picks2 = [
        s2.select_deployment(_make_ctx(f"r{i}"))["litellm_params"]["model"]
        for i in range(10)
    ]
    assert picks1 == picks2
