"""Unit tests for per-request region-aware / data-residency routing (RouteIQ-60cc).

The region filter is a PRE-SCORING candidate filter on the same
``filter_routable_candidates`` seam every RouteIQ strategy already calls, so it
composes with gov-ban + cooldown and benefits EVERY strategy (no new strategy
subclass). Acceptance:

  - A residency-CONSTRAINED request STAYS in-region (never routed out-of-region;
    if no in-region arm exists the candidate set is emptied -> None / fallback).
  - An UNCONSTRAINED request PREFERS in-region but MAY fall back to the full set
    when no in-region arm exists (availability over locality).
  - Default OFF / identity: a byte-stable no-op when disabled OR no region token.

All cred-free: settings are injected via ``get_settings(region_routing=...)``
after ``reset_settings()`` (the conftest autouse fixture resets between tests);
the RoutingContext is built directly (no Router needed for the filter itself).
"""

from __future__ import annotations

from typing import Any, Dict, List

from litellm_llmrouter.candidate_filter import (
    drop_out_of_region,
    filter_routable_candidates,
    resolve_request_region,
)
from litellm_llmrouter.settings import get_settings, reset_settings
from litellm_llmrouter.strategy_registry import RoutingContext


def _dep(model: str, region: str, dep_id: str, model_name: str = "claude") -> Dict:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "aws_region_name": region},
        "model_info": {"id": dep_id},
    }


def _ctx(
    *,
    request_kwargs: Dict | None = None,
    metadata: Dict | None = None,
) -> RoutingContext:
    return RoutingContext(
        router=None,
        model="claude",
        request_kwargs=request_kwargs,
        metadata=metadata or {},
    )


def _configure(**region_routing: Any) -> None:
    """Reset + create the settings singleton with a region_routing override."""
    reset_settings()
    get_settings(region_routing=region_routing)


# ---------------------------------------------------------------------------
# Default OFF / identity (byte-stable)
# ---------------------------------------------------------------------------


def test_disabled_is_identity_no_op():
    """enabled=False -> the input list OBJECT is returned unchanged."""
    _configure(enabled=False)
    cands: List[Dict] = [_dep("m", "us-east-1", "d1"), _dep("m", "eu-west-1", "d2")]
    ctx = _ctx(metadata={"region": "eu"})
    out = drop_out_of_region(ctx, cands)
    assert out is cands  # same object -> byte-stable


def test_enabled_but_no_region_token_is_no_op():
    """enabled=True but no region token on the request -> no-op (same object)."""
    _configure(enabled=True)
    cands = [_dep("m", "us-east-1", "d1"), _dep("m", "eu-west-1", "d2")]
    ctx = _ctx(metadata={})  # no region / header
    out = drop_out_of_region(ctx, cands)
    assert out is cands


def test_none_context_is_no_op():
    """A None context (legacy call shape) -> no-op even when enabled."""
    _configure(enabled=True)
    cands = [_dep("m", "us-east-1", "d1")]
    out = drop_out_of_region(None, cands)
    assert out is cands


# ---------------------------------------------------------------------------
# Hard residency constraint (RouteIQ-60cc acceptance #1)
# ---------------------------------------------------------------------------


def test_hard_residency_stays_in_region():
    """A residency-flagged request keeps ONLY in-region candidates."""
    _configure(enabled=True, region_map={"eu": ["eu-west-1", "eu-central-1"]})
    us = _dep("m-us", "us-east-1", "d1")
    eu1 = _dep("m-eu1", "eu-west-1", "d2")
    eu2 = _dep("m-eu2", "eu-central-1", "d3")
    ctx = _ctx(metadata={"region": "eu", "residency": True})
    out = drop_out_of_region(ctx, [us, eu1, eu2])
    assert out == [eu1, eu2]
    assert us not in out


def test_hard_residency_no_in_region_returns_empty():
    """Acceptance: hard residency + no in-region arm -> EMPTY (never out-of-region).

    The strategy then returns None / triggers a fallback; a residency violation
    must never silently leak by routing to the only (out-of-region) arm.
    """
    _configure(enabled=True, region_map={"eu": ["eu-west-1"]})
    us1 = _dep("m-us1", "us-east-1", "d1")
    us2 = _dep("m-us2", "us-west-2", "d2")
    ctx = _ctx(metadata={"region": "eu", "data_residency": "true"})
    out = drop_out_of_region(ctx, [us1, us2])
    assert out == []


def test_hard_residency_default_makes_plain_region_hard():
    """hard_residency_default=True: an UNFLAGGED region is treated as hard."""
    _configure(
        enabled=True,
        hard_residency_default=True,
        region_map={"eu": ["eu-west-1"]},
    )
    us = _dep("m-us", "us-east-1", "d1")
    ctx = _ctx(metadata={"region": "eu"})  # no residency flag
    out = drop_out_of_region(ctx, [us])
    assert out == []  # hard-by-default -> never out-of-region


# ---------------------------------------------------------------------------
# Soft preference (RouteIQ-60cc acceptance #2)
# ---------------------------------------------------------------------------


def test_soft_prefers_in_region():
    """An unconstrained request keeps the in-region subset when one exists."""
    _configure(enabled=True, region_map={"eu": ["eu-west-1"]})
    us = _dep("m-us", "us-east-1", "d1")
    eu = _dep("m-eu", "eu-west-1", "d2")
    ctx = _ctx(metadata={"region": "eu"})  # no residency flag -> soft
    out = drop_out_of_region(ctx, [us, eu])
    assert out == [eu]


def test_soft_falls_back_to_full_set_when_no_in_region():
    """Acceptance: unconstrained + no in-region arm -> the FULL set (fallback)."""
    _configure(enabled=True, region_map={"eu": ["eu-west-1"]})
    us1 = _dep("m-us1", "us-east-1", "d1")
    us2 = _dep("m-us2", "us-west-2", "d2")
    cands = [us1, us2]
    ctx = _ctx(metadata={"region": "eu"})
    out = drop_out_of_region(ctx, cands)
    assert out is cands  # availability over locality -> original object


def test_all_in_region_is_byte_stable():
    """When every candidate is already in-region the original object is returned."""
    _configure(enabled=True, region_map={"eu": ["eu-west-1", "eu-central-1"]})
    eu1 = _dep("m-eu1", "eu-west-1", "d1")
    eu2 = _dep("m-eu2", "eu-central-1", "d2")
    cands = [eu1, eu2]
    ctx = _ctx(metadata={"region": "eu"})
    out = drop_out_of_region(ctx, cands)
    assert out is cands


# ---------------------------------------------------------------------------
# Region resolution: header > metadata, verbatim token, map
# ---------------------------------------------------------------------------


def test_region_from_header_case_insensitive():
    """The configured header carries the region token (case-insensitive key)."""
    _configure(enabled=True, region_header="X-RouteIQ-Region")
    us = _dep("m-us", "us-east-1", "d1")
    eu = _dep("m-eu", "eu-west-1", "d2")
    ctx = _ctx(request_kwargs={"headers": {"x-routeiq-region": "eu-west-1"}})
    out = drop_out_of_region(ctx, [us, eu])
    assert out == [eu]  # verbatim token (no map entry needed)


def test_header_takes_priority_over_metadata_region():
    """Header is resolved before the metadata region key."""
    _configure(enabled=True)
    eu = _dep("m-eu", "eu-west-1", "d1")
    us = _dep("m-us", "us-east-1", "d2")
    ctx = RoutingContext(
        router=None,
        model="claude",
        request_kwargs={"headers": {"X-RouteIQ-Region": "eu-west-1"}},
        metadata={"region": "us-east-1"},
    )
    token, _hard = resolve_request_region(ctx, get_settings().region_routing)
    assert token == "eu-west-1"
    out = drop_out_of_region(ctx, [eu, us])
    assert out == [eu]


def test_verbatim_region_token_needs_no_map():
    """A token equal to an aws_region_name matches with zero region_map config."""
    _configure(enabled=True)  # empty region_map
    eu = _dep("m-eu", "eu-west-1", "d1")
    us = _dep("m-us", "us-east-1", "d2")
    ctx = _ctx(metadata={"region": "eu-west-1", "residency": True})
    out = drop_out_of_region(ctx, [eu, us])
    assert out == [eu]


def test_resolve_request_region_returns_hard_flag():
    """resolve_request_region surfaces the (token, is_hard) tuple correctly."""
    _configure(enabled=True)
    settings = get_settings().region_routing
    hard_ctx = _ctx(metadata={"region": "eu", "residency": "yes"})
    assert resolve_request_region(hard_ctx, settings) == ("eu", True)
    soft_ctx = _ctx(metadata={"region": "eu"})
    assert resolve_request_region(soft_ctx, settings) == ("eu", False)
    none_ctx = _ctx(metadata={})
    assert resolve_request_region(none_ctx, settings) == ("", False)


# ---------------------------------------------------------------------------
# Composition through filter_routable_candidates (the real seam)
# ---------------------------------------------------------------------------


class _NoCooldownRouter:
    def __init__(self) -> None:
        self.model_list: list = []


def test_filter_routable_candidates_threads_region(monkeypatch):
    """filter_routable_candidates applies the region filter when context is passed.

    Cooldown is stubbed empty + no gov-ban, so the region filter is the only
    active stage; a hard-residency request drops the out-of-region arm.
    """
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    _configure(enabled=True, region_map={"eu": ["eu-west-1"]})
    us = _dep("m-us", "us-east-1", "d1")
    eu = _dep("m-eu", "eu-west-1", "d2")
    ctx = _ctx(metadata={"region": "eu", "residency": True})
    out = filter_routable_candidates(_NoCooldownRouter(), [us, eu], ctx)
    assert out == [eu]


def test_filter_routable_candidates_no_context_is_legacy_no_op(monkeypatch):
    """Without a context arg the region filter never runs (legacy callers safe)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    _configure(enabled=True, region_map={"eu": ["eu-west-1"]})
    us = _dep("m-us", "us-east-1", "d1")
    eu = _dep("m-eu", "eu-west-1", "d2")
    cands = [us, eu]
    out = filter_routable_candidates(_NoCooldownRouter(), cands)
    assert out == cands  # region filter inert without a context


# ---------------------------------------------------------------------------
# END-TO-END residency through the ACTUAL strategies (RouteIQ-60cc)
# ---------------------------------------------------------------------------
#
# The unit tests above exercise the filter primitive. These build the REAL
# strategy candidate sources -- the DEFAULT strategy
# (``DefaultStrategy._get_deployments``) and the Kumaraswamy-Thompson bandit
# (``KumaraswamyThompsonStrategy.select_deployment``) -- with a HARD-residency
# request (X-RouteIQ-Region header + residency:true) and an out-of-region-ONLY
# deployment group, and assert the out-of-region arm is NEVER selected: the
# candidate set empties so the strategy yields None / falls back, rather than
# leaking a residency violation.


class _StaticRouter:
    """Minimal Router stub: a static healthy_deployments alias (the RouteIQ path
    reads this, bypassing LiteLLM's healthy-deployment pipeline)."""

    def __init__(self, deployments: List[Dict]) -> None:
        self.model_list = deployments
        self.healthy_deployments = deployments


def _hard_residency_ctx(router: Any, deployments: List[Dict]) -> RoutingContext:
    """A RoutingContext carrying a HARD-residency signal via the configured
    header AND a truthy residency flag (request_kwargs is the first source the
    region filter scans)."""
    return RoutingContext(
        router=router,
        model="claude",
        request_kwargs={
            "headers": {"X-RouteIQ-Region": "eu"},
            "residency": True,
        },
    )


def test_default_strategy_get_deployments_drops_out_of_region(monkeypatch):
    """ACCEPTANCE: the DEFAULT strategy candidate source empties for a HARD
    residency request when only out-of-region arms exist (never leaks)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.strategy_registry import DefaultStrategy

    _configure(
        enabled=True,
        region_header="X-RouteIQ-Region",
        region_map={"eu": ["eu-west-1", "eu-central-1"]},
    )
    # Out-of-region ONLY (both us-*); the request demands eu (hard residency).
    us1 = _dep("us-east/claude", "us-east-1", "d1")
    us2 = _dep("us-west/claude", "us-west-2", "d2")
    router = _StaticRouter([us1, us2])
    ctx = _hard_residency_ctx(router, [us1, us2])

    model_list, deployment_map = DefaultStrategy()._get_deployments(ctx)

    # The candidate set MUST be empty -> the strategy returns None / falls back.
    assert model_list == []
    assert deployment_map == {}
    # And neither out-of-region arm key leaked into the scored set.
    assert "us-east/claude" not in deployment_map
    assert "us-west/claude" not in deployment_map


def test_default_strategy_get_deployments_keeps_in_region(monkeypatch):
    """The DEFAULT strategy keeps ONLY the in-region arm for a HARD residency
    request when an in-region arm exists (control: filter is active, not a
    blanket empty)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.strategy_registry import DefaultStrategy

    _configure(
        enabled=True,
        region_header="X-RouteIQ-Region",
        region_map={"eu": ["eu-west-1"]},
    )
    us = _dep("us/claude", "us-east-1", "d1")
    eu = _dep("eu/claude", "eu-west-1", "d2")
    router = _StaticRouter([us, eu])
    ctx = _hard_residency_ctx(router, [us, eu])

    model_list, deployment_map = DefaultStrategy()._get_deployments(ctx)

    assert model_list == ["eu/claude"]
    assert "us/claude" not in deployment_map


def test_kumaraswamy_thompson_never_selects_out_of_region(monkeypatch):
    """ACCEPTANCE: KumaraswamyThompson.select_deployment NEVER returns an
    out-of-region arm for a HARD residency request -- the candidate set empties
    and select_deployment yields None (no leak, no fallback to out-of-region)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.kumaraswamy_thompson import KumaraswamyThompsonStrategy

    _configure(
        enabled=True,
        region_header="X-RouteIQ-Region",
        region_map={"eu": ["eu-west-1", "eu-central-1"]},
    )
    # Out-of-region ONLY: a HARD-residency eu request must reject both.
    us1 = _dep("us-east/claude", "us-east-1", "d1")
    us2 = _dep("us-west/claude", "us-west-2", "d2")
    router = _StaticRouter([us1, us2])
    ctx = _hard_residency_ctx(router, [us1, us2])

    selected = KumaraswamyThompsonStrategy().select_deployment(ctx)

    # The bandit MUST NOT pick an out-of-region arm; the set empties -> None.
    assert selected is None


def test_kumaraswamy_thompson_selects_in_region_arm(monkeypatch):
    """KumaraswamyThompson selects ONLY from the in-region subset for a HARD
    residency request (control: an in-region arm IS selectable)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.kumaraswamy_thompson import KumaraswamyThompsonStrategy

    _configure(
        enabled=True,
        region_header="X-RouteIQ-Region",
        region_map={"eu": ["eu-west-1"]},
    )
    us = _dep("us/claude", "us-east-1", "d1")
    eu = _dep("eu/claude", "eu-west-1", "d2")
    router = _StaticRouter([us, eu])
    ctx = _hard_residency_ctx(router, [us, eu])

    selected = KumaraswamyThompsonStrategy().select_deployment(ctx)

    assert selected is not None
    # Whatever arm is sampled, it is the in-region one (the only one allowed).
    assert selected["litellm_params"]["model"] == "eu/claude"
    assert selected["litellm_params"]["aws_region_name"] == "eu-west-1"
