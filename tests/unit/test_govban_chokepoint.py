"""Gov-ban chokepoint tests (RouteIQ-a073 + RouteIQ-513e).

The standing safety invariant: a gov-banned arm (the always-on Fable 5 family
ban, plus operator ``banned_models``) must NEVER be the deployment RouteIQ
returns to LiteLLM -- regardless of which routing strategy selected it.

RouteIQ-a073 found the pre-scoring filter was wired seam-by-seam and the
CentroidRoutingStrategy + CostAwareRoutingStrategy seams were missed. The fix is
defense-in-depth:

1. Each candidate source pre-filters (banned arms are never scored).
2. A SINGLE post-selection chokepoint (``RouteIQRoutingStrategy._guard_selected``)
   at both public entry methods refuses any banned selection -- so even a
   strategy whose own candidate source forgot to filter cannot route a ban.

These tests assert (2): they force each path to hand back a banned arm and prove
the entry point refuses it. They also assert the two found gaps (centroid,
cost-aware) now pre-filter (1).

All cred-free: the Router is a MagicMock; settings come from ``get_settings``
(reset between tests via the conftest autouse fixture).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.custom_routing_strategy import RouteIQRoutingStrategy
from litellm_llmrouter.settings import get_settings, reset_settings

FABLE5 = "bedrock/global.anthropic.claude-fable-5"


def _dep(model_name: str, arm: str, dep_id: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "litellm_params": {"model": arm, "api_key": "test-key"},
        "model_info": {"id": dep_id},
    }


def _make_router(healthy: List[Dict[str, Any]]) -> MagicMock:
    router = MagicMock()
    router.model_list = healthy
    router.healthy_deployments = healthy
    return router


_LEGAL = _dep("gpt-4", "openai/gpt-4", "d1")
_BANNED = _dep("gpt-4", FABLE5, "d2")


# ---------------------------------------------------------------------------
# The chokepoint: _guard_selected refuses a banned selection from ANY path
# ---------------------------------------------------------------------------


def test_guard_refuses_banned_selection_zero_config():
    """A banned arm handed to the guard is refused, falling back to a legal arm.

    Zero operator config -> relies on the always-on Fable 5 family ban.
    """
    reset_settings()
    get_settings()  # banned_models empty
    router = _make_router([_LEGAL, _BANNED])
    strategy = RouteIQRoutingStrategy(router_instance=router)

    # Simulate a strategy that (wrongly) selected the banned arm.
    guarded = strategy._guard_selected(_BANNED, "gpt-4")

    assert guarded is not None
    assert guarded["litellm_params"]["model"] != FABLE5
    assert guarded["litellm_params"]["model"] == "openai/gpt-4"


def test_guard_returns_none_for_banned_only_group():
    """A group whose ONLY arm is banned yields no deployment (fail-closed)."""
    reset_settings()
    get_settings()
    router = _make_router([_BANNED])
    strategy = RouteIQRoutingStrategy(router_instance=router)

    assert strategy._guard_selected(_BANNED, "gpt-4") is None


def test_guard_passes_through_legal_selection():
    """A legal selection is returned unchanged (no false refusal)."""
    reset_settings()
    get_settings()
    router = _make_router([_LEGAL, _BANNED])
    strategy = RouteIQRoutingStrategy(router_instance=router)

    assert strategy._guard_selected(_LEGAL, "gpt-4") is _LEGAL


@pytest.mark.asyncio
async def test_async_entry_refuses_banned_arm_from_a_rogue_strategy():
    """End-to-end: a strategy that returns a banned arm is refused at the entry.

    This is the 'unbypassable' proof — we patch the INNER selection to return a
    banned arm directly (simulating a future strategy that forgot to filter),
    and assert the public entry method never hands it to LiteLLM.
    """
    reset_settings()
    get_settings()
    router = _make_router([_LEGAL, _BANNED])
    strategy = RouteIQRoutingStrategy(router_instance=router)

    async def _rogue(**_kwargs: Any) -> Optional[Dict[str, Any]]:
        return _BANNED  # a strategy bug: hands back a banned arm

    with patch.object(strategy, "_async_select", side_effect=_rogue):
        result = await strategy.async_get_available_deployment(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert result is not None
    assert result["litellm_params"]["model"] != FABLE5


def test_sync_entry_refuses_banned_arm_from_a_rogue_strategy():
    """Sync entry point applies the same chokepoint."""
    reset_settings()
    get_settings()
    router = _make_router([_LEGAL, _BANNED])
    strategy = RouteIQRoutingStrategy(router_instance=router)

    with patch.object(strategy, "_sync_select", return_value=_BANNED):
        result = strategy.get_available_deployment(model="gpt-4")

    assert result is not None
    assert result["litellm_params"]["model"] != FABLE5


def test_guard_honors_operator_configured_ban():
    """The chokepoint also enforces operator ``banned_models`` (not just Fable 5)."""
    reset_settings()
    get_settings(governance={"banned_models": ["openai/gpt-4"]})
    router = _make_router(
        [_dep("grp", "openai/gpt-4", "d1"), _dep("grp", "openai/gpt-4o", "d2")]
    )
    strategy = RouteIQRoutingStrategy(router_instance=router)

    guarded = strategy._guard_selected(_dep("grp", "openai/gpt-4", "d1"), "grp")
    assert guarded is not None
    assert guarded["litellm_params"]["model"] == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# The two found gaps now pre-filter (RouteIQ-a073)
# ---------------------------------------------------------------------------


def test_centroid_get_healthy_deployments_excludes_banned():
    """centroid _get_healthy_deployments drops the banned arm (the named gap)."""
    from litellm_llmrouter.centroid_routing import CentroidRoutingStrategy

    reset_settings()
    get_settings()
    router = _make_router([_LEGAL, _BANNED])
    ctx = MagicMock()
    ctx.router = router
    ctx.model = "gpt-4"

    out = CentroidRoutingStrategy._get_healthy_deployments(ctx)
    arms = [d["litellm_params"]["model"] for d in out]
    assert FABLE5 not in arms
    assert "openai/gpt-4" in arms


def test_centroid_fallback_excludes_banned():
    """centroid _fallback_deployment never returns the banned arm."""
    from litellm_llmrouter.centroid_routing import CentroidRoutingStrategy

    reset_settings()
    get_settings()
    router = _make_router([_BANNED, _LEGAL])
    ctx = MagicMock()
    ctx.router = router
    ctx.model = "gpt-4"

    for _ in range(20):  # _fallback uses random.choice — sample repeatedly
        out = CentroidRoutingStrategy._fallback_deployment(ctx)
        assert out is not None
        assert out["litellm_params"]["model"] != FABLE5


def test_cost_aware_get_candidates_excludes_banned():
    """CostAwareRoutingStrategy _get_candidates drops the banned arm (found gap)."""
    from litellm_llmrouter.strategies import CostAwareRoutingStrategy

    reset_settings()
    get_settings()
    router = _make_router([_LEGAL, _BANNED])
    ctx = MagicMock()
    ctx.router = router
    ctx.model = "gpt-4"

    strategy = CostAwareRoutingStrategy()
    out = strategy._get_candidates(ctx)
    arms = [d["litellm_params"]["model"] for d in out]
    assert FABLE5 not in arms
    assert "openai/gpt-4" in arms
