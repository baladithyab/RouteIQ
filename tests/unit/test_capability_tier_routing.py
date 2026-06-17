"""Unit tests for INTELLIGENCE-FIRST capability-tier routing (RouteIQ-8e37).

The capability-tier FLOOR is a PRE-SCORING candidate filter on the same
``filter_routable_candidates`` seam every RouteIQ strategy already calls (the
region filter — RouteIQ-60cc — is the precedent), so it composes with gov-ban +
cooldown + region and benefits EVERY strategy (default / ML / bandit / centroid /
FALLBACK / LinUCB) with no new strategy subclass. Acceptance:

  (1) a HARD-reasoning request NEVER selects a sub-tier model on the
      default / ML / bandit / centroid / FALLBACK paths (tested each, like
      region); a SIMPLE request can use any tier.
  (2) capability tiers are config-driven with safe defaults; an UNKNOWN model is
      never wrongly excluded (resolves to the safe MIDDLE ``default_tier``).
  (3) the ``intelligence_first`` reward profile is a one-flag switch; the default
      stays ``balanced`` (0.5/0.4/0.1, byte-stable).
  (4) graceful degrade (SOFT) vs STRICT is configurable; never 500 on empty.

All cred-free: settings are injected via ``get_settings(capability_routing=...)``
after ``reset_settings()`` (conftest autouse fixture resets between tests);
difficulty is driven by reasoning MARKERS (pure regex, no centroid cold-load).
"""

from __future__ import annotations

from typing import Any, Dict, List

from litellm_llmrouter.candidate_filter import (
    drop_below_capability_tier,
    filter_routable_candidates,
)
from litellm_llmrouter.centroid_routing import (
    MODEL_CAPABILITY_TIERS,
    get_model_tier,
    resolve_request_difficulty,
)
from litellm_llmrouter.settings import get_settings, reset_settings
from litellm_llmrouter.strategy_registry import RoutingContext

# A prompt with >= 2 distinct reasoning markers -> resolves to "complex"
# difficulty WITHOUT needing the centroid model warm (pure-regex markers).
_HARD_PROMPT = "Please reason step-by-step and prove that the result is correct."
_SIMPLE_PROMPT = "hi there, what time is it?"


def _dep(arm: str, dep_id: str, model_name: str = "claude") -> Dict:
    return {
        "model_name": model_name,
        "litellm_params": {"model": arm},
        "model_info": {"id": dep_id},
    }


def _ctx(prompt: str, *, model: str = "claude") -> RoutingContext:
    return RoutingContext(
        router=None,
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )


def _configure(**capability_routing: Any) -> None:
    """Reset + create the settings singleton with a capability_routing override."""
    reset_settings()
    get_settings(capability_routing=capability_routing)


# ---------------------------------------------------------------------------
# Difficulty resolution (warm-only centroid + reasoning markers)
# ---------------------------------------------------------------------------


def test_difficulty_complex_from_reasoning_markers():
    """>= 2 reasoning markers -> 'complex' (no centroid warm-load needed)."""
    assert resolve_request_difficulty(_ctx(_HARD_PROMPT)) == "complex"


def test_difficulty_simple_for_plain_prompt():
    """A plain prompt with no markers (and cold centroid) -> 'simple'."""
    assert resolve_request_difficulty(_ctx(_SIMPLE_PROMPT)) == "simple"


def test_difficulty_empty_context_is_simple():
    """No prompt text -> the most permissive 'simple' (never blocks routing)."""
    assert (
        resolve_request_difficulty(RoutingContext(router=None, model="m")) == "simple"
    )


def test_difficulty_reads_request_kwargs_messages():
    """Difficulty falls back to request_kwargs['messages'] (ML-path context)."""
    ctx = RoutingContext(
        router=None,
        model="m",
        request_kwargs={"messages": [{"role": "user", "content": _HARD_PROMPT}]},
    )
    assert resolve_request_difficulty(ctx) == "complex"


# ---------------------------------------------------------------------------
# Config-driven tiers + safe defaults (acceptance #2)
# ---------------------------------------------------------------------------


def test_known_family_tiers_defaults():
    """Known families resolve to their built-in tier (opus=expert, haiku=fast)."""
    assert get_model_tier("claude-opus-4-6-20250918", {}, "advanced") == "expert"
    assert get_model_tier("claude-3-5-haiku-20241022", {}, "advanced") == "fast"
    assert get_model_tier("gpt-4o", {}, "advanced") == "advanced"


def test_region_prefixed_arm_matches_family_substring():
    """A provider/region-prefixed arm matches its family by substring."""
    arm = "bedrock/global.anthropic.claude-opus-4-6-20250918"
    assert get_model_tier(arm, {}, "advanced") == "expert"


def test_unknown_model_resolves_to_safe_default():
    """ACCEPTANCE: an unknown model is NEVER wrongly excluded -> default tier."""
    assert get_model_tier("totally-unknown-model", {}, "advanced") == "advanced"


def test_operator_override_wins_over_default():
    """ACCEPTANCE: operator model_tiers override the built-in defaults."""
    # Override a normally-fast model UP to expert.
    assert (
        get_model_tier("claude-3-5-haiku-20241022", {"haiku": "expert"}, "advanced")
        == "expert"
    )
    # Pin an unknown model.
    assert get_model_tier("my-7b", {"my-7b": "fast"}, "advanced") == "fast"


def test_deepseek_minimax_kimi_o_series_are_expert():
    """The reasoning families called out in the directive default to expert."""
    for arm in (
        "deepseek-r1",
        "deepseek-reasoner",
        "minimax-m1",
        "kimi-k2",
        "o1",
        "o3-mini",
    ):
        assert get_model_tier(arm, {}, "advanced") == "expert", arm
    # nova-lite is a fast tier.
    assert get_model_tier("bedrock/amazon.nova-lite-v1", {}, "advanced") == "fast"


# ---------------------------------------------------------------------------
# The gate primitive: drop_below_capability_tier
# ---------------------------------------------------------------------------


def test_disabled_is_identity_no_op():
    """enabled=False -> the input list OBJECT is returned unchanged (byte-stable)."""
    _configure(enabled=False)
    cands = [
        _dep("claude-opus-4-6-20250918", "d1"),
        _dep("claude-3-5-haiku-20241022", "d2"),
    ]
    out = drop_below_capability_tier(_ctx(_HARD_PROMPT), cands)
    assert out is cands


def test_none_context_is_no_op():
    """A None context (legacy call shape) -> no-op even when enabled."""
    _configure(enabled=True)
    cands = [_dep("claude-opus-4-6-20250918", "d1")]
    assert drop_below_capability_tier(None, cands) is cands


def test_simple_request_accepts_any_tier():
    """ACCEPTANCE: a SIMPLE request keeps ALL tiers (lowest floor, byte-stable)."""
    _configure(enabled=True)
    cands = [
        _dep("claude-3-5-haiku-20241022", "d1"),
        _dep("gpt-4o", "d2"),
        _dep("claude-opus-4-6-20250918", "d3"),
    ]
    out = drop_below_capability_tier(_ctx(_SIMPLE_PROMPT), cands)
    assert out is cands  # simple->fast floor drops nothing -> same object


def test_hard_request_drops_sub_expert_models():
    """ACCEPTANCE: a HARD-reasoning request keeps ONLY expert-tier arms."""
    _configure(enabled=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    gpt4o = _dep("gpt-4o", "d2")  # advanced
    opus = _dep("claude-opus-4-6-20250918", "d3")  # expert
    out = drop_below_capability_tier(_ctx(_HARD_PROMPT), [haiku, gpt4o, opus])
    assert out == [opus]
    assert haiku not in out and gpt4o not in out


def test_all_capable_is_byte_stable():
    """When every candidate meets the floor the original object is returned."""
    _configure(enabled=True)
    opus = _dep("claude-opus-4-6-20250918", "d1")
    o1 = _dep("o1", "d2")
    cands = [opus, o1]
    out = drop_below_capability_tier(_ctx(_HARD_PROMPT), cands)
    assert out is cands


# ---------------------------------------------------------------------------
# Graceful degrade (SOFT) vs STRICT (acceptance #4) — never 500 on empty
# ---------------------------------------------------------------------------


def test_soft_degrades_to_full_set_when_no_capable_model():
    """ACCEPTANCE: SOFT (default) + no capable model -> the FULL set (never 500)."""
    _configure(enabled=True)  # strict defaults False
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    gpt4o = _dep("gpt-4o", "d2")
    cands = [haiku, gpt4o]  # neither is expert
    out = drop_below_capability_tier(_ctx(_HARD_PROMPT), cands)
    assert out is cands  # degrade to full set (availability over capability)


def test_strict_empties_when_no_capable_model():
    """ACCEPTANCE: STRICT + no capable model -> EMPTY (strategy yields None)."""
    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    gpt4o = _dep("gpt-4o", "d2")
    out = drop_below_capability_tier(_ctx(_HARD_PROMPT), [haiku, gpt4o])
    assert out == []


def test_strict_keeps_capable_when_one_exists():
    """STRICT keeps the capable arm when one exists (control: not a blanket empty)."""
    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    opus = _dep("claude-opus-4-6-20250918", "d2")
    out = drop_below_capability_tier(_ctx(_HARD_PROMPT), [haiku, opus])
    assert out == [opus]


def test_custom_min_tier_loosens_floor():
    """complex_min_tier='advanced' lets advanced models through for a hard req."""
    _configure(enabled=True, complex_min_tier="advanced")
    haiku = _dep("claude-3-5-haiku-20241022", "d1")  # fast -> dropped
    gpt4o = _dep("gpt-4o", "d2")  # advanced -> kept
    opus = _dep("claude-opus-4-6-20250918", "d3")  # expert -> kept
    out = drop_below_capability_tier(_ctx(_HARD_PROMPT), [haiku, gpt4o, opus])
    assert out == [gpt4o, opus]


# ---------------------------------------------------------------------------
# Composition through filter_routable_candidates (the real seam)
# ---------------------------------------------------------------------------


class _NoCooldownRouter:
    def __init__(self) -> None:
        self.model_list: list = []


def test_filter_routable_candidates_threads_capability(monkeypatch):
    """filter_routable_candidates applies the capability floor when enabled."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    opus = _dep("claude-opus-4-6-20250918", "d2")
    out = filter_routable_candidates(
        _NoCooldownRouter(), [haiku, opus], _ctx(_HARD_PROMPT)
    )
    assert out == [opus]


def test_filter_routable_candidates_no_context_is_legacy_no_op(monkeypatch):
    """Without a context arg the capability floor never runs (legacy callers safe)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    opus = _dep("claude-opus-4-6-20250918", "d2")
    cands = [haiku, opus]
    out = filter_routable_candidates(_NoCooldownRouter(), cands)
    assert out == cands  # floor inert without a context


def test_capability_composes_with_gov_ban(monkeypatch):
    """gov-ban runs FIRST: a banned expert arm is removed before the floor."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    _configure(enabled=True, strict=True)
    # Fable 5 is always-banned even though it would tier as expert-ish.
    fable = {
        "model_name": "claude",
        "litellm_params": {"model": "bedrock/global.anthropic.claude-fable-5"},
        "model_info": {"id": "df"},
    }
    opus = _dep("claude-opus-4-6-20250918", "d2")
    out = filter_routable_candidates(
        _NoCooldownRouter(), [fable, opus], _ctx(_HARD_PROMPT)
    )
    assert out == [opus]  # banned arm gone, expert arm kept


# ---------------------------------------------------------------------------
# END-TO-END through the ACTUAL strategies (acceptance #1) — like region
# ---------------------------------------------------------------------------


class _StaticRouter:
    """Minimal Router stub: a static healthy_deployments alias (the RouteIQ path
    reads this, bypassing LiteLLM's healthy-deployment pipeline)."""

    def __init__(self, deployments: List[Dict]) -> None:
        self.model_list = deployments
        self.healthy_deployments = deployments


def _hard_ctx(router: Any) -> RoutingContext:
    return RoutingContext(
        router=router,
        model="claude",
        messages=[{"role": "user", "content": _HARD_PROMPT}],
    )


def test_default_strategy_drops_sub_tier_for_hard_request(monkeypatch):
    """ACCEPTANCE (default path): DefaultStrategy._get_deployments keeps only the
    expert arm for a HARD-reasoning request (strict empties when none capable)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.strategy_registry import DefaultStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    opus = _dep("claude-opus-4-6-20250918", "d2")
    router = _StaticRouter([haiku, opus])
    model_list, dep_map = DefaultStrategy()._get_deployments(_hard_ctx(router))
    assert model_list == ["claude-opus-4-6-20250918"]
    assert "claude-3-5-haiku-20241022" not in dep_map


def test_default_strategy_strict_empties_when_no_expert(monkeypatch):
    """ACCEPTANCE (default path): strict + no expert arm -> empty candidate set."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.strategy_registry import DefaultStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    gpt4o = _dep("gpt-4o", "d2")
    router = _StaticRouter([haiku, gpt4o])
    model_list, dep_map = DefaultStrategy()._get_deployments(_hard_ctx(router))
    assert model_list == []
    assert dep_map == {}


def test_kumaraswamy_thompson_never_selects_sub_tier(monkeypatch):
    """ACCEPTANCE (bandit path): KumaraswamyThompson NEVER scores a sub-tier model
    for a HARD-reasoning request -- in strict mode the set empties -> None."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.kumaraswamy_thompson import KumaraswamyThompsonStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    gpt4o = _dep("gpt-4o", "d2")
    router = _StaticRouter([haiku, gpt4o])  # no expert arm
    selected = KumaraswamyThompsonStrategy().select_deployment(_hard_ctx(router))
    assert selected is None


def test_kumaraswamy_thompson_selects_expert_arm(monkeypatch):
    """ACCEPTANCE (bandit path): KumaraswamyThompson selects ONLY the expert arm
    for a HARD-reasoning request (control: an expert arm IS selectable)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.kumaraswamy_thompson import KumaraswamyThompsonStrategy

    _configure(enabled=True)  # soft (default): expert exists, no degrade needed
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    opus = _dep("claude-opus-4-6-20250918", "d2")
    router = _StaticRouter([haiku, opus])
    selected = KumaraswamyThompsonStrategy().select_deployment(_hard_ctx(router))
    assert selected is not None
    assert selected["litellm_params"]["model"] == "claude-opus-4-6-20250918"


def test_linucb_never_selects_sub_tier(monkeypatch):
    """ACCEPTANCE (LinUCB path): the contextual bandit also honours the floor."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.kumaraswamy_thompson import LinUCBRoutingStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    gpt4o = _dep("gpt-4o", "d2")
    router = _StaticRouter([haiku, gpt4o])
    assert LinUCBRoutingStrategy().select_deployment(_hard_ctx(router)) is None


def test_linucb_selects_expert_arm(monkeypatch):
    """LinUCB selects from the expert subset for a HARD request (control)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.kumaraswamy_thompson import LinUCBRoutingStrategy

    _configure(enabled=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    opus = _dep("claude-opus-4-6-20250918", "d2")
    router = _StaticRouter([haiku, opus])
    selected = LinUCBRoutingStrategy().select_deployment(_hard_ctx(router))
    assert selected is not None
    assert selected["litellm_params"]["model"] == "claude-opus-4-6-20250918"


def test_centroid_get_healthy_deployments_drops_sub_tier(monkeypatch):
    """ACCEPTANCE (centroid path): the centroid candidate source applies the
    capability floor (strict empties when no expert arm exists)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.centroid_routing import CentroidRoutingStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    gpt4o = _dep("gpt-4o", "d2")
    router = _StaticRouter([haiku, gpt4o])
    out = CentroidRoutingStrategy._get_healthy_deployments(_hard_ctx(router))
    assert out == []


def test_centroid_get_healthy_deployments_keeps_expert(monkeypatch):
    """centroid candidate source keeps the expert arm for a hard request."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.centroid_routing import CentroidRoutingStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    opus = _dep("claude-opus-4-6-20250918", "d2")
    router = _StaticRouter([haiku, opus])
    out = CentroidRoutingStrategy._get_healthy_deployments(_hard_ctx(router))
    assert [d["litellm_params"]["model"] for d in out] == ["claude-opus-4-6-20250918"]


def test_centroid_fallback_drops_sub_tier(monkeypatch):
    """ACCEPTANCE (centroid FALLBACK path): even the terminal centroid fallback
    honours the capability floor (strict empties -> None)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.centroid_routing import CentroidRoutingStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    gpt4o = _dep("gpt-4o", "d2")
    router = _StaticRouter([haiku, gpt4o])
    assert CentroidRoutingStrategy._fallback_deployment(_hard_ctx(router)) is None


# ---------------------------------------------------------------------------
# custom_routing_strategy paths: ML _get_model_list + terminal _fallback
# ---------------------------------------------------------------------------


def test_custom_strategy_get_model_list_drops_sub_tier(monkeypatch):
    """ACCEPTANCE (ML path): RouteIQRoutingStrategy._get_model_list applies the
    floor when messages + an enabled floor are threaded."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.custom_routing_strategy import RouteIQRoutingStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    opus = _dep("claude-opus-4-6-20250918", "d2")
    router = _StaticRouter([haiku, opus])
    strat = RouteIQRoutingStrategy(router_instance=router)
    model_list, _ = strat._get_model_list(
        "claude", None, [{"role": "user", "content": _HARD_PROMPT}]
    )
    assert model_list == ["claude-opus-4-6-20250918"]


def test_custom_strategy_fallback_drops_sub_tier(monkeypatch):
    """ACCEPTANCE (terminal FALLBACK path): _fallback_deployment honours the floor
    for a HARD-reasoning request (strict -> None when no expert arm)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.custom_routing_strategy import RouteIQRoutingStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    gpt4o = _dep("gpt-4o", "d2")
    router = _StaticRouter([haiku, gpt4o])
    strat = RouteIQRoutingStrategy(router_instance=router)
    out = strat._fallback_deployment(
        "claude", None, [{"role": "user", "content": _HARD_PROMPT}]
    )
    assert out is None


def test_custom_strategy_fallback_keeps_expert(monkeypatch):
    """terminal fallback keeps the expert arm for a hard request (control)."""
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    from litellm_llmrouter.custom_routing_strategy import RouteIQRoutingStrategy

    _configure(enabled=True, strict=True)
    haiku = _dep("claude-3-5-haiku-20241022", "d1")
    opus = _dep("claude-opus-4-6-20250918", "d2")
    router = _StaticRouter([haiku, opus])
    strat = RouteIQRoutingStrategy(router_instance=router)
    out = strat._fallback_deployment(
        "claude", None, [{"role": "user", "content": _HARD_PROMPT}]
    )
    assert out is not None
    assert out["litellm_params"]["model"] == "claude-opus-4-6-20250918"


# ---------------------------------------------------------------------------
# Intelligence-first reward profile (acceptance #3) — one-flag switch
# ---------------------------------------------------------------------------


def test_reward_profile_default_is_balanced_byte_stable():
    """ACCEPTANCE: the DEFAULT reward profile is balanced (0.5/0.4/0.1)."""
    reset_settings()
    kts = get_settings().kumaraswamy_thompson
    assert kts.reward_profile == "balanced"
    assert kts.resolved_reward_weights() == (0.5, 0.4, 0.1)


def test_reward_profile_respects_explicit_weights_on_balanced():
    """balanced leaves explicit w_* untouched (operator-tuned weights honoured)."""
    reset_settings()
    kts = get_settings(
        kumaraswamy_thompson={"w_quality": 0.7, "w_cost": 0.2, "w_latency": 0.1}
    ).kumaraswamy_thompson
    assert kts.resolved_reward_weights() == (0.7, 0.2, 0.1)


def test_intelligence_first_profile_is_one_flag_switch():
    """ACCEPTANCE: intelligence_first overrides weights to quality~1.0/cost~0."""
    reset_settings()
    kts = get_settings(
        kumaraswamy_thompson={"reward_profile": "intelligence_first"}
    ).kumaraswamy_thompson
    assert kts.resolved_reward_weights() == (1.0, 0.0, 0.0)


def test_cost_aware_profile_weights_cost():
    """cost_aware is the cost-weighted preset."""
    reset_settings()
    kts = get_settings(
        kumaraswamy_thompson={"reward_profile": "cost_aware"}
    ).kumaraswamy_thompson
    assert kts.resolved_reward_weights() == (0.2, 0.7, 0.1)


def test_unknown_profile_falls_back_to_explicit_weights():
    """An unknown profile is treated as balanced (never silently zeroes a weight)."""
    reset_settings()
    kts = get_settings(
        kumaraswamy_thompson={
            "reward_profile": "bogus",
            "w_quality": 0.6,
            "w_cost": 0.3,
            "w_latency": 0.1,
        }
    ).kumaraswamy_thompson
    assert kts.resolved_reward_weights() == (0.6, 0.3, 0.1)


def test_intelligence_first_flows_into_registered_bandit(monkeypatch):
    """The reward profile reaches the registered KumaraswamyThompsonStrategy."""
    from litellm_llmrouter import kumaraswamy_thompson as kt

    reset_settings()
    get_settings(
        kumaraswamy_thompson={"enabled": True, "reward_profile": "intelligence_first"}
    )

    captured: Dict[str, Any] = {}

    class _Registry:
        def register(self, *a: Any, **k: Any) -> None:
            pass

    real_cls = kt.KumaraswamyThompsonStrategy

    def _spy_cls(*args: Any, **kwargs: Any):
        captured.update(kwargs)
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(kt, "KumaraswamyThompsonStrategy", _spy_cls)
    monkeypatch.setattr(
        "litellm_llmrouter.strategy_registry.get_routing_registry",
        lambda: _Registry(),
    )

    assert kt.register_kumaraswamy_thompson_strategy() is True
    # intelligence_first -> quality~1.0, cost~0, latency~0 (post sum-normalize the
    # strategy stores (1.0, 0.0, 0.0)).
    assert captured["w_quality"] == 1.0
    assert captured["w_cost"] == 0.0
    assert captured["w_latency"] == 0.0


# ---------------------------------------------------------------------------
# Sanity: the tier table covers the directive's named families
# ---------------------------------------------------------------------------


def test_tier_table_has_expected_families():
    """Spot-check the built-in tier table for the directive's named models."""
    assert MODEL_CAPABILITY_TIERS["claude-opus-4-6-20250918"] == "expert"
    assert MODEL_CAPABILITY_TIERS["claude-3-5-haiku-20241022"] == "fast"
    assert MODEL_CAPABILITY_TIERS["deepseek-r1"] == "expert"
    assert MODEL_CAPABILITY_TIERS["minimax-m1"] == "expert"
    assert MODEL_CAPABILITY_TIERS["kimi-k2"] == "expert"
    assert MODEL_CAPABILITY_TIERS["nova-lite"] == "fast"
