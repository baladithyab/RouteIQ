"""Unit tests for the pre-scoring candidate filter (RouteIQ-99e8 + RouteIQ-badb).

RouteIQ-99e8 (cooldown): a deployment LiteLLM has cooled down must be EXCLUDED
from the candidate set the strategy scores -- not merely retried after a failed
selection. Fail-open if every candidate is cooled down.

RouteIQ-badb (gov-ban): a gov-banned model (config-driven via
``get_settings().governance.banned_models``, e.g. Fable 5) must be REMOVED from
the arm set BEFORE scoring and NEVER re-added, even if it is the sole candidate.
Default-empty config is a byte-stable no-op.

All cred-free: the Router is a stub; cooldown state is injected; settings come
from ``get_settings`` (reset between tests via the conftest autouse fixture).
"""

from __future__ import annotations

from litellm_llmrouter.candidate_filter import (
    banned_model_keys,
    cooled_down_ids,
    drop_cooled_down,
    drop_gov_banned,
    filter_routable_candidates,
    is_gov_banned,
)
from litellm_llmrouter.settings import get_settings, reset_settings

FABLE5 = "bedrock/global.anthropic.claude-fable-5"


def _dep(model_name: str, arm: str, dep_id: str) -> dict:
    return {
        "model_name": model_name,
        "litellm_params": {"model": arm},
        "model_info": {"id": dep_id},
    }


class _CooldownRouter:
    """Stub Router whose cooldown source reports a fixed set of model_info.id.

    Mirrors what LiteLLM's ``_get_cooldown_deployments`` reads: it calls
    ``router.cooldown_cache.get_active_cooldowns(...)``. We stub the RouteIQ
    helper directly via monkeypatch in the tests instead, so this Router only
    needs to exist as an opaque object.
    """

    def __init__(self) -> None:
        self.model_list: list = []


# ---------------------------------------------------------------------------
# RouteIQ-99e8 — cooldown pre-filter
# ---------------------------------------------------------------------------


def test_cooldown_excludes_arm_before_scoring(monkeypatch):
    """The 99e8 acceptance: a cooled-down deployment is EXCLUDED from the set."""
    d1 = _dep("gpt-4", "bedrock/a", "d1")
    d2 = _dep("gpt-4", "bedrock/b", "d2")

    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: {"d2"},
    )
    kept = drop_cooled_down(_CooldownRouter(), [d1, d2])
    kept_ids = [d["model_info"]["id"] for d in kept]
    assert "d2" not in kept_ids
    assert kept_ids == ["d1"]


def test_cooldown_fail_open_when_all_cooled(monkeypatch):
    """If EVERY candidate is cooled down, return the original set (fail-open)."""
    d1 = _dep("gpt-4", "bedrock/a", "d1")
    d2 = _dep("gpt-4", "bedrock/b", "d2")
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: {"d1", "d2"},
    )
    kept = drop_cooled_down(_CooldownRouter(), [d1, d2])
    assert kept == [d1, d2]


def test_cooldown_no_cooldown_is_byte_stable(monkeypatch):
    """Empty cooldown set -> the same list object is returned (byte-stable)."""
    cands = [_dep("gpt-4", "bedrock/a", "d1"), _dep("gpt-4", "bedrock/b", "d2")]
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: set(),
    )
    kept = drop_cooled_down(_CooldownRouter(), cands)
    assert kept is cands


def test_cooled_down_ids_fail_open_on_error():
    """A cooldown-lookup failure must fail OPEN (empty set), never wedge routing."""

    class _Boom:
        pass

    # No litellm cooldown handler resolvable against this stub -> empty set.
    assert cooled_down_ids(_Boom()) == set()


# ---------------------------------------------------------------------------
# RouteIQ-badb — gov-ban pre-filter
# ---------------------------------------------------------------------------


def test_gov_ban_removes_fable5():
    """The badb acceptance: a gov-banned arm is removed; legal arms kept."""
    reset_settings()
    get_settings(governance={"banned_models": [FABLE5]})
    legal = _dep("claude", "bedrock/anthropic.claude-3-sonnet", "d1")
    banned = _dep("fable5", FABLE5, "d2")
    out = drop_gov_banned([legal, banned])
    assert out == [legal]


def test_gov_ban_default_empty_byte_stable():
    """Default-empty config -> the same list object is returned (no filtering)."""
    reset_settings()
    get_settings()  # default GovernanceSettings -> banned_models == []
    assert banned_model_keys() == set()
    cands = [_dep("claude", "bedrock/anthropic.claude-3-sonnet", "d1")]
    assert drop_gov_banned(cands) is cands


def test_gov_ban_removes_sole_banned_candidate():
    """Compliance fail-closed-to-removal: a banned arm is removed even when sole."""
    reset_settings()
    get_settings(governance={"banned_models": [FABLE5]})
    banned = _dep("fable5", FABLE5, "d1")
    assert drop_gov_banned([banned]) == []


def test_gov_ban_matches_model_name_or_arm_key():
    """A banned key matches EITHER litellm_params.model OR model_name."""
    reset_settings()
    get_settings(governance={"banned_models": ["fable5-group", FABLE5]})
    by_arm = _dep("safe-name", FABLE5, "d1")
    by_name = _dep("fable5-group", "bedrock/some-other-arm", "d2")
    legal = _dep("claude", "bedrock/anthropic.claude-3-sonnet", "d3")
    out = drop_gov_banned([by_arm, by_name, legal])
    assert out == [legal]


def test_gov_ban_via_env(monkeypatch):
    """Config also threads through the nested env var (ADR-0013)."""
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE__BANNED_MODELS", f'["{FABLE5}"]')
    reset_settings()
    assert FABLE5 in banned_model_keys()


# ---------------------------------------------------------------------------
# Composition: gov-ban FIRST then cooldown (a banned arm can never be re-added)
# ---------------------------------------------------------------------------


def test_filter_composition_ban_then_cooldown(monkeypatch):
    """gov-ban runs first; cooldown's fail-open re-add cannot re-add a ban."""
    reset_settings()
    get_settings(governance={"banned_models": [FABLE5]})
    legal = _dep("claude", "bedrock/anthropic.claude-3-sonnet", "d1")
    banned = _dep("fable5", FABLE5, "d2")
    # cooldown reports the legal arm cooled -> drop_cooled_down would fail open
    # on its (already ban-filtered) input, re-adding only [legal], never banned.
    monkeypatch.setattr(
        "litellm_llmrouter.candidate_filter.cooled_down_ids",
        lambda router: {"d1"},
    )
    out = filter_routable_candidates(_CooldownRouter(), [legal, banned])
    out_arms = [d["litellm_params"]["model"] for d in out]
    assert FABLE5 not in out_arms
    assert out == [legal]  # fail-open re-add from the ban-filtered set only


# ---------------------------------------------------------------------------
# RouteIQ-513e — Fable 5 banned with ZERO operator config (always-on family ban)
# ---------------------------------------------------------------------------


def test_fable5_banned_with_zero_config():
    """513e acceptance: Fable 5 is removed even when banned_models is EMPTY.

    The standing constraint must hold out-of-the-box; relying on operator
    config alone left it unenforced.
    """
    reset_settings()
    get_settings()  # default GovernanceSettings -> banned_models == []
    assert banned_model_keys() == set()  # operator config is empty
    legal = _dep("claude", "bedrock/anthropic.claude-3-sonnet", "d1")
    banned = _dep("fable5", FABLE5, "d2")
    out = drop_gov_banned([legal, banned])
    assert out == [legal]  # banned by the always-on family ban, not config


def test_fable5_family_ban_catches_all_prefix_variants():
    """Every provider/region-prefixed + bare Fable 5 variant is caught."""
    reset_settings()
    get_settings()
    variants = [
        "bedrock/global.anthropic.claude-fable-5",
        "anthropic/claude-fable-5",
        "claude-fable-5",
        "Bedrock/Global.Anthropic.Claude-Fable-5",  # case-insensitive
        "  claude-fable-5  ",  # whitespace-insensitive
    ]
    for arm in variants:
        assert is_gov_banned(_dep("grp", arm, "d")), arm
    # And by model_name too (group named for fable5).
    assert is_gov_banned(_dep("claude-fable-5-group", "bedrock/safe-arm", "d"))


def test_is_gov_banned_legal_model_is_false():
    """A legal model is not banned (no false positives) with default config."""
    reset_settings()
    get_settings()
    assert not is_gov_banned(_dep("claude", "bedrock/anthropic.claude-3-sonnet", "d1"))
    assert not is_gov_banned(_dep("gpt", "openai/gpt-4o", "d2"))


def test_fable5_banned_even_as_sole_candidate_zero_config():
    """A Fable-5-only group yields NO deployment, even with empty config."""
    reset_settings()
    get_settings()
    assert drop_gov_banned([_dep("fable5", FABLE5, "d1")]) == []


def test_drop_gov_banned_byte_stable_when_nothing_banned():
    """No banned candidate (and empty config) -> same list object (byte-stable)."""
    reset_settings()
    get_settings()
    cands = [_dep("claude", "bedrock/anthropic.claude-3-sonnet", "d1")]
    assert drop_gov_banned(cands) is cands
