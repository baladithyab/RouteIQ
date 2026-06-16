"""
Pre-scoring candidate filter (RouteIQ-99e8 cooldown + RouteIQ-badb gov-ban)
===========================================================================

The RouteIQ custom-routing-strategy path BYPASSES LiteLLM's built-in
healthy-deployment pipeline: ``router.set_custom_routing_strategy`` hot-swaps
``async_get_available_deployment``, so LiteLLM's cooldown filter
(``_get_healthy_deployments`` -> ``_filter_cooldown_deployments``) AND its
``async_filter_deployments`` ``CustomLogger`` hook NEVER run on the RouteIQ
path. Every RouteIQ candidate source instead reads the STATIC
``getattr(router, "healthy_deployments", router.model_list)``, which LiteLLM
sets to ``self.model_list`` at construction and never re-derives per request.

Consequences this module fixes:

- **RouteIQ-99e8 (cooldown):** a deployment LiteLLM has cooled down (provider
  429/5xx) still appears in ``healthy_deployments``, so the bandit/strategy
  scores it, may select it, the call fails, and only THEN does retry kick in
  (reactive). :func:`drop_cooled_down` removes cooled-down arms BEFORE scoring
  by consulting LiteLLM's live cooldown set.
- **RouteIQ-badb (gov-ban):** data-retention / compliance bans (e.g. Fable 5)
  cannot be enforced via LiteLLM's ``async_filter_deployments`` hook because
  that hook lives on the bypassed built-in path. :func:`drop_gov_banned`
  removes config-banned arms (``get_settings().governance.banned_models``)
  BEFORE scoring, RouteIQ-native.

Both are PURE functions reading ``get_settings()`` directly -> no new
singleton, no ``conftest`` reset needed (``reset_settings()`` is already wired).

Fail behaviour (deliberately asymmetric):
- Cooldown fails OPEN (availability): if every candidate is cooled down, the
  original set is returned (no worse than today; avoids zero-candidate
  dead-ends -- mirrors ``KumaraswamyThompsonStrategy._drop_open_breakers``).
- Gov-ban fails CLOSED-to-removal (compliance): a banned arm is removed even if
  it is the sole candidate. Returning no deployment for a gov-banned-only group
  is the correct outcome -- a banned model must NEVER be selected.

Composition order (locked): gov-ban FIRST, then cooldown, so the cooldown
fail-open re-add can never re-introduce a banned arm (it only re-adds from its
own already-ban-filtered input).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Always-on compliance ban (RouteIQ-513e)
# ---------------------------------------------------------------------------
# The standing constraint "Fable 5 is GOV-BANNED as a routable arm — never add
# it to live routing config" must hold with ZERO operator config. Relying on
# ``GovernanceSettings.banned_models`` alone left it unenforced out-of-the-box
# (the default is empty), so a bare deployment that happened to list a Fable 5
# arm could route it. The family is matched (normalized, see ``_tokens``)
# against BOTH the arm key (``litellm_params.model``) and the group name
# (``model_name``) IN ADDITION TO the config-driven bans, so the control survives
# an empty / cleared / glitched ``banned_models``.
#
# Matched on a TOKEN/SEGMENT BOUNDARY (RouteIQ-2e1f), not a raw substring. The
# arm + group name are normalized by splitting on any of ``[/._- ]`` (and lower-
# casing), then the Fable 5 family is matched as a BOUNDED token SUBSEQUENCE. So
# every provider/region-prefixed and separator variant is caught --
# ``bedrock/global.anthropic.claude-fable-5``, ``anthropic/claude-fable-5``, the
# bare ``claude-fable-5``, ``claude_fable_5``, ``claude.fable.5`` -- while a
# look-alike whose terminal version TOKEN differs is NOT (``claude-fable-50``
# tokenizes to ``[..., fable, 50]`` and ``50`` != the token ``5``; the fused
# ``claude-fable50`` -> ``[..., fable50]`` likewise misses ``(fable5,)``). The
# match keys on the token VALUE, not on whether a trailing segment follows, so a
# clean ``(fable, 5)`` run anywhere (e.g. the ``claude-fable-5-group`` group
# name, or a ``claude-fable-5-mini`` variant) IS banned -- the safe direction for
# a compliance ban.
#
# Each entry is the family expressed as the ordered token tuple a model name
# tokenizes to: ``claude-fable-5`` -> ``(fable, 5)`` and ``claude-fable5`` ->
# ``(fable5,)`` (the no-separator spelling is its own single token).
_ALWAYS_BANNED_TOKEN_SEQS: tuple[tuple[str, ...], ...] = (
    ("fable", "5"),
    ("fable5",),
)

# Split on the separators that appear in provider/region-prefixed model ids:
# ``/`` (provider), ``.`` (region/vendor segments), ``-`` and ``_`` (name parts),
# whitespace, AND the version-glue separators ``:`` / ``@`` / ``#`` (RouteIQ-9fce).
# Version-tagged spellings glue the version onto the family with one of these --
# ``claude-fable-5:v1`` / ``claude-fable-5@1`` / ``claude-fable-5#latest`` -- so
# without them the terminal token fuses (``5:v1``) and the ``(fable, 5)`` run is
# missed. Reused to tokenize both the arm and the group name.
_TOKEN_SPLIT_RE = re.compile(r"[/._\-:@# ]+")


def _normalize(s: str) -> str:
    """Lowercase + strip whitespace for case/format-insensitive ban matching."""
    return str(s or "").strip().lower()


def _tokens(s: str) -> list[str]:
    """Tokenize a (normalized) model id on ``[/._- ]`` boundaries, dropping empties."""
    return [t for t in _TOKEN_SPLIT_RE.split(_normalize(s)) if t]


def _contains_token_seq(tokens: list[str], seq: tuple[str, ...]) -> bool:
    """True if ``seq`` appears as a contiguous run within ``tokens``.

    A bounded match: ``(fable, 5)`` matches ``[..., fable, 5, ...]`` but NOT
    ``[..., fable, 50]`` (token ``50`` != token ``5``) nor a fused ``fable5``.
    """
    n = len(seq)
    if n == 0:
        return False
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i : i + n]) == seq:
            return True
    return False


def _is_always_banned(*names: str) -> bool:
    """True if ANY of the given model names tokenizes to a banned Fable 5 family
    token sequence (RouteIQ-513e always-on ban, RouteIQ-2e1f boundary-matched)."""
    for name in names:
        toks = _tokens(name)
        for seq in _ALWAYS_BANNED_TOKEN_SEQS:
            if _contains_token_seq(toks, seq):
                return True
    return False


def cooled_down_ids(router: Any) -> set[str]:
    """Return LiteLLM's live cooldown set as ``model_info.id`` strings.

    Reuses LiteLLM's exact cooldown logic
    (``litellm.router_utils.cooldown_handlers._get_cooldown_deployments``),
    which reads ``router.cooldown_cache.get_active_cooldowns(...)`` and returns
    a list of ``model_info.id`` values. Cred-free and mockable.

    Fails OPEN (returns the empty set) on any error -- a cooldown-lookup failure
    must never wedge routing.
    """
    try:
        from litellm.router_utils.cooldown_handlers import (
            _get_cooldown_deployments,
        )

        cooled = _get_cooldown_deployments(
            litellm_router_instance=router, parent_otel_span=None
        )
        return {str(cid) for cid in (cooled or [])}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("cooled_down_ids lookup failed (fail-open): %s", exc)
        return set()


def drop_cooled_down(
    router: Any, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """RouteIQ-99e8: remove cooled-down deployments BEFORE scoring.

    A candidate is dropped when its ``model_info.id`` is in LiteLLM's live
    cooldown set (mirrors LiteLLM ``_filter_cooldown_deployments``). Fail-OPEN:
    if EVERY candidate is cooled down, the original set is returned unchanged
    (no worse than today; avoids zero-candidate dead-ends).

    Byte-stable when nothing is cooled down (returns the input list object).
    """
    cooled = cooled_down_ids(router)
    if not cooled:
        return candidates
    kept = [
        d
        for d in candidates
        if str((d.get("model_info") or {}).get("id", "")) not in cooled
    ]
    return kept if kept else candidates


def banned_model_keys() -> set[str]:
    """Config-driven gov-ban set from ``get_settings().governance.banned_models``.

    NOTE: this is the OPERATOR-CONFIGURED set only; it does NOT include the
    always-on Fable 5 family ban (:data:`_ALWAYS_BANNED_SUBSTRINGS`). Use
    :func:`is_gov_banned` / :func:`drop_gov_banned` for the full ban check.
    Default empty. Fails to the empty set on any error (no spurious bans from a
    settings glitch).
    """
    try:
        from litellm_llmrouter.settings import get_settings

        return {str(m) for m in (get_settings().governance.banned_models or [])}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("banned_model_keys lookup failed: %s", exc)
        return set()


def is_gov_banned(deployment: Dict[str, Any]) -> bool:
    """True if ``deployment`` is gov-banned — the single ban predicate.

    Banned when EITHER the arm key (``litellm_params.model``) OR the group name
    (``model_name``) is banned, by EITHER:

    - the always-on Fable 5 family ban (:data:`_ALWAYS_BANNED_TOKEN_SEQS`,
      token/segment-boundary match — RouteIQ-513e holds with zero config,
      RouteIQ-2e1f tightens it so a different terminal version token
      ``claude-fable-50`` is NOT banned while separator variants
      ``claude_fable_5`` / ``claude.fable.5`` ARE), OR
    - the operator config (:func:`banned_model_keys`, exact normalized match —
      RouteIQ-badb).

    This is the chokepoint primitive: every candidate source AND the post-
    selection backstop call it, so no strategy can route a banned arm.
    """
    arm = _normalize((deployment.get("litellm_params") or {}).get("model", ""))
    name = _normalize(deployment.get("model_name", ""))

    # Always-on family ban (token/segment boundary, zero-config).
    if _is_always_banned(arm, name):
        return True

    # Operator-configured ban (exact, normalized).
    configured = {_normalize(m) for m in banned_model_keys()}
    return arm in configured or name in configured


def drop_gov_banned(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """RouteIQ-badb + RouteIQ-513e: remove gov-banned arms BEFORE scoring.

    Removes any deployment for which :func:`is_gov_banned` is True (the always-on
    Fable 5 family ban OR the operator-configured ``banned_models``).

    Byte-stable no-op when NO candidate is banned -> returns the input list
    object unchanged. NO fail-open re-add: a banned arm must NEVER be selected
    even if it is the only candidate (compliance fail-closed-to-removal).
    """
    if not any(is_gov_banned(d) for d in candidates):
        return candidates
    return [d for d in candidates if not is_gov_banned(d)]


def filter_routable_candidates(
    router: Any, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Compose both pre-scoring filters: gov-ban (hard) then cooldown (soft).

    Gov-ban runs FIRST (a banned arm is never re-added), then cooldown
    (fail-open). Byte-stable no-op when both ``banned_models`` is empty and
    nothing is cooled down -- returns the original list object.
    """
    return drop_cooled_down(router, drop_gov_banned(candidates))
