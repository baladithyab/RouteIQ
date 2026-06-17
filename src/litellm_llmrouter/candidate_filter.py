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


# ---------------------------------------------------------------------------
# Region-aware / data-residency pre-filter (RouteIQ-60cc)
# ---------------------------------------------------------------------------
#
# Bedrock cross-region inference profiles are already used STATICALLY (each
# ``model_list`` arm pins ``litellm_params.aws_region_name``). This filter adds a
# PER-REQUEST region preference + a HARD residency constraint on top, run at the
# SAME pre-scoring seam as gov-ban + cooldown so it benefits EVERY RouteIQ
# strategy without a new strategy subclass.
#
# Semantics (settings.region_routing, default OFF -> identity):
#   - UNCONSTRAINED request: PREFER in-region; fall back to the FULL set when no
#     in-region arm exists (availability over locality -- a soft preference).
#   - CONSTRAINED (hard residency) request: NEVER route out-of-region; if no
#     in-region arm exists the set is EMPTIED so the strategy returns None /
#     fall back triggers -- a residency violation must never silently leak.
#
# A candidate's region is read from ``litellm_params.aws_region_name``. The
# request region token is normalized through ``region_map`` (token -> concrete
# AWS regions); a token absent from the map matches the candidate region
# verbatim (so an exact-region token works with zero map config).

# Source keys (besides the configured header) carrying an explicit region token.
_REGION_KEYS: tuple[str, ...] = ("region", "residency_region", "aws_region")
# Truthy keys (in the SAME source as the region) marking a HARD residency need.
_RESIDENCY_FLAG_KEYS: tuple[str, ...] = ("residency", "data_residency", "hard_region")
_TRUTHY: frozenset[str] = frozenset({"true", "1", "yes", "on"})


def _region_settings() -> Any:
    """Return ``get_settings().region_routing`` or ``None`` (fail-open) on error."""
    try:
        from litellm_llmrouter.settings import get_settings

        return get_settings().region_routing
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("region_routing settings lookup failed (fail-open): %s", exc)
        return None


def _ctx_sources(context: Any) -> List[Dict[str, Any]]:
    """Ordered context dicts to read region signals from (request_kwargs first).

    Mirrors ``TagRegexUserAgentRoutingStrategy._sources`` so the region signal is
    read from the same places (request_kwargs, then metadata).
    """
    sources: List[Dict[str, Any]] = []
    rk = getattr(context, "request_kwargs", None)
    if isinstance(rk, dict):
        sources.append(rk)
    md = getattr(context, "metadata", None)
    if isinstance(md, dict):
        sources.append(md)
    return sources


def _header_value(source: Dict[str, Any], header: str) -> str:
    """Case-insensitive header lookup inside a context source's ``headers`` dict."""
    headers = source.get("headers")
    if not isinstance(headers, dict):
        return ""
    needle = header.lower()
    for key, value in headers.items():
        if str(key).lower() == needle and value:
            return str(value)
    return ""


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUTHY


def resolve_request_region(context: Any, settings: Any) -> tuple[str, bool]:
    """Resolve ``(region_token, is_hard_residency)`` for the request.

    FIRST match wins, in order: the configured header, then a ``region`` /
    ``residency_region`` / ``aws_region`` key, scanning request_kwargs before
    metadata. The constraint is HARD when EITHER the matched source carries a
    truthy residency flag OR ``settings.hard_residency_default`` is True.

    Returns ``("", False)`` when no region token is present (a request with no
    region preference -- the filter then no-ops for it).
    """
    header = getattr(settings, "region_header", "") or ""
    hard_default = bool(getattr(settings, "hard_residency_default", False))

    for source in _ctx_sources(context):
        # 1. Header (configured name, case-insensitive).
        if header:
            token = _header_value(source, header)
            if token:
                hard = hard_default or any(
                    _is_truthy(source.get(k)) for k in _RESIDENCY_FLAG_KEYS
                )
                return token.strip().lower(), hard
        # 2. Explicit region key in the source.
        for key in _REGION_KEYS:
            raw = source.get(key)
            if raw:
                hard = hard_default or any(
                    _is_truthy(source.get(k)) for k in _RESIDENCY_FLAG_KEYS
                )
                return str(raw).strip().lower(), hard
    return "", False


def _allowed_regions(token: str, settings: Any) -> set[str]:
    """Concrete AWS regions that satisfy ``token`` (normalized, lowercased).

    Uses ``region_map[token]`` when present; otherwise the token itself
    (verbatim match against the candidate ``aws_region_name``).
    """
    region_map = getattr(settings, "region_map", None) or {}
    mapped = region_map.get(token)
    if mapped:
        return {str(r).strip().lower() for r in mapped if str(r).strip()}
    return {token} if token else set()


def _candidate_region(deployment: Dict[str, Any]) -> str:
    """The candidate's AWS region (normalized) from ``litellm_params.aws_region_name``."""
    params = deployment.get("litellm_params") or {}
    return str(params.get("aws_region_name", "") or "").strip().lower()


def drop_out_of_region(
    context: Any, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """RouteIQ-60cc: per-request region-aware / data-residency pre-filter.

    Default OFF / identity: returns the input list OBJECT unchanged when the
    feature is disabled, the context is absent, or the request carries no region
    token (byte-stable -- no allocation, no reordering).

    When a region token IS present:
      - HARD residency: keep ONLY in-region candidates; if none, return ``[]``
        (the strategy returns None / falls back -- never an out-of-region leak).
      - SOFT preference: keep in-region candidates; if none, return the ORIGINAL
        list object unchanged (availability over locality).
    """
    if context is None:
        return candidates
    settings = _region_settings()
    if settings is None or not getattr(settings, "enabled", False):
        return candidates

    token, hard = resolve_request_region(context, settings)
    if not token:
        return candidates  # no region preference on this request -> no-op

    allowed = _allowed_regions(token, settings)
    if not allowed:
        return candidates  # token resolved to nothing -> no-op (fail-open)

    in_region = [d for d in candidates if _candidate_region(d) in allowed]
    if in_region:
        # Byte-stable when ALL candidates are already in-region (same membership
        # AND order) -> return the original object so downstream stays identical.
        if len(in_region) == len(candidates):
            return candidates
        return in_region
    # No in-region candidate.
    if hard:
        return []  # residency fail-closed: never route out-of-region
    return candidates  # soft: availability over locality


# ---------------------------------------------------------------------------
# Capability-tier floor (RouteIQ-8e37, intelligence-first routing)
# ---------------------------------------------------------------------------
#
# Makes the RL router pick the most CAPABLE model for the task, not the cheapest,
# by enforcing a per-request capability-tier FLOOR at the SAME pre-scoring seam
# as gov-ban + cooldown + region. Each model carries an ordered tier (``fast`` <
# ``advanced`` < ``expert``); the request difficulty (warm centroid classifier +
# reasoning markers) maps to a REQUIRED minimum tier. A request that needs
# ``>=`` a tier DROPS sub-tier models BEFORE scoring, so a reasoning task NEVER
# scores a sub-tier model — even in fallback / exploration.
#
# UNLIKE region/residency, capability is a PREFERENCE floor, not a hard deny:
# in the default SOFT mode an empty floor result DEGRADES GRACEFULLY (logs +
# returns the full set) rather than emptying the set / 500ing. ``strict`` mode
# fails closed-to-empty (the strategy then yields None / triggers a fallback).
#
# Default OFF / identity: when ``settings.capability_routing.enabled`` is False,
# or no context is supplied, the filter is a byte-stable no-op.


def _capability_settings() -> Any:
    """Return ``get_settings().capability_routing`` or ``None`` (fail-open)."""
    try:
        from litellm_llmrouter.settings import get_settings

        return get_settings().capability_routing
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("capability_routing settings lookup failed (fail-open): %s", exc)
        return None


def _required_min_tier(difficulty: str, settings: Any) -> str:
    """Map request ``difficulty`` -> the required MINIMUM capability tier.

    ``complex`` (centroid complex tier OR reasoning markers) -> the configured
    ``complex_min_tier`` (default ``expert``). Anything else (``simple``) -> the
    configured ``simple_min_tier`` (default ``fast`` — the lowest floor, so a
    simple task accepts any tier).
    """
    if difficulty == "complex":
        return str(getattr(settings, "complex_min_tier", "expert") or "expert")
    return str(getattr(settings, "simple_min_tier", "fast") or "fast")


def _tier_rank(tier: str, tier_order: List[str]) -> int:
    """Index of ``tier`` in ``tier_order``; an unknown tier ranks at the bottom.

    An unknown tier returning ``-1`` means it is treated as BELOW every known
    tier — but the only producer of tiers (``get_model_tier``) maps unknown
    models to the safe ``default_tier`` (a known tier), so a candidate never
    reaches here with an unknown tier in practice.
    """
    try:
        return tier_order.index(tier)
    except ValueError:
        return -1


def drop_below_capability_tier(
    context: Any, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """RouteIQ-8e37: per-request capability-tier FLOOR pre-filter.

    Default OFF / identity: returns the input list OBJECT unchanged when the
    feature is disabled or the context is absent (byte-stable — no allocation,
    no reordering).

    When ENABLED, resolves the request difficulty (warm centroid classifier +
    reasoning markers — never a cold load) -> a required minimum tier, then DROPS
    every candidate whose tier ranks BELOW it. Behaviour when the floor empties
    the set:

      - SOFT (default): DEGRADE GRACEFULLY — log + return the ORIGINAL list
        object unchanged (capability is a preference floor; never 500 on empty).
      - STRICT: return ``[]`` (the strategy yields None / triggers a fallback).

    Byte-stable when every candidate already meets the floor (same membership AND
    order) -> returns the original object.
    """
    if context is None:
        return candidates
    settings = _capability_settings()
    if settings is None or not getattr(settings, "enabled", False):
        return candidates
    if not candidates:
        return candidates

    try:
        from litellm_llmrouter.centroid_routing import (
            get_model_tier,
            resolve_request_difficulty,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("capability tier deps unavailable (fail-open): %s", exc)
        return candidates

    difficulty = resolve_request_difficulty(context)
    required = _required_min_tier(difficulty, settings)
    tier_order = list(
        getattr(settings, "tier_order", None) or ["fast", "advanced", "expert"]
    )
    required_rank = _tier_rank(required, tier_order)
    # A required tier that is the LOWEST in the order (e.g. simple->fast) can
    # never drop anything -> byte-stable no-op without per-candidate work.
    if required_rank <= 0:
        return candidates

    model_tiers = dict(getattr(settings, "model_tiers", None) or {})
    default_tier = str(getattr(settings, "default_tier", "advanced") or "advanced")

    def _arm_name(dep: Dict[str, Any]) -> str:
        return str(
            (dep.get("litellm_params") or {}).get("model", "")
            or dep.get("model_name", "")
        )

    capable = [
        d
        for d in candidates
        if _tier_rank(
            get_model_tier(_arm_name(d), model_tiers, default_tier), tier_order
        )
        >= required_rank
    ]

    if capable:
        # Byte-stable when EVERY candidate already meets the floor.
        if len(capable) == len(candidates):
            return candidates
        return capable

    # No candidate meets the floor.
    if getattr(settings, "strict", False):
        logger.info(
            "capability floor (strict): no candidate meets tier >= %r for a "
            "%s request -> emptying set (None / fallback)",
            required,
            difficulty,
        )
        return []
    logger.info(
        "capability floor (soft): no candidate meets tier >= %r for a %s "
        "request -> degrading to full set (availability over capability)",
        required,
        difficulty,
    )
    return candidates


def filter_routable_candidates(
    router: Any,
    candidates: List[Dict[str, Any]],
    context: Any = None,
) -> List[Dict[str, Any]]:
    """Compose the pre-scoring filters: gov-ban, cooldown, capability, region.

    Order (locked): gov-ban FIRST (a banned arm is never re-added), then cooldown
    (fail-open), then the per-request capability-tier FLOOR (RouteIQ-8e37), then
    the per-request region filter (RouteIQ-60cc).

    Capability runs BEFORE region so that the intelligence floor narrows to the
    capable set first; region (hard residency) then applies its request-scoped
    constraint LAST, since its fail-closed empty result is the correct terminal
    outcome (a residency violation must not be re-added). The capability floor's
    SOFT degrade re-adds only from its OWN already-(gov-ban+cooldown)-filtered
    input, so it can never re-introduce a banned / cooled-down arm; a STRICT
    empty likewise flows through region unchanged.

    ``context`` is OPTIONAL: when ``None`` (the legacy call shape) BOTH the
    capability and region filters no-op, so every existing caller is byte-stable.
    When supplied AND the respective feature is enabled, that filter activates.

    Byte-stable no-op when ``banned_models`` is empty, nothing is cooled down,
    and (capability disabled OR no tier floor applies) and (region disabled OR no
    per-request region token) -- returns the original list object.
    """
    routable = drop_cooled_down(router, drop_gov_banned(candidates))
    routable = drop_below_capability_tier(context, routable)
    return drop_out_of_region(context, routable)
