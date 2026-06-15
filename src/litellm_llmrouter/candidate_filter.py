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
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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

    Default empty -> a byte-stable no-op for :func:`drop_gov_banned`. Fails to
    the empty set on any error (no spurious bans from a settings glitch).
    """
    try:
        from litellm_llmrouter.settings import get_settings

        return {str(m) for m in (get_settings().governance.banned_models or [])}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("banned_model_keys lookup failed: %s", exc)
        return set()


def drop_gov_banned(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """RouteIQ-badb: remove gov-banned arms BEFORE scoring.

    Matches a banned key against BOTH ``litellm_params.model`` (the bandit arm
    key / ``_arm_key``) AND ``model_name`` (the group name), so a config of
    ``["bedrock/global.anthropic.claude-fable-5"]`` removes that arm.

    Default-empty config -> returns the input list object unchanged (byte
    stable). NO fail-open re-add: a banned arm must NEVER be selected even if it
    is the only candidate (compliance fail-closed-to-removal).
    """
    banned = banned_model_keys()
    if not banned:
        return candidates
    out: List[Dict[str, Any]] = []
    for d in candidates:
        arm = str((d.get("litellm_params") or {}).get("model", ""))
        name = str(d.get("model_name", ""))
        if arm in banned or name in banned:
            continue
        out.append(d)
    return out


def filter_routable_candidates(
    router: Any, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Compose both pre-scoring filters: gov-ban (hard) then cooldown (soft).

    Gov-ban runs FIRST (a banned arm is never re-added), then cooldown
    (fail-open). Byte-stable no-op when both ``banned_models`` is empty and
    nothing is cooled down -- returns the original list object.
    """
    return drop_cooled_down(router, drop_gov_banned(candidates))
