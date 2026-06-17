"""Cost-cascade mode-(a) RETRY consumer (RouteIQ-3ff5).

``CostCascadeRoutingStrategy`` (mode (a)) returns the CHEAPEST capable rung and
attaches the ordered escalation ladder under
``deployment["metadata"]["routeiq_cascade"]``. That ladder was dark in
production: nothing read it, so a low-quality cheap-rung answer was returned as
final. This module closes mode (a): it issues the cheapest-rung completion,
scores the response confidence, and -- when confidence is BELOW the threshold and
a stronger rung exists -- RE-ISSUES the request pinned to the next rung, climbing
up to ``max_rungs`` rungs.

Gated, default OFF (``cost_cascade.retry_consumer_enabled``): with it off there is
NO wrapper and a single cheapest-rung completion is returned unchanged
(byte-stable). The consumer makes no decisions of its own about which models
exist -- it only follows the ladder the strategy already computed, so it stays a
pure consumer of the existing mechanism.

No side effects on import.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "extract_response_confidence",
    "CascadeRetryConsumer",
    "get_cascade_retry_consumer",
    "reset_cascade_retry_consumer",
]


def extract_response_confidence(response: Any) -> Optional[float]:
    """Best-effort confidence in ``[0, 1]`` extracted from an LLM response.

    Resolution order (first hit wins):
    1. an explicit ``response["routeiq_confidence"]`` / attribute (a caller- or
       judge-provided score);
    2. the mean per-token probability derived from ``choices[0].logprobs`` (if
       the provider returned logprobs);
    3. ``None`` when no confidence signal is available (the consumer treats
       ``None`` as "no signal" and does NOT escalate on it).
    """
    # 1. explicit score on the response object/dict
    explicit = _get(response, "routeiq_confidence")
    if explicit is not None:
        try:
            return max(0.0, min(1.0, float(explicit)))
        except (TypeError, ValueError):
            pass

    # 2. derive from logprobs of the first choice, if present
    try:
        choices = _get(response, "choices") or []
        if choices:
            logprobs = _get(choices[0], "logprobs")
            content = _get(logprobs, "content") if logprobs is not None else None
            probs = []
            for tok in content or []:
                lp = _get(tok, "logprob")
                if lp is not None:
                    import math

                    probs.append(math.exp(float(lp)))
            if probs:
                return max(0.0, min(1.0, sum(probs) / len(probs)))
    except Exception:  # never let confidence extraction raise
        pass

    return None


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict (item) or object (attribute)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class CascadeRetryConsumer:
    """Climbs the cost-cascade ladder by re-issuing low-confidence completions.

    The consumer is intentionally thin: it asks a *selector* for the cheapest
    rung (and its ladder), issues a completion via a *completion* callable, scores
    confidence, and -- if below threshold and a stronger rung exists -- re-issues
    pinned to the next rung. Both collaborators are injected so the hot-path wires
    real ones (the live router) while tests inject fakes.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.7,
        max_rungs: int = 4,
    ) -> None:
        self._confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self._max_rungs = max(1, int(max_rungs))

    async def complete_with_cascade(
        self,
        *,
        select_rung: Callable[[int], Optional[dict]],
        complete: Callable[[dict, int], Any],
        confidence_fn: Callable[[Any], Optional[float]] = extract_response_confidence,
    ) -> Any:
        """Issue a cascade completion, climbing rungs on low confidence.

        Args:
            select_rung: ``rung_index -> deployment`` for the requested model.
                Rung 0 is the cheapest; the returned deployment carries the
                ``metadata['routeiq_cascade']['ladder']`` the consumer follows.
                Returns ``None`` when no rung exists.
            complete: ``(deployment, rung_index) -> response`` -- issues the
                actual completion against the chosen rung's model.
            confidence_fn: maps a response to a confidence in ``[0, 1]`` (or
                ``None`` for "no signal" -> no escalation).

        Returns:
            The final (possibly escalated) response. Never raises on escalation
            logic -- a re-issue failure falls back to the last good response.
        """
        rung_index = 0
        deployment = select_rung(rung_index)
        if deployment is None:
            return None

        ladder = _ladder_of(deployment)
        # The number of climbable rungs is bounded by the ladder length AND the
        # configured cap.
        max_index = min(self._max_rungs, len(ladder) or 1) - 1

        response = await _maybe_await(complete(deployment, rung_index))
        while rung_index < max_index:
            confidence = confidence_fn(response)
            if confidence is None or confidence >= self._confidence_threshold:
                break  # confident enough (or no signal) -> stop climbing
            next_index = rung_index + 1
            next_dep = select_rung(next_index)
            if next_dep is None:
                break
            logger.info(
                "Cascade retry: confidence %.3f < %.3f -> escalating to rung %d",
                confidence,
                self._confidence_threshold,
                next_index,
            )
            try:
                escalated = await _maybe_await(complete(next_dep, next_index))
            except Exception as exc:  # re-issue failure -> keep last good response
                logger.warning(
                    "Cascade retry re-issue failed at rung %d: %s", next_index, exc
                )
                break
            response = escalated
            deployment = next_dep
            rung_index = next_index

        return response


def _ladder_of(deployment: dict) -> list:
    """Extract the ordered ladder rung descriptors from a deployment, or []."""
    meta = deployment.get("metadata") or {}
    cascade = meta.get("routeiq_cascade") or {}
    return list(cascade.get("ladder") or [])


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, else return it as-is."""
    if hasattr(value, "__await__"):
        return await value
    return value


# ---------------------------------------------------------------------------
# Singleton (gated construction; reset_* per the RouteIQ convention)
# ---------------------------------------------------------------------------

_consumer: Optional[CascadeRetryConsumer] = None


def get_cascade_retry_consumer() -> Optional[CascadeRetryConsumer]:
    """Return the process-wide consumer, or ``None`` when disabled (default).

    Constructed from :class:`~litellm_llmrouter.settings.CostCascadeSettings`
    only when BOTH ``enabled`` and ``retry_consumer_enabled`` are True; otherwise
    returns ``None`` so the caller takes the byte-stable single-shot path.
    """
    global _consumer
    if _consumer is not None:
        return _consumer
    try:
        from litellm_llmrouter.settings import get_settings

        cfg = get_settings().cost_cascade
    except Exception:
        return None
    if not (
        getattr(cfg, "enabled", False) and getattr(cfg, "retry_consumer_enabled", False)
    ):
        return None
    _consumer = CascadeRetryConsumer(
        confidence_threshold=getattr(cfg, "confidence_threshold", 0.7),
        max_rungs=getattr(cfg, "max_rungs", 4),
    )
    return _consumer


def reset_cascade_retry_consumer() -> None:
    """Reset the singleton (tests; ``autouse`` reset fixture)."""
    global _consumer
    _consumer = None
