"""
Upstream LiteLLM Router Delegation (RouteIQ-8539)
=================================================

Upstream LiteLLM ships its OWN online-bandit / quality-feedback routing modes
that live ON the LiteLLM ``Router`` instance, NOT in RouteIQ's strategy family:

- ``adaptive_router`` -- a per-(request_type, model) Beta-Bernoulli Thompson
  bandit (``Router.adaptive_routers``) that learns from conversation signals and
  buffers bandit-cell + session deltas in an in-memory
  ``AdaptiveRouterUpdateQueue`` (``router.queue``), drained to Postgres by a
  background flusher.
- ``quality_router`` -- a quality-tier router (``Router.quality_routers``).
- ``auto_router`` -- a semantic (embedding) router (``Router.auto_routers``).

Each is configured per-deployment in the LiteLLM ``model_list`` via the
``auto_router/...`` model prefix. This module is the RouteIQ-side GLUE that:

1. lets an operator SELECT one of those upstream modes as a delegation target
   (alongside the RouteIQ ML strategies), and
2. FLUSHES the selected adaptive router's update queue to the durable store on
   the MLOps feedback cadence -- closing the upstream router's online-learning
   loop through RouteIQ's existing FEEDBACK arm.

Settings-gated under ``settings.mlops.upstream_router`` and DEFAULT OFF: with
``enabled=False`` this module never touches the LiteLLM Router, so importing it
or wiring it is a byte-stable no-op. It NEVER raises into the request/feedback
path -- a missing upstream router or durable client degrades to a no-op.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "UPSTREAM_ROUTER_MODES",
    "UpstreamRouterDelegate",
    "FlushResult",
    "get_upstream_router_delegate",
    "reset_upstream_router_delegate",
    "wire_upstream_router_flush",
]

# Map a RouteIQ-facing mode name to the attribute on the LiteLLM ``Router`` that
# holds the per-router-name dict for that family.
UPSTREAM_ROUTER_MODES: Dict[str, str] = {
    "adaptive": "adaptive_routers",
    "quality": "quality_routers",
    "auto": "auto_routers",
}


@dataclass
class FlushResult:
    """Outcome of one upstream-queue flush."""

    flushed: bool = False
    """True iff a queue was found and a drain was attempted."""

    router_name: str = ""
    """The upstream router_name that was flushed."""

    state_rows: int = 0
    """Number of bandit-cell state rows flushed to the durable store."""

    session_rows: int = 0
    """Number of session-snapshot rows flushed to the durable store."""

    reason: str = ""
    """Why the flush was skipped, when ``flushed`` is False."""


class UpstreamRouterDelegate:
    """Selects + flushes an upstream LiteLLM router (RouteIQ-8539).

    Holds the operator's chosen ``mode`` (adaptive/quality/auto) and optional
    ``router_name``. Resolves the concrete upstream router off the live LiteLLM
    ``Router`` lazily (so a router that is hot-reloaded after startup is still
    found), and drives ``AdaptiveRouterUpdateQueue`` flushes through the durable
    prisma client.

    Args:
        mode: Upstream family to delegate to (key of ``UPSTREAM_ROUTER_MODES``).
        router_name: Specific upstream router_name, or empty for the first.
        flush_queue: Whether the feedback cadence flushes the update queue.
    """

    def __init__(
        self,
        *,
        mode: str = "adaptive",
        router_name: str = "",
        flush_queue: bool = True,
    ) -> None:
        if mode not in UPSTREAM_ROUTER_MODES:
            mode = "adaptive"
        self.mode = mode
        self.router_name = router_name
        self.flush_queue = flush_queue

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _registry_attr(self) -> str:
        return UPSTREAM_ROUTER_MODES[self.mode]

    def resolve_router(self, llm_router: Any = None) -> Optional[Any]:
        """Resolve the concrete upstream router instance, or None.

        Reads the per-name dict (``Router.adaptive_routers`` etc.) off the live
        LiteLLM Router. Selects ``router_name`` when set, else the first
        registered router of the mode. Returns None when none is configured
        (the operator enabled delegation but never added the ``auto_router/...``
        deployment) -- a no-op rather than an error.
        """
        router = llm_router if llm_router is not None else self._litellm_router()
        if router is None:
            return None
        registry = getattr(router, self._registry_attr(), None)
        if not isinstance(registry, dict) or not registry:
            return None
        if self.router_name:
            return registry.get(self.router_name)
        # First registered router of this mode (insertion order).
        first_name = next(iter(registry))
        return registry.get(first_name)

    def is_available(self, llm_router: Any = None) -> bool:
        """True iff a concrete upstream router of the selected mode is present."""
        return self.resolve_router(llm_router) is not None

    @staticmethod
    def _litellm_router() -> Optional[Any]:
        """Read the process-wide LiteLLM Router (None when not initialized)."""
        try:
            from litellm.proxy.proxy_server import llm_router

            return llm_router
        except Exception:  # pragma: no cover - defensive
            return None

    @staticmethod
    def _prisma_client() -> Optional[Any]:
        """Read the proxy's durable prisma client (None when not initialized)."""
        try:
            from litellm.proxy.proxy_server import prisma_client

            return prisma_client
        except Exception:  # pragma: no cover - defensive
            return None

    # ------------------------------------------------------------------
    # Queue flush (durable-store persistence of online updates)
    # ------------------------------------------------------------------

    async def flush(
        self,
        *,
        llm_router: Any = None,
        prisma_client: Any = None,
    ) -> FlushResult:
        """Drain the selected adaptive router's update queue to the durable store.

        The adaptive router buffers bandit-cell + session deltas in an in-memory
        ``AdaptiveRouterUpdateQueue`` (``router.queue``); this drains both halves
        to Postgres via the prisma client, exactly as the upstream proxy flusher
        does, but on RouteIQ's MLOps cadence. Quality/auto routers have no queue,
        so the flush short-circuits with a reason.

        Fully fail-open: a missing router / queue / client returns a non-flushed
        result and never raises.
        """
        if not self.flush_queue:
            return FlushResult(reason="flush_disabled")

        router = self.resolve_router(llm_router)
        if router is None:
            return FlushResult(reason="no_upstream_router")

        queue = getattr(router, "queue", None)
        if queue is None:
            return FlushResult(
                reason="mode_has_no_queue",
                router_name=getattr(router, "router_name", "") or "",
            )

        client = prisma_client if prisma_client is not None else self._prisma_client()
        if client is None:
            return FlushResult(
                reason="no_durable_client",
                router_name=getattr(router, "router_name", "") or "",
            )

        result = FlushResult(
            flushed=True, router_name=getattr(router, "router_name", "") or ""
        )
        try:
            flush_state = getattr(queue, "flush_state_to_db", None)
            if callable(flush_state):
                result.state_rows = int(await flush_state(client) or 0)
        except Exception as e:  # pragma: no cover - durable flush must not raise
            logger.warning("Upstream adaptive-router state flush failed: %s", e)
        try:
            flush_session = getattr(queue, "flush_session_to_db", None)
            if callable(flush_session):
                result.session_rows = int(await flush_session(client) or 0)
        except Exception as e:  # pragma: no cover - durable flush must not raise
            logger.warning("Upstream adaptive-router session flush failed: %s", e)

        if result.flushed:
            logger.debug(
                "Flushed upstream router %r queue: state=%d session=%d",
                result.router_name,
                result.state_rows,
                result.session_rows,
            )
        return result


# ---------------------------------------------------------------------------
# Settings-gated singleton
# ---------------------------------------------------------------------------

_delegate: Optional[UpstreamRouterDelegate] = None


def get_upstream_router_delegate() -> Optional[UpstreamRouterDelegate]:
    """Get the upstream-router delegate singleton, or None when disabled.

    Returns None unless ``settings.mlops.upstream_router.enabled`` is true.
    Settings read failures degrade to disabled (None), so a misconfig never
    silently enables delegation.
    """
    global _delegate
    if _delegate is not None:
        return _delegate
    cfg = _upstream_settings()
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    _delegate = UpstreamRouterDelegate(
        mode=getattr(cfg, "mode", "adaptive"),
        router_name=getattr(cfg, "router_name", "") or "",
        flush_queue=bool(getattr(cfg, "flush_queue", True)),
    )
    return _delegate


def _upstream_settings():  # type: ignore[no-untyped-def]
    """Read ``settings.mlops.upstream_router`` (None on any failure)."""
    try:
        from litellm_llmrouter.settings import get_settings

        mlops = getattr(get_settings(), "mlops", None)
        return getattr(mlops, "upstream_router", None) if mlops is not None else None
    except Exception:  # pragma: no cover - defensive
        return None


def reset_upstream_router_delegate() -> None:
    """Reset the singleton (MUST be called in the autouse test fixture)."""
    global _delegate
    _delegate = None


# ---------------------------------------------------------------------------
# FEEDBACK-arm wiring: flush the upstream queue on the eval cadence
# ---------------------------------------------------------------------------


@dataclass
class _FlushSubscription:
    """Holds the async callback subscribed into the eval pipeline."""

    delegate: UpstreamRouterDelegate
    results: List[FlushResult] = field(default_factory=list)


async def _flush_on_feedback(model_qualities: Dict[str, float]) -> None:
    """Eval-pipeline feedback callback that flushes the upstream queue.

    Signature matches ``EvalPipeline.feedback_callbacks`` (``{model: quality}``);
    the aggregate is unused -- this callback exists to flush the upstream
    adaptive router's update queue to the durable store every time the eval loop
    pushes feedback. A no-op when delegation is disabled or no queue exists.
    Never raises.
    """
    delegate = get_upstream_router_delegate()
    if delegate is None:
        return
    try:
        await delegate.flush()
    except Exception as e:  # pragma: no cover - feedback callback must not raise
        logger.debug("Upstream router flush on feedback failed: %s", e)


def wire_upstream_router_flush(
    *,
    eval_pipeline: Any = None,
    force: bool = False,
) -> bool:
    """Subscribe the upstream-queue flush into the eval-pipeline FEEDBACK arm.

    On each ``push_feedback()`` the selected upstream adaptive router's update
    queue is drained to the durable store. Gated on
    ``settings.mlops.upstream_router.enabled`` (pass ``force=True`` to bypass in
    tests). Idempotent and never raises -- a wiring failure must not block
    startup.

    Returns:
        True if the flush callback was newly subscribed into a live eval
        pipeline; False otherwise (disabled, pipeline absent, or already wired).
    """
    if not force and get_upstream_router_delegate() is None:
        return False

    pipeline = eval_pipeline
    if pipeline is None:
        try:
            from litellm_llmrouter.eval_pipeline import get_eval_pipeline

            pipeline = get_eval_pipeline()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Upstream router: eval pipeline unavailable: %s", e)
            pipeline = None
    if pipeline is None:
        return False

    add_cb = getattr(pipeline, "add_feedback_callback", None)
    if not callable(add_cb):
        return False
    try:
        added = add_cb(_flush_on_feedback)
        if added:
            logger.info("Upstream router: subscribed queue-flush callback to eval loop")
        return bool(added)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Upstream router: flush wiring failed: %s", e)
        return False
