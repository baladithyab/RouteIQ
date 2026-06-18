"""
Strategy-Agnostic MLOps Loop Glue
=================================

The ``MLOpsCoordinator`` is the STRATEGY-AGNOSTIC machinery that closes the
learning loop. It subscribes to ``EvalPipeline.feedback_callbacks`` and fans
``(model, quality)`` out to ANY registered adapter that declares ``learns`` (by
calling ``adapter.update(...)``), and drives one-time / continuous artifact
reloads. It is NOT bandit-coupled: the Kumaraswamy-Thompson bandit is one
consumer; a stub classifier with an ``update()`` is fanned out to identically.

Per-strategy ``train_mode``:
- ``one_time``  — fit once, freeze; does NOT subscribe to the feedback fan-out.
- ``continuous`` — subscribes to feedback; updated on every aggregate.

Design reference:
``docs/architecture/aws-rearchitecture/40-pluggable-routing-and-mlops.md`` §3.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from litellm_llmrouter.adapters.contract import (
    TRAIN_MODE_CONTINUOUS,
    AdapterManifest,
    ArtifactRef,
    RoutingFeedback,
)

logger = logging.getLogger(__name__)


class MLOpsCoordinator:
    """Generalized feedback fan-out + artifact reload driver for adapters.

    Holds a registry of learning adapters keyed by name. ``on_aggregate_feedback``
    is the callback to register into ``EvalPipeline.feedback_callbacks``.
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, Any] = {}
        self._manifests: Dict[str, AdapterManifest] = {}
        self._lock = threading.RLock()
        # The eval pipeline this coordinator was wired to (RouteIQ-e019). Set by
        # ``wire_mlops_feedback_loop`` so ``_bucket_qualities`` reads the
        # per-(bucket, model) aggregate from the SAME pipeline that fires the
        # feedback callback (the firing pipeline is authoritative; the module
        # singleton may differ in tests). Falls back to the singleton when unset.
        self._eval_pipeline: Any = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_learning_adapter(self, name: str, adapter: Any) -> bool:
        """Register an adapter for the feedback fan-out.

        Only registered if the adapter declares ``learns=True`` and exposes an
        ``update`` method. Returns False otherwise (silently skipped).
        """
        manifest = self._safe_manifest(adapter)
        if manifest is None or not manifest.learns:
            return False
        if not (hasattr(adapter, "update") or hasattr(adapter, "update_from_feedback")):
            return False
        with self._lock:
            self._adapters[name] = adapter
            self._manifests[name] = manifest
        logger.info(
            "MLOps: registered learning adapter %r (train_mode=%s)",
            name,
            manifest.train_mode,
        )
        return True

    def unregister(self, name: str) -> bool:
        with self._lock:
            existed = name in self._adapters
            self._adapters.pop(name, None)
            self._manifests.pop(name, None)
        return existed

    def list_adapters(self) -> List[str]:
        with self._lock:
            return list(self._adapters.keys())

    def discover_learning_adapters_from_registry(self) -> List[str]:
        """Register every learning strategy currently in the routing registry.

        STRATEGY-AGNOSTIC: scans the routing registry and registers any strategy
        whose ``declare_capabilities()`` reports ``learns=True`` (the bandit, a
        stub classifier, the personalized router, or any future adapter). This is
        how active strategies opt into the feedback fan-out without the eval
        pipeline or this coordinator hard-coding their names.

        Strategies that do not expose ``declare_capabilities`` (legacy ABC-only
        strategies) are skipped silently — they simply don't participate.

        Returns:
            The list of strategy names newly registered as learning adapters.
        """
        registered: List[str] = []
        try:
            from litellm_llmrouter.strategy_registry import get_routing_registry

            registry = get_routing_registry()
            names = registry.list_strategies()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("MLOps: registry scan failed: %s", e)
            return registered

        for name in names:
            if name in self._adapters:
                continue
            try:
                strategy = registry.get(name)
            except Exception:
                continue
            if strategy is None:
                continue
            if self.register_learning_adapter(name, strategy):
                registered.append(name)
        return registered

    @staticmethod
    def _safe_manifest(adapter: Any) -> Optional[AdapterManifest]:
        declare = getattr(adapter, "declare_capabilities", None)
        if not callable(declare):
            return None
        try:
            manifest = declare()
            return manifest if isinstance(manifest, AdapterManifest) else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Feedback fan-out (the EvalPipeline callback)
    # ------------------------------------------------------------------

    def on_aggregate_feedback(self, model_qualities: Dict[str, float]) -> None:
        """Fan aggregate quality scores out to every CONTINUOUS learning adapter.

        ``model_qualities`` is ``{model: quality}`` where ``quality`` is in
        ``[0, 1]`` (the eval pipeline's ``ModelQualityTracker`` scale). It is
        converted to the ``[-1, 1]`` feedback ``score`` before dispatch.

        This is the signature ``EvalPipeline`` invokes its feedback callbacks
        with. It is generic — works for the bandit AND a stub classifier.

        Per-bucket attribution (RouteIQ-e019): the bandit keys its posteriors by
        ``(bucket, model)`` and its hot path READS ``(_bucket(context), model)``.
        A per-model-only fan-out (``RoutingFeedback`` with no ``bucket``) makes
        the bandit ``update`` resolve ``bucket or ... or "default"`` and land EVERY
        eval-derived update on the inert ``("default", model)`` posterior the
        router never samples. To fix that, when the eval pipeline tracked quality
        per ``(bandit_bucket, model)`` (a decision-time bucket was recovered at
        COLLECT), this fans out one ``RoutingFeedback(model, score, bucket=bucket)``
        per ``(bucket, model)`` so ``update`` keys the SAME posterior the router
        samples. Models with NO per-bucket data fall back to the byte-stable
        per-model dispatch (``bucket=None``) so non-bandit adapters and the prior
        behaviour are unchanged.
        """
        if not model_qualities:
            return
        with self._lock:
            targets = [
                (name, self._adapters[name], self._manifests[name])
                for name in self._adapters
            ]

        # Pull the per-(bucket, model) aggregate from the live eval pipeline (the
        # same seam ``_mlops_aggregate_feedback`` uses to reach per-strategy
        # quality). The build is grouped per-model so a model with per-bucket data
        # is dispatched ONLY per-bucket (no double-counting) and a model without
        # falls back to a single per-model dispatch.
        bucket_qualities = self._bucket_qualities_for_dispatch()
        per_model_buckets: Dict[str, list] = {}
        for (bucket, model), quality in bucket_qualities.items():
            per_model_buckets.setdefault(model, []).append((bucket, quality))

        for name, adapter, manifest in targets:
            if manifest.train_mode != TRAIN_MODE_CONTINUOUS:
                continue
            for model, quality in model_qualities.items():
                buckets = per_model_buckets.get(model)
                if buckets:
                    # Per-(bucket, model): key the posterior the router samples.
                    for bucket, b_quality in buckets:
                        score = b_quality * 2.0 - 1.0  # [0,1] -> [-1,1]
                        feedback = RoutingFeedback(
                            model=model,
                            score=score,
                            bucket=bucket,
                            metadata={"source": "eval_aggregate"},
                        )
                        self._dispatch_update(name, adapter, feedback)
                else:
                    # No bandit bucket for this model -> byte-stable per-model path.
                    score = quality * 2.0 - 1.0  # [0,1] -> [-1,1]
                    feedback = RoutingFeedback(
                        model=model, score=score, metadata={"source": "eval_aggregate"}
                    )
                    self._dispatch_update(name, adapter, feedback)

    def _bucket_qualities_for_dispatch(self) -> Dict[tuple, float]:
        """Per-``(bandit_bucket, model)`` aggregate from the live eval pipeline.

        Prefers the pipeline this coordinator was wired to (``self._eval_pipeline``
        — the pipeline that actually fires the feedback callback), then the module
        singleton. Best-effort: returns ``{}`` when no pipeline is available /
        disabled or it carried no per-bucket samples, in which case
        ``on_aggregate_feedback`` degrades to the byte-stable per-model fan-out.
        Never raises.
        """
        pipeline = self._eval_pipeline
        if pipeline is None:
            try:
                from litellm_llmrouter.eval_pipeline import get_eval_pipeline

                pipeline = get_eval_pipeline()
            except Exception:  # pragma: no cover - defensive
                pipeline = None
        if pipeline is None:
            return {}
        try:
            tracker = getattr(pipeline, "tracker", None)
            getter = getattr(tracker, "get_all_bucket_qualities", None)
            if not callable(getter):
                return {}
            data = getter()
            return data if isinstance(data, dict) else {}
        except Exception:  # pragma: no cover - defensive, never break the loop
            return {}

    @staticmethod
    def _dispatch_update(name: str, adapter: Any, feedback: RoutingFeedback) -> None:
        """Call the adapter's update hook, preferring the typed entry."""
        try:
            update_fb = getattr(adapter, "update_from_feedback", None)
            if callable(update_fb):
                update_fb(feedback)
                return
            update = getattr(adapter, "update", None)
            if callable(update):
                update(
                    feedback.model,
                    feedback.score,
                    request_id=feedback.request_id,
                    bucket=feedback.bucket,
                )
        except TypeError:
            # Adapter has a simpler update signature — fall back to positional.
            try:
                adapter.update(feedback.model, feedback.score)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("MLOps update for %r failed: %s", name, e)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("MLOps update for %r failed: %s", name, e)

    # ------------------------------------------------------------------
    # Artifact reload (one-time / continuous)
    # ------------------------------------------------------------------

    def apply_artifact(
        self, name: str, ref: ArtifactRef, *, verify: bool = True
    ) -> bool:
        """Verify (optional) then reload a trained artifact into an adapter."""
        with self._lock:
            adapter = self._adapters.get(name)
        if adapter is None:
            return False

        if verify and ref.path:
            try:
                from litellm_llmrouter.model_artifacts import get_artifact_verifier

                verifier = get_artifact_verifier()
                verify_fn = getattr(verifier, "verify_artifact", None)
                if callable(verify_fn):
                    ok = verify_fn(ref.path)
                    valid = ok[0] if isinstance(ok, tuple) else bool(ok)
                    if not valid:
                        logger.warning(
                            "MLOps: artifact verification failed for %r", name
                        )
                        return False
            except Exception as e:
                logger.debug("MLOps: artifact verification skipped (%s)", e)

        reload_fn = getattr(adapter, "reload", None) or getattr(
            adapter, "load_artifact", None
        )
        if not callable(reload_fn):
            return False
        try:
            return bool(reload_fn(ref))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("MLOps reload for %r failed: %s", name, e)
            return False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_coordinator: Optional[MLOpsCoordinator] = None
_coordinator_lock = threading.Lock()


def get_mlops_coordinator() -> MLOpsCoordinator:
    """Get or create the MLOps coordinator singleton."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            _coordinator = MLOpsCoordinator()
        return _coordinator


def reset_mlops_coordinator() -> None:
    """Reset the singleton (MUST be called in the autouse test fixture)."""
    global _coordinator
    with _coordinator_lock:
        _coordinator = None


# ---------------------------------------------------------------------------
# FEEDBACK-arm wiring (eval pipeline -> coordinator -> learning adapters)
# ---------------------------------------------------------------------------


def wire_mlops_feedback_loop(
    *,
    coordinator: Optional[MLOpsCoordinator] = None,
    eval_pipeline: Any = None,
    force: bool = False,
) -> bool:
    """Wire the eval-pipeline FEEDBACK arm into the MLOps coordinator.

    This is the strategy-agnostic glue that closes the loop end-to-end:
    COLLECT/EVALUATE/AGGREGATE happen in ``EvalPipeline``; the aggregate
    ``{model: quality}`` is fanned out here to every CONTINUOUS learning adapter
    discovered from the routing registry (the bandit AND any other learning
    strategy). It does NOT couple the loop to the bandit.

    Behind the ``adapter_framework.mlops_feedback_loop`` flag (default ON as of
    RouteIQ-3b4d — the intelligent-routing use case needs the bandit to learn
    automatically; opt out with
    ``ROUTEIQ_ADAPTER_FRAMEWORK__MLOPS_FEEDBACK_LOOP=false``). It is still a
    no-op unless a continuous learning strategy is registered AND the eval
    pipeline is enabled, so the default flag costs nothing on its own. Pass
    ``force=True`` to bypass the flag (used by tests). Idempotent and never
    raises — a wiring failure must not block startup.

    Steps:
      1. Discover learning adapters from the routing registry (continuous ones
         subscribe; one-time ones are registered but skipped by the fan-out).
      2. Subscribe ``coordinator.on_aggregate_feedback`` into the eval pipeline's
         ``feedback_callbacks`` so each ``push_feedback()`` reaches the adapters.

    Args:
        coordinator: The coordinator (defaults to the process singleton).
        eval_pipeline: The eval pipeline (defaults to ``get_eval_pipeline()``;
            ``None`` when the pipeline is disabled — then only step 1 runs).
        force: Bypass the settings flag (test convenience).

    Returns:
        True if the callback was subscribed into a live eval pipeline; False
        otherwise (flag off, pipeline disabled, or already wired).
    """
    if not force:
        try:
            from litellm_llmrouter.settings import get_settings

            af = getattr(get_settings(), "adapter_framework", None)
            if af is None or not getattr(af, "mlops_feedback_loop", False):
                return False
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("MLOps: settings read failed, skipping wiring: %s", e)
            return False

    coord = coordinator or get_mlops_coordinator()

    # Step 1: discover + register learning strategies from the registry.
    try:
        discovered = coord.discover_learning_adapters_from_registry()
        if discovered:
            logger.info("MLOps: registered learning adapters %s", discovered)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("MLOps: adapter discovery failed: %s", e)

    # Step 2: subscribe the coordinator to the eval pipeline's feedback arm.
    pipeline = eval_pipeline
    if pipeline is None:
        try:
            from litellm_llmrouter.eval_pipeline import get_eval_pipeline

            pipeline = get_eval_pipeline()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("MLOps: eval pipeline unavailable: %s", e)
            pipeline = None

    if pipeline is None:
        # Eval pipeline disabled; adapters are still registered for any direct
        # online updates, but there is no aggregate feedback source to subscribe.
        return False

    add_cb = getattr(pipeline, "add_feedback_callback", None)
    if not callable(add_cb):
        logger.debug("MLOps: eval pipeline has no add_feedback_callback")
        return False

    # Remember the firing pipeline so the per-(bucket, model) aggregate read in
    # on_aggregate_feedback comes from the SAME pipeline that fires the callback
    # (RouteIQ-e019) -- authoritative even when it differs from the singleton.
    coord._eval_pipeline = pipeline

    try:
        added = add_cb(coord.on_aggregate_feedback)
        if added:
            logger.info("MLOps: subscribed coordinator to eval-pipeline feedback arm")
        return bool(added)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("MLOps: feedback subscription failed: %s", e)
        return False
