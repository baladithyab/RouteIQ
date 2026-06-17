"""
MLOps Drift Detection (RouteIQ-6dce)
====================================

Detects two distinct degradations of a deployed routing configuration and emits
OTel drift signals so CloudWatch / Prometheus can alarm:

(a) **Input drift** -- a shift in the request / task-bucket distribution vs a
    captured baseline. Scored with the **Population Stability Index (PSI)**, the
    textbook population-shift measure::

        PSI = sum_i (cur_i - base_i) * ln(cur_i / base_i)

    over the shared coarse-bucket support (with a small epsilon floor so empty
    buckets do not blow up the log). PSI < 0.1 is "no shift", 0.1-0.2 "minor",
    >= 0.2 "moderate+ shift" (the default ``input_drift_threshold``).

(b) **Routing-quality regression** -- the aggregated routing quality (the eval
    loop's ``ModelQualityTracker`` ``[0, 1]`` scale) dropping below a captured
    baseline by an absolute ``quality_regression_threshold``.

The detector is STRATEGY-AGNOSTIC and builds on the EXISTING eval loop: it does
not re-score anything, it consumes the aggregate ``{model/strategy: quality}``
that ``EvalPipeline.push_feedback`` already produces, plus coarse request-bucket
observations recorded on the routing hot path.

Pure stdlib (no numpy), settings-gated, DEFAULT OFF. Importing this module or
calling its methods with the detector disabled is a byte-stable no-op.

Design: the detector keeps a *baseline* snapshot (frozen at ``capture_baseline``)
and a *current* sliding window. ``evaluate()`` compares the two and returns a
:class:`DriftReport`; it emits gauges + signal counters via ``metrics.py`` only
when the windows are large enough (``min_samples``) -- so a cold start never
fires a false alarm.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DriftDetector",
    "DriftReport",
    "get_drift_detector",
    "reset_drift_detector",
]

# Floor applied to bucket proportions before the PSI log so an empty/near-empty
# bucket cannot drive the term to +/-inf. Standard PSI practice.
_PSI_EPSILON = 1e-4


@dataclass
class DriftReport:
    """Outcome of one drift evaluation.

    All scores are PII-safe aggregates (bucket proportions + quality scalars).
    """

    input_drift_score: float = 0.0
    """Population Stability Index of the current vs baseline bucket dist."""

    input_drift_detected: bool = False
    """True iff ``input_drift_score >= input_drift_threshold``."""

    quality_baseline: Optional[float] = None
    """Captured baseline aggregated quality (None if no baseline)."""

    quality_current: Optional[float] = None
    """Current aggregated quality (None if insufficient samples)."""

    quality_regression: float = 0.0
    """``quality_baseline - quality_current`` (positive => quality dropped)."""

    quality_regression_detected: bool = False
    """True iff ``quality_regression >= quality_regression_threshold``."""

    evaluated: bool = False
    """False when skipped for insufficient samples (no signal emitted)."""

    reason: str = ""
    """Why the evaluation was skipped, when ``evaluated`` is False."""

    signals: List[str] = field(default_factory=list)
    """Drift signal kinds that fired (``input_drift`` / ``quality_regression``)."""

    @property
    def any_drift(self) -> bool:
        """True if either drift signal fired."""
        return self.input_drift_detected or self.quality_regression_detected


class DriftDetector:
    """Input-drift + quality-regression detector (RouteIQ-6dce).

    Thread-safe. Records coarse request-bucket observations and aggregate
    quality values, compares the current sliding window against a frozen
    baseline, and emits OTel drift gauges/signals via ``metrics.py``.

    Args:
        input_drift_threshold: PSI threshold for an input-drift signal.
        quality_regression_threshold: Absolute quality drop for a regression
            signal.
        min_samples: Minimum observations in BOTH baseline and current windows
            before any evaluation fires.
        window_size: Sliding-window size for the current bucket/quality samples.
    """

    def __init__(
        self,
        *,
        input_drift_threshold: float = 0.2,
        quality_regression_threshold: float = 0.1,
        min_samples: int = 30,
        window_size: int = 200,
    ) -> None:
        self._input_drift_threshold = input_drift_threshold
        self._quality_regression_threshold = quality_regression_threshold
        self._min_samples = min_samples
        self._window_size = window_size
        self._lock = threading.RLock()

        # Baseline (frozen) -- bucket counts + a single aggregated quality scalar.
        self._baseline_buckets: Dict[str, int] = {}
        self._baseline_quality: Optional[float] = None

        # Current sliding windows.
        self._cur_buckets: Deque[str] = deque(maxlen=window_size)
        self._cur_quality: Deque[float] = deque(maxlen=window_size)

        # Last-emitted gauge values (so we can push signed deltas to the
        # UpDownCounter gauge model in metrics.py).
        self._last_input_drift: float = 0.0
        self._last_quality_regression: float = 0.0

    # ------------------------------------------------------------------
    # Observation recording (routing hot path)
    # ------------------------------------------------------------------

    def record_request_bucket(self, bucket: str) -> None:
        """Record one coarse request-bucket observation into the current window.

        ``bucket`` is a LOW-cardinality label (e.g. task class / prompt-length
        bucket / requested tier) -- never raw user text.
        """
        if not bucket:
            return
        with self._lock:
            self._cur_buckets.append(bucket)

    def record_quality(self, quality: float) -> None:
        """Record one aggregated-quality observation into the current window.

        ``quality`` is the eval loop's ``[0, 1]`` ModelQualityTracker scale.
        """
        with self._lock:
            self._cur_quality.append(float(quality))

    def observe_aggregate(self, model_qualities: Dict[str, float]) -> None:
        """Feed an eval-loop aggregate ``{model/strategy: quality}`` snapshot.

        This is the signature ``EvalPipeline`` invokes its feedback callbacks
        with, so the detector can subscribe to the FEEDBACK arm directly. The
        snapshot's MEAN quality is recorded as one current-quality observation.
        """
        if not model_qualities:
            return
        mean_q = sum(model_qualities.values()) / len(model_qualities)
        self.record_quality(mean_q)

    # ------------------------------------------------------------------
    # Baseline capture
    # ------------------------------------------------------------------

    def capture_baseline(
        self,
        *,
        buckets: Optional[Dict[str, int]] = None,
        quality: Optional[float] = None,
    ) -> None:
        """Freeze the reference distribution + quality the current window is
        compared against.

        If ``buckets`` / ``quality`` are omitted, the CURRENT window is snapshot
        as the baseline (the common "first stable window is the baseline" flow).
        """
        with self._lock:
            if buckets is not None:
                self._baseline_buckets = dict(buckets)
            else:
                self._baseline_buckets = self._count_buckets(self._cur_buckets)

            if quality is not None:
                self._baseline_quality = float(quality)
            elif self._cur_quality:
                self._baseline_quality = sum(self._cur_quality) / len(self._cur_quality)
            else:
                self._baseline_quality = None

    @staticmethod
    def _count_buckets(samples: Deque[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for b in samples:
            counts[b] = counts.get(b, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # PSI
    # ------------------------------------------------------------------

    @staticmethod
    def population_stability_index(
        baseline: Dict[str, int],
        current: Dict[str, int],
    ) -> float:
        """Population Stability Index over the union of bucket keys.

        Proportions are floored at :data:`_PSI_EPSILON` to keep the log finite.
        Returns 0.0 when either side is empty (cannot measure a shift).
        """
        base_total = sum(baseline.values())
        cur_total = sum(current.values())
        if base_total <= 0 or cur_total <= 0:
            return 0.0

        keys = set(baseline) | set(current)
        psi = 0.0
        for k in keys:
            base_p = max(baseline.get(k, 0) / base_total, _PSI_EPSILON)
            cur_p = max(current.get(k, 0) / cur_total, _PSI_EPSILON)
            psi += (cur_p - base_p) * math.log(cur_p / base_p)
        # PSI is non-negative by construction; clamp tiny negative float noise.
        return max(psi, 0.0)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> DriftReport:
        """Compare the current window against the baseline and emit signals.

        Returns a :class:`DriftReport`. Skips (``evaluated=False``, no signal)
        when either window is below ``min_samples`` so a cold start never fires
        a false alarm. Emits the drift gauges + per-kind signal counters via
        ``metrics.py`` (best-effort; telemetry never raises).
        """
        with self._lock:
            cur_bucket_count = len(self._cur_buckets)
            cur_quality_count = len(self._cur_quality)
            baseline_bucket_total = sum(self._baseline_buckets.values())

            report = DriftReport()

            # --- input drift (needs baseline + current buckets) ---
            bucket_evaluable = (
                baseline_bucket_total >= self._min_samples
                and cur_bucket_count >= self._min_samples
            )
            if bucket_evaluable:
                cur_counts = self._count_buckets(self._cur_buckets)
                psi = self.population_stability_index(
                    self._baseline_buckets, cur_counts
                )
                report.input_drift_score = psi
                report.input_drift_detected = psi >= self._input_drift_threshold

            # --- quality regression (needs baseline quality + current samples) ---
            baseline_quality = self._baseline_quality
            quality_evaluable = (
                baseline_quality is not None and cur_quality_count >= self._min_samples
            )
            if quality_evaluable and baseline_quality is not None:
                cur_q = sum(self._cur_quality) / cur_quality_count
                report.quality_baseline = baseline_quality
                report.quality_current = cur_q
                regression = baseline_quality - cur_q
                report.quality_regression = regression
                report.quality_regression_detected = (
                    regression >= self._quality_regression_threshold
                )

            if not bucket_evaluable and not quality_evaluable:
                report.evaluated = False
                report.reason = "insufficient_samples"
                return report

            report.evaluated = True
            if report.input_drift_detected:
                report.signals.append("input_drift")
            if report.quality_regression_detected:
                report.signals.append("quality_regression")

            # Emit metrics OUTSIDE-but-with the lock held is fine: the metric
            # helpers are non-blocking and must never raise.
            self._emit_metrics(report, bucket_evaluable, quality_evaluable)
            return report

    def _emit_metrics(
        self,
        report: DriftReport,
        bucket_evaluable: bool,
        quality_evaluable: bool,
    ) -> None:
        """Best-effort OTel emission. Telemetry must never break the loop."""
        try:
            from litellm_llmrouter.metrics import get_gateway_metrics

            m = get_gateway_metrics()
            if m is None:
                return
            if bucket_evaluable:
                m.set_input_drift_score(
                    report.input_drift_score, self._last_input_drift
                )
                self._last_input_drift = report.input_drift_score
            if quality_evaluable:
                m.set_quality_regression(
                    report.quality_regression, self._last_quality_regression
                )
                self._last_quality_regression = report.quality_regression
            for kind in report.signals:
                m.record_drift_signal(kind)
        except Exception:  # pragma: no cover - telemetry must not break flow
            pass

    def get_stats(self) -> Dict[str, object]:
        """PII-safe snapshot of detector state for admin/debugging."""
        with self._lock:
            return {
                "baseline_bucket_total": sum(self._baseline_buckets.values()),
                "baseline_distinct_buckets": len(self._baseline_buckets),
                "baseline_quality": self._baseline_quality,
                "current_bucket_samples": len(self._cur_buckets),
                "current_quality_samples": len(self._cur_quality),
                "min_samples": self._min_samples,
                "window_size": self._window_size,
                "input_drift_threshold": self._input_drift_threshold,
                "quality_regression_threshold": self._quality_regression_threshold,
            }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_detector: Optional[DriftDetector] = None
_detector_lock = threading.Lock()


def get_drift_detector() -> Optional[DriftDetector]:
    """Get the drift-detector singleton, or None when disabled.

    Returns None unless ``settings.mlops.drift.enabled`` is true. Settings read
    failures degrade to disabled (None) so a misconfig never silently enables
    drift detection.
    """
    global _detector
    with _detector_lock:
        if _detector is not None:
            return _detector
        cfg = _drift_settings()
        if cfg is None or not cfg.enabled:
            return None
        _detector = DriftDetector(
            input_drift_threshold=cfg.input_drift_threshold,
            quality_regression_threshold=cfg.quality_regression_threshold,
            min_samples=cfg.min_samples,
            window_size=cfg.window_size,
        )
        return _detector


def _drift_settings():  # type: ignore[no-untyped-def]
    """Read ``settings.mlops.drift`` (None on any failure)."""
    try:
        from litellm_llmrouter.settings import get_settings

        mlops = getattr(get_settings(), "mlops", None)
        return getattr(mlops, "drift", None) if mlops is not None else None
    except Exception:  # pragma: no cover - defensive
        return None


def reset_drift_detector() -> None:
    """Reset the singleton (MUST be called in the autouse test fixture)."""
    global _detector
    with _detector_lock:
        _detector = None
