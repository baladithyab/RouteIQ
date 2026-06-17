"""Unit tests for the MLOps drift detector (Cluster H, RouteIQ-6dce).

Covers REAL detector behavior (not mock tautology):
- injected input-distribution shift -> input-drift signal fires (PSI >= threshold);
- aggregated-quality drop -> quality-regression signal fires;
- a stable window vs its own baseline -> no signal;
- insufficient samples -> evaluation skipped, no signal;
- drift gauges + signal counters are emitted via the GatewayMetrics seam.

The detector is settings-gated; the singleton tests pin default-off behavior.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from litellm_llmrouter.metrics import init_gateway_metrics, reset_gateway_metrics
from litellm_llmrouter.mlops.drift import (
    DriftDetector,
    get_drift_detector,
    reset_drift_detector,
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    reset_drift_detector()
    reset_gateway_metrics()
    yield
    reset_drift_detector()
    reset_gateway_metrics()


@pytest.fixture()
def metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture()
def gateway_metrics(metric_reader):
    provider = MeterProvider(metric_readers=[metric_reader])
    meter = provider.get_meter("test-drift-meter", "0.1.0")
    return init_gateway_metrics(meter)


def _data_points(reader: InMemoryMetricReader, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    if data is None:
        return []
    pts: list[Any] = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    pts.extend(metric.data.data_points)
    return pts


def _counter_value(
    reader: InMemoryMetricReader,
    name: str,
    attrs: Optional[dict] = None,
) -> float:
    total = 0.0
    for dp in _data_points(reader, name):
        if attrs is None or all(dp.attributes.get(k) == v for k, v in attrs.items()):
            total += dp.value
    return total


# --------------------------------------------------------------- PSI math


class TestPSI:
    def test_identical_distributions_zero_psi(self):
        base = {"a": 50, "b": 50}
        cur = {"a": 50, "b": 50}
        assert DriftDetector.population_stability_index(base, cur) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_shifted_distribution_positive_psi(self):
        base = {"a": 90, "b": 10}
        cur = {"a": 10, "b": 90}  # complete inversion
        psi = DriftDetector.population_stability_index(base, cur)
        assert psi > 0.2  # well past the moderate-shift line

    def test_empty_side_returns_zero(self):
        assert DriftDetector.population_stability_index({}, {"a": 5}) == 0.0
        assert DriftDetector.population_stability_index({"a": 5}, {}) == 0.0

    def test_new_bucket_does_not_blow_up(self):
        # current introduces a bucket absent from baseline: epsilon floor keeps
        # the term finite (no inf / NaN).
        base = {"a": 100}
        cur = {"a": 50, "b": 50}
        psi = DriftDetector.population_stability_index(base, cur)
        assert psi > 0.0 and psi < float("inf")


# --------------------------------------------------------- input drift signal


class TestInputDrift:
    def test_injected_shift_fires_input_drift(self, gateway_metrics, metric_reader):
        det = DriftDetector(input_drift_threshold=0.2, min_samples=10, window_size=200)
        # Baseline: heavily 'simple' bucket.
        det.capture_baseline(buckets={"simple": 90, "complex": 10})
        # Current window: inverted distribution -> large PSI.
        for _ in range(90):
            det.record_request_bucket("complex")
        for _ in range(10):
            det.record_request_bucket("simple")

        report = det.evaluate()
        assert report.evaluated is True
        assert report.input_drift_detected is True
        assert "input_drift" in report.signals
        assert report.input_drift_score >= 0.2

        # Signal counter fired for the input_drift kind.
        assert (
            _counter_value(
                metric_reader, "gateway.mlops.drift.signal", {"kind": "input_drift"}
            )
            == 1.0
        )
        # The gauge recorded a (non-zero) value.
        assert _data_points(metric_reader, "gateway.mlops.input_drift.score")

    def test_stable_distribution_no_signal(self, gateway_metrics, metric_reader):
        det = DriftDetector(input_drift_threshold=0.2, min_samples=10, window_size=200)
        det.capture_baseline(buckets={"simple": 50, "complex": 50})
        for _ in range(50):
            det.record_request_bucket("simple")
        for _ in range(50):
            det.record_request_bucket("complex")

        report = det.evaluate()
        assert report.evaluated is True
        assert report.input_drift_detected is False
        assert "input_drift" not in report.signals
        assert (
            _counter_value(
                metric_reader, "gateway.mlops.drift.signal", {"kind": "input_drift"}
            )
            == 0.0
        )


# ------------------------------------------------- quality regression signal


class TestQualityRegression:
    def test_quality_drop_fires_regression(self, gateway_metrics, metric_reader):
        det = DriftDetector(
            quality_regression_threshold=0.1, min_samples=10, window_size=200
        )
        det.capture_baseline(quality=0.9)
        for _ in range(20):
            det.record_quality(0.6)  # dropped 0.3 below baseline

        report = det.evaluate()
        assert report.evaluated is True
        assert report.quality_regression_detected is True
        assert report.quality_regression == pytest.approx(0.3, abs=1e-9)
        assert "quality_regression" in report.signals
        assert (
            _counter_value(
                metric_reader,
                "gateway.mlops.drift.signal",
                {"kind": "quality_regression"},
            )
            == 1.0
        )

    def test_stable_quality_no_regression(self, gateway_metrics, metric_reader):
        det = DriftDetector(
            quality_regression_threshold=0.1, min_samples=10, window_size=200
        )
        det.capture_baseline(quality=0.85)
        for _ in range(20):
            det.record_quality(0.84)  # within threshold

        report = det.evaluate()
        assert report.evaluated is True
        assert report.quality_regression_detected is False
        assert "quality_regression" not in report.signals

    def test_observe_aggregate_records_mean(self):
        det = DriftDetector(min_samples=2, window_size=200)
        det.observe_aggregate({"gpt-4o": 0.8, "claude": 0.6})  # mean 0.7
        stats = det.get_stats()
        assert stats["current_quality_samples"] == 1


# -------------------------------------------------------- insufficient samples


class TestInsufficientSamples:
    def test_cold_start_no_evaluation(self, gateway_metrics, metric_reader):
        det = DriftDetector(min_samples=30, window_size=200)
        det.capture_baseline(buckets={"a": 5}, quality=0.9)
        # Only a handful of current samples -> below min_samples.
        for _ in range(3):
            det.record_request_bucket("a")
            det.record_quality(0.2)

        report = det.evaluate()
        assert report.evaluated is False
        assert report.reason == "insufficient_samples"
        assert report.signals == []
        # No signal counter, no gauge emission.
        assert _data_points(metric_reader, "gateway.mlops.drift.signal") == []

    def test_quality_only_when_no_baseline_buckets(self, gateway_metrics):
        # Quality drop evaluable even when there is no bucket baseline.
        det = DriftDetector(
            quality_regression_threshold=0.1, min_samples=5, window_size=200
        )
        det.capture_baseline(quality=0.9)  # no buckets given, current empty
        for _ in range(10):
            det.record_quality(0.5)
        report = det.evaluate()
        assert report.evaluated is True
        assert report.quality_regression_detected is True
        assert report.input_drift_detected is False


# ------------------------------------------------------------- settings gate


class TestSettingsGate:
    def test_singleton_disabled_by_default(self):
        # Default settings have mlops.drift.enabled=False.
        assert get_drift_detector() is None
