"""Unit tests for resilience OTel instruments (RouteIQ-6cd5).

These tests pin the new Prometheus-bound instruments that unblock KEDA scaling:

  * ``gateway.backpressure.active_requests`` (UpDownCounter) — in-flight request
    gauge driven by the DrainManager acquire/release path.
  * ``gateway.backpressure.rejections`` (Counter) — load-shed rejections emitted
    when the BackpressureMiddleware sends a 503 (``over_capacity`` / ``draining``).
  * ``gateway.circuit_breaker.state`` (UpDownCounter) — per-breaker 0/1 gauge
    keyed by the ``state`` label, driven at every CircuitBreaker transition.

Same in-memory-reader pattern as ``tests/unit/test_metrics_dark_subsystems.py``:
a ``MeterProvider`` backed by an ``InMemoryMetricReader`` is the metric backend,
``init_gateway_metrics`` installs the singleton the subsystems fetch via
``get_gateway_metrics()``, and we drive the REAL subsystem at its state-change
point and assert the instrument recorded the expected data point + labels.

Telemetry must never raise when OTel is disabled (no singleton); the
``no_metrics_*`` tests pin that the subsystem still works without the singleton.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Optional

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from litellm_llmrouter.metrics import (
    init_gateway_metrics,
    reset_gateway_metrics,
)
from litellm_llmrouter.resilience import (
    BackpressureMiddleware,
    CircuitBreaker,
    DrainManager,
    ResilienceConfig,
    reset_drain_manager,
    reset_shared_circuit_breaker_state,
)

# The OTel->Prometheus bridge is an OPTIONAL dependency (the ``otel`` extra). The
# end-to-end /metrics exposure test skips cleanly when it is absent.
_PROM_EXPORTER_AVAILABLE = (
    importlib.util.find_spec("opentelemetry.exporter.prometheus") is not None
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset GatewayMetrics + resilience singletons around each test."""
    reset_gateway_metrics()
    reset_drain_manager()
    reset_shared_circuit_breaker_state()
    yield
    reset_gateway_metrics()
    reset_drain_manager()
    reset_shared_circuit_breaker_state()


@pytest.fixture()
def metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture()
def gateway_metrics(metric_reader):
    """Install the GatewayMetrics singleton backed by an in-memory reader."""
    provider = MeterProvider(metric_readers=[metric_reader])
    meter = provider.get_meter("test-resilience-meter", "0.1.0")
    return init_gateway_metrics(meter)


def _data_points(reader: InMemoryMetricReader, metric_name: str) -> list[Any]:
    """Return all data points recorded for ``metric_name`` (empty if none)."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    points: list[Any] = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == metric_name:
                    points.extend(metric.data.data_points)
    return points


def _sum_value(
    reader: InMemoryMetricReader,
    metric_name: str,
    attrs: Optional[dict[str, Any]] = None,
) -> float:
    """Sum the (cumulative) value for ``metric_name`` filtered by ``attrs``.

    Works for both monotonic Counter and non-monotonic UpDownCounter data
    points, which both aggregate as a ``Sum`` whose ``.value`` is the running
    total of the recorded deltas.
    """
    total = 0.0
    for dp in _data_points(reader, metric_name):
        if attrs is None or all(dp.attributes.get(k) == v for k, v in attrs.items()):
            total += dp.value
    return total


# ---------------------------------------------------------------------------
# Backpressure: in-flight active-request gauge (DrainManager acquire/release)
# ---------------------------------------------------------------------------


class TestBackpressureActiveGauge:
    @pytest.mark.asyncio
    async def test_acquire_increments_active_gauge(
        self, gateway_metrics, metric_reader
    ):
        dm = DrainManager()
        acquired = await dm.acquire()
        assert acquired is True

        assert _sum_value(metric_reader, "gateway.backpressure.active_requests") == 1

    @pytest.mark.asyncio
    async def test_release_decrements_back_to_zero(
        self, gateway_metrics, metric_reader
    ):
        dm = DrainManager()
        await dm.acquire()
        await dm.acquire()
        await dm.release()
        await dm.release()

        # Net in-flight is zero after every slot is released.
        assert _sum_value(metric_reader, "gateway.backpressure.active_requests") == 0

    @pytest.mark.asyncio
    async def test_steady_state_reflects_outstanding_requests(
        self, gateway_metrics, metric_reader
    ):
        dm = DrainManager()
        await dm.acquire()
        await dm.acquire()
        await dm.acquire()
        await dm.release()

        # 3 acquired, 1 released -> 2 outstanding.
        assert _sum_value(metric_reader, "gateway.backpressure.active_requests") == 2

    @pytest.mark.asyncio
    async def test_acquire_while_draining_does_not_increment(
        self, gateway_metrics, metric_reader
    ):
        dm = DrainManager()
        await dm.start_drain()
        acquired = await dm.acquire()
        assert acquired is False
        # Rejected acquire never touches the in-flight gauge.
        assert not _data_points(metric_reader, "gateway.backpressure.active_requests")

    @pytest.mark.asyncio
    async def test_no_metrics_singleton_does_not_raise(self):
        """acquire/release must be a no-op (never raise) when OTel is disabled."""
        reset_gateway_metrics()  # ensure no singleton
        dm = DrainManager()
        assert await dm.acquire() is True
        await dm.release()

    @pytest.mark.asyncio
    async def test_gauge_never_negative_on_double_release(
        self, gateway_metrics, metric_reader
    ):
        """A spurious extra release must not drive the in-flight gauge below 0.

        RouteIQ-9e21: ``release`` previously emitted ``-1`` unconditionally,
        even when the active-request counter was already clamped to 0, so a
        double-release (or a drain/release race) drifted the gauge negative.
        The delta is now computed from the ACTUAL change in the counter, so an
        already-zero release emits 0 and the gauge floors at 0.
        """
        dm = DrainManager()
        await dm.acquire()
        await dm.release()  # back to 0, gauge == 0
        await dm.release()  # spurious extra release: counter clamps, gauge stays 0

        assert _sum_value(metric_reader, "gateway.backpressure.active_requests") == 0

    @pytest.mark.asyncio
    async def test_gauge_never_negative_under_simulated_drain_release_race(
        self, gateway_metrics, metric_reader
    ):
        """Under a flurry of concurrent releases the gauge never reports < 0.

        Simulates the drain/release race: more releases are issued than the
        counter ever rose to (e.g. a slot released twice while draining). The
        net gauge must be >= 0 at every observable point and == 0 at rest, with
        inc/dec strictly paired.
        """
        import asyncio

        dm = DrainManager()
        # Two real acquisitions...
        await dm.acquire()
        await dm.acquire()
        # ...then FOUR concurrent releases (two of them spurious, modelling the
        # drain/release race where a slot is released more than once).
        await asyncio.gather(dm.release(), dm.release(), dm.release(), dm.release())

        # Net delta of the gauge can never be negative: +2 acquired, at most -2
        # actually released, the 2 spurious releases emit 0.
        net = _sum_value(metric_reader, "gateway.backpressure.active_requests")
        assert net >= 0
        assert net == 0
        # The DrainManager's own counter is also clamped at 0 (never negative).
        assert dm.active_requests == 0


# ---------------------------------------------------------------------------
# Backpressure: rejection counter (BackpressureMiddleware 503 funnel)
# ---------------------------------------------------------------------------


async def _drive_asgi(app, scope) -> dict:
    """Drive a single ASGI request through ``app`` and capture the response."""
    captured: dict[str, Any] = {"status": None, "body": b""}

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            captured["status"] = message["status"]
        elif message["type"] == "http.response.body":
            captured["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return captured


def _http_scope(path: str = "/v1/chat/completions") -> dict:
    return {
        "type": "http",
        "path": path,
        "headers": [],
        "method": "POST",
    }


class TestBackpressureRejectionCounter:
    @pytest.mark.asyncio
    async def test_over_capacity_rejection_records_reason_over_capacity(
        self, gateway_metrics, metric_reader
    ):
        # Capacity of 1, already exhausted -> next request is shed with 503.
        config = ResilienceConfig(max_concurrent_requests=1)
        dm = DrainManager()
        mw = BackpressureMiddleware(app=None, config=config, drain_manager=dm)
        # Drain the single semaphore slot so the next request is at capacity.
        assert mw._semaphore is not None
        await mw._semaphore.acquire()

        resp = await _drive_asgi(mw, _http_scope())
        assert resp["status"] == 503

        assert (
            _sum_value(
                metric_reader,
                "gateway.backpressure.rejections",
                {"reason": "over_capacity"},
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_draining_rejection_records_reason_draining(
        self, gateway_metrics, metric_reader
    ):
        config = ResilienceConfig(max_concurrent_requests=10)
        dm = DrainManager()
        await dm.start_drain()
        mw = BackpressureMiddleware(app=None, config=config, drain_manager=dm)

        resp = await _drive_asgi(mw, _http_scope())
        assert resp["status"] == 503

        assert (
            _sum_value(
                metric_reader,
                "gateway.backpressure.rejections",
                {"reason": "draining"},
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_no_metrics_singleton_rejection_does_not_raise(self):
        """A 503 rejection must not raise when OTel is disabled (no singleton)."""
        reset_gateway_metrics()
        config = ResilienceConfig(max_concurrent_requests=10)
        dm = DrainManager()
        await dm.start_drain()
        mw = BackpressureMiddleware(app=None, config=config, drain_manager=dm)
        resp = await _drive_asgi(mw, _http_scope())
        assert resp["status"] == 503


# ---------------------------------------------------------------------------
# Circuit breaker: per-breaker state gauge (0/1 per state label)
# ---------------------------------------------------------------------------


class TestCircuitBreakerStateGauge:
    @pytest.mark.asyncio
    async def test_force_open_drives_open_to_one_closed_to_zero(
        self, gateway_metrics, metric_reader
    ):
        cb = CircuitBreaker("svc-open")
        await cb.force_open()

        # The breaker's current state series is driven to 1...
        assert (
            _sum_value(
                metric_reader,
                "gateway.circuit_breaker.state",
                {"breaker": "svc-open", "state": "open"},
            )
            == 1
        )
        # ...and the previous (closed) state series back to 0.
        assert (
            _sum_value(
                metric_reader,
                "gateway.circuit_breaker.state",
                {"breaker": "svc-open", "state": "closed"},
            )
            == -1
        )

    @pytest.mark.asyncio
    async def test_open_then_close_returns_state_to_closed(
        self, gateway_metrics, metric_reader
    ):
        cb = CircuitBreaker("svc-cycle")
        await cb.force_open()
        await cb.force_closed()

        # After open->close, the open series nets back to 0 and closed to 0
        # (closed -1 on open, +1 on close).
        assert (
            _sum_value(
                metric_reader,
                "gateway.circuit_breaker.state",
                {"breaker": "svc-cycle", "state": "open"},
            )
            == 0
        )
        assert (
            _sum_value(
                metric_reader,
                "gateway.circuit_breaker.state",
                {"breaker": "svc-cycle", "state": "closed"},
            )
            == 0
        )

    @pytest.mark.asyncio
    async def test_per_breaker_isolation(self, gateway_metrics, metric_reader):
        """State series are keyed per breaker; one open breaker does not leak.

        ``breaker-b`` is constructed but never transitioned, so it must never
        emit a state data point — proving the gauge is keyed per breaker.
        """
        breaker_a = CircuitBreaker("breaker-a")
        CircuitBreaker("breaker-b")  # constructed, never transitioned
        await breaker_a.force_open()

        # breaker-a's open series is at 1.
        assert (
            _sum_value(
                metric_reader,
                "gateway.circuit_breaker.state",
                {"breaker": "breaker-a", "state": "open"},
            )
            == 1
        )
        # breaker-b never emitted any state series.
        assert not [
            dp
            for dp in _data_points(metric_reader, "gateway.circuit_breaker.state")
            if dp.attributes.get("breaker") == "breaker-b"
        ]

    @pytest.mark.asyncio
    async def test_no_state_emitted_when_transition_is_noop(
        self, gateway_metrics, metric_reader
    ):
        cb = CircuitBreaker("svc-noop")
        # Already CLOSED; closing again is a no-op transition -> no state point.
        await cb.force_closed()
        assert not _data_points(metric_reader, "gateway.circuit_breaker.state")

    @pytest.mark.asyncio
    async def test_no_metrics_singleton_transition_does_not_raise(self):
        """force_open must not raise when OTel is disabled (no singleton)."""
        reset_gateway_metrics()
        cb = CircuitBreaker("svc-disabled")
        await cb.force_open()  # must not raise


# ---------------------------------------------------------------------------
# End-to-end: the new instruments surface on GET /metrics (RouteIQ-6cd5 -> KEDA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PROM_EXPORTER_AVAILABLE,
    reason="opentelemetry-exporter-prometheus (otel extra) not installed",
)
@pytest.mark.asyncio
async def test_resilience_instruments_surface_on_metrics_endpoint(monkeypatch):
    """After recording, the resilience instruments appear in the /metrics scrape.

    Mirrors test_metrics_endpoint's bridge contract: the OTel
    PrometheusMetricReader wired into the MeterProvider at construction routes
    the resilience instruments into the default REGISTRY the /metrics endpoint
    serves, so a KEDA Prometheus scaler can read them.
    """
    from fastapi.testclient import TestClient
    from prometheus_client import REGISTRY

    from litellm_llmrouter.gateway.app import create_standalone_app
    from litellm_llmrouter.metrics import get_gateway_metrics, reset_gateway_metrics
    from litellm_llmrouter.observability import (
        init_observability,
        reset_observability_manager,
    )
    from litellm_llmrouter.routes.metrics_endpoint import (
        reset_metrics_endpoint_state,
    )

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_TIMEOUT", "1")

    import opentelemetry.metrics._internal as mi

    def _reset_global_meter_provider() -> None:
        mi._METER_PROVIDER = None
        mi._METER_PROVIDER_SET_ONCE = mi.Once()

    _reset_global_meter_provider()
    reset_gateway_metrics()
    reset_metrics_endpoint_state()

    init_observability(
        service_name="test-routeiq-6cd5",
        enable_traces=False,
        enable_logs=False,
        enable_metrics=True,
    )
    try:
        gm = get_gateway_metrics()
        assert gm is not None
        # Drive each new instrument once.
        gm.inc_backpressure_active(1)
        gm.record_backpressure_rejection("over_capacity")
        gm.record_circuit_breaker_state("svc", "closed", "open")

        app = create_standalone_app(enable_plugins=False, enable_resilience=False)
        client = TestClient(app, raise_server_exceptions=False)
        body = client.get("/metrics").text

        assert "gateway_backpressure_active_requests" in body
        assert "gateway_backpressure_rejections_total" in body
        assert "gateway_circuit_breaker_state" in body
    finally:
        # Shut the provider down + sweep its collectors so nothing leaks.
        from opentelemetry import metrics as _otel_metrics

        provider = _otel_metrics.get_meter_provider()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        for collector in list(getattr(REGISTRY, "_collector_to_names", {})):
            if type(collector).__name__ == "_CustomCollector":
                try:
                    REGISTRY.unregister(collector)
                except KeyError:
                    pass
        reset_observability_manager()
        reset_gateway_metrics()
        reset_metrics_endpoint_state()
        _reset_global_meter_provider()
