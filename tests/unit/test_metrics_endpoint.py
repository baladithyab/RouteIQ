"""
Tests for the Prometheus scrape endpoint (``GET /metrics``) — RouteIQ-bd7d /
RouteIQ-f60a.

The chart's ServiceMonitor scrapes ``path: /metrics`` on the main http port;
before this endpoint existed that scrape 404'd, and once it existed it exposed
ONLY the process-level prometheus_client default registry — none of the RouteIQ
OTel instruments. These tests assert the endpoint exists, is registered on the
unauthenticated health router (scrapers carry no user key), returns a valid
Prometheus text-exposition content type, returns a parseable body, AND that the
RouteIQ OTel instruments surface once the OTel ``PrometheusMetricReader`` is
wired into the ``MeterProvider`` at construction (observability.py).
"""

import importlib.util

import pytest
from fastapi.testclient import TestClient

from litellm_llmrouter.gateway.app import create_standalone_app
from litellm_llmrouter.routes.metrics_endpoint import reset_metrics_endpoint_state

# The OTel->Prometheus bridge is an OPTIONAL dependency (the `otel` extra). The
# instrument-exposure tests below skip cleanly when it is absent (RouteIQ-bd7d
# partial-resolution contract), while the floor tests (registration, auth,
# content type, default-registry exposure) always run.
_PROM_EXPORTER_AVAILABLE = (
    importlib.util.find_spec("opentelemetry.exporter.prometheus") is not None
)


@pytest.fixture(autouse=True)
def _reset_metrics_endpoint():
    """Reset module-level endpoint state between tests (reset_* convention)."""
    reset_metrics_endpoint_state()
    yield
    reset_metrics_endpoint_state()


@pytest.fixture()
def unauthed_client():
    """A TestClient with NO auth overrides — proves /metrics needs no credential."""
    app = create_standalone_app(
        enable_plugins=False,
        enable_resilience=False,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_metrics_endpoint_is_registered(unauthed_client):
    """GET /metrics resolves (is registered) — no longer a 404 scrape target."""
    resp = unauthed_client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_endpoint_is_unauthenticated(unauthed_client):
    """Scrapers carry no user key; /metrics must NOT require auth (200, not 401/403)."""
    resp = unauthed_client.get("/metrics")
    assert resp.status_code == 200
    assert resp.status_code not in (401, 403)


def test_metrics_endpoint_content_type(unauthed_client):
    """Content-Type is the Prometheus text exposition format (text/plain; version=...)."""
    resp = unauthed_client.get("/metrics")
    content_type = resp.headers.get("content-type", "")
    assert content_type.startswith("text/plain")
    assert "version=" in content_type


def test_metrics_endpoint_body_is_parseable_exposition(unauthed_client):
    """Body is valid Prometheus exposition text (parses without error)."""
    resp = unauthed_client.get("/metrics")
    body = resp.text
    # Prometheus text format: every non-empty line is either a ``# HELP``/``# TYPE``
    # comment or a ``metric_name ... value`` sample. An empty registry is also
    # valid exposition. Assert the body parses under the official text parser.
    from prometheus_client.parser import text_string_to_metric_families

    families = list(text_string_to_metric_families(body))
    # families may be empty (degraded install) but parsing must not raise.
    assert isinstance(families, list)


def test_metrics_endpoint_exposes_default_registry(unauthed_client):
    """A counter registered in the prometheus_client default registry is scraped."""
    from prometheus_client import Counter, REGISTRY

    metric_name = "routeiq_test_f60a_scrape_total"
    # Guard against duplicate registration across test re-runs in one process.
    if metric_name not in {
        getattr(c, "_name", None) for c in list(REGISTRY._collector_to_names)
    }:
        try:
            Counter(metric_name, "Test counter for the /metrics scrape test.")
        except ValueError:
            # Already registered (collector survives across tests) — fine.
            pass

    resp = unauthed_client.get("/metrics")
    assert resp.status_code == 200
    assert metric_name in resp.text


def test_metrics_endpoint_excluded_from_openapi_schema(unauthed_client):
    """The scrape endpoint is operational, not part of the public OpenAPI surface."""
    schema = unauthed_client.get("/openapi.json").json()
    assert "/metrics" not in schema.get("paths", {})


def test_reset_metrics_endpoint_state_is_idempotent_noop():
    """reset_metrics_endpoint_state() is a stable idempotent no-op.

    The endpoint holds no mutable module-level latch: the OTel
    PrometheusMetricReader is wired once at MeterProvider construction
    (observability.py), not lazily per-scrape, so there is nothing to clear.
    Retained for the reset_* convention / autouse fixture hook.
    """
    assert reset_metrics_endpoint_state() is None
    # Calling twice must not raise (idempotent).
    assert reset_metrics_endpoint_state() is None


# ---------------------------------------------------------------------------
# OTel instrument exposure (RouteIQ-bd7d): the construction-site PrometheusMetricReader
# ---------------------------------------------------------------------------


def _reset_global_meter_provider() -> None:
    """Reset the OTel global MeterProvider set-once latch (test isolation).

    ``metrics.set_meter_provider()`` is once-per-process; to deterministically
    exercise observability.py's construction branch (which wires the
    PrometheusMetricReader) we clear the SDK's module-level latch. Test-only —
    mirrors how OTel's own test suite resets the global provider.
    """
    import opentelemetry.metrics._internal as mi

    mi._METER_PROVIDER = None
    mi._METER_PROVIDER_SET_ONCE = mi.Once()


def _teardown_meter_provider(registry) -> None:
    """Shut down the live MeterProvider and clear any stray OTel collectors.

    Shutting the provider down stops the background OTLP PeriodicExportingMetricReader
    (no localhost push retries leaking past the test) and unregisters the
    PrometheusMetricReader's collector from the default REGISTRY. A defensive
    sweep then removes any collector the shutdown left behind, so the wired
    instruments never leak into a later test's scrape.
    """
    from opentelemetry import metrics as _otel_metrics

    provider = _otel_metrics.get_meter_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            # shutdown() may raise if a reader's collector was already removed;
            # the defensive sweep below still leaves the registry clean.
            pass
    for collector in list(getattr(registry, "_collector_to_names", {})):
        if type(collector).__name__ == "_CustomCollector":
            try:
                registry.unregister(collector)
            except KeyError:
                pass


@pytest.mark.skipif(
    not _PROM_EXPORTER_AVAILABLE,
    reason="opentelemetry-exporter-prometheus (otel extra) not installed",
)
def test_build_prometheus_metric_reader_returns_reader():
    """ObservabilityManager._build_prometheus_metric_reader() returns a reader.

    The reader self-registers an OTel collector with the prometheus_client
    default REGISTRY; we unregister it afterward so it does not leak into other
    tests' scrapes.
    """
    from opentelemetry.exporter.prometheus import PrometheusMetricReader

    from litellm_llmrouter.observability import ObservabilityManager

    mgr = ObservabilityManager(service_name="test-routeiq")
    reader = mgr._build_prometheus_metric_reader()
    assert isinstance(reader, PrometheusMetricReader)
    # Cleanup: unregister the collector this reader added to the default REGISTRY.
    from prometheus_client import REGISTRY

    REGISTRY.unregister(reader._collector)


@pytest.mark.skipif(
    not _PROM_EXPORTER_AVAILABLE,
    reason="opentelemetry-exporter-prometheus (otel extra) not installed",
)
def test_metrics_endpoint_exposes_routeiq_otel_instruments(
    unauthed_client, monkeypatch
):
    """After init_observability + recording gateway.request.total, GET /metrics
    contains the Prometheus name ``gateway_request_total`` (RouteIQ-bd7d).

    This is the core metrics-3 bridge contract: the OTel PrometheusMetricReader
    wired into the MeterProvider at construction routes RouteIQ instruments into
    the default REGISTRY that the /metrics endpoint serves.
    """
    from prometheus_client import REGISTRY

    from litellm_llmrouter.metrics import get_gateway_metrics, reset_gateway_metrics
    from litellm_llmrouter.observability import (
        init_observability,
        reset_observability_manager,
    )

    # The OTLP push reader observability.py also wires has no collector at
    # localhost:4317 in unit tests; cap its timeout so provider shutdown's final
    # flush bails immediately instead of retrying with backoff.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_TIMEOUT", "1")

    # Force observability.py's construction branch (wires the Prometheus reader).
    _reset_global_meter_provider()
    reset_gateway_metrics()

    init_observability(
        service_name="test-routeiq-bd7d",
        enable_traces=False,
        enable_logs=False,
        enable_metrics=True,
    )
    try:
        gm = get_gateway_metrics()
        assert gm is not None, "GatewayMetrics not initialized by init_observability"
        # Record a known instrument; gateway.request.total -> gateway_request_total.
        gm.request_total.add(1, {"model": "gpt-4", "status": "success"})

        resp = unauthed_client.get("/metrics")
        assert resp.status_code == 200
        assert "gateway_request_total" in resp.text
    finally:
        _teardown_meter_provider(REGISTRY)
        reset_observability_manager()
        reset_gateway_metrics()
        _reset_global_meter_provider()


@pytest.mark.skipif(
    not _PROM_EXPORTER_AVAILABLE,
    reason="opentelemetry-exporter-prometheus (otel extra) not installed",
)
def test_metrics_endpoint_has_no_raw_user_text_labels(unauthed_client, monkeypatch):
    """The exposition uses only bounded labels — no raw per-user/per-key text.

    Records instruments with the bounded label set the catalog models (model /
    status / strategy) and asserts the scrape body contains those label KEYS but
    none of the high-cardinality identifiers (user id / api key) that would blow
    up cardinality if ever leaked as a label.
    """
    from prometheus_client import REGISTRY

    from litellm_llmrouter.metrics import get_gateway_metrics, reset_gateway_metrics
    from litellm_llmrouter.observability import (
        init_observability,
        reset_observability_manager,
    )

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_TIMEOUT", "1")
    _reset_global_meter_provider()
    reset_gateway_metrics()
    init_observability(
        service_name="test-routeiq-card",
        enable_traces=False,
        enable_logs=False,
        enable_metrics=True,
    )
    try:
        gm = get_gateway_metrics()
        assert gm is not None
        gm.request_total.add(1, {"model": "gpt-4", "status": "success"})
        gm.routing_strategy_usage.add(1, {"strategy": "knn", "outcome": "success"})

        body = unauthed_client.get("/metrics").text
        # Bounded label keys present; raw user/key identifiers never appear.
        assert 'model="gpt-4"' in body
        assert "user_id" not in body
        assert "api_key" not in body
        assert "sk-" not in body
    finally:
        _teardown_meter_provider(REGISTRY)
        reset_observability_manager()
        reset_gateway_metrics()
        _reset_global_meter_provider()
