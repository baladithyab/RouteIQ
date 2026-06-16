"""
Tests for the Prometheus scrape endpoint (``GET /metrics``) — RouteIQ-f60a.

The chart's ServiceMonitor scrapes ``path: /metrics`` on the main http port;
before this endpoint existed that scrape 404'd. These tests assert the endpoint
exists, is registered on the unauthenticated health router (scrapers carry no
user key), returns a valid Prometheus text-exposition content type, and returns
a parseable body.
"""

import pytest
from fastapi.testclient import TestClient

from litellm_llmrouter.gateway.app import create_standalone_app
from litellm_llmrouter.routes.metrics_endpoint import reset_metrics_endpoint_state


@pytest.fixture(autouse=True)
def _reset_metrics_endpoint():
    """Reset the module-level OTel-bridge latch between tests (reset_* convention)."""
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


def test_reset_metrics_endpoint_state_clears_bridge_latch():
    """reset_metrics_endpoint_state() clears the OTel-bridge wired latch."""
    import litellm_llmrouter.routes.metrics_endpoint as me

    me._otel_bridge_wired = True
    reset_metrics_endpoint_state()
    assert me._otel_bridge_wired is False
