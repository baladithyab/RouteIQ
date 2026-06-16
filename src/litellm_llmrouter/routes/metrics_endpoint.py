"""
Prometheus scrape endpoint (``GET /metrics``)
=============================================

RouteIQ-bd7d / RouteIQ-f60a (metrics-3): historically the gateway pushed
metrics to an OTLP collector ONLY (``observability.py`` wired a
``PeriodicExportingMetricReader`` onto the SDK ``MeterProvider``), and even once
this ``GET /metrics`` endpoint existed it exposed only the process-level
prometheus_client default registry (Python GC/process collectors) — none of the
RouteIQ OTel instruments (``metrics.py``) appeared, because an OTel
``MetricReader`` can only be attached to a ``MeterProvider`` at CONSTRUCTION
(there is no public post-construction registration API).

The fix wires an OTel ``PrometheusMetricReader`` into the ``MeterProvider`` at
construction in ``observability.py`` (``_init_metrics``), ALONGSIDE the OTLP
``PeriodicExportingMetricReader`` (additive — the OTLP push is never replaced).
That reader self-registers with the ``prometheus_client`` default ``REGISTRY``,
so the RouteIQ instruments now appear here. This module serves
``generate_latest(REGISTRY)`` over that same default registry. When the optional
``opentelemetry-exporter-prometheus`` package is absent the construction-site
wiring skips the Prometheus reader (OTLP push unaffected) and this endpoint
still serves the process-level default registry, so the scrape target is never a
404.

Scope / auth: the endpoint is registered on the **unauthenticated**
``health_router`` (same scope as the K8s probes). Prometheus scrapers do not
carry per-user API keys, so user-auth would make the target permanently fail;
network-scoping (NetworkPolicy / ServiceMonitor namespace selector) is the
intended access control, mirroring how ``/_health/*`` is scoped. The RouteIQ
instruments use bounded, low-cardinality labels only (strategy / model /
status / reason / check_type) — no raw per-user or per-key text is ever a label,
so the unauthenticated exposition leaks no high-cardinality user data.
"""

from __future__ import annotations

import logging

from fastapi import Response

from . import health_router

logger = logging.getLogger(__name__)

# Prometheus text exposition content type. ``prometheus_client`` exports the
# canonical value, but we pin a stable fallback so the endpoint always advertises
# a valid ``text/plain; version=...`` content type even if the symbol moves.
try:  # pragma: no cover - import shape only
    from prometheus_client import (
        CONTENT_TYPE_LATEST as _PROM_CONTENT_TYPE,
        REGISTRY as _PROM_REGISTRY,
        generate_latest as _prom_generate_latest,
    )

    _PROM_CLIENT_AVAILABLE = True
except Exception:  # pragma: no cover - prometheus_client is a core dep
    _PROM_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
    _PROM_REGISTRY = None
    _prom_generate_latest = None
    _PROM_CLIENT_AVAILABLE = False


@health_router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus pull-based scrape endpoint (RouteIQ-bd7d / RouteIQ-f60a).

    Returns the Prometheus text exposition format (content type
    ``text/plain; version=...``) so the chart's ServiceMonitor target resolves
    instead of 404ing. Unauthenticated (network/admin-scoped) — Prometheus
    scrapers do not carry user API keys, mirroring the ``/_health/*`` probes.

    Serves ``generate_latest(REGISTRY)`` over the prometheus_client default
    registry. The OTel ``PrometheusMetricReader`` wired at MeterProvider
    construction in ``observability.py`` self-registers with that same default
    registry, so the RouteIQ OTel instruments (``metrics.py``) surface here in
    addition to the process-level collectors. The OTLP metric push configured by
    ``observability.py`` is unaffected; this endpoint is an additive pull
    surface.
    """
    if _PROM_CLIENT_AVAILABLE and _prom_generate_latest is not None:
        payload = _prom_generate_latest(_PROM_REGISTRY)
        return Response(content=payload, media_type=_PROM_CONTENT_TYPE)

    # prometheus_client is a declared core dependency, so this branch is only
    # reached in a degraded install. Emit a single, parseable comment line so the
    # endpoint is still a valid (empty) Prometheus exposition rather than a 500.
    return Response(
        content="# prometheus_client unavailable; no metrics exported\n",
        media_type=_PROM_CONTENT_TYPE,
    )


def reset_metrics_endpoint_state() -> None:
    """Reset module-level state (singleton hygiene for tests).

    Retained for the RouteIQ ``reset_*()`` test convention and as the autouse
    fixture hook. The endpoint now holds no mutable module-level latch: the OTel
    ``PrometheusMetricReader`` is wired once at ``MeterProvider`` construction in
    ``observability.py`` (no lazy per-scrape bridge), so there is nothing to
    clear here. Kept as a stable, idempotent no-op so callers/fixtures do not
    break.
    """
    return None
