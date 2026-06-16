"""
Prometheus scrape endpoint (``GET /metrics``)
=============================================

RouteIQ-f60a (metrics-3): historically the gateway pushed metrics to an OTLP
collector ONLY (``observability.py`` wires a ``PeriodicExportingMetricReader``
onto the SDK ``MeterProvider``), yet the Helm chart shipped a ``ServiceMonitor``
that scraped ``path: /metrics`` — which 404'd because no such endpoint existed.

This module adds the missing pull-based Prometheus exposition endpoint so the
ServiceMonitor scrape target is real. It is **additive** to the OTLP push: the
OTLP ``PeriodicExportingMetricReader`` on the ``MeterProvider`` is left untouched
(``observability.py`` is unchanged). When the OTel Prometheus exporter
(``opentelemetry-exporter-prometheus``) is importable it is wired as an
*additional* metric reader against the existing OTel ``MeterProvider`` so the
RouteIQ OTel instruments (``metrics.py``) surface in the scrape; when it is not
available the endpoint still serves the process-level Prometheus default
registry (``prometheus_client``, a declared core dependency) so the scrape
target is never a 404.

Scope / auth: the endpoint is registered on the **unauthenticated**
``health_router`` (same scope as the K8s probes). Prometheus scrapers do not
carry per-user API keys, so user-auth would make the target permanently fail;
network-scoping (NetworkPolicy / ServiceMonitor namespace selector) is the
intended access control, mirroring how ``/_health/*`` is scoped.
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

# Whether the OTel->Prometheus bridge has been wired against the OTel
# MeterProvider. Wired lazily on first scrape so the RouteIQ OTel instruments
# (metrics.py) are exported in addition to the prometheus_client default
# registry. The OTel Prometheus exporter is an OPTIONAL dependency; absence is
# not an error (the prometheus_client default registry still serves).
_otel_bridge_wired: bool = False


def _try_wire_otel_prometheus_bridge() -> bool:
    """Attempt to attach an OTel ``PrometheusMetricReader`` to the live provider.

    The bridge surfaces the RouteIQ OTel instruments (``metrics.py``) in the
    Prometheus scrape *in addition to* the OTLP push reader already configured by
    ``observability.py`` — the OTLP push is never replaced. Returns ``True`` if
    the bridge is (or was already) wired, ``False`` when the optional exporter is
    unavailable or the provider cannot accept a reader.

    Idempotent and never raises: a telemetry-wiring failure must not break the
    scrape endpoint (it falls back to the prometheus_client default registry).
    """
    global _otel_bridge_wired
    if _otel_bridge_wired:
        return True
    try:
        from opentelemetry.exporter.prometheus import (  # type: ignore[import-not-found]
            PrometheusMetricReader,
        )
    except Exception:
        # Optional dependency not installed — prometheus_client default registry
        # is the exposition source. Not an error.
        return False

    try:
        from opentelemetry import metrics as _otel_metrics

        provider = _otel_metrics.get_meter_provider()
        # The SDK MeterProvider does not expose a public post-construction
        # reader-registration API; we attach via the documented private hook
        # only when present, guarding against SDK shape drift.
        register = getattr(provider, "_register_metric_reader", None)
        reader = PrometheusMetricReader()
        if callable(register):
            register(reader)
            _otel_bridge_wired = True
            logger.info(
                "Wired OTel PrometheusMetricReader as an ADDITIONAL reader "
                "(OTLP push reader left intact)"
            )
            return True
        logger.debug(
            "OTel MeterProvider does not accept a post-construction reader; "
            "serving prometheus_client default registry only"
        )
        return False
    except Exception:  # pragma: no cover - defensive: never break the scrape
        logger.debug("Failed to wire OTel Prometheus bridge", exc_info=True)
        return False


@health_router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus pull-based scrape endpoint (RouteIQ-f60a).

    Returns the Prometheus text exposition format (content type
    ``text/plain; version=...``) so the chart's ServiceMonitor target resolves
    instead of 404ing. Unauthenticated (network/admin-scoped) — Prometheus
    scrapers do not carry user API keys, mirroring the ``/_health/*`` probes.

    The OTLP metric push configured by ``observability.py`` is unaffected; this
    endpoint is an additive pull surface. When the optional OTel Prometheus
    exporter is installed its reader is wired as an ADDITIONAL reader so the
    RouteIQ OTel instruments surface here too.
    """
    # Best-effort: surface OTel instruments via the bridge when available. The
    # prometheus_client default registry is always exported as the floor so the
    # scrape never returns an empty/error body.
    _try_wire_otel_prometheus_bridge()

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
    """Reset module-level bridge state (singleton hygiene for tests).

    Per the RouteIQ ``reset_*()`` test convention: clears the
    ``_otel_bridge_wired`` latch so a test that exercises the OTel bridge does
    not leak the wired state into the next test.
    """
    global _otel_bridge_wired
    _otel_bridge_wired = False
