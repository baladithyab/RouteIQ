"""
Service Discovery and Health Probing for External Dependencies
===============================================================

RouteIQ operates with optional external services. At startup, it probes
configured services and reports their availability. Features degrade
gracefully when optional services are unavailable.

Required services: None (RouteIQ can run standalone with just LLM provider keys)
Optional services: PostgreSQL, Redis, OTel Collector, OIDC Provider

Usage:
    from litellm_llmrouter.service_discovery import (
        probe_all_services,
        get_feature_availability,
        format_service_status_table,
    )

    # At startup
    services = await probe_all_services()
    features = get_feature_availability()
    print(format_service_status_table(services, features))

    # At runtime (health endpoint)
    services = await probe_all_services()
    return {"services": services, "features": get_feature_availability()}
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ServiceStatus:
    """Status of a single external service probe."""

    name: str
    available: bool
    url: str  # sanitized (no passwords)
    latency_ms: Optional[float] = None
    version: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


class ExternalServices(str, Enum):
    """Known external services that RouteIQ can integrate with."""

    POSTGRESQL = "postgresql"
    REDIS = "redis"
    OTEL_COLLECTOR = "otel-collector"
    OIDC_PROVIDER = "oidc-provider"


# ---------------------------------------------------------------------------
# Module-level cache of last probe results
# ---------------------------------------------------------------------------

_last_probe: Dict[str, ServiceStatus] = {}
_last_probe_lock: Optional[asyncio.Lock] = None


def _get_last_probe_lock() -> asyncio.Lock:
    """Lazily initialize the probe lock to avoid binding to wrong event loop."""
    global _last_probe_lock
    if _last_probe_lock is None:
        _last_probe_lock = asyncio.Lock()
    return _last_probe_lock


def get_last_probe_results() -> Dict[str, ServiceStatus]:
    """Return the most recent probe results (non-blocking, may be stale)."""
    return dict(_last_probe)


def reset_service_discovery() -> None:
    """Reset module state for testing."""
    global _last_probe, _last_probe_lock
    _last_probe = {}
    _last_probe_lock = None


# ---------------------------------------------------------------------------
# URL sanitization
# ---------------------------------------------------------------------------


def _sanitize_url(url: str) -> str:
    """Remove credentials from a URL for safe logging/display.

    Examples:
        postgresql://user:pass@host:5432/db -> postgresql://***@host:5432/db
        redis://:secret@host:6379/0         -> redis://***@host:6379/0
    """
    try:
        parsed = urlparse(url)
        if parsed.password or parsed.username:
            # Rebuild without credentials
            netloc = f"***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            sanitized = parsed._replace(netloc=netloc)
            return urlunparse(sanitized)
        return url
    except Exception:
        return "<invalid-url>"


# ---------------------------------------------------------------------------
# Individual service probes
# ---------------------------------------------------------------------------


async def probe_postgresql(db_url: Optional[str] = None) -> ServiceStatus:
    """Probe PostgreSQL connectivity. Uses shared pool if available.

    Args:
        db_url: Database URL override. Falls back to DATABASE_URL env var.

    Returns:
        ServiceStatus for the PostgreSQL connection.
    """
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        return ServiceStatus(
            name=ExternalServices.POSTGRESQL.value,
            available=False,
            url="",
            error="Not configured (DATABASE_URL not set)",
        )

    sanitized = _sanitize_url(url)
    start = time.monotonic()

    try:
        from litellm_llmrouter.database import get_db_pool

        pool = await asyncio.wait_for(get_db_pool(url), timeout=5.0)
        if pool is None:
            return ServiceStatus(
                name=ExternalServices.POSTGRESQL.value,
                available=False,
                url=sanitized,
                error="Pool creation failed (asyncpg not installed?)",
            )

        async with pool.acquire() as conn:
            row = await asyncio.wait_for(conn.fetchrow("SELECT version()"), timeout=3.0)
            version = str(row[0]) if row else None
            latency = (time.monotonic() - start) * 1000

        return ServiceStatus(
            name=ExternalServices.POSTGRESQL.value,
            available=True,
            url=sanitized,
            latency_ms=round(latency, 1),
            version=version,
        )

    except asyncio.TimeoutError:
        return ServiceStatus(
            name=ExternalServices.POSTGRESQL.value,
            available=False,
            url=sanitized,
            latency_ms=round((time.monotonic() - start) * 1000, 1),
            error="Connection timeout",
        )
    except ImportError:
        return ServiceStatus(
            name=ExternalServices.POSTGRESQL.value,
            available=False,
            url=sanitized,
            error="asyncpg not installed",
        )
    except Exception as e:
        return ServiceStatus(
            name=ExternalServices.POSTGRESQL.value,
            available=False,
            url=sanitized,
            latency_ms=round((time.monotonic() - start) * 1000, 1),
            error=str(e)[:200],
        )


async def probe_redis(
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> ServiceStatus:
    """Probe Redis connectivity.

    Args:
        host: Redis host override. Falls back to REDIS_HOST env var.
        port: Redis port override. Falls back to REDIS_PORT env var.

    Returns:
        ServiceStatus for the Redis connection.
    """
    redis_host = host or os.getenv("REDIS_HOST")
    if not redis_host:
        return ServiceStatus(
            name=ExternalServices.REDIS.value,
            available=False,
            url="",
            error="Not configured (REDIS_HOST not set)",
        )

    redis_port = port or int(os.getenv("REDIS_PORT", "6379"))
    ssl = os.getenv("REDIS_SSL", "false").lower() in ("true", "1", "yes")
    scheme = "rediss" if ssl else "redis"
    display_url = f"{scheme}://{redis_host}:{redis_port}"

    start = time.monotonic()

    try:
        from litellm_llmrouter.redis_pool import get_async_redis_client

        client = await asyncio.wait_for(get_async_redis_client(), timeout=5.0)
        if client is None:
            return ServiceStatus(
                name=ExternalServices.REDIS.value,
                available=False,
                url=display_url,
                error="Client creation failed",
            )

        info = await asyncio.wait_for(client.info("server"), timeout=3.0)
        latency = (time.monotonic() - start) * 1000
        version = info.get("redis_version") if isinstance(info, dict) else None

        return ServiceStatus(
            name=ExternalServices.REDIS.value,
            available=True,
            url=display_url,
            latency_ms=round(latency, 1),
            version=version,
        )

    except asyncio.TimeoutError:
        return ServiceStatus(
            name=ExternalServices.REDIS.value,
            available=False,
            url=display_url,
            latency_ms=round((time.monotonic() - start) * 1000, 1),
            error="Connection timeout",
        )
    except ImportError:
        return ServiceStatus(
            name=ExternalServices.REDIS.value,
            available=False,
            url=display_url,
            error="redis package not installed",
        )
    except Exception as e:
        return ServiceStatus(
            name=ExternalServices.REDIS.value,
            available=False,
            url=display_url,
            latency_ms=round((time.monotonic() - start) * 1000, 1),
            error=str(e)[:200],
        )


async def probe_otel_collector(endpoint: Optional[str] = None) -> ServiceStatus:
    """Probe OTel collector endpoint via HTTP health check.

    Attempts a lightweight HTTP GET to the collector's health endpoint.
    The standard OTLP gRPC endpoint (4317) doesn't have a REST health
    check, so we probe the HTTP endpoint (4318) at /v1/traces with
    an empty OPTIONS-style request.

    Args:
        endpoint: OTLP endpoint override. Falls back to
                  OTEL_EXPORTER_OTLP_ENDPOINT env var.

    Returns:
        ServiceStatus for the OTel collector.
    """
    raw_endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not raw_endpoint:
        return ServiceStatus(
            name=ExternalServices.OTEL_COLLECTOR.value,
            available=False,
            url="",
            error="Not configured (OTEL_EXPORTER_OTLP_ENDPOINT not set)",
        )

    sanitized = _sanitize_url(raw_endpoint)
    start = time.monotonic()

    try:
        import httpx

        # Try to probe the health endpoint.  Many collectors expose a
        # /v1/health or simply respond to a GET on the base URL.
        # We try the base URL first — a connection success is enough.
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Strip trailing slashes and try a simple GET
            probe_url = raw_endpoint.rstrip("/")
            # For gRPC endpoints, the host may be reachable but not serve HTTP.
            # A successful TCP connect is sufficient evidence.
            resp = await client.get(probe_url)
            latency = (time.monotonic() - start) * 1000

            # Any response (even 404/405) means the collector is reachable
            return ServiceStatus(
                name=ExternalServices.OTEL_COLLECTOR.value,
                available=True,
                url=sanitized,
                latency_ms=round(latency, 1),
                version=resp.headers.get("server"),
            )

    except ImportError:
        # httpx not available — try a raw TCP connect
        try:
            parsed = urlparse(raw_endpoint)
            host = parsed.hostname or "localhost"
            port = parsed.port or (4317 if "grpc" in raw_endpoint.lower() else 4318)
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=3.0
            )
            writer.close()
            await writer.wait_closed()
            latency = (time.monotonic() - start) * 1000

            return ServiceStatus(
                name=ExternalServices.OTEL_COLLECTOR.value,
                available=True,
                url=sanitized,
                latency_ms=round(latency, 1),
            )
        except Exception as e:
            return ServiceStatus(
                name=ExternalServices.OTEL_COLLECTOR.value,
                available=False,
                url=sanitized,
                latency_ms=round((time.monotonic() - start) * 1000, 1),
                error=str(e)[:200],
            )

    except Exception as e:
        return ServiceStatus(
            name=ExternalServices.OTEL_COLLECTOR.value,
            available=False,
            url=sanitized,
            latency_ms=round((time.monotonic() - start) * 1000, 1),
            error=str(e)[:200],
        )


async def probe_oidc_provider(issuer_url: Optional[str] = None) -> ServiceStatus:
    """Probe OIDC provider's .well-known/openid-configuration.

    Args:
        issuer_url: OIDC issuer URL override. Falls back to
                    ROUTEIQ_OIDC_ISSUER_URL env var.

    Returns:
        ServiceStatus for the OIDC provider.
    """
    oidc_enabled = os.getenv("ROUTEIQ_OIDC_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    raw_url = issuer_url or os.getenv("ROUTEIQ_OIDC_ISSUER_URL")

    if not oidc_enabled or not raw_url:
        return ServiceStatus(
            name=ExternalServices.OIDC_PROVIDER.value,
            available=False,
            url="",
            error="Not configured (ROUTEIQ_OIDC_ENABLED=false or ROUTEIQ_OIDC_ISSUER_URL not set)",
        )

    sanitized = _sanitize_url(raw_url)
    well_known = raw_url.rstrip("/") + "/.well-known/openid-configuration"
    start = time.monotonic()

    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(well_known)
            latency = (time.monotonic() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                issuer = data.get("issuer", "")
                return ServiceStatus(
                    name=ExternalServices.OIDC_PROVIDER.value,
                    available=True,
                    url=sanitized,
                    latency_ms=round(latency, 1),
                    version=issuer,
                )
            else:
                return ServiceStatus(
                    name=ExternalServices.OIDC_PROVIDER.value,
                    available=False,
                    url=sanitized,
                    latency_ms=round(latency, 1),
                    error=f"HTTP {resp.status_code} from .well-known endpoint",
                )

    except ImportError:
        return ServiceStatus(
            name=ExternalServices.OIDC_PROVIDER.value,
            available=False,
            url=sanitized,
            error="httpx not installed",
        )
    except Exception as e:
        return ServiceStatus(
            name=ExternalServices.OIDC_PROVIDER.value,
            available=False,
            url=sanitized,
            latency_ms=round((time.monotonic() - start) * 1000, 1),
            error=str(e)[:200],
        )


# ---------------------------------------------------------------------------
# Aggregate probing
# ---------------------------------------------------------------------------


async def probe_all_services() -> Dict[str, ServiceStatus]:
    """Probe all configured external services and return their status.

    Runs all probes concurrently for minimal startup latency.

    Returns:
        Dict mapping service name to its ServiceStatus.
    """
    global _last_probe

    results = await asyncio.gather(
        probe_postgresql(),
        probe_redis(),
        probe_otel_collector(),
        probe_oidc_provider(),
        return_exceptions=True,
    )

    probes = {}
    service_names = [
        ExternalServices.POSTGRESQL.value,
        ExternalServices.REDIS.value,
        ExternalServices.OTEL_COLLECTOR.value,
        ExternalServices.OIDC_PROVIDER.value,
    ]

    for name, result in zip(service_names, results):
        if isinstance(result, ServiceStatus):
            probes[name] = result
        elif isinstance(result, Exception):
            probes[name] = ServiceStatus(
                name=name,
                available=False,
                url="",
                error=f"Probe error: {result!s}"[:200],
            )
        else:
            probes[name] = ServiceStatus(
                name=name,
                available=False,
                url="",
                error="Unknown probe result",
            )

    # Cache results
    async with _get_last_probe_lock():
        _last_probe = dict(probes)

    return probes


# ---------------------------------------------------------------------------
# Feature availability
# ---------------------------------------------------------------------------


def get_feature_availability(
    services: Optional[Dict[str, ServiceStatus]] = None,
) -> Dict[str, bool]:
    """Return which features are available based on service status.

    If *services* is not provided, uses the last cached probe results.

    Returns:
        Dict mapping feature name to availability boolean.
    """
    svc = services or _last_probe

    pg_available = svc.get(
        ExternalServices.POSTGRESQL.value, ServiceStatus("", False, "")
    ).available
    redis_available = svc.get(
        ExternalServices.REDIS.value, ServiceStatus("", False, "")
    ).available
    otel_available = svc.get(
        ExternalServices.OTEL_COLLECTOR.value, ServiceStatus("", False, "")
    ).available
    oidc_available = svc.get(
        ExternalServices.OIDC_PROVIDER.value, ServiceStatus("", False, "")
    ).available

    # Check ML model artifacts
    models_dir = os.getenv("LLMROUTER_MODELS_PATH", "models")
    ml_routing = False
    try:
        if os.path.isdir(models_dir):
            # Check for any .pkl, .pt, .onnx, or .joblib files
            for entry in os.listdir(models_dir):
                if entry.endswith((".pkl", ".pt", ".onnx", ".joblib", ".json")):
                    ml_routing = True
                    break
    except OSError:
        pass

    # Check centroid routing vectors
    centroids_dir = os.path.join(models_dir, "centroids")
    centroid_routing = os.getenv("ROUTEIQ_CENTROID_ROUTING", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    # Centroid routing can work without pre-computed vectors (builds them lazily)
    # but we note if vectors are pre-loaded
    centroids_preloaded = False
    try:
        if os.path.isdir(centroids_dir) and os.listdir(centroids_dir):
            centroids_preloaded = True
    except OSError:
        pass

    return {
        "persistence": pg_available,
        "caching": redis_available,
        "ha_leader_election": redis_available,
        "observability": otel_available,
        "sso_login": oidc_available,
        "ml_routing": ml_routing,
        "centroid_routing": centroid_routing,
        "centroid_vectors_preloaded": centroids_preloaded,
    }


# ---------------------------------------------------------------------------
# Startup display formatting
# ---------------------------------------------------------------------------


def format_service_status_table(
    services: Dict[str, ServiceStatus],
    features: Dict[str, bool],
) -> str:
    """Format service status as a terminal-friendly table.

    Returns a multi-line string suitable for startup logging.
    """
    # Column widths
    name_width = 14
    status_width = 20
    latency_width = 11

    lines = []
    lines.append("")
    lines.append("RouteIQ Gateway — Service Status")
    lines.append(
        f"{'':>2}{'Service':<{name_width}} {'Status':<{status_width}} {'Latency':<{latency_width}}"
    )
    lines.append(
        f"{'':>2}{'-' * name_width} {'-' * status_width} {'-' * latency_width}"
    )

    display_names = {
        ExternalServices.POSTGRESQL.value: "PostgreSQL",
        ExternalServices.REDIS.value: "Redis",
        ExternalServices.OTEL_COLLECTOR.value: "OTel Collector",
        ExternalServices.OIDC_PROVIDER.value: "OIDC Provider",
    }

    for svc_key in [
        ExternalServices.POSTGRESQL.value,
        ExternalServices.REDIS.value,
        ExternalServices.OTEL_COLLECTOR.value,
        ExternalServices.OIDC_PROVIDER.value,
    ]:
        status = services.get(svc_key)
        if status is None:
            continue

        display_name = display_names.get(svc_key, svc_key)

        if status.available:
            status_str = "OK"
            latency_str = (
                f"{status.latency_ms:.0f}ms" if status.latency_ms is not None else ""
            )
        elif status.error and "Not configured" in status.error:
            status_str = "Not configured"
            latency_str = ""
        else:
            status_str = f"FAIL: {status.error or 'unknown'}"[:status_width]
            latency_str = (
                f"{status.latency_ms:.0f}ms" if status.latency_ms is not None else ""
            )

        lines.append(
            f"  {display_name:<{name_width}} {status_str:<{status_width}} {latency_str:<{latency_width}}"
        )

    # Feature summary
    feature_parts = []
    feature_labels = {
        "persistence": "persistence",
        "caching": "caching",
        "ha_leader_election": "ha",
        "observability": "observability",
        "sso_login": "sso",
        "ml_routing": "ml_routing",
        "centroid_routing": "centroid",
    }

    for key, label in feature_labels.items():
        val = features.get(key, False)
        feature_parts.append(f"{label}={'yes' if val else 'no'}")

    lines.append("")
    lines.append(f"  Features: {' '.join(feature_parts)}")
    lines.append("")

    return "\n".join(lines)
