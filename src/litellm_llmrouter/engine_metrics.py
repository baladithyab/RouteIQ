"""
Engine-Side Metrics Scrape (self-hosted serving engines)
========================================================

Scrapes the Prometheus ``/metrics`` exposition of a **self-hosted inference
engine** (vLLM, vLLM Production Stack, AIBrix, llm-d — anything that exports the
``vllm:*`` metric family) so RouteIQ can read the engine's queue depth and
KV-cache pressure. Those signals are what a *future* KV/queue-aware Layer-1
router (or an autoscaler-into-the-engine) consumes.

Layering (see ``docs/architecture/aws-rearchitecture/51-multinode-large-model-serving.md``
Part 3): RouteIQ does **Layer-1 model selection**; the engine does **Layer-2
replica scheduling** below one OpenAI ``api_base``. This module does NOT reach
inside that boundary to pick a replica — it reads the engine *frontend's*
aggregate ``/metrics`` so Layer-1 can decide *whether to send more load to this
engine arm at all* (or autoscale it). It never scrapes individual worker pods.

Design constraints (mirrors ``bedrock_discovery`` / ``service_discovery``):
- **Cred-free.** A plain ``httpx`` GET of the engine ``/metrics`` URL; the
  in-cluster engine is unauthenticated (the ``fake-api-key`` sentinel). No AWS
  creds, no boto3, no secrets.
- **Default OFF.** ``ROUTEIQ_ENGINE_METRICS__ENABLED=false`` by default. With it
  off, nothing scrapes and importing this module is a no-op (no I/O on import).
- **Graceful on unreachable.** A down/timed-out/garbage engine yields an EMPTY
  snapshot (``EngineMetricsSnapshot(reachable=False, gauges={})``), never an
  exception that could crash a routing decision or a probe.

The parser is a tolerant subset of the Prometheus text exposition format
(https://prometheus.io/docs/instrumenting/exposition_formats/) — enough to read
the scalar gauges/counters RouteIQ cares about, not a full OpenMetrics parser.

Usage::

    from litellm_llmrouter.engine_metrics import get_engine_metrics_scraper

    scraper = get_engine_metrics_scraper()
    snap = await scraper.scrape("http://aibrix-gateway.aibrix-system.svc:8000/metrics")
    if snap.reachable:
        waiting = snap.gauges.get("vllm:num_requests_waiting")
        kv_used = snap.kv_cache_usage()   # handles the v0/v1 rename
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# The gauges a KV/queue-aware router (or engine autoscaler) actually wants.
# =============================================================================
#
# vLLM v1 RENAMED ``vllm:gpu_cache_usage_perc`` -> ``vllm:kv_cache_usage_perc``
# (verified against vllm-project/vllm v1 metrics: ``kv_cache_usage_perc`` is the
# v1 Gauge "Fraction of used KV cache blocks (0-1)"; the old ``gpu_cache_usage_perc``
# does not exist in v1). We accept BOTH so the scraper works across engine
# versions, and ``kv_cache_usage()`` returns whichever is present.
KV_CACHE_USAGE_V1 = "vllm:kv_cache_usage_perc"
KV_CACHE_USAGE_V0 = "vllm:gpu_cache_usage_perc"

# The scalar metrics we parse out (a small allowlist keeps the snapshot tight
# and avoids retaining the engine's entire — possibly large — metric surface).
# All are vLLM-native names; vLLM Production Stack, AIBrix, and llm-d all serve
# vLLM underneath and re-export this family.
ENGINE_GAUGE_NAMES: frozenset[str] = frozenset(
    {
        "vllm:num_requests_waiting",  # Gauge: requests queued (queue depth)
        "vllm:num_requests_running",  # Gauge: requests in the running batch
        "vllm:num_requests_swapped",  # Gauge: requests swapped to CPU
        KV_CACHE_USAGE_V1,  # Gauge: KV-cache fraction used (v1)
        KV_CACHE_USAGE_V0,  # Gauge: GPU-cache fraction used (pre-v1 alias)
        "vllm:prefix_cache_hits",  # Counter (v1): cumulative prefix-cache hits
        "vllm:prefix_cache_queries",  # Counter (v1): cumulative prefix-cache queries
        "vllm:gpu_prefix_cache_hit_rate",  # Gauge: hit rate (Production-Stack router export)
        "vllm:gpu_cache_usage_perc",  # (alias above, kept for set-membership clarity)
    }
)


# =============================================================================
# Parsed snapshot
# =============================================================================
@dataclass(frozen=True)
class EngineMetricsSnapshot:
    """One scrape of an engine ``/metrics`` endpoint.

    ``reachable=False`` + empty ``gauges`` is the graceful-failure shape returned
    for any unreachable / timed-out / unparseable engine. Callers branch on
    ``reachable`` (or simply ``.get(name)`` which is ``None`` when absent), never
    on an exception.
    """

    endpoint: str
    reachable: bool
    gauges: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def get(self, name: str) -> Optional[float]:
        """Return a parsed metric value, or ``None`` if absent."""
        return self.gauges.get(name)

    def kv_cache_usage(self) -> Optional[float]:
        """KV-cache fraction-used (0..1), resolving the v1 rename.

        Prefers the v1 name (``vllm:kv_cache_usage_perc``) and falls back to the
        pre-v1 ``vllm:gpu_cache_usage_perc`` so the same caller works across
        engine versions.
        """
        v1 = self.gauges.get(KV_CACHE_USAGE_V1)
        if v1 is not None:
            return v1
        return self.gauges.get(KV_CACHE_USAGE_V0)

    def num_waiting(self) -> Optional[float]:
        """Queue depth — the primary queue-aware-routing signal."""
        return self.gauges.get("vllm:num_requests_waiting")


# =============================================================================
# Prometheus text-exposition parser (tolerant subset)
# =============================================================================
def parse_prometheus_metrics(
    body: str,
    wanted: frozenset[str] = ENGINE_GAUGE_NAMES,
) -> dict[str, float]:
    """Parse scalar gauges/counters from a Prometheus text-exposition body.

    Tolerant subset of the exposition format:
    - Skips ``# HELP`` / ``# TYPE`` comment lines and blank lines.
    - A sample line is ``<name>[{labels}] <value> [timestamp]``. We read the
      metric name (up to ``{`` or whitespace), keep it only if in ``wanted``,
      and parse the value token. Trailing timestamps are ignored.
    - **Labelled series are collapsed**: for a ``wanted`` name with labels (e.g.
      per-``model_name`` series in the Production-Stack router export) we keep the
      LAST sample seen. Engine-frontend exports are aggregate / single-series for
      these names, so this is the right read; a future per-model router can pass a
      narrower ``wanted`` or parse labels itself.
    - Prometheus value tokens ``NaN`` / ``+Inf`` / ``-Inf`` are dropped (not
      useful routing signals); malformed value tokens are skipped, not raised.

    Returns a (possibly empty) ``{name: float}`` dict. Never raises on bad input.
    """
    out: dict[str, float] = {}
    if not body:
        return out

    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Split metric token from the rest (value [timestamp]).
        # The name ends at the first '{' (labels) or whitespace.
        brace = line.find("{")
        ws = line.find(" ")
        if brace != -1 and (ws == -1 or brace < ws):
            name = line[:brace]
            close = line.find("}", brace)
            if close == -1:
                continue  # malformed label set; skip
            rest = line[close + 1 :].strip()
        elif ws != -1:
            name = line[:ws]
            rest = line[ws + 1 :].strip()
        else:
            continue  # no value token

        if name not in wanted:
            continue

        # value is the first whitespace-delimited token of ``rest``
        value_tok = rest.split()[0] if rest else ""
        lowered = value_tok.lower()
        if lowered in ("nan", "+inf", "-inf", "inf"):
            continue
        try:
            out[name] = float(value_tok)
        except ValueError:
            continue  # unparseable value; skip this sample, keep going

    return out


# =============================================================================
# Scraper
# =============================================================================
class EngineMetricsScraper:
    """Credless HTTP scraper for self-hosted-engine ``/metrics``.

    Holds no engine state and no connection; each ``scrape()`` is a one-shot GET.
    Default-off via :class:`EngineMetricsSettings`. The scraper is constructed
    lazily by :func:`get_engine_metrics_scraper` and reset by
    :func:`reset_engine_metrics_scraper` (tests).
    """

    def __init__(self, *, enabled: bool, timeout: float) -> None:
        self._enabled = enabled
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def scrape(self, endpoint: str) -> EngineMetricsSnapshot:
        """GET ``endpoint`` and parse the engine gauges.

        Always returns an :class:`EngineMetricsSnapshot`:
        - default-off -> ``reachable=False, error="engine metrics scrape disabled"``
        - unreachable / non-200 / timeout / bad body -> ``reachable=False`` with a
          short ``error`` string; ``gauges={}``.
        - success -> ``reachable=True`` with the parsed allowlisted gauges.
        """
        if not self._enabled:
            return EngineMetricsSnapshot(
                endpoint=endpoint,
                reachable=False,
                error="engine metrics scrape disabled",
            )

        body = await self._fetch(endpoint)
        if body is None:
            # _fetch already logged the cause; return the graceful-empty shape.
            return EngineMetricsSnapshot(
                endpoint=endpoint,
                reachable=False,
                error="engine unreachable",
            )

        gauges = parse_prometheus_metrics(body)
        return EngineMetricsSnapshot(
            endpoint=endpoint,
            reachable=True,
            gauges=gauges,
        )

    async def _fetch(self, endpoint: str) -> Optional[str]:
        """Issue the GET. Returns the body text, or ``None`` on any failure.

        Uses the shared pooled httpx client when available (per
        ``http_client_pool``), falling back to a per-request client otherwise —
        the same pattern other outbound callers use. All exceptions are caught
        and turned into ``None`` so a scrape can never crash the caller.
        """
        try:
            import httpx

            from litellm_llmrouter import http_client_pool

            if http_client_pool.is_pooling_enabled():
                client = http_client_pool.get_http_client()
                resp = await client.get(endpoint, timeout=self._timeout)
            else:
                async with http_client_pool.create_fallback_client(
                    timeout=self._timeout
                ) as client:
                    resp = await client.get(endpoint)

            if resp.status_code != 200:
                logger.debug(
                    "engine metrics scrape: %s returned HTTP %s",
                    endpoint,
                    resp.status_code,
                )
                return None
            return resp.text
        except httpx.HTTPError as exc:  # timeouts, connect errors, etc.
            logger.debug("engine metrics scrape: %s unreachable: %s", endpoint, exc)
            return None
        except Exception as exc:  # belt-and-suspenders: never crash a scrape
            logger.debug("engine metrics scrape: %s failed: %s", endpoint, exc)
            return None


# =============================================================================
# Singleton (reset_* for tests, per the RouteIQ singleton convention)
# =============================================================================
_scraper: Optional[EngineMetricsScraper] = None


def get_engine_metrics_scraper() -> EngineMetricsScraper:
    """Return the process-wide :class:`EngineMetricsScraper`, constructing it from
    :class:`litellm_llmrouter.settings.GatewaySettings` on first use."""
    global _scraper
    if _scraper is None:
        try:
            from litellm_llmrouter.settings import get_settings

            cfg = get_settings().engine_metrics
            _scraper = EngineMetricsScraper(
                enabled=cfg.enabled,
                timeout=cfg.scrape_timeout,
            )
        except Exception:  # settings unavailable -> safe default-off scraper
            _scraper = EngineMetricsScraper(enabled=False, timeout=2.0)
    return _scraper


def reset_engine_metrics_scraper() -> None:
    """Reset the singleton (tests; ``autouse`` reset fixture)."""
    global _scraper
    _scraper = None
