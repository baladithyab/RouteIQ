"""
Router Decision Telemetry for TG4.1/TG4.2
==========================================

This module provides router decision telemetry emission for TG4.1 acceptance criteria.

Two mechanisms are provided:
1. RouterDecisionMiddleware - FastAPI middleware that emits router.* attributes on
   ALL LLM API requests (deterministic, works with any routing strategy)
2. RouterDecisionCallback - LiteLLM callback that emits metrics and span attributes

The middleware is the preferred approach for E2E testing because it:
- Fires before the LLM API call, so mock API keys don't prevent telemetry
- Works with LiteLLM's built-in routing strategies (simple-shuffle, etc.)
- Doesn't require trained ML models

Usage:
    # In gateway/app.py:
    from litellm_llmrouter.router_decision_callback import RouterDecisionMiddleware
    app.add_middleware(RouterDecisionMiddleware)

Environment Variables:
    LLMROUTER_ROUTER_CALLBACK_ENABLED: Enable the middleware (default: true if OTEL configured)
    LLMROUTER_ROUTER_CALLBACK_STRATEGY: Override strategy name in telemetry
"""

import logging
import os
import random
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Strategy name used by the centroid router (centroid decision identification).
_CENTROID_STRATEGY_NAME = "llmrouter-nadirclaw-centroid"

# Cap on the response text captured into an EvalSample (COLLECT arm).  Keeps the
# captured pair cheap; the LLM-as-judge re-truncates downstream regardless.
_EVAL_RESPONSE_CONTENT_CAP = 8192

# Feature flag: Enable router decision callback
# Default to true if OTEL is configured, false otherwise
_OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
ROUTER_CALLBACK_ENABLED = (
    os.getenv(
        "LLMROUTER_ROUTER_CALLBACK_ENABLED", "true" if _OTEL_ENDPOINT else "false"
    ).lower()
    == "true"
)

# Override strategy name in telemetry (useful for testing)
OVERRIDE_STRATEGY_NAME = os.getenv(
    "LLMROUTER_ROUTER_CALLBACK_STRATEGY", "simple-shuffle"
)


def _governance_spend_tracking_env() -> bool:
    """Raw-env reader for the spend-tracking gate (graceful fallback path).

    Env ``LLMROUTER_GOVERNANCE_SPEND_TRACKING`` (true/1/yes/on); default ON.
    """
    return os.getenv("LLMROUTER_GOVERNANCE_SPEND_TRACKING", "true").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


def _governance_spend_tracking_enabled() -> bool:
    """Whether post-response governance spend tracking is enabled (default ON).

    This is INDEPENDENT of ROUTER_CALLBACK_ENABLED (which is OTEL-gated): the
    spend write path is the WRITER for the governance budget / rpm counters and
    must work even when OTEL telemetry is off.

    RouteIQ-9f9f: settings-first per ADR-0013 -- reads
    ``get_settings().llmrouter_governance_spend_tracking`` (bound via
    ``validation_alias`` to the same ``LLMROUTER_GOVERNANCE_SPEND_TRACKING`` env
    + default).  Falls back to the raw env read if ``get_settings()`` throws,
    matching the module's other settings-first-with-env-fallback patterns.

    RESOLUTION TIMING (RouteIQ-9fce): the flag is a BOOT-TIME setting per
    ADR-0013, NOT a per-request toggle.  Although this function is called fresh
    on every success event, ``get_settings()`` returns the process-wide cached
    Settings instance built once at startup -- so flipping the
    ``LLMROUTER_GOVERNANCE_SPEND_TRACKING`` env var mid-process has NO effect on
    a running gateway.  To change it: set the env var and RESTART the process
    (production), or call ``reset_settings()`` after monkeypatching the env in
    tests so the next ``get_settings()`` rebuilds with the new value.  (The
    raw-env fallback only fires when ``get_settings()`` raises, which does not
    happen on the normal path, so it is NOT a live-toggle escape hatch.)
    """
    try:
        from litellm_llmrouter.settings import get_settings

        return bool(get_settings().llmrouter_governance_spend_tracking)
    except Exception:
        return _governance_spend_tracking_env()


# =============================================================================
# In-process Routing Stats Accumulator (RouteIQ-aba9)
# =============================================================================

# Bound on the number of distinct per-key / per-user / per-model rollup entries
# retained.  Keeps the accumulator's memory footprint constant under high key
# cardinality (eviction is oldest-first, LRU-ish on insert order).
_DEFAULT_MAX_ROLLUP_ENTRIES = 10_000

# Bound on how many recent model names are remembered per key (for /me/stats).
_MAX_RECENT_MODELS_PER_KEY = 10


def _stats_max_rollup_entries() -> int:
    """Resolve the per-rollup-map size cap (settings-first, env fallback).

    Reads ``ROUTEIQ_STATS_MAX_ROLLUP_ENTRIES`` to let operators bound the
    accumulator under pathological key cardinality.  Defaults to
    ``_DEFAULT_MAX_ROLLUP_ENTRIES``.  Never raises -- a bad value falls back to
    the default so the hot path is unaffected.
    """
    try:
        raw = os.getenv("ROUTEIQ_STATS_MAX_ROLLUP_ENTRIES")
        if raw is None:
            return _DEFAULT_MAX_ROLLUP_ENTRIES
        value = int(raw)
        return value if value > 0 else _DEFAULT_MAX_ROLLUP_ENTRIES
    except Exception:
        return _DEFAULT_MAX_ROLLUP_ENTRIES


class _KeyRollup:
    """Per-key usage rollup (process-local).

    Holds the decision count and a bounded recent-models list for ONE resolved
    key_id.  Mutated only under :class:`RoutingStatsAccumulator`'s lock.
    """

    __slots__ = ("decisions", "recent_models")

    def __init__(self) -> None:
        self.decisions: int = 0
        # Most-recent-last; bounded to _MAX_RECENT_MODELS_PER_KEY distinct names.
        self.recent_models: "OrderedDict[str, None]" = OrderedDict()

    def record(self, model: str) -> None:
        self.decisions += 1
        if model:
            # Move-to-end so the list reflects recency; cap the size.
            self.recent_models.pop(model, None)
            self.recent_models[model] = None
            while len(self.recent_models) > _MAX_RECENT_MODELS_PER_KEY:
                self.recent_models.popitem(last=False)


class RoutingStatsAccumulator:
    """Thread-safe, bounded, process-local routing-decision counter store.

    Fed by :class:`RouterDecisionCallback` / :class:`RouterDecisionMiddleware`
    on every routing decision.  Read back by the stats API endpoints
    (admin global stats + the caller-scoped ``/me/stats``).

    All mutation/read goes through a single :class:`threading.Lock` so it is
    safe to feed from sync callbacks and read from async route handlers.  The
    per-key / per-user / per-model maps are bounded (oldest-evicted) so memory
    stays constant regardless of key cardinality.

    This is a SINGLETON (see :func:`get_stats_accumulator`); tests MUST call
    :func:`reset_stats_accumulator` in an autouse fixture.

    .. warning:: **Per-worker (process-local) scope — counts are NOT cluster-wide.**

        This accumulator lives in a single Python process.  Each uvicorn worker
        and each replica (``replicaCount > 1`` / ``--workers N``) keeps its OWN
        instance, and the load balancer fans requests across all of them.  So
        :meth:`global_snapshot` (``/stats/global``) and :meth:`key_snapshot`
        (``/me/stats``) report ONLY the decisions that landed on the serving
        worker for that scrape — under multiple workers/replicas they
        **undercount** the true cluster total, and successive reads can hit
        different workers and return different numbers.

        These endpoints are a convenience / debug view, NOT the cluster-wide
        source of truth.  The authoritative cluster-wide counts come from the
        metrics backend: the RouteIQ OTel instruments (``metrics.py``) are pushed
        to the OTLP collector and exposed at ``/metrics`` (Prometheus/AMP), where
        the scraper aggregates across every worker and replica.  See
        ``docs/operations/observability.md`` ("Routing-stats endpoints are
        per-worker").  A shared-store backing (Redis/Aurora) for true cluster-wide
        stats endpoints is deliberately out of scope here (tracked separately).
    """

    def __init__(self, max_entries: Optional[int] = None) -> None:
        self._lock = threading.Lock()
        self._max_entries = max_entries or _stats_max_rollup_entries()
        self._total_decisions: int = 0
        self._centroid_decisions: int = 0
        self._latency_sum_ms: float = 0.0
        self._latency_count: int = 0
        self._strategy_counts: Dict[str, int] = {}
        self._profile_counts: Dict[str, int] = {}
        self._model_counts: "OrderedDict[str, int]" = OrderedDict()
        self._key_rollups: "OrderedDict[str, _KeyRollup]" = OrderedDict()
        self._user_counts: "OrderedDict[str, int]" = OrderedDict()

    @staticmethod
    def _bump_bounded(
        store: "OrderedDict[str, int]", key: str, max_entries: int
    ) -> None:
        """Increment ``store[key]`` with oldest-first eviction past *max_entries*."""
        store[key] = store.get(key, 0) + 1
        while len(store) > max_entries:
            store.popitem(last=False)

    def record_decision(
        self,
        *,
        strategy: Optional[str] = None,
        model: Optional[str] = None,
        profile: Optional[str] = None,
        latency_ms: Optional[float] = None,
        key_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Record a single routing decision.

        Every argument is optional so partial telemetry (e.g. the middleware
        path, which has no resolved model) still increments the global total.
        Never raises -- the caller wraps it, but we also guard internally so a
        bad value cannot break the request path.
        """
        try:
            with self._lock:
                self._total_decisions += 1

                if strategy:
                    self._strategy_counts[strategy] = (
                        self._strategy_counts.get(strategy, 0) + 1
                    )
                    if strategy == _CENTROID_STRATEGY_NAME:
                        self._centroid_decisions += 1

                if profile:
                    self._profile_counts[profile] = (
                        self._profile_counts.get(profile, 0) + 1
                    )

                if latency_ms is not None and latency_ms >= 0:
                    self._latency_sum_ms += float(latency_ms)
                    self._latency_count += 1

                if model:
                    self._bump_bounded(self._model_counts, model, self._max_entries)

                if key_id:
                    rollup = self._key_rollups.get(key_id)
                    if rollup is None:
                        rollup = _KeyRollup()
                        self._key_rollups[key_id] = rollup
                        while len(self._key_rollups) > self._max_entries:
                            self._key_rollups.popitem(last=False)
                    rollup.record(model or "")

                if user_id:
                    self._bump_bounded(self._user_counts, user_id, self._max_entries)
        except Exception:  # pragma: no cover - stats must never break routing
            logger.debug(
                "RoutingStatsAccumulator.record_decision failed", exc_info=True
            )

    def record_latency(self, latency_ms: float) -> None:
        """Record a routing/response latency sample (decoupled from decisions).

        Fed from the post-response success path where the REAL duration is
        known, so the average reflects actual latency rather than the pre-call
        placeholder.  Does NOT increment the decision counter.
        """
        try:
            if latency_ms is None or latency_ms < 0:
                return
            with self._lock:
                self._latency_sum_ms += float(latency_ms)
                self._latency_count += 1
        except Exception:  # pragma: no cover - stats must never break routing
            logger.debug("RoutingStatsAccumulator.record_latency failed", exc_info=True)

    def average_latency_ms(self) -> float:
        with self._lock:
            if self._latency_count == 0:
                return 0.0
            return self._latency_sum_ms / self._latency_count

    def global_snapshot(self) -> Dict[str, Any]:
        """Return a copy of the global aggregates (admin / dashboard view).

        PER-WORKER SCOPE: these aggregates cover only the decisions seen by THIS
        process.  Under multiple uvicorn workers / ``replicaCount > 1`` the
        returned totals undercount the cluster; the cluster-wide source of truth
        is the metrics backend (Prometheus/AMP scrape of ``/metrics``).  See the
        class docstring and ``docs/operations/observability.md``.
        """
        with self._lock:
            avg = (
                self._latency_sum_ms / self._latency_count
                if self._latency_count
                else 0.0
            )
            return {
                "total_decisions": self._total_decisions,
                "strategy_distribution": dict(self._strategy_counts),
                "profile_distribution": dict(self._profile_counts),
                "centroid_decisions": self._centroid_decisions,
                "average_latency_ms": avg,
                "model_distribution": dict(self._model_counts),
                "key_distribution": {
                    k: r.decisions for k, r in self._key_rollups.items()
                },
                "tracked_keys": len(self._key_rollups),
            }

    def key_snapshot(self, key_id: str) -> Dict[str, Any]:
        """Return the per-key usage rollup for *key_id* (caller-scoped view).

        Returns zeroed counters (not an error) when the key has no recorded
        decisions yet, so a freshly-issued key still gets a valid response.

        PER-WORKER SCOPE: the rollup reflects only the requests for *key_id* that
        this process served, so under multiple workers / replicas the caller's
        ``/me/stats`` undercounts their true usage and may vary between reads.
        For authoritative per-key/cluster-wide usage consult the metrics backend
        (or the governance spend counters).  See the class docstring.
        """
        with self._lock:
            rollup = self._key_rollups.get(key_id)
            if rollup is None:
                return {"decisions": 0, "recent_models": []}
            # recent_models is most-recent-last; surface most-recent-first.
            return {
                "decisions": rollup.decisions,
                "recent_models": list(reversed(rollup.recent_models.keys())),
            }


# Singleton --------------------------------------------------------------------

_stats_accumulator: Optional[RoutingStatsAccumulator] = None
_stats_accumulator_lock = threading.Lock()


def get_stats_accumulator() -> RoutingStatsAccumulator:
    """Get or create the process-local routing stats accumulator singleton."""
    global _stats_accumulator
    if _stats_accumulator is None:
        with _stats_accumulator_lock:
            if _stats_accumulator is None:
                _stats_accumulator = RoutingStatsAccumulator()
    return _stats_accumulator


def reset_stats_accumulator() -> None:
    """Reset the stats accumulator singleton (for tests)."""
    global _stats_accumulator
    with _stats_accumulator_lock:
        _stats_accumulator = None


# =============================================================================
# Cluster-wide shared stats store (RouteIQ-78fd)
# =============================================================================
#
# The :class:`RoutingStatsAccumulator` above is PER-WORKER: each uvicorn worker
# / replica keeps its own copy, so the admin Routing-stats panel only ever shows
# the worker that served the scrape.  This store layers a Redis-backed
# *cluster-wide* aggregate on top: every worker mirrors its decision counters
# into shared Redis keys (write-through), and the admin ``/stats/global``
# endpoint reads the SUM across all workers back out.
#
# Lower-risk-than-Prometheus design (per the seed): reuse the existing
# ``redis_pool`` async client.  When Redis is unavailable the cluster snapshot
# falls back to the in-memory per-worker snapshot, so the panel still renders
# (degraded to one worker) rather than erroring.

# Redis key namespace + TTL for the shared counters.  Counters are refreshed on
# every write so an idle cluster eventually expires the keys (avoids unbounded
# growth of per-strategy / per-model fields after a config change).
_CLUSTER_STATS_PREFIX = "routeiq:stats:cluster"
_CLUSTER_STATS_TTL_SECONDS = 7 * 24 * 3600  # 7 days; refreshed on every write


def _cluster_stats_enabled() -> bool:
    """Whether cluster-wide stats write-through is enabled (default ON).

    Reads ``ROUTEIQ_CLUSTER_STATS_ENABLED`` (true/1/yes/on).  When OFF the
    accumulator does NOT mirror to Redis and the global endpoint serves the
    in-memory per-worker snapshot (the historical behaviour).  Never raises.
    """
    return os.getenv("ROUTEIQ_CLUSTER_STATS_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


def _k(*parts: str) -> str:
    """Build a namespaced cluster-stats Redis key."""
    return ":".join((_CLUSTER_STATS_PREFIX, *parts))


async def mirror_decision_to_cluster(
    *,
    strategy: Optional[str] = None,
    model: Optional[str] = None,
    profile: Optional[str] = None,
    key_id: Optional[str] = None,
    centroid: bool = False,
) -> bool:
    """Write-through ONE routing decision into the shared cluster counters.

    Mirrors the same dimensions the in-memory accumulator tracks into shared
    Redis keys so every worker contributes to a single cluster-wide aggregate:

      * ``<prefix>:totals``  -- a hash with ``total_decisions`` /
        ``centroid_decisions`` fields,
      * ``<prefix>:strategy`` / ``:profile`` / ``:model`` -- hashes of per-name
        counts,
      * ``<prefix>:keys``    -- a hash of per-key decision counts (for
        ``tracked_keys`` / ``key_distribution``).

    Uses a pipeline so the mirror is one network round-trip per decision and
    every touched key gets its TTL refreshed.  Fully fail-open: returns
    ``False`` (and never raises) when Redis is absent/disabled so the request
    path is never broken by cluster bookkeeping.
    """
    if not _cluster_stats_enabled():
        return False
    try:
        from litellm_llmrouter.redis_pool import get_async_redis_client

        redis = await get_async_redis_client()
        if redis is None:
            return False

        totals_key = _k("totals")
        pipe = redis.pipeline()
        pipe.hincrby(totals_key, "total_decisions", 1)
        if centroid:
            pipe.hincrby(totals_key, "centroid_decisions", 1)
        pipe.expire(totals_key, _CLUSTER_STATS_TTL_SECONDS)

        if strategy:
            sk = _k("strategy")
            pipe.hincrby(sk, strategy, 1)
            pipe.expire(sk, _CLUSTER_STATS_TTL_SECONDS)
        if profile:
            pk = _k("profile")
            pipe.hincrby(pk, profile, 1)
            pipe.expire(pk, _CLUSTER_STATS_TTL_SECONDS)
        if model:
            mk = _k("model")
            pipe.hincrby(mk, model, 1)
            pipe.expire(mk, _CLUSTER_STATS_TTL_SECONDS)
        if key_id:
            kk = _k("keys")
            pipe.hincrby(kk, key_id, 1)
            pipe.expire(kk, _CLUSTER_STATS_TTL_SECONDS)

        await pipe.execute()
        return True
    except Exception:  # pragma: no cover - cluster stats must never break routing
        logger.debug("Failed to mirror decision to cluster store", exc_info=True)
        return False


def _coerce_int_hash(raw: Any) -> Dict[str, int]:
    """Coerce a Redis HGETALL result into a ``{str: int}`` map.

    Tolerates bytes keys/values (``decode_responses=False`` clients) and
    non-numeric junk (skipped), so a malformed field never breaks the read.
    """
    out: Dict[str, int] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            name = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            out[name] = int(v)
        except Exception:
            continue
    return out


async def cluster_global_snapshot() -> Dict[str, Any]:
    """Return the CLUSTER-WIDE global aggregate, or the per-worker fallback.

    Reads the shared Redis counters (summed across every worker that mirrored
    into them) and shapes them into the SAME dict ``global_snapshot`` returns,
    so the admin ``/stats/global`` endpoint is a drop-in.  Two fields cannot be
    reconstructed cluster-wide from the counters and are taken from the local
    accumulator as a best-effort overlay:

      * ``profile_distribution`` is cluster-wide (mirrored), and
      * ``average_latency_ms`` is LOCAL only (latency is a per-worker rolling
        mean that does not sum), so it reflects the serving worker.

    Fail-open: when cluster stats are disabled or Redis is unavailable this
    returns the local :meth:`RoutingStatsAccumulator.global_snapshot` verbatim,
    annotated with ``cluster_wide=False`` so the caller/UI can show scope.
    """
    local = get_stats_accumulator().global_snapshot()
    local["cluster_wide"] = False

    if not _cluster_stats_enabled():
        return local

    try:
        from litellm_llmrouter.redis_pool import get_async_redis_client

        redis = await get_async_redis_client()
        if redis is None:
            return local

        totals = _coerce_int_hash(await redis.hgetall(_k("totals")))
        strategy = _coerce_int_hash(await redis.hgetall(_k("strategy")))
        profile = _coerce_int_hash(await redis.hgetall(_k("profile")))
        model = _coerce_int_hash(await redis.hgetall(_k("model")))
        keys = _coerce_int_hash(await redis.hgetall(_k("keys")))

        total_decisions = totals.get("total_decisions", 0)
        if total_decisions == 0 and not keys and not strategy:
            # Nothing mirrored yet (fresh cluster / Redis flushed): the local
            # snapshot is at least as informative.
            return local

        return {
            "total_decisions": total_decisions,
            "strategy_distribution": strategy,
            "profile_distribution": profile,
            "centroid_decisions": totals.get("centroid_decisions", 0),
            # Latency does not aggregate via counters; surface the local mean.
            "average_latency_ms": local["average_latency_ms"],
            "model_distribution": model,
            "key_distribution": keys,
            "tracked_keys": len(keys),
            "cluster_wide": True,
        }
    except Exception:  # pragma: no cover - cluster read must never break the panel
        logger.debug("Failed to read cluster stats snapshot", exc_info=True)
        return local


async def reset_cluster_stats() -> bool:
    """Delete all shared cluster-stats keys (for tests / operator reset).

    Returns ``True`` when a delete was issued, ``False`` when Redis is
    unavailable.  Fail-open: never raises.
    """
    try:
        from litellm_llmrouter.redis_pool import get_async_redis_client

        redis = await get_async_redis_client()
        if redis is None:
            return False
        await redis.delete(
            _k("totals"),
            _k("strategy"),
            _k("profile"),
            _k("model"),
            _k("keys"),
        )
        return True
    except Exception:  # pragma: no cover
        logger.debug("Failed to reset cluster stats", exc_info=True)
        return False


# =============================================================================
# LLM API Path Registry
# =============================================================================

# Maps all LLM API endpoint paths to their operation type.
# Imported from the canonical source in telemetry_contracts.
from litellm_llmrouter.telemetry_contracts import LLM_API_PATHS  # noqa: E402


# =============================================================================
# RouterDecisionMiddleware - FastAPI Middleware (Recommended for E2E tests)
# =============================================================================


# Custom header names for routing transparency
HEADER_ROUTEIQ_MODEL = "X-RouteIQ-Model"
HEADER_ROUTEIQ_TIER = "X-RouteIQ-Tier"
HEADER_ROUTEIQ_STRATEGY = "X-RouteIQ-Strategy"

# All custom RouteIQ headers (for CORS expose_headers)
ROUTEIQ_RESPONSE_HEADERS = [
    HEADER_ROUTEIQ_MODEL,
    HEADER_ROUTEIQ_TIER,
    HEADER_ROUTEIQ_STRATEGY,
]


class RouterDecisionMiddleware:
    """
    Raw ASGI middleware that emits TG4.1 router decision span attributes
    and injects ``X-RouteIQ-*`` routing transparency headers into responses.

    Uses the raw ASGI pattern (same as BackpressureMiddleware) instead of
    BaseHTTPMiddleware, which buffers the entire response and breaks streaming.

    This middleware intercepts LLM API requests and emits router.*
    span attributes BEFORE the LLM API call happens. It also increments
    gateway.request.total and gateway.routing.strategy.usage metrics.

    Response headers:
        - ``X-RouteIQ-Model``: The model selected by the router.
        - ``X-RouteIQ-Tier``: The complexity tier (simple/complex/reasoning).
        - ``X-RouteIQ-Strategy``: The routing strategy that made the decision.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Only instrument POST requests to LLM API endpoints
        if method != "POST" or path not in LLM_API_PATHS:
            await self.app(scope, receive, send)
            return

        if not ROUTER_CALLBACK_ENABLED:
            await self.app(scope, receive, send)
            return

        # Emit router decision attributes on the current span
        self._emit_router_telemetry(path)

        # Increment gateway metrics
        self._increment_metrics(path)

        # Resolve routing context for response headers
        model, tier, strategy = self._resolve_routing_context()

        # Wrap send to inject X-RouteIQ-* headers into response
        async def send_with_routing_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if model:
                    headers.append(
                        (
                            HEADER_ROUTEIQ_MODEL.lower().encode("latin-1"),
                            model.encode("latin-1"),
                        )
                    )
                if tier:
                    headers.append(
                        (
                            HEADER_ROUTEIQ_TIER.lower().encode("latin-1"),
                            tier.encode("latin-1"),
                        )
                    )
                if strategy:
                    headers.append(
                        (
                            HEADER_ROUTEIQ_STRATEGY.lower().encode("latin-1"),
                            strategy.encode("latin-1"),
                        )
                    )
                message = {**message, "headers": headers}
            await send(message)

        # Pass through to the next ASGI app (streaming-safe)
        await self.app(scope, receive, send_with_routing_headers)

    def _resolve_routing_context(self) -> tuple:
        """Resolve the current routing context for response headers.

        Inspects the LiteLLM router (if available) to determine the
        model, tier, and strategy that will handle the current request.

        Returns:
            Tuple of ``(model, tier, strategy)`` strings. Empty strings
            for values that cannot be determined.
        """
        strategy = OVERRIDE_STRATEGY_NAME or "litellm-builtin"
        model = ""
        tier = ""

        try:
            from litellm.proxy.proxy_server import llm_router

            if llm_router is not None:
                model_list = getattr(llm_router, "model_list", []) or []
                if model_list:
                    first_model = model_list[0]
                    model = first_model.get("litellm_params", {}).get("model", "")
        except Exception:
            pass

        # Derive tier heuristic from model name
        if model:
            model_lower = model.lower()
            simple_indicators = ["mini", "nano", "haiku", "flash", "small", "light"]
            reasoning_indicators = ["o1", "o3", "o4", "reasoner"]
            if any(ind in model_lower for ind in reasoning_indicators):
                tier = "reasoning"
            elif any(ind in model_lower for ind in simple_indicators):
                tier = "simple"
            else:
                tier = "complex"

        return model, tier, strategy

    def _emit_router_telemetry(self, path: str) -> None:
        """
        Emit TG4.1 router decision span attributes and gen_ai.* attributes.

        This is called BEFORE the LLM API call, ensuring telemetry is
        always emitted for LLM API requests.
        """
        try:
            from litellm_llmrouter.observability import set_router_decision_attributes

            span = trace.get_current_span()
            if not span or not span.is_recording():
                return

            operation_name = LLM_API_PATHS.get(path, "unknown")
            strategy = OVERRIDE_STRATEGY_NAME or "litellm-builtin"

            set_router_decision_attributes(
                span,
                strategy=strategy,
                model_selected="pending",
                candidates_evaluated=1,
                outcome="success",
                reason="middleware_routing",
                latency_ms=0.1,
                strategy_version="v1-middleware",
                fallback_triggered=False,
            )

            # GenAI semantic convention attributes (ADR-0019)
            from litellm_llmrouter.telemetry_contracts import GenAIAttributes as GA

            span.set_attribute(GA.OPERATION_NAME, operation_name)

            # Try to resolve the system (provider) from the router config
            try:
                from litellm.proxy.proxy_server import llm_router

                if llm_router is not None:
                    model_list = getattr(llm_router, "model_list", []) or []
                    if model_list:
                        first_model = model_list[0]
                        litellm_model = first_model.get("litellm_params", {}).get(
                            "model", ""
                        )
                        provider = first_model.get("litellm_params", {}).get(
                            "custom_llm_provider", ""
                        )
                        if not provider and "/" in litellm_model:
                            provider = litellm_model.split("/")[0]
                        if provider:
                            span.set_attribute(GA.SYSTEM, provider)
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Failed to emit router telemetry in middleware: {e}")

    def _increment_metrics(self, path: str) -> None:
        """Increment gateway.request.total and gateway.routing.strategy.usage."""
        try:
            from litellm_llmrouter.metrics import get_gateway_metrics

            metrics = get_gateway_metrics()
            if metrics is None:
                return

            operation = LLM_API_PATHS.get(path, "unknown")
            strategy = OVERRIDE_STRATEGY_NAME or "litellm-builtin"

            metrics.request_total.add(1, {"operation": operation})
            metrics.strategy_usage.add(1, {"strategy": strategy})

        except Exception:
            pass


def register_router_decision_middleware(app: Any) -> bool:
    """
    Register the RouterDecisionMiddleware with a FastAPI app.

    Wraps the ASGI app directly (same pattern as BackpressureMiddleware)
    to avoid BaseHTTPMiddleware's response buffering that breaks streaming.

    Args:
        app: FastAPI application instance

    Returns:
        True if middleware was registered, False if disabled
    """
    if not ROUTER_CALLBACK_ENABLED:
        logger.debug("Router decision middleware disabled")
        return False

    try:
        app.app = RouterDecisionMiddleware(app.app)
        logger.info("Registered RouterDecisionMiddleware for TG4.1 telemetry (ASGI)")
        return True
    except Exception as e:
        logger.warning(f"Failed to register router decision middleware: {e}")
        return False


# =============================================================================
# RouterDecisionCallback - LiteLLM Callback (Legacy)
# =============================================================================


class RouterDecisionCallback:
    """
    LiteLLM custom callback that emits TG4.1 router decision span attributes
    and records OTel metrics for request duration, token usage, and errors.

    Compatible with LiteLLM's custom callback interface.
    """

    def __init__(
        self,
        strategy_name: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize the router decision callback.

        Args:
            strategy_name: Override strategy name in telemetry
            enabled: Whether the callback is active
        """
        self._strategy_name = (
            strategy_name or OVERRIDE_STRATEGY_NAME or "litellm-builtin"
        )
        self._enabled = enabled and ROUTER_CALLBACK_ENABLED
        self._call_count = 0
        # Per-call start times keyed by litellm_call_id
        self._start_times: Dict[str, float] = {}
        logger.info(
            f"RouterDecisionCallback initialized: enabled={self._enabled}, "
            f"strategy={self._strategy_name}"
        )

    def log_pre_api_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Called before each API call - emits router decision telemetry
        and records start time + increments the active request gauge.
        """
        if not self._enabled:
            return

        try:
            self._emit_router_telemetry(model, messages, kwargs)
        except Exception as e:
            logger.debug(f"Failed to emit router telemetry: {e}")

        # Record start time and increment active gauge
        try:
            call_id = kwargs.get("litellm_call_id", "")
            if call_id:
                self._start_times[call_id] = time.perf_counter()

            from litellm_llmrouter.metrics import get_gateway_metrics

            gm = get_gateway_metrics()
            if gm:
                gm.request_active.add(1, {"model": model})
        except Exception as e:
            logger.debug(f"Failed to record pre-call metrics: {e}")

    def _emit_router_telemetry(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Emit TG4.1 router decision span attributes and gen_ai.* attributes.

        Sets router.* span attributes for routing visibility and gen_ai.*
        attributes for GenAI Semantic Convention compliance.
        """
        from litellm_llmrouter.observability import set_router_decision_attributes

        span = trace.get_current_span()
        if not span or not span.is_recording():
            logger.debug("No active span for router telemetry")
            return

        self._call_count += 1

        # Extract metadata from kwargs
        metadata = kwargs.get("metadata", {}) or {}

        # Determine number of candidates (if available from router)
        candidates = metadata.get("model_group_size", 1)
        if isinstance(candidates, str):
            try:
                candidates = int(candidates)
            except ValueError:
                candidates = 1

        # Determine routing strategy from metadata or config
        strategy = metadata.get("routing_strategy") or self._strategy_name

        # Determine outcome - if we got here, routing succeeded
        outcome = "success"
        reason = "model_selected"

        # Check for specific deployment selection
        if metadata.get("specific_deployment"):
            reason = "specific_deployment_requested"
        elif metadata.get("fallback"):
            reason = "fallback_triggered"

        # Approximate latency (not accurate, but provides a value)
        latency_ms = 0.1  # Placeholder

        # Set TG4.1 span attributes
        set_router_decision_attributes(
            span,
            strategy=strategy,
            model_selected=model,
            candidates_evaluated=candidates,
            outcome=outcome,
            reason=reason,
            latency_ms=latency_ms,
            strategy_version=f"v1-callback-{self._call_count}",
            fallback_triggered=bool(metadata.get("fallback")),
        )

        # Set gen_ai.* span attributes (GenAI Semantic Conventions — ADR-0019)
        from litellm_llmrouter.telemetry_contracts import GenAIAttributes as GA

        span.set_attribute(GA.REQUEST_MODEL, model)
        # Extract provider from litellm_params if available
        litellm_params = kwargs.get("litellm_params", {}) or {}
        provider = litellm_params.get("custom_llm_provider", "")
        if provider:
            span.set_attribute(GA.SYSTEM, provider)

        # Feed the in-process stats accumulator (RouteIQ-aba9).  This is the
        # per-decision seam with the resolved model + strategy + governance
        # scope, so the global stats + /me/stats endpoints read real aggregates.
        _record_decision_stats(model, strategy, metadata)

        # Feed the MLOps hot-path observers (RouteIQ-fc5c part b): the drift
        # detector's request-bucket window and the shadow/mirror candidate. Both
        # are settings-gated singletons that return None when disabled, so this
        # is a byte-stable no-op until an operator opts in. Cheap + fully
        # fail-open: never breaks the routing/telemetry path.
        _record_mlops_hot_path(model, strategy, messages, metadata)

        logger.debug(
            f"Emitted router telemetry: model={model}, strategy={strategy}, "
            f"candidates={candidates}, outcome={outcome}"
        )

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Called on successful API response.

        Records duration histogram, token usage, success counter, cost,
        and gen_ai.* span attributes from the response.
        """
        if not self._enabled:
            return

        try:
            self._record_success_metrics(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            logger.debug(f"Failed to record success metrics: {e}")

    def _record_success_metrics(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Record metrics from a successful LLM API response."""
        from litellm_llmrouter.metrics import get_gateway_metrics

        model = kwargs.get("model", "unknown")
        litellm_params = kwargs.get("litellm_params", {}) or {}
        provider = litellm_params.get("custom_llm_provider", "unknown")

        # Compute duration from perf_counter if available, else from timestamps
        call_id = kwargs.get("litellm_call_id", "")
        perf_start = self._start_times.pop(call_id, None) if call_id else None
        if perf_start is not None:
            duration_s = time.perf_counter() - perf_start
        else:
            # Fallback: start_time/end_time are datetime or float from LiteLLM
            duration_s = _compute_duration(start_time, end_time)

        # Extract token usage from response object
        input_tokens = 0
        output_tokens = 0
        if hasattr(response_obj, "usage") and response_obj.usage is not None:
            usage = response_obj.usage
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Extract finish reasons from response object
        finish_reasons: List[str] = []
        if hasattr(response_obj, "choices") and response_obj.choices:
            for choice in response_obj.choices:
                reason = getattr(choice, "finish_reason", None)
                if reason:
                    finish_reasons.append(str(reason))

        from litellm_llmrouter.telemetry_contracts import GenAIAttributes as GA

        attrs = {
            GA.REQUEST_MODEL: model,
            GA.SYSTEM: provider,
        }

        # Is this a streamed response? LiteLLM stamps the resolved request params
        # onto ``kwargs["optional_params"]`` (same source the upstream OTel
        # integration reads to gate its TTFT metric). For streamed responses the
        # combined throughput metric is wired below via ``record_streaming_metrics``.
        optional_params = kwargs.get("optional_params", {}) or {}
        is_streaming = bool(optional_params.get("stream", False))

        # Feed the real response latency into the stats accumulator so the
        # global / dashboard average reflects actual durations (RouteIQ-aba9).
        try:
            get_stats_accumulator().record_latency(duration_s * 1000.0)
        except Exception:  # pragma: no cover - stats must never break routing
            logger.debug("Failed to record latency stats", exc_info=True)

        gm = get_gateway_metrics()
        if gm:
            # Duration histogram
            gm.request_duration.record(duration_s, attrs)

            # Token usage histograms (per-request, split by direction).
            if input_tokens > 0:
                gm.token_usage.record(
                    input_tokens, {**attrs, "gen_ai.token.type": "input"}
                )
            if output_tokens > 0:
                gm.token_usage.record(
                    output_tokens, {**attrs, "gen_ai.token.type": "output"}
                )

            # Streaming throughput (RouteIQ-f55a): for streamed responses wire the
            # previously-uncalled ``record_streaming_metrics`` helper so the
            # tokens_per_second histogram (and the combined token_usage sample the
            # helper records) finally has a live caller. TTFT is computed from
            # LiteLLM's ``completion_start_time`` (when the first chunk arrived);
            # it is recorded ONCE here, not also via ``observability.record_ttft``,
            # so there is no double-record. Best-effort: any error is swallowed so
            # a malformed timestamp never breaks the response path.
            if is_streaming:
                self._record_streaming_throughput(
                    gm,
                    kwargs=kwargs,
                    output_tokens=output_tokens,
                    duration_s=duration_s,
                    attrs=attrs,
                )

            # Success counter
            gm.request_total.add(
                1,
                {
                    "model": model,
                    "provider": provider,
                    "status": "success",
                },
            )

            # Decrement active gauge
            gm.request_active.add(-1, {"model": model})

        # Set gen_ai.* span attributes on the current span (ADR-0019)
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute(GA.USAGE_INPUT_TOKENS, input_tokens)
            span.set_attribute(GA.USAGE_OUTPUT_TOKENS, output_tokens)
            total_tokens = input_tokens + output_tokens
            if total_tokens > 0:
                span.set_attribute(GA.USAGE_TOTAL_TOKENS, total_tokens)
            if hasattr(response_obj, "model") and response_obj.model:
                span.set_attribute(GA.RESPONSE_MODEL, response_obj.model)
            if hasattr(response_obj, "id") and response_obj.id:
                span.set_attribute(GA.RESPONSE_ID, response_obj.id)
            if finish_reasons:
                span.set_attribute(GA.RESPONSE_FINISH_REASONS, finish_reasons)

    def _record_streaming_throughput(
        self,
        gm: Any,
        *,
        kwargs: Dict[str, Any],
        output_tokens: int,
        duration_s: float,
        attrs: Dict[str, Any],
    ) -> None:
        """Record streaming throughput via ``metrics.record_streaming_metrics``.

        Wires the combined TTFT + tokens_per_second + token_usage helper
        (RouteIQ-f55a) into the streaming success path, which is the only
        post-response seam that carries the completed token count, the total
        duration, and LiteLLM's ``completion_start_time`` (the first-chunk
        timestamp).

        Throughput is generated-tokens / generation-time, where generation time
        is ``end_time - completion_start_time`` (the post-first-token window) when
        ``completion_start_time`` is available, else the full request duration as a
        floor. TTFT is ``completion_start_time - api_call_start_time`` and is
        recorded ONCE here (``observability.record_ttft`` is not also called for
        this response), so the time_to_first_token series is not double-counted.

        Best-effort: never raises. Labels are the low-cardinality ``attrs``
        (model + provider) only -- no per-request / per-user identifiers.
        """
        try:
            ttft_s = _streaming_ttft_seconds(kwargs)
            gen_window_s = _streaming_generation_seconds(kwargs, duration_s)
            tps = (output_tokens / gen_window_s) if gen_window_s > 0 else 0.0
            gm.record_streaming_metrics(
                ttft_ms=ttft_s * 1000.0,
                tps=tps,
                total_tokens=output_tokens,
                attrs=attrs,
            )
        except Exception:  # pragma: no cover - metrics must never break routing
            logger.debug("Failed to record streaming throughput", exc_info=True)

    def log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Called on failed API response.

        Records error counter with model/provider/error_type dimensions.
        """
        if not self._enabled:
            return

        try:
            self._record_failure_metrics(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            logger.debug(f"Failed to record failure metrics: {e}")

    def _record_failure_metrics(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Record metrics from a failed LLM API response."""
        from litellm_llmrouter.metrics import get_gateway_metrics

        model = kwargs.get("model", "unknown")
        litellm_params = kwargs.get("litellm_params", {}) or {}
        provider = litellm_params.get("custom_llm_provider", "unknown")

        # Determine error type
        exception = kwargs.get("exception", None)
        error_type = type(exception).__name__ if exception else "unknown"

        # Clean up start time tracking
        call_id = kwargs.get("litellm_call_id", "")
        if call_id:
            self._start_times.pop(call_id, None)

        gm = get_gateway_metrics()
        if gm:
            gm.request_error.add(
                1,
                {
                    "model": model,
                    "provider": provider,
                    "error_type": error_type,
                },
            )

            # Decrement active gauge
            gm.request_active.add(-1, {"model": model})

        # Set gen_ai.response.finish_reasons = ["error"] on the span (ADR-0019)
        from litellm_llmrouter.telemetry_contracts import GenAIAttributes as GA

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute(GA.RESPONSE_FINISH_REASONS, ["error"])

    async def async_log_pre_api_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        """Async version of log_pre_api_call."""
        self.log_pre_api_call(model, messages, kwargs)

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Async version of log_success_event.

        Also drives the governance/usage-policy spend WRITE path (P4) on a
        feature gate INDEPENDENT of the OTEL-gated telemetry ``_enabled`` flag,
        so budget/RPM counters are written even when OTEL is off.  Fail-open.

        This is ALSO the COLLECT arm of the eval pipeline (RouteIQ-295a): the
        success callback is the only post-response seam that carries both the
        request (``kwargs``: model + messages + metadata) and the response
        (``response_obj``: usage + content), so it is where request/response
        pairs are captured into the closed MLOps loop.  Gated behind the eval
        pipeline being enabled (``get_eval_pipeline()`` may return ``None`` --
        no-op) and the pipeline's own sample rate; fully fail-open.
        """
        if self._enabled:
            self.log_success_event(kwargs, response_obj, start_time, end_time)

        if _governance_spend_tracking_enabled():
            try:
                await _record_post_response_spend(kwargs, response_obj)
            except Exception as e:  # fail-open: never break the response path
                logger.debug("Governance spend tracking skipped: %s", e)

        # COLLECT arm: feed the request/response pair into the eval pipeline.
        # Safe no-op when the pipeline is disabled/absent; never raises.
        _collect_eval_sample(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Async version of log_failure_event.

        Also drives the eval FAILURE-path capture (RouteIQ-d365): a parallel,
        clearly-labeled, low-rate capture from the error/timeout path so
        AGGREGATE/FEEDBACK can downweight error-prone models/strategies. Gated
        behind ``settings.mlops.failure_capture.enabled`` (default off) and
        sampled at a LOW rate; safe no-op when disabled. Fully fail-open.
        """
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

        # FAILURE-path eval capture (independent of the OTEL-gated _enabled flag,
        # mirroring the success-path COLLECT/spend pattern).
        _collect_failure_eval_sample(kwargs, start_time, end_time)

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> None:
        """Called after successful API call. Required by LiteLLM callback interface."""
        pass

    async def async_post_call_failure_hook(
        self,
        request_data: Dict[str, Any],
        original_exception: Exception,
        user_api_key_dict: Any,
    ) -> None:
        """Called after failed API call. Required by LiteLLM callback interface."""
        pass


def _compute_duration(start_time: Any, end_time: Any) -> float:
    """
    Compute duration in seconds from LiteLLM start/end times.

    LiteLLM passes datetime objects or floats depending on the code path.

    Args:
        start_time: Start time (datetime or float)
        end_time: End time (datetime or float)

    Returns:
        Duration in seconds, or 0.0 if computation fails.
    """
    try:
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            return max(0.0, float(end_time) - float(start_time))
        # datetime objects
        if hasattr(start_time, "timestamp") and hasattr(end_time, "timestamp"):
            return max(0.0, end_time.timestamp() - start_time.timestamp())
    except Exception:
        pass
    return 0.0


def _to_epoch_seconds(value: Any) -> Optional[float]:
    """Coerce a LiteLLM timestamp (float, int, or datetime) to epoch seconds.

    Returns ``None`` when the value is missing or cannot be coerced, so callers
    can fall back rather than record a garbage sample.
    """
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
    except Exception:
        return None
    return None


def _streaming_ttft_seconds(kwargs: Dict[str, Any]) -> float:
    """Time-to-first-token in seconds for a streamed response.

    Computed as ``completion_start_time - api_call_start_time`` (the first-chunk
    arrival relative to the LLM API call start), mirroring how upstream LiteLLM's
    OTel integration derives its TTFT. Returns ``0.0`` when either timestamp is
    absent or non-coercible (the helper's ``ttft_ms`` then records a 0 sample,
    which the throughput half does not depend on).
    """
    start = _to_epoch_seconds(kwargs.get("api_call_start_time"))
    first_chunk = _to_epoch_seconds(kwargs.get("completion_start_time"))
    if start is None or first_chunk is None:
        return 0.0
    return max(0.0, first_chunk - start)


def _streaming_generation_seconds(kwargs: Dict[str, Any], duration_s: float) -> float:
    """Generation window in seconds used as the tokens_per_second denominator.

    For a streamed response throughput is generated-tokens / generation-time,
    where generation-time is the post-first-token window
    (``end_time - completion_start_time``). Falls back to the full request
    ``duration_s`` when ``completion_start_time`` / ``end_time`` are unavailable,
    so a tps value is still produced. Never returns a negative window.
    """
    first_chunk = _to_epoch_seconds(kwargs.get("completion_start_time"))
    end = _to_epoch_seconds(kwargs.get("end_time"))
    if first_chunk is not None and end is not None:
        window = end - first_chunk
        if window > 0:
            return window
    return max(0.0, duration_s)


def _record_decision_stats(
    model: str,
    strategy: str,
    metadata: Dict[str, Any],
) -> None:
    """Feed one routing decision into the process-local stats accumulator.

    Resolves the caller's key scope from the SAME ``_governance_ctx`` stamp the
    spend writer uses (raw ``key_id``, never LiteLLM's hashed token), and the
    routing profile from the stamp's ``effective_profile`` (falling back to the
    ``_routing_profile`` hint).  Fully fail-open: any error is swallowed so the
    request path is never broken by stats bookkeeping.
    """
    try:
        key_id: Optional[str] = None
        user_id: Optional[str] = None
        profile: Optional[str] = None

        gctx = metadata.get("_governance_ctx") if isinstance(metadata, dict) else None
        if isinstance(gctx, dict):
            key_id = gctx.get("key_id")
            profile = gctx.get("effective_profile")

        if isinstance(metadata, dict):
            if not profile:
                profile = metadata.get("_routing_profile")
            if not key_id:
                key_id = metadata.get("user_api_key") or metadata.get("_key")
            user_id = (
                metadata.get("user_api_key_user_id")
                or metadata.get("user_id")
                or metadata.get("_user")
            )

        resolved_key = str(key_id) if key_id else None
        get_stats_accumulator().record_decision(
            strategy=strategy,
            model=model,
            profile=profile,
            key_id=resolved_key,
            user_id=str(user_id) if user_id else None,
        )

        # Write-through to the cluster-wide shared store (RouteIQ-78fd) so every
        # worker contributes to a single cross-replica aggregate.  Fire-and-forget
        # on the running loop; a no-op when no loop / Redis is available.
        _schedule_cluster_mirror(
            strategy=strategy,
            model=model,
            profile=profile,
            key_id=resolved_key,
            centroid=strategy == _CENTROID_STRATEGY_NAME,
        )
    except Exception:  # pragma: no cover - stats must never break routing
        logger.debug("Failed to record decision stats", exc_info=True)


def _prompt_length_bucket(messages: Optional[List[Dict[str, Any]]]) -> str:
    """Map a request to a LOW-cardinality prompt-length bucket label.

    Never emits raw user text -- only a coarse size class -- so the drift
    detector's bucket window stays PII-safe and bounded-cardinality.
    """
    try:
        total = 0
        for m in messages or []:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):  # multimodal content parts
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += len(part["text"])
    except Exception:  # pragma: no cover - defensive
        return "unknown"
    if total <= 0:
        return "empty"
    if total < 256:
        return "xs"
    if total < 1024:
        return "s"
    if total < 4096:
        return "m"
    if total < 16384:
        return "l"
    return "xl"


def _record_mlops_hot_path(
    model: str,
    strategy: str,
    messages: Optional[List[Dict[str, Any]]],
    metadata: Dict[str, Any],
) -> None:
    """Feed the MLOps hot-path observers from one routing decision (RouteIQ-fc5c).

    Wires two previously-dark mechanisms into the per-decision seam:

    * :meth:`DriftDetector.record_request_bucket` -- records one coarse
      request-bucket (the requested tier when present, else a prompt-length
      bucket) into the drift detector's current window.
    * :meth:`ShadowMirror.mirror` -- mirrors the decision to a candidate strategy
      (computes its counterfactual choice silently, never affecting serving).

    Both consumers are settings-gated singletons that return ``None`` when
    disabled, so this is a byte-stable no-op until ``settings.mlops.drift`` /
    ``settings.mlops.shadow`` are turned on. Fully fail-open: any error is
    swallowed so MLOps bookkeeping never breaks the routing/telemetry path.
    """
    try:
        from litellm_llmrouter.mlops.drift import get_drift_detector

        detector = get_drift_detector()
        if detector is not None:
            tier = (
                metadata.get("_routing_profile") if isinstance(metadata, dict) else None
            )
            bucket = str(tier) if tier else _prompt_length_bucket(messages)
            detector.record_request_bucket(bucket)
    except Exception:  # pragma: no cover - drift must never break routing
        logger.debug("Failed to record drift bucket", exc_info=True)

    try:
        from litellm_llmrouter.strategy_registry import (
            RoutingContext,
            RoutingResult,
            get_shadow_mirror,
        )

        mirror = get_shadow_mirror()
        if mirror is not None:
            context = RoutingContext(
                router=None,
                model=model,
                messages=messages,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
            served = RoutingResult(
                deployment={"litellm_params": {"model": model}},
                strategy_name=strategy,
            )
            mirror.mirror(context, served)
    except Exception:  # pragma: no cover - shadow must never break routing
        logger.debug("Failed to mirror shadow decision", exc_info=True)


def _schedule_cluster_mirror(
    *,
    strategy: Optional[str],
    model: Optional[str],
    profile: Optional[str],
    key_id: Optional[str],
    centroid: bool,
) -> None:
    """Fire-and-forget the async cluster mirror without blocking the caller.

    ``_record_decision_stats`` runs on the sync callback path which MAY be
    inside the request event loop (async callback) or on a plain thread (sync
    callback).  When a running loop is present we schedule the coroutine as a
    task; otherwise we drop the mirror (the local accumulator already recorded
    it, and the next loop-bound decision will refresh the shared keys).  Never
    raises -- cluster bookkeeping must never break routing.
    """
    if not _cluster_stats_enabled():
        return
    try:
        import asyncio

        coro = mirror_decision_to_cluster(
            strategy=strategy,
            model=model,
            profile=profile,
            key_id=key_id,
            centroid=centroid,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop on this thread: close the coroutine to avoid a
            # "coroutine was never awaited" warning and skip the mirror.
            coro.close()
            return
        loop.create_task(coro)
    except Exception:  # pragma: no cover - mirror scheduling must never raise
        logger.debug("Failed to schedule cluster mirror", exc_info=True)


def _extract_response_content(response_obj: Any) -> str:
    """Best-effort extraction of the assistant text from an LLM response.

    Reads ``response_obj.choices[0].message.content`` (the OpenAI-compatible
    shape LiteLLM normalizes to).  Returns ``""`` on any structural surprise so
    a malformed/streamed response never breaks collection.  Bounded to keep the
    captured sample cheap; the judge re-truncates downstream regardless.
    """
    try:
        choices = getattr(response_obj, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            return content[:_EVAL_RESPONSE_CONTENT_CAP]
    except Exception:
        return ""
    return ""


def _resolve_tier_from_model(model: str) -> str:
    """Derive a coarse complexity tier from the model name (heuristic).

    Mirrors the middleware's tier heuristic so collected samples carry a tier
    even when none was stamped into metadata.  Returns ``""`` for an empty
    model name.
    """
    if not model:
        return ""
    model_lower = model.lower()
    if any(ind in model_lower for ind in ("o1", "o3", "o4", "reasoner")):
        return "reasoning"
    if any(
        ind in model_lower
        for ind in ("mini", "nano", "haiku", "flash", "small", "light")
    ):
        return "simple"
    return "complex"


def _collect_eval_sample(
    kwargs: Dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """COLLECT arm: capture one request/response pair into the eval pipeline.

    This is the live wiring (RouteIQ-295a) of the previously-dead COLLECT stage:
    EVALUATE/AGGREGATE/FEEDBACK were already running, but nothing was feeding
    ``EvalPipeline.collect``.  Called from the success callback, which is the
    only post-response seam carrying both the request and the response.

    Gating + safety contract:
      * ``get_eval_pipeline()`` returns ``None`` when the pipeline is disabled
        (``ROUTEIQ_EVAL_PIPELINE`` off) -- this function then no-ops.
      * Sampling is delegated to the pipeline's own ``should_sample()`` so the
        configured ``sample_rate`` is honored and most requests are skipped
        cheaply (the EvalSample is only built for sampled requests).
      * Fully fail-open: any error is swallowed so the hot response path is
        never broken by eval bookkeeping.

    PII discipline: only the fields the ``EvalSample`` schema asks for are
    captured (model/strategy/tier, messages, response text, token counts,
    latency, and the caller's key/user/workspace scope).  No secrets, headers,
    or governance stamps are copied; nothing is logged.
    """
    try:
        from litellm_llmrouter.eval_pipeline import EvalSample, get_eval_pipeline

        pipeline = get_eval_pipeline()
        if pipeline is None:
            return

        # Honor the pipeline's configured sample rate BEFORE building a sample
        # so the un-sampled common case stays cheap.
        should_sample = getattr(pipeline, "should_sample", None)
        if callable(should_sample) and not should_sample():
            return

        metadata = kwargs.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        model = kwargs.get("model", "") or ""
        strategy = (
            metadata.get("routing_strategy") or OVERRIDE_STRATEGY_NAME or "unknown"
        )
        tier = metadata.get("_routing_tier") or _resolve_tier_from_model(model)

        messages = kwargs.get("messages", []) or []
        if not isinstance(messages, list):
            messages = []

        # Token usage from the response usage object (best-effort).
        input_tokens = 0
        output_tokens = 0
        usage = getattr(response_obj, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Caller scope: prefer the resolved governance stamp (raw key_id), then
        # fall back to the raw metadata hints -- same precedence the stats path
        # uses.  These are identifiers the EvalSample schema models, not secrets.
        key_id: Optional[str] = None
        user_id: Optional[str] = None
        workspace_id: Optional[str] = None
        gctx = metadata.get("_governance_ctx")
        if isinstance(gctx, dict):
            key_id = gctx.get("key_id")
            workspace_id = gctx.get("workspace_id")
        if not key_id:
            key_id = metadata.get("user_api_key") or metadata.get("_key")
        if not workspace_id:
            workspace_id = metadata.get("workspace_id") or metadata.get("_workspace")
        user_id = (
            metadata.get("user_api_key_user_id")
            or metadata.get("user_id")
            or metadata.get("_user")
        )

        latency_ms = _compute_duration(start_time, end_time) * 1000.0

        sample = EvalSample(
            sample_id=str(
                kwargs.get("litellm_call_id") or f"eval-{time.time()}-{id(kwargs)}"
            ),
            timestamp=time.time(),
            model=str(model),
            strategy=str(strategy),
            tier=str(tier),
            messages=messages,
            request_tokens=int(input_tokens),
            response_content=_extract_response_content(response_obj),
            response_tokens=int(output_tokens),
            latency_ms=float(latency_ms),
            user_id=str(user_id) if user_id else None,
            workspace_id=str(workspace_id) if workspace_id else None,
            prompt_name=metadata.get("prompt_name"),
        )
        pipeline.collect(sample)
    except Exception:  # pragma: no cover - collection must never break routing
        logger.debug("Failed to collect eval sample", exc_info=True)


def _failure_capture_settings() -> tuple[bool, float]:
    """Resolve ``settings.mlops.failure_capture`` (enabled, sample_rate).

    Returns ``(False, 0.0)`` on any read failure so a misconfig never silently
    enables failure capture.  Settings-first per ADR-0013.
    """
    try:
        from litellm_llmrouter.settings import get_settings

        mlops = getattr(get_settings(), "mlops", None)
        fc = getattr(mlops, "failure_capture", None) if mlops is not None else None
        if fc is None or not getattr(fc, "enabled", False):
            return False, 0.0
        return True, float(getattr(fc, "sample_rate", 0.0))
    except Exception:  # pragma: no cover - defensive
        return False, 0.0


def _collect_failure_eval_sample(
    kwargs: Dict[str, Any],
    start_time: Any,
    end_time: Any,
) -> None:
    """FAILURE arm: capture one error/timeout decision into the eval pipeline.

    The success-only COLLECT arm leaves AGGREGATE/FEEDBACK blind to which
    models/strategies are error-prone (RouteIQ-d365).  This captures a parallel,
    clearly-labeled (``outcome="failure"`` + ``error_type``), LOW-rate sample
    from the failure path so the loop can downweight error-prone targets.

    Gating + safety contract:
      * Independent gate ``settings.mlops.failure_capture.enabled`` (default off)
        -- distinct from the eval pipeline's own success sample rate, so a
        failure storm cannot ride the success rate.
      * Sampled at the LOW ``failure_capture.sample_rate`` so a failure storm
        cannot flood the eval queue.
      * ``get_eval_pipeline()`` must be present (eval pipeline enabled), else
        no-op -- the captured sample has nowhere to go otherwise.
      * Fully fail-open: any error is swallowed so the failure path is never
        broken by eval bookkeeping.

    No response is captured (there is none on the failure path); the sample
    carries the request + the error_type only.  PII discipline matches the
    success arm.
    """
    try:
        enabled, sample_rate = _failure_capture_settings()
        if not enabled:
            return
        if sample_rate <= 0.0 or random.random() >= sample_rate:
            return

        from litellm_llmrouter.eval_pipeline import EvalSample, get_eval_pipeline

        pipeline = get_eval_pipeline()
        if pipeline is None:
            return

        metadata = kwargs.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        model = kwargs.get("model", "") or ""
        strategy = (
            metadata.get("routing_strategy") or OVERRIDE_STRATEGY_NAME or "unknown"
        )
        tier = metadata.get("_routing_tier") or _resolve_tier_from_model(model)

        exception = kwargs.get("exception", None)
        error_type = type(exception).__name__ if exception is not None else "unknown"

        messages = kwargs.get("messages", []) or []
        if not isinstance(messages, list):
            messages = []

        # Caller scope (same precedence the success arm uses).
        key_id: Optional[str] = None
        user_id: Optional[str] = None
        workspace_id: Optional[str] = None
        gctx = metadata.get("_governance_ctx")
        if isinstance(gctx, dict):
            key_id = gctx.get("key_id")
            workspace_id = gctx.get("workspace_id")
        if not key_id:
            key_id = metadata.get("user_api_key") or metadata.get("_key")
        if not workspace_id:
            workspace_id = metadata.get("workspace_id") or metadata.get("_workspace")
        user_id = (
            metadata.get("user_api_key_user_id")
            or metadata.get("user_id")
            or metadata.get("_user")
        )

        latency_ms = _compute_duration(start_time, end_time) * 1000.0

        sample = EvalSample(
            sample_id=str(
                kwargs.get("litellm_call_id") or f"evalfail-{time.time()}-{id(kwargs)}"
            ),
            timestamp=time.time(),
            model=str(model),
            strategy=str(strategy),
            tier=str(tier),
            messages=messages,
            response_content="",
            latency_ms=float(latency_ms),
            user_id=str(user_id) if user_id else None,
            workspace_id=str(workspace_id) if workspace_id else None,
            prompt_name=metadata.get("prompt_name"),
            outcome="failure",
            error_type=error_type,
        )
        pipeline.collect(sample)
    except Exception:  # pragma: no cover - capture must never break the failure path
        logger.debug("Failed to collect failure eval sample", exc_info=True)


def _derive_spend_scope(metadata: Dict[str, Any]) -> tuple[str, str]:
    """Derive the governance spend scope + scope_type from request metadata.

    The WRITE scope MUST be byte-identical to the READ scope the enforce path
    checks, or the budget counter the read consults is always 0.0 (fail-open).
    The enforce path (``custom_routing_strategy._enforce_governance``) stamps the
    RESOLVED governance scope into ``metadata["_governance_ctx"]``
    (``workspace_id`` + the RAW ``key_id``).  We prefer that stamp and apply the
    SAME precedence the read path uses (``workspace_id`` -> ``key_id`` ->
    ``"global"``, via ``derive_spend_scope_from_ctx``), so:
      * workspace budgets are enforced (RouteIQ-ed7a), and
      * key budgets use the raw api_key, not LiteLLM's hashed token (RouteIQ-08dd).

    Legacy/back-compat fallthrough (raw ``metadata`` keys) is preserved for
    non-enforce callers that never produced a ``_governance_ctx`` stamp.
    """
    from litellm_llmrouter.governance import derive_spend_scope_from_ctx

    gctx = metadata.get("_governance_ctx")
    if isinstance(gctx, dict):
        # Delegate to the SINGLE source of truth the read path uses so WRITE ==
        # READ by construction -- no inline precedence to drift (RouteIQ-9738).
        # derive_spend_scope_from_ctx accepts the dict stamp directly.
        return derive_spend_scope_from_ctx(gctx)

    # Legacy fallback: requests that bypassed the enforce stamp.
    workspace_id = metadata.get("workspace_id") or metadata.get("_workspace")
    if workspace_id:
        return str(workspace_id), "workspace"
    key_id = (
        metadata.get("user_api_key")
        or metadata.get("_key")
        or metadata.get("_user")
        or metadata.get("user_api_key_user_id")
    )
    if key_id:
        return str(key_id), "key"
    return "global", "global"


async def _record_post_response_spend(
    kwargs: Dict[str, Any],
    response_obj: Any,
) -> None:
    """Write post-response spend to the governance + usage-policy counters.

    This is the WRITER that closes the latent gap where ``governance:spend:`` /
    ``governance:rpm:`` were read but never written, and where
    ``UsagePolicyEngine.record_usage`` had no call site.  Both are wired here
    because this LiteLLM success callback is the only post-response seam that
    carries cost (``kwargs["response_cost"]``) + tokens (``response_obj.usage``).

    NEVER logs cost/token VALUES.  Fail-open is the caller's responsibility.
    """
    metadata = kwargs.get("metadata", {}) or {}

    # Cost: LiteLLM stamps response_cost on kwargs post-response.
    cost = kwargs.get("response_cost")
    if cost is None:
        cost = (kwargs.get("standard_logging_object", {}) or {}).get("response_cost")
    cost = float(cost) if cost else 0.0

    # Tokens from the response usage object.
    input_tokens = 0
    output_tokens = 0
    usage = getattr(response_obj, "usage", None)
    if usage is not None:
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = input_tokens + output_tokens

    scope, scope_type = _derive_spend_scope(metadata)

    # 1. Governance spend/rpm counters (ElastiCache hot + Aurora durable rollup).
    from litellm_llmrouter.governance import record_governance_spend

    await record_governance_spend(
        scope,
        scope_type,
        cost=cost,
        tokens=total_tokens,
        requests=1,
    )

    # 2. Usage-policy cost/token counters (wires the previously-dead record_usage).
    try:
        from litellm_llmrouter.usage_policies import get_usage_policy_engine

        request_context = {
            "model": kwargs.get("model", ""),
            "metadata": metadata,
            "workspace_id": metadata.get("workspace_id"),
        }
        await get_usage_policy_engine().record_usage(
            request_context,
            tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
    except Exception as e:
        logger.debug("Usage-policy record_usage skipped: %s", e)


def register_router_decision_callback() -> Optional[RouterDecisionCallback]:
    """
    Register the router decision callback with LiteLLM.

    Returns:
        The registered callback instance, or None if disabled.
    """
    if not ROUTER_CALLBACK_ENABLED:
        logger.debug("Router decision callback disabled")
        return None

    try:
        import litellm

        callback = RouterDecisionCallback()

        # Append to LiteLLM's callbacks list
        if not hasattr(litellm, "callbacks"):
            litellm.callbacks = []

        # Avoid duplicate registration
        for existing in litellm.callbacks:
            if isinstance(existing, RouterDecisionCallback):
                logger.debug("Router decision callback already registered")
                return existing

        litellm.callbacks.append(callback)
        logger.info("Registered router decision callback with LiteLLM")
        return callback

    except ImportError:
        logger.warning("LiteLLM not available, cannot register callback")
        return None
    except Exception as e:
        logger.error(f"Failed to register router decision callback: {e}")
        return None


def get_router_decision_callback() -> type:
    """
    Get the RouterDecisionCallback class for manual registration.

    Returns:
        The RouterDecisionCallback class
    """
    return RouterDecisionCallback
