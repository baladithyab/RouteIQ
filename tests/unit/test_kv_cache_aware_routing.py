"""Unit tests for the KV-cache-/queue-aware routing strategy (RouteIQ-08d6/6a89).

The engine-metrics scraper (RouteIQ-ffaa) and the strategy were both built; these
tests prove the wiring closes the loop:

* 08d6: ``KVCacheAwareRoutingStrategy`` consumes the scraped vLLM gauges
  (num_requests_waiting + kv_cache_usage) and selects the LEAST-LOADED engine arm;
* 6a89: the selection is fed by LIVE scraped snapshots (the scraper is mocked to
  return per-endpoint gauges -- no network I/O);
* default OFF -> ``register_kv_cache_aware_strategy`` registers nothing
  (byte-stable);
* registration is gated by ``engine_metrics.kv_aware_routing_enabled``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from litellm_llmrouter import engine_metrics as em_mod
from litellm_llmrouter.engine_metrics import EngineMetricsSnapshot
from litellm_llmrouter.settings import reset_settings
from litellm_llmrouter.strategies import (
    KVCacheAwareRoutingStrategy,
    register_kv_cache_aware_strategy,
)
from litellm_llmrouter.strategy_registry import (
    RoutingContext,
    get_routing_registry,
    reset_routing_singletons,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    reset_routing_singletons()
    yield
    reset_settings()
    reset_routing_singletons()


class _Router:
    """Minimal router exposing the candidate model_list the strategy reads."""

    def __init__(self, model_list):
        self.model_list = list(model_list)
        self.healthy_deployments = list(model_list)


def _arm(model_name, model, metrics_url):
    return {
        "model_name": model_name,
        "litellm_params": {"model": model},
        "model_info": {"engine_metrics_url": metrics_url},
    }


def _snap(endpoint, *, waiting, kv):
    return EngineMetricsSnapshot(
        endpoint=endpoint,
        reachable=True,
        gauges={
            "vllm:num_requests_waiting": waiting,
            "vllm:kv_cache_usage_perc": kv,
        },
    )


def _model_list():
    return [
        _arm("llama", "hosted/llama-busy", "http://engine-a:8000/metrics"),
        _arm("llama", "hosted/llama-idle", "http://engine-b:8000/metrics"),
    ]


# ---------------------------------------------------------------------------
# Selection (the live consumer of scraped gauges)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selects_least_loaded_arm(monkeypatch):
    """Given two arms, the lower queue-depth/KV-pressure arm is selected."""
    router = _Router(_model_list())

    async def _fake_scrape(endpoint):
        if endpoint.startswith("http://engine-a"):
            return _snap(endpoint, waiting=20, kv=0.9)  # busy
        return _snap(endpoint, waiting=1, kv=0.1)  # idle

    scraper = AsyncMock()
    scraper.scrape = AsyncMock(side_effect=_fake_scrape)
    monkeypatch.setattr(em_mod, "get_engine_metrics_scraper", lambda: scraper)

    strategy = KVCacheAwareRoutingStrategy()
    ctx = RoutingContext(router=router, model="llama")
    selected = await strategy.select_least_loaded(ctx)

    assert selected is not None
    assert selected["litellm_params"]["model"] == "hosted/llama-idle"
    assert scraper.scrape.await_count == 2


@pytest.mark.asyncio
async def test_prefers_arm_with_live_signal_over_unreachable(monkeypatch):
    """An arm with a reachable snapshot beats one whose engine is unreachable."""
    router = _Router(_model_list())

    async def _fake_scrape(endpoint):
        if endpoint.startswith("http://engine-a"):
            return EngineMetricsSnapshot(
                endpoint=endpoint, reachable=False, error="down"
            )
        return _snap(endpoint, waiting=50, kv=0.99)  # reachable but busy

    scraper = AsyncMock()
    scraper.scrape = AsyncMock(side_effect=_fake_scrape)
    monkeypatch.setattr(em_mod, "get_engine_metrics_scraper", lambda: scraper)

    strategy = KVCacheAwareRoutingStrategy()
    ctx = RoutingContext(router=router, model="llama")
    selected = await strategy.select_least_loaded(ctx)
    # engine-b is reachable (live signal) -> wins over the unreachable engine-a.
    assert selected["litellm_params"]["model"] == "hosted/llama-idle"


@pytest.mark.asyncio
async def test_sync_path_uses_cached_load(monkeypatch):
    """The sync select_deployment reads the cache the async path refreshed."""
    router = _Router(_model_list())

    async def _fake_scrape(endpoint):
        if endpoint.startswith("http://engine-a"):
            return _snap(endpoint, waiting=2, kv=0.2)  # idle
        return _snap(endpoint, waiting=30, kv=0.95)  # busy

    scraper = AsyncMock()
    scraper.scrape = AsyncMock(side_effect=_fake_scrape)
    monkeypatch.setattr(em_mod, "get_engine_metrics_scraper", lambda: scraper)

    strategy = KVCacheAwareRoutingStrategy()
    ctx = RoutingContext(router=router, model="llama")
    # Refresh cache via the async path...
    await strategy.select_least_loaded(ctx)
    # ...then the sync path picks the same least-loaded arm from the cache.
    selected = strategy.select_deployment(ctx)
    assert selected["litellm_params"]["model"] == "hosted/llama-busy"


@pytest.mark.asyncio
async def test_no_candidates_returns_none(monkeypatch):
    router = _Router([])
    scraper = AsyncMock()
    monkeypatch.setattr(em_mod, "get_engine_metrics_scraper", lambda: scraper)
    strategy = KVCacheAwareRoutingStrategy()
    ctx = RoutingContext(router=router, model="llama")
    assert await strategy.select_least_loaded(ctx) is None


# ---------------------------------------------------------------------------
# Registration gating
# ---------------------------------------------------------------------------


def test_register_disabled_by_default(monkeypatch):
    monkeypatch.delenv(
        "ROUTEIQ_ENGINE_METRICS__KV_AWARE_ROUTING_ENABLED", raising=False
    )
    reset_settings()
    assert register_kv_cache_aware_strategy() is False
    assert get_routing_registry().get("llmrouter-kv-cache-aware") is None


def test_register_when_enabled(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_ENGINE_METRICS__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_ENGINE_METRICS__KV_AWARE_ROUTING_ENABLED", "true")
    reset_settings()
    assert register_kv_cache_aware_strategy() is True
    strategy = get_routing_registry().get("llmrouter-kv-cache-aware")
    assert isinstance(strategy, KVCacheAwareRoutingStrategy)
