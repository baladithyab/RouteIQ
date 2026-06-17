"""
Tests for the self-hosted-engine /metrics scraper (RouteIQ-ffaa).

Covers the cred-free engine-metrics scrape module
(``litellm_llmrouter.engine_metrics``):
- the Prometheus text-exposition parser against a realistic vLLM /metrics body
  (the v1 ``vllm:kv_cache_usage_perc`` rename AND the pre-v1
  ``vllm:gpu_cache_usage_perc`` alias, plus queue gauges and prefix-cache
  counters), including labelled series, HELP/TYPE comments, and NaN/Inf;
- the scraper's default-OFF no-op (zero network I/O);
- graceful empty snapshots on unreachable / non-200 / timeout engines (never an
  exception that could crash a routing decision);
- a successful scrape via a stubbed httpx client.

All tests are cred-free: no live engine, no AWS, no secrets. The httpx layer is
stubbed; the parser is exercised directly on fixture bodies.
"""

import httpx
import pytest

from litellm_llmrouter.engine_metrics import (
    EngineMetricsScraper,
    EngineMetricsSnapshot,
    get_engine_metrics_scraper,
    parse_prometheus_metrics,
    reset_engine_metrics_scraper,
)


# A realistic-shaped vLLM v1 /metrics body. Mixes the metrics we want with noise
# (HELP/TYPE comments, unrelated metrics, labelled series, histogram buckets) so
# the parser is exercised against a body that looks like a real scrape.
VLLM_V1_METRICS_BODY = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-70B-Instruct"} 7.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="meta-llama/Llama-3.1-70B-Instruct"} 12.0
# HELP vllm:num_requests_swapped Number of requests swapped to CPU.
# TYPE vllm:num_requests_swapped gauge
vllm:num_requests_swapped{model_name="meta-llama/Llama-3.1-70B-Instruct"} 0.0
# HELP vllm:kv_cache_usage_perc Fraction of used KV cache blocks (0-1).
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="meta-llama/Llama-3.1-70B-Instruct"} 0.83
# HELP vllm:prefix_cache_queries Number of prefix cache queries.
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{model_name="meta-llama/Llama-3.1-70B-Instruct"} 1500.0
# HELP vllm:prefix_cache_hits Number of prefix cache hits.
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{model_name="meta-llama/Llama-3.1-70B-Instruct"} 1200.0
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop"} 9001.0
# HELP python_gc_objects_collected_total Objects collected during gc.
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 42.0
"""

# Pre-v1 body: the GPU-cache alias instead of the renamed kv_cache name, and an
# unlabelled (single-series) shape.
VLLM_V0_METRICS_BODY = """\
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.55
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 3
"""


# =============================================================================
# Parser
# =============================================================================
def test_parser_reads_v1_gauges_from_fixture():
    """The v1 fixture yields exactly the allowlisted gauges with correct values."""
    parsed = parse_prometheus_metrics(VLLM_V1_METRICS_BODY)

    assert parsed["vllm:num_requests_running"] == 7.0
    assert parsed["vllm:num_requests_waiting"] == 12.0
    assert parsed["vllm:num_requests_swapped"] == 0.0
    assert parsed["vllm:kv_cache_usage_perc"] == pytest.approx(0.83)
    assert parsed["vllm:prefix_cache_queries"] == 1500.0
    assert parsed["vllm:prefix_cache_hits"] == 1200.0

    # Non-allowlisted metrics (engine internals, process collectors) are dropped.
    assert "vllm:request_success_total" not in parsed
    assert "python_gc_objects_collected_total" not in parsed


def test_parser_reads_pre_v1_gpu_cache_alias():
    """The pre-v1 ``gpu_cache_usage_perc`` name is parsed (no v1 rename present)."""
    parsed = parse_prometheus_metrics(VLLM_V0_METRICS_BODY)
    assert parsed["vllm:gpu_cache_usage_perc"] == pytest.approx(0.55)
    assert parsed["vllm:num_requests_waiting"] == 3.0
    assert "vllm:kv_cache_usage_perc" not in parsed


def test_parser_skips_comments_blanks_and_nan_inf():
    body = (
        "# HELP vllm:num_requests_waiting q\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        "\n"
        "vllm:num_requests_waiting NaN\n"  # NaN dropped
        "vllm:num_requests_running +Inf\n"  # +Inf dropped
        "vllm:kv_cache_usage_perc 0.5\n"  # kept
    )
    parsed = parse_prometheus_metrics(body)
    assert parsed == {"vllm:kv_cache_usage_perc": 0.5}


def test_parser_collapses_labelled_series_to_last():
    """Labelled multi-series for a wanted name collapses to the last sample."""
    body = (
        'vllm:num_requests_waiting{model_name="a"} 1\n'
        'vllm:num_requests_waiting{model_name="b"} 9\n'
    )
    parsed = parse_prometheus_metrics(body)
    assert parsed["vllm:num_requests_waiting"] == 9.0


def test_parser_ignores_trailing_timestamp():
    body = "vllm:num_requests_waiting 4 1718000000000\n"
    parsed = parse_prometheus_metrics(body)
    assert parsed["vllm:num_requests_waiting"] == 4.0


def test_parser_tolerates_malformed_lines():
    """Malformed value tokens / label sets are skipped, not raised; good lines kept."""
    body = (
        "vllm:num_requests_waiting not_a_number\n"  # bad value -> skipped
        "vllm:kv_cache_usage_perc{unterminated 0.5\n"  # bad labels -> skipped
        "vllm:num_requests_running 5\n"  # good -> kept
    )
    parsed = parse_prometheus_metrics(body)
    assert parsed == {"vllm:num_requests_running": 5.0}


def test_parser_empty_body_is_empty_dict():
    assert parse_prometheus_metrics("") == {}
    assert parse_prometheus_metrics("\n\n# only comments\n") == {}


# =============================================================================
# Snapshot helpers
# =============================================================================
def test_snapshot_kv_cache_usage_prefers_v1():
    snap = EngineMetricsSnapshot(
        endpoint="x",
        reachable=True,
        gauges={
            "vllm:kv_cache_usage_perc": 0.9,
            "vllm:gpu_cache_usage_perc": 0.1,
        },
    )
    assert snap.kv_cache_usage() == 0.9


def test_snapshot_kv_cache_usage_falls_back_to_v0():
    snap = EngineMetricsSnapshot(
        endpoint="x",
        reachable=True,
        gauges={"vllm:gpu_cache_usage_perc": 0.42},
    )
    assert snap.kv_cache_usage() == 0.42


def test_snapshot_helpers_none_when_absent():
    snap = EngineMetricsSnapshot(endpoint="x", reachable=True, gauges={})
    assert snap.kv_cache_usage() is None
    assert snap.num_waiting() is None
    assert snap.get("vllm:num_requests_waiting") is None


# =============================================================================
# Scraper: default-off
# =============================================================================
@pytest.mark.asyncio
async def test_disabled_scraper_is_noop(monkeypatch):
    """Default-off scraper returns an empty snapshot and does ZERO network I/O."""

    def _boom(*_a, **_k):  # pragma: no cover - asserts it is never called
        raise AssertionError("disabled scraper must not touch the network")

    # Patch the http client pool so any network attempt fails the test loudly.
    monkeypatch.setattr("litellm_llmrouter.http_client_pool.is_pooling_enabled", _boom)

    scraper = EngineMetricsScraper(enabled=False, timeout=2.0)
    snap = await scraper.scrape("http://engine.internal:8000/metrics")

    assert snap.reachable is False
    assert snap.gauges == {}
    assert snap.error == "engine metrics scrape disabled"


# =============================================================================
# Scraper: success / graceful failure (httpx stubbed)
# =============================================================================
class _StubClient:
    """Minimal async httpx-like client returning a canned response or raising."""

    def __init__(self, *, response=None, exc=None):
        self._response = response
        self._exc = exc

    async def get(self, url, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._response


def _install_stub_client(monkeypatch, *, response=None, exc=None):
    """Force the scraper down the pooled-client path with a stub client."""
    monkeypatch.setattr(
        "litellm_llmrouter.http_client_pool.is_pooling_enabled", lambda: True
    )
    monkeypatch.setattr(
        "litellm_llmrouter.http_client_pool.get_http_client",
        lambda: _StubClient(response=response, exc=exc),
    )


@pytest.mark.asyncio
async def test_scrape_success_parses_gauges(monkeypatch):
    resp = httpx.Response(200, text=VLLM_V1_METRICS_BODY)
    _install_stub_client(monkeypatch, response=resp)

    scraper = EngineMetricsScraper(enabled=True, timeout=2.0)
    snap = await scraper.scrape("http://engine.internal:8000/metrics")

    assert snap.reachable is True
    assert snap.num_waiting() == 12.0
    assert snap.kv_cache_usage() == pytest.approx(0.83)
    assert snap.get("vllm:num_requests_running") == 7.0


@pytest.mark.asyncio
async def test_scrape_unreachable_is_graceful_empty(monkeypatch):
    """A connect error yields reachable=False + empty gauges, NOT an exception."""
    _install_stub_client(monkeypatch, exc=httpx.ConnectError("connection refused"))

    scraper = EngineMetricsScraper(enabled=True, timeout=2.0)
    snap = await scraper.scrape("http://down.engine:8000/metrics")

    assert snap.reachable is False
    assert snap.gauges == {}
    assert snap.error == "engine unreachable"


@pytest.mark.asyncio
async def test_scrape_timeout_is_graceful_empty(monkeypatch):
    _install_stub_client(monkeypatch, exc=httpx.ReadTimeout("too slow"))

    scraper = EngineMetricsScraper(enabled=True, timeout=0.1)
    snap = await scraper.scrape("http://slow.engine:8000/metrics")

    assert snap.reachable is False
    assert snap.gauges == {}


@pytest.mark.asyncio
async def test_scrape_non_200_is_graceful_empty(monkeypatch):
    resp = httpx.Response(503, text="Service Unavailable")
    _install_stub_client(monkeypatch, response=resp)

    scraper = EngineMetricsScraper(enabled=True, timeout=2.0)
    snap = await scraper.scrape("http://engine.internal:8000/metrics")

    assert snap.reachable is False
    assert snap.gauges == {}


# =============================================================================
# Singleton
# =============================================================================
def test_singleton_default_off_from_settings(monkeypatch):
    """The singleton is constructed default-off from GatewaySettings."""
    reset_engine_metrics_scraper()
    scraper = get_engine_metrics_scraper()
    assert isinstance(scraper, EngineMetricsScraper)
    assert scraper.enabled is False  # default OFF


def test_reset_clears_singleton():
    first = get_engine_metrics_scraper()
    reset_engine_metrics_scraper()
    second = get_engine_metrics_scraper()
    assert first is not second
