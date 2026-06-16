"""Tests for the RouteIQ control-plane stats client (stats_client.py).

Drives the three-surface reader through a MockTransport — NO live control plane,
NO credentials. Asserts it NAMES the active strategy, merges the distributions,
sends the admin header, and degrades when a surface is unreachable. — RouteIQ-4f19 (d).
"""

from __future__ import annotations

import asyncio

import httpx

from stress_harness.stats_client import RouteIQStatsClient

from .conftest import make_stats_transport


def _run(coro):
    return asyncio.run(coro)


def test_reads_active_strategy_and_distributions(make_async_client):
    handler = make_stats_transport(
        active_strategy="kumaraswamy-thompson",
        model_distribution={"model-a": 30, "model-b": 20},
        strategy_distribution={"kumaraswamy-thompson": 50},
        total_decisions=50,
    )
    client = RouteIQStatsClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    stats = _run(client.fetch())
    assert stats.active_strategy == "kumaraswamy-thompson"
    assert stats.model_distribution == {"model-a": 30, "model-b": 20}
    assert stats.strategy_distribution == {"kumaraswamy-thompson": 50}
    assert stats.total_decisions == 50
    assert "kumaraswamy-thompson" in stats.available_strategies


def test_sends_admin_header(make_async_client):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.setdefault("admin", request.headers.get("x-admin-api-key"))
        return make_stats_transport()(request)

    client = RouteIQStatsClient(
        "http://routeiq.local",
        admin_key="test-admin-key",
        http_client=make_async_client(handler),
    )
    _run(client.fetch())
    assert captured["admin"] == "test-admin-key"


def test_degrades_when_config_surface_unreachable(make_async_client):
    # /routing/config 404s -> active_strategy unknown, but model_distribution
    # from /stats/global still read; a note is recorded, nothing raises.
    handler = make_stats_transport(
        config_status=404,
        model_distribution={"m": 5},
        active_strategy="llmrouter-knn",
    )
    client = RouteIQStatsClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    stats = _run(client.fetch())
    # config 404 means active_strategy cannot come from config; the stats
    # payloads in this fixture also DON'T carry active_strategy, so it stays None.
    assert stats.active_strategy is None
    assert stats.model_distribution == {"m": 5}
    assert any("routing/config" in n for n in stats.notes)


def test_active_strategy_falls_back_to_stats_payload(make_async_client):
    # config 404s but a stats payload carries active_strategy.
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/routing/config"):
            return httpx.Response(404, json={})
        if path.endswith("/stats/global"):
            return httpx.Response(
                200,
                json={
                    "active_strategy": "llmrouter-mlp",
                    "model_distribution": {"x": 1},
                    "strategy_distribution": {},
                    "total_decisions": 1,
                },
            )
        return httpx.Response(404, json={})

    client = RouteIQStatsClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    stats = _run(client.fetch())
    assert stats.active_strategy == "llmrouter-mlp"


def test_all_surfaces_unreachable_does_not_raise(make_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = RouteIQStatsClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    stats = _run(client.fetch())
    assert stats.active_strategy is None
    assert stats.model_distribution == {}
    assert len(stats.notes) >= 1
