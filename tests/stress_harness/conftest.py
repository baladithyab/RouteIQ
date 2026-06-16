"""Shared pytest fixtures for the RouteIQ stress-harness tests.

Provides:
  * a sys.path shim so ``import stress_harness`` resolves: the package lives at
    ``tools/stress_harness``, so its parent ``tools/`` must be importable (the
    harness is a standalone tool, not part of the installed ``routeiq`` package).
  * a ``make_async_client`` factory building an ``httpx.MockTransport`` so the
    client's request/response-parsing path is exercised with NO live endpoint
    and NO credentials.
  * fixture builders for OpenAI-compat chat responses and RouteIQ stats payloads.

cred-free: every test flows through a mocked transport; no socket is opened.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

# tests/stress_harness/conftest.py -> parents[2] == repo root; tools/ is the
# importable parent of the stress_harness package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _REPO_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


@pytest.fixture
def make_async_client() -> Callable[
    [Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient
]:
    """Factory: given a request handler, return an ``httpx.AsyncClient`` wired to
    a ``MockTransport``. No socket is opened."""

    def _factory(
        handler: Callable[[httpx.Request], httpx.Response],
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return _factory


def make_chat_response(
    *,
    request_id: str = "chatcmpl-test-0001",
    model: str = "anthropic.claude-opus-4-8",
    prompt_tokens: int = 12,
    completion_tokens: int = 34,
    status: int = 200,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> httpx.Response:
    """Build a realistic OpenAI-compat chat completion response for the mock
    transport. ``model`` is the concrete backend RouteIQ chose for the
    ``model:auto`` request."""
    payload = body or {
        "id": request_id,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return httpx.Response(status, json=payload, headers=headers or {})


def make_stats_transport(
    *,
    active_strategy: str | None = "llmrouter-knn",
    available_strategies: list[str] | None = None,
    model_distribution: dict[str, int] | None = None,
    strategy_distribution: dict[str, int] | None = None,
    total_decisions: int = 0,
    config_status: int = 200,
    global_status: int = 200,
    routing_status: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a MockTransport handler that serves RouteIQ's three control-plane
    stats surfaces (routing/config, stats/global, routing/stats) from fixture
    data. Lets tests drive the stats client with NO live control plane."""
    avail = (
        available_strategies
        if available_strategies is not None
        else [
            "llmrouter-knn",
            "llmrouter-cost-aware",
            "kumaraswamy-thompson",
        ]
    )
    md = model_distribution or {}
    sd = strategy_distribution or {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/routing/config"):
            return httpx.Response(
                config_status,
                json={
                    "active_strategy": active_strategy,
                    "available_strategies": avail,
                    "routing_profile": "auto",
                    "centroid_routing_enabled": True,
                    "ab_testing": {"enabled": False, "weights": {}},
                },
            )
        if path.endswith("/stats/global"):
            return httpx.Response(
                global_status,
                json={
                    "total_decisions": total_decisions,
                    "strategy_distribution": sd,
                    "profile_distribution": {},
                    "model_distribution": md,
                    "key_distribution": {},
                    "centroid_decisions": 0,
                    "average_latency_ms": 0.0,
                    "tracked_keys": 0,
                },
            )
        if path.endswith("/routing/stats"):
            return httpx.Response(
                routing_status,
                json={
                    "total_decisions": total_decisions,
                    "strategy_distribution": sd,
                    "profile_distribution": {},
                    "centroid_decisions": 0,
                    "average_latency_ms": 0.0,
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    return handler
