"""Unit tests for upstream LiteLLM router delegation (RouteIQ-8539).

Verifies RouteIQ can SELECT an upstream LiteLLM adaptive/quality/auto router
(off the live ``Router`` instance) and FLUSH the selected adaptive router's
update queue to the durable store -- both cred-free against a mock LiteLLM
router + a mock adaptive-router queue.

Default OFF: ``get_upstream_router_delegate()`` returns None until
``settings.mlops.upstream_router.enabled`` is set; the eval-loop flush wiring
is a no-op until then.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm_llmrouter.mlops.upstream_router as ur
from litellm_llmrouter.eval_pipeline import EvalPipeline, reset_eval_pipeline
from litellm_llmrouter.mlops.upstream_router import (
    UPSTREAM_ROUTER_MODES,
    UpstreamRouterDelegate,
    get_upstream_router_delegate,
    reset_upstream_router_delegate,
    wire_upstream_router_flush,
)
from litellm_llmrouter.settings import reset_settings


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    reset_upstream_router_delegate()
    reset_eval_pipeline()
    yield
    reset_settings()
    reset_upstream_router_delegate()
    reset_eval_pipeline()


def _mock_adaptive_router(
    *, name: str = "router-1", state_rows: int = 3, session_rows: int = 2
) -> SimpleNamespace:
    """A mock upstream adaptive router with a flushable update queue."""
    queue = MagicMock()
    queue.flush_state_to_db = AsyncMock(return_value=state_rows)
    queue.flush_session_to_db = AsyncMock(return_value=session_rows)
    return SimpleNamespace(router_name=name, queue=queue)


def _mock_llm_router(**registries) -> SimpleNamespace:
    """A mock LiteLLM Router carrying per-mode router registries."""
    return SimpleNamespace(
        adaptive_routers=registries.get("adaptive", {}),
        quality_routers=registries.get("quality", {}),
        auto_routers=registries.get("auto", {}),
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_modes_cover_all_three_upstream_families():
    assert set(UPSTREAM_ROUTER_MODES) == {"adaptive", "quality", "auto"}


def test_resolve_first_router_of_mode():
    r = _mock_adaptive_router(name="adaptive-1")
    router = _mock_llm_router(adaptive={"adaptive-1": r})
    delegate = UpstreamRouterDelegate(mode="adaptive")
    assert delegate.resolve_router(router) is r
    assert delegate.is_available(router) is True


def test_resolve_named_router():
    a = _mock_adaptive_router(name="a")
    b = _mock_adaptive_router(name="b")
    router = _mock_llm_router(adaptive={"a": a, "b": b})
    delegate = UpstreamRouterDelegate(mode="adaptive", router_name="b")
    assert delegate.resolve_router(router) is b


def test_resolve_none_when_mode_empty():
    router = _mock_llm_router(adaptive={})
    delegate = UpstreamRouterDelegate(mode="adaptive")
    assert delegate.resolve_router(router) is None
    assert delegate.is_available(router) is False


def test_invalid_mode_falls_back_to_adaptive():
    delegate = UpstreamRouterDelegate(mode="nonsense")
    assert delegate.mode == "adaptive"


# ---------------------------------------------------------------------------
# Queue flush to the durable store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_drains_state_and_session_queues():
    r = _mock_adaptive_router(state_rows=5, session_rows=4)
    router = _mock_llm_router(adaptive={"router-1": r})
    prisma = MagicMock()  # opaque durable client; passed straight through
    delegate = UpstreamRouterDelegate(mode="adaptive")

    result = await delegate.flush(llm_router=router, prisma_client=prisma)

    assert result.flushed is True
    assert result.state_rows == 5
    assert result.session_rows == 4
    assert result.router_name == "router-1"
    # Both queue halves were drained THROUGH the provided durable client.
    r.queue.flush_state_to_db.assert_awaited_once_with(prisma)
    r.queue.flush_session_to_db.assert_awaited_once_with(prisma)


@pytest.mark.asyncio
async def test_flush_noop_when_no_upstream_router():
    router = _mock_llm_router(adaptive={})
    delegate = UpstreamRouterDelegate(mode="adaptive")
    result = await delegate.flush(llm_router=router, prisma_client=MagicMock())
    assert result.flushed is False
    assert result.reason == "no_upstream_router"


@pytest.mark.asyncio
async def test_flush_noop_when_mode_has_no_queue():
    # quality/auto routers have no .queue attribute.
    quality_router = SimpleNamespace(router_name="q1")
    router = _mock_llm_router(quality={"q1": quality_router})
    delegate = UpstreamRouterDelegate(mode="quality")
    result = await delegate.flush(llm_router=router, prisma_client=MagicMock())
    assert result.flushed is False
    assert result.reason == "mode_has_no_queue"


@pytest.mark.asyncio
async def test_flush_noop_when_no_durable_client():
    r = _mock_adaptive_router()
    router = _mock_llm_router(adaptive={"router-1": r})
    delegate = UpstreamRouterDelegate(mode="adaptive")
    result = await delegate.flush(llm_router=router, prisma_client=None)
    assert result.flushed is False
    assert result.reason == "no_durable_client"
    r.queue.flush_state_to_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_flush_disabled_short_circuits():
    r = _mock_adaptive_router()
    router = _mock_llm_router(adaptive={"router-1": r})
    delegate = UpstreamRouterDelegate(mode="adaptive", flush_queue=False)
    result = await delegate.flush(llm_router=router, prisma_client=MagicMock())
    assert result.flushed is False
    assert result.reason == "flush_disabled"


# ---------------------------------------------------------------------------
# Settings gating (default off)
# ---------------------------------------------------------------------------


def test_delegate_disabled_by_default():
    assert get_upstream_router_delegate() is None


def test_delegate_enabled_via_settings(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_MLOPS__UPSTREAM_ROUTER__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_MLOPS__UPSTREAM_ROUTER__MODE", "quality")
    monkeypatch.setenv("ROUTEIQ_MLOPS__UPSTREAM_ROUTER__ROUTER_NAME", "q1")
    reset_settings()
    reset_upstream_router_delegate()
    delegate = get_upstream_router_delegate()
    assert delegate is not None
    assert delegate.mode == "quality"
    assert delegate.router_name == "q1"


# ---------------------------------------------------------------------------
# Eval-loop flush wiring (feedback arm)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wire_flush_subscribes_and_flushes_on_feedback(monkeypatch):
    r = _mock_adaptive_router(state_rows=2, session_rows=1)
    router = _mock_llm_router(adaptive={"router-1": r})
    prisma = MagicMock()

    delegate = UpstreamRouterDelegate(mode="adaptive")
    # Route the singleton + LiteLLM globals to our mocks.
    monkeypatch.setattr(ur, "get_upstream_router_delegate", lambda: delegate)
    monkeypatch.setattr(
        UpstreamRouterDelegate, "_litellm_router", staticmethod(lambda: router)
    )
    monkeypatch.setattr(
        UpstreamRouterDelegate, "_prisma_client", staticmethod(lambda: prisma)
    )

    pipeline = EvalPipeline(sample_rate=1.0)
    pipeline.tracker.record("gpt-4o", 0.8)

    assert wire_upstream_router_flush(eval_pipeline=pipeline, force=True) is True
    # Second wire is a no-op (callback already subscribed).
    assert wire_upstream_router_flush(eval_pipeline=pipeline, force=True) is False

    await pipeline.push_feedback()

    # The upstream queue was flushed via the FEEDBACK arm.
    r.queue.flush_state_to_db.assert_awaited_once_with(prisma)
    r.queue.flush_session_to_db.assert_awaited_once_with(prisma)


def test_wire_skipped_when_disabled():
    pipeline = EvalPipeline(sample_rate=1.0)
    # Delegation disabled (default settings) -> wiring refuses without force.
    assert wire_upstream_router_flush(eval_pipeline=pipeline, force=False) is False
