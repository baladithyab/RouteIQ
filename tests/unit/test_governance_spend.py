"""Unit tests for the governance spend WRITE path (the latent-gap fix).

Before P4 nothing wrote ``governance:spend:`` / ``governance:rpm:`` -- so the
GovernanceEngine budget/RPM reads always saw 0.0 (fail-open).  These tests use a
mocked async Redis to prove:
  * record_governance_spend writes the EXACT key shape the read path consumes,
  * a write then becomes visible to GovernanceEngine._get_current_spend,
  * the Aurora rollup is skipped when the store is disabled (still no raise),
  * a missing Redis client is a fail-open no-op,
  * the async_log_success_event spend hook honours the feature flag.
"""

import time
from unittest.mock import AsyncMock, patch

import pytest

from litellm_llmrouter.governance import (
    GovernanceContext,
    GovernanceEngine,
    record_governance_spend,
)


def _make_fake_redis(initial: dict | None = None):
    """An AsyncMock Redis that tracks INCRBYFLOAT/INCRBY into a backing dict."""
    store = dict(initial or {})
    redis = AsyncMock()

    async def _incrbyfloat(key, amount):
        store[key] = float(store.get(key, 0.0)) + float(amount)
        return store[key]

    async def _incrby(key, amount):
        store[key] = int(store.get(key, 0)) + int(amount)
        return store[key]

    async def _get(key):
        val = store.get(key)
        return str(val) if val is not None else None

    redis.incrbyfloat = AsyncMock(side_effect=_incrbyfloat)
    redis.incrby = AsyncMock(side_effect=_incrby)
    redis.expire = AsyncMock(return_value=True)
    redis.get = AsyncMock(side_effect=_get)
    redis._backing = store
    return redis


@pytest.mark.asyncio
async def test_record_spend_writes_exact_key_shape(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)  # store disabled
    redis = _make_fake_redis()

    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=redis),
    ):
        await record_governance_spend(
            "ws-1", "workspace", cost=1.5, tokens=100, requests=1
        )

    spend_bucket = int(time.time() // 2_592_000)
    minute_bucket = int(time.time() // 60)
    spend_key = f"governance:spend:ws-1:{spend_bucket}"
    rpm_key = f"governance:rpm:ws-1:{minute_bucket}"

    # The key shape MUST match the read path exactly.
    redis.incrbyfloat.assert_awaited_once_with(spend_key, 1.5)
    redis.incrby.assert_awaited_once_with(rpm_key, 1)
    # TTLs applied.
    assert redis.expire.await_count == 2
    assert redis._backing[spend_key] == 1.5
    assert redis._backing[rpm_key] == 1


@pytest.mark.asyncio
async def test_write_then_read_roundtrip_through_engine(monkeypatch):
    """A spend write is visible to GovernanceEngine._get_current_spend (read path
    that previously always saw 0.0)."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    redis = _make_fake_redis()

    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=redis),
    ):
        await record_governance_spend(
            "ws-7", "workspace", cost=4.25, tokens=10, requests=1
        )

        engine = GovernanceEngine()
        ctx = GovernanceContext(workspace_id="ws-7")
        spend = await engine._get_current_spend(ctx)

    assert spend == 4.25


@pytest.mark.asyncio
async def test_store_disabled_still_writes_redis_no_raise(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    redis = _make_fake_redis()
    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=redis),
    ):
        # Aurora rollup skipped (store disabled) but Redis still written, no raise.
        await record_governance_spend("k-1", "key", cost=0.5, requests=1)
    assert redis.incrbyfloat.await_count == 1


@pytest.mark.asyncio
async def test_redis_unavailable_is_failopen_noop(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=None),
    ):
        # No redis client -> no-op, must not raise.
        await record_governance_spend("ws-x", "workspace", cost=9.9, requests=1)


@pytest.mark.asyncio
async def test_zero_cost_only_writes_rpm(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    redis = _make_fake_redis()
    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=redis),
    ):
        await record_governance_spend("ws-2", "workspace", cost=0.0, requests=1)
    # cost==0 -> no spend incr, but rpm still increments.
    redis.incrbyfloat.assert_not_awaited()
    redis.incrby.assert_awaited_once()


# -- Feature-flag gating of the success-callback hook -----------------------


@pytest.mark.asyncio
async def test_success_event_calls_spend_when_flag_on(monkeypatch):
    monkeypatch.setenv("LLMROUTER_GOVERNANCE_SPEND_TRACKING", "true")
    from litellm_llmrouter.router_decision_callback import RouterDecisionCallback

    cb = RouterDecisionCallback(enabled=False)  # OTEL telemetry off

    recorded = AsyncMock()
    with patch(
        "litellm_llmrouter.router_decision_callback._record_post_response_spend",
        recorded,
    ):

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5

        class _Resp:
            usage = _Usage()

        await cb.async_log_success_event(
            {"metadata": {"workspace_id": "ws-1"}, "response_cost": 0.01},
            _Resp(),
            0.0,
            1.0,
        )
    recorded.assert_awaited_once()


@pytest.mark.asyncio
async def test_success_event_skips_spend_when_flag_off(monkeypatch):
    monkeypatch.setenv("LLMROUTER_GOVERNANCE_SPEND_TRACKING", "false")
    from litellm_llmrouter.router_decision_callback import RouterDecisionCallback

    cb = RouterDecisionCallback(enabled=False)

    recorded = AsyncMock()
    with patch(
        "litellm_llmrouter.router_decision_callback._record_post_response_spend",
        recorded,
    ):
        await cb.async_log_success_event(
            {"metadata": {"workspace_id": "ws-1"}}, object(), 0.0, 1.0
        )
    recorded.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_post_response_spend_derives_scope(monkeypatch):
    """_record_post_response_spend extracts cost+tokens and derives workspace
    scope, then calls record_governance_spend."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from litellm_llmrouter.router_decision_callback import (
        _derive_spend_scope,
        _record_post_response_spend,
    )

    assert _derive_spend_scope({"workspace_id": "ws-9"}) == ("ws-9", "workspace")
    assert _derive_spend_scope({"_user": "u-1"}) == ("u-1", "key")
    assert _derive_spend_scope({}) == ("global", "global")

    gov_spy = AsyncMock()

    class _Usage:
        prompt_tokens = 7
        completion_tokens = 3

    class _Resp:
        usage = _Usage()

    with patch("litellm_llmrouter.governance.record_governance_spend", gov_spy):
        await _record_post_response_spend(
            {"metadata": {"workspace_id": "ws-9"}, "response_cost": 0.02, "model": "m"},
            _Resp(),
        )
    gov_spy.assert_awaited_once()
    _args, kwargs = gov_spy.await_args
    assert _args[0] == "ws-9"
    assert _args[1] == "workspace"
    assert kwargs["cost"] == 0.02
    assert kwargs["tokens"] == 10
