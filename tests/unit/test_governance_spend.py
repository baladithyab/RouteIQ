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

from fastapi import HTTPException

from litellm_llmrouter.governance import (
    GovernanceContext,
    GovernanceEngine,
    KeyGovernance,
    WorkspaceConfig,
    derive_spend_scope_from_ctx,
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


# -- Canonical scope: WRITE scope == READ scope (RouteIQ-ed7a + 08dd) --------


def test_derive_spend_scope_from_ctx_precedence():
    """The single source-of-truth helper used by BOTH read and write paths."""
    # workspace wins over key
    ctx = GovernanceContext(workspace_id="ws-1", key_id="sk-1")
    assert derive_spend_scope_from_ctx(ctx) == ("ws-1", "workspace")
    # key-only -> raw key (not a hash / user-id)
    ctx = GovernanceContext(key_id="sk-raw-2")
    assert derive_spend_scope_from_ctx(ctx) == ("sk-raw-2", "key")
    # neither -> global
    assert derive_spend_scope_from_ctx(GovernanceContext()) == ("global", "global")


def test_writer_prefers_governance_ctx_stamp():
    """The enforce-path stamp is honoured with the read precedence (ed7a/08dd)."""
    from litellm_llmrouter.router_decision_callback import _derive_spend_scope

    # workspace stamp wins (ed7a)
    assert _derive_spend_scope(
        {"_governance_ctx": {"workspace_id": "ws-b", "key_id": "sk-1"}}
    ) == ("ws-b", "workspace")
    # key stamp with no workspace -> raw key (08dd), NOT LiteLLM's hashed token
    assert _derive_spend_scope(
        {
            "_governance_ctx": {"workspace_id": None, "key_id": "sk-key-only"},
            "user_api_key": "hashed-deadbeef",  # legacy candidate must be ignored
        }
    ) == ("sk-key-only", "key")
    # explicit stamp with neither -> global
    assert _derive_spend_scope({"_governance_ctx": {}}) == ("global", "global")
    # no stamp -> legacy fallback preserved (back-compat)
    assert _derive_spend_scope({"workspace_id": "ws-legacy"}) == (
        "ws-legacy",
        "workspace",
    )


@pytest.mark.asyncio
async def test_workspace_budget_write_seen_by_read_and_enforced(monkeypatch):
    """RouteIQ-ed7a: spend written via the success-callback path lands on the
    workspace scope the budget READ checks, so an over-budget workspace is denied."""
    monkeypatch.delenv("DATABASE_URL", raising=False)  # Aurora off
    from litellm_llmrouter.governance import reset_governance_engine
    from litellm_llmrouter.router_decision_callback import (
        _record_post_response_spend,
    )

    reset_governance_engine()
    engine = GovernanceEngine()
    engine.register_workspace(
        WorkspaceConfig(workspace_id="ws-b", name="Beta", max_budget_usd=10.0)
    )
    engine.register_key_governance(
        KeyGovernance(key_id="sk-test-1", workspace_id="ws-b")
    )

    redis = _make_fake_redis()

    class _Usage:
        prompt_tokens = 5
        completion_tokens = 5

    class _Resp:
        usage = _Usage()

    # Isolate the governance spend seam: stub the usage-policy arm of the writer
    # (a separate counter store) so this test asserts only the governance
    # write==read scope wiring (ed7a) and not the usage-policy side effect.
    usage_engine = AsyncMock()
    usage_engine.record_usage = AsyncMock(return_value=0)

    with (
        patch(
            "litellm_llmrouter.redis_pool.get_async_redis_client",
            AsyncMock(return_value=redis),
        ),
        patch(
            "litellm_llmrouter.usage_policies.get_usage_policy_engine",
            return_value=usage_engine,
        ),
    ):
        # Write spend through the success-callback writer using the stamp the
        # enforce path produces (resolved workspace_id + raw key_id).
        await _record_post_response_spend(
            {
                "metadata": {
                    "_governance_ctx": {
                        "workspace_id": "ws-b",
                        "key_id": "sk-test-1",
                    }
                },
                "response_cost": 12.0,
                "model": "m",
            },
            _Resp(),
        )

        # The write must have landed on the workspace read scope.
        spend_bucket = int(time.time() // 2_592_000)
        assert redis._backing[f"governance:spend:ws-b:{spend_bucket}"] == 12.0

        # The workspace is over its $10 budget -> enforce DENIES.
        with pytest.raises(HTTPException) as ei:
            await engine.enforce("sk-test-1", "gpt-4o")
        assert ei.value.status_code == 429
        assert ei.value.detail["error"] == "budget_exceeded"

    reset_governance_engine()


@pytest.mark.asyncio
async def test_workspace_under_budget_allows(monkeypatch):
    """Control: a workspace under budget resolves without raising."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from litellm_llmrouter.governance import reset_governance_engine

    reset_governance_engine()
    engine = GovernanceEngine()
    engine.register_workspace(
        WorkspaceConfig(workspace_id="ws-ok", name="OK", max_budget_usd=100.0)
    )
    engine.register_key_governance(
        KeyGovernance(key_id="sk-ok-1", workspace_id="ws-ok")
    )

    redis = _make_fake_redis(
        {f"governance:spend:ws-ok:{int(time.time() // 2_592_000)}": 5.0}
    )
    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=redis),
    ):
        ctx = await engine.enforce("sk-ok-1", "gpt-4o")
    assert ctx.workspace_id == "ws-ok"
    reset_governance_engine()


@pytest.mark.asyncio
async def test_key_budget_write_seen_by_read_and_enforced(monkeypatch):
    """RouteIQ-08dd: key-level spend (no workspace) written via the callback path
    lands on the RAW key scope the read checks, so an over-budget key is denied."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from litellm_llmrouter.governance import reset_governance_engine
    from litellm_llmrouter.router_decision_callback import (
        _record_post_response_spend,
    )

    reset_governance_engine()
    engine = GovernanceEngine()
    engine.register_key_governance(
        KeyGovernance(key_id="sk-key-only", max_budget_usd=5.0)  # no workspace
    )

    redis = _make_fake_redis()

    class _Usage:
        prompt_tokens = 3
        completion_tokens = 2

    class _Resp:
        usage = _Usage()

    # Isolate the governance spend seam (see ed7a test) -- stub the usage-policy arm.
    usage_engine = AsyncMock()
    usage_engine.record_usage = AsyncMock(return_value=0)

    with (
        patch(
            "litellm_llmrouter.redis_pool.get_async_redis_client",
            AsyncMock(return_value=redis),
        ),
        patch(
            "litellm_llmrouter.usage_policies.get_usage_policy_engine",
            return_value=usage_engine,
        ),
    ):
        await _record_post_response_spend(
            {
                "metadata": {
                    "_governance_ctx": {
                        "workspace_id": None,
                        "key_id": "sk-key-only",
                    }
                },
                "response_cost": 6.0,
                "model": "m",
            },
            _Resp(),
        )

        spend_bucket = int(time.time() // 2_592_000)
        assert redis._backing[f"governance:spend:sk-key-only:{spend_bucket}"] == 6.0

        with pytest.raises(HTTPException) as ei:
            await engine.enforce("sk-key-only", "gpt-4o")
        assert ei.value.status_code == 429
        assert ei.value.detail["error"] == "budget_exceeded"

    reset_governance_engine()


# -- Fail-closed mode when the spend store is down (RouteIQ-24fc) -------------


@pytest.mark.asyncio
async def test_budget_fail_closed_denies_when_store_down(monkeypatch):
    """RouteIQ-24fc: store unavailable + a budget configured + fail_mode=closed
    -> check_budget denies (no spend leak)."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_FAIL_MODE", "closed")
    from litellm_llmrouter.settings import reset_settings

    reset_settings()

    engine = GovernanceEngine()
    ctx = GovernanceContext(workspace_id="ws-fc")
    ctx.effective_max_budget_usd = 10.0  # a budget IS set

    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=None),  # store unavailable
    ):
        assert await engine.check_budget(ctx) is False


@pytest.mark.asyncio
async def test_budget_fail_open_allows_when_store_down(monkeypatch):
    """RouteIQ-24fc: default fail_mode=open allows when the store is down."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ROUTEIQ_GOVERNANCE_FAIL_MODE", raising=False)
    from litellm_llmrouter.settings import reset_settings

    reset_settings()

    engine = GovernanceEngine()
    ctx = GovernanceContext(workspace_id="ws-fo")
    ctx.effective_max_budget_usd = 10.0

    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=None),
    ):
        assert await engine.check_budget(ctx) is True


@pytest.mark.asyncio
async def test_budget_fail_closed_no_limit_still_allows(monkeypatch):
    """RouteIQ-24fc guard: fail-closed must NOT deny when NO budget is configured."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_FAIL_MODE", "closed")
    from litellm_llmrouter.settings import reset_settings

    reset_settings()

    engine = GovernanceEngine()
    ctx = GovernanceContext(workspace_id="ws-nolimit")  # effective_max_budget_usd=None

    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=None),
    ):
        assert await engine.check_budget(ctx) is True


@pytest.mark.asyncio
async def test_rate_limit_fail_closed_denies_when_store_down(monkeypatch):
    """RouteIQ-24fc: rate-limit mirrors budget -- closed denies on store-down."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_FAIL_MODE", "closed")
    from litellm_llmrouter.settings import reset_settings

    reset_settings()

    engine = GovernanceEngine()
    ctx = GovernanceContext(workspace_id="ws-rl")
    ctx.effective_max_rpm = 5

    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=None),
    ):
        assert await engine.check_rate_limit(ctx) is False


@pytest.mark.asyncio
async def test_rate_limit_fail_open_allows_when_store_down(monkeypatch):
    """RouteIQ-24fc: default open allows rate-limit check when store down."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ROUTEIQ_GOVERNANCE_FAIL_MODE", raising=False)
    from litellm_llmrouter.settings import reset_settings

    reset_settings()

    engine = GovernanceEngine()
    ctx = GovernanceContext(workspace_id="ws-rl-open")
    ctx.effective_max_rpm = 5

    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=None),
    ):
        assert await engine.check_rate_limit(ctx) is True


@pytest.mark.asyncio
async def test_enforce_fail_closed_denies_when_store_down(monkeypatch):
    """RouteIQ-24fc end-to-end: enforce() denies a budgeted key when the store is
    down under fail_mode=closed (the spend leak is closed at the enforce seam)."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_FAIL_MODE", "closed")
    from litellm_llmrouter.governance import reset_governance_engine
    from litellm_llmrouter.settings import reset_settings

    reset_settings()
    reset_governance_engine()
    engine = GovernanceEngine()
    engine.register_key_governance(
        KeyGovernance(key_id="sk-fc-enforce", max_budget_usd=10.0)
    )

    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=None),  # store down
    ):
        with pytest.raises(HTTPException) as ei:
            await engine.enforce("sk-fc-enforce", "gpt-4o")
        assert ei.value.status_code == 429
        assert ei.value.detail["error"] == "budget_exceeded"

    reset_governance_engine()
