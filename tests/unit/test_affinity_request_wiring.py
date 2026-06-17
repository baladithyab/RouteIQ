"""Live request-path wiring for engine/conversation affinity (RouteIQ-bdd0 + 3316).

These tests drive the REAL :class:`PluginCallbackBridge` deployment + success
hooks (not mocks of them) to prove the affinity mechanisms reach the request
path, while preserving the default-OFF + byte-stable contract and the
unconditional fail-closed input-guardrail seam.

Proves:
  (a) flag OFF  => deployment hook is byte-stable (kwargs unchanged, no tracker)
  (b) flag ON + known affinity => headers injected, caller header NOT clobbered
  (c) success event records response_id -> deployment mapping
  (d) disagg signal => kv_transfer params passthrough, truthy-only
  (e) guardrail evaluation STILL runs unconditionally with affinity ON
  (f) double-fire idempotency (deployment hook + logging hook)
"""

from __future__ import annotations

import pytest

from litellm_llmrouter.conversation_affinity import (
    get_affinity_tracker,
    init_affinity_tracker,
    reset_affinity_tracker,
)
from litellm_llmrouter.engine_affinity import (
    AFFINITY_KEY_HEADER,
    WORKER_INSTANCE_HEADER,
)
from litellm_llmrouter.gateway.plugin_callback_bridge import (
    PluginCallbackBridge,
    reset_callback_bridge,
)
from litellm_llmrouter.settings import reset_settings


@pytest.fixture(autouse=True)
def _reset_wiring_singletons(monkeypatch: pytest.MonkeyPatch):
    """Reset bridge / tracker / settings singletons around each test."""
    monkeypatch.delenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", raising=False)
    reset_callback_bridge()
    reset_affinity_tracker()
    reset_settings()
    yield
    reset_callback_bridge()
    reset_affinity_tracker()
    reset_settings()


def _enable_multinode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "true")
    reset_settings()


# ---------------------------------------------------------------------------
# (a) Flag OFF => deployment hook is byte-stable, no tracker calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_is_byte_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    """OFF: no plugins, no guardrails => hook returns None and kwargs unchanged."""
    # A tracker exists, but the OFF gate must short-circuit before touching it.
    init_affinity_tracker()
    tracker = get_affinity_tracker()
    calls: list[str] = []
    original = tracker.get_affinity

    async def _spy(key: str):  # pragma: no cover - asserted not called
        calls.append(key)
        return await original(key)

    monkeypatch.setattr(tracker, "get_affinity", _spy)

    bridge = PluginCallbackBridge([])  # no plugins
    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "previous_response_id": "resp_prev",
    }
    snapshot = dict(kwargs)

    result = await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    assert result is None  # byte-stable: no kwargs replacement signalled
    assert kwargs == snapshot  # not mutated
    assert "extra_headers" not in kwargs
    assert calls == []  # tracker never consulted when OFF


# ---------------------------------------------------------------------------
# (b) Flag ON + known affinity => headers injected, caller header not clobbered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_known_affinity_injects_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_multinode(monkeypatch)
    tracker = init_affinity_tracker()
    # Prior turn landed on this deployment/worker.
    await tracker.record_response("resp_prev", "openai/gpt-4-decode-7", "gpt-4")

    bridge = PluginCallbackBridge([])  # no plugins -> proves affinity stands alone
    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "next turn"}],
        "previous_response_id": "resp_prev",
    }

    result = await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    # Affinity mutated kwargs => hook MUST return kwargs so litellm picks it up,
    # even though there are zero plugins.
    assert result is kwargs
    headers = kwargs["extra_headers"]
    assert headers[WORKER_INSTANCE_HEADER] == "openai/gpt-4-decode-7"
    assert headers[AFFINITY_KEY_HEADER] == "resp_prev"


@pytest.mark.asyncio
async def test_on_does_not_clobber_caller_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_multinode(monkeypatch)
    tracker = init_affinity_tracker()
    await tracker.record_response("conv-9", "openai/gpt-4-decode-3", "gpt-4")

    bridge = PluginCallbackBridge([])
    kwargs = {
        "model": "gpt-4",
        "messages": [],
        "conversation_id": "conv-9",
        # Caller already set a worker pin — affinity must NOT overwrite it.
        "extra_headers": {WORKER_INSTANCE_HEADER: "caller-pinned-worker"},
    }

    await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    headers = kwargs["extra_headers"]
    assert headers[WORKER_INSTANCE_HEADER] == "caller-pinned-worker"  # preserved
    # Affinity key (a key the caller did NOT set) still gets filled in.
    assert headers[AFFINITY_KEY_HEADER] == "conv-9"


@pytest.mark.asyncio
async def test_on_no_known_affinity_is_byte_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ON but no affinity key and no disagg signal => still a byte-stable no-op."""
    _enable_multinode(monkeypatch)
    init_affinity_tracker()

    bridge = PluginCallbackBridge([])
    kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "stateless"}]}
    snapshot = dict(kwargs)

    result = await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    assert result is None
    assert kwargs == snapshot
    assert "extra_headers" not in kwargs


@pytest.mark.asyncio
async def test_on_affinity_key_without_record_echoes_key_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ON + affinity key present but no stored mapping => echo key, no worker hint."""
    _enable_multinode(monkeypatch)
    init_affinity_tracker()  # empty tracker

    bridge = PluginCallbackBridge([])
    kwargs = {"model": "gpt-4", "messages": [], "session_id": "sess-42"}

    result = await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    assert result is kwargs
    headers = kwargs["extra_headers"]
    assert headers[AFFINITY_KEY_HEADER] == "sess-42"
    # No prior mapping => no sticky worker hint.
    assert WORKER_INSTANCE_HEADER not in headers


# ---------------------------------------------------------------------------
# (c) Success event records response_id -> deployment mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_event_records_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_multinode(monkeypatch)
    tracker = init_affinity_tracker()

    bridge = PluginCallbackBridge([])
    kwargs = {
        "model": "gpt-4",
        "litellm_params": {"model": "openai/gpt-4-decode-5"},
    }
    response = {"id": "resp_new_123", "choices": [{"message": {"content": "ok"}}]}

    await bridge.async_log_success_event(kwargs, response, None, None)

    record = await tracker.get_affinity("resp_new_123")
    assert record is not None
    # The concrete provider model is recorded as the sticky deployment hint.
    assert record.provider_deployment == "openai/gpt-4-decode-5"
    assert record.model == "gpt-4"


@pytest.mark.asyncio
async def test_success_record_then_next_turn_sticky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: record on success, then the next turn injects the worker hint."""
    _enable_multinode(monkeypatch)
    tracker = init_affinity_tracker()

    bridge = PluginCallbackBridge([])

    # Turn 1: success records resp_T1 -> deployment.
    await bridge.async_log_success_event(
        {"model": "gpt-4", "litellm_params": {"model": "openai/gpt-4-decode-9"}},
        {"id": "resp_T1"},
        None,
        None,
    )
    assert await tracker.get_affinity("resp_T1") is not None

    # Turn 2: previous_response_id=resp_T1 => sticky header injected.
    kwargs = {"model": "gpt-4", "messages": [], "previous_response_id": "resp_T1"}
    await bridge.async_pre_call_deployment_hook(kwargs, "completion")
    assert kwargs["extra_headers"][WORKER_INSTANCE_HEADER] == "openai/gpt-4-decode-9"


@pytest.mark.asyncio
async def test_success_event_off_does_not_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag OFF: success event records nothing."""
    # OFF (no enable). Tracker exists to prove the gate, not the tracker, is the no-op.
    tracker = init_affinity_tracker()
    bridge = PluginCallbackBridge([])

    await bridge.async_log_success_event(
        {"model": "gpt-4", "litellm_params": {"model": "openai/gpt-4"}},
        {"id": "resp_off"},
        None,
        None,
    )

    assert await tracker.get_affinity("resp_off") is None


# ---------------------------------------------------------------------------
# (d) Disagg signal => kv_transfer params passthrough, truthy-only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disagg_per_request_metadata_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_multinode(monkeypatch)
    init_affinity_tracker()

    bridge = PluginCallbackBridge([])
    kwargs = {
        "model": "gpt-4",
        "messages": [],
        "metadata": {
            "do_remote_prefill": True,
            "do_remote_decode": True,
            "kv_transfer_params": {"engine_id": "e1"},
        },
    }

    result = await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    assert result is kwargs
    assert kwargs["do_remote_prefill"] is True
    assert kwargs["do_remote_decode"] is True
    assert kwargs["kv_transfer_params"] == {"engine_id": "e1"}


@pytest.mark.asyncio
async def test_disagg_truthy_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only truthy disagg flags are emitted; a falsy flag adds nothing."""
    _enable_multinode(monkeypatch)
    init_affinity_tracker()

    bridge = PluginCallbackBridge([])
    kwargs = {
        "model": "gpt-4",
        "messages": [],
        "metadata": {"do_remote_decode": True, "do_remote_prefill": False},
    }

    await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    assert kwargs["do_remote_decode"] is True
    # Falsy prefill flag must NOT be carried (truthy-only contract).
    assert "do_remote_prefill" not in kwargs


@pytest.mark.asyncio
async def test_disagg_no_signal_byte_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ON, no affinity key, no disagg signal => zero params added (byte-stable)."""
    _enable_multinode(monkeypatch)
    init_affinity_tracker()

    bridge = PluginCallbackBridge([])
    kwargs = {"model": "gpt-4", "messages": []}
    snapshot = dict(kwargs)

    result = await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    assert result is None
    assert kwargs == snapshot


@pytest.mark.asyncio
async def test_disagg_default_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings-level disagg default carries the flag with no per-request hint."""
    monkeypatch.setenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "true")
    monkeypatch.setenv(
        "ROUTEIQ_MULTINODE_AFFINITY__DISAGG_DEFAULT_REMOTE_DECODE", "true"
    )
    reset_settings()
    init_affinity_tracker()

    bridge = PluginCallbackBridge([])
    kwargs = {"model": "gpt-4", "messages": []}

    result = await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    assert result is kwargs
    assert kwargs["do_remote_decode"] is True
    assert "do_remote_prefill" not in kwargs  # default prefill stays OFF


@pytest.mark.asyncio
async def test_disagg_metadata_overrides_settings_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-request metadata flag overrides the settings-level default."""
    monkeypatch.setenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "true")
    monkeypatch.setenv(
        "ROUTEIQ_MULTINODE_AFFINITY__DISAGG_DEFAULT_REMOTE_DECODE", "true"
    )
    reset_settings()
    init_affinity_tracker()

    bridge = PluginCallbackBridge([])
    kwargs = {
        "model": "gpt-4",
        "messages": [],
        "metadata": {"do_remote_decode": False},
    }

    result = await bridge.async_pre_call_deployment_hook(kwargs, "completion")

    # Per-request explicit False overrides settings default True => nothing carried.
    assert result is None
    assert "do_remote_decode" not in kwargs


# ---------------------------------------------------------------------------
# (e) Guardrail evaluation STILL runs unconditionally with affinity ON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrails_still_fail_closed_with_affinity_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured DENY input guardrail still blocks even with affinity ON and
    zero plugins (fail-closed preserved; affinity does not short-circuit it)."""
    from fastapi import HTTPException

    from litellm_llmrouter.guardrail_policies import (
        GuardrailAction,
        GuardrailPhase,
        GuardrailPolicy,
        GuardrailType,
        get_guardrail_policy_engine,
        reset_guardrail_policy_engine,
    )

    _enable_multinode(monkeypatch)
    init_affinity_tracker()
    reset_guardrail_policy_engine()
    engine = get_guardrail_policy_engine()
    # A regex-deny DENY policy that matches the request content.
    engine.add_policy(
        GuardrailPolicy(
            guardrail_id="deny-secret",
            name="deny-secret",
            phase=GuardrailPhase.INPUT,
            check_type=GuardrailType.REGEX_DENY,
            action=GuardrailAction.DENY,
            parameters={"patterns": ["(?i)forbidden"]},
        )
    )
    try:
        bridge = PluginCallbackBridge([])  # zero plugins
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "this is forbidden"}],
            "previous_response_id": "resp_prev",
        }

        with pytest.raises(HTTPException) as exc_info:
            await bridge.async_pre_call_deployment_hook(kwargs, "completion")

        assert exc_info.value.status_code == 446
        # Guardrail denied BEFORE affinity headers were attached to the wire.
        assert "extra_headers" not in kwargs
    finally:
        reset_guardrail_policy_engine()


# ---------------------------------------------------------------------------
# (f) Double-fire idempotency (deployment hook + logging hook)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_double_fire_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-firing the deployment hook (and the logging hook) does not double-apply."""
    _enable_multinode(monkeypatch)
    tracker = init_affinity_tracker()
    await tracker.record_response("resp_prev", "openai/gpt-4-decode-1", "gpt-4")

    bridge = PluginCallbackBridge([])
    kwargs = {
        "model": "gpt-4",
        "messages": [],
        "previous_response_id": "resp_prev",
    }

    # First fire: injects headers.
    await bridge.async_pre_call_deployment_hook(kwargs, "completion")
    headers_after_first = dict(kwargs["extra_headers"])
    assert headers_after_first[WORKER_INSTANCE_HEADER] == "openai/gpt-4-decode-1"

    # Re-fire the SAME kwargs through both seams: sentinel must short-circuit, so
    # the headers dict is unchanged (idempotent).
    await bridge.async_pre_call_deployment_hook(kwargs, "completion")
    await bridge.async_log_pre_api_call("gpt-4", [], kwargs)

    assert dict(kwargs["extra_headers"]) == headers_after_first
