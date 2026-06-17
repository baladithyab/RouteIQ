"""Unit tests for A2A -> AgentCore Memory persistence WIRING (RouteIQ-60e7).

The bedrock-agentcore-mcp plugin exposes ``memory_put_event`` / ``memory_list_events``
/ ``identity_get_oauth2_token``, but they had no caller in the agent invocation
flow. These tests prove the LIVE wiring on the A2A invocation path:

* gated default OFF -> the invocation path issues NO AgentCore call (byte-stable);
* gated ON + the plugin present + Memory enabled -> a successful A2A invocation
  persists the turn via the plugin's ``memory_put_event`` (CreateEvent);
* fail-open -> a memory error never fails the agent invocation.

All boto3 / HTTP is mocked -- no live AWS, no live agent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter import a2a_gateway as a2a_mod
from litellm_llmrouter.a2a_gateway import (
    A2AAgent,
    A2AGateway,
    JSONRPCRequest,
    persist_a2a_event_to_agentcore,
)


class _FakePlugin:
    """Stand-in for the bedrock-agentcore-mcp plugin."""

    def __init__(self, *, memory_enabled=True, raises=False):
        self._memory_enabled = memory_enabled
        self._raises = raises
        self.metadata = MagicMock(name="bedrock-agentcore-mcp")
        self.metadata.name = "bedrock-agentcore-mcp"
        self.put_calls = []

    def is_memory_enabled(self):
        return self._memory_enabled

    async def memory_put_event(self, *, actor_id, session_id, payload):
        if self._raises:
            raise RuntimeError("memory boom")
        self.put_calls.append((actor_id, session_id, payload))
        return {"event": {"eventId": "evt-1"}}


def _patch_plugin(plugin):
    return patch.object(a2a_mod, "_get_agentcore_plugin", lambda: plugin)


# ---------------------------------------------------------------------------
# persist_a2a_event_to_agentcore (the helper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_helper_noop_when_disabled(monkeypatch):
    """Default OFF -> no plugin call, returns False (byte-stable)."""
    monkeypatch.setenv("A2A_AGENTCORE_MEMORY_ENABLED", "false")
    plugin = _FakePlugin()
    with _patch_plugin(plugin):
        ok = await persist_a2a_event_to_agentcore("a1", "s1", {"x": 1})
    assert ok is False
    assert plugin.put_calls == []


@pytest.mark.asyncio
async def test_helper_persists_when_enabled(monkeypatch):
    """ON + plugin present + Memory enabled -> CreateEvent called."""
    monkeypatch.setenv("A2A_AGENTCORE_MEMORY_ENABLED", "true")
    plugin = _FakePlugin(memory_enabled=True)
    with _patch_plugin(plugin):
        ok = await persist_a2a_event_to_agentcore("a1", "sess-9", {"turn": 1})
    assert ok is True
    assert plugin.put_calls == [("a1", "sess-9", {"turn": 1})]


@pytest.mark.asyncio
async def test_helper_noop_when_plugin_memory_disabled(monkeypatch):
    """ON but the plugin's Memory surface is disabled -> no call."""
    monkeypatch.setenv("A2A_AGENTCORE_MEMORY_ENABLED", "true")
    plugin = _FakePlugin(memory_enabled=False)
    with _patch_plugin(plugin):
        ok = await persist_a2a_event_to_agentcore("a1", "s1", {})
    assert ok is False
    assert plugin.put_calls == []


@pytest.mark.asyncio
async def test_helper_fail_open_on_error(monkeypatch):
    """A memory error is swallowed (returns False), never raises."""
    monkeypatch.setenv("A2A_AGENTCORE_MEMORY_ENABLED", "true")
    plugin = _FakePlugin(raises=True)
    with _patch_plugin(plugin):
        ok = await persist_a2a_event_to_agentcore("a1", "s1", {})
    assert ok is False


@pytest.mark.asyncio
async def test_helper_noop_when_plugin_absent(monkeypatch):
    """ON but no plugin loaded -> no-op."""
    monkeypatch.setenv("A2A_AGENTCORE_MEMORY_ENABLED", "true")
    with _patch_plugin(None):
        ok = await persist_a2a_event_to_agentcore("a1", "s1", {})
    assert ok is False


# ---------------------------------------------------------------------------
# Live callsite: invoke_agent persists on success when enabled
# ---------------------------------------------------------------------------


def _mock_http_ok():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={"jsonrpc": "2.0", "id": 1, "result": {"text": "hi"}}
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


@pytest.mark.asyncio
async def test_invoke_persists_to_agentcore_when_enabled(monkeypatch):
    """A successful A2A invocation persists the turn when memory is enabled."""
    monkeypatch.setenv("A2A_AGENTCORE_MEMORY_ENABLED", "true")
    gw = A2AGateway()
    gw.enabled = True
    gw.register_agent(
        A2AAgent(agent_id="a1", name="A1", description="", url="https://x/a")
    )
    plugin = _FakePlugin(memory_enabled=True)

    with (
        patch.object(a2a_mod, "get_client_for_request", return_value=_mock_http_ok()),
        patch.object(a2a_mod, "validate_outbound_url_async", new_callable=AsyncMock),
        _patch_plugin(plugin),
    ):
        req = JSONRPCRequest(
            method="message/send", params={"message": {"text": "hi"}}, id=1
        )
        resp = await gw.invoke_agent("a1", req)

    assert resp.error is None
    assert len(plugin.put_calls) == 1
    actor_id, _session, payload = plugin.put_calls[0]
    assert actor_id == "a1"
    assert payload["result"] == {"text": "hi", "_task": payload["result"]["_task"]}


@pytest.mark.asyncio
async def test_invoke_does_not_persist_when_disabled(monkeypatch):
    """Default OFF -> a successful invocation issues NO AgentCore call."""
    monkeypatch.setenv("A2A_AGENTCORE_MEMORY_ENABLED", "false")
    gw = A2AGateway()
    gw.enabled = True
    gw.register_agent(
        A2AAgent(agent_id="a1", name="A1", description="", url="https://x/a")
    )
    plugin = _FakePlugin(memory_enabled=True)

    with (
        patch.object(a2a_mod, "get_client_for_request", return_value=_mock_http_ok()),
        patch.object(a2a_mod, "validate_outbound_url_async", new_callable=AsyncMock),
        _patch_plugin(plugin),
    ):
        req = JSONRPCRequest(
            method="message/send", params={"message": {"text": "hi"}}, id=1
        )
        resp = await gw.invoke_agent("a1", req)

    assert resp.error is None
    assert plugin.put_calls == []
