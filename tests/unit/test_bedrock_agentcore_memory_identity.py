"""Unit tests for AgentCore Memory + Identity integration (RouteIQ-60e7).

Building on the retargeted AgentCore connector (RouteIQ-e5a4), this adds:
- AgentCore Memory: managed agent/conversation state via the data-plane
  CreateEvent (put) / ListEvents (get) APIs.
- AgentCore Identity: agent OAuth2 token exchange via GetResourceOauth2Token.

Both are settings-gated default-OFF and ride the SAME SigV4-signing
``bedrock-agentcore`` data-plane client. All boto3 is MOCKED via an injected
``client_factory`` -- cred-free, no live AWS.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.gateway.plugins.bedrock_agentcore_mcp import (
    BedrockAgentCoreMCPPlugin,
)


def _plugin(
    factory, *, memory=False, identity=False, memory_id="mem-1", region="us-east-1"
):
    plugin = BedrockAgentCoreMCPPlugin(client_factory=factory)
    plugin._region = region
    plugin._memory_enabled = memory
    plugin._memory_id = memory_id
    plugin._identity_enabled = identity
    return plugin


class TestGatingDefaults:
    def test_memory_and_identity_default_off(self, monkeypatch):
        monkeypatch.delenv(
            "ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_MEMORY_ENABLED", raising=False
        )
        monkeypatch.delenv(
            "ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_IDENTITY_ENABLED", raising=False
        )
        plugin = BedrockAgentCoreMCPPlugin(client_factory=MagicMock())
        assert plugin.is_memory_enabled() is False
        assert plugin.is_identity_enabled() is False

    def test_env_enables(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_MEMORY_ENABLED", "true")
        monkeypatch.setenv("ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_IDENTITY_ENABLED", "true")
        monkeypatch.setenv("ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_MEMORY_ID", "mem-xyz")
        plugin = BedrockAgentCoreMCPPlugin(client_factory=MagicMock())
        assert plugin.is_memory_enabled() is True
        assert plugin.is_identity_enabled() is True
        assert plugin._memory_id == "mem-xyz"


class TestMemory:
    @pytest.mark.asyncio
    async def test_put_event_calls_create_event(self):
        client = MagicMock()
        client.create_event.return_value = {"event": {"eventId": "e1"}}
        captured = {}

        def factory(region, endpoint_url=None):
            captured["region"] = region
            captured["endpoint_url"] = endpoint_url
            return client

        plugin = _plugin(factory, memory=True)
        resp = await plugin.memory_put_event(
            "actor-1", "sess-1", {"role": "user", "text": "hi"}
        )

        assert captured["endpoint_url"] == (
            "https://bedrock-agentcore.us-east-1.amazonaws.com"
        )
        client.create_event.assert_called_once()
        kwargs = client.create_event.call_args.kwargs
        assert kwargs["memoryId"] == "mem-1"
        assert kwargs["actorId"] == "actor-1"
        assert kwargs["sessionId"] == "sess-1"
        # Single dict payload is normalised to the list shape the API expects.
        assert kwargs["payload"] == [{"role": "user", "text": "hi"}]
        assert resp == {"event": {"eventId": "e1"}}

    @pytest.mark.asyncio
    async def test_list_events_returns_events(self):
        client = MagicMock()
        client.list_events.return_value = {
            "events": [{"eventId": "e1"}, {"eventId": "e2"}]
        }
        plugin = _plugin(lambda r, endpoint_url=None: client, memory=True)
        events = await plugin.memory_list_events("actor-1", "sess-1", max_results=10)
        assert [e["eventId"] for e in events] == ["e1", "e2"]
        kwargs = client.list_events.call_args.kwargs
        assert kwargs["memoryId"] == "mem-1"
        assert kwargs["includePayloads"] is True
        assert kwargs["maxResults"] == 10

    @pytest.mark.asyncio
    async def test_put_raises_when_disabled(self):
        plugin = _plugin(MagicMock(), memory=False)
        with pytest.raises(RuntimeError):
            await plugin.memory_put_event("a", "s", {})

    @pytest.mark.asyncio
    async def test_put_raises_without_memory_id(self):
        plugin = _plugin(MagicMock(), memory=True, memory_id="")
        with pytest.raises(ValueError):
            await plugin.memory_put_event("a", "s", {})

    @pytest.mark.asyncio
    async def test_explicit_memory_id_overrides(self):
        client = MagicMock()
        client.create_event.return_value = {}
        plugin = _plugin(lambda r, endpoint_url=None: client, memory=True)
        await plugin.memory_put_event("a", "s", {}, memory_id="override-mem")
        assert client.create_event.call_args.kwargs["memoryId"] == "override-mem"


class TestIdentity:
    @pytest.mark.asyncio
    async def test_get_oauth2_token_calls_api(self):
        client = MagicMock()
        client.get_resource_oauth2_token.return_value = {
            "accessToken": "tok-123",
            "sessionStatus": "COMPLETED",
        }
        plugin = _plugin(lambda r, endpoint_url=None: client, identity=True)
        resp = await plugin.identity_get_oauth2_token(
            "wlit-token",
            "my-provider",
            ["scope-a", "scope-b"],
            oauth2_flow="M2M",
        )
        assert resp["accessToken"] == "tok-123"
        kwargs = client.get_resource_oauth2_token.call_args.kwargs
        assert kwargs["workloadIdentityToken"] == "wlit-token"
        assert kwargs["resourceCredentialProviderName"] == "my-provider"
        assert kwargs["scopes"] == ["scope-a", "scope-b"]
        assert kwargs["oauth2Flow"] == "M2M"

    @pytest.mark.asyncio
    async def test_raises_when_disabled(self):
        plugin = _plugin(MagicMock(), identity=False)
        with pytest.raises(RuntimeError):
            await plugin.identity_get_oauth2_token("t", "p", ["s"])


class TestNoLiveBoto3:
    def test_no_api_called_from_init(self):
        """Constructing the plugin (even enabled) calls NO AgentCore API."""
        factory = MagicMock()
        BedrockAgentCoreMCPPlugin(client_factory=factory)
        # Factory is only invoked at memory/identity call time, never in __init__.
        factory.assert_not_called()
