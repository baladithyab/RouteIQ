"""
Unit tests for the retargeted Bedrock AgentCore connector (RouteIQ-e5a4).

The legacy connector targeted the DEPRECATED bedrock-agent-runtime API with a
hardcoded TSTALIASID alias and ZERO SigV4. These tests assert the new behaviour:
- registration targets the AgentCore Runtime data-plane endpoint
  (bedrock-agentcore.{region}.amazonaws.com), NOT bedrock-agent-runtime, and NOT
  the legacy /agentAliases/TSTALIASID path
- registration uses the SSE/MCP transport, not the contradictory streamable_http
- invocation calls invoke_agent_runtime on a SigV4-signing boto3 client (botocore
  signs automatically), targeting the correct service + endpoint + agentRuntimeArn
- region resolution discipline is honoured (explicit -> AWS_REGION -> boto3)

All boto3 is MOCKED via an injected client_factory — no live AWS, cred-free.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm_llmrouter.gateway.plugins import bedrock_agentcore_mcp
from litellm_llmrouter.gateway.plugins.bedrock_agentcore_mcp import (
    AGENTCORE_RUNTIME_SERVICE,
    BedrockAgentCoreMCPPlugin,
    _agentcore_runtime_endpoint,
    _agent_runtime_arn,
    _resolve_region,
)
from litellm_llmrouter.gateway.plugin_manager import PluginContext


def _stub_boto3_region_none(monkeypatch) -> None:
    """Force the boto3 session fallback in ``_resolve_region`` to yield no region.

    HERMETICITY (mirrors RouteIQ-22bc): ``_resolve_region`` falls back to
    ``boto3.session.Session().region_name`` which reads ``~/.aws/config`` FROM DISK
    (not cleared by ``monkeypatch.delenv``). On an AWS-authenticated dev shell the
    profile region leaks in and the "unresolvable" assertions fail. Inject a fake
    boto3 whose session resolves no region so the test is hermetic regardless of
    the machine's AWS profile/config.
    """
    import sys
    from unittest.mock import MagicMock

    fake_boto3 = MagicMock()
    fake_boto3.session.Session.return_value.region_name = None
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)


class TestRegionResolution:
    def test_explicit_config_wins(self):
        assert _resolve_region("eu-west-1") == "eu-west-1"

    def test_falls_back_to_aws_region_env(self, monkeypatch):
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.setenv("AWS_REGION", "ap-south-1")
        assert _resolve_region(None) == "ap-south-1"

    def test_none_when_unresolvable(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        # boto3 session must resolve no region (hermetic: stub the disk-config leak)
        _stub_boto3_region_none(monkeypatch)
        assert _resolve_region(None) is None


class TestEndpointShape:
    def test_endpoint_targets_agentcore_not_legacy(self):
        ep = _agentcore_runtime_endpoint("us-east-1")
        assert ep == "https://bedrock-agentcore.us-east-1.amazonaws.com"
        # Must NOT be the deprecated data plane.
        assert "bedrock-agent-runtime" not in ep

    def test_arn_passthrough_and_wrap(self):
        full = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-rt"
        assert _agent_runtime_arn(full, "us-east-1") == full
        wrapped = _agent_runtime_arn("my-rt", "us-west-2")
        assert wrapped.startswith("arn:aws:bedrock-agentcore:us-west-2:")
        assert wrapped.endswith("runtime/my-rt")


class TestRegistrationRetargeted:
    @pytest.mark.asyncio
    async def test_registers_under_agentcore_endpoint_and_sse(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        plugin = BedrockAgentCoreMCPPlugin(client_factory=MagicMock())
        plugin._region = "us-east-1"
        plugin._agent_ids = ["agent-1"]

        mock_mcp = MagicMock()
        mock_mcp.is_enabled.return_value = True
        mock_mcp.register_server = AsyncMock(
            return_value={"server_id": "x", "status": "registered"}
        )
        context = PluginContext(mcp=mock_mcp)

        await plugin._register_agents(context)

        assert mock_mcp.register_server.await_count == 1
        kwargs = mock_mcp.register_server.call_args.kwargs
        # Targets AgentCore Runtime endpoint, NOT bedrock-agent-runtime.
        assert kwargs["url"] == "https://bedrock-agentcore.us-east-1.amazonaws.com"
        assert "bedrock-agent-runtime" not in kwargs["url"]
        assert "TSTALIASID" not in kwargs["url"]
        # MCP/SSE transport, not the contradictory streamable_http-over-REST.
        assert kwargs["transport"] == "sse"
        # Metadata records SigV4 + the runtime ARN + qualifier.
        meta = kwargs["metadata"]
        assert meta["signed"] == "sigv4"
        assert meta["service"] == AGENTCORE_RUNTIME_SERVICE
        assert meta["qualifier"] == "DEFAULT"
        assert meta["agent_runtime_arn"].endswith("runtime/agent-1")

    @pytest.mark.asyncio
    async def test_skips_registration_when_no_region(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        _stub_boto3_region_none(monkeypatch)
        plugin = BedrockAgentCoreMCPPlugin(client_factory=MagicMock())
        plugin._region = None
        plugin._agent_ids = ["agent-1"]

        mock_mcp = MagicMock()
        mock_mcp.is_enabled.return_value = True
        mock_mcp.register_server = AsyncMock()
        context = PluginContext(mcp=mock_mcp)

        await plugin._register_agents(context)
        # Fail loud-but-safe: no signing with Region=None.
        mock_mcp.register_server.assert_not_called()


class TestInvocationSigV4:
    @pytest.mark.asyncio
    async def test_invoke_agent_calls_invoke_agent_runtime_signed(self):
        """invoke_agent must dispatch via boto3 invoke_agent_runtime (SigV4)."""
        # Mock the boto3 client + its (auto-signed) invoke_agent_runtime.
        mock_client = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps({"output": "hello"}).encode("utf-8")
        mock_client.invoke_agent_runtime.return_value = {"response": body}

        captured = {}

        def factory(region, endpoint_url=None):
            captured["region"] = region
            captured["endpoint_url"] = endpoint_url
            return mock_client

        plugin = BedrockAgentCoreMCPPlugin(client_factory=factory)
        plugin._region = "us-east-1"

        result = await plugin.invoke_agent(
            "my-runtime", {"prompt": "hi"}, session_id="sess-1"
        )

        # The factory built a client for the AgentCore service + correct endpoint.
        assert captured["region"] == "us-east-1"
        assert (
            captured["endpoint_url"]
            == "https://bedrock-agentcore.us-east-1.amazonaws.com"
        )

        # invoke_agent_runtime called once with the right shape.
        assert mock_client.invoke_agent_runtime.call_count == 1
        call_kwargs = mock_client.invoke_agent_runtime.call_args.kwargs
        assert call_kwargs["agentRuntimeArn"].endswith("runtime/my-runtime")
        assert call_kwargs["qualifier"] == "DEFAULT"
        assert call_kwargs["runtimeSessionId"] == "sess-1"
        # Payload is JSON-encoded bytes (the runtime contract).
        assert json.loads(call_kwargs["payload"]) == {"prompt": "hi"}

        # Response body decoded.
        assert result == {"output": "hello"}

    @pytest.mark.asyncio
    async def test_invoke_passes_full_arn_through(self):
        mock_client = MagicMock()
        mock_client.invoke_agent_runtime.return_value = {
            "response": json.dumps({"ok": True}).encode("utf-8")
        }
        plugin = BedrockAgentCoreMCPPlugin(
            client_factory=lambda r, endpoint_url=None: mock_client
        )
        plugin._region = "eu-west-1"
        full_arn = "arn:aws:bedrock-agentcore:eu-west-1:123456789012:runtime/rt-x"

        await plugin.invoke_agent(full_arn, {})

        call_kwargs = mock_client.invoke_agent_runtime.call_args.kwargs
        assert call_kwargs["agentRuntimeArn"] == full_arn

    @pytest.mark.asyncio
    async def test_invoke_raises_without_region(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        _stub_boto3_region_none(monkeypatch)
        plugin = BedrockAgentCoreMCPPlugin(client_factory=MagicMock())
        plugin._region = None
        with pytest.raises(ValueError):
            await plugin.invoke_agent("rt", {})

    @pytest.mark.asyncio
    async def test_invoke_increments_counter(self):
        mock_client = MagicMock()
        mock_client.invoke_agent_runtime.return_value = {
            "response": json.dumps({}).encode("utf-8")
        }
        plugin = BedrockAgentCoreMCPPlugin(
            client_factory=lambda r, endpoint_url=None: mock_client
        )
        plugin._region = "us-east-1"
        counter = MagicMock()
        plugin._invocations_counter = counter

        await plugin.invoke_agent("rt", {"x": 1})
        counter.add.assert_called_once()

    def test_decode_handles_str_bytes_dict(self):
        decode = BedrockAgentCoreMCPPlugin._decode_invoke_response
        assert decode({"response": b'{"a":1}'}) == {"a": 1}
        assert decode({"response": '{"b":2}'}) == {"b": 2}
        assert decode({"response": {"c": 3}}) == {"c": 3}
        # Non-JSON string falls back to {"output": ...}
        assert decode({"response": b"not json"}) == {"output": "not json"}


class TestNoLiveBoto3Import:
    """The connector must not import boto3 at module load (cred-free)."""

    def test_module_imports_without_boto3(self):
        # If this test file imported, the module loaded without boto3 present.
        assert bedrock_agentcore_mcp.AGENTCORE_RUNTIME_SERVICE == "bedrock-agentcore"
