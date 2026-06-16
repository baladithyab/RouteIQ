"""
Unit tests for the real MCP JSON-RPC client + gateway invocation (RouteIQ-6ccd).

These verify that RouteIQ speaks REAL MCP JSON-RPC 2.0 over HTTP (with SSE
support) rather than the legacy custom REST stub:
- tools/call goes out as a JSON-RPC 2.0 envelope (method="tools/call",
  params={"name", "arguments"})
- the JSON-RPC result is parsed (content blocks, isError)
- SSE (text/event-stream) responses are parsed
- transport/protocol errors map to MCPToolResult(success=False)
- SSRF reject at invocation time
- feature-flag gating (tool_invocation_enabled)

All HTTP is mocked — no live MCP server.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm_llmrouter import mcp_jsonrpc_client
from litellm_llmrouter.mcp_jsonrpc_client import (
    MCPJSONRPCClient,
    MCPProtocolError,
    _parse_sse_payload,
    _normalize_endpoint,
)
from litellm_llmrouter.mcp_gateway import (
    MCPGateway,
    MCPServer,
    MCPTransport,
)


def _make_cm_client(response: MagicMock) -> AsyncMock:
    """Build an async-context-manager mock httpx client returning ``response``."""
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _json_response(payload: dict, status: int = 200, headers=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {"content-type": "application/json"}
    resp.text = json.dumps(payload)
    resp.json = MagicMock(return_value=payload)
    return resp


class TestEndpointNormalization:
    def test_strips_trailing_slash_no_suffix_appended(self):
        # The legacy bug appended /mcp/tools/call. Real MCP uses a single endpoint.
        assert _normalize_endpoint("https://mcp.example.com/mcp/") == (
            "https://mcp.example.com/mcp"
        )
        assert _normalize_endpoint("https://mcp.example.com/rpc") == (
            "https://mcp.example.com/rpc"
        )


class TestSSEParsing:
    def test_parses_single_data_frame(self):
        body = 'data: {"jsonrpc":"2.0","id":1,"result":{"content":[]}}\n\n'
        msg = _parse_sse_payload(body)
        assert msg["result"] == {"content": []}

    def test_picks_last_response_among_notifications(self):
        body = (
            'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
            'data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"ok"}]}}\n\n'
        )
        msg = _parse_sse_payload(body)
        assert msg["result"]["content"][0]["text"] == "ok"

    def test_raises_when_no_response_in_stream(self):
        with pytest.raises(MCPProtocolError):
            _parse_sse_payload('data: {"jsonrpc":"2.0","method":"ping"}\n\n')


class TestMCPJSONRPCClientCallTool:
    @pytest.mark.asyncio
    async def test_tools_call_sends_jsonrpc_envelope(self):
        """A tools/call must go out as a JSON-RPC 2.0 request, NOT the REST stub."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "42"}]},
        }
        response = _json_response(payload)
        client_cm = _make_cm_client(response)

        with patch.object(
            mcp_jsonrpc_client, "get_client_for_request", return_value=client_cm
        ):
            client = MCPJSONRPCClient("https://mcp.example.com/rpc")
            result = await client.call_tool("add", {"a": 1, "b": 2})

        # Assert the OUTBOUND request was a JSON-RPC tools/call envelope.
        assert client_cm.post.await_count == 1
        _args, kwargs = client_cm.post.call_args
        sent = kwargs["json"]
        assert sent["jsonrpc"] == "2.0"
        assert sent["method"] == "tools/call"
        assert sent["params"] == {"name": "add", "arguments": {"a": 1, "b": 2}}
        assert "id" in sent
        # No legacy {tool_name, arguments} REST stub body.
        assert "tool_name" not in sent

        # Accept header negotiates both JSON and SSE.
        assert "text/event-stream" in kwargs["headers"]["Accept"]

        # Result parsed from JSON-RPC result.content.
        assert result.is_error is False
        assert result.content == [{"type": "text", "text": "42"}]

    @pytest.mark.asyncio
    async def test_initialize_sends_handshake(self):
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": {"name": "test-server"},
            },
        }
        client_cm = _make_cm_client(_json_response(payload))
        with patch.object(
            mcp_jsonrpc_client, "get_client_for_request", return_value=client_cm
        ):
            client = MCPJSONRPCClient("https://mcp.example.com/rpc")
            result = await client.initialize()

        _args, kwargs = client_cm.post.call_args
        assert kwargs["json"]["method"] == "initialize"
        assert "protocolVersion" in kwargs["json"]["params"]
        assert result["serverInfo"]["name"] == "test-server"

    @pytest.mark.asyncio
    async def test_tools_call_parses_sse_response(self):
        sse_body = (
            'data: {"jsonrpc":"2.0","id":1,'
            '"result":{"content":[{"type":"text","text":"sse-ok"}]}}\n\n'
        )
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/event-stream"}
        resp.text = sse_body
        client_cm = _make_cm_client(resp)

        with patch.object(
            mcp_jsonrpc_client, "get_client_for_request", return_value=client_cm
        ):
            client = MCPJSONRPCClient("https://mcp.example.com/rpc")
            result = await client.call_tool("echo", {})

        assert result.content == [{"type": "text", "text": "sse-ok"}]

    @pytest.mark.asyncio
    async def test_jsonrpc_error_raises_protocol_error(self):
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }
        client_cm = _make_cm_client(_json_response(payload))
        with patch.object(
            mcp_jsonrpc_client, "get_client_for_request", return_value=client_cm
        ):
            client = MCPJSONRPCClient("https://mcp.example.com/rpc")
            with pytest.raises(MCPProtocolError) as exc:
                await client.call_tool("nope", {})
        assert exc.value.code == -32601

    @pytest.mark.asyncio
    async def test_http_error_status_raises(self):
        resp = MagicMock()
        resp.status_code = 502
        resp.headers = {"content-type": "application/json"}
        resp.text = "bad gateway"
        client_cm = _make_cm_client(resp)
        with patch.object(
            mcp_jsonrpc_client, "get_client_for_request", return_value=client_cm
        ):
            client = MCPJSONRPCClient("https://mcp.example.com/rpc")
            with pytest.raises(MCPProtocolError):
                await client.call_tool("x", {})

    @pytest.mark.asyncio
    async def test_session_id_echoed_on_subsequent_calls(self):
        init_resp = _json_response(
            {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}},
            headers={
                "content-type": "application/json",
                "mcp-session-id": "sess-123",
            },
        )
        call_resp = _json_response(
            {"jsonrpc": "2.0", "id": 2, "result": {"content": []}}
        )
        client_cm = AsyncMock()
        client_cm.post = AsyncMock(side_effect=[init_resp, call_resp])
        client_cm.__aenter__ = AsyncMock(return_value=client_cm)
        client_cm.__aexit__ = AsyncMock(return_value=None)

        with patch.object(
            mcp_jsonrpc_client, "get_client_for_request", return_value=client_cm
        ):
            client = MCPJSONRPCClient("https://mcp.example.com/rpc")
            await client.initialize()
            await client.call_tool("x", {})

        second_headers = client_cm.post.call_args_list[1].kwargs["headers"]
        assert second_headers.get("Mcp-Session-Id") == "sess-123"


class TestGatewayInvocationViaJSONRPC:
    def _make_gateway(self) -> MCPGateway:
        gw = MCPGateway()
        gw.enabled = True
        gw._tool_invocation_enabled = True
        gw._ha_sync_enabled = False
        gw._db_persistence_enabled = False
        return gw

    @pytest.mark.asyncio
    async def test_invoke_tool_routes_through_jsonrpc_client(self):
        gw = self._make_gateway()
        gw.register_server(
            MCPServer(
                server_id="s1",
                name="srv",
                url="https://mcp.example.com/rpc",
                transport=MCPTransport.STREAMABLE_HTTP,
                tools=["add"],
            )
        )

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "done"}]},
        }
        client_cm = _make_cm_client(_json_response(payload))

        with (
            patch.object(
                mcp_jsonrpc_client, "get_client_for_request", return_value=client_cm
            ),
            patch(
                "litellm_llmrouter.mcp_gateway.validate_outbound_url_async",
                new_callable=AsyncMock,
            ),
        ):
            result = await gw.invoke_tool("add", {"a": 1})

        assert result.success is True
        assert result.result == [{"type": "text", "text": "done"}]
        # The outbound call was JSON-RPC tools/call.
        sent = client_cm.post.call_args.kwargs["json"]
        assert sent["method"] == "tools/call"
        assert sent["params"]["name"] == "add"

    @pytest.mark.asyncio
    async def test_tool_level_error_maps_to_failure(self):
        gw = self._make_gateway()
        gw.register_server(
            MCPServer(
                server_id="s1",
                name="srv",
                url="https://mcp.example.com/rpc",
                tools=["boom"],
            )
        )
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "tool failed"}],
            },
        }
        client_cm = _make_cm_client(_json_response(payload))
        with (
            patch.object(
                mcp_jsonrpc_client, "get_client_for_request", return_value=client_cm
            ),
            patch(
                "litellm_llmrouter.mcp_gateway.validate_outbound_url_async",
                new_callable=AsyncMock,
            ),
        ):
            result = await gw.invoke_tool("boom", {})
        assert result.success is False
        assert "tool failed" in result.error

    @pytest.mark.asyncio
    async def test_invoke_disabled_by_feature_flag(self):
        gw = self._make_gateway()
        gw._tool_invocation_enabled = False
        gw.register_server(
            MCPServer(server_id="s1", name="srv", url="https://x/rpc", tools=["t"])
        )
        result = await gw.invoke_tool("t", {})
        assert result.success is False
        assert "tool_invocation_disabled" in result.error

    @pytest.mark.asyncio
    async def test_invoke_ssrf_reject_at_invocation(self):
        from litellm_llmrouter.url_security import SSRFBlockedError

        gw = self._make_gateway()
        gw.register_server(
            MCPServer(
                server_id="s1",
                name="srv",
                url="https://mcp.example.com/rpc",
                tools=["t"],
            )
        )

        async def _raise(*a, **k):
            raise SSRFBlockedError("https://mcp.example.com/rpc", "blocked")

        with patch(
            "litellm_llmrouter.mcp_gateway.validate_outbound_url_async",
            new=_raise,
        ):
            result = await gw.invoke_tool("t", {})
        assert result.success is False
        assert "blocked for security reasons" in result.error

    @pytest.mark.asyncio
    async def test_transport_timeout_maps_to_failure(self):
        gw = self._make_gateway()
        gw.register_server(
            MCPServer(
                server_id="s1",
                name="srv",
                url="https://mcp.example.com/rpc",
                tools=["t"],
            )
        )
        client_cm = AsyncMock()
        client_cm.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        client_cm.__aenter__ = AsyncMock(return_value=client_cm)
        client_cm.__aexit__ = AsyncMock(return_value=None)
        with (
            patch.object(
                mcp_jsonrpc_client, "get_client_for_request", return_value=client_cm
            ),
            patch(
                "litellm_llmrouter.mcp_gateway.validate_outbound_url_async",
                new_callable=AsyncMock,
            ),
        ):
            result = await gw.invoke_tool("t", {})
        assert result.success is False
        assert "Timeout" in result.error
