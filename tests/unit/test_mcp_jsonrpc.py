"""
Unit tests for MCP Native JSON-RPC Transport.

Tests the /mcp endpoint with JSON-RPC 2.0 protocol.
"""

import json
import os
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Set environment variables before importing modules
os.environ.setdefault("MCP_GATEWAY_ENABLED", "true")


@pytest.fixture
def app():
    """Create a FastAPI app with MCP JSON-RPC router."""
    from litellm_llmrouter.mcp_jsonrpc import mcp_jsonrpc_router

    app = FastAPI()

    # Mock authentication
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    async def mock_auth():
        return None

    app.dependency_overrides[user_api_key_auth] = mock_auth
    app.include_router(mcp_jsonrpc_router, prefix="")

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_gateway():
    """Create a mock MCP gateway."""
    gateway = MagicMock()
    gateway.is_enabled.return_value = True
    gateway.is_tool_invocation_enabled.return_value = True
    gateway.list_servers.return_value = []
    gateway.list_resources.return_value = []
    gateway.get_server.return_value = None
    gateway.get_tool.return_value = None
    return gateway


class TestMCPJSONRPCInfo:
    """Test GET /mcp endpoint (server info)."""

    def test_get_mcp_returns_server_info(self, client, mock_gateway):
        """GET /mcp returns server info."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.get("/mcp")
            assert response.status_code == 200

            data = response.json()
            assert "name" in data
            assert "version" in data
            assert "protocolVersion" in data
            assert data["transport"] == "streamable-http"
            assert "capabilities" in data

    def test_get_mcp_shows_disabled_status(self, client, mock_gateway):
        """GET /mcp shows disabled status when gateway is disabled."""
        mock_gateway.is_enabled.return_value = False

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.get("/mcp")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "disabled"


class TestMCPJSONRPCInitialize:
    """Test initialize method."""

    def test_initialize_returns_valid_response(self, client, mock_gateway):
        """initialize returns protocol version and capabilities."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0.0"},
                    },
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 1
            assert "result" in data
            assert "error" not in data

            result = data["result"]
            assert "protocolVersion" in result
            assert "capabilities" in result
            assert "serverInfo" in result
            assert result["capabilities"]["tools"]["listChanged"] is True

    def test_initialize_fails_when_gateway_disabled(self, client, mock_gateway):
        """initialize returns error when gateway is disabled."""
        mock_gateway.is_enabled.return_value = False

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32003  # MCP_GATEWAY_DISABLED


class TestMCPJSONRPCToolsList:
    """Test tools/list method."""

    def test_tools_list_returns_empty_list(self, client, mock_gateway):
        """tools/list returns empty list when no servers registered."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 1
            assert "result" in data

            result = data["result"]
            assert "tools" in result
            assert result["tools"] == []

    def test_tools_list_returns_namespaced_tools(self, client, mock_gateway):
        """tools/list returns tools namespaced with server_id."""
        # Create mock server with tools
        mock_server = MagicMock()
        mock_server.server_id = "test-server"
        mock_server.name = "Test Server"
        mock_server.tools = ["echo", "sum"]
        mock_server.tool_definitions = {}

        mock_gateway.list_servers.return_value = [mock_server]

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                },
            )
            assert response.status_code == 200

            data = response.json()
            result = data["result"]
            tools = result["tools"]

            assert len(tools) == 2
            tool_names = [t["name"] for t in tools]
            assert "test-server.echo" in tool_names
            assert "test-server.sum" in tool_names

            # Check tool structure
            for tool in tools:
                assert "name" in tool
                assert "description" in tool
                assert "inputSchema" in tool


class TestMCPJSONRPCToolsCall:
    """Test tools/call method."""

    def test_tools_call_requires_name(self, client, mock_gateway):
        """tools/call returns error when name is missing."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"arguments": {}},
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32602  # Invalid params

    def test_tools_call_returns_not_found(self, client, mock_gateway):
        """tools/call returns error when tool not found."""
        mock_gateway.get_tool.return_value = None
        mock_gateway.get_server.return_value = None

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "nonexistent.tool", "arguments": {}},
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32001  # MCP_TOOL_NOT_FOUND

    def test_tools_call_disabled_returns_error(self, client, mock_gateway):
        """tools/call returns error when tool invocation is disabled."""
        mock_gateway.is_tool_invocation_enabled.return_value = False

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "test.tool", "arguments": {}},
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32002  # MCP_TOOL_INVOCATION_DISABLED


class TestMCPJSONRPCResourcesList:
    """Test resources/list method."""

    def test_resources_list_returns_empty_list(self, client, mock_gateway):
        """resources/list returns empty list when no resources."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "resources/list",
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert "result" in data

            result = data["result"]
            assert "resources" in result
            assert result["resources"] == []


class TestMCPJSONRPCErrorHandling:
    """Test JSON-RPC error handling."""

    def test_invalid_json_returns_parse_error(self, client):
        """Invalid JSON returns parse error."""
        response = client.post(
            "/mcp",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32700  # Parse error

    def test_wrong_jsonrpc_version_returns_error(self, client, mock_gateway):
        """Wrong JSON-RPC version returns invalid request error."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "1.0",  # Wrong version
                    "id": 1,
                    "method": "initialize",
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32600  # Invalid request

    def test_missing_method_returns_error(self, client, mock_gateway):
        """Missing method returns invalid request error."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32600  # Invalid request

    def test_unknown_method_returns_not_found(self, client, mock_gateway):
        """Unknown method returns method not found error."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "nonexistent/method",
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32601  # Method not found

    def test_empty_body_returns_parse_error(self, client):
        """Empty request body returns parse error."""
        response = client.post(
            "/mcp",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32700  # Parse error


class TestMCPJSONRPCNotifications:
    """Test JSON-RPC notifications (requests without id)."""

    def test_notification_has_null_id_in_response(self, client, mock_gateway):
        """Notification (no id) returns response with null id."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    # No id - this is a notification
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert data.get("id") is None
