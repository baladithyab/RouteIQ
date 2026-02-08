"""
MCP Protocol Compliance Tests (2025-11-25 Spec)
================================================

Tests verifying RouteIQ's compliance with MCP specification 2025-11-25.
Covers: version negotiation, initialized notification, error codes,
pagination, tool annotations, resources/read proxy, resources/list pagination,
Mcp-Session-Id header, notifications/tools/list_changed, session management,
and find_server_for_resource.
"""

import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch

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
    gateway.find_server_for_resource.return_value = None
    gateway.proxy_resource_read = AsyncMock(
        return_value={
            "error": {
                "code": -32002,
                "message": "No server found for resource URI: unknown",
            }
        }
    )
    return gateway


@pytest.fixture(autouse=True)
def _reset_sessions():
    """Reset MCP session state between tests."""
    from litellm_llmrouter.mcp_jsonrpc import reset_sessions

    reset_sessions()
    yield
    reset_sessions()


def _jsonrpc(method, params=None, request_id=1):
    """Helper to build a JSON-RPC 2.0 request dict."""
    req = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        req["params"] = params
    return req


# ============================================================================
# G1: Protocol Version Negotiation
# ============================================================================


class TestProtocolVersionNegotiation:
    """Test MCP 2025-03-26 protocol version negotiation."""

    def test_initialize_with_2025_03_26(self, client, mock_gateway):
        """Server accepts 2025-03-26 and responds with same version."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "initialize",
                    {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                ),
            )
            assert response.status_code == 200
            data = response.json()
            assert "result" in data
            assert data["result"]["protocolVersion"] == "2025-03-26"

    def test_initialize_with_2024_11_05_backward_compat(self, client, mock_gateway):
        """Server accepts 2024-11-05 for backward compatibility."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "old-client", "version": "0.9"},
                    },
                ),
            )
            assert response.status_code == 200
            data = response.json()
            assert "result" in data
            assert data["result"]["protocolVersion"] == "2024-11-05"

    def test_initialize_with_unsupported_version_returns_error(
        self, client, mock_gateway
    ):
        """Server returns -32602 for unsupported protocol version."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "initialize",
                    {
                        "protocolVersion": "2099-01-01",
                        "capabilities": {},
                        "clientInfo": {"name": "future-client", "version": "9.9"},
                    },
                ),
            )
            assert response.status_code == 200
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32602  # JSONRPC_INVALID_PARAMS
            assert "2099-01-01" in data["error"]["message"]

    def test_initialize_without_version_uses_default(self, client, mock_gateway):
        """Server uses default version when client omits protocolVersion."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("initialize", {"capabilities": {}}),
            )
            assert response.status_code == 200
            data = response.json()
            assert "result" in data
            assert data["result"]["protocolVersion"] == "2025-11-25"

    def test_initialize_capabilities_include_tools(self, client, mock_gateway):
        """Initialize response includes tools capability with listChanged."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "initialize",
                    {"protocolVersion": "2025-03-26", "capabilities": {}},
                ),
            )
            data = response.json()
            caps = data["result"]["capabilities"]
            assert "tools" in caps
            assert caps["tools"]["listChanged"] is True

    def test_initialize_capabilities_include_resources(self, client, mock_gateway):
        """Initialize response declares resources capability with listChanged."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "initialize",
                    {"protocolVersion": "2025-03-26", "capabilities": {}},
                ),
            )
            data = response.json()
            caps = data["result"]["capabilities"]
            assert "resources" in caps
            assert caps["resources"]["listChanged"] is True
            assert caps["resources"]["subscribe"] is False


# ============================================================================
# G2: Initialized Notification
# ============================================================================


class TestInitializedNotification:
    """Test notifications/initialized handling."""

    def test_initialized_notification_accepted(self, client, mock_gateway):
        """Server accepts notifications/initialized without error."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    # No id - this is a notification
                },
            )
            # Notifications return 202 Accepted
            assert response.status_code == 202

    def test_initialized_notification_with_params(self, client, mock_gateway):
        """Server accepts notifications/initialized with optional params."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
            )
            assert response.status_code == 202


# ============================================================================
# G5: Error Code -32004 for Tool Invocation Disabled
# ============================================================================


class TestErrorCodes:
    """Test MCP-specific error codes compliance."""

    def test_tool_invocation_disabled_returns_32004(self, client, mock_gateway):
        """Tool invocation disabled returns -32004 (not -32002)."""
        mock_gateway.is_tool_invocation_enabled.return_value = False

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "tools/call",
                    {"name": "test.tool", "arguments": {}},
                ),
            )
            assert response.status_code == 200
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32004

    def test_resource_not_found_returns_32002(self, client, mock_gateway):
        """Resource not found returns -32002 (MCP spec reserved code)."""
        mock_gateway.proxy_resource_read = AsyncMock(
            return_value={
                "error": {
                    "code": -32002,
                    "message": "No server found for resource URI: file:///nonexistent",
                }
            }
        )

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "resources/read",
                    {"uri": "file:///nonexistent"},
                ),
            )
            assert response.status_code == 200
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32002

    def test_gateway_disabled_returns_32003(self, client, mock_gateway):
        """Gateway disabled returns -32003."""
        mock_gateway.is_enabled.return_value = False

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("initialize"),
            )
            data = response.json()
            assert data["error"]["code"] == -32003

    def test_tool_not_found_returns_32001(self, client, mock_gateway):
        """Tool not found returns -32001."""
        mock_gateway.get_server.return_value = None

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "tools/call",
                    {"name": "missing.tool", "arguments": {}},
                ),
            )
            data = response.json()
            assert data["error"]["code"] == -32001


# ============================================================================
# G8: Pagination for tools/list
# ============================================================================


class TestToolsListPagination:
    """Test cursor-based pagination for tools/list."""

    def _make_server_with_tools(self, server_id, tool_count):
        """Create a mock server with N tools."""
        server = MagicMock()
        server.server_id = server_id
        server.name = f"Server {server_id}"
        server.tools = [f"tool_{i}" for i in range(tool_count)]
        server.tool_definitions = {}
        return server

    def test_first_page_returns_all_when_under_limit(self, client, mock_gateway):
        """Returns all tools without nextCursor when under page size."""
        server = self._make_server_with_tools("s1", 3)
        mock_gateway.list_servers.return_value = [server]

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("tools/list"),
            )
            data = response.json()
            result = data["result"]
            assert len(result["tools"]) == 3
            assert "nextCursor" not in result

    def test_pagination_returns_next_cursor(self, client, mock_gateway):
        """Returns nextCursor when more tools are available."""
        server = self._make_server_with_tools("s1", 5)
        mock_gateway.list_servers.return_value = [server]

        with (
            patch(
                "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
                return_value=mock_gateway,
            ),
            patch("litellm_llmrouter.mcp_jsonrpc.MCP_TOOLS_PAGE_SIZE", 2),
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("tools/list"),
            )
            data = response.json()
            result = data["result"]
            assert len(result["tools"]) == 2
            assert "nextCursor" in result

    def test_pagination_second_page(self, client, mock_gateway):
        """Second page uses cursor from first page."""
        server = self._make_server_with_tools("s1", 5)
        mock_gateway.list_servers.return_value = [server]

        with (
            patch(
                "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
                return_value=mock_gateway,
            ),
            patch("litellm_llmrouter.mcp_jsonrpc.MCP_TOOLS_PAGE_SIZE", 2),
        ):
            # First page
            resp1 = client.post(
                "/mcp",
                json=_jsonrpc("tools/list"),
            )
            cursor = resp1.json()["result"]["nextCursor"]

            # Second page
            resp2 = client.post(
                "/mcp",
                json=_jsonrpc("tools/list", {"cursor": cursor}),
            )
            data2 = resp2.json()
            result2 = data2["result"]
            assert len(result2["tools"]) == 2
            # Should have cursor for page 3
            assert "nextCursor" in result2

    def test_pagination_last_page_no_cursor(self, client, mock_gateway):
        """Last page has no nextCursor."""
        server = self._make_server_with_tools("s1", 3)
        mock_gateway.list_servers.return_value = [server]

        with (
            patch(
                "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
                return_value=mock_gateway,
            ),
            patch("litellm_llmrouter.mcp_jsonrpc.MCP_TOOLS_PAGE_SIZE", 2),
        ):
            # First page
            resp1 = client.post(
                "/mcp",
                json=_jsonrpc("tools/list"),
            )
            cursor = resp1.json()["result"]["nextCursor"]

            # Second (last) page
            resp2 = client.post(
                "/mcp",
                json=_jsonrpc("tools/list", {"cursor": cursor}),
            )
            result2 = resp2.json()["result"]
            assert len(result2["tools"]) == 1
            assert "nextCursor" not in result2

    def test_pagination_empty_list(self, client, mock_gateway):
        """Empty tools list returns no tools and no cursor."""
        mock_gateway.list_servers.return_value = []

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("tools/list"),
            )
            data = response.json()
            result = data["result"]
            assert result["tools"] == []
            assert "nextCursor" not in result

    def test_cursor_encoding_is_base64(self, client, mock_gateway):
        """Cursors are valid base64-encoded offsets."""
        server = self._make_server_with_tools("s1", 5)
        mock_gateway.list_servers.return_value = [server]

        with (
            patch(
                "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
                return_value=mock_gateway,
            ),
            patch("litellm_llmrouter.mcp_jsonrpc.MCP_TOOLS_PAGE_SIZE", 2),
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("tools/list"),
            )
            cursor = response.json()["result"]["nextCursor"]
            # Verify it's base64 that decodes to a number
            decoded = base64.b64decode(cursor).decode()
            assert decoded == "2"

    def test_invalid_cursor_treated_as_zero(self, client, mock_gateway):
        """Invalid cursor is treated as offset 0 (start from beginning)."""
        server = self._make_server_with_tools("s1", 3)
        mock_gateway.list_servers.return_value = [server]

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("tools/list", {"cursor": "not-valid-base64!!!"}),
            )
            data = response.json()
            assert len(data["result"]["tools"]) == 3


# ============================================================================
# G12: Tool Annotations
# ============================================================================


class TestToolAnnotations:
    """Test tool annotations passthrough (MCP 2025-03-26)."""

    def test_tool_with_annotations_includes_them(self, client, mock_gateway):
        """Tools with annotations include them in tools/list response."""
        from litellm_llmrouter.mcp_gateway import MCPToolDefinition

        server = MagicMock()
        server.server_id = "annotated-server"
        server.name = "Annotated Server"
        server.tools = ["safe_read"]
        server.tool_definitions = {
            "safe_read": MCPToolDefinition(
                name="safe_read",
                description="Read-only data access",
                input_schema={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
                server_id="annotated-server",
                annotations={
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            )
        }
        mock_gateway.list_servers.return_value = [server]

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("tools/list"),
            )
            data = response.json()
            tools = data["result"]["tools"]
            assert len(tools) == 1
            tool = tools[0]
            assert "annotations" in tool
            assert tool["annotations"]["readOnlyHint"] is True
            assert tool["annotations"]["destructiveHint"] is False
            assert tool["annotations"]["idempotentHint"] is True
            assert tool["annotations"]["openWorldHint"] is False

    def test_tool_without_annotations_omits_field(self, client, mock_gateway):
        """Tools without annotations do not include annotations field."""
        from litellm_llmrouter.mcp_gateway import MCPToolDefinition

        server = MagicMock()
        server.server_id = "plain-server"
        server.name = "Plain Server"
        server.tools = ["basic_tool"]
        server.tool_definitions = {
            "basic_tool": MCPToolDefinition(
                name="basic_tool",
                description="A basic tool",
                input_schema={"type": "object"},
                server_id="plain-server",
                annotations=None,
            )
        }
        mock_gateway.list_servers.return_value = [server]

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("tools/list"),
            )
            data = response.json()
            tools = data["result"]["tools"]
            assert len(tools) == 1
            assert "annotations" not in tools[0]

    def test_annotations_serialized_in_tool_definition(self):
        """MCPToolDefinition serializes annotations properly."""
        from litellm_llmrouter.mcp_gateway import MCPToolDefinition

        tool = MCPToolDefinition(
            name="test",
            description="Test tool",
            annotations={
                "readOnlyHint": True,
                "destructiveHint": False,
            },
        )
        d = tool.to_dict()
        assert "annotations" in d
        assert d["annotations"]["readOnlyHint"] is True

        # Round-trip
        restored = MCPToolDefinition.from_dict(d)
        assert restored.annotations == tool.annotations

    def test_annotations_none_not_in_dict(self):
        """MCPToolDefinition with None annotations omits from dict."""
        from litellm_llmrouter.mcp_gateway import MCPToolDefinition

        tool = MCPToolDefinition(name="test", description="No annotations")
        d = tool.to_dict()
        assert "annotations" not in d


# ============================================================================
# G3: resources/read (proxy to upstream)
# ============================================================================


class TestResourcesRead:
    """Test resources/read handler with upstream proxying."""

    def test_resources_read_proxies_to_upstream(self, client, mock_gateway):
        """resources/read proxies request and returns upstream content."""
        mock_gateway.proxy_resource_read = AsyncMock(
            return_value={
                "contents": [
                    {
                        "uri": "file:///data/config.json",
                        "mimeType": "text/plain",
                        "text": "upstream content here",
                    }
                ]
            }
        )

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "resources/read",
                    {"uri": "file:///data/config.json"},
                ),
            )
            data = response.json()
            assert "result" in data
            contents = data["result"]["contents"]
            assert len(contents) == 1
            assert contents[0]["uri"] == "file:///data/config.json"
            assert contents[0]["text"] == "upstream content here"
            assert contents[0]["mimeType"] == "text/plain"

    def test_resources_read_not_found(self, client, mock_gateway):
        """resources/read returns -32002 for unknown URI."""
        mock_gateway.proxy_resource_read = AsyncMock(
            return_value={
                "error": {
                    "code": -32002,
                    "message": "No server found for resource URI: file:///nonexistent",
                }
            }
        )

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "resources/read",
                    {"uri": "file:///nonexistent"},
                ),
            )
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32002

    def test_resources_read_missing_uri(self, client, mock_gateway):
        """resources/read returns -32602 when uri is missing."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("resources/read", {}),
            )
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32602

    def test_resources_read_upstream_error(self, client, mock_gateway):
        """resources/read returns error when upstream server fails."""
        mock_gateway.proxy_resource_read = AsyncMock(
            return_value={
                "error": {
                    "code": -32603,
                    "message": "Upstream server error: HTTP 500: Internal error",
                }
            }
        )

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc(
                    "resources/read",
                    {"uri": "file:///data/broken"},
                ),
            )
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32603

    def test_resources_read_calls_proxy(self, client, mock_gateway):
        """resources/read calls gateway.proxy_resource_read with correct URI."""
        mock_gateway.proxy_resource_read = AsyncMock(
            return_value={
                "contents": [
                    {
                        "uri": "file:///test/file.txt",
                        "mimeType": "text/plain",
                        "text": "content",
                    }
                ]
            }
        )

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            client.post(
                "/mcp",
                json=_jsonrpc(
                    "resources/read",
                    {"uri": "file:///test/file.txt"},
                ),
            )
            mock_gateway.proxy_resource_read.assert_called_once_with(
                "file:///test/file.txt"
            )


# ============================================================================
# G4: notifications/tools/list_changed
# ============================================================================


class TestToolsListChanged:
    """Test tool change notification mechanism."""

    def test_gateway_notifies_on_register(self):
        """MCPGateway fires callback when server with tools is registered."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer

        gateway = MCPGateway()
        gateway.enabled = True

        callback_called = []
        gateway.on_tools_changed(lambda: callback_called.append(True))

        server = MCPServer(
            server_id="test",
            name="Test",
            url="https://example.com",
            tools=["echo"],
        )
        gateway.register_server(server)
        assert len(callback_called) == 1

    def test_gateway_notifies_on_unregister(self):
        """MCPGateway fires callback when server is unregistered."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer

        gateway = MCPGateway()
        gateway.enabled = True

        server = MCPServer(
            server_id="test",
            name="Test",
            url="https://example.com",
            tools=["echo"],
        )
        gateway.register_server(server)

        callback_called = []
        gateway.on_tools_changed(lambda: callback_called.append(True))

        gateway.unregister_server("test")
        assert len(callback_called) == 1

    def test_gateway_no_notify_on_register_without_tools(self):
        """MCPGateway does not fire callback for server with no tools."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer

        gateway = MCPGateway()
        gateway.enabled = True

        callback_called = []
        gateway.on_tools_changed(lambda: callback_called.append(True))

        server = MCPServer(
            server_id="empty",
            name="Empty",
            url="https://example.com",
            tools=[],
        )
        gateway.register_server(server)
        assert len(callback_called) == 0

    def test_sse_notify_function_exists(self):
        """notify_tools_list_changed function is importable."""
        from litellm_llmrouter.mcp_sse_transport import notify_tools_list_changed

        # Should be callable
        assert callable(notify_tools_list_changed)

    async def test_emit_tools_list_changed_queues_notifications(self):
        """emit_tools_list_changed queues notification for active sessions."""
        from litellm_llmrouter.mcp_jsonrpc import (
            emit_tools_list_changed,
            get_or_create_session,
            get_active_sessions,
        )

        # Create a session
        session_id = get_or_create_session({})
        sessions = get_active_sessions()
        assert session_id in sessions

        # Emit tools list changed
        await emit_tools_list_changed()

        # Check pending notifications
        session = sessions[session_id]
        assert "pending_notifications" in session
        assert len(session["pending_notifications"]) == 1
        notification = session["pending_notifications"][0]
        assert notification["method"] == "notifications/tools/list_changed"
        assert notification["jsonrpc"] == "2.0"


# ============================================================================
# G9: resources/list Pagination
# ============================================================================


class TestResourcesListPagination:
    """Test cursor-based pagination for resources/list."""

    def _make_resources(self, count):
        """Create a list of mock resources."""
        return [
            {
                "server_id": f"s{i}",
                "server_name": f"Server {i}",
                "resource": f"file:///data/resource_{i}.json",
            }
            for i in range(count)
        ]

    def test_first_page_returns_all_when_under_limit(self, client, mock_gateway):
        """Returns all resources without nextCursor when under page size."""
        mock_gateway.list_resources.return_value = self._make_resources(3)

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("resources/list"),
            )
            data = response.json()
            result = data["result"]
            assert len(result["resources"]) == 3
            assert "nextCursor" not in result

    def test_pagination_returns_next_cursor(self, client, mock_gateway):
        """Returns nextCursor when more resources are available."""
        mock_gateway.list_resources.return_value = self._make_resources(5)

        with (
            patch(
                "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
                return_value=mock_gateway,
            ),
            patch("litellm_llmrouter.mcp_jsonrpc.MCP_RESOURCES_PAGE_SIZE", 2),
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("resources/list"),
            )
            data = response.json()
            result = data["result"]
            assert len(result["resources"]) == 2
            assert "nextCursor" in result

    def test_pagination_second_page(self, client, mock_gateway):
        """Second page uses cursor from first page."""
        mock_gateway.list_resources.return_value = self._make_resources(5)

        with (
            patch(
                "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
                return_value=mock_gateway,
            ),
            patch("litellm_llmrouter.mcp_jsonrpc.MCP_RESOURCES_PAGE_SIZE", 2),
        ):
            # First page
            resp1 = client.post(
                "/mcp",
                json=_jsonrpc("resources/list"),
            )
            cursor = resp1.json()["result"]["nextCursor"]

            # Second page
            resp2 = client.post(
                "/mcp",
                json=_jsonrpc("resources/list", {"cursor": cursor}),
            )
            result2 = resp2.json()["result"]
            assert len(result2["resources"]) == 2
            # Should have cursor for page 3
            assert "nextCursor" in result2

    def test_pagination_last_page_no_cursor(self, client, mock_gateway):
        """Last page has no nextCursor."""
        mock_gateway.list_resources.return_value = self._make_resources(3)

        with (
            patch(
                "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
                return_value=mock_gateway,
            ),
            patch("litellm_llmrouter.mcp_jsonrpc.MCP_RESOURCES_PAGE_SIZE", 2),
        ):
            # First page
            resp1 = client.post(
                "/mcp",
                json=_jsonrpc("resources/list"),
            )
            cursor = resp1.json()["result"]["nextCursor"]

            # Second (last) page
            resp2 = client.post(
                "/mcp",
                json=_jsonrpc("resources/list", {"cursor": cursor}),
            )
            result2 = resp2.json()["result"]
            assert len(result2["resources"]) == 1
            assert "nextCursor" not in result2

    def test_pagination_empty_list(self, client, mock_gateway):
        """Empty resources list returns no resources and no cursor."""
        mock_gateway.list_resources.return_value = []

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("resources/list"),
            )
            data = response.json()
            result = data["result"]
            assert result["resources"] == []
            assert "nextCursor" not in result

    def test_pagination_page_size_capped_at_100(self, client, mock_gateway):
        """Client-requested pageSize is capped at 100."""
        mock_gateway.list_resources.return_value = self._make_resources(150)

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("resources/list", {"pageSize": 200}),
            )
            data = response.json()
            result = data["result"]
            # Should cap at 100
            assert len(result["resources"]) == 100
            assert "nextCursor" in result

    def test_pagination_custom_page_size(self, client, mock_gateway):
        """Client can request a smaller pageSize."""
        mock_gateway.list_resources.return_value = self._make_resources(10)

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("resources/list", {"pageSize": 3}),
            )
            data = response.json()
            result = data["result"]
            assert len(result["resources"]) == 3
            assert "nextCursor" in result

    def test_pagination_boundary_exact_page(self, client, mock_gateway):
        """When resources exactly fill the page, no nextCursor is returned."""
        mock_gateway.list_resources.return_value = self._make_resources(4)

        with (
            patch(
                "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
                return_value=mock_gateway,
            ),
            patch("litellm_llmrouter.mcp_jsonrpc.MCP_RESOURCES_PAGE_SIZE", 4),
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("resources/list"),
            )
            data = response.json()
            result = data["result"]
            assert len(result["resources"]) == 4
            assert "nextCursor" not in result


# ============================================================================
# G10: Mcp-Session-Id Header Management
# ============================================================================


class TestMcpSessionId:
    """Test Mcp-Session-Id header management."""

    def test_response_includes_session_id_header(self, client, mock_gateway):
        """JSON-RPC responses include Mcp-Session-Id header."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("initialize", {"capabilities": {}}),
            )
            assert response.status_code == 200
            assert "mcp-session-id" in response.headers

    def test_session_id_is_uuid_format(self, client, mock_gateway):
        """Session ID is a valid UUID."""
        import uuid

        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("initialize", {"capabilities": {}}),
            )
            session_id = response.headers["mcp-session-id"]
            # Should not raise
            uuid.UUID(session_id)

    def test_session_id_reused_on_subsequent_requests(self, client, mock_gateway):
        """Same session ID is returned when client sends it back."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            # First request - get session ID
            resp1 = client.post(
                "/mcp",
                json=_jsonrpc("initialize", {"capabilities": {}}),
            )
            session_id = resp1.headers["mcp-session-id"]

            # Second request - send session ID back
            resp2 = client.post(
                "/mcp",
                json=_jsonrpc("tools/list"),
                headers={"Mcp-Session-Id": session_id},
            )
            assert resp2.headers["mcp-session-id"] == session_id

    def test_new_session_for_unknown_id(self, client, mock_gateway):
        """A new session is created when client sends unknown session ID."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json=_jsonrpc("initialize", {"capabilities": {}}),
                headers={"Mcp-Session-Id": "nonexistent-session-id"},
            )
            session_id = response.headers["mcp-session-id"]
            assert session_id != "nonexistent-session-id"

    def test_notification_includes_session_id(self, client, mock_gateway):
        """Notification responses include Mcp-Session-Id header."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                },
            )
            assert response.status_code == 202
            assert "mcp-session-id" in response.headers


# ============================================================================
# G11: Session Management
# ============================================================================


class TestSessionManagement:
    """Test session lifecycle management."""

    def test_get_or_create_session_creates_new(self):
        """get_or_create_session creates a new session when none exists."""
        from litellm_llmrouter.mcp_jsonrpc import (
            get_or_create_session,
            get_active_sessions,
        )

        session_id = get_or_create_session({})
        assert session_id
        sessions = get_active_sessions()
        assert session_id in sessions
        assert "created_at" in sessions[session_id]
        assert "last_active" in sessions[session_id]

    def test_get_or_create_session_reuses_existing(self):
        """get_or_create_session reuses session when valid ID is provided."""
        from litellm_llmrouter.mcp_jsonrpc import get_or_create_session

        # Create session
        session_id = get_or_create_session({})

        # Reuse it
        reused_id = get_or_create_session({"mcp-session-id": session_id})
        assert reused_id == session_id

    def test_get_or_create_session_expired_creates_new(self):
        """get_or_create_session creates new session when existing is expired."""
        import time

        from litellm_llmrouter.mcp_jsonrpc import (
            get_or_create_session,
            get_active_sessions,
        )

        # Create session
        session_id = get_or_create_session({})

        # Expire it by backdating last_active
        sessions = get_active_sessions()
        sessions[session_id]["last_active"] = time.time() - 99999

        # Should create new
        new_id = get_or_create_session({"mcp-session-id": session_id})
        assert new_id != session_id

    def test_reset_sessions_clears_all(self):
        """reset_sessions clears all active sessions."""
        from litellm_llmrouter.mcp_jsonrpc import (
            get_or_create_session,
            get_active_sessions,
            reset_sessions,
        )

        get_or_create_session({})
        get_or_create_session({})
        assert len(get_active_sessions()) >= 1

        reset_sessions()
        assert len(get_active_sessions()) == 0


# ============================================================================
# G13: find_server_for_resource
# ============================================================================


class TestFindServerForResource:
    """Test MCPGateway.find_server_for_resource method."""

    def test_find_resource_owner(self):
        """find_server_for_resource returns correct server."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer

        gateway = MCPGateway()
        gateway.enabled = True

        server = MCPServer(
            server_id="data-server",
            name="Data Server",
            url="https://data.example.com",
            resources=["file:///data/config.json", "file:///data/users.json"],
        )
        gateway.register_server(server)

        found = gateway.find_server_for_resource("file:///data/config.json")
        assert found is not None
        assert found.server_id == "data-server"

    def test_find_resource_not_found(self):
        """find_server_for_resource returns None for unknown URI."""
        from litellm_llmrouter.mcp_gateway import MCPGateway

        gateway = MCPGateway()
        gateway.enabled = True

        found = gateway.find_server_for_resource("file:///nonexistent")
        assert found is None

    def test_find_resource_multi_server(self):
        """find_server_for_resource finds correct server among multiple."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer

        gateway = MCPGateway()
        gateway.enabled = True

        server1 = MCPServer(
            server_id="s1",
            name="Server 1",
            url="https://s1.example.com",
            resources=["file:///s1/data.json"],
        )
        server2 = MCPServer(
            server_id="s2",
            name="Server 2",
            url="https://s2.example.com",
            resources=["file:///s2/data.json"],
        )
        gateway.register_server(server1)
        gateway.register_server(server2)

        found = gateway.find_server_for_resource("file:///s2/data.json")
        assert found is not None
        assert found.server_id == "s2"


# ============================================================================
# Protocol surface: method dispatch
# ============================================================================


class TestMethodDispatch:
    """Test that all required MCP methods are registered."""

    def test_all_required_methods_exist(self, client, mock_gateway):
        """All MCP 2025-03-26 required methods return valid responses."""
        required_methods = [
            "initialize",
            "tools/list",
            "tools/call",
            "resources/list",
            "resources/read",
        ]
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            for method in required_methods:
                response = client.post(
                    "/mcp",
                    json=_jsonrpc(method, {}),
                )
                data = response.json()
                # Should not get "method not found"
                if "error" in data:
                    assert data["error"]["code"] != -32601, (
                        f"Method {method} returned method-not-found"
                    )

    def test_notification_methods_accepted(self, client, mock_gateway):
        """Notification methods are accepted without error."""
        with patch(
            "litellm_llmrouter.mcp_jsonrpc.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                },
            )
            assert response.status_code == 202
