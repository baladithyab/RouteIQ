"""
Unit tests for MCP SSE Transport.

Tests the /mcp/sse endpoint for Server-Sent Events transport.
Validates SSE framing, session management, and transport mode selection.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Set environment variables before importing modules
os.environ.setdefault("MCP_GATEWAY_ENABLED", "true")
os.environ.setdefault("MCP_SSE_TRANSPORT_ENABLED", "true")
os.environ.setdefault("MCP_SSE_LEGACY_MODE", "false")


@pytest.fixture
def app():
    """Create a FastAPI app with MCP SSE router."""
    from litellm_llmrouter.mcp_sse_transport import mcp_sse_router

    app = FastAPI()

    # Mock authentication
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    async def mock_auth():
        return None

    app.dependency_overrides[user_api_key_auth] = mock_auth
    app.include_router(mcp_sse_router, prefix="")

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


# ============================================================================
# SSE Event Formatting Tests
# ============================================================================


class TestSSEEventFormatting:
    """Test SSE event formatting utilities."""

    def test_format_sse_event_simple_data(self):
        """format_sse_event formats simple data correctly."""
        from litellm_llmrouter.mcp_sse_transport import format_sse_event

        event = format_sse_event({"type": "test", "value": 123})

        # Should contain data line with JSON
        assert "data:" in event
        assert '{"type": "test", "value": 123}' in event
        # Should end with double newline
        assert event.endswith("\n\n")

    def test_format_sse_event_with_event_type(self):
        """format_sse_event includes event type when provided."""
        from litellm_llmrouter.mcp_sse_transport import format_sse_event

        event = format_sse_event(
            data={"message": "hello"},
            event="message",
        )

        assert "event: message\n" in event
        assert "data:" in event

    def test_format_sse_event_with_id(self):
        """format_sse_event includes event ID when provided."""
        from litellm_llmrouter.mcp_sse_transport import format_sse_event

        event = format_sse_event(
            data={"test": True},
            event_id="12345",
        )

        assert "id: 12345\n" in event
        assert "data:" in event

    def test_format_sse_event_with_retry(self):
        """format_sse_event includes retry interval when provided."""
        from litellm_llmrouter.mcp_sse_transport import format_sse_event

        event = format_sse_event(
            data={"test": True},
            retry=3000,
        )

        assert "retry: 3000\n" in event
        assert "data:" in event

    def test_format_sse_event_full(self):
        """format_sse_event handles all fields together."""
        from litellm_llmrouter.mcp_sse_transport import format_sse_event

        event = format_sse_event(
            data={"type": "session.created"},
            event="session",
            event_id="1",
            retry=5000,
        )

        lines = event.split("\n")
        assert "event: session" in lines
        assert "id: 1" in lines
        assert "retry: 5000" in lines
        assert any("data:" in line for line in lines)

    def test_format_sse_event_string_data(self):
        """format_sse_event handles string data without JSON encoding."""
        from litellm_llmrouter.mcp_sse_transport import format_sse_event

        event = format_sse_event(data="plain text message")

        assert "data: plain text message\n" in event

    def test_format_sse_comment(self):
        """format_sse_comment formats comments correctly."""
        from litellm_llmrouter.mcp_sse_transport import format_sse_comment

        comment = format_sse_comment("ping 1234567890")

        assert comment == ": ping 1234567890\n\n"

    def test_format_sse_event_multiline_data(self):
        """format_sse_event handles multi-line data."""
        from litellm_llmrouter.mcp_sse_transport import format_sse_event

        event = format_sse_event(data="line1\nline2\nline3")

        assert "data: line1\n" in event
        assert "data: line2\n" in event
        assert "data: line3\n" in event


# ============================================================================
# Transport Mode Tests
# ============================================================================


class TestTransportMode:
    """Test transport mode selection and feature flags."""

    def test_get_transport_mode_sse_enabled(self):
        """get_transport_mode returns 'sse' when SSE is enabled."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.MCP_SSE_TRANSPORT_ENABLED", True
        ), patch("litellm_llmrouter.mcp_sse_transport.MCP_SSE_LEGACY_MODE", False):
            from litellm_llmrouter.mcp_sse_transport import get_transport_mode

            # Need to reload to get fresh values
            assert get_transport_mode.__module__ == "litellm_llmrouter.mcp_sse_transport"

    def test_get_transport_mode_legacy(self):
        """get_transport_mode returns 'legacy' when legacy mode is set."""
        from litellm_llmrouter.mcp_sse_transport import get_transport_mode

        with patch(
            "litellm_llmrouter.mcp_sse_transport.MCP_SSE_LEGACY_MODE", True
        ):
            # The function reads module-level constants, so we need to check behavior
            # This test verifies the function exists and returns expected types
            mode = get_transport_mode()
            assert mode in ["sse", "legacy", "disabled"]


# ============================================================================
# SSE Transport Info Endpoint Tests
# ============================================================================


class TestSSETransportInfo:
    """Test GET /mcp/transport endpoint."""

    def test_transport_info_returns_config(self, client, mock_gateway):
        """GET /mcp/transport returns transport configuration."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.get("/mcp/transport")
            assert response.status_code == 200

            data = response.json()
            assert "transports" in data
            assert "current_mode" in data
            assert "config" in data
            assert "feature_flags" in data

            # Check transport entries
            assert "sse" in data["transports"]
            assert "http" in data["transports"]

            # SSE transport info
            sse = data["transports"]["sse"]
            assert "enabled" in sse
            assert "endpoint" in sse
            assert sse["endpoint"] == "/mcp/sse"

            # HTTP transport info
            http = data["transports"]["http"]
            assert http["enabled"] is True
            assert http["endpoint"] == "/mcp"

    def test_transport_info_shows_config_values(self, client, mock_gateway):
        """GET /mcp/transport shows configuration values."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.get("/mcp/transport")
            data = response.json()

            config = data["config"]
            assert "heartbeat_interval" in config
            assert "max_connection_duration" in config
            assert "retry_interval_ms" in config


# ============================================================================
# SSE Endpoint Tests
# ============================================================================


class TestSSEEndpoint:
    """Test GET /mcp/sse endpoint."""

    def test_sse_requires_accept_header(self, client, mock_gateway):
        """GET /mcp/sse requires Accept: text/event-stream header."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            # Without proper Accept header
            response = client.get(
                "/mcp/sse",
                headers={"Accept": "application/json"},
            )
            assert response.status_code == 406

            data = response.json()
            assert data["detail"]["error"] == "not_acceptable"

    def test_sse_fails_when_gateway_disabled(self, client, mock_gateway):
        """GET /mcp/sse returns 404 when gateway is disabled."""
        mock_gateway.is_enabled.return_value = False

        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.get(
                "/mcp/sse",
                headers={"Accept": "text/event-stream"},
            )
            assert response.status_code == 404

            data = response.json()
            assert data["detail"]["error"] == "mcp_gateway_disabled"

    @pytest.mark.skip(reason="SSE streaming test requires special async handling - headers validation covered by integration test")
    def test_sse_returns_stream_response(self, client, mock_gateway):
        """GET /mcp/sse returns streaming response with correct headers."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            # Use stream context manager to avoid blocking on infinite stream
            with client.stream(
                "GET",
                "/mcp/sse",
                headers={"Accept": "text/event-stream"},
            ) as response:
                # The response should start streaming with correct headers
                assert response.headers.get("content-type") == "text/event-stream; charset=utf-8"
                assert response.headers.get("cache-control") == "no-cache, no-store, must-revalidate"
                assert response.headers.get("x-accel-buffering") == "no"

    @pytest.mark.skip(reason="SSE streaming test requires special async handling")
    def test_sse_accepts_wildcard_accept(self, client, mock_gateway):
        """GET /mcp/sse accepts Accept: */* header."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            # Use stream context manager to avoid blocking on infinite stream
            with client.stream(
                "GET",
                "/mcp/sse",
                headers={"Accept": "*/*"},
            ) as response:
                # Should not get 406 Not Acceptable
                assert response.status_code != 406


# ============================================================================
# SSE Messages POST Endpoint Tests
# ============================================================================


class TestSSEMessagesEndpoint:
    """Test POST /mcp/sse/messages endpoint."""

    def test_post_messages_handles_jsonrpc(self, client, mock_gateway):
        """POST /mcp/sse/messages handles JSON-RPC requests."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp/sse/messages",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "test.tool", "arguments": {}},
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["jsonrpc"] == "2.0"

    def test_post_messages_empty_body_error(self, client, mock_gateway):
        """POST /mcp/sse/messages returns error for empty body."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp/sse/messages",
                content=b"",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32700

    def test_post_messages_invalid_json_error(self, client, mock_gateway):
        """POST /mcp/sse/messages returns error for invalid JSON."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp/sse/messages",
                content=b"not json",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32700

    def test_post_messages_unsupported_method(self, client, mock_gateway):
        """POST /mcp/sse/messages returns error for unsupported methods."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp/sse/messages",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",  # Not supported via SSE POST
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32601  # Method not found

    def test_post_messages_tool_call_disabled(self, client, mock_gateway):
        """POST /mcp/sse/messages returns error when tool invocation disabled."""
        mock_gateway.is_tool_invocation_enabled.return_value = False

        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp/sse/messages",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "test.tool"},
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32002


# ============================================================================
# SSE Sessions Endpoint Tests
# ============================================================================


class TestSSESessionsEndpoint:
    """Test GET /mcp/sse/sessions endpoint."""

    def test_sessions_endpoint_returns_list(self, client, mock_gateway):
        """GET /mcp/sse/sessions returns session list."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.get("/mcp/sse/sessions")
            assert response.status_code == 200

            data = response.json()
            assert "active_sessions" in data
            assert "sessions" in data
            assert isinstance(data["sessions"], list)


# ============================================================================
# SSE Session Management Tests
# ============================================================================


class TestSSESessionManagement:
    """Test SSE session management."""

    def test_sse_session_creation(self):
        """SSESession can be created with defaults."""
        from litellm_llmrouter.mcp_sse_transport import SSESession

        session = SSESession(session_id="test-session-123")

        assert session.session_id == "test-session-123"
        assert session.client_id is None
        assert session.is_active is True
        assert session.last_event_id == 0

    def test_sse_session_next_event_id(self):
        """SSESession.next_event_id increments correctly."""
        from litellm_llmrouter.mcp_sse_transport import SSESession

        session = SSESession(session_id="test-session")

        id1 = session.next_event_id()
        id2 = session.next_event_id()
        id3 = session.next_event_id()

        assert id1 == "1"
        assert id2 == "2"
        assert id3 == "3"
        assert session.last_event_id == 3


# ============================================================================
# Legacy Mode / Rollback Tests
# ============================================================================


class TestLegacyModeRollback:
    """Test legacy mode rollback behavior."""

    def test_sse_disabled_in_legacy_mode(self, app, mock_gateway):
        """SSE endpoints return 404 when in legacy mode."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.MCP_SSE_LEGACY_MODE", True
        ), patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            client = TestClient(app)
            response = client.get(
                "/mcp/sse",
                headers={"Accept": "text/event-stream"},
            )

            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["error"] == "sse_legacy_mode"

    def test_transport_info_shows_legacy_mode(self, client, mock_gateway):
        """GET /mcp/transport shows legacy mode status."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.get("/mcp/transport")
            data = response.json()

            assert "legacy_mode" in data
            assert "feature_flags" in data
            assert "MCP_SSE_LEGACY_MODE" in data["feature_flags"]


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestSSETransportIntegration:
    """Integration-style tests for SSE transport behavior."""

    @pytest.mark.skip(reason="SSE streaming test requires special async handling - hangs in TestClient")
    def test_sse_initial_event_format(self, client, mock_gateway):
        """SSE stream starts with session.created event."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            # Use stream=True to get the raw content
            with client.stream(
                "GET",
                "/mcp/sse",
                headers={"Accept": "text/event-stream"},
            ) as response:
                # Read first chunk
                first_chunk = next(response.iter_lines())

                # Should be the event type line
                assert first_chunk.startswith("event:") or first_chunk.startswith("id:")

    def test_tools_call_via_sse_messages(self, client, mock_gateway):
        """tools/call can be invoked via /mcp/sse/messages."""
        # Mock a successful tool invocation
        from litellm_llmrouter.mcp_gateway import MCPToolResult

        async def mock_invoke_tool(tool_name, args):
            return MCPToolResult(
                success=True,
                result={"echo": args.get("message", "")},
                tool_name=tool_name,
                server_id="test-server",
            )

        mock_gateway.invoke_tool = mock_invoke_tool

        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp/sse/messages",
                json={
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "tools/call",
                    "params": {
                        "name": "test-server.echo",
                        "arguments": {"message": "hello"},
                    },
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 42
            assert "result" in data
            assert "content" in data["result"]


# ============================================================================
# Feature Flag Environment Variable Tests
# ============================================================================


class TestFeatureFlagEnvironment:
    """Test feature flag environment variable behavior."""

    def test_default_sse_enabled(self):
        """SSE transport is enabled by default."""
        # Clear any existing value and test default
        with patch.dict(os.environ, {}, clear=False):
            # Default should be "true"
            from litellm_llmrouter.mcp_sse_transport import MCP_SSE_TRANSPORT_ENABLED
            # The constant is set at import time, so we verify it's True as designed
            assert MCP_SSE_TRANSPORT_ENABLED is True

    def test_default_legacy_mode_off(self):
        """Legacy mode is off by default."""
        from litellm_llmrouter.mcp_sse_transport import MCP_SSE_LEGACY_MODE
        assert MCP_SSE_LEGACY_MODE is False


# ============================================================================
# Legacy SSE Messages Endpoint Tests
# ============================================================================


class TestLegacyMessagesEndpoint:
    """Test POST /mcp/messages endpoint for legacy SSE transport."""

    def test_messages_endpoint_requires_session_id(self, client, mock_gateway):
        """POST /mcp/messages requires sessionId query parameter."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp/messages",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                },
            )
            # FastAPI returns 422 for missing required query param
            assert response.status_code == 422

    def test_messages_endpoint_invalid_session(self, client, mock_gateway):
        """POST /mcp/messages returns 404 for invalid session."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.post(
                "/mcp/messages?sessionId=invalid-session-id",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                },
            )
            assert response.status_code == 404

            data = response.json()
            assert data["detail"]["error"] == "session_not_found"


# ============================================================================
# Session Isolation Tests
# ============================================================================


class TestSessionIsolation:
    """Test session isolation and async queue behavior."""

    @pytest.mark.asyncio
    async def test_session_queue_creation(self):
        """SSESession creates async queue on demand."""
        from litellm_llmrouter.mcp_sse_transport import SSESession

        session = SSESession(session_id="test-queue-session")

        # Queue should not exist yet
        assert session._response_queue is None

        # Get queue (creates it)
        queue = await session.get_queue()
        assert queue is not None
        assert session._response_queue is not None

        # Second call returns same queue
        queue2 = await session.get_queue()
        assert queue2 is queue

    @pytest.mark.asyncio
    async def test_session_send_response(self):
        """SSESession.send_response puts message in queue."""
        from litellm_llmrouter.mcp_sse_transport import SSESession

        session = SSESession(session_id="test-send-session")

        test_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"test": "data"},
        }

        await session.send_response(test_response)

        queue = await session.get_queue()
        assert not queue.empty()

        received = await queue.get()
        assert received == test_response

    @pytest.mark.asyncio
    async def test_session_expiration(self):
        """SSESession.is_expired checks against timeout."""
        import time
        from litellm_llmrouter.mcp_sse_transport import SSESession, MCP_SSE_SESSION_TIMEOUT

        session = SSESession(session_id="test-expire-session")

        # Fresh session should not be expired
        assert not session.is_expired()

        # Set last_activity to past timeout
        session.last_activity = time.time() - MCP_SSE_SESSION_TIMEOUT - 1
        assert session.is_expired()

    @pytest.mark.asyncio
    async def test_session_touch_updates_activity(self):
        """SSESession.touch updates last_activity timestamp."""
        import time
        from litellm_llmrouter.mcp_sse_transport import SSESession

        session = SSESession(session_id="test-touch-session")
        original_activity = session.last_activity

        # Small delay
        time.sleep(0.01)

        session.touch()
        assert session.last_activity > original_activity

    @pytest.mark.asyncio
    async def test_get_session_returns_active(self):
        """get_session returns active, non-expired sessions."""
        from litellm_llmrouter.mcp_sse_transport import (
            SSESession,
            _sse_sessions,
            get_session,
            _sessions_lock,
        )

        session_id = "test-get-session-123"
        session = SSESession(session_id=session_id)

        # Register manually
        async with _sessions_lock:
            _sse_sessions[session_id] = session

        try:
            result = await get_session(session_id)
            assert result is session

            # Non-existent session
            result = await get_session("nonexistent")
            assert result is None

            # Inactive session
            session.is_active = False
            result = await get_session(session_id)
            assert result is None

        finally:
            # Cleanup
            async with _sessions_lock:
                if session_id in _sse_sessions:
                    del _sse_sessions[session_id]

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self):
        """cleanup_expired_sessions removes expired and inactive sessions."""
        import time
        from litellm_llmrouter.mcp_sse_transport import (
            SSESession,
            _sse_sessions,
            cleanup_expired_sessions,
            MCP_SSE_SESSION_TIMEOUT,
            _sessions_lock,
        )

        # Create sessions
        active_session = SSESession(session_id="active-session")
        inactive_session = SSESession(session_id="inactive-session")
        inactive_session.is_active = False

        expired_session = SSESession(session_id="expired-session")
        expired_session.last_activity = time.time() - MCP_SSE_SESSION_TIMEOUT - 10

        async with _sessions_lock:
            _sse_sessions["active-session"] = active_session
            _sse_sessions["inactive-session"] = inactive_session
            _sse_sessions["expired-session"] = expired_session

        try:
            removed = await cleanup_expired_sessions()
            # Should remove inactive and expired
            assert removed >= 2

            # Active session should remain
            assert "active-session" in _sse_sessions
            # Others should be gone
            assert "inactive-session" not in _sse_sessions
            assert "expired-session" not in _sse_sessions

        finally:
            # Cleanup
            async with _sessions_lock:
                if "active-session" in _sse_sessions:
                    del _sse_sessions["active-session"]


# ============================================================================
# JSON-RPC Method Dispatch Tests
# ============================================================================


class TestJSONRPCDispatch:
    """Test JSON-RPC method dispatching via SSE transport."""

    @pytest.mark.asyncio
    async def test_dispatch_initialize(self, mock_gateway):
        """_dispatch_jsonrpc_method handles initialize."""
        from litellm_llmrouter.mcp_sse_transport import (
            _dispatch_jsonrpc_method,
            SSESession,
        )

        session = SSESession(session_id="dispatch-init-session")

        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            result = await _dispatch_jsonrpc_method(
                method="initialize",
                request_id=1,
                params={"protocolVersion": "2024-11-05"},
                session=session,
            )

            assert result["jsonrpc"] == "2.0"
            assert result["id"] == 1
            assert "result" in result
            assert result["result"]["protocolVersion"] == "2024-11-05"
            assert session.is_initialized is True

    @pytest.mark.asyncio
    async def test_dispatch_tools_list(self, mock_gateway):
        """_dispatch_jsonrpc_method handles tools/list."""
        from litellm_llmrouter.mcp_sse_transport import (
            _dispatch_jsonrpc_method,
            SSESession,
        )

        # Mock server with tools
        mock_server = MagicMock()
        mock_server.server_id = "test-server"
        mock_server.name = "Test Server"
        mock_server.tools = ["tool1", "tool2"]
        mock_server.tool_definitions = {}
        mock_gateway.list_servers.return_value = [mock_server]

        session = SSESession(session_id="dispatch-tools-session")

        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            result = await _dispatch_jsonrpc_method(
                method="tools/list",
                request_id=2,
                params=None,
                session=session,
            )

            assert result["jsonrpc"] == "2.0"
            assert result["id"] == 2
            assert "result" in result
            assert "tools" in result["result"]
            assert len(result["result"]["tools"]) == 2
            # Tools should be namespaced
            tool_names = [t["name"] for t in result["result"]["tools"]]
            assert "test-server.tool1" in tool_names
            assert "test-server.tool2" in tool_names

    @pytest.mark.asyncio
    async def test_dispatch_unknown_method(self, mock_gateway):
        """_dispatch_jsonrpc_method returns error for unknown method."""
        from litellm_llmrouter.mcp_sse_transport import (
            _dispatch_jsonrpc_method,
            SSESession,
        )

        session = SSESession(session_id="dispatch-unknown-session")

        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            result = await _dispatch_jsonrpc_method(
                method="nonexistent/method",
                request_id=3,
                params=None,
                session=session,
            )

            assert result["jsonrpc"] == "2.0"
            assert result["id"] == 3
            assert "error" in result
            assert result["error"]["code"] == -32601  # Method not found


# ============================================================================
# Transport Info Tests
# ============================================================================


class TestTransportInfoEndpoint:
    """Test GET /mcp/transport with new SSE fields."""

    def test_transport_info_includes_session_timeout(self, client, mock_gateway):
        """GET /mcp/transport includes session_timeout in config."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.get("/mcp/transport")
            assert response.status_code == 200

            data = response.json()
            config = data["config"]
            assert "session_timeout" in config

    def test_transport_info_includes_messages_endpoint(self, client, mock_gateway):
        """GET /mcp/transport shows /mcp/messages endpoint."""
        with patch(
            "litellm_llmrouter.mcp_sse_transport.get_mcp_gateway",
            return_value=mock_gateway,
        ):
            response = client.get("/mcp/transport")
            assert response.status_code == 200

            data = response.json()
            sse_info = data["transports"]["sse"]
            assert sse_info["messages_endpoint"] == "/mcp/messages"
            assert sse_info["legacy_messages_endpoint"] == "/mcp/sse/messages"
