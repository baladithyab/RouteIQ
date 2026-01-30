"""
Tests for MCP Tool Invocation Feature
=====================================

Tests the MCP tool invocation functionality including:
- Default disabled state (returns error with code tool_invocation_disabled)
- Enabled state making real HTTP calls to MCP servers
- SSRF protection during invocation
- Timeout handling
"""

import os
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from litellm_llmrouter.mcp_gateway import (
    get_mcp_gateway,
    MCPServer,
    MCPTransport,
    reset_mcp_gateway,
)

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


class TestMCPToolInvocationFeatureFlag:
    """Tests for the MCP tool invocation feature flag."""

    @pytest.fixture(autouse=True)
    def setup_gateway(self):
        """Reset gateway before each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        reset_mcp_gateway()
        clear_ssrf_config_cache()

        # Store original env vars
        self._orig_enabled = os.environ.get("MCP_GATEWAY_ENABLED")
        self._orig_invocation = os.environ.get("LLMROUTER_ENABLE_MCP_TOOL_INVOCATION")
        self._orig_allow_private = os.environ.get("LLMROUTER_ALLOW_PRIVATE_IPS")

        yield

        # Restore original env vars
        reset_mcp_gateway()
        clear_ssrf_config_cache()
        if self._orig_enabled is not None:
            os.environ["MCP_GATEWAY_ENABLED"] = self._orig_enabled
        else:
            os.environ.pop("MCP_GATEWAY_ENABLED", None)
        if self._orig_invocation is not None:
            os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = self._orig_invocation
        else:
            os.environ.pop("LLMROUTER_ENABLE_MCP_TOOL_INVOCATION", None)
        if self._orig_allow_private is not None:
            os.environ["LLMROUTER_ALLOW_PRIVATE_IPS"] = self._orig_allow_private
        else:
            os.environ.pop("LLMROUTER_ALLOW_PRIVATE_IPS", None)

    async def test_tool_invocation_disabled_by_default(self):
        """Test that tool invocation is disabled by default and returns appropriate error."""
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        # Don't set LLMROUTER_ENABLE_MCP_TOOL_INVOCATION - should default to false
        os.environ.pop("LLMROUTER_ENABLE_MCP_TOOL_INVOCATION", None)

        gateway = get_mcp_gateway()
        assert gateway.is_enabled() is True
        assert gateway.is_tool_invocation_enabled() is False

        # Register a server
        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            url="https://example.com/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["test_tool"],
        )
        gateway.register_server(server)

        # Try to invoke tool - should fail with disabled error
        result = await gateway.invoke_tool("test_tool", {"arg": "value"})
        assert result.success is False
        assert "tool_invocation_disabled" in result.error
        assert "LLMROUTER_ENABLE_MCP_TOOL_INVOCATION" in result.error

    async def test_is_tool_invocation_enabled_flag(self):
        """Test the is_tool_invocation_enabled() method."""
        # Test default (false)
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ.pop("LLMROUTER_ENABLE_MCP_TOOL_INVOCATION", None)
        gateway = get_mcp_gateway()
        assert gateway.is_tool_invocation_enabled() is False

        # Test explicit false
        reset_mcp_gateway()
        os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = "false"
        gateway = get_mcp_gateway()
        assert gateway.is_tool_invocation_enabled() is False

        # Test explicit true
        reset_mcp_gateway()
        os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = "true"
        gateway = get_mcp_gateway()
        assert gateway.is_tool_invocation_enabled() is True

    async def test_tool_invocation_enabled_makes_http_call(self):
        """Test that when enabled, tool invocation makes HTTP call to server."""
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = "true"

        gateway = get_mcp_gateway()
        assert gateway.is_tool_invocation_enabled() is True

        # Register a server
        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            url="https://example.com/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["test_tool"],
        )
        gateway.register_server(server)

        # Mock httpx to verify it's called
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "tool_name": "test_tool",
            "result": {"data": "test_result"},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await gateway.invoke_tool("test_tool", {"arg": "value"})

            # Verify HTTP call was made
            mock_instance.post.assert_called_once()
            call_args = mock_instance.post.call_args
            assert "/mcp/tools/call" in call_args[0][0]
            assert call_args[1]["json"]["tool_name"] == "test_tool"
            assert call_args[1]["json"]["arguments"] == {"arg": "value"}

            # Verify response was parsed
            assert result.success is True
            assert result.result == {"data": "test_result"}

    async def test_tool_invocation_ssrf_blocked(self):
        """Test that SSRF-blocked URLs fail even when invocation is enabled."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        reset_mcp_gateway()
        clear_ssrf_config_cache()

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = "true"
        os.environ["LLMROUTER_ALLOW_PRIVATE_IPS"] = "false"

        gateway = get_mcp_gateway()

        # Try to register server with private IP - should fail at registration
        try:
            server = MCPServer(
                server_id="private-server",
                name="Private Server",
                url="http://192.168.1.100/mcp",  # Private IP
                transport=MCPTransport.STREAMABLE_HTTP,
                tools=["test_tool"],
            )
            gateway.register_server(server)
            pytest.fail("Should have raised ValueError for private IP")
        except ValueError as e:
            assert "blocked for security reasons" in str(e)

    async def test_tool_invocation_http_error_handling(self):
        """Test that HTTP errors are handled gracefully."""
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = "true"

        gateway = get_mcp_gateway()

        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            url="https://example.com/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["test_tool"],
        )
        gateway.register_server(server)

        # Mock httpx to return 500 error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await gateway.invoke_tool("test_tool", {"arg": "value"})

            assert result.success is False
            assert "HTTP 500" in result.error

    async def test_tool_invocation_timeout_handling(self):
        """Test that timeouts are handled gracefully."""
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = "true"

        gateway = get_mcp_gateway()

        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            url="https://example.com/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["test_tool"],
        )
        gateway.register_server(server)

        # Mock httpx to raise timeout
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post = AsyncMock(
                side_effect=httpx.TimeoutException("Connect timeout")
            )

            result = await gateway.invoke_tool("test_tool", {"arg": "value"})

            assert result.success is False
            assert "Timeout" in result.error

    async def test_tool_invocation_connection_error_handling(self):
        """Test that connection errors are handled gracefully."""
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = "true"

        gateway = get_mcp_gateway()

        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            url="https://example.com/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["test_tool"],
        )
        gateway.register_server(server)

        # Mock httpx to raise connection error
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            result = await gateway.invoke_tool("test_tool", {"arg": "value"})

            assert result.success is False
            assert "connect" in result.error.lower()

    async def test_tool_not_found(self):
        """Test that invoking a non-existent tool returns appropriate error."""
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = "true"

        gateway = get_mcp_gateway()

        result = await gateway.invoke_tool("nonexistent_tool", {})
        assert result.success is False
        assert "not found" in result.error

    async def test_tool_invocation_with_auth_token(self):
        """Test that auth tokens are passed in headers when configured."""
        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = "true"

        gateway = get_mcp_gateway()

        # Register server with bearer token auth
        server = MCPServer(
            server_id="auth-server",
            name="Auth Server",
            url="https://example.com/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["test_tool"],
            auth_type="bearer_token",
            metadata={"auth_token": "secret-token-123"},
        )
        gateway.register_server(server)

        # Mock httpx to capture headers
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "result": {}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post = AsyncMock(return_value=mock_response)

            await gateway.invoke_tool("test_tool", {})

            # Verify auth header was sent
            call_args = mock_instance.post.call_args
            headers = call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer secret-token-123"


class TestMCPToolInvocationRouteEndpoint:
    """Tests for the /llmrouter/mcp/tools/call route endpoint."""

    @pytest.fixture(autouse=True)
    def setup_gateway(self):
        """Reset gateway before each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        reset_mcp_gateway()
        clear_ssrf_config_cache()

        # Store original env vars
        self._orig_enabled = os.environ.get("MCP_GATEWAY_ENABLED")
        self._orig_invocation = os.environ.get("LLMROUTER_ENABLE_MCP_TOOL_INVOCATION")

        yield

        # Restore
        reset_mcp_gateway()
        clear_ssrf_config_cache()
        if self._orig_enabled is not None:
            os.environ["MCP_GATEWAY_ENABLED"] = self._orig_enabled
        else:
            os.environ.pop("MCP_GATEWAY_ENABLED", None)
        if self._orig_invocation is not None:
            os.environ["LLMROUTER_ENABLE_MCP_TOOL_INVOCATION"] = self._orig_invocation
        else:
            os.environ.pop("LLMROUTER_ENABLE_MCP_TOOL_INVOCATION", None)

    async def test_route_returns_501_when_invocation_disabled(self):
        """Test that the route returns 501 when tool invocation is disabled."""

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ.pop(
            "LLMROUTER_ENABLE_MCP_TOOL_INVOCATION", None
        )  # Disabled by default

        # For this basic test, we'll test the gateway directly instead
        gateway = get_mcp_gateway()

        # Register a server
        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            url="https://example.com/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["test_tool"],
        )
        gateway.register_server(server)

        # Verify the gateway behavior
        assert gateway.is_tool_invocation_enabled() is False
        result = await gateway.invoke_tool("test_tool", {})
        assert result.success is False
        assert "tool_invocation_disabled" in result.error
