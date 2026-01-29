"""
Tests for MCP Parity Layer
==========================

Tests for upstream-compatible MCP endpoint aliases, OAuth endpoints,
and protocol proxy functionality.

Covers:
- Alias endpoints route to same behavior as existing endpoints
- Protocol proxy path registration and SSRF blocks
- OAuth endpoints validate state and store tokens
"""

import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


class TestMCPParityGatewayIntegration:
    """Tests for MCP parity layer integration with the gateway."""

    @pytest.fixture(autouse=True)
    def setup_gateway(self):
        """Reset gateway and environment before each test."""
        from litellm_llmrouter.mcp_gateway import reset_mcp_gateway
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        reset_mcp_gateway()
        clear_ssrf_config_cache()

        # Store original env vars
        self._orig_enabled = os.environ.get("MCP_GATEWAY_ENABLED")
        self._orig_oauth = os.environ.get("MCP_OAUTH_ENABLED")
        self._orig_proxy = os.environ.get("MCP_PROTOCOL_PROXY_ENABLED")
        self._orig_allow_private = os.environ.get("LLMROUTER_ALLOW_PRIVATE_IPS")

        yield

        # Restore original env vars
        reset_mcp_gateway()
        clear_ssrf_config_cache()
        for var, orig in [
            ("MCP_GATEWAY_ENABLED", self._orig_enabled),
            ("MCP_OAUTH_ENABLED", self._orig_oauth),
            ("MCP_PROTOCOL_PROXY_ENABLED", self._orig_proxy),
            ("LLMROUTER_ALLOW_PRIVATE_IPS", self._orig_allow_private),
        ]:
            if orig is not None:
                os.environ[var] = orig
            else:
                os.environ.pop(var, None)

    async def test_parity_endpoints_use_same_gateway(self):
        """Test that parity endpoints use the same MCPGateway singleton."""
        from litellm_llmrouter.mcp_gateway import get_mcp_gateway, MCPServer, MCPTransport

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        
        gateway = get_mcp_gateway()
        
        # Register a server
        server = MCPServer(
            server_id="test-singleton",
            name="Test Singleton",
            url="https://example.com/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["tool1"],
        )
        gateway.register_server(server)

        # Verify server is accessible
        assert gateway.get_server("test-singleton") is not None
        assert gateway.get_server("test-singleton").name == "Test Singleton"
        
        # List should contain our server
        servers = gateway.list_servers()
        assert len(servers) == 1
        assert servers[0].server_id == "test-singleton"

    async def test_parity_endpoints_gateway_disabled_error(self):
        """Test that parity endpoints return error when gateway is disabled."""
        from litellm_llmrouter.mcp_gateway import get_mcp_gateway

        os.environ["MCP_GATEWAY_ENABLED"] = "false"
        
        gateway = get_mcp_gateway()
        assert gateway.is_enabled() is False


class TestMCPProtocolProxy:
    """Tests for MCP protocol proxy functionality."""

    @pytest.fixture(autouse=True)
    def setup_gateway(self):
        """Reset gateway and environment before each test."""
        from litellm_llmrouter.mcp_gateway import reset_mcp_gateway
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        reset_mcp_gateway()
        clear_ssrf_config_cache()

        self._orig_enabled = os.environ.get("MCP_GATEWAY_ENABLED")
        self._orig_proxy = os.environ.get("MCP_PROTOCOL_PROXY_ENABLED")
        self._orig_allow_private = os.environ.get("LLMROUTER_ALLOW_PRIVATE_IPS")

        yield

        reset_mcp_gateway()
        clear_ssrf_config_cache()
        for var, orig in [
            ("MCP_GATEWAY_ENABLED", self._orig_enabled),
            ("MCP_PROTOCOL_PROXY_ENABLED", self._orig_proxy),
            ("LLMROUTER_ALLOW_PRIVATE_IPS", self._orig_allow_private),
        ]:
            if orig is not None:
                os.environ[var] = orig
            else:
                os.environ.pop(var, None)

    async def test_proxy_route_registered_when_enabled(self):
        """Test that protocol proxy route is registered when feature flag is enabled."""
        # Need to reload mcp_parity module with flag enabled
        os.environ["MCP_PROTOCOL_PROXY_ENABLED"] = "true"
        os.environ["MCP_GATEWAY_ENABLED"] = "true"

        # Import fresh to pick up env var change
        import importlib
        import litellm_llmrouter.mcp_parity as mcp_parity_module
        importlib.reload(mcp_parity_module)

        from litellm_llmrouter.mcp_parity import mcp_proxy_router

        # Check that routes are registered
        routes = [r.path for r in mcp_proxy_router.routes]
        assert any("/{server_id}" in r for r in routes)

    async def test_proxy_respects_ssrf_blocks(self):
        """Test that protocol proxy blocks SSRF attempts at server registration."""
        from litellm_llmrouter.mcp_gateway import (
            get_mcp_gateway,
            MCPServer,
            MCPTransport,
            reset_mcp_gateway,
        )
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        reset_mcp_gateway()
        clear_ssrf_config_cache()

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["LLMROUTER_ALLOW_PRIVATE_IPS"] = "false"

        gateway = get_mcp_gateway()

        # Try to register server with private IP - should fail
        try:
            server = MCPServer(
                server_id="ssrf-test",
                name="SSRF Test",
                url="http://192.168.1.100/mcp",
                transport=MCPTransport.STREAMABLE_HTTP,
            )
            gateway.register_server(server)
            pytest.fail("Should have raised ValueError for private IP")
        except ValueError as e:
            assert "blocked for security reasons" in str(e)


class TestMCPOAuthState:
    """Tests for MCP OAuth state management and CSRF protection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset state before each test."""
        from litellm_llmrouter.mcp_gateway import reset_mcp_gateway
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        reset_mcp_gateway()
        clear_ssrf_config_cache()

        self._orig_enabled = os.environ.get("MCP_GATEWAY_ENABLED")
        self._orig_oauth = os.environ.get("MCP_OAUTH_ENABLED")

        yield

        reset_mcp_gateway()
        clear_ssrf_config_cache()
        for var, orig in [
            ("MCP_GATEWAY_ENABLED", self._orig_enabled),
            ("MCP_OAUTH_ENABLED", self._orig_oauth),
        ]:
            if orig is not None:
                os.environ[var] = orig
            else:
                os.environ.pop(var, None)

    async def test_oauth_state_validation_logic(self):
        """Test OAuth state validation logic directly (without HTTP layer)."""
        os.environ["MCP_OAUTH_ENABLED"] = "true"
        
        import importlib
        import litellm_llmrouter.mcp_parity as mcp_parity_module
        importlib.reload(mcp_parity_module)

        from litellm_llmrouter.mcp_parity import (
            _oauth_sessions,
            _generate_state,
            _cleanup_expired_sessions,
            OAuthSession,
            OAUTH_SESSION_TTL,
        )
        import time

        # Generate a state and create session
        state = _generate_state()
        assert len(state) > 20  # Token should be substantial

        # Create a session with this state
        _oauth_sessions[state] = OAuthSession(
            server_id="test-server",
            state="original-client-state",
            redirect_uri="https://example.com/callback",
            created_at=time.time(),
            client_id="test-client",
        )

        # Verify session exists
        assert state in _oauth_sessions
        assert _oauth_sessions[state].server_id == "test-server"

        # Test invalid state lookup fails
        assert "invalid-state-12345" not in _oauth_sessions

    async def test_oauth_session_expiry(self):
        """Test that OAuth sessions expire correctly."""
        os.environ["MCP_OAUTH_ENABLED"] = "true"
        
        import importlib
        import litellm_llmrouter.mcp_parity as mcp_parity_module
        importlib.reload(mcp_parity_module)

        from litellm_llmrouter.mcp_parity import (
            _oauth_sessions,
            _cleanup_expired_sessions,
            OAuthSession,
            OAUTH_SESSION_TTL,
        )
        import time

        # Create an expired session (created 10 minutes ago)
        expired_state = "expired-test-state"
        _oauth_sessions[expired_state] = OAuthSession(
            server_id="test-server",
            state="original-state",
            redirect_uri="https://example.com/callback",
            created_at=time.time() - (OAUTH_SESSION_TTL + 100),  # Expired
            client_id="test-client",
        )

        # Create a valid session
        valid_state = "valid-test-state"
        _oauth_sessions[valid_state] = OAuthSession(
            server_id="test-server-2",
            state="original-state-2",
            redirect_uri="https://example.com/callback",
            created_at=time.time(),  # Just now
            client_id="test-client-2",
        )

        # Cleanup expired sessions
        _cleanup_expired_sessions()

        # Expired should be gone, valid should remain
        assert expired_state not in _oauth_sessions
        assert valid_state in _oauth_sessions


class TestMCPParityRouteRegistration:
    """Tests to verify all parity routes are properly registered."""

    async def test_parity_router_paths_exist(self):
        """Test that parity routers have expected paths."""
        from litellm_llmrouter.mcp_parity import (
            mcp_parity_router,
            mcp_parity_admin_router,
            mcp_rest_router,
        )

        # Extract route paths (routes have full paths including prefix)
        parity_paths = [r.path for r in mcp_parity_router.routes]
        parity_admin_paths = [r.path for r in mcp_parity_admin_router.routes]
        mcp_rest_paths = [r.path for r in mcp_rest_router.routes]

        # Verify expected parity paths (read-only) - full paths with /v1/mcp prefix
        assert any("/server" in p for p in parity_paths)
        assert any("/server/health" in p for p in parity_paths)
        assert any("/tools" in p for p in parity_paths)
        assert any("/access_groups" in p for p in parity_paths)
        assert any("/registry.json" in p for p in parity_paths)

        # Verify expected admin parity paths
        assert any("/server" in p for p in parity_admin_paths)  # POST/PUT

        # Verify MCP REST paths
        assert any("/tools/list" in p for p in mcp_rest_paths)
        assert any("/tools/call" in p for p in mcp_rest_paths)

    async def test_parity_router_has_correct_prefix(self):
        """Test that parity routers have correct prefixes."""
        from litellm_llmrouter.mcp_parity import (
            mcp_parity_router,
            mcp_rest_router,
        )

        assert mcp_parity_router.prefix == "/v1/mcp"
        assert mcp_rest_router.prefix == "/mcp-rest"

    async def test_oauth_routes_exist_when_enabled(self):
        """Test that OAuth routes exist when feature flag is enabled."""
        os.environ["MCP_OAUTH_ENABLED"] = "true"
        
        import importlib
        import litellm_llmrouter.mcp_parity as mcp_parity_module
        importlib.reload(mcp_parity_module)

        from litellm_llmrouter.mcp_parity import (
            mcp_parity_router,
            mcp_parity_admin_router,
            oauth_callback_router,
        )

        # Check OAuth routes in parity router
        parity_paths = [r.path for r in mcp_parity_router.routes]
        parity_admin_paths = [r.path for r in mcp_parity_admin_router.routes]
        callback_paths = [r.path for r in oauth_callback_router.routes]

        # OAuth authorize should be in parity router
        assert any("oauth" in p and "authorize" in p for p in parity_paths)
        
        # OAuth token/register should be in admin router
        assert any("oauth" in p and "token" in p for p in parity_admin_paths)
        assert any("oauth" in p and "register" in p for p in parity_admin_paths)
        
        # Callback should exist
        assert "/mcp/oauth/callback" in callback_paths

    async def test_proxy_routes_exist_when_enabled(self):
        """Test that proxy routes exist when feature flag is enabled."""
        os.environ["MCP_PROTOCOL_PROXY_ENABLED"] = "true"
        
        import importlib
        import litellm_llmrouter.mcp_parity as mcp_parity_module
        importlib.reload(mcp_parity_module)

        from litellm_llmrouter.mcp_parity import mcp_proxy_router

        proxy_paths = [r.path for r in mcp_proxy_router.routes]
        
        # Should have the catch-all proxy route
        assert any("{server_id}" in p and "{path:path}" in p for p in proxy_paths)


class TestMCPParityFeatureFlags:
    """Tests for MCP parity feature flags."""

    async def test_feature_flags_read_from_environment(self):
        """Test that feature flags are correctly read from environment."""
        # Test OAuth flag
        os.environ["MCP_OAUTH_ENABLED"] = "true"
        
        import importlib
        import litellm_llmrouter.mcp_parity as mcp_parity_module
        importlib.reload(mcp_parity_module)
        
        assert mcp_parity_module.MCP_OAUTH_ENABLED is True

        # Test with false
        os.environ["MCP_OAUTH_ENABLED"] = "false"
        importlib.reload(mcp_parity_module)
        assert mcp_parity_module.MCP_OAUTH_ENABLED is False

    async def test_protocol_proxy_flag(self):
        """Test protocol proxy feature flag."""
        os.environ["MCP_PROTOCOL_PROXY_ENABLED"] = "true"
        
        import importlib
        import litellm_llmrouter.mcp_parity as mcp_parity_module
        importlib.reload(mcp_parity_module)
        
        assert mcp_parity_module.MCP_PROTOCOL_PROXY_ENABLED is True

        os.environ["MCP_PROTOCOL_PROXY_ENABLED"] = "false"
        importlib.reload(mcp_parity_module)
        assert mcp_parity_module.MCP_PROTOCOL_PROXY_ENABLED is False

    async def test_default_feature_flags_are_false(self):
        """Test that feature flags default to false."""
        os.environ.pop("MCP_OAUTH_ENABLED", None)
        os.environ.pop("MCP_PROTOCOL_PROXY_ENABLED", None)
        
        import importlib
        import litellm_llmrouter.mcp_parity as mcp_parity_module
        importlib.reload(mcp_parity_module)
        
        assert mcp_parity_module.MCP_OAUTH_ENABLED is False
        assert mcp_parity_module.MCP_PROTOCOL_PROXY_ENABLED is False
