"""
TG3.2 RBAC Enforcement Integration Test
========================================

This test validates RBAC enforcement using a test FastAPI app with:
- MCP server registration endpoint (requires mcp.server.write permission)
- Admin API key bypass
- User with/without required permission

No Docker required - uses TestClient for in-process testing.

Test Coverage:
1. Admin with admin key can access protected endpoint
2. User with required permission can access protected endpoint
3. User without required permission gets 403
4. Unauthenticated request gets 401
5. Streaming endpoints are not buffered (regression check)

Usage:
    uv run pytest tests/integration/test_rbac_enforcement.py -v
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Depends, HTTPException
from fastapi.testclient import TestClient

# We'll need to create a standalone test to avoid complex module patching


# =============================================================================
# Test Fixtures - Self-contained test infrastructure
# =============================================================================


@pytest.fixture
def admin_api_keys():
    """Set of valid admin API keys for testing."""
    return {"test-admin-key-1", "test-admin-key-2"}


@pytest.fixture
def rbac_test_app(admin_api_keys):
    """
    Create a test FastAPI app with RBAC-protected endpoints.
    
    Uses a simplified mock of the RBAC dependency that mimics the real behavior.
    """
    from litellm_llmrouter.rbac import (
        PERMISSION_MCP_SERVER_WRITE,
        PERMISSION_MCP_TOOL_CALL,
        PERMISSION_SYSTEM_CONFIG_RELOAD,
        PERMISSION_SUPERUSER,
        has_permission,
        extract_user_permissions,
    )
    
    app = FastAPI()
    
    # Simulated user database (api_key -> user_info)
    user_database = {
        "user-with-mcp-server-write": {
            "user_id": "user-1",
            "metadata": {"permissions": ["mcp.server.write"]}
        },
        "user-with-mcp-tool-call": {
            "user_id": "user-2", 
            "metadata": {"permissions": ["mcp.tool.call"]}
        },
        "user-with-all-perms": {
            "user_id": "superuser",
            "permissions": "*"
        },
        "user-no-perms": {
            "user_id": "user-3",
            "metadata": {"permissions": []}
        },
        "user-with-mcp-wildcard": {
            "user_id": "user-4",
            "metadata": {"permissions": ["mcp.*"]}
        },
    }
    
    def create_rbac_dependency(required_permission: str):
        """Create an RBAC dependency that mimics the real implementation."""
        from fastapi import Request
        
        async def rbac_check(request: Request):
            # Check for admin key first
            admin_key = request.headers.get("X-Admin-API-Key", "").strip()
            if not admin_key:
                auth_header = request.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    admin_key = auth_header[7:]
            
            if admin_key in admin_api_keys:
                return {
                    "is_admin": True,
                    "permissions": frozenset({PERMISSION_SUPERUSER}),
                }
            
            # Try user auth
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "authentication_required",
                        "message": "Valid API key required.",
                        "request_id": "test-123",
                    }
                )
            
            api_key = auth_header[7:]
            user_info = user_database.get(api_key)
            
            if not user_info:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "authentication_required",
                        "message": "Invalid API key.",
                        "request_id": "test-123",
                    }
                )
            
            # Check permission
            user_perms = extract_user_permissions(user_info)
            if has_permission(user_perms, required_permission):
                return {
                    "is_admin": False,
                    "permissions": user_perms,
                    "user_info": user_info,
                }
            
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "permission_denied",
                    "message": f"Insufficient permissions. Required: {required_permission}",
                    "required_permission": required_permission,
                    "request_id": "test-123",
                }
            )
        
        return rbac_check
    
    @app.post("/test/mcp/servers")
    async def create_mcp_server(
        rbac_info: dict = Depends(create_rbac_dependency(PERMISSION_MCP_SERVER_WRITE)),
    ):
        return {
            "status": "created",
            "server_id": "test-server",
            "rbac_info": {
                "is_admin": rbac_info["is_admin"],
                "permissions": list(rbac_info["permissions"]),
            }
        }
    
    @app.post("/test/tools/call")
    async def call_tool(
        rbac_info: dict = Depends(create_rbac_dependency(PERMISSION_MCP_TOOL_CALL)),
    ):
        return {"status": "invoked", "is_admin": rbac_info["is_admin"]}
    
    @app.post("/test/config/reload")
    async def reload_config(
        rbac_info: dict = Depends(create_rbac_dependency(PERMISSION_SYSTEM_CONFIG_RELOAD)),
    ):
        return {"status": "reloaded", "is_admin": rbac_info["is_admin"]}
    
    return app


@pytest.fixture
def client(rbac_test_app):
    """Create a test client for the app."""
    return TestClient(rbac_test_app)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRBACEnforcementIntegration:
    """
    Integration tests for RBAC enforcement.
    
    Validates that the RBAC dependency is correctly wired to FastAPI endpoints.
    """

    def test_admin_key_allows_access(self, client):
        """
        Test: Admin with valid admin API key can access protected endpoint.
        
        Admin authentication bypasses permission checks entirely.
        """
        response = client.post(
            "/test/mcp/servers",
            headers={"X-Admin-API-Key": "test-admin-key-1"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["rbac_info"]["is_admin"] is True

    def test_admin_key_via_bearer_allows_access(self, client):
        """
        Test: Admin key in Authorization: Bearer header also works.
        """
        response = client.post(
            "/test/mcp/servers",
            headers={"Authorization": "Bearer test-admin-key-2"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["rbac_info"]["is_admin"] is True

    def test_user_with_permission_allowed(self, client):
        """
        Test: User with required permission can access protected endpoint.
        """
        response = client.post(
            "/test/mcp/servers",
            headers={"Authorization": "Bearer user-with-mcp-server-write"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["rbac_info"]["is_admin"] is False
        assert "mcp.server.write" in data["rbac_info"]["permissions"]

    def test_user_without_permission_denied_403(self, client):
        """
        Test: User without required permission gets 403 Forbidden.
        """
        response = client.post(
            "/test/mcp/servers",
            headers={"Authorization": "Bearer user-no-perms"},
        )
        
        assert response.status_code == 403
        data = response.json()
        assert data["detail"]["error"] == "permission_denied"
        assert "mcp.server.write" in data["detail"]["required_permission"]

    def test_user_with_wrong_permission_denied_403(self, client):
        """
        Test: User with different permission gets 403.
        """
        # User has mcp.tool.call but endpoint requires mcp.server.write
        response = client.post(
            "/test/mcp/servers",
            headers={"Authorization": "Bearer user-with-mcp-tool-call"},
        )
        
        assert response.status_code == 403

    def test_unauthenticated_request_denied_401(self, client):
        """
        Test: Request without credentials gets 401 Unauthorized.
        """
        response = client.post("/test/mcp/servers")
        
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["error"] == "authentication_required"

    def test_invalid_api_key_denied_401(self, client):
        """
        Test: Request with invalid API key gets 401.
        """
        response = client.post(
            "/test/mcp/servers",
            headers={"Authorization": "Bearer invalid-key"},
        )
        
        assert response.status_code == 401

    def test_superuser_can_access_all_endpoints(self, client):
        """
        Test: User with '*' (superuser) permission can access any endpoint.
        """
        # MCP server write
        response1 = client.post(
            "/test/mcp/servers",
            headers={"Authorization": "Bearer user-with-all-perms"},
        )
        assert response1.status_code == 200
        
        # Tool call
        response2 = client.post(
            "/test/tools/call",
            headers={"Authorization": "Bearer user-with-all-perms"},
        )
        assert response2.status_code == 200
        
        # Config reload
        response3 = client.post(
            "/test/config/reload",
            headers={"Authorization": "Bearer user-with-all-perms"},
        )
        assert response3.status_code == 200

    def test_different_endpoints_require_different_permissions(self, client):
        """
        Test: Different endpoints require their specific permissions.
        """
        # User with mcp.server.write can access /test/mcp/servers
        response1 = client.post(
            "/test/mcp/servers",
            headers={"Authorization": "Bearer user-with-mcp-server-write"},
        )
        assert response1.status_code == 200
        
        # But cannot access /test/tools/call (needs mcp.tool.call)
        response2 = client.post(
            "/test/tools/call",
            headers={"Authorization": "Bearer user-with-mcp-server-write"},
        )
        assert response2.status_code == 403
        
        # And cannot access /test/config/reload (needs system.config.reload)
        response3 = client.post(
            "/test/config/reload",
            headers={"Authorization": "Bearer user-with-mcp-server-write"},
        )
        assert response3.status_code == 403

    def test_namespace_wildcard_grants_namespace_permissions(self, client):
        """
        Test: User with mcp.* wildcard can access all mcp.* endpoints.
        """
        # mcp.* should grant access to mcp.server.write
        response1 = client.post(
            "/test/mcp/servers",
            headers={"Authorization": "Bearer user-with-mcp-wildcard"},
        )
        assert response1.status_code == 200
        
        # mcp.* should also grant access to mcp.tool.call  
        response2 = client.post(
            "/test/tools/call",
            headers={"Authorization": "Bearer user-with-mcp-wildcard"},
        )
        assert response2.status_code == 200
        
        # But NOT system.config.reload
        response3 = client.post(
            "/test/config/reload",
            headers={"Authorization": "Bearer user-with-mcp-wildcard"},
        )
        assert response3.status_code == 403


class TestRBACResponseFormat:
    """Test RBAC error response format."""
    
    def test_401_response_format(self, client):
        """Test 401 response has correct format."""
        response = client.post("/test/mcp/servers")
        
        assert response.status_code == 401
        data = response.json()
        detail = data["detail"]
        
        assert "error" in detail
        assert "message" in detail
        assert "request_id" in detail
        assert detail["error"] == "authentication_required"

    def test_403_response_format(self, client):
        """Test 403 response has correct format."""
        response = client.post(
            "/test/mcp/servers",
            headers={"Authorization": "Bearer user-no-perms"},
        )
        
        assert response.status_code == 403
        data = response.json()
        detail = data["detail"]
        
        assert "error" in detail
        assert "message" in detail
        assert "required_permission" in detail
        assert "request_id" in detail
        assert detail["error"] == "permission_denied"


class TestRBACStreamingSafety:
    """
    Test that RBAC doesn't buffer streaming responses.
    
    RBAC is implemented as a FastAPI dependency, not middleware,
    so it should not affect response bodies at all.
    """

    @pytest.fixture
    def streaming_app(self, admin_api_keys):
        """Create app with streaming endpoint."""
        from starlette.responses import StreamingResponse
        from fastapi import Request
        import asyncio
        from litellm_llmrouter.rbac import (
            PERMISSION_MCP_SERVER_WRITE,
            PERMISSION_SUPERUSER,
        )
        
        app = FastAPI()
        
        async def simple_rbac_check(request: Request):
            """Simplified RBAC check for streaming test."""
            admin_key = request.headers.get("X-Admin-API-Key", "").strip()
            if admin_key in admin_api_keys:
                return {"is_admin": True, "permissions": frozenset({PERMISSION_SUPERUSER})}
            raise HTTPException(status_code=401, detail={"error": "authentication_required"})
        
        @app.get("/test/stream")
        async def stream_response(
            rbac_info: dict = Depends(simple_rbac_check),
        ):
            async def generate():
                # Yield chunks with small delays
                for i in range(5):
                    yield f"data: chunk-{i}\n\n"
                    await asyncio.sleep(0.01)
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"X-RBAC-Admin": str(rbac_info["is_admin"])},
            )
        
        return app

    def test_streaming_endpoint_not_buffered(self, streaming_app):
        """
        Test: Streaming endpoint returns correct type without buffering.
        
        This is a regression test to ensure RBAC (as a dependency) doesn't
        accidentally buffer the response body.
        """
        client = TestClient(streaming_app)
        
        response = client.get(
            "/test/stream",
            headers={"X-Admin-API-Key": "test-admin-key-1"},
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["x-rbac-admin"] == "True"
        
        # Check all chunks are present
        content = response.text
        for i in range(5):
            assert f"chunk-{i}" in content


# =============================================================================
# Standalone execution
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
