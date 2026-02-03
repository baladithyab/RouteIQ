"""
Unit tests for RBAC (Role-Based Access Control) module.

Tests cover:
1. Permission normalization from various input formats
2. Permission checking including wildcards
3. User permission extraction from metadata
4. RBAC dependency function behavior
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm_llmrouter.rbac import (
    # Permission constants
    PERMISSION_MCP_SERVER_WRITE,
    PERMISSION_MCP_TOOL_WRITE,
    PERMISSION_MCP_TOOL_CALL,
    PERMISSION_SYSTEM_CONFIG_RELOAD,
    PERMISSION_A2A_AGENT_WRITE,
    PERMISSION_SUPERUSER,
    ALL_PERMISSIONS,
    # Functions
    normalize_permissions,
    extract_user_permissions,
    has_permission,
    requires_permission,
)


# =============================================================================
# Permission Normalization Tests
# =============================================================================


class TestNormalizePermissions:
    """Tests for normalize_permissions function."""

    def test_none_returns_empty_set(self):
        """None input returns empty frozenset."""
        result = normalize_permissions(None)
        assert result == frozenset()

    def test_empty_string_returns_empty_set(self):
        """Empty string returns empty frozenset."""
        result = normalize_permissions("")
        assert result == frozenset()

    def test_single_permission_string(self):
        """Single permission string is normalized."""
        result = normalize_permissions("mcp.server.write")
        assert result == frozenset({"mcp.server.write"})

    def test_comma_separated_string(self):
        """Comma-separated string is split and normalized."""
        result = normalize_permissions("mcp.server.write, mcp.tool.write, a2a.agent.write")
        assert result == frozenset({
            "mcp.server.write",
            "mcp.tool.write",
            "a2a.agent.write",
        })

    def test_list_of_permissions(self):
        """List of strings is normalized."""
        result = normalize_permissions(["mcp.server.write", "mcp.tool.write"])
        assert result == frozenset({"mcp.server.write", "mcp.tool.write"})

    def test_set_of_permissions(self):
        """Set of strings is normalized."""
        result = normalize_permissions({"mcp.server.write", "mcp.tool.write"})
        assert result == frozenset({"mcp.server.write", "mcp.tool.write"})

    def test_tuple_of_permissions(self):
        """Tuple of strings is normalized."""
        result = normalize_permissions(("mcp.server.write", "mcp.tool.write"))
        assert result == frozenset({"mcp.server.write", "mcp.tool.write"})

    def test_uppercase_normalized_to_lowercase(self):
        """Uppercase permissions are normalized to lowercase."""
        result = normalize_permissions("MCP.SERVER.WRITE")
        assert result == frozenset({"mcp.server.write"})

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped."""
        result = normalize_permissions("  mcp.server.write  ")
        assert result == frozenset({"mcp.server.write"})

    def test_empty_items_excluded(self):
        """Empty items after split are excluded."""
        result = normalize_permissions("mcp.server.write,,mcp.tool.write,")
        assert result == frozenset({"mcp.server.write", "mcp.tool.write"})

    def test_unsupported_type_returns_empty(self):
        """Unsupported types return empty frozenset with warning."""
        result = normalize_permissions(12345)
        assert result == frozenset()

    def test_mixed_valid_invalid_in_list(self):
        """List with non-string items - only strings are included."""
        result = normalize_permissions(["mcp.server.write", 123, None, "mcp.tool.write"])
        assert result == frozenset({"mcp.server.write", "mcp.tool.write"})


# =============================================================================
# User Permission Extraction Tests
# =============================================================================


class TestExtractUserPermissions:
    """Tests for extract_user_permissions function."""

    def test_direct_permissions_field(self):
        """Permissions from direct 'permissions' field."""
        user_info = {"permissions": ["mcp.server.write", "mcp.tool.write"]}
        result = extract_user_permissions(user_info)
        assert result == frozenset({"mcp.server.write", "mcp.tool.write"})

    def test_nested_metadata_permissions(self):
        """Permissions from metadata.permissions field."""
        user_info = {
            "metadata": {
                "permissions": ["mcp.server.write", "system.config.reload"]
            }
        }
        result = extract_user_permissions(user_info)
        assert result == frozenset({"mcp.server.write", "system.config.reload"})

    def test_direct_takes_precedence_over_metadata(self):
        """Direct permissions field takes precedence over metadata."""
        user_info = {
            "permissions": ["direct.permission"],
            "metadata": {
                "permissions": ["metadata.permission"]
            }
        }
        result = extract_user_permissions(user_info)
        assert result == frozenset({"direct.permission"})

    def test_empty_user_info(self):
        """Empty user info returns empty set."""
        result = extract_user_permissions({})
        assert result == frozenset()

    def test_no_permissions_field(self):
        """User info without permissions returns empty set."""
        user_info = {"user_id": "123", "name": "Test User"}
        result = extract_user_permissions(user_info)
        assert result == frozenset()

    def test_metadata_not_dict(self):
        """Non-dict metadata is handled gracefully."""
        user_info = {"metadata": "not-a-dict"}
        result = extract_user_permissions(user_info)
        assert result == frozenset()

    def test_comma_separated_in_metadata(self):
        """Comma-separated string in metadata is normalized."""
        user_info = {
            "metadata": {
                "permissions": "mcp.server.write, mcp.tool.write"
            }
        }
        result = extract_user_permissions(user_info)
        assert result == frozenset({"mcp.server.write", "mcp.tool.write"})


# =============================================================================
# Permission Checking Tests
# =============================================================================


class TestHasPermission:
    """Tests for has_permission function."""

    def test_exact_match(self):
        """Exact permission match."""
        perms = frozenset({"mcp.server.write"})
        assert has_permission(perms, "mcp.server.write") is True

    def test_exact_match_case_insensitive(self):
        """Exact match is case-insensitive."""
        perms = frozenset({"mcp.server.write"})
        assert has_permission(perms, "MCP.SERVER.WRITE") is True

    def test_no_match(self):
        """No matching permission returns False."""
        perms = frozenset({"mcp.server.write"})
        assert has_permission(perms, "mcp.tool.write") is False

    def test_superuser_wildcard(self):
        """Superuser '*' grants all permissions."""
        perms = frozenset({"*"})
        assert has_permission(perms, "mcp.server.write") is True
        assert has_permission(perms, "any.permission.here") is True

    def test_namespace_wildcard(self):
        """Namespace wildcard 'mcp.*' grants all mcp.* permissions."""
        perms = frozenset({"mcp.*"})
        assert has_permission(perms, "mcp.server.write") is True
        assert has_permission(perms, "mcp.tool.write") is True
        assert has_permission(perms, "mcp.tool.call") is True
        assert has_permission(perms, "system.config.reload") is False
        assert has_permission(perms, "a2a.agent.write") is False

    def test_empty_permissions(self):
        """Empty permission set returns False."""
        perms = frozenset()
        assert has_permission(perms, "mcp.server.write") is False

    def test_multiple_permissions(self):
        """Multiple permissions, one matches."""
        perms = frozenset({"mcp.server.write", "mcp.tool.write", "system.config.reload"})
        assert has_permission(perms, "mcp.tool.write") is True

    def test_all_standard_permissions_defined(self):
        """All standard permissions are in ALL_PERMISSIONS."""
        assert PERMISSION_MCP_SERVER_WRITE in ALL_PERMISSIONS
        assert PERMISSION_MCP_TOOL_WRITE in ALL_PERMISSIONS
        assert PERMISSION_MCP_TOOL_CALL in ALL_PERMISSIONS
        assert PERMISSION_SYSTEM_CONFIG_RELOAD in ALL_PERMISSIONS
        assert PERMISSION_A2A_AGENT_WRITE in ALL_PERMISSIONS


# =============================================================================
# RBAC Dependency Tests
# =============================================================================


class TestRequiresPermission:
    """Tests for requires_permission dependency factory."""

    @pytest.mark.asyncio
    async def test_admin_auth_bypasses_permission_check(self):
        """Admin authentication bypasses permission check."""
        # Mock admin auth to succeed
        with patch("litellm_llmrouter.rbac._is_admin_auth_enabled", return_value=True), \
             patch("litellm_llmrouter.rbac._load_admin_api_keys", return_value={"test-admin-key"}), \
             patch("litellm_llmrouter.rbac.get_request_id", return_value="test-req-id"):
            
            # Create mock request with admin key
            mock_request = MagicMock()
            mock_request.headers = {"X-Admin-API-Key": "test-admin-key"}
            mock_request.url.path = "/test/endpoint"
            
            # Get the dependency function
            dep_func = requires_permission(PERMISSION_MCP_SERVER_WRITE)
            
            # Call it
            result = await dep_func(mock_request)
            
            # Admin should be allowed
            assert result["is_admin"] is True
            assert PERMISSION_SUPERUSER in result["permissions"]

    @pytest.mark.asyncio
    async def test_user_with_permission_allowed(self):
        """User with required permission is allowed."""
        # Mock admin auth to fail, user auth to succeed with permission
        with patch("litellm_llmrouter.rbac._is_admin_auth_enabled", return_value=True), \
             patch("litellm_llmrouter.rbac._load_admin_api_keys", return_value={"test-admin-key"}), \
             patch("litellm_llmrouter.rbac.get_request_id", return_value="test-req-id"):
            
            # Create mock request without admin key
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": "Bearer user-api-key"}
            mock_request.url.path = "/test/endpoint"
            
            # Mock user_api_key_auth to return user with permission
            mock_user_info = {
                "user_id": "test-user",
                "metadata": {
                    "permissions": ["mcp.server.write"]
                }
            }
            
            # Patch at the import location inside _try_user_auth
            with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", 
                       new_callable=AsyncMock, return_value=mock_user_info):
                # Get the dependency function
                dep_func = requires_permission(PERMISSION_MCP_SERVER_WRITE)
                
                # Call it
                result = await dep_func(mock_request)
                
                # User should be allowed with their permissions
                assert result["is_admin"] is False
                assert "mcp.server.write" in result["permissions"]

    @pytest.mark.asyncio
    async def test_user_without_permission_denied_403(self):
        """User without required permission gets 403."""
        with patch("litellm_llmrouter.rbac._is_admin_auth_enabled", return_value=True), \
             patch("litellm_llmrouter.rbac._load_admin_api_keys", return_value={"test-admin-key"}), \
             patch("litellm_llmrouter.rbac.get_request_id", return_value="test-req-id"):
            
            # Create mock request without admin key
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": "Bearer user-api-key"}
            mock_request.url.path = "/test/endpoint"
            
            # Mock user_api_key_auth to return user WITHOUT permission
            mock_user_info = {
                "user_id": "test-user",
                "metadata": {
                    "permissions": ["some.other.permission"]
                }
            }
            
            with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", 
                       new_callable=AsyncMock, return_value=mock_user_info):
                dep_func = requires_permission(PERMISSION_MCP_SERVER_WRITE)
                
                with pytest.raises(HTTPException) as exc_info:
                    await dep_func(mock_request)
                
                assert exc_info.value.status_code == 403
                assert exc_info.value.detail["error"] == "permission_denied"
                assert "mcp.server.write" in exc_info.value.detail["required_permission"]

    @pytest.mark.asyncio
    async def test_unauthenticated_request_denied_401(self):
        """Unauthenticated request gets 401."""
        with patch("litellm_llmrouter.rbac._is_admin_auth_enabled", return_value=True), \
             patch("litellm_llmrouter.rbac._load_admin_api_keys", return_value={"test-admin-key"}), \
             patch("litellm_llmrouter.rbac.get_request_id", return_value="test-req-id"):
            
            # Create mock request with no credentials
            mock_request = MagicMock()
            mock_request.headers = {}
            mock_request.url.path = "/test/endpoint"
            
            # Mock user_api_key_auth to raise HTTPException (auth failed)
            auth_exc = HTTPException(status_code=401, detail="Invalid API key")
            
            with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", 
                       new_callable=AsyncMock, side_effect=auth_exc):
                dep_func = requires_permission(PERMISSION_MCP_SERVER_WRITE)
                
                with pytest.raises(HTTPException) as exc_info:
                    await dep_func(mock_request)
                
                assert exc_info.value.status_code == 401
                assert exc_info.value.detail["error"] == "authentication_required"

    @pytest.mark.asyncio
    async def test_admin_via_bearer_token(self):
        """Admin authentication via Bearer token in Authorization header."""
        with patch("litellm_llmrouter.rbac._is_admin_auth_enabled", return_value=True), \
             patch("litellm_llmrouter.rbac._load_admin_api_keys", return_value={"test-admin-key"}), \
             patch("litellm_llmrouter.rbac.get_request_id", return_value="test-req-id"):
            
            # Create mock request with admin key in Authorization header
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": "Bearer test-admin-key"}
            mock_request.url.path = "/test/endpoint"
            
            dep_func = requires_permission(PERMISSION_MCP_SERVER_WRITE)
            result = await dep_func(mock_request)
            
            assert result["is_admin"] is True

    @pytest.mark.asyncio
    async def test_user_with_superuser_permission(self):
        """User with superuser '*' permission can access anything."""
        with patch("litellm_llmrouter.rbac._is_admin_auth_enabled", return_value=True), \
             patch("litellm_llmrouter.rbac._load_admin_api_keys", return_value={"test-admin-key"}), \
             patch("litellm_llmrouter.rbac.get_request_id", return_value="test-req-id"):
            
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": "Bearer user-api-key"}
            mock_request.url.path = "/test/endpoint"
            
            # User has superuser permission
            mock_user_info = {
                "user_id": "superuser",
                "permissions": "*"  # Superuser
            }
            
            with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", 
                       new_callable=AsyncMock, return_value=mock_user_info):
                dep_func = requires_permission(PERMISSION_MCP_SERVER_WRITE)
                result = await dep_func(mock_request)
                
                assert result["is_admin"] is False
                assert "*" in result["permissions"]

    @pytest.mark.asyncio
    async def test_user_with_namespace_wildcard(self):
        """User with namespace wildcard 'mcp.*' can access mcp.* permissions."""
        with patch("litellm_llmrouter.rbac._is_admin_auth_enabled", return_value=True), \
             patch("litellm_llmrouter.rbac._load_admin_api_keys", return_value={"test-admin-key"}), \
             patch("litellm_llmrouter.rbac.get_request_id", return_value="test-req-id"):
            
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": "Bearer user-api-key"}
            mock_request.url.path = "/test/endpoint"
            
            # User has mcp.* wildcard
            mock_user_info = {
                "user_id": "mcp-admin",
                "permissions": ["mcp.*"]
            }
            
            with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", 
                       new_callable=AsyncMock, return_value=mock_user_info):
                # Should allow mcp.server.write
                dep_func = requires_permission(PERMISSION_MCP_SERVER_WRITE)
                result = await dep_func(mock_request)
                assert result["is_admin"] is False
                
                # Should also allow mcp.tool.write
                dep_func2 = requires_permission(PERMISSION_MCP_TOOL_WRITE)
                result2 = await dep_func2(mock_request)
                assert result2["is_admin"] is False

    @pytest.mark.asyncio
    async def test_user_with_namespace_wildcard_denied_other_namespace(self):
        """User with 'mcp.*' wildcard cannot access 'system.*' permissions."""
        with patch("litellm_llmrouter.rbac._is_admin_auth_enabled", return_value=True), \
             patch("litellm_llmrouter.rbac._load_admin_api_keys", return_value={"test-admin-key"}), \
             patch("litellm_llmrouter.rbac.get_request_id", return_value="test-req-id"):
            
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": "Bearer user-api-key"}
            mock_request.url.path = "/test/endpoint"
            
            # User has mcp.* wildcard only
            mock_user_info = {
                "user_id": "mcp-admin",
                "permissions": ["mcp.*"]
            }
            
            with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", 
                       new_callable=AsyncMock, return_value=mock_user_info):
                # Should NOT allow system.config.reload
                dep_func = requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)
                
                with pytest.raises(HTTPException) as exc_info:
                    await dep_func(mock_request)
                
                assert exc_info.value.status_code == 403


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_permission_constants_are_lowercase(self):
        """All permission constants are lowercase."""
        assert PERMISSION_MCP_SERVER_WRITE == "mcp.server.write"
        assert PERMISSION_MCP_TOOL_WRITE == "mcp.tool.write"
        assert PERMISSION_MCP_TOOL_CALL == "mcp.tool.call"
        assert PERMISSION_SYSTEM_CONFIG_RELOAD == "system.config.reload"
        assert PERMISSION_A2A_AGENT_WRITE == "a2a.agent.write"

    def test_all_permissions_is_frozenset(self):
        """ALL_PERMISSIONS is a frozenset (immutable)."""
        assert isinstance(ALL_PERMISSIONS, frozenset)

    def test_normalize_handles_frozenset_input(self):
        """normalize_permissions handles frozenset input."""
        input_set = frozenset({"mcp.server.write", "mcp.tool.write"})
        result = normalize_permissions(input_set)
        assert result == frozenset({"mcp.server.write", "mcp.tool.write"})

    @pytest.mark.asyncio
    async def test_admin_auth_disabled_falls_through_to_user_auth(self):
        """When admin auth is disabled, falls through to user auth."""
        with patch("litellm_llmrouter.rbac._is_admin_auth_enabled", return_value=False), \
             patch("litellm_llmrouter.rbac.get_request_id", return_value="test-req-id"):
            
            mock_request = MagicMock()
            mock_request.headers = {"Authorization": "Bearer user-api-key"}
            mock_request.url.path = "/test/endpoint"
            
            mock_user_info = {
                "user_id": "test-user",
                "permissions": ["mcp.server.write"]
            }
            
            with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth", 
                       new_callable=AsyncMock, return_value=mock_user_info):
                dep_func = requires_permission(PERMISSION_MCP_SERVER_WRITE)
                result = await dep_func(mock_request)
                
                # Should use user auth when admin auth is disabled
                assert result["is_admin"] is False


# =============================================================================
# Standalone execution
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
