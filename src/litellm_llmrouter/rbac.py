"""
Role-Based Access Control (RBAC) for RouteIQ Gateway
=====================================================

This module provides fine-grained permission checking for control-plane endpoints.
It extends the existing admin-vs-user authentication with granular permissions.

Permission Taxonomy:
- mcp.server.write: Create/update/delete MCP servers
- mcp.tool.write: Register MCP tool definitions
- mcp.tool.call: Invoke MCP tools (remote execution)
- system.config.reload: Trigger config/sync reload
- a2a.agent.write: Create/update/delete A2A agents

Usage:
    from litellm_llmrouter.rbac import requires_permission

    @router.post("/mcp/servers", dependencies=[Depends(requires_permission("mcp.server.write"))])
    async def register_mcp_server(...):
        ...

Design:
- Admin users (authenticated via admin_api_key_auth) are always allowed.
- Regular users (authenticated via user_api_key_auth) need explicit permissions
  in their metadata["permissions"] list.
- Missing credentials => 401 Unauthorized
- Authenticated but missing permission => 403 Forbidden
- Implemented as FastAPI dependencies (no response body middleware/buffering)
"""

import logging
import os
from typing import Any

from fastapi import Depends, HTTPException, Request

from .auth import (
    ADMIN_API_KEY_HEADER,
    AUTHORIZATION_HEADER,
    get_request_id,
    _extract_bearer_token,
    _load_admin_api_keys,
    _is_admin_auth_enabled,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Permission Constants
# =============================================================================

# Standard permission strings following a hierarchical namespace
PERMISSION_MCP_SERVER_WRITE = "mcp.server.write"
PERMISSION_MCP_TOOL_WRITE = "mcp.tool.write"
PERMISSION_MCP_TOOL_CALL = "mcp.tool.call"
PERMISSION_SYSTEM_CONFIG_RELOAD = "system.config.reload"
PERMISSION_A2A_AGENT_WRITE = "a2a.agent.write"

# All known permissions (used for validation/documentation)
ALL_PERMISSIONS = frozenset({
    PERMISSION_MCP_SERVER_WRITE,
    PERMISSION_MCP_TOOL_WRITE,
    PERMISSION_MCP_TOOL_CALL,
    PERMISSION_SYSTEM_CONFIG_RELOAD,
    PERMISSION_A2A_AGENT_WRITE,
})

# Superuser permission that grants all access (alternative to admin key)
PERMISSION_SUPERUSER = "*"


# =============================================================================
# Permission Normalization
# =============================================================================


def normalize_permissions(raw_permissions: Any) -> frozenset[str]:
    """
    Normalize permissions from various input formats to a consistent frozenset.

    Handles:
    - None => empty set
    - str => single permission (comma-separated also supported)
    - list[str] => list of permissions
    - set[str] => set of permissions

    Permissions are lowercased and whitespace-stripped for robustness.

    Args:
        raw_permissions: Raw permissions from metadata

    Returns:
        Normalized frozenset of permission strings
    """
    if raw_permissions is None:
        return frozenset()

    if isinstance(raw_permissions, str):
        # Handle comma-separated string: "mcp.server.write, mcp.tool.write"
        perms = [p.strip().lower() for p in raw_permissions.split(",")]
        return frozenset(p for p in perms if p)

    if isinstance(raw_permissions, (list, tuple, set, frozenset)):
        perms = []
        for item in raw_permissions:
            if isinstance(item, str):
                perms.append(item.strip().lower())
        return frozenset(p for p in perms if p)

    logger.warning(
        f"Unexpected permissions type: {type(raw_permissions).__name__}, treating as empty"
    )
    return frozenset()


def extract_user_permissions(user_info: dict[str, Any]) -> frozenset[str]:
    """
    Extract permissions from user info/metadata.

    Looks for permissions in multiple locations:
    1. user_info["permissions"] (direct)
    2. user_info["metadata"]["permissions"] (nested in metadata)

    Args:
        user_info: User info dict from authentication

    Returns:
        Normalized frozenset of permission strings
    """
    # Try direct permissions field first
    if "permissions" in user_info:
        return normalize_permissions(user_info["permissions"])

    # Try nested in metadata
    metadata = user_info.get("metadata", {})
    if isinstance(metadata, dict) and "permissions" in metadata:
        return normalize_permissions(metadata["permissions"])

    return frozenset()


def has_permission(user_permissions: frozenset[str], required: str) -> bool:
    """
    Check if user has the required permission.

    Supports:
    - Exact match: "mcp.server.write"
    - Wildcard: "*" grants all permissions
    - Namespace wildcard: "mcp.*" grants all mcp.* permissions

    Args:
        user_permissions: User's granted permissions
        required: Required permission string

    Returns:
        True if permission is granted
    """
    # Superuser has all permissions
    if PERMISSION_SUPERUSER in user_permissions:
        return True

    # Exact match
    if required.lower() in user_permissions:
        return True

    # Namespace wildcard (e.g., "mcp.*" grants "mcp.server.write")
    required_lower = required.lower()
    required_parts = required_lower.split(".")
    for perm in user_permissions:
        if perm.endswith(".*"):
            prefix = perm[:-2]  # Remove ".*"
            if required_lower.startswith(prefix + "."):
                return True

    return False


# =============================================================================
# Authentication Helpers
# =============================================================================


async def _try_admin_auth(request: Request) -> dict[str, Any] | None:
    """
    Try to authenticate as admin without raising exceptions.

    Returns admin info dict on success, None on failure.
    """
    if not _is_admin_auth_enabled():
        return None

    admin_keys = _load_admin_api_keys()
    if not admin_keys:
        return None

    # Try X-Admin-API-Key header first
    admin_key = request.headers.get(ADMIN_API_KEY_HEADER, "").strip()

    # Fallback to Authorization header
    if not admin_key:
        auth_header = request.headers.get(AUTHORIZATION_HEADER, "")
        admin_key = _extract_bearer_token(auth_header) or ""

    if admin_key and admin_key in admin_keys:
        return {"admin_key": admin_key, "is_admin": True}

    return None


async def _try_user_auth(request: Request) -> dict[str, Any] | None:
    """
    Try to authenticate as regular user without raising exceptions.

    Uses LiteLLM's user_api_key_auth under the hood.

    Returns user info dict on success, None on failure.
    """
    try:
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        # user_api_key_auth returns a dict with user info including metadata
        user_info = await user_api_key_auth(request)
        if user_info:
            return {"user_info": user_info, "is_admin": False}
    except HTTPException:
        # User auth failed
        pass
    except Exception as e:
        logger.debug(f"User auth failed with exception: {e}")

    return None


# =============================================================================
# RBAC Dependency Factory
# =============================================================================


def requires_permission(permission: str):
    """
    Factory function that creates a FastAPI dependency for permission checking.

    The returned dependency:
    1. Tries admin authentication first - if valid, allows access (admin bypass)
    2. Falls back to user authentication - checks for required permission
    3. Returns 401 if no valid credentials
    4. Returns 403 if authenticated but missing permission

    This is a dependency, not middleware - it doesn't touch the response body
    and is safe for streaming endpoints.

    Args:
        permission: Required permission string (e.g., "mcp.server.write")

    Returns:
        FastAPI dependency function

    Example:
        @router.post("/mcp/servers")
        async def create_server(
            ...,
            auth_info: dict = Depends(requires_permission("mcp.server.write"))
        ):
            # auth_info contains {"is_admin": bool, "permissions": frozenset, ...}
    """

    async def permission_dependency(request: Request) -> dict[str, Any]:
        """Check if request has required permission."""
        request_id = get_request_id() or "unknown"
        path = str(request.url.path)

        # Try admin auth first (admin bypass)
        admin_result = await _try_admin_auth(request)
        if admin_result is not None:
            logger.debug(
                f"RBAC: Admin access granted for {path}",
                extra={"request_id": request_id, "permission": permission},
            )
            return {
                "is_admin": True,
                "permissions": frozenset({PERMISSION_SUPERUSER}),
                "admin_key": admin_result.get("admin_key"),
            }

        # Try user auth
        user_result = await _try_user_auth(request)
        if user_result is None:
            # No valid credentials at all
            logger.warning(
                f"RBAC: No valid credentials for {path}",
                extra={"request_id": request_id, "permission": permission},
            )
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "authentication_required",
                    "message": "Valid API key required. Provide via Authorization header or X-Admin-API-Key.",
                    "request_id": request_id,
                },
            )

        # User authenticated - check permissions
        user_info = user_result.get("user_info", {})
        user_permissions = extract_user_permissions(user_info)

        if has_permission(user_permissions, permission):
            logger.debug(
                f"RBAC: User access granted for {path} with permission {permission}",
                extra={"request_id": request_id, "permissions": list(user_permissions)},
            )
            return {
                "is_admin": False,
                "permissions": user_permissions,
                "user_info": user_info,
            }

        # Authenticated but missing permission
        logger.warning(
            f"RBAC: Permission denied for {path} - requires {permission}",
            extra={"request_id": request_id, "user_permissions": list(user_permissions)},
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error": "permission_denied",
                "message": f"Insufficient permissions. Required: {permission}",
                "required_permission": permission,
                "request_id": request_id,
            },
        )

    return permission_dependency


# =============================================================================
# Convenience Dependency Factories
# =============================================================================

# Pre-defined dependencies for common permissions
mcp_server_write = requires_permission(PERMISSION_MCP_SERVER_WRITE)
mcp_tool_write = requires_permission(PERMISSION_MCP_TOOL_WRITE)
mcp_tool_call = requires_permission(PERMISSION_MCP_TOOL_CALL)
system_config_reload = requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)
a2a_agent_write = requires_permission(PERMISSION_A2A_AGENT_WRITE)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main factory
    "requires_permission",
    # Permission constants
    "PERMISSION_MCP_SERVER_WRITE",
    "PERMISSION_MCP_TOOL_WRITE",
    "PERMISSION_MCP_TOOL_CALL",
    "PERMISSION_SYSTEM_CONFIG_RELOAD",
    "PERMISSION_A2A_AGENT_WRITE",
    "PERMISSION_SUPERUSER",
    "ALL_PERMISSIONS",
    # Helpers
    "normalize_permissions",
    "extract_user_permissions",
    "has_permission",
    # Pre-defined dependencies
    "mcp_server_write",
    "mcp_tool_write",
    "mcp_tool_call",
    "system_config_reload",
    "a2a_agent_write",
]
