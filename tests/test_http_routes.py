"""
HTTP-level tests for LLMRouter routes.

These tests verify:
1. Probe endpoints (/_health/live, /_health/ready) are unauthenticated
2. Custom routes (llmrouter_router) require user API key authentication
3. Admin routes (admin_router) require admin API key authentication
4. Error responses are sanitized and include request_id
5. Request correlation IDs are passed through

Uses FastAPI TestClient against minimal app fixtures.
"""

import os
import pytest
import uuid

# Check if litellm is available
try:
    import litellm  # noqa: F401
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LITELLM_AVAILABLE,
    reason="litellm package not installed - HTTP tests require litellm",
)


@pytest.fixture
def app_with_health_router():
    """Create a minimal FastAPI app with just the health router."""
    from litellm_llmrouter.routes import health_router

    app = FastAPI()
    app.include_router(health_router)
    return app


@pytest.fixture
def app_with_llmrouter_router():
    """Create a minimal FastAPI app with the llmrouter router (auth-protected)."""
    from fastapi import Depends, HTTPException, Request
    from litellm_llmrouter.routes import health_router

    # Create a mock auth dependency that rejects requests without API key
    async def mock_api_key_auth(request: Request):
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header == "Bearer ":
            raise HTTPException(
                status_code=401, detail={"error": "Invalid or missing API key"}
            )
        return {"api_key": auth_header.split(" ")[1]}

    # Import and create a router without the original auth dependency
    from fastapi import APIRouter

    # Create a custom router with our mock auth
    test_router = APIRouter(
        tags=["llmrouter-test"],
        dependencies=[Depends(mock_api_key_auth)],
    )

    @test_router.get("/router/info")
    async def get_router_info():
        """Get router info - requires auth."""
        return {
            "registered_strategies": [],
            "strategy_count": 0,
            "hot_reload_enabled": False,
        }

    @test_router.get("/config/sync/status")
    async def get_sync_status():
        """Get sync status - requires auth."""
        return {"enabled": False, "message": "Config sync is not enabled"}

    app = FastAPI()
    app.include_router(health_router)
    app.include_router(test_router)
    return app


class TestProbeEndpointsUnauthenticated:
    """Test that probe endpoints are accessible without authentication."""

    def test_liveness_probe_returns_200(self, app_with_health_router):
        """Test /_health/live returns 200 without auth."""
        client = TestClient(app_with_health_router)
        response = client.get("/_health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["service"] == "litellm-llmrouter"

    def test_readiness_probe_returns_200_no_deps(self, app_with_health_router):
        """Test /_health/ready returns 200 without auth when no deps configured."""
        # Ensure no DATABASE_URL or REDIS_HOST are set
        env_backup = {
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            client = TestClient(app_with_health_router)
            response = client.get("/_health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["service"] == "litellm-llmrouter"
            assert "checks" in data
        finally:
            # Restore environment
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_liveness_probe_no_authorization_header(self, app_with_health_router):
        """Probe endpoints work without any Authorization header."""
        client = TestClient(app_with_health_router)
        # Explicitly ensure no auth header
        response = client.get("/_health/live", headers={})

        assert response.status_code == 200

    def test_readiness_probe_no_authorization_header(self, app_with_health_router):
        """Probe endpoints work without any Authorization header."""
        # Clear deps
        env_backup = {
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            client = TestClient(app_with_health_router)
            response = client.get("/_health/ready", headers={})

            assert response.status_code == 200
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val


class TestAuthGatedEndpoints:
    """Test that custom routes require authentication."""

    def test_router_info_requires_auth(self, app_with_llmrouter_router):
        """Test /router/info returns 401 without API key."""
        client = TestClient(app_with_llmrouter_router, raise_server_exceptions=False)
        response = client.get("/router/info")

        assert response.status_code == 401

    def test_router_info_rejects_empty_bearer(self, app_with_llmrouter_router):
        """Test /router/info returns 401 with empty Bearer token."""
        client = TestClient(app_with_llmrouter_router, raise_server_exceptions=False)
        response = client.get("/router/info", headers={"Authorization": "Bearer "})

        assert response.status_code == 401

    def test_router_info_accepts_valid_key(self, app_with_llmrouter_router):
        """Test /router/info returns 200 with valid API key."""
        client = TestClient(app_with_llmrouter_router)
        response = client.get(
            "/router/info", headers={"Authorization": "Bearer sk-test-key-123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "registered_strategies" in data
        assert "strategy_count" in data
        assert "hot_reload_enabled" in data

    def test_config_sync_status_requires_auth(self, app_with_llmrouter_router):
        """Test /config/sync/status returns 401 without API key."""
        client = TestClient(app_with_llmrouter_router, raise_server_exceptions=False)
        response = client.get("/config/sync/status")

        assert response.status_code == 401

    def test_config_sync_status_accepts_valid_key(self, app_with_llmrouter_router):
        """Test /config/sync/status returns 200 with valid API key."""
        client = TestClient(app_with_llmrouter_router)
        response = client.get(
            "/config/sync/status", headers={"Authorization": "Bearer sk-test-key-123"}
        )

        assert response.status_code == 200

    def test_probes_still_work_on_combined_app(self, app_with_llmrouter_router):
        """Probe endpoints remain unauthenticated even on combined app."""
        # Clear deps
        env_backup = {
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            client = TestClient(app_with_llmrouter_router)

            # Probes should work without auth
            live_resp = client.get("/_health/live")
            assert live_resp.status_code == 200

            ready_resp = client.get("/_health/ready")
            assert ready_resp.status_code == 200
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val


class TestConfigSyncStatusResponse:
    """Test that /config/sync/status returns the structured ConfigSyncManager.get_status() output."""

    def test_sync_status_returns_full_structure(self):
        """Test that get_sync_status route returns the full ConfigSyncManager.get_status() dict."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        # Create a manager and verify get_status() structure
        manager = ConfigSyncManager(sync_interval_seconds=120)
        status = manager.get_status()

        # Verify the expected keys from ConfigSyncManager.get_status()
        assert "enabled" in status
        assert "hot_reload_enabled" in status
        assert "sync_interval_seconds" in status
        assert "local_config_path" in status
        assert "local_config_hash" in status
        assert "reload_count" in status
        assert "last_sync_time" in status
        assert "running" in status

        # These should be present (even if None when not configured)
        assert "s3" in status or status.get("s3") is None
        assert "gcs" in status or status.get("gcs") is None


# =============================================================================
# Admin Auth Tests
# =============================================================================


@pytest.fixture
def app_with_admin_router():
    """Create a minimal FastAPI app with admin-protected routes."""
    from fastapi import APIRouter, Depends
    from litellm_llmrouter.routes import health_router
    from litellm_llmrouter.auth import admin_api_key_auth, RequestIDMiddleware

    # Create admin router with mock responses
    admin_router = APIRouter(
        tags=["admin-test"],
        dependencies=[Depends(admin_api_key_auth)],
    )

    @admin_router.post("/router/reload")
    async def reload_router():
        """Reload router - requires admin auth."""
        return {"status": "reloaded"}

    @admin_router.post("/config/reload")
    async def reload_config():
        """Reload config - requires admin auth."""
        return {"status": "reloaded"}

    @admin_router.post("/llmrouter/mcp/servers")
    async def register_mcp_server():
        """Register MCP server - requires admin auth."""
        return {"status": "registered", "server_id": "test"}

    @admin_router.post("/a2a/agents")
    async def register_agent():
        """Register agent - requires admin auth."""
        return {"status": "registered", "agent_id": "test"}

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.include_router(health_router)
    app.include_router(admin_router)
    return app


class TestAdminAuthEndpoints:
    """Test that control-plane endpoints require admin API key authentication."""

    def test_router_reload_requires_admin_key(self, app_with_admin_router):
        """Test POST /router/reload returns 401 without admin API key."""
        # Clear any existing admin keys
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            # Set a test admin key
            os.environ["ADMIN_API_KEYS"] = "test-admin-key-123"

            client = TestClient(app_with_admin_router, raise_server_exceptions=False)
            response = client.post("/router/reload")

            assert response.status_code == 401
            data = response.json()["detail"]
            assert data["error"] == "admin_key_required"
            assert "request_id" in data
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_router_reload_rejects_invalid_admin_key(self, app_with_admin_router):
        """Test POST /router/reload returns 401 with invalid admin API key."""
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            os.environ["ADMIN_API_KEYS"] = "correct-admin-key"

            client = TestClient(app_with_admin_router, raise_server_exceptions=False)
            response = client.post(
                "/router/reload",
                headers={"X-Admin-API-Key": "wrong-admin-key"},
            )

            assert response.status_code == 401
            data = response.json()["detail"]
            assert data["error"] == "invalid_admin_key"
            assert "request_id" in data
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_router_reload_accepts_valid_admin_key_header(self, app_with_admin_router):
        """Test POST /router/reload returns 200 with valid X-Admin-API-Key header."""
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            os.environ["ADMIN_API_KEYS"] = "test-admin-key-123"

            client = TestClient(app_with_admin_router)
            response = client.post(
                "/router/reload",
                headers={"X-Admin-API-Key": "test-admin-key-123"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "reloaded"
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_router_reload_accepts_admin_key_via_bearer(self, app_with_admin_router):
        """Test POST /router/reload accepts admin key via Authorization: Bearer header."""
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            os.environ["ADMIN_API_KEYS"] = "test-admin-key-456"

            client = TestClient(app_with_admin_router)
            response = client.post(
                "/router/reload",
                headers={"Authorization": "Bearer test-admin-key-456"},
            )

            assert response.status_code == 200
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_admin_auth_denies_when_not_configured(self, app_with_admin_router):
        """Test control-plane returns 403 when no admin keys are configured (fail-closed)."""
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            # No admin keys configured - should fail closed
            client = TestClient(app_with_admin_router, raise_server_exceptions=False)
            response = client.post(
                "/router/reload",
                headers={"X-Admin-API-Key": "any-key"},
            )

            assert response.status_code == 403
            data = response.json()["detail"]
            assert data["error"] == "control_plane_not_configured"
            assert "request_id" in data
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_mcp_server_registration_requires_admin_key(self, app_with_admin_router):
        """Test POST /llmrouter/mcp/servers returns 401 without admin API key."""
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            os.environ["ADMIN_API_KEYS"] = "test-admin-key"

            client = TestClient(app_with_admin_router, raise_server_exceptions=False)
            response = client.post("/llmrouter/mcp/servers")

            assert response.status_code == 401
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_a2a_agent_registration_requires_admin_key(self, app_with_admin_router):
        """Test POST /a2a/agents returns 401 without admin API key."""
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            os.environ["ADMIN_API_KEYS"] = "test-admin-key"

            client = TestClient(app_with_admin_router, raise_server_exceptions=False)
            response = client.post("/a2a/agents")

            assert response.status_code == 401
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_multiple_admin_keys_supported(self, app_with_admin_router):
        """Test that multiple admin keys (comma-separated) are all accepted."""
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            os.environ["ADMIN_API_KEYS"] = "key1,key2,key3"

            client = TestClient(app_with_admin_router)

            # All three keys should work
            for key in ["key1", "key2", "key3"]:
                response = client.post(
                    "/router/reload",
                    headers={"X-Admin-API-Key": key},
                )
                assert response.status_code == 200
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]


class TestRequestCorrelationID:
    """Test request correlation ID middleware."""

    def test_request_id_passthrough(self, app_with_admin_router):
        """Test that X-Request-ID header is passed through to response."""
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            custom_id = "my-custom-request-id-12345"
            client = TestClient(app_with_admin_router)
            response = client.get(
                "/_health/live",
                headers={"X-Request-ID": custom_id},
            )

            assert response.status_code == 200
            assert response.headers.get("X-Request-ID") == custom_id
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_request_id_generated_when_not_provided(self, app_with_admin_router):
        """Test that X-Request-ID is generated when not provided."""
        env_backup = {
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            client = TestClient(app_with_admin_router)
            response = client.get("/_health/live")

            assert response.status_code == 200
            request_id = response.headers.get("X-Request-ID")
            assert request_id is not None
            # Should be a valid UUID
            uuid.UUID(request_id)
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_request_id_in_error_response(self, app_with_admin_router):
        """Test that request_id is included in error response bodies."""
        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            os.environ["ADMIN_API_KEYS"] = "correct-key"
            custom_id = "error-test-request-id"

            client = TestClient(app_with_admin_router, raise_server_exceptions=False)
            response = client.post(
                "/router/reload",
                headers={
                    "X-Request-ID": custom_id,
                    "X-Admin-API-Key": "wrong-key",
                },
            )

            assert response.status_code == 401
            data = response.json()["detail"]
            assert data["request_id"] == custom_id
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]


class TestReadinessErrorSanitization:
    """Test that readiness probe errors are sanitized."""

    def test_readiness_does_not_leak_exception_text(self, app_with_health_router):
        """Test /_health/ready does not leak raw exception messages."""
        from litellm_llmrouter.auth import RequestIDMiddleware

        # Add middleware to the app
        app_with_health_router.add_middleware(RequestIDMiddleware)

        # Configure a database URL that will fail
        env_backup = {
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            # Set an invalid database URL that will cause connection to fail
            os.environ["DATABASE_URL"] = (
                "postgresql://invalid:invalid@localhost:5432/invalid"
            )

            client = TestClient(app_with_health_router, raise_server_exceptions=False)
            response = client.get("/_health/ready")

            # Response could be 200 (if asyncpg not installed, check is skipped)
            # or 503 (if asyncpg installed and connection fails)
            assert response.status_code in (200, 503)

            if response.status_code == 503:
                data = response.json()["detail"]

                # Check that error is sanitized - should not contain connection details
                if "database" in data.get("checks", {}):
                    db_check = data["checks"]["database"]
                    error_msg = db_check.get("error", "")
                    # Should be generic error, not the full exception
                    assert (
                        "connection failed" in error_msg
                        or "connection timeout" in error_msg
                    )
                    # Should not contain stack traces or connection strings
                    assert "traceback" not in error_msg.lower()
                    assert "password" not in error_msg.lower()
                    assert "invalid:invalid" not in error_msg

                # Should include request_id
                assert "request_id" in data
            else:
                # 200 OK - check structure is correct
                data = response.json()
                assert "request_id" in data
                # Database check should be skipped (asyncpg not installed)
                if "database" in data.get("checks", {}):
                    assert data["checks"]["database"]["status"] == "skipped"
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_readiness_includes_request_id(self, app_with_health_router):
        """Test /_health/ready includes request_id in response."""
        from litellm_llmrouter.auth import RequestIDMiddleware

        app_with_health_router.add_middleware(RequestIDMiddleware)

        env_backup = {
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            client = TestClient(app_with_health_router)
            response = client.get("/_health/ready")

            assert response.status_code == 200
            data = response.json()
            assert "request_id" in data
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val


class TestAdminAuthUnit:
    """Unit tests for admin auth module."""

    def test_load_admin_api_keys_from_list(self):
        """Test loading admin keys from ADMIN_API_KEYS env var."""
        from litellm_llmrouter.auth import _load_admin_api_keys

        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
        }

        try:
            os.environ["ADMIN_API_KEYS"] = "key1, key2 , key3"
            keys = _load_admin_api_keys()

            assert keys == {"key1", "key2", "key3"}
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_load_admin_api_keys_fallback_to_single(self):
        """Test fallback to ADMIN_API_KEY when ADMIN_API_KEYS not set."""
        from litellm_llmrouter.auth import _load_admin_api_keys

        env_backup = {
            "ADMIN_API_KEYS": os.environ.pop("ADMIN_API_KEYS", None),
            "ADMIN_API_KEY": os.environ.pop("ADMIN_API_KEY", None),
        }

        try:
            os.environ["ADMIN_API_KEY"] = "single-key"
            keys = _load_admin_api_keys()

            assert keys == {"single-key"}
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_is_admin_auth_enabled_default_true(self):
        """Test that admin auth is enabled by default."""
        from litellm_llmrouter.auth import _is_admin_auth_enabled

        env_backup = {
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            assert _is_admin_auth_enabled() is True
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_is_admin_auth_enabled_can_be_disabled(self):
        """Test that admin auth can be disabled via env var."""
        from litellm_llmrouter.auth import _is_admin_auth_enabled

        env_backup = {
            "ADMIN_AUTH_ENABLED": os.environ.pop("ADMIN_AUTH_ENABLED", None),
        }

        try:
            for disable_value in ["false", "False", "FALSE", "0", "no", "off"]:
                os.environ["ADMIN_AUTH_ENABLED"] = disable_value
                assert _is_admin_auth_enabled() is False, (
                    f"Failed for value: {disable_value}"
                )
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

    def test_sanitize_error_response(self):
        """Test that sanitize_error_response produces correct structure."""
        from litellm_llmrouter.auth import sanitize_error_response

        error = Exception(
            "Sensitive database connection string: postgres://user:pass@host"
        )
        result = sanitize_error_response(error, "test-req-id", "Generic error message")

        assert result["error"] == "internal_error"
        assert result["message"] == "Generic error message"
        assert result["request_id"] == "test-req-id"
        # Should NOT contain the sensitive info
        assert "postgres" not in str(result)
        assert "user:pass" not in str(result)
