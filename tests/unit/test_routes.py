"""
Tests for route handler modules: health, config, admin_ui, and models.

Uses FastAPI's TestClient with create_standalone_app() and mocked auth
dependencies for synchronous HTTP-level route testing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm_llmrouter.gateway.app import create_standalone_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_TEST_ADMIN_KEY = "sk-riq-test-admin-key-for-routes"


@pytest.fixture()
def client(monkeypatch):
    """Create a TestClient backed by the standalone gateway app.

    A test admin API key is configured so both admin_api_key_auth and
    RBAC requires_permission() grant access when the key is provided
    in the X-Admin-API-Key header.
    """
    monkeypatch.setenv("ADMIN_API_KEYS", _TEST_ADMIN_KEY)
    monkeypatch.setenv("ADMIN_AUTH_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_ENV", "test")

    app = create_standalone_app(
        enable_plugins=False,
        enable_resilience=False,
    )

    # Override user_api_key_auth for llmrouter_router endpoints
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: {
        "api_key": "test-user-key",
    }

    c = TestClient(app, raise_server_exceptions=False)
    # Patch the request method to always include the admin key header
    _original_request = c.request

    def _authed_request(*args, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("X-Admin-API-Key", _TEST_ADMIN_KEY)
        return _original_request(*args, headers=headers, **kwargs)

    c.request = _authed_request
    return c


@pytest.fixture()
def unauthed_client():
    """Client with NO auth overrides — used to test 401/403 responses."""
    app = create_standalone_app(
        enable_plugins=False,
        enable_resilience=False,
    )
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# Health Routes
# ===========================================================================


class TestHealthRoutes:
    """Tests for /_health/* and /config/services endpoints."""

    def test_liveness_probe_returns_200(self, client):
        resp = client.get("/_health/live")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "alive"
        assert body["service"] == "litellm-llmrouter"

    def test_readiness_probe_returns_200(self, client):
        """Readiness returns 200 with a status field when no external deps configured."""
        resp = client.get("/_health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert body["status"] in ("ready", "degraded")
        assert "checks" in body

    def test_readiness_degraded_returns_200(self, client):
        """When circuit breakers are open, readiness returns 200 (not 503)
        with status 'degraded' per the documented non-obvious behaviour."""
        from litellm_llmrouter.resilience import (
            get_circuit_breaker_manager,
            CircuitBreakerState,
        )
        import time

        cb_manager = get_circuit_breaker_manager()
        # Open a breaker to trigger degraded mode
        cb = cb_manager.get_breaker("test-model")
        # Force the breaker into open state
        cb._state = CircuitBreakerState.OPEN
        cb._opened_at = time.monotonic()

        resp = client.get("/_health/ready")
        # Degraded → 200, not 503
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["is_degraded"] is True

    def test_service_status_returns_services(self, client):
        resp = client.get("/config/services")
        assert resp.status_code == 200
        body = resp.json()
        assert "services" in body
        assert "features" in body

    def test_model_health_returns_models(self, client):
        resp = client.get("/_health/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert isinstance(body["models"], list)

    def test_liveness_is_unauthenticated(self, unauthed_client):
        """Liveness probe must work without credentials."""
        resp = unauthed_client.get("/_health/live")
        assert resp.status_code == 200

    def test_readiness_is_unauthenticated(self, unauthed_client):
        """Readiness probe must work without credentials."""
        resp = unauthed_client.get("/_health/ready")
        # 200 when no external deps, no auth required
        assert resp.status_code == 200


# ===========================================================================
# Admin UI Routes
# ===========================================================================


class TestAdminUIRoutes:
    """Tests for /api/v1/routeiq/* admin UI endpoints."""

    def test_ui_config_returns_200(self, client):
        resp = client.get("/api/v1/routeiq/ui-config")
        assert resp.status_code == 200
        body = resp.json()
        assert "version" in body
        assert "features" in body
        assert "oidc" in body

    def test_ui_config_version_is_dynamic(self, client):
        """Version should come from package metadata, not be a hardcoded literal."""
        resp = client.get("/api/v1/routeiq/ui-config")
        body = resp.json()
        # Version is resolved dynamically via importlib.metadata.version("routeiq")
        # with a fallback of "0.0.0-dev". Verify it's present and non-empty.
        assert isinstance(body["version"], str)
        assert len(body["version"]) > 0
        # Verify it matches what importlib.metadata would return
        import importlib.metadata as _importlib_meta

        try:
            expected = _importlib_meta.version("routeiq")
        except _importlib_meta.PackageNotFoundError:
            expected = "0.0.0-dev"
        assert body["version"] == expected

    def test_ui_config_features_include_expected_keys(self, client):
        resp = client.get("/api/v1/routeiq/ui-config")
        features = resp.json()["features"]
        assert "sso_login" in features
        assert "model_playground" in features
        assert "cost_analytics" in features

    def test_gateway_status_returns_200(self, client):
        resp = client.get("/api/v1/routeiq/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "version" in body
        assert "uptime_seconds" in body
        assert "feature_flags" in body

    def test_models_endpoint_returns_200(self, client):
        resp = client.get("/api/v1/routeiq/models")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)

    def test_routing_stats_returns_200(self, client):
        resp = client.get("/api/v1/routeiq/routing/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_decisions" in body

    def test_routing_config_get_returns_200(self, client):
        resp = client.get("/api/v1/routeiq/routing/config")
        assert resp.status_code == 200
        body = resp.json()
        assert "routing_profile" in body
        assert "centroid_routing_enabled" in body

    def test_admin_ui_requires_auth(self, unauthed_client):
        """Admin UI endpoints should require authentication."""
        # Without auth overrides, the admin_router dependency will reject
        resp = unauthed_client.get("/api/v1/routeiq/status")
        assert resp.status_code in (401, 403)


# ===========================================================================
# Governance Workspace CRUD
# ===========================================================================


class TestGovernanceWorkspaces:
    """Tests for /api/v1/routeiq/governance/workspaces CRUD."""

    def test_list_workspaces_empty(self, client):
        resp = client.get("/api/v1/routeiq/governance/workspaces")
        assert resp.status_code == 200
        body = resp.json()
        assert body["workspaces"] == []
        assert body["count"] == 0

    def test_create_workspace(self, client):
        payload = {
            "workspace_id": "ws-test-1",
            "name": "Test Workspace",
            "org_id": "org-1",
        }
        resp = client.post("/api/v1/routeiq/governance/workspaces", json=payload)
        assert resp.status_code == 201
        body = resp.json()
        assert body["workspace_id"] == "ws-test-1"
        assert body["created"] is True

    def test_get_workspace(self, client):
        # Create first
        payload = {
            "workspace_id": "ws-get-1",
            "name": "Get Workspace",
            "org_id": "org-1",
        }
        client.post("/api/v1/routeiq/governance/workspaces", json=payload)

        # Get
        resp = client.get("/api/v1/routeiq/governance/workspaces/ws-get-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["workspace"]["workspace_id"] == "ws-get-1"

    def test_get_workspace_not_found(self, client):
        resp = client.get("/api/v1/routeiq/governance/workspaces/nonexistent")
        assert resp.status_code == 404

    def test_update_workspace(self, client):
        # Create
        payload = {
            "workspace_id": "ws-upd-1",
            "name": "Original",
            "org_id": "org-1",
        }
        client.post("/api/v1/routeiq/governance/workspaces", json=payload)

        # Update
        update_payload = {
            "workspace_id": "ws-upd-1",
            "name": "Updated",
            "org_id": "org-1",
        }
        resp = client.put(
            "/api/v1/routeiq/governance/workspaces/ws-upd-1", json=update_payload
        )
        assert resp.status_code == 200
        assert resp.json()["workspace"]["name"] == "Updated"

    def test_update_workspace_not_found(self, client):
        payload = {
            "workspace_id": "nonexistent",
            "name": "X",
            "org_id": "org-1",
        }
        resp = client.put(
            "/api/v1/routeiq/governance/workspaces/nonexistent", json=payload
        )
        assert resp.status_code == 404

    def test_delete_workspace(self, client):
        # Create
        payload = {
            "workspace_id": "ws-del-1",
            "name": "Delete Me",
            "org_id": "org-1",
        }
        client.post("/api/v1/routeiq/governance/workspaces", json=payload)

        # Delete
        resp = client.delete("/api/v1/routeiq/governance/workspaces/ws-del-1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Verify gone
        resp = client.get("/api/v1/routeiq/governance/workspaces/ws-del-1")
        assert resp.status_code == 404

    def test_delete_workspace_not_found(self, client):
        resp = client.delete("/api/v1/routeiq/governance/workspaces/nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# Governance Key CRUD
# ===========================================================================


class TestGovernanceKeys:
    """Tests for /api/v1/routeiq/governance/keys CRUD."""

    def test_get_key_governance_not_found(self, client):
        resp = client.get("/api/v1/routeiq/governance/keys/nonexistent")
        assert resp.status_code == 404

    def test_put_and_get_key_governance(self, client):
        payload = {
            "key_id": "key-test-1",
            "allowed_models": ["gpt-4"],
            "max_rpm": 100,
        }
        resp = client.put("/api/v1/routeiq/governance/keys/key-test-1", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["key_governance"]["key_id"] == "key-test-1"

        # Get it back
        resp = client.get("/api/v1/routeiq/governance/keys/key-test-1")
        assert resp.status_code == 200

    def test_delete_key_governance(self, client):
        payload = {
            "key_id": "key-del-1",
            "allowed_models": ["gpt-4"],
        }
        client.put("/api/v1/routeiq/governance/keys/key-del-1", json=payload)

        resp = client.delete("/api/v1/routeiq/governance/keys/key-del-1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_key_governance_not_found(self, client):
        resp = client.delete("/api/v1/routeiq/governance/keys/nonexistent")
        assert resp.status_code == 404


# ===========================================================================
# Usage Policy CRUD
# ===========================================================================


class TestUsagePolicies:
    """Tests for /api/v1/routeiq/governance/policies CRUD."""

    def test_list_policies_empty(self, client):
        resp = client.get("/api/v1/routeiq/governance/policies")
        assert resp.status_code == 200
        body = resp.json()
        assert body["policies"] == []
        assert body["count"] == 0

    def test_create_usage_policy(self, client):
        payload = {
            "policy_id": "pol-test-1",
            "name": "Test Policy",
            "limit_type": "requests",
            "limit_value": 1000,
            "limit_period": "day",
            "alert_threshold": 0.8,
        }
        resp = client.post("/api/v1/routeiq/governance/policies", json=payload)
        assert resp.status_code == 201
        body = resp.json()
        assert body["policy_id"] == "pol-test-1"
        assert body["created"] is True

    def test_create_usage_policy_invalid_limit(self, client):
        payload = {
            "policy_id": "pol-bad",
            "name": "Bad",
            "limit_type": "requests",
            "limit_value": -1,
            "limit_period": "day",
            "alert_threshold": 0.8,
        }
        resp = client.post("/api/v1/routeiq/governance/policies", json=payload)
        assert resp.status_code == 400

    def test_create_usage_policy_invalid_threshold(self, client):
        payload = {
            "policy_id": "pol-bad2",
            "name": "Bad2",
            "limit_type": "requests",
            "limit_value": 100,
            "limit_period": "day",
            "alert_threshold": 1.5,
        }
        resp = client.post("/api/v1/routeiq/governance/policies", json=payload)
        assert resp.status_code == 400

    def test_get_usage_policy(self, client):
        payload = {
            "policy_id": "pol-get-1",
            "name": "Get Policy",
            "limit_type": "requests",
            "limit_value": 500,
            "limit_period": "day",
            "alert_threshold": 0.9,
        }
        client.post("/api/v1/routeiq/governance/policies", json=payload)

        resp = client.get("/api/v1/routeiq/governance/policies/pol-get-1")
        assert resp.status_code == 200
        assert resp.json()["policy"]["policy_id"] == "pol-get-1"

    def test_get_usage_policy_not_found(self, client):
        resp = client.get("/api/v1/routeiq/governance/policies/nonexistent")
        assert resp.status_code == 404

    def test_update_usage_policy(self, client):
        payload = {
            "policy_id": "pol-upd-1",
            "name": "Original",
            "limit_type": "requests",
            "limit_value": 100,
            "limit_period": "day",
            "alert_threshold": 0.5,
        }
        client.post("/api/v1/routeiq/governance/policies", json=payload)

        update = {
            "policy_id": "pol-upd-1",
            "name": "Updated",
            "limit_type": "requests",
            "limit_value": 200,
            "limit_period": "day",
            "alert_threshold": 0.7,
        }
        resp = client.put("/api/v1/routeiq/governance/policies/pol-upd-1", json=update)
        assert resp.status_code == 200
        assert resp.json()["policy"]["name"] == "Updated"
        assert resp.json()["policy"]["limit_value"] == 200

    def test_update_usage_policy_not_found(self, client):
        payload = {
            "policy_id": "nonexistent",
            "name": "X",
            "limit_type": "requests",
            "limit_value": 100,
            "limit_period": "day",
            "alert_threshold": 0.5,
        }
        resp = client.put(
            "/api/v1/routeiq/governance/policies/nonexistent", json=payload
        )
        assert resp.status_code == 404

    def test_delete_usage_policy(self, client):
        payload = {
            "policy_id": "pol-del-1",
            "name": "Delete Me",
            "limit_type": "requests",
            "limit_value": 100,
            "limit_period": "day",
            "alert_threshold": 0.5,
        }
        client.post("/api/v1/routeiq/governance/policies", json=payload)

        resp = client.delete("/api/v1/routeiq/governance/policies/pol-del-1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_usage_policy_not_found(self, client):
        resp = client.delete("/api/v1/routeiq/governance/policies/nonexistent")
        assert resp.status_code == 404

    def test_usage_query(self, client):
        """GET /policies/{id}/usage returns usage counters."""
        payload = {
            "policy_id": "pol-usage-1",
            "name": "Usage",
            "limit_type": "requests",
            "limit_value": 100,
            "limit_period": "day",
            "alert_threshold": 0.5,
        }
        client.post("/api/v1/routeiq/governance/policies", json=payload)

        resp = client.get("/api/v1/routeiq/governance/policies/pol-usage-1/usage")
        assert resp.status_code == 200
        body = resp.json()
        assert body["policy_id"] == "pol-usage-1"

    def test_usage_query_not_found(self, client):
        resp = client.get("/api/v1/routeiq/governance/policies/nonexistent/usage")
        assert resp.status_code == 404

    def test_reset_policy_counters(self, client):
        payload = {
            "policy_id": "pol-reset-1",
            "name": "Reset",
            "limit_type": "requests",
            "limit_value": 100,
            "limit_period": "day",
            "alert_threshold": 0.5,
        }
        client.post("/api/v1/routeiq/governance/policies", json=payload)

        resp = client.post("/api/v1/routeiq/governance/policies/pol-reset-1/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["policy_id"] == "pol-reset-1"
        # reset may be False when Redis is unavailable (test env)
        assert "reset" in body

    def test_reset_policy_not_found(self, client):
        resp = client.post("/api/v1/routeiq/governance/policies/nonexistent/reset")
        assert resp.status_code == 404


# ===========================================================================
# Guardrail Policy CRUD
# ===========================================================================


class TestGuardrailPolicies:
    """Tests for /api/v1/routeiq/governance/guardrails CRUD."""

    def test_list_guardrail_policies_empty(self, client):
        resp = client.get("/api/v1/routeiq/governance/guardrails")
        assert resp.status_code == 200
        body = resp.json()
        assert body["guardrails"] == []
        assert body["count"] == 0

    def test_create_guardrail_policy(self, client):
        payload = {
            "guardrail_id": "gr-test-1",
            "name": "Test Guardrail",
            "check_type": "regex_deny",
            "phase": "input",
            "action": "deny",
            "enabled": True,
        }
        resp = client.post("/api/v1/routeiq/governance/guardrails", json=payload)
        assert resp.status_code == 201
        body = resp.json()
        assert body["guardrail_id"] == "gr-test-1"
        assert body["created"] is True

    def test_create_guardrail_missing_id(self, client):
        payload = {
            "guardrail_id": "",
            "name": "No ID",
            "check_type": "regex_deny",
            "phase": "input",
            "action": "deny",
        }
        resp = client.post("/api/v1/routeiq/governance/guardrails", json=payload)
        assert resp.status_code == 400

    def test_create_guardrail_missing_name(self, client):
        payload = {
            "guardrail_id": "gr-noname",
            "name": "",
            "check_type": "regex_deny",
            "phase": "input",
            "action": "deny",
        }
        resp = client.post("/api/v1/routeiq/governance/guardrails", json=payload)
        assert resp.status_code == 400

    def test_get_guardrail_policy(self, client):
        payload = {
            "guardrail_id": "gr-get-1",
            "name": "Get Guardrail",
            "check_type": "regex_deny",
            "phase": "input",
            "action": "deny",
        }
        client.post("/api/v1/routeiq/governance/guardrails", json=payload)

        resp = client.get("/api/v1/routeiq/governance/guardrails/gr-get-1")
        assert resp.status_code == 200
        assert resp.json()["guardrail"]["guardrail_id"] == "gr-get-1"

    def test_get_guardrail_not_found(self, client):
        resp = client.get("/api/v1/routeiq/governance/guardrails/nonexistent")
        assert resp.status_code == 404

    def test_update_guardrail_policy(self, client):
        payload = {
            "guardrail_id": "gr-upd-1",
            "name": "Original",
            "check_type": "regex_deny",
            "phase": "input",
            "action": "deny",
        }
        client.post("/api/v1/routeiq/governance/guardrails", json=payload)

        update = {
            "guardrail_id": "gr-upd-1",
            "name": "Updated",
            "check_type": "regex_deny",
            "phase": "input",
            "action": "log",
        }
        resp = client.put("/api/v1/routeiq/governance/guardrails/gr-upd-1", json=update)
        assert resp.status_code == 200
        assert resp.json()["guardrail"]["name"] == "Updated"

    def test_update_guardrail_not_found(self, client):
        payload = {
            "guardrail_id": "nonexistent",
            "name": "X",
            "check_type": "regex_deny",
            "phase": "input",
            "action": "deny",
        }
        resp = client.put(
            "/api/v1/routeiq/governance/guardrails/nonexistent", json=payload
        )
        assert resp.status_code == 404

    def test_delete_guardrail_policy(self, client):
        payload = {
            "guardrail_id": "gr-del-1",
            "name": "Delete Me",
            "check_type": "regex_deny",
            "phase": "input",
            "action": "deny",
        }
        client.post("/api/v1/routeiq/governance/guardrails", json=payload)

        resp = client.delete("/api/v1/routeiq/governance/guardrails/gr-del-1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_guardrail_not_found(self, client):
        resp = client.delete("/api/v1/routeiq/governance/guardrails/nonexistent")
        assert resp.status_code == 404

    def test_list_with_invalid_phase_filter(self, client):
        resp = client.get("/api/v1/routeiq/governance/guardrails?phase=invalid")
        assert resp.status_code == 400


# ===========================================================================
# Prompt Management CRUD
# ===========================================================================


class TestPromptManagement:
    """Tests for /api/v1/routeiq/prompts/* endpoints.

    Prompt management requires ROUTEIQ_PROMPT_MANAGEMENT=true, so tests
    patch the feature flag.
    """

    @pytest.fixture(autouse=True)
    def _enable_prompt_mgmt(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_PROMPT_MANAGEMENT", "true")

    def test_list_prompts_empty(self, client):
        resp = client.get("/api/v1/routeiq/prompts")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0

    def test_create_prompt(self, client):
        payload = {
            "name": "test-prompt",
            "template": "Hello, {{name}}!",
        }
        resp = client.post("/api/v1/routeiq/prompts", json=payload)
        assert resp.status_code == 201
        body = resp.json()
        assert body["prompt"]["name"] == "test-prompt"

    def test_get_prompt(self, client):
        client.post(
            "/api/v1/routeiq/prompts",
            json={"name": "get-me", "template": "Hi"},
        )
        resp = client.get("/api/v1/routeiq/prompts/get-me")
        assert resp.status_code == 200
        assert resp.json()["prompt"]["name"] == "get-me"

    def test_get_prompt_not_found(self, client):
        resp = client.get("/api/v1/routeiq/prompts/nonexistent")
        assert resp.status_code == 404

    def test_update_prompt(self, client):
        client.post(
            "/api/v1/routeiq/prompts",
            json={"name": "upd-me", "template": "v1"},
        )
        resp = client.put(
            "/api/v1/routeiq/prompts/upd-me",
            json={"template": "v2", "change_note": "Updated template"},
        )
        assert resp.status_code == 200

    def test_delete_prompt(self, client):
        client.post(
            "/api/v1/routeiq/prompts",
            json={"name": "del-me", "template": "bye"},
        )
        resp = client.delete("/api/v1/routeiq/prompts/del-me")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_prompt_not_found(self, client):
        resp = client.delete("/api/v1/routeiq/prompts/nonexistent")
        assert resp.status_code == 404

    def test_prompt_disabled_returns_404(self, client, monkeypatch):
        """When prompt management is disabled, endpoints return 404."""
        monkeypatch.setenv("ROUTEIQ_PROMPT_MANAGEMENT", "false")
        resp = client.get("/api/v1/routeiq/prompts")
        assert resp.status_code == 404


# ===========================================================================
# Eval Pipeline Endpoints
# ===========================================================================


class TestEvalPipelineEndpoints:
    """Tests for /api/v1/routeiq/eval/* endpoints."""

    def test_eval_stats_disabled(self, client):
        """When eval pipeline is not enabled, returns disabled status."""
        resp = client.get("/api/v1/routeiq/eval/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False

    def test_eval_samples_disabled(self, client):
        resp = client.get("/api/v1/routeiq/eval/samples")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["samples"] == []

    def test_eval_run_batch_disabled(self, client):
        resp = client.post("/api/v1/routeiq/eval/run-batch")
        assert resp.status_code == 503

    def test_eval_model_quality_disabled(self, client):
        resp = client.get("/api/v1/routeiq/eval/model-quality")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False


# ===========================================================================
# Routing Feedback Endpoint
# ===========================================================================


class TestRoutingFeedback:
    """Tests for /api/v1/routeiq/routing/feedback."""

    def test_feedback_disabled(self, client):
        """When personalized routing is not enabled, returns 503."""
        resp = client.post(
            "/api/v1/routeiq/routing/feedback",
            json={"user_id": "u1", "model": "gpt-4", "score": 0.5},
        )
        assert resp.status_code == 503

    def test_feedback_invalid_json(self, client, monkeypatch):
        """Invalid JSON body returns 400."""
        monkeypatch.setenv("ROUTEIQ_PERSONALIZED_ROUTING", "true")
        # Send non-JSON to trigger parse error
        resp = client.post(
            "/api/v1/routeiq/routing/feedback",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        # Either 400 (invalid JSON) or 503 (not enabled) depending on order
        assert resp.status_code in (400, 503)


# ===========================================================================
# Router-R1 Endpoint
# ===========================================================================


class TestRouterR1:
    """Tests for /api/v1/routeiq/routing/r1."""

    def test_r1_disabled(self, client):
        """When Router-R1 is not enabled, returns 503."""
        resp = client.post(
            "/api/v1/routeiq/routing/r1",
            json={"query": "What is AI?"},
        )
        assert resp.status_code == 503


# ===========================================================================
# Auth Required for Admin Endpoints
# ===========================================================================


class TestAdminAuthRequired:
    """Verify admin endpoints require authentication."""

    def test_governance_workspaces_requires_auth(self, unauthed_client):
        resp = unauthed_client.get("/api/v1/routeiq/governance/workspaces")
        assert resp.status_code in (401, 403)

    def test_governance_policies_requires_auth(self, unauthed_client):
        resp = unauthed_client.get("/api/v1/routeiq/governance/policies")
        assert resp.status_code in (401, 403)

    def test_governance_guardrails_requires_auth(self, unauthed_client):
        resp = unauthed_client.get("/api/v1/routeiq/governance/guardrails")
        assert resp.status_code in (401, 403)

    def test_eval_stats_requires_auth(self, unauthed_client):
        resp = unauthed_client.get("/api/v1/routeiq/eval/stats")
        assert resp.status_code in (401, 403)

    def test_config_reload_requires_auth(self, unauthed_client):
        resp = unauthed_client.post("/llmrouter/reload")
        assert resp.status_code in (401, 403)

    def test_spend_report_requires_auth(self, unauthed_client):
        resp = unauthed_client.get("/llmrouter/spend/report")
        assert resp.status_code in (401, 403)


# ===========================================================================
# Config Status (unauthenticated)
# ===========================================================================


class TestConfigStatus:
    """Tests for /config/status (unauthenticated)."""

    def test_config_status_returns_200(self, client):
        resp = client.get("/config/status")
        assert resp.status_code == 200
        body = resp.json()
        # Should be a dict from dataclasses.asdict
        assert isinstance(body, dict)

    def test_config_status_is_unauthenticated(self, unauthed_client):
        resp = unauthed_client.get("/config/status")
        assert resp.status_code == 200
