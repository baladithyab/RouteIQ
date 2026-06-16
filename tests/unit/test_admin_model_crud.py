"""Unit tests for admin model CRUD + real health status (RouteIQ-eb2d / c8d5).

Covers:
  * POST / PUT / DELETE mutate the in-memory model catalog (litellm.model_list),
    with the live router sync stubbed out (no real LiteLLM router required).
  * Validation: duplicate add -> 409, missing model param -> 400, missing
    target -> 404.
  * Admin-auth is enforced on the CRUD endpoints (via the admin_router) — an
    unauthenticated request is rejected 401/403.
  * Model Overview Status reflects REAL deployment health derived from the
    per-provider circuit-breaker state (closed=active, open=unavailable,
    half_open=degraded), not a cosmetic always-active.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from litellm_llmrouter.gateway.app import create_standalone_app

_TEST_ADMIN_KEY = "sk-riq-test-admin-key-model-crud"


@pytest.fixture()
def fake_model_list(monkeypatch):
    """Point ``litellm.model_list`` at an isolated list for each test.

    The router sync is stubbed so the catalog mutation is exercised without a
    live LiteLLM router.
    """
    import litellm

    monkeypatch.setattr(litellm, "model_list", [], raising=False)
    # Stub the router sync helpers so they are no-ops (no live router in tests).
    monkeypatch.setattr(
        "litellm_llmrouter.routes.admin_ui._sync_router_add",
        lambda entry: None,
    )
    monkeypatch.setattr(
        "litellm_llmrouter.routes.admin_ui._sync_router_remove",
        lambda model_name: None,
    )
    return litellm.model_list


@pytest.fixture()
def client(monkeypatch, fake_model_list):
    """Admin-authed TestClient backed by the standalone gateway app."""
    monkeypatch.setenv("ADMIN_API_KEYS", _TEST_ADMIN_KEY)
    monkeypatch.setenv("ADMIN_AUTH_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_ENV", "test")

    app = create_standalone_app(enable_plugins=False, enable_resilience=False)
    c = TestClient(app, raise_server_exceptions=False)
    _original_request = c.request

    def _authed_request(*args, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("X-Admin-API-Key", _TEST_ADMIN_KEY)
        return _original_request(*args, headers=headers, **kwargs)

    c.request = _authed_request
    return c


@pytest.fixture()
def unauthed_client(monkeypatch, fake_model_list):
    """Client with NO admin key header — used to test 401/403."""
    monkeypatch.setenv("ADMIN_API_KEYS", _TEST_ADMIN_KEY)
    monkeypatch.setenv("ADMIN_AUTH_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_ENV", "test")
    app = create_standalone_app(enable_plugins=False, enable_resilience=False)
    return TestClient(app, raise_server_exceptions=False)


# --- CRUD: add ---------------------------------------------------------------


class TestAddModel:
    def test_add_appends_to_model_list(self, client, fake_model_list):
        resp = client.post(
            "/api/v1/routeiq/models",
            json={
                "model_name": "claude-3-5-sonnet",
                "litellm_params": {"model": "anthropic/claude-3-5-sonnet"},
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["action"] == "added"
        assert body["model_name"] == "claude-3-5-sonnet"
        assert body["model_count"] == 1
        # The in-memory catalog was actually mutated.
        assert len(fake_model_list) == 1
        assert fake_model_list[0]["model_name"] == "claude-3-5-sonnet"
        assert (
            fake_model_list[0]["litellm_params"]["model"]
            == "anthropic/claude-3-5-sonnet"
        )

    def test_add_duplicate_returns_409(self, client, fake_model_list):
        payload = {
            "model_name": "gpt-4o",
            "litellm_params": {"model": "openai/gpt-4o"},
        }
        assert client.post("/api/v1/routeiq/models", json=payload).status_code == 201
        dup = client.post("/api/v1/routeiq/models", json=payload)
        assert dup.status_code == 409
        # No second entry was appended.
        assert len(fake_model_list) == 1

    def test_add_missing_model_param_returns_400(self, client, fake_model_list):
        resp = client.post(
            "/api/v1/routeiq/models",
            json={"model_name": "bad", "litellm_params": {}},
        )
        assert resp.status_code == 400
        assert len(fake_model_list) == 0

    def test_add_carries_model_info(self, client, fake_model_list):
        resp = client.post(
            "/api/v1/routeiq/models",
            json={
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"id": "deploy-1", "mode": "chat"},
            },
        )
        assert resp.status_code == 201
        assert fake_model_list[0]["model_info"]["id"] == "deploy-1"


# --- CRUD: update ------------------------------------------------------------


class TestUpdateModel:
    def test_update_replaces_entry_in_place(self, client, fake_model_list):
        client.post(
            "/api/v1/routeiq/models",
            json={"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}},
        )
        resp = client.put(
            "/api/v1/routeiq/models/gpt-4o",
            json={
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o-2024-11-20"},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["action"] == "updated"
        assert len(fake_model_list) == 1
        assert (
            fake_model_list[0]["litellm_params"]["model"] == "openai/gpt-4o-2024-11-20"
        )

    def test_update_can_rename(self, client, fake_model_list):
        client.post(
            "/api/v1/routeiq/models",
            json={"model_name": "old", "litellm_params": {"model": "openai/gpt-4o"}},
        )
        resp = client.put(
            "/api/v1/routeiq/models/old",
            json={"model_name": "new", "litellm_params": {"model": "openai/gpt-4o"}},
        )
        assert resp.status_code == 200
        assert fake_model_list[0]["model_name"] == "new"

    def test_update_missing_returns_404(self, client, fake_model_list):
        resp = client.put(
            "/api/v1/routeiq/models/ghost",
            json={"model_name": "ghost", "litellm_params": {"model": "openai/gpt-4o"}},
        )
        assert resp.status_code == 404


# --- CRUD: delete ------------------------------------------------------------


class TestDeleteModel:
    def test_delete_removes_entry(self, client, fake_model_list):
        client.post(
            "/api/v1/routeiq/models",
            json={"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}},
        )
        resp = client.delete("/api/v1/routeiq/models/gpt-4o")
        assert resp.status_code == 200
        assert resp.json()["action"] == "removed"
        assert resp.json()["model_count"] == 0
        assert len(fake_model_list) == 0

    def test_delete_missing_returns_404(self, client, fake_model_list):
        resp = client.delete("/api/v1/routeiq/models/ghost")
        assert resp.status_code == 404


# --- Admin-auth enforcement --------------------------------------------------


class TestModelCrudAdminAuth:
    def test_add_requires_admin_auth(self, unauthed_client):
        resp = unauthed_client.post(
            "/api/v1/routeiq/models",
            json={"model_name": "x", "litellm_params": {"model": "openai/gpt-4o"}},
        )
        assert resp.status_code in (401, 403)

    def test_update_requires_admin_auth(self, unauthed_client):
        resp = unauthed_client.put(
            "/api/v1/routeiq/models/x",
            json={"model_name": "x", "litellm_params": {"model": "openai/gpt-4o"}},
        )
        assert resp.status_code in (401, 403)

    def test_delete_requires_admin_auth(self, unauthed_client):
        resp = unauthed_client.delete("/api/v1/routeiq/models/x")
        assert resp.status_code in (401, 403)


# --- Real health status (RouteIQ-c8d5) --------------------------------------


class TestModelOverviewHealth:
    def test_status_active_when_no_breaker(self, client):
        """A provider with no breaker (no failures) reports active."""
        import litellm

        litellm.model_list = [
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o"},
            }
        ]
        resp = client.get("/api/v1/routeiq/models")
        assert resp.status_code == 200
        models = resp.json()
        assert models[0]["status"] == "active"

    def test_status_reflects_open_breaker_as_unavailable(self, client):
        """An OPEN provider breaker maps the model to unavailable."""
        import litellm

        litellm.model_list = [
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o"},
            },
            {
                "model_name": "claude-3-5-sonnet",
                "litellm_params": {"model": "anthropic/claude-3-5-sonnet"},
            },
        ]
        # openai breaker open (unavailable), anthropic healthy (active).
        with patch(
            "litellm_llmrouter.routes.admin_ui._build_provider_status_map",
            return_value={"openai": "unavailable"},
        ):
            resp = client.get("/api/v1/routeiq/models")
        assert resp.status_code == 200
        by_name = {m["model_name"]: m for m in resp.json()}
        assert by_name["gpt-4o"]["status"] == "unavailable"
        assert by_name["claude-3-5-sonnet"]["status"] == "active"

    def test_status_half_open_breaker_is_degraded(self, client):
        import litellm

        litellm.model_list = [
            {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}}
        ]
        with patch(
            "litellm_llmrouter.routes.admin_ui._build_provider_status_map",
            return_value={"openai": "degraded"},
        ):
            resp = client.get("/api/v1/routeiq/models")
        assert resp.json()[0]["status"] == "degraded"
