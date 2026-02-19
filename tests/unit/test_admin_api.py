"""Unit tests for admin UI API endpoints."""

import os
import time
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm_llmrouter.routes.admin_ui import (
    get_gateway_status,
    get_models,
    get_routing_config,
    get_routing_stats,
    update_routing_config,
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons before each test to avoid cross-test contamination."""
    from litellm_llmrouter.centroid_routing import reset_centroid_strategy
    from litellm_llmrouter.strategy_registry import reset_routing_singletons

    reset_routing_singletons()
    reset_centroid_strategy()
    yield
    reset_routing_singletons()
    reset_centroid_strategy()


@pytest.fixture
def app():
    """Create a test FastAPI app with admin UI routes."""
    test_app = FastAPI()
    test_app.get("/api/v1/routeiq/status")(get_gateway_status)
    test_app.get("/api/v1/routeiq/routing/stats")(get_routing_stats)
    test_app.get("/api/v1/routeiq/routing/config")(get_routing_config)
    test_app.post("/api/v1/routeiq/routing/config")(update_routing_config)
    test_app.get("/api/v1/routeiq/models")(get_models)
    return test_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestGatewayStatus:
    """Tests for GET /api/v1/routeiq/status."""

    def test_returns_status(self, client):
        """Test basic status response."""
        response = client.get("/api/v1/routeiq/status")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "uptime_seconds" in data
        assert "uptime_formatted" in data
        assert "worker_count" in data
        assert "feature_flags" in data
        assert isinstance(data["feature_flags"], dict)

    def test_uptime_increases(self, client):
        """Test that uptime increases between calls."""
        r1 = client.get("/api/v1/routeiq/status")
        time.sleep(0.1)
        r2 = client.get("/api/v1/routeiq/status")
        assert r2.json()["uptime_seconds"] >= r1.json()["uptime_seconds"]

    def test_worker_count_from_env(self, client):
        """Test worker count reads from env."""
        with patch.dict(os.environ, {"ROUTEIQ_WORKERS": "4"}):
            response = client.get("/api/v1/routeiq/status")
            assert response.json()["worker_count"] == 4

    def test_feature_flags(self, client):
        """Test feature flags are populated."""
        with patch.dict(
            os.environ,
            {
                "MCP_GATEWAY_ENABLED": "true",
                "A2A_GATEWAY_ENABLED": "false",
            },
        ):
            response = client.get("/api/v1/routeiq/status")
            flags = response.json()["feature_flags"]
            assert flags["mcp_gateway"] is True
            assert flags["a2a_gateway"] is False

    def test_version_present(self, client):
        """Test version is returned."""
        response = client.get("/api/v1/routeiq/status")
        assert response.json()["version"] == "0.2.0"

    def test_routing_profile_default(self, client):
        """Test default routing profile."""
        response = client.get("/api/v1/routeiq/status")
        assert response.json()["routing_profile"] == "auto"

    def test_centroid_routing_enabled_default(self, client):
        """Test centroid routing is enabled by default."""
        response = client.get("/api/v1/routeiq/status")
        assert response.json()["centroid_routing_enabled"] is True


class TestRoutingStats:
    """Tests for GET /api/v1/routeiq/routing/stats."""

    def test_returns_stats(self, client):
        """Test basic stats response."""
        response = client.get("/api/v1/routeiq/routing/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_decisions" in data
        assert "strategy_distribution" in data
        assert "centroid_decisions" in data
        assert "average_latency_ms" in data

    def test_mvp_defaults(self, client):
        """Test MVP defaults are zeros."""
        response = client.get("/api/v1/routeiq/routing/stats")
        data = response.json()
        assert data["total_decisions"] == 0
        assert data["centroid_decisions"] == 0
        assert data["average_latency_ms"] == 0.0

    def test_profile_distribution_empty(self, client):
        """Test profile_distribution is empty for MVP."""
        response = client.get("/api/v1/routeiq/routing/stats")
        assert response.json()["profile_distribution"] == {}


class TestRoutingConfig:
    """Tests for GET/POST /api/v1/routeiq/routing/config."""

    def test_get_config(self, client):
        """Test basic config response."""
        response = client.get("/api/v1/routeiq/routing/config")
        assert response.status_code == 200
        data = response.json()
        assert "active_strategy" in data
        assert "available_strategies" in data
        assert "routing_profile" in data
        assert "centroid_routing_enabled" in data
        assert "ab_testing" in data

    def test_update_routing_profile(self, client):
        """Test updating routing profile."""
        response = client.post(
            "/api/v1/routeiq/routing/config",
            json={"routing_profile": "eco"},
        )
        assert response.status_code == 200
        assert response.json()["routing_profile"] == "eco"

    def test_invalid_routing_profile(self, client):
        """Test invalid routing profile returns 400."""
        response = client.post(
            "/api/v1/routeiq/routing/config",
            json={"routing_profile": "invalid"},
        )
        assert response.status_code == 400

    def test_update_centroid_routing(self, client):
        """Test toggling centroid routing."""
        response = client.post(
            "/api/v1/routeiq/routing/config",
            json={"centroid_routing_enabled": False},
        )
        assert response.status_code == 200
        assert response.json()["centroid_routing_enabled"] is False

    def test_update_active_strategy_not_found(self, client):
        """Test setting non-existent strategy returns 400."""
        response = client.post(
            "/api/v1/routeiq/routing/config",
            json={"active_strategy": "nonexistent-strategy"},
        )
        assert response.status_code == 400

    def test_ab_testing_defaults(self, client):
        """Test A/B testing defaults when no experiment configured."""
        response = client.get("/api/v1/routeiq/routing/config")
        ab = response.json()["ab_testing"]
        assert ab["enabled"] is False
        assert ab["weights"] == {}

    def test_empty_update_returns_config(self, client):
        """Test POST with empty body returns current config."""
        response = client.post(
            "/api/v1/routeiq/routing/config",
            json={},
        )
        assert response.status_code == 200
        data = response.json()
        assert "active_strategy" in data
        assert "routing_profile" in data

    def test_update_multiple_fields(self, client):
        """Test updating routing profile and centroid together."""
        response = client.post(
            "/api/v1/routeiq/routing/config",
            json={
                "routing_profile": "premium",
                "centroid_routing_enabled": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["routing_profile"] == "premium"
        assert data["centroid_routing_enabled"] is True


class TestModels:
    """Tests for GET /api/v1/routeiq/models."""

    def test_returns_models_list(self, client):
        """Test models endpoint returns a list."""
        response = client.get("/api/v1/routeiq/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_model_structure_with_mock(self, client):
        """Test model response structure when models exist."""
        import litellm_llmrouter.routes.admin_ui as admin_ui_module

        with patch.object(
            admin_ui_module,
            "_get_models",
            return_value=[
                {
                    "model_name": "claude-3-opus",
                    "provider": "anthropic",
                    "model_id": "anthropic/claude-3-opus-20240229",
                    "status": "active",
                }
            ],
        ):
            response = client.get("/api/v1/routeiq/models")
            assert response.status_code == 200
            models = response.json()
            assert len(models) == 1
            assert models[0]["model_name"] == "claude-3-opus"
            assert models[0]["provider"] == "anthropic"
            assert models[0]["model_id"] == "anthropic/claude-3-opus-20240229"
            assert models[0]["status"] == "active"

    def test_empty_models_list(self, client):
        """Test empty models list when nothing configured."""
        import litellm_llmrouter.routes.admin_ui as admin_ui_module

        with patch.object(admin_ui_module, "_get_models", return_value=[]):
            response = client.get("/api/v1/routeiq/models")
            assert response.status_code == 200
            assert response.json() == []


class TestFormatUptime:
    """Tests for _format_uptime helper."""

    def test_seconds_only(self):
        from litellm_llmrouter.routes.admin_ui import _format_uptime

        assert _format_uptime(45) == "45s"

    def test_minutes_and_seconds(self):
        from litellm_llmrouter.routes.admin_ui import _format_uptime

        assert _format_uptime(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        from litellm_llmrouter.routes.admin_ui import _format_uptime

        assert _format_uptime(3661) == "1h 1m 1s"

    def test_days(self):
        from litellm_llmrouter.routes.admin_ui import _format_uptime

        assert _format_uptime(90061) == "1d 1h 1m 1s"

    def test_zero(self):
        from litellm_llmrouter.routes.admin_ui import _format_uptime

        assert _format_uptime(0) == "0s"
