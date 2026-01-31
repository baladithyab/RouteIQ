"""
E2E API Validation Tests for RouteIQ Gateway.

These tests validate critical API endpoints against a running Docker compose stack.
Run against docker-compose.local-test.yml on http://localhost:4010

Test Coverage:
- Admin vs User AuthZ boundary
- SSRF/outbound URL policy enforcement  
- Skills discovery endpoint paths
- HA compose stack validation

Usage:
    pytest tests/integration/test_e2e_api_validation.py -v
    
Environment Variables:
    GATEWAY_URL: Gateway base URL (default: http://localhost:4010)
    ADMIN_KEY: Admin API key (default: sk-test-master-key)
"""

import os
import pytest
import requests
from typing import Dict, Any
import time

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:4010")
ADMIN_KEY = os.getenv("ADMIN_KEY", "sk-test-master-key")
REQUEST_TIMEOUT = 5  # seconds


class TestAuthZBoundary:
    """Test admin vs user/invalid key authorization boundaries."""

    @pytest.fixture
    def admin_headers(self) -> Dict[str, str]:
        """Admin API key headers."""
        return {"Authorization": f"Bearer {ADMIN_KEY}"}

    @pytest.fixture
    def invalid_headers(self) -> Dict[str, str]:
        """Invalid API key headers (should be rejected)."""
        return {"Authorization": "Bearer sk-invalid-user-key-12345"}

    def test_router_reload_with_invalid_key(self, invalid_headers):
        """
        Admin endpoint /router/reload should reject invalid keys with 401/403.
        
        Validates:
        - Invalid keys are rejected
        - Error response is properly formatted
        """
        response = requests.post(
            f"{GATEWAY_URL}/router/reload",
            headers=invalid_headers,
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code in [401, 403], \
            f"Expected 401/403 for invalid key, got {response.status_code}"
        
        # Validate error response structure
        data = response.json()
        assert "error" in data or "detail" in data, \
            "Error response should contain error/detail field"

    def test_router_reload_with_admin_key(self, admin_headers):
        """
        Admin endpoint /router/reload should accept admin keys with 200.
        
        Validates:
        - Admin keys are accepted
        - Reload operation succeeds
        """
        response = requests.post(
            f"{GATEWAY_URL}/router/reload",
            headers=admin_headers,
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code == 200, \
            f"Expected 200 for admin key, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "status" in data or "message" in data, \
            "Success response should contain status/message"

    def test_config_reload_with_invalid_key(self, invalid_headers):
        """
        Admin endpoint /config/reload should reject invalid keys with 401/403.
        """
        response = requests.post(
            f"{GATEWAY_URL}/config/reload",
            headers=invalid_headers,
            json={"force_sync": False},
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code in [401, 403], \
            f"Expected 401/403 for invalid key, got {response.status_code}"

    def test_config_reload_with_admin_key(self, admin_headers):
        """
        Admin endpoint /config/reload should accept admin keys with 200.
        """
        response = requests.post(
            f"{GATEWAY_URL}/config/reload",
            headers=admin_headers,
            json={"force_sync": False},
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code == 200, \
            f"Expected 200 for admin key, got {response.status_code}: {response.text}"

    def test_mcp_server_registration_with_invalid_key(self, invalid_headers):
        """
        MCP server registration should reject invalid keys with 401/403.
        """
        response = requests.post(
            f"{GATEWAY_URL}/llmrouter/mcp/servers",
            headers=invalid_headers,
            json={
                "server_id": "test-auth-check",
                "name": "Test Server",
                "url": "https://example.com/mcp",
                "transport": "http"
            },
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code in [401, 403], \
            f"Expected 401/403 for invalid key, got {response.status_code}"


class TestSSRFProtection:
    """Test SSRF/outbound URL policy enforcement."""

    @pytest.fixture
    def admin_headers(self) -> Dict[str, str]:
        """Admin API key headers."""
        return {
            "Authorization": f"Bearer {ADMIN_KEY}",
            "Content-Type": "application/json"
        }

    @pytest.mark.parametrize("private_url,description", [
        ("http://127.0.0.1:9999/mcp", "loopback address"),
        ("http://10.0.0.1:8080/mcp", "private class A network"),
        ("http://192.168.1.1:8080/mcp", "private class C network"),
        ("http://172.16.0.1:8080/mcp", "private class B network"),
        ("http://localhost:8080/mcp", "localhost hostname"),
    ])
    def test_mcp_registration_blocks_private_ips(
        self, admin_headers, private_url, description
    ):
        """
        MCP server registration should block private/internal URLs.
        
        Validates:
        - SSRF protection blocks requests to private IPs
        - Error response indicates blocking reason
        - Even admin keys cannot bypass SSRF protection
        
        Ref: src/litellm_llmrouter/url_security.py:validate_outbound_url()
        """
        payload = {
            "server_id": f"test-ssrf-{hash(private_url)}",
            "name": f"SSRF Test - {description}",
            "url": private_url,
            "transport": "http"
        }
        
        response = requests.post(
            f"{GATEWAY_URL}/llmrouter/mcp/servers",
            headers=admin_headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        
        # Should be blocked with 400/403/422
        assert response.status_code in [400, 403, 422], \
            f"Private IP {private_url} should be blocked, got {response.status_code}: {response.text}"
        
        # Error message should mention SSRF or private IP
        response_text = response.text.lower()
        assert any(keyword in response_text for keyword in [
            "ssrf", "private", "blocked", "not allowed", "invalid"
        ]), f"Error message should mention SSRF/blocking: {response.text}"

    def test_mcp_registration_allows_public_urls(self, admin_headers):
        """
        MCP server registration should allow public URLs (when reachable).
        
        Note: This may fail with connection errors if the URL is unreachable,
        but should NOT fail with SSRF blocking.
        """
        payload = {
            "server_id": "test-public-url",
            "name": "Public URL Test",
            "url": "https://example.com/mcp",
            "transport": "http"
        }
        
        response = requests.post(
            f"{GATEWAY_URL}/llmrouter/mcp/servers",
            headers=admin_headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        
        # Should NOT be blocked by SSRF (may fail for other reasons)
        assert response.status_code not in [403], \
            f"Public URL should not be SSRF-blocked, got {response.status_code}: {response.text}"


class TestSkillsDiscovery:
    """Test skills discovery endpoint paths."""

    @pytest.fixture
    def admin_headers(self) -> Dict[str, str]:
        """Admin API key headers."""
        return {"Authorization": f"Bearer {ADMIN_KEY}"}

    def test_v1_skills_endpoint(self, admin_headers):
        """
        Test LiteLLM native /v1/skills endpoint.
        
        Expected: Either 200 with skills list, or 404/501 if not implemented.
        """
        response = requests.get(
            f"{GATEWAY_URL}/v1/skills",
            headers=admin_headers,
            timeout=REQUEST_TIMEOUT,
        )
        
        # Acceptable responses: 200 (implemented), 404/501 (not implemented)
        assert response.status_code in [200, 404, 501], \
            f"Unexpected status {response.status_code} for /v1/skills"
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (dict, list)), \
                "Skills response should be dict or list"

    def test_well_known_skills_index(self, admin_headers):
        """
        Test /.well-known/skills/index.json (skills discovery plugin).
        
        Expected: Either 200 with index, or 404 if plugin not enabled.
        """
        response = requests.get(
            f"{GATEWAY_URL}/.well-known/skills/index.json",
            headers=admin_headers,
            timeout=REQUEST_TIMEOUT,
        )
        
        # Acceptable: 200 (plugin enabled), 404 (not enabled)
        assert response.status_code in [200, 404], \
            f"Unexpected status {response.status_code} for /.well-known/skills/index.json"
        
        if response.status_code == 200:
            data = response.json()
            assert "skills" in data, \
                "Skills index should have 'skills' field"


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_live_probe(self):
        """
        Test /_health/live (liveness probe).
        
        Validates:
        - Endpoint is accessible without auth
        - Returns 200 status
        - Response is sanitized JSON (no exceptions leaking)
        """
        response = requests.get(
            f"{GATEWAY_URL}/_health/live",
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code == 200, \
            f"Liveness probe should return 200, got {response.status_code}"
        
        data = response.json()
        assert "status" in data, \
            "Health response should have status field"
        assert data["status"] == "alive", \
            "Liveness status should be 'alive'"
        
        # Should not contain stack traces or raw exceptions
        response_text = response.text.lower()
        assert "traceback" not in response_text, \
            "Health response should not leak stack traces"

    def test_health_ready_probe(self):
        """
        Test /_health/ready (readiness probe).
        
        Validates:
        - Endpoint is accessible without auth
        - Returns 200 status when ready
        - Shows component health status
        """
        response = requests.get(
            f"{GATEWAY_URL}/_health/ready",
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code in [200, 503], \
            f"Readiness probe should return 200/503, got {response.status_code}"
        
        data = response.json()
        assert "status" in data, \
            "Health response should have status field"
        
        if response.status_code == 200:
            assert "checks" in data, \
                "Ready response should include component checks"


class TestMCPEndpoints:
    """Test MCP gateway endpoints."""

    @pytest.fixture
    def admin_headers(self) -> Dict[str, str]:
        """Admin API key headers."""
        return {"Authorization": f"Bearer {ADMIN_KEY}"}

    def test_list_mcp_servers(self, admin_headers):
        """
        Test GET /llmrouter/mcp/servers returns server list.
        """
        response = requests.get(
            f"{GATEWAY_URL}/llmrouter/mcp/servers",
            headers=admin_headers,
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "servers" in data, \
            "Response should have 'servers' field"
        assert isinstance(data["servers"], list), \
            "Servers should be a list"

    def test_list_mcp_tools(self, admin_headers):
        """
        Test GET /llmrouter/mcp/tools returns tools list.
        """
        response = requests.get(
            f"{GATEWAY_URL}/llmrouter/mcp/tools",
            headers=admin_headers,
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "tools" in data, \
            "Response should have 'tools' field"


class TestRouterEndpoints:
    """Test router management endpoints."""

    @pytest.fixture
    def admin_headers(self) -> Dict[str, str]:
        """Admin API key headers."""
        return {"Authorization": f"Bearer {ADMIN_KEY}"}

    def test_router_info(self, admin_headers):
        """
        Test GET /router/info returns routing strategy information.
        """
        response = requests.get(
            f"{GATEWAY_URL}/router/info",
            headers=admin_headers,
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        expected_fields = ["registered_strategies", "strategy_count"]
        for field in expected_fields:
            assert field in data, \
                f"Router info should include '{field}'"


# Mark all tests as integration tests
pytestmark = pytest.mark.integration
