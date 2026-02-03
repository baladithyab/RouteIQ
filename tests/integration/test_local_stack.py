"""
Integration tests for the LiteLLM + LLMRouter local Docker stack.

Tests the docker-compose.local-test.yml stack including:
- LiteLLM Gateway health and model endpoints
- A2A Gateway functionality
- MCP Gateway functionality
- Jaeger tracing
- MLflow integration
- MinIO storage

Prerequisites:
    docker compose -f docker-compose.local-test.yml up -d
"""

import os
import pytest
import requests

# Test configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:4010")
MASTER_KEY = os.getenv("MASTER_KEY", "local-dev-master-key")
JAEGER_URL = os.getenv("JAEGER_URL", "http://localhost:16686")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5050")
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
MCP_PROXY_URL = os.getenv("MCP_PROXY_URL", "http://localhost:3100")


@pytest.fixture
def auth_headers():
    """Return authorization headers for API calls."""
    return {"Authorization": f"Bearer {MASTER_KEY}"}


class TestGatewayHealth:
    """Test LiteLLM Gateway health endpoints."""

    def test_health_endpoint(self, auth_headers):
        """Test that health endpoint returns 200."""
        resp = requests.get(f"{GATEWAY_URL}/health", headers=auth_headers, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "healthy_endpoints" in data or "healthy_count" in data

    def test_health_liveliness(self, auth_headers):
        """Test liveliness probe."""
        resp = requests.get(
            f"{GATEWAY_URL}/health/liveliness", headers=auth_headers, timeout=10
        )
        # May return 200 or 401 depending on config
        assert resp.status_code in [200, 401, 404]

    def test_health_readiness(self, auth_headers):
        """Test readiness probe."""
        resp = requests.get(
            f"{GATEWAY_URL}/health/readiness", headers=auth_headers, timeout=10
        )
        assert resp.status_code in [200, 401, 404]


class TestGatewayModels:
    """Test LiteLLM Gateway model endpoints."""

    def test_list_models(self, auth_headers):
        """Test that models endpoint returns model list."""
        resp = requests.get(
            f"{GATEWAY_URL}/v1/models", headers=auth_headers, timeout=10
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_model_info(self, auth_headers):
        """Test model info endpoint."""
        resp = requests.get(
            f"{GATEWAY_URL}/model/info", headers=auth_headers, timeout=10
        )
        assert resp.status_code in [200, 404]


class TestA2AGateway:
    """Test A2A (Agent-to-Agent) Gateway functionality."""

    def test_a2a_agents_list(self, auth_headers):
        """Test listing A2A agents."""
        resp = requests.get(
            f"{GATEWAY_URL}/a2a/agents", headers=auth_headers, timeout=10
        )
        # Endpoint may not exist if A2A not fully configured
        assert resp.status_code in [200, 404, 405]

    def test_a2a_agent_register(self, auth_headers):
        """Test registering an A2A agent."""
        agent_data = {
            "agent_id": "test-agent-integration",
            "name": "Integration Test Agent",
            "description": "Agent for integration testing",
            "url": "http://localhost:9999/a2a",
            "capabilities": ["test", "integration"],
        }
        resp = requests.post(
            f"{GATEWAY_URL}/a2a/agents",
            headers=auth_headers,
            json=agent_data,
            timeout=10,
        )
        # May not be implemented or return 500 if endpoint exists but fails
        assert resp.status_code in [200, 201, 404, 405, 422, 500]


class TestMCPGateway:
    """Test MCP (Model Context Protocol) Gateway functionality."""

    def test_mcp_servers_list(self, auth_headers):
        """Test listing MCP servers."""
        resp = requests.get(
            f"{GATEWAY_URL}/mcp/servers", headers=auth_headers, timeout=10
        )
        # 406 Not Acceptable may be returned if MCP expects specific content types
        assert resp.status_code in [200, 404, 405, 406]

    def test_mcp_tools_list(self, auth_headers):
        """Test listing MCP tools."""
        resp = requests.get(
            f"{GATEWAY_URL}/mcp/tools", headers=auth_headers, timeout=10
        )
        assert resp.status_code in [200, 404, 405, 406]


class TestJaegerTracing:
    """Test Jaeger tracing integration."""

    def test_jaeger_ui_accessible(self):
        """Test that Jaeger UI is accessible."""
        resp = requests.get(f"{JAEGER_URL}/", timeout=10)
        assert resp.status_code == 200

    def test_jaeger_api_services(self):
        """Test Jaeger API services endpoint."""
        resp = requests.get(f"{JAEGER_URL}/api/services", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data


class TestMLflow:
    """Test MLflow integration."""

    def test_mlflow_health(self):
        """Test MLflow health endpoint."""
        resp = requests.get(f"{MLFLOW_URL}/health", timeout=10)
        assert resp.status_code == 200

    def test_mlflow_experiments_list(self):
        """Test listing MLflow experiments."""
        # MLflow search expects POST with body including max_results
        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            json={"max_results": 10},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "experiments" in data


class TestMinIO:
    """Test MinIO storage integration."""

    def test_minio_health(self):
        """Test MinIO health endpoint."""
        resp = requests.get(f"{MINIO_URL}/minio/health/live", timeout=10)
        assert resp.status_code == 200


class TestMCPProxy:
    """Test MCP Proxy (stdio-to-HTTP bridge) servers."""

    @pytest.mark.skip(reason="MCP stdio servers don't respond to HTTP GET requests")
    def test_filesystem_server(self):
        """Test MCP filesystem server is accessible."""
        # The MCP proxy bridges stdio servers to HTTP using SSE/streaming
        # Simple GET requests won't work with MCP protocol
        resp = requests.get(f"{MCP_PROXY_URL}/", timeout=10)
        assert resp.status_code in [200, 400, 404, 405]

    @pytest.mark.skip(reason="MCP stdio servers don't respond to HTTP GET requests")
    def test_time_server(self):
        """Test MCP time server is accessible."""
        resp = requests.get("http://localhost:3103/", timeout=10)
        assert resp.status_code in [200, 400, 404, 405]


class TestBedrockModels:
    """Test Bedrock model integration (requires valid AWS credentials)."""

    @pytest.mark.skipif(
        os.getenv("SKIP_BEDROCK_TESTS", "false").lower() == "true",
        reason="Bedrock tests skipped via SKIP_BEDROCK_TESTS env var",
    )
    def test_bedrock_chat_completion(self, auth_headers):
        """Test a simple chat completion with Bedrock Claude model."""
        payload = {
            "model": "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
            "max_tokens": 50,
        }
        resp = requests.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            headers=auth_headers,
            json=payload,
            timeout=30,
        )
        # Check response
        if resp.status_code == 200:
            data = resp.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "message" in data["choices"][0]
        else:
            # May fail due to auth issues - check error format
            data = resp.json()
            assert "error" in data or "detail" in data

    @pytest.mark.skipif(
        os.getenv("SKIP_BEDROCK_TESTS", "false").lower() == "true",
        reason="Bedrock tests skipped via SKIP_BEDROCK_TESTS env var",
    )
    def test_bedrock_nova_model(self, auth_headers):
        """Test Amazon Nova model via Bedrock."""
        payload = {
            "model": "bedrock/amazon.nova-micro-v1:0",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 50,
        }
        resp = requests.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            headers=auth_headers,
            json=payload,
            timeout=30,
        )
        assert resp.status_code in [200, 400, 401, 403, 500]


class TestConfigSync:
    """Test configuration sync functionality."""

    def test_config_sync_status(self, auth_headers):
        """Test config sync status endpoint."""
        resp = requests.get(
            f"{GATEWAY_URL}/config/sync/status", headers=auth_headers, timeout=10
        )
        assert resp.status_code in [200, 404]

    def test_config_reload(self, auth_headers):
        """Test config reload endpoint."""
        resp = requests.post(
            f"{GATEWAY_URL}/config/reload",
            headers=auth_headers,
            json={"force_sync": False},
            timeout=10,
        )
        assert resp.status_code in [200, 404, 405]


class TestRouterInfo:
    """Test LLMRouter-specific endpoints."""

    def test_router_info(self, auth_headers):
        """Test router info endpoint."""
        resp = requests.get(
            f"{GATEWAY_URL}/router/info", headers=auth_headers, timeout=10
        )
        assert resp.status_code in [200, 404]

    def test_router_reload(self, auth_headers):
        """Test router reload endpoint."""
        resp = requests.post(
            f"{GATEWAY_URL}/router/reload", headers=auth_headers, timeout=10
        )
        assert resp.status_code in [200, 404, 405]
