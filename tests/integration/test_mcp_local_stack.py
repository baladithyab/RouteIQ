"""
Integration tests for MCP Gateway with Docker Compose local-test stack.

This test module manages its own docker-compose lifecycle:
- Boots docker-compose.local-test.yml for the duration of the test session
- Waits for all required services to be healthy
- Exercises MCP management + tool invocation end-to-end
- Tears down the stack after all tests complete

Prerequisites:
    - Docker and docker compose CLI installed
    - Ports 4010 and 9100 available (or override via env vars)

Usage:
    pytest tests/integration/test_mcp_local_stack.py -v

Environment Variables:
    GATEWAY_URL: Override gateway URL (default: http://localhost:4010)
    MASTER_KEY: Override master key (default: local-dev-master-key)
    ADMIN_API_KEY: Override admin key (default: local-dev-admin-key)
    MCP_STUB_URL: Override MCP stub server URL (default: http://localhost:9100)
    MCP_DOCKER_STARTUP_TIMEOUT: Override startup timeout in seconds (default: 120)
    MCP_SKIP_DOCKER_LIFECYCLE: Set to "true" to skip docker compose up/down
                               (useful when stack is already running)
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import pytest
import requests

if TYPE_CHECKING:
    from typing import Any

# =============================================================================
# Test Configuration
# =============================================================================

# Gateway configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:4010")
MASTER_KEY = os.getenv("MASTER_KEY", "local-dev-master-key")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "local-dev-admin-key")

# MCP stub server configuration (the container URL for gateway to reach it)
MCP_STUB_CONTAINER_URL = os.getenv(
    "MCP_STUB_CONTAINER_URL", "http://mcp-stub-server:9100/mcp"
)
# MCP stub server local URL (for direct test verification)
MCP_STUB_LOCAL_URL = os.getenv("MCP_STUB_URL", "http://localhost:9100")

# Timeouts
DOCKER_STARTUP_TIMEOUT = int(os.getenv("MCP_DOCKER_STARTUP_TIMEOUT", "120"))
HTTP_TIMEOUT = 10
HEALTH_CHECK_INTERVAL = 2

# Skip docker lifecycle if already running (for faster local development)
SKIP_DOCKER_LIFECYCLE = os.getenv("MCP_SKIP_DOCKER_LIFECYCLE", "").lower() == "true"

# Project root (where docker-compose.local-test.yml lives)
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# Helper Functions
# =============================================================================


def is_docker_available() -> bool:
    """Check if docker and docker compose are available."""
    docker_cmd = shutil.which("docker")
    if not docker_cmd:
        return False

    # Check if docker daemon is running
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def is_docker_compose_available() -> bool:
    """Check if docker compose (v2) is available."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open on the given host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except (OSError, socket.error):
        return False


def wait_for_health(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = DOCKER_STARTUP_TIMEOUT,
    interval: int = HEALTH_CHECK_INTERVAL,
) -> bool:
    """Wait for a service to become healthy by polling a URL."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    return False


def docker_compose_up() -> bool:
    """Start the docker-compose.local-test.yml stack."""
    compose_file = PROJECT_ROOT / "docker-compose.local-test.yml"
    if not compose_file.exists():
        return False

    try:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "up",
                "-d",
                "--build",
                "--wait",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=300,  # 5 minutes for build + startup
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def docker_compose_down() -> bool:
    """Stop and remove the docker-compose.local-test.yml stack."""
    compose_file = PROJECT_ROOT / "docker-compose.local-test.yml"
    if not compose_file.exists():
        return False

    try:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "down",
                "-v",  # Remove volumes for clean state
                "--remove-orphans",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def docker_compose_logs(service: str | None = None) -> str:
    """Get logs from the docker-compose stack."""
    compose_file = PROJECT_ROOT / "docker-compose.local-test.yml"
    cmd = ["docker", "compose", "-f", str(compose_file), "logs", "--tail", "100"]
    if service:
        cmd.append(service)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=30,
            text=True,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


# =============================================================================
# Skip Conditions
# =============================================================================

# Skip entire module if docker is not available
docker_available = is_docker_available() and is_docker_compose_available()
pytestmark = pytest.mark.skipif(
    not docker_available,
    reason="Docker or docker compose not available. Install Docker to run these tests.",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def docker_stack() -> Generator[bool, None, None]:
    """
    Manage the docker-compose.local-test.yml stack lifecycle.

    This fixture:
    1. Starts the stack before all tests in the module
    2. Waits for all required services to become healthy
    3. Yields control to tests
    4. Stops and removes the stack after all tests complete

    Yields:
        True if stack is ready, False otherwise
    """
    if SKIP_DOCKER_LIFECYCLE:
        # Assume stack is already running
        yield True
        return

    # Start the stack
    print("\n[MCP Integration] Starting docker-compose.local-test.yml stack...")
    if not docker_compose_up():
        print("[MCP Integration] Failed to start docker compose stack")
        print(f"[MCP Integration] Logs:\n{docker_compose_logs()}")
        yield False
        return

    # Wait for gateway health
    print(f"[MCP Integration] Waiting for gateway at {GATEWAY_URL}...")
    auth_headers = {"Authorization": f"Bearer {MASTER_KEY}"}
    gateway_healthy = wait_for_health(
        f"{GATEWAY_URL}/health",
        headers=auth_headers,
        timeout=DOCKER_STARTUP_TIMEOUT,
    )

    if not gateway_healthy:
        print(f"[MCP Integration] Gateway did not become healthy within {DOCKER_STARTUP_TIMEOUT}s")
        print(f"[MCP Integration] Gateway logs:\n{docker_compose_logs('litellm-gateway')}")
        docker_compose_down()
        yield False
        return

    # Wait for MCP stub server health
    print(f"[MCP Integration] Waiting for MCP stub server at {MCP_STUB_LOCAL_URL}...")
    stub_healthy = wait_for_health(
        f"{MCP_STUB_LOCAL_URL}/mcp/health",
        timeout=30,
    )

    if not stub_healthy:
        print("[MCP Integration] MCP stub server did not become healthy within 30s")
        print(f"[MCP Integration] MCP stub logs:\n{docker_compose_logs('mcp-stub-server')}")
        docker_compose_down()
        yield False
        return

    print("[MCP Integration] All services are healthy!")
    yield True

    # Teardown: stop the stack
    print("\n[MCP Integration] Stopping docker compose stack...")
    docker_compose_down()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return authorization headers for API calls."""
    return {"Authorization": f"Bearer {MASTER_KEY}"}


@pytest.fixture
def admin_headers() -> dict[str, str]:
    """Return headers for admin API calls (control plane operations)."""
    return {
        "Authorization": f"Bearer {MASTER_KEY}",
        "X-Admin-API-Key": ADMIN_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestMCPRegistryEndpoints:
    """Test MCP registry and discovery endpoints."""

    def test_registry_json_returns_200(
        self, docker_stack: bool, auth_headers: dict[str, str]
    ) -> None:
        """Test GET /v1/llmrouter/mcp/registry.json returns 200."""
        assert docker_stack, "Docker stack not ready"

        resp = requests.get(
            f"{GATEWAY_URL}/v1/llmrouter/mcp/registry.json",
            headers=auth_headers,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert isinstance(data, dict), "Registry should return a JSON object"

    def test_mcp_servers_list_returns_200(
        self, docker_stack: bool, auth_headers: dict[str, str]
    ) -> None:
        """Test GET /llmrouter/mcp/servers returns 200."""
        assert docker_stack, "Docker stack not ready"

        resp = requests.get(
            f"{GATEWAY_URL}/llmrouter/mcp/servers",
            headers=auth_headers,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )


class TestMCPServerRegistration:
    """Test MCP server registration with admin auth."""

    @pytest.fixture(autouse=True)
    def cleanup_registered_server(
        self, docker_stack: bool, admin_headers: dict[str, str]
    ) -> Generator[None, None, None]:
        """Cleanup any previously registered test server."""
        yield
        # Cleanup after test
        if docker_stack:
            try:
                requests.delete(
                    f"{GATEWAY_URL}/llmrouter/mcp/servers/test-stub-mcp",
                    headers=admin_headers,
                    timeout=HTTP_TIMEOUT,
                )
            except requests.RequestException:
                pass  # Best effort cleanup

    def test_server_registration_returns_200(
        self, docker_stack: bool, admin_headers: dict[str, str]
    ) -> None:
        """Test POST /llmrouter/mcp/servers returns 200 with valid admin auth."""
        assert docker_stack, "Docker stack not ready"

        register_payload = {
            "server_id": "test-stub-mcp",
            "name": "Integration Test Stub",
            "url": MCP_STUB_CONTAINER_URL,
            "transport": "streamable_http",
            "tools": ["stub.echo", "stub.sum"],
            "resources": ["stub://resource/demo"],
            "metadata": {"environment": "integration-test"},
        }

        resp = requests.post(
            f"{GATEWAY_URL}/llmrouter/mcp/servers",
            headers=admin_headers,
            json=register_payload,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Server registration failed with {resp.status_code}: {resp.text}"
        )

    def test_registered_server_appears_in_list(
        self, docker_stack: bool, admin_headers: dict[str, str], auth_headers: dict[str, str]
    ) -> None:
        """Test that a registered server appears in the servers list."""
        assert docker_stack, "Docker stack not ready"

        # Register the server
        register_payload = {
            "server_id": "test-stub-mcp",
            "name": "Integration Test Stub",
            "url": MCP_STUB_CONTAINER_URL,
            "transport": "streamable_http",
            "tools": ["stub.echo"],
        }

        resp = requests.post(
            f"{GATEWAY_URL}/llmrouter/mcp/servers",
            headers=admin_headers,
            json=register_payload,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, f"Registration failed: {resp.text}"

        # Allow time for registration to propagate
        time.sleep(1)

        # Verify it appears in the list
        list_resp = requests.get(
            f"{GATEWAY_URL}/llmrouter/mcp/servers",
            headers=auth_headers,
            timeout=HTTP_TIMEOUT,
        )
        assert list_resp.status_code == 200
        data = list_resp.json()

        # Check if our server is in the list (response format may vary)
        server_ids = []
        if isinstance(data, list):
            server_ids = [s.get("server_id", s.get("id", "")) for s in data]
        elif isinstance(data, dict):
            servers = data.get("servers", data.get("data", []))
            server_ids = [s.get("server_id", s.get("id", "")) for s in servers]

        assert "test-stub-mcp" in server_ids, (
            f"Registered server not found in list. Got: {data}"
        )


class TestMCPToolInvocation:
    """Test MCP tool invocation via the gateway."""

    @pytest.fixture(autouse=True)
    def register_mcp_server(
        self, docker_stack: bool, admin_headers: dict[str, str]
    ) -> Generator[None, None, None]:
        """Register the MCP stub server before tool invocation tests."""
        if not docker_stack:
            yield
            return

        register_payload = {
            "server_id": "stub-for-tool-test",
            "name": "Stub for Tool Test",
            "url": MCP_STUB_CONTAINER_URL,
            "transport": "streamable_http",
            "tools": ["stub.echo", "stub.sum"],
        }

        try:
            resp = requests.post(
                f"{GATEWAY_URL}/llmrouter/mcp/servers",
                headers=admin_headers,
                json=register_payload,
                timeout=HTTP_TIMEOUT,
            )
            if resp.status_code != 200:
                print(f"[MCP Integration] Warning: Server registration returned {resp.status_code}")
        except requests.RequestException as e:
            print(f"[MCP Integration] Warning: Server registration failed: {e}")

        # Allow time for registration
        time.sleep(2)

        yield

        # Cleanup
        try:
            requests.delete(
                f"{GATEWAY_URL}/llmrouter/mcp/servers/stub-for-tool-test",
                headers=admin_headers,
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException:
            pass

    def test_stub_echo_tool_returns_200_with_expected_payload(
        self, docker_stack: bool, admin_headers: dict[str, str]
    ) -> None:
        """Test POST /llmrouter/mcp/tools/call with stub.echo returns 200 and echoes input."""
        assert docker_stack, "Docker stack not ready"

        tool_call_payload = {
            "tool_name": "stub.echo",
            "arguments": {"text": "Hello from integration test"},
        }

        resp = requests.post(
            f"{GATEWAY_URL}/llmrouter/mcp/tools/call",
            headers=admin_headers,
            json=tool_call_payload,
            timeout=HTTP_TIMEOUT,
        )

        assert resp.status_code == 200, (
            f"Tool invocation failed with {resp.status_code}: {resp.text}"
        )

        data = resp.json()
        # Verify echo response structure
        assert "result" in data or "echo" in str(data), (
            f"Expected echo result in response. Got: {data}"
        )

        # Check that our input text appears somewhere in the response
        response_str = str(data)
        assert "Hello from integration test" in response_str, (
            f"Echo text not found in response. Got: {data}"
        )

    def test_stub_sum_tool_returns_200_with_correct_sum(
        self, docker_stack: bool, admin_headers: dict[str, str]
    ) -> None:
        """Test POST /llmrouter/mcp/tools/call with stub.sum returns correct sum."""
        assert docker_stack, "Docker stack not ready"

        tool_call_payload = {
            "tool_name": "stub.sum",
            "arguments": {"values": [1, 2, 3, 4, 5]},
        }

        resp = requests.post(
            f"{GATEWAY_URL}/llmrouter/mcp/tools/call",
            headers=admin_headers,
            json=tool_call_payload,
            timeout=HTTP_TIMEOUT,
        )

        assert resp.status_code == 200, (
            f"Tool invocation failed with {resp.status_code}: {resp.text}"
        )

        data = resp.json()
        # Navigate to result - structure may be nested
        result = data
        if "result" in data:
            result = data["result"]
        if "result" in result:
            result = result["result"]

        # Check for sum value
        if isinstance(result, dict) and "sum" in result:
            assert result["sum"] == 15.0, f"Expected sum=15.0, got {result['sum']}"
        else:
            # Check the raw value
            assert "15" in str(data), f"Sum of 15 not found in response: {data}"

    def test_unknown_tool_returns_404(
        self, docker_stack: bool, admin_headers: dict[str, str]
    ) -> None:
        """Test POST /llmrouter/mcp/tools/call with unknown tool returns 404."""
        assert docker_stack, "Docker stack not ready"

        tool_call_payload = {
            "tool_name": "stub.nonexistent",
            "arguments": {},
        }

        resp = requests.post(
            f"{GATEWAY_URL}/llmrouter/mcp/tools/call",
            headers=admin_headers,
            json=tool_call_payload,
            timeout=HTTP_TIMEOUT,
        )

        assert resp.status_code == 404, (
            f"Expected 404 for unknown tool, got {resp.status_code}: {resp.text}"
        )


class TestMCPToolsAndResources:
    """Test MCP tools and resources aggregation endpoints."""

    @pytest.fixture(autouse=True)
    def register_mcp_server(
        self, docker_stack: bool, admin_headers: dict[str, str]
    ) -> Generator[None, None, None]:
        """Register the MCP stub server for tools/resources tests."""
        if not docker_stack:
            yield
            return

        register_payload = {
            "server_id": "stub-for-aggregation",
            "name": "Stub for Aggregation Test",
            "url": MCP_STUB_CONTAINER_URL,
            "transport": "streamable_http",
            "tools": ["stub.echo", "stub.sum"],
            "resources": ["stub://resource/demo"],
        }

        try:
            requests.post(
                f"{GATEWAY_URL}/llmrouter/mcp/servers",
                headers=admin_headers,
                json=register_payload,
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException:
            pass

        time.sleep(1)
        yield

        try:
            requests.delete(
                f"{GATEWAY_URL}/llmrouter/mcp/servers/stub-for-aggregation",
                headers=admin_headers,
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException:
            pass

    def test_mcp_tools_list_returns_200(
        self, docker_stack: bool, auth_headers: dict[str, str]
    ) -> None:
        """Test GET /llmrouter/mcp/tools returns 200."""
        assert docker_stack, "Docker stack not ready"

        resp = requests.get(
            f"{GATEWAY_URL}/llmrouter/mcp/tools",
            headers=auth_headers,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

    def test_mcp_resources_list_returns_200(
        self, docker_stack: bool, auth_headers: dict[str, str]
    ) -> None:
        """Test GET /llmrouter/mcp/resources returns 200."""
        assert docker_stack, "Docker stack not ready"

        resp = requests.get(
            f"{GATEWAY_URL}/llmrouter/mcp/resources",
            headers=auth_headers,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )


class TestMCPStubServerDirect:
    """Direct tests against the MCP stub server (bypassing gateway)."""

    def test_stub_server_health(self, docker_stack: bool) -> None:
        """Test that MCP stub server is healthy."""
        assert docker_stack, "Docker stack not ready"

        resp = requests.get(
            f"{MCP_STUB_LOCAL_URL}/mcp/health",
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Stub server health check failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert data.get("status") == "ok", f"Unexpected health response: {data}"

    def test_stub_server_tools_list(self, docker_stack: bool) -> None:
        """Test that MCP stub server lists its tools."""
        assert docker_stack, "Docker stack not ready"

        resp = requests.get(
            f"{MCP_STUB_LOCAL_URL}/mcp/tools",
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data, f"Expected 'tools' key in response: {data}"

        tool_names = [t.get("name", "") for t in data["tools"]]
        assert "stub.echo" in tool_names, f"stub.echo not in tools: {tool_names}"
        assert "stub.sum" in tool_names, f"stub.sum not in tools: {tool_names}"

    def test_stub_server_direct_echo(self, docker_stack: bool) -> None:
        """Test direct tool call to MCP stub server."""
        assert docker_stack, "Docker stack not ready"

        resp = requests.post(
            f"{MCP_STUB_LOCAL_URL}/mcp/tools/call",
            json={
                "tool_name": "stub.echo",
                "arguments": {"text": "direct test"},
            },
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "success", f"Unexpected response: {data}"
        assert "direct test" in str(data), f"Echo text not in response: {data}"
