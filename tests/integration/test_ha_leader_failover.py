"""
Integration tests for HA leader election failover.

This test module exercises leader election failover in the HA Docker compose stack.
It validates that when the current leader is killed, another replica takes over
leadership and the load balancer continues to serve requests.

Test Scenario:
1. Boot docker-compose.ha.yml with leader election enabled
2. Wait for all services to be healthy
3. Query Postgres to determine which gateway holds the leader lease
4. Stop/kill the leader gateway container
5. Assert a different replica becomes leader within bounded timeout
6. Assert the nginx load balancer continues to serve health checks

Prerequisites:
    - Docker and docker compose CLI installed
    - Ports 4000, 4001, 8080, 5432 available (or override via env vars)

Usage:
    pytest tests/integration/test_ha_leader_failover.py -v

Environment Variables:
    HA_LB_URL: Load balancer URL (default: http://localhost:8080)
    HA_GATEWAY1_URL: Gateway 1 URL (default: http://localhost:4000)
    HA_GATEWAY2_URL: Gateway 2 URL (default: http://localhost:4001)
    HA_POSTGRES_HOST: PostgreSQL host (default: localhost)
    HA_POSTGRES_PORT: PostgreSQL port (default: 5432)
    HA_MASTER_KEY: Master API key (default: test-ha-master-key)
    HA_DOCKER_STARTUP_TIMEOUT: Startup timeout in seconds (default: 180)
    HA_FAILOVER_TIMEOUT: Failover timeout in seconds (default: 60)
    HA_SKIP_DOCKER_LIFECYCLE: Set to "true" to skip docker compose up/down
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest

# Optional: psycopg2 for direct DB queries (we'll fall back if not available)
try:
    import psycopg2

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

# Requests for HTTP health checks
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# =============================================================================
# Test Configuration
# =============================================================================

# Load balancer (nginx) URL
HA_LB_URL = os.getenv("HA_LB_URL", "http://localhost:8080")

# Individual gateway URLs (for direct checks)
HA_GATEWAY1_URL = os.getenv("HA_GATEWAY1_URL", "http://localhost:4000")
HA_GATEWAY2_URL = os.getenv("HA_GATEWAY2_URL", "http://localhost:4001")

# PostgreSQL connection settings
HA_POSTGRES_HOST = os.getenv("HA_POSTGRES_HOST", "localhost")
HA_POSTGRES_PORT = int(os.getenv("HA_POSTGRES_PORT", "5432"))
HA_POSTGRES_USER = os.getenv("HA_POSTGRES_USER", "litellm")
HA_POSTGRES_PASSWORD = os.getenv("HA_POSTGRES_PASSWORD", "litellm_password")
HA_POSTGRES_DB = os.getenv("HA_POSTGRES_DB", "litellm")

# Master API key
HA_MASTER_KEY = os.getenv("HA_MASTER_KEY", "test-ha-master-key")

# Timeouts
DOCKER_STARTUP_TIMEOUT = int(os.getenv("HA_DOCKER_STARTUP_TIMEOUT", "180"))
FAILOVER_TIMEOUT = int(os.getenv("HA_FAILOVER_TIMEOUT", "60"))
HTTP_TIMEOUT = 10
HEALTH_CHECK_INTERVAL = 2

# Skip docker lifecycle if already running (for faster local development)
SKIP_DOCKER_LIFECYCLE = os.getenv("HA_SKIP_DOCKER_LIFECYCLE", "").lower() == "true"

# Project root (where docker-compose.ha.yml lives)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Leader election table and lock name (from leader_election.py)
LEADER_LOCK_NAME = "config_sync"

# Container names from docker-compose.ha.yml
CONTAINER_GATEWAY_1 = "litellm-gateway-1"
CONTAINER_GATEWAY_2 = "litellm-gateway-2"


# =============================================================================
# Helper Functions
# =============================================================================


def is_docker_available() -> bool:
    """Check if docker daemon is available and running."""
    docker_cmd = shutil.which("docker")
    if not docker_cmd:
        return False

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


def wait_for_url_healthy(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = DOCKER_STARTUP_TIMEOUT,
    interval: int = HEALTH_CHECK_INTERVAL,
    expected_codes: tuple[int, ...] = (200,),
) -> bool:
    """
    Wait for a URL to return an expected HTTP status code.

    Args:
        url: URL to poll
        headers: Optional HTTP headers
        timeout: Maximum seconds to wait
        interval: Seconds between polls
        expected_codes: Tuple of acceptable HTTP status codes

    Returns:
        True if URL returned expected code within timeout, False otherwise
    """
    if not REQUESTS_AVAILABLE:
        return False

    start_time = time.time()
    last_error = None

    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            if resp.status_code in expected_codes:
                return True
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            last_error = str(e)

        time.sleep(interval)

    print(f"[HA Failover] URL {url} did not become healthy. Last error: {last_error}")
    return False


def wait_for_postgres_healthy(
    host: str = HA_POSTGRES_HOST,
    port: int = HA_POSTGRES_PORT,
    timeout: int = DOCKER_STARTUP_TIMEOUT,
) -> bool:
    """Wait for PostgreSQL to be accepting connections."""
    if not PSYCOPG2_AVAILABLE:
        # Fall back to simple port check
        start_time = time.time()
        while time.time() - start_time < timeout:
            if is_port_open(host, port):
                return True
            time.sleep(2)
        return False

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=HA_POSTGRES_USER,
                password=HA_POSTGRES_PASSWORD,
                dbname=HA_POSTGRES_DB,
                connect_timeout=5,
            )
            conn.close()
            return True
        except Exception:
            pass
        time.sleep(2)
    return False


def get_current_leader_holder_id() -> str | None:
    """
    Query the Postgres leader election table to get the current leader's holder_id.

    Returns:
        holder_id string if a valid leader exists, None otherwise
    """
    if not PSYCOPG2_AVAILABLE:
        print("[HA Failover] psycopg2 not installed, cannot query leader from DB")
        return None

    try:
        conn = psycopg2.connect(
            host=HA_POSTGRES_HOST,
            port=HA_POSTGRES_PORT,
            user=HA_POSTGRES_USER,
            password=HA_POSTGRES_PASSWORD,
            dbname=HA_POSTGRES_DB,
            connect_timeout=5,
        )
        try:
            with conn.cursor() as cur:
                # Query for valid (non-expired) leader lease
                cur.execute(
                    """
                    SELECT holder_id, expires_at
                    FROM config_sync_leader
                    WHERE lock_name = %s AND expires_at > NOW()
                    """,
                    (LEADER_LOCK_NAME,),
                )
                row = cur.fetchone()
                if row:
                    return row[0]  # holder_id
                return None
        finally:
            conn.close()
    except Exception as e:
        print(f"[HA Failover] Error querying leader: {e}")
        return None


def get_container_name_from_holder_id(holder_id: str) -> str | None:
    """
    Map a holder_id to a container name.

    The holder_id format is typically: {hostname}-{uuid8}
    The hostname is typically the container hostname which matches the container name.
    """
    if not holder_id:
        return None

    # The holder_id is typically like "litellm-gateway-1-abcd1234"
    # We need to extract the container name prefix
    for container in [CONTAINER_GATEWAY_1, CONTAINER_GATEWAY_2]:
        if holder_id.startswith(container):
            return container

    # Fallback: try to determine from docker inspect
    for container in [CONTAINER_GATEWAY_1, CONTAINER_GATEWAY_2]:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.Config.Hostname}}", container],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                hostname = result.stdout.strip()
                if holder_id.startswith(hostname):
                    return container
        except (subprocess.TimeoutExpired, OSError):
            pass

    return None


def stop_container(container_name: str) -> bool:
    """Stop a docker container."""
    try:
        result = subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def kill_container(container_name: str) -> bool:
    """Force kill a docker container (simulates crash)."""
    try:
        result = subprocess.run(
            ["docker", "kill", container_name],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def start_container(container_name: str) -> bool:
    """Start a stopped docker container."""
    try:
        result = subprocess.run(
            ["docker", "start", container_name],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def get_container_status(container_name: str) -> str:
    """Get the status of a docker container."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def docker_compose_up() -> bool:
    """
    Start the docker-compose.ha.yml stack with leader election enabled.

    This adds the necessary environment variables for HA leader election.
    """
    compose_file = PROJECT_ROOT / "docker-compose.ha.yml"
    if not compose_file.exists():
        print(f"[HA Failover] Compose file not found: {compose_file}")
        return False

    # Environment for HA leader election
    env = os.environ.copy()
    env.update({
        "LITELLM_MASTER_KEY": HA_MASTER_KEY,
        "POSTGRES_PASSWORD": HA_POSTGRES_PASSWORD,
        "POSTGRES_USER": HA_POSTGRES_USER,
        "POSTGRES_DB": HA_POSTGRES_DB,
        # Enable leader election mode
        "LLMROUTER_HA_MODE": "leader_election",
        # Faster lease times for quicker failover testing
        "LLMROUTER_CONFIG_SYNC_LEASE_SECONDS": "15",
        "LLMROUTER_CONFIG_SYNC_RENEW_INTERVAL_SECONDS": "5",
    })

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
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            timeout=300,  # 5 minutes for build + startup
        )
        if result.returncode != 0:
            print(f"[HA Failover] docker compose up failed: {result.stderr.decode()}")
            return False
        return True
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[HA Failover] docker compose up error: {e}")
        return False


def docker_compose_down() -> bool:
    """Stop and remove the docker-compose.ha.yml stack."""
    compose_file = PROJECT_ROOT / "docker-compose.ha.yml"
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


def docker_compose_logs(service: str | None = None, tail: int = 100) -> str:
    """Get logs from the docker-compose stack."""
    compose_file = PROJECT_ROOT / "docker-compose.ha.yml"
    cmd = ["docker", "compose", "-f", str(compose_file), "logs", "--tail", str(tail)]
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
        return result.stdout + result.stderr
    except (subprocess.TimeoutExpired, OSError):
        return ""


# =============================================================================
# Skip Conditions
# =============================================================================

# Skip entire module if docker is not available
docker_available = is_docker_available() and is_docker_compose_available()

pytestmark = [
    pytest.mark.skipif(
        not docker_available,
        reason="Docker or docker compose not available. Install Docker to run these tests.",
    ),
    pytest.mark.skipif(
        not REQUESTS_AVAILABLE,
        reason="requests library not installed. Install with: pip install requests",
    ),
    pytest.mark.skipif(
        not PSYCOPG2_AVAILABLE,
        reason="psycopg2 not installed. Install with: pip install psycopg2-binary",
    ),
    pytest.mark.integration,
    pytest.mark.ha,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def ha_stack() -> Generator[bool, None, None]:
    """
    Manage the docker-compose.ha.yml stack lifecycle.

    This fixture:
    1. Starts the HA stack before all tests in the module
    2. Waits for all required services to become healthy
    3. Yields control to tests
    4. Stops and removes the stack after all tests complete

    Yields:
        True if stack is ready, False otherwise
    """
    if SKIP_DOCKER_LIFECYCLE:
        print("[HA Failover] Skipping docker lifecycle (HA_SKIP_DOCKER_LIFECYCLE=true)")
        yield True
        return

    # Ensure clean state
    print("\n[HA Failover] Cleaning up any existing HA stack...")
    docker_compose_down()

    # Start the stack
    print("[HA Failover] Starting docker-compose.ha.yml stack...")
    if not docker_compose_up():
        print("[HA Failover] Failed to start docker compose stack")
        print(f"[HA Failover] Logs:\n{docker_compose_logs()}")
        yield False
        docker_compose_down()
        return

    # Wait for PostgreSQL
    print(f"[HA Failover] Waiting for PostgreSQL at {HA_POSTGRES_HOST}:{HA_POSTGRES_PORT}...")
    if not wait_for_postgres_healthy():
        print("[HA Failover] PostgreSQL did not become healthy")
        print(f"[HA Failover] Postgres logs:\n{docker_compose_logs('postgres')}")
        docker_compose_down()
        yield False
        return

    # Wait for both gateways to be healthy
    auth_headers = {"Authorization": f"Bearer {HA_MASTER_KEY}"}

    print(f"[HA Failover] Waiting for Gateway 1 at {HA_GATEWAY1_URL}...")
    gateway1_healthy = wait_for_url_healthy(
        f"{HA_GATEWAY1_URL}/health/liveliness",
        headers=auth_headers,
        timeout=DOCKER_STARTUP_TIMEOUT,
    )
    if not gateway1_healthy:
        print(f"[HA Failover] Gateway 1 did not become healthy within {DOCKER_STARTUP_TIMEOUT}s")
        print(f"[HA Failover] Gateway 1 logs:\n{docker_compose_logs('litellm-gateway-1')}")
        docker_compose_down()
        yield False
        return

    print(f"[HA Failover] Waiting for Gateway 2 at {HA_GATEWAY2_URL}...")
    gateway2_healthy = wait_for_url_healthy(
        f"{HA_GATEWAY2_URL}/health/liveliness",
        headers=auth_headers,
        timeout=60,  # Shorter timeout since stack is already up
    )
    if not gateway2_healthy:
        print(f"[HA Failover] Gateway 2 did not become healthy")
        print(f"[HA Failover] Gateway 2 logs:\n{docker_compose_logs('litellm-gateway-2')}")
        docker_compose_down()
        yield False
        return

    # Wait for nginx load balancer
    print(f"[HA Failover] Waiting for nginx LB at {HA_LB_URL}...")
    lb_healthy = wait_for_url_healthy(
        f"{HA_LB_URL}/health",
        headers=auth_headers,
        timeout=30,
    )
    if not lb_healthy:
        print("[HA Failover] Nginx LB did not become healthy")
        print(f"[HA Failover] Nginx logs:\n{docker_compose_logs('nginx')}")
        docker_compose_down()
        yield False
        return

    # Wait for leader election to establish (give time for leases to be acquired)
    print("[HA Failover] Waiting for leader election to establish...")
    time.sleep(10)

    print("[HA Failover] All services are healthy!")
    yield True

    # Teardown: always stop the stack
    print("\n[HA Failover] Stopping docker compose stack...")
    docker_compose_down()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return authorization headers for API calls."""
    return {"Authorization": f"Bearer {HA_MASTER_KEY}"}


@pytest.fixture
def restart_killed_containers(ha_stack: bool) -> Generator[None, None, None]:
    """
    Fixture that ensures any stopped/killed containers are restarted after test.
    """
    yield
    if not ha_stack:
        return

    # Restart both gateways if they were stopped
    for container in [CONTAINER_GATEWAY_1, CONTAINER_GATEWAY_2]:
        status = get_container_status(container)
        if status in ("exited", "dead"):
            print(f"[HA Failover] Restarting {container} after test...")
            start_container(container)
            time.sleep(5)  # Give time to start


# =============================================================================
# Test Classes
# =============================================================================


class TestHAStackHealth:
    """Basic health checks for the HA stack."""

    def test_gateway_1_is_healthy(
        self,
        ha_stack: bool,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that Gateway 1 responds to health checks."""
        assert ha_stack, "HA stack not ready"

        resp = requests.get(
            f"{HA_GATEWAY1_URL}/health/liveliness",
            headers=auth_headers,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Gateway 1 health check failed: {resp.status_code} - {resp.text}"
        )

    def test_gateway_2_is_healthy(
        self,
        ha_stack: bool,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that Gateway 2 responds to health checks."""
        assert ha_stack, "HA stack not ready"

        resp = requests.get(
            f"{HA_GATEWAY2_URL}/health/liveliness",
            headers=auth_headers,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Gateway 2 health check failed: {resp.status_code} - {resp.text}"
        )

    def test_load_balancer_is_healthy(
        self,
        ha_stack: bool,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that the nginx load balancer responds to health checks."""
        assert ha_stack, "HA stack not ready"

        resp = requests.get(
            f"{HA_LB_URL}/health",
            headers=auth_headers,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Load balancer health check failed: {resp.status_code} - {resp.text}"
        )

    def test_leader_election_established(self, ha_stack: bool) -> None:
        """Test that a leader has been elected."""
        assert ha_stack, "HA stack not ready"

        leader_id = get_current_leader_holder_id()
        assert leader_id is not None, (
            "No leader found in config_sync_leader table. "
            "Leader election may not be enabled or working."
        )
        print(f"[HA Failover] Current leader: {leader_id}")


class TestHALeaderFailover:
    """Tests for leader election failover scenarios."""

    def test_failover_after_leader_killed(
        self,
        ha_stack: bool,
        auth_headers: dict[str, str],
        restart_killed_containers: None,
    ) -> None:
        """
        Test that leadership fails over when the current leader is killed.

        Scenario:
        1. Identify the current leader
        2. Kill the leader container
        3. Wait for a new leader to be elected
        4. Verify the new leader is different from the original
        5. Verify the load balancer still serves requests
        """
        assert ha_stack, "HA stack not ready"

        # Step 1: Get current leader
        original_leader_id = get_current_leader_holder_id()
        assert original_leader_id is not None, (
            "No leader found. Cannot test failover without an established leader."
        )
        print(f"[HA Failover] Original leader: {original_leader_id}")

        # Determine which container is the leader
        leader_container = get_container_name_from_holder_id(original_leader_id)
        if leader_container is None:
            # Fallback: try to determine leader from gateway health endpoints
            # by checking which one reports is_leader=True
            for container, url in [
                (CONTAINER_GATEWAY_1, HA_GATEWAY1_URL),
                (CONTAINER_GATEWAY_2, HA_GATEWAY2_URL),
            ]:
                try:
                    resp = requests.get(
                        f"{url}/config/sync/status",
                        headers=auth_headers,
                        timeout=HTTP_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("leader_election", {}).get("is_leader"):
                            leader_container = container
                            break
                except requests.RequestException:
                    pass

        assert leader_container is not None, (
            f"Could not determine which container is the leader. "
            f"Leader ID: {original_leader_id}"
        )
        print(f"[HA Failover] Leader container: {leader_container}")

        # Step 2: Kill the leader container
        print(f"[HA Failover] Killing leader container: {leader_container}")
        kill_success = kill_container(leader_container)
        assert kill_success, f"Failed to kill container {leader_container}"

        # Verify container is stopped
        time.sleep(2)
        status = get_container_status(leader_container)
        assert status in ("exited", "dead"), (
            f"Container {leader_container} is still running: {status}"
        )

        # Step 3: Wait for new leader election
        print(f"[HA Failover] Waiting up to {FAILOVER_TIMEOUT}s for new leader...")
        new_leader_id = None
        start_time = time.time()

        while time.time() - start_time < FAILOVER_TIMEOUT:
            new_leader_id = get_current_leader_holder_id()
            if new_leader_id is not None and new_leader_id != original_leader_id:
                break
            time.sleep(2)

        # Step 4: Verify new leader
        assert new_leader_id is not None, (
            f"No new leader elected within {FAILOVER_TIMEOUT}s. "
            f"The remaining gateway may not have acquired leadership."
        )
        assert new_leader_id != original_leader_id, (
            f"Leader ID did not change after killing leader container. "
            f"Original: {original_leader_id}, Current: {new_leader_id}"
        )
        print(f"[HA Failover] New leader elected: {new_leader_id}")

        # Step 5: Verify load balancer still serves requests
        print("[HA Failover] Verifying load balancer still serves health checks...")
        lb_healthy = wait_for_url_healthy(
            f"{HA_LB_URL}/health",
            headers=auth_headers,
            timeout=30,
            interval=1,
        )
        assert lb_healthy, (
            "Load balancer stopped serving requests after leader failover. "
            "The remaining gateway may not be handling requests correctly."
        )

        # Also verify we can list models through the LB
        resp = requests.get(
            f"{HA_LB_URL}/v1/models",
            headers=auth_headers,
            timeout=HTTP_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"Could not list models through LB after failover: "
            f"{resp.status_code} - {resp.text}"
        )
        print("[HA Failover] Load balancer verified healthy after failover")

    def test_failover_after_leader_graceful_stop(
        self,
        ha_stack: bool,
        auth_headers: dict[str, str],
        restart_killed_containers: None,
    ) -> None:
        """
        Test failover with graceful stop (allows leader to release lease).

        This tests the graceful shutdown path where the leader has a chance
        to release its lease before stopping.
        """
        assert ha_stack, "HA stack not ready"

        # Get current leader
        original_leader_id = get_current_leader_holder_id()
        assert original_leader_id is not None, "No leader found"

        leader_container = get_container_name_from_holder_id(original_leader_id)
        if leader_container is None:
            pytest.skip("Could not determine leader container")

        print(f"[HA Failover] Gracefully stopping leader: {leader_container}")

        # Graceful stop (allows SIGTERM handling)
        stop_success = stop_container(leader_container)
        assert stop_success, f"Failed to stop container {leader_container}"

        # Wait for new leader
        print(f"[HA Failover] Waiting for new leader after graceful stop...")
        new_leader_id = None
        start_time = time.time()

        while time.time() - start_time < FAILOVER_TIMEOUT:
            new_leader_id = get_current_leader_holder_id()
            if new_leader_id is not None and new_leader_id != original_leader_id:
                break
            time.sleep(2)

        assert new_leader_id is not None and new_leader_id != original_leader_id, (
            f"No new leader after graceful stop within {FAILOVER_TIMEOUT}s"
        )
        print(f"[HA Failover] New leader after graceful stop: {new_leader_id}")

    def test_load_balancer_serves_during_failover(
        self,
        ha_stack: bool,
        auth_headers: dict[str, str],
        restart_killed_containers: None,
    ) -> None:
        """
        Test that the load balancer continues to serve requests during failover.

        This test makes continuous requests while killing the leader to verify
        availability is maintained.
        """
        assert ha_stack, "HA stack not ready"

        original_leader_id = get_current_leader_holder_id()
        assert original_leader_id is not None, "No leader found"

        leader_container = get_container_name_from_holder_id(original_leader_id)
        if leader_container is None:
            pytest.skip("Could not determine leader container")

        # Track request success/failures during failover
        success_count = 0
        failure_count = 0
        total_requests = 0

        print(f"[HA Failover] Testing LB availability during failover...")
        print(f"[HA Failover] Killing leader container: {leader_container}")

        # Kill the leader
        kill_container(leader_container)

        # Make requests for the next 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            try:
                resp = requests.get(
                    f"{HA_LB_URL}/health",
                    headers=auth_headers,
                    timeout=5,
                )
                total_requests += 1
                if resp.status_code == 200:
                    success_count += 1
                else:
                    failure_count += 1
            except requests.RequestException:
                total_requests += 1
                failure_count += 1

            time.sleep(0.5)  # 2 requests per second

        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        print(
            f"[HA Failover] Results: {success_count}/{total_requests} successful "
            f"({success_rate:.1f}%)"
        )

        # Allow some failures during the brief transition period,
        # but overall availability should be >80%
        assert success_rate >= 80, (
            f"Load balancer availability dropped below 80% during failover. "
            f"Success rate: {success_rate:.1f}% "
            f"({success_count}/{total_requests} requests)"
        )


class TestHALeaderElectionDetails:
    """Detailed tests for leader election state and behavior."""

    def test_leader_lease_table_exists(self, ha_stack: bool) -> None:
        """Test that the config_sync_leader table was created."""
        assert ha_stack, "HA stack not ready"

        conn = psycopg2.connect(
            host=HA_POSTGRES_HOST,
            port=HA_POSTGRES_PORT,
            user=HA_POSTGRES_USER,
            password=HA_POSTGRES_PASSWORD,
            dbname=HA_POSTGRES_DB,
            connect_timeout=5,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'config_sync_leader'
                    )
                    """
                )
                exists = cur.fetchone()[0]
                assert exists, "config_sync_leader table does not exist"
        finally:
            conn.close()

    def test_lease_has_valid_expiry(self, ha_stack: bool) -> None:
        """Test that the leader lease has a valid future expiry time."""
        assert ha_stack, "HA stack not ready"

        conn = psycopg2.connect(
            host=HA_POSTGRES_HOST,
            port=HA_POSTGRES_PORT,
            user=HA_POSTGRES_USER,
            password=HA_POSTGRES_PASSWORD,
            dbname=HA_POSTGRES_DB,
            connect_timeout=5,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT holder_id, acquired_at, expires_at, (expires_at > NOW()) as is_valid
                    FROM config_sync_leader
                    WHERE lock_name = %s
                    """,
                    (LEADER_LOCK_NAME,),
                )
                row = cur.fetchone()
                assert row is not None, "No leader lease found in database"

                holder_id, acquired_at, expires_at, is_valid = row
                assert is_valid, (
                    f"Leader lease is expired. "
                    f"Acquired: {acquired_at}, Expires: {expires_at}"
                )
                print(
                    f"[HA Failover] Lease valid: holder={holder_id}, "
                    f"acquired={acquired_at}, expires={expires_at}"
                )
        finally:
            conn.close()

    def test_only_one_leader(self, ha_stack: bool) -> None:
        """Test that only one gateway reports as leader."""
        assert ha_stack, "HA stack not ready"

        auth_headers = {"Authorization": f"Bearer {HA_MASTER_KEY}"}
        leaders = []

        for container, url in [
            (CONTAINER_GATEWAY_1, HA_GATEWAY1_URL),
            (CONTAINER_GATEWAY_2, HA_GATEWAY2_URL),
        ]:
            status = get_container_status(container)
            if status != "running":
                continue

            try:
                resp = requests.get(
                    f"{url}/config/sync/status",
                    headers=auth_headers,
                    timeout=HTTP_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("leader_election", {}).get("is_leader"):
                        leaders.append(container)
            except requests.RequestException:
                pass

        assert len(leaders) <= 1, (
            f"Multiple gateways report as leader: {leaders}. "
            "Split-brain detected!"
        )
        if len(leaders) == 1:
            print(f"[HA Failover] Single leader verified: {leaders[0]}")
        else:
            print("[HA Failover] No gateway reports as leader (may be in transition)")
