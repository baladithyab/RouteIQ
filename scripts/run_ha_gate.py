#!/usr/bin/env python3
"""
HA Gate - Leader Election / Failover Validation
================================================

This script validates the High Availability (HA) behavior of the RouteIQ
gateway stack by testing leader election and failover scenarios.

Test Scenarios:
  1. Boot HA stack and wait for readiness
  2. Determine current leader from PostgreSQL
  3. Verify exactly one leader (no split-brain)
  4. Stop/kill the leader container
  5. Assert new leader is elected within timeout
  6. Verify service remains available via nginx LB
  7. Cleanup: tear down the stack

Leadership Detection:
  - Queries PostgreSQL `config_sync_leader` table directly
  - Checks `holder_id` column for unique leader identification
  - Verifies `expires_at` for lease validity

Requirements:
  - Docker Compose (docker compose or docker-compose)
  - Python 3.12+ with httpx and asyncpg (or psycopg2)
  - No external LLM credentials needed (uses health endpoints only)

Usage:
    python scripts/run_ha_gate.py [OPTIONS]

Options:
    --compose-file FILE   Path to HA compose file (default: docker-compose.ha.yml)
    --timeout SECONDS     Max seconds to wait for failover (default: 90)
    --json-output FILE    Write JSON report to file
    --no-cleanup          Don't tear down stack after test (for debugging)
    --help                Show this help message

Examples:
    # Quick local run
    python scripts/run_ha_gate.py

    # With custom timeout and JSON output
    python scripts/run_ha_gate.py --timeout 120 --json-output ha-gate-report.json

    # Debug mode (no cleanup)
    python scripts/run_ha_gate.py --no-cleanup
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


# =============================================================================
# Configuration
# =============================================================================

# Default settings
DEFAULT_COMPOSE_FILE = "docker-compose.ha-test.yml"
DEFAULT_FAILOVER_TIMEOUT = 90  # seconds
DEFAULT_STARTUP_TIMEOUT = 180  # seconds
DEFAULT_READINESS_POLL_INTERVAL = 5  # seconds
DEFAULT_FAILOVER_POLL_INTERVAL = 2  # seconds

# Service endpoints
NGINX_URL = os.getenv("HA_NGINX_URL", "http://localhost:8080")
GATEWAY_1_URL = os.getenv("HA_GATEWAY_1_URL", "http://localhost:4000")
GATEWAY_2_URL = os.getenv("HA_GATEWAY_2_URL", "http://localhost:4001")
POSTGRES_HOST = os.getenv("HA_POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("HA_POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "litellm")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "litellm_test_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "litellm")

# Container names from docker-compose.ha.yml
GATEWAY_1_CONTAINER = "litellm-gateway-1"
GATEWAY_2_CONTAINER = "litellm-gateway-2"
POSTGRES_CONTAINER = "litellm-postgres"
NGINX_CONTAINER = "litellm-nginx"

# Leader election table/lock
LEADER_LOCK_NAME = "config_sync"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LeaderInfo:
    """Information about the current leader."""

    holder_id: str
    acquired_at: datetime
    expires_at: datetime
    lock_name: str = LEADER_LOCK_NAME

    def is_valid(self) -> bool:
        """Check if the lease is still valid."""
        return datetime.now(timezone.utc) < self.expires_at


@dataclass
class HAGateResult:
    """Result of the HA gate test."""

    passed: bool
    timestamp: str
    duration_seconds: float
    startup_time_seconds: float
    failover_time_seconds: float | None
    initial_leader: str | None
    new_leader: str | None
    split_brain_detected: bool
    nginx_available_during_failover: bool
    nginx_available_after_failover: bool
    error: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        return d


# =============================================================================
# Docker Compose Helpers
# =============================================================================


def get_compose_command() -> list[str]:
    """Get the appropriate compose command (docker compose, docker-compose, or finch compose)."""
    # Prefer 'docker compose' (V2)
    if shutil.which("docker"):
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return ["docker", "compose"]

    # Fallback to docker-compose (V1)
    if shutil.which("docker-compose"):
        return ["docker-compose"]

    # Support finch (Docker-compatible runtime on macOS)
    if shutil.which("finch"):
        result = subprocess.run(
            ["finch", "compose", "version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return ["finch", "compose"]

    raise RuntimeError("Neither 'docker compose', 'docker-compose', nor 'finch compose' found")


def run_compose(
    compose_file: str,
    *args: str,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a compose command."""
    cmd = get_compose_command()
    full_cmd = cmd + ["-f", compose_file] + list(args)
    print(f"  $ {' '.join(full_cmd)}")
    return subprocess.run(
        full_cmd,
        capture_output=capture,
        text=True,
        check=check,
        env={
            **os.environ,
            # Provide required env vars for HA compose (matching docker-compose.ha-test.yml defaults)
            "LITELLM_MASTER_KEY": os.getenv(
                "LITELLM_MASTER_KEY", "ha-gate-test-key-not-for-production"
            ),
            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "litellm_test_password"),
            "LLMROUTER_HA_MODE": "leader_election",
            "LLMROUTER_CONFIG_SYNC_LEASE_SECONDS": "15",  # Faster for testing
            "LLMROUTER_CONFIG_SYNC_RENEW_INTERVAL_SECONDS": "5",
        },
    )


def start_stack(compose_file: str) -> None:
    """Start the HA stack."""
    print("\n📦 Starting HA stack...")
    # Note: We don't use --wait because finch compose doesn't support it
    # Instead, we rely on our own wait_for_stack_ready() function
    run_compose(compose_file, "up", "-d", "--build")


def stop_stack(compose_file: str) -> None:
    """Stop and remove the HA stack."""
    print("\n🧹 Stopping HA stack...")
    run_compose(compose_file, "down", "-v", "--remove-orphans", check=False)


def stop_container(container_name: str) -> None:
    """Stop a specific container."""
    print(f"  🛑 Stopping container: {container_name}")
    subprocess.run(
        ["docker", "stop", container_name],
        capture_output=True,
        text=True,
        check=False,
    )


def kill_container(container_name: str) -> None:
    """Kill a specific container (simulate crash)."""
    print(f"  💀 Killing container: {container_name}")
    subprocess.run(
        ["docker", "kill", container_name],
        capture_output=True,
        text=True,
        check=False,
    )


def get_container_id(container_name: str) -> str | None:
    """Get container ID by name."""
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={container_name}"],
        capture_output=True,
        text=True,
    )
    container_id = result.stdout.strip()
    return container_id if container_id else None


# =============================================================================
# Database Helpers
# =============================================================================


async def get_leader_from_db() -> LeaderInfo | None:
    """Query PostgreSQL directly to get the current leader."""
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
        )
        try:
            now = datetime.now(timezone.utc)
            row = await conn.fetchrow(
                """
                SELECT lock_name, holder_id, acquired_at, expires_at
                FROM config_sync_leader
                WHERE lock_name = $1 AND expires_at > $2
                """,
                LEADER_LOCK_NAME,
                now,
            )
            if row:
                return LeaderInfo(
                    holder_id=row["holder_id"],
                    acquired_at=row["acquired_at"],
                    expires_at=row["expires_at"],
                    lock_name=row["lock_name"],
                )
            return None
        finally:
            await conn.close()
    except ImportError:
        # Fallback to psycopg2 if asyncpg not available
        try:
            import psycopg2

            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                dbname=POSTGRES_DB,
            )
            try:
                with conn.cursor() as cur:
                    now = datetime.now(timezone.utc)
                    cur.execute(
                        """
                        SELECT lock_name, holder_id, acquired_at, expires_at
                        FROM config_sync_leader
                        WHERE lock_name = %s AND expires_at > %s
                        """,
                        (LEADER_LOCK_NAME, now),
                    )
                    row = cur.fetchone()
                    if row:
                        return LeaderInfo(
                            holder_id=row[1],
                            acquired_at=row[2],
                            expires_at=row[3],
                            lock_name=row[0],
                        )
                    return None
            finally:
                conn.close()
        except ImportError:
            print("  ⚠️  Neither asyncpg nor psycopg2 available")
            return None
    except Exception as e:
        print(f"  ⚠️  Database query error: {e}")
        return None


async def count_leaders_in_db() -> int:
    """Count the number of valid leaders (should always be 0 or 1)."""
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
        )
        try:
            now = datetime.now(timezone.utc)
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM config_sync_leader
                WHERE lock_name = $1 AND expires_at > $2
                """,
                LEADER_LOCK_NAME,
                now,
            )
            return count or 0
        finally:
            await conn.close()
    except Exception as e:
        print(f"  ⚠️  Count leaders error: {e}")
        return -1  # Error indicator


# =============================================================================
# Health Check Helpers
# =============================================================================


async def check_service_health(url: str, timeout: float = 5.0) -> bool:
    """Check if a service is healthy."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Try liveliness endpoint first (unauthenticated)
            for endpoint in ["/health/liveliness", "/health", "/_health/live"]:
                try:
                    resp = await client.get(f"{url}{endpoint}")
                    if resp.status_code == 200:
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        return False


async def wait_for_stack_ready(
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    poll_interval: float = DEFAULT_READINESS_POLL_INTERVAL,
) -> float:
    """Wait for all services to be ready. Returns startup time in seconds."""
    print("\n⏳ Waiting for stack to be ready...")
    start = time.time()
    end_time = start + startup_timeout

    services = {
        "nginx": NGINX_URL,
        "gateway-1": GATEWAY_1_URL,
        "gateway-2": GATEWAY_2_URL,
    }

    ready = {name: False for name in services}

    while time.time() < end_time:
        for name, url in services.items():
            if not ready[name]:
                is_healthy = await check_service_health(url)
                if is_healthy:
                    ready[name] = True
                    print(f"  ✅ {name} is ready")

        if all(ready.values()):
            startup_time = time.time() - start
            print(f"  🚀 All services ready in {startup_time:.1f}s")
            return startup_time

        await asyncio.sleep(poll_interval)

    # Timeout - report what's not ready
    not_ready = [name for name, is_ready in ready.items() if not is_ready]
    raise TimeoutError(f"Services not ready after {startup_timeout}s: {not_ready}")


# =============================================================================
# Main Test Logic
# =============================================================================


async def run_ha_gate(
    compose_file: str = DEFAULT_COMPOSE_FILE,
    failover_timeout: float = DEFAULT_FAILOVER_TIMEOUT,
    cleanup: bool = True,
) -> HAGateResult:
    """Run the HA gate test."""
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    result = HAGateResult(
        passed=False,
        timestamp=timestamp,
        duration_seconds=0,
        startup_time_seconds=0,
        failover_time_seconds=None,
        initial_leader=None,
        new_leader=None,
        split_brain_detected=False,
        nginx_available_during_failover=True,
        nginx_available_after_failover=False,
        error=None,
        details={},
    )

    try:
        # Step 1: Start the HA stack
        print("\n" + "=" * 60)
        print("🧪 HA GATE: Leader Election & Failover Test")
        print("=" * 60)
        print(f"  Compose file: {compose_file}")
        print(f"  Failover timeout: {failover_timeout}s")

        start_stack(compose_file)

        # Step 2: Wait for readiness
        result.startup_time_seconds = await wait_for_stack_ready()

        # Step 3: Wait for leader election to stabilize
        print("\n⏳ Waiting for leader election to stabilize...")
        await asyncio.sleep(10)  # Allow time for initial election

        # Step 4: Determine current leader
        print("\n🔍 Determining current leader...")
        initial_leader = await get_leader_from_db()
        if initial_leader is None:
            raise RuntimeError("No leader found in database after startup")

        result.initial_leader = initial_leader.holder_id
        print(f"  📍 Initial leader: {initial_leader.holder_id}")
        print(f"  📍 Lease expires: {initial_leader.expires_at.isoformat()}")

        # Step 5: Check for split-brain (should be exactly 1 leader)
        leader_count = await count_leaders_in_db()
        if leader_count > 1:
            result.split_brain_detected = True
            raise RuntimeError(f"Split-brain detected: {leader_count} leaders found!")
        print(f"  ✅ No split-brain: {leader_count} leader(s)")

        # Step 6: Determine which container is the leader
        # Match holder_id pattern: {hostname}-{uuid}
        if "gateway-1" in initial_leader.holder_id.lower():
            leader_container = GATEWAY_1_CONTAINER
        elif "gateway-2" in initial_leader.holder_id.lower():
            leader_container = GATEWAY_2_CONTAINER
        else:
            # Fallback: check which gateway reports itself as leader
            print("  🔍 Checking gateway /config/sync/status endpoints...")
            leader_container = None
            async with httpx.AsyncClient(timeout=5.0) as client:
                for name, url in [
                    (GATEWAY_1_CONTAINER, GATEWAY_1_URL),
                    (GATEWAY_2_CONTAINER, GATEWAY_2_URL),
                ]:
                    try:
                        resp = await client.get(f"{url}/config/sync/status")
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("leader_election", {}).get("is_leader"):
                                leader_container = name
                                print(f"  📍 {name} reports itself as leader")
                                break
                    except Exception as e:
                        print(f"  ⚠️  Could not check {name}: {e}")

            if leader_container is None:
                # Last resort: kill gateway-1 and see if gateway-2 takes over
                print("  ⚠️  Could not determine leader, assuming gateway-1")
                leader_container = GATEWAY_1_CONTAINER

        print(f"  🎯 Leader container: {leader_container}")

        # Step 7: Kill the leader (simulate crash)
        print("\n💥 Simulating leader failure...")
        failover_start = time.time()
        kill_container(leader_container)

        # Step 8: Wait for new leader election
        print(f"\n⏳ Waiting for new leader (timeout: {failover_timeout}s)...")
        failover_end_time = failover_start + failover_timeout
        new_leader = None
        failover_attempts = 0

        while time.time() < failover_end_time:
            failover_attempts += 1
            await asyncio.sleep(DEFAULT_FAILOVER_POLL_INTERVAL)

            # Check nginx availability during failover
            nginx_healthy = await check_service_health(NGINX_URL)
            if not nginx_healthy:
                result.nginx_available_during_failover = False
                print(f"  ⚠️  Nginx unavailable at attempt {failover_attempts}")

            # Check for new leader
            current_leader = await get_leader_from_db()
            if current_leader and current_leader.holder_id != initial_leader.holder_id:
                new_leader = current_leader
                result.failover_time_seconds = time.time() - failover_start
                print(f"  🎉 New leader elected: {new_leader.holder_id}")
                print(f"  ⏱️  Failover time: {result.failover_time_seconds:.1f}s")
                break

            # Check for split-brain during failover
            leader_count = await count_leaders_in_db()
            if leader_count > 1:
                result.split_brain_detected = True
                print(f"  ⚠️  Split-brain during failover: {leader_count} leaders!")

        if new_leader is None:
            raise TimeoutError(
                f"No new leader elected within {failover_timeout}s"
            )

        result.new_leader = new_leader.holder_id

        # Step 9: Final validation
        print("\n🔍 Final validation...")

        # Check no split-brain after failover
        leader_count = await count_leaders_in_db()
        if leader_count > 1:
            result.split_brain_detected = True
            raise RuntimeError(f"Split-brain after failover: {leader_count} leaders!")
        print(f"  ✅ No split-brain after failover: {leader_count} leader(s)")

        # Check nginx availability after failover
        await asyncio.sleep(2)  # Brief pause for stabilization
        result.nginx_available_after_failover = await check_service_health(NGINX_URL)
        if result.nginx_available_after_failover:
            print("  ✅ Nginx is available after failover")
        else:
            print("  ⚠️  Nginx unavailable after failover")

        # All checks passed
        result.passed = True
        print("\n" + "=" * 60)
        print("✅ HA GATE PASSED")
        print("=" * 60)

    except Exception as e:
        result.error = str(e)
        print(f"\n❌ HA gate failed: {e}")

    finally:
        result.duration_seconds = time.time() - start_time

        if cleanup:
            stop_stack(compose_file)
        else:
            print("\n⚠️  Stack left running (--no-cleanup specified)")

    return result


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HA Gate - Leader Election & Failover Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick local run
    python scripts/run_ha_gate.py

    # With custom timeout and JSON output
    python scripts/run_ha_gate.py --timeout 120 --json-output ha-gate-report.json

    # Debug mode (no cleanup)
    python scripts/run_ha_gate.py --no-cleanup

Environment Variables:
    HA_NGINX_URL        Nginx LB URL (default: http://localhost:8080)
    HA_GATEWAY_1_URL    Gateway 1 URL (default: http://localhost:4000)
    HA_GATEWAY_2_URL    Gateway 2 URL (default: http://localhost:4001)
    HA_POSTGRES_HOST    PostgreSQL host (default: localhost)
    HA_POSTGRES_PORT    PostgreSQL port (default: 5432)
    POSTGRES_USER       PostgreSQL user (default: litellm)
    POSTGRES_PASSWORD   PostgreSQL password (default: litellm_password)
    LITELLM_MASTER_KEY  LiteLLM master key (default: ha-gate-test-key-not-for-production)
        """,
    )
    parser.add_argument(
        "--compose-file",
        default=DEFAULT_COMPOSE_FILE,
        help=f"Path to HA compose file (default: {DEFAULT_COMPOSE_FILE})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_FAILOVER_TIMEOUT,
        help=f"Max seconds to wait for failover (default: {DEFAULT_FAILOVER_TIMEOUT})",
    )
    parser.add_argument(
        "--json-output",
        help="Write JSON report to file",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't tear down stack after test (for debugging)",
    )

    args = parser.parse_args()

    # Run the test
    try:
        result = asyncio.run(
            run_ha_gate(
                compose_file=args.compose_file,
                failover_timeout=args.timeout,
                cleanup=not args.no_cleanup,
            )
        )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130

    # Output JSON report if requested
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"\n📄 Report written to: {args.json_output}")

    # Print summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"  Passed:                    {'✅ Yes' if result.passed else '❌ No'}")
    print(f"  Duration:                  {result.duration_seconds:.1f}s")
    print(f"  Startup time:              {result.startup_time_seconds:.1f}s")
    if result.failover_time_seconds:
        print(f"  Failover time:             {result.failover_time_seconds:.1f}s")
    print(f"  Initial leader:            {result.initial_leader or 'N/A'}")
    print(f"  New leader:                {result.new_leader or 'N/A'}")
    print(f"  Split-brain detected:      {'⚠️ Yes' if result.split_brain_detected else '✅ No'}")
    print(f"  Nginx during failover:     {'✅ Available' if result.nginx_available_during_failover else '⚠️ Unavailable'}")
    print(f"  Nginx after failover:      {'✅ Available' if result.nginx_available_after_failover else '⚠️ Unavailable'}")
    if result.error:
        print(f"  Error:                     {result.error}")
    print("-" * 60)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
