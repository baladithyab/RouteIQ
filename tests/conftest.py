"""
Pytest configuration for LiteLLM + LLMRouter tests.

Automatically skips integration tests when the local Docker stack is not running.
"""

import os
import socket

import pytest


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a port is open on the given host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except (OSError, socket.error):
        return False


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if the local stack is not running."""
    # Check if we're in CI or if the gateway is not available
    in_ci = os.getenv("CI", "").lower() == "true"
    gateway_port = 4010  # Default gateway port from docker-compose.local-test.yml

    # Try to detect if the gateway is running
    gateway_available = is_port_open("localhost", gateway_port)

    # Tests that manage their own compose stack (should not be auto-skipped)
    self_managed_tests = {
        "test_streaming_perf_gate",  # TG10.5 - manages its own compose stack
    }

    if not gateway_available:
        skip_integration = pytest.mark.skip(
            reason="Integration tests skipped: Local Docker stack not running. "
            "Run 'docker compose -f docker-compose.local-test.yml up -d' first."
        )
        for item in items:
            # Skip tests in the integration folder only (not property tests)
            # Check for /integration/ or \integration\ in the path
            fspath_str = str(item.fspath)
            if "/integration/" in fspath_str or "\\integration\\" in fspath_str:
                # Don't skip tests that manage their own compose stack
                parent_name = getattr(item.parent, "name", "") if item.parent else ""
                if parent_name not in self_managed_tests and item.name not in self_managed_tests:
                    # Check module name as well
                    module_name = item.fspath.purebasename if hasattr(item.fspath, 'purebasename') else ""
                    if module_name not in self_managed_tests:
                        item.add_marker(skip_integration)

    if in_ci and not gateway_available:
        # In CI, we expect integration tests to be skipped
        print("CI environment detected, skipping integration tests")
