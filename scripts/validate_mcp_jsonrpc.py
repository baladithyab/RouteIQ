#!/usr/bin/env python3
"""
MCP Native JSON-RPC Surface Validator
======================================

Validates the native MCP JSON-RPC 2.0 endpoint at /mcp.

This script exercises the MCP protocol surface that Claude Desktop and
IDE MCP clients expect:
- initialize: Session initialization
- tools/list: List available tools
- tools/call: Invoke a tool

Usage:
    # Against local-test (default):
    python scripts/validate_mcp_jsonrpc.py

    # Custom endpoint:
    GATEWAY_URL=http://localhost:8080 python scripts/validate_mcp_jsonrpc.py

    # With authentication:
    GATEWAY_URL=http://localhost:4010 API_KEY=local-dev-master-key \\
    LITELLM_MASTER_KEY=local-dev-master-key python scripts/validate_mcp_jsonrpc.py

Environment Variables:
    GATEWAY_URL     - Gateway URL (default: http://localhost:4010)
    API_KEY         - API key for authentication (default: local-dev-master-key)
    LITELLM_MASTER_KEY    - Admin API key for tool invocation (default: local-dev-master-key)
    STUB_SERVER_ID  - Server ID for stub server (default: stub-mcp-1)
    STUB_URL        - Stub server URL as seen by gateway (default: http://mcp-stub-server:9100/mcp)
    VERBOSE         - Set to "true" for detailed output

Exit Codes:
    0 - All validations passed
    1 - One or more validations failed
    2 - Script error (network, config, etc.)
"""

import json
import os
import sys
from typing import Any

import httpx

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:4010")
API_KEY = os.getenv("API_KEY", "local-dev-master-key")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "local-dev-master-key")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", API_KEY)
STUB_SERVER_ID = os.getenv("STUB_SERVER_ID", "stub-mcp-1")
STUB_URL = os.getenv("STUB_URL", "http://mcp-stub-server:9100/mcp")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# ANSI colors
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"  # No Color


def log_info(msg: str) -> None:
    """Print info message."""
    print(f"{BLUE}[INFO]{NC} {msg}")


def log_pass(msg: str) -> None:
    """Print pass message."""
    print(f"{GREEN}[PASS]{NC} {msg}")


def log_fail(msg: str) -> None:
    """Print fail message."""
    print(f"{RED}[FAIL]{NC} {msg}")


def log_warn(msg: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}[WARN]{NC} {msg}")


def jsonrpc_request(
    method: str,
    params: dict[str, Any] | None = None,
    request_id: int = 1,
    use_admin_key: bool = False,
) -> dict[str, Any]:
    """
    Send a JSON-RPC 2.0 request to /mcp.

    Args:
        method: JSON-RPC method name
        params: Method parameters
        request_id: Request ID
        use_admin_key: Whether to use admin API key

    Returns:
        JSON-RPC response as dict
    """
    url = f"{GATEWAY_URL.rstrip('/')}/mcp"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ADMIN_API_KEY if use_admin_key else API_KEY}",
    }

    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params:
        payload["params"] = params

    if VERBOSE:
        log_info(f"Request: {method}")
        log_info(f"  URL: {url}")
        log_info(f"  Payload: {json.dumps(payload, indent=2)}")

    response = httpx.post(url, json=payload, headers=headers, timeout=30.0)

    if VERBOSE:
        log_info(f"  Status: {response.status_code}")
        log_info(f"  Response: {response.text[:500]}")

    return response.json()


def rest_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    use_admin_key: bool = False,
) -> httpx.Response:
    """
    Send a REST request for setup/teardown operations.

    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        path: URL path
        payload: Request payload
        use_admin_key: Whether to use admin API key

    Returns:
        httpx Response object
    """
    url = f"{GATEWAY_URL.rstrip('/')}{path}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    if use_admin_key:
        headers["X-Admin-API-Key"] = ADMIN_API_KEY

    if method.upper() == "GET":
        return httpx.get(url, headers=headers, timeout=30.0)
    elif method.upper() == "POST":
        return httpx.post(url, json=payload, headers=headers, timeout=30.0)
    elif method.upper() == "DELETE":
        return httpx.delete(url, headers=headers, timeout=30.0)
    else:
        raise ValueError(f"Unsupported method: {method}")


def setup_stub_server() -> bool:
    """
    Register the stub MCP server if not already registered.

    Returns:
        True if setup successful, False otherwise
    """
    log_info("Setting up stub MCP server...")

    # Check if already registered
    try:
        response = rest_request("GET", f"/llmrouter/mcp/servers/{STUB_SERVER_ID}")
        if response.status_code == 200:
            log_info(f"Stub server '{STUB_SERVER_ID}' already registered")
            return True
    except Exception:
        pass

    # Register stub server
    payload = {
        "server_id": STUB_SERVER_ID,
        "name": "test-stub",
        "url": STUB_URL,
        "transport": "streamable_http",
        "tools": ["stub.echo", "stub.sum"],
        "resources": ["stub://resource/demo"],
        "metadata": {"environment": "e2e-jsonrpc"},
    }

    try:
        response = rest_request(
            "POST",
            "/llmrouter/mcp/servers",
            payload=payload,
            use_admin_key=True,
        )
        if response.status_code == 200:
            log_info(f"Registered stub server '{STUB_SERVER_ID}'")
            return True
        else:
            log_warn(f"Failed to register stub server: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        log_warn(f"Error registering stub server: {e}")
        return False


def cleanup_stub_server() -> None:
    """Unregister the stub MCP server."""
    try:
        response = rest_request(
            "DELETE",
            f"/llmrouter/mcp/servers/{STUB_SERVER_ID}",
            use_admin_key=True,
        )
        if response.status_code == 200:
            log_info(f"Cleaned up stub server '{STUB_SERVER_ID}'")
        elif response.status_code == 404:
            log_info("Stub server already cleaned up")
        else:
            log_warn(f"Cleanup returned {response.status_code}")
    except Exception as e:
        log_warn(f"Error during cleanup: {e}")


# ============================================================================
# Test Cases
# ============================================================================


def test_get_info() -> bool:
    """Test GET /mcp returns server info."""
    log_info("Testing GET /mcp (server info)...")

    url = f"{GATEWAY_URL.rstrip('/')}/mcp"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        if response.status_code != 200:
            log_fail(f"GET /mcp returned {response.status_code}")
            return False

        data = response.json()
        if data.get("protocolVersion") and data.get("transport") == "streamable-http":
            log_pass("GET /mcp returns valid server info")
            return True
        else:
            log_fail(f"GET /mcp missing expected fields: {data}")
            return False
    except Exception as e:
        log_fail(f"GET /mcp error: {e}")
        return False


def test_initialize() -> bool:
    """Test initialize method."""
    log_info("Testing JSON-RPC initialize...")

    try:
        response = jsonrpc_request(
            "initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validate-mcp-jsonrpc", "version": "1.0.0"},
            },
        )

        # Check for valid JSON-RPC response
        if response.get("jsonrpc") != "2.0":
            log_fail(f"Invalid jsonrpc version: {response.get('jsonrpc')}")
            return False

        if "error" in response:
            log_fail(f"initialize returned error: {response['error']}")
            return False

        result = response.get("result", {})
        if result.get("protocolVersion") and result.get("capabilities") and result.get("serverInfo"):
            log_pass("initialize returns valid response")
            if VERBOSE:
                log_info(f"  protocolVersion: {result.get('protocolVersion')}")
                log_info(f"  serverInfo: {result.get('serverInfo')}")
            return True
        else:
            log_fail(f"initialize missing expected fields: {result}")
            return False

    except Exception as e:
        log_fail(f"initialize error: {e}")
        return False


def test_tools_list() -> bool:
    """Test tools/list method."""
    log_info("Testing JSON-RPC tools/list...")

    try:
        response = jsonrpc_request("tools/list")

        if response.get("jsonrpc") != "2.0":
            log_fail(f"Invalid jsonrpc version: {response.get('jsonrpc')}")
            return False

        if "error" in response:
            log_fail(f"tools/list returned error: {response['error']}")
            return False

        result = response.get("result", {})
        tools = result.get("tools", [])

        if not isinstance(tools, list):
            log_fail(f"tools/list.tools is not a list: {type(tools)}")
            return False

        # Check for stub tools if stub server is registered
        stub_tools = [t for t in tools if t.get("name", "").startswith(f"{STUB_SERVER_ID}.")]
        if stub_tools:
            log_pass(f"tools/list returns {len(tools)} tools (includes {len(stub_tools)} stub tools)")
            if VERBOSE:
                for tool in stub_tools:
                    log_info(f"  - {tool.get('name')}: {tool.get('description', '')[:50]}")
        else:
            log_pass(f"tools/list returns {len(tools)} tools (stub tools not yet registered)")

        return True

    except Exception as e:
        log_fail(f"tools/list error: {e}")
        return False


def test_tools_call_stub_echo() -> bool:
    """Test tools/call with stub.echo tool."""
    log_info("Testing JSON-RPC tools/call (stub.echo)...")

    # Tool name is namespaced as server_id.tool_name
    tool_name = f"{STUB_SERVER_ID}.stub.echo"

    try:
        response = jsonrpc_request(
            "tools/call",
            params={
                "name": tool_name,
                "arguments": {"text": "Hello from validation script"},
            },
            use_admin_key=True,  # Tool invocation requires admin
        )

        if response.get("jsonrpc") != "2.0":
            log_fail(f"Invalid jsonrpc version: {response.get('jsonrpc')}")
            return False

        if "error" in response:
            error = response["error"]
            # Check for expected "disabled" error if tool invocation is off
            if error.get("code") == -32002:  # MCP_TOOL_INVOCATION_DISABLED
                log_warn("tools/call returned 'tool invocation disabled' - this is expected if LLMROUTER_ENABLE_MCP_TOOL_INVOCATION is not set")
                return True  # Pass - expected behavior
            else:
                log_fail(f"tools/call returned error: {error}")
                return False

        result = response.get("result", {})
        content = result.get("content", [])
        is_error = result.get("isError", False)

        if is_error:
            log_fail(f"tools/call returned isError=true: {content}")
            return False

        if not content:
            log_fail("tools/call returned empty content")
            return False

        # Check content has text
        text_content = [c for c in content if c.get("type") == "text"]
        if text_content:
            log_pass(f"tools/call (stub.echo) returned content: {text_content[0].get('text', '')[:100]}")
            return True
        else:
            log_fail(f"tools/call returned no text content: {content}")
            return False

    except Exception as e:
        log_fail(f"tools/call error: {e}")
        return False


def test_resources_list() -> bool:
    """Test resources/list method."""
    log_info("Testing JSON-RPC resources/list...")

    try:
        response = jsonrpc_request("resources/list")

        if response.get("jsonrpc") != "2.0":
            log_fail(f"Invalid jsonrpc version: {response.get('jsonrpc')}")
            return False

        if "error" in response:
            log_fail(f"resources/list returned error: {response['error']}")
            return False

        result = response.get("result", {})
        resources = result.get("resources", [])

        if not isinstance(resources, list):
            log_fail(f"resources/list.resources is not a list: {type(resources)}")
            return False

        log_pass(f"resources/list returns {len(resources)} resources")
        return True

    except Exception as e:
        log_fail(f"resources/list error: {e}")
        return False


def test_invalid_method() -> bool:
    """Test that invalid methods return proper error."""
    log_info("Testing JSON-RPC invalid method handling...")

    try:
        response = jsonrpc_request("nonexistent/method")

        if response.get("jsonrpc") != "2.0":
            log_fail(f"Invalid jsonrpc version: {response.get('jsonrpc')}")
            return False

        if "error" not in response:
            log_fail("Invalid method should return error")
            return False

        error = response["error"]
        if error.get("code") == -32601:  # Method not found
            log_pass("Invalid method returns proper JSON-RPC error")
            return True
        else:
            log_fail(f"Unexpected error code: {error.get('code')}")
            return False

    except Exception as e:
        log_fail(f"Invalid method test error: {e}")
        return False


def test_malformed_request() -> bool:
    """Test that malformed requests return proper error."""
    log_info("Testing JSON-RPC malformed request handling...")

    url = f"{GATEWAY_URL.rstrip('/')}/mcp"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    # Send request with wrong jsonrpc version
    payload = {
        "jsonrpc": "1.0",  # Wrong version
        "id": 1,
        "method": "initialize",
    }

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        data = response.json()

        if "error" not in data:
            log_fail("Malformed request should return error")
            return False

        error = data["error"]
        if error.get("code") == -32600:  # Invalid request
            log_pass("Malformed request returns proper JSON-RPC error")
            return True
        else:
            log_fail(f"Unexpected error code: {error.get('code')}")
            return False

    except Exception as e:
        log_fail(f"Malformed request test error: {e}")
        return False


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """
    Run all validation tests.

    Returns:
        0 if all tests pass, 1 if any fail, 2 on script error
    """
    print("=" * 50)
    print("  MCP Native JSON-RPC Validation")
    print("=" * 50)
    print(f"Gateway URL: {GATEWAY_URL}")
    print(f"API Key:     {API_KEY[:8]}...")
    print(f"Stub Server: {STUB_SERVER_ID}")
    print()

    pass_count = 0
    fail_count = 0

    try:
        # Setup
        setup_success = setup_stub_server()
        if not setup_success:
            log_warn("Stub server setup failed - some tests may be limited")

        # Run tests
        tests = [
            ("GET /mcp info", test_get_info),
            ("initialize", test_initialize),
            ("tools/list", test_tools_list),
            ("tools/call (stub.echo)", test_tools_call_stub_echo),
            ("resources/list", test_resources_list),
            ("invalid method", test_invalid_method),
            ("malformed request", test_malformed_request),
        ]

        print()
        print("-" * 50)
        print("  Running Tests")
        print("-" * 50)
        print()

        for name, test_fn in tests:
            try:
                if test_fn():
                    pass_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                log_fail(f"{name}: {e}")
                fail_count += 1
            print()

        # Cleanup
        print("-" * 50)
        print("  Cleanup")
        print("-" * 50)
        cleanup_stub_server()

        # Summary
        print()
        print("=" * 50)
        print("  Validation Summary")
        print("=" * 50)
        print(f"{GREEN}Passed: {pass_count}{NC}")
        print(f"{RED}Failed: {fail_count}{NC}")
        print()

        if fail_count == 0:
            print(f"{GREEN}All validations passed!{NC}")
            return 0
        else:
            print(f"{RED}Some validations failed.{NC}")
            return 1

    except httpx.ConnectError as e:
        print()
        log_fail(f"Cannot connect to gateway at {GATEWAY_URL}")
        log_info("Make sure the gateway is running:")
        log_info("  docker compose -f docker-compose.local-test.yml up -d")
        return 2

    except Exception as e:
        print()
        log_fail(f"Script error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
