#!/usr/bin/env python3
"""
Test script for LiteLLM + LLMRouter Gateway features.
Tests: A2A Gateway, MCP Gateway, Config Sync, Hot Reload, Routing Strategies
"""

import sys
from typing import Any

import requests

BASE_URL = "http://localhost:4000"
MASTER_KEY = "local-dev-master-key"


def request(
    method: str, path: str, data: dict[str, Any] | None = None, auth: bool = True
) -> dict[str, Any]:
    """Make a request to the gateway."""
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {MASTER_KEY}"

    url = f"{BASE_URL}{path}"
    if method == "GET":
        resp = requests.get(url, headers=headers, params=data, timeout=30)
    elif method == "POST":
        resp = requests.post(url, headers=headers, json=data, timeout=30)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers, timeout=30)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {"status": resp.status_code, "body": resp.json() if resp.text else {}}


def test_health():
    """Test health endpoint."""
    print("\n=== Test: Health Check ===")
    resp = request("GET", "/health", auth=False)
    assert resp["status"] == 200, f"Health check failed: {resp}"
    print(f"✓ Health: {resp['body']}")


def test_models():
    """Test model listing."""
    print("\n=== Test: Model List ===")
    resp = request("GET", "/v1/models")
    assert resp["status"] == 200, f"Model list failed: {resp}"
    models = [m["id"] for m in resp["body"].get("data", [])]
    print(f"✓ Models: {models}")


def test_a2a_gateway():
    """Test A2A Gateway endpoints."""
    print("\n=== Test: A2A Gateway ===")

    # Register agent
    agent_data = {
        "agent_id": "test-agent-1",
        "name": "Test Agent",
        "description": "A test agent",
        "url": "http://localhost:9000/a2a",
        "capabilities": ["chat", "code"],
    }
    resp = request("POST", "/a2a/agents", agent_data)
    print(f"  Register agent: {resp['status']} - {resp['body']}")

    # List agents
    resp = request("GET", "/a2a/agents")
    print(f"  List agents: {resp['status']} - {resp['body']}")

    # Discover by capability
    resp = request("GET", "/a2a/agents", {"capability": "chat"})
    print(f"  Discover (chat): {resp['status']} - {resp['body']}")


def test_mcp_gateway():
    """Test MCP Gateway endpoints."""
    print("\n=== Test: MCP Gateway ===")

    # Register server
    server_data = {
        "server_id": "test-mcp-1",
        "name": "Test MCP Server",
        "url": "http://localhost:8080/mcp",
        "transport": "streamable_http",
        "tools": ["search", "fetch"],
    }
    resp = request("POST", "/mcp/servers", server_data)
    print(f"  Register server: {resp['status']} - {resp['body']}")

    # List servers
    resp = request("GET", "/mcp/servers")
    print(f"  List servers: {resp['status']} - {resp['body']}")

    # List tools
    resp = request("GET", "/mcp/tools")
    print(f"  List tools: {resp['status']} - {resp['body']}")


def test_config_sync():
    """Test Config Sync endpoints."""
    print("\n=== Test: Config Sync ===")

    resp = request("GET", "/config/sync/status")
    print(f"  Sync status: {resp['status']} - {resp['body']}")


def test_hot_reload():
    """Test Hot Reload endpoints."""
    print("\n=== Test: Hot Reload ===")

    resp = request("POST", "/config/reload", {"force_sync": False})
    print(f"  Config reload: {resp['status']} - {resp['body']}")

    resp = request("POST", "/router/reload")
    print(f"  Router reload: {resp['status']} - {resp['body']}")


def test_router_info():
    """Test Router info endpoint."""
    print("\n=== Test: Router Info ===")

    resp = request("GET", "/router/info")
    print(f"  Router info: {resp['status']} - {resp['body']}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("LiteLLM + LLMRouter Gateway Test Suite")
    print("=" * 60)

    try:
        test_health()
        test_models()
        test_a2a_gateway()
        test_mcp_gateway()
        test_config_sync()
        test_hot_reload()
        test_router_info()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except requests.ConnectionError:
        print("\n✗ ERROR: Cannot connect to gateway at", BASE_URL)
        print("  Make sure the gateway is running.")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
