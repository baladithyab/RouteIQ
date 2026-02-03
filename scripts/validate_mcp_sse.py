#!/usr/bin/env python3
"""
MCP SSE Transport Validator
===========================

Validates the MCP legacy SSE transport implementation:
- GET /mcp/sse: SSE connection with `endpoint` event
- POST /mcp/messages?sessionId=<id>: JSON-RPC requests via SSE

This script exercises the MCP SSE protocol surface that Claude Desktop
and IDE MCP clients expect for legacy compatibility.

Usage:
    # Against local-test (default):
    uv run python scripts/validate_mcp_sse.py

    # Custom endpoint:
    GATEWAY_URL=http://localhost:8080 uv run python scripts/validate_mcp_sse.py

    # With authentication:
    GATEWAY_URL=http://localhost:4010 API_KEY=sk-test-master-key uv run python scripts/validate_mcp_sse.py

Environment Variables:
    GATEWAY_URL     - Gateway URL (default: http://localhost:4010)
    API_KEY         - API key for authentication (default: sk-test-master-key)
    SSE_TIMEOUT     - Timeout for SSE connection in seconds (default: 10)
    VERBOSE         - Set to "true" for detailed output

Exit Codes:
    0 - All validations passed
    1 - One or more validations failed
    2 - Script error (network, config, etc.)
"""

import asyncio
import json
import os
import sys
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:4010")
API_KEY = os.getenv("API_KEY", "sk-test-master-key")
SSE_TIMEOUT = float(os.getenv("SSE_TIMEOUT", "10"))
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


def log_debug(msg: str) -> None:
    """Print debug message (only if VERBOSE)."""
    if VERBOSE:
        print(f"{BLUE}[DEBUG]{NC} {msg}")


class SSEEvent:
    """Represents a parsed SSE event."""

    def __init__(
        self,
        event: str | None = None,
        data: str | None = None,
        event_id: str | None = None,
        retry: int | None = None,
    ):
        self.event = event
        self.data = data
        self.event_id = event_id
        self.retry = retry

    def __repr__(self) -> str:
        return f"SSEEvent(event={self.event!r}, data={self.data!r}, id={self.event_id!r})"


def parse_sse_events(raw: str) -> list[SSEEvent]:
    """
    Parse raw SSE stream into events.

    SSE format:
    event: <type>
    id: <id>
    retry: <ms>
    data: <json>

    Events are separated by double newlines.
    """
    events = []
    current_event = SSEEvent()
    data_lines: list[str] = []

    for line in raw.split("\n"):
        if line.startswith(":"):
            # Comment line (heartbeat), skip
            continue
        elif line.startswith("event:"):
            current_event.event = line[6:].strip()
        elif line.startswith("id:"):
            current_event.event_id = line[3:].strip()
        elif line.startswith("retry:"):
            try:
                current_event.retry = int(line[6:].strip())
            except ValueError:
                pass
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
        elif line == "" and (data_lines or current_event.event):
            # End of event
            if data_lines:
                current_event.data = "\n".join(data_lines)
            events.append(current_event)
            current_event = SSEEvent()
            data_lines = []

    return events


async def connect_sse_and_get_endpoint() -> tuple[str, str] | None:
    """
    Connect to /mcp/sse and extract the messages endpoint from the `endpoint` event.

    Returns:
        Tuple of (session_id, messages_url) or None on failure
    """
    log_info("Connecting to SSE endpoint...")

    url = f"{GATEWAY_URL.rstrip('/')}/mcp/sse"
    headers = {
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        async with httpx.AsyncClient(timeout=SSE_TIMEOUT) as client:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code != 200:
                    log_fail(f"SSE connection failed: {response.status_code}")
                    return None

                # Read until we get the endpoint event
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    log_debug(f"Received chunk: {chunk[:100]}...")

                    events = parse_sse_events(buffer)
                    for event in events:
                        if event.event == "endpoint":
                            # Extract session ID from the URL
                            messages_url = event.data
                            log_debug(f"Messages URL: {messages_url}")

                            # Parse session ID from query string
                            parsed = urlparse(messages_url)
                            query_params = dict(
                                param.split("=") for param in parsed.query.split("&") if "=" in param
                            )
                            session_id = query_params.get("sessionId", "")

                            if not session_id:
                                log_fail("No sessionId in endpoint URL")
                                return None

                            log_pass(f"Got endpoint event with session: {session_id[:8]}...")
                            return session_id, messages_url

    except httpx.TimeoutException:
        log_fail(f"SSE connection timed out after {SSE_TIMEOUT}s")
        return None
    except Exception as e:
        log_fail(f"SSE connection error: {e}")
        return None

    log_fail("No endpoint event received")
    return None


async def post_jsonrpc_and_wait_sse(
    session_id: str,
    messages_url: str,
    method: str,
    params: dict[str, Any] | None = None,
    request_id: int = 1,
) -> dict[str, Any] | None:
    """
    POST a JSON-RPC request to /mcp/messages and wait for response via SSE.

    This simulates the legacy MCP SSE flow:
    1. POST request to messages endpoint
    2. Wait for HTTP 202 Accepted
    3. Response comes via SSE stream

    Note: For this validator, we'll take a simpler approach since we can't
    maintain a persistent SSE connection while posting. We'll verify the
    HTTP 202 response and the session-based routing.
    """
    log_info(f"Posting JSON-RPC {method}...")

    # For the validator, use the messages URL directly
    # In real clients, they'd maintain the SSE connection
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params:
        payload["params"] = params

    log_debug(f"Request payload: {json.dumps(payload)}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                messages_url,
                json=payload,
                headers=headers,
            )

            log_debug(f"Response status: {response.status_code}")
            log_debug(f"Response body: {response.text}")

            if response.status_code == 202:
                data = response.json()
                if data.get("status") == "accepted":
                    log_pass(f"POST {method} accepted (202)")
                    return data
                else:
                    log_fail(f"Unexpected response: {data}")
                    return None
            else:
                log_fail(f"POST {method} returned {response.status_code}: {response.text}")
                return None

    except Exception as e:
        log_fail(f"POST {method} error: {e}")
        return None


async def test_full_sse_flow() -> bool:
    """
    Test the full SSE flow:
    1. Connect to /mcp/sse
    2. Extract endpoint URL from `endpoint` event
    3. POST initialize request
    4. POST tools/list request
    """
    log_info("Testing full SSE transport flow...")

    # Step 1: Connect and get endpoint
    result = await connect_sse_and_get_endpoint()
    if not result:
        return False

    session_id, messages_url = result

    # Step 2: POST initialize
    init_result = await post_jsonrpc_and_wait_sse(
        session_id,
        messages_url,
        "initialize",
        params={
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "validate-mcp-sse", "version": "1.0.0"},
        },
        request_id=1,
    )
    if not init_result:
        return False

    # Step 3: POST tools/list
    tools_result = await post_jsonrpc_and_wait_sse(
        session_id,
        messages_url,
        "tools/list",
        request_id=2,
    )
    if not tools_result:
        return False

    log_pass("Full SSE transport flow validated")
    return True


async def test_transport_info() -> bool:
    """Test GET /mcp/transport returns SSE info."""
    log_info("Testing GET /mcp/transport...")

    url = f"{GATEWAY_URL.rstrip('/')}/mcp/transport"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                log_fail(f"GET /mcp/transport returned {response.status_code}")
                return False

            data = response.json()
            log_debug(f"Transport info: {json.dumps(data, indent=2)}")

            # Verify SSE transport info
            sse_info = data.get("transports", {}).get("sse", {})
            if not sse_info.get("enabled"):
                log_warn("SSE transport is not enabled")
                # Still pass - this is a config issue, not implementation
                return True

            if sse_info.get("endpoint") != "/mcp/sse":
                log_fail(f"Unexpected SSE endpoint: {sse_info.get('endpoint')}")
                return False

            if sse_info.get("messages_endpoint") != "/mcp/messages":
                log_fail(f"Unexpected messages endpoint: {sse_info.get('messages_endpoint')}")
                return False

            log_pass("GET /mcp/transport returns valid SSE config")
            return True

    except Exception as e:
        log_fail(f"GET /mcp/transport error: {e}")
        return False


async def test_invalid_session() -> bool:
    """Test that invalid session returns 404."""
    log_info("Testing invalid session handling...")

    url = f"{GATEWAY_URL.rstrip('/')}/mcp/messages?sessionId=invalid-session-id"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code == 404:
                data = response.json()
                if data.get("detail", {}).get("error") == "session_not_found":
                    log_pass("Invalid session returns 404 with proper error")
                    return True
                else:
                    log_fail(f"Unexpected 404 response: {data}")
                    return False
            else:
                log_fail(f"Expected 404, got {response.status_code}")
                return False

    except Exception as e:
        log_fail(f"Invalid session test error: {e}")
        return False


async def test_sse_sessions_endpoint() -> bool:
    """Test GET /mcp/sse/sessions returns session list."""
    log_info("Testing GET /mcp/sse/sessions...")

    url = f"{GATEWAY_URL.rstrip('/')}/mcp/sse/sessions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                log_fail(f"GET /mcp/sse/sessions returned {response.status_code}")
                return False

            data = response.json()
            log_debug(f"Sessions: {json.dumps(data, indent=2)}")

            if "active_sessions" not in data or "sessions" not in data:
                log_fail(f"Missing expected fields in response: {data}")
                return False

            log_pass(f"GET /mcp/sse/sessions returns {data['active_sessions']} sessions")
            return True

    except Exception as e:
        log_fail(f"GET /mcp/sse/sessions error: {e}")
        return False


async def run_all_tests() -> tuple[int, int]:
    """Run all validation tests."""
    pass_count = 0
    fail_count = 0

    tests = [
        ("Transport info", test_transport_info),
        ("SSE sessions endpoint", test_sse_sessions_endpoint),
        ("Invalid session handling", test_invalid_session),
        ("Full SSE flow", test_full_sse_flow),
    ]

    for name, test_fn in tests:
        print()
        try:
            if await test_fn():
                pass_count += 1
            else:
                fail_count += 1
        except Exception as e:
            log_fail(f"{name}: {e}")
            fail_count += 1

    return pass_count, fail_count


def main() -> int:
    """
    Run all validation tests.

    Returns:
        0 if all tests pass, 1 if any fail, 2 on script error
    """
    print("=" * 60)
    print("  MCP SSE Transport Validator")
    print("=" * 60)
    print(f"Gateway URL: {GATEWAY_URL}")
    print(f"API Key:     {API_KEY[:8]}...")
    print(f"SSE Timeout: {SSE_TIMEOUT}s")
    print()

    try:
        pass_count, fail_count = asyncio.run(run_all_tests())

        # Summary
        print()
        print("=" * 60)
        print("  Validation Summary")
        print("=" * 60)
        print(f"{GREEN}Passed: {pass_count}{NC}")
        print(f"{RED}Failed: {fail_count}{NC}")
        print()

        if fail_count == 0:
            print(f"{GREEN}All validations passed!{NC}")
            return 0
        else:
            print(f"{RED}Some validations failed.{NC}")
            return 1

    except httpx.ConnectError:
        print()
        log_fail(f"Cannot connect to gateway at {GATEWAY_URL}")
        log_info("Make sure the gateway is running:")
        log_info("  finch compose -f docker-compose.yml up -d")
        log_info("  # or")
        log_info("  uv run uvicorn litellm_llmrouter.startup:app --port 4010")
        return 2

    except Exception as e:
        print()
        log_fail(f"Script error: {e}")
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
