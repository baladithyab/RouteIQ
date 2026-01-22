import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_LB_URL = os.getenv("MCP_GATEWAY_URL", "http://localhost:8080")
DEFAULT_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-master-key-change-me")
DEFAULT_REPLICA_URLS = ["http://localhost:4000", "http://localhost:4001"]
# MCP stub server URL as seen by containers (service name in Docker network)
DEFAULT_STUB_URL = os.getenv("MCP_STUB_URL", "http://mcp-stub-server:9100/mcp")
DEFAULT_STUB_LOCAL_URL = os.getenv("MCP_STUB_LOCAL_URL", "http://localhost:9100/mcp")
STREAMING_CONTENT_TYPE = "text/event-stream"

# NOTE: LLMRouter MCP REST endpoints use /llmrouter/mcp/* prefix
# to avoid conflicts with LiteLLM's native /mcp endpoint.


@dataclass
class TestResult:
    name: str
    passed: bool
    status_code: int | None
    details: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class RequestResult:
    status_code: int | None
    body: dict[str, Any] | None
    content_type: str | None
    elapsed_ms: float | None
    error: str | None


logger = logging.getLogger("validate_mcp_gateway")


def _configure_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _headers(master_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {master_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = dict(headers)
    if "Authorization" in redacted:
        redacted["Authorization"] = "Bearer ***"
    return redacted


def _curl(method: str, url: str, payload: dict[str, Any] | None) -> str:
    headers = "-H 'Authorization: Bearer ***' -H 'Content-Type: application/json'"
    if payload is None:
        return f"curl -sS -X {method} {headers} '{url}'"
    data = json.dumps(payload)
    return f"curl -sS -X {method} {headers} '{url}' -d '{data}'"


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        resp.read()
        return resp.json()
    except Exception:
        try:
            text = resp.text
        except Exception:
            text = ""
        logger.debug("Non-JSON response body: %s", text)
        return {"raw": text} if text else {}


def _request(
    client: httpx.Client,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    name: str | None = None,
    allow_sse: bool = False,
) -> RequestResult:
    request_headers = headers or {}
    label = name or f"{method} {url}"
    logger.info("Request start: %s", label)
    logger.debug("Request headers: %s", _redact_headers(request_headers))
    if payload is not None:
        logger.debug("Request payload: %s", payload)
    start = time.perf_counter()
    resp: httpx.Response | None = None
    try:
        resp = client.request(
            method,
            url,
            headers=request_headers,
            json=payload,
        )
        elapsed = (time.perf_counter() - start) * 1000
        content_type = resp.headers.get("content-type", "")
        logger.info(
            "Request end: %s status=%s elapsed_ms=%.1f content_type=%s",
            label,
            resp.status_code,
            elapsed,
            content_type,
        )
        logger.debug("Response headers: %s", dict(resp.headers))
        body: dict[str, Any] | None = None
        error: str | None = None
        if STREAMING_CONTENT_TYPE in content_type:
            logger.warning(
                "Streaming response detected for %s; skipping body read", label
            )
            body = {
                "streaming": True,
                "note": "SSE response not consumed",
            }
            # If we expect SSE (allow_sse=True), then this is a success case for streaming endpoints
            # Otherwise, it's an error because we expected a JSON response
            if not allow_sse:
                error = (
                    "Unexpected streaming response " f"(content-type={content_type})"
                )
        else:
            body = _safe_json(resp)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Response body: %s", body)
        return RequestResult(
            status_code=resp.status_code,
            body=body,
            content_type=content_type,
            elapsed_ms=elapsed,
            error=error,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        logger.error("Request error: %s elapsed_ms=%.1f error=%s", label, elapsed, exc)
        return RequestResult(
            status_code=None,
            body=None,
            content_type=None,
            elapsed_ms=elapsed,
            error=str(exc),
        )
    finally:
        if resp is not None:
            resp.close()


def _record(results: list[TestResult], result: TestResult) -> None:
    results.append(result)
    status = "PASS" if result.passed else "FAIL"
    code = result.status_code if result.status_code is not None else "N/A"
    print(f"[{status}] {result.name} (status={code})")
    if result.details:
        print(f"  details: {json.dumps(result.details, indent=2)}")
    if result.error:
        print(f"  error: {result.error}")


def _contains_server(servers: list[dict[str, Any]], server_id: str) -> bool:
    return any(server.get("server_id") == server_id for server in servers)


def _contains_tool(tools: list[dict[str, Any]], tool_name: str) -> bool:
    return any(tool.get("tool") == tool_name for tool in tools)


def _contains_resource(resources: list[dict[str, Any]], resource_name: str) -> bool:
    return any(resource.get("resource") == resource_name for resource in resources)


async def _run_perf(
    base_url: str, headers: dict[str, str], timeout: httpx.Timeout
) -> dict[str, Any]:
    url = f"{base_url}/llmrouter/mcp/tools/call"
    payload = {"tool_name": "stub.echo", "arguments": {"text": "ping"}}
    total = 20
    concurrency = 5
    latencies: list[float] = []
    status_codes: list[int] = []

    logger.info("Perf sanity start total=%s concurrency=%s", total, concurrency)

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def _worker(seq: int) -> None:
            async with semaphore:
                start = time.perf_counter()
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed)
                    status_codes.append(resp.status_code)
                    logger.debug(
                        "Perf call %s status=%s elapsed_ms=%.1f",
                        seq,
                        resp.status_code,
                        elapsed,
                    )
                except Exception as exc:
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed)
                    status_codes.append(0)
                    logger.error(
                        "Perf call %s failed elapsed_ms=%.1f error=%s",
                        seq,
                        elapsed,
                        exc,
                    )

        await asyncio.gather(*[_worker(index) for index in range(total)])

    successes = sum(1 for code in status_codes if code == 200)
    success_rate = successes / total if total else 0.0
    latencies_sorted = sorted(latencies)
    p50 = (
        latencies_sorted[int(len(latencies_sorted) * 0.50)]
        if latencies_sorted
        else None
    )
    p95 = (
        latencies_sorted[int(len(latencies_sorted) * 0.95)]
        if latencies_sorted
        else None
    )

    logger.info("Perf sanity complete success_rate=%.2f", success_rate)

    return {
        "total": total,
        "concurrency": concurrency,
        "success_rate": success_rate,
        "p50_ms": round(p50, 1) if p50 is not None else None,
        "p95_ms": round(p95, 1) if p95 is not None else None,
        "status_codes": {
            "200": status_codes.count(200),
            "non_200": total - status_codes.count(200),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MCP gateway via HTTP")
    parser.add_argument("--url", default=DEFAULT_LB_URL, help="Load balancer URL")
    parser.add_argument("--master-key", default=DEFAULT_MASTER_KEY, help="API key")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--replica-url",
        action="append",
        default=DEFAULT_REPLICA_URLS,
        help="Replica URL (repeatable)",
    )
    parser.add_argument(
        "--stub-url",
        default=DEFAULT_STUB_URL,
        help="Stub MCP URL reachable from containers",
    )
    parser.add_argument(
        "--stub-local-url",
        default=DEFAULT_STUB_LOCAL_URL,
        help="Stub MCP URL reachable locally",
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger.info("MCP Gateway validation starting")
    logger.info("Timeout seconds: %s", args.timeout)

    base_url = args.url.rstrip("/")
    headers = _headers(args.master_key)
    timeout = httpx.Timeout(args.timeout)
    server_id = "stub-mcp-1"
    tool_name = "stub.echo"
    resource_name = "stub://resource/demo"

    register_payload = {
        "server_id": server_id,
        "name": "Stub MCP",
        "url": args.stub_url,
        "transport": "streamable_http",
        "tools": [tool_name, "stub.sum"],
        "resources": [resource_name],
        "metadata": {"environment": "e2e"},
    }

    print("MCP Gateway validation")
    print(f"  LB URL: {base_url}")
    print(f"  Replicas: {args.replica_url}")
    print(f"  Stub URL (container): {args.stub_url}")
    print(f"  Stub URL (local): {args.stub_local_url}")
    print()
    print("NOTE: Using /llmrouter/mcp/* REST endpoints")
    print("      (avoiding LiteLLM's native /mcp JSON-RPC endpoint)")
    print()

    results: list[TestResult] = []

    with httpx.Client(timeout=timeout) as client:
        # Test registry endpoint
        registry_url = f"{base_url}/v1/llmrouter/mcp/registry.json"
        print(_curl("GET", registry_url, None))
        result = _request(
            client,
            "GET",
            registry_url,
            headers=headers,
            name="GET /v1/llmrouter/mcp/registry.json",
        )
        if result.error:
            _record(
                results,
                TestResult(
                    name="GET /v1/llmrouter/mcp/registry.json",
                    passed=False,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                    error=result.error,
                ),
            )
        else:
            passed = result.status_code == 200
            _record(
                results,
                TestResult(
                    name="GET /v1/llmrouter/mcp/registry.json",
                    passed=passed,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                ),
            )

        # Test servers list
        servers_url = f"{base_url}/llmrouter/mcp/servers"
        print(_curl("GET", servers_url, None))
        result = _request(
            client,
            "GET",
            servers_url,
            headers=headers,
            name="GET /llmrouter/mcp/servers (LB)",
        )
        if result.error:
            _record(
                results,
                TestResult(
                    name="GET /llmrouter/mcp/servers (LB)",
                    passed=False,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                    error=result.error,
                ),
            )
        else:
            passed = result.status_code == 200
            _record(
                results,
                TestResult(
                    name="GET /llmrouter/mcp/servers (LB)",
                    passed=passed,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                ),
            )

        # Register server
        print(_curl("POST", servers_url, register_payload))
        result = _request(
            client,
            "POST",
            servers_url,
            headers=headers,
            payload=register_payload,
            name="POST /llmrouter/mcp/servers register stub",
        )
        if result.error:
            _record(
                results,
                TestResult(
                    name="POST /llmrouter/mcp/servers register stub",
                    passed=False,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                    error=result.error,
                ),
            )
        else:
            passed = result.status_code == 200
            _record(
                results,
                TestResult(
                    name="POST /llmrouter/mcp/servers register stub",
                    passed=passed,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                ),
            )

        # Verify server is listed
        print(_curl("GET", servers_url, None))
        result = _request(
            client,
            "GET",
            servers_url,
            headers=headers,
            name="GET /llmrouter/mcp/servers (LB after register)",
        )
        if result.error:
            _record(
                results,
                TestResult(
                    name="GET /llmrouter/mcp/servers (LB after register)",
                    passed=False,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                    error=result.error,
                ),
            )
        else:
            body = result.body or {}
            servers = body.get("servers", []) if isinstance(body, dict) else []
            passed = result.status_code == 200 and _contains_server(servers, server_id)
            _record(
                results,
                TestResult(
                    name="GET /llmrouter/mcp/servers (LB after register)",
                    passed=passed,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "servers": servers,
                    },
                ),
            )

        # Test replicas see the server (HA sync verification)
        for replica_url in args.replica_url:
            replica_url = replica_url.rstrip("/")
            replica_servers_url = f"{replica_url}/llmrouter/mcp/servers"
            print(_curl("GET", replica_servers_url, None))
            result = _request(
                client,
                "GET",
                replica_servers_url,
                headers=headers,
                name=f"GET /llmrouter/mcp/servers ({replica_url})",
            )
            if result.error:
                _record(
                    results,
                    TestResult(
                        name=f"GET /llmrouter/mcp/servers ({replica_url})",
                        passed=False,
                        status_code=result.status_code,
                        details={
                            "content_type": result.content_type,
                            "body": result.body,
                        },
                        error=result.error,
                    ),
                )
            else:
                body = result.body or {}
                servers = body.get("servers", []) if isinstance(body, dict) else []
                passed = result.status_code == 200 and _contains_server(
                    servers, server_id
                )
                _record(
                    results,
                    TestResult(
                        name=f"GET /llmrouter/mcp/servers ({replica_url})",
                        passed=passed,
                        status_code=result.status_code,
                        details={
                            "content_type": result.content_type,
                            "servers": servers,
                        },
                    ),
                )

        # Test tools list
        tools_url = f"{base_url}/llmrouter/mcp/tools"
        print(_curl("GET", tools_url, None))
        result = _request(
            client,
            "GET",
            tools_url,
            headers=headers,
            name="GET /llmrouter/mcp/tools",
        )
        if result.error:
            _record(
                results,
                TestResult(
                    name="GET /llmrouter/mcp/tools",
                    passed=False,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                    error=result.error,
                ),
            )
        else:
            body = result.body or {}
            tools = body.get("tools", []) if isinstance(body, dict) else []
            passed = result.status_code == 200 and _contains_tool(tools, tool_name)
            _record(
                results,
                TestResult(
                    name="GET /llmrouter/mcp/tools",
                    passed=passed,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "tools": tools,
                    },
                ),
            )

        # Test resources list
        resources_url = f"{base_url}/llmrouter/mcp/resources"
        print(_curl("GET", resources_url, None))
        result = _request(
            client,
            "GET",
            resources_url,
            headers=headers,
            name="GET /llmrouter/mcp/resources",
        )
        if result.error:
            _record(
                results,
                TestResult(
                    name="GET /llmrouter/mcp/resources",
                    passed=False,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                    error=result.error,
                ),
            )
        else:
            body = result.body or {}
            resources = body.get("resources", []) if isinstance(body, dict) else []
            passed = result.status_code == 200 and _contains_resource(
                resources, resource_name
            )
            _record(
                results,
                TestResult(
                    name="GET /llmrouter/mcp/resources",
                    passed=passed,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "resources": resources,
                    },
                ),
            )

        # Test stub server directly (if running locally)
        stub_tool_call_payload = {
            "tool_name": "stub.echo",
            "arguments": {"text": "hello"},
        }
        stub_direct_url = f"{args.stub_local_url}/tools/call"
        print(_curl("POST", stub_direct_url, stub_tool_call_payload))
        result = _request(
            client,
            "POST",
            stub_direct_url,
            payload=stub_tool_call_payload,
            name="POST stub /mcp/tools/call (direct)",
        )
        if result.error:
            _record(
                results,
                TestResult(
                    name="POST stub /mcp/tools/call (direct)",
                    passed=False,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                    error=result.error,
                ),
            )
        else:
            _record(
                results,
                TestResult(
                    name="POST stub /mcp/tools/call (direct)",
                    passed=result.status_code == 200,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                ),
            )

        # Test tool call via gateway
        tool_call_url = f"{base_url}/llmrouter/mcp/tools/call"
        print(_curl("POST", tool_call_url, stub_tool_call_payload))
        result = _request(
            client,
            "POST",
            tool_call_url,
            headers=headers,
            payload=stub_tool_call_payload,
            name="POST /llmrouter/mcp/tools/call (stub.echo)",
        )
        if result.error:
            _record(
                results,
                TestResult(
                    name="POST /llmrouter/mcp/tools/call (stub.echo)",
                    passed=False,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                    error=result.error,
                ),
            )
        else:
            _record(
                results,
                TestResult(
                    name="POST /llmrouter/mcp/tools/call (stub.echo)",
                    passed=result.status_code == 200,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                ),
            )

        # Test unknown tool returns 404
        unknown_tool_payload = {"tool_name": "unknown.tool", "arguments": {}}
        print(_curl("POST", tool_call_url, unknown_tool_payload))
        result = _request(
            client,
            "POST",
            tool_call_url,
            headers=headers,
            payload=unknown_tool_payload,
            name="POST /llmrouter/mcp/tools/call (unknown tool)",
        )
        if result.error:
            _record(
                results,
                TestResult(
                    name="POST /llmrouter/mcp/tools/call (unknown tool)",
                    passed=False,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                    error=result.error,
                ),
            )
        else:
            passed = result.status_code in {400, 404}
            _record(
                results,
                TestResult(
                    name="POST /llmrouter/mcp/tools/call (unknown tool)",
                    passed=passed,
                    status_code=result.status_code,
                    details={
                        "content_type": result.content_type,
                        "body": result.body,
                    },
                ),
            )

    # Performance test
    perf_results = asyncio.run(_run_perf(base_url, headers, timeout))
    _record(
        results,
        TestResult(
            name="Perf sanity (20 calls, concurrency=5)",
            passed=perf_results["success_rate"] == 1.0,
            status_code=None,
            details=perf_results,
        ),
    )

    failures = [result for result in results if not result.passed]
    print("\nSummary")
    print(f"  Total: {len(results)}")
    print(f"  Failed: {len(failures)}")
    if failures:
        for result in failures:
            print(f"  - {result.name}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
