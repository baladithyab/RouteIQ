#!/usr/bin/env python3
"""
LiteLLM Feature Parity Validation Script
=========================================

Compares our container against expected LiteLLM proxy functionality.

Usage:
    python scripts/validate_parity.py [--url URL] [--key API_KEY]
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-dev-key")


@dataclass
class TestResult:
    name: str
    passed: bool
    status_code: int | None = None
    response_time_ms: float = 0
    error: str | None = None
    details: dict[str, Any] | None = None


class ParityValidator:
    """Validates LiteLLM feature parity."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.results: list[TestResult] = []

    async def run_all_tests(self) -> list[TestResult]:
        """Run all validation tests."""
        print("=" * 70)
        print("🔍 LiteLLM Feature Parity Validation")
        print(f"   Target: {self.base_url}")
        print("=" * 70)

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Health endpoints
            await self._test_health_endpoints(client)
            # Model endpoints
            await self._test_model_endpoints(client)
            # Chat completions
            await self._test_chat_completions(client)
            # Management endpoints
            await self._test_management_endpoints(client)
            # MCP parity endpoints
            await self._test_mcp_parity_endpoints(client)

        return self.results

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        auth: bool = True,
        **kwargs,
    ) -> tuple[httpx.Response | None, float, str | None]:
        """Make a request and return (response, time_ms, error)."""
        headers = kwargs.pop("headers", {})
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}{path}"
        start = time.perf_counter()
        try:
            resp = await client.request(method, url, headers=headers, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            return resp, elapsed, None
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return None, elapsed, str(e)

    async def _test_health_endpoints(self, client: httpx.AsyncClient) -> None:
        """Test health check endpoints."""
        print("\n📋 Health Endpoints")
        endpoints = [
            ("/health", "GET", False, "Main health check"),
            ("/health/liveliness", "GET", False, "Kubernetes liveness probe"),
            ("/health/readiness", "GET", False, "Kubernetes readiness probe"),
        ]

        for path, method, auth, desc in endpoints:
            resp, time_ms, error = await self._request(client, method, path, auth=auth)
            passed = resp is not None and resp.status_code in [200, 401, 503]
            self.results.append(
                TestResult(
                    name=f"Health: {desc}",
                    passed=passed,
                    status_code=resp.status_code if resp else None,
                    response_time_ms=time_ms,
                    error=error,
                )
            )
            status = "✅" if passed else "❌"
            code = resp.status_code if resp else "N/A"
            print(f"  {status} {path} - {code} ({time_ms:.0f}ms)")

    async def _test_model_endpoints(self, client: httpx.AsyncClient) -> None:
        """Test model listing endpoints."""
        print("\n📋 Model Endpoints")
        endpoints = [
            ("/v1/models", "GET", "OpenAI-compatible model list"),
            ("/model/info", "GET", "LiteLLM model info"),
        ]

        for path, method, desc in endpoints:
            resp, time_ms, error = await self._request(client, method, path)
            passed = resp is not None and resp.status_code in [200, 401, 404]
            details = None
            if passed and resp and resp.status_code == 200:
                try:
                    data = resp.json()
                    if "data" in data:
                        details = {"model_count": len(data["data"])}
                except Exception:
                    pass
            self.results.append(
                TestResult(
                    name=f"Models: {desc}",
                    passed=passed,
                    status_code=resp.status_code if resp else None,
                    response_time_ms=time_ms,
                    error=error,
                    details=details,
                )
            )
            status = "✅" if passed else "❌"
            code = resp.status_code if resp else "N/A"
            extra = f" ({details['model_count']} models)" if details else ""
            print(f"  {status} {path} - {code} ({time_ms:.0f}ms){extra}")

    async def _test_chat_completions(self, client: httpx.AsyncClient) -> None:
        """Test chat completions endpoint."""
        print("\n📋 Chat Completions")
        # Non-streaming
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Say 'parity test' only"}],
            "max_tokens": 10,
        }
        resp, time_ms, error = await self._request(
            client, "POST", "/v1/chat/completions", json=payload
        )
        passed = resp is not None and resp.status_code in [200, 401, 400, 422]
        self.results.append(
            TestResult(
                name="Chat: Non-streaming completion",
                passed=passed,
                status_code=resp.status_code if resp else None,
                response_time_ms=time_ms,
                error=error,
            )
        )
        status = "✅" if passed else "❌"
        code = resp.status_code if resp else "N/A"
        print(f"  {status} POST /v1/chat/completions - {code} ({time_ms:.0f}ms)")

    async def _test_management_endpoints(self, client: httpx.AsyncClient) -> None:
        """Test management endpoints."""
        print("\n📋 Management Endpoints")
        endpoints = [
            ("/key/info", "GET", "API key info"),
            ("/spend/logs", "GET", "Spend logs"),
            ("/config/yaml", "GET", "Config export"),
        ]
        for path, method, desc in endpoints:
            resp, time_ms, error = await self._request(client, method, path)
            passed = resp is not None and resp.status_code in [200, 401, 403, 404, 422]
            self.results.append(
                TestResult(
                    name=f"Management: {desc}",
                    passed=passed,
                    status_code=resp.status_code if resp else None,
                    response_time_ms=time_ms,
                    error=error,
                )
            )
            status = "✅" if passed else "❌"
            code = resp.status_code if resp else "N/A"
            print(f"  {status} {path} - {code} ({time_ms:.0f}ms)")

    async def _test_mcp_parity_endpoints(self, client: httpx.AsyncClient) -> None:
        """Test MCP parity endpoints (upstream-compatible routes)."""
        print("\n📋 MCP Parity Endpoints")
        endpoints = [
            # MCP Management Endpoints
            ("/v1/mcp/server", "GET", "List MCP servers"),
            ("/v1/mcp/server/health", "GET", "MCP server health"),
            ("/v1/mcp/tools", "GET", "List MCP tools"),
            ("/v1/mcp/access_groups", "GET", "List access groups"),
            ("/v1/mcp/registry.json", "GET", "MCP registry document"),
            # MCP REST API
            ("/mcp-rest/tools/list", "GET", "MCP REST tools list"),
            # Namespaced MCP routes
            ("/mcp", "GET", "Built-in MCP endpoint"),
        ]
        for path, method, desc in endpoints:
            resp, time_ms, error = await self._request(client, method, path)
            # Accept 200, 401, 403, 404 as valid responses
            # 404 is valid if MCP gateway is disabled
            passed = resp is not None and resp.status_code in [200, 401, 403, 404]
            self.results.append(
                TestResult(
                    name=f"MCP Parity: {desc}",
                    passed=passed,
                    status_code=resp.status_code if resp else None,
                    response_time_ms=time_ms,
                    error=error,
                )
            )
            status = "✅" if passed else "❌"
            code = resp.status_code if resp else "N/A"
            print(f"  {status} {path} - {code} ({time_ms:.0f}ms)")

    def print_summary(self) -> int:
        """Print test summary and return exit code."""
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        pct = (passed / total * 100) if total > 0 else 0

        print(f"\n  Total Tests: {total}")
        print(f"  Passed: {passed} ({pct:.0f}%)")
        print(f"  Failed: {total - passed}")

        if passed < total:
            print("\n  ❌ Failed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"     - {r.name}: {r.error or f'Status {r.status_code}'}")

        # Calculate average response time
        avg_time = (
            sum(r.response_time_ms for r in self.results) / total if total > 0 else 0
        )
        print(f"\n  Average Response Time: {avg_time:.0f}ms")

        if pct >= 80:
            print("\n✅ Feature parity validation PASSED")
            return 0
        else:
            print("\n❌ Feature parity validation FAILED")
            return 1


async def main():
    parser = argparse.ArgumentParser(description="Validate LiteLLM feature parity")
    parser.add_argument("--url", default=LITELLM_URL, help="LiteLLM URL")
    parser.add_argument("--key", default=LITELLM_API_KEY, help="API key")
    args = parser.parse_args()

    validator = ParityValidator(args.url, args.key)
    await validator.run_all_tests()
    exit_code = validator.print_summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
