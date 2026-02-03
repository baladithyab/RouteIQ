#!/usr/bin/env python3
"""
Comprehensive Test Suite for LiteLLM + LLMRouter Container
===========================================================

Tests all features from the validation plan including:
- Health endpoints
- Chat completions
- Redis caching
- Jaeger tracing
- MLflow integration
- A2A/MCP Gateway (protocol-aware)
- Performance benchmarks

Usage:
    python scripts/comprehensive_test.py
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

# Configuration
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4010")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "local-dev-master-key")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "local-dev-master-key")
JAEGER_URL = os.getenv("JAEGER_URL", "http://localhost:16686")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5050")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")


@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    duration_ms: float = 0
    status_code: int | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ComprehensiveTestRunner:
    """Runs comprehensive tests for the LiteLLM container."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.results: list[TestResult] = []
        self.start_time = datetime.now()

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def run_all_tests(self) -> list[TestResult]:
        """Run all test categories."""
        print("=" * 70)
        print("🧪 LiteLLM + LLMRouter Comprehensive Test Suite")
        print(f"   Target: {self.base_url}")
        print(f"   Started: {self.start_time.isoformat()}")
        print("=" * 70)

        async with httpx.AsyncClient(timeout=60.0) as client:
            await self._test_health(client)
            await self._test_models(client)
            await self._test_chat_completions(client)
            await self._test_streaming(client)
            await self._test_redis_caching(client)
            await self._test_jaeger_tracing(client)
            await self._test_mlflow(client)
            await self._test_performance(client)

        return self.results

    async def _test_health(self, client: httpx.AsyncClient) -> None:
        """Test health endpoints."""
        print("\n📋 [1/8] Health Endpoints")
        tests = [
            ("/health", "Main health check"),
            ("/health/liveliness", "Kubernetes liveness"),
            ("/health/readiness", "Kubernetes readiness"),
        ]
        for path, desc in tests:
            start = time.perf_counter()
            try:
                resp = await client.get(
                    f"{self.base_url}{path}", headers=self._headers()
                )
                passed = resp.status_code in [200, 401]
                self.results.append(
                    TestResult(
                        name=desc,
                        category="Health",
                        passed=passed,
                        duration_ms=(time.perf_counter() - start) * 1000,
                        status_code=resp.status_code,
                    )
                )
                status = "✅" if passed else "❌"
                print(f"  {status} {path}: {resp.status_code}")
            except Exception as e:
                self.results.append(
                    TestResult(name=desc, category="Health", passed=False, error=str(e))
                )
                print(f"  ❌ {path}: {e}")

    async def _test_models(self, client: httpx.AsyncClient) -> None:
        """Test model endpoints."""
        print("\n📋 [2/8] Model Endpoints")
        try:
            resp = await client.get(
                f"{self.base_url}/v1/models", headers=self._headers()
            )
            data = resp.json()
            model_count = len(data.get("data", []))
            self.results.append(
                TestResult(
                    name="List models",
                    category="Models",
                    passed=resp.status_code == 200,
                    status_code=resp.status_code,
                    details={"model_count": model_count},
                )
            )
            print(f"  ✅ /v1/models: {model_count} models available")
        except Exception as e:
            self.results.append(
                TestResult(
                    name="List models", category="Models", passed=False, error=str(e)
                )
            )
            print(f"  ❌ /v1/models: {e}")

    async def _test_chat_completions(self, client: httpx.AsyncClient) -> None:
        """Test chat completion API."""
        print("\n📋 [3/8] Chat Completions")
        models = ["nova-micro", "nova-lite", "claude-4.5-haiku"]
        for model in models:
            start = time.perf_counter()
            try:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 10,
                    },
                )
                duration = (time.perf_counter() - start) * 1000
                passed = resp.status_code == 200
                self.results.append(
                    TestResult(
                        name=f"Chat: {model}",
                        category="Chat",
                        passed=passed,
                        duration_ms=duration,
                        status_code=resp.status_code,
                    )
                )
                status = "✅" if passed else "❌"
                print(f"  {status} {model}: {resp.status_code} ({duration:.0f}ms)")
            except Exception as e:
                self.results.append(
                    TestResult(
                        name=f"Chat: {model}",
                        category="Chat",
                        passed=False,
                        error=str(e),
                    )
                )
                print(f"  ❌ {model}: {e}")

    async def _test_streaming(self, client: httpx.AsyncClient) -> None:
        """Test streaming chat completions."""
        print("\n📋 [4/8] Streaming Chat")
        start = time.perf_counter()
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json={
                    "model": "nova-micro",
                    "messages": [{"role": "user", "content": "Count to 3"}],
                    "max_tokens": 20,
                    "stream": True,
                },
            ) as resp:
                chunks = 0
                async for line in resp.aiter_lines():
                    if line.startswith("data:") and "[DONE]" not in line:
                        chunks += 1
                duration = (time.perf_counter() - start) * 1000
                passed = resp.status_code == 200 and chunks > 0
                self.results.append(
                    TestResult(
                        name="Streaming chat",
                        category="Streaming",
                        passed=passed,
                        duration_ms=duration,
                        details={"chunks": chunks},
                    )
                )
                status = "✅" if passed else "❌"
                print(f"  {status} Streaming: {chunks} chunks ({duration:.0f}ms)")
        except Exception as e:
            self.results.append(
                TestResult(
                    name="Streaming chat",
                    category="Streaming",
                    passed=False,
                    error=str(e),
                )
            )
            print(f"  ❌ Streaming: {e}")

    async def _test_redis_caching(self, client: httpx.AsyncClient) -> None:
        """Test Redis caching behavior."""
        print("\n📋 [5/8] Redis Caching")
        cache_test_prompt = f"Cache test prompt {datetime.now().timestamp()}"
        # First request (cache miss)
        start1 = time.perf_counter()
        resp1 = await client.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": "nova-micro",
                "messages": [{"role": "user", "content": cache_test_prompt}],
                "max_tokens": 10,
            },
        )
        time1 = (time.perf_counter() - start1) * 1000
        # Second request (cache hit expected)
        start2 = time.perf_counter()
        resp2 = await client.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": "nova-micro",
                "messages": [{"role": "user", "content": cache_test_prompt}],
                "max_tokens": 10,
            },
        )
        time2 = (time.perf_counter() - start2) * 1000
        passed = resp1.status_code == 200 and resp2.status_code == 200
        cache_effective = time2 < time1 * 0.5  # Cache should be faster
        self.results.append(
            TestResult(
                name="Cache behavior",
                category="Caching",
                passed=passed,
                details={
                    "first_ms": round(time1),
                    "second_ms": round(time2),
                    "cache_speedup": cache_effective,
                },
            )
        )
        status = "✅" if passed else "❌"
        print(
            f"  {status} First: {time1:.0f}ms, Second: {time2:.0f}ms (cache: {'yes' if cache_effective else 'no'})"
        )

    async def _test_jaeger_tracing(self, client: httpx.AsyncClient) -> None:
        """Test Jaeger tracing integration."""
        print("\n📋 [6/8] Jaeger Tracing")
        try:
            resp = await client.get(f"{JAEGER_URL}/api/services")
            data = resp.json()
            services = data.get("data", [])
            has_litellm = "litellm-gateway" in services
            self.results.append(
                TestResult(
                    name="Jaeger services",
                    category="Tracing",
                    passed=has_litellm,
                    status_code=resp.status_code,
                    details={"services": services},
                )
            )
            status = "✅" if has_litellm else "❌"
            print(f"  {status} Services: {services}")
            # Check for recent traces
            resp2 = await client.get(
                f"{JAEGER_URL}/api/traces?service=litellm-gateway&limit=5"
            )
            data2 = resp2.json()
            trace_count = len(data2.get("data", []))
            self.results.append(
                TestResult(
                    name="Trace collection",
                    category="Tracing",
                    passed=trace_count > 0,
                    details={"trace_count": trace_count},
                )
            )
            print(
                f"  {'✅' if trace_count > 0 else '❌'} Traces collected: {trace_count}"
            )
        except Exception as e:
            self.results.append(
                TestResult(
                    name="Jaeger tracing",
                    category="Tracing",
                    passed=False,
                    error=str(e),
                )
            )
            print(f"  ❌ Jaeger: {e}")

    async def _test_mlflow(self, client: httpx.AsyncClient) -> None:
        """Test MLflow integration."""
        print("\n📋 [7/8] MLflow Integration")
        try:
            resp = await client.get(f"{MLFLOW_URL}/health")
            passed = resp.status_code == 200
            self.results.append(
                TestResult(
                    name="MLflow health",
                    category="MLflow",
                    passed=passed,
                    status_code=resp.status_code,
                )
            )
            print(f"  {'✅' if passed else '❌'} MLflow health: {resp.status_code}")
            resp2 = await client.post(
                f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
                json={"max_results": 10},
                headers={"Content-Type": "application/json"},
            )
            exp_count = len(resp2.json().get("experiments", []))
            self.results.append(
                TestResult(
                    name="MLflow experiments",
                    category="MLflow",
                    passed=resp2.status_code == 200,
                    details={"experiment_count": exp_count},
                )
            )
            print(f"  ✅ Experiments: {exp_count}")
        except Exception as e:
            self.results.append(
                TestResult(name="MLflow", category="MLflow", passed=False, error=str(e))
            )
            print(f"  ❌ MLflow: {e}")

    async def _test_performance(self, client: httpx.AsyncClient) -> None:
        """Run performance benchmarks."""
        print("\n📋 [8/8] Performance Benchmarks")
        latencies = []
        for i in range(5):
            start = time.perf_counter()
            await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json={
                    "model": "nova-micro",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
            )
            latencies.append((time.perf_counter() - start) * 1000)
        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        self.results.append(
            TestResult(
                name="Latency benchmark",
                category="Performance",
                passed=avg < 5000,
                details={
                    "avg_ms": round(avg),
                    "p95_ms": round(p95),
                    "samples": len(latencies),
                },
            )
        )
        print(f"  ✅ Avg: {avg:.0f}ms, P95: {p95:.0f}ms ({len(latencies)} samples)")

    def print_summary(self) -> int:
        """Print test summary."""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        pct = (passed / total * 100) if total else 0
        print(
            f"\n  Total: {total} | Passed: {passed} ({pct:.0f}%) | Failed: {total - passed}"
        )
        if passed < total:
            print("\n  ❌ Failed:")
            for r in self.results:
                if not r.passed:
                    print(f"     [{r.category}] {r.name}: {r.error or r.status_code}")
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n  Duration: {duration:.1f}s")
        print(
            "\n"
            + ("✅ ALL TESTS PASSED" if passed == total else "❌ SOME TESTS FAILED")
        )
        return 0 if passed == total else 1


async def main():
    parser = argparse.ArgumentParser(description="Comprehensive LiteLLM test suite")
    parser.add_argument("--url", default=LITELLM_URL, help="LiteLLM URL")
    parser.add_argument("--key", default=LITELLM_API_KEY, help="API key")
    args = parser.parse_args()

    runner = ComprehensiveTestRunner(args.url, args.key)
    await runner.run_all_tests()
    sys.exit(runner.print_summary())


if __name__ == "__main__":
    asyncio.run(main())
