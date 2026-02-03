#!/usr/bin/env python3
"""
End-to-End Test Script for LiteLLM + LLMRouter Stack

This script performs comprehensive testing of the local Docker Compose stack:
1. Health checks for all services
2. Sends 100 requests across different models
3. Exports Jaeger traces to MinIO/S3
4. Trains a router model and logs to MLflow
5. Uploads artifacts to MinIO

Usage:
    python scripts/e2e_test.py [--requests 100] [--export-traces] [--train-router]
"""

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import boto3
from botocore.client import Config

# Configuration
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4010")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "local-dev-master-key")
JAEGER_URL = os.getenv("JAEGER_URL", "http://localhost:16686")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5050")
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# Test prompts for variety
TEST_PROMPTS = [
    "What is 2+2?",
    "Write a haiku about coding.",
    "Explain machine learning in one sentence.",
    "What is the capital of France?",
    "List 3 programming languages.",
    "What is Python used for?",
    "Define artificial intelligence.",
    "What is a database?",
    "Explain REST API briefly.",
    "What is cloud computing?",
]

# Models to test
MODELS = ["nova-micro", "nova-lite", "nova-pro", "claude-4.5-haiku"]


class E2ETestRunner:
    """End-to-end test runner for the LiteLLM stack."""

    def __init__(self, num_requests: int = 100):
        self.num_requests = num_requests
        self.results: list[dict[str, Any]] = []
        self.start_time = datetime.now()

    async def check_health(self) -> dict[str, bool]:
        """Check health of all services."""
        print("\n" + "=" * 60)
        print("🏥 HEALTH CHECKS")
        print("=" * 60)

        services = {
            "LiteLLM Gateway": f"{LITELLM_URL}/health",
            "Jaeger": f"{JAEGER_URL}/api/services",
            "MLflow": f"{MLFLOW_URL}/health",
        }

        health_status = {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            for name, url in services.items():
                try:
                    headers = {}
                    if "litellm" in name.lower():
                        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"
                    resp = await client.get(url, headers=headers)
                    healthy = resp.status_code == 200
                    health_status[name] = healthy
                    status = "✅" if healthy else "❌"
                    print(f"  {status} {name}: {resp.status_code}")
                except Exception as e:
                    health_status[name] = False
                    print(f"  ❌ {name}: {e}")

        # Check MinIO
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=MINIO_URL,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
                config=Config(signature_version="s3v4"),
            )
            s3.list_buckets()
            health_status["MinIO"] = True
            print("  ✅ MinIO: Connected")
        except Exception as e:
            health_status["MinIO"] = False
            print(f"  ❌ MinIO: {e}")

        return health_status

    async def send_request(
        self, client: httpx.AsyncClient, model: str, prompt: str, request_id: int
    ) -> dict[str, Any]:
        """Send a single chat completion request."""
        start = time.time()
        try:
            resp = await client.post(
                f"{LITELLM_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {LITELLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                },
                timeout=60.0,
            )
            latency = time.time() - start
            data = resp.json()

            if resp.status_code == 200 and "choices" in data:
                return {
                    "request_id": request_id,
                    "model": model,
                    "prompt": prompt[:50],
                    "success": True,
                    "latency_ms": round(latency * 1000, 2),
                    "tokens": data.get("usage", {}).get("total_tokens", 0),
                    "response": data["choices"][0]["message"]["content"][:100],
                }
            else:
                return {
                    "request_id": request_id,
                    "model": model,
                    "success": False,
                    "latency_ms": round(latency * 1000, 2),
                    "error": str(data.get("error", data))[:100],
                }
        except Exception as e:
            return {
                "request_id": request_id,
                "model": model,
                "success": False,
                "latency_ms": round((time.time() - start) * 1000, 2),
                "error": str(e)[:100],
            }

    async def run_load_test(self) -> None:
        """Run load test with specified number of requests."""
        print("\n" + "=" * 60)
        print(f"🚀 LOAD TEST ({self.num_requests} requests)")
        print("=" * 60)

        async with httpx.AsyncClient() as client:
            tasks = []
            for i in range(self.num_requests):
                model = random.choice(MODELS)
                prompt = random.choice(TEST_PROMPTS)
                tasks.append(self.send_request(client, model, prompt, i))

            # Run with concurrency limit
            semaphore = asyncio.Semaphore(10)

            async def limited_request(task):
                async with semaphore:
                    return await task

            self.results = await asyncio.gather(*[limited_request(t) for t in tasks])

        # Print summary
        successful = sum(1 for r in self.results if r.get("success"))
        failed = len(self.results) - successful
        avg_latency = sum(r.get("latency_ms", 0) for r in self.results) / len(
            self.results
        )

        print("\n📊 Results:")
        print(f"   ✅ Successful: {successful}/{self.num_requests}")
        print(f"   ❌ Failed: {failed}/{self.num_requests}")
        print(f"   ⏱️  Avg Latency: {avg_latency:.2f}ms")

        # Per-model breakdown
        print("\n📈 Per-Model Breakdown:")
        for model in MODELS:
            model_results = [r for r in self.results if r.get("model") == model]
            if model_results:
                success_rate = sum(1 for r in model_results if r.get("success")) / len(
                    model_results
                )
                avg_lat = sum(r.get("latency_ms", 0) for r in model_results) / len(
                    model_results
                )
                print(
                    f"   {model}: {len(model_results)} reqs, {success_rate * 100:.1f}% success, {avg_lat:.0f}ms avg"
                )

    async def export_traces(self) -> str | None:
        """Export Jaeger traces to MinIO."""
        print("\n" + "=" * 60)
        print("📤 EXPORTING JAEGER TRACES")
        print("=" * 60)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get traces from Jaeger
                resp = await client.get(
                    f"{JAEGER_URL}/api/traces",
                    params={"service": "litellm-gateway", "limit": 100},
                )
                if resp.status_code != 200:
                    print(f"   ⚠️  Could not fetch traces: {resp.status_code}")
                    return None

                traces = resp.json()
                print(f"   Found {len(traces.get('data', []))} traces")

                # Save to MinIO
                s3 = boto3.client(
                    "s3",
                    endpoint_url=MINIO_URL,
                    aws_access_key_id=MINIO_ACCESS_KEY,
                    aws_secret_access_key=MINIO_SECRET_KEY,
                    config=Config(signature_version="s3v4"),
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                key = f"traces/jaeger_traces_{timestamp}.json"

                s3.put_object(
                    Bucket="mlflow",
                    Key=key,
                    Body=json.dumps(traces, indent=2),
                    ContentType="application/json",
                )
                print(f"   ✅ Uploaded to s3://mlflow/{key}")
                return key

        except Exception as e:
            print(f"   ❌ Error exporting traces: {e}")
            return None

    def save_results(self) -> str:
        """Save test results to file and MinIO."""
        print("\n" + "=" * 60)
        print("💾 SAVING RESULTS")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_data = {
            "timestamp": timestamp,
            "num_requests": self.num_requests,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "summary": {
                "successful": sum(1 for r in self.results if r.get("success")),
                "failed": sum(1 for r in self.results if not r.get("success")),
                "avg_latency_ms": sum(r.get("latency_ms", 0) for r in self.results)
                / max(len(self.results), 1),
            },
            "results": self.results,
        }

        # Save locally
        local_path = Path(f"/tmp/e2e_results_{timestamp}.json")
        local_path.write_text(json.dumps(results_data, indent=2))
        print(f"   📁 Local: {local_path}")

        # Upload to MinIO
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=MINIO_URL,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
                config=Config(signature_version="s3v4"),
            )
            key = f"test-results/e2e_results_{timestamp}.json"
            s3.put_object(
                Bucket="mlflow",
                Key=key,
                Body=json.dumps(results_data, indent=2),
                ContentType="application/json",
            )
            print(f"   ☁️  MinIO: s3://mlflow/{key}")
        except Exception as e:
            print(f"   ⚠️  MinIO upload failed: {e}")

        return str(local_path)


async def main():
    parser = argparse.ArgumentParser(description="E2E Test for LiteLLM + LLMRouter")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument(
        "--export-traces", action="store_true", help="Export Jaeger traces"
    )
    parser.add_argument("--skip-load-test", action="store_true", help="Skip load test")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🧪 LiteLLM + LLMRouter End-to-End Test")
    print("=" * 60)
    print(f"   Requests: {args.requests}")
    print(f"   LiteLLM: {LITELLM_URL}")
    print(f"   Jaeger: {JAEGER_URL}")
    print(f"   MLflow: {MLFLOW_URL}")
    print(f"   MinIO: {MINIO_URL}")

    runner = E2ETestRunner(num_requests=args.requests)

    # Health checks
    health = await runner.check_health()
    if not all(health.values()):
        print("\n⚠️  Some services are unhealthy. Continuing anyway...")

    # Load test
    if not args.skip_load_test:
        await runner.run_load_test()

    # Export traces
    if args.export_traces:
        await runner.export_traces()

    # Save results
    runner.save_results()

    print("\n" + "=" * 60)
    print("✅ E2E TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
