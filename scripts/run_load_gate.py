#!/usr/bin/env python3
"""
TG6.1 Load/Soak Gate Runner

Runs deterministic load testing against the streaming perf stack and enforces
latency (TTFB p95/p99) and error rate thresholds.

This script:
1. Starts the docker-compose.streaming-perf.yml stack (or assumes it's running)
2. Generates concurrent streaming requests over a configurable duration
3. Measures TTFB (Time-To-First-Byte) latency and error rates
4. Enforces configurable thresholds, exiting non-zero on failure

Modes:
- load (default): Quick PR gate (~30s, lower concurrency)
- soak: Nightly extended test (~300s, sustained load)

Usage:
    # Run quick load gate (default, for PR)
    uv run python scripts/run_load_gate.py

    # Run soak gate (for nightly)
    uv run python scripts/run_load_gate.py --mode soak

    # With custom thresholds
    uv run python scripts/run_load_gate.py --ttfb-p95-ms 300 --error-rate-pct 1.0

    # Show help
    uv run python scripts/run_load_gate.py --help

Environment Variables (override defaults):
    LOAD_GATE_MODE: load or soak (default: load)
    LOAD_GATE_DURATION_S: Test duration in seconds
    LOAD_GATE_CONCURRENCY: Number of concurrent requests
    LOAD_GATE_TTFB_P95_MS: p95 TTFB threshold in ms (default: 500)
    LOAD_GATE_TTFB_P99_MS: p99 TTFB threshold in ms (default: 1000)
    LOAD_GATE_ERROR_RATE_PCT: Max error rate percentage (default: 1.0)
    LOAD_GATE_STUB_URL: Stub server URL (default: http://localhost:9200)
    LOAD_GATE_SKIP_COMPOSE: If "1", skip compose up/down (assume stack is running)

Exit Codes:
    0: All thresholds passed
    1: Threshold violation(s) detected
    2: Infrastructure/setup error

Requirements:
    - finch or docker CLI installed
    - docker-compose.streaming-perf.yml available
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Run: uv add httpx")
    sys.exit(2)


# =============================================================================
# Configuration & Defaults
# =============================================================================

# Mode presets
MODE_PRESETS = {
    "load": {
        "duration_s": 30,
        "concurrency": 5,
        "description": "Quick load test for PR gate",
    },
    "soak": {
        "duration_s": 300,  # 5 minutes
        "concurrency": 10,
        "description": "Extended soak test for nightly",
    },
}

# Default thresholds (can be overridden via CLI/env)
DEFAULT_TTFB_P95_MS = 500.0
DEFAULT_TTFB_P99_MS = 1000.0
DEFAULT_ERROR_RATE_PCT = 1.0  # 1% max error rate

# Streaming stub config
DEFAULT_STUB_URL = "http://localhost:9200"
DEFAULT_MARKER_COUNT = 10  # Shorter streams for load test
DEFAULT_INTERVAL_MS = 50  # 50ms between markers

# Compose config
COMPOSE_FILE = "docker-compose.streaming-perf.yml"
COMPOSE_CMD: Optional[str] = None
for cmd in ("finch", "docker"):
    if shutil.which(cmd):
        COMPOSE_CMD = cmd
        break


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RequestResult:
    """Result from a single streaming request."""

    request_id: int
    ttfb_ms: float = 0.0
    total_ms: float = 0.0
    marker_count: int = 0
    success: bool = False
    error: Optional[str] = None


@dataclass
class LoadTestResult:
    """Aggregated results from load test."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    ttfb_values: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def error_rate_pct(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    @property
    def ttfb_p50(self) -> float:
        if not self.ttfb_values:
            return 0.0
        return statistics.median(self.ttfb_values)

    @property
    def ttfb_p95(self) -> float:
        if not self.ttfb_values:
            return 0.0
        return statistics.quantiles(self.ttfb_values, n=20)[18]  # 95th percentile

    @property
    def ttfb_p99(self) -> float:
        if not self.ttfb_values:
            return 0.0
        return statistics.quantiles(self.ttfb_values, n=100)[98]  # 99th percentile

    @property
    def ttfb_max(self) -> float:
        if not self.ttfb_values:
            return 0.0
        return max(self.ttfb_values)

    @property
    def requests_per_second(self) -> float:
        if self.duration_s == 0:
            return 0.0
        return self.total_requests / self.duration_s


@dataclass
class ThresholdConfig:
    """Threshold configuration for gate validation."""

    ttfb_p95_ms: float = DEFAULT_TTFB_P95_MS
    ttfb_p99_ms: float = DEFAULT_TTFB_P99_MS
    error_rate_pct: float = DEFAULT_ERROR_RATE_PCT


@dataclass
class ThresholdViolation:
    """A single threshold violation."""

    metric: str
    threshold: float
    actual: float
    unit: str


# =============================================================================
# Streaming Request Logic
# =============================================================================


async def make_streaming_request(
    client: httpx.AsyncClient,
    stub_url: str,
    request_id: int,
    marker_count: int = DEFAULT_MARKER_COUNT,
    interval_ms: float = DEFAULT_INTERVAL_MS,
) -> RequestResult:
    """Make a single streaming request and measure TTFB."""
    result = RequestResult(request_id=request_id)
    start_time = time.monotonic()

    try:
        async with client.stream(
            "GET",
            f"{stub_url}/stream",
            params={"markers": marker_count, "interval_ms": interval_ms},
            timeout=httpx.Timeout(30.0),
        ) as response:
            response.raise_for_status()

            first_byte_received = False
            async for line in response.aiter_lines():
                if not first_byte_received:
                    result.ttfb_ms = (time.monotonic() - start_time) * 1000
                    first_byte_received = True

                if line.strip():
                    result.marker_count += 1

            result.total_ms = (time.monotonic() - start_time) * 1000
            result.success = True

    except httpx.HTTPStatusError as e:
        result.error = f"HTTP {e.response.status_code}: {e.response.text[:100]}"
        result.total_ms = (time.monotonic() - start_time) * 1000
    except httpx.RequestError as e:
        result.error = f"Request error: {type(e).__name__}: {str(e)[:100]}"
        result.total_ms = (time.monotonic() - start_time) * 1000
    except Exception as e:
        result.error = f"Unexpected error: {type(e).__name__}: {str(e)[:100]}"
        result.total_ms = (time.monotonic() - start_time) * 1000

    return result


async def run_load_test(
    stub_url: str,
    duration_s: float,
    concurrency: int,
    marker_count: int = DEFAULT_MARKER_COUNT,
    interval_ms: float = DEFAULT_INTERVAL_MS,
) -> LoadTestResult:
    """Run load test for specified duration with given concurrency."""
    result = LoadTestResult()
    start_time = time.monotonic()
    end_time = start_time + duration_s
    request_counter = 0
    active_tasks: set[asyncio.Task] = set()

    async with httpx.AsyncClient() as client:
        while time.monotonic() < end_time or active_tasks:
            # Start new tasks if under concurrency limit and still in duration
            while len(active_tasks) < concurrency and time.monotonic() < end_time:
                request_counter += 1
                task = asyncio.create_task(
                    make_streaming_request(
                        client,
                        stub_url,
                        request_counter,
                        marker_count,
                        interval_ms,
                    )
                )
                active_tasks.add(task)

            # Wait for at least one task to complete
            if active_tasks:
                done, active_tasks = await asyncio.wait(
                    active_tasks,
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    req_result: RequestResult = task.result()
                    result.total_requests += 1

                    if req_result.success:
                        result.successful_requests += 1
                        result.ttfb_values.append(req_result.ttfb_ms)
                    else:
                        result.failed_requests += 1
                        if req_result.error:
                            result.errors.append(req_result.error)

            # Small sleep to prevent tight loop
            await asyncio.sleep(0.01)

    result.duration_s = time.monotonic() - start_time
    return result


# =============================================================================
# Compose Management
# =============================================================================


def start_compose_stack() -> bool:
    """Start the streaming perf compose stack."""
    if COMPOSE_CMD is None:
        print("ERROR: Neither finch nor docker CLI found")
        return False

    compose_base = [COMPOSE_CMD, "compose", "-f", COMPOSE_FILE]
    print(f"🚀 Starting compose stack with {COMPOSE_CMD}...")

    try:
        subprocess.run(
            compose_base + ["up", "-d", "--build"],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to start compose stack: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("ERROR: Compose up timed out after 3 minutes")
        return False

    # Wait for health
    print("⏳ Waiting for services to be healthy...")
    stub_url = os.environ.get("LOAD_GATE_STUB_URL", DEFAULT_STUB_URL)
    max_wait = 60
    start = time.monotonic()

    while time.monotonic() - start < max_wait:
        try:
            resp = httpx.get(f"{stub_url}/health", timeout=5.0)
            if resp.status_code == 200:
                print("✅ Services healthy")
                return True
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(2)

    print("ERROR: Services did not become healthy within 60 seconds")
    return False


def stop_compose_stack() -> None:
    """Stop the streaming perf compose stack."""
    if COMPOSE_CMD is None:
        return

    compose_base = [COMPOSE_CMD, "compose", "-f", COMPOSE_FILE]
    print("\n🧹 Tearing down compose stack...")
    subprocess.run(
        compose_base + ["down", "-v"],
        capture_output=True,
        timeout=60,
    )


# =============================================================================
# Threshold Evaluation
# =============================================================================


def evaluate_thresholds(
    result: LoadTestResult,
    config: ThresholdConfig,
) -> list[ThresholdViolation]:
    """Evaluate results against thresholds, return list of violations."""
    violations = []

    # TTFB p95 check
    if result.ttfb_p95 > config.ttfb_p95_ms:
        violations.append(
            ThresholdViolation(
                metric="TTFB p95",
                threshold=config.ttfb_p95_ms,
                actual=result.ttfb_p95,
                unit="ms",
            )
        )

    # TTFB p99 check
    if result.ttfb_p99 > config.ttfb_p99_ms:
        violations.append(
            ThresholdViolation(
                metric="TTFB p99",
                threshold=config.ttfb_p99_ms,
                actual=result.ttfb_p99,
                unit="ms",
            )
        )

    # Error rate check
    if result.error_rate_pct > config.error_rate_pct:
        violations.append(
            ThresholdViolation(
                metric="Error rate",
                threshold=config.error_rate_pct,
                actual=result.error_rate_pct,
                unit="%",
            )
        )

    return violations


# =============================================================================
# Result Reporting
# =============================================================================


def print_results(
    result: LoadTestResult,
    config: ThresholdConfig,
    violations: list[ThresholdViolation],
) -> None:
    """Print detailed results and threshold evaluation."""
    print("\n" + "=" * 60)
    print("📊 LOAD/SOAK GATE RESULTS")
    print("=" * 60)

    print("\n📈 Test Summary:")
    print(f"   Duration: {result.duration_s:.1f}s")
    print(f"   Total requests: {result.total_requests}")
    print(f"   Successful: {result.successful_requests}")
    print(f"   Failed: {result.failed_requests}")
    print(f"   Requests/sec: {result.requests_per_second:.1f}")

    print("\n⏱️  TTFB Latency:")
    print(f"   p50: {result.ttfb_p50:.1f}ms")
    print(f"   p95: {result.ttfb_p95:.1f}ms (threshold: {config.ttfb_p95_ms:.1f}ms)")
    print(f"   p99: {result.ttfb_p99:.1f}ms (threshold: {config.ttfb_p99_ms:.1f}ms)")
    print(f"   max: {result.ttfb_max:.1f}ms")

    print("\n❌ Error Rate:")
    print(f"   {result.error_rate_pct:.2f}% (threshold: {config.error_rate_pct:.1f}%)")

    if result.errors:
        print("\n⚠️  Sample Errors (first 5):")
        for err in result.errors[:5]:
            print(f"   - {err}")

    print("\n" + "-" * 60)
    if violations:
        print("🔴 GATE FAILED - Threshold violations:")
        for v in violations:
            print(f"   ❌ {v.metric}: {v.actual:.2f}{v.unit} > {v.threshold:.1f}{v.unit}")
    else:
        print("🟢 GATE PASSED - All thresholds met")
    print("=" * 60)


def output_json_report(
    result: LoadTestResult,
    config: ThresholdConfig,
    violations: list[ThresholdViolation],
    output_path: Optional[str],
) -> None:
    """Output JSON report for CI artifacts."""
    report = {
        "summary": {
            "duration_s": result.duration_s,
            "total_requests": result.total_requests,
            "successful_requests": result.successful_requests,
            "failed_requests": result.failed_requests,
            "requests_per_second": result.requests_per_second,
        },
        "ttfb": {
            "p50_ms": result.ttfb_p50,
            "p95_ms": result.ttfb_p95,
            "p99_ms": result.ttfb_p99,
            "max_ms": result.ttfb_max,
        },
        "error_rate_pct": result.error_rate_pct,
        "thresholds": {
            "ttfb_p95_ms": config.ttfb_p95_ms,
            "ttfb_p99_ms": config.ttfb_p99_ms,
            "error_rate_pct": config.error_rate_pct,
        },
        "violations": [
            {
                "metric": v.metric,
                "threshold": v.threshold,
                "actual": v.actual,
                "unit": v.unit,
            }
            for v in violations
        ],
        "passed": len(violations) == 0,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 JSON report written to: {output_path}")
    else:
        # Print to stdout for CI parsing if no file specified
        print("\n--- JSON Report ---")
        print(json.dumps(report, indent=2))


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["load", "soak"],
        default=os.environ.get("LOAD_GATE_MODE", "load"),
        help="Test mode: 'load' (quick PR gate) or 'soak' (nightly). Default: load",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Test duration in seconds (overrides mode preset)",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent requests (overrides mode preset)",
    )

    parser.add_argument(
        "--ttfb-p95-ms",
        type=float,
        default=float(os.environ.get("LOAD_GATE_TTFB_P95_MS", DEFAULT_TTFB_P95_MS)),
        help=f"TTFB p95 threshold in ms. Default: {DEFAULT_TTFB_P95_MS}",
    )

    parser.add_argument(
        "--ttfb-p99-ms",
        type=float,
        default=float(os.environ.get("LOAD_GATE_TTFB_P99_MS", DEFAULT_TTFB_P99_MS)),
        help=f"TTFB p99 threshold in ms. Default: {DEFAULT_TTFB_P99_MS}",
    )

    parser.add_argument(
        "--error-rate-pct",
        type=float,
        default=float(os.environ.get("LOAD_GATE_ERROR_RATE_PCT", DEFAULT_ERROR_RATE_PCT)),
        help=f"Max error rate percentage. Default: {DEFAULT_ERROR_RATE_PCT}",
    )

    parser.add_argument(
        "--stub-url",
        default=os.environ.get("LOAD_GATE_STUB_URL", DEFAULT_STUB_URL),
        help=f"Streaming stub URL. Default: {DEFAULT_STUB_URL}",
    )

    parser.add_argument(
        "--skip-compose",
        action="store_true",
        default=os.environ.get("LOAD_GATE_SKIP_COMPOSE", "0") == "1",
        help="Skip compose up/down (assume stack is already running)",
    )

    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to write JSON report (optional)",
    )

    parser.add_argument(
        "--marker-count",
        type=int,
        default=DEFAULT_MARKER_COUNT,
        help=f"Markers per stream. Default: {DEFAULT_MARKER_COUNT}",
    )

    parser.add_argument(
        "--interval-ms",
        type=float,
        default=DEFAULT_INTERVAL_MS,
        help=f"Interval between markers in ms. Default: {DEFAULT_INTERVAL_MS}",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Get mode preset
    preset = MODE_PRESETS[args.mode]
    print(f"🔧 Mode: {args.mode} - {preset['description']}")

    # Resolve duration and concurrency (CLI overrides preset)
    duration_s = args.duration if args.duration is not None else preset["duration_s"]
    concurrency = args.concurrency if args.concurrency is not None else preset["concurrency"]

    # Override from env if set
    if os.environ.get("LOAD_GATE_DURATION_S"):
        duration_s = float(os.environ["LOAD_GATE_DURATION_S"])
    if os.environ.get("LOAD_GATE_CONCURRENCY"):
        concurrency = int(os.environ["LOAD_GATE_CONCURRENCY"])

    print(f"   Duration: {duration_s}s")
    print(f"   Concurrency: {concurrency}")
    print(f"   Stub URL: {args.stub_url}")

    # Threshold config
    threshold_config = ThresholdConfig(
        ttfb_p95_ms=args.ttfb_p95_ms,
        ttfb_p99_ms=args.ttfb_p99_ms,
        error_rate_pct=args.error_rate_pct,
    )

    print("\n📏 Thresholds:")
    print(f"   TTFB p95: {threshold_config.ttfb_p95_ms}ms")
    print(f"   TTFB p99: {threshold_config.ttfb_p99_ms}ms")
    print(f"   Error rate: {threshold_config.error_rate_pct}%")

    # Start compose if needed
    compose_started = False
    if not args.skip_compose:
        if not start_compose_stack():
            return 2
        compose_started = True
    else:
        print("\n⏭️  Skipping compose (--skip-compose)")
        # Quick health check
        try:
            resp = httpx.get(f"{args.stub_url}/health", timeout=5.0)
            if resp.status_code != 200:
                print(f"ERROR: Stub server not healthy at {args.stub_url}")
                return 2
        except Exception as e:
            print(f"ERROR: Cannot reach stub server at {args.stub_url}: {e}")
            return 2

    try:
        # Run load test
        print(f"\n🏃 Running load test for {duration_s}s...")
        result = asyncio.run(
            run_load_test(
                stub_url=args.stub_url,
                duration_s=duration_s,
                concurrency=concurrency,
                marker_count=args.marker_count,
                interval_ms=args.interval_ms,
            )
        )

        # Evaluate thresholds
        violations = evaluate_thresholds(result, threshold_config)

        # Print results
        print_results(result, threshold_config, violations)

        # Output JSON if requested
        if args.json_output or os.environ.get("CI"):
            output_json_report(
                result,
                threshold_config,
                violations,
                args.json_output,
            )

        # Return exit code based on violations
        return 1 if violations else 0

    finally:
        # Cleanup compose if we started it
        if compose_started:
            stop_compose_stack()


if __name__ == "__main__":
    sys.exit(main())
