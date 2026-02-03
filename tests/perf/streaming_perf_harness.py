#!/usr/bin/env python3
"""
Streaming Performance Harness (TG10.5)

A developer-runnable performance measurement tool for streaming endpoints.

Measures:
- TTFB (Time To First Byte)
- Chunk cadence (inter-chunk timing statistics)
- Modest concurrency throughput (10-50 concurrent streaming requests)

Target Selection:
- In-process ASGI test server (default): Uses httpx ASGI transport
- Running gateway URL: Set STREAMING_PERF_TARGET_URL environment variable

Usage:
    # In-process mock server (default)
    uv run python -m tests.perf.streaming_perf_harness
    
    # Against running gateway
    STREAMING_PERF_TARGET_URL=http://localhost:4010 uv run python -m tests.perf.streaming_perf_harness
    
    # With custom concurrency
    STREAMING_PERF_CONCURRENCY=25 uv run python -m tests.perf.streaming_perf_harness

Output: Concise summary with TTFB + chunk cadence stats in tabular format.
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# =============================================================================
# Configuration
# =============================================================================

# Target URL - if set, uses external server; otherwise uses in-process ASGI
TARGET_URL = os.getenv("STREAMING_PERF_TARGET_URL", "")

# Concurrency level for load testing
CONCURRENCY = int(os.getenv("STREAMING_PERF_CONCURRENCY", "10"))

# Number of requests per concurrency test
REQUESTS_PER_TEST = int(os.getenv("STREAMING_PERF_REQUESTS", "50"))

# Chunk configuration for mock server
MOCK_CHUNK_COUNT = int(os.getenv("STREAMING_PERF_MOCK_CHUNKS", "20"))
MOCK_CHUNK_SIZE = int(os.getenv("STREAMING_PERF_MOCK_CHUNK_SIZE", "1024"))
MOCK_CHUNK_DELAY_MS = float(os.getenv("STREAMING_PERF_MOCK_DELAY_MS", "10"))


# =============================================================================
# Data Classes for Metrics
# =============================================================================


@dataclass
class StreamingMetrics:
    """Metrics collected from a single streaming request."""
    
    ttfb_ms: float = 0.0
    total_time_ms: float = 0.0
    chunk_count: int = 0
    total_bytes: int = 0
    chunk_times_ms: list[float] = field(default_factory=list)
    error: str | None = None
    
    @property
    def avg_chunk_interval_ms(self) -> float:
        """Average time between chunks."""
        if len(self.chunk_times_ms) < 2:
            return 0.0
        intervals = []
        for i in range(1, len(self.chunk_times_ms)):
            intervals.append(self.chunk_times_ms[i] - self.chunk_times_ms[i-1])
        return statistics.mean(intervals) if intervals else 0.0
    
    @property
    def chunk_interval_stddev_ms(self) -> float:
        """Standard deviation of inter-chunk intervals."""
        if len(self.chunk_times_ms) < 3:
            return 0.0
        intervals = []
        for i in range(1, len(self.chunk_times_ms)):
            intervals.append(self.chunk_times_ms[i] - self.chunk_times_ms[i-1])
        return statistics.stdev(intervals) if len(intervals) >= 2 else 0.0


@dataclass
class AggregateMetrics:
    """Aggregated metrics from multiple requests."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # TTFB stats
    ttfb_min_ms: float = float('inf')
    ttfb_max_ms: float = 0.0
    ttfb_avg_ms: float = 0.0
    ttfb_p50_ms: float = 0.0
    ttfb_p95_ms: float = 0.0
    ttfb_p99_ms: float = 0.0
    
    # Chunk cadence stats
    avg_chunk_interval_ms: float = 0.0
    chunk_interval_stddev_ms: float = 0.0
    
    # Throughput
    total_bytes: int = 0
    total_time_ms: float = 0.0
    requests_per_second: float = 0.0
    bytes_per_second: float = 0.0
    
    @classmethod
    def from_metrics(cls, metrics: list[StreamingMetrics], wall_time_ms: float) -> "AggregateMetrics":
        """Compute aggregate statistics from individual metrics."""
        agg = cls()
        
        agg.total_requests = len(metrics)
        successful = [m for m in metrics if m.error is None]
        agg.successful_requests = len(successful)
        agg.failed_requests = len(metrics) - len(successful)
        
        if not successful:
            return agg
        
        # TTFB stats
        ttfbs = sorted([m.ttfb_ms for m in successful])
        agg.ttfb_min_ms = min(ttfbs)
        agg.ttfb_max_ms = max(ttfbs)
        agg.ttfb_avg_ms = statistics.mean(ttfbs)
        
        # Percentiles
        agg.ttfb_p50_ms = ttfbs[len(ttfbs) // 2]
        agg.ttfb_p95_ms = ttfbs[int(len(ttfbs) * 0.95)]
        agg.ttfb_p99_ms = ttfbs[int(len(ttfbs) * 0.99)]
        
        # Chunk cadence
        all_intervals = []
        for m in successful:
            if len(m.chunk_times_ms) >= 2:
                for i in range(1, len(m.chunk_times_ms)):
                    all_intervals.append(m.chunk_times_ms[i] - m.chunk_times_ms[i-1])
        
        if all_intervals:
            agg.avg_chunk_interval_ms = statistics.mean(all_intervals)
            if len(all_intervals) >= 2:
                agg.chunk_interval_stddev_ms = statistics.stdev(all_intervals)
        
        # Throughput
        agg.total_bytes = sum(m.total_bytes for m in successful)
        agg.total_time_ms = wall_time_ms
        
        if wall_time_ms > 0:
            agg.requests_per_second = (agg.successful_requests / wall_time_ms) * 1000
            agg.bytes_per_second = (agg.total_bytes / wall_time_ms) * 1000
        
        return agg


# =============================================================================
# Mock ASGI Application for In-Process Testing
# =============================================================================


class MockStreamingApp:
    """
    Mock ASGI application that emits streaming responses.
    
    This simulates an upstream server with configurable:
    - Number of chunks
    - Chunk size
    - Inter-chunk delay
    """
    
    def __init__(
        self,
        chunk_count: int = 20,
        chunk_size: int = 1024,
        chunk_delay_ms: float = 10.0,
    ):
        self.chunk_count = chunk_count
        self.chunk_size = chunk_size
        self.chunk_delay_s = chunk_delay_ms / 1000.0
    
    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        """ASGI application entrypoint."""
        if scope["type"] != "http":
            return
        
        path = scope.get("path", "/")
        
        if path == "/stream":
            await self._handle_stream(scope, receive, send)
        elif path == "/health":
            await self._handle_health(scope, receive, send)
        else:
            await self._handle_not_found(scope, receive, send)
    
    async def _handle_stream(self, scope: dict, receive: Any, send: Any) -> None:
        """Handle streaming endpoint."""
        # Send response headers
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/octet-stream"),
                (b"transfer-encoding", b"chunked"),
            ],
        })
        
        # Stream chunks with delays - send first chunk immediately for accurate TTFB
        for i in range(self.chunk_count):
            chunk = f"chunk-{i:04d}-".encode() + b"X" * (self.chunk_size - 12)
            
            # Delay BEFORE sending (except first chunk for TTFB accuracy)
            if i > 0 and self.chunk_delay_s > 0:
                await asyncio.sleep(self.chunk_delay_s)
            
            await send({
                "type": "http.response.body",
                "body": chunk,
                "more_body": i < self.chunk_count - 1,
            })
    
    async def _handle_health(self, scope: dict, receive: Any, send: Any) -> None:
        """Handle health check."""
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"status": "ok"}',
            "more_body": False,
        })
    
    async def _handle_not_found(self, scope: dict, receive: Any, send: Any) -> None:
        """Handle 404."""
        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"error": "not found"}',
            "more_body": False,
        })


# =============================================================================
# Performance Test Functions
# =============================================================================


async def measure_streaming_request(
    client: httpx.AsyncClient,
    url: str,
) -> StreamingMetrics:
    """
    Measure a single streaming request.
    
    Returns:
        StreamingMetrics with TTFB, chunk timing, etc.
    """
    metrics = StreamingMetrics()
    start_time = time.monotonic()
    first_chunk_time = None
    
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            
            async for chunk in response.aiter_bytes():
                current_time = time.monotonic()
                
                if first_chunk_time is None:
                    first_chunk_time = current_time
                    metrics.ttfb_ms = (first_chunk_time - start_time) * 1000
                
                metrics.chunk_times_ms.append((current_time - start_time) * 1000)
                metrics.chunk_count += 1
                metrics.total_bytes += len(chunk)
        
        metrics.total_time_ms = (time.monotonic() - start_time) * 1000
        
    except Exception as e:
        metrics.error = str(e)
        metrics.total_time_ms = (time.monotonic() - start_time) * 1000
    
    return metrics


async def run_concurrent_test(
    client: httpx.AsyncClient,
    url: str,
    concurrency: int,
    total_requests: int,
) -> tuple[list[StreamingMetrics], float]:
    """
    Run concurrent streaming requests.
    
    Returns:
        Tuple of (list of metrics, wall clock time in ms)
    """
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request() -> StreamingMetrics:
        async with semaphore:
            return await measure_streaming_request(client, url)
    
    start_time = time.monotonic()
    
    tasks = [asyncio.create_task(bounded_request()) for _ in range(total_requests)]
    results = await asyncio.gather(*tasks)
    
    wall_time_ms = (time.monotonic() - start_time) * 1000
    
    return list(results), wall_time_ms


# =============================================================================
# Output Formatting
# =============================================================================


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_metrics(agg: AggregateMetrics, test_name: str) -> None:
    """Print metrics in a formatted table."""
    print()
    print(f"Test: {test_name}")
    print("-" * 50)
    
    print(f"  Requests:      {agg.total_requests:>8} total")
    print(f"                 {agg.successful_requests:>8} successful")
    print(f"                 {agg.failed_requests:>8} failed")
    print()
    
    print("  TTFB (Time To First Byte):")
    print(f"    Min:         {agg.ttfb_min_ms:>8.2f} ms")
    print(f"    Max:         {agg.ttfb_max_ms:>8.2f} ms")
    print(f"    Avg:         {agg.ttfb_avg_ms:>8.2f} ms")
    print(f"    P50:         {agg.ttfb_p50_ms:>8.2f} ms")
    print(f"    P95:         {agg.ttfb_p95_ms:>8.2f} ms")
    print(f"    P99:         {agg.ttfb_p99_ms:>8.2f} ms")
    print()
    
    print("  Chunk Cadence:")
    print(f"    Avg Interval:{agg.avg_chunk_interval_ms:>8.2f} ms")
    print(f"    Stddev:      {agg.chunk_interval_stddev_ms:>8.2f} ms")
    print()
    
    print("  Throughput:")
    print(f"    Total Bytes: {agg.total_bytes:>8,} bytes")
    print(f"    Wall Time:   {agg.total_time_ms:>8.2f} ms")
    print(f"    RPS:         {agg.requests_per_second:>8.2f} req/s")
    print(f"    Bandwidth:   {agg.bytes_per_second / 1024:>8.2f} KB/s")


def print_summary(results: dict[str, AggregateMetrics]) -> None:
    """Print a final summary table."""
    print_header("SUMMARY")
    
    # Header row
    print(f"{'Test':<25} {'TTFB P50':>10} {'TTFB P95':>10} {'Chunk Int':>10} {'RPS':>10}")
    print("-" * 70)
    
    for name, agg in results.items():
        print(
            f"{name:<25} "
            f"{agg.ttfb_p50_ms:>9.2f}ms "
            f"{agg.ttfb_p95_ms:>9.2f}ms "
            f"{agg.avg_chunk_interval_ms:>9.2f}ms "
            f"{agg.requests_per_second:>9.2f}"
        )


# =============================================================================
# Main Test Runner
# =============================================================================


async def run_harness(
    target_url: str | None = None,
    concurrency_levels: list[int] | None = None,
    requests_per_test: int = REQUESTS_PER_TEST,
) -> dict[str, AggregateMetrics]:
    """
    Run the complete performance harness.
    
    Args:
        target_url: URL to test. If None, uses in-process mock server.
        concurrency_levels: List of concurrency levels to test.
        requests_per_test: Number of requests per test.
    
    Returns:
        Dict mapping test name to aggregate metrics.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 10, 25, 50]
    
    results: dict[str, AggregateMetrics] = {}
    
    print_header("Streaming Performance Harness (TG10.5)")
    
    # Determine target
    if target_url:
        base_url = target_url.rstrip("/")
        stream_url = f"{base_url}/stream"
        print(f"Target: External server at {base_url}")
        transport = None
    else:
        # Use in-process mock server
        print("Target: In-process mock ASGI server")
        print(f"  Mock config: {MOCK_CHUNK_COUNT} chunks × {MOCK_CHUNK_SIZE} bytes, {MOCK_CHUNK_DELAY_MS}ms delay")
        
        mock_app = MockStreamingApp(
            chunk_count=MOCK_CHUNK_COUNT,
            chunk_size=MOCK_CHUNK_SIZE,
            chunk_delay_ms=MOCK_CHUNK_DELAY_MS,
        )
        transport = httpx.ASGITransport(app=mock_app)
        base_url = "http://testserver"
        stream_url = f"{base_url}/stream"
    
    print(f"Requests per test: {requests_per_test}")
    print(f"Concurrency levels: {concurrency_levels}")
    
    # Create client
    client_kwargs = {"timeout": httpx.Timeout(60.0)}
    if transport:
        client_kwargs["transport"] = transport
        client_kwargs["base_url"] = base_url
    
    async with httpx.AsyncClient(**client_kwargs) as client:
        
        # Run tests at different concurrency levels
        for concurrency in concurrency_levels:
            test_name = f"Concurrency={concurrency}"
            print_header(f"Running: {test_name}")
            
            metrics, wall_time = await run_concurrent_test(
                client=client,
                url=stream_url,
                concurrency=concurrency,
                total_requests=requests_per_test,
            )
            
            agg = AggregateMetrics.from_metrics(metrics, wall_time)
            results[test_name] = agg
            
            print_metrics(agg, test_name)
    
    # Final summary
    print_summary(results)
    
    return results


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Streaming Performance Harness for TG10.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with in-process mock server
  uv run python -m tests.perf.streaming_perf_harness
  
  # Run against external server
  uv run python -m tests.perf.streaming_perf_harness --url http://localhost:4010
  
  # Custom concurrency
  uv run python -m tests.perf.streaming_perf_harness --concurrency 1,10,50
  
Environment Variables:
  STREAMING_PERF_TARGET_URL     External server URL
  STREAMING_PERF_CONCURRENCY    Default concurrency level
  STREAMING_PERF_REQUESTS       Requests per test (default: 50)
  STREAMING_PERF_MOCK_CHUNKS    Mock server chunk count (default: 20)
  STREAMING_PERF_MOCK_CHUNK_SIZE Mock server chunk size (default: 1024)
  STREAMING_PERF_MOCK_DELAY_MS  Mock server inter-chunk delay (default: 10)
""",
    )
    
    parser.add_argument(
        "--url",
        type=str,
        default=TARGET_URL or None,
        help="Target server URL. If not set, uses in-process mock server.",
    )
    
    parser.add_argument(
        "--concurrency",
        type=str,
        default=None,
        help="Comma-separated concurrency levels (e.g., '1,10,25,50')",
    )
    
    parser.add_argument(
        "--requests",
        type=int,
        default=REQUESTS_PER_TEST,
        help=f"Number of requests per test (default: {REQUESTS_PER_TEST})",
    )
    
    args = parser.parse_args()
    
    # Parse concurrency levels
    concurrency_levels = None
    if args.concurrency:
        concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",")]
    
    # Run the harness
    try:
        asyncio.run(run_harness(
            target_url=args.url,
            concurrency_levels=concurrency_levels,
            requests_per_test=args.requests,
        ))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
