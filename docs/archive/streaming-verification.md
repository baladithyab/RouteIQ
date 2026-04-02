# Streaming Verification Guide (TG10.5)

This document covers how to run streaming correctness tests and performance harnesses for verifying the raw streaming passthrough implementation.

## Overview

TG10.5 provides:
1. **Streaming Correctness Tests** - Validate no line buffering, incremental yields, byte integrity, and cancellation behavior
2. **Performance Harness** - Measure TTFB, chunk cadence, and concurrent throughput

## Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager

## Running Streaming Correctness Tests

The correctness tests validate:
- **No line buffering**: Newlines inside chunks don't delay emission
- **Incremental yields**: No full-response buffering
- **Byte integrity**: Hash(input) == Hash(output)
- **Cancellation/backpressure**: Client disconnect stops upstream read

### Run all streaming correctness tests

```bash
uv run pytest tests/unit/test_streaming_correctness.py -v
```

### Run specific test categories

```bash
# No line buffering tests
uv run pytest tests/unit/test_streaming_correctness.py::TestNoLineBuffering -v

# Incremental yield tests
uv run pytest tests/unit/test_streaming_correctness.py::TestIncrementalYields -v

# Byte integrity tests
uv run pytest tests/unit/test_streaming_correctness.py::TestByteIntegrity -v

# Cancellation/backpressure tests
uv run pytest tests/unit/test_streaming_correctness.py::TestCancellationBackpressure -v

# Edge cases
uv run pytest tests/unit/test_streaming_correctness.py::TestStreamingEdgeCases -v

# Mode comparison (raw vs buffered)
uv run pytest tests/unit/test_streaming_correctness.py::TestModeComparison -v
```

### Run with existing A2A streaming passthrough tests

```bash
uv run pytest tests/unit/test_a2a_streaming_passthrough.py tests/unit/test_streaming_correctness.py -v
```

## Running the Performance Harness

The performance harness measures:
- **TTFB** (Time To First Byte)
- **Chunk cadence** (inter-chunk timing statistics)
- **Concurrent throughput** (requests per second at various concurrency levels)

### Target Selection

The harness can target:
1. **In-process ASGI test server** (default) - Uses httpx ASGI transport with a mock streaming server
2. **Running gateway URL** - Set via `STREAMING_PERF_TARGET_URL` environment variable

### Basic Usage

```bash
# Run with in-process mock server (default)
uv run python -m tests.perf.streaming_perf_harness

# Run against a running gateway
STREAMING_PERF_TARGET_URL=http://localhost:4010 uv run python -m tests.perf.streaming_perf_harness

# Custom concurrency levels
uv run python -m tests.perf.streaming_perf_harness --concurrency 1,10,25,50

# More requests per test (for more stable statistics)
uv run python -m tests.perf.streaming_perf_harness --requests 100
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMING_PERF_TARGET_URL` | (empty) | External server URL. If empty, uses in-process mock |
| `STREAMING_PERF_CONCURRENCY` | `10` | Default concurrency level |
| `STREAMING_PERF_REQUESTS` | `50` | Requests per test |
| `STREAMING_PERF_MOCK_CHUNKS` | `20` | Mock server: number of chunks per response |
| `STREAMING_PERF_MOCK_CHUNK_SIZE` | `1024` | Mock server: bytes per chunk |
| `STREAMING_PERF_MOCK_DELAY_MS` | `10` | Mock server: milliseconds between chunks |

### Example Output

```
======================================================================
  Streaming Performance Harness (TG10.5)
======================================================================
Target: In-process mock ASGI server
  Mock config: 20 chunks × 1024 bytes, 10.0ms delay
Requests per test: 50
Concurrency levels: [1, 10, 25, 50]

======================================================================
  Running: Concurrency=1
======================================================================

Test: Concurrency=1
--------------------------------------------------
  Requests:            50 total
                       50 successful
                        0 failed

  TTFB (Time To First Byte):
    Min:            10.23 ms
    Max:            15.67 ms
    Avg:            11.45 ms
    P50:            11.12 ms
    P95:            14.23 ms
    P99:            15.01 ms

  Chunk Cadence:
    Avg Interval:   10.12 ms
    Stddev:          0.87 ms

  Throughput:
    Total Bytes:  1,024,000 bytes
    Wall Time:    10234.56 ms
    RPS:             4.89 req/s
    Bandwidth:      100.12 KB/s

[... more concurrency levels ...]

======================================================================
  SUMMARY
======================================================================
Test                      TTFB P50   TTFB P95  Chunk Int        RPS
----------------------------------------------------------------------
Concurrency=1               11.12ms    14.23ms    10.12ms       4.89
Concurrency=10              12.45ms    18.67ms    10.34ms      42.15
Concurrency=25              15.78ms    24.12ms    10.89ms      89.34
Concurrency=50              23.45ms    45.67ms    11.23ms     142.67
```

## Running with a Local Stack (Finch Compose)

If you need to test against the full gateway stack:

### Start the stack with Finch

```bash
# Start the local test stack
finch compose -f docker-compose.yml up -d

# Wait for services to be ready
sleep 10

# Verify the gateway is running
curl http://localhost:4010/health

# Run the performance harness against the gateway
STREAMING_PERF_TARGET_URL=http://localhost:4010 uv run python -m tests.perf.streaming_perf_harness

# When done, stop the stack
finch compose -f docker-compose.yml down
```

### Using the local test compose file (if available)

```bash
finch compose -f docker-compose.local-test.yml up -d
STREAMING_PERF_TARGET_URL=http://localhost:4010 uv run python -m tests.perf.streaming_perf_harness
finch compose -f docker-compose.local-test.yml down
```

## Running Linters

```bash
# Run ruff linter
uv run ruff check tests/unit/test_streaming_correctness.py tests/perf/

# Run ruff with fixes
uv run ruff check --fix tests/unit/test_streaming_correctness.py tests/perf/

# Run mypy type checking
uv run mypy tests/unit/test_streaming_correctness.py tests/perf/
```

## CI Integration

For CI pipelines, add these steps:

```yaml
# Run streaming correctness tests
- name: Run streaming correctness tests
  run: uv run pytest tests/unit/test_streaming_correctness.py -v --tb=short

# Optional: Run performance harness (may add latency to CI)
- name: Run streaming performance harness
  run: |
    uv run python -m tests.perf.streaming_perf_harness \
      --concurrency 1,10 \
      --requests 20
```

## Troubleshooting

### Tests failing with import errors

Ensure you're running from the project root and have installed dependencies:

```bash
uv sync
uv run pytest tests/unit/test_streaming_correctness.py -v
```

### Performance harness shows high TTFB

1. Check if the mock server delay is configured correctly
2. For external targets, verify network latency
3. Ensure no resource contention on the test machine

### Cancellation tests flaky

The cancellation tests verify backpressure behavior which can be timing-sensitive. If tests are flaky:

1. Increase the inter-chunk delay in the mock
2. Run tests in isolation: `uv run pytest tests/unit/test_streaming_correctness.py::TestCancellationBackpressure -v --count=10`

## Related Documentation

- [A2A Gateway](./a2a-gateway.md) - A2A protocol implementation details
- [Configuration](./configuration.md) - Feature flags including `A2A_RAW_STREAMING_ENABLED`
- [Observability](./observability.md) - Streaming span tracing
