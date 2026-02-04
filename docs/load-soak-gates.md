# Load and Soak Gates (TG6.1)

This document describes the Load/Soak gate infrastructure for CI quality validation.

## Overview

The load/soak gates validate streaming performance metrics under concurrent load:

| Metric | Default Threshold | Description |
|--------|-------------------|-------------|
| **TTFB p95** | 500ms | 95th percentile Time-To-First-Byte |
| **TTFB p99** | 1000ms | 99th percentile Time-To-First-Byte |
| **Error Rate** | 1.0% | Maximum allowed error percentage |

## Test Modes

| Mode | Duration | Concurrency | Use Case |
|------|----------|-------------|----------|
| **load** | 30s | 5 | Quick PR gate validation |
| **soak** | 300s (5 min) | 10 | Nightly extended testing |

## How Thresholds Are Computed

1. **TTFB (Time-To-First-Byte)**: Measured from request start to first streaming marker
   received. Collected for all successful requests, then p50/p95/p99 percentiles computed.

2. **Error Rate**: `(failed_requests / total_requests) * 100`. A request is "failed" if
   it returns HTTP error, times out, or experiences connection issues.

3. **Cadence Check**: Beyond thresholds, the test validates that streaming markers arrive
   with proper cadence (not burst), catching buffering issues.

## Local Usage

### Prerequisites

- `finch` or `docker` CLI installed
- Python 3.14+ with `uv` package manager

### Quick Commands

```bash
# Run quick load gate (30s, for PR validation)
uv run python scripts/run_load_gate.py

# Run extended soak gate (5 min, for deep validation)
uv run python scripts/run_load_gate.py --mode soak

# With custom thresholds
uv run python scripts/run_load_gate.py \
  --ttfb-p95-ms 300 \
  --ttfb-p99-ms 800 \
  --error-rate-pct 0.5

# Skip compose up/down (when stack is already running)
uv run python scripts/run_load_gate.py --skip-compose

# Output JSON report
uv run python scripts/run_load_gate.py --json-output my-report.json

# Show all options
uv run python scripts/run_load_gate.py --help
```

### Manual Stack Management

If you want to manage the compose stack separately:

```bash
# Start the stack
finch compose -f docker-compose.streaming-perf.yml up -d --build

# Run gate (skip compose)
LOAD_GATE_SKIP_COMPOSE=1 uv run python scripts/run_load_gate.py

# View logs
finch compose -f docker-compose.streaming-perf.yml logs -f

# Stop the stack
finch compose -f docker-compose.streaming-perf.yml down -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOAD_GATE_MODE` | `load` | Test mode: `load` or `soak` |
| `LOAD_GATE_DURATION_S` | Mode-dependent | Override test duration |
| `LOAD_GATE_CONCURRENCY` | Mode-dependent | Override concurrency |
| `LOAD_GATE_TTFB_P95_MS` | `500` | p95 TTFB threshold |
| `LOAD_GATE_TTFB_P99_MS` | `1000` | p99 TTFB threshold |
| `LOAD_GATE_ERROR_RATE_PCT` | `1.0` | Max error rate % |
| `LOAD_GATE_STUB_URL` | `http://localhost:9200` | Stub server URL |
| `LOAD_GATE_SKIP_COMPOSE` | `0` | Set to `1` to skip compose |

## CI Integration

### GitHub Actions

The gate runs automatically via `.github/workflows/load-gate.yml`:

- **PR Gate**: Runs quick load test (30s) on PRs touching relevant paths
- **Nightly Soak**: Runs extended soak test (5 min) at 2 AM UTC daily
- **Manual Dispatch**: Configurable via GitHub Actions UI

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All thresholds passed |
| `1` | Threshold violation(s) detected |
| `2` | Infrastructure/setup error |

## JSON Report Schema

```json
{
  "summary": {
    "duration_s": 30.5,
    "total_requests": 150,
    "successful_requests": 149,
    "failed_requests": 1,
    "requests_per_second": 4.9
  },
  "ttfb": {
    "p50_ms": 12.5,
    "p95_ms": 45.2,
    "p99_ms": 89.3,
    "max_ms": 123.4
  },
  "error_rate_pct": 0.67,
  "thresholds": {
    "ttfb_p95_ms": 500.0,
    "ttfb_p99_ms": 1000.0,
    "error_rate_pct": 1.0
  },
  "violations": [],
  "passed": true
}
```

## Troubleshooting

### Stack Won't Start

```bash
# Check if ports are in use
lsof -i :9200  # Stub port
lsof -i :4020  # Gateway port

# Force cleanup
finch compose -f docker-compose.streaming-perf.yml down -v --remove-orphans
```

### High TTFB or Errors

1. Check stub server logs: `finch compose -f docker-compose.streaming-perf.yml logs streaming-stub`
2. Verify network: `curl http://localhost:9200/health`
3. Increase timeout: `--marker-count 5 --interval-ms 200`

### Deterministic Testing

The streaming stub emits markers at precise intervals (default: 50ms). Any buffering
in the gateway would cause:
- High TTFB (first byte delayed)
- Burst arrival (all markers at once at the end)

Both are caught by the gate's validation logic.

## Related Files

- [`scripts/run_load_gate.py`](../scripts/run_load_gate.py) - Main gate runner
- [`scripts/streaming_stub_server.py`](../scripts/streaming_stub_server.py) - Deterministic stub
- [`docker-compose.streaming-perf.yml`](../docker-compose.streaming-perf.yml) - Compose stack
- [`config/config.streaming-perf.yaml`](../config/config.streaming-perf.yaml) - Gateway config
- [`.github/workflows/load-gate.yml`](../.github/workflows/load-gate.yml) - CI workflow
