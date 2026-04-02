# HA CI Gate

> **TG6.2 Implementation**

This document describes the High Availability (HA) CI gate that validates leader
election and failover behavior in the RouteIQ gateway stack.

## Overview

The HA gate tests the following scenarios:

1. **Stack Boot** - Starts the HA stack (`docker-compose.ha.yml`)
2. **Leader Detection** - Queries PostgreSQL to identify the current leader
3. **Split-Brain Check** - Verifies exactly one leader exists (no split-brain)
4. **Leader Failure** - Kills the leader container to simulate a crash
5. **Failover Validation** - Asserts a new leader is elected within timeout
6. **Service Availability** - Verifies nginx LB remains available during/after failover

## How Leader Election Works

The gateway uses a database-backed lease lock mechanism:

- **Table**: `config_sync_leader` in PostgreSQL
- **Lease Duration**: Configurable via `LLMROUTER_CONFIG_SYNC_LEASE_SECONDS` (default: 30s)
- **Renewal Interval**: `LLMROUTER_CONFIG_SYNC_RENEW_INTERVAL_SECONDS` (default: 10s)
- **Lock Name**: `config_sync`

When a leader fails to renew its lease, the lock expires and another replica can
acquire leadership through an atomic INSERT/UPDATE operation.

## Running Locally

### Prerequisites

- Docker Compose (v2.0+)
- Python 3.12+ with `httpx` and `asyncpg` packages
- uv package manager (recommended)

### Quick Start

```bash
# Install dependencies
uv sync --frozen --extra dev
uv pip install httpx asyncpg

# Run the HA gate
uv run python scripts/run_ha_gate.py
```

### Command Line Options

```bash
uv run python scripts/run_ha_gate.py --help

Options:
  --compose-file FILE   Path to HA compose file (default: docker-compose.ha.yml)
  --timeout SECONDS     Max seconds to wait for failover (default: 90)
  --json-output FILE    Write JSON report to file
  --no-cleanup          Don't tear down stack after test (for debugging)
```

### Examples

```bash
# Quick local run
uv run python scripts/run_ha_gate.py

# With custom timeout and JSON output
uv run python scripts/run_ha_gate.py --timeout 120 --json-output ha-gate-report.json

# Debug mode (no cleanup)
uv run python scripts/run_ha_gate.py --no-cleanup
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HA_NGINX_URL` | `http://localhost:8080` | Nginx LB URL |
| `HA_GATEWAY_1_URL` | `http://localhost:4000` | Gateway 1 URL |
| `HA_GATEWAY_2_URL` | `http://localhost:4001` | Gateway 2 URL |
| `HA_POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `HA_POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `litellm` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `litellm_password` | PostgreSQL password |
| `LITELLM_MASTER_KEY` | (test key) | LiteLLM master key |

## CI Integration

The HA gate runs automatically via GitHub Actions:

- **PR Gate**: Runs on pull requests when HA-related files change
- **Nightly**: Runs at 3 AM UTC for extended validation
- **Manual**: Can be triggered manually with custom timeout

### Workflow File

See [`.github/workflows/ha-gate.yml`](../.github/workflows/ha-gate.yml)

### Triggers

The PR gate triggers on changes to:
- `src/**`
- `docker-compose.ha.yml`
- `config/nginx.conf`
- `docker/Dockerfile`
- `scripts/run_ha_gate.py`
- `src/litellm_llmrouter/leader_election.py`
- `src/litellm_llmrouter/config_sync.py`

## Test Report

The gate generates a JSON report with the following fields:

```json
{
  "passed": true,
  "timestamp": "2026-02-04T09:00:00.000000+00:00",
  "duration_seconds": 45.2,
  "startup_time_seconds": 25.1,
  "failover_time_seconds": 12.3,
  "initial_leader": "litellm-gateway-1-abc12345",
  "new_leader": "litellm-gateway-2-def67890",
  "split_brain_detected": false,
  "nginx_available_during_failover": true,
  "nginx_available_after_failover": true,
  "error": null
}
```

## Troubleshooting

### Stack won't start

1. Check Docker is running
2. Verify no conflicting containers: `docker ps -a | grep litellm`
3. Check port availability: 4000, 4001, 5432, 8080

### No leader detected

1. Check PostgreSQL is healthy: `docker logs litellm-postgres`
2. Verify `LLMROUTER_HA_MODE=leader_election` is set
3. Check gateway logs: `docker logs litellm-gateway-1`

### Failover timeout

1. Increase timeout: `--timeout 180`
2. Check lease settings - shorter leases mean faster failover
3. Verify the surviving gateway is healthy

### Debug mode

Run with `--no-cleanup` to inspect the stack after failure:

```bash
uv run python scripts/run_ha_gate.py --no-cleanup

# Inspect containers
docker logs litellm-gateway-1
docker logs litellm-gateway-2
docker exec litellm-postgres psql -U litellm -c "SELECT * FROM config_sync_leader;"

# Manual cleanup when done
docker compose -f docker-compose.ha.yml down -v
```

## Related

- [High Availability Setup](deployment.md)
- [Leader Election Implementation](../src/litellm_llmrouter/leader_election.py)
- [Config Sync](../src/litellm_llmrouter/config_sync.py)
