# Resume Checkpoint: TG Backlog & Workflow

**Date:** 2026-02-03
**HEAD:** `6ac6e240df210aa15958200a1a84ffbcdbb67242`
**Context:** Resume point for ongoing Task Groups (TGs) and validation workflows.

## Recent Changes
- **Docs:** Updated `AGENTS.md` with `rr` push workflow and post-push sync instructions.
- **Tooling:** Integrated `rr` (Road Runner) for remote git operations.
- **Streaming:** Added performance gate for TTFB and chunk cadence (`test_streaming_perf_gate.py`).
- **MCP:** Added legacy SSE transport and validator (`validate_mcp_sse.py`).

---

## 1. Key Validation Commands

Run these locally to verify core functionality before pushing.

| Component | Command | Description |
|-----------|---------|-------------|
| **MCP JSON-RPC** | `uv run python scripts/validate_mcp_jsonrpc.py` | Validates standard MCP JSON-RPC protocol. |
| **MCP SSE** | `uv run python scripts/validate_mcp_sse.py` | Validates MCP over Server-Sent Events (SSE). |
| **Streaming Perf** | `uv run pytest tests/integration/test_streaming_perf_gate.py` | Checks Time-To-First-Byte (TTFB) and chunk cadence. <br> *Requires:* `docker-compose -f docker-compose.streaming-perf.yml up -d` |
| **HA Failover** | `uv run pytest tests/integration/test_ha_leader_failover.py` | Verifies leader election and failover logic. |

---

## 2. Push Workflow (Mandatory)

Due to Code Defender restrictions, local `git push` may be blocked. Use the **Road Runner (rr)** workflow.

### Step 1: Push via rr
```bash
# Normal push
rr push

# Force push (use sparingly)
rr push-force
```

### Step 2: Sync Local Repo (CRITICAL)
After `rr push`, your local `origin/main` ref is updated, but your local branch is behind.

**After Normal Push:**
```bash
git pull
```

**After Force Push:**
```bash
# ⚠️ Warning: This resets your working tree. Ensure it is clean.
git fetch origin
git reset --hard origin/main
```

> See [`docs/rr-workflow.md`](../docs/rr-workflow.md) for full details.

---

## 3. Task Group (TG) Status

| TG | Goal | Status | Key Files | Next Acceptance Criteria |
|----|------|--------|-----------|--------------------------|
| **TG2.3** | **Streaming Performance** | **Pending** | [`tests/integration/test_streaming_perf_gate.py`](../tests/integration/test_streaming_perf_gate.py) <br> [`docker-compose.streaming-perf.yml`](../docker-compose.streaming-perf.yml) | Pass TTFB < 50ms and variance < 10% in CI environment. |
| **TG3** | **High Availability** | **Pending** | [`tests/integration/test_ha_leader_failover.py`](../tests/integration/test_ha_leader_failover.py) | Reliable leader election recovery under 5s during chaos testing. |
| **TG4** | **MCP SSE Transport** | **Validating** | [`src/litellm_llmrouter/mcp_sse_transport.py`](../src/litellm_llmrouter/mcp_sse_transport.py) <br> [`scripts/validate_mcp_sse.py`](../scripts/validate_mcp_sse.py) | Full pass of `validate_mcp_sse.py` with concurrent clients. |
| **TG5** | **Security Hardening** | **In Progress** | [`src/litellm_llmrouter/auth.py`](../src/litellm_llmrouter/auth.py) <br> [`src/litellm_llmrouter/url_security.py`](../src/litellm_llmrouter/url_security.py) | Audit log export to S3 verified (P1-04). |
| **TG9** | **Docker E2E** | **Done** | [`plans/tg10-6-e2e-verification-report.md`](tg10-6-e2e-verification-report.md) | Finalize gap closure report (Gate 10). |

---

## 4. Next Priorities

1. **Complete TG2.3 (Streaming):** Tune performance gate thresholds and ensure stability.
2. **Finalize TG4 (MCP SSE):** Ensure SSE transport is robust and matches JSON-RPC parity.
3. **Advance TG3 (HA):** Hardening of Redis-based leader election.
