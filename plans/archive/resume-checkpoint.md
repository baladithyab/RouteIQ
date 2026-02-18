# Resume Checkpoint: TG Backlog & Workflow

**Date:** 2026-02-04
**HEAD:** *(updated on squash merge to main)*
**Context:** Resume point for ongoing Task Groups (TGs) and validation workflows.

## Recent Changes
- **TG3 Family (Tenant Controls):** Completed quotas, RBAC, and audit logging implementation.
- **Epic Plans:** Queued TG4–TG9 with detailed sub-TG breakdowns and acceptance criteria.
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
| **TG3** | **Tenant Controls (Quotas/RBAC/Audit)** | **✅ Done** | [`src/litellm_llmrouter/quota.py`](../src/litellm_llmrouter/quota.py) <br> [`src/litellm_llmrouter/rbac.py`](../src/litellm_llmrouter/rbac.py) <br> [`src/litellm_llmrouter/audit.py`](../src/litellm_llmrouter/audit.py) | Completed: Quota enforcement, RBAC policies, audit logging. |
| **TG9 (Legacy)** | **Docker E2E** | **✅ Done** | [`plans/tg10-6-e2e-verification-report.md`](tg10-6-e2e-verification-report.md) | Finalized gap closure report (Gate 10). |

---

## 4. Queued Epic Orchestrations (TG4–TG9)

These epics are queued for implementation. Each contains sub-TG breakdowns, acceptance criteria, and validation commands.

| Epic | Goal | Epic Plan | Priority |
|------|------|-----------|----------|
| **TG4** | **Observability** | [tg4-observability-epic.md](tg4-observability-epic.md) | High |
| **TG5** | **Security Policy** | [tg5-security-policy-epic.md](tg5-security-policy-epic.md) | High |
| **TG6** | **CI Quality Gates** | [tg6-ci-quality-gates-epic.md](tg6-ci-quality-gates-epic.md) | Medium |
| **TG7** | **Cloud-Native Deployment** | [tg7-cloud-native-deploy-epic.md](tg7-cloud-native-deploy-epic.md) | Medium |
| **TG8** | **Routing & MLOps** | [tg8-routing-mlops-epic.md](tg8-routing-mlops-epic.md) | Medium |
| **TG9** | **Extensibility** | [tg9-extensibility-epic.md](tg9-extensibility-epic.md) | Low |

### How to Resume an Epic

1. Read the epic plan file for the TG you want to work on.
2. Create the feature branch as documented (e.g., `git checkout -b tg4-observability`).
3. Work through sub-TGs sequentially, committing locally.
4. Run validation commands from the epic plan to verify completion.
5. Squash merge to `main` when the epic is complete.
6. Use `rr push` if local push is blocked, then `git pull`.

---

## 5. Next Priorities

1. **Start TG4 (Observability):** Implement routing decision visibility and OTel pipeline.
2. **Complete TG2.3 (Streaming):** Tune performance gate thresholds and ensure stability.
3. **Advance TG5 (Security):** Implement durable audit export and secret rotation.
