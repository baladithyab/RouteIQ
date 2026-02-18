# RouteIQ Implementation Beads

> Tracked by `bd` CLI (v0.52.0). Run `bd list` for live status, `bd ready` for unblocked work.
> This file is a human-readable companion to the `.beads/` database.

## Dependency Graph

```
TG-IMPL-A (P0) ──blocks──▶ TG-IMPL-C (P1) ──blocks──▶ TG-IMPL-D (P2)
       │
       └──────blocks──▶ TG-IMPL-F (P2)

TG-IMPL-G (P1) ──blocks──▶ TG-IMPL-B (P1)

TG-IMPL-E (P2) ─── (no blockers)
```

## Active Beads

### TG-IMPL-A: P0 Critical Fixes `RouteIQ-oe6`
- **Status**: 🔴 Not Started
- **Priority**: P0 (Critical)
- **Dependencies**: None (ready to start)
- **Tasks**:
  - [ ] Fix middleware ordering (BackpressureMiddleware must wrap inside RequestID)
  - [ ] Fix `unpatch_litellm_router()` to restore all 3 methods (get_available_deployment, async_get_available_deployment, _common_checks_available_deployment)
  - [ ] Make ML strategies actually register at runtime (verify strategies appear in Router.routing_strategy_args)
  - [ ] Solve multi-worker scaling (move routing state to Redis or use CustomRoutingStrategyBase)

### TG-IMPL-B: Documentation Cleanup `RouteIQ-d4i`
- **Status**: 🔴 Not Started
- **Priority**: P1 (High)
- **Dependencies**: TG-IMPL-G (must reduce code before documenting it)
- **Tasks**:
  - [ ] Consolidate redundant doc files (remove placeholder stubs with `# TODO` content)
  - [ ] Remove outdated plans/ files that no longer apply
  - [ ] Update AGENTS.md to reflect actual architecture post-cleanup
  - [ ] Ensure docs/ reflects current state, not aspirational state
  - [ ] Update README.md quick start section

### TG-IMPL-C: LiteLLM Plugin Architecture Migration `RouteIQ-n5d`
- **Status**: 🔴 Not Started
- **Priority**: P1 (High)
- **Dependencies**: TG-IMPL-A (critical fixes must land first)
- **Tasks**:
  - [ ] Implement RouteIQ strategies using `CustomRoutingStrategyBase` interface
  - [ ] Remove `routing_strategy_patch.py` monkey-patch
  - [ ] Remove single-worker constraint from documentation and entrypoint
  - [ ] Register strategies via LiteLLM's `custom_routing_strategy_class` config
  - [ ] Update MLOps pipeline to work with new strategy interface
  - [ ] Verify multi-worker deployment works with new architecture

### TG-IMPL-D: NadirClaw Integration `RouteIQ-9m8`
- **Status**: 🔴 Not Started
- **Priority**: P2 (Medium)
- **Dependencies**: TG-IMPL-C (plugin arch needed for clean integration)
- **Tasks**:
  - [ ] Implement centroid-based routing strategy
  - [ ] Add routing profiles (cost-optimized, quality-optimized, balanced)
  - [ ] Implement prompt-complexity analysis for intelligent model selection
  - [ ] Add zero-config defaults for common use cases
  - [ ] Integration tests with real model endpoints

### TG-IMPL-E: Admin UI MVP `RouteIQ-a5p`
- **Status**: 🔴 Not Started
- **Priority**: P2 (Medium)
- **Dependencies**: None (ready to start)
- **Tasks**:
  - [ ] Scaffold React+Vite+TypeScript project in `ui/` directory
  - [ ] Dashboard page: routing stats, model health, request volume
  - [ ] Routing config page: strategy selection, A/B test weights
  - [ ] Model management page: view configured models, health status
  - [ ] Connect to RouteIQ admin API endpoints
  - [ ] Docker build integration (multi-stage with static serve)

### TG-IMPL-F: Cloud-Native Hardening `RouteIQ-y4c`
- **Status**: 🔴 Not Started
- **Priority**: P2 (Medium)
- **Dependencies**: TG-IMPL-A (multi-worker fix required first)
- **Tasks**:
  - [ ] Consolidate env vars (audit ROUTEIQ_* vs LITELLM_* vs LLMROUTER_*)
  - [ ] Externalize routing state to Redis (strategy weights, A/B config)
  - [ ] Update Helm charts for true stateless horizontal scaling
  - [ ] Improve health check semantics (readiness vs liveness clarity)
  - [ ] Add Kubernetes-native leader election (replace Redis-based)

### TG-IMPL-G: Codebase Reduction `RouteIQ-2qz`
- **Status**: ✅ Done
- **Priority**: P1 (High)
- **Dependencies**: None (ready to start)
- **Completed**: 2026-02-18
- **Tasks**:
  - [x] Phase 1: Removed 23 obsolete files (GATE reports, stub docs, CRITICAL_ARCHITECTURE_ASSESSMENT.md) — ~3,766 lines
  - [x] Phase 1: Archived 10 completed plan files to `plans/archive/`
  - [x] Phase 2: MCP/A2A redundancy evaluated — 3 modules (3,379 lines) identified as fully redundant (`mcp_jsonrpc.py`, `mcp_sse_transport.py`, `mcp_parity.py`); 2 modules flagged for future removal; 2 tracing modules kept (unique value)
  - [x] Phase 3: Version identity unified — Helm Chart appVersion, Dockerfile labels, entrypoint banners, `pyproject.toml` description all aligned to RouteIQ 0.2.0

## Execution Order (Recommended)

| Phase | Beads | Rationale |
|-------|-------|-----------|
| 1 (Now) | **TG-IMPL-A** + **TG-IMPL-G** | Unblocked P0/P1 work — fix critical bugs, reduce code |
| 2 | **TG-IMPL-C** + **TG-IMPL-B** + **TG-IMPL-E** | After A unblocks C; after G unblocks B; E is independent |
| 3 | **TG-IMPL-D** + **TG-IMPL-F** | After C unblocks D; after A unblocks F |

## bd Quick Reference

```bash
bd list                    # List all beads
bd ready                   # Show unblocked work
bd show RouteIQ-oe6        # Show bead details
bd update RouteIQ-oe6 --status in_progress  # Start working
bd close RouteIQ-oe6       # Mark complete
bd dep tree RouteIQ-9m8    # View dependency tree
```
