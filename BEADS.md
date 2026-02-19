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
- **Status**: ✅ Done
- **Priority**: P0 (Critical)
- **Dependencies**: None
- **Completed**: 2026-02-18
- **Tasks**:
  - [x] Fix middleware ordering (BackpressureMiddleware must wrap inside RequestID)
  - [x] Fix `unpatch_litellm_router()` to restore all 3 methods (get_available_deployment, async_get_available_deployment, _common_checks_available_deployment)
  - [x] Make ML strategies actually register at runtime (verify strategies appear in Router.routing_strategy_args)
  - [x] Solve multi-worker scaling (move routing state to Redis or use CustomRoutingStrategyBase)

### TG-IMPL-B: Documentation Cleanup `RouteIQ-d4i`
- **Status**: ✅ Done
- **Priority**: P1 (High)
- **Dependencies**: TG-IMPL-G (must reduce code before documenting it)
- **Completed**: 2026-02-18
- **Tasks**:
  - [x] Fixed AGENTS.md/CLAUDE.md drift — synchronized both files with actual codebase architecture
  - [x] Consolidated redundant docs — merged plugin-development-guide.md into plugins.md, deleted docs-consolidation-plan.md
  - [x] Updated project-state.md, parity-roadmap.md, and quickstart links to reflect current state
  - [x] Added cross-references between related docs (plugins, skills, security)
  - [x] Rewrote docs/index.md as comprehensive documentation hub — 39 docs across 7 categories with descriptions and status indicators
  - [x] Ensured docs/ reflects current state, not aspirational state

### TG-IMPL-C: LiteLLM Plugin Architecture Migration `RouteIQ-n5d`
- **Status**: ✅ Done
- **Priority**: P1 (High)
- **Dependencies**: TG-IMPL-A (critical fixes must land first)
- **Completed**: 2026-02-18
- **Tasks**:
  - [x] Implement `RouteIQRoutingStrategy` using `CustomRoutingStrategyBase` plugin API (`custom_routing_strategy.py`)
  - [x] Add `ROUTEIQ_USE_PLUGIN_STRATEGY` feature flag (default: `true`) with fallback to legacy monkey-patch
  - [x] Add `ROUTEIQ_WORKERS` env var for multi-worker uvicorn support
  - [x] Wire plugin strategy into `startup.py` (post-init install) and `gateway/app.py` (conditional monkey-patch skip)
  - [x] Update Docker entrypoints, AGENTS.md, `.env.example`, and env_validation.py
  - [x] Add 62 unit tests (`test_custom_routing_strategy.py`, `test_multi_worker.py`)
  - [x] Legacy monkey-patch preserved as fallback (`ROUTEIQ_USE_PLUGIN_STRATEGY=false`)

### TG-IMPL-D: NadirClaw Integration `RouteIQ-9m8`
- **Status**: ✅ Done
- **Priority**: P2 (Medium)
- **Dependencies**: TG-IMPL-C (plugin arch needed for clean integration)
- **Completed**: 2026-02-19
- **Tasks**:
  - [x] Implement centroid-based routing strategy (`centroid_routing.py` — CentroidRoutingStrategy, CentroidClassifier, AgenticDetector, ReasoningDetector, SessionCache, RoutingProfile)
  - [x] Add routing profiles (auto, eco, premium, free, reasoning)
  - [x] Implement prompt-complexity analysis via cosine similarity against centroid vectors (~2ms)
  - [x] Add zero-config defaults — works immediately with pre-trained centroids
  - [x] Integration in `custom_routing_strategy.py` — centroid as fallback in progressive enhancement chain
  - [x] Agentic detection (tool use patterns) and reasoning detection (math/logic markers)
  - [x] Session persistence for conversation affinity
  - [x] 76 new tests (60 centroid + 16 integration)

### TG-IMPL-E: Admin UI MVP `RouteIQ-a5p`
- **Status**: ✅ Done
- **Priority**: P2 (Medium)
- **Dependencies**: None
- **Completed**: 2026-02-18
- **Tasks**:
  - [x] Scaffold React+Vite+TypeScript project in `ui/` directory
  - [x] Dashboard page: routing stats, model health, request volume
  - [x] Routing config page: strategy selection, A/B test weights
  - [x] Model management page: view configured models, health status
  - [x] Connect to RouteIQ admin API endpoints
  - [x] Docker build integration (multi-stage with static serve)

### TG-IMPL-F: Cloud-Native Hardening `RouteIQ-y4c`
- **Status**: ✅ Done
- **Priority**: P2 (Medium)
- **Dependencies**: TG-IMPL-A (multi-worker fix required first)
- **Completed**: 2026-02-19
- **Tasks**:
  - [x] Consolidate env vars (audit ROUTEIQ_* vs LITELLM_* vs LLMROUTER_*) — added comprehensive `.env.example` with 150+ documented variables
  - [x] Update Helm charts for plugin strategy, centroid routing, admin UI, and multi-worker support
  - [x] Update docker-compose files for plugin strategy and multi-worker configuration
  - [x] Improve health check semantics (readiness vs liveness clarity)
  - [x] Kubernetes-native configuration for stateless horizontal scaling

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
| 1 (Done) | **TG-IMPL-G** ✅ + **TG-IMPL-B** ✅ | Codebase reduction and documentation cleanup complete |
| 2 (Done) | **TG-IMPL-A** ✅ + **TG-IMPL-C** ✅ | P0 critical fixes + plugin architecture migration complete |
| 3 (Done) | **TG-IMPL-E** ✅ + **TG-IMPL-D** ✅ + **TG-IMPL-F** ✅ | Admin UI, centroid routing, and cloud-native hardening all complete |

## Completion Summary

> **All implementation task groups (TG-IMPL-A through TG-IMPL-G) are now complete.**
>
> - **Date**: 2026-02-19
> - **Total beads**: 7 / 7 done
> - **Key deliverables**:
>   - P0 critical fixes (middleware, unpatch, strategy registration, multi-worker)
>   - Plugin architecture migration (CustomRoutingStrategyBase, feature flags)
>   - NadirClaw centroid-based routing (~2ms zero-config routing)
>   - Admin UI MVP (React+Vite+TypeScript dashboard)
>   - Cloud-native hardening (env consolidation, Helm updates, docker-compose updates)
>   - Documentation cleanup and codebase reduction
>
> RouteIQ v0.2.0 implementation phase is complete. Next: validation and release.

## bd Quick Reference

```bash
bd list                    # List all beads
bd ready                   # Show unblocked work
bd show RouteIQ-oe6        # Show bead details
bd update RouteIQ-oe6 --status in_progress  # Start working
bd close RouteIQ-oe6       # Mark complete
bd dep tree RouteIQ-9m8    # View dependency tree
```
