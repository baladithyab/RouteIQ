# RouteIQ Architecture Review — February 2026

> **Date:** February 2026  
> **Status:** Final  
> **Scope:** Full architecture review — current state assessment, rearchitecture recommendation, migration roadmap  
> **Task Groups:** TG0 (Submodules), TG1 (Research), TG2 (Evaluation), TG3 (Rearchitecture), TG4 (Summary)

---

## Executive Summary

A four-phase architecture review of RouteIQ v0.2.0 — a 36,207-line Python AI gateway built on LiteLLM and LLMRouter — found the system **not production-ready** (goal alignment score: 3.05/5), with four critical blockers: fragile monkey-patching of LiteLLM internals, a hardcoded single-worker constraint, 124 environment variables (70 undocumented), and a middleware ordering defect that makes load-shed responses untraceable. Approximately 7,578 lines of MCP and A2A code are now redundant with upstream LiteLLM. The recommended path forward is **Option A: LiteLLM Plugin/Extension** — restructuring RouteIQ as a ~5,000-line package using LiteLLM's official `CustomRoutingStrategyBase` extension point, with NadirClaw centroid-based zero-config routing, a React admin UI, and four self-hosting tiers. This eliminates the monkey-patch, enables multi-worker scaling, reduces the codebase by 86%, and raises the goal alignment score to an estimated 4.45/5. The migration spans 18 weeks across 6 phases at approximately 28–32 person-weeks of effort.

---

## Current State Assessment

### Component Inventory

| Component | Role | Status | Lines of Code |
|-----------|------|--------|:-------------:|
| **LiteLLM** | OpenAI-compatible proxy for 100+ LLM providers | Upstream dependency (v1.81.3) | N/A (external) |
| **LLMRouter** | ML-based routing algorithms (academic research) | Upstream dependency | N/A (external) |
| **NadirClaw** | Centroid-based zero-config routing engine | Reference (added TG0) | ~2,000 (reference) |
| **RouteIQ** | Gateway integrating all three components | v0.2.0 — not production-ready | ~36,207 |

RouteIQ comprises 36 Python source files, 10 gateway files, 12 plugins, and 82+ test files (2,095 passing).

### Current Architecture

```
+------------------------------------------------------------------+
|                   RouteIQ Gateway (startup.py)                    |
|                   workers=1 (HARDCODED)                           |
|                                                                   |
|  +--------------------+  +------------------+  +---------------+  |
|  | MCP Gateway        |  | A2A Gateway      |  | Plugin System |  |
|  | 5 surfaces, 5958L  |  | 1620L (wrapper)  |  | 12 plugins    |  |
|  | (REDUNDANT)        |  | (REDUNDANT)      |  |               |  |
|  +--------------------+  +------------------+  +---------------+  |
|                                                                   |
|  +------------------------------------------------------------+  |
|  | routing_strategy_patch.py — MONKEY-PATCHES 3 Router methods |  |
|  | (class-level patches prevent multi-worker operation)         |  |
|  +------------------------------------------------------------+  |
|           |                    |                    |              |
|  +--------v-------+  +--------v-------+  +---------v---------+   |
|  | strategies.py   |  | strategy_      |  | router_decision_  |   |
|  | 18+ ML algos    |  | registry.py    |  | callback.py       |   |
|  | 1594 lines      |  | A/B testing    |  | OTel telemetry    |   |
|  +-----------------+  +----------------+  +-------------------+   |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |                  LiteLLM Proxy (upstream)                   |  |
|  |     100+ providers | Auth | Spend tracking | Config         |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

### Goal Alignment Scorecard

| Dimension | Score | Weight | Weighted | Verdict |
|-----------|:-----:|:------:|:--------:|---------|
| Cloud-Native | 3/5 | 20% | 0.60 | Artifacts present; architecture contradicts |
| Self-Hostable | 3/5 | 15% | 0.45 | Many compose variants; hard to actually run |
| Generalized AI Gateway | 4/5 | 15% | 0.60 | Strongest — inherited from LiteLLM |
| Intelligent Routing | 3/5 | 20% | 0.60 | Code exists; runtime evidence missing |
| Extensible / Customizable | 2/5 | 10% | 0.20 | Sound design; incomplete execution |
| Enterprise-Ready | 3/5 | 10% | 0.30 | Features implemented; untested at scale |
| MCP / A2A Support | 3/5 | 10% | 0.30 | 5 surfaces exist; protocol mismatch in POST |
| **Overall** | | **100%** | **3.05** | **NOT PRODUCTION-READY** |

### Critical Blockers

1. **CB-1 — Monkey-Patch Fragility:** RouteIQ patches 3 methods on LiteLLM's `Router` class at the class level. The `unpatch()` function only restores 1 of 3 methods. Async path blocks the event loop with synchronous ML inference. Any upstream LiteLLM update can silently break routing.

2. **CB-2 — Single-Worker Bottleneck:** `workers=1` is hardcoded because monkey-patches don't survive the `os.execvp()` used by multi-worker uvicorn. One Python process handles all requests — no CPU parallelism, no horizontal scaling per pod.

3. **CB-3 — Configuration Sprawl:** 124 environment variables across 20 prefixes, only 54 documented. No startup validation. Three competing configuration systems (env vars, YAML, LiteLLM config) with no precedence documentation.

4. **CB-4 — Middleware Ordering Defect:** `BackpressureMiddleware` wraps outermost, so 503 load-shed responses bypass all other middleware — no request ID, no CORS headers, no audit logging. During peak load, responses become untraceable.

---

## Recommended Architecture

### Option A: LiteLLM Plugin/Extension (Score: 4.45/5)

Replace the monkey-patch with LiteLLM's official `CustomRoutingStrategyBase` extension point and `Router.set_custom_routing_strategy()`. This uses `setattr()` on the Router *instance* (not class), eliminating the single-worker mandate entirely.

```
+------------------------------------------------------+
|              LiteLLM Proxy (upstream)                 |
|                                                       |
|  +----------+  +---------+  +----------+  +-------+  |
|  | Auth/RBAC|  | MCP GW  |  | A2A GW   |  | Admin |  |
|  | (native) |  | (native)|  | (native) |  | UI    |  |
|  +----------+  +---------+  +----------+  +-------+  |
|                                                       |
|  +---------------------------------------------------+|
|  |              litellm.Router                        ||
|  |                                                   ||
|  |  router.set_custom_routing_strategy(              ||
|  |      RouteIQRoutingStrategy()                     ||
|  |  )                                                ||
|  |                                                   ||
|  |  +-----------------------------------------------+||
|  |  |  RouteIQRoutingStrategy                       |||
|  |  |    extends CustomRoutingStrategyBase           |||
|  |  |                                               |||
|  |  |  - Centroid routing: zero-config, ~2.1ms      |||
|  |  |  - 18 ML strategies: KNN, MLP, SVM, etc.     |||
|  |  |  - A/B testing pipeline                       |||
|  |  |  - 5 routing profiles                         |||
|  |  |  - Routing telemetry via OTel                 |||
|  |  +-----------------------------------------------+||
|  +---------------------------------------------------+|
|                                                       |
|  +---------------------------------------------------+|
|  |  CustomLogger callbacks                           ||
|  |  - NadirClaw pre-classifier                       ||
|  |  - Policy engine (pre-call evaluation)            ||
|  |  - Observability (routing decision spans)         ||
|  +---------------------------------------------------+|
+------------------------------------------------------+
            |               |               |
       OpenAI API     Anthropic API    Bedrock API
```

### Key Benefits

- **Multi-worker operation:** Instance-level strategy binding survives `os.execvp()` — uvicorn with N workers
- **No monkey-patching:** Uses LiteLLM's official extension API (`CustomRoutingStrategyBase`)
- **86% code reduction:** ~36K lines → ~5K lines (router) + ~3K lines (UI)
- **Redundant code eliminated:** ~7,578 lines of MCP/A2A code deleted — LiteLLM native handles these
- **PyPI distributable:** `pip install routeiq-router` for any LiteLLM deployment

### Admin UI Summary

**Stack:** React 18 + TypeScript 5.5 + Vite 6 + TanStack Query + Tremor charts + Tailwind CSS

**Default deployment:** Static assets bundled in a `routeiq-ui` Python wheel (~5MB gzipped), served by FastAPI `StaticFiles` at `/ui/*` — single container, single port, zero CORS configuration.

| Page | Purpose | Priority |
|------|---------|:--------:|
| Dashboard | Operational metrics, model distribution, strategy performance | P0 |
| Routing Config | Strategy selector, A/B experiment manager, model candidates | P0 |
| Routing Explorer | Search and inspect individual routing decisions | P1 |
| Plugin Manager | Plugin lifecycle, health, and configuration | P1 |
| Config Editor | Monaco YAML editor with diff and hot-reload | P1 |
| Policy Editor | OPA-style rule editor with test interface | P2 |
| MLOps Pipeline | Training jobs, model artifacts, deployment | P2 |

**Backend:** ~43 API endpoints across all pages, with 8 P0 endpoints for MVP.

### NadirClaw Zero-Config Routing

Pre-trained centroids (~100KB) ship with the package, providing immediate intelligent routing from the first request — no training data required.

**Pipeline:** Prompt → Embed (all-MiniLM-L6-v2, ~2ms) → Cosine similarity to 8 category centroids → Model-tier affinity mapping → Selected deployment

**Performance:** ~2.1ms total routing overhead. **Accuracy:** 72–78% (vs. 87–91% for trained ML with 10K samples). Five routing profiles (auto, eco, premium, free, reasoning) selectable per-request via header, per-team, or per-key configuration.

---

## Self-Hosting Tiers

| Tier | Target | Infrastructure | Resources | Throughput | Complexity |
|:----:|--------|---------------|-----------|:----------:|:----------:|
| **1 — Minimal** | Dev / evaluation | Single container, SQLite, in-memory cache | ~256MB RAM, 1 CPU | ~20–50 req/s | ★☆☆☆☆ |
| **2 — Standard** | Small-medium production | Docker Compose, PostgreSQL, Redis, 2+ workers | ~2GB RAM, 2 CPU | ~100–200 req/s | ★★☆☆☆ |
| **3 — Production** | Kubernetes + Helm | 3–20 pods, HPA, PDB, ExternalSecrets, monitoring | Per-pod resources | ~500–2,000 req/s | ★★★☆☆ |
| **4 — Enterprise** | Multi-region HA | 5–50 pods/region, ArgoCD, mTLS, custom plugins, SLA tooling | Per-region resources | ~5,000–20,000 req/s | ★★★★★ |

**Tier 1 experience:** `docker compose up` with 3 environment variables — zero-config centroid routing active from first request, embedded admin UI at `/ui/`.

---

## Migration Roadmap

| Phase | Scope | Weeks | Key Deliverables |
|:-----:|-------|:-----:|-----------------|
| **0 — Foundation** | Plugin scaffold, CI/CD, config consolidation | 1–2 | `RouteIQRoutingStrategy(CustomRoutingStrategyBase)` wrapper; Pydantic Settings model (~30 validated env vars); multi-worker integration tests; package scaffolding |
| **1 — Core Routing** | ML strategies as LiteLLM plugin | 3–5 | Centroid routing with 8 pre-trained centroids; 5 routing profiles; 5-tier complexity classification; unified 6-stage routing pipeline; session persistence via Redis |
| **2 — NadirClaw** | Centroid routing integration | 6–8 | NadirClaw pre-classifier; agentic/reasoning detection; progressive centroid refinement; A/B testing centroid vs. trained ML |
| **3 — Admin UI** | React dashboard | 9–12 | 7 UI pages (Dashboard, Routing Config, Explorer, Plugin Manager, Config Editor, Policy Editor, MLOps); ~43 API endpoints; embedded deployment mode |
| **4 — Cloud-Native** | Self-hosting tiers, Helm charts | 13–15 | Delete ~7,578 lines redundant code; Redis-backed shared state; updated Helm chart for v1.0; Tier 1–4 deployment artifacts; ExternalSecrets integration |
| **5 — Migration & Cutover** | Data migration, rollback plan, GA release | 16–18 | v0.2.0 → v1.0 migration guide; PyPI release (`routeiq-router`, `routeiq-ui`); Docker images on GHCR; updated documentation; E2E test suite |

**Total effort:** ~28–32 person-weeks. Phases 1 and 2 can run in parallel with Phase 3 (routing vs. UI workstreams).

---

## Key Metrics

| Metric | Before (v0.2.0) | After (v1.0) |
|--------|:----------------:|:------------:|
| Goal alignment | 3.05 / 5 | 4.45 / 5 |
| Production readiness | Not ready | Production-ready |
| Time-to-first-routing | ~30 min config | < 5 min (zero-config NadirClaw) |
| Workers | 1 (hard limit) | N (horizontal scaling) |
| Codebase | ~36K lines | ~5K lines (router) + ~3K (UI) |
| Environment variables | 124 (70 undocumented) | ~30 (all documented) |
| Redundant code | ~7,578 lines | 0 |
| Routing latency overhead | ~5ms (KNN) | ~2.1ms (centroid) |
| Self-hosting experience | Complex (16 env vars min) | `docker compose up` (3 env vars) |
| Admin UI | None | 7 pages, embedded SPA |

---

## Documents Produced

| Document | Description | Task Group |
|----------|-------------|:----------:|
| `CRITICAL_ARCHITECTURE_ASSESSMENT.md` | Critical findings summary — blockers, competitive positioning, strategic direction | TG2 |
| `docs/architecture/tg2-architecture-evaluation.md` | Full architecture evaluation — goal alignment scorecard, 18 recommendations, 90-day plan | TG2 |
| `docs/architecture/tg3-alternative-patterns.md` | Architecture pattern analysis — 5 options scored, Option A recommended (4.45/5) | TG3 |
| `docs/architecture/tg3-rearchitecture-proposal.md` | Unified rearchitecture proposal — migration phases, effort estimates, risk assessment | TG3 |
| `docs/architecture/tg3-admin-ui-design.md` | Admin UI design specification — React/TypeScript/Vite, 4 deployment modes, 7 pages | TG3 |
| `docs/architecture/tg3-cloud-native-design.md` | Cloud-native design specification — 4 self-hosting tiers, 12-factor compliance, state externalization | TG3 |
| `docs/architecture/tg3-nadirclaw-integration.md` | NadirClaw integration design — centroid routing, profiles, 6-stage pipeline, session persistence | TG3 |
| `docs/architecture/ROUTEIQ-ARCHITECTURE-REVIEW-2026-02.md` | This executive summary | TG4 |

---

## Next Steps

### Immediate (Weeks 1–2)

1. **Validate LiteLLM `CustomRoutingStrategyBase` API** with latest upstream release — confirm instance-level `set_custom_routing_strategy()` supports async routing and multi-worker operation
2. **Set up plugin development environment** — package scaffolding for `routeiq-router` with CI/CD pipeline
3. **Prototype NadirClaw centroid routing as LiteLLM plugin** — prove the zero-config path works end-to-end with 8 pre-trained centroids

### Decision Points

1. **Confirm Option A (plugin) vs. Option B (fork)** — prototype results from weeks 1–2 should validate or invalidate the `CustomRoutingStrategyBase` approach
2. **Budget allocation for 18-week migration** — ~28–32 person-weeks across 6 phases; two-engineer team is the minimum viable staffing
3. **Admin UI scope: full (7 pages) vs. MVP (2 pages)** — Dashboard + Routing Config delivers 80% of the value; defer P1/P2 pages if resources are constrained
4. **Self-hosting tier prioritization** — Tier 1 (single container) and Tier 2 (Docker Compose) cover the majority of initial users; Tier 3/4 can follow post-GA
