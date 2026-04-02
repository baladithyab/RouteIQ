# TG3: RouteIQ v1.0 Rearchitecture Proposal

> **Version**: 1.0
> **Date**: 2026-02-18
> **Status**: Proposal — awaiting CTO/VP Engineering approval
> **Scope**: Unified rearchitecture plan synthesizing TG3 subtask findings

---

## Executive Summary

RouteIQ v0.2.0 is a 36,207-line Python monolith that extends LiteLLM via class-level
monkey-patching of three Router methods. This design forces a single-worker constraint
(`workers=1`), duplicates ~7,578 lines of MCP/A2A functionality that LiteLLM already
provides, exposes 124 environment variables (70 undocumented), and has no admin UI.
An independent assessment scores it 3.05/5 for goal alignment — **not production-ready**.

**The solution**: Restructure RouteIQ as a ~5,000-line `pip install routeiq-router`
package that integrates with LiteLLM via its official
[`CustomRoutingStrategyBase`](../../reference/litellm/litellm/types/router.py:671)
extension point. This eliminates the monkey-patch, enables multi-worker operation,
sheds redundant code, and adds zero-config intelligent routing via NadirClaw's
centroid-based approach — all while preserving RouteIQ's genuine differentiators:
18 ML routing strategies, A/B testing, policy engine, and resilience middleware.

**Key outcomes:**

| Metric | v0.2.0 (Current) | v1.0 (Target) |
|--------|:-----------------:|:-------------:|
| Goal alignment score | 3.05/5 | 4.5+/5 |
| Production readiness | NOT READY | READY |
| Time to first intelligent routing | Days–weeks (requires training) | 0 (centroid zero-config) |
| Uvicorn workers | 1 (hardcoded) | N (multi-worker) |
| Codebase | ~36,207 lines | ~5,000 lines (router) + ~3,000 (UI) |
| Environment variables | 124 (70 undocumented) | ~30 (all documented) |
| Self-hosting experience | Complex | `docker compose up` |
| Admin UI | None | React + TypeScript SPA |

**Timeline**: 18 weeks across 6 phases. **Effort**: ~32 person-weeks total.

---

## 1. Recommended Architecture

### 1.1 Architecture Decision: LiteLLM Plugin/Extension

Five architecture options were evaluated in the
[Alternative Patterns analysis](tg3-alternative-patterns.md):

| Option | Score | Effort | Key Tradeoff |
|--------|:-----:|:------:|-------------|
| **A: LiteLLM Plugin** ⭐ | **4.45/5** | 6–8 pw | Simplest path; coupled to LiteLLM API |
| B: Sidecar | 3.35/5 | 8–10 pw | Clean separation; extra network hop |
| C: Microservices | 2.10/5 | 16–24 pw | Best scaling; crushing complexity |
| D: Fork & Own | 2.95/5 | 4–6 pw + ongoing | Full control; 50–100 upstream commits/week to track |
| E: Hybrid Plugin + External | 3.95/5 | 10–14 pw | Scaling path; higher initial cost |

**Option A wins** because it delivers the most value with the least effort and ongoing
cost. The critical discovery is that LiteLLM ships
[`CustomRoutingStrategyBase`](../../reference/litellm/litellm/types/router.py:671) and
[`Router.set_custom_routing_strategy()`](../../reference/litellm/litellm/router.py:8782)
as its official extension point. These use `setattr()` on the **instance** (not class),
eliminating the single-worker mandate entirely. RouteIQ's current monkey-patch at
[`routing_strategy_patch.py:476-488`](../../src/litellm_llmrouter/routing_strategy_patch.py:476)
is replaced by a single `router.set_custom_routing_strategy()` call.

**Evolution path**: When scaling demands exceed in-process capacity (e.g., KNN embedding
models consuming excessive memory), evolve to Option E by extracting ML inference into
a standalone service. The `CustomRoutingStrategyBase` interface stays identical; only the
internal implementation changes from local computation to an HTTP call.

```
                    +----------------------------------------------+
                    |            LiteLLM Proxy (upstream)           |
                    |                                              |
                    |  +----------+  +--------+  +----------+     |
                    |  | Auth/RBAC|  | MCP GW |  | A2A GW   |     |
                    |  +----------+  +--------+  +----------+     |
                    |                                              |
                    |  +------------------------------------------+|
                    |  |             litellm.Router                ||
                    |  |                                          ||
                    |  |  router.set_custom_routing_strategy(     ||
                    |  |      RouteIQRoutingStrategy()            ||
                    |  |  )                                      ||
                    |  |                                          ||
                    |  |  +------------------------------------+ ||
                    |  |  | RouteIQRoutingStrategy              | ||
                    |  |  |   CustomRoutingStrategyBase         | ||
                    |  |  |                                    | ||
                    |  |  |  - Centroid routing (zero-config)  | ||
                    |  |  |  - 18 ML strategies (KNN,SVM,etc) | ||
                    |  |  |  - A/B testing pipeline            | ||
                    |  |  |  - Routing profiles                | ||
                    |  |  |  - Routing telemetry               | ||
                    |  |  +------------------------------------+ ||
                    |  +------------------------------------------+|
                    |                                              |
                    |  +------------------------------------------+|
                    |  | CustomLogger callbacks                   ||
                    |  |  - NadirClaw pre-classifier              ||
                    |  |  - Policy engine (pre-call)              ||
                    |  |  - Observability (decision spans)        ||
                    |  +------------------------------------------+|
                    +----------------------------------------------+
                                        |
                        +---------------+---------------+
                        |               |               |
                   OpenAI API     Anthropic API    Bedrock API
```

### 1.2 What Changes

| Concern | v0.2.0 | v1.0 |
|---------|--------|------|
| **Routing integration** | Monkey-patch 3 Router methods (class-level) | `CustomRoutingStrategyBase` (instance-level) |
| **Worker model** | `workers=1` hardcoded in [`startup.py`](../../src/litellm_llmrouter/startup.py) | Multi-worker uvicorn (N workers) |
| **MCP gateway** | Custom 5,958-line reimplementation | LiteLLM native (deleted) |
| **A2A gateway** | Custom 1,620-line reimplementation | LiteLLM native (deleted) |
| **Routing cold-start** | Requires MLOps training pipeline | Centroid zero-config from day 0 |
| **Admin UI** | None | React + TypeScript + Vite SPA |
| **Configuration** | 124 env vars, mixed file + env | ~30 core vars, Pydantic Settings validation |
| **Singletons** | 16 module-level singletons | 8 kept (per-worker), 1 externalized, 7 deleted |
| **Entry point** | Custom [`startup.py`](../../src/litellm_llmrouter/startup.py) with in-process uvicorn | `pip install routeiq-router` + standard LiteLLM startup |
| **Distribution** | Monolithic Docker image | PyPI package + Docker image |

### 1.3 What Gets Deleted

~7,578 lines of code redundant with upstream LiteLLM are removed:

| Module | Lines | Reason |
|--------|------:|--------|
| [`mcp_gateway.py`](../../src/litellm_llmrouter/mcp_gateway.py) | 1,267 | LiteLLM native MCP |
| [`mcp_jsonrpc.py`](../../src/litellm_llmrouter/mcp_jsonrpc.py) | 1,200 | LiteLLM native MCP |
| [`mcp_sse_transport.py`](../../src/litellm_llmrouter/mcp_sse_transport.py) | 1,100 | LiteLLM native MCP |
| [`mcp_parity.py`](../../src/litellm_llmrouter/mcp_parity.py) | 800 | LiteLLM native MCP |
| [`mcp_tracing.py`](../../src/litellm_llmrouter/mcp_tracing.py) | 591 | LiteLLM native |
| [`a2a_gateway.py`](../../src/litellm_llmrouter/a2a_gateway.py) | 1,200 | LiteLLM native A2A |
| [`a2a_tracing.py`](../../src/litellm_llmrouter/a2a_tracing.py) | 420 | LiteLLM native |
| [`routing_strategy_patch.py`](../../src/litellm_llmrouter/routing_strategy_patch.py) | 546 | Replaced by `CustomRoutingStrategyBase` |
| [`startup.py`](../../src/litellm_llmrouter/startup.py) | ~300 | LiteLLM standard startup |
| [`leader_election.py`](../../src/litellm_llmrouter/leader_election.py) | ~617 | LiteLLM native HA |
| [`quota.py`](../../src/litellm_llmrouter/quota.py) | ~971 | LiteLLM native quota |
| [`http_client_pool.py`](../../src/litellm_llmrouter/http_client_pool.py) | ~163 | LiteLLM manages connections |

Additionally, 7 of 16 singletons are deleted entirely, and the remaining 8 become
per-worker stateless instances (details in
[Cloud-Native Design §State Externalization](tg3-cloud-native-design.md#state-externalization)).

### 1.4 What Gets Built

| Component | Lines (est.) | Description |
|-----------|:------------:|-------------|
| `RouteIQRoutingStrategy` | ~500 | `CustomRoutingStrategyBase` wrapper for all strategies |
| `CentroidRoutingStrategy` | ~400 | Zero-config centroid-based routing |
| `NadirClawPreClassifier` | ~300 | Prompt classification via `CustomLogger` |
| `RoutingProfile` system | ~300 | 5 built-in profiles + custom profile support |
| `TierClassifier` | ~250 | Heuristic + centroid 5-tier complexity classification |
| `NadirClawPipeline` | ~300 | 6-stage unified routing pipeline orchestrator |
| Session persistence extensions | ~200 | Model-level affinity in `ConversationAffinityTracker` |
| Pydantic Settings model | ~150 | ~30 validated env vars replacing 124 |
| Redis state adapters | ~300 | A/B counters, circuit breakers, session state |
| Centroid data files | ~100KB | 8 pre-trained centroids + affinity maps |
| Admin UI (React + TypeScript) | ~3,000 | 7 pages, embedded in gateway |
| Admin API endpoints | ~800 | ~20 new P0 endpoints for UI |
| **Total** | **~6,500** | Core routing + UI |

---

## 2. Admin UI Design Summary

Full design in [Admin UI Architecture](tg3-admin-ui-design.md).

### 2.1 Technology: React + TypeScript + Vite

**Stack**: React 18, TypeScript 5.5, Vite 6, TanStack Query 5, Tremor charts, Tailwind
CSS, Monaco Editor for YAML/policy editing. **No runtime dependencies** — production
artifact is static HTML/CSS/JS served by FastAPI `StaticFiles`.

### 2.2 Default Deployment: Embedded in Gateway

The `routeiq-ui` Python wheel bundles pre-built static assets (~5MB gzipped). On
startup, FastAPI mounts them at `/ui/*`. Single container, single port, CORS-free.

Four deployment modes supported:

| Mode | Use Case | Complexity |
|------|----------|:----------:|
| **Embedded** (default) | Self-hosted standalone | ★☆☆☆☆ |
| Docker Sidecar | K8s / Helm deployments | ★★☆☆☆ |
| Standalone SPA | Managed / SaaS offering | ★★★☆☆ |
| LiteLLM Extension | RouteIQ as LiteLLM plugin | ★★☆☆☆ |

### 2.3 MVP Feature Set (Phase 1)

- **Dashboard**: Requests/sec, latency percentiles, cost/hour, model distribution,
  strategy performance via SSE real-time metrics stream
- **Routing Configuration**: Active strategy selector, A/B experiment manager,
  model candidate list with health status
- **Auth**: Admin API key validation + localStorage persistence; RBAC-aware tab visibility

**Backend**: 8 new API endpoints (strategy CRUD, experiment CRUD, auth, metrics).

### 2.4 Full Feature Set (Phase 3)

All 7 pages operational across ~43 API endpoints:

| Page | Purpose | Priority |
|------|---------|:--------:|
| Dashboard | Operational metrics at a glance | P0 |
| Routing Config | Strategy + A/B testing management | P0 |
| Routing Explorer | Search/inspect routing decisions | P1 |
| Plugin Manager | Plugin lifecycle + health + config | P1 |
| Config Editor | Monaco YAML editor + diff + hot-reload | P1 |
| Policy Editor | OPA-style rule editor + tester | P2 |
| MLOps Pipeline | Training jobs + model artifacts + deploy | P2 |

---

## 3. Deployment Architecture

Full design in [Cloud-Native Design](tg3-cloud-native-design.md).

### 3.1 Self-Hosting Tiers

| Tier | Target | Containers | Throughput | Complexity |
|------|--------|:----------:|:----------:|:----------:|
| **1: Minimal** | Dev / eval | 1 | ~20–50 req/s | ★☆☆☆☆ |
| **2: Standard** | Small-med prod | 4–6 | ~100–200 req/s | ★★☆☆☆ |
| **3: Production** | K8s + Helm | 3–20 pods | ~500–2,000 req/s | ★★★☆☆ |
| **4: Enterprise** | Multi-region HA | 5–50 pods/region | ~5,000–20,000 req/s | ★★★★★ |

### 3.2 Tier 1: docker compose up (the 5-minute experience)

```yaml
services:
  routeiq:
    image: ghcr.io/routeiq/routeiq-gateway:latest
    ports: ["4000:4000"]
    volumes:
      - ./config:/app/config:ro
    environment:
      - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
    deploy:
      resources:
        limits: { memory: 2G, cpus: "1.0" }
```

Single container, no PostgreSQL, no Redis, in-memory cache, SQLite or no DB.
Embedded admin UI at `/ui/`. Zero-config centroid routing active from first request.

### 3.3 Tier 3: Production Kubernetes

- **HPA**: 3–20 replicas scaling on CPU (70%), request rate, and p95 latency
- **PDB**: `minAvailable: 2` for zero-downtime deployments
- **External backing services**: PostgreSQL (RDS/CloudSQL), Redis (ElastiCache), S3/GCS
- **ExternalSecrets**: Vault/AWS SM/GCP SM integration for secret rotation
- **Startup probe**: 2-min timeout for ML model loading (KNN sentence-transformers)
- **Network policies**: Restrict egress to backing services + LLM providers

Deploy: `helm install routeiq-gateway ./deploy/charts/routeiq-gateway -f values-production.yaml`

### 3.4 State Externalization Summary

| State | Current | v1.0 Target | Consistency |
|-------|---------|-------------|:-----------:|
| Active strategy | In-memory singleton | Redis key | Eventual (<1s) |
| A/B traffic counters | In-memory dict | Redis HINCRBY | **Strong** (atomic) |
| Circuit breaker state | In-memory dict | Redis hash + TTL | Eventual (<1s) |
| Session affinity | In-memory LRU | Redis hash + TTL | Eventual (<1s) |
| Policy rules | YAML file | ConfigMap / Redis | Reload interval |
| ML model artifacts | Loaded into RAM | S3/GCS → per-worker RAM | Sync interval |
| Plugin registry | In-memory list | Per-worker (config-driven) | Per-worker |
| Semantic cache | In-memory LRU | L1 memory + L2 Redis | L1 per-worker, L2 shared |

**12-Factor compliance**: Two critical blockers resolved:
- **Factor 6 (Processes)**: `workers=1` → multi-worker via `CustomRoutingStrategyBase`
- **Factor 8 (Concurrency)**: Single-process → horizontal scaling via HPA

---

## 4. NadirClaw Integration Plan

Full design in [NadirClaw Integration Strategy](tg3-nadirclaw-integration.md).

### 4.1 Zero-Config Routing (centroid-based)

Pre-trained centroids ship with the `routeiq-router` package (~100KB), providing
immediate prompt-to-model routing via embedding similarity:

```
User Prompt → Embed (all-MiniLM-L6-v2) → Cosine sim to 8 centroids → Model affinity → Selected Model
```

| Category | Description | Typical Best Tier |
|----------|-------------|:-----------------:|
| `code_generation` | Code writing, debugging, refactoring | premium |
| `creative_writing` | Stories, poetry, marketing copy | premium |
| `analysis` | Data analysis, comparison, evaluation | mid |
| `summarization` | Condensing documents, TL;DR | budget |
| `translation` | Language conversion, localization | mid |
| `conversation` | Casual chat, simple Q&A | budget |
| `reasoning` | Logic, math, multi-step analysis | premium |
| `tool_use` | Function calling, API interaction, agentic | premium |

**Performance**: ~2.1ms total routing latency (2ms embed + 0.1ms cosine similarity).
**Accuracy**: 72–78% (vs 87–91% for trained MLP with 10K samples), but available
from day zero with no training data required.

### 4.2 Routing Profiles

Five built-in profiles selectable via header, metadata, team/key config, or global default:

| Profile | Strategy | Cost Behavior | Quality Floor |
|---------|----------|:------------:|:-------------:|
| `auto` (default) | Centroid or trained ML | No limit | None |
| `eco` | Cost-aware | Aggressive cap | 0.7 |
| `premium` | Top-tier always | No limit | 0.9 |
| `free` | Free/open-source only | $0 | None |
| `reasoning` | Reasoning-capable models | No limit | 0.85 |

Selection priority: `X-RouteIQ-Profile` header → request metadata → per-team →
per-key → `ROUTEIQ_DEFAULT_PROFILE` env var.

### 4.3 Unified Routing Pipeline

Six stages, designed to short-circuit early for performance:

```
Stage 1: Pre-Classification     Structural + content heuristics (<0.5ms)
    |                           Detects agentic, reasoning, code patterns
    v
Stage 2: Profile Resolution     Resolve from headers/metadata/team/key/env
    |
    v
Stage 3: Session Check          Lookup ConversationAffinityTracker
    |                           If found → short-circuit to Stage 6
    v
Stage 4: Tier Classification    5-tier complexity: trivial → expert (<2ms)
    |                           Heuristic fast path + centroid fallback
    v
Stage 5: Strategy Routing       Centroid, KNN, MLP, or hybrid via A/B
    |                           Delegates to RoutingPipeline
    v
Stage 6: Cost/Quality Filter    Apply profile constraints + tier mapping
    |                           Circuit breaker check, model selection
    v
Selected Deployment             Record session affinity
```

### 4.4 Progressive Enhancement Path

```
Day 0               Week 1              Month 1             Month 3+
  |                   |                   |                   |
  v                   v                   v                   v
Centroid           Centroid +          Centroid +           Trained ML
routing            user-defined        auto-refined         (KNN/MLP)
only               affinity maps       centroids from       + centroid
(ships free)       (config.yaml)       production traces    pre-filter
```

A/B testing enables safe transitions. Example: run centroid (90%) vs trained KNN (10%),
compare quality metrics, ramp up when KNN proves superior.

---

## 5. Migration Path

### 5.1 Phase 0: Foundation (weeks 1–2)

Eliminate the monkey-patch and establish the new integration pattern.

**Deliverables:**
- `RouteIQRoutingStrategy(CustomRoutingStrategyBase)` wrapping existing strategy logic
- `RouteIQPolicyCallback(CustomLogger)` for policy engine hooks
- Pydantic Settings model validating ~30 core env vars at startup
- Multi-worker validation with `workers=4` integration tests
- Package scaffolding for `routeiq-router` v0.1.0

**Key files:**
- `src/routeiq/strategy.py` (new — `CustomRoutingStrategyBase` implementation)
- `src/routeiq/settings.py` (new — Pydantic Settings)
- `src/routeiq/callbacks.py` (new — `CustomLogger` hooks)

**Exit criteria:** `uv run pytest tests/unit/ -x` passes with `workers=4` uvicorn.

### 5.2 Phase 1: Core Routing (weeks 3–5)

Implement NadirClaw centroid routing and the profile/tier system.

**Deliverables:**
- `CentroidRoutingStrategy` with 8 pre-trained centroids
- 5 routing profiles (auto, eco, premium, free, reasoning)
- Multi-tier complexity classification (5 tiers: trivial → expert)
- Session persistence via `ModelAffinityRecord` + Redis backend
- Unified 6-stage routing pipeline
- `NadirClawPreClassifier` for agentic/reasoning detection
- Unit tests for all new components

**Exit criteria:** Zero-config routing operational; centroid accuracy ≥ 70% on benchmark.

### 5.3 Phase 2: Admin UI MVP (weeks 6–8)

Deliver the first two UI pages with embedded deployment.

**Deliverables:**
- React + Vite project scaffolding with Tailwind + Tremor
- Auth flow (admin API key + localStorage)
- Dashboard page (metric cards, model distribution, strategy performance)
- Routing Configuration page (strategy selector, A/B experiment manager)
- 8 new backend API endpoints
- Embedded deployment mode (FastAPI `StaticFiles` mount)
- SSE metrics stream for real-time dashboard

**Exit criteria:** Dashboard loads at `/ui/` with live metrics; strategy changes via UI.

### 5.4 Phase 3: Production Hardening (weeks 9–12)

State externalization, code shed, and expanded UI.

**Deliverables:**
- Delete MCP gateway modules (~5,958 lines)
- Delete A2A gateway modules (~1,620 lines)
- Delete [`routing_strategy_patch.py`](../../src/litellm_llmrouter/routing_strategy_patch.py) (546 lines)
- Redis-backed state for A/B counters, circuit breakers, session affinity
- Updated Helm chart for v1.0 (values, deployment, HPA, startup probes)
- Tier 1 + Tier 2 docker-compose files
- Admin UI: Routing Explorer + Plugin Manager + Config Editor pages
- A/B testing: centroid vs trained ML strategies

**Exit criteria:** Multi-replica deployment with shared state via Redis; code deleted.

### 5.5 Phase 4: Advanced (weeks 13–16)

MLOps integration, advanced UI pages, LiteLLM dashboard extension.

**Deliverables:**
- Agentic detection pipeline (structural + content + centroid signals)
- MLOps pipeline integration (training job API, model artifact management)
- Admin UI: MLOps Pipeline + Policy Editor pages
- LiteLLM dashboard extension mode (iframe/sidebar integration)
- ExternalSecrets integration (Vault, AWS SM, GCP SM)
- Custom metrics HPA (request rate, p95 latency)
- Progressive centroid refinement from production traces

**Exit criteria:** Full admin UI operational; MLOps pipeline manageable via UI.

### 5.6 Phase 5: Polish & Release (weeks 17–18)

Documentation, migration guide, and public release.

**Deliverables:**
- v0.2.0 → v1.0 migration guide (config mapping, env var migration)
- PyPI release: `routeiq-router` v1.0.0
- PyPI release: `routeiq-ui` v1.0.0
- Updated Docker images on GHCR
- Updated Helm chart on artifact registry
- Updated deployment documentation (all 4 tiers)
- E2E tests (Playwright for UI, integration for API)
- Enterprise deployment guide (multi-region, mTLS, ArgoCD)

**Exit criteria:** `pip install routeiq-router` works; Docker images published;
docs complete.

---

## 6. Effort Estimates

| Phase | Deliverables | Person-Weeks | Dependencies |
|:-----:|-------------|:------------:|:------------:|
| **0** | `CustomRoutingStrategyBase` wrapper, env var consolidation, multi-worker | 3 | None |
| **1** | Centroid routing, profiles, tiers, sessions, pipeline | 5 | Phase 0 |
| **2** | Admin UI MVP (Dashboard + Routing Config) | 3 | Phase 0 |
| **3** | Code shed, Redis state, Helm, UI Explorer/Plugins/Config | 7 | Phases 0–1 |
| **4** | Agentic detection, MLOps, Policy Editor, enterprise Helm | 7 | Phases 1–3 |
| **5** | Docs, migration guide, PyPI/Docker release | 3 | Phases 0–4 |
| | | **~28–32 pw total** | |

Phases 1 and 2 can run in parallel (routing vs UI), as can components within Phases 3
and 4. With two engineers, calendar time is approximately 18 weeks.

```
Week:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18
       |--Phase 0--|
                   |----Phase 1----|
                   |--Phase 2--|
                               |-------Phase 3---------|
                                                 |-------Phase 4---------|
                                                                   |Phase 5|
```

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| **LiteLLM `CustomRoutingStrategyBase` API changes** | Medium | High | Pin LiteLLM version; abstract behind our own interface; track upstream releases weekly |
| **Sentence-transformers memory in multi-worker** | High | Medium | Memory-mapped model loading; model quantization (ONNX); reduce workers for KNN strategy |
| **Centroid accuracy insufficient for production** | Low | Medium | A/B test centroids vs random routing; progressive enhancement to trained ML; accuracy floor of 70% acceptable as baseline |
| **Redis as single point of failure** | Medium | High | Graceful degradation to in-memory state in Tier 1; Redis Sentinel/Cluster for Tier 3+ |
| **Pickle model loading security** | Low | High | JSON centroid format eliminates pickle for new models; existing `LLMROUTER_ENFORCE_SIGNED_MODELS` for legacy models |

### 7.2 Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| **Admin UI scope creep** | High | Medium | Strict P0/P1/P2 prioritization; MVP is 2 pages only; defer P2 pages to post-v1.0 |
| **LiteLLM upstream breaking changes** | Medium | High | Pin litellm version in `pyproject.toml`; test against upstream nightly in CI; maintain compatibility shim |
| **ML model training pipeline drift** | Low | Low | Centroid routing is the primary v1.0 strategy; MLOps integration is Phase 4 |
| **Two-engineer team velocity** | Medium | Medium | Phases 1/2 parallelizable; Phase 5 is buffer; cut P2 UI pages if behind |

### 7.3 Mitigations Summary

1. **Version pinning + abstraction layer**: RouteIQ wraps `CustomRoutingStrategyBase`
   behind its own `RouteIQStrategy` abstract class. If LiteLLM changes the interface,
   only the adapter layer changes.

2. **Memory-mapped models**: Use `torch.load(mmap=True)` or ONNX runtime for
   sentence-transformers to share physical pages across workers via OS page cache.

3. **Graceful degradation**: Every external dependency (Redis, S3, PostgreSQL) has an
   in-memory fallback. Tier 1 runs with zero external services.

4. **Weekly upstream tracking**: CI runs RouteIQ tests against `litellm@main` nightly.
   Breaking changes are detected within 24 hours.

---

## 8. Success Criteria

| Criterion | Current (v0.2.0) | Target (v1.0) | Measurement |
|-----------|:----------------:|:-------------:|-------------|
| **Goal alignment** | 3.05/5 | 4.5+/5 | Re-run TG2 evaluation rubric |
| **Production readiness** | NOT READY | READY | Pass all TG quality gates |
| **Time to first route** | ∞ (requires training) | 0s (centroid) | New deployment → first intelligent route |
| **Workers** | 1 | N (default 4) | `uvicorn --workers N` passes integration tests |
| **Codebase** | ~36,207 lines | ~5,000 (router) + ~3,000 (UI) | `tokei src/` line count |
| **Env vars** | 124 (70 undocumented) | ~30 (all documented) | Pydantic Settings model |
| **Self-hosting** | Complex (16 env vars min) | `docker compose up` (3 env vars) | Time from clone to running gateway |
| **Routing latency overhead** | ~5ms (KNN) | ~2.1ms (centroid) | p50 of `routeiq.routing.decision.duration` |
| **Multi-replica consistency** | Not supported (single worker) | Redis-backed shared state | A/B counter accuracy across replicas |
| **Admin UI** | None | 7 pages operational | Manual verification |
| **PyPI installable** | No | `pip install routeiq-router` | PyPI listing |

---

## 9. Decision Log

Key architectural decisions made during TG3 analysis:

| # | Decision | Rationale | Alternatives Rejected |
|---|----------|-----------|----------------------|
| D1 | **Use `CustomRoutingStrategyBase` instead of monkey-patching** | Official LiteLLM extension point; per-instance; enables multi-worker | Sidecar (extra hop), Fork (maintenance burden), Microservices (premature) |
| D2 | **Ship centroid-based zero-config routing** | Solves cold-start problem; 72–78% accuracy with zero setup | Require training data (current); random routing (no intelligence) |
| D3 | **React + Vite for admin UI (not extend LiteLLM Next.js)** | Independently versioned; decoupled from LiteLLM release cadence | Next.js (coupling), Streamlit (limited UX), LiteLLM fork (merge conflicts) |
| D4 | **Embedded UI as default deployment** | Single container simplicity; zero additional infrastructure | Sidecar (extra container), CDN (requires CORS) |
| D5 | **Redis for cross-replica state** | Standard, well-understood; LiteLLM already uses Redis | PostgreSQL (latency), etcd (operational complexity), in-memory (no sharing) |
| D6 | **~30 env vars via Pydantic Settings** | 76% reduction; validated at startup; clear error messages | Keep 124 vars (confusing), single config file only (less 12-factor) |
| D7 | **5-tier complexity classification** | Finer-grained than binary simple/complex; maps to model tier selection | Binary (too coarse), continuous score (harder to configure) |
| D8 | **5 routing profiles** | User-selectable cost/quality presets; per-team/per-key defaults | Single strategy for all (inflexible), per-request config (complex) |
| D9 | **JSON centroid format (not pickle)** | Human-auditable, no security concerns, cross-platform | Pickle (security risk), protobuf (overengineered for 100KB data) |
| D10 | **4 self-hosting tiers** | Not everyone runs K8s; `docker compose up` must work | K8s-only (excludes small teams), single tier (one size fits none) |

---

## Appendix A: Document Index

| Document | Scope | Lines |
|----------|-------|:-----:|
| [TG3 Alternative Patterns](tg3-alternative-patterns.md) | 5 architecture options evaluated; Option A recommended | 737 |
| [TG3 Admin UI Design](tg3-admin-ui-design.md) | React + TypeScript + Vite; 4 deployment modes; 7 pages; ~43 endpoints | 1,292 |
| [TG3 Cloud-Native Design](tg3-cloud-native-design.md) | 12-factor compliance; 4 self-hosting tiers; state externalization; serverless | 1,568 |
| [TG3 NadirClaw Integration](tg3-nadirclaw-integration.md) | Centroid routing; profiles; 6-stage pipeline; session persistence | 1,112 |

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **Centroid** | Mean embedding vector for a category of prompts; enables classification via cosine similarity |
| **`CustomRoutingStrategyBase`** | LiteLLM's official abstract class for custom routing; at [`router.py:671`](../../reference/litellm/litellm/types/router.py:671) |
| **`CustomLogger`** | LiteLLM's callback hook system for pre-call, post-call, and routing events |
| **NadirClaw** | Centroid-based routing engine providing zero-config intelligent routing |
| **Routing Profile** | User-selectable cost/quality preset (auto, eco, premium, free, reasoning) |
| **Tier** | Prompt complexity level (0=trivial, 1=simple, 2=moderate, 3=complex, 4=expert) |
| **A/B Testing** | Traffic splitting between routing strategies for comparative evaluation |
| **routeiq-router** | PyPI package name for the RouteIQ routing library (~5,000 lines) |
| **routeiq-ui** | PyPI package name for the Admin UI static assets |
| **HPA** | Kubernetes Horizontal Pod Autoscaler |
| **PDB** | Kubernetes Pod Disruption Budget |
| **ESO** | External Secrets Operator (syncs secrets from Vault/AWS SM/GCP SM to K8s) |
| **pw** | Person-week (unit of effort) |
