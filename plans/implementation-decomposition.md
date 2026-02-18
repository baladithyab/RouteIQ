# RouteIQ Implementation Decomposition — Master Plan

> **Date:** 2026-02-18
> **Purpose:** Detailed, actionable implementation plans for 7 task groups derived from the TG0–TG4 architecture review
> **Input documents:** Executive summary, Critical Assessment, TG2 Evaluation, TG3 Rearchitecture Proposal, NadirClaw Integration Design, Admin UI Design, Cloud-Native Design
> **Bead tracking:** See `BEADS.md` for dependency graph and bead IDs

---

## Table of Contents

1. [TG-IMPL-A: P0 Critical Fixes (RouteIQ-oe6)](#tg-impl-a-p0-critical-fixes)
2. [TG-IMPL-B: Documentation Cleanup (RouteIQ-d4i)](#tg-impl-b-documentation-cleanup)
3. [TG-IMPL-C: LiteLLM Plugin Architecture (RouteIQ-n5d)](#tg-impl-c-litellm-plugin-architecture)
4. [TG-IMPL-D: NadirClaw Integration (RouteIQ-9m8)](#tg-impl-d-nadirclaw-integration)
5. [TG-IMPL-E: Admin UI MVP (RouteIQ-a5p)](#tg-impl-e-admin-ui-mvp)
6. [TG-IMPL-F: Cloud-Native Hardening (RouteIQ-y4c)](#tg-impl-f-cloud-native-hardening)
7. [TG-IMPL-G: Codebase Reduction (RouteIQ-2qz)](#tg-impl-g-codebase-reduction)

---

## TG-IMPL-A: P0 Critical Fixes

**Bead:** `RouteIQ-oe6` | **Priority:** P0 (Critical) | **Dependencies:** None | **Est. effort:** 3–5 days

### A1. Fix Middleware Ordering (CB-4)

**Problem:** [`BackpressureMiddleware`](src/litellm_llmrouter/resilience.py:312) wraps outermost via `app.app = BackpressureMiddleware(original_app, ...)` at [`resilience.py:509`](src/litellm_llmrouter/resilience.py:509). This means 503 load-shed responses bypass `RequestIDMiddleware`, `CORSMiddleware`, `PolicyMiddleware`, and all plugin hooks. During peak load, responses become untraceable.

**Current middleware order** (outermost → innermost):

```
BackpressureMiddleware    (ASGI wrap at app.py:449-450, resilience.py:509)
  → RouterDecisionMiddleware  (last add_middleware)
    → PluginMiddleware
      → ManagementMiddleware
        → PolicyMiddleware
          → RequestIDMiddleware
            → CORSMiddleware  (first add_middleware)
              → Routes
```

**Required order** (RequestID + CORS must wrap backpressure):

```
CORSMiddleware            (outermost - always present on responses)
  → RequestIDMiddleware   (always sets X-Request-ID)
    → BackpressureMiddleware
      → PolicyMiddleware
        → PluginMiddleware
          → ManagementMiddleware
            → RouterDecisionMiddleware
              → Routes
```

**Files to modify:**

| File | Function | Change |
|------|----------|--------|
| [`gateway/app.py:84-141`](src/litellm_llmrouter/gateway/app.py:84) | `_configure_middleware()` | Reorder middleware additions; add backpressure INSIDE RequestID/CORS |
| [`gateway/app.py:448-450`](src/litellm_llmrouter/gateway/app.py:448) | `create_app()` step 8 | Move `add_backpressure_middleware(app)` call BEFORE other middleware OR refactor to use `add_middleware()` |
| [`resilience.py:487-524`](src/litellm_llmrouter/resilience.py:487) | `add_backpressure_middleware()` | Option A: convert to use `add_middleware()` instead of ASGI wrapping. Option B: have 503 responses manually include `X-Request-ID` and CORS headers |

**Recommended approach (Option A):**
1. In `_configure_middleware()`, call `add_backpressure_middleware(app)` AFTER adding `CORSMiddleware` and `RequestIDMiddleware` but BEFORE `PolicyMiddleware`
2. Refactor `add_backpressure_middleware()` to use `app.add_middleware(BackpressureMiddleware, ...)` instead of `app.app = BackpressureMiddleware(app.app, ...)`
3. This requires making `BackpressureMiddleware` compatible with Starlette's middleware interface — it already implements `__call__(scope, receive, send)` so it should work

**Alternative (Option B — faster, lower risk):**
1. In [`_send_503_response()`](src/litellm_llmrouter/resilience.py:349), add `X-Request-ID` and CORS headers to the 503 response
2. Import CORS origin config and generate a UUID for X-Request-ID in the 503 path
3. Downside: duplicates CORS/RequestID logic, but fixes the immediate bug quickly

**Test:** Unit test that sends a request when backpressure is at capacity and verifies the 503 response has both `X-Request-ID` and `Access-Control-Allow-Origin` headers.

### A2. Fix `unpatch_litellm_router()` (CB-1 partial)

**Problem:** [`unpatch_litellm_router()`](src/litellm_llmrouter/routing_strategy_patch.py:505) only restores `routing_strategy_init`. The other two original methods (`get_available_deployment`, `async_get_available_deployment`) are stored as **local variables** inside `patch_litellm_router()` at [lines 466-473](src/litellm_llmrouter/routing_strategy_patch.py:466) and are lost when the function returns.

**Files to modify:**

| File | Line | Change |
|------|------|--------|
| [`routing_strategy_patch.py:40`](src/litellm_llmrouter/routing_strategy_patch.py:40) | Module globals | Add `_original_get_available_deployment = None` and `_original_async_get_available_deployment = None` |
| [`routing_strategy_patch.py:465-473`](src/litellm_llmrouter/routing_strategy_patch.py:465) | `patch_litellm_router()` | Change local vars to use `global` keyword for all 3 original methods |
| [`routing_strategy_patch.py:505-531`](src/litellm_llmrouter/routing_strategy_patch.py:505) | `unpatch_litellm_router()` | Restore all 3 methods from module globals, then reset all 3 globals to `None` |

**Concrete code change:**

```python
# Module level (line ~40)
_original_routing_strategy_init = None
_original_get_available_deployment = None
_original_async_get_available_deployment = None

# In patch_litellm_router() (line ~448)
def patch_litellm_router() -> bool:
    global _patch_applied, _original_routing_strategy_init
    global _original_get_available_deployment, _original_async_get_available_deployment
    # ... store to globals instead of locals ...

# In unpatch_litellm_router() (line ~505)
def unpatch_litellm_router() -> bool:
    global _patch_applied, _original_routing_strategy_init
    global _original_get_available_deployment, _original_async_get_available_deployment
    # ... restore all 3, then reset all 3 globals ...
```

**Test:** Unit test that patches, verifies methods are replaced, unpatches, verifies ALL 3 methods are restored to original.

### A3. Fix Strategy Registration at Runtime

**Problem:** `register_llmrouter_strategies()` in [`strategies.py`](src/litellm_llmrouter/strategies.py:1) registers strategy names in a dict but the Router doesn't know about them until `routing_strategy_init` is called with an `llmrouter-*` prefix. The monkey-patch intercepts this, but the strategies never appear in `Router.routing_strategy_args` or any discoverable listing.

**What "register at runtime" means concretely:**
1. [`register_llmrouter_strategies()`](src/litellm_llmrouter/strategies.py:1) populates the `LLMROUTER_STRATEGIES` dict (an enum-like map of strategy name → class)
2. The patched [`routing_strategy_init()`](src/litellm_llmrouter/routing_strategy_patch.py:72) checks `is_llmrouter_strategy()` and stores the strategy on `self._llmrouter_strategy`
3. But there's no way to **list** available llmrouter strategies from the Router or API

**Fix (immediate):**
- Add an endpoint or function that returns available strategies: merge `LLMROUTER_STRATEGIES.keys()` with LiteLLM's built-in `RoutingStrategy` enum values
- Expose in health/status endpoint

**Fix (strategic — for TG-IMPL-C):**
- Use `CustomRoutingStrategyBase` which is set on the Router **instance** via `router.set_custom_routing_strategy()`, making it properly discoverable

### A4. Fix Multi-Worker State

**Problem:** [`startup.py:355`](src/litellm_llmrouter/startup.py:355) defaults to `workers=1`. The monkey-patch in `routing_strategy_patch.py` patches **class-level** methods on `Router`. When uvicorn forks workers with `os.execvp()`, the patches are lost.

**What state needs to move to Redis (for multi-worker):**

| State | Current Location | Move To | Method |
|-------|-----------------|---------|--------|
| `_routing_attempts` dict | [`routing_strategy_patch.py:150`](src/litellm_llmrouter/routing_strategy_patch.py:150) | **Delete** (will be per-instance in `CustomRoutingStrategyBase`) | N/A |
| Active strategy name | `router._llmrouter_strategy` (in-memory) | Config (set once per worker) | `set_custom_routing_strategy()` handles this per-instance |
| A/B traffic counters | `strategy_registry.py` RoutingStrategyRegistry | Redis HINCRBY | Atomic increment |
| Circuit breaker state | [`resilience.py`](src/litellm_llmrouter/resilience.py:626) `CircuitBreaker._failures` deque | Redis sorted set + TTL | Eventually consistent |
| Session affinity | `conversation_affinity.py` in-memory LRU | Redis hash + TTL | Eventually consistent |
| Plugin registry | `plugin_manager.py` singleton | Per-worker from config | Stateless (config-driven) |
| Drain manager | [`resilience.py:294`](src/litellm_llmrouter/resilience.py:294) `_drain_manager` | Per-worker (correct as-is) | Instance-level |

**Immediate fix for multi-worker (minimal):**
1. Remove the `workers=1` default — allow `workers=N` in [`startup.py:355`](src/litellm_llmrouter/startup.py:355) and [`startup.py:434`](src/litellm_llmrouter/startup.py:434)
2. Document that `workers > 1` requires TG-IMPL-C (CustomRoutingStrategyBase) OR acceptance that routing state is per-worker
3. Add a startup warning if `workers > 1` and monkey-patch is active

### A5. Fix Version Identity Inconsistency

**Problem:** Three conflicting version strings:

| Location | Value |
|----------|-------|
| [`__init__.py:101`](src/litellm_llmrouter/__init__.py:101) | `"0.0.5"` |
| [`pyproject.toml:7`](pyproject.toml:7) | `"0.2.0"` |
| [`gateway/app.py:464`](src/litellm_llmrouter/gateway/app.py:464) | `"0.0.3"` (in `create_standalone_app`) |

**Fix:**
1. `__init__.py`: Read version from `importlib.metadata.version("litellm-llmrouter")` as single source of truth
2. `gateway/app.py`: Import `__version__` from `__init__`
3. Add test that asserts all versions match

### A6. Fix `http_pool_setup` Dead Code

**Problem:** [`gateway/app.py:441-446`](src/litellm_llmrouter/gateway/app.py:441) stores `http_pool_setup` as a function on `app.state` but **never calls it**. The function would set `app.state.llmrouter_http_pool_startup`, but since it's never called, [`startup.py:291`](src/litellm_llmrouter/startup.py:291) checking `hasattr(app.state, "llmrouter_http_pool_startup")` always fails. Test logs confirm: `WARNING: HTTP client pool not initialized`.

**Fix:** In `create_app()`, replace:

```python
app.state.http_pool_setup = http_pool_setup  # stored but never called
```

with:

```python
http_pool_setup(app)  # actually call it to set the lifecycle hooks
```

### A7. Add Missing Singleton Resets to Test Conftest

**Problem:** Only 5 of 21 `reset_*()` functions are called in [`tests/unit/conftest.py`](tests/unit/conftest.py:40). The other 16 singletons leak between tests.

**Fix:** Add all reset calls to the `autouse=True` fixture:

```python
# In tests/unit/conftest.py
from litellm_llmrouter.a2a_gateway import reset_a2a_gateway
from litellm_llmrouter.mcp_gateway import reset_mcp_gateway
from litellm_llmrouter.strategy_registry import reset_routing_singletons
from litellm_llmrouter.resilience import reset_drain_manager, reset_circuit_breaker_manager
from litellm_llmrouter.policy_engine import reset_policy_engine
from litellm_llmrouter.quota import reset_quota_enforcer
from litellm_llmrouter.gateway.plugin_manager import reset_plugin_manager
from litellm_llmrouter.gateway.plugin_callback_bridge import reset_callback_bridge
from litellm_llmrouter.gateway.plugin_middleware import reset_plugin_middleware
from litellm_llmrouter.http_client_pool import reset_http_client_pool
from litellm_llmrouter.audit import reset_audit_repository
```

---

## TG-IMPL-B: Documentation Cleanup

**Bead:** `RouteIQ-d4i` | **Priority:** P1 | **Dependencies:** TG-IMPL-G (reduce code first) | **Est. effort:** 2–3 days

### B1. Documentation Inventory

#### Stub Files (< 400 chars, just `# Title` + Attribution boilerplate — DELETE or FILL)

| File | Size | Action |
|------|------|--------|
| `docs/TECHNICAL_ROADMAP.md` | 331 chars | **DELETE** — superseded by `plans/technical-roadmap.md` and architecture docs |
| `docs/VALIDATION_PLAN.md` | 325 chars | **DELETE** — superseded by `plans/validation-plan.md` |
| `docs/api-parity-analysis.md` | 319 chars | **DELETE** — superseded by `plans/api-parity.md` |
| `docs/high-availability.md` | 313 chars | **FILL** — should be real HA guide (reference `docs/tutorials/ha-quickstart.md`) |
| `docs/hot-reloading.md` | 329 chars | **FILL** — hot reload feature exists, needs real docs |
| `docs/implementation-backlog.md` | 316 chars | **DELETE** — superseded by `BEADS.md` and `plans/backlog.md` |
| `docs/litellm-cloud-native-enhancements.md` | 353 chars | **DELETE** — superseded by `docs/architecture/tg3-cloud-native-design.md` |
| `docs/moat-mode.md` | 372 chars | **DELETE** — never implemented |
| `docs/observability-training.md` | 354 chars | **DELETE** — superseded by `docs/observability.md` |
| `docs/parity-roadmap.md` | 332 chars | **DELETE** — superseded by `plans/parity-roadmap.md` |
| `docs/quickstart-ha-compose.md` | 353 chars | **DELETE** — superseded by `docs/tutorials/ha-quickstart.md` |
| `docs/quickstart-otel-compose.md` | 379 chars | **DELETE** — superseded by `docs/tutorials/observability-quickstart.md` |
| `docs/release-checklist.md` | 331 chars | **DELETE** — superseded by `plans/release-checklist.md` |
| `docs/architecture/aws-deployment.md` | 318 chars | **DELETE** — superseded by `docs/deployment/aws.md` |
| `docs/architecture/ml-routing-cloud-native.md` | 308 chars | **DELETE** — superseded by `docs/architecture/tg3-cloud-native-design.md` |

**Total stubs to delete: 15 files**

#### Substantial Docs (Keep & Update)

| File | Size | Status | Notes |
|------|------|--------|-------|
| `docs/index.md` | 1,724 | **UPDATE** | Quick start section outdated |
| `docs/configuration.md` | 13,955 | **UPDATE** | Missing ~70 env vars |
| `docs/routing-strategies.md` | 13,644 | **UPDATE** | Needs to document current strategy registration |
| `docs/plugins.md` | 17,674 | ✅ Current | Well-written |
| `docs/security.md` | 13,067 | ✅ Current | |
| `docs/observability.md` | 16,135 | **UPDATE** | Contains unverified claims (MCP tracing, CB events) |
| `docs/mcp-gateway.md` | 3,671 | **UPDATE** | Documents REST API but native endpoint is JSON-RPC |
| `docs/a2a-gateway.md` | 5,111 | **UPDATE** | Mark as wrapper around LiteLLM native |
| `docs/api-reference.md` | 10,576 | **UPDATE** | Needs to match actual endpoints |
| `docs/skills-gateway.md` | 7,511 | ✅ Current | |
| `docs/mlops-training.md` | 10,690 | ✅ Current | |
| `docs/streaming-verification.md` | 7,604 | **FLAG** | Claims streaming works but test FAILS |
| `docs/vector-stores.md` | 5,395 | ✅ Current | |

#### Architecture Docs (Keep — produced by TG0–TG4)

All 8 architecture docs in `docs/architecture/` should be preserved as-is. They are the output of the architecture review and remain current.

#### Redundant Docs (resolve overlaps)

| Docs Pair | Resolution |
|-----------|------------|
| `docs/quickstart-docker-compose.md` ↔ `docs/index.md` | Merge into `docs/index.md` |
| `docs/ha-ci-gate.md` ↔ `docs/load-soak-gates.md` | Merge into `docs/load-soak-gates.md` |
| `docs/project-state.md` ↔ `docs/docs-consolidation-plan.md` | Delete `docs-consolidation-plan.md` (superseded by this document) |
| `docs/deployment.md` ↔ `docs/deployment/aws.md` + `docs/aws-production-guide.md` | Consolidate into `docs/deployment/` folder |

### B2. AGENTS.md Drift Items

Current `AGENTS.md` references:

| Reference | Actual State | Fix |
|-----------|-------------|-----|
| "monolithic `routes.py`" | Routes split into `routes/` package with 6 modules | Update structure diagram |
| `download_custom_router_from_s3` | Returns `None` always — dead code | Remove from AGENTS.md API |
| 30 circular import workarounds | May have changed since assessment | Re-audit and update count |
| `reference/litellm/` described as "git submodule" | Actually a regular directory | Clarify |
| Version `0.2.0` | Conflicting versions (0.0.3, 0.0.5, 0.2.0) | Fix after A5, then update AGENTS.md |
| `docs/deployment.md` listed | Now also has `docs/deployment/` directory | Update references |

### B3. Proposed Consolidated Doc Structure

```
docs/
├── index.md                    # Getting started (merge quickstart-docker-compose)
├── configuration.md            # All config (update with all 131 env vars)
├── routing-strategies.md       # ML strategies + registration
├── plugins.md                  # Plugin system guide
├── plugin-development-guide.md # Plugin development
├── security.md                 # Security considerations
├── observability.md            # OTel setup (fix unverified claims)
├── mcp-gateway.md              # MCP guide (fix protocol docs)
├── a2a-gateway.md              # A2A guide (note wrapper status)
├── api-reference.md            # API reference (update endpoints)
├── skills-gateway.md           # Anthropic skills
├── mlops-training.md           # MLOps pipeline
├── vector-stores.md            # Vector store support
├── streaming-verification.md   # Streaming (flag test failure)
├── rr-workflow.md              # Remote push workflow
├── project-state.md            # Current project state
├── architecture/               # 8 TG docs (keep as-is)
│   ├── overview.md
│   ├── deep-review-v0.2.0.md
│   ├── bedrock-discovery.md
│   ├── cloud-native.md
│   ├── mlops-loop.md
│   ├── ROUTEIQ-ARCHITECTURE-REVIEW-2026-02.md
│   ├── tg2-architecture-evaluation.md
│   ├── tg3-*.md (5 files)
├── deployment/                 # Deployment guides
│   ├── aws.md                  # (merge aws-production-guide.md)
│   ├── air-gapped.md
│   └── cloudfront-alb.md
└── tutorials/
    ├── ha-quickstart.md
    └── observability-quickstart.md
```

**Deletions:** 15 stub files + 3 redundant files = **18 files deleted**
**Merges:** 3 file merges

---

## TG-IMPL-C: LiteLLM Plugin Architecture

**Bead:** `RouteIQ-n5d` | **Priority:** P1 | **Dependencies:** TG-IMPL-A | **Est. effort:** 5–8 days

### C1. `CustomRoutingStrategyBase` API (from LiteLLM source)

**Location:** [`reference/litellm/litellm/types/router.py:671-718`](reference/litellm/litellm/types/router.py:671)

```python
class CustomRoutingStrategyBase:
    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """Returns an element from litellm.router.model_list"""
        pass

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """Returns an element from litellm.router.model_list"""
        pass
```

**`set_custom_routing_strategy()` at [`reference/litellm/litellm/router.py:8782-8806`](reference/litellm/litellm/router.py:8782):**

```python
def set_custom_routing_strategy(self, CustomRoutingStrategy: CustomRoutingStrategyBase):
    """Sets get_available_deployment and async_get_available_deployment
    on an instance of litellm.Router"""
    setattr(self, "get_available_deployment",
            CustomRoutingStrategy.get_available_deployment)
    setattr(self, "async_get_available_deployment",
            CustomRoutingStrategy.async_get_available_deployment)
```

**Key insight:** This uses `setattr()` on the Router **instance**, not the class. Each worker gets its own Router instance with its own strategy binding. This is what enables multi-worker operation.

### C2. Code to DELETE from `routing_strategy_patch.py`

The **entire file** (546 lines) can be deleted once the `CustomRoutingStrategyBase` wrapper is in place:

| Function | Lines | Reason for deletion |
|----------|-------|--------------------|
| `is_llmrouter_strategy()` | 47-51 | No longer needed — strategies set via `set_custom_routing_strategy()` |
| `get_llmrouter_strategy_name()` | 54-58 | No longer needed |
| `create_patched_routing_strategy_init()` | 61-100 | Replaced by `CustomRoutingStrategyBase` |
| `_initialize_pipeline_strategy()` | 103-137 | Pipeline routing built into new class |
| `create_patched_get_available_deployment()` | 140-220 | Replaced by `CustomRoutingStrategyBase.get_available_deployment()` |
| `_get_deployment_via_pipeline()` | 223-279 | Integrated into new class |
| `_get_deployment_via_llmrouter()` | 282-364 | Core logic moves to new class |
| `_async_get_deployment_via_llmrouter()` | 367-397 | Replaced by `async_get_available_deployment()` |
| `create_patched_async_get_available_deployment()` | 400-438 | Replaced by `CustomRoutingStrategyBase` |
| `patch_litellm_router()` | 441-502 | No longer needed |
| `unpatch_litellm_router()` | 505-531 | No longer needed |

### C3. Code to ADAPT from `strategies.py`

[`strategies.py`](src/litellm_llmrouter/strategies.py:1) (1,594 lines) contains the actual ML strategy logic. The key pieces to preserve:

| Component | Lines (est.) | Keep/Adapt |
|-----------|:---:|------------|
| `LLMRouterStrategyFamily` class | ~600 | **ADAPT** — becomes the engine inside `RouteIQRoutingStrategy` |
| `LLMROUTER_STRATEGIES` dict | ~50 | **KEEP** — strategy registry |
| `register_llmrouter_strategies()` | ~30 | **ADAPT** — no longer patches Router, instead configures the new class |
| `_load_custom_router()` | ~10 | **DELETE** — always returns `None` (dead code) |
| Enum `LLMRouterStrategyName` | ~40 | **KEEP** — strategy names |
| `route_with_observability()` | ~100 | **MOVE** into `RouteIQRoutingStrategy` |

### C4. New Class Hierarchy

```python
# src/routeiq/strategy.py (new file)

from litellm.types.router import CustomRoutingStrategyBase

class RouteIQRoutingStrategy(CustomRoutingStrategyBase):
    """
    LiteLLM-compatible routing strategy wrapping all RouteIQ routing logic.
    
    This is the single integration point with LiteLLM.
    Usage: router.set_custom_routing_strategy(RouteIQRoutingStrategy(config))
    """
    
    def __init__(self, config: RouteIQConfig):
        self.config = config
        self.strategy_family = LLMRouterStrategyFamily(...)
        self.pipeline = RoutingPipeline(...)
        # NadirClaw centroid routing (Phase 2)
        self.centroid_router = None  # CentroidRoutingStrategy
    
    async def async_get_available_deployment(
        self, model, messages=None, input=None,
        specific_deployment=False, request_kwargs=None
    ):
        """Async routing — delegates to pipeline or strategy family."""
        # 1. Extract query from messages/input
        # 2. Run through routing pipeline (A/B testing)
        # 3. Map selected model to LiteLLM deployment
        # 4. Return deployment dict from router.model_list
        ...
    
    def get_available_deployment(
        self, model, messages=None, input=None,
        specific_deployment=False, request_kwargs=None
    ):
        """Sync routing — same logic as async."""
        ...
```

### C5. Integration with LiteLLM Startup

In the new architecture, `startup.py` simplifies dramatically:

```python
# After LiteLLM creates its Router:
from litellm.proxy.proxy_server import llm_router
from routeiq.strategy import RouteIQRoutingStrategy

strategy = RouteIQRoutingStrategy(config)
llm_router.set_custom_routing_strategy(strategy)
```

No more monkey-patching, no more `workers=1` constraint.

### C6. Migration Steps

1. Create `src/routeiq/strategy.py` with `RouteIQRoutingStrategy(CustomRoutingStrategyBase)`
2. Move routing logic from `_get_deployment_via_llmrouter()` and `_get_deployment_via_pipeline()` into the new class
3. Move `route_with_observability()` from `strategies.py` into the new class
4. Add integration hook in `startup.py` to call `router.set_custom_routing_strategy()`
5. Write tests with `workers=4` to verify multi-worker operation
6. Delete `routing_strategy_patch.py` entirely
7. Update `__init__.py` to remove patch-related exports
8. Remove `workers=1` documentation and defaults

---

## TG-IMPL-D: NadirClaw Integration

**Bead:** `RouteIQ-9m8` | **Priority:** P2 | **Dependencies:** TG-IMPL-C | **Est. effort:** 5–7 days

### D1. NadirClaw Reference Files to Study

| File | Lines | Purpose |
|------|-------|---------|
| [`reference/NadirClaw/nadirclaw/classifier.py`](reference/NadirClaw/nadirclaw/classifier.py:1) | 175 | Binary complexity classifier using sentence embedding centroids |
| [`reference/NadirClaw/nadirclaw/routing.py`](reference/NadirClaw/nadirclaw/routing.py:1) | 508 | Routing intelligence: agentic detection, reasoning detection, profiles, sessions |
| [`reference/NadirClaw/nadirclaw/prototypes.py`](reference/NadirClaw/nadirclaw/prototypes.py:1) | 146 | Seed prompts for centroid training (80 simple + 65 complex) |
| [`reference/NadirClaw/nadirclaw/encoder.py`](reference/NadirClaw/nadirclaw/encoder.py:1) | 22 | Shared sentence-transformers encoder (all-MiniLM-L6-v2) |
| [`reference/NadirClaw/nadirclaw/settings.py`](reference/NadirClaw/nadirclaw/settings.py:1) | 114 | Settings class with SIMPLE_MODEL, COMPLEX_MODEL, etc. |
| [`reference/NadirClaw/nadirclaw/server.py`](reference/NadirClaw/nadirclaw/server.py:1) | 1,137 | FastAPI server (we don't need this — we integrate directly) |

### D2. How Centroid Routing Maps to `CustomRoutingStrategyBase`

NadirClaw's `BinaryComplexityClassifier.classify(prompt)` returns `(is_complex, confidence)`. The mapping:

```python
class CentroidRoutingStrategy:
    """Wraps NadirClaw's binary classifier for use inside RouteIQRoutingStrategy."""
    
    def __init__(self, centroids_path: str):
        self.simple_centroid = np.load(f"{centroids_path}/simple_centroid.npy")
        self.complex_centroid = np.load(f"{centroids_path}/complex_centroid.npy")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def route(self, query: str, deployments: List[Dict]) -> Dict:
        """Returns a deployment dict from the model_list."""
        emb = self.encoder.encode([query])[0]
        emb = emb / np.linalg.norm(emb)
        
        sim_simple = float(np.dot(emb, self.simple_centroid))
        sim_complex = float(np.dot(emb, self.complex_centroid))
        
        is_complex = sim_complex > sim_simple
        
        # Map to deployment: complex → premium tier, simple → budget tier
        tier = "premium" if is_complex else "budget"
        return self._select_deployment(tier, deployments)
```

This gets called from `RouteIQRoutingStrategy.async_get_available_deployment()` as the default routing path when no trained ML model is available.

### D3. Pre-trained Centroids to Ship

From NadirClaw's [`prototypes.py`](reference/NadirClaw/nadirclaw/prototypes.py:1):
- **80 simple prototypes** (factual Q&A, definitions, translations, simple code tasks)
- **65 complex prototypes** (architecture design, debugging, security audits, multi-step tasks)

**Centroid files to ship:**
- `data/simple_centroid.npy` (~1.5KB — 384-dim float32 vector)
- `data/complex_centroid.npy` (~1.5KB — 384-dim float32 vector)
- `data/centroid_metadata.json` (~500B — training info, prototype counts)

**Extended centroids** (from architecture design — 8 categories instead of 2):
- `data/centroids/code_generation.npy`
- `data/centroids/creative_writing.npy`
- `data/centroids/analysis.npy`
- `data/centroids/summarization.npy`
- `data/centroids/translation.npy`
- `data/centroids/conversation.npy`
- `data/centroids/reasoning.npy`
- `data/centroids/tool_use.npy`

Total: ~100KB shipped with the package.

### D4. Additional NadirClaw Components to Port

| NadirClaw Component | Port to RouteIQ | Effort |
|---------------------|-----------------|--------|
| `detect_agentic()` from [`routing.py:134-191`](reference/NadirClaw/nadirclaw/routing.py:134) | `NadirClawPreClassifier` — agentic detection | 1 day |
| `detect_reasoning()` from [`routing.py:237-253`](reference/NadirClaw/nadirclaw/routing.py:237) | `NadirClawPreClassifier` — reasoning detection | 0.5 day |
| `SessionCache` from [`routing.py:295-388`](reference/NadirClaw/nadirclaw/routing.py:295) | Enhance existing `ConversationAffinityTracker` | 1 day |
| `ROUTING_PROFILES` from [`routing.py:84`](reference/NadirClaw/nadirclaw/routing.py:84) | `RoutingProfile` system | 1 day |
| `apply_routing_modifiers()` from [`routing.py:417-508`](reference/NadirClaw/nadirclaw/routing.py:417) | Stage 6 of unified pipeline | 1 day |
| `MODEL_REGISTRY` from [`routing.py:19-53`](reference/NadirClaw/nadirclaw/routing.py:19) | Use LiteLLM's model info instead | 0 (use upstream) |

---

## TG-IMPL-E: Admin UI MVP

**Bead:** `RouteIQ-a5p` | **Priority:** P2 | **Dependencies:** None (independent workstream) | **Est. effort:** 5–8 days

### E1. Scaffolding Structure

```
ui/
├── package.json                # React 18, TypeScript 5.5, Vite 6
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.ts
├── index.html
├── public/
│   └── favicon.ico
├── src/
│   ├── main.tsx               # Entry point
│   ├── App.tsx                # Root layout + routing
│   ├── api/
│   │   ├── client.ts          # API client (fetch + auth header)
│   │   └── types.ts           # TypeScript types for API responses
│   ├── hooks/
│   │   ├── useAuth.ts         # Admin key auth hook
│   │   └── useMetrics.ts      # SSE metrics stream hook
│   ├── pages/
│   │   ├── Dashboard.tsx      # P0: Operational metrics
│   │   ├── RoutingConfig.tsx  # P0: Strategy + A/B testing
│   │   ├── RoutingExplorer.tsx # P1: Decision search
│   │   ├── PluginManager.tsx  # P1: Plugin lifecycle
│   │   ├── ConfigEditor.tsx   # P1: Monaco YAML editor
│   │   ├── PolicyEditor.tsx   # P2: OPA rule editor
│   │   └── MLOps.tsx          # P2: Training pipeline
│   ├── components/
│   │   ├── MetricCard.tsx
│   │   ├── ModelDistribution.tsx
│   │   ├── StrategySelector.tsx
│   │   ├── ABTestManager.tsx
│   │   └── Layout/
│   │       ├── Sidebar.tsx
│   │       ├── Header.tsx
│   │       └── ErrorBoundary.tsx
│   └── styles/
│       └── globals.css        # Tailwind imports
├── Dockerfile                 # Multi-stage build → static files
└── README.md
```

### E2. MVP Pages vs Later Phases

| Page | Priority | Phase | Backend Endpoints Required |
|------|:--------:|:-----:|---------------------------|
| **Dashboard** | P0 | MVP | `GET /admin/metrics`, `SSE /admin/metrics/stream` |
| **Routing Config** | P0 | MVP | `GET/PUT /admin/strategy`, `GET/POST/DELETE /admin/experiments` |
| Routing Explorer | P1 | Phase 2 | `GET /admin/routing-decisions` (paginated, filterable) |
| Plugin Manager | P1 | Phase 2 | `GET /admin/plugins`, `POST /admin/plugins/{id}/restart` |
| Config Editor | P1 | Phase 2 | `GET/PUT /admin/config` |
| Policy Editor | P2 | Phase 3 | `GET/PUT /admin/policies`, `POST /admin/policies/test` |
| MLOps Pipeline | P2 | Phase 3 | `GET/POST /admin/training-jobs`, `GET /admin/model-artifacts` |

### E3. MVP API Endpoints (8 total)

| Method | Endpoint | Body/Params | Response |
|--------|----------|-------------|----------|
| `POST` | `/admin/auth/verify` | `{ "api_key": "..." }` | `{ "valid": true, "role": "admin" }` |
| `GET` | `/admin/metrics` | — | `{ "requests_per_sec": 42, "p50_ms": 120, ... }` |
| `GET` | `/admin/metrics/stream` | SSE | `event: metrics\ndata: {...}\n\n` (every 5s) |
| `GET` | `/admin/strategy` | — | `{ "active": "llmrouter-knn", "available": [...] }` |
| `PUT` | `/admin/strategy` | `{ "strategy": "llmrouter-knn" }` | `{ "success": true }` |
| `GET` | `/admin/experiments` | — | `{ "experiments": [...] }` |
| `POST` | `/admin/experiments` | `{ "name": "...", "weights": {...} }` | `{ "id": "..." }` |
| `DELETE` | `/admin/experiments/{id}` | — | `{ "success": true }` |

### E4. Deployment — Embedded Mode

Build static assets with Vite, package in a Python wheel (`routeiq-ui`), serve via FastAPI `StaticFiles`:

```python
# In gateway/app.py
from importlib.resources import files

ui_dir = files("routeiq_ui") / "dist"
app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True))
```

---

## TG-IMPL-F: Cloud-Native Hardening

**Bead:** `RouteIQ-y4c` | **Priority:** P2 | **Dependencies:** TG-IMPL-A | **Est. effort:** 5–7 days

### F1. Environment Variable Audit

**Total env vars found in source: 131** (via `os.getenv()` grep)

Categorized by prefix:

| Prefix | Count | Category | Notes |
|--------|:-----:|----------|-------|
| `ROUTEIQ_` | 32 | RouteIQ application config | Core app settings |
| `MCP_` | 19 | MCP gateway settings | Potentially deletable after TG-IMPL-G |
| `LLMROUTER_` | 14 | LLMRouter/routing settings | Routing-specific |
| `A2A_` | 8 | A2A gateway settings | Potentially deletable after TG-IMPL-G |
| `OTEL_` | 8 | OpenTelemetry (standard) | Keep — standard OTel |
| `LITELLM_` | 6 | LiteLLM settings | Keep — upstream config |
| `REDIS_` | 5 | Redis connection | Keep |
| `CONFIG_` | 6 | Config loading/sync | Keep |
| `CACHE_` | 7 | Caching settings | Keep |
| `ADMIN_` | 3 | Admin auth | Keep |
| `AUDIT_` | 2 | Audit logging | Keep |
| `CONTENT_FILTER_` | 3 | Content filter plugin | Keep |
| `CONVERSATION_AFFINITY_` | 3 | Session persistence | Keep |
| `BEDROCK_GUARDRAIL_` | 3 | Bedrock guardrails plugin | Keep |
| `LLAMAGUARD_` | 4 | LlamaGuard plugin | Keep |
| `HTTP_CLIENT_` | 3 | HTTP client pool | Keep |
| `DATABASE_` | 1 | Database | Keep |
| `POLICY_` | 3 | Policy engine | Keep |
| `COST_TRACKER_` | 1 | Cost tracking plugin | Keep |
| `LOG_QUERIES` | 1 | Logging | Keep |
| `AWS_REGION` | 1 | AWS region | Keep |

**Target after consolidation:** ~30 core vars (per rearchitecture proposal)

**What to consolidate:**
- All `MCP_*` (19 vars) → deletable when MCP gateway removed
- All `A2A_*` (8 vars) → deletable when A2A gateway removed
- `LLMROUTER_*` → rename to `ROUTEIQ_ROUTING_*` for consistency
- Create Pydantic `BaseSettings` model with validation and defaults

### F2. Singletons to Externalize to Redis

| Singleton | Current Module | Redis Key Pattern | Consistency |
|-----------|---------------|-------------------|:-----------:|
| A/B traffic counters | `strategy_registry.py` | `routeiq:ab:{experiment_id}` HINCRBY | **Strong** |
| Circuit breaker state | `resilience.py` | `routeiq:cb:{name}` hash + TTL | Eventual |
| Session affinity | `conversation_affinity.py` | `routeiq:session:{key}` hash + TTL | Eventual |
| Active strategy | `strategy_registry.py` | `routeiq:strategy:active` string | Eventual |
| Drain manager | `resilience.py` | **Per-worker** (keep in-memory) | N/A |
| Plugin registry | `plugin_manager.py` | **Per-worker** (config-driven) | N/A |
| Policy engine rules | `policy_engine.py` | ConfigMap/file (keep) | Reload interval |
| Quota state | `quota.py` | `routeiq:quota:{team}` hash | Strong |

### F3. Helm Chart Changes

| File | Change |
|------|--------|
| `deploy/charts/routeiq-gateway/values.yaml` | Add `replicaCount: 3` default, `redis.enabled: true`, update env var list |
| `deploy/charts/routeiq-gateway/templates/deployment.yaml` | Add startup probe with 2-min timeout for ML model loading; add `REDIS_HOST` env |
| `deploy/charts/routeiq-gateway/templates/hpa.yaml` | Add custom metrics: `routeiq_requests_per_second`, `routeiq_p95_latency_ms` |
| `deploy/charts/routeiq-gateway/Chart.yaml` | Bump chart version, add Redis subchart dependency |
| NEW: `values-tier1.yaml` | Minimal: 1 replica, no Redis, SQLite |
| NEW: `values-tier2.yaml` | Standard: 2 replicas, Redis, PostgreSQL |
| NEW: `values-tier3.yaml` | Production: 3-20 replicas, HPA, PDB, ExternalSecrets |

---

## TG-IMPL-G: Codebase Reduction

**Bead:** `RouteIQ-2qz` | **Priority:** P1 | **Dependencies:** None | **Est. effort:** 3–5 days

### G1. Lines of Code in Each Potentially Redundant Module

| Module | Lines | Function | Redundancy Status |
|--------|------:|----------|------------------|
| [`a2a_gateway.py`](src/litellm_llmrouter/a2a_gateway.py:1) | 1,620 | A2A agent registry & task management | **REDUNDANT** — LiteLLM provides `/v1/agents`, `/a2a/{id}` natively |
| [`a2a_tracing.py`](src/litellm_llmrouter/a2a_tracing.py:1) | 720 | OTel tracing for A2A | Redundant if A2A gateway removed |
| [`mcp_gateway.py`](src/litellm_llmrouter/mcp_gateway.py:1) | 1,292 | MCP server registry & tool discovery | **REDUNDANT** — LiteLLM has native MCP |
| [`mcp_jsonrpc.py`](src/litellm_llmrouter/mcp_jsonrpc.py:1) | 868 | MCP JSON-RPC 2.0 handler | **REDUNDANT** — LiteLLM native |
| [`mcp_sse_transport.py`](src/litellm_llmrouter/mcp_sse_transport.py:1) | 1,434 | MCP SSE transport | **REDUNDANT** — LiteLLM native |
| [`mcp_parity.py`](src/litellm_llmrouter/mcp_parity.py:1) | 1,072 | MCP REST compatibility aliases | **REDUNDANT** — thin wrappers |
| [`mcp_tracing.py`](src/litellm_llmrouter/mcp_tracing.py:1) | 466 | OTel tracing for MCP | Redundant if MCP removed |
| [`routes/mcp.py`](src/litellm_llmrouter/routes/mcp.py:1) | 826 | MCP REST routes | **REDUNDANT** |
| [`routes/a2a.py`](src/litellm_llmrouter/routes/a2a.py:1) | ~600* | A2A routes | **REDUNDANT** |
| [`routing_strategy_patch.py`](src/litellm_llmrouter/routing_strategy_patch.py:1) | 546 | Monkey-patch (after TG-IMPL-C) | **DELETE** after plugin migration |
| [`leader_election.py`](src/litellm_llmrouter/leader_election.py:1) | 684 | Redis-based leader election | **EVALUATE** — LiteLLM may handle natively |
| [`quota.py`](src/litellm_llmrouter/quota.py:1) | 1,212 | Per-team quota enforcement | **EVALUATE** — LiteLLM has spend tracking |
| [`http_client_pool.py`](src/litellm_llmrouter/http_client_pool.py:1) | 408 | Shared httpx pool | **EVALUATE** — LiteLLM manages connections |

**Total potentially redundant: ~10,748 lines**

### G2. What LiteLLM Upstream Provides That We Duplicate

| Feature | RouteIQ Implementation | LiteLLM Native (upstream) |
|---------|----------------------|--------------------------|
| MCP Gateway | 5,958 lines across 6 files | Built-in MCP support (v1.80+) |
| A2A Gateway | 2,340 lines across 2 files + routes | Built-in `/v1/agents`, `/a2a/{id}` endpoints |
| Leader election | 684 lines (`leader_election.py`) | Proxy handles DB migrations with single-writer |
| Quota enforcement | 1,212 lines (`quota.py`) | Built-in spend tracking per key/team |
| HTTP client pool | 408 lines (`http_client_pool.py`) | LiteLLM manages its own httpx clients |
| Auth/RBAC | Partial in `auth.py`/`rbac.py` | Full auth system with key management |

### G3. Safe Deletion Plan

**Phase 1: Safe to delete immediately (no functionality loss)**

| Module | Lines | What Breaks | Mitigation |
|--------|------:|-------------|------------|
| `mcp_parity.py` | 1,072 | `/v1/mcp/*` aliases stop working | Users use LiteLLM's native MCP endpoints instead |
| `mcp_tracing.py` | 466 | MCP OTel spans stop | LiteLLM provides its own tracing |
| Routes: `routes/mcp.py` partial | ~400 | REST MCP routes | Use native MCP |

**Phase 2: Delete with feature flag migration**

| Module | Lines | What Breaks | Mitigation |
|--------|------:|-------------|------------|
| `mcp_gateway.py` | 1,292 | MCP server registry | Feature flag `MCP_GATEWAY_ENABLED` already governs this; verify LiteLLM native covers all use cases |
| `mcp_jsonrpc.py` | 868 | JSON-RPC surface at `/mcp` | Verify Claude Desktop works with LiteLLM native |
| `mcp_sse_transport.py` | 1,434 | SSE transport at `/mcp/sse` | Verify LiteLLM native SSE |
| `a2a_gateway.py` | 1,620 | `/a2a/agents` wrapper | Map to LiteLLM's `/v1/agents` |
| `a2a_tracing.py` | 720 | A2A OTel spans | Accept loss or verify LiteLLM tracing covers it |

**Phase 3: Delete after TG-IMPL-C**

| Module | Lines | What Breaks | Mitigation |
|--------|------:|-------------|------------|
| `routing_strategy_patch.py` | 546 | Monkey-patching mechanism | Replaced by `CustomRoutingStrategyBase` |

**Phase 4: Evaluate and potentially delete**

| Module | Lines | Decision Criteria |
|--------|------:|------------------|
| `leader_election.py` | 684 | Does LiteLLM handle leader election for DB migrations? |
| `quota.py` | 1,212 | Is LiteLLM's spend tracking sufficient? |
| `http_client_pool.py` | 408 | Is the dead-code path the only usage? |

### G4. Version Identity Fix

Fix the three conflicting version strings (see A5 above). After fixing, update:
- `__init__.py` — import from `importlib.metadata`
- `gateway/app.py` — use imported version
- Add CI check that versions are consistent

### G5. Dead Code Removal

| Dead Code | File | Line | Fix |
|-----------|------|------|-----|
| `_load_custom_router()` always returns `None` | `strategies.py:883-889` | 883 | Delete function and `download_custom_router_from_s3` export |
| `http_pool_setup` stored but never called | `gateway/app.py:441-446` | 441 | Call it (see A6) or inline |
| `lifespan_with_plugins` defined but never used | `gateway/app.py:416-432` | 416 | Delete dead context manager |
| 7x `asyncio.get_event_loop()` | `startup.py:284`, etc. | 284 | Replace with `asyncio.run()` |
| SIGTERM handler race condition | `startup.py:336` | 336 | Await drain before delegating |

---

## Execution Order Summary

```
Phase 1 (Now):   TG-IMPL-A (P0 Critical Fixes) + TG-IMPL-G (Codebase Reduction)
                 ↓ unblocks ↓                     ↓ unblocks ↓
Phase 2:         TG-IMPL-C (Plugin Arch) + TG-IMPL-E (Admin UI) + TG-IMPL-B (Docs)
                 ↓ unblocks ↓
Phase 3:         TG-IMPL-D (NadirClaw) + TG-IMPL-F (Cloud-Native)
```

**Total estimated effort:** 28–43 days across 7 task groups

| TG | Effort | Dependencies | Parallelizable With |
|----|:------:|:------------:|---------------------|
| **A** | 3–5d | None | G |
| **G** | 3–5d | None | A |
| **C** | 5–8d | A | E, B |
| **E** | 5–8d | None | C, B |
| **B** | 2–3d | G | C, E |
| **D** | 5–7d | C | F |
| **F** | 5–7d | A | D |

---

## Appendix: Key File Reference

| File | Lines | Role | Modified By TG |
|------|------:|------|:--------------:|
| `routing_strategy_patch.py` | 546 | Monkey-patch (delete) | A, C, G |
| `startup.py` | 484 | Entry point | A, C |
| `gateway/app.py` | 537 | App factory | A, E |
| `resilience.py` | 1,138 | Backpressure + CB | A, F |
| `strategies.py` | 1,594 | ML strategies | C, D |
| `strategy_registry.py` | 1,453 | A/B testing pipeline | C, F |
| `__init__.py` | 153 | Public API | A, C, G |
| `mcp_gateway.py` | 1,292 | MCP (delete) | G |
| `a2a_gateway.py` | 1,620 | A2A (delete) | G |
| `mcp_jsonrpc.py` | 868 | MCP JSON-RPC (delete) | G |
| `mcp_sse_transport.py` | 1,434 | MCP SSE (delete) | G |
| `mcp_parity.py` | 1,072 | MCP aliases (delete) | G |
| `routes/mcp.py` | 826 | MCP routes (delete) | G |
| `routes/a2a.py` | ~600 | A2A routes (delete) | G |
| `tests/unit/conftest.py` | — | Test fixtures | A |
