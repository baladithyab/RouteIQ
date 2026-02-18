# RouteIQ v0.2.0 — Critical Architecture Assessment

**Date**: 2026-02-18  
**Assessor**: Code Skeptic (Kilo Code)  
**Method**: Full static analysis of 36,207 LoC across 63 Python files + runtime unit test execution  
**Motto**: *"Show me the logs or it didn't happen."*

---

## Executive Summary

RouteIQ v0.2.0 is an ambitious AI gateway wrapping LiteLLM with 18+ ML routing strategies, 5 MCP protocol surfaces, an A2A gateway, a 12-plugin system, policy engine, RBAC, and full observability. The codebase is **large** (36K LoC), **feature-rich on paper**, but has **critical architectural debts** that undermine its "production-grade" claims.

**Verdict: 🔴 NOT PRODUCTION-READY** — 4 critical blockers, 5 architectural smells, 3 dead code areas, and 70 undocumented env vars.

### Quick Scorecard

| # | Area | Rating | Key Issue |
|---|------|--------|-----------|
| 1 | Monkey-Patching | 🔴 Critical | Fragile, `unpatch()` broken, sync-in-async, silent fallback |
| 2 | Single-Worker | 🔴 Critical | `workers=1` hardcoded, can't scale a single process |
| 3 | Env Var Sprawl | 🔴 Critical | 124 in code, 54 documented, 20 prefixes, no validation |
| 4 | Middleware Ordering | 🔴 Critical | 503 load-shed has no X-Request-ID, no CORS |
| 5 | Singleton Pattern | 🟡 Concerning | 16 singletons, only 5/21 reset in tests |
| 6 | MCP Surfaces | 🟡 Concerning | 5 surfaces x 5,958 lines, protocol mismatch |
| 7 | Test Coverage | 🟡 Concerning | 1 failing test, 208 warnings, singleton leaks |
| 8 | GATE Reports | 🟡 Concerning | 64.7% security pass rate called "no defects" |
| 9 | Code Organization | 🟡 Concerning | God modules (1600 LoC), 30 circular import workarounds |
| 10 | Plugin System | 🟢 Acceptable | Well-designed, but dead code in http_pool_setup |
| 11 | Dead Code | 🟡 Concerning | Custom router loading, lifespan_with_plugins, version x3 |

---

## 1. The Monkey-Patching Approach 🔴

**Files**: [`routing_strategy_patch.py`](src/litellm_llmrouter/routing_strategy_patch.py:1)

### What's Built
RouteIQ patches **3 methods** on `litellm.router.Router` at runtime:
- [`routing_strategy_init`](src/litellm_llmrouter/routing_strategy_patch.py:476) — intercepts strategy initialization
- [`get_available_deployment`](src/litellm_llmrouter/routing_strategy_patch.py:479) — sync deployment selection
- [`async_get_available_deployment`](src/litellm_llmrouter/routing_strategy_patch.py:485) — async deployment selection

### What's Broken

1. **[`unpatch_litellm_router()`](src/litellm_llmrouter/routing_strategy_patch.py:505) is incomplete** — It only restores `routing_strategy_init` (stored in module global `_original_routing_strategy_init`). The other two originals (`get_available_deployment`, `async_get_available_deployment`) are stored as **local variables** inside `patch_litellm_router()` and are lost. Calling `unpatch()` leaves 2 of 3 methods permanently patched.

2. **No actual signature validation** — The docstring at line 17 claims "checks for method signature compatibility" but the code only does `hasattr(Router, "routing_strategy_init")` at [line 458](src/litellm_llmrouter/routing_strategy_patch.py:458). If LiteLLM changes parameter names or order, the wrapped function silently accepts the wrong arguments.

3. **Sync blocking in async path** — [`_async_get_deployment_via_llmrouter()`](src/litellm_llmrouter/routing_strategy_patch.py:367) at line 390 says *"Fallback to sync version as LLMRouter doesn't have async API"* and calls the synchronous `_get_deployment_via_llmrouter()`. This **blocks the event loop** during ML inference (embedding computation for KNN routing).

4. **Memory leak** — [`_routing_attempts`](src/litellm_llmrouter/routing_strategy_patch.py:150) dict grows unbounded to 10K entries, then does a full `clear()`. Individual completed requests are never cleaned up. Under sustained load this creates periodic ~10K-entry allocations.

5. **Silent fallback everywhere** — If pipeline routing fails ([line 278](src/litellm_llmrouter/routing_strategy_patch.py:278)), it falls back silently. If LLMRouter returns no model ([line 343](src/litellm_llmrouter/routing_strategy_patch.py:343)), it falls back to first deployment. No metrics, no alerts, no degradation signal.

6. **LiteLLM version pin is razor-thin** — [`pyproject.toml:20`](pyproject.toml:20): `litellm>=1.81.3,<1.82.0`. Any upstream patch release could break the 3-method surface. No CI job validates against LiteLLM HEAD.

### Recommendation
Replace monkey-patching with a **Router subclass** or **composition pattern**. LiteLLM's Router can be extended — this was a shortcut that became the foundation.

---

## 2. Single-Worker Limitation 🔴

**Files**: [`startup.py:355`](src/litellm_llmrouter/startup.py:355), [`docker/entrypoint.sh:168-179`](docker/entrypoint.sh:168)

### What's Built
- [`startup.py:355`](src/litellm_llmrouter/startup.py:355): `"workers": kwargs.get("workers", 1)` — defaults to 1 worker
- [`startup.py:434`](src/litellm_llmrouter/startup.py:434): argparse default is also `1`
- [`docker/entrypoint.sh`](docker/entrypoint.sh:178): Just `exec python -m litellm_llmrouter.startup "$@"` — no worker override

### What This Means
A single uvicorn worker means:
- **One Python process** handles ALL requests
- **One event loop** — any sync blocking (like KNN embedding computation in the monkey-patch) stalls everything
- **No CPU parallelism** — Python's GIL limits CPU-bound tasks to one core
- **HA workaround** (N pods x 1 worker) works for stateless routing but wastes resources (each pod carries full memory overhead of LiteLLM + LLMRouter + ML models)

### What's Missing
No throughput benchmarks exist. The `docs/load-soak-gates.md` doc talks about load gates but no actual results are provided. "Production-grade" is a claim without evidence.

### Recommendation
The HA workaround is viable but should be explicitly documented as a **limitation**, not hidden. Performance benchmarks under realistic load are mandatory before any production deployment.

---

## 3. Environment Variable Sprawl 🔴

**Files**: [`.env.example`](.env.example), all source files

### The Numbers

| Metric | Count |
|--------|-------|
| Env vars referenced in code | **124** |
| Env vars documented in `.env.example` | **54** |
| **Undocumented env vars** | **70** |
| Unique prefixes used | **20** |

### Prefix Chaos (top 10)

| Prefix | Count | Purpose |
|--------|-------|---------|
| `ROUTEIQ_` | 30 | RouteIQ-specific settings |
| `MCP_` | 27 | MCP gateway settings |
| `LLMROUTER_` | 16 | LLMRouter/routing settings |
| `OTEL_` | 11 | OpenTelemetry (standard) |
| `REDIS_` | 10 | Redis connection |
| `LITELLM_` | 8 | LiteLLM settings |
| `CONFIG_` | 8 | Config loading |
| `CACHE_` | 8 | Caching |
| `ADMIN_` | 7 | Admin auth |
| `DATABASE_` | 6 | Database |

### What's Missing
- **No startup validation** — Incorrect values silently produce wrong behavior (e.g., `ROUTEIQ_MAX_CONCURRENT_REQUESTS=abc` would need investigation)
- **No schema** — No Pydantic settings model, no JSON schema, no type checking
- **Three competing configuration systems**: env vars, YAML config, and LiteLLM's own config

### Recommendation
Create a `pydantic.BaseSettings` model for all RouteIQ env vars with validation, type coercion, and documentation generation. Consolidate to 2-3 prefixes max.

---

## 4. Middleware Stack Ordering 🔴

**Files**: [`gateway/app.py:84-141`](src/litellm_llmrouter/gateway/app.py:84), [`resilience.py:487-515`](src/litellm_llmrouter/resilience.py:487)

### The Order (outside to inside)

Starlette's `add_middleware()` stacks in reverse — last added is outermost. Then `add_backpressure_middleware()` wraps `app.app` outermost of all:

```
Request -> BackpressureMiddleware (ASGI wrap at line 509)
        -> RouterDecisionMiddleware (last add_middleware)
        -> PluginMiddleware
        -> ManagementMiddleware
        -> PolicyMiddleware
        -> RequestIDMiddleware
        -> CORSMiddleware (first add_middleware)
        -> Routes
```

### The Defect

When [`BackpressureMiddleware`](src/litellm_llmrouter/resilience.py:312) returns a **503 load-shed response**, it bypasses ALL other middleware. This means:
- **No `X-Request-ID` header** — Responses are untraceable during the exact moments you need observability most
- **No CORS headers** — Browser clients get CORS errors instead of 503, masking the real problem
- **No policy enforcement** — Though arguably correct for load shedding
- **No plugin hooks** — Cost tracking, audit logging all miss load-shed events

This is confirmed by the code: [`app.app = BackpressureMiddleware(original_app, config, drain_manager)`](src/litellm_llmrouter/resilience.py:509) replaces the inner ASGI app.

### Recommendation
Move `add_backpressure_middleware()` call BEFORE `_configure_middleware()`, or have BackpressureMiddleware manually set `X-Request-ID` and CORS headers on 503 responses.

---

## 5. Singleton Pattern 🟡

### The Numbers

| Metric | Count |
|--------|-------|
| Module-level singletons | **16** |
| `reset_*()` functions | **21** |
| Singletons reset in unit conftest | **5** |
| **Singletons leaking between tests** | **16** |

### What's Reset (in [`tests/unit/conftest.py:40-56`](tests/unit/conftest.py:40))
1. `reset_observability_manager()`
2. `reset_config_sync_manager()`
3. `reset_hot_reload_manager()`
4. `reset_gateway_metrics()`
5. `reset_migration_state()`

### What's NOT Reset
- [`reset_a2a_gateway()`](src/litellm_llmrouter/a2a_gateway.py:1596)
- [`reset_mcp_gateway()`](src/litellm_llmrouter/mcp_gateway.py:1283)
- [`reset_routing_singletons()`](src/litellm_llmrouter/strategy_registry.py:1410)
- [`reset_drain_manager()`](src/litellm_llmrouter/resilience.py:306)
- [`reset_circuit_breaker_manager()`](src/litellm_llmrouter/resilience.py:997)
- [`reset_policy_engine()`](src/litellm_llmrouter/policy_engine.py:899)
- [`reset_quota_enforcer()`](src/litellm_llmrouter/quota.py:979)
- [`reset_plugin_manager()`](src/litellm_llmrouter/gateway/plugin_manager.py:1235)
- [`reset_callback_bridge()`](src/litellm_llmrouter/gateway/plugin_callback_bridge.py:246)
- [`reset_plugin_middleware()`](src/litellm_llmrouter/gateway/plugin_middleware.py:313)
- [`reset_http_client_pool()`](src/litellm_llmrouter/http_client_pool.py:391)
- [`reset_audit_repository()`](src/litellm_llmrouter/audit.py:444)
- [`reset_affinity_tracker()`](src/litellm_llmrouter/conversation_affinity.py:360)
- [`reset_sessions()`](src/litellm_llmrouter/mcp_jsonrpc.py:125)
- [`reset_client_instantiation_count()`](src/litellm_llmrouter/http_client_pool.py:95)
- `clear_evaluator_plugins()` (evaluator.py)

### Concurrency Safety
- [`_active_sessions`](src/litellm_llmrouter/mcp_jsonrpc.py:83): Plain `dict` — no lock
- [`_a2a_gateway`](src/litellm_llmrouter/a2a_gateway.py:1577): Protected by `threading.Lock()`
- [`_mcp_gateway`](src/litellm_llmrouter/mcp_gateway.py:1264): Protected by `threading.Lock()`
- [`_registry_instance`](src/litellm_llmrouter/strategy_registry.py:1386): Protected by `threading.Lock()`
- Most others: No lock

### Recommendation
Add ALL `reset_*()` calls to the conftest. Consider dependency injection (FastAPI's `Depends()`) for testability.

---

## 6. MCP Surfaces 🟡

**Files**: 6 files, 5,958 lines total

| Surface | File | Lines | Endpoint | Protocol |
|---------|------|-------|----------|----------|
| Core Gateway | [`mcp_gateway.py`](src/litellm_llmrouter/mcp_gateway.py:1) | 1,292 | Internal | Python API |
| JSON-RPC | [`mcp_jsonrpc.py`](src/litellm_llmrouter/mcp_jsonrpc.py:1) | 868 | `POST /mcp` | JSON-RPC 2.0 |
| SSE | [`mcp_sse_transport.py`](src/litellm_llmrouter/mcp_sse_transport.py:1) | 1,434 | `/mcp/sse` | SSE + POST |
| REST/Parity | [`mcp_parity.py`](src/litellm_llmrouter/mcp_parity.py:1) | 1,072 | `/v1/mcp/*`, `/mcp-rest/*` | REST |
| Routes | [`routes/mcp.py`](src/litellm_llmrouter/routes/mcp.py:1) | 826 | `/llmrouter/mcp/*` | REST |
| Tracing | [`mcp_tracing.py`](src/litellm_llmrouter/mcp_tracing.py:1) | 466 | N/A | OTel |

### Issues
1. **Protocol mismatch** — GATE6 acknowledges `POST /mcp` uses JSON-RPC 2.0, but `docs/mcp-gateway.md` describes REST. Users following the docs will get 400 errors.
2. **5 surfaces for one feature** — JSON-RPC, SSE, REST (parity), REST (routes/mcp), proxy. Is this complexity justified for how many users actually use MCP?
3. **`_active_sessions`** in mcp_jsonrpc.py is a plain dict with no lock — safe only because single-worker, but fragile assumption.

### What's Good
- Feature flags (`MCP_GATEWAY_ENABLED`, `MCP_SSE_TRANSPORT_ENABLED`, etc.) allow selective enablement
- Separate tracing module with proper OTel instrumentation
- Clear separation between native MCP and REST compatibility

---

## 7. Test Coverage 🟡

### Runtime Results (verified)
```
$ uv run pytest tests/unit/ -x --tb=short -q
FAILED tests/unit/test_streaming_correctness.py::TestIncrementalYields::test_first_chunk_available_before_last_sent
1 failed, 2095 passed, 14 skipped, 208 warnings in 15.53s
```

### What's Good
- 2,095 passing tests is substantial
- 86 test files covering most modules
- Property-based testing with Hypothesis
- Auto-skip for integration tests when Docker stack is down

### What's Concerning
1. **[`test_first_chunk_available_before_last_sent`](tests/unit/test_streaming_correctness.py:363) FAILS** — `AssertionError: TTFB (42.24ms) is too close to total time (134.11ms). This suggests full buffering rather than incremental yields.` — Streaming correctness is broken.

2. **208 warnings** — Mostly `DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated` from dependencies. 7 from RouteIQ's own code using deprecated [`asyncio.get_event_loop()`](src/litellm_llmrouter/startup.py:284).

3. **HTTP client pool not initialized during tests** — Log warning: `HTTP client pool not initialized; falling back to per-request client.` This confirms the dead code path in `create_app()`.

4. **16 singletons leaking** — Cross-test contamination risk (see section 5).

---

## 8. GATE Reports 🟡

### Assessment: Theater-Adjacent

| Gate | Tests | Passed | Rate | Claimed Outcome |
|------|-------|--------|------|-----------------|
| GATE6 (MCP) | ~15 | ~12 | ~80% | Acknowledges protocol mismatch |
| GATE7 (Security) | 17 | 11 | **64.7%** | "No security defects found" |
| GATE9 (E2E) | ~30 | ~25 | ~83% | "PASS with credential limitations" |
| GATE11 (Final) | ~20 | ~18 | ~90% | "PASS with documented constraints" |

### Issues
1. **GATE7 claims "No security defects found"** despite 6/17 failures (35.3%). The failures are attributed to "missing LLM provider API keys" — but a security validation that can't actually test budget enforcement, rate limiting, or key rotation in the test environment is an **incomplete security validation**, not a passing one.

2. **GATE9 tested without credentials** — For an AI gateway, testing without any LLM provider credentials means you can't verify: routing works, model fallback works, cost tracking works, or that streaming actually streams. It's infrastructure testing, not E2E.

3. **GATE6 is the most honest** — It clearly documents the protocol mismatch and doesn't claim success where there isn't any.

### Recommendation
Rewrite GATE7 executive summary to honestly reflect the 64.7% pass rate. Add a mock LLM provider to enable credential-free E2E testing.

---

## 9. Code Organization 🟡

### Scale
| Metric | Value |
|--------|-------|
| Total Python files | 63 |
| Total lines of code | 36,207 |
| Files > 1,000 lines | 15 |
| Largest file | [`a2a_gateway.py`](src/litellm_llmrouter/a2a_gateway.py:1) (1,620 lines) |
| Circular import workarounds | **30** |
| Stub documentation files | 10 |

### God Modules (>1,000 lines)
1. [`a2a_gateway.py`](src/litellm_llmrouter/a2a_gateway.py:1) — 1,620 lines
2. [`strategies.py`](src/litellm_llmrouter/strategies.py:1) — 1,594 lines
3. [`model_artifacts.py`](src/litellm_llmrouter/model_artifacts.py:1) — 1,484 lines
4. [`strategy_registry.py`](src/litellm_llmrouter/strategy_registry.py:1) — 1,453 lines
5. [`mcp_sse_transport.py`](src/litellm_llmrouter/mcp_sse_transport.py:1) — 1,434 lines

### 30 Circular Import Workarounds
The codebase uses `TYPE_CHECKING` guards, lazy imports, and deferred imports 30 times. This is a strong signal that the module boundaries are wrong. Subsystems that need each other at runtime are in separate modules that can't import each other cleanly.

### 3 Version Strings
- [`__init__.py:101`](src/litellm_llmrouter/__init__.py:101): `__version__ = "0.0.5"`
- [`pyproject.toml:7`](pyproject.toml:7): `version = "0.2.0"`
- [`gateway/app.py:465`](src/litellm_llmrouter/gateway/app.py:465): `create_standalone_app(version="0.0.3")`

This is not just sloppy — it means any version-checking logic is unreliable.

### Recommendation
Break god modules into subpackages: `routing/`, `mcp/`, `security/`, `observability/`. Fix version to single source of truth.

---

## 10. Plugin System 🟢

**Files**: [`plugin_manager.py`](src/litellm_llmrouter/gateway/plugin_manager.py:1) (1,242 lines), 12 plugins (4,086 lines total)

### What's Good
- Topological dependency sort (Kahn's algorithm)
- Per-plugin failure modes (continue/abort/quarantine)
- Security policy (allowlist + capability restrictions)
- Startup timeouts (configurable)
- 11 capability types covering real use cases
- `PluginContext` with 7 subsystem accessors

### What's Concerning
- 12 built-in plugins, but [`evaluator.py`](src/litellm_llmrouter/gateway/plugins/evaluator.py:1) is only an abstract framework (no concrete evaluator ships)
- [`prompt_injection_guard.py`](src/litellm_llmrouter/gateway/plugins/prompt_injection_guard.py:1) is only 139 lines — minimal
- Plugin system is 1,242 lines for a feature that has no third-party consumers yet

---

## 11. Dead Code and Unimplemented Features

### Custom Router Loading — NOT IMPLEMENTED
[`strategies.py:883-889`](src/litellm_llmrouter/strategies.py:883):
```python
def _load_custom_router(self):
    """Load a custom router from the custom routers directory."""
    custom_path = os.environ.get("LLMROUTER_CUSTOM_ROUTERS_PATH", "/app/custom_routers")
    verbose_proxy_logger.info(f"Loading custom router from: {custom_path}")
    return None  # <-- ALWAYS returns None
```
This is exposed in `__init__.py` as `download_custom_router_from_s3` and documented in AGENTS.md. **The feature does not exist.**

### HTTP Client Pool Setup — Dead Code Path
[`gateway/app.py:441-446`](src/litellm_llmrouter/gateway/app.py:441):
```python
def http_pool_setup(app: FastAPI) -> None:
    app.state.llmrouter_http_pool_startup = _startup_http_client_pool
    app.state.llmrouter_http_pool_shutdown = _shutdown_http_client_pool
app.state.http_pool_setup = http_pool_setup  # Stored but never called
```
In [`startup.py:291`](src/litellm_llmrouter/startup.py:291), the code checks `hasattr(app.state, "llmrouter_http_pool_startup")` — but that attribute is never set because `http_pool_setup()` is stored but never invoked. Test logs confirm: `WARNING: HTTP client pool not initialized; falling back to per-request client.`

### Lifespan Context Manager — Never Used
[`gateway/app.py:416-432`](src/litellm_llmrouter/gateway/app.py:416): `lifespan_with_plugins` is defined but the code immediately comments: *"Note: We don't replace the lifespan here since LiteLLM manages its own."* Dead code.

### Deprecated API Usage
7 uses of [`asyncio.get_event_loop()`](src/litellm_llmrouter/startup.py:284) — deprecated since Python 3.10, will be removed in Python 3.16. Given `requires-python = ">=3.14"`, this is already on borrowed time.

### SIGTERM Handler Race Condition
[`startup.py:336`](src/litellm_llmrouter/startup.py:336): `loop.create_task(app.state.graceful_shutdown())` creates an async task but does NOT await it. Then immediately delegates to uvicorn's signal handler which may shut down the event loop before the drain completes. This means graceful shutdown may not actually be graceful.

---

## 12. What's Actually Good

Despite the critical issues, several things are well-done:

1. **Plugin manager architecture** — Topological sort, failure modes, quarantine, security policy. This is production-quality design.
2. **SSRF protection** — [`url_security.py`](src/litellm_llmrouter/url_security.py:1) (979 lines) validates at registration AND invocation time to catch DNS rebinding. Thorough.
3. **Policy engine** — [`policy_engine.py`](src/litellm_llmrouter/policy_engine.py:1) (994 lines) with OPA-style rules at ASGI layer. Real middleware, not theater.
4. **Audit logging** — [`audit.py`](src/litellm_llmrouter/audit.py:1) with structured events and file output.
5. **Feature flags** — MCP, A2A, SSE, Proxy are all independently toggleable.
6. **Helm chart** — Complete K8s deployment package with HPA, PDB, NetworkPolicy, ExternalSecret.
7. **Health probes** — Proper K8s liveness + readiness with degraded state reporting.
8. **2,095 passing unit tests** — Substantial coverage, even if singleton resets are incomplete.
9. **Entrypoint.sh** — HA-safe DB migrations, S3 config loading, exponential backoff. Well-engineered.

---

## 13. Unverified Claims

These claims appear in documentation but have no runtime evidence in the repository:

| Claim | Source | Evidence |
|-------|--------|----------|
| MCP tracing spans in Jaeger | docs/observability.md | No Jaeger screenshots or trace exports |
| HA failover < 30s | docs/high-availability.md (stub) | No failover test results |
| Streaming backpressure works | docs/streaming-verification.md | Unit test FAILS (TTFB too close to total) |
| Atomic config hot-reload | docs/configuration.md | No atomicity test |
| Circuit breaker OTel events | docs/observability.md | No span export evidence |
| "18+ ML strategies" | README.md | strategies.py exists but many are enum entries, not tested strategies |
| "Production-grade" | README.md | No load test results, no benchmark data |
| Custom router loading | AGENTS.md | `_load_custom_router()` returns None |

---

## 14. Recommendations (Priority Order)

### Immediate (Before any production deployment)
1. **Fix middleware ordering** — Move backpressure inside RequestID + CORS, or have it emit those headers
2. **Fix `unpatch_litellm_router()`** — Store all 3 original methods in module globals
3. **Fix http_pool_setup dead code** — Call the function or inline the setup
4. **Add all reset_*() calls to conftest** — Prevent cross-test contamination
5. **Fix version string** — Single source of truth from `pyproject.toml`
6. **Document all 124 env vars** — Or create Pydantic BaseSettings model

### Strategic (Before v1.0)
7. **Replace monkey-patching** — Router subclass or composition pattern
8. **Add async wrapper for LLMRouter** — Don't block event loop during routing
9. **Add startup env var validation** — Fail-fast on misconfiguration
10. **Break god modules** — Create `routing/`, `mcp/`, `security/` subpackages
11. **Implement or remove custom router loading** — Don't ship dead features
12. **Add mock LLM provider for E2E testing** — Enable credential-free validation
13. **Fix deprecated `asyncio.get_event_loop()`** — 7 occurrences
14. **Fix SIGTERM handler race** — Await drain before delegating to uvicorn

### Aspirational
15. **Multi-worker support** — This requires solving the monkey-patch problem first
16. **Configuration schema** — JSON Schema or Pydantic for all config
17. **Consolidate env var prefixes** — 20 to 3 (ROUTEIQ_, LITELLM_, OTEL_)
18. **Load testing** — Prove "production-grade" with actual benchmarks

---

## Appendix A: Raw Data

### File Sizes (source only)
```
63 Python files, 36,207 total lines
15 files > 1,000 lines
5 files > 1,400 lines
```

### Test Results (captured 2026-02-18)
```
1 failed, 2095 passed, 14 skipped, 208 warnings in 15.53s
FAILED: test_streaming_correctness.py::TestIncrementalYields::test_first_chunk_available_before_last_sent
Error: AssertionError: TTFB (42.24ms) is too close to total time (134.11ms)
```

### Singleton Inventory
```
16 module-level singletons
21 reset functions
5 reset in conftest (24%)
16 leaking between tests (76%)
```

### Environment Variable Inventory
```
124 env vars referenced in code
54 documented in .env.example
70 undocumented (57%)
20 distinct prefixes
```

### Plugin Inventory (12 plugins, 4,086 lines)
```
skills_discovery.py      508 lines  (substantial)
cache_plugin.py          467 lines  (substantial)
content_filter.py        411 lines  (substantial)
upskill_evaluator.py     422 lines  (substantial)
evaluator.py             399 lines  (abstract framework only)
cost_tracker.py          385 lines  (substantial)
guardrails_base.py       288 lines  (base class)
llamaguard_plugin.py     279 lines  (substantial)
bedrock_agentcore_mcp.py 278 lines  (substantial)
pii_guard.py             247 lines  (moderate)
bedrock_guardrails.py    223 lines  (moderate)
prompt_injection_guard.py 139 lines (minimal)
```
