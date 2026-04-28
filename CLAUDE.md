# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

**RouteIQ Gateway** - A production-grade, cloud-native General AI Gateway built on
[LiteLLM](https://github.com/BerriAI/litellm) (proxy/API compatibility) and
[LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (ML-based routing intelligence).

Always refer to this project as **RouteIQ**. Do not rename `LITELLM_*` environment
variables. Do not claim RouteIQ implements features only inherited from upstream LiteLLM.

## Development Commands

### Install & Run

```bash
uv sync                            # Install dependencies
uv sync --extra dev                # Install with dev tools
uv run python -m litellm_llmrouter.startup --config config/config.yaml --port 4000
```

### Testing

```bash
uv run pytest tests/unit/ -x                    # Unit tests (fast, no external deps)
uv run pytest tests/integration/                # Integration tests (needs Docker stack)
uv run pytest tests/ -x -v                      # All tests
uv run pytest -k "test_name"                    # Run specific test by name
uv run pytest tests/unit/test_file.py -v -s     # Single file, verbose + stdout
uv run pytest tests/unit/test_file.py --trace   # With debugger
uv run pytest tests/property/                   # Property-based tests (hypothesis)
```

Integration tests auto-skip when Docker stack is not running on port 4010.
Start the test stack with: `docker compose -f docker-compose.local-test.yml up -d`

### Code Quality

```bash
uv run ruff format src/ tests/                                 # Format
uv run ruff check src/ tests/                                  # Lint
uv run ruff check --fix src/ tests/                            # Auto-fix
uv run mypy src/litellm_llmrouter/ --ignore-missing-imports    # Type check
```

### Git Hooks

Lefthook manages git hooks. Pre-commit runs ruff, yamllint, secret detection in parallel.
Pre-push runs unit tests, mypy, and security scanning sequentially.

```bash
./scripts/install_lefthook.sh      # Install
lefthook run pre-commit            # Manual run
```

### Docker

```bash
docker compose up -d                                                          # Basic
docker compose -f docker-compose.local-test.yml up -d                          # Local test stack
docker compose -f examples/docker/ha/docker-compose.ha.yml up -d               # HA (Redis/Postgres/Nginx)
docker compose -f examples/docker/observability/docker-compose.otel.yml up -d   # Observability (OTel/Jaeger)
docker build -f docker/Dockerfile -t litellm-llmrouter:latest . # Build
```

### Pushing Changes

Local `git push` may be blocked by Code Defender. Use Road Runner to push:

```bash
rr push                            # Sync and push
rr push-force                      # Force push (--force-with-lease)
```

After `rr push`, always sync local: `git pull`

## Architecture Overview

> **As of v1.0.0rc1 (2026-04-02).** Major rearchitecture (22-commit squash, 25 ADRs) shipped
> via commit `ba89b9e`. Key inversions since v0.2.0: monkey-patch → plugin API (ADR-0002),
> custom MCP surfaces → upstream LiteLLM delegation (ADR-0003, ADR-0017), embedded-in-LiteLLM
> → own FastAPI app (ADR-0012), scattered `os.environ.get()` → Pydantic `BaseSettings`
> (ADR-0013). See `docs/adr/` for all 25 decisions.

### Core Entry Points

- **`cli.py`** - Unified CLI: `routeiq start|validate-config|version|probe-services` (project.scripts entry point)
- **`startup.py`** - Legacy CLI entry: `python -m litellm_llmrouter.startup`
- **`gateway/app.py`** - App factory: `create_app()` (own FastAPI; LiteLLM Router installed as plugin)
- **`routes/`** - FastAPI routers: `health.py`, `a2a.py`, `mcp.py`, `config.py`, `models.py`, `admin_ui.py`

### Startup Load Order

1. Load Pydantic `Settings` from env + YAML (`settings.py`)
2. Probe optional external services (`service_discovery.py`) — feature availability report
3. Initialize LiteLLM (`litellm.proxy.proxy_server.initialize()`)
4. Install RouteIQ routing strategy on LiteLLM Router via `CustomRoutingStrategyBase` plugin API (`custom_routing_strategy.py`)
5. Create own FastAPI app (ADR-0012)
6. Add backpressure middleware (innermost, wraps ASGI directly)
7. Configure middleware (CORS, RequestID, Policy, Governance, Management, Plugin, RouterDecision)
8. Load gateway plugins (discovery + validation)
9. Register routes (health, llmrouter, governance, prompts, eval, admin UI)
10. Setup lifecycle hooks (plugin startup, HTTP pool, drain)

### Source Layout (`src/litellm_llmrouter/`)

**Routing (dataplane):**

| Module | Purpose |
|--------|---------|
| `custom_routing_strategy.py` | LiteLLM `CustomRoutingStrategyBase` plugin — routing entry point (replaces deleted monkey-patch per ADR-0002) |
| `strategies.py` | 18+ ML routing strategies (KNN, MLP, SVM, ELO, MF, hybrid) |
| `strategy_registry.py` | A/B testing, hot-swap, routing pipeline |
| `router_decision_callback.py` | Routing decision telemetry (TG4.1) |
| `centroid_routing.py` | Zero-config centroid-based routing (ADR-0010) |
| `personalized_routing.py` | Per-user preference learning via EMA + feedback endpoint (ADR-0025) |
| `router_r1.py` | Router-R1 iterative reasoning router (NeurIPS 2025, native over LiteLLM) |
| `conversation_affinity.py` | Conversation-based routing affinity |

**Control plane (multi-tenant governance):**

| Module | Purpose |
|--------|---------|
| `governance.py` | Workspaces, API keys, org hierarchy (ADR-0020) |
| `usage_policies.py` | Dynamic rate limits + budgets with condition matching (ADR-0022) |
| `guardrail_policies.py` | Config-driven input/output guardrails, 14 check types (ADR-0023) |
| `oidc.py` | OIDC/SSO: Keycloak, Auth0, Okta, Azure AD (ADR-0008) |
| `prompt_management.py` | Prompt templates + versioning + A/B + rollback |
| `rbac.py` | Role-based access control |
| `policy_engine.py` | OPA-style policy evaluation middleware |
| `quota.py` | Per-team/per-key quota enforcement |
| `audit.py` | Audit logging |
| `management_classifier.py` | Classifies LiteLLM management endpoints |
| `management_middleware.py` | RBAC/audit middleware for management ops |
| `auth.py` | Admin auth, RequestID middleware (raw ASGI), secret scrubbing |

**Gateway & plugins:**

| Module | Purpose |
|--------|---------|
| `gateway/app.py` | Own FastAPI app factory (composition root, ADR-0012) |
| `gateway/plugin_manager.py` | Plugin lifecycle with dependency resolution |
| `gateway/plugin_adapters.py` | Plugin adapter implementations |
| `gateway/plugin_callback_bridge.py` | Bridge between plugins and LiteLLM callbacks |
| `gateway/plugin_middleware.py` | Plugin middleware integration |
| `gateway/plugin_protocols.py` | Plugin protocol definitions (typing) |
| `gateway/plugins/` | Built-in plugins (14 total — see Plugin System below) |
| `semantic_cache.py` | Semantic caching for LLM responses |

**MCP & A2A (delegated to upstream LiteLLM per ADR-0017):**

| Module | Purpose |
|--------|---------|
| `mcp_gateway.py` | MCP server registry and tool discovery (REST) |
| `mcp_tracing.py` | OTel instrumentation for MCP |
| `a2a_gateway.py` | A2A agent registry |
| `a2a_tracing.py` | OTel instrumentation for A2A |

> Deleted in v1.0.0rc1 per ADR-0003: `mcp_jsonrpc.py`, `mcp_sse_transport.py`, `mcp_parity.py`.
> MCP JSON-RPC + SSE are now served by upstream LiteLLM. Only REST gateway + tracing remain.

**Evaluation & feedback loop:**

| Module | Purpose |
|--------|---------|
| `eval_pipeline.py` | COLLECT / EVALUATE / AGGREGATE / FEEDBACK loop with LLM-as-judge |

**Observability:**

| Module | Purpose |
|--------|---------|
| `observability.py` | OpenTelemetry init (traces, metrics, logs) |
| `metrics.py` | OTel metric helpers |
| `telemetry_contracts.py` | Versioned telemetry event schemas |

**Infra:**

| Module | Purpose |
|--------|---------|
| `settings.py` | Pydantic `BaseSettings` — centralized config (ADR-0013, replaces scattered env lookups) |
| `service_discovery.py` | Startup-time probing of Postgres/Redis/OTel/OIDC with graceful degradation (ADR-0011) |
| `cli.py` | Unified CLI entry point |
| `resilience.py` | Backpressure, drain manager, circuit breakers |
| `http_client_pool.py` | Shared httpx.AsyncClient pool |
| `hot_reload.py` | Filesystem-watching config hot-reload |
| `config_loader.py` | YAML config + S3/GCS download |
| `config_sync.py` | Background config sync (S3 ETag-based) — leader-only |
| `model_artifacts.py` | ML model verification (hash, signature) |
| `url_security.py` | SSRF protection (fail-closed DNS) |
| `database.py` | Database connection helpers |
| `migrations.py` | Database migration utilities (leader-only) |
| `leader_election.py` | HA leader election — K8s Lease API primary, Redis fallback (ADR-0015) |
| `redis_pool.py` | Redis connection pool |
| `env_validation.py` | Startup env var validation (advisory only) |

## Key Patterns

### Routing Strategy Integration

RouteIQ uses LiteLLM's official `CustomRoutingStrategyBase` plugin API
(ADR-0002). `install_plugin_routing_strategy(app)` wires RouteIQ's strategy
family into LiteLLM's Router at startup. **No monkey-patch required**;
multi-worker safe.

Critical: `install_plugin_routing_strategy()` must receive the `app` arg —
the TypeError is silently swallowed otherwise, and ML routing falls back to
LiteLLM default without error. See commits `59f80e9` + `7844419` for the
historical bug fix.

### MCP Surfaces

RouteIQ exposes the REST MCP gateway at `/llmrouter/mcp/*` and `/v1/llmrouter/mcp/*`
(`routes/mcp.py`). JSON-RPC and SSE transports are served by **upstream LiteLLM**
(ADR-0017). The RouteIQ-native `mcp_jsonrpc.py`, `mcp_sse_transport.py`,
`mcp_parity.py` were deleted in v1.0.0rc1 (ADR-0003).

MCP tool invocation is feature-flagged: `MCP_GATEWAY_ENABLED=true` +
`LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true`.

### Plugin System

Plugins extend the gateway via `GatewayPlugin` base class. They are loaded from
config BEFORE routes (deterministic ordering) and started during app lifespan.

**14 built-in plugins:** `agentic_pipeline`, `bedrock_agentcore_mcp`,
`bedrock_guardrails`, `cache_plugin`, `content_filter`, `context_optimizer`,
`cost_tracker`, `evaluator`, `guardrails_base`, `llamaguard_plugin`,
`pii_guard`, `prompt_injection_guard`, `skills_discovery`, `upskill_evaluator`.

`context_optimizer` (ADR-0024) applies 6 lossless transforms to reduce tokens
30–70%: JSON minification, tool schema dedup, system prompt dedup, whitespace
norm, chat history trim, semantic dedup.

### Governance Layer (ADR-0020)

Multi-tenant primitives are RouteIQ-native (not LiteLLM-inherited):
- Workspaces + API keys (`governance.py`)
- Usage policies — dynamic rate limits + budgets with condition matching (`usage_policies.py`)
- Guardrail policies — 14 check types, deny/log/alert actions (`guardrail_policies.py`)
- OIDC/SSO — Keycloak, Auth0, Okta, Azure AD (`oidc.py`)

State is file-backed by default; env paths: `ROUTEIQ_GOVERNANCE_STATE_PATH`,
`ROUTEIQ_USAGE_POLICIES_STATE_PATH`, `ROUTEIQ_GUARDRAIL_POLICIES_STATE_PATH`,
`ROUTEIQ_PROMPTS_STATE_PATH`. Auto-loaded at startup, saved after every CRUD
mutation.

### Evaluation & Feedback Loop

`eval_pipeline.py` implements COLLECT → EVALUATE → AGGREGATE → FEEDBACK:
- Collects routing decisions via `router_decision_callback.py`
- Evaluates via `gateway/plugins/evaluator.py` (LLM-as-judge)
- Aggregates quality scores per model/strategy
- Feeds verdicts into `personalized_routing.py` + centroid weights

API: `/api/v1/routeiq/eval/{stats,samples,run-batch,model-quality,push-feedback}`.
Admin UI: Observability page renders model-quality rankings + run-batch trigger.

### Settings (ADR-0013)

Pydantic `BaseSettings` in `settings.py` replaces scattered `os.environ.get()`
calls (124+ call sites collapsed). Supports profiles, validation, defaults.
**Do not** reintroduce direct env lookups in new code.

### Policy Engine

OPA-style pre-request policy evaluation at the ASGI layer. Runs before routing
and FastAPI auth. Supports fail-open (default) and fail-closed modes.
Configured via `POLICY_ENGINE_ENABLED` and `POLICY_CONFIG_PATH`.

### Auth Model

Two-tier auth: admin auth (`ADMIN_API_KEYS`, `X-Admin-API-Key` header) for
control-plane endpoints; user auth (LiteLLM's `user_api_key_auth`) for
data-plane endpoints. Admin auth is fail-closed.

### Config & Hot Reload

Config loaded from YAML files via `config_loader.py`. Supports S3/GCS download
with ETag-based change detection. Hot-reload watches filesystem for changes.
Background sync via `config_sync.py`.

## Important Constraints

### READ-ONLY Reference

`reference/litellm/` is a git submodule. **Never modify** files in this directory.

### Security

- No real secrets in code or tests (use `test-api-key` placeholders)
- SSRF protection via `url_security.py` for all external requests
- Pickle loading disabled by default (`LLMROUTER_ALLOW_PICKLE_MODELS=false`)
- Secret scrubbing in error logs
- Admin auth fail-closed when no keys configured

### Testing

- All new features require unit tests
- `asyncio_mode = "auto"` in pytest config (async tests are auto-detected)
- `hypothesis` for property-based testing (max_examples: 100)
- Integration tests require Docker stack (auto-skip otherwise)
- Some integration tests manage their own compose stack

## Code Style

- Python 3.12+ floor (`pyproject.toml` `requires-python = ">=3.12"`)
- Ruff formatter + linter (line-length: 88); Black removed in v1.0.0rc1
- Type hints required for public APIs
- Pydantic v2 for data validation; `pydantic-settings` for config (ADR-0013)
- Async/await throughout FastAPI routes
- No side effects on import
- New config values go through `settings.py` (Pydantic `BaseSettings`), not direct `os.environ.get()`

## Development Workflow

### Task Group (TG) Pattern

Each feature follows: create branch (`tg<id>-desc`) -> develop locally ->
squash merge to main -> commit as `feat: complete TG<id> description` ->
push via `rr push` if blocked.

### Common Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LITELLM_MASTER_KEY` | (required) | Admin access key |
| `OTEL_ENABLED` | `true` | OpenTelemetry |
| `A2A_GATEWAY_ENABLED` | `false` | A2A protocol |
| `MCP_GATEWAY_ENABLED` | `false` | MCP protocol |
| `POLICY_ENGINE_ENABLED` | `false` | Policy engine |
| `CONFIG_HOT_RELOAD` | `false` | Config hot-reload |
| `LLMROUTER_ALLOW_PICKLE_MODELS` | `false` | ML model pickle loading |
| `LLMROUTER_ROUTER_CALLBACK_ENABLED` | `true` | Routing telemetry |

## Quick Reference

### Adding an Endpoint

1. Add route in the appropriate module under `routes/` (`health.py`, `a2a.py`, `mcp.py`, `config.py`, or `models.py`)
2. Add auth dependency (`admin_api_key_auth` or `user_api_key_auth`)
3. Register in `gateway/app.py` `_register_routes()` if new router
4. Add unit test in `tests/unit/`

### Adding a Routing Strategy

1. Implement in `strategies.py`
2. Add to `LLMROUTER_STRATEGIES` dict + `LLMRouterStrategyFamily` enum
3. Auto-registered via `register_llmrouter_strategies()`
4. Add unit tests

### Adding a Plugin

1. Create in `gateway/plugins/`
2. Extend `GatewayPlugin`, implement `startup()` / `shutdown()`
3. Define `metadata` (capabilities, priority, dependencies)
4. Add unit tests

### Running the Gateway Locally

```bash
uv sync
docker compose -f docker-compose.local-test.yml up -d  # Dependencies
uv run python -m litellm_llmrouter.startup --config config/config.local-test.yaml --port 4000
```

## Non-Obvious Behaviors

These are critical gotchas that are easy to miss:

- **Own FastAPI app, not embedded in LiteLLM** (ADR-0012). RouteIQ creates its own app and installs LiteLLM Router as a routing-strategy plugin — inverted from pre-v1.0 where RouteIQ mounted on top of `litellm.proxy.proxy_server.app`.
- **`install_plugin_routing_strategy(app)` must receive `app`** — if the arg is missing, the TypeError is swallowed and ML routing silently falls back to LiteLLM default. Historical bug fixed in `59f80e9` + `7844419`; guard against regression.
- **Multi-worker safe.** No monkey-patch to preserve. The `CustomRoutingStrategyBase` plugin registers per-worker.
- **BackpressureMiddleware is the innermost middleware** registered first via `add_backpressure_middleware()` before `_configure_middleware()`, wrapping the ASGI app directly (replaces `app.app`). This is because `BaseHTTPMiddleware` breaks streaming.
- **Plugin hooks live on `app.state`** as lambdas, not in lifespan, because LiteLLM manages its own lifespan.
- **`/_health/ready` returns 200 for degraded state** (circuit breakers open), not 503.
- **MCP surfaces consolidated.** JSON-RPC / SSE now served by upstream LiteLLM (ADR-0003, ADR-0017). RouteIQ's remaining MCP surface is the REST gateway at `/llmrouter/mcp/*` and `/v1/llmrouter/mcp/*`.
- **SSRF checks happen twice**: at registration (no DNS) and invocation (with DNS) to catch rebinding. DNS failures are fail-closed (security fix in v1.0.0rc1).
- **MCP tool invocation is off by default** even when `MCP_GATEWAY_ENABLED=true`. Needs `LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true`.
- **Config sync only runs on HA leader** — K8s Lease API primary, Redis fallback (ADR-0015). Non-leader replicas skip it.
- **Prisma migrations are leader-gated** via `ROUTEIQ_LEADER_MIGRATIONS`.
- **Governance state is file-backed by default.** Auto-loaded at startup, saved after every CRUD mutation. See env paths in the Governance Layer section.
- **CORS default is `""` (deny-all)** as of v1.0.0rc1 — set `CORS_ORIGINS` explicitly. Previous `"*"` default is a CSRF vuln when combined with credentials.
- **Singletons need `reset_*()`** - every subsystem uses singletons. Tests MUST call `reset_*()` in `autouse=True` fixtures.
- **OTel provider reuse** - `ObservabilityManager` reuses existing TracerProvider if LiteLLM set one up.
- **Unit test OTel** - use `shared_span_exporter` fixture from `tests/unit/conftest.py`, never call `trace.set_tracer_provider()` in test files.

<!-- mulch:start -->
## Project Expertise (Mulch)
<!-- mulch-onboard-v:1 -->

This project uses [Mulch](https://github.com/jayminwest/mulch) for structured expertise management.

**At the start of every session**, run:
```bash
mulch prime
```

This injects project-specific conventions, patterns, decisions, and other learnings into your context.
Use `mulch prime --files src/foo.ts` to load only records relevant to specific files.

**Before completing your task**, review your work for insights worth preserving — conventions discovered,
patterns applied, failures encountered, or decisions made — and record them:
```bash
mulch record <domain> --type <convention|pattern|failure|decision|reference|guide> --description "..."
```

Link evidence when available: `--evidence-commit <sha>`, `--evidence-bead <id>`

Run `mulch status` to check domain health and entry counts.
Run `mulch --help` for full usage.
Mulch write commands use file locking and atomic writes — multiple agents can safely record to the same domain concurrently.

### Before You Finish

1. Discover what to record:
   ```bash
   mulch learn
   ```
2. Store insights from this work session:
   ```bash
   mulch record <domain> --type <convention|pattern|failure|decision|reference|guide> --description "..."
   ```
3. Validate and commit:
   ```bash
   mulch sync
   ```
<!-- mulch:end -->

<!-- seeds:start -->
## Issue Tracking (Seeds)
<!-- seeds-onboard-v:1 -->

This project uses [Seeds](https://github.com/jayminwest/seeds) for git-native issue tracking.

**At the start of every session**, run:
```
sd prime
```

This injects session context: rules, command reference, and workflows.

**Quick reference:**
- `sd ready` — Find unblocked work
- `sd create --title "..." --type task --priority 2` — Create issue
- `sd update <id> --status in_progress` — Claim work
- `sd close <id>` — Complete work
- `sd dep add <id> <depends-on>` — Add dependency between issues
- `sd sync` — Sync with git (run before pushing)

### Before You Finish
1. Close completed issues: `sd close <id>`
2. File issues for remaining work: `sd create --title "..."`
3. Sync and push: `sd sync && git push`
<!-- seeds:end -->

## gstack

Use the `/browse` skill from gstack for **all web browsing**. Never use `mcp__claude-in-chrome__*` tools.

### Available Skills

`/office-hours` `/plan-ceo-review` `/plan-eng-review` `/plan-design-review`
`/design-consultation` `/design-shotgun` `/review` `/ship` `/land-and-deploy`
`/canary` `/benchmark` `/browse` `/connect-chrome` `/qa` `/qa-only`
`/design-review` `/setup-browser-cookies` `/setup-deploy` `/retro`
`/investigate` `/document-release` `/codex` `/cso` `/autoplan` `/careful`
`/freeze` `/guard` `/unfreeze` `/gstack-upgrade`
