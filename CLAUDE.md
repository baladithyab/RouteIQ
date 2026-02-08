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
docker compose up -d                                           # Basic
docker compose -f docker-compose.ha.yml up -d                  # HA (Redis/Postgres/Nginx)
docker compose -f docker-compose.otel.yml up -d                # Observability (OTel/Jaeger)
docker compose -f docker-compose.local-test.yml up -d          # Local test stack
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

### Core Entry Points

- **`startup.py`** - CLI entry point: `python -m litellm_llmrouter.startup`
- **`gateway/app.py`** - App factory: `create_app()` (with LiteLLM) / `create_standalone_app()` (testing)
- **`routes/`** - FastAPI routers split into focused modules (health, a2a, mcp, config)

### Startup Load Order

1. Apply LiteLLM Router monkey-patch (`routing_strategy_patch.py`)
2. Get/create FastAPI app
3. Add middleware: RequestID -> Policy -> RouterDecision
4. Load plugins (discovery + validation)
5. Register routes (health, llmrouter, admin, MCP surfaces)
6. Setup lifecycle hooks (plugins, HTTP pool, drain)
7. Add backpressure middleware

### Source Layout (`src/litellm_llmrouter/`)

| Module | Purpose |
|--------|---------|
| `gateway/app.py` | FastAPI app factory (composition root) |
| `gateway/plugin_manager.py` | Plugin lifecycle with dependency resolution |
| `gateway/plugins/` | Built-in plugins (evaluator, skills, upskill, guardrails) |
| `startup.py` | CLI entry point, initialization orchestration |
| `routes/` | API routers split into: `health.py`, `a2a.py`, `mcp.py`, `config.py`, `models.py` |
| `strategies.py` | 18+ ML routing strategies (KNN, MLP, SVM, ELO, MF, hybrid) |
| `strategy_registry.py` | A/B testing, hot-swap, routing pipeline |
| `routing_strategy_patch.py` | Monkey-patch for LiteLLM Router integration |
| `router_decision_callback.py` | Routing decision telemetry (TG4.1) |
| `mcp_gateway.py` | MCP server registry and tool discovery |
| `mcp_jsonrpc.py` | Native MCP JSON-RPC 2.0 (for Claude Desktop) |
| `mcp_sse_transport.py` | MCP SSE streaming transport |
| `mcp_parity.py` | Upstream-compatible `/v1/mcp/*` aliases |
| `mcp_tracing.py` | OTel instrumentation for MCP |
| `a2a_gateway.py` | A2A agent registry (wraps LiteLLM) |
| `a2a_tracing.py` | OTel instrumentation for A2A |
| `observability.py` | OpenTelemetry init (traces, metrics, logs) |
| `telemetry_contracts.py` | Versioned telemetry event schemas |
| `auth.py` | Admin auth, RequestID middleware (raw ASGI), secret scrubbing |
| `rbac.py` | Role-based access control |
| `policy_engine.py` | OPA-style policy evaluation middleware |
| `quota.py` | Per-team/per-key quota enforcement |
| `audit.py` | Audit logging |
| `resilience.py` | Backpressure, drain manager, circuit breakers |
| `http_client_pool.py` | Shared httpx.AsyncClient pool |
| `hot_reload.py` | Filesystem-watching config hot-reload |
| `config_loader.py` | YAML config + S3/GCS download |
| `config_sync.py` | Background config sync (S3 ETag-based) |
| `model_artifacts.py` | ML model verification (hash, signature) |
| `url_security.py` | SSRF protection |
| `leader_election.py` | HA leader election (Redis-based) |

## Key Patterns

### Routing Strategy Integration

LiteLLM's Router is monkey-patched at runtime to support `llmrouter-*` strategies.
Critical constraint: **always run 1 uvicorn worker** (patches don't survive `os.execvp()`).
`patch_litellm_router()` must be called BEFORE creating Router instances.
`create_app()` handles this automatically.

### MCP Multiple Surfaces

MCP is exposed through 5 surfaces: JSON-RPC (`/mcp`), SSE (`/mcp/sse`),
REST (`/mcp-rest/*`), parity (`/v1/mcp/*`), and proxy (`/mcp-proxy/*`).
Each is feature-flagged via environment variables.

### Plugin System

Plugins extend the gateway via `GatewayPlugin` base class. They are loaded from
config BEFORE routes (deterministic ordering) and started during app lifespan.
Built-in: evaluator, skills_discovery, upskill_evaluator.

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

- Python 3.14+ (target version in pyproject.toml)
- Ruff formatter + linter (line-length: 88)
- Type hints required for public APIs
- Pydantic v2 for data validation
- Async/await throughout FastAPI routes
- No side effects on import

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

1. Add route to `routes.py` (pick the right router: `health_router`, `admin_router`, `llmrouter_router`)
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

- **In-process uvicorn is mandatory** - `startup.py` runs LiteLLM in-process (not via `os.execvp()`) to preserve monkey-patches. Always use 1 worker.
- **BackpressureMiddleware wraps ASGI directly** (replaces `app.app`), not via `add_middleware()`, because `BaseHTTPMiddleware` breaks streaming.
- **Plugin hooks live on `app.state`** as lambdas, not in lifespan, because LiteLLM manages its own lifespan.
- **`/_health/ready` returns 200 for degraded state** (circuit breakers open), not 503.
- **Two MCP systems exist**: `/llmrouter/mcp/*` (REST gateway) and `/mcp` (JSON-RPC for Claude Desktop) are separate.
- **SSRF checks happen twice**: at registration (no DNS) and invocation (with DNS) to catch rebinding.
- **MCP tool invocation is off by default** even when `MCP_GATEWAY_ENABLED=true`. Needs `LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true`.
- **Config sync only runs on HA leader** - non-leader replicas skip it.
- **Singletons need `reset_*()`** - every subsystem uses singletons. Tests MUST call `reset_*()` in `autouse=True` fixtures.
- **OTel provider reuse** - `ObservabilityManager` reuses existing TracerProvider if LiteLLM set one up.
- **Unit test OTel** - use `shared_span_exporter` fixture from `tests/unit/conftest.py`, never call `trace.set_tracer_provider()` in test files.
