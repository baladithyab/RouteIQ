# AGENTS.md - Instructions for AI Coding Agents

This document provides comprehensive instructions for AI coding agents (Claude Code,
Cursor, Codex, etc.) working in the RouteIQ Gateway repository.

## OVERVIEW

**RouteIQ Gateway** is a production-grade, cloud-native General AI Gateway built on
[LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and
[LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML-based routing. It provides:

- **Unified AI Gateway**: OpenAI-compatible proxy for 100+ LLM providers
- **Intelligent Routing**: ML-based routing strategies (KNN, MLP, SVM, ELO, MF, hybrid, etc.)
- **MCP Gateway**: Model Context Protocol support (JSON-RPC, SSE, REST surfaces)
- **A2A Protocol**: Agent-to-Agent communication via LiteLLM's built-in endpoints
- **Skills Gateway**: Anthropic Computer Use, Bash, and Text Editor skill execution
- **Plugin System**: Extensible plugin architecture with lifecycle management
- **Policy Engine**: OPA-style pre-request policy evaluation (allow/deny rules)
- **MLOps Pipeline**: Scripts for telemetry collection, model training, and deployment
- **Enterprise Features**: RBAC, quotas, audit logging, observability, resilience, HA

> **Attribution**: RouteIQ is built on upstream LiteLLM and LLMRouter. Refer to this
> product as **RouteIQ**. Do not rename `LITELLM_*` environment variables. Do not claim
> RouteIQ implements features only inherited from upstream.

## REPOSITORY STRUCTURE

```
RouteIQ/
├── src/litellm_llmrouter/         # Main application code
│   ├── gateway/                   # Composition root & plugin system
│   │   ├── app.py                 # FastAPI app factory (create_app / create_standalone_app)
│   │   ├── plugin_manager.py      # Plugin lifecycle, dependency resolution
│   │   ├── plugin_adapters.py     # Plugin adapter implementations
│   │   ├── plugin_callback_bridge.py # Bridge between plugins and LiteLLM callbacks
│   │   ├── plugin_middleware.py   # Plugin middleware integration
│   │   ├── plugin_protocols.py    # Plugin protocol definitions (typing)
│   │   └── plugins/               # Built-in plugins (13 total)
│   │       ├── evaluator.py       # LLM-as-judge evaluation plugin
│   │       ├── skills_discovery.py # Anthropic skills (bash, computer, editor)
│   │       ├── upskill_evaluator.py # Combined skill + evaluation plugin
│   │       ├── bedrock_agentcore_mcp.py # Bedrock AgentCore MCP integration
│   │       ├── bedrock_guardrails.py # AWS Bedrock Guardrails plugin
│   │       ├── cache_plugin.py    # Caching plugin
│   │       ├── content_filter.py  # Content filtering plugin
│   │       ├── cost_tracker.py    # Cost tracking plugin
│   │       ├── guardrails_base.py # Base guardrails plugin class
│   │       ├── llamaguard_plugin.py # LlamaGuard safety plugin
│   │       ├── pii_guard.py       # PII detection/redaction plugin
│   │       └── prompt_injection_guard.py # Prompt injection detection plugin
│   ├── startup.py                 # Entry point: python -m litellm_llmrouter.startup
│   ├── routes/                    # FastAPI routers (package)
│   │   ├── __init__.py            # Re-exports all routers + feature flags
│   │   ├── health.py              # Health/readiness probes
│   │   ├── a2a.py                 # A2A agent routes
│   │   ├── mcp.py                 # MCP gateway routes (all surfaces)
│   │   ├── config.py              # Config/admin routes
│   │   └── models.py              # Model management routes
│   ├── strategies.py              # ML routing strategies (18+ algorithms)
│   ├── strategy_registry.py       # A/B testing, hot-swap, routing pipeline
│   ├── routing_strategy_patch.py  # Monkey-patch to LiteLLM's Router for ML strategies
│   ├── router_decision_callback.py # TG4.1 telemetry: router.* span attributes
│   ├── mcp_gateway.py             # MCP protocol: server registry, tool discovery
│   ├── mcp_jsonrpc.py             # MCP JSON-RPC 2.0 handler (for Claude Desktop)
│   ├── mcp_sse_transport.py       # MCP SSE transport for streaming
│   ├── mcp_parity.py              # Upstream-compatible /v1/mcp/* aliases
│   ├── mcp_tracing.py             # OpenTelemetry instrumentation for MCP
│   ├── a2a_gateway.py             # A2A agent registry (wraps LiteLLM's global_agent_registry)
│   ├── a2a_tracing.py             # OpenTelemetry instrumentation for A2A
│   ├── observability.py           # OpenTelemetry init (traces, metrics, logs)
│   ├── metrics.py                 # OTel metric helpers
│   ├── telemetry_contracts.py     # Versioned telemetry event schemas
│   ├── auth.py                    # Admin auth, RequestID middleware, secret scrubbing
│   ├── rbac.py                    # Role-based access control
│   ├── policy_engine.py           # OPA-style policy evaluation middleware
│   ├── quota.py                   # Per-team/per-key quota enforcement
│   ├── audit.py                   # Audit logging (file + structured events)
│   ├── resilience.py              # Backpressure middleware, drain manager, circuit breakers
│   ├── http_client_pool.py        # Shared httpx.AsyncClient pool for outbound requests
│   ├── hot_reload.py              # Filesystem-watching config hot-reload
│   ├── config_loader.py           # YAML config loading + S3/GCS download
│   ├── config_sync.py             # Background config sync (S3 ETag-based)
│   ├── model_artifacts.py         # ML model verification (hash, signature, manifest)
│   ├── url_security.py            # SSRF protection for external requests
│   ├── database.py                # Database connection helpers
│   ├── migrations.py              # Database migration utilities
│   ├── leader_election.py         # HA leader election (Redis-based)
│   ├── redis_pool.py              # Redis connection pool
│   ├── env_validation.py          # Startup env var validation (advisory only)
│   ├── conversation_affinity.py   # Conversation-based routing affinity
│   ├── management_classifier.py   # Classifies LiteLLM management endpoints
│   ├── management_middleware.py   # RBAC/audit middleware for management ops
│   ├── semantic_cache.py          # Semantic caching for LLM responses
│   ├── centroid_routing.py        # NadirClaw-inspired centroid-based routing (~2ms)
│   ├── custom_routing_strategy.py # LiteLLM CustomRoutingStrategyBase plugin adapter
│   └── __init__.py                # Public API exports
├── tests/
│   ├── conftest.py                # Root conftest: auto-skip integration if stack not running
│   ├── unit/                      # Unit tests (fast, no external deps)
│   │   ├── conftest.py            # Unit test fixtures
│   │   └── test_*.py              # ~30 unit test files
│   ├── integration/               # Integration tests (require Docker stack)
│   │   └── test_*.py              # ~12 integration test files
│   ├── property/                  # Property-based tests (hypothesis)
│   ├── perf/                      # Performance tests
│   └── test_*.py                  # Root-level test files (strategies, security, MCP)
├── config/
│   ├── config.yaml                # Main config (model_list, router_settings, general_settings)
│   ├── config.bedrock.yaml        # AWS Bedrock-specific config
│   ├── config.local-test.yaml     # Local test stack config
│   ├── config.otel-test.yaml      # OTel test config
│   ├── config.quota-test.yaml     # Quota testing config
│   ├── config.streaming-perf.yaml # Streaming perf test config
│   ├── llm_candidates.json        # LLM candidate list for ML routing
│   ├── nginx.conf                 # Nginx reverse proxy config (HA setup)
│   ├── otel-collector-config.yaml # OpenTelemetry Collector pipeline
│   └── policy.example.yaml        # Example policy engine rules
├── scripts/                       # Utility scripts (lint, test, validate, secrets)
├── examples/mlops/                # MLOps training pipeline
│   ├── scripts/                   # Training scripts (extract traces, train, deploy)
│   ├── configs/                   # Training configs (knn, mlp, svm, mf)
│   └── docker-compose.mlops.yml   # MLOps Docker stack
├── docker/
│   ├── Dockerfile                 # Production multi-stage build
│   ├── Dockerfile.local           # Local dev build
│   ├── entrypoint.sh              # Production entrypoint
│   └── entrypoint.local.sh        # Local dev entrypoint
├── docker-compose.yml             # Basic stack
├── docker-compose.ha.yml          # HA: multi-replica + Redis + Postgres + Nginx
├── docker-compose.otel.yml        # Observability: OTel Collector + Jaeger
├── docker-compose.ha-otel.yml     # HA + Observability combined
├── docker-compose.ha-test.yml     # HA integration testing
├── docker-compose.local-test.yml  # Local development testing
├── docker-compose.quota-test.yml  # Quota enforcement testing
├── docker-compose.streaming-perf.yml # Streaming performance testing
├── deploy/charts/                 # Helm charts for Kubernetes
├── docs/                          # Comprehensive documentation (~35 files)
├── plans/                         # Development planning (TG epics, roadmaps)
│   └── archive/                   # Archived plan files (completed TG epics)
├── models/                        # Trained ML models (empty .gitkeep placeholder)
│   └── centroids/                 # Pre-computed centroid vectors for zero-config routing
├── custom_routers/                # Custom routing strategies (empty .gitkeep placeholder)
├── reference/litellm/             # Upstream LiteLLM submodule (READ-ONLY)
├── pyproject.toml                 # Build config, deps, tool settings
├── lefthook.yml                   # Git hooks (pre-commit, pre-push, post-commit)
└── GATE*.md / TG*.md              # Quality gate validation reports
```

## DEVELOPMENT COMMANDS

### Package Management (uv - preferred)

```bash
uv sync                            # Install all dependencies
uv sync --extra dev                # Install with dev dependencies
uv add <package>                   # Add a dependency
uv run python -m <module>          # Run a module
uv run python -m litellm_llmrouter.startup --config config/config.yaml  # Start gateway
```

### Testing

```bash
uv run pytest tests/unit/ -x       # Run unit tests (stop on first failure)
uv run pytest tests/integration/   # Run integration tests (needs Docker stack)
uv run pytest tests/ -x -v         # Run all tests verbose
uv run pytest -k "test_name"       # Run specific test by name
uv run pytest tests/unit/test_file.py::TestClass::test_method -v  # Run exact test
uv run pytest tests/property/      # Run property-based tests (hypothesis)
```

**Integration tests** auto-skip if the local Docker stack is not running (port 4010).
Some integration tests manage their own compose stack and always run.

### Linting and Formatting

```bash
uv run ruff format src/ tests/     # Format with ruff
uv run ruff check src/ tests/      # Lint with ruff
uv run ruff check --fix src/ tests/ # Auto-fix lint issues
uv run mypy src/litellm_llmrouter/ --ignore-missing-imports  # Type checking
./scripts/lint.sh format           # Format via lint script
./scripts/lint.sh check            # Check via lint script
```

### Git Hooks (Lefthook)

```bash
./scripts/install_lefthook.sh      # Install lefthook
lefthook run pre-commit            # Run pre-commit hooks manually
lefthook run pre-push              # Run pre-push hooks manually
```

**Pre-commit hooks** (parallel): ruff format, ruff check, yamllint, detect-secrets,
private key detection, trailing whitespace fix, merge conflict check, large file check.

**Pre-push hooks** (sequential): unit tests, mypy type checking, security scan.

### Docker

```bash
docker compose up -d                                           # Basic stack
docker compose -f docker-compose.ha.yml up -d                  # HA stack
docker compose -f docker-compose.otel.yml up -d                # Observability stack
docker compose -f docker-compose.ha-otel.yml up -d             # HA + Observability
docker compose -f docker-compose.local-test.yml up -d          # Local test stack (port 4010)
docker build -f docker/Dockerfile -t litellm-llmrouter:latest . # Build production image
```

### Remote Execution (rr - Road Runner)

When local `git push` is blocked by Code Defender:

```bash
rr push                            # Sync code and push to GitHub
rr push-force                      # Force push (--force-with-lease)
rr test                            # Run unit tests on remote
rr ci                              # Run full CI pipeline on remote
```

**Always sync local after `rr push`**: `git pull` or `git fetch origin && git reset --hard origin/main`

## CODING GUIDELINES

### Code Style

- **Python 3.14+** required (target in pyproject.toml)
- **Ruff** for linting and formatting (line-length: 88)
- **Type hints** required for all public APIs
- **Pydantic v2** for data validation
- **Async/await** patterns throughout FastAPI routes
- **No side effects on import** - patches applied explicitly via `patch_litellm_router()`

### File Patterns

- Source files: `src/litellm_llmrouter/*.py`
- Gateway subsystem: `src/litellm_llmrouter/gateway/*.py`
- Plugins: `src/litellm_llmrouter/gateway/plugins/*.py`
- Unit tests: `tests/unit/test_*.py`
- Integration tests: `tests/integration/test_*.py`
- Property tests: `tests/property/test_*.py`
- Config files: `config/*.yaml`

### Import Order

```python
# Standard library
import asyncio
from typing import Optional, Dict, Any

# Third-party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# Local
from litellm_llmrouter.observability import setup_telemetry
from litellm_llmrouter.auth import verify_token
```

### Error Handling

- Use `fastapi.HTTPException` with meaningful status codes and messages
- Admin auth uses fail-closed pattern: no keys configured = deny all
- Policy engine supports fail-open (default) or fail-closed modes
- Log errors with secret scrubbing via `auth._scrub_secrets()`
- Handle both sync and async errors consistently
- Use `RequestIDMiddleware` for request correlation in error responses

### Testing Conventions

- Tests must be async where testing async code (`asyncio_mode = "auto"`)
- Use `pytest-asyncio` with `@pytest.mark.asyncio` (or rely on auto mode)
- Use `hypothesis` for property-based testing
- Mock external services (LLM providers, databases, S3, Redis)
- Unit tests in `tests/unit/` must not require external services
- Integration tests auto-skip when Docker stack is not running
- Use `conftest.py` fixtures; unit tests have their own `conftest.py`

## KEY PATTERNS

### 1. Application Startup (Composition Root)

The gateway uses a factory pattern in `gateway/app.py`:

```python
from litellm_llmrouter.gateway import create_app

# In-process with LiteLLM proxy (production)
app = create_app()

# Standalone without LiteLLM (testing)
app = create_standalone_app()
```

**Load order**: Patch LiteLLM Router -> Get FastAPI app -> Add backpressure
middleware (innermost, registered first) -> Configure middleware (CORS,
RequestID, Policy, Management, Plugin, RouterDecision) -> Load plugins ->
Register routes -> Setup lifecycle hooks

The entry point is `startup.py`:
```bash
uv run python -m litellm_llmrouter.startup --config config/config.yaml --port 4000
```

### 2. Routing Strategies

18+ ML strategies registered via the strategy registry. Strategies are monkey-patched
into LiteLLM's Router class via `routing_strategy_patch.py`:

```python
from litellm_llmrouter import patch_litellm_router, register_llmrouter_strategies

patch_litellm_router()          # Must be called BEFORE creating Router instances
register_llmrouter_strategies() # Register all llmrouter-* strategies
```

**A/B testing** via the routing pipeline:
```python
from litellm_llmrouter import get_routing_registry

registry = get_routing_registry()
registry.set_weights({"baseline": 90, "candidate": 10})
```

**Strategy families**: KNN, MLP, SVM, ELO, MF (matrix factorization), hybrid, custom.
KNN uses sentence-transformers for embedding-based similarity routing.

### 3. MCP Gateway (Multiple Surfaces)

MCP is exposed through several protocol surfaces:

| Surface | Endpoint | Use Case |
|---------|----------|----------|
| JSON-RPC | `POST /mcp` | Native MCP clients (Claude Desktop, IDEs) |
| SSE | `/mcp/sse` | Real-time streaming events |
| REST | `/mcp-rest/*` | RESTful access to MCP operations |
| Parity | `/v1/mcp/*` | Upstream LiteLLM-compatible aliases |
| Proxy | `/mcp-proxy/*` | Protocol-level MCP server proxy (admin) |

**Feature flags**: `MCP_GATEWAY_ENABLED`, `MCP_SSE_TRANSPORT_ENABLED`,
`MCP_SSE_LEGACY_MODE`, `MCP_PROTOCOL_PROXY_ENABLED`, `MCP_OAUTH_ENABLED`

### 4. Plugin System

Plugins extend gateway functionality with lifecycle management:

```python
from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin

class MyPlugin(GatewayPlugin):
    async def startup(self, app):
        ...
    async def shutdown(self, app):
        ...
```

Built-in plugins (13 total): `evaluator` (LLM-as-judge), `skills_discovery` (Anthropic
skills), `upskill_evaluator` (combined), `bedrock_agentcore_mcp`, `bedrock_guardrails`,
`cache_plugin`, `content_filter`, `cost_tracker`, `guardrails_base`, `llamaguard_plugin`,
`pii_guard`, `prompt_injection_guard`. Plugins are loaded from config before routes and
started during app lifespan.

### 5. Observability

All operations should be traced via OpenTelemetry:

```python
from litellm_llmrouter.observability import init_observability, set_router_decision_attributes

# Init at startup
init_observability(service_name="litellm-gateway", enable_traces=True)

# Emit routing decision telemetry (TG4.1)
set_router_decision_attributes(strategy="llmrouter-knn", model="claude-3-opus")
```

**Telemetry contracts** in `telemetry_contracts.py` define versioned event schemas
for structured telemetry emission.

### 6. Policy Engine

OPA-style policy evaluation at the ASGI middleware layer:

```python
# Configured via POLICY_ENGINE_ENABLED=true and POLICY_CONFIG_PATH
# Evaluates: subject, route, model, headers, source IP
# Supports: allow/deny rules, fail-open/fail-closed modes
```

### 7. Resilience

- **Backpressure middleware**: Limits concurrent requests
- **Drain manager**: Graceful shutdown with request draining
- **Circuit breakers**: Per-provider circuit breaker pattern
- **HTTP client pool**: Shared connection pool via `httpx.AsyncClient`

### 8. Configuration

```python
from litellm_llmrouter.config_loader import load_config

# Load from file
config = load_config("config/config.yaml")

# Config hot-reload: watches filesystem for changes
# Config sync: pulls from S3/GCS with ETag-based change detection
```

## IMPORTANT CONSTRAINTS

### 1. Reference Directory is READ-ONLY

The `reference/litellm/` directory is a git submodule containing upstream LiteLLM.
**DO NOT MODIFY** files in this directory. It exists for reference only.

### 2. Security Requirements

- **No real secrets** in code or tests (use placeholders like `test-api-key`)
- **SSRF protection** via `url_security.py` for all external requests
- **Input validation** for all API endpoints
- **Pickle loading disabled by default** - set `LLMROUTER_ALLOW_PICKLE_MODELS=true`
  only in trusted environments. Use `LLMROUTER_ENFORCE_SIGNED_MODELS=true` for manifest verification.
- **Secret scrubbing** in error logs via `auth._scrub_secrets()`
- **Admin auth fail-closed** - no keys configured = deny all control-plane requests

### 3. Test Requirements

- All new features require unit tests
- Integration tests for API endpoints
- Security-sensitive code requires security tests
- Pre-push hooks run unit tests and type checking

### 4. Monkey-Patch Constraint

LiteLLM's Router is patched at runtime via `routing_strategy_patch.py`. This means:
- **When using legacy monkey-patch mode** (`ROUTEIQ_USE_PLUGIN_STRATEGY=false`), only 1 uvicorn worker is supported. When using the plugin strategy (default), multiple workers can be configured via `ROUTEIQ_WORKERS`.
- **Call `patch_litellm_router()` BEFORE creating Router instances**
- The `create_app()` factory handles this automatically

### 5. Branding & Attribution

- Refer to this product as **RouteIQ**
- Refer to upstream as **LiteLLM** or **LLMRouter**
- Do not rename `LITELLM_*` environment variables
- Do not claim RouteIQ implements features only inherited from upstream

## COMMON TASKS

### Adding a New Endpoint

1. Add route in the appropriate module under `src/litellm_llmrouter/routes/` (choose correct file: `health.py`, `a2a.py`, `mcp.py`, `config.py`, or `models.py`)
2. Add Pydantic models for request/response
3. Add appropriate auth dependency (`admin_api_key_auth` for control-plane, `user_api_key_auth` for data-plane)
4. Register router in `gateway/app.py` `_register_routes()` if using a new router
5. Add unit test in `tests/unit/test_*.py`
6. Add integration test if endpoint requires external services

### Adding a New Routing Strategy

1. Implement strategy class in `src/litellm_llmrouter/strategies.py`
2. Add to `LLMROUTER_STRATEGIES` dict and `LLMRouterStrategyFamily` enum
3. Strategy is auto-registered via `register_llmrouter_strategies()`
4. Add configuration support in `config/config.yaml`
5. Add unit tests in `tests/unit/`
6. Update `docs/routing-strategies.md`

### Adding a New Plugin

1. Create plugin file in `src/litellm_llmrouter/gateway/plugins/`
2. Extend `GatewayPlugin` base class
3. Define `metadata` with capabilities, priority, dependencies
4. Implement `startup(app)` and `shutdown(app)` hooks
5. Register in plugin configuration
6. Add unit tests in `tests/unit/`

### Debugging Test Failures

```bash
uv run pytest tests/unit/test_file.py -v -s     # Verbose with stdout
uv run pytest tests/unit/test_file.py --trace    # With debugger
uv run pytest tests/unit/test_file.py -x --tb=long  # Stop on first failure, full traceback
```

## WORKFLOW

### Task Group (TG) Workflow

Development follows a Task Group pattern with quality gates:

1. **Create Feature Branch**: `git checkout -b tg<id>-<short-desc>`
2. **Local Development**: Commit freely, run tests locally
3. **Squash Merge to Main**: `git checkout main && git merge --squash tg<id>-branch`
4. **Single Commit**: `git commit -m "feat: complete TG<id> description"`
5. **Push via `rr push`** if local push is blocked

### Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `LITELLM_MASTER_KEY` | Yes | Master API key for admin access |
| `LITELLM_CONFIG_PATH` | No | Default config path |
| `DATABASE_URL` | No | PostgreSQL connection string |
| `REDIS_HOST` / `REDIS_PORT` | No | Redis for caching |
| `OTEL_ENABLED` | No | Enable OpenTelemetry (default: true) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | OTLP collector endpoint |
| `OTEL_SERVICE_NAME` | No | Service name (default: litellm-gateway) |
| `A2A_GATEWAY_ENABLED` | No | Enable A2A gateway (default: false) |
| `MCP_GATEWAY_ENABLED` | No | Enable MCP gateway (default: false) |
| `MCP_SSE_TRANSPORT_ENABLED` | No | Enable MCP SSE transport |
| `MCP_PROTOCOL_PROXY_ENABLED` | No | Enable MCP protocol proxy (admin) |
| `CONFIG_HOT_RELOAD` | No | Enable config hot-reload (default: false) |
| `CONFIG_S3_BUCKET` / `CONFIG_S3_KEY` | No | S3 config sync |
| `POLICY_ENGINE_ENABLED` | No | Enable policy engine (default: false) |
| `POLICY_CONFIG_PATH` | No | Policy YAML config path |
| `ADMIN_API_KEYS` | No | Comma-separated admin API keys |
| `LLMROUTER_ALLOW_PICKLE_MODELS` | No | Allow pickle model loading (default: false) |
| `LLMROUTER_ENFORCE_SIGNED_MODELS` | No | Require manifest verification |
| `LLMROUTER_ROUTER_CALLBACK_ENABLED` | No | Router decision telemetry (default: true) |
| `ROUTEIQ_CORS_ORIGINS` | No | Allowed CORS origins |
| `ROUTEIQ_SKIP_ENV_VALIDATION` | No | Skip startup env validation |
| `ROUTEIQ_EVALUATOR_ENABLED` | No | Enable LLM-as-judge evaluator plugin |
| `ROUTEIQ_USE_PLUGIN_STRATEGY` | No | Use plugin routing strategy instead of monkey-patch (default: true) |
| `ROUTEIQ_WORKERS` | No | Number of uvicorn workers (default: 1, multi-worker requires plugin strategy) |
| `ROUTEIQ_CENTROID_ROUTING` | No | Enable centroid routing fallback (default: true) |
| `ROUTEIQ_ROUTING_PROFILE` | No | Default routing profile: auto/eco/premium/free/reasoning (default: auto) |
| `ROUTEIQ_CENTROID_WARMUP` | No | Pre-warm centroid classifier at startup (default: false) |

## DOCUMENTATION

| Document | Purpose |
|----------|---------|
| [`README.md`](README.md) | Project overview and quick start |
| [`docs/index.md`](docs/index.md) | Comprehensive getting started guide |
| [`docs/deployment.md`](docs/deployment.md) | Docker, K8s deployment guide |
| [`docs/configuration.md`](docs/configuration.md) | Configuration options |
| [`docs/routing-strategies.md`](docs/routing-strategies.md) | ML routing strategies |
| [`docs/mcp-gateway.md`](docs/mcp-gateway.md) | MCP Protocol guide |
| [`docs/a2a-gateway.md`](docs/a2a-gateway.md) | A2A Protocol guide |
| [`docs/security.md`](docs/security.md) | Security considerations |
| [`docs/observability.md`](docs/observability.md) | OpenTelemetry setup |
| [`docs/plugins.md`](docs/plugins.md) | Plugin system guide |
| [`docs/skills-gateway.md`](docs/skills-gateway.md) | Skills gateway guide |
| [`docs/mlops-training.md`](docs/mlops-training.md) | MLOps training loop |
| [`docs/rr-workflow.md`](docs/rr-workflow.md) | Remote push workflow |
| [`docs/aws-production-guide.md`](docs/aws-production-guide.md) | AWS production deployment guide |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guidelines |

## NON-OBVIOUS BEHAVIORS & GOTCHAS

1. **In-process uvicorn is mandatory**: `startup.py` uses `uvicorn.run(app=app)` instead
   of `os.execvp()`. This is critical because `os.execvp()` replaces the process and would
   lose all monkey-patches to LiteLLM's Router class. When using the plugin strategy
   (default), multiple workers are supported via `ROUTEIQ_WORKERS` since `os.fork()`
   preserves the app state including installed strategies. Legacy monkey-patch mode
   is restricted to 1 worker.
2. **BackpressureMiddleware is the innermost middleware**: It is registered first via
   `add_backpressure_middleware()` before `_configure_middleware()`, wrapping the ASGI app
   directly (replaces `app.app`). This is required because `BaseHTTPMiddleware` does NOT
   properly handle streaming responses.

3. **Plugin hooks on `app.state`, not lifespan**: LiteLLM manages its own lifespan, so plugin
   startup/shutdown are stored as lambdas on `app.state` and called by `startup.py`.

4. **Readiness returns 200 for degraded state**: When circuit breakers are open, `/_health/ready`
   returns `status: "degraded"` with HTTP 200, not 503.

5. **Two separate MCP surfaces**: `/llmrouter/mcp/*` (REST for LLMRouter's MCP gateway) and
   `/mcp` (native JSON-RPC for Claude Desktop) are distinct systems.

6. **SSRF validation happens twice**: At registration time (no DNS) and at invocation time
   (with DNS resolution) to catch DNS rebinding attacks.

7. **MCP tool invocation disabled by default**: Even with `MCP_GATEWAY_ENABLED=true`, tool
   invocation requires `LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true`. Server registration and
   discovery work without it.

8. **A2A `/a2a/agents` wraps LiteLLM's registry**: These are thin wrappers around LiteLLM's
   `global_agent_registry`, NOT the custom `A2AGateway`. The gateway is a separate system.

9. **Config sync only runs on leader**: In HA mode, non-leader replicas skip config sync
   entirely.

10. **Singletons everywhere with `reset_*()` for testing**: Every subsystem uses module-level
    singletons with `get_*()` accessors and `reset_*()` functions. Tests MUST use `autouse=True`
    fixtures calling `reset_*()` to avoid cross-test contamination.

11. **OTel provider reuse**: `ObservabilityManager` detects if LiteLLM already set up a
    `TracerProvider` and reuses it to avoid duplicate spans.

12. **SSE uses async queues**: POST to `/mcp/messages` pushes to an asyncio.Queue and returns
    202 immediately. The response is emitted on the SSE stream.

## WHEN IN DOUBT

1. Follow existing patterns in the codebase
2. Check similar implementations in `reference/litellm/` (read-only reference)
3. Run tests before committing: `uv run pytest tests/unit/ -x`
4. Ensure comprehensive test coverage
5. Keep security in mind for all changes
6. Use `uv` for all Python operations

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
