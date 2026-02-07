# Contributing to RouteIQ Gateway

Thank you for your interest in contributing! This document provides guidelines
for development on the RouteIQ Gateway project.

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm)
> for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter)
> for ML routing.

## Development Setup

### Prerequisites

- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager (preferred over pip)
- **Docker** and **Docker Compose**
- **Git**
- **[Lefthook](https://github.com/evilmartians/lefthook)** - Git hooks manager (installed via script)

### Local Development

```bash
# Clone the repository
git clone https://github.com/baladithyab/RouteIQ.git
cd RouteIQ

# Install dependencies with uv
uv sync

# Install git hooks
./scripts/install_lefthook.sh

# Start local development stack (Postgres, Redis, Jaeger, MinIO, etc.)
docker compose -f docker-compose.local-test.yml up -d

# Run the gateway
uv run python -m litellm_llmrouter.startup --config config/config.local-test.yaml --port 4000
```

### Git Hooks (Lefthook)

This project uses [Lefthook](https://github.com/evilmartians/lefthook) for git hooks.
Lefthook runs fast, parallel checks on staged files before each commit.

```bash
# Install Lefthook and configure git hooks
./scripts/install_lefthook.sh

# Manual hook runs
lefthook run pre-commit    # Lint, format, secret detection
lefthook run pre-push      # Unit tests, type checking, security scan
```

**Pre-commit hooks** (parallel):
- Ruff format and lint on Python files (auto-fix staged)
- YAML linting
- Secret detection (API keys, passwords, tokens)
- Private key detection
- Trailing whitespace fix
- Merge conflict marker check
- Large file check (>1MB)

**Pre-push hooks** (sequential):
- Unit tests (`uv run pytest tests/unit/ -x`)
- Type checking (`uv run mypy src/litellm_llmrouter/`)
- Full repo security scan

### Building the Container

```bash
# Production build (multi-stage, optimized)
docker build -f docker/Dockerfile -t routeiq-gateway:latest .

# Local development build
docker build -f docker/Dockerfile.local -t routeiq-gateway:local .
```

## Project Structure

```
RouteIQ/
├── src/litellm_llmrouter/   # Main application code
│   ├── gateway/              # App factory, plugin system
│   │   ├── app.py            # Composition root (create_app)
│   │   ├── plugin_manager.py # Plugin lifecycle management
│   │   └── plugins/          # Built-in plugins
│   ├── startup.py            # CLI entry point
│   ├── routes.py             # FastAPI routes
│   ├── strategies.py         # ML routing strategies (18+)
│   ├── strategy_registry.py  # A/B testing, routing pipeline
│   ├── mcp_gateway.py        # MCP protocol gateway
│   ├── a2a_gateway.py        # Agent-to-Agent gateway
│   ├── observability.py      # OpenTelemetry integration
│   ├── auth.py               # Authentication & request ID
│   ├── policy_engine.py      # OPA-style policy engine
│   ├── resilience.py         # Backpressure, circuit breakers
│   └── ...                   # See AGENTS.md for full listing
├── tests/
│   ├── unit/                 # Unit tests (fast, no external deps)
│   ├── integration/          # Integration tests (require Docker stack)
│   ├── property/             # Property-based tests (hypothesis)
│   └── perf/                 # Performance tests
├── config/                   # YAML configs and infrastructure configs
├── docker/                   # Dockerfiles and entrypoints
├── deploy/charts/            # Helm charts for Kubernetes
├── docs/                     # Documentation
├── examples/mlops/           # MLOps training pipeline
├── plans/                    # Development planning (TG epics)
├── scripts/                  # Utility scripts
├── models/                   # Trained ML models (deployed at runtime)
├── custom_routers/           # Custom routing strategies
└── reference/litellm/        # Upstream LiteLLM submodule (READ-ONLY)
```

## Making Changes

### Code Style

- **Python 3.14+** with type hints for all public APIs
- **Ruff** for linting and formatting (line-length: 88)
- **Pydantic v2** for data validation
- **Async/await** patterns throughout FastAPI routes
- No side effects on import (patches applied explicitly)

### Running Tests

```bash
# Unit tests (fast, recommended during development)
uv run pytest tests/unit/ -x -v

# All tests (excluding integration)
uv run pytest tests/ --ignore=tests/integration -x -v

# Specific test file
uv run pytest tests/unit/test_plugin_manager.py -v

# Specific test
uv run pytest tests/unit/test_plugin_manager.py::TestPluginStartup::test_startup_calls_plugins -v

# Property-based tests
uv run pytest tests/property/ -v

# With coverage
uv run pytest tests/unit/ --cov=src/litellm_llmrouter
```

**Integration tests** require the Docker stack running on port 4010:
```bash
docker compose -f docker-compose.local-test.yml up -d
uv run pytest tests/integration/ -v
```

### Linting and Type Checking

```bash
uv run ruff format src/ tests/                              # Format
uv run ruff check src/ tests/                               # Lint
uv run ruff check --fix src/ tests/                         # Auto-fix
uv run mypy src/litellm_llmrouter/ --ignore-missing-imports # Type check
```

### Branding & Attribution

We maintain strict attribution to upstream projects.

**Rules:**
- Refer to this product as **RouteIQ**
- Refer to upstream components as **LiteLLM** or **LLMRouter**
- **Do not** rename environment variables (keep `LITELLM_*`, e.g., `LITELLM_MASTER_KEY`)
- **Do not** claim RouteIQ implements features only inherited from upstream

**Verify attribution:**
```bash
git grep -n "LiteLLM" docs plans README.md
```

## Development Workflow

### Task Group (TG) Pattern

Development follows a Task Group pattern with quality gates:

1. **Create feature branch**: `git checkout -b tg<id>-<short-desc>`
2. **Develop locally**: Commit freely, run tests
3. **Squash merge to main**: `git checkout main && git merge --squash tg<id>-branch`
4. **Commit**: `git commit -m "feat: complete TG<id> description"`
5. **Push**: `git push` (or `rr push` if blocked by Code Defender)

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes with tests
4. Ensure pre-commit and pre-push hooks pass
5. Push to your fork
6. Open a Pull Request

### PR Checklist

- [ ] Unit tests pass (`uv run pytest tests/unit/ -x`)
- [ ] Linting passes (`uv run ruff check src/ tests/`)
- [ ] Type checking passes (`uv run mypy src/litellm_llmrouter/`)
- [ ] New features have unit tests
- [ ] Documentation updated for user-facing changes
- [ ] No sensitive data committed (secrets, API keys)
- [ ] Attribution rules followed
- [ ] Commit messages are clear and descriptive

## Adding New Features

### New Routing Strategy

1. Implement strategy in `src/litellm_llmrouter/strategies.py`
2. Add to `LLMROUTER_STRATEGIES` dict and `LLMRouterStrategyFamily` enum
3. Strategy is auto-registered via `register_llmrouter_strategies()`
4. Add unit tests in `tests/unit/`
5. Update `docs/routing-strategies.md`

### New Gateway Plugin

1. Create plugin file in `src/litellm_llmrouter/gateway/plugins/`
2. Extend `GatewayPlugin` base class
3. Define `metadata` with capabilities, priority, and dependencies
4. Implement `startup(app, context)` and `shutdown(app, context)` hooks
5. Add unit tests in `tests/unit/`
6. Update `docs/plugins.md`

### New API Endpoint

1. Add route in `src/litellm_llmrouter/routes.py` (choose the right router)
2. Add Pydantic models for request/response
3. Add auth dependency (`admin_api_key_auth` or `user_api_key_auth`)
4. Register router in `gateway/app.py` if using a new router
5. Add unit test in `tests/unit/`
6. Update `docs/api-reference.md`

## Docker Compose Variants

| File | Purpose | Use When |
|------|---------|----------|
| `docker-compose.yml` | Basic single instance | Quick local testing |
| `docker-compose.ha.yml` | HA with Redis/Postgres/Nginx | Testing production-like setup |
| `docker-compose.otel.yml` | Observability with Jaeger | Debugging traces |
| `docker-compose.local-test.yml` | Full local dev stack | Running integration tests |
| `docker-compose.ha-otel.yml` | HA + Observability | Full production simulation |

## Security

- No real secrets in code or tests (use `test-api-key` placeholders)
- SSRF protection is deny-by-default for private IPs
- Pickle model loading is disabled by default (security risk)
- Admin auth fails closed (no keys = deny all)
- Pre-commit hooks scan for secrets automatically

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- See [AGENTS.md](AGENTS.md) for comprehensive AI agent instructions
- See [CLAUDE.md](CLAUDE.md) for Claude Code-specific guidance
