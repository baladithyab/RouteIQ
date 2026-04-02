# Contributing

Thank you for your interest in contributing to RouteIQ Gateway!

## Development Setup

### Prerequisites

- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package manager
- **Docker** and **Docker Compose**
- **Git**

### Local Development

```bash
# Clone the repository
git clone https://github.com/baladithyab/RouteIQ.git
cd routeiq

# Install dependencies
uv sync --extra dev

# Install git hooks
./scripts/install_lefthook.sh

# Start local dev stack
docker compose -f docker-compose.local-test.yml up -d

# Run the gateway
uv run python -m litellm_llmrouter.startup \
  --config config/config.local-test.yaml --port 4000
```

## Running Tests

```bash
# Unit tests
uv run pytest tests/unit/ -x -v

# All tests (unit + integration)
uv run pytest tests/ -x -v

# Specific test
uv run pytest tests/unit/test_file.py::TestClass::test_method -v

# Property-based tests
uv run pytest tests/property/
```

!!! note
    Integration tests auto-skip if the Docker stack is not running.

## Code Quality

```bash
# Format
uv run ruff format src/ tests/

# Lint
uv run ruff check src/ tests/

# Auto-fix
uv run ruff check --fix src/ tests/

# Type check
uv run mypy src/litellm_llmrouter/ --ignore-missing-imports
```

## Git Hooks

Pre-commit hooks run automatically:

- Ruff format and lint
- YAML linting
- Secret detection
- Trailing whitespace fix
- Large file check (>1MB)

Pre-push hooks:

- Unit tests
- Type checking
- Security scan

## Project Structure

```
src/litellm_llmrouter/         # Main application code
  gateway/                     # Composition root & plugin system
    plugins/                   # Built-in plugins (13 total)
  routes/                      # FastAPI routers
  strategies.py                # ML routing strategies
tests/
  unit/                        # Fast, no external deps
  integration/                 # Require Docker stack
  property/                    # Hypothesis property tests
config/                        # Configuration files
docs/                          # Documentation
```

## Coding Guidelines

- **Ruff** for formatting (line-length: 88)
- **Type hints** required for all public APIs
- **Async/await** patterns throughout FastAPI routes
- **No side effects on import**
- Use `get_settings()` from `settings.py` for configuration

## Adding Features

### New Endpoint

1. Add route in `src/litellm_llmrouter/routes/`
2. Add Pydantic request/response models
3. Add auth dependency
4. Register router in `gateway/app.py` if new
5. Add unit test

### New Routing Strategy

1. Implement in `src/litellm_llmrouter/strategies.py`
2. Add to `LLMROUTER_STRATEGIES` dict
3. Add configuration support
4. Add unit tests

### New Plugin

1. Create in `src/litellm_llmrouter/gateway/plugins/`
2. Extend `GatewayPlugin` base class
3. Define `metadata` with capabilities and priority
4. Implement `startup(app)` and `shutdown(app)`
5. Add unit tests
