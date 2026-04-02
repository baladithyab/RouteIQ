.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help setup dev test test-all lint fix typecheck docker-up docker-down docker-build docker-build-slim clean validate-config

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup: install deps + git hooks
	uv sync --extra dev
	./scripts/install_lefthook.sh 2>/dev/null || true
	@echo "Setup complete. Run 'make dev' to start."

dev: ## Start local dev environment
	docker compose -f docker-compose.local-test.yml up -d 2>/dev/null || true
	uv run python -m litellm_llmrouter.startup --config config/config.local-test.yaml --port 4000

test: ## Run unit tests
	uv run pytest tests/unit/ -x -v --tb=short

test-all: ## Run all tests (unit + integration)
	uv run pytest tests/ -x -v

lint: ## Run linters (ruff check + format check)
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fix: ## Auto-fix lint issues
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

typecheck: ## Run mypy type checking
	uv run mypy src/litellm_llmrouter/ --ignore-missing-imports

docker-up: ## Start Docker test stack
	docker compose -f docker-compose.local-test.yml up -d

docker-down: ## Stop Docker test stack
	docker compose -f docker-compose.local-test.yml down

docker-build: ## Build full Docker image
	docker build -f docker/Dockerfile -t routeiq:latest .

docker-build-slim: ## Build slim Docker image (~500MB, no ML deps)
	docker build -f docker/Dockerfile.slim -t routeiq:slim .

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

validate-config: ## Validate config files
	uv run python -c "from litellm_llmrouter.settings import GatewaySettings; s = GatewaySettings(); print(f'Settings loaded: {len(s.model_fields)} fields')"
