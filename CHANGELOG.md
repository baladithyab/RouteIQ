# Changelog

All notable changes to RouteIQ Gateway will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0-rc1] — 2026-04-02

### Fixed
- Python 2 except syntax in policy_engine.py (tuple form required)
- CORS wildcard + credentials security vulnerability (wildcard now forces credentials=false)
- SSRF DNS fail-open changed to fail-closed (deny by default on timeout/failure)
- Helm deployment: added preStop lifecycle hook for zero-downtime rolling updates
- Helm deployment: automountServiceAccountToken now conditional on leader election backend
- Python version floor lowered from 3.14 to 3.12 (3.14 is pre-release)
- Repository URL consistency (routeiq/routeiq -> baladithyab/RouteIQ)
- Removed dead `black` dependency from dev extras (ruff is the formatter)

### Added
- File-based persistence for governance, usage policies, guardrail policies, and prompts
  - `ROUTEIQ_GOVERNANCE_STATE_PATH` — persist workspace and key governance config
  - `ROUTEIQ_USAGE_POLICIES_STATE_PATH` — persist usage policy definitions
  - `ROUTEIQ_GUARDRAIL_POLICIES_STATE_PATH` — persist guardrail policy definitions
  - `ROUTEIQ_PROMPTS_STATE_PATH` — persist prompt templates and versions
  - State is automatically loaded at startup and saved after every CRUD mutation
- CORS origins default changed from `"*"` to `""` (must be explicitly configured)
- CHANGELOG.md

### Changed
- Version bumped to 1.0.0-rc1 across all packages
- `uv` is now the primary/recommended package manager (pip remains as alternative)
- `routeiq-routing` standalone package bumped to 1.0.0rc1

### Changed
- Version bumped to 1.0.0-rc1 across all packages
- `uv` is now the primary/recommended package manager (pip remains as alternative)
- `routeiq-routing` standalone package bumped to 1.0.0rc1

## [0.2.0] — 2026-03-28

### Added
- Three-layer architecture (ADR-0001): LiteLLM upstream -> RouteIQ plugins -> Gateway
- 18+ ML routing strategies with inference-only adapters (KNN, SVM, MLP, MF, ELO)
- Centroid zero-config routing (~2ms, 5 tiers, 5 profiles)
- MCP Gateway (JSON-RPC, SSE, REST surfaces) on top of LiteLLM native MCP
- A2A Gateway for Agent-to-Agent protocol communication
- Skills Gateway: Anthropic Computer Use, Bash, and Text Editor skill execution
- Plugin system with lifecycle management (13 built-in plugins)
- Policy engine (OPA-style pre-request evaluation)
- Governance layer: workspace isolation, API key governance, usage policies
- Guardrail policy engine (14 check types)
- Prompt management with versioning and A/B testing
- Context optimization (30-70% token savings via 6 lossless transforms)
- Evaluation pipeline (LLM-as-judge quality scoring with feedback to routing)
- Personalized routing (per-user/per-team model preference learning)
- Router-R1 iterative reasoning-based routing
- Admin dashboard (6-page React UI at `/ui/`)
- OIDC/SSO integration (Keycloak, Auth0, Okta, Azure AD, Google)
- RBAC, quotas, audit logging, SSRF protection, circuit breakers
- Helm charts for Kubernetes deployment
- HA with leader election (Redis and Kubernetes Lease API)
- OpenTelemetry traces, metrics, and structured logging
- CLI: `routeiq start/validate-config/version/probe-services`
- Standalone package: `uv add routeiq-routing` (or `pip install routeiq-routing`) for ML routing without full gateway
