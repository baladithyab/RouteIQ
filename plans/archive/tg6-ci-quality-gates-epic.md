# TG6: CI Quality Gates Epic

**Status:** Queued  
**Epic Owner:** DevOps / Backend  
**Last Updated:** 2026-02-04

---

## Goal

Establish robust CI/CD quality gates with automated testing, coverage enforcement, and security scanning to ensure production-ready code quality.

### Non-Goals

- Implementing CD pipelines for specific cloud providers (separate epic)
- Building custom CI infrastructure (use GitHub Actions)
- Creating a test framework (use existing pytest + hypothesis)

---

## Sub-TG Breakdown

### TG6.1: Test Coverage Gates

**Description:** Enforce minimum test coverage thresholds in CI.

**Acceptance Criteria:**
- [ ] Coverage reporting integrated with pytest-cov
- [ ] CI fails if coverage drops below 80% (configurable threshold)
- [ ] Coverage report uploaded as artifact for review
- [ ] Diff coverage reported for PRs (new code must meet threshold)
- [ ] Documentation describes coverage requirements

**Key Files:**
- `.github/workflows/ci.yml`
- `pyproject.toml` (coverage config)
- `docs/CONTRIBUTING.md`

---

### TG6.2: Security Scanning Gates

**Description:** Automated security scanning for dependencies and code.

**Acceptance Criteria:**
- [ ] Dependabot enabled for dependency updates
- [ ] SAST scanning with bandit or semgrep
- [ ] CI fails on high/critical vulnerabilities
- [ ] Secret scanning enabled on repository
- [ ] Weekly vulnerability report generated

**Key Files:**
- `.github/workflows/security.yml` (new)
- `.github/dependabot.yml`
- `pyproject.toml` (bandit config)

---

### TG6.3: Performance Regression Gates

**Description:** Prevent performance regressions with automated benchmarks.

**Acceptance Criteria:**
- [ ] Benchmark suite for critical paths (routing, auth, streaming)
- [ ] CI compares benchmark results against baseline
- [ ] CI warns on >10% regression, fails on >25%
- [ ] Benchmark results stored for trend analysis
- [ ] Documentation for running benchmarks locally

**Key Files:**
- `.github/workflows/benchmark.yml` (new)
- `tests/benchmark/` (new directory)
- `docs/development.md`

---

### TG6.4: Integration Test Matrix

**Description:** Comprehensive integration test matrix across Python versions and dependencies.

**Acceptance Criteria:**
- [ ] CI matrix tests Python 3.12, 3.13, 3.14
- [ ] CI matrix tests with latest and pinned dependency versions
- [ ] Docker build verification in CI
- [ ] Integration tests run against containerized services
- [ ] Test matrix documented in CI config

**Key Files:**
- `.github/workflows/ci.yml`
- `docker-compose.local-test.yml`
- `tests/integration/`

---

## Branch + Squash Workflow

```bash
# Create feature branch
git checkout -b tg6-ci-quality-gates

# Work on sub-TGs, commit locally as needed
git add .
git commit -m "feat(tg6.1): add coverage gates to CI"
# ... more commits ...

# When TG6 is complete, squash merge to main
git checkout main
git merge --squash tg6-ci-quality-gates
git commit -m "feat: complete TG6 CI quality gates epic"
```

---

## Test/Validation Commands

```bash
# Run full test suite with coverage
uv run pytest tests/ --cov=src/litellm_llmrouter --cov-report=html --cov-fail-under=80

# Run linting and type checking
./scripts/lint.sh check
uv run ruff check src/ tests/
uv run mypy src/litellm_llmrouter/

# Security scanning
uv run bandit -r src/litellm_llmrouter/ -ll

# Run benchmarks locally
uv run pytest tests/benchmark/ --benchmark-only

# Verify Docker build
docker build -f docker/Dockerfile -t routeiq-gateway:test .

# Run integration tests with Docker
docker-compose -f docker-compose.local-test.yml up -d
uv run pytest tests/integration/ -v
docker-compose -f docker-compose.local-test.yml down
```

---

## Bulk Publish Workflow

If local `git push` is blocked by Code Defender:

```bash
# Push via Road Runner
rr push

# MANDATORY: Sync local repo after rr push
git pull
```

> ⚠️ **Critical:** Always run `git pull` after `rr push` to sync your local repo with the remote state.

---

## Related Documents

- [Contributing Guide](../CONTRIBUTING.md)
- [Validation Plan](validation-plan.md)
- [GitHub Actions Workflows](../.github/workflows/)
- [Backlog - CI Hardening](backlog.md)
