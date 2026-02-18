# TG5: Security Policy Epic

**Status:** Queued  
**Epic Owner:** Security / Backend  
**Last Updated:** 2026-02-04

---

## Goal

Harden RouteIQ Gateway security posture with durable audit logging, secret rotation, and advanced authentication policies for enterprise compliance (SOC2/HIPAA).

### Non-Goals

- Implementing a secrets management service (use external vaults)
- Building a custom audit log viewer (logs export to external systems)
- Replacing LiteLLM's existing auth mechanisms (extend them)

---

## Sub-TG Breakdown

### TG5.1: Durable Audit Export

**Description:** Implement async export of audit logs to S3 for compliance requirements.

**Acceptance Criteria:**
- [ ] Audit events serialized to JSON with timestamp, actor, action, resource, outcome
- [ ] Async S3 uploader with configurable bucket and prefix
- [ ] Retry logic with exponential backoff for failed uploads
- [ ] Integration test validates S3 upload (using LocalStack or mocks)
- [ ] Documentation for audit log schema and retention policy

**Key Files:**
- `src/litellm_llmrouter/audit.py`
- `src/litellm_llmrouter/audit_export.py` (new)
- `tests/integration/test_audit_export.py`
- `docs/security.md`

---

### TG5.2: Secret Rotation Patterns

**Description:** Support dynamic reloading of secrets from file mounts without restart.

**Acceptance Criteria:**
- [ ] File watcher detects changes to mounted secret files
- [ ] API keys and database credentials reload without pod restart
- [ ] Graceful handling of invalid/malformed secrets (log error, keep old value)
- [ ] Integration test simulates secret rotation during active requests
- [ ] Documentation describes K8s Secret rotation workflow

**Key Files:**
- `src/litellm_llmrouter/hot_reload.py`
- `src/litellm_llmrouter/config_loader.py`
- `tests/integration/test_secret_rotation.py`
- `docs/security.md`

---

### TG5.3: Enhanced RBAC Policies

**Description:** Extend RBAC with fine-grained permissions for model access and rate limits.

**Acceptance Criteria:**
- [ ] RBAC policies support per-model access control
- [ ] RBAC policies support per-tenant rate limit overrides
- [ ] Policy evaluation cached with configurable TTL
- [ ] Admin API for policy CRUD operations
- [ ] Unit tests cover edge cases (deny-by-default, policy conflicts)

**Key Files:**
- `src/litellm_llmrouter/rbac.py`
- `src/litellm_llmrouter/routes.py`
- `tests/unit/test_rbac.py`
- `tests/integration/test_rbac_enforcement.py`

---

### TG5.4: Input Validation Hardening

**Description:** Comprehensive input validation for all API endpoints to prevent injection attacks.

**Acceptance Criteria:**
- [ ] Pydantic models with strict validation for all request bodies
- [ ] Path/query parameters validated with explicit constraints
- [ ] Reject requests with unexpected fields (extra="forbid")
- [ ] Fuzz testing with hypothesis for critical endpoints
- [ ] Security test validates rejection of malformed inputs

**Key Files:**
- `src/litellm_llmrouter/routes.py`
- `tests/security/test_input_validation.py` (new)
- `tests/property/test_api_properties.py`

---

## Branch + Squash Workflow

```bash
# Create feature branch
git checkout -b tg5-security-policy

# Work on sub-TGs, commit locally as needed
git add .
git commit -m "feat(tg5.1): implement audit log S3 export"
# ... more commits ...

# When TG5 is complete, squash merge to main
git checkout main
git merge --squash tg5-security-policy
git commit -m "feat: complete TG5 security policy epic"
```

---

## Test/Validation Commands

```bash
# Unit tests for security modules
uv run pytest tests/unit/test_rbac.py tests/unit/test_audit.py -v

# Integration tests
uv run pytest tests/integration/test_audit_logging.py tests/integration/test_rbac_enforcement.py -v

# Security-specific tests
uv run pytest tests/security/ -v

# Lint and type check
./scripts/lint.sh check
uv run mypy src/litellm_llmrouter/audit.py src/litellm_llmrouter/rbac.py

# Property-based testing
uv run pytest tests/property/test_api_properties.py -v

# Docker Compose smoke test with audit enabled
docker-compose -f docker-compose.yml up -d
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}'
docker-compose logs gateway | grep -i audit
docker-compose down
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

- [Security Docs](../docs/security.md)
- [RBAC Implementation](../src/litellm_llmrouter/rbac.py)
- [Audit Implementation](../src/litellm_llmrouter/audit.py)
- [Gate 7 Security Report](../GATE7_SECURITY_VALIDATION_REPORT.md)
- [Backlog P1-04, P1-05](backlog.md)
