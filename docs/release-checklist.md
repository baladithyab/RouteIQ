# Release Checklist

This document provides a comprehensive checklist for validating and releasing RouteIQ Gateway.

## Version Information

- **Helm Chart Version**: `0.1.0`
- **Application Version**: `1.81.3` (LiteLLM upstream compatibility)
- **Release Date**: 2026-01-30

## Pre-Release Validation

### 1. Code Quality Gates

Run all quality gates before any release:

```bash
# Linting (must pass with zero errors)
uv run ruff check src/ tests/ --no-fix

# Format check (must pass with zero errors)
uv run ruff format --check src/ tests/

# Unit tests
uv run pytest tests/unit/ -q --tb=short

# Full test suite (includes property tests)
uv run pytest tests/ -q --tb=short
```

**Expected Results:**
- ✅ `ruff check`: All checks passed!
- ✅ `ruff format`: 62 files already formatted
- ✅ Full test suite: 871 passed, 33 skipped, 0 failed

**Skipped Tests:**
- Integration tests requiring external services (22 skipped)
- MCP parity tests for not-yet-implemented features (4 skipped): protocol proxy routes, MCP REST routes, OAuth register route

### 2. Docker Build Verification

```bash
# Build the production image
docker build -f docker/Dockerfile . --quiet

# Expected output: sha256:<hash>
```

### 3. Helm Chart Validation

```bash
# Lint chart
helm lint deploy/charts/routeiq-gateway

# Template validation (renders all manifests)
helm template routeiq deploy/charts/routeiq-gateway > /dev/null && echo "SUCCESS"
```

**Expected Results:**
- ✅ 1 chart(s) linted, 0 chart(s) failed
- ℹ️ INFO: Chart.yaml: icon is recommended (non-blocking)

### 4. Integration Script Requirements

The following validation scripts require a running service stack:

#### Parity Validation (`scripts/validate_parity.py`)

**Prerequisites:**
- RouteIQ Gateway running at `http://localhost:4000`
- Valid API key configured

```bash
# Start the stack first
docker compose up -d

# Run parity validation
uv run python scripts/validate_parity.py
```

#### MCP Gateway Validation (`scripts/validate_mcp_gateway.py`)

**Prerequisites:**
- HA stack running (load balancer at `:8080`, replicas at `:4000`, `:4001`)
- MCP stub server at `:9100`
- Valid admin API key configured

```bash
# Start HA stack with MCP stub
docker compose -f docker-compose.ha.yml up -d

# Run MCP gateway validation
uv run python scripts/validate_mcp_gateway.py
```

---

## Required Environment Variables

### Critical Security Variables

| Variable | Required | Description | Generation Command |
|----------|----------|-------------|-------------------|
| `LITELLM_MASTER_KEY` | ✅ Yes | Master API key | `openssl rand -hex 32` |
| `ADMIN_API_KEYS` | ✅ For control-plane | Comma-separated admin keys | `openssl rand -hex 32` |
| `POSTGRES_PASSWORD` | ✅ For HA | Database password | `openssl rand -hex 16` |

### Feature Flags (Defaults Shown)

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_GATEWAY_ENABLED` | `false` | Enable MCP gateway |
| `A2A_GATEWAY_ENABLED` | `false` | Enable A2A gateway |
| `LLMROUTER_ENABLE_MCP_TOOL_INVOCATION` | `false` | Allow remote tool calls |
| `LLMROUTER_ALLOW_PRIVATE_IPS` | `false` | Allow private IP connections |
| `ADMIN_AUTH_ENABLED` | `true` | Require admin key for control-plane |

### LLM Provider Keys

Configure at least one provider:

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `AZURE_API_KEY` | Azure OpenAI |
| `AWS_DEFAULT_REGION` | AWS Bedrock (uses IAM) |

---

## Security Posture

### Authentication Model

1. **User API Key Authentication** (`Authorization: Bearer sk-xxx`)
   - All inference endpoints
   - Read-only monitoring/listing endpoints

2. **Admin API Key Authentication** (`X-Admin-API-Key: <key>`)
   - `POST /router/reload`, `POST /config/reload`
   - All MCP server management endpoints
   - All A2A agent management endpoints

3. **Unauthenticated** (Health probes only)
   - `/_health/live`
   - `/_health/ready`

### Fail-Closed Defaults

| Feature | Default State | Security Impact |
|---------|---------------|-----------------|
| Admin authentication | Enabled | Control-plane endpoints return 403 if keys not configured |
| MCP tool invocation | Disabled | POST `/llmrouter/mcp/tools/call` returns 501 |
| Private IP access | Blocked | SSRF protection blocks 10.x, 172.x, 192.168.x by default |
| Loopback/link-local | Always blocked | Cannot be overridden via allowlist |

### SSRF Protection Configuration

```bash
# For internal MCP servers, explicitly allowlist:
LLMROUTER_SSRF_ALLOWLIST_HOSTS=mcp.internal,.trusted-corp.com
LLMROUTER_SSRF_ALLOWLIST_CIDRS=10.100.0.0/16,192.168.50.0/24
```

---

## CI/CD Checks

### GitHub Actions Workflow

The following checks run automatically on PR/push:

```yaml
# .github/workflows/docker-build.yml
jobs:
  lint:
    - uv lock --check            # Lockfile integrity
    - uv run ruff check src/ tests/
    - uv run ruff format --check src/ tests/
  
  test:
    - uv run pytest tests/ -q
  
  docker:
    - docker build -f docker/Dockerfile .
    - docker buildx build --sbom=true --provenance=true  # Supply chain
  
  helm:
    - helm lint deploy/charts/routeiq-gateway
    - helm template routeiq deploy/charts/routeiq-gateway
```

### Manual Validation Commands

For CI environments without Docker/Helm:

```bash
# CI lint (document these in pipeline config)
uv run ruff check src/ tests/ --no-fix
uv run ruff format --check src/ tests/

# CI tests
uv run pytest tests/ -q --tb=no -x

# Docker build (requires docker daemon)
docker build -f docker/Dockerfile . --quiet

# Helm (requires helm CLI)
helm lint deploy/charts/routeiq-gateway
helm template routeiq deploy/charts/routeiq-gateway > /tmp/manifests.yaml
```

---

## Deployment Configurations

### Docker Compose Options

| File | Use Case | Components |
|------|----------|------------|
| `docker-compose.yml` | Development | Gateway only |
| `docker-compose.ha.yml` | HA testing | Gateway x2 + Redis + Postgres + nginx |
| `docker-compose.otel.yml` | Observability | Gateway + Jaeger + Prometheus |
| `docker-compose.ha-otel.yml` | Full stack | HA + Observability |

### Kubernetes Deployment

```bash
# Install with Helm
helm install routeiq deploy/charts/routeiq-gateway \
  --set secrets.litellmMasterKey="$(openssl rand -hex 32)" \
  --set secrets.adminApiKeys="$(openssl rand -hex 32)"

# Verify deployment
kubectl get pods -l app.kubernetes.io/name=routeiq-gateway
kubectl logs -l app.kubernetes.io/name=routeiq-gateway --tail=50
```

---

## Rollback Procedures

### Docker Compose Rollback

```bash
# Stop current version
docker compose down

# Pull previous version
docker pull ghcr.io/baladithyab/litellm-llm-router:<previous-tag>

# Update docker-compose.yml image tag and restart
docker compose up -d
```

### Helm Rollback

```bash
# List release history
helm history routeiq

# Rollback to previous revision
helm rollback routeiq <revision-number>

# Verify rollback
kubectl rollout status deployment/routeiq-routeiq-gateway
```

### Model Rollback (ML Routing)

If using ML-based routing with signed manifests:

```bash
# Verify model artifacts
uv run python -c "
from litellm_llmrouter.model_artifacts import verify_manifest
manifest = verify_manifest('/app/models/manifest.json')
print(f'Active model: {manifest.artifacts[0].path}')
"

# Rollback: Update manifest to point to previous model version
# Then trigger reload:
curl -X POST -H "X-Admin-API-Key: $ADMIN_KEY" \
  http://localhost:4000/router/reload
```

---

## Post-Release Validation

After deployment, verify:

1. **Health endpoints respond**:
   ```bash
   curl http://localhost:4000/_health/live
   curl http://localhost:4000/_health/ready
   ```

2. **Inference works**:
   ```bash
   curl -X POST http://localhost:4000/v1/chat/completions \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

3. **Admin endpoints protected**:
   ```bash
   # Should return 401 without admin key
   curl -X POST http://localhost:4000/router/reload
   ```

4. **Observability flowing** (if OTEL enabled):
   - Check Jaeger UI for traces
   - Check Prometheus/Grafana for metrics

---

## Known Limitations

1. **MCP Tool Invocation**: Disabled by default. Enable only after configuring SSRF allowlists.
2. **MCP Protocol Proxy**: Feature routes not yet fully implemented; tests are skipped.
3. **Helm Chart Icon**: Chart.yaml missing icon field (non-blocking INFO warning).

---

## Release Artifacts

- Docker Image: `ghcr.io/baladithyab/litellm-llm-router:latest`
- Helm Chart: `deploy/charts/routeiq-gateway`
- Documentation: `docs/` directory
