# TG4: Observability Epic

**Status:** Queued  
**Epic Owner:** SRE / Backend  
**Last Updated:** 2026-02-04

---

## Goal

Establish production-grade observability for RouteIQ Gateway including comprehensive tracing, metrics, and logging with full OpenTelemetry integration.

### Non-Goals

- Building a custom observability backend (use standard collectors/exporters)
- Replacing existing LiteLLM metrics (extend them)
- Implementing custom alerting rules (out of scope for gateway code)

---

## Sub-TG Breakdown

### TG4.1: Routing Decision Visibility

**Description:** Add OpenTelemetry attributes to traces that explain routing decisions.

**Acceptance Criteria:**
- [ ] Traces include `router.strategy` attribute (e.g., `knn`, `mlp`, `random`)
- [ ] Traces include `router.model_selected` attribute
- [ ] Traces include `router.score` for ML-based strategies
- [ ] Traces include `router.candidates_evaluated` count
- [ ] Unit tests validate attribute presence

**Key Files:**
- `src/litellm_llmrouter/strategies.py`
- `src/litellm_llmrouter/observability.py`
- `tests/unit/test_observability.py`

---

### TG4.2: OTel Collector Pipeline

**Description:** Standardize OpenTelemetry export pipeline with sidecar/DaemonSet patterns.

**Acceptance Criteria:**
- [ ] `config/otel-collector-config.yaml` supports exporters for Jaeger, OTLP, and Prometheus
- [ ] Docker Compose includes OTel collector sidecar
- [ ] Helm chart includes OTel collector configuration
- [ ] Smoke test validates trace export to collector

**Key Files:**
- `config/otel-collector-config.yaml`
- `docker-compose.otel.yml`
- `deploy/charts/routeiq-gateway/templates/`

---

### TG4.3: Multi-Replica Trace Correlation

**Description:** Propagate trace context across ingress, gateway replicas, and database calls.

**Acceptance Criteria:**
- [ ] W3C Trace Context headers propagated through all HTTP handlers
- [ ] Database spans include parent trace context
- [ ] Integration test with multi-replica setup validates correlation
- [ ] Documentation updated with trace correlation examples

**Key Files:**
- `src/litellm_llmrouter/observability.py`
- `tests/integration/test_observability_correlation.py`

---

### TG4.4: Custom Metrics for Autoscaling

**Description:** Expose custom metrics for HPA/KEDA-based autoscaling.

**Acceptance Criteria:**
- [ ] `/metrics` endpoint exposes `routeiq_active_streams` gauge
- [ ] `/metrics` endpoint exposes `routeiq_request_queue_depth` gauge
- [ ] `/metrics` endpoint exposes `routeiq_routing_latency_histogram`
- [ ] Prometheus scrape config validated in smoke test

**Key Files:**
- `src/litellm_llmrouter/routes.py`
- `src/litellm_llmrouter/observability.py`
- `config/prometheus.yml`

---

## Branch + Squash Workflow

```bash
# Create feature branch
git checkout -b tg4-observability

# Work on sub-TGs, commit locally as needed
git add .
git commit -m "feat(tg4.1): add routing decision trace attributes"
# ... more commits ...

# When TG4 is complete, squash merge to main
git checkout main
git merge --squash tg4-observability
git commit -m "feat: complete TG4 observability epic"
```

---

## Test/Validation Commands

```bash
# Unit tests for observability
uv run pytest tests/unit/test_observability.py -v

# Integration tests for trace correlation
uv run pytest tests/integration/test_observability_correlation.py -v

# Lint and type check
./scripts/lint.sh check
uv run mypy src/litellm_llmrouter/observability.py

# Docker Compose smoke test (OTel stack)
docker-compose -f docker-compose.otel.yml up -d
curl -s http://localhost:4000/health | jq .
docker-compose -f docker-compose.otel.yml down

# Validate metrics endpoint
curl -s http://localhost:4000/metrics | grep routeiq_
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

- [Observability Docs](../docs/observability.md)
- [OTel Collector Config](../config/otel-collector-config.yaml)
- [MLOps Loop Architecture](../docs/architecture/mlops-loop.md)
- [Backlog P0-04, P0-06](backlog.md)
