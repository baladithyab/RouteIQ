# Implementation Backlog

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

**Status:** Active  
**Last Updated:** 2026-02-03  
**Resume Checkpoint:** [See TG Backlog & Workflow](resume-checkpoint.md)
**Context:** Consolidated backlog for Cloud-Native, Security, and MLOps enhancements.

---

## What’s Already Done (WS2 / WS3 / WS5)

### WS2: Security Hardening
- **Pickle Opt-in:** Enforced explicit opt-in for pickle serialization to prevent RCE.
- **SSRF Guards:** Implemented validation for outbound requests.
- **CI Hardening:** Enhanced GitHub Actions security.
- **Auth Gating:** Secured custom routes with authentication checks.
- **Validation:** Passed [Gate 7 Security Validation](../GATE7_SECURITY_VALIDATION_REPORT.md).

### WS3: Cloud-Native Improvements
- **K8s Probes:** Added `/health/liveliness` and `/health/readiness` endpoints.
- **Docker Hardening:** Migrated to non-root user (UID 1000) and `tini` init system.
- **Docs Updates:** Refreshed deployment and architecture documentation.
- **Validation:** Passed [Gate 6 MCP Validation](../GATE6_MCP_VALIDATION_REPORT.md).

### WS5: Rebranding & Documentation
- **Rebranding:** Unified "RouteIQ" identity.
- **Docs Refresh:** Updated [API Reference](../docs/api-reference.md) and [Deployment Guide](../docs/deployment.md).
- **Roadmap:** Established [Technical Roadmap](technical-roadmap.md).

---

## Consolidated Backlog

### P0: Critical Infrastructure & Observability (High Priority)

| ID | Item | Description | Impact | Risk | Evidence / Owner |
|----|------|-------------|--------|------|------------------|
| **P0-01** | **K8s Deployment Manifests** | Create standard `Deployment`, `Service`, and `ConfigMap` manifests for Kubernetes. | Enables scalable production deployment. | Low | [Deployment Guide](../docs/deployment.md) <br> **Owner:** DevOps |
| **P0-02** | **PostgreSQL HA Setup** | Configure production-grade Postgres (RDS/StatefulSet) with failover. | Ensures data durability and high availability. | Medium | [Config](../config/config.yaml) <br> **Owner:** DevOps |
| **P0-03** | **Redis Cluster Setup** | Configure Redis Cluster for distributed caching and locking. | Critical for rate limiting and model sync locks. | Medium | [High Availability](../docs/deployment.md) <br> **Owner:** DevOps |
| **P0-04** | **OTel Collector Deployment** | Standardize OpenTelemetry export pipeline (Sidecar/DaemonSet). | Provides visibility into system performance. | Low | [Observability](../docs/observability.md) <br> **Owner:** SRE |
| **P0-05** | **Hot-Reload Config Sync** | Implement sidecar to sync config from S3 without restarting pods. | Zero-downtime configuration updates. | High | [Hot Reloading](../docs/configuration.md) <br> **Owner:** Backend |
| **P0-06** | **Routing Decision Visibility** | Add OTel attributes (`router.strategy`, `router.score`) to traces. | Explains "why" a model was chosen. | Low | [ML Routing Arch](../docs/architecture/mlops-loop.md) <br> **Owner:** ML Engineer |

### P1: Resilience & Security Hardening (Medium Priority)

| ID | Item | Description | Impact | Risk | Evidence / Owner |
|----|------|-------------|--------|------|------------------|
| **P1-01** | **Model Sync with Redis Locks** | Implement distributed locking for model artifact downloads. | Prevents "thundering herd" on S3 during scaling. | Medium | [ML Routing Arch](../docs/architecture/mlops-loop.md) <br> **Owner:** Backend |
| **P1-02** | **Streaming-Aware Shutdown** | Handle `SIGTERM` to allow active LLM streams to complete. | Prevents user-facing errors during deployments. | Low | [Entrypoint Script](../docker/entrypoint.sh) <br> **Owner:** Backend |
| **P1-03** | **Multi-Replica Trace Correlation** | Propagate `trace_id` across Ingress, Gateway, and DB. | Enables end-to-end debugging in distributed setup. | Low | [Observability](../docs/observability.md) <br> **Owner:** SRE |
| **P1-04** | **Durable Audit Export** | Async export of audit logs to S3 for compliance. | Required for enterprise compliance (SOC2/HIPAA). | Low | [Security Docs](../docs/security.md) <br> **Owner:** Security |
| **P1-05** | **Secret Rotation Patterns** | Support dynamic reloading of secrets from file mounts. | Reduces risk of long-lived credentials. | Medium | [Security Docs](../docs/security.md) <br> **Owner:** Security |
| **P1-06** | **Backpressure / Load Shedding** | Reject requests (HTTP 503) when queue depth exceeds limit. | Prevents cascading failures under load. | High | [Validation Plan](validation-plan.md) <br> **Owner:** Backend |

### P2: Advanced MLOps & Scale (Lower Priority)

| ID | Item | Description | Impact | Risk | Evidence / Owner |
|----|------|-------------|--------|------|------------------|
| **P2-01** | **Automated Training Pipeline** | End-to-end loop: Trace Export -> Train -> Deploy. | Automates model improvement. | Medium | [MLOps Training](../docs/mlops-training.md) <br> **Owner:** ML Engineer |
| **P2-02** | **Model Rollback Mechanism** | One-click revert to previous model version via config. | Fast recovery from bad model deployments. | Low | [Hot Reloading](../docs/configuration.md) <br> **Owner:** DevOps |
| **P2-03** | **Degraded Mode Circuit Breakers** | Fallback logic (e.g., local cache) when Redis/DB fails. | Maintains service during dependency outages. | High | [High Availability](../docs/deployment.md) <br> **Owner:** Backend |
| **P2-04** | **Autoscaling Guidance** | Expose custom metrics (`active_streams`) for HPA/KEDA. | Scales based on actual load, not just CPU. | Medium | [Deployment Guide](../docs/deployment.md) <br> **Owner:** SRE |
| **P2-05** | **Moat-Mode Hardening** | Air-gapped artifacts and "no-internet" startup validation. | Enables deployment in secure/gov environments. | Low | [Moat Mode](../docs/deployment/air-gapped.md) <br> **Owner:** Security |

---

## Quick Wins Executed in This Step

*   [x] **Consolidated Backlog Creation:** Created this document to centralize tracking.
*   [x] **Config Sync Status Route:** Updated `/config/sync/status` to use [`ConfigSyncManager.get_status()`](../src/litellm_llmrouter/config_sync.py:199) in [`src/litellm_llmrouter/routes.py`](../src/litellm_llmrouter/routes.py:839).
*   [x] **API Reference Updates:** Fixed MCP paths, A2A schema, and `/router/info` in [`docs/api-reference.md`](../docs/api-reference.md:1).
*   [x] **HTTP Route Tests:** Added new HTTP-level tests in [`tests/test_http_routes.py`](../tests/test_http_routes.py:1).
*   [x] **Test Isolation Fixes:** Resolved `sys.modules` pollution in [`tests/unit/test_observability.py`](../tests/unit/test_observability.py:1), [`tests/property/test_observability_properties.py`](../tests/property/test_observability_properties.py:1), and [`tests/property/test_core_integration_properties.py`](../tests/property/test_core_integration_properties.py:1).

---

## Next 2–5 Quick Wins

These items are high-impact and safe to implement immediately:

1.  **P0-06: Routing Decision Visibility**
    *   *Action:* Add `span.set_attribute("router.model", ...)` in `reference/LLMRouter/llmrouter/models/knnrouter/router.py`.
    *   *Benefit:* Immediate visibility into ML routing logic.

2.  **P1-02: Streaming-Aware Shutdown**
    *   *Action:* Update `docker/entrypoint.sh` to trap `SIGTERM` and wait for active connections.
    *   *Benefit:* Zero user errors during next deployment.

3.  **P0-01: K8s Deployment Manifests**
    *   *Action:* Create `deploy/k8s/` directory with standard YAMLs based on `docker-compose.ha.yml`.
    *   *Benefit:* Unlocks Kubernetes testing.

4.  **P1-05: Secret Rotation (Docs)**
    *   *Action:* Add a section to `docs/security.md` describing how to use K8s Secrets for rotation.
    *   *Benefit:* Clarifies security posture without code changes.
