# Cloud-Native Architectural Roadmap
## Transitioning LiteLLM + LLM-Router to Production-Grade Infrastructure

**Document Version:** 1.1  
**Last Updated:** 2026-01-30  
**Status:** Live / Iterating

## Current Status

**Overall Status:** ✅ Milestones A, B, and C are **Completed**. Milestone D is **Planned**.

For the authoritative release proof and validation steps, please refer to the [Release Checklist](../docs/release-checklist.md).

### Status Legend
- ✅ **Completed**: Fully implemented, tested, and merged.
- 🟡 **In Progress**: Currently being worked on.
- ⬜ **Planned**: Scheduled for future work.

---

## Executive Summary

This roadmap provides a comprehensive, phased approach to transform the LiteLLM + LLM-Router combination into a robust, production-grade cloud-native system. The transformation emphasizes High Availability (HA), observability, MLOps excellence, and operational resilience while maintaining the system's core value proposition of intelligent ML-powered routing.

**Update (Jan 2026):** Milestones A, B, and C (P0-P2) have been executed. See [`docs/release-checklist.md`](../docs/release-checklist.md) for release verification and [`plans/p1-remove-import-side-effects-plan.md`](p1-remove-import-side-effects-plan.md) for the architectural cleanup details. The focus now shifts to Milestone D+ (Post-MVP).

### Key Objectives
- **Zero-downtime Operations**: Hot-reloading configurations and ML models without service interruptions
- **Production Observability**: Full OpenTelemetry integration with traces, metrics, logs, and SLO tracking
- **MLOps Automation**: Closed-loop pipeline from trace collection → training → deployment → monitoring
- **High Availability**: Multi-replica stateless architecture with durable state management
- **Cloud-Native Standards**: Kubernetes-first design with standardized patterns for config, secrets, and persistence

### Success Metrics
- **Availability**: 99.95% uptime (< 4.38 hours downtime/year)
- **Latency**: P99 < 500ms (including ML routing overhead < 50ms)
- **Scalability**: Support 10K+ req/min per gateway replica
- **Hot-Reload**: Config/model updates propagate within 60 seconds with zero dropped requests
- **Observability**: 100% trace coverage with routing decision metadata

---

## Current State Assessment

### Architecture Baseline (As of Q1 2026)

**Deployed Components** (per [`docker-compose.ha.yml`](../docker-compose.ha.yml)):
- **Gateway Replicas**: 2x LiteLLM + LLMRouter pods
- **Load Balancer**: Nginx with least-connections
- **State Layer**: PostgreSQL 16 (user/team config, API keys, logs)
- **Cache Layer**: Redis 7 (response cache, rate limiting)
- **ML Routing**: KNN/SVM routers with file-based hot-reload

**Capabilities Delivered** (per [`docs/architecture/overview.md`](../docs/architecture/overview.md)):
- ✅ 100+ LLM provider integrations
- ✅ 18+ ML routing strategies
- ✅ OpenTelemetry tracing to Jaeger/Tempo/X-Ray
- ✅ Hot-reload for routing models (file-watch mechanism)
- ✅ Basic HA with shared Redis/Postgres

**Identified Gaps** (per [`docs/litellm-cloud-native-enhancements.md`](../docs/litellm-cloud-native-enhancements.md)):
- ❌ **Config Sync**: No standardized mechanism for multi-replica config distribution
- ❌ **Graceful Shutdown**: Active LLM streams terminated on SIGTERM
- ❌ **MLOps Automation**: Fully automated training/deployment workflow (currently script-driven)
- ❌ **Observability**: Missing routing decision visibility and SLO dashboards
- ❌ **State Management**: No distributed locks for model downloads
- ❌ **Persistence**: S3/object storage integration not standardized

---

## Target State Architecture

### Conceptual Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                     INGRESS & TRAFFIC MANAGEMENT                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │   Ingress    │──▶│   Service    │──▶│  HPA/KEDA Scaler     │   │
│  │  (TLS Term)  │   │   Mesh (LB)  │   │ (Active Streams)     │   │
│  └──────────────┘   └──────────────┘   └──────────────────────┘   │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│               GATEWAY PODS (Stateless, Multi-Replica)               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Main Container: LiteLLM + LLMRouter                         │   │
│  │  - ML Routing Inference (KNN/MLP/Graph/etc.)                │   │
│  │  - Request Authentication & Rate Limiting                    │   │
│  │  - OTel Instrumentation (Traces/Metrics/Logs)               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Config Sync Sidecar                                         │   │
│  │  - Watches S3/MinIO for config.yaml + llm_candidates.json   │   │
│  │  - Validates config schema before SIGHUP                     │   │
│  │  - Exposes /health/config-sync endpoint                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Model Sync Sidecar                                          │   │
│  │  - Polls model registry (S3/MinIO) for *.pkl/*.pt artifacts  │   │
│  │  - Uses distributed lock (Redis) to avoid stampedes          │   │
│  │  - Validates artifact signatures (Moat-Mode)                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  OTel Collector Sidecar (Optional)                           │   │
│  │  - Batches & exports traces/metrics to backend               │   │
│  │  - Supports Jaeger/Tempo/CloudWatch/Grafana Cloud            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────┐
│                        STATE & PERSISTENCE LAYER                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ PostgreSQL │  │   Redis    │  │ S3/MinIO   │  │ Secrets Mgr │  │
│  │  (RDS/HA)  │  │(ElastiCache│  │ (Configs & │  │ (Vault/AWS) │  │
│  │            │  │  Cluster)  │  │  Models)   │  │             │  │
│  │ - User DB  │  │ - Cache    │  │ - config/* │  │ - API Keys  │  │
│  │ - API Keys │  │ - Locks    │  │ - models/* │  │ - Certs     │  │
│  │ - Audit Log│  │ - Sessions │  │ - Backups  │  │             │  │
│  └────────────┘  └────────────┘  └────────────┘  └─────────────┘  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY & CONTROL PLANE                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Tempo/Jaeger │  │  Prometheus  │  │       Grafana            │  │
│  │   (Traces)   │  │   (Metrics)  │  │  - Dashboards            │  │
│  │              │  │              │  │  - Alerts (SLO Targets)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Logging Pipeline: Fluentd/Loki → Correlation via trace_id   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                          MLOps PIPELINE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Trace Export │─▶│   Training   │─▶│   Model Registry         │  │
│  │ (OTEL → S3)  │  │   Jobs (K8s) │  │  (MLflow + S3 Backend)   │  │
│  └──────────────┘  └──────────────┘  └──────┬───────────────────┘  │
│                                              │                      │
│  ┌───────────────────────────────────────────▼──────────────────┐  │
│  │  Canary Rollout: 5% traffic → 50% → 100% (Argo Rollouts)     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Component Inventory

| Component | Technology Options | Purpose | HA Requirements |
|-----------|-------------------|---------|-----------------|
| **Gateway Pods** | LiteLLM + LLMRouter (custom image) | Request routing & LLM proxying | Min 3 replicas, HPA-enabled |
| **Ingress Controller** | AWS ALB / Nginx Ingress / Istio | TLS termination, L7 routing | Multi-AZ load balancer |
| **PostgreSQL** | RDS Multi-AZ / CloudNative-PG | User config, API keys, audit logs | Multi-AZ with automatic failover |
| **Redis** | ElastiCache Cluster / Redis Sentinel | Cache, rate limiting, distributed locks | 3+ node cluster (quorum) |
| **Object Storage** | S3 / MinIO / GCS | Config files, ML artifacts, trace backups | Versioning enabled, CRR for DR |
| **Secrets Manager** | AWS Secrets Manager / Vault / K8s Secrets | API keys, TLS certs, DB credentials | Encrypted at rest, auto-rotation |
| **OTel Collector** | OpenTelemetry Collector / ADOT | Trace/metric aggregation & export | Per-pod sidecar or DaemonSet |
| **Trace Backend** | Jaeger / Tempo (S3) / CloudWatch X-Ray | Distributed tracing storage & query | S3 backend for durability |
| **Metrics Backend** | Prometheus + Thanos / CloudWatch | Time-series metrics storage | Remote write to long-term storage |
| **Dashboards** | Grafana / CloudWatch Dashboards | Visualization, alerting, SLO tracking | Backed by Git (IaC) |
| **Model Registry** | MLflow + S3 / S3 only | Versioned ML artifact storage | S3 versioning + lifecycle policies |
| **Training Platform** | Kubernetes Jobs / SageMaker Training | ML model training environment | Ephemeral, stateless jobs |
| **Core Proxy** | Unified API (OpenAI compatible), 100+ providers. | ML-based Routing ([strategies.py](../src/litellm_llmrouter/strategies.py)), A2A/MCP protocol support. |

---

## Execution Roadmap

This roadmap decomposes the architectural vision into concrete, PR-sized work items. Each item includes an assigned "Owner Mode" for execution and specific validation commands.

### Milestone A: Foundation & Containerization
**Status:** ✅ **Completed**
**Goal:** Establish a reproducible, containerized environment for LiteLLM.

- [x] **Dockerization**
  - [x] Create optimized `Dockerfile` (multi-stage build).
  - [x] Ensure non-root user execution for security.
  - [x] Minimize image size (alpine/slim variants).
- [x] **Docker Compose**
  - [x] Create `docker-compose.yml` for local development.
  - [x] Include Redis service for caching/rate-limiting.
  - [x] Include Postgres service (optional, for logging).
- [x] **Configuration Management**
  - [x] Externalize configuration (move out of code).
  - [x] Support `.env` files and environment variable overrides.
  - [x] Create `config.yaml` template.

### Milestone B: Observability & Monitoring
**Status:** ✅ **Completed**
**Goal:** Gain visibility into system performance, errors, and usage.

- [x] **Structured Logging**
  - [x] Configure JSON logging for machine parsing.
  - [x] Ensure correlation IDs are propagated.
- [x] **Metrics (Prometheus)**
  - [x] Expose `/metrics` endpoint.
  - [x] Define key metrics: Request Latency, Error Rate, Throughput (RPM).
  - [x] Set up Prometheus scraping config (example).
- [x] **Tracing (OpenTelemetry)**
  - [x] Instrument code with OpenTelemetry SDK.
  - [x] Configure exporter (OTLP/Jaeger).
  - [x] Verify trace propagation across services.

### Milestone C: Security & Production Hardening
**Status:** ✅ **Completed**
**Goal:** Secure the deployment for production use.

- [x] **Secret Management**
  - [x] Integrate with a secret manager (or secure env var injection).
  - [x] Ensure no secrets are hardcoded or logged.
- [x] **Rate Limiting & Throttling**
  - [x] Configure global and per-user/key rate limits.
  - [x] Test Redis-backed rate limiting.
- [x] **Health Checks**
  - [x] Implement `/health/liveness` and `/health/readiness` probes.
  - [x] Configure Docker/K8s health checks.
- [x] **Documentation**
  - [x] Write "Deployment Guide".
  - [x] Write "Configuration Reference".

### Milestone D: Advanced Features (Post-MVP)
**Status:** ⬜ **Planned / Not Started**
**Goal:** Scale and extend capabilities.

- [ ] **Kubernetes Helm Chart**
  - [ ] Create Helm chart for scalable deployment.
  - [ ] Support HPA (Horizontal Pod Autoscaling).
- [ ] **Service Mesh Integration**
  - [ ] Istio/Linkerd configuration examples.
- [ ] **Advanced Caching**
  - [ ] Semantic Caching implementation.
- [ ] **GitOps Workflow**
  - [ ] ArgoCD/Flux examples.

#### Detailed Work Items (Milestone D+)

##### D.1: Control-Plane OIDC SSO + RBAC
- **Scope**: Integrate OIDC provider (Keycloak/Auth0) for admin routes.
- **Acceptance Criteria**:
  - [ ] Admin endpoints require valid JWT.
  - [ ] RBAC roles (Admin, Viewer, Editor) enforced.
- **Validation**:
  ```bash
  curl -H "Authorization: Bearer $JWT" http://localhost:4000/admin/config
  ```

##### D.2: Distributed Registry State
- **Scope**: Move MCP/A2A registry state from in-memory/local to Redis/Postgres.
- **Acceptance Criteria**:
  - [ ] Registry updates on one pod visible to others immediately.
  - [ ] State survives pod restarts.

##### D.3: MCP Protocol Parity
- **Scope**: Implement full OAuth 2.0 flow for MCP tools and remove protocol compliance skips.
- **Acceptance Criteria**:
  - [ ] All MCP compliance tests pass without skips.
  - [ ] OAuth-protected tools function correctly.

##### D.4: Plugin Sandboxing & Provenance
- **Scope**: Enforce signature verification for plugins and explore WASM/process isolation.
- **Acceptance Criteria**:
  - [ ] Unsigned plugins rejected in production mode.
  - [ ] Plugins cannot access host filesystem outside allowed paths.

##### D.5: Vector Store Extension Model
- **Scope**: Create plugin interface for custom vector stores and verify parity across implementations.
- **Acceptance Criteria**:
  - [ ] Plugin can register new vector store backend.
  - [ ] Standard test suite passes for custom backend.

##### D.6: Management UI
- **Scope**: React-based admin dashboard for system management.
- **Acceptance Criteria**:
  - [ ] View/edit rate limits and quotas.
  - [ ] Searchable audit logs.

---

## Tracking Table

| ID | Milestone | Task | Owner | Status | PR | Risk |
|----|-----------|------|-------|--------|----|------|
| A.1 | A (P0) | K8s Manifests | `architect` | ✅ Done | - | Low |
| A.2 | A (P0) | HA State | `code` | ✅ Done | - | Med |
| A.3 | A (P0) | Basic OTel | `code` | ✅ Done | - | Low |
| A.4 | A (P0) | Security Baseline | `code` | ✅ Done | - | Low |
| B.1 | B (P1) | Config Sync Sidecar | `code` | ✅ Done | - | High |
| B.2 | B (P1) | Routing Decision Visibility | `code` | ✅ Done | - | Low |
| B.3 | B (P1) | Security Hardening | `code` | ✅ Done | - | Med |
| B.4 | B (P1) | Streaming Shutdown | `architect` | ✅ Done | - | Med |
| C.1 | C (P2) | MLOps Tooling (Scripts) | `architect`, `Services Team` | ✅ Done | - | High |
| C.2 | C (P2) | Circuit Breakers | `code` | ✅ Done | - | Med |
| C.3 | C (P2) | Autoscaling | `code` | ✅ Done | - | Low |
| D.1 | D+ | OIDC SSO + RBAC | `code` | ⚪ Backlog | - | Med |
| D.2 | D+ | Distributed Registry | `architect` | ⚪ Backlog | - | High |
| D.3 | D+ | MCP Protocol Parity | `code` | ⚪ Backlog | - | Low |
| D.4 | D+ | Plugin Sandboxing | `architect` | ⚪ Backlog | - | High |
| D.5 | D+ | Vector Store Extension | `code` | ⚪ Backlog | - | Med |
| D.6 | D+ | Management UI | `frontend-specialist` | ⚪ Backlog | - | Low |

## Appendix: Code References

- **Auth Boundary**: [`src/litellm_llmrouter/auth.py`](../src/litellm_llmrouter/auth.py)
- **SSRF Validation**: [`src/litellm_llmrouter/url_security.py`](../src/litellm_llmrouter/url_security.py)
- **MCP Gateway**: [`src/litellm_llmrouter/mcp_gateway.py`](../src/litellm_llmrouter/mcp_gateway.py)
- **A2A Gateway**: [`src/litellm_llmrouter/a2a_gateway.py`](../src/litellm_llmrouter/a2a_gateway.py)
- **Startup Logic**: [`src/litellm_llmrouter/startup.py`](../src/litellm_llmrouter/startup.py:275)
- **Backend Configuration**: `DATABASE_URL`, `LLMFS`, `REDIS_URL`, `LOG_DIR`, `ADMIN_API_KEYS`, `SSRF_PROTECTION_ENABLED`
- **Backend Estado**: [`src/litellm_llmrouter/backendmanager.py`](../src/litellm_llmrouter/backendmanager.py:193), [`src/litellm_llmrouter/backendmanager.py`](../src/litellm_llmrouter/backendmanager.py:216), [`src/audit_exporter/exporter.py`](../src/audit_exporter/exporter.py:44), [`src/litellm_llmrouter/signatures.py`](../src/litellm_llmrouter/signatures.py:176)
- **C2B and MCP**: [`src/litellm_llmrouter/backendmanager.py`](../src/litellm_llmrouter/backendmanager.py:334), [`src/litellm_llmrouter/gateway/__init__.py`](../src/litellm_llmrouter/gateway/__init__.py:121), [`src/litellm_llmrouter/plugins.py`](../src/litellm_llmrouter/plugins.py:741), [`src/litellm_llmrouter/cli.py`](../src/litellm_llmrouter/cli.py:253), [`src/litellm_llmrouter/backendmanager.py`](../src/litellm_llmrouter/backendmanager.py:193), [`src/litellm_llmrouter/auth.py`](../src/litellm_llmrouter/auth.py), [`src/litellm_llmrouter/url_security.py`](../src/litellm_llmrouter/url_security.py), [`src/litellm_llmrouter/custom_proxies.py`](../src/litellm_llmrouter/custom_proxies.py), [`src/litellm_llmrouter/mcp_client.py`](../src/litellm_llmrouter/mcp_client.py), [`src/audit_exporter/exporter.py`](../src/audit_exporter/exporter.py), [`src/litellm_llmrouter/logger/aws_connect.py`](../src/litellm_llmrouter/logger/aws_connect.py), [`src/litellm_llmrouter/logger/audit_logger.py`](../src/litellm_llmrouter/logger/audit_logger.py)
