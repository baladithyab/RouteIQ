# Cloud-Native Architectural Roadmap
## Transitioning LiteLLM + LLM-Router to Production-Grade Infrastructure

**Document Version:** 1.1  
**Last Updated:** 2026-01-30  
**Status:** Live / Iterating

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
- ❌ **MLOps Pipeline**: Manual training/deployment workflow
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

---

## Execution Roadmap

This roadmap decomposes the architectural vision into concrete, PR-sized work items. Each item includes an assigned "Owner Mode" for execution and specific validation commands.

### Milestone A: Production Hardening (P0)
**Goal:** Establish a stable, observable, and secure foundation on Kubernetes.

| ID | Task | Owner Mode | Dependencies |
|----|------|------------|--------------|
| A.1 | **K8s Manifest Standardization**<br>Create production-grade Helm chart or Kustomize base. | `architect` | None |
| A.2 | **HA State Layer**<br>Configure Redis Sentinel/Cluster & Postgres HA. | `code` | A.1 |
| A.3 | **Basic Observability**<br>Ensure OTel traces reach Tempo/Jaeger. | `code` | A.1 |
| A.4 | **Security Baseline**<br>Enforce non-root execution & read-only root filesystem. | `code` | A.1 |

#### Detailed Work Items (Milestone A)

##### A.1: Kubernetes Manifests
- **Scope**: Convert `docker-compose.ha.yml` to `deploy/k8s/` (Deployment, Service, ConfigMap, Secret).
- **Acceptance Criteria**:
  - [ ] Pods start successfully with readiness probes passing.
  - [ ] Service routes traffic to all replicas.
  - [ ] Secrets mounted correctly from K8s Secrets.
- **Validation**:
  ```bash
  kubectl apply -f deploy/k8s/base/ --dry-run=client
  kubectl get pods -l app=litellm-gateway
  ```

##### A.2: HA State Configuration
- **Scope**: Update [`src/litellm_llmrouter/startup.py`](src/litellm_llmrouter/startup.py:275) to respect `REDIS_SENTINEL_HOSTS` and `DATABASE_URL` for HA.
- **Acceptance Criteria**:
  - [ ] Gateway connects to Redis Sentinel.
  - [ ] Gateway connects to Postgres Read/Write replicas.
- **Validation**:
  ```bash
  # Verify Redis connection
  python -c "from litellm_llmrouter.gateway import get_redis; print(get_redis().ping())"
  ```

##### A.3: OTel Pipeline Verification
- **Scope**: Verify trace propagation in [`src/litellm_llmrouter/observability.py`](src/litellm_llmrouter/observability.py:1).
- **Acceptance Criteria**:
  - [ ] Traces contain `service.name` = `litellm-gateway`.
  - [ ] Attributes visible in Tempo.
- **Validation**:
  ```python
  # Manual trace check
  from opentelemetry import trace
  span = trace.get_current_span()
  print(span.attributes.get("router.strategy"))
  ```

##### A.4: Security Hardening (SSRF & Auth)
- **Scope**:
  - Enforce [`python.validate_outbound_url()`](src/litellm_llmrouter/url_security.py:281) on all user-supplied URLs.
  - Protect admin routes with [`python.admin_api_key_auth()`](src/litellm_llmrouter/auth.py:111).
- **Acceptance Criteria**:
  - [ ] Requests to `169.254.169.254` blocked.
  - [ ] Admin routes return 401 without key.
- **Validation**:
  ```bash
  # Test SSRF
  curl -X POST http://localhost:4000/v1/agents -d '{"url": "http://169.254.169.254"}'
  # Expect 400/403
  ```

---

### Milestone B: Platform Readiness (P1)
**Goal:** Enable zero-downtime operations and deep visibility into routing logic.

| ID | Task | Owner Mode | Dependencies |
|----|------|------------|--------------|
| B.1 | **Config Sync Sidecar**<br>Hot-reload config from S3. | `code` | A.1 |
| B.2 | **Routing Visibility**<br>Add OTel attributes for ML decisions. | `code` | A.3 |
| B.3 | **Security Hardening**<br>SSRF protection & Admin Auth. | `code` | A.4 |
| B.4 | **Streaming Shutdown**<br>Graceful SIGTERM handling. | `code` | A.1 |

#### Detailed Work Items (Milestone B)

##### B.1: Config Sync Sidecar
- **Scope**: Implement sidecar to poll S3 and trigger SIGHUP.
- **Acceptance Criteria**:
  - [ ] Config change in S3 propagates to pod volume.
  - [ ] Gateway reloads config without dropping requests.
- **Validation**:
  ```bash
  # Simulate S3 update
  aws s3 cp config-v2.yaml s3://bucket/config.yaml
  # Check logs for reload
  kubectl logs -l app=litellm-gateway | grep "Config reloaded"
  ```

##### B.2: Routing Decision Visibility
- **Scope**: Instrument [`src/litellm_llmrouter/strategies.py`](src/litellm_llmrouter/strategies.py:1) to add span attributes.
- **Attributes**: `router.strategy`, `router.score`, `router.selected_model`.
- **Acceptance Criteria**:
  - [ ] 100% of routing decisions traced.
  - [ ] Attributes visible in Tempo.
- **Validation**:
  ```bash
  curl -v http://localhost:4000/_health/liveliness
  # Check Tempo/Jaeger for new trace
  ```

##### B.3: Security Hardening (SSRF & Auth)
- **Scope**: Update [`src/litellm_llmrouter/startup.py`](src/litellm_llmrouter/startup.py:275) to enforce security validation on LLM provider URLs.
- **Acceptance Criteria**:
  - [ ] All SSRF attempts fail.
  - [ ] Unauthorized admin access logs but fails.
- **Validation**:
  ```bash
  # Test SSRF validation
  python src/litellm_llmrouter/startup.py --validate --url "http://169.254.169.254"
  ```

##### B.4: Streaming-Aware Shutdown
- **Scope**: Update [`docker/entrypoint.sh`](docker/entrypoint.sh:1) trap SIGTERM, and [`src/litellm_llmrouter/startup.py`](src/litellm_llmrouter/startup.py:275) cleanup active streams.
- **Acceptance Criteria**:
  - [ ] Graceful SIGTERM terminates existing streams.
  - [ ] New streams blocked immediately upon shutdown.
- **Validation**:
  ```bash
  # Start long-running stream
  export STREAM_ID=$(python examples/get_stream_id.py)
  curl ... -H "X-MCP-STREAM-ID: $STREAM_ID"
  # Kill pod with SIGTERM handler
  kubectl delete pod <pod-id>
  # Verify cleanup logs
  kubectl logs -l app=litellm-gateway | grep "Graceful shutdown"
  ```

---

### Milestone C: MLOps & Maturity (P2)
**Goal:** Automate the feedback loop and ensure resilience at scale.

| ID | Task | Owner Mode | Dependencies |
|----|------|------------|--------------|
| C.1 | **MLOps Pipeline**<br>Trace -> Train -> Deploy loop. | `test-engineer` | B.2 |
| C.2 | **Circuit Breakers**<br>Degraded mode for Redis/DB. | `code` | A.2 |
| C.3 | **Autoscaling**<br>KEDA metrics for active streams. | `architect` | A.1 |

#### Detailed Work Items (Milestone C)

##### C.1: Automated Training Pipeline
- **Scope**: Create scripts to extract traces and retrain routers.
- **Acceptance Criteria**:
  - [ ] Training job runs successfully on trace data.
  - [ ] New model artifact produced and versioned.
- **Validation**:
  ```bash
  # Export traces from S3
  python examples/mlops/scripts/export_traces.py --bucket s3://logs/config-v1 \
                                              --base-name test_v1 \
                                              --output traces.jsonl
  # Run training
  python examples/mlops/scripts/train_router.py --input traces.jsonl \
                                             --model knn \
                                             --output repo/litellm_llmrouter/candidates/triggers/model_v2.pkl \
                                             --metric representative_accuracy
  ```

##### C.2: Circuit Breakers & Degraded Mode
- **Scope**: Implement resilience patterns in the Gateway to prevent the entire system from failing if a dependency (e.g., Redis) fails.
- **Acceptance Criteria**:
  - [ ] Gateway functions read-only when Redis unavailable.
  - [ ] Auth continues with local config when DB unavailable.
  - [ ] Degraded mode toggles correctly and logs issues.
- **Validation**:
  ```bash
  # Stop Redis
  docker stop redis
  # Verify gateway still responds to health check
  curl http://localhost:4000/_health/liveliness
  ```

##### C.3: Autoscaling (KEDA)
- **Scope**: Expose `active_streams_count` metric in [`/src/litellm_llmrouter/gateway/__init__.py`](src/litellm_llmrouter/gateway/__init__.py:763) for KEDA scalelib.
- **Acceptance Criteria**:
  - [ ] Active stream count available at `/metrics`.
  - [ ] HPA metrics endpoint receives correct values.
  - [ ] HPA makes correct scale decisions based on metrics.
- **Validation**:
  ```bash
  # Check metrics endpoint
  curl http://localhost:4000/metrics | grep active_streams_count
  # Verify KEDA successfully scales
  kubectl describe hpa litellm-gateway-hpa
  kubectl get hpa -w # watch scaling
  ```

---

### Milestone D+: Post-MVP Backlog
**Goal:** Advanced enterprise features, security hardening, and extensibility.

These items represent the next phase of development after the core cloud-native transition is complete.

| ID | Task | Owner Mode | Dependencies |
|----|------|------------|--------------|
| D.1 | **Control-Plane OIDC SSO + RBAC**<br>Replace admin break-glass key with true identity management. | `code` | A.4 |
| D.2 | **Distributed Registry State**<br>Persisted/shared-state registries for MCP/A2A for true HA. | `architect` | A.2 |
| D.3 | **MCP Protocol Parity**<br>Complete OAuth flows and remove remaining protocol skips. | `code` | - |
| D.4 | **Plugin Sandboxing**<br>Code signing, attestation, and isolation for plugins. | `architect` | - |
| D.5 | **Vector Store Extension**<br>Plugin-based model for custom vector DBs + parity verification. | `code` | - |
| D.6 | **Management UI**<br>Admin surface for rate limits, quotas, and audit logs. | `frontend-specialist` | D.1 |

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
| C.1 | C (P2) | MLOps Pipeline | `architect`, `Services Team` | ✅ Done | - | High |
| C.2 | C (P2) | Circuit Breakers | `code` | ✅ Done | - | Med |
| C.3 | C (P2) | Autoscaling | `code` | ✅ Done | - | Low |
| D.1 | D+ | OIDC SSO + RBAC | `code` | ⚪ Backlog | - | Med |
| D.2 | D+ | Distributed Registry | `architect` | ⚪ Backlog | - | High |
| D.3 | D+ | MCP Protocol Parity | `code` | ⚪ Backlog | - | Low |
| D.4 | D+ | Plugin Sandboxing | `architect` | ⚪ Backlog | - | High |
| D.5 | D+ | Vector Store Extension | `code` | ⚪ Backlog | - | Med |
| D.6 | D+ | Management UI | `frontend-specialist` | ⚪ Backlog | - | Low |

## Appendix: Code References

- **Auth Boundary**: [`src/litellm_llmrouter/auth.py`](src/litellm_llmrouter/auth.py)
- **SSRF Validation**: [`src/litellm_llmrouter/url_security.py`](src/litellm_llmrouter/url_security.py)
- **MCP Gateway**: [`src/litellm_llmrouter/mcp_gateway.py`](src/litellm_llmrouter/mcp_gateway.py)
- **A2A Gateway**: [`src/litellm_llmrouter/a2a_gateway.py`](src/litellm_llmrouter/a2a_gateway.py)
- **Startup Logic**: [`src/litellm_llmrouter/startup.py`](src/litellm_llmrouter/startup.py:275)
- **Backend Configuration**: `DATABASE_URL`, `LLMFS`, `REDIS_URL`, `LOG_DIR`, `ADMIN_API_KEYS`, `SSRF_PROTECTION_ENABLED`
- **Backend Estado**: [`src/litellm_llmrouter/backendmanager.py`](src/litellm_llmrouter/backendmanager.py:193), [`src/litellm_llmrouter/backendmanager.py`](src/litellm_llmrouter/backendmanager.py:216), [`src/audit_exporter/exporter.py`](src/audit_exporter/exporter.py:44), [`src/litellm_llmrouter/signatures.py`](src/litellm_llmrouter/signatures.py:176)
- **C2B and MCP**: [`src/litellm_llmrouter/backendmanager.py`](src/litellm_llmrouter/backendmanager.py:334), [`src/litellm_llmrouter/gateway/__init__.py`](src/litellm_llmrouter/gateway/__init__.py:121), [`src/litellm_llmrouter/plugins.py`](src/litellm_llmrouter/plugins.py:741), [`src/litellm_llmrouter/cli.py`](src/litellm_llmrouter/cli.py:253), [`src/litellm_llmrouter/backendmanager.py`](src/litellm_llmrouter/backendmanager.py:193), [`src/litellm_llmrouter/auth.py`](src/litellm_llmrouter/auth.py), [`src/litellm_llmrouter/url_security.py`](src/litellm_llmrouter/url_security.py), [`src/litellm_llmrouter/custom_proxies.py`](src/litellm_llmrouter/custom_proxies.py), [`src/litellm_llmrouter/mcp_client.py`](src/litellm_llmrouter/mcp_client.py), [`src/audit_exporter/exporter.py`](src/audit_exporter/exporter.py), [`src/litellm_llmrouter/logger/aws_connect.py`](src/litellm_llmrouter/logger/aws_connect.py), [`src/litellm_llmrouter/logger/audit_logger.py`](src/litellm_llmrouter/logger/audit_logger.py)
