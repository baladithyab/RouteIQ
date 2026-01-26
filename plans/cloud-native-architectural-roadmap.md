# Cloud-Native Architectural Roadmap
## Transitioning LiteLLM + LLM-Router to Production-Grade Infrastructure

**Document Version:** 1.0  
**Last Updated:** 2026-01-26  
**Status:** Approved for Implementation

---

## Executive Summary

This roadmap provides a comprehensive, phased approach to transform the LiteLLM + LLM-Router combination into a robust, production-grade cloud-native system. The transformation emphasizes High Availability (HA), observability, MLOps excellence, and operational resilience while maintaining the system's core value proposition of intelligent ML-powered routing.

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

## Phased Roadmap

### Phase 0: Foundation (Weeks 1-3)
**Goal:** Establish baseline cloud-native primitives and observability

#### Deliverables

##### D0.1: Kubernetes Migration
- **Task**: Convert Docker Compose to Kubernetes manifests
- **Artifacts**:
  - [`deploy/k8s/base/deployment.yaml`] - Gateway Deployment with 3 replicas
  - [`deploy/k8s/base/service.yaml`] - ClusterIP service exposing port 4000
  - [`deploy/k8s/base/configmap.yaml`] - ConfigMap for [`config/config.yaml`](../config/config.yaml)
  - [`deploy/k8s/base/secrets.yaml`] - Sealed Secrets for API keys
- **Acceptance Criteria**:
  - [ ] Gateway pods start successfully with health checks passing
  - [ ] Service routes traffic to all pods via round-robin
  - [ ] ConfigMap changes trigger rolling update
  - [ ] Secrets mounted as env vars or files (per LiteLLM requirements)

##### D0.2: Persistent State Layer
- **Task**: Deploy PostgreSQL and Redis with HA configuration
- **Artifacts**:
  - [`deploy/k8s/state/postgres-statefulset.yaml`] OR RDS Terraform module
  - [`deploy/k8s/state/redis-cluster.yaml`] OR ElastiCache Terraform module
  - [`deploy/k8s/state/pvc-storage-class.yaml`] - StorageClass for local dev
- **Acceptance Criteria**:
  - [ ] PostgreSQL: Multi-AZ deployment OR RDS Multi-AZ enabled
  - [ ] PostgreSQL: Automated backups configured (daily, 7-day retention)
  - [ ] Redis: 3-node cluster OR ElastiCache cluster mode enabled
  - [ ] Redis: Persistence enabled (AOF) with snapshot frequency = 60s
  - [ ] Connection pooling configured (e.g., PgBouncer sidecar OR RDS Proxy)

##### D0.3: OpenTelemetry Pipeline
- **Task**: Standardize OTel instrumentation and export pipeline
- **Artifacts**:
  - [`deploy/k8s/otel/collector-daemonset.yaml`] - OTel Collector per node
  - [`config/otel-collector-config.yaml`](../config/otel-collector-config.yaml) - Updated for Tempo backend
  - Grafana dashboard JSON: [`dashboards/gateway-overview.json`]
- **Acceptance Criteria**:
  - [ ] Traces exported to Tempo with S3 backend configured
  - [ ] Metrics scraped by Prometheus (gateway pod `/metrics` endpoint)
  - [ ] Grafana dashboard shows: request rate, P50/P99 latency, error rate (RED metrics)
  - [ ] Trace attributes include: `llm.model`, `llm.provider`, token counts (per [`docs/observability.md`](../docs/observability.md))

##### D0.4: Ingress & TLS
- **Task**: Configure Ingress with TLS termination
- **Artifacts**:
  - [`deploy/k8s/ingress/ingress.yaml`] - Ingress resource with TLS
  - [`deploy/k8s/ingress/certificate.yaml`] - cert-manager Certificate OR ACM integration
- **Acceptance Criteria**:
  - [ ] HTTPS endpoint accessible at `https://gateway.example.com`
  - [ ] TLS certificate auto-renewed (cert-manager OR ACM)
  - [ ] HTTP → HTTPS redirect configured
  - [ ] Health check endpoint `/health/readiness` returns 200

**Priority Backlog (Phase 0)**

| ID | Item | Priority | Acceptance Criteria |
|----|------|----------|---------------------|
| P0-01 | Kubernetes Deployment manifests | P0 | Pods start with readiness probe passing |
| P0-02 | PostgreSQL HA setup (RDS OR StatefulSet) | P0 | Automatic failover tested (kill primary) |
| P0-03 | Redis Cluster setup | P0 | Cluster survives single-node failure |
| P0-04 | OTel Collector deployment | P0 | Traces visible in Tempo UI within 30s |
| P0-05 | Ingress with TLS | P0 | HTTPS request completes successfully |
| P0-06 | Basic Grafana dashboard | P1 | Dashboard shows live metrics (refresh interval 5s) |

---

### Phase 1: Hot Reload & Config Management (Weeks 4-6)
**Goal:** Implement zero-downtime configuration and model updates

#### Deliverables

##### D1.1: Config Sync Sidecar
- **Task**: Implement sidecar to sync config from S3/MinIO to gateway pods
- **Artifacts**:
  - [`src/config_sync/sync_agent.py`] - Python/Go agent watching S3 bucket
  - [`deploy/k8s/gateway/deployment-sidecar.yaml`] - Updated Deployment with sidecar
  - [`config/config-sync.yaml`] - Sidecar configuration (poll interval, validation schema)
- **Implementation Pattern**:
  1. Sidecar polls S3 bucket every 60s (configurable via `CONFIG_SYNC_INTERVAL`)
  2. On ETag change, downloads `config.yaml` to shared EmptyDir volume
  3. Validates config against schema (JSON Schema OR Pydantic model)
  4. If valid, triggers SIGHUP to gateway process for reload
  5. If invalid, logs error and retains current config
- **Acceptance Criteria**:
  - [ ] Config update in S3 propagates to all pods within 120s (2x poll interval)
  - [ ] Invalid config rejected with structured error log
  - [ ] Zero dropped requests during config reload (tested with load test: 100 req/s)
  - [ ] Sidecar health endpoint `/health/config-sync` returns timestamp of last successful sync

##### D1.2: Model Sync with Distributed Locks
- **Task**: Implement model artifact sync with Redis distributed locks
- **Artifacts**:
  - [`src/model_sync/sync_agent.py`] - Agent for model download with lock acquisition
  - Updated [`src/litellm_llmrouter/strategies.py`](../src/litellm_llmrouter/strategies.py) - Atomic model swap logic
  - [`deploy/k8s/gateway/deployment-model-sidecar.yaml`] - Model sync sidecar
- **Implementation Pattern**:
  1. Sidecar polls model registry (S3) every 300s (per [`docs/hot-reloading.md`](../docs/hot-reloading.md))
  2. On new model version detected, acquires Redis lock: `model-sync:{model_name}`
  3. Downloads model to temp location, validates signature (optional, Moat-Mode)
  4. Moves to `/app/models/{model_name}.pkl` (atomic rename)
  5. Releases lock, main container detects file change via inotify/mtime
  6. InferenceKNNRouter reloads model (thread-safe swap per [`docs/architecture/ml-routing-cloud-native.md`](../docs/architecture/ml-routing-cloud-native.md))
- **Acceptance Criteria**:
  - [ ] Model update tested with 50 replica deployment (no stampede to S3)
  - [ ] Lock acquisition/release logged with trace_id for correlation
  - [ ] Model reload latency < 5s (tested with 500MB .pkl file)
  - [ ] Fallback to previous model if new model load fails (logged as ERROR)

##### D1.3: Canary Config Rollout
- **Task**: Implement canary rollout for config changes (optional P1)
- **Artifacts**:
  - [`scripts/canary_rollout.sh`] - Script to update config in stages
  - Updated Ingress with weighted routing OR Argo Rollouts integration
- **Implementation Pattern**:
  1. Upload new config to S3 with version tag: `config-v2.yaml`
  2. Update 1 pod (10%) to point to `config-v2.yaml` via env var override
  3. Monitor error rate for 5 minutes
  4. If error_rate < threshold (default 1%), proceed to 50% → 100%
  5. If error_rate exceeds threshold, rollback to `config-v1.yaml`
- **Acceptance Criteria**:
  - [ ] Canary rollout script tested with intentionally broken config
  - [ ] Automatic rollback triggered within 60s of error threshold breach
  - [ ] Rollout logged to audit trail (PostgreSQL OR S3)

**Priority Backlog (Phase 1)**

| ID | Item | Priority | Acceptance Criteria |
|----|------|----------|---------------------|
| P1-01 | Config Sync Sidecar implementation | P0 | Config change propagates within 120s, zero dropped requests |
| P1-02 | Model Sync with Redis locks | P0 | 50 replicas sync without S3 rate limit errors |
| P1-03 | Schema validation for config | P0 | Invalid config rejected before reload |
| P1-04 | Canary rollout automation | P1 | Rollback executed within 60s of failure |
| P1-05 | Observability for sync operations | P1 | Trace spans for config/model sync in Tempo |

---

### Phase 2: Full-Stack Observability & SLOs (Weeks 7-10)
**Goal:** Achieve comprehensive visibility and establish SLO tracking

#### Deliverables

##### D2.1: Routing Decision Visibility
- **Task**: Instrument LLMRouter with OTel attributes for routing decisions
- **Artifacts**:
  - Updated [`src/litellm_llmrouter/strategies.py`](../src/litellm_llmrouter/strategies.py) - Add OTel span attributes
  - [`docs/observability-routing.md`] - Documentation for routing trace analysis
- **OTel Attributes** (per [`docs/litellm-cloud-native-enhancements.md`](../docs/litellm-cloud-native-enhancements.md)):
  ```python
  span.set_attribute("router.strategy", "llmrouter-knn")
  span.set_attribute("router.score", 0.87)
  span.set_attribute("router.candidates", ["gpt-4", "claude-3-opus", "claude-haiku"])
  span.set_attribute("router.selected_model", "claude-haiku")
  span.set_attribute("router.latency_ms", 23.5)
  ```
- **Acceptance Criteria**:
  - [ ] 100% of routing decisions include attributes in traces
  - [ ] Grafana dashboard query: "Top models by selection frequency (24h)"
  - [ ] Tempo TraceQL query: `{ router.strategy="llmrouter-knn" && router.score > 0.8 }`
  - [ ] Routing latency P50/P99 tracked per strategy

##### D2.2: Multi-Replica Trace Correlation
- **Task**: Ensure `trace_id` propagated across Ingress → Gateway → DB
- **Artifacts**:
  - Updated Nginx config: [`config/nginx.conf`](../config/nginx.conf) - Add `traceparent` header
  - Updated gateway code: Middleware to extract and propagate `trace_id` to structured logs
  - [`dashboards/trace-correlation.json`] - Grafana dashboard with Loki + Tempo correlation
- **Acceptance Criteria**:
  - [ ] Single request traced end-to-end: Nginx → Gateway → LLM Provider
  - [ ] Structured logs include `trace_id` field (JSON logs)
  - [ ] Grafana Explore: Click trace span → Jump to correlated logs in Loki

##### D2.3: SLO Dashboards & Alerting
- **Task**: Define and monitor Service Level Objectives
- **SLOs Defined**:
  | Objective | Target | Measurement Window |
  |-----------|--------|-------------------|
  | Availability | 99.95% | 30-day rolling |
  | Request Success Rate | 99.9% | 1-hour window |
  | P99 Latency | < 500ms | 5-minute window |
  | ML Routing Latency | < 50ms | 5-minute window |
- **Artifacts**:
  - [`dashboards/slo-dashboard.json`] - Grafana SLO dashboard
  - [`alerts/slo-alerts.yaml`] - Prometheus AlertManager rules
- **Acceptance Criteria**:
  - [ ] SLO dashboard shows current status (green/yellow/red)
  - [ ] Alert fired when P99 latency > 500ms for 5 consecutive minutes
  - [ ] Error budget calculation visible (allowed error minutes remaining)

##### D2.4: Durable Audit Export
- **Task**: Export audit logs to S3 for compliance
- **Artifacts**:
  - [`src/audit_exporter/exporter.py`] - Background task for log export
  - [`deploy/k8s/audit/cronjob.yaml`] - CronJob to batch export logs daily
- **Implementation Pattern**:
  1. Background task queries PostgreSQL for audit logs (last 24h)
  2. Batches into JSONL format with schema version
  3. Uploads to S3: `s3://audit-logs/{year}/{month}/{day}/audit-{timestamp}.jsonl`
  4. Logs uploaded with SSE-S3 encryption
  5. S3 lifecycle policy transitions to Glacier after 90 days
- **Acceptance Criteria**:
  - [ ] Audit logs exported to S3 within 24h of creation
  - [ ] Logs immutable (S3 Object Lock OR versioning + deletion protection)
  - [ ] Export failures trigger alert (AlertManager)

**Priority Backlog (Phase 2)**

| ID | Item | Priority | Acceptance Criteria |
|----|------|----------|---------------------|
| P2-01 | Router decision OTel attributes | P0 | 100% trace coverage with routing metadata |
| P2-02 | Multi-replica trace correlation | P1 | End-to-end trace spans across all services |
| P2-03 | SLO dashboard creation | P1 | Dashboard deployed, alerts configured |
| P2-04 | Audit log export to S3 | P1 | Daily batch export with encryption |
| P2-05 | Alert escalation policy | P2 | PagerDuty/Slack integration for P0 alerts |

---

### Phase 3: MLOps Automation & Resilience (Weeks 11-16)
**Goal:** Close the loop from data → training → deployment; harden for production

#### Deliverables

##### D3.1: Automated Training Pipeline
- **Task**: Implement end-to-end MLOps pipeline
- **Artifacts**:
  - [`examples/mlops/scripts/export_traces.py`] - Export OTEL traces to S3 as training data
  - [`examples/mlops/scripts/train_router.py`](../examples/mlops/scripts/train_router.py) - Parameterized training script
  - [`examples/mlops/scripts/deploy_model.py`](../examples/mlops/scripts/deploy_model.py) - Upload trained model to registry
  - [`deploy/k8s/mlops/training-job.yaml`] - Kubernetes Job for training
  - [`deploy/k8s/mlops/model-registry.yaml`] - MLflow deployment OR S3-only registry
- **Pipeline Flow** (as per [`docs/mlops-training.md`](../docs/mlops-training.md)):
  ```mermaid
  graph LR
    A[Trace Export CronJob] -->|Daily| B[S3: traces/YYYY-MM-DD/]
    B --> C[Training Job Triggered]
    C --> D[Train KNN/MLP/Graph Router]
    D --> E[Register in MLflow]
    E --> F[Deploy to S3: models/knn-router-v{version}.pkl]
    F --> G[Canary Rollout 5% -> 100%]
    G --> H[Monitor Routing Quality]
    H -->|If quality drops| I[Rollback to Previous Version]
  ```
- **Acceptance Criteria**:
  - [ ] Training job executes on schedule (weekly OR on-demand)
  - [ ] Model artifacts versioned with metadata: training date, dataset size, metrics
  - [ ] MLflow UI shows experiment runs with accuracy/F1 metrics
  - [ ] Automated deployment to staging environment (separate K8s namespace)

##### D3.2: Model Rollback Mechanism
- **Task**: Implement one-click rollback for model versions
- **Artifacts**:
  - [`scripts/rollback_model.sh`] - Script to revert to previous model version
  - Updated Config Sync to tag deployed model version in PostgreSQL
- **Implementation Pattern**:
  1. Config includes `model_version: v42` field
  2. On rollback trigger, script updates config to `model_version: v41`
  3. Config Sync propagates change to all pods
  4. Model Sync downloads v41 artifact from S3
  5. Gateway reloads model (hot-reload mechanism from Phase 1)
- **Acceptance Criteria**:
  - [ ] Rollback tested: v2 → v1 completes within 300s (2x poll interval + download time)
  - [ ] Active requests continue using old model until new model loaded
  - [ ] Rollback event logged to audit trail with timestamp + user

##### D3.3: Experiment Tracking Integration
- **Task**: Integrate MLflow for experiment management
- **Artifacts**:
  - [`deploy/k8s/mlops/mlflow-server.yaml`] - MLflow server with S3 backend
  - Updated training scripts to log metrics to MLflow
- **MLflow Configuration**:
  - **Backend Store**: PostgreSQL (metadata: experiments, runs, params, metrics)
  - **Artifact Store**: S3 (models, plots, datasets)
- **Acceptance Criteria**:
  - [ ] MLflow UI accessible at `https://mlflow.example.com`
  - [ ] Training run logs: hyperparameters, accuracy, training time
  - [ ] Model artifacts downloadable from MLflow UI
  - [ ] Model lineage tracked: training dataset → model version → deployment timestamp

##### D3.4: Streaming-Aware Shutdown
- **Task**: Implement graceful shutdown for LLM streaming requests
- **Artifacts**:
  - Updated [`docker/entrypoint.sh`] - Trap SIGTERM signal
  - [`src/shutdown_handler.py`] - Middleware to track active streams
- **Implementation Logic** (per [`docs/litellm-cloud-native-enhancements.md`](../docs/litellm-cloud-native-enhancements.md)):
  ```python
  def handle_sigterm():
      logging.info("SIGTERM received, entering graceful shutdown")
      stop_accepting_new_requests()  # Health check returns 503
      wait_for_active_streams(timeout=30s)  # Wait for streams to complete
      if active_streams_remain():
          logging.warning(f"Force-closing {len(active_streams)} streams")
      sys.exit(0)
  ```
- **Acceptance Criteria**:
  - [ ] Streaming request started before SIGTERM completes successfully
  - [ ] New requests rejected with HTTP 503 after SIGTERM
  - [ ] Pod terminates within 35s (30s grace period + 5s cleanup)
  - [ ] Zero error logs related to "connection closed unexpectedly"

##### D3.5: Degraded Mode & Circuit Breakers
- **Task**: Implement fallback logic for dependency failures
- **Failure Scenarios**:
  | Failure | Degraded Mode Behavior |
  |---------|------------------------|
  | Redis down | Use in-memory LRU cache (max 100MB), disable rate limiting |
  | PostgreSQL down | Use local fallback API keys (read from ConfigMap), disable audit logging |
  | S3 unreachable | Continue with last known good config/model, emit warning metrics |
  | LLM Provider timeout | Fallback to next candidate model (per LiteLLM retry logic) |
- **Artifacts**:
  - [`src/resilience/circuit_breakers.py`] - Circuit breaker logic
  - [`config/degraded-mode.yaml`] - Fallback configuration
- **Acceptance Criteria**:
  - [ ] Redis failure: Gateway continues serving with in-memory cache
  - [ ] PostgreSQL failure: Gateway serves requests for 1 hour with cached keys
  - [ ] Circuit breaker opens after 5 consecutive failures, half-open after 60s
  - [ ] Degraded mode logged to metrics: `gateway_degraded_mode{service="redis"} = 1`

##### D3.6: Autoscaling Based on Active Streams
- **Task**: Configure HPA/KEDA to scale on custom metrics
- **Artifacts**:
  - [`deploy/k8s/autoscaling/hpa-custom-metrics.yaml`] - HPA with custom metrics
  - [`deploy/k8s/autoscaling/keda-scaledobject.yaml`] - KEDA ScaledObject
  - Updated gateway to expose `active_streams_count` metric
- **Scaling Logic**:
  - **Scale Up**: When `active_streams_per_pod > 100` (sustained for 30s)
  - **Scale Down**: When `active_streams_per_pod < 20` (sustained for 300s)
  - **Min Replicas**: 3 (always)
  - **Max Replicas**: 50
- **Acceptance Criteria**:
  - [ ] Load test: 1000 concurrent streams triggers scale-up to 10+ pods
  - [ ] Scale-up latency < 90s (pod startup + readiness)
  - [ ] Scale-down respects graceful shutdown (no stream termination)

**Priority Backlog (Phase 3)**

| ID | Item | Priority | Acceptance Criteria |
|----|------|----------|---------------------|
| P3-01 | Automated training pipeline | P0 | Weekly training job succeeds, model registered in MLflow |
| P3-02 | Model rollback mechanism | P0 | Rollback completes within 300s, zero dropped requests |
| P3-03 | Streaming-aware shutdown | P1 | Active streams complete during pod termination |
| P3-04 | Degraded mode circuit breakers | P1 | Gateway operational during Redis/PostgreSQL outage |
| P3-05 | Custom metrics HPA/KEDA | P2 | Autoscaling based on active streams validated |
| P3-06 | Experiment tracking (MLflow) | P1 | MLflow UI accessible, experiments logged |
| P3-07 | Model A/B testing framework | P2 | 10% traffic to model-v2, 90% to model-v1 |

---

## Kubernetes Manifests & Primitives

### Core Resources

#### Deployment
```yaml
# deploy/k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-gateway
  labels:
    app: litellm-gateway
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero downtime
  selector:
    matchLabels:
      app: litellm-gateway
  template:
    metadata:
      labels:
        app: litellm-gateway
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "4000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: litellm-gateway
      terminationGracePeriodSeconds: 35  # Allow streams to complete
      initContainers:
      - name: wait-for-postgres
        image: busybox:1.36
        command: ['sh', '-c', 'until nc -z postgres 5432; do sleep 2; done']
      containers:
      - name: gateway
        image: ghcr.io/baladithyab/litellm-llm-router:v1.2.0
        ports:
        - containerPort: 4000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: litellm-secrets
              key: database-url
        - name: LITELLM_CONFIG_PATH
          value: /app/config/config.yaml
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health/liveliness
            port: 4000
          initialDelaySeconds: 15
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 4000
          initialDelaySeconds: 10
          periodSeconds: 10
      - name: config-sync
        image: config-sync-agent:v1.0
        env:
        - name: S3_BUCKET
          value: "litellm-configs"
        - name: S3_KEY
          value: "prod/config.yaml"
        - name: POLL_INTERVAL
          value: "60"
        volumeMounts:
        - name: config
          mountPath: /app/config
      - name: model-sync
        image: model-sync-agent:v1.0
        env:
        - name: S3_BUCKET
          value: "litellm-models"
        - name: MODEL_PATH
          value: "knn-router.pkl"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: litellm-secrets
              key: redis-url
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: config
        emptyDir: {}
      - name: models
        emptyDir: {}
```

#### Service
```yaml
# deploy/k8s/base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: litellm-gateway
  labels:
    app: litellm-gateway
spec:
  type: ClusterIP
  ports:
  - port: 4000
    targetPort: 4000
    protocol: TCP
    name: http
  selector:
    app: litellm-gateway
```

#### HorizontalPodAutoscaler
```yaml
# deploy/k8s/autoscaling/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: litellm-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: litellm-gateway
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: active_llm_streams
      target:
        type: AverageValue
        averageValue: "100"
```

---

## Operational Patterns

### Config Management
- **Source of Truth**: S3 bucket `s3://litellm-configs/{env}/config.yaml`
- **Versioning**: S3 versioning enabled, tags for semantic versions (v1.0.0)
- **Validation**: JSON Schema validation before propagation
- **Propagation**: Config Sync sidecar polls every 60s, atomic update via SIGHUP
- **Rollback**: Update S3 object to previous version ID

### Secrets Management
- **Storage**: AWS Secrets Manager OR HashiCorp Vault
- **Rotation**: Automatic rotation every 90 days
- **Access**: Kubernetes ExternalSecrets Operator syncs to K8s Secrets
- **Injection**: Secrets mounted as env vars OR files per LiteLLM requirements

### Model Lifecycle
1. **Training**: Kubernetes Job with GPU (optional)
2. **Registration**: Upload to MLflow (metadata) + S3 (artifacts)
3. **Deployment**:
   - Staging: Update staging namespace config
   - Canary: 5% traffic to new model for 1 hour
   - Full: 100% traffic if error rate < 1%
4. **Monitoring**: Track routing quality metrics (accuracy proxy via success rate)
5. **Rollback**: Revert config to previous model version, propagates within 300s

---

## Prioritized Backlog: Consolidated View

| ID | Phase | Item | Priority | Dependencies | Acceptance Criteria |
|----|-------|------|----------|--------------|---------------------|
| **P0-01** | 0 | Kubernetes Deployment manifests | **P0** | None | Pods start, health checks pass |
| **P0-02** | 0 | PostgreSQL HA (RDS Multi-AZ) | **P0** | None | Automatic failover tested |
| **P0-03** | 0 | Redis Cluster setup | **P0** | None | Survives single-node failure |
| **P0-04** | 0 | OTel Collector deployment | **P0** | None | Traces visible in Tempo <30s |
| **P0-05** | 0 | Ingress with TLS | **P0** | None | HTTPS endpoint functional |
| **P1-01** | 1 | Config Sync Sidecar | **P0** | P0-01 | Config propagates <120s, zero dropped requests |
| **P1-02** | 1 | Model Sync with Redis locks | **P0** | P0-03, P1-01 | 50 replicas sync without rate limits |
| **P2-01** | 2 | Router decision OTel attributes | **P0** | P0-04 | 100% trace coverage with routing metadata |
| **P3-01** | 3 | Automated training pipeline | **P0** | P2-01 | Weekly training job succeeds |
| **P3-02** | 3 | Model rollback mechanism | **P0** | P1-02 | Rollback <300s, zero dropped requests |
| **P0-06** | 0 | Basic Grafana dashboard | **P1** | P0-04 | Dashboard shows RED metrics |
| **P1-03** | 1 | Config schema validation | **P0** | P1-01 | Invalid config rejected |
| **P1-04** | 1 | Canary config rollout | **P1** | P1-01 | Automatic rollback on errors |
| **P1-05** | 1 | Observability for sync ops | **P1** | P0-04, P1-01 | Trace spans for sync operations |
| **P2-02** | 2 | Multi-replica trace correlation | **P1** | P0-04 | End-to-end traces across services |
| **P2-03** | 2 | SLO dashboard & alerting | **P1** | P0-06 | Alerts fire on SLO breach |
| **P2-04** | 2 | Audit log export to S3 | **P1** | P0-02 | Daily batch export with encryption |
| **P3-03** | 3 | Streaming-aware shutdown | **P1** | P0-01 | Active streams complete on SIGTERM |
| **P3-04** | 3 | Degraded mode circuit breakers | **P1** | P0-02, P0-03 | Gateway functional during dependency outage |
| **P3-06** | 3 | Experiment tracking (MLflow) | **P1** | P3-01 | MLflow UI accessible, runs logged |
| **P2-05** | 2 | Alert escalation (PagerDuty) | **P2** | P2-03 | P0 alerts trigger pages |
| **P3-05** | 3 | Custom metrics HPA/KEDA | **P2** | P0-01 | Scales based on active_streams |
| **P3-07** | 3 | Model A/B testing framework | **P2** | P3-02 | Traffic split validated (10%/90%) |

---

## Success Criteria & Validation

### Phase 0 Validation
- [ ] Deploy to staging K8s cluster (3 gateway replicas)
- [ ] Execute load test: 1000 req/min sustained for 1 hour
- [ ] Verify P99 latency < 500ms
- [ ] Trace end-to-end request in Tempo UI
- [ ] Simulate pod failure: kill 1 replica, verify traffic redirected

### Phase 1 Validation
- [ ] Update config in S3, verify propagation to all pods <120s
- [ ] Upload new model to S3, verify hot-reload with zero errors
- [ ] Deploy intentionally invalid config, verify rejection
- [ ] Load test during config reload: zero dropped requests
- [ ] Verify Redis distributed lock prevents S3 stampede (test with 50 replicas)

### Phase 2 Validation
- [ ] Execute 100 requests, verify 100% have routing metadata in traces
- [ ] Query Grafana: "Show requests where router.score > 0.8"
- [ ] Verify SLO dashboard shows current availability percentage
- [ ] Trigger P99 latency alert by load testing with slow backend
- [ ] Verify audit logs exported to S3 within 24 hours

### Phase 3 Validation
- [ ] Execute training pipeline end-to-end: trace export → train → deploy
- [ ] Register model in MLflow, verify metadata (accuracy, F1 score)
- [ ] Rollback model version, verify completion <300s
- [ ] Simulate Redis failure, verify gateway continues with in-memory cache
- [ ] Load test with 1000 concurrent streams, verify autoscaling triggers
- [ ] Send SIGTERM to pod with active stream, verify stream completes

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **S3 outage prevents config/model updates** | High | S3 versioning + cross-region replication; gateway caches last known good config/model |
| **Model hot-reload causes memory spike** | Medium | Load new model in separate process, memory check before swap; rollback on OOM |
| **PostgreSQL failover causes 30s downtime** | Medium | Use RDS Proxy for connection pooling; implement degraded mode with cached keys |
| **OTel Collector becomes bottleneck** | Low | Deploy as DaemonSet (1 per node); configure backpressure handling |
| **Training pipeline produces bad model** | High | Automated quality checks (accuracy threshold); canary deployment with rollback |
| **Redis cluster split-brain** | Medium | Use Redis Sentinel with odd number of sentinels (3+); automatic failover |

---

## Appendix: Reference Links

### Existing Documentation
- [Architecture Overview](../docs/architecture/overview.md)
- [ML Routing Cloud-Native Architecture](../docs/architecture/ml-routing-cloud-native.md)
- [High Availability Setup](../docs/high-availability.md)
- [Observability Guide](../docs/observability.md)
- [Hot Reloading Guide](../docs/hot-reloading.md)
- [LiteLLM Cloud-Native Enhancements](../docs/litellm-cloud-native-enhancements.md)
- [MLOps Training Guide](../docs/mlops-training.md)

### Configuration Files
- [Main Config](../config/config.yaml)
- [LLM Candidates](../config/llm_candidates.json)
- [OTel Collector Config](../config/otel-collector-config.yaml)
- [Nginx Config](../config/nginx.conf)

### Deployment Artifacts
- [HA Docker Compose](../docker-compose.ha.yml)
- [OTel Docker Compose](../docker-compose.otel.yml)

### Code References
- [Custom Routing Strategies](../src/litellm_llmrouter/strategies.py)
- [Training Scripts](../examples/mlops/scripts/train_router.py)
- [Validation Scripts](../scripts/validate_routing.py)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-26 | Architecture Team | Initial roadmap based on existing documentation |

**Approval Required From:**
- [ ] Platform Engineering Lead
- [ ] MLOps Lead  
- [ ] Security Team (for Secrets Management & Moat-Mode patterns)
- [ ] SRE Team (for SLO definitions & alerting)

**Next Review Date:** 2026-02-26 (4 weeks)
