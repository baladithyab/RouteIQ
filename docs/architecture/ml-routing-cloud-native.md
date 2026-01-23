# ML-Based Routing Architecture (Cloud-Native)

This document outlines the architecture for deploying Machine Learning (ML) based routing strategies (KNN, MLP, SVM, etc.) within the [LiteLLM Cloud-Native Enhancement Layer](../litellm-cloud-native-enhancements.md). It details the end-to-end lifecycle from trace data to production inference, emphasizing High Availability (HA) and "Moat-Mode" (air-gapped) compatibility.

## 1. ML Routing Lifecycle

The ML routing pipeline transforms historical traffic data into deployable routing artifacts that optimize for cost, latency, or quality.

### 1.1 Data Sources & Feature Extraction
The primary input for training is the **Trace History** of the gateway.

*   **Sources**:
    *   **OTEL/Jaeger Traces**: Captured via OpenTelemetry instrumentation. Contains `query`, `model`, `latency`, `status_code`.
    *   **Spend Logs**: LiteLLM's internal logging (Postgres/S3) for cost data.
*   **Extraction**:
    *   Scripts in [`examples/mlops/scripts/extract_jaeger_traces.py`](../../examples/mlops/scripts/extract_jaeger_traces.py:1) pull raw traces.
    *   [`examples/mlops/scripts/convert_traces_to_llmrouter.py`](../../examples/mlops/scripts/convert_traces_to_llmrouter.py:1) normalizes data into the `JSONL` format required by `llmrouter`.
    *   **Features**: Query text (embedded via `sentence-transformers`), historical latency, and success/failure signals.

### 1.2 Training & Artifact Registry
Training is decoupled from the serving layer to ensure stability.

*   **Training Jobs**: Run as ephemeral containers (e.g., Kubernetes Jobs, Docker Compose) using [`examples/mlops/scripts/train_router.py`](../../examples/mlops/scripts/train_router.py:1).
*   **Artifacts**:
    *   **Model File**: `.pkl` (sklearn) or `.pt` (PyTorch) files containing the trained router logic.
    *   **Config**: YAML metadata describing hyperparameters and label mappings.
*   **Registry**:
    *   **Standard**: S3-compatible object storage (AWS S3, MinIO, GCS).
    *   **Moat-Mode**: Local filesystem or internal MinIO instance.

### 1.3 Deployment & Hot Reload
The gateway loads models dynamically without restarting.

*   **Mechanism**: [`src/litellm_llmrouter/strategies.py`](../../src/litellm_llmrouter/strategies.py:1) implements `LLMRouterStrategyFamily`.
*   **Hot Reload**:
    *   The strategy checks for file modifications (`mtime`) or polls S3 at a configurable `reload_interval`.
    *   **Zero-Downtime**: The old model services requests while the new model loads in the background. Swapping is atomic via thread locks.

---

## 2. Reference Architectures

We support three primary deployment patterns ranging from simple file-based setups to complex, air-gapped enterprise environments.

### A. File/Artifact-Based (Sidecar Pattern)
*Best for: Kubernetes deployments, stateless gateways.*

*   **Architecture**:
    *   **Gateway Pod**: Runs LiteLLM Proxy + `llmrouter` logic.
    *   **Sidecar**: A lightweight agent (e.g., `aws s3 sync` loop or custom Go binary) watches an S3 bucket.
    *   **Shared Volume**: `ReadWriteMany` or `ReadWriteOnce` volume shared between Sidecar and Gateway.
*   **Flow**:
    1.  CI/CD pipeline pushes new `router.pkl` to S3.
    2.  Sidecar detects change, downloads to `/app/models/router.pkl`.
    3.  Gateway's `InferenceKNNRouter` detects file change and reloads.
*   **Enhancements**:
    *   **Config Sync**: Leverages the **Hot-Reload Config Sync** (P0) enhancement to coordinate model updates with general proxy config changes.

### B. DB-Config + Object Storage
*Best for: Dynamic environments, centralized management.*

*   **Architecture**:
    *   **Postgres**: Stores LiteLLM Proxy configuration (models, keys, routing rules) enabled via [`store_model_in_db`](../../reference/litellm/litellm/proxy/proxy_server.py:2557).
    *   **Object Storage (S3)**: Stores large ML artifacts (too big for DB).
*   **Flow**:
    1.  Admin updates routing config in Postgres to point to `s3://my-bucket/v2/router.pkl`.
    2.  Gateway polls DB, sees new config.
    3.  Gateway downloads the specific S3 object referenced in the DB config.
*   **Enhancements**:
    *   **Distributed Locks**: Ensures multiple gateway replicas don't stampede S3 simultaneously during a rollout.

### C. Moat-Mode (Air-Gapped)
*Best for: Defense, Finance, High-Security Enterprise.*

*   **Architecture**:
    *   **No External Internet**: All dependencies vendored.
    *   **Internal Services**: Self-hosted Postgres, Redis, and MinIO running within the VPC/Cluster.
    *   **Offline Training**: Training happens in a separate secure zone; artifacts are manually transferred or scanned before entering the production registry (MinIO).
*   **Flow**:
    1.  Artifacts are placed in the internal MinIO.
    2.  Gateway is configured to trust *only* the internal CA and registry.
    3.  **Fallback**: If ML router fails (e.g., corruption), system falls back to a static `weighted-round-robin` strategy defined in local YAML.
*   **Enhancements**:
    *   **Moat-Mode Hardening**: Strict validation of artifact signatures before loading.
    *   **Degraded Mode**: Circuit breakers ensure the gateway functions even if the internal MinIO is unreachable.

---

## 3. Integration with Enhancement Backlog

This architecture relies on and drives several items in the [Enhancement Backlog](../litellm-cloud-native-enhancements.md).

| Feature | Relevance to ML Routing |
| :--- | :--- |
| **Streaming-Aware Shutdown** (P1) | Critical for ML models. If a model is reloading or the pod is terminating, in-flight inferences (which may be part of a long-running stream) must complete. |
| **Distributed Locks** (P2) | Prevents race conditions when 50+ replicas try to download a new 500MB router model simultaneously. |
| **Observability Semantic Conventions** (P0) | The `route_with_observability` method in [`src/litellm_llmrouter/strategies.py`](../../src/litellm_llmrouter/strategies.py:645) emits standard OTel attributes (`router.strategy`, `router.score`) to debug ML decisions. |
| **Config Sync** (P0) | The mechanism that triggers the ML model reload is the same sidecar/polling logic used for general config. |

---

## 4. Validation

All ML routing deployments must pass the validation checklist defined in [`docs/VALIDATION_PLAN.md`](../VALIDATION_PLAN.md:1).

### Checklist
- [ ] **Hot Reload Test**: Update `.pkl` file while load testing; ensure 0 errors and eventual convergence to new routing logic.
- [ ] **Fallback Test**: Corrupt the model file; ensure gateway reverts to static routing or returns safe error (not crash).
- [ ] **Latency Budget**: ML inference overhead must be < 50ms (P99).
- [ ] **Observability**: Traces must show `llm.routing.selected_model` and `llm.routing.latency_ms`.
