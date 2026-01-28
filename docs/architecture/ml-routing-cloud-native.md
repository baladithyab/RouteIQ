# ML-Based Routing Architecture (Cloud-Native)

This document describes the closed-loop MLOps architecture for **RouteIQ Gateway** routing.

Key principle: the **Data Plane** (serving) is decoupled from the **Control Plane** (training, registry, and rollout). The data plane should continue serving traffic even if control-plane systems are unavailable.

## 1. Planes and Responsibilities

### 1.1 Data Plane (Gateway Runtime)

The data plane is the in-path serving runtime:

- Receives client requests and enforces security controls.
- Executes the **Routing Intelligence Layer** inline to select a provider/model per request.
- Proxies requests via LiteLLM.
- Emits telemetry (OpenTelemetry traces/logs) used for offline training.

Routing strategies are plugged into LiteLLM routing using [`python.LLMRouterStrategyFamily()`](../../src/litellm_llmrouter/strategies.py:318).

### 1.2 Control Plane (MLOps + Delivery)

The control plane is out-of-path:

- Extracts and labels training datasets.
- Trains routing models.
- Stores artifacts in a registry (MLflow and/or object storage).
- Rolls out new configuration and model artifacts to the data plane (CI/CD, sidecars, init containers, or rolling deploys).

## 2. ML Routing Lifecycle (Telemetry → Train → Registry → Rollout)

### 2.1 Data Sources & Feature Extraction

Primary input is the gateway's trace/log history.

- **Sources**:
  - **OTel/Jaeger traces** emitted by the data plane.
  - Optional outcome signals (success/failure, latency, cost) depending on what you log/export.
- **Extraction**:
  - [`examples/mlops/scripts/extract_jaeger_traces.py`](../../examples/mlops/scripts/extract_jaeger_traces.py:1) pulls traces from Jaeger.
  - [`examples/mlops/scripts/convert_traces_to_llmrouter.py`](../../examples/mlops/scripts/convert_traces_to_llmrouter.py:1) converts data into the JSONL + embeddings format expected by `llmrouter` training.

### 2.2 Training & Artifact Registry

Training is decoupled from serving to preserve stability.

- **Training jobs**: run as ephemeral containers (Kubernetes Jobs, Docker Compose, CI runners) using [`examples/mlops/scripts/train_router.py`](../../examples/mlops/scripts/train_router.py:1).
- **Artifacts** (examples):
  - **KNN (sklearn)**: `.pkl` model files.
  - **Torch-based routers**: `.pt` model files (router-dependent).
  - **Config/metadata**: YAML and JSON files describing hyperparameters, label mappings, and candidate model keys.
- **Registry / storage**:
  - The example pipeline uses MLflow + object storage (see [`examples/mlops/scripts/deploy_model.py`](../../examples/mlops/scripts/deploy_model.py:1)).

**Security note (pickle)**: Serving sklearn `.pkl` artifacts requires explicit opt-in. Pickle deserialization is disabled by default in the gateway and must be enabled via `LLMROUTER_ALLOW_PICKLE_MODELS=true`.

### 2.3 Deployment & Hot Reload (Data Plane)

The gateway loads routing artifacts from the **local filesystem** and can hot-reload when local artifacts change.

- **Mechanism**: [`python.LLMRouterStrategyFamily()`](../../src/litellm_llmrouter/strategies.py:318) checks a local artifact file's modification time (`mtime`) and reloads under a lock.
- **Reload trigger**: controlled by `hot_reload` + `reload_interval` in `routing_strategy_args`.
- **Important**: the routing strategy does **not** fetch artifacts from S3/GCS at request time. If you want object-storage-backed rollouts, use a delivery mechanism that updates the local file (sidecar sync, init container, or rolling deploy).
  - This repository also includes a one-time S3 model download in the container entrypoint when `LLMROUTER_MODEL_S3_BUCKET` and `LLMROUTER_MODEL_S3_KEY` are set (see [`docker/entrypoint.sh`](../../docker/entrypoint.sh:114)).

## 3. Reference Deployment Patterns (Model Delivery)

### A. File/Artifact Sync (Sidecar Pattern)

*Best for: Kubernetes deployments, stateless gateways.*

- **Gateway Pod**: runs the data plane.
- **Sidecar / init container**: syncs model artifacts from object storage to a shared volume.
- **Gateway reload**: the routing strategy observes local file changes (`mtime`) and reloads.

Example flow:

1. CI/CD publishes a new router artifact (e.g., `router.pkl`) to object storage.
2. Sidecar syncs it to the shared volume (e.g., `/app/models/router.pkl`).
3. The data plane reloads the local artifact on the next reload check.

### B. DB-Backed Config + Object Storage (Conceptual)

*Best for: centralized config management with separate artifact delivery.*

- Postgres can be used for centralized configuration (upstream LiteLLM capability).
- Large routing artifacts still live in object storage.
- A delivery mechanism (sidecar/CI/CD) must place artifacts onto the data plane filesystem; this repository does not automatically download routing artifacts from object storage based solely on DB config.

### C. Moat-Mode (Air-Gapped)

*Best for: defense/finance/high-security enterprise.*

- No external internet.
- Use internal-only services (self-hosted Postgres/Redis/MinIO) and controlled artifact transfer.
- **Hardening recommendation**: treat the artifact registry as a trusted boundary. Signature verification for routing artifacts is an optional control-plane hardening step and is not enforced by default in the gateway runtime.

## 4. Validation

All ML routing deployments should follow the validation checklist in [`docs/VALIDATION_PLAN.md`](../VALIDATION_PLAN.md:1).

### Checklist

- [ ] **Hot Reload Test**: replace the local routing artifact while load testing; ensure successful reload and eventual convergence to new routing behavior.
- [ ] **Fallback Test**: corrupt or remove the artifact; ensure the gateway fails safe (fallback strategy or safe error) rather than crashing.
- [ ] **Latency Budget**: routing overhead stays within an agreed budget.
- [ ] **Observability**: traces include routing decision metadata (e.g., `llm.routing.strategy`, `llm.routing.selected_model`, `llm.routing.latency_ms`).
