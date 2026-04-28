# Technical Roadmap

This document outlines the technical direction for RouteIQ Gateway. It is
reality-synced to **v1.0.0rc1** (2026-04-02) and reflects the decisions in
`docs/architecture/vsr-vs-routeiq-decision-v3.md`.

For historical strategic deliberation, see:
- [v1 decision doc](../docs/architecture/vsr-vs-routeiq-decision.md) — 2026-04-27 initial 5-agent research
- [v2 decision doc](../docs/architecture/vsr-vs-routeiq-decision-v2.md) — 2026-04-27 8-investigator + 4-deliberator team
- [v3 decision doc](../docs/architecture/vsr-vs-routeiq-decision-v3.md) — 2026-04-27 reality-sync after v1.0.0rc1 reconciliation

For the product-level "why," see [Product Vision](../docs/product-vision.md)
and [Evidence Console Design](../docs/architecture/evidence-console-design.md).

---

## Shipped in v1.0.0rc1 (2026-04-02)

The 22-commit rearchitecture squash delivered most of what earlier roadmap
versions scheduled for Q1–Q2 2026:

- **Own FastAPI app** (ADR-0012) — RouteIQ owns the ASGI composition root
- **Plugin-based routing** (ADR-0002) — `CustomRoutingStrategyBase` replaces the deleted monkey-patch; multi-worker safe
- **MCP consolidation** (ADR-0003, ADR-0017) — custom JSON-RPC / SSE / parity surfaces deleted; upstream LiteLLM used
- **Pydantic `BaseSettings`** (ADR-0013) — 124+ `os.environ.get()` call sites collapsed
- **K8s-native leader election** (ADR-0015) — Lease API primary, Redis fallback
- **Centroid zero-config routing** (ADR-0010), **personalized routing** (ADR-0025), **Router-R1** — native ML routing stack
- **Context optimizer** (ADR-0024) — 6 lossless token-reduction transforms, 30–70% reduction
- **Governance layer** (ADR-0020, ADR-0022) — workspaces, API keys, usage policies
- **Guardrail policy pipeline** (ADR-0023) — 14 check types with deny/log/alert
- **OIDC / SSO** (ADR-0008) — Keycloak, Auth0, Okta, Azure AD
- **Service discovery** (ADR-0011) — graceful degradation based on optional-dep probes
- **Prompt management** — CRUD + versioning + A/B + rollback + import/export
- **Evaluation pipeline** — COLLECT / EVALUATE / AGGREGATE / FEEDBACK with LLM-as-judge; `/api/v1/routeiq/eval/*`
- **Admin UI** — 6 pages: Dashboard, Routing Config, Governance, Guardrails, Prompts, Observability
- **Helm chart** — Grafana dashboard, PrometheusRule, ServiceMonitor, leader-election RBAC, PDB, NetworkPolicy, ExternalSecret
- **OTel GenAI conventions** (ADR-0019)
- **Disaggregated UI** (ADR-0018) — independent UI container image
- **25 ADRs** (0001–0025)

---

## Q2 2026: Evidence gap closure & v1.0 GA — the remaining 20%

**Goal:** Close the gap between v1.0.0rc1 (80% of the evidence story) and
the product vision's "routing you can prove." See v3 decision doc §"The
remaining 20%" for the eight gaps.

### M1 (weeks 1–4): CPTC plumbing

*   [ ] **P0: Per-request CPTC attribution** — OTel span schema in `telemetry_contracts.py`; trace-joined (prompt_tokens × input_price) + (completion_tokens × output_price) + routing_overhead + judge_quality + latency
*   [ ] **P0: `routing_backend` span attribute** — single attribute emitted from `router_decision_callback.py`; required for experiment rubrics
*   [ ] **P0: Dispatcher-sprawl guardrail** — arch test / lint rule enforcing pure-function dispatcher discipline before any additional backend arms land
*   [ ] **P1: Streaming-aware shutdown** — graceful termination for long-running LLM streams (carryover from earlier roadmap)

### M2 (weeks 5–8): Experiment harness + Evidence UI

*   [ ] **P0: Tenant-level routing-backend A/B harness** — distinct from the shipped prompt A/B; hash-based traffic split on `tenant_id + request_id`; arm stamped as `routing_arm` span attribute
*   [ ] **P0: Evidence Console UI pages** — Experiments (list/create/assign), Evidence (CPTC time-series per backend), Judge (verdict review + correction); complements the shipped Governance/Guardrails/Prompts/Observability quartet
*   [ ] **P0: Evidence API endpoints** — `/api/v1/routeiq/experiments/*`, `/evidence/*`, `/judge/*` (distinct from the per-model `/eval/*` that already ships)

### M3 (weeks 9–12): MLOps feedback loop

*   [ ] **P0: Judge-verdict → offline dataset** — Parquet export from `eval_pipeline.py` samples to S3/GCS
*   [ ] **P0: Nightly LoRA training job** — K8s GPU Job (HuggingFace trainer / PEFT) producing LoRA safetensors + signed manifest
*   [ ] **P0: Hot-reload LoRA weights** — strategies receive new weights via `model_artifacts.py` trust chain without restart
*   [ ] **P1: Cold-start task classifier** — narrow scope: pre-personalization feature extraction + modality gating for image/audio/embedding routing

### M4 (weeks 13–16): Measure, decide, GA

*   [ ] **P0: Collect ≥6 weeks of tenant-experiment data** — validate the M4 promotion rubric on real traffic
*   [ ] **P0: VSR integration decision** — gate based on judge-verdict CPTC uplift vs *RouteIQ's shipped trained arm*, **not** vs LiteLLM auto_router. Thresholds: ≥15% → integrate as opt-in backend; 5–15% → product judgment; <5% → formally close the VSR path and port selected BERT+LoRA weights into RouteIQ's plugin system instead (weights are Apache-licensed; Envoy runtime is not required)
*   [ ] **P0: Breaking-changes migration guide** — `MIGRATION.md` documenting v0.2.0 → v1.0.0rc1 deltas (MCP surfaces deleted, plugin-strategy signature, governance state file paths)
*   [ ] **P0: v1.0.0 GA tag**

### Phase 0 — Demand validation (ongoing, parallel to M1–M4)

*   [ ] Find 3–5 teams spending $2,000+/mo on LLM APIs
*   [ ] Produce manual CPTC analysis reports from their traffic
*   [ ] Validate evidence-based routing resonates as a wedge

---

## Q3 2026: Enterprise hardening

*   **Goal**: Advanced security, resilience, and compliance for "Moat-Mode" deployments.
*   **Key Deliverables**:
    *   [ ] **P1: Backpressure & Load Shedding** — harden the existing `resilience.py` primitives
    *   [ ] **P1: Multi-Replica Trace Correlation** — end-to-end tracing across load balancers and replicas
    *   [ ] **P1: Durable Audit Export** — compliance logging to S3/Kafka
    *   [ ] **P1: Secret Rotation** — patterns for dynamic secret updates
    *   [ ] **P1: Cost Analyzer** — deep CPTC breakdown, historical trends, anomaly detection (UI layer on top of M1 plumbing)

---

## Q4 2026: HA/DR hardening & federation

*   **Goal**: Production-grade HA, disaster recovery, and multi-region.
*   **Key Deliverables**:
    *   [ ] **P1: HA hardening** — K8s Lease election proven under failure injection; Aurora Multi-AZ / ElastiCache cluster mode validated
    *   [ ] **P1: DR runbook** — cross-region S3 CRR for config + model artifacts; quarterly DR drill cadence; RPO 5min / RTO 15–30min
    *   [ ] **P2: Federation** — Route53 latency + geolocation pinning; regional ClickHouse with global rollup; multi-region OIDC JWKS cache
    *   [ ] **P2: Autoscaling Guidance** — KEDA/HPA metrics for token throughput
    *   [ ] **P2: Multi-Region Sync** — patterns for global deployments

---

## Long Term

*   **Goal**: Fully autonomous, self-optimizing gateway — the AI gateway that configures itself.
*   **Themes**:
    *   Online Learning for Routers (Bandit algorithms beyond EMA)
    *   AI-assisted configuration (gateway suggests routing changes based on traffic patterns)
    *   Managed service (hosted RouteIQ for teams that don't want to self-host)
    *   Marketplace for Routing Models (community LoRA artifacts)
    *   Deep integration with Kubernetes Operators (Operator/CRD scaffolding — deferred from rc1)

---

## Deferred / Rejected

- **VSR-as-default-substrate** — rejected in v3 decision doc. VSR is gated at M4 on measured CPTC uplift. Remains opt-in at best.
- **Topology A (VSR upstream via Envoy ExtProc)** — rejected: Python-plugin-coherence killshot (v2 §5.1, retained in v3). Go classifier upstream of Python guardrails is a governance smell.
- **Kubernetes Operator / CRDs** — deferred until Helm-based deployment proves insufficient at scale.
- **Migrate off LiteLLM** — rejected: control plane is already RouteIQ-native, provider adapter matrix (115 providers) is load-bearing, `litellm.model_cost` cost ledger is the source of truth for spend.
