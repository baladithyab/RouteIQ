# Evidence Console: Architecture Design

> **Version:** 1.0
> **Date:** 2026-03-29
> **Status:** Approved
> **Supersedes:** [TG3 Admin UI Design](tg3-admin-ui-design.md) (extends, does not replace — TG3 covers general admin UI architecture; this document covers the Evidence Console product layer)
> **Origin:** Office hours design session with adversarial review (3 rounds, 8/10 final score)

---

## Overview

The Routing Evidence Console is RouteIQ's primary product surface — a dashboard that makes intelligent routing visible and actionable. It answers one question: **"Can I safely switch models and save money?"**

This document covers the engineering design: metrics, data model, API surface, experiment execution, task classification, and quality evaluation. For product positioning and competitive context, see [Product Vision](../product-vision.md).

### Approach

**A+B Hybrid:** Ship Phase 1 scope (Evidence Console) with Phase 2 architecture (Full Control Plane). Build the experiment workflow first. Structure the codebase so Cost Analyzer, Routing Observatory, Strategy Manager, and Model Registry are natural extensions.

---

## Key Definitions

### Cost-Per-Task-Completion (CPTC)

The core metric. Not price-per-token — the total cost to produce a successful response for a logical request, including retries, tool calls, and reasoning tokens.

| Term | Definition |
|------|-----------|
| **Task** | A single logical user request (one chat completion, one classification, one code generation). Multi-turn conversations: each turn is a separate task. |
| **Completion** | A response with HTTP 200 and `finish_reason` in (`stop`, `tool_calls`). Responses with `finish_reason=length` or error codes are incomplete. |
| **CPTC** | Sum of token costs across all attempts for one task. |

**Example:** Model A needs 1 attempt at $0.02. Model B needs 3 attempts at $0.005 each. CPTC(A) = $0.02. CPTC(B) = $0.015. Model B is cheaper despite more attempts.

**OTel integration:** A parent span groups all attempts for one logical request. The gateway's retry logic (LiteLLM's `num_retries` parameter) creates child spans under the original request span. Client-initiated retries are separate tasks unless correlated by a client-provided `X-Request-ID` header.

**Implementation:** Extend `router_decision_callback.py` spans with:
- `routeiq.cptc.attempt_count` (int)
- `routeiq.cptc.cumulative_token_cost` (float)
- `routeiq.cptc.task_type` (string)

### Task Type Taxonomy

Auto-classification of requests into 5 categories for spend analysis.

| Type | Heuristic Rules |
|------|----------------|
| **Simple Q&A** | Single-turn factual questions, short responses (< 200 tokens output) |
| **Classification** | Structured output requests (JSON, labels, categories, yes/no) |
| **Code Generation** | Requests containing code blocks, programming language keywords, or `code` in system prompt |
| **Complex Reasoning** | Multi-step reasoning, chain-of-thought, long output (> 500 tokens), or explicit reasoning instructions |
| **Other** | Everything else |

**Phase 1 implementation:** Regex + heuristic rules on prompt content and response metadata. No ML model needed. If classification confidence is low, tag as "Other." Accuracy target: 80%+ on the 4 named categories. Good enough for meaningful spend breakdowns without introducing a separate ML dependency.

**Phase 2 upgrade path:** Embedding-based classifier using the existing `strategies.py` embedding infrastructure.

### LLM-as-a-Judge

Quality evaluation for A/B experiments.

- **Judge model:** Different model family than baseline or challenger to avoid self-bias. Default: if baseline is Claude, judge with GPT-4o-mini. Configurable per experiment.
- **Evaluation prompt:** Rubric-based scoring: "Given this user request, rate each response independently on correctness (0-10), completeness (0-10), helpfulness (0-10). Then state which is better overall or if they are equivalent. Output: score_A, score_B, verdict (A/B/tie), confidence (0-1)."
- **Historical baseline matching:** The judge compares the challenger response against a historical baseline response matched by task type + prompt embedding similarity. Uses the existing `strategies.py` embedding infrastructure (same embeddings used for KNN routing). The judge prompt frames this as: "Response A is from a previous similar request; Response B is from the current request."
- **Transition trigger:** Experiment status moves from `running` to `review` automatically when sample count reaches the configured minimum (default: 200 per task type present). Time-based fallback: auto-transition after 14 days with "low confidence" flag if below threshold.
- **Failure mode:** Judge confidence < 70% flags for human review. Judge model unavailable queues as "pending human review."
- **Cost:** ~$0.001/call. At 5% traffic split on 10K requests/month = 500 judge calls = $0.50/month.

**Gateway integration:** Implemented as a `GatewayPlugin` extending the existing plugin base class. See [Plugin System](../plugins.md) for lifecycle details.

---

## Data Model

```
Experiment:
  id: UUID
  name: str
  baseline_model: str              # LiteLLM model name
  challenger_model: str
  traffic_split_pct: int           # Enforced range: 1-10
  judge_model: str
  status: enum(draft, running, review, approved, rejected)
  min_samples_per_type: int        # Default: 200
  created_at, updated_at: datetime
  created_by: str                  # Admin key ID

ExperimentResult:
  id: UUID
  experiment_id: FK -> Experiment
  request_hash: str                # Anonymized request fingerprint
  task_type: enum(simple_qa, classification, code_gen, complex_reasoning, other)
  baseline_cptc: Decimal           # From historical match
  challenger_cptc: Decimal         # From live experiment
  baseline_latency_ms: int
  challenger_latency_ms: int
  judge_score_baseline: float      # 0-10 rubric average
  judge_score_challenger: float    # 0-10 rubric average
  judge_confidence: float          # 0-1
  judge_verdict: enum(baseline_better, challenger_better, tie, needs_human_review)
  created_at: datetime

RoutingApproval:
  id: UUID
  experiment_id: FK -> Experiment
  approved_by: str                 # Admin key ID
  config_patch: JSON               # The YAML patch applied via hot-reload
  savings_estimate_monthly: Decimal
  created_at: datetime
```

**Storage:** PostgreSQL (already a dependency for HA mode). SQLite fallback for single-node dev.

---

## API Surface

### Phase 1 Endpoints

```
GET  /api/v1/evidence/overview                    # Dashboard stats + spend breakdown
GET  /api/v1/evidence/experiments                 # List experiments
POST /api/v1/evidence/experiments                 # Create experiment (status: draft)
POST /api/v1/evidence/experiments/{id}/start      # Start experiment (draft -> running)
GET  /api/v1/evidence/experiments/{id}            # Experiment detail + results
POST /api/v1/evidence/experiments/{id}/approve    # Approve + generate config patch
POST /api/v1/evidence/experiments/{id}/reject     # Reject experiment
GET  /api/v1/evidence/experiments/{id}/samples    # Side-by-side sample pairs
```

**Auth:** All endpoints require admin auth (`X-Admin-API-Key` header). Read-only dashboard (`/overview`) accessible to any authenticated user.

**Polling:** Results update via `GET` with `?since=<timestamp>` parameter. WebSocket deferred to Phase 2.

**Config generation:** `POST .../approve` generates a YAML config patch in `config_loader.py` format, applied via the existing `hot_reload.py` filesystem watcher.

**Implementation:** New router module at `src/litellm_llmrouter/routes/evidence.py`, registered in `gateway/app.py` via `_register_routes()`.

### Phase 2 Extensions

```
GET  /api/v1/evidence/cost-analysis               # Deep CPTC breakdown, trends
GET  /api/v1/evidence/routing-traces               # OTel trace query (Jaeger/Tempo API proxy)
GET  /api/v1/evidence/strategies                   # Available strategies + performance baselines
PUT  /api/v1/evidence/strategies/{name}/activate   # Hot-swap active strategy
GET  /api/v1/evidence/models                       # Connected models + BYOK status
```

---

## Experiment Execution Model

During an experiment, the gateway uses **true A/B testing** (not shadow traffic).

1. A request arrives at the gateway.
2. The experiment middleware checks if the request matches the experiment scope and rolls the dice against the traffic split percentage.
3. **If selected (1-10% of traffic):** Route to the challenger model. Serve the challenger response to the user.
4. **If not selected (90-99%):** Route normally (baseline model). Store the response as a historical baseline sample.
5. **Quality evaluation:** The judge compares the challenger response against a historical baseline response matched by task type and prompt embedding similarity.
6. **When sample threshold is reached:** Experiment transitions to `review` status. Results are aggregated and presented for human approval.
7. **On approval:** A YAML config patch is generated and applied via hot-reload, making the routing change permanent.

**Why true A/B, not shadow traffic:** Shadow traffic (send to both models, serve only baseline) doubles inference cost for experiment traffic. True A/B serves the challenger to a small percentage of real users. The traffic split (1-10%) bounds the blast radius.

**Zero-disruption guarantee:** At most 10% of requests go to the challenger. If the challenger underperforms, the experiment can be rejected and all traffic returns to the baseline immediately.

---

## Statistical Validity

- **Minimum sample size:** 200 requests per task type before results are shown as "confident." Below 200: results display with a "low confidence" warning.
- **At 5% traffic split on 10K requests/month:** ~500 experiment samples. If evenly distributed across 5 task types, 100/type — below confidence threshold. Recommendation: start with 10% split or run experiments longer.
- **Phase 1 methodology:** Simple win-rate (% of samples where challenger matches or beats baseline) with confidence intervals displayed in UI.
- **Phase 2 upgrade:** Proper statistical significance tests (e.g., paired t-test on judge scores).

---

## UI Architecture

### Phase 1 Views

| View | Purpose | Key Components |
|------|---------|---------------|
| **Overview** | Dashboard with spend breakdown, active experiments, savings | Stat cards, bar chart (spend by task type), experiment summary table |
| **Experiments** | Experiment list with status filters | Table with status badges, actions (start/approve/reject) |
| **Experiment Detail** | Results for a single experiment | Side-by-side quality samples, CPTC/latency/judge deltas, approve/reject CTA |

### Navigation (sidebar)

```
EVIDENCE
  ● Overview          (Phase 1 — active)
  ○ Experiments        (Phase 1 — active)

CONTROL PLANE
  ○ Cost Analyzer      (Phase 2 — grayed)
  ○ Routing Observatory (Phase 2 — grayed)
  ○ Strategy Manager   (Phase 2 — grayed)
  ○ Model Registry     (Phase 2 — grayed)

SETTINGS
  ○ API Keys
  ○ Team
```

Phase 2 views are visible in the sidebar from day one — grayed out but present. This communicates the product's direction and prevents layout shifts when new views ship.

### Technology

- **Frontend:** React + Vite + TypeScript (existing scaffold in `admin-ui/`)
- **State management:** TanStack Query (server state), React context (UI state)
- **Packaging:** Pre-built static assets served via FastAPI `StaticFiles` mount (zero Node.js in production)
- **Auth:** `X-Admin-API-Key` header, same as existing admin endpoints

See [TG3 Admin UI Design](tg3-admin-ui-design.md) for detailed technology evaluation and deployment modes (embedded, sidecar, standalone, LiteLLM plugin).

---

## Implementation Phases

### Phase 0: Demand Validation (weeks 0-2)
Find 3-5 teams spending $2,000+/month on LLM APIs. Analyze their traffic manually (OTel exports or request logs). Produce CPTC reports. Runs IN PARALLEL with Phase 1 scaffolding.

### Phase 1: Evidence Console (weeks 1-4)
- `routes/evidence.py` — API endpoints
- `evidence_plugin.py` — Experiment execution middleware (GatewayPlugin)
- `task_classifier.py` — Rule-based task type classification
- `judge_plugin.py` — LLM-as-a-judge evaluation plugin
- Admin UI: Overview dashboard, Experiment builder, Results view
- Database migrations for Experiment/ExperimentResult/RoutingApproval tables

### Phase 2: Control Plane Extensions (weeks 5-10)
- Cost Analyzer: Historical CPTC trends, anomaly detection
- Routing Observatory: OTel trace query integration (Jaeger/Tempo API, batch query, no real-time streaming)
- Strategy Manager: Expose 3-5 key strategies with configuration UI
- Model Registry: BYOK key management, performance baselines

---

## Open Questions

1. **Multi-turn CPTC:** Each turn is a separate task for now. Tasks spanning turns (e.g., iterative code refinement) need user research to determine if per-turn attribution is sufficient.
2. **Judge bias:** Different model families may systematically prefer their own style. Mitigation: always use a judge from a different family. Needs empirical validation.
3. **Embedding model for historical matching:** The existing `strategies.py` infrastructure provides embeddings. If embedding quality is insufficient for baseline matching, may need a dedicated embedding model.

---

*See also: [Product Vision](../product-vision.md) | [TG3 Admin UI Design](tg3-admin-ui-design.md) | [Routing Strategies](../routing-strategies.md) | [Observability Guide](../observability.md)*
