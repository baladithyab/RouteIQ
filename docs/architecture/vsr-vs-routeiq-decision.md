# Strategic Decision: VSR vs. RouteIQ

**Date:** 2026-04-27
**Status:** Recommendation — not yet ratified
**Decision owner:** Baladithya Balamurugan
**Method:** 5-agent parallel research fan-out (VSR-Investigator, RouteIQ-Auditor,
LiteLLM-LockIn-Analyst, Migration-Architect, VSR-Steelman — adversarial)

---

## TL;DR

The user's hypothesis — *"we pinned ourselves to LiteLLM, which restricts us;
vllm-semantic-router (VSR) built our routing vision better than we could;
we should build the application layer on top of VSR"* — is **partly right, but
the prescribed action is wrong**.

**Right:** VSR's signal taxonomy, BERT/LoRA classifier guardrails, and
boolean-DSL decision engine are architecturally cleaner than our
LiteLLM+LLMRouter monkey-patched stack for the routing half of the vision.

**Wrong:** Migrating to VSR forfeits ~70-110 engineer-weeks of inherited
LiteLLM functionality (115 providers, virtual keys, teams, budgets,
spend/audit, guardrails SDK, non-chat endpoints, SSO) to gain a routing
engine we already out-feature, while the **actual** differentiator — the
Evidence Console with Cost-Per-Task-Completion (CPTC) measurement — has
zero lines of implementation and is backend-agnostic regardless.

**Recommendation: Path C — Evidence-Layer First, Substrate Later.**
Build Evidence Console as a portable OTel-consuming service that works
against either dataplane. Absorb VSR's best ideas (signal taxonomy,
classifier-first guardrails) into our existing router. Revisit a VSR-backed
enterprise SKU in 6 months once VSR ships an admin API and we have shipped
evidence.

---

## 1. The Question, Restated Precisely

> "Is RouteIQ's current LiteLLM+LLMRouter substrate a strategic mistake, and
> should we migrate our efforts to vllm-project/semantic-router (VSR)?"

Two claims are embedded here. The team evaluated both separately:

| Claim | Verdict | Evidence |
|---|---|---|
| (1) VSR covers the routing/classification vision better than we could | **Partially true** | VSR's 16-signal taxonomy, BERT+LoRA jailbreak/PII classifiers, and semantic cache (memory/redis/valkey/milvus/hybrid) are stronger than RouteIQ's regex-first guards and single-strategy routing. But VSR lacks cost-aware Pareto selection, conversation affinity, and CPTC emission — all RouteIQ-native. |
| (2) Therefore we should migrate efforts to VSR | **False as framed** | Migration cost (70-110 eng-weeks control-plane rebuild + ~55% of src/ ported with rework + 115→7 provider regression) is paid up front. Evidence Console — the actual moat — still has to be built from scratch on either substrate. Migrating substrates delays Evidence Console. |

---

## 2. Triangulated Findings (5-Agent Synthesis)

### 2.1 VSR is architecturally serious but a narrow product

Source: VSR-Investigator, cross-referenced with deepwiki + GitHub + docs.

- **Deployment:** Envoy ExtProc is **mandatory** for the real routing path. "Pure
  API mode" on :8080 exists but exposes only `/v1/models`, `/health`,
  `/config/router` — not an OpenAI-compatible proxy. K8s/Helm/Operator is
  the advertised production path.
- **Language stack:** Go (core + ExtProc gRPC server) + Rust FFI bindings
  (Candle, ONNX, ML). Python is CLI/orchestrator only — **no in-process
  Python plugin runtime, no WASM**.
- **Signals:** 16 families (keyword, embedding, domain/MMLU, fact-check,
  user_feedback, preference, jailbreak, PII, language, context, structure,
  complexity, modality, authz, KB, re-ask, conversation, projection).
  Broader than RouteIQ's current taxonomy.
- **Selectors:** KNN, KMeans, SVM, MLP, Elo (Bradley-Terry), RouterDC,
  AutoMix (POMDP), Hybrid, Static, GMTRouter, LatencyAware, RLDriven. Covers
  ~70% of LLMRouter's 18 strategies natively. **No Matrix Factorization.**
- **Cache:** memory / redis / valkey / milvus / hybrid. Mature.
- **Guardrails:** BERT+LoRA classifiers for Jailbreak (Prompt Guard), PII
  (token-level BIO), Hallucination (Halugate Sentinel+Detector). Classifier-
  first — a real capability advantage over regex-first RouteIQ plugins. **No
  LlamaGuard or Bedrock Guardrails integration.**
- **Extensibility:** In-tree Go or out-of-tree MCP/gRPC sidecar. No Python
  plugin SDK.
- **Backers:** Red Hat (7+ engineers), IBM Research, AMD, vLLM maintainers.
  Latest release v0.2.0 "Athena" March 2026. Pre-v1.

### 2.2 What VSR does NOT have — the load-bearing gaps

| Gap | Impact |
|---|---|
| No LiteLLM-compatible management API | `/v1/key/generate`, `/v1/team/*`, `/v1/organization/*`, budgets, spend attribution — all absent. Multi-tenancy explicitly "future enhancement." |
| ~7 backends vs LiteLLM's 115 | Only `api_format: openai \| anthropic`. No Bedrock SigV4, Vertex GCP-auth, Azure AD, Cohere rerank, audio/image/batches, Groq headers, DeepSeek cache tokens, Ollama, Databricks, SageMaker, Watsonx, XAI. ~80-90% of enterprise customers break day-one. |
| No CPTC, no shipped LLM-as-a-judge | Router Replay captures `PromptTokens/CompletionTokens/ActualCost/BaselineCost/CostSavings`, but there is no task-level quality-aware cost aggregation. Elo uses user preference feedback, not judge scoring. Router-R1 LLM-as-router is experimental, not an eval service. |
| No Python extension runtime | RouteIQ's 13 Python plugins (Bedrock Guardrails, LlamaGuard, pii_guard, prompt_injection_guard, content_filter, cost_tracker, evaluator, cache_plugin, skills_discovery, upskill_evaluator, bedrock_agentcore_mcp, guardrails_base) must be rewritten in Go, or run as MCP/gRPC sidecars (latency + ops tax). |
| No `docker run` / `pip install` path | Envoy + K8s Operator is the real production path. Narrows TAM relative to a pip-install proxy. |

### 2.3 RouteIQ's LLMRouter lock-in is thinner than feared

Source: RouteIQ-Auditor.

- Only **2 import lines** touch LLMRouter (`strategies.py:804`, `:812`).
- ~**3% of strategy code** invokes UIUC classes (lazy-loaded only when a
  `llmrouter-*` strategy is explicitly selected).
- The narrative of "18 UIUC academic strategies" overstates the dependency.
  Most of what RouteIQ actually uses in production — Centroid, cost-aware
  Pareto, hybrid — is **RouteIQ-native code in `strategies.py`**.
- **Implication:** We can drop LLMRouter tomorrow without migrating
  anywhere. It's optional.

### 2.4 RouteIQ's LiteLLM lock-in is thick — and that's a moat

Source: LiteLLM-LockIn-Analyst, verified against `reference/litellm/`.

- **181** `from litellm.*` imports, **65** `litellm.` call sites.
- **115 provider adapters** inherited (Bedrock, Azure, Vertex, Anthropic,
  Cohere, Mistral, Groq, Together, Fireworks, Perplexity, DeepSeek, Ollama,
  Databricks, SageMaker, Watsonx, XAI, Cerebras, NIM, Voyage, Jina,
  Deepgram, ElevenLabs, Stability, Recraft, Runway...).
- **21 management-endpoint modules** inherited (keys, teams, orgs, budgets,
  rate limits, fallbacks, callbacks, caching, guardrails SDK, spend logs,
  SSO, SCIM, admin UI, prometheus, pass-through, `/embeddings`, `/rerank`,
  `/audio`, `/images`, `/v1/fine_tuning`, `/v1/batches`, `/v1/files`).
- **Control-plane rebuild estimate on VSR: 70-110 engineer-weeks** — plus
  maintenance tail.

**The lock-in is deep because we extend LiteLLM, not despite it.** Every
feature enterprise customers actually buy (virtual keys, teams, budgets,
spend, SSO, 115 providers, guardrails SDK, admin UI) is LiteLLM surface
area we inherit for free and differentiate on top of.

### 2.5 LiteLLM is eroding routing differentiation from inside

This is the **genuinely new** finding — and it changes the strategic picture:

- **`feat: add adaptive routing to litellm`** merged **2026-04-18**
  (`litellm_adaptive_routing` branch, v1.83.12-nightly 2026-04-23).
- **`feat(router): add auto_router/quality_router for quality-tier routing`**
  (PR #25993, merged 2026-04-20).
- **`fix(adaptive_router): P1 flusher hot-reload + P2 hook accumulation`**
  2026-04-22 — indicating live feedback loop, not static config.
- Active guardrails work: #26466, #26448, #26390, #26272 (team-level +
  global + streaming post-call) all merged 2026-04-25.

Within 2-3 quarters, "LiteLLM + adaptive router" subsumes our KNN/MLP/ELO
differentiation for 80% of customers. **The real threat is not VSR — it's
the substrate we depend on growing into our niche.**

### 2.6 Evidence Console — the actual moat — has ZERO velocity

Source: RouteIQ-Auditor, git history analysis.

- `docs/architecture/evidence-console-design.md` exists. **No code.**
- Zero commits in the last 3 months reference evidence/CPTC/judge/experiment.
- `routes/evidence.py`, `plugins/judge.py`, `task_classifier.py`,
  experiment-DB schema — **none exist**.
- Telemetry contracts define `routeiq.cptc.*` span attributes. **No code
  populates them.**
- Active commits in the last 3 months: admin UI scaffolding, centroid
  routing, plugin API migration, SSRF hardening. All substrate work.

**Diagnosis: the team has been hardening the foundation instead of building
the tower.** Migrating substrates delays the tower further.

### 2.7 Code disposition under migration

Source: RouteIQ-Auditor on 39,174 src LOC.

| Disposition | LOC | % |
|---|---|---|
| Thrown away (monkey-patches, callback bridges, ASGI wrappers) | ~2,000 | ~6% |
| Ported with rework (plugin system, control plane, API routes rewritten for VSR's hooks) | ~22,000 | ~55% |
| Kept as-is (telemetry contracts, RBAC, audit, cost logic, React UI) | ~4,500 | ~12% |
| Evidence Console / CPTC — doesn't exist yet, backend-agnostic | N/A | ~22% of target scope |

Plugin system migration is the single biggest rework cost: **13 plugins × ~300-500
LOC rewrite each** = 4,000-6,500 LOC port, plus translating LiteLLM's callback
hooks into VSR's MCP/gRPC sidecar model.

---

## 3. The Three Paths

### Path A — Stay on LiteLLM+LLMRouter

**Action:** Drop LLMRouter (2 import lines). Keep the LiteLLM control plane.
Ship Evidence Console as the product.

- **Pros:**
  - Zero migration cost. Preserves 115 providers, 21 management endpoints,
    guardrails SDK, admin UI, SSO/SCIM, non-chat endpoints.
  - All "kept as-is" (12%) + "scaffolded" (22%) code ships value immediately.
  - Customers unaffected.
- **Cons:**
  - LiteLLM adaptive routing (shipped April 2026) erodes our routing-layer
    differentiation over 2-3 quarters.
  - Stuck competing with LiteLLM Enterprise on keys/budgets — losing fight.
  - Single-worker uvicorn constraint unresolved; pre-v1 academic feel.
  - We don't inherit VSR's classifier-first guardrails (demonstrably better
    than regex for PII/jailbreak).
- **Strategic framing:** "Enterprise-grade LiteLLM with evidence layer."

### Path B — Full Migration to VSR

**Action:** Adopt Envoy+VSR as dataplane. Rebuild control plane in Go or
as a Python sidecar. Rewrite 13 plugins as MCP/gRPC sidecars.

- **Pros:**
  - Architecturally cleanest. Envoy ExtProc is the correct L7 substrate.
  - Classifier-first guardrails (BERT+LoRA) are a real capability upgrade.
  - Broader signal taxonomy (16 families) is a future-proofing bet.
  - Red Hat / IBM / AMD / vLLM backing suggests durable OSS trajectory.
- **Cons:**
  - **70-110 engineer-weeks control-plane rebuild** before feature parity.
  - **80-90% of enterprise customers break day-one** on provider gaps
    (Bedrock SigV4, Vertex, Azure AD, Cohere rerank, audio/image/batches).
  - No LlamaGuard / Bedrock Guardrails integration.
  - Python plugin ecosystem dies; 13 plugins rewritten or sidecar'd.
  - Forces K8s + Envoy + Operator as baseline deploy. Kills `pip install` /
    `docker run` TAM.
  - Evidence Console still greenfield — migration doesn't accelerate it.
  - Pre-v1 upstream; we'd be riding release trains we don't control.
- **Strategic framing:** "The trust layer for vLLM Semantic Router."
  (Steelman's pitch — strong if the Envoy-K8s customer is the only
  customer that matters.)

### Path C — Evidence-Layer First, Substrate Later ← **RECOMMENDED**

**Action:** Freeze substrate choice. Ship Evidence Console as a
**backend-agnostic OTel consumer** that works against LiteLLM or VSR.
Absorb VSR's best ideas into RouteIQ's existing routing layer. Revisit
VSR as an optional enterprise SKU in 6 months.

- **Pros:**
  - Shortest path to differentiated, shippable product. Evidence Console +
    CPTC + LLM-as-a-judge are backend-agnostic from day one.
  - Preserves LiteLLM's 115 providers + control plane → zero customer
    disruption.
  - Absorbs VSR's classifier-first guardrails (port their BERT+LoRA signal
    models as a Python plugin — the models are portable even if the
    runtime isn't).
  - Positions RouteIQ's moat (evidence + CPTC + experiments) above whatever
    substrate wins — including LiteLLM's adaptive routing.
  - Keeps the VSR option live: if VSR ships an admin API in 12 months and
    Envoy-K8s becomes the obvious enterprise baseline, we bolt VSR on as
    an alternate dataplane without re-architecting the control plane.
- **Cons:**
  - Doesn't cleanly answer "why not just stay LiteLLM forever?" — requires
    discipline to keep Evidence Console portable, not LiteLLM-shaped.
  - We still ship a FastAPI single-worker gateway in the short term; the
    architectural critique doesn't go away.
  - Upstream VSR moves faster than we can absorb — some ideas (e.g., LoRA
    multi-task classification) may be out of reach as ported features.
- **Strategic framing:** "Evidence-first AI gateway. Bring your own
  dataplane."

---

## 4. Decision Rubric

| Criterion | Weight | Path A | Path B | Path C |
|---|---|---|---|---|
| Time to shippable differentiator | 30% | Medium (Evidence Console still to build) | Low (control-plane rebuild first) | **High** |
| Customer disruption | 20% | **None** | Severe (providers, APIs break) | **None** |
| Technical architecture quality | 15% | Low (single-worker, monkey-patch) | **High** (Envoy ExtProc) | Medium (unchanged short-term) |
| Moat durability vs LiteLLM adaptive routing | 15% | Weak (inside their tent) | **Strong** (different substrate) | **Strong** (above the substrate) |
| Option value preserved | 10% | Low (locked to LiteLLM) | Low (locked to VSR) | **High** (both live) |
| Operational complexity | 10% | **Low** | High (Envoy+K8s+Go) | Low |

**Weighted scores (rough):** A ≈ 5.5 / 10, B ≈ 5.0 / 10, **C ≈ 8.0 / 10**.

---

## 5. Recommendation — Path C, Concretely

### 5.1 What to do in the next 4 weeks

1. **Stop substrate-hardening sprints.** No more monkey-patch refinement, no
   more ASGI-middleware reshuffling. Freeze the LiteLLM integration layer.
2. **Ship Evidence Console Phase 1 — OTel-consumer architecture:**
   - A separate FastAPI service (`evidence-service/`) that consumes OTel
     spans from a Collector — agnostic to whether the producer is RouteIQ
     or VSR.
   - Postgres for experiments + CPTC aggregates. ClickHouse optional later.
   - Span contract: `routeiq.cptc.*` attributes already designed —
     implement the emitter in `router_decision_callback.py` and the
     consumer in `evidence-service/`.
3. **Drop LLMRouter.** Remove the 2 imports in `strategies.py:804,:812`.
   LLMRouter's KNN/SVM/MLP/ELO are either covered by RouteIQ-native code
   or can be replaced by VSR-ported equivalents later.
4. **Port one VSR guardrail model.** Take VSR's BERT+LoRA PII classifier
   (Apache-licensed ONNX weights) and ship it as a Python plugin alongside
   the existing regex `pii_guard`. Demonstrate that VSR's best ideas
   transfer without requiring substrate migration.

### 5.2 Phase 2 (months 2-3) — The actual product

5. **Task classifier.** Implement the rules-based 5-class task classifier
   from the Evidence Console design doc. Emit as span attribute
   `routeiq.task.class`. Consumed by evidence-service.
6. **LLM-as-a-judge plugin.** Wire the scaffolded `evaluator.py` into a
   real judge service. Async, sampled (1-10% of traffic), scores
   challenger vs. baseline with rubric.
7. **A/B experiment middleware.** Hash-based traffic split keyed on
   `tenant_id + request_id`. Arm assignment emitted as span attribute.
   Experiment lifecycle managed by evidence-service.
8. **Admin UI views.** Evidence overview, experiment builder, CPTC
   dashboard. React+Vite — already scaffolded.

### 5.3 Phase 3 (months 4-6) — VSR as optional SKU, only if warranted

Revisit migration only if ALL of the following are true at month 6:
- VSR has shipped a real multi-tenant management API (keys/teams/budgets).
- VSR provider coverage has grown beyond OpenAI/Anthropic native to
  include at least Bedrock-SigV4 and Vertex.
- At least one enterprise customer explicitly requests Envoy+K8s
  deployment.

If those gates fire, offer an **optional "RouteIQ Enterprise on VSR"**
deployment: Evidence Console service (unchanged) consumes OTel from a
VSR dataplane instead of LiteLLM. Same product surface, different
router backend. The preparatory work to make evidence-service portable
in Phase 1 is what unlocks this cleanly.

---

## 6. Killshot Risks (What Would Invalidate This)

| Risk | Signal to watch | Trigger reassessment |
|---|---|---|
| LiteLLM ships a first-class CPTC / judge-based A/B system | LiteLLM release notes | Within 3 months — accelerate Evidence Console ship date |
| VSR ships LiteLLM-compatible management API | VSR milestone "Multi-tenancy" moves from "future" to "in progress" | Re-evaluate Path B cost |
| Enterprise customer demands Envoy+K8s as prerequisite | Actual sales conversation | Stand up Phase 3 VSR SKU now instead of waiting |
| Portkey Gateway OSS (250+ providers, MIT, virtual keys) matures into a credible substrate swap | Portkey GitHub activity | Evaluate as alternate Path B target — architecturally closer to LiteLLM than VSR is |

---

## 7. What the User Was Right About

The framing contains two genuinely important truths that survive the
analysis:

1. **"LiteLLM is restricting us."** It is — not because LiteLLM is bad, but
   because **we extend LiteLLM's feature surface instead of differentiating
   above it**. The remedy is not substrate migration. The remedy is to
   define RouteIQ as the evidence/experiment/CPTC layer — a different
   product category — so LiteLLM shipping adaptive routing doesn't
   commoditize us.
2. **"We should build an application layer."** Yes. Evidence Console is
   that application layer. Build it portably, above whatever dataplane
   wins.

The part to drop: the assumption that "application layer" requires
switching dataplanes. It doesn't. That's the whole point of OTel.

---

## Appendix A — Agent Transcripts

Full 5-agent research outputs (~6,000 words combined) retained in:

- `/private/tmp/claude-503/.../a2b542f6f4d6cad55.output` (VSR-Investigator)
- `/private/tmp/claude-503/.../a5bdfad23ecfeda0d.output` (RouteIQ-Auditor)
- `/private/tmp/claude-503/.../a81b2da8199d05582.output` (LiteLLM-LockIn-Analyst)
- `/private/tmp/claude-503/.../a42cbec1a581b012a.output` (Migration-Architect)
- `/private/tmp/claude-503/.../aae056b13f4d25e82.output` (VSR-Steelman)

## Appendix B — Where Agents Disagreed

| Claim | Steelman | Investigator | LockIn | Architect | Auditor |
|---|---|---|---|---|---|
| VSR is the correct substrate | **Yes** | Partial | No | Partial | Silent |
| LiteLLM is eroding routing moat | Yes | Silent | **Yes (primary source)** | Silent | Silent |
| Evidence Console is the moat | **Yes** | **Yes** | Implicit | **Yes** | **Yes** |
| Migration pays off | **Yes** | No | No | No | No |
| Keep LiteLLM long-term | No | Yes | **Yes** | Yes (short) | Neutral |

Only VSR-Steelman recommends full migration. The four non-adversarial
agents converge on Path A or Path C. The disagreement Steelman raises
that survives scrutiny: **LiteLLM shipping adaptive routing IS a
differentiation threat, and doing nothing about it is not safe.** Path C
addresses this by moving the moat above the substrate.
