# Four-Way Comparison: LiteLLM, LLMRouter, RouteIQ, VSR

> **Scope:** April 2026 snapshot. Research produced by a parallel 4-agent fan-out
> (one deep-dive researcher per project). Synthesized from their reports.
> **Companion:** This doc is descriptive. For the strategic decision built on top
> of it, see `vsr-vs-routeiq-decision-v3.md`.
>
> **Versions covered:** LiteLLM 1.81.3 · LLMRouter (llmrouter-lib) 0.3.1 ·
> RouteIQ 1.0.0rc1 · VSR ("Athena") 0.2.0.

---

## 1. Project identities at a glance

| Project | What it is | Primary abstraction | Not what it is |
|---|---|---|---|
| **LiteLLM** | Unified Python SDK + FastAPI proxy that normalizes ~112 providers to OpenAI schemas, plus a full enterprise control plane (keys/teams/budgets/spend/SSO/~28 guardrails/~67 observability integrations) | `litellm.completion()` and `litellm.proxy.proxy_server:app` | An intelligent router |
| **LLMRouter** (UIUC `llmrouter-lib`) | Academic Python library offering 16+ published routing algorithms (KNN/SVM/MLP/MF/Elo/RouterDC/Hybrid/Automix/GNN/Router-R1/…) behind a `MetaRouter` base class, with training + inference + batch evaluator + an OpenAI-compatible FastAPI serve mode in v0.3.1 | `class MyRouter(MetaRouter)` | A production gateway |
| **RouteIQ** | Cloud-native AI gateway built on LiteLLM (proxy, providers, spend) with its own FastAPI app, pluggable routing strategy, native multi-tenant governance, 14 Python plugins, LLM-as-judge eval pipeline, and 6-page React admin UI | `custom_routing_strategy.py` plugin into LiteLLM Router + `/api/v1/routeiq/*` | An Envoy-native dataplane |
| **VSR** (vllm-project/semantic-router) | Envoy ExtProc (Go core + Rust FFI) with 16 signal families, BERT+LoRA classifiers, Boolean-DSL decision engine, ML-driven selectors, in-process guardrails, Router-Replay evaluation, K8s Operator + Helm | Envoy ExtProc filter → `DecisionEngine` → selector | A multi-tenant management plane |

Each project answers a different load-bearing question. LiteLLM answers "how do I talk to every LLM with one schema and bill for it?" LLMRouter answers "what routing algorithms can I test/compare academically?" VSR answers "how do I route LLM traffic with signal-driven classifiers at Envoy line rate?" RouteIQ answers "how do I combine provider breadth with trained routing intelligence and governance into a product I can operate?"

---

## 2. Layered architecture comparison

```
                  ┌──────────────────── USER / CLIENT ────────────────────┐
                  │                                                        │
                  │   OpenAI-compatible HTTP:                              │
                  │   /v1/chat/completions   /v1/embeddings   /v1/rerank   │
                  │   /v1/audio   /v1/images   /v1/batches   /v1/files     │
                  └───────────────────────────┬────────────────────────────┘
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        │                                     │                                     │
        ▼ LiteLLM proxy (FastAPI)             ▼ RouteIQ (own FastAPI)               ▼ VSR + Envoy
  ┌────────────────┐                   ┌────────────────────┐               ┌────────────────────┐
  │ auth / keys    │                   │ auth / admin-key   │               │ Envoy L7 proxy     │
  │ teams/orgs     │                   │ governance (native)│               │      │             │
  │ budgets / spend│                   │ usage_policies     │               │      ▼             │
  │ guardrails SDK │                   │ guardrail_policies │               │ ExtProc (Go gRPC)  │
  │ callbacks      │                   │ oidc               │               │  │                 │
  │ Next.js UI     │                   │ eval_pipeline      │               │  ├─ 16 signals     │
  │ 28 gr + 67 obs │                   │ React UI (6 pages) │               │  ├─ Boolean DSL    │
  │ Router class   │                   │                    │               │  ├─ selectors      │
  │  ├ shuffle     │                   │ CustomRoutingStrat │               │  ├─ semantic cache │
  │  ├ least_busy  │                   │ │   plugin         │               │  ├─ LoRA guards    │
  │  ├ lowest_*    │                   │ ├─ 18+ strategies  │               │  └─ Router Replay  │
  │  ├ auto_router │                   │ ├─ centroid        │               │      │             │
  │  ├ adaptive    │                   │ ├─ personalized    │               │      ▼             │
  │  └ CustomRSB   │                   │ ├─ router_r1       │               │  LiteLLM / vLLM /   │
  └───────┬────────┘                   │ └─ context_opt     │               │  OpenAI-compat     │
          │                            └──────────┬─────────┘               │  backend           │
          ▼                                       │                         └────────────────────┘
 ┌─────────────────┐                              ▼
 │ 112 providers   │                      ┌──────────────┐
 │ (Bedrock, Vtx,  │                      │ LiteLLM      │◄──── RouteIQ wraps LiteLLM at /v1/*
 │  Azure, Anthro, │                      │ (provider    │
 │  Cohere, Groq…) │                      │  breadth)    │
 └─────────────────┘                      └──────────────┘


LLMRouter (library):
      import llmrouter.models
      router = KNNRouter(yaml_path=...)
      router.load_router("weights.pt")
      router.route_batch(queries)      # ← inference
      trainer = KNNRouterTrainer(...)  # ← or training
```

**Substrate pattern by project:**

- LiteLLM: single-process Python FastAPI + Prisma(Postgres/SQLite) + optional Redis.
- LLMRouter: Python library, no server (except optional FastAPI serve mode), no persistence.
- RouteIQ: Python FastAPI + Pydantic BaseSettings + optional Postgres/Redis/Milvus + LiteLLM installed as routing-strategy plugin + K8s-native HA (Lease API).
- VSR: Envoy + Go/Rust binary + K8s Operator + Helm + optional Redis/Milvus/Valkey.

---

## 3. Provider coverage matrix

| Capability | LiteLLM | LLMRouter | RouteIQ | VSR |
|---|---|---|---|---|
| Direct provider adapters | **~112** (OpenAI, Anthropic, Bedrock SigV4, Vertex GCP-auth, Azure AD, Cohere, Mistral, Groq, Together, Fireworks, Perplexity, DeepSeek, Ollama, Databricks, SageMaker, Watsonx, XAI, Cerebras, NIM, Voyage, Jina, Deepgram, ElevenLabs, Gemini, OpenRouter, Snowflake, Replicate, HF, Copilot, OCI, SAP, Z.AI, Moonshot, Vercel AI Gateway, plus search/image/video providers) | N/A (not a proxy) | **All of LiteLLM's** (inherited) | ~18 backends, OpenAI-compatible only (no native Bedrock SigV4, no Vertex GCP-auth, no Azure AD) |
| `/v1/chat/completions` | ✅ | (via `llmrouter/serve/server.py` in v0.3.1) | ✅ (via LiteLLM) | ✅ |
| `/v1/embeddings` | ✅ | — | ✅ (via LiteLLM) | ❌ (not signal-routed) |
| `/v1/rerank` | ✅ | — | ✅ (via LiteLLM) | ❌ |
| `/v1/audio/*` | ✅ (speech, TTS, STT) | — | ✅ (via LiteLLM) | ❌ |
| `/v1/images/*` | ✅ (generation, edits) | — | ✅ (via LiteLLM) | ❌ (modality classifier exists, but routing only on chat/completions + /v1/responses) |
| `/v1/videos/*` | ✅ (RunwayML, Fal, Topaz) | — | ✅ (via LiteLLM) | ❌ |
| `/v1/batches` | ✅ | — | ✅ (via LiteLLM) | ❌ |
| `/v1/files` | ✅ | — | ✅ (via LiteLLM) | ❌ |
| `/v1/fine_tuning` | ✅ | — | ✅ (via LiteLLM) | ❌ |
| `/v1/realtime` | ✅ | — | ✅ (via LiteLLM) | ❌ |
| `/v1/responses` (OpenAI Responses API) | ✅ | — | ✅ (via LiteLLM) | ✅ |
| `/v1/ocr` | ✅ (new April 2026) | — | ✅ (via LiteLLM) | ❌ |
| `/v1/vector_stores` | ✅ | — | ✅ (via LiteLLM) | ❌ |

**Finding.** LiteLLM's breadth is unmatched. RouteIQ inherits all of it at zero cost. VSR's scope is narrow by design: classify-and-route chat / responses only; everything non-chat must bypass VSR via Envoy route config.

---

## 4. Routing strategies comparison

| Strategy family | LiteLLM (native) | LLMRouter (library) | RouteIQ (shipped) | VSR (shipped) |
|---|---|---|---|---|
| Random / shuffle | simple_shuffle | SmallestLLM / LargestLLM baselines | (via LiteLLM fallback) | StaticSelector |
| Least-busy / load | least_busy, lowest_tpm_rpm_v2 | — | (via LiteLLM fallback) | — |
| Lowest-cost | lowest_cost | — | **CostAwareRoutingStrategy** (native, Pareto) | cost factor in HybridSelector |
| Lowest-latency | lowest_latency | — | (available via LiteLLM) | LatencyAwareSelector (TPOT/TTFT percentiles) |
| Tag / metadata | tag_based_routing | — | (available via LiteLLM) | keyword signal + decision tree |
| Embedding similarity | **auto_router** (April 2026, semantic-router wrapper, no online feedback) | — | **centroid_routing** (zero-config, ~2ms, 5 tiers + 5 profiles) | **embedding signal** (cosine) |
| Adaptive / bandit | **adaptive_router** (April 2026, Beta-Bernoulli + request-type buckets, persisted in `LiteLLM_AdaptiveRouterState`) | — | **personalized_routing** (per-user EMA + online feedback endpoint) | **RLDrivenSelector** (Router-R1, Thompson Sampling) |
| KNN | — | KNNRouter (trains + infers) | RouteIQ-native `InferenceKNNRouter` (deliberately bypasses UIUC KNN due to data-loader bugs) | KNN selector (ML layer) |
| SVM | — | SVMRouter | via LLMRouter lazy-load | SVM selector |
| MLP | — | MLPRouter | via LLMRouter lazy-load | MLP selector |
| Matrix factorization | — | MFRouter | via LLMRouter lazy-load | — |
| Elo / Bradley-Terry | — | EloRouter | via LLMRouter lazy-load | EloSelector with UpdateFeedback |
| Dual contrastive | — | DCRouter / RouterDC | via LLMRouter lazy-load | RouterDCSelector |
| Hybrid small-vs-large | — | HybridLLMRouter | via LLMRouter lazy-load | HybridSelector |
| AutoMix (POMDP cascaded) | — | AutomixRouter | via LLMRouter lazy-load | AutoMixSelector |
| Graph neural net | — | GraphRouter | via LLMRouter lazy-load | GMTRouterSelector |
| Router-R1 (NeurIPS 2025 iterative reasoning) | — | RouterR1 (requires vLLM + torch) | **native, no vLLM required** (`router_r1.py`) | RLDrivenSelector (Router-R1 integrated) |
| Multi-turn agentic | — | KNNMultiRoundRouter, LLMMultiRoundRouter | `conversation_affinity.py` | SessionID / TurnIndex / PreviousModel signals |
| MMLU domain classifier | — | — | — | domain signal (~14 classes) |
| Jailbreak classifier | (via guardrails) | — | (via `prompt_injection_guard` plugin) | ModernBERT + LoRA Prompt Guard |
| PII classifier | (via Presidio guardrail) | — | (via `pii_guard` plugin) | token-level BIO ModernBERT + LoRA |
| Hallucination detector | — | — | — | Halugate Sentinel + Detector + Explainer |
| Modality classifier (text/image/audio) | — | — | — | mmBERT-32K (AR / DIFFUSION / BOTH) |
| Token reducer | — | — | **context_optimizer** (6 lossless transforms, 30-70%) | — |

**Finding.** The "18+ RouteIQ strategies" narrative breaks down. RouteIQ has three kinds of strategies: (a) deeply-native — centroid, cost-aware Pareto, personalized, context_optimizer, router_r1, InferenceKNNRouter; (b) LLMRouter-delegated lazy imports — SVM, MLP, MF, Elo, RouterDC, Hybrid, Automix, Graph, CausalLM, GMT; (c) branded RouteIQ prefix over native code — `llmrouter-cost-aware`, `llmrouter-nadirclaw-centroid`. VSR ships the broadest classifier-as-first-class-signal taxonomy. LiteLLM's shipped routing is heuristic + a new embedding-similarity router + a new Beta-Bernoulli bandit.

---

## 5. Control plane / governance comparison

| Capability | LiteLLM | LLMRouter | RouteIQ | VSR |
|---|---|---|---|---|
| Virtual keys (`/v1/key/generate`) | ✅ (`key_management_endpoints.py`) | ❌ | ✅ (inherited from LiteLLM) + `governance.py` API-key layer on top | ❌ |
| Teams (`/v1/team/*`) | ✅ | ❌ | ✅ (inherited) | ❌ |
| Organizations | ✅ | ❌ | ✅ (inherited) + `governance.py` three-tier (Org → Workspace → Key) | ❌ |
| Budgets | ✅ | ❌ | ✅ (inherited) + RouteIQ `usage_policies.py` dynamic with condition matching | ❌ |
| Rate limits | ✅ (lowest_tpm_rpm_v2) | ❌ | ✅ + native with Redis-backed counters | rate-limiting delegated to Envoy AI Gateway |
| Spend tracking | ✅ (`LiteLLM_SpendLogs`, `/v1/spend/logs`) | ❌ | ✅ (inherited) | UsageCost captured in Router Replay |
| SSO (OAuth/JWT/SAML) | ✅ | ❌ | ✅ native (`oidc.py`: Keycloak, Auth0, Okta, Azure AD, Google) | JWT dashboard auth only |
| SCIM v2 | ✅ (`scim/scim_v2.py`) | ❌ | ❌ | ❌ |
| RBAC role-bindings | ✅ | ❌ | ✅ (`rbac.py`, `policy_engine.py`) | ✅ (compiles to router config fragments) |
| Pass-through endpoints | ✅ | ❌ | ✅ (inherited) | ❌ |
| Admin UI | ✅ Next.js dashboard | ❌ (Gradio / ComfyUI demos only) | ✅ React + Vite (6 pages: Dashboard, Routing Config, Governance, Guardrails, Prompts, Observability) | ✅ (onboarding, playground, replay, insights, topology) |
| Multi-tenant primitives shipped | ✅ | ❌ | ✅ native | ❌ ("future enhancement") |
| Prompt template management | — | — | ✅ native (`prompt_management.py`: CRUD + versioning + A/B + rollback + import/export) | "system_prompt" per-decision plugin only |

**Finding.** LiteLLM and RouteIQ both ship full multi-tenant control planes. RouteIQ's is **natively portable** (imports only RouteIQ modules) on top of LiteLLM's inherited surface. VSR has no LiteLLM-compatible management API — it's a routing engine, not a governance platform. LLMRouter has none of this; it's a research library.

---

## 6. Guardrails comparison

| Provider / model | LiteLLM | LLMRouter | RouteIQ | VSR |
|---|---|---|---|---|
| Aporia | ✅ | — | ✅ (via LiteLLM) | — |
| Azure Content Safety / Prompt Shields | ✅ | — | ✅ (via LiteLLM) | — |
| Bedrock Guardrails | ✅ | — | ✅ native (`bedrock_guardrails.py` plugin) | — |
| Google Model Armor | ✅ | — | ✅ (via LiteLLM) | — |
| Guardrails AI | ✅ | — | ✅ (via LiteLLM) | — |
| HiddenLayer | ✅ | — | ✅ (via LiteLLM) | — |
| IBM Guardrails | ✅ | — | ✅ (via LiteLLM) | — |
| Javelin | ✅ | — | ✅ (via LiteLLM) | — |
| Lakera / Lakera v2 | ✅ | — | ✅ (via LiteLLM) | — |
| LlamaGuard | (available) | — | ✅ native (`llamaguard_plugin.py`) | ❌ (would need MCP sidecar) |
| OpenAI Moderation | ✅ | — | ✅ (via LiteLLM) | — |
| Pangea | ✅ | — | ✅ (via LiteLLM) | — |
| Prisma AIRS (Palo Alto) | ✅ | — | ✅ (via LiteLLM) | — |
| Presidio (PII) | ✅ | — | ✅ (via LiteLLM) + native `pii_guard.py` | ✅ token-level BIO ModernBERT+LoRA (in-process) |
| Pillar | ✅ | — | ✅ | — |
| Qualifire | ✅ | — | ✅ | — |
| Zscaler AI Guard | ✅ | — | ✅ | — |
| Prompt injection classifier | (via guardrail vendors) | — | ✅ native (`prompt_injection_guard.py`) | ✅ ModernBERT+LoRA Prompt Guard (in-process) |
| Content filter | ✅ (LiteLLM content filter) | — | ✅ native (`content_filter.py`) | — |
| Hallucination detection | — | — | — | ✅ Halugate Sentinel + Detector + Explainer (in-process) |
| Tool-use permission | ✅ (`tool_permission.py`) | — | — | — |
| Streaming post-call guardrails | ✅ (April 2026) | — | (via LiteLLM) | — |

**Finding.** LiteLLM wins on vendor breadth (~28 providers, Apache-2.0 integrations). VSR wins on in-process classifier latency (ModernBERT+LoRA runs inline in the data path, no network round-trip). RouteIQ bridges both: inherits LiteLLM's vendor breadth AND ships native Python classifier plugins (LlamaGuard, Presidio, Bedrock Guardrails) with its own plugin lifecycle.

---

## 7. Evaluation & experimentation comparison

| Capability | LiteLLM | LLMRouter | RouteIQ | VSR |
|---|---|---|---|---|
| Offline batch evaluator | — | ✅ (`batch_evaluator.py`, 11 benchmark datasets) | — | Fleet simulator + trace workload |
| LLM-as-judge plugin | — | (manual) | ✅ native (`evaluator.py` plugin) | ❌ (planned: Router-R1 LLM-as-Router is experimental) |
| Per-prompt A/B | — | — | ✅ (`/prompts/{name}/ab-test` start/stop) | ❌ |
| Per-request CPTC attribution | — | — | ❌ (the v3 gap) | ❌ (UsageCost at decision granularity only) |
| Per-tenant routing experiment | — | — | ❌ (the v3 gap) | ❌ |
| User-feedback-driven updates | — | (via training) | ✅ `personalized_routing.py` EMA + `/routing/feedback` | ✅ Elo UpdateFeedback (Bradley-Terry) |
| Online feedback loop | ❌ (`auto_router` is static; `adaptive_router` is bucketed bandit) | ❌ (training-time only) | ✅ (eval_pipeline → personalized_routing + centroid weights) | ✅ (Elo via user prefs; Router-R1 RL experimental) |
| Replay / capture | — | — | — | ✅ Router Replay with `UsageCost {PromptTokens, CompletionTokens, ActualCost, BaselineCost, CostSavings}` |
| Model quality dashboard UI | — | — | ✅ Observability page (model-quality ranking, Run-Batch) | ✅ Insights dashboard |
| Trace-level cost dashboard | (via observability integration) | — | ❌ (the v3 gap) | Prometheus metrics |

**Finding.** VSR's Router Replay is the sharpest *capture-and-aggregate* mechanism. RouteIQ's eval_pipeline is the only *judge-scored feedback loop*. Neither project ships per-request CPTC attribution yet. LiteLLM defers evaluation to its ~67 observability integration partners.

---

## 8. Extensibility comparison

| Extension point | LiteLLM | LLMRouter | RouteIQ | VSR |
|---|---|---|---|---|
| Python plugin SDK | ✅ `CustomLogger` + `CustomRoutingStrategyBase` | ✅ (subclass `MetaRouter`) | ✅ (`GatewayPlugin` base + 14 shipped plugins) + `CustomRoutingStrategyBase` into LiteLLM | ❌ |
| In-tree language | Python | Python | Python | Go (core) / Rust (ML) |
| Out-of-tree extension | Via `CustomLogger` subclass, pip install | Subclass + PyPI | Python plugin discovery + LiteLLM hook bridge | MCP servers + gRPC sidecars |
| WASM runtime | — | — | — | ❌ |
| Custom signals / classifiers | — | (any LLMRouter trainer) | (any `gateway/plugins/` module) | In-tree Go only (no stable plugin ABI) |
| Config-only rules | ✅ (guardrails, fallbacks) | (YAML) | YAML + Pydantic Settings | ✅ DSL over signals (primary extension model) |

**Finding.** Python ecosystems (LiteLLM, LLMRouter, RouteIQ) have soft extension boundaries: any Python developer can write a plugin. VSR has hard in-tree boundaries: new signals require upstream Go contributions or side-car processes. This is the single biggest architectural philosophy split across the four projects.

---

## 9. Deployment & operational posture

| Capability | LiteLLM | LLMRouter | RouteIQ | VSR |
|---|---|---|---|---|
| `pip install` path | ✅ (`pip install litellm[proxy]`) | ✅ (`pip install llmrouter-lib`) | ✅ (`pip install routeiq[...]`) | ❌ (binary only) |
| `docker run` path | ✅ | ❌ | ✅ | Envoy sidecar required |
| Docker Compose | ✅ + hardened variant | ❌ | ✅ (10 scenarios: basic, batteries-included, disaggregated-ui, full-stack, ha, local-dev, observability, plug-in, slim, testing) | ✅ |
| Helm chart | ✅ (`deploy/charts/litellm-helm/`) | ❌ | ✅ (12+ templates: leader-election RBAC, ServiceMonitor, PrometheusRule, PDB, NetworkPolicy, ExternalSecret) | ✅ (Athena added full Helm) |
| K8s Operator / CRD | ❌ | ❌ | ❌ (deferred per v3 doc) | ✅ `SemanticRouter` CRD via controller-runtime |
| AWS Lambda | ✅ (`litellm/proxy/lambda.py`) | ❌ | ❌ | ❌ |
| Multi-worker safe | ⚠️ (DB-hot paths, callback fan-out) | N/A | ✅ (post-rc1: `CustomRoutingStrategyBase` plugin replaces monkey-patch) | ✅ (stateless Go server) |
| HA within-region | ✅ (via Redis / Postgres) | N/A | ✅ K8s-native Lease API primary, Redis fallback (ADR-0015) | N/A (assumed stateless) |
| Multi-region / federation | ❌ (no first-class federation primitive) | N/A | Manual today; roadmapped Q4 2026 | ❌ single-cluster only |
| Graceful drain | ✅ | N/A | ✅ (`resilience.py`) | (Envoy's drain) |
| Service discovery w/ graceful degradation | — | N/A | ✅ (`service_discovery.py` probes Postgres/Redis/OTel/OIDC at startup; ADR-0011) | — |
| Hot-reload config | ⚠️ (reload endpoint) | ❌ | ✅ filesystem + S3/GCS ETag | ConfigMap rolling updates |
| Hot-reload ML model weights | ❌ | — | ✅ via `model_artifacts.py` verification | ⚠️ pod-rotation only (not live adapter swap) |

**Finding.** LiteLLM and RouteIQ share the "pip + docker + Helm" triad. VSR forces K8s + Envoy. LLMRouter is a library, not a service.

---

## 10. Observability comparison

| Capability | LiteLLM | LLMRouter | RouteIQ | VSR |
|---|---|---|---|---|
| OpenTelemetry traces | ✅ | ❌ | ✅ (+ GenAI semantic conventions, ADR-0019) | ✅ (`vllm-sr` service name, Envoy → VSR → backend) |
| Prometheus metrics | ✅ | ❌ | ✅ (ServiceMonitor in Helm) | ✅ (`:9190/metrics` with classification/cache/routing metrics) |
| Grafana dashboards | (via Prometheus) | ❌ | ✅ (Helm template) | ✅ (embedded at `:8700/embedded/grafana`) |
| Routing-decision telemetry | ⚠️ (basic log lines) | ❌ | ✅ TG4.1 attributes on spans: `router.strategy`, `router.model_selected`, `router.score`, `router.candidates_evaluated`, `router.decision_outcome`, `router.latency_ms` | ✅ response headers: `X-VSR-Category`, `X-VSR-Model`, `x-vsr-selected-*`, `x-vsr-cache-hit` |
| Third-party logger integrations | **~67** (Datadog, Langfuse, LangSmith, Arize, Helicone, Logfire, Lunary, MLflow, Opik, PostHog, S3, GCS, Dynamo, SQS, Supabase, Weave, Traceloop, Sentinel, CloudZero…) | ❌ | (via LiteLLM callback bridge) | ❌ |
| Versioned telemetry contracts | ❌ | ❌ | ✅ (`telemetry_contracts.py`) | ❌ |

**Finding.** LiteLLM dominates on observability vendor breadth (~67 integrations). RouteIQ's advantage is versioned telemetry contracts + OTel GenAI conventions + TG4.1 routing-decision attributes. VSR ships Prometheus + OTel natively but no third-party partner integrations.

---

## 11. License & governance

| | LiteLLM | LLMRouter | RouteIQ | VSR |
|---|---|---|---|---|
| License (core) | MIT | MIT | MIT | Apache-2.0 |
| License (enterprise overlay) | Proprietary (`enterprise/`) | — | — | — |
| Dual versioning | `litellm` + `litellm-enterprise` + `litellm-proxy-extras` | `llmrouter-lib` | `routeiq` | monolithic |
| Primary maintainer | Berri AI (Y Combinator S23, ~2 core maintainers) | UIUC U Lab (Tao Feng + Haozhen Zhang) | Single core maintainer (Baladithya Balamurugan) | vLLM project + Red Hat + IBM Research + AMD contributors |
| Python floor | ≥3.9, <4.0 | — | ≥3.12 | N/A (Go/Rust) |
| Release cadence | Weekly patches (1.81.3 current) | Dec 2025 → v0.3.1 Feb 2026 | v0.2.0 March 2026 → v1.0.0rc1 April 2026 | Codename releases: Iris Jan → Athena March 2026 |
| Development status | Active / production | Alpha (`Development Status :: 3`) | Pre-v1 release candidate | v0.2, pre-v1 |
| Contributor community | High velocity, small bus factor | ~2 named authors + PRs | Solo core | Multi-company (Red Hat, IBM, AMD, vLLM) |

**Finding.** Each project has governance risk of a different kind. LiteLLM's is enterprise-feature gating (adaptive routing could move behind the proprietary overlay). LLMRouter's is academic-project drift. RouteIQ's is bus factor of one. VSR's is multi-organization coordination but with more mass.

---

## 12. Where each project is strongest

**LiteLLM is strongest at provider breadth + operational control plane.** No other project in the space matches ~112 provider adapters normalized to OpenAI schemas with full streaming/tool-call/structured-output fidelity, plus virtual keys / teams / orgs / budgets / SCIM v2 / ~28 guardrail vendors / ~67 observability integrations / a Next.js dashboard / Helm / Docker / Lambda. LiteLLM is the *default* LLM gateway of record in enterprises that want one thing in front of every model.

**LLMRouter is strongest at academic routing-algorithm breadth.** Sixteen+ published routing algorithms (KNN, SVM, MLP, MF, Elo, RouterDC, Hybrid, Automix, GNN, Router-R1, GMT, multi-turn variants, baselines) behind a unified `MetaRouter` abstraction, each with its own trainer and documented paper. The data-generation pipeline across 11 benchmark datasets plus batch evaluator makes it the natural substrate for *reproducing and benchmarking* routing research.

**RouteIQ is strongest at combining LiteLLM's breadth with trained routing + governance + evaluation.** 18+ strategies (centroid zero-config ~2ms, personalized EMA, native Router-R1 without vLLM, context_optimizer 30-70% token reduction, cost-aware Pareto, plus LLMRouter-delegated classifiers), multi-tenant governance (workspaces, dynamic policies, 14 guardrail types, OIDC), live LLM-as-judge eval pipeline with feedback into routing, 14 Python plugins, 6-page React admin UI, K8s-native HA, 25 ADRs, 3,222 tests. First OSS AI gateway shipping Portkey-grade governance on top of LiteLLM.

**VSR is strongest at signal taxonomy + classifier-based guardrails in the data path.** 16 signal families including PII BIO tagging, ModernBERT+LoRA jailbreak Prompt Guard, three-stage Halugate hallucination detection, MMLU-Pro domain classifier, mmBERT-32K modality router, RL-driven selection, Bradley-Terry Elo — all first-class primitives with sub-10ms classification latency. Envoy-native deployment inherits Envoy's maturity (retries, mTLS, load balancing, observability) for free. Dedicated HuggingFace LoRA training scripts + auto-merge pipeline → Rust-deployable artifacts make VSR unusually credible for *measured* routing.

---

## 13. Where each project is weakest

**LiteLLM's routing intelligence is shallow.** Strategies are rules (shuffle, latency, cost, TPM/RPM, tags, budgets) or thin wrappers (`auto_router` = semantic-router with no online feedback; `adaptive_router` = bucketed bandit). No native ML-driven router, no offline model comparator, no principled cost/quality optimization beyond the bandit. The proxy's single-process Python + Prisma architecture concentrates operational risk. Enterprise features are gated behind a proprietary overlay.

**LLMRouter is not production-ready.** No multi-tenancy, rate limiting, quotas, RBAC, audit logging, OTel, circuit breakers, health probes, HA, secret scrubbing, hot-reload, backpressure, or SLA story. `MetaRouter.__init__` is side-effectful (loads datasets at construction). Alpha status; model weights loaded from arbitrary binary format files — handle only trusted checkpoints; research-grade error handling. Not designed to be the outer edge of a platform.

**RouteIQ has evidence and experimentation gaps.** No per-request CPTC (cost/quality/latency trace-joined), no tenant-level routing-backend experiment harness (prompt A/B only), no `/experiments` / `/judge` / `/evidence` endpoints. No Kubernetes Operator/CRDs (Helm only). No VSR integration (gated at M4 per v3 doc). Task classifier for cold-start absent. Single-maintainer bus factor.

**VSR lacks control plane, tenant primitives, and provider breadth.** No LiteLLM-compatible management API (no key/team/org endpoints). No shipped multi-tenancy. ~18 backends vs LiteLLM's ~112. Pre-v1 (Athena is v0.2). No Python plugin SDK, no WASM runtime, no stable out-of-tree selector ABI — new signals/selectors require in-tree Go contributions. Third-party guardrails (LlamaGuard, Bedrock Guardrails) must run as MCP/gRPC sidecars. K8s + Envoy essentially mandatory. `/v1/images`, `/v1/audio`, `/v1/embeddings`, `/v1/rerank`, `/v1/batches` not signal-routed.

---

## 14. What to adopt from each project (for RouteIQ specifically)

Because RouteIQ is the only project whose mission spans all four axes, these are concrete "pull upstream ideas in" recommendations. Alignment with the v3 decision doc's M1-M4 plan is called out where relevant.

### From LiteLLM

- **Watch `adaptive_router`.** The Beta-Bernoulli bandit with request-type bucketing + persisted state (`LiteLLM_AdaptiveRouterState`) is a good production pattern. RouteIQ's `personalized_routing.py` already uses EMA; consider whether Thompson Sampling on a per-task-type basis adds signal.
- **Port the 28-vendor guardrail catalog** into RouteIQ's `guardrail_policies.py` policy engine. Most are already hookable via LiteLLM's callback bridge — exposing them as first-class policy actions is an M2 UI win.
- **Streaming post-call guardrails** (April 2026). If RouteIQ's guardrail plugins block on full response today, retrofit to support SSE chunk-level inspection.
- **SCIM v2.** Enterprise procurement requires it. Not in RouteIQ's governance layer today; worth an ADR.
- **Non-chat endpoints gap parity.** RouteIQ inherits everything, but the admin UI, observability, and eval pipeline are chat-only. Cover `/v1/batches`, `/v1/images`, `/v1/audio` in the evidence pipeline.

### From LLMRouter

- **The `batch_evaluator.py` pattern.** RouteIQ's eval_pipeline is online + judge-scored; LLMRouter's is offline + benchmark-scored. They're complementary, not competing. Add an offline benchmark mode (`/api/v1/routeiq/eval/benchmark`) that replays a RouterArena-style labeled set.
- **The 11 benchmark datasets.** Don't rebuild; reference them in `docs/evaluation.md` and let users ingest their own via the `routeiq` CLI.
- **Router-R1 training recipe.** RouteIQ ships `router_r1.py` native; LLMRouter documents the paper + training config. Cross-reference.
- **Do NOT deepen the LLMRouter runtime dependency.** The two-import lazy-load pattern is correct. Do not let `MetaRouter` leak further into RouteIQ internals.

### From VSR

- **Signal taxonomy naming.** VSR's 16 signal families are a good public vocabulary (keyword, embedding, domain, fact_check, user_feedback, preference, jailbreak, pii, language, context, structure, complexity, modality, authz, kb, reask, conversation, projection). RouteIQ's plugins and policies could adopt the same taxonomy so third-party artifacts are portable.
- **Router Replay schema.** VSR's `UsageCost {PromptTokens, CompletionTokens, TotalTokens, ActualCost, BaselineCost, CostSavings, Currency, BaselineModel}` is close to what RouteIQ's per-request CPTC should emit. Adopt the field names so cross-tool comparisons work.
- **BERT+LoRA classifier weights.** Apache-2.0. RouteIQ can **port** (not integrate) selected VSR-trained LoRAs into its plugin system (`pii_guard.py`, `prompt_injection_guard.py`). Classifier weights are portable; Envoy runtime is not required.
- **Boolean DSL for policies.** VSR's Boolean tree over signals is a clean way to express routing rules. RouteIQ's `policy_engine.py` is OPA-style; consider whether signal-Boolean-tree is a better UX for the routing layer specifically.
- **Do NOT integrate VSR as a substrate** before M4 measurement. Port, don't stack, until judge-CPTC uplift proves VSR adds >15% value vs RouteIQ's trained arm.

---

## 15. When to pick which project

| If you are … | Pick | Why |
|---|---|---|
| A startup wanting "ChatGPT access to 100+ providers, bill by key, ship today" | **LiteLLM** | Zero-to-production on day one. No routing intelligence, but breadth and mgmt plane are unrivaled. |
| A research team benchmarking routing algorithms on published datasets | **LLMRouter** | 16+ papers in one library, batch evaluator, training pipelines. Do not put it in production. |
| An enterprise wanting multi-tenant governance + trained routing + LLM-as-judge evaluation + evidence of savings on top of LiteLLM's provider breadth | **RouteIQ** | First OSS AI gateway with that combination. Bus-factor-of-one risk; v1.0 GA pending. |
| A platform team running Envoy + K8s, needing classifier-driven signal routing at line rate | **VSR** | The cleanest signal-driven dataplane in 2026. Bring your own management plane. |
| Some combination of the above | **RouteIQ**, adopting ideas from the others per §14 | RouteIQ is the only project designed to subsume pieces of all three. |

---

## Appendix — Research provenance

This document synthesizes four parallel deep-dive reports produced by the
`agent-teams-vs-subagents` skill fan-out pattern:

- `litellm-researcher` — 1,485 words; sources: `reference/litellm/`, GitHub, `litellm.ai/docs/`, deepwiki.
- `llmrouter-researcher` — 1,100 words; sources: `reference/LLMRouter/`, `llmrouter-lib` PyPI 0.3.1.
- `routeiq-researcher` — 1,400 words; sources: repo code, 25 ADRs, CHANGELOG, decision docs v1–v3.
- `vsr-researcher` — 1,485 words; sources: deepwiki on `vllm-project/semantic-router`, GitHub releases v0.1.0 "Iris" + v0.2.0 "Athena".

Sibling artifacts:
- `vsr-vs-routeiq-decision.md` (v1, 2026-04-27)
- `vsr-vs-routeiq-decision-v2.md` (v2, 2026-04-27)
- `vsr-vs-routeiq-decision-v3.md` (v3, 2026-04-27 reality-sync post v1.0.0rc1)
- `evidence-console-design.md` (design doc)
- `product-vision.md`
