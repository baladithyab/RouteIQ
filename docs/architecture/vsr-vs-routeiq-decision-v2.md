# VSR vs RouteIQ Decision — v2

> **Supersedes:** `docs/architecture/vsr-vs-routeiq-decision.md` (v1)
> **Status:** Decision doc, produced by `routeiq-deliberation` team (3-role debate on top of 8-investigator dossier).
> **Date:** 2026-04-27

---

## TL;DR

The user's hypothesis — "stack VSR upstream of LiteLLM (Topology A) as the default dataplane" — is **rejected** by the deliberation team. After three rounds of cross-examination between `stacking-bull`, `stacking-bear`, and `evidence-layer`, all three deliberators converged on a single recommendation:

> **Ship Evidence Console first. Ship Topology C (opt-in dispatcher) second. Decide VSR promotion third, from data, not advocacy.**

The moat is not the routing substrate; the moat is the **Evidence Console + CPTC + judge-labeled retraining loop**, which consumes OpenTelemetry spans and is substrate-neutral by construction. VSR becomes a feature-flagged enterprise-tier backend behind a hard SSE benchmark gate, not the default path. Integration-Architect-v2's "Topology B.2" and Stacking-Architect's "Topology C" collapse into the same architecture once you ask where the dispatcher lives — *inside LiteLLM's process, as a `CustomRoutingStrategyBase` implementation*. The apparent conflict between investigators was label-deep, not architecture-deep.

**Verdict:**
- Topology A: REJECTED (three killshots — see §5).
- Topology B.2 / Topology C: equivalent; CHOSEN as C.
- Sequencing: Evidence Console (M1-M2) → Topology C dispatcher (M3) → MLOps + substrate decision (M4) → HA/DR/federation (M5-M6).

---

## What the user was right about

Before the corrections and killshots, it matters to honor what the original hypothesis got right, because the revision is narrow — it is a change of *ordering*, not a rejection of the underlying bet:

- **Stacking is viable.** Migration off LiteLLM would be 55% rework; stacking preserves ~63% of existing code. The instinct that RouteIQ cannot afford to throw away LiteLLM's provider breadth, auth, spend tracking, multi-modal surface, MCP, and A2A is correct.
- **LiteLLM's breadth matters.** 181 `from litellm` imports and 65 `litellm.*` calls in `src/litellm_llmrouter/` reflect real leverage, not accidental coupling. The `CustomRoutingStrategyBase` plugin API (in use at `src/litellm_llmrouter/custom_routing_strategy.py:114`, shipped by commit `70080c8`) is an **official** integration point, not a monkey-patch.
- **VSR's routing intelligence is real.** BERT + LoRA classifiers, mmBERT-32K modality routing, `hasImageParts`, session affinity through `RequestContext.SessionID/TurnIndex/PreviousModel`, PII/jailbreak/intent/fact-check training scripts — this is a credible ML asset that LiteLLM's upstream `auto_router` does not match on online feedback.
- **OpenRouter-clone framing.** The product north star — a gateway that routes intelligently across many providers, with observable evidence of quality — is the right target. Nothing in this decision contests that.

What the team reframed is **ordering**. The user's instinct put VSR (substrate) ahead of Evidence (moat). The team inverted that.

---

## Corrections to v1

### 1. B.2 ≈ C (the biggest single correction)

v1 presented Topology A, B, and C as three distinct architectures, with Stackability-Investigator scoring A=8/10 / B=3/10 and Integration-Architect-v2 independently recommending B.2. That looked like a genuine disagreement. It is not.

**Stackability-Investigator's B** was an Envoy-less subprocess-call arrangement (LiteLLM invoking VSR binary as a child process). **Integration-Architect-v2's B.2** is materially different: LiteLLM in front, a *same-process* direct route for non-VSR traffic, and VSR registered as a `routeiq-smart` backend reachable via `CustomRoutingStrategyBase`. Once the dispatcher lives inside the LiteLLM process as a pluggable strategy, "Topology B.2" and "Topology C" describe the same binaries and the same deployment shape — only the label differs. The v1 conflict between those two investigators dissolves when B.2 is re-categorized as C.

### 2. LLMRouter is not LiteLLM

v1 occasionally used "LLMRouter" and "LiteLLM router" interchangeably. They are not the same. RouteIQ's production lock-in is to **LiteLLM** (proxy / API compatibility / callback bridge / user_api_key_auth / router model), not to `LLMRouter` (the upstream research project whose strategies were adapted into `src/litellm_llmrouter/strategies.py`). This matters because *where the substrate is replaceable* depends on what is actually imported. The lock-in is LiteLLM-shaped, and the escape hatch is LiteLLM's official plugin surface.

### 3. LiteLLM auto_router reassessment

v1 treated LiteLLM's upstream routing as "heuristic." It is not. Version 1.83.x ships `auto_router`, which is embedding-similarity based (aurelio-labs/semantic-router derived). The gap between that and VSR's trained BERT+LoRA classifiers is **smaller than v1 assumed** — but still material on one axis: `auto_router` has **no online feedback loop**. Embeddings are frozen at model training time; there is no judge-verdict → retraining path. That is exactly the gap VSR+RouteIQ's MLOps loop fills.

### 4. Three-path rubric reframed

v1 framed "A (stack) vs B (stack) vs full migration." The team reframes this as **"three ways to deploy the moat"**: (a) native LiteLLM-only, (b) LiteLLM with VSR upstream (Topology A), or (c) LiteLLM with VSR as an opt-in backend (Topology C). Migration is off the table — Evidence Console is the asset, not the substrate.

---

## Triangulated findings (8 investigators + 3 deliberators)

Converged facts the team treats as load-bearing:

1. **VSR ExtProc** rewrites `model` and sets `x-vsr-destination-endpoint`; classification targets `<10ms`, hardening at `<50ms`. Streaming/SSE behavior through the response path is **unconfirmed** (VSR-Investigator + Stackability-Investigator).
2. **VSR owns only `/v1/chat/completions` and `/v1/responses`**. `/v1/images`, `/v1/audio`, `/v1/embeddings`, `/v1/rerank`, `/v1/batches` must bypass VSR via Envoy route match. MCP/A2A pass through untouched.
3. **VSR federation is absent** — single-cluster. Multi-region routing is a non-primitive; must be handled at LiteLLM or global-LB layer (Cloud-Native-Architect).
4. **VSR hot-reload of classifier weights = pod rotation.** No documented in-place LoRA swap. Argo Rollouts delivers new weights; floor ~5-15 min.
5. **LiteLLM in-process hot reload** via `hot_reload.py` is seconds-class (filesystem watch + `CustomRoutingStrategyBase` swap).
6. **RouteIQ code disposition under stacking:** ~63% kept, ~3% thrown away, ~4% rework, ~10% new code (Evidence service + dispatcher). Migration path is ~55% rework — far worse (RouteIQ-Auditor + Stackability-Investigator).
7. **LiteLLM `auto_router`** (v1.83.x, embedding similarity, no online feedback). Dual-licensed (MIT core + proprietary `enterprise/` modules). Routing intelligence is upstream territory going forward (LiteLLM-Deep-Audit).
8. **RouteIQ has 13 Python plugins** in `src/litellm_llmrouter/gateway/plugins/` (pii_guard, prompt_injection_guard, llamaguard, bedrock_guardrails, cache_plugin, content_filter, cost_tracker, evaluator, skills_discovery, upskill_evaluator, bedrock_agentcore_mcp, guardrails_base, semantic_cache). All bind to LiteLLM's callback bridge. VSR has **no Python plugin runtime**.
9. **Evidence Console is greenfield** — design doc only, zero code. That is a *feature* for sequencing (substrate-neutral OTel consumer can be built regardless of dataplane).
10. **Portable moat principle** (from `strategic-replatform-deliberation`): "If the user's differentiator consumes telemetry rather than produces it, it is substrate-independent by construction."

---

## Three topologies as the team resolved them

```
Topology A — VSR upstream via Envoy ExtProc (REJECTED)
───────────────────────────────────────────────────────
   client ─► Envoy ──ExtProc──► VSR (Go, BERT+LoRA)
                                  │ rewrites model + x-vsr-destination-endpoint
                                  ▼
                              LiteLLM (Python) ──► providers
                                  │
                                  └─► 13 Python plugins fire AFTER routing
                                       (PII guardrails downstream of classifier)
```

**Why rejected:**
- Python plugin coherence killshot (§5.1). PII content routed by Go classifier *before* Python guardrails fire.
- SSE through Envoy upstream is unmeasured; no Phase-0 benchmark exists.
- Envoy-mandatory eliminates the `pip install routeiq` / `docker run` deployment shape — narrows TAM to K8s-native buyers only.
- Forces disabling LiteLLM router features (fallbacks, cost-based, `auto_router`).

```
Topology B.2 / Topology C — same-process dispatcher (CHOSEN)
────────────────────────────────────────────────────────────
   client ─► LiteLLM
                │
                ▼
        CustomRoutingStrategyBase  ◄── dispatcher (thin, per-tenant)
                │
       ┌────────┼────────────────┬─────────────────────┐
       ▼        ▼                ▼                     ▼
   auto_router  native           VSR opt-in            (bypass for
   (control)    strategies.py    (enterprise K8s)      /v1/images etc.)
                (trained, MLOps                         ──► providers direct
                 retrained)
                │
                ▼ (all three emit OTel spans with routing_backend attr)
   ──────────────────────────────────────────────────────────────
                                 ▲
                                 │ consumes spans
                             Evidence Console
                             (judge, CPTC, retrain)
```

**Why chosen:**
- Every backend emits OTel; Evidence Console is substrate-neutral.
- Dispatcher is thin (pure tenant/flag selector); no third dataplane.
- VSR risk (SSE, v0.2, pod-rotation hot-reload, Envoy dependency) is bounded to opt-in enterprise tenants.
- 13 Python plugins continue firing pre-routing in native path. Governance coherence preserved.
- `pip install routeiq` / `docker run` ships with LiteLLM-native path only (B.2 shape with VSR backend absent).

### Decision matrix

| Criterion | Topology A | Topology C |
|---|---|---|
| Plugin coherence (13 Python guardrails) | Split across Go+Python; PII routed before guardrails | Native path preserves order |
| SSE streaming | Unmeasured risk on response path | Native path inherits LiteLLM's proven behavior |
| Deploy shape | Envoy mandatory → K8s only | Dual shape (pip/docker + K8s) |
| Blast radius of VSR risk | All tenants | Opt-in tenants only |
| MLOps hot-reload | Pod rotation (~5-15 min) | Seconds in native path; pod-rotation in VSR arm |
| Routing intelligence | VSR dominant | Three arms: auto_router / native / VSR |
| Evidence moat coupling | Coupled to VSR dataplane | Substrate-neutral (consumes OTel) |
| Disables LiteLLM router features | Yes | No |
| Federation path | Not in VSR; needs external layer | Inherited from LiteLLM / global-LB |
| Code disposition | ~10% new + ~15% rework | ~10% new + ~4% rework |

---

## Three paths — the path rubric

The user's v1 framing asked "migrate, stack-A, or stack-B?" The team's rubric is reshaped:

| Path | Preserves LiteLLM breadth? | Ships moat? | Contains VSR risk? | 6-month ship odds |
|---|---|---|---|---|
| **P1 — Full migrate to VSR-native** | No (loses providers, auth, spend, MCP, A2A, plugins) | Maybe — depends on rebuild | Risk becomes the whole product | Low |
| **P2 — Stack A (VSR upstream always-on)** | Yes | Only after substrate ships | All tenants carry risk | Medium |
| **P3 — Stack C (Evidence-first, VSR opt-in)** | Yes | M1-M2 (before substrate) | Opt-in tenants only | High |

P3 is the chosen path. P1 is eliminated on code disposition (~55% rework, brand hit). P2 is eliminated by the three debate killshots.

---

## Three killshots the debate surfaced

These three were not in the investigator dossier. They emerged from cross-examination and are the highest-leverage new content in this revision.

### §5.1 Python plugin coherence (kills Topology A)

RouteIQ's 13 Python plugins in `src/litellm_llmrouter/gateway/plugins/` bind to LiteLLM's callback bridge (`plugin_callback_bridge.py`) and run in the LiteLLM process. Under Topology A, VSR classifies the request upstream of LiteLLM — meaning PII-tagged content is **routed to a downstream model before `pii_guard`, `prompt_injection_guard`, and `llamaguard_plugin` ever fire**. The classifier may have pre-labeled the content as containing PII and chosen a smaller/cheaper model; the guardrails then redact the content downstream. Governance chain is inverted: the routing decision sees raw data that the guardrail system has not yet sanitized.

Under Topology C, the dispatcher runs *inside* the LiteLLM process. Plugin callbacks fire in their documented order relative to routing. The native path preserves the existing guardrail-then-route invariant; the VSR path can be opted into only when tenants accept the inverted order.

Stackability-Investigator's A=8/10 verdict did not weight this. The team treats it as a categorical defect of Topology A, not a gap to mitigate.

### §5.2 Evidence-without-active-RouteIQ-arm trap

Evidence-layer's sharpest contribution. If Evidence Console ships M1-M2 consuming only OTel spans from LiteLLM's `auto_router` (embedding similarity, upstream, frozen weights), every judge verdict grades a commodity routing decision. The retraining dataset contains no RouteIQ-labeled divergence from upstream behavior. In that world, Evidence is a fancy dashboard over someone else's routing, and the MLOps loop has nothing novel to learn.

The mitigation is a **gating requirement on M1-M2 ship**: `src/litellm_llmrouter/strategies.py`'s trained-strategy family (KNN, MLP, SVM, ELO, MF, hybrid — 18+ strategies) must be wired as an active routing arm *from day one of Evidence*, via `CustomRoutingStrategyBase`. Judge verdicts then compare auto_router vs native-trained vs (when opt-in) VSR — that's a real label signal, not a grade on LiteLLM's work.

This is the **sharpest killshot** in the entire decision. Missing it would ship a dashboard that looks like a moat but produces no defensible training data.

### §5.3 Retrain-cadence as substrate-forcing function

The M4 decision point (promote VSR or keep it opt-in) is not an advocacy call. It is forced by one measurable property: how fast the MLOps loop needs to deliver new classifier weights to production.

- **VSR hot-reload** = pod rotation via Argo Rollouts. Floor ~5-15 minutes.
- **LiteLLM in-process** via `hot_reload.py` = filesystem watch + `CustomRoutingStrategyBase` swap. Seconds.

If Evidence-labeled retraining exposes regressions requiring sub-hour rollback (e.g., a bad LoRA shipped at 02:00 has to be pulled by 02:30), the native in-process path wins and VSR K8s-default is never defensible. If retrain cadence is daily-or-slower, VSR's pod-rotation cost is invisible and promotion becomes fine.

This is **decided by Evidence data in M4**, not by any of the three deliberators' preferences. That is the team's answer to the question "when does this decision get made?"

### §5.4 The dispatcher is an A/B harness, not load balancing

Added in `stacking-bull`'s closing addendum and ratified by all three agents. The M3 dispatcher is not "backend selection for load distribution" — it is a **measurement instrument**. The claim "a trained classifier + judge-labeled retraining loop beats upstream embedding-similarity routing" is a falsifiable hypothesis, and the dispatcher is the apparatus that tests it. Every request is labeled with its `routing_backend` span attribute; judge verdicts then produce a per-backend quality time-series. M4's VSR-promotion decision falls out of that signal directly.

The team adopted an explicit numeric rubric for the M4 promotion call:

| Judge-scored CPTC uplift of VSR vs LiteLLM-native | M4 decision |
|---|---|
| ≥ 15% | Promote VSR to K8s-tier default. Native remains as fallback. |
| 5% – 15% | Product-judgment call (pricing, plugin coherence, operational cost). |
| < 5% | Keep VSR opt-in only, or sunset the VSR backend. |

Three consequences of this framing:

1. It resolves the apparent Integration-Architect-v2 vs Stackability-Investigator conflict — substrate choice becomes data-driven, not argued.
2. It eliminates the "we chose VSR because the investigator dossier scored A=8/10" failure mode. Scoring the architecture is not the same as measuring routing quality.
3. It gives the team a clean *sunset path* for VSR if upstream LiteLLM Enterprise ships an adaptive feedback router in H2 2026 — judge verdicts will show the delta narrowing, and the <5% threshold triggers sunset without drama.

---

## Phased recommendation

| Month | Deliverable | Substrate state |
|---|---|---|
| **M1** | Evidence Console scaffolding + CPTC schema + judge pipeline skeleton. Wire `strategies.py` as an active routing arm via `CustomRoutingStrategyBase` (§5.2 mitigation). | Single substrate: LiteLLM-native. |
| **M2** | Evidence Console ships. OTel ingestion from LiteLLM (via `observability.py`) produces labeled judge verdicts. First real CPTC data. | Single substrate. |
| **M3** | Topology C dispatcher ships inside LiteLLM. Three backends wired: auto_router (control), native strategies (trained arm), VSR (feature-flagged, opt-in only). Two deployment shapes: `pip`/`docker` = native-only; enterprise K8s = all three. VSR SSE Phase-0 benchmark is a gate for enabling the VSR backend in any tenant. | Dispatcher active; VSR opt-in gated. |
| **M4** | MLOps retraining loop: judge-labeled dataset → nightly LoRA → canary rollout. Substrate-promotion decision locked in here, based on observed retrain-cadence requirement (§5.3). | Decision artifact: retain opt-in OR promote VSR to K8s default. |
| **M5** | HA/DR primitives at LiteLLM / global-LB layer (topology-agnostic per Cloud-Native-Architect). Conversation affinity, config sync on leader-only, circuit-breaker and drain hardening. | As M4. |
| **M6** | Multi-region federation (DNS/global-LB layer, not VSR — VSR is single-cluster). Classifier-versioning cadence stable; VSR promotion-or-hold decision published. | Stable. |

**Gating rules:**
- Evidence ships M1-M2 or M3 does not start. The dispatcher without Evidence is a selector with no signal.
- Native trained arm ships alongside Evidence, not later. §5.2.
- VSR backend ships in M3 behind an SSE Phase-0 gate (TTFB delta ≤10%, inter-token jitter within native envelope, on streamed 4k-token completion, under load). Gate failure → drop VSR backend from that release; native path continues.

---

## Where agents disagreed

### Investigator phase

| Claim | VSR-Investigator | RouteIQ-Auditor | LockIn-Analyst | Integration-Arch-v2 | Steelman-v2 | Stackability | Stacking-Arch | Cloud-Native | Competitive | LiteLLM-Deep-Audit |
|---|---|---|---|---|---|---|---|---|---|---|
| VSR SSE/streaming OK | ? | — | — | ? | — | ✗ | ? | — | — | — |
| Topology A default | ○ | — | — | ✗ (B.2) | ○ | ✓ (8/10) | — | ○ | — | — |
| Topology B.2 coherent | — | — | — | ✓ | — | ✗ (3/10) | — | — | — | — |
| Topology C substrate-neutral | — | — | — | — | — | ✓ (7/10) | ✓ | ✓ | ✓ | — |
| LiteLLM auto_router commoditizes routing | — | — | ✓ | — | ○ | — | — | — | ✓ | ✓ |
| Python plugin coherence is A's problem | — | — | — | ○ | — | — | — | — | — | — |
| HA/MLOps is topology-agnostic | — | — | — | — | — | — | — | ✓ | — | — |
| Migration dominates stacking | — | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | — | ✗ | ✗ |

`✓` = supports / `✗` = rejects / `○` = hedged / `?` = flagged as open / `—` = not addressed.

### Deliberator phase

| Position | stacking-bull (final) | stacking-bear (final) | evidence-layer (final) |
|---|---|---|---|
| Topology A as default | ✗ (conceded) | ✗ | ✗ |
| Topology C as default | ✓ | ✓ | ✓ |
| Evidence Console first | ✓ | ✓ | ✓ (thesis) |
| VSR K8s-default at M3 | ✓ (held weakly) | ✗ | ✗ |
| VSR promotion conditional on M4 Evidence data | ○ | ✓ | ✓ (thesis) |
| SSE Phase-0 gate is non-negotiable | ✓ (conceded) | ✓ | ✓ |
| Dual deploy shape (pip + K8s) | ✓ (conceded) | ✓ | ✓ |
| Label-only disagreement (A-with-tiers vs C-with-enterprise) | ✓ (holds name "A") | ✗ (wants "C") | ✗ (wants "C") |

**Three remaining disagreements at doc-write time:**

1. **Labeling:** bull wants the recommendation called "Topology A with tiers"; bear and evidence-layer want "Topology C with VSR-as-enterprise-tier." Same binaries, different marketing. This doc resolves in favor of **C** — more honest about blast-radius containment and matches the actual per-tenant opt-in semantics.
2. **M3 VSR promotion timing:** bull wants VSR as the K8s-tier default at M3; evidence-layer wants promotion conditional on M4 retrain-cadence data; bear wants no promotion commitment. This doc resolves toward **evidence-layer** — decide from data, not advocacy.
3. **Dispatcher-sprawl guardrail:** bear was asked to name one testable discipline preventing the dispatcher from growing into "RouteIQ's own router." Bear's final message converged on C before answering. **Open design-discipline decision for M3 kickoff.** Proposed shape: "dispatcher MUST be a pure function of (tenant_id, feature_flags, request headers), MUST NOT parse request bodies, MUST emit routing_backend as a single OTel span attribute." The team must settle this before the M3 merge.

---

## Killshot risks + kill criteria

| Risk | Likelihood | Impact | Kill criterion |
|---|---|---|---|
| **Evidence-without-active-arm trap** (§5.2) | High if ignored | Nullifies the moat | Native `strategies.py` arm is a GATING requirement on M1-M2 ship. Non-negotiable. |
| **VSR SSE regression** | Medium | Streaming breakage on opt-in path | SSE Phase-0 gate at M3. TTFB delta ≤10%, inter-token jitter within native envelope, streamed 4k-token completion under load. Gate fail → drop VSR backend. |
| **LiteLLM Enterprise ships adaptive router with feedback (H2 2026)** | Medium | VSR delta shrinks further | Topology C keeps native in-process path alive. Pivot cost low — drop VSR backend, keep Evidence + native arm. |
| **Python plugin coherence under VSR** (§5.1) | High if Topology A | Governance inversion (PII routed pre-guard) | Topology A rejected. In Topology C, VSR opt-in carries explicit governance disclosure. |
| **Dispatcher sprawl** | Medium if unguarded | C collapses into its own third dataplane | M3 kickoff design discipline (pending). Proposed: pure-function dispatcher, no body parsing, single OTel span attribute. |
| **VSR v0.2 pre-v1 breakage** | Medium | Opt-in tenants affected by VSR upgrades | Pin VSR version per enterprise release; upgrade gated on SSE re-benchmark. |
| **Retrain cadence mismatch** (§5.3) | Low-medium | VSR pod-rotation bottlenecks MLOps loop | Decided by M4 Evidence data; if regressions require sub-hour rollback, VSR promotion is blocked. |

---

## Links

- v1 decision doc (superseded): `docs/architecture/vsr-vs-routeiq-decision.md`
- Evidence Console design: `docs/architecture/evidence-console-design.md`
- MLOps loop design: `docs/architecture/mlops-loop.md`
- Cloud-native review: `docs/architecture/cloud-native.md`
- LiteLLM custom routing strategy (plugin API hook): `src/litellm_llmrouter/custom_routing_strategy.py:114`
- In-process config hot reload: `src/litellm_llmrouter/hot_reload.py`
- Plugin callback bridge (the governance-ordering constraint from §5.1): `src/litellm_llmrouter/gateway/plugin_callback_bridge.py`
- OTel setup reused by Evidence Console: `src/litellm_llmrouter/observability.py`

---

*Produced by `routeiq-deliberation` team (decision-scribe synthesizer). 3 deliberators × 3 rounds over 8-investigator dossier. All three deliberators converged on Topology C with Evidence-first sequencing before this doc was written.*
