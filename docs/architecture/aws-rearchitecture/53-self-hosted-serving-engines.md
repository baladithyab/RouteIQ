# 53 — Self-Hosted Serving Engines: vLLM Production Stack, AIBrix, llm-d (the vLLM-native control planes)

> **Status**: Architecture / forward roadmap. **Date**: 2026-06-17.
> **Thesis**: The C3 self-hosted tier has **four** engine options, and three of
> them (**vLLM Production Stack**, **AIBrix**, **llm-d**) are *vLLM-native control
> planes* that share one shape: a KV-aware **router/gateway** over a fleet of vLLM
> serving pods, registered to RouteIQ as **ONE OpenAI `api_base` per model**. The
> fourth, **NVIDIA Dynamo**, is the backend-agnostic, multi-node-disaggregation
> deep tier already covered in
> [`51-multinode-large-model-serving.md`](./51-multinode-large-model-serving.md).
> This doc fills the gap: `50-...` §4.3 gives a *one-row summary* of all four;
> `51-...` deep-dives Dynamo; **nothing deep-dives Production Stack / AIBrix /
> llm-d** — until now.
>
> **Honesty contract.** Claims are tagged **[VERIFIED]** (grounded in the engine
> repos' own docs read 2026-06-17 via DeepWiki against `vllm-project/production-stack`,
> `vllm-project/aibrix`, `llm-d/llm-d`, and `vllm-project/vllm` — cited inline) or
> **[SPECULATIVE]** (reasoned synthesis / a design we have not built). The ranked
> `engineVerdict` and the EKS GPU-gap facts are owned by `50-...` §4.3/§4.4 and the
> `research/notes/final_report_eks-inference-engines-litellm-6a7af4.md` report; this
> doc does not re-adjudicate them — it expands the three vLLM-native rows.
>
> **Sources.**
> - [`50-litellm-universal-surface.md`](./50-litellm-universal-surface.md) §4 — the
>   ranked `engineVerdict` (raw vLLM → Production Stack → AIBrix → llm-d → Dynamo),
>   the two-layer routing model, the "one `api_base` per model" hard rule, and the
>   EKS GPU NodePool gap (§4.4). **The layering + verdict authority — not re-derived.**
> - [`51-multinode-large-model-serving.md`](./51-multinode-large-model-serving.md) —
>   the Dynamo C3-deep multi-node + EFA + KVBM deep-dive, and the **Grove+KAI vs
>   LWS+Volcano** gang-scheduling pairing this doc reuses (§1.2). **The gang-scheduler
>   authority.**
> - [`52-self-hosted-engine-kvbm.md`](./52-self-hosted-engine-kvbm.md) — the cred-free
>   vs operator-gated split this doc mirrors; the `config/config.self-hosted-engine.yaml`
>   contract.
> - DeepWiki reads (2026-06-17) of `vllm-project/production-stack`,
>   `vllm-project/aibrix`, `llm-d/llm-d`, `vllm-project/vllm` — cited inline as
>   **[VERIFIED — DeepWiki <repo>]**.

---

## 0. Where this sits — the four-engine fork under one `api_base`

`50-...` §4.3 establishes the ranked adoption order and the one invariant that makes
all four engines interchangeable *to RouteIQ*: **every engine exposes an
OpenAI-compatible `/v1` endpoint, and RouteIQ registers the engine's
gateway/router Service as exactly ONE `model_list` arm.** The choice between them is
**operational / platform / maturity — never API compatibility.**

```
caller: model="oss-70b"
        │
   ┌────▼───────────────────────────────────────┐
   │ RouteIQ  Layer 1: model SELECTION           │  (unchanged — 50-... §4.2)
   │ K-Thompson bandit over healthy_deployments  │
   └────┬───────────────────────────────────────┘
        │ picks ONE deployment: api_base = http://<engine-gateway>/v1
   ┌────▼──────────────────────────────────────────────────────────────┐
   │ ONE engine gateway/router  (ONE api_base — everything below opaque) │
   │   Production Stack router │ AIBrix Router (Envoy ext-proc) │        │
   │   llm-d EPP (GAIE)        │ Dynamo Smart Router (→ 51-...)          │
   │   Layer 2: replica SCHEDULING (KV-aware) over a fleet of vLLM pods  │
   └────────────────────────────────────────────────────────────────────┘
```

This doc covers the **three vLLM-native control planes** (Production Stack, AIBrix,
llm-d) at the depth `51-...` gives Dynamo. The decision spine:

| Engine | One-line "pick this when" | K8s integration weight | Multi-node primitive | Maturity (2026-06) |
|---|---|---|---|---|
| **raw `vllm serve`** | A single replica is enough; you want zero control-plane. | Device-plugin only | none (single replica) | engine GA |
| **vLLM Production Stack** | Smallest step up from raw vLLM to **KV-cache-aware routing + KEDA autoscaling**, Helm-only, no new gateway CRDs. | Helm + KEDA | **LWS** (LMCache/NIXL disagg) | reference impl |
| **AIBrix** | **Most production-proven**; you need **LoRA-at-scale**, distributed KV (L1 DRAM/L2 remote), **11 routing algos**, or **P/D disaggregation without Ray**. | 5 CRDs + controller-manager; **default scheduler** | **StormService** (no-Ray P/D) or **KubeRay** gang | ByteDance-proven, v0.3+ |
| **llm-d** | You **standardize on the Gateway API Inference Extension (GAIE)**, want **precise prefix-cache routing**, P/D disaggregation, or **cross-vendor accelerators** (NVIDIA/AMD/TPU/HPU). | **STRICTLY requires GAIE + Gateway API CRDs + a Gateway provider** | **LWS** (EP/TP over RDMA) | CNCF Sandbox v0.7 |
| **NVIDIA Dynamo** | **Multi-node, model-too-big-for-one-node**, backend-agnostic, disaggregation at ~8+ GPU nodes. | GPU Operator + EFA + Grove/KAI or LWS/Volcano | **Grove+KAI** (or LWS+Volcano) | 1.0 GA — see `51-...` |

> **The first cut is single-node vs multi-node.** Production Stack and AIBrix are the
> **single-node-replica common path** (C3a/C3b in `50-...`/`51-...`): each replica fits
> one node, the router load-balances across replicas with KV-cache awareness, and you
> never touch EFA. llm-d and Dynamo are the **multi-node** options (a replica spans
> nodes via LWS/Grove) — and at that point the EFA + gang-scheduling machinery of
> `51-...` Part 1 applies. **Adopt KV-aware routing (Production Stack / AIBrix) WITHOUT
> adopting disaggregation** (`50-...` §4.3 corollary): disaggregation is scale-gated.

---

# PART 1 — vLLM Production Stack (the smallest step up)

## 1.1 What it is

**[VERIFIED — DeepWiki vllm-project/production-stack]** The vLLM Production Stack is a
**reference implementation** for an LLM inference stack on Kubernetes, deployed via a
**Helm chart**. Three components:

1. **vLLM serving engines** — vLLM instances (one model each) as K8s pods + Services.
2. **Request router** — a **FastAPI** app that does service discovery, load balancing,
   and routing; it directs each request to the best backend engine and exposes its own
   `/metrics`. **This router Service is the `api_base` RouteIQ registers.**
3. **Observability stack** — Prometheus + Grafana over the backends.

## 1.2 KV-cache-aware routing

**[VERIFIED — DeepWiki]** The router's headline feature is **KV-cache-aware routing**:
it directs each incoming request to the vLLM instance with the **highest KV-cache hit
rate** for that request's prefix, maximizing cache reuse. This is distinct from plain
*prefix-aware* routing (which routes on the prefix even after the cache was evicted).
Enabled by setting `lmcacheConfig` with a unique `instanceId` per engine in the Helm
values. The router scrapes the engines' `vllm:*` Prometheus metrics (queue depth,
prefix-cache hit rate — see Part 4) to make these decisions.

## 1.3 Kubernetes integration + multi-node

**[VERIFIED — DeepWiki]**
- **Helm-only, no new gateway CRDs.** `values-*.yaml` defines model specs, resource
  requests, and per-engine vLLM config. This is the *lightest* K8s footprint of the
  three — no Gateway API, no operator CRDs beyond the chart. **This is exactly why
  `50-...` §4.3 ranks it 1a: smallest step up from raw vLLM.**
- **Multi-node via LMCache + NIXL.** The `lmcacheConfig` exposes `enableController`,
  `controllerPort`, `workerPort`, `kvRole`, `enableNixl`, `nixlRole`, `nixlPeerHost`,
  `nixlPeerPort` — i.e. **distributed KV-cache management + disaggregated prefill over
  NIXL across nodes** (LMCache is the KV-offload/transfer layer). The multi-node primitive
  for "leader + workers as one unit" is **LeaderWorkerSet (LWS)** — the same sig-apps
  primitive `51-...` §1.2 names as the **LWS+Volcano** fallback pairing for Dynamo.
  **[SPECULATIVE]** On AWS, that cross-node NIXL path lands on **EFA** with the exact
  `51-...` Part 1 plumbing (LIBFABRIC backend, the silent ~98s-vs-~1s TTFT trap); the
  Production Stack does not change that calculus.

## 1.4 Autoscaling

**[VERIFIED — DeepWiki]** Autoscaling is via **KEDA** (Kubernetes Event-driven
Autoscaling), configured under `servingEngineSpec.modelSpec[].keda` in the Helm values:
min/max replicas, polling interval, cooldown, and **Prometheus-based triggers with
PromQL queries** over the vLLM metrics (e.g. scale on `vllm:num_requests_waiting`). So
the autoscaler is **metric-driven off the same `/metrics` RouteIQ's engine-metrics
scraper reads** (Part 4) — queue-depth-aware scaling is built in.

---

# PART 2 — AIBrix (the production-proven, feature-rich option)

## 2.1 What it is

**[VERIFIED — DeepWiki vllm-project/aibrix]** AIBrix (originated at **ByteDance**,
now under the vllm-project org) is a **Kubernetes-native platform** that sits as a
**middleware layer between Kubernetes and the inference engines** (vLLM, SGLang). It is
`50-...` §4.3's **most production-proven** option. It is built from five CRDs and a
single `controller-manager` binary hosting six controllers.

## 2.2 Gateway, router, and the 11 routing algorithms

**[VERIFIED — DeepWiki]**
- **Gateway = Envoy Gateway** with a custom **AIBrix Router** embedded as an **Envoy
  external-processing (`ext-proc`) extension** — the single entry point for all
  inference requests. The gateway is exposed via a **LoadBalancer Service in
  `envoy-gateway-system`** and serves the OpenAI surface (`/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`, `/v1/models`). **This gateway Service is the
  `api_base` RouteIQ registers.** No GAIE required (its own Envoy plugin).
- **11 routing algorithms**: `random`, `least-request`, `throughput`, `prefix-cache`,
  `least-busy-time`, `least-kv-cache`, `least-latency`, `prefix-cache-preble`,
  `vtc-basic`, `pd` (prefill-decode disaggregation), `session-affinity`. The strategy
  can be selected **per-request via a header**.
- **KV-cache-aware routing**: `prefix-cache` routes to a pod that already holds the
  request's prefix in KV cache; `least-kv-cache` routes to the pod with the smallest
  current KV-cache footprint (VRAM relief). **[VERIFIED — DeepWiki]** The router reads
  **cached metrics from Redis** (high-frequency local cache via periodic pulls +
  subscriptions) so routing decisions never block on live pod queries — low hot-path
  overhead.

## 2.3 LoRA-aware routing + distributed KV cache

**[VERIFIED — DeepWiki]**
- **LoRA at scale** via the **ModelAdapter Controller**: dynamic load/unload of LoRA
  adapters through the `aibrix_runtime` sidecar, creating a **Service + EndpointSlice
  per adapter** for high-density multi-LoRA. The router routes to the right adapter
  instance by the requested model/adapter — **this is the unique reason to pick AIBrix
  when you serve many LoRA variants behind one base model.**
- **Distributed KV cache** (v0.3.0+): an **L1 DRAM** cache + optional **L2 remote**
  cache for **cross-engine KV reuse** — analogous in spirit to Dynamo KVBM's G2/G4 tiers
  (`51-...` Part 2) but AIBrix-native, not KVBM.

## 2.4 Kubernetes integration — CRDs, scheduler, gang scheduling

**[VERIFIED — DeepWiki]**
- **5 CRDs**: `ModelAdapter`, `PodAutoscaler`, `RayClusterFleet`,
  `RayClusterReplicaSet`, `StormService`. One `controller-manager` runs six controllers
  (pod-autoscaler, distributed-inference, model-adapter, kv-cache, stormservice,
  modelrouter).
- **Scheduler = the DEFAULT kube-scheduler.** Controllers emit standard Deployments /
  Services / HPA-KPA objects; the default scheduler places them. **This is a key
  contrast with `51-...`**: AIBrix does **not** require KAI/Volcano/Grove for the common
  path. Gang scheduling enters only for distributed inference:
  - **KubeRay** (optional) for Ray-based distributed inference (`RayClusterFleet` /
    `RayClusterReplicaSet`); the distributed-inference controller is **auto-skipped if
    KubeRay CRDs are absent** — so you only take the Ray dependency if you want it.
  - **StormService** orchestrates **prefill-decode disaggregation with Adaptive Flow
    Detection (AFD) for multi-node inference WITHOUT requiring Ray** — the no-Ray P/D
    path. This is AIBrix's answer to the `51-...` Dynamo disaggregation tier, but
    self-contained.

## 2.5 Autoscaling

**[VERIFIED — DeepWiki]** Three strategies via the `PodAutoscaler` CRD:
- **HPA** — native K8s HPA on CPU.
- **KPA** — Knative-style with a **panic window** for rapid scale-up.
- **APA** — AIBrix's **Advanced Pod Autoscaler** with fluctuation parameters to **damp
  oscillation** + profile-based **proactive** scaling.

It supports **all vLLM metrics** and **multi-metric autoscaling** (final replica count =
the metric demanding the most replicas), using LLM-specific signals like
`gpu_cache_usage_perc` / `gpu_kv_cache_utilization` and token throughput — again the same
`/metrics` family the engine-metrics scraper (Part 4) parses.

---

# PART 3 — llm-d (the Gateway-API-Inference-Extension option)

## 3.1 What it is

**[VERIFIED — DeepWiki llm-d/llm-d]** llm-d is a **CNCF Sandbox** project founded by
**Red Hat, Google Cloud, IBM Research, CoreWeave, and NVIDIA** — a Kubernetes-native
distributed-inference stack that integrates **vLLM (default engine) + the Kubernetes
Gateway API + Kubernetes as the control plane**, validated across **NVIDIA / AMD / Google
TPU / Intel HPU** accelerators. Three tiers:

1. **Gateway layer** — K8s **Gateway API** (`Gateway` + `HTTPRoute` CRDs), TLS, ingress.
2. **Inference control plane (the llm-d Router)** — a **Proxy** (Envoy) that consults an
   **Endpoint Picker (EPP)** over the `ext-proc` protocol; **the EPP is the scheduler.**
3. **Model serving layer** — vLLM/SGLang pods on accelerators.

The OpenAI surface (`/v1/chat/completions`, etc.) is served **through the Gateway → EPP →
selected model server**; the EPP has out-of-the-box parsers for the OpenAI HTTP API.
**The Gateway Service is the `api_base` RouteIQ registers.**

## 3.2 The hard dependency: Gateway API Inference Extension (GAIE)

**[VERIFIED — DeepWiki + `50-...` §4.3]** llm-d **STRICTLY requires** the **Gateway API
Inference Extension (GAIE)** — Gateway API CRDs + the **`InferencePool`** CRD + a Gateway
provider (Istio / agentgateway / GKE Gateway). The `InferencePool` is the central
service-discovery resource that bridges Gateway ↔ EPP ↔ model-server pods via label
selectors and target ports (port 8000 for vLLM). Two deployment modes:
- **Standalone** — proxy as a sidecar to the EPP in the same pod (`ext-proc` over
  localhost).
- **Gateway (Inference Gateway)** — the official K8s Gateway API for shared
  infrastructure, multi-cluster LB, advanced traffic management.

> **This is the load-bearing "pick-when".** Choose llm-d **iff you are standardizing on
> GAIE / Gateway API as your org's inference-networking substrate** — otherwise the
> Gateway-provider + GAIE CRD dependency is pure overhead vs Production Stack/AIBrix
> (which ship their own gateway and need no GAIE). On EKS this means installing a Gateway
> provider; `50-...` §4.3 notes the single-node path "drops on cleanly," but the GAIE/CRD
> prerequisite is real.

## 3.3 KV-cache-aware routing (the EPP scoring pipeline)

**[VERIFIED — DeepWiki]** The EPP makes per-request routing via a **pluggable plugin
pipeline**: **Filters** (pre-filter candidate endpoints) → **Scorers** → **Pickers**.
Built-in scorers: **queue-scorer** (queue depth), **kv-cache-utilization-scorer** (HBM +
host-memory pressure), **precise-prefix-cache-scorer** (prefix locality),
**predicted-latency-scorer** (XGBoost latency prediction). Two KV-cache-aware
implementations:
- **Approximate** — char-to-token ratios + rolling hash chains, **no external deps**.
- **Precise** — 100% accuracy using actual token data + **real-time state from model
  servers over ZeroMQ events** (a global index of vLLM KV-cache state).

## 3.4 Multi-node + gang scheduling

**[VERIFIED — DeepWiki]**
- **LeaderWorkerSet (LWS)** for models exceeding one node (e.g. DeepSeek-R1): LWS manages
  a group of pods as one logical unit, enabling **Expert Parallelism (EP) + Tensor
  Parallelism (TP) across an RDMA mesh**. The Wide-EP guide deploys DeepSeek-R1 with LWS +
  vLLM **P/D disaggregation over NIXL** — i.e. llm-d's multi-node path is **LWS-based**,
  matching the `51-...` §1.2 **LWS** primitive (and on AWS the same EFA/NIXL-LIBFABRIC
  fabric applies).
- **Disaggregated prefill/decode (P/D)** — separate prefill and decode fleets, scaled
  independently, KV transferred over **NIXL/RDMA**: lower TTFT, more predictable TPOT.
- **Flow Control** — intelligent request queuing with **multi-tenant fairness, strict
  priority dispatch, late-binding scheduling** (llm-d's gang/admission layer; not a
  cluster gang-scheduler like KAI/Volcano but an in-EPP queueing discipline).

## 3.5 Autoscaling — well-lit paths

**[VERIFIED — DeepWiki]** llm-d ships **tested, benchmarked "well-lit path" recipes**.
Autoscaling has two complementary patterns:
- **HPA on llm-d Router metrics** — K8s-native scaling on **queue depth + request
  counts** from the Router's internal metrics.
- **Workload Variant Autoscaler (WVA)** — **multi-model, SLO-aware** scaling on
  heterogeneous hardware, optimizing cost by routing across model variants.

The "Workload Autoscaling" well-lit path is explicitly **proactive, SLO-aware, driven by
queue depth + KV-cache pressure** — the same signals Part 4's scraper exposes.

---

# PART 4 — How they plug into RouteIQ (the integration that is identical for all four)

## 4.1 One `api_base` per model — the invariant (carried from `50-...` §4.2 / `51-...` §3.2)

For **every** engine, RouteIQ registers the engine's **gateway/router Service** as **ONE**
`model_list` arm and delegates ALL per-replica / KV-aware / disaggregation scheduling
**downward**:

```yaml
model_list:
  - model_name: oss-70b
    litellm_params:
      model: hosted_vllm/meta-llama/Llama-3.1-70B-Instruct  # the model id the ENGINE serves
      api_base: http://<engine-gateway-or-router-service>/v1 # ONE per model; NEVER a pod IP
      api_key: fake-api-key                                   # hosted_vllm sentinel
# Production Stack: http://<release>-router-service.<ns>.svc.cluster.local/v1
# AIBrix:          http://<envoy-gateway-lb>.envoy-gateway-system.svc.cluster.local:8000/v1
# llm-d:           http://<gateway>.<ns>.svc.cluster.local/v1   (the GAIE Gateway)
# Dynamo:          http://dynamo-frontend.dynamo-system.svc.cluster.local:8000/v1  (→ 51-...)
```

**The one config error to avoid** (verbatim from `52-...` §2 and `51-...` §3.2): never
register individual workers / replicas / prefill+decode pods as separate `model_list`
rows. That collapses Layer-1 model-selection into Layer-2 replica-scheduling and makes
RouteIQ take **cache-blind, topology-blind** decisions that **fight** the engine's own
KV-aware router (Production-Stack router / AIBrix Router / llm-d EPP / Dynamo Smart
Router). It is *fine* to have a second `model_list` row under the same `model_name` when
it is a genuinely separate **capacity source** (a Bedrock arm; a second engine) — never
a second row pointing *inside* one engine group.

## 4.2 The cred-free config arms

`config/config.self-hosted-engine.yaml` carries **three commented, cred-free engine arms
(vLLM Production Stack / AIBrix / llm-d)** plus the live default arm — each a single
`hosted_vllm/<model>` row whose `api_base` is the engine's in-cluster gateway Service.
Uncommenting one (with the real Service DNS) is the entire RouteIQ-side change; the live
engine deploy (GPU NodePool, Helm/CRDs, GAIE for llm-d) is **operator-gated** (`52-...`
§4). Default render ships **zero** self-hosted arms (byte-stable when off).

## 4.3 The engine-side metrics scrape (RouteIQ-ffaa) — feeding a future KV/queue-aware router

**[VERIFIED — DeepWiki vllm-project/vllm]** All four engines re-export the **vLLM
`vllm:*` Prometheus metric family** on the serving pods' `/metrics`, and the routers
(Production-Stack router, AIBrix Router, llm-d EPP) consume exactly these to make Layer-2
decisions. RouteIQ ships a **cred-free, default-off scraper**
(`src/litellm_llmrouter/engine_metrics.py`, gated by `ROUTEIQ_ENGINE_METRICS__ENABLED`)
that GETs an engine `/metrics` endpoint and parses the key gauges so a **future
KV/queue-aware Layer-1 router** (or an autoscaler-into-the-engine) can consume them:

| Metric | Type | What it tells RouteIQ |
|---|---|---|
| `vllm:num_requests_waiting` | Gauge | **queue depth** — the primary queue-aware-routing / autoscale signal |
| `vllm:num_requests_running` | Gauge | requests in the running batch |
| `vllm:num_requests_swapped` | Gauge | requests swapped to CPU (pressure) |
| `vllm:kv_cache_usage_perc` | Gauge | **KV-cache fraction used (0–1)** — vLLM **v1** name |
| `vllm:gpu_cache_usage_perc` | Gauge | the **pre-v1 alias** for the above — scraper accepts BOTH |
| `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` | Counter | v1 prefix-cache counters (hit rate = `rate(hits)/rate(queries)`) |

> **[VERIFIED — DeepWiki vllm-project/vllm] The v1 rename trap.** vLLM **v1** renamed
> `vllm:gpu_cache_usage_perc` → **`vllm:kv_cache_usage_perc`** (the old name does **not**
> exist in v1), and replaced the single prefix-cache-hit-rate gauge with two counters
> (`vllm:prefix_cache_hits`, `vllm:prefix_cache_queries`). The scraper parses **both**
> cache-usage names and `EngineMetricsSnapshot.kv_cache_usage()` returns whichever is
> present — so it works across engine/vLLM versions. AIBrix autoscaling docs still cite
> the v0 `gpu_cache_usage_perc`; the dual-name handling is load-bearing, not cosmetic.

The scraper reads the **engine frontend's aggregate** `/metrics` (so Layer-1 can decide
*whether to send more load to this engine arm at all*, or autoscale it) — it does **not**
scrape individual worker pods, which would collapse the two-layer model (§4.1). It is
**graceful on unreachable**: a down/timed-out/garbage engine yields an empty
`reachable=False` snapshot, never an exception that could crash a routing decision.

---

# PART 5 — Decision tree + phased slotting

## 5.1 Decision tree — which engine?

```
            START: a self-hosted model to serve on EKS
                            │
            Is ONE replica enough (no cross-replica routing,
            no autoscaling, no KV-aware scheduling)?
                  ┌─────────┴──────────┐
                YES                    NO
                  │                     │
          ┌───────▼───────┐    Does ONE replica fit ONE node?
          │ raw vllm serve │      ┌──────┴───────────┐
          │ (the floor)    │     YES                 NO (too big for one node)
          └───────────────┘       │                   │
                          Need many LoRA variants /    │
                          distributed KV / no-Ray P/D? │
                            ┌──────┴───────┐           │
                          YES             NO           │
                            │              │           │
                     ┌──────▼─────┐  Standardizing on  │
                     │  AIBrix    │  GAIE / Gateway API?│
                     │ (most      │   ┌──────┴──────┐   │
                     │  proven;   │  YES           NO   │
                     │  default   │   │             │   │
                     │  scheduler;│ ┌─▼────┐  ┌──────▼────────┐
                     │  StormSvc/ │ │llm-d │  │ vLLM Prod.    │
                     │  KubeRay)  │ │(EPP, │  │ Stack         │
                     └────────────┘ │ GAIE,│  │ (Helm + KEDA, │
                                    │ LWS) │  │  KV-aware,    │
                                    └──────┘  │  smallest step│
                                              │  up, LWS for  │
                                              │  multi-node)  │
                                              └───────────────┘
                                                          │
                                       (multi-node, backend-agnostic,
                                        disaggregation at 8+ GPU nodes)
                                                          │
                                              ┌───────────▼────────────┐
                                              │ NVIDIA Dynamo → 51-...  │
                                              │ (EFA + Grove/KAI or     │
                                              │  LWS/Volcano + KVBM)    │
                                              └─────────────────────────┘
```

## 5.2 Slotting against the C0–C3 roadmap (`50-...` §5)

- **C3a/C3b (single-node common path):** **vLLM Production Stack** *or* **AIBrix** — both
  EKS-Auto-Mode-clean (device plugin auto-installed on GPU nodes, no GPU Operator), no
  EFA, no node group, registered as ONE `api_base`. Start with whichever the team's
  needs select per §5.1; `50-...` §4.3 leans AIBrix for production-proven, Production
  Stack for the smallest step. **The engine-metrics scraper (Part 4) lands here, default
  off** — cred-free, byte-stable, ready for a future KV/queue-aware router to consume.
- **C3-deep (multi-node, EFA):** **llm-d** (LWS + GAIE + EFA/NIXL) **or** **Dynamo**
  (`51-...`: Grove+KAI / LWS+Volcano + EFA + KVBM). Both inherit `51-...` Part 1's EFA +
  gang-scheduling + LIBFABRIC-backend machinery unchanged. **Build the EFA node source
  only when a model actually needs multi-node serving** (`51-...` §4.2).

## 5.3 Gang-scheduling pairing — the established repo convention

`51-...` §1.2 records the verified pairing for the **Dynamo** multi-node tier — **Grove
(PodCliqueSet) + KAI (recommended)** or **LWS + Volcano (fallback)** — and corrects the
non-existent "Grove-Volcano" pairing. For the three engines here:
- **Production Stack** and **llm-d** use **LWS** as their multi-node leader+worker
  primitive (LWS+Volcano-style gang scheduling on AWS, same as the `51-...` fallback).
- **AIBrix** uses the **default kube-scheduler** for the common path, **KubeRay** for
  Ray-based distributed inference, and **StormService (AFD)** for no-Ray P/D — it does
  **not** need KAI/Grove/Volcano.
- **Dynamo** is the one that prefers **Grove+KAI** (`51-...`). Do not cross these wires.

## 5.4 Acceptance criteria (cred-free half)

- **One `api_base` per model** — never per-worker/replica (§4.1); the chart render test
  (`52-...` §2) guards the seam; default render ships zero self-hosted arms.
- **Engine-metrics scraper** — parses the vLLM `vllm:*` gauges from a fixture `/metrics`
  body (both the v1 `kv_cache_usage_perc` and the pre-v1 `gpu_cache_usage_perc`), is
  **default-off** (zero network I/O when disabled), and returns a graceful empty snapshot
  on an unreachable engine. Covered by `tests/unit/test_engine_metrics.py`.
- **Operator-gated** — the live engine deploy (GPU NodePool, Helm/CRDs, GAIE for llm-d,
  Envoy Gateway for AIBrix, EFA for the multi-node tiers) is out of scope here (`52-...`
  §4 split).

---

## 6. One-paragraph synthesis

The C3 self-hosted tier has four engines and they are **interchangeable to RouteIQ** —
each exposes an OpenAI `/v1` and is registered as **ONE `api_base` per model**, with all
KV-aware / disaggregation scheduling delegated to the engine below that one endpoint — so
the choice is **operational, not API**. **vLLM Production Stack** is the smallest step up
(Helm + KEDA, KV-cache-aware FastAPI router, LWS for multi-node, no new gateway CRDs);
**AIBrix** is the most production-proven (ByteDance-origin, Envoy `ext-proc` router with
11 routing algos, LoRA-at-scale via ModelAdapter, distributed L1/L2 KV cache, **default
kube-scheduler** with KubeRay/StormService only for distributed inference, HPA/KPA/APA
autoscaling); **llm-d** is the **Gateway-API-Inference-Extension** option (Red Hat/Google/
IBM CNCF Sandbox, EPP scorer pipeline with precise prefix-cache routing, LWS + NIXL P/D
across vendors) — pick it **iff you standardize on GAIE**, which is its load-bearing
prerequisite; and **NVIDIA Dynamo** (`51-...`) is the backend-agnostic multi-node
disaggregation deep tier with Grove+KAI/LWS+Volcano + EFA + KVBM. All four re-export the
vLLM `vllm:*` metric family, which RouteIQ's **cred-free, default-off engine-metrics
scraper** (`engine_metrics.py`, RouteIQ-ffaa) parses — handling the **vLLM-v1
`gpu_cache_usage_perc`→`kv_cache_usage_perc` rename** — so a future KV/queue-aware Layer-1
router (or autoscaler-into-the-engine) has the queue-depth + KV-pressure signals to
consume, with the live engine deploy left operator-gated.
