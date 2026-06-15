# 50 — LiteLLM as the Universal Surface: One Arm-Set, Trained Routing Above It

> **Status**: Architecture / forward roadmap. **Date**: 2026-06-15.
> **Thesis**: LiteLLM is the **one** OpenAI-compatible surface in front of *everything*.
> RouteIQ's trained routing (Kumaraswamy-Thompson + the 18-strategy toolkit) and
> governance sit **above** that surface and route uniformly across **all** capacity
> sources. Every backend — a cloud-provider API key, a cross-region Bedrock, a
> cross-account assume-role Bedrock, an AgentCore-fronted path, a self-hosted
> vLLM / Dynamo / llm-d / AIBrix endpoint — is *just another LiteLLM deployment*:
> an **arm** in `healthy_deployments` that the bandit routes over.
>
> **Honesty contract.** Claims are tagged **[VERIFIED]** (grounded in
> `reference/litellm/` source read at a cited file:line, in the discover notes
> under `research/universal-surface/`, or in the research report) or
> **[SPECULATIVE]** (reasoned synthesis / inferred from API semantics / a design
> we have not yet built). The **engine** facts (Dynamo / llm-d / AIBrix / vLLM)
> are owned by the research report
> `research/notes/final_report_eks-inference-engines-litellm-6a7af4.md`; this doc
> does not re-adjudicate them.
>
> **Sources.**
> - `research/universal-surface/discover-litellm-capacity.md` — LiteLLM Router
>   federation primitive, cited against `reference/litellm/` (read-only submodule).
> - `research/universal-surface/discover-capacity-eks-gap.md` — VSR cross-account
>   capacity pattern + RouteIQ's EKS GPU NodePool gap.
> - `research/universal-surface/synth-digest.json` — engine `engineVerdict` +
>   key findings (digest of the research report).
> - Sibling doc: `agentcore-integration-and-arch-2026-06-15.md` (AgentCore — §3 here
>   is a pointer, not a duplicate).

---

## 0. The single picture

```
                    caller sends  model="claude-sonnet"  (OpenAI /v1/chat/completions)
                                    │
            ┌───────────────────────▼──────────────────────────────┐
            │  RouteIQ  (own FastAPI app, ADR-0012)                  │   ← LAYER 1: model SELECTION
            │  governance · usage policies · guardrails · OIDC      │     (which arm, by quality/cost/latency)
            │  CustomRoutingStrategyBase → Kumaraswamy-Thompson      │
            │  bandit + 18 strategies routes over healthy_deployments│
            └───────────────────────┬──────────────────────────────┘
                                    │  picks ONE deployment dict
            ┌───────────────────────▼──────────────────────────────┐
            │  LiteLLM Router  (installed as routing-strategy plugin)│   ← the SURFACE / capacity substrate
            │  model_list → model_group → healthy_deployments        │
            │  cooldown · failover · per-deploy rpm/tpm              │
            └──┬─────────┬──────────┬──────────┬─────────────┬──────┘
               │         │          │          │             │
       arm: API-key  arm: x-region arm: x-acct arm:AgentCore arm: self-hosted api_base
        pool (§2a)    Bedrock (§2b) Bedrock(§2c)  path (§3)    vLLM/Dynamo/llm-d/AIBrix (§4)
                                                                   │
                                                                   ▼  ← LAYER 2: replica SCHEDULING
                                                        engine's KV-aware / disagg router
                                                        (Dynamo Smart Router / llm-d EPP /
                                                         AIBrix Router) — sits BELOW Layer 1
```

The entire design rests on one LiteLLM primitive — **N deployments may share one
`model_name`** — and one RouteIQ seam — **`CustomRoutingStrategyBase` reads
`healthy_deployments` as a dynamic arm set**. Everything else is "what field do you
vary to add an arm."

---

## 1. The arm-set model — every capacity source is a LiteLLM deployment

**[VERIFIED]** LiteLLM's `Router` is built from a `model_list`: a flat list of
**deployments**. Each deployment is a dict with two load-bearing keys
(`discover-litellm-capacity.md` §0):

- `model_name` — the **logical alias** the caller asks for (e.g. `"claude-sonnet"`).
- `litellm_params` — the **physical wiring** (`model`, `api_key`, `api_base`,
  `aws_region_name`, `aws_role_name`, `rpm`, `tpm`, …).

**N deployments sharing one `model_name` = a *model group* = a capacity pool.** The
caller sends `model="claude-sonnet"`; the Router picks exactly one concrete
deployment per request. This is the whole federation primitive: *one surface name,
many backing endpoints.*

### How a deployment becomes a distinct arm

**[VERIFIED]** `_generate_model_id(model_group, litellm_params)`
(`reference/litellm/litellm/router.py:5592-5621`) gives every deployment a stable,
unique `model_info["id"] = sha256(model_group + all litellm_params)`. Two deployments
under the same `model_name` that differ in **any** `litellm_param` — a different
`api_key`, a different `aws_region_name`, a different `aws_role_name`, a different
`api_base` — therefore hash to **different ids** and are **distinct, independently
cool-downable arms**. This single hash is the mechanism that turns "same model,
different key / region / account / endpoint" into separable capacity units.

The group is indexed by `self.model_name_to_deployment_indices: Dict[str, List[int]]`
(`router.py:447-452`), built automatically by `set_model_list()`.

### Cooldown / failover — the arm set self-heals per request

**[VERIFIED]** (`discover-litellm-capacity.md` §1):

- A failed call runs `deployment_callback_on_failure` (`router.py:5180-5248`);
  `_should_cooldown_deployment` (`router_utils/cooldown_handlers.py:166-257`) decides.
  A **429** cools the arm immediately *unless it is the only arm in the group*
  (`is_single_deployment_model_group`, `cooldown_handlers.py:191-194`) — you don't
  cool down your only capacity source.
- `add_deployment_to_cooldown` (`cooldown_cache.py:69-105`) writes a TTL'd key to the
  DualCache (in-memory + optional Redis). **The TTL *is* the cooldown**: the key
  auto-expires and the arm silently rejoins. Redis-backing makes cooldown state
  **shared across workers** — multi-worker safe.
- Per request, `async_get_healthy_deployments` (`router.py:7838-7848`) computes
  `healthy_deployments = model_group − cooldown − filters`;
  `_filter_cooldown_deployments` (`router.py:8507-8527`) is the O(1) set-membership
  drop.
- Capacity reach is two-tier: `num_retries` re-enters `async_get_available_deployment`
  and lands on a *different healthy sibling in the same group*; `fallbacks`
  (`router.py:501-508`) fail over to a *different model group* once the primary is
  exhausted.

### How RouteIQ sees the arm set — `CustomRoutingStrategyBase`

**[VERIFIED]** The seam (`discover-litellm-capacity.md` §7):

- `CustomRoutingStrategyBase` (`reference/litellm/litellm/types/router.py:674-718`)
  declares `async_get_available_deployment(...)` (+ sync twin); both must "return an
  element from `litellm.router.model_list`" (`:694`, `:718`) — pick exactly one arm.
- `Router.set_custom_routing_strategy(...)` (`router.py:8648-8669`) sets
  `async_get_available_deployment` to RouteIQ's method, **bypassing the built-in
  load-balancer dispatch** (`simple-shuffle`, `least-busy`, `latency-based`, etc.).
  RouteIQ installs this per ADR-0002 (multi-worker safe — per-Router-instance attribute,
  not a global monkey-patch).
- RouteIQ's `RouteIQRoutingStrategy._get_model_list(model)`
  (`src/litellm_llmrouter/custom_routing_strategy.py:745-765`) reads
  `getattr(self._router, "healthy_deployments", self._router.model_list)` — it
  **explicitly prefers `healthy_deployments` over the static `model_list` "for
  freshness"** (`:749`, `:754-756`), filters to the requested `model_name`, scores the
  candidates, then `_match_deployment(...)` (`:797`) maps the winner back to one
  deployment dict.

**Why "dynamic arm set" is precise.** The set RouteIQ scores changes per request with
**zero RouteIQ-side bookkeeping**: LiteLLM has already *added* arms (a new
`model_list` row from config / hot-reload), *removed* arms (a 429'd deployment dropped
for its cooldown TTL), and *re-added* arms (TTL expiry). The bandit never tracks keys,
regions, accounts, or endpoint health — those are LiteLLM's federation substrate,
surfaced as one list. **RouteIQ contributes reward intelligence (which arm is best for
*this* prompt); LiteLLM contributes capacity intelligence (which arms exist and are
alive *right now*).**

### Concrete config sketch — one alias, four kinds of arm

The point of the universal surface is that *heterogeneous* backends can sit in **one
group**. A single `model_name` can front an API-key pool, cross-region Bedrock,
cross-account Bedrock, and a self-hosted box at once:

```yaml
model_list:
  # arm A — cloud provider API key (capacity pool member; §2a)
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_KEY_A
      rpm: 10000

  # arm B — cross-region Bedrock (different aws_region_name; §2b)
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-west-2

  # arm C — cross-account Bedrock (assume-role into a capacity account; §2c)
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1
      aws_role_name: arn:aws:iam::222222222222:role/VllmSrBedrockCapacity-prod
      aws_session_name: routeiq

  # arm D — self-hosted EKS inference (api_base = in-cluster Service; §4)
  - model_name: claude-sonnet
    litellm_params:
      model: hosted_vllm/anthropic/claude-equivalent-oss-70b
      api_base: http://aibrix-gateway.aibrix-system.svc.cluster.local:8000/v1
      api_key: fake-api-key            # vLLM-family needs none
```

All four hash to distinct `model_info.id`s, all four are independently cool-downable,
and RouteIQ's bandit picks among whichever survive cooldown this request.
**[SPECULATIVE]** Mixing a closed-weight cloud arm and an OSS self-hosted arm under
one alias is a quality/cost trade RouteIQ's reward model is *designed* to make, but the
"equivalent OSS model" mapping is a product decision, not a LiteLLM mechanism — we have
not validated reward parity across such mixed groups.

---

## 2. Capacity federation (VERIFIED — native LiteLLM)

All four patterns are **the same primitive with a different `litellm_param` varied**
(`discover-litellm-capacity.md` §8). No special "key rotation" or "multi-account"
feature exists in LiteLLM — it is `model_list` *shape*.

### 2a. Multiple API keys per provider = a capacity pool

**[VERIFIED]** Stack N deployments with the same `model_name` + same `model`, different
`api_key`, each carrying its own `rpm` (`discover-litellm-capacity.md` §3):

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params: {model: openai/gpt-4o, api_key: os.environ/OPENAI_KEY_A, rpm: 10000}
  - model_name: gpt-4o
    litellm_params: {model: openai/gpt-4o, api_key: os.environ/OPENAI_KEY_B, rpm: 10000}
  # pool = 20000 rpm
```

`_generate_model_id` hashes `api_key`, so the two keys are distinct arms; per-key `rpm`
semaphores cap each independently (`router.py:1581-1592`); a 429 on `KEY_A` cools that
id and traffic auto-shifts to `KEY_B` (`_filter_cooldown_deployments`), then `KEY_A`
rejoins on TTL expiry. **Per-key quotas sum into the pool, with automatic 429-driven
failover between keys.**

### 2b. Bedrock cross-region — different `aws_region_name`

**[VERIFIED]** Bedrock throughput is per-region; list one deployment per region under
one alias (`discover-litellm-capacity.md` §4). `aws_region_name` is a first-class auth
param (`reference/litellm/litellm/llms/bedrock/common_utils.py:152-164`); resolution
precedence is explicit `aws_region_name` > `AWS_REGION_NAME` > `AWS_REGION`
(`common_utils.py:202-207`), passed to `boto3.client("bedrock-runtime",
region_name=...)` on every auth branch (`common_utils.py:287`, `:300`, `:310`, `:321`).
Each region is a distinct, independently cool-downable arm; a regional throttle cools
just that region.

### 2c. Bedrock cross-account — `aws_role_name` assume-role

**[VERIFIED]** To reach Bedrock quota in *other AWS accounts*, each deployment carries
an `aws_role_name` (a role ARN in the target account); LiteLLM STS-assumes it before
calling Bedrock (`discover-litellm-capacity.md` §5):

- **Plain assume-role path**: `aws_role_name` + `aws_session_name` set →
  `sts_client.assume_role(RoleArn=aws_role_name, ...)`
  (`bedrock/common_utils.py:269-291`); the `bedrock-runtime` client is built with the
  temporary credentials. In `base_aws_llm.py` this is the
  `elif aws_role_name is not None: → _auth_with_aws_role()` branch (`:213`, `:243`).
- **Web-identity (IRSA) path**: `aws_web_identity_token` + `aws_role_name` +
  `aws_session_name` → `assume_role_with_web_identity(...)`
  (`common_utils.py:236-266`; `base_aws_llm.py:200`, `:205`).

#### Map to the VSR `bedrock_capacity_member_stack` pattern (port reference)

**[VERIFIED]** (`discover-capacity-eks-gap.md` (a)). vLLM-SR-on-AWS already solved
cross-account Bedrock onboarding as reproducible IaC: `BedrockCapacityMemberStack`
(`vllm-sr-on-aws/cdk/lib/bedrock_capacity_member_stack.py`) is a small standalone stack
deployed **into each capacity account** that creates exactly one stable-named role,
`VllmSrBedrockCapacity-<env>` (→ predictable ARN
`arn:aws:iam::<acct>:role/VllmSrBedrockCapacity-<env>`), with permission to invoke
Bedrock in *that* account and a trust statement for the home cluster. The home stack
(`vllm_sr_eks_stack.py`) drives a `capacity_account_ids: list` loop, computes the
predictable member ARNs, and emits each as a `CfnOutput`.

The mapping to LiteLLM is **direct** (`discover-capacity-eks-gap.md` (a) table):

| VSR construct | LiteLLM config | LiteLLM code path |
|---|---|---|
| `VllmSrBedrockCapacity-<env>` role ARN (per capacity account) | one `model_list` entry, `aws_role_name: <that ARN>` | `_auth_with_aws_role()` (plain `sts:AssumeRole`) — `base_aws_llm.py:213,243` |
| web-identity trust variant | `aws_role_name` + `aws_web_identity_token` | `_auth_with_web_identity_token()` — `base_aws_llm.py:200,205` |

**N accounts = N deployments behind one logical model name.** The VSR member role is
**directly consumable by LiteLLM with no Envoy AI Gateway in front of it**.

> **[VERIFIED] RouteIQ caveat — Pod Identity, not IRSA.** RouteIQ's cluster construct
> deliberately uses **EKS Pod Identity** (no `OpenIdConnectProvider`, no web-identity
> trust — `deploy/cdk/lib/eks_cluster_construct.py`,
> `90-roadmap-completion-2026-06-15.md` P0). The VSR *native/web-identity* member path
> assumes a cluster OIDC issuer RouteIQ does not provision. **The clean fit is the
> plain `sts:AssumeRole` path** = `aws_role_name`-alone in LiteLLM: the pod's Pod
> Identity role gets `sts:AssumeRole` on the member ARNs, the member role trusts that
> role's ARN. (Re-introducing an OIDC provider to use the web-identity path is the
> alternative, not the default.)

#### What RouteIQ ADDS on top of native federation

LiteLLM gives you the arms and keeps them healthy. RouteIQ adds the *intelligence and
control* the bare Router does not have:

- **The bandit routes by quality / cost / latency across the pool**, not round-robin.
  Where LiteLLM's `simple-shuffle` splits traffic by `rpm` weight
  (`simple_shuffle.py:42-58`), RouteIQ's Kumaraswamy-Thompson bandit + 18 ML strategies
  (KNN / MLP / SVM / ELO / MF / centroid / personalized) score *which arm is best for
  this prompt* and learn per-arm reward online — the same "shift away from a throttled
  account" behavior VSR gets from its `rl_driven` arms, but as a trained policy over the
  LiteLLM deployment list (`30-migration-roadmap.md` P3,
  `20-kumaraswamy-thompson-router.md`). **[VERIFIED]** the bandit + adapter framework
  shipped (P3, commit `506fb89`, `90-roadmap-completion-2026-06-15.md`).
- **Governance caps per tenant.** RouteIQ's native multi-tenant layer (ADR-0020:
  `governance.py`, `usage_policies.py`) enforces per-workspace / per-key budgets and
  rate limits *above* the capacity pool — capacity is shared substrate, governance
  decides who may draw from it and how much.
- **The gov-ban as a non-routable arm.** **[SPECULATIVE]** A model under a data-handling
  / compliance ban (e.g. the Fable-5-class data-retention gate) is expressed as a
  **guardrail / usage policy that removes it from the candidate set** before the bandit
  scores — i.e. governance can make an arm *exist in `model_list` but be un-routable for
  a given tenant*. This is a RouteIQ control-plane decision layered on top of LiteLLM's
  health-only filtering; the precise wiring (policy → `async_filter_deployments` hook,
  `custom_logger.py:248-256`, vs. RouteIQ pre-scoring filter) is a design choice we have
  not yet implemented.

---

## 3. AgentCore — see the sibling doc

**[VERIFIED]** Amazon Bedrock **AgentCore** integration (Gateway/Registry for tools via
MCP, Runtime for agents) is designed in detail in the sibling document
**`docs/architecture/agentcore-integration-and-arch-2026-06-15.md`** (Part 2). In the
arm-set frame: an AgentCore-fronted path is *another LiteLLM deployment / surface* that
RouteIQ routes to uniformly — AgentCore tools arrive over MCP (which upstream LiteLLM
serves per ADR-0017), and AgentCore Runtime agents are reachable as a backend like any
other. **Do not duplicate the AgentCore design here**; that doc is the authority,
including its VERIFIED/SPECULATIVE tags and the seeds it files.

---

## 4. Self-hosted EKS inference — Dynamo vs llm-d vs AIBrix vs vLLM

> The research report
> `research/notes/final_report_eks-inference-engines-litellm-6a7af4.md` is the
> **authority** for all engine facts in this section. The `engineVerdict` and key
> findings below are quoted from `synth-digest.json`.

### 4.1 OpenAI-compatibility is a non-discriminator (VERIFIED)

**[VERIFIED]** All four engines **and** raw `vllm serve` expose OpenAI
`/v1/chat/completions` + `/v1/completions`, and LiteLLM's `hosted_vllm` / `openai_like`
provider consumes every one **directly via `api_base` with no translation shim**
(`synth-digest.json` finding 1; research report §1). LiteLLM confirms the handshake:
"If an endpoint already exposes /v1/chat/completions in OpenAI format, LiteLLM can proxy
route to it directly via api_base without any translation shim." So each self-hosted
engine is registered exactly like §1 arm D: a `hosted_vllm/<model>` (or `openai_like`)
row whose `litellm_params.api_base` points at the engine's **in-cluster Service**
(`HostedVLLMChatConfig` subclasses `OpenAIGPTConfig` and treats the api_key as optional
`"fake-api-key"` — `reference/litellm/litellm/llms/hosted_vllm/chat/transformation.py:48-55`).
**The engine choice is operational / platform / maturity, NOT API-compatibility.**

### 4.2 The KEY architectural insight — two-layer routing that COMPOSES, not competes

**[VERIFIED]** (`synth-digest.json` findings 2, 3; research report §5). There are two
structurally distinct routing decisions:

- **Layer 1 — model SELECTION ("which model / api_base / InferencePool").** LiteLLM's
  router, **RouteIQ's** Kumaraswamy-Thompson + KNN / MLP / SVM / ELO / centroid /
  personalized strategies, and the Gateway API Inference Extension's *Body-Based Router*
  live here. Reads the requested model name / RouteIQ quality+cost features.
- **Layer 2 — replica SCHEDULING ("which replica").** Takes a chosen model and picks
  the optimal *replica* by KV-cache locality, KV utilization, and queue depth. This is
  the job of **Dynamo's Smart Router, llm-d's Endpoint Picker (EPP), AIBrix's Router,
  and the GAIE Endpoint Picker**. Reads runtime KV-cache/queue state that "the gateway
  is unaware of" from above.

**The official Gateway API Inference Extension README settles the layering**: the
inference gateway integrates self-hosted models "in a higher level AI Gateway like
**LiteLLM**, Solo AI Gateway, or Apigee." The project that *builds* the KV-aware
endpoint picker names LiteLLM as the layer **above** it. The layers cannot collapse —
they consume different information. **KV-cache-aware / disaggregated routing is a LAYER
BELOW RouteIQ's model-selection routing (intra-model replica scheduling vs. inter-model
selection). They COMPOSE.**

> **The ONLY duplication risk is a configuration error** (`synth-digest.json` finding 3;
> report §5). Because LiteLLM is itself a load-balancer, a team can *naturally* register
> each vLLM **pod** as a separate LiteLLM deployment — at which point RouteIQ makes
> **cache-blind replica decisions that fight** the engine's KV-aware router and the two
> layers collapse. **Correct design (a hard rule): RouteIQ targets exactly ONE
> `api_base` per model — the engine's gateway/router Service — and delegates per-replica
> scheduling downward.** Never register individual replicas as separate LiteLLM
> deployments.

### 4.3 The ranked `engineVerdict` (from the research report)

**[VERIFIED]** adoption order (`synth-digest.json` `engineVerdict`; report §7). Each
engine is a single LiteLLM `hosted_vllm`/`openai_like` deployment, `api_base` = the
engine's in-cluster gateway Service:

| Order | Engine | Why / what it adds over raw vLLM | EKS Auto Mode fit |
|---|---|---|---|
| **0** | **raw `vllm serve`** | The serving floor — engine + OpenAI server, but single-replica: no cross-replica routing, KV-aware scheduling, or autoscaling. | Clean (device-plugin-only). |
| **1a** | **vLLM Production Stack** | Smallest step up: KV-cache-aware routing (routes to highest KV-cache-hit instance). Helm chart / CRD operator; does **not** need GAIE. | Clean. |
| **1b** | **AIBrix** | Most production-proven today (ByteDance 6+ mo by Feb 2025, v0.5.0, arxiv 2504.03648); LoRA-aware autoscaling + distributed KV cache (L1 DRAM / L2 remote); own Envoy gateway-plugins (no GAIE). | Clean (device plugin auto-installed on GPU nodes; no GPU operator). |
| **2** | **llm-d** | When you standardize on the Gateway API Inference Extension or need precise prefix-cache scheduling + cross-vendor accelerators. **STRICTLY requires** GAIE v1.5.0 + Gateway API v1.5.1 CRDs + a Gateway provider (Istio/agentgateway/GKE); its scheduler *is* the EPP. CNCF Sandbox v0.7 (May 2026). | Single-node path drops on cleanly. |
| **3** | **NVIDIA Dynamo** | Only at large multi-node scale (~8+ GPU nodes; reasoning / agentic / long-context) where disaggregated prefill/decode's 7–30x gains justify the cost. **NOT built on vLLM** — backend-agnostic (vLLM/SGLang/TensorRT-LLM workers), three-tier Frontend/Router/Worker, NIXL KV transfer. 1.0 GA Mar 16 2026. | **Friction case** — assumes the NVIDIA GPU Operator; canonical AWS ref uses self-managed Karpenter nodes + EFA. The topology-aware EFA-DRA driver is "not supported with Karpenter or EKS Auto Mode" → use managed/self-managed node groups for full disaggregation. |

Two corollaries from the report worth pinning:

- **Disaggregation is scale-gated, not a default** (`synth-digest.json` finding 9): it
  pays off at ~8+ GPU nodes with high-concurrency/long-context/reasoning/agentic
  workloads and effectively requires NVLink + InfiniBand/400GbE (or EFA on AWS). You can
  adopt broadly-useful **KV-aware routing (Production Stack / AIBrix) WITHOUT adopting
  disaggregation.**
- **Backend-agnosticism is unique to Dynamo**; Production Stack / llm-d / AIBrix are
  vLLM-native control planes. All four COMPOSE with vLLM.

### 4.4 The EKS GPU gap (VERIFIED — RouteIQ-specific blocker)

**[VERIFIED]** (`discover-capacity-eks-gap.md` (b)). RouteIQ's cluster
(`deploy/cdk/lib/eks_cluster_construct.py`) is **EKS Auto Mode** with only the two
built-in AWS-managed node pools:

```python
_AUTO_MODE_NODE_POOLS = ["general-purpose", "system"]   # CPU families only
```

These provision **general-purpose CPU instances only** and Auto Mode does **not**
install the NVIDIA device plugin for them. Concrete consequence: a pod requesting
`nvidia.com/gpu: 1` sits **`Pending` forever** — no node advertises the resource and the
built-in pools won't scale a GPU node. **So today RouteIQ is a Bedrock/managed-endpoint
gateway only at the substrate level — it cannot host its own GPU model server.** (This
matches VSR's own "Day-1 = CPU-only, Bedrock backend; GPU pool is an additive later
change" posture; RouteIQ inherited it but has *not* authored the GPU additive change.)

**What's needed (additive — neither changes the CPU path)**:

- **Option 1 (preferred, Auto-Mode-native):** a custom Auto Mode **GPU `NodePool` +
  `NodeClass`** (K8s CRs applied out-of-band) selecting GPU families (`g6`/`g6e`/`p5`),
  `NodeClass.role:` = the existing Auto Mode node role (the construct **already emits
  `NodeRoleName`** as a `CfnOutput` exactly for this), with a `nvidia.com/gpu` taint.
  EKS Auto Mode includes the NVIDIA device plugin via Bottlerocket Accelerated AMIs
  ("runs automatically, isn't visible as a daemon set") — verify advertisement for the
  chosen family; install the device-plugin DaemonSet only if absent. Preserves the
  managed-Karpenter benefit that motivated Auto Mode.
- **Option 2:** a classic **managed node group** of GPU instances + EKS GPU-optimized
  AMI + self-installed NVIDIA device plugin. Heavier; partly defeats Auto Mode. Required
  for Dynamo's full disaggregated/EFA path.

Either way the missing pieces are the same two: **(1)** a node source that provisions
GPU instances, and **(2)** the NVIDIA device plugin so the GPU is advertised as
`nvidia.com/gpu`. Until both exist, any vLLM/Dynamo/llm-d/AIBrix serving pod stays
`Pending` and the self-hosted story is non-functional.

---

## 5. Phased plan — a NEW roadmap beyond P0–P4

> **This is a new roadmap.** The P0–P4 migration roadmap
> (`30-migration-roadmap.md`, `90-roadmap-completion-2026-06-15.md`) is **complete and
> merged**. The phases below (call them **C0–C3** for "capacity") are *additive*,
> ordered by **effort/value**: capacity federation is mostly config (lowest effort,
> immediate value); self-hosted GPU needs new CDK + an engine (highest effort).

| Phase | Scope | Effort | Value | Nature |
|---|---|---|---|---|
| **C0 — API-key + cross-region capacity pools** | §2a + §2b: stack N `model_list` rows (extra keys, extra `aws_region_name`) under existing aliases. Pure config + governance per-tenant caps. | **Lowest** (config only) | High — immediate quota headroom + 429 failover, no infra | Config |
| **C1 — Cross-account Bedrock arms** | §2c: per-capacity-account `aws_role_name` deployments; port the VSR `BedrockCapacityMemberStack` *plain-`sts:AssumeRole`* path (Pod-Identity-clean); grant the gateway Pod Identity role `sts:AssumeRole` on member ARNs. | **Low–Med** (1 new CDK construct + per-account `cdk deploy` + config) | High — multiplies Bedrock quota across accounts | CDK port + config |
| **C2 — Gov-ban / non-routable-arm control** | §2c-ADD: express the Fable-5-class data-retention ban as a usage/guardrail policy that removes an arm from the candidate set before the bandit scores. | **Low** (control-plane code) | Med-High — compliance-gated routing | App (RouteIQ-native) |
| **C3 — Self-hosted EKS inference** | §4: close the **GPU NodePool gap** (new CDK construct / out-of-band CRs, Option 1) **+** deploy one engine (start Production Stack *or* AIBrix per the verdict) **+** register it as ONE `api_base`-per-model LiteLLM deployment. | **Highest** (GPU NodePool + engine + new CDK construct + Helm/CRDs) | High but scale-gated — only pays off once OSS-hosted capacity is wanted | CDK new-build + platform + config |

**Ordering rationale.** C0 is *almost free* and delivers the headline "expanded
capacity" with zero infra change — do it first. C1 reuses a *tested* VSR construct and a
*verified* LiteLLM auth path; the only RouteIQ-specific work is choosing the
plain-AssumeRole (Pod-Identity-clean) trust path. C2 is small RouteIQ control-plane code
that makes the governance story honest. C3 is gated on closing the EKS GPU gap and
standing up an engine — highest effort, and per the report its *value is scale-gated*
(KV-aware routing pays off broadly; disaggregation only at 8+ GPU nodes), so it is last.

**The composition discipline carries into C3 as an acceptance criterion**: when an
engine lands, RouteIQ MUST register **one `api_base` per model** (the engine gateway),
never per-replica — otherwise Layer 1 and Layer 2 collide (§4.2).

---

## 6. One-paragraph synthesis

LiteLLM federates capacity by letting **N deployments share one `model_name`**; each is
a `model_list` row whose `litellm_params` get a stable unique `model_info.id` via sha256
(`router.py:5592-5621`), making "same model, different key/region/account/endpoint" into
separable, independently cool-downable arms. The four federation patterns are the same
primitive with a different field varied — `api_key` (key-pool, §2a), `aws_region_name`
(cross-region, §2b), `aws_role_name` (cross-account assume-role mapping to the VSR
`BedrockCapacityMemberStack`, §2c), and `api_base` under `hosted_vllm`/`openai_like`
(self-hosted, §4). Per request LiteLLM computes
`healthy_deployments = model_group − cooldown − filters`; RouteIQ's
`CustomRoutingStrategyBase` reads that list "for freshness" and lets the
Kumaraswamy-Thompson bandit pick the best surviving arm — so `healthy_deployments` is
the bandit's dynamic arm set. Self-hosted engines' KV-cache-aware / disaggregated
routing is **Layer 2 (replica scheduling)** sitting **below** RouteIQ's **Layer 1
(model selection)** — they compose iff RouteIQ targets one `api_base` per model. The new
C0–C3 roadmap adds arms in effort/value order: config-only key+region pools first, the
cross-account CDK port next, the gov-ban control, and self-hosted EKS inference last
(gated on closing the verified GPU NodePool gap).
