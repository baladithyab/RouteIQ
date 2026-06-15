# RouteIQ Architecture (post-AWS-rearch) + AgentCore Gateway/Registry Integration Design

> **As of 2026-06-15.** Two parts, one document.
>
> - **Part 1** updates the April-2026 `four-way-comparison.md` baseline with the
>   AWS substrate we just shipped (P0–P4, merged to `main`), and re-contrasts
>   RouteIQ-current against vllm-sr-on-aws-current.
> - **Part 2** is the design for connecting **Amazon Bedrock AgentCore Gateway +
>   Agent Registry** to LiteLLM/RouteIQ — verdict, three ranked directions, the
>   concrete smallest-shippable step, and the seeds to file.
>
> **Honesty contract.** All AgentCore facts are sourced from the research report
> (`research/notes/final_report_agentcore-gateway-litellm-integration-18d379.md`),
> which is grounded in official AWS AgentCore docs + LiteLLM MCP docs + RouteIQ
> source. Claims are tagged **[VERIFIED]** (documented in an AWS/LiteLLM doc or
> present on disk) or **[SPECULATIVE]** (reasoned synthesis of two documented
> halves, or inferred from API semantics). The research report is the authority
> for AgentCore facts.

---

# PART 1 — RouteIQ-current vs vllm-sr-on-aws-current (post-AWS-rearch)

## 1.1 What changed since the April baseline

The April-2026 `four-way-comparison.md` is still a correct description of the
RouteIQ **application layer** (the framework: own FastAPI + LiteLLM-as-plugin,
the strategy toolkit, native governance). It simply **predates the AWS
substrate**. The P0–P4 roadmap (28 commits ahead of `origin/main`, merged but
never pushed, operator-gated) added a 3-CDK-stack AWS-native substrate ported
from vllm-sr-on-aws. RouteIQ is now **two layers of one platform**:

```
┌──── APPLICATION LAYER (RouteIQ framework, src/litellm_llmrouter/) ───────────────┐
│  own FastAPI app (ADR-0012) · LiteLLM 1.82.3 mounted at /v1 as a plugin           │
│  ~18 routing strategies + Kumaraswamy-Thompson bandit (P3)                        │
│  strategy-agnostic adapter/MLOps loop (P3) · three-tier governance + OIDC + RBAC  │
│  14 plugins · eval pipeline (LLM-as-judge) · 6-page React admin UI                │
└───────────────────────────────────────┬───────────────────────────────────────────┘
                                         │ runs as a stateless pod on
                                         ▼
┌──── AWS SUBSTRATE (3 CDK stacks, deploy/cdk/lib/, ported from vllm-sr-on-aws) ────┐
│  P0 RouteIqStack         — VPC/SGs/endpoints · EKS Auto Mode · EKS Pod Identity   │
│  P1 RouteIqStateStack    — Aurora PG Serverless v2 · ElastiCache Serverless · KMS │
│  P2 RouteIqObservability — AppConfig · AMP+AMG+CW filters/alarms/SNS · Firehose   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

The substrate is the realization of the gap-doc synthesis path #2: **RouteIQ
adopts vllm-sr-on-aws's deployment substrate, keeps its superior framework
capabilities.** Everything *inside* the pod is RouteIQ's existing framework
code; everything *around* it is new CDK-provisioned AWS-native infrastructure.

A symmetric fact closes the loop: vllm-sr-on-aws's own latest strategic
decisions (`multitenant-front-layer-decision.md`, 2026-06-10; the
`responses-only-models-litellm-bridge.md` "SOLVED 2026-06-14") are pulling
**LiteLLM** (RouteIQ's exact substrate) onto VSR's AWS substrate to solve its
Responses-API gap and to gain a multi-tenant front layer. The two projects are
converging on the same shape from opposite ends.

## 1.2 The four contrast axes

### (a) Dataplane — the request hot path

| | RouteIQ-current | vllm-sr-on-aws-current |
|---|---|---|
| **Edge** | **Own Python FastAPI app** (ADR-0012, `gateway/app.py::create_gateway_app()`), LiteLLM proxy mounted at `/v1/`. Routing = a `CustomRoutingStrategyBase` plugin into the LiteLLM Router, in-process; `install_plugin_routing_strategy(app)` called in `_routeiq_lifespan` after `litellm.proxy.proxy_server.initialize(...)`. | **Envoy ExtProc**, line-rate C++ data plane. The routing decision is a gRPC `ExternalProcessor` call to the upstream Go router on `:50051`; the router rewrites `body.model` + `:path` in flight. Fail-CLOSED (`failure_mode_allow=false`), `suppress_envoy_headers`. |
| **Language / mechanism** | Python throughout; soft plugin boundary (any Python dev writes a strategy or plugin). | Go (router core) + Rust/Candle + ONNX (in-process classifiers) + C++ (Envoy). Hard boundary — new signals/selectors need upstream Go contributions or sidecars. Fork count of routing logic = **0** (tracks upstream). |
| **Routing intelligence** | A *toolkit*: ~9 genuinely-native strategies (centroid zero-config ~2ms, personalized EMA + online-feedback endpoint, native Router-R1 without vLLM, cost-aware Pareto, `context_optimizer` 30–70% token reduction, 5 native ML routers KNN/SVM/MLP/MF/Elo) + ~10 delegated upstream LLMRouter routers + an A/B + staged-rollout registry (`strategy_registry.py`) + an LLM-as-judge eval/feedback loop + the **new Kumaraswamy-Thompson bandit** (`kumaraswamy_thompson.py`, P3) + the **strategy-agnostic MLOps loop** (`adapters/{contract,loader,mlops}.py`, P3). | **One** deployed engine: upstream VSR's `rl_driven` **Thompson-sampling bandit**, keyed on `(decision_name, model)`. Online-learning via Aurora-replay → feedback POST → Beta-posterior update (idle on EKS until replay wired). The upstream **16-signal taxonomy** (keyword/embedding/domain-MMLU/jailbreak/PII/language/modality/complexity + BERT/LoRA classifiers + Boolean-DSL decision engine) is a richer first-class *signal vocabulary* but is largely **not exercised** in the live single-arm Bedrock deployment. |
| **Provider breadth** | **~112 inherited from `litellm==1.82.3`** (OpenAI, Anthropic, Bedrock SigV4, Vertex, Azure AD, Cohere, Mistral, Groq, Together, Fireworks…), all OpenAI-schema-normalized; full non-chat surface (`/embeddings`, `/rerank`, `/audio`, `/images`, `/batches`, `/files`, `/responses`); **chat/messages → Responses API translation** (`LiteLLMCompletionResponsesConfig` + Anthropic-messages↔OpenAI adapters) — the headline capability VSR's passthrough-only EAIG cannot do. | **~18 backends, OpenAI-compatible only** at the engine level; the live deployment routes to a *single* Bedrock "Mantle" egress (`bedrock-mantle.us-west-2.api.aws:443`, 33 arms → 1 host) + multi-account capacity arms. Chat/messages/responses only; **cannot reshape chat/messages → Responses API** (EAIG's `NewResponsesOpenAIToOpenAITranslator` is passthrough-only; Responses-only models like gpt-5.5 404/quarantine). |

**The Kumaraswamy-Thompson bandit (P3) is the routing news.** RouteIQ now has a
Thompson-sampling bandit that samples each `(bucket, model)` arm via the
closed-form Kumaraswamy inverse-CDF (`kumaraswamy_quantile`, log-space
stabilized) over a conjugate Beta posterior. It is **one pluggable strategy
among many** behind `strategy_registry.py` — not the whole router — and it is
fed by a **strategy-agnostic** `MLOpsCoordinator` (`adapters/mlops.py`) that
scans the registry for any strategy declaring `learns=True` and fans `(model,
quality)` aggregates out to *continuous* learners via `on_aggregate_feedback()`,
wired into the eval loop. This is the structural difference from VSR's deployed
bandit, which is a single hardwired `rl_driven` arm.

**Philosophy split (sharpest, unchanged).** RouteIQ extends in **Python at the
application layer**; VSR extends in **Go/Rust in-tree at the proxy layer**. VSR
wins the latency budget (Envoy line-rate + sub-10ms in-process classification);
RouteIQ wins dev velocity (write a strategy, no fork). The bandit ↔ bandit
comparison is the closest the two get to head-to-head — VSR's is more
production-hardened (live against Bedrock), RouteIQ's is more architecturally
flexible (one strategy in a pluggable MLOps framework).

### (b) AWS substrate — the new layer (this is what changed)

Both projects are now **EKS-Auto-Mode + 3-CDK-stack + cdk-nag** shops. The April
baseline's framing ("RouteIQ has no cloud IaC; VSR is ECS-default/EKS-alt") is
**doubly stale**: RouteIQ now has a full CDK substrate, and VSR **tore ECS down**
(its `ADR-0018-eks-auto-mode-supersedes-ecs.md` supersedes ADR-0001 — EKS Auto
Mode is its *only* live serving substrate; the `compute_construct.py` ECS task
remains in-tree but is historical/superseded).

| Substrate dimension | RouteIQ-current (P0–P4 shipped) | vllm-sr-on-aws-current |
|---|---|---|
| **Compute** | **EKS Auto Mode**, L1 `CfnCluster` (Compute/BlockStorage/ELB blocks) in `eks_cluster_construct.py`; `routeiq-gateway` Helm chart (anti-affinity, topology spread, PDB, HPA); stateless pods (`emptyDir` only — the hard-won VSR lesson: RWO-EBS + RollingUpdate = Multi-Attach deadlock). | **EKS Auto Mode only** (ECS torn down). Single replica (`replicaCount=1`) because Thompson state is replica-local on an EBS gp3 PVC (`/srv/policy/elo_ratings.json`, RWO/Retain — chosen over EFS/S3 because the router saves via atomic `os.Rename`, ADR-0019). |
| **Identity** | **EKS Pod Identity, NOT IRSA** (the central P0 divergence from VSR). `CfnPodIdentityAssociation` with a *static* `pods.eks.amazonaws.com` service-principal trust (`sts:AssumeRole` + `sts:TagSession`) — **no `OpenIdConnectProvider`, no `WebIdentityPrincipal`, no `CfnJson` trust map**. Bound to one `(namespace=routeiq, serviceAccount=routeiq-gateway)`. Least-privilege: `BedrockInvoke`, `SecretsRead`, optional `ConfigS3Read`, scoped `Logs`. | **IRSA via `CfnJson` trust-key** (OIDC provider), per-workload (router→bedrock-mantle, EAIG gateway→native Bedrock, bearer-minter). This is the seam EAIG's `BackendSecurityPolicy` authenticates against. |
| **State** | **Aurora PG Serverless v2** (`replay_store_construct.py`, `PRIVATE_ISOLATED`, IAM DB auth, KMS, 30-day master-secret rotation, schema-bootstrap Lambda; engine `VER_16_13` pinned) — governance/keys/spend/**bandit posteriors** persist here; **ElastiCache Serverless (Valkey)** (`cache_construct.py`, TLS, IAM-auth user group) — rate-limit/budget windows, session affinity, circuit-breaker state, hot EMA, L2 semantic cache. DB-optional: `governance_store.py` (P4) no-ops when `DATABASE_URL` unset. | Aurora Serverless v2 (replay) **provisioned but not wired on the EKS path** (no reader endpoint wired — open NEEDS-DEPLOY); ElastiCache Serverless **provisioned, not wired** (upstream client IAM+TLS gap); EBS PVC is the live bandit state. |
| **Config / Observability** | **AppConfig** (`config_state_construct.py`, hosted config + validator Lambda — the routing-strategy hot-reload contract, ADR-0026; pod polls via `StartConfigurationSession`/`GetLatestConfiguration`) + **AMP** remote-write + flag-gated **AMG** + **CW MetricFilters/alarms/SNS** including a per-model filter keyed on the OTel `gen_ai.response.model` field + **Firehose → S3 Parquet → Glue → Athena** data lake (the MLOps telemetry sink). | AppConfig live (delivers `router.yaml`; ConfigMap on EKS, no agent sidecar). **EKS observability is deliberately CloudWatch-only** (ADR-0021) — Container Insights/Fluent Bit → CW Logs → 3 metric filters (incl. `routing_latency_ms_by_model` dimensioned per `selected_model`) → baseline-free alarms. AMP/AMG/ADOT are ECS-era, **not on the EKS path**. Same Firehose→S3→Glue→Athena lake (flag-gated). |
| **Bedrock ops** | Inherits **generic LiteLLM SigV4** — the VPC carries a `BEDROCK_RUNTIME` interface endpoint; zero mantle/bearer-minter/multi-account code. | **Live mantle bearer-minting** (init `token-mint` + `token-refresher` sidecar, 6h re-mint), **multi-account Bedrock capacity** (`bedrock_capacity_member_stack.py`, 4 capacity accounts, cross-account `bedrock-mantle:CreateInference`/`CallWithBearerToken`), multi-region. |
| **Stack hygiene** | 3 separate stacks = 3 blast radii / ~30-min rollback units; cross-stack wiring **cred-free by reference** (`Export`/`Fn::ImportValue` at synth, never `from_lookup`); `AwsSolutionsChecks` (cdk-nag) app-wide. Gate on `main`: `deploy/cdk/tests/` = **228 passed**, `tests/unit/` = **3620 passed**, `helm lint` = 0. | 3 CDK stacks, **~446 CDK tests**, cdk-nag enforced; CodeBuild/CodePipeline provisioned in-CDK (image bake + model-promotion + deploy). |

**Substrate net.** The two are now substrate-siblings. The remaining substrate
deltas are (1) **Pod Identity vs IRSA** (RouteIQ's deliberate simplification —
no OIDC provider to manage), (2) **state actually wired** (RouteIQ wires Aurora
via `governance_store.py` + ElastiCache; VSR has both provisioned-but-unwired on
EKS), and (3) **Bedrock ops depth** (VSR has live mantle/minter/multi-account;
RouteIQ rides generic LiteLLM SigV4 through a VPC endpoint). RouteIQ's observability
is AMP+AMG+CW (richer); VSR's EKS path is CW-only by choice.

### (c) Control plane / governance — RouteIQ's largest forward-advantage axis

| Capability | RouteIQ-current | vllm-sr-on-aws-current |
|---|---|---|
| Multi-tenant primitives | **Shipped, native:** three-tier Org→Workspace→Key (`governance.py`), now **optionally Aurora-backed** (`governance_store.py`, P4; file-backed default). | **None live.** Single-tenant, single-region, single-account; CIDR-locked NLB to one operator IP *is* the security boundary. |
| Virtual keys / teams / budgets | Shipped (inherited LiteLLM + native `usage_policies.py` dynamic rate limits + budgets; the P1-convergence commit wired spend write-scope == read-scope so workspace/key budgets actually enforce, plus a fail-closed governance mode). | None deployed. |
| SSO / OIDC | Shipped native (`oidc.py`: Keycloak/Auth0/Okta/Azure AD/Google, live JWKS). | JWT dashboard auth only. |
| Guardrail policy engine | Shipped — 14 `_check_` handlers (`guardrail_policies.py`) + plugins (`bedrock_guardrails`, `pii_guard`, `llamaguard`, `prompt_injection_guard`, `content_filter`). | Construct exists (`native_guardrail_construct.py`, gated, native-arm-only). Rich in-engine VSR classifiers exist upstream but are not deployed as a tenant guardrail plane. |
| RBAC / policy engine | Shipped (`rbac.py`, `policy_engine.py` OPA-style). | None. |
| Admin UI | Shipped **6-page React** (Dashboard/RoutingConfig/Governance/Guardrails/Prompts/Observability). | EAIG operator dashboard `:8700` (ClusterIP, port-forward); no tenant UI. |

**Net:** RouteIQ ships a full native multi-tenant control plane; VSR-on-AWS
ships **none** (it's a routing engine + AWS substrate, not a governance
platform). This is unchanged from April and is precisely why VSR's own
2026-06-10 decision reaches for LiteLLM as its multi-tenant front layer.

### (d) When to pick which

- **Pick RouteIQ** when you need a **multi-provider gateway with a real control
  plane** — multi-tenant keys/budgets/OIDC/RBAC/guardrails/UI, ~112 providers,
  chat↔Responses↔Anthropic translation, a *toolkit* of routing strategies you
  can A/B and a pluggable online-learning MLOps loop — and you want to extend in
  Python. The new AWS substrate means it now also ships as a production EKS-Auto-Mode
  deployment, not just a Helm chart.
- **Pick vllm-sr-on-aws** when you need a **single hardened line-rate routing
  endpoint** in front of Bedrock with the lowest possible per-request overhead
  (Envoy ExtProc, sub-10ms classification), live mantle bearer-minting +
  multi-account Bedrock capacity, and you are comfortable with single-tenant /
  one-engine and extending in Go/Rust.
- **Use both** when the synthesis is the point: RouteIQ as the multi-tenant
  governed front + provider breadth + translation, VSR as a downstream
  line-rate Bedrock routing arm. This is the direction both repos are
  independently converging on.

---

# PART 2 — Connecting AgentCore Gateway + Registry to LiteLLM/RouteIQ

## 2.1 Integration verdict (lead)

**Yes — and it is well-supported in all three directions, because both systems
were built to the same MCP standard (streamable-HTTP at the shared 2025-11-25
protocol version). [VERIFIED]**

The decisive fact: **Amazon Bedrock AgentCore Gateway *is itself* an MCP
server.** `create_mcp_gateway(...)` returns a `gatewayUrl` that speaks MCP
streamable-HTTP — POST `/mcp` JSON-RPC, `Mcp-Session-Id` session routing — and
exposes the full MCP op set (`tools/list`, `tools/call`, prompts, resources,
elicitation, sampling) plus a built-in semantic-discovery tool
`x_amz_bedrock_agentcore_search`. Any MCP client can connect. **[VERIFIED — AWS:
Introducing AgentCore Gateway; Use an AgentCore gateway]**

**The cleanest seam is Direction A: consume the AgentCore Gateway as an MCP
server from LiteLLM's MCP client.** LiteLLM ships a *named, purpose-built mode*
for exactly this: its MCP-client docs say verbatim *"For MCP servers hosted on
AWS Bedrock AgentCore, select AWS SigV4 … service name (defaults to
`bedrock-agentcore`) … falls back to the boto3 credential chain."* Add one
`mcp_servers` entry (`url=gatewayUrl`, `transport: http`, auth matched to the
gateway authorizer) and the gateway's tools become available to **any model
LiteLLM routes to**. **[VERIFIED — LiteLLM MCP Overview docs]**

**The one rule that decides success is auth coherence:** LiteLLM's `auth_type`
must MATCH the gateway's create-time inbound authorizer (SigV4 if gateway=IAM;
OAuth 2LO / static bearer access-token if gateway=`CUSTOM_JWT`). A mismatch
returns the gateway's RFC 6750 `401 WWW-Authenticate` challenge. Plus one
config nuance: **pin `transport: http`** — LiteLLM defaults to `sse`, and
standalone SSE is deprecated in MCP 2025-03-26+ while AgentCore is
streamable-HTTP-only. **[VERIFIED]**

**Two distinct AgentCore planes — do not conflate them. [VERIFIED]**
- **Gateway** = the runtime tool-call plane (an MCP server at `gatewayUrl`).
  There is **NO separate "tool registry" API** — tools are registered
  *implicitly* via `CreateGatewayTarget` and discovered at runtime via MCP
  `tools/list`. The "registry" *is* the gateway's MCP listing surface.
- **AWS Agent Registry** (Preview, announced 2026-04-09) = a *distinct*
  discovery/governance catalog with its own `CreateRegistryRecord` API
  (`descriptorType ∈ {MCP, A2A, CUSTOM, AGENT_SKILLS}`, `descriptors` payload,
  `synchronizationConfiguration.fromUrl`), registries + records, approval
  workflows, and **its own MCP endpoint** queryable by any MCP client. The
  Registry-only deep-dive (R1 publish / R2 consume, ranked, the
  `CreateRegistryRecord` shape, the single-`search_registry_records`-tool +
  LiteLLM-no-SigV4 catch, the smallest-shippable, and the record-API gotchas)
  lives in the companion doc
  [`agentcore-registry-integration-2026-06-15.md`](./agentcore-registry-integration-2026-06-15.md).

**Honest caveat.** This is an additive arrangement, not a free one: AgentCore
Gateway is a **second managed hop** with consumption cost, latency, and AWS
dependency *on top of* LiteLLM's own MCP gateway. It earns its place when
consolidating many or AWS-fronted tools behind one auth surface with semantic
search + governance, and adds little when LiteLLM could reach a single remote
MCP server directly. **[VERIFIED — research report §11]**

## 2.2 The existing RouteIQ MCP/AgentCore surface (what we already have)

Four current seams touch MCP/tool-use/AgentCore (from `discover-mcp-surface.md`):

| Seam | Owner | Shape | Default |
|---|---|---|---|
| **(a)** RouteIQ REST MCP gateway | RouteIQ-native | Plain REST/JSON (`/llmrouter/mcp/*`), **NOT** MCP JSON-RPC — custom `{"tool_name","arguments"}` POST body | off (`MCP_GATEWAY_ENABLED=false`) |
| **(b)** Upstream LiteLLM MCP server | upstream LiteLLM (ADR-0017) | **Real MCP** JSON-RPC over Streamable-HTTP/SSE; `global_mcp_server_manager`, DB/config-backed, RBAC, hooks | config/DB-driven |
| **(c)** LiteLLM tool-call normalization | upstream LiteLLM | OpenAI `tools`/`tool_choice` → per-provider transforms | always-on for `/v1/chat/completions` |
| **(d)** Inbound AgentCore plugin | RouteIQ-native (reference) | Registers AgentCore *agents* as servers INTO seam (a) | off (`ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_ENABLED=false`) |

**The existing inbound plugin (d) — `gateway/plugins/bedrock_agentcore_mcp.py` —
is a stub, and it is INBOUND.** It registers AgentCore *agents* as MCP servers
INTO RouteIQ's own REST gateway (a). Critically, its `_register_agents` builds a
**fabricated legacy URL** (`bedrock_agentcore_mcp.py:240-241`):

```
https://bedrock-agent-runtime.{region}.amazonaws.com/agents/{agent_id}/agentAliases/TSTALIASID/sessions/test/text
```

— literally `TSTALIASID`, session `test` — which is the **old
`bedrock-agent-runtime` InvokeAgent REST shape, NOT the new AgentCore
Gateway/Runtime API.** It then calls `register_server(..., transport="streamable_http")`
(lines 253-257) with an **empty tools list**. There is **no boto3, no
`bedrock-agentcore` SDK, no SigV4, no live `tools/list` discovery** — the
`health_check` comment says outright *"We don't actually call AgentCore here."*
So today (d) is a demonstration of `PluginContext.mcp`, not a working connector.

**How the new directions differ from (d):** (d) points *inward* at a single
legacy agent-alias and dead-ends in seam (a)'s non-MCP custom REST protocol
(which cannot sign AWS requests — its adapter supports only
`none/api_key/bearer_token`). The directions below instead treat the AgentCore
**Gateway** as a real MCP server and wire it into **seam (b)** (the real MCP
manager with real transports, RBAC, hooks, and OAuth) — or expose RouteIQ
through AgentCore. The smallest of them (2.4) is a targeted **repoint** of (d).

## 2.3 The three integration directions, ranked

### Direction A (RANK 1) — AgentCore Gateway as an MCP server LiteLLM/RouteIQ consumes

- **Seam:** LiteLLM's MCP *client* / `global_mcp_server_manager` (seam **(b)**),
  configured via the top-level `mcp_servers` config key (loaded by
  `proxy_server.py::_init_non_llm_configs` → `load_servers_from_config`) or
  `POST /v1/mcp/server`. **AgentCore API:** `CreateGateway(protocolType='MCP',
  authorizerType=...)` returns `gatewayUrl`; tools discovered via MCP
  `tools/list` off that URL; the built-in `x_amz_bedrock_agentcore_search`
  helps with tool overload.
- **VERIFIED:** the gateway IS an MCP server over streamable-HTTP at 2025-11-25;
  LiteLLM ships the named AWS SigV4 / `bedrock-agentcore` MCP mode + OAuth +
  static-bearer; version overlap is exact; auth-coherence rule + `transport: http`
  pin. **[research report §§1, 6, 7, 10]**
- **SPECULATIVE:** the *exact* end-to-end LiteLLM↔AgentCore-Gateway pairing is a
  reasoned synthesis of two documented halves (AWS documents the gateway as an
  MCP server; LiteLLM documents itself as an MCP client) — no single doc shows
  the full join. Treat the first live connection as the verification step.
- **Auth model:** match the gateway's *create-time* inbound authorizer.
  IAM gateway → LiteLLM SigV4 (`service=bedrock-agentcore`, boto3 cred chain →
  RouteIQ's EKS Pod Identity role on AWS). `CUSTOM_JWT` gateway → OAuth 2LO
  client-credentials or static `Authorization: Bearer <2LO access token>` from
  the gateway's OIDC provider. Mismatch → RFC 6750 `401`.
- **Effort:** **Small.** A config entry + auth wiring; no new RouteIQ code if
  using seam (b) directly. The SigV4 gap noted in `discover-mcp-surface.md` (§3
  "Auth gap") applies to seam (a)'s adapter, **not** to LiteLLM's MCP client,
  which has native SigV4 — so consuming through (b) sidesteps it.
- **NET-NEW vs existing plugin:** **Mostly reuses** — it does not need the (d)
  plugin at all if wired through (b). If you *want* it to flow through the (d)
  plugin's `register_server` path, that is the 2.4 repoint (small).
- **Gotcha to respect:** if any gateway target is a Lambda you author, the
  **Gateway tool-name contract** applies — the invoked tool name arrives in
  `context.client_context.custom['bedrockAgentCoreToolName']` (NOT
  `event['tool_name']`, which is empty) and args are top-level event keys; strip
  the `<target>___<tool>` federated prefix. **[agentcore-gateway-lambda-toolname-contract]**

### Direction C (RANK 2) — AgentCore Runtime agents using LiteLLM/RouteIQ as the model backend

- **Seam:** the agent framework's model object (Strands LiteLLM provider;
  LangGraph `ChatLiteLLM` or an OpenAI-compatible client with
  `base_url → RouteIQ /v1`). **AgentCore API:** the agent draws its *tools* from
  the gateway over streamable-HTTP + bearer (the canonical
  `streamablehttp_client(gateway_url, headers={Authorization: Bearer ...})`
  pattern), optionally hosted on **AgentCore Runtime**.
- **VERIFIED:** AgentCore is model-agnostic (any model in/out of Bedrock incl.
  OpenAI/Gemini); AWS docs show Strands/LangGraph drawing tools from the gateway
  while using a model object. **[research report §9]**
- **SPECULATIVE:** the specific Strands/LangGraph *LiteLLM-provider* wiring is
  established framework behavior, not shown in an AWS doc end-to-end. The
  plane-separation is clean: **model plane = LiteLLM/RouteIQ, tool plane =
  Gateway.**
- **Auth model:** two independent auth surfaces — the agent→gateway inbound auth
  (bearer/SigV4, as Direction A) and the agent→LiteLLM model-backend auth
  (RouteIQ virtual key / OIDC). They do not interact.
- **Effort:** **Medium** — lives mostly *outside* RouteIQ (in the agent app /
  AgentCore Runtime). RouteIQ's only job is to be a model backend, which it
  already is. If hosting on AgentCore Runtime, respect the **runtime CFN schema
  + invoke-URL** contract: `Code.S3.{Bucket,Prefix}` (not `{Uri}`),
  `ProtocolConfiguration` is a String enum (`MCP|HTTP|A2A|AGUI`),
  `agentRuntimeName` is underscores-only `<=48` chars, and the `/invocations`
  URL must be percent-encoded in code (CFN intrinsics can't URL-encode the ARN).
  **[agentcore-runtime-cfn-schema-and-invoke-url]**
- **NET-NEW vs existing plugin:** **NET-NEW** but orthogonal to RouteIQ's
  codebase — it does not touch the (d) plugin; RouteIQ participates only as a
  model backend.

### Direction B (RANK 3) — RouteIQ/LiteLLM exposed THROUGH AgentCore

- **Seam:** `CreateGatewayTarget` with an **MCP-server `targetConfiguration`**
  pointing at RouteIQ's MCP endpoint (seam (b), reached through the `/v1`
  sub-mount), **preferred** over an OpenAPI target on `/v1`. Plus a
  `CreateRegistryRecord` (`descriptorType: MCP`) for discovery.
- **VERIFIED:** Gateway treats MCP servers as native targets (incl.
  Gateway-to-Gateway federation); Agent Registry's `CreateRegistryRecord`
  accepts an MCP descriptor with `synchronizationConfiguration.fromUrl`.
  **[research report §§3, 4, 8]**
- **SPECULATIVE:** that the OpenAPI-on-`/v1` sub-seam is *awkward* is inference
  from API semantics — `/v1/chat/completions` is a model-invocation API, not a
  tool; wrapping it as an MCP "tool" produces a tool that *is itself a chat
  call*. Better reserved for RouteIQ's tool-shaped control/eval/governance
  endpoints. Not a documented anti-pattern.
- **Auth model:** the gateway's *outbound* auth to the RouteIQ MCP-server target
  — OAuth 2LO/3LO/token-exchange or SigV4 via the gateway service role
  (`GATEWAY_IAM_ROLE` with `iamCredentialProvider.service`). RouteIQ must accept
  whichever the target binds.
- **Effort:** **Medium-to-Large** — two control-plane resources +
  RouteIQ-side auth acceptance + the ASGI mount-path verification
  (`discover-mcp-surface.md` §6: RouteIQ's `/v1` mount over LiteLLM's root-level
  `/mcp`/`/sse`/`/{name}/mcp` mounts needs empirical path confirmation).
- **NET-NEW vs existing plugin:** **NET-NEW** — (d) is inbound; this is outbound
  (RouteIQ-as-a-target). Reuses nothing from (d).
- **Gotcha to respect:** if you register a `CreateRegistryRecord`, the
  **registry record API contract** bites — `createRegistryRecord` returns only
  `recordArn` (no top-level `recordId`); the `recordId` regex accepts BOTH a
  bare 12-char id AND an ARN (so a recordId-regex error is rarely actually about
  the id form — probe the live API); a record's own `onUpdate` is **not safely
  deployable** with vanilla `AwsCustomResource` (use onCreate+onDelete only, or a
  custom provider returning the bare id); A2A cards need the full 0.3 schema with
  non-empty `skills`. **[agentcore-registry-record-api-contract]**

## 2.4 Recommended next step — the smallest shippable integration

**Ship Direction A as a config-only consumer first; the (d)-plugin repoint is
the fast-follow.**

**Step 1 (smallest, config-only, no RouteIQ code change).** Add one
`mcp_servers` entry to the LiteLLM config that points at a real AgentCore
`gatewayUrl`, `transport: http`, `auth_type` matched to the gateway's inbound
authorizer (start with an **IAM** gateway → LiteLLM SigV4 `service=bedrock-agentcore`,
because RouteIQ's EKS Pod Identity role already carries `BedrockInvoke` and the
boto3 cred chain resolves it — no new secret to manage). Verify the gateway's
tools appear in `GET /mcp-rest/tools/list` and a `tools/call` succeeds. This is
the canonical Direction-A join and exercises the auth-coherence rule live (the
one SPECULATIVE link in the chain).

**Step 2 (fast-follow, small code).** Repoint the existing
`bedrock_agentcore_mcp.py` plugin from the fabricated `bedrock-agent-runtime`
URL (lines 240-241) to a real `gatewayUrl` (config-supplied), and add SigV4 /
OAuth-bearer support to the registration path so the (d) plugin becomes a
**working Gateway consumer** instead of a stub. This is the "small, concrete
change that turns the existing plumbing into a working AgentCore-Gateway
consumer" the research report calls out — and it makes RouteIQ's *own* REST MCP
surface (a) expose the gateway's tools to RouteIQ-native clients. (Note: prefer
routing real invocations through seam (b)'s real MCP transport, since (a)'s
`invoke_tool` uses a non-MCP custom REST body and cannot sign AWS requests.)

Everything after that (Direction C agent apps, Direction B federation +
registry record) is incremental and gated on a real need, not on capability.

## 2.5 Seeds to file

1. **`feat: Direction-A AgentCore Gateway as LiteLLM mcp_servers entry`** (P1,
   small) — add the config-only consumer (Step 1), document the IAM-vs-CUSTOM_JWT
   auth-coherence matrix and the `transport: http` pin, add an integration smoke
   test that lists + calls one gateway tool. *Verifies the single SPECULATIVE
   link (the end-to-end LiteLLM↔Gateway pairing).*
2. **`feat: repoint bedrock_agentcore_mcp plugin to a real gatewayUrl + SigV4`**
   (P2, small) — Step 2; replace the `TSTALIASID` legacy URL, add SigV4/OAuth to
   `_register_agents`, add live `tools/list` discovery to populate the empty
   tools list, add MagicMock-context tests. *Depends on seed 1.*
3. **`chore: verify ASGI mount-path for the real MCP surface under /v1`** (P2,
   small) — empirically confirm the external paths for `/v1/mcp`, `/v1/sse`,
   `/v1/{server}/mcp` (the `/v1` sub-mount over LiteLLM's root mounts) before any
   client wiring. *Blocks Direction B; de-risks Direction A.*
4. **`spike: Direction-B federate RouteIQ MCP server as an AgentCore MCP-server
   target + Agent Registry record`** (P3, medium) — only if there is a need to
   expose RouteIQ tools through AgentCore; carries the
   `agentcore-registry-record-api-contract` gotcha (onCreate+onDelete only).
5. **`docs: Direction-C reference — Strands/LangGraph agent on AgentCore Runtime
   using RouteIQ as model backend`** (P3, medium) — a worked example; carries the
   `agentcore-runtime-cfn-schema-and-invoke-url` gotcha.

---

## Source map

- **Part 1 sources:** `research/arch-agentcore/discover-routeiq-current.md`,
  `discover-vsr-current.md`; `docs/architecture/four-way-comparison.md` (April
  baseline being updated); `docs/architecture/aws-rearchitecture/{10-aws-native-target-architecture,
  20-kumaraswamy-thompson-router,40-pluggable-routing-and-mlops,90-roadmap-completion-2026-06-15}.md`;
  `deploy/cdk/lib/` (3 stacks); vllm-sr-on-aws `ADRs/0018-eks-auto-mode-supersedes-ecs.md`.
- **Part 2 sources:** `research/notes/final_report_agentcore-gateway-litellm-integration-18d379.md`
  (AgentCore authority); `research/arch-agentcore/discover-mcp-surface.md` (the
  four current seams); `src/litellm_llmrouter/gateway/plugins/bedrock_agentcore_mcp.py`
  (the (d) stub, lines 240-241, 253-257);
  `reference/litellm/.../mcp_server/{server,mcp_server_manager}.py` (seam (b),
  READ-ONLY).
- **Gotchas cited:** `agentcore-gateway-lambda-toolname-contract`,
  `agentcore-registry-record-api-contract`, `agentcore-runtime-cfn-schema-and-invoke-url`.
