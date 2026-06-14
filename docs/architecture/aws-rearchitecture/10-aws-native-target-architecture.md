# 10 — AWS-Native Target Architecture for RouteIQ

> **Status**: Design (target state). **Date**: 2026-06-14.
> **Thesis**: RouteIQ-the-framework **adopts vllm-sr-on-aws's deployment substrate**
> (a hardened CDK + EKS Auto Mode estate) **while keeping its superior framework
> capabilities** (LiteLLM API-translation, ~9 routing strategies + Kumaraswamy-Thompson,
> the three-tier governance/OIDC control plane, the pluggable-adapter + MLOps loop).
> This is exactly the synthesis the gap analysis recommended — see
> `vllm-sr-on-aws/docs/architecture/routeiq-vs-vllmsr-aws-gap.md` §6.2 path #2.
>
> **Sibling docs** (this folder; reference, do not depend on file existence yet):
> - ADRs `0026-appconfig.md`, `0027-otel-amp-amg.md`, `0028-aurora-serverless.md`,
>   `0029-elasticache-serverless.md`, `0030-eks-auto-mode.md` — the deployment-pattern decisions.
> - `20-kumaraswamy-thompson-router.md` — the new bandit algorithm.
> - `40-pluggable-routing-and-mlops.md` — the adapter framework + closed-loop MLOps.
> - `30-migration-roadmap.md` — the phased plan to get here.

---

## 0. Where we are starting from

RouteIQ today is **framework-complete but Helm-only on AWS** (gap doc §1):

- A generic Helm chart (`deploy/charts/routeiq-gateway/`, 15 templates) that emits
  **K8s objects only** — and already *expects* an AWS-native cluster: IRSA SA
  annotations (`values.yaml:206-207`), External Secrets Operator
  (`templates/externalsecret.yaml`, `external-secrets.io/v1beta1`), a Prometheus-Operator
  `ServiceMonitor` (`templates/servicemonitor.yaml`), `coordination.k8s.io` Lease RBAC
  for leader election (`templates/leader-election-rbac.yaml`), PDB, anti-affinity,
  `readOnlyRootFilesystem` security context, and a graceful-drain `preStop` hook
  (`templates/deployment.yaml:128-131`).
- **Zero AWS-provisioning IaC** — the only AWS story is a 640-line manual runbook of
  copy-paste `aws … create-*` commands (`docs/deployment/aws.md`).

vllm-sr-on-aws is the mirror image: **deep CDK substrate, thin routing**. RouteIQ's
chart is a tenant that has been waiting for exactly the landlord vllm-sr-on-aws built.
The target architecture **provisions that landlord in CDK** and runs RouteIQ as the
front-layer control plane on it.

---

## 1. The layered target

```
                              Internet / VPC clients
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  AWS WAFv2 WebACL             │  port: waf_construct.py
                        │  (3 managed groups + rate)    │  (3 mgd groups + rate-limit)
                        └──────────────┬───────────────┘
                                       ▼
        ┌──────────────────────────────────────────────────────────┐
        │  ALB (HTTPS/ACM)  →  AWS LB Controller Ingress (chart      │
        │  ingress.className: alb)   OR   NLB for L4 + CIDR-lock      │
        │  (spec.loadBalancerSourceRanges — Auto Mode CIDR-lock)     │
        └──────────────────────────┬───────────────────────────────┘
                                   ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │                EKS Auto Mode cluster  (multi-AZ, private)               │
   │  L1 CfnCluster, ComputeConfig/BlockStorage/ELB enabled                  │
   │  (port: eks_cluster_construct.py — ADR-0030)                            │
   │                                                                         │
   │   ┌─────────────────────────────────────────────────────────────────┐ │
   │   │  RouteIQ Gateway Deployment  (deploy/charts/routeiq-gateway)      │ │
   │   │  replicaCount 2+, PDB, anti-affinity, topology spread, HPA        │ │
   │   │                                                                   │ │
   │   │   ┌───────────────────────────────────────────────────────────┐ │ │
   │   │   │  RouteIQ pod (single container, uvicorn ROUTEIQ_WORKERS)   │ │ │
   │   │   │  • LiteLLM proxy mounted at /v1  (gateway/app.py:784)      │ │ │
   │   │   │    - /v1/chat/completions, /v1/messages, /v1/responses    │ │ │
   │   │   │    - chat/messages → Responses TRANSLATION (the edge)     │ │ │
   │   │   │  • Routing Intelligence: ~9 strategies + registry         │ │ │
   │   │   │    + Kumaraswamy-Thompson (20-...md)                      │ │ │
   │   │   │  • Governance: Org→Workspace→Key, OIDC, budgets, RBAC     │ │ │
   │   │   │  • Guardrail engine (14 handlers) + 12 plugins            │ │ │
   │   │   │  • React admin UI at /ui/                                 │ │ │
   │   │   │  IRSA SA (eks.amazonaws.com/role-arn)  ─────────────┐     │ │ │
   │   │   └────────────────────────────────────────────────────│─────┘ │ │
   │   └────────────────────────────────────────────────────────│───────┘ │
   │                                                             │         │
   │   Pod Identity / IRSA  (port: vllm_sr_eks_stack.py IRSA factory)      │
   └─────────────────────────────────────────────────────────────│───────┘
            │              │               │            │         │
            ▼              ▼               ▼            ▼         ▼
   ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ ┌────────┐ ┌──────────┐
   │ Aurora PG    │ │ ElastiCache │ │  AppConfig   │ │  KMS   │ │ Bedrock  │
   │ Serverless v2│ │ Serverless  │ │ (config +    │ │ CMKs   │ │ (+ other │
   │ scale-to-zero│ │ (Valkey)    │ │  routing-    │ │        │ │ providers│
   │ KMS+IAM-auth │ │ IAM-auth    │ │  strategy    │ │ envelope│ │ via      │
   │              │ │             │ │  hot-reload) │ │  for DB │ │ LiteLLM) │
   │ keys/spend/  │ │ cache /     │ │              │ │ cache  │ │          │
   │ governance/  │ │ ratelimit / │ │ ADR-0026     │ │ secrets│ │          │
   │ bandit-state │ │ EMA / CB    │ │              │ │        │ │          │
   │ ADR-0028     │ │ ADR-0029    │ │              │ │        │ │          │
   └──────────────┘ └─────────────┘ └──────────────┘ └────────┘ └──────────┘
            │              │                                          │
            └──────────────┴──────────────┐                          │
                                          ▼                          ▼
                  ┌────────────────────────────────────┐   ┌──────────────────┐
                  │  Observability (ADR-0027)           │   │  Secrets via      │
                  │  AMP (CfnWorkspace) ← /metrics      │   │  External Secrets │
                  │  AMG (Grafana) dashboards           │   │  Operator →       │
                  │  CloudWatch Logs + MetricFilters    │   │  Secrets Manager  │
                  │  ServiceMonitor → AMP remote-write  │   │  (chart already   │
                  └──────────────┬─────────────────────┘   │  expects it)      │
                                 ▼                          └──────────────────┘
                  ┌────────────────────────────────────┐
                  │  Data Lake (MLOps telemetry)        │
                  │  routing_decision logs → Firehose   │
                  │  → S3 Parquet → Glue → Athena       │
                  │  (port: data_lake_construct.py)     │
                  │  feeds eval_pipeline + 40-...md loop │
                  └─────────────────────────────────────┘
```

**Reading the diagram.** Everything in the boxes outside the RouteIQ pod is a CDK
construct **ported from vllm-sr-on-aws** (cited per box, full map in §2). Everything
*inside* the RouteIQ pod is RouteIQ's existing framework code (cited in §4 as the
"edge"). The substrate is vllm-sr's; the brains are RouteIQ's.

---

## 2. Component map — RouteIQ module → AWS landing

Each RouteIQ subsystem maps to a CDK construct from
`vllmsr-patterns.md` (the construct→pattern map) and the gap-doc inventory
(`routeiq-vs-vllmsr-aws-gap.md` §1.2). "Port" = lift the construct largely as-is;
"reuse" = the construct already does what RouteIQ needs.

| RouteIQ module / artifact | What it needs | AWS landing (construct → service) | Port/reuse |
|---|---|---|---|
| **LiteLLM proxy** (`gateway/app.py:784`, mounts `/v1`) | container runtime, ALB, autoscale, drain | EKS Auto Mode Deployment via the existing Helm chart; cluster from `eks_cluster_construct.py` (L1 CfnCluster, ELB/Compute/BlockStorage enabled); ALB via AWS LB Controller (chart `ingress.className: alb`) | Port (cluster) |
| **governance.py** (718 LOC: `KeyGovernance`, `WorkspaceConfig`, `GovernanceContext`) | durable keys/spend/workspace/org rows | **Aurora PostgreSQL Serverless v2** via `replay_store_construct.py` (scale-to-zero, KMS, IAM-auth, 30-day rotation, schema-bootstrap Lambda) | Port |
| **usage_policies.py** (781 LOC: budgets, rate-limit periods) | shared counters across pods/workers (ADR-0021) | **ElastiCache Serverless (Valkey)** via `cache_construct.py` (IAM-auth user group, KMS) for rate-limit + budget windows; durable budgets in Aurora | Port |
| **strategy_registry.py** (1453 LOC: A/B + staged rollout, `VersionedStrategyEntry`, `ExperimentConfig`) | hot-reloadable strategy/version selection without redeploy | **AWS AppConfig** via `config_state_construct.py` (Application/Environment/Profile/DeploymentStrategy/HostedConfigurationVersion + validator Lambda) — routing-strategy hot-reload (ADR-0026); strategy *state* in Aurora | Port |
| **centroid_routing.py** `SessionCache` + **resilience.py** `SharedCircuitBreakerState` (ADR-0021) | externalized session affinity + circuit-breaker state | **ElastiCache Serverless** (`cache_construct.py`) — `routeiq:session:*`, `routeiq:cb:*` keys | Reuse |
| **Kumaraswamy-Thompson router** (`20-...md`) | persistent posteriors / bandit-arm state | Aurora (`replay_store_construct.py`) for posteriors; ElastiCache for hot EMA | Port |
| **eval_pipeline.py** (617 LOC: COLLECT/EVALUATE/AGGREGATE/FEEDBACK, `EvalJudge`, `ModelQualityTracker`) + MLOps loop (`40-...md`) | routing telemetry capture + offline training corpus | **Data Lake** via `data_lake_construct.py` (routing_decision CW Logs → Firehose → S3 Parquet → Glue → Athena); training as ephemeral K8s Jobs | Port |
| **semantic_cache.py** (744 LOC: L1 LRU + L2 vector store) | L2 distributed vector cache | **ElastiCache Serverless** (`cache_construct.py`) | Reuse |
| **oidc.py** (1475 LOC: JWKS, token exchange) + admin **React UI** (`ui/`, `routes/admin_ui.py`) | TLS ingress, secret storage, OIDC client-secret | ALB/ACM via chart ingress; client-secret via **External Secrets Operator** → Secrets Manager (chart `externalsecret.yaml`); KMS CMK from `security_construct.py` | Reuse + Port |
| **guardrail_policies.py** (1285 LOC, 14 handlers) + 12 gateway plugins | optional native Bedrock Guardrail | `native_guardrail_construct.py` (`bedrock.CfnGuardrail`, flag-gated) | Port (optional) |
| **telemetry_contracts.py** (OTel GenAI `gen_ai.*` spans) | metrics/traces backend | **AMP + AMG + CloudWatch** via `observability_construct.py` (CfnWorkspace AMP, Grafana CfnWorkspace, 9 alarms, MetricFilters, SNS); chart `ServiceMonitor` remote-writes to AMP (ADR-0027) | Port |
| **Secrets** (master key, provider keys, DB/Redis creds — `values.yaml:365-409`) | secret delivery + rotation | **KMS + Secrets Manager** via `security_construct.py`; chart ESO `ClusterSecretStore: aws-secrets-manager` | Reuse + Port |
| **Container images** | immutable, scanned registry | **7 ECR repos** via `ecr_construct.py` (immutable, scan-on-push, GHCR pull-through cache) — RouteIQ publishes to GHCR today; pull-through cache bridges it | Port |
| **Build/deploy** | deploy-to-AWS CI/CD (RouteIQ has none — GHCR only) | **CodeBuild + CodePipeline** via `build_pipeline_construct.py` (8 projects, 2 pipelines) | Port |
| **Network isolation** | VPC, SGs, Bedrock endpoint | `network_construct.py` (VPC, SGs, 11 interface endpoints incl. `BEDROCK_RUNTIME`, S3 gateway endpoint); EKS stack owns its VPC (`vllm_sr_eks_stack.py`, 10.30/16) | Port |

cdk-nag (`AwsSolutionsChecks`) and the evidenced suppression pattern
(`nag_suppressions.py`) come along with the constructs — RouteIQ inherits
policy-as-code it has never had.

---

## 3. HA story

RouteIQ already designed for HA at the K8s layer; the target wires that design to
AWS-native durable backends. Three layers:

**3.1 Compute HA — EKS Auto Mode multi-AZ.** The cluster (`eks_cluster_construct.py`,
ADR-0030) runs across the VPC's private subnets in multiple AZs; Auto Mode manages
node provisioning/bin-packing. The chart already spreads pods: `podAntiAffinity`
(`values.yaml:157-164`, soft hostname spread), `topologySpreadConstraints` hook for
zone spread (`values.yaml:166-173`), `PodDisruptionBudget` (`values.yaml:59-63`),
and HPA (`values.yaml:47-56`). For production, set anti-affinity `type: hard` /
add a `topology.kubernetes.io/zone` spread constraint so a single-AZ loss never
takes all replicas.

**3.2 Coordination HA — RouteIQ's existing K8s-native leader election (ADR-0015).**
The chart ships `coordination.k8s.io` Lease RBAC (`leader-election-rbac.yaml`) and the
deployment auto-mounts the SA token only when the Lease backend is active
(`deployment.yaml:34`). `leader_election.py` (1529 LOC: `K8sLeaseLeaderElection`,
`RedisLeaderElection`, singleton fallback) elects a single config-sync/leader-only-task
holder. On Auto Mode this needs **no external dependency** — the Lease lives in the
K8s API server, eliminating the PostgreSQL coupling ADR-0015 set out to remove.

**3.3 State HA — externalized state (ADR-0021) → Aurora / ElastiCache.** ADR-0021
already moved `SessionCache` and `SharedCircuitBreakerState` out of process to Redis.
In the target, "Redis" **is ElastiCache Serverless** (`cache_construct.py`, multi-AZ,
IAM-auth) and durable governance/spend/bandit state **is Aurora Serverless v2**
(`replay_store_construct.py`, scale-to-zero, automated backups, secret rotation).
Both are managed, multi-AZ, and survive pod churn — so a rolling deploy or node
replacement preserves routing affinity, circuit state, budgets, and bandit posteriors.

**3.4 Graceful drain.** The chart's `preStop: sleep 5` (`deployment.yaml:128-131`) +
`terminationGracePeriodSeconds: 30` (`values.yaml:541`) bleed connections before SIGTERM;
combined with the leader handover in ADR-0015 this gives zero-drop rollouts.

**3.5 The two hard-won lessons from vllm-sr-on-aws** (`vllmsr-patterns.md` HA notes):

- **RWO-EBS + RollingUpdate = Multi-Attach deadlock.** RouteIQ pods must stay
  **stateless** (all durable state is in Aurora/ElastiCache). The chart's volumes are
  `emptyDir` only (`deployment.yaml:154-163`) — keep it that way. If any future feature
  needs a PVC, set the workload `strategy: Recreate`, never `RollingUpdate`, on a
  ReadWriteOnce EBS volume, or two pods will deadlock trying to attach the same volume.
- **Auto Mode CIDR-lock = `spec.loadBalancerSourceRanges`**, NOT the load-balancer
  annotation. When exposing RouteIQ on an NLB (e.g. an internal control-plane endpoint),
  restrict source IPs via the Service's `loadBalancerSourceRanges`, and allowlist the
  NAT-pool /24, not a single /32 (the second lesson from the live EAIG cutover).

---

## 4. What RouteIQ KEEPS as its edge

These are the framework capabilities **vllm-sr-on-aws lacks** and that justify running
RouteIQ as the front layer rather than just deploying the VSR bandit (gap doc §3, §4, §5).
None of this changes in the re-architecture — it simply lands on a better substrate.

**4.1 The LiteLLM chat/messages → Responses translation (the headline edge).**
vllm-sr-on-aws's EAIG gateway is **passthrough-only** and architecturally **cannot**
reshape a `/v1/chat/completions` or `/anthropic/v1/messages` request into the Responses
API — so Responses-only models like `openai.gpt-5.5` **404** through it (gap doc §4.1).
RouteIQ inherits the exact bridge in its hard dependency `litellm==1.82.3`
(`pyproject.toml:21`): `LiteLLMCompletionResponsesConfig` and the Anthropic-messages↔OpenAI
adapters, with `/v1/responses` and `/v1/messages` mounted by the LiteLLM proxy
(gap doc §4.2). **RouteIQ closes the gpt-5.5 gap vllm-sr-on-aws cannot** — and vllm-sr's
own 2026-06-14 runbook reached back to LiteLLM to solve it (gap doc §4.3).

**4.2 ~9 routing strategies + the new Kumaraswamy-Thompson.** Centroid (`centroid_routing.py`,
1759 LOC), personalized EMA (`personalized_routing.py`, 783 LOC), Router-R1
(`router_r1.py`, 353 LOC), cost-aware Pareto, 5 native ML routers, the strategy A/B +
staged-rollout registry (`strategy_registry.py`, 1453 LOC), plus the new
Kumaraswamy-Thompson bandit (`20-kumaraswamy-thompson-router.md`). vllm-sr-on-aws
deploys *one* engine (the upstream VSR `rl_driven` Thompson keyed on `decision_name`);
RouteIQ is a routing **toolkit** where the bandit is one pluggable strategy among many.

**4.3 Governance / OIDC / UI control plane.** Three-tier Org→Workspace→Key governance
(`governance.py`, 718 LOC), OIDC SSO with live JWKS (`oidc.py`, 1475 LOC),
condition-based budgets/rate-limits (`usage_policies.py`, 781 LOC), 14 guardrail
handlers (`guardrail_policies.py`, 1285 LOC), RBAC, and a 6-page React admin UI. This is
a **shipped** multi-tenant control plane; vllm-sr-on-aws's tenancy is generator-only and
flag-gated OFF (gap doc §5).

**4.4 Pluggable-adapter + MLOps framework.** The `CustomRoutingStrategyBase` plugin
contract (ADR-0002), the LLM-as-judge eval loop (`eval_pipeline.py`, 617 LOC:
COLLECT/EVALUATE/AGGREGATE/FEEDBACK), and the decoupled data-plane/control-plane MLOps
lifecycle (`docs/architecture/mlops-loop.md`). On the new substrate the telemetry sink
becomes the **data lake** (`data_lake_construct.py`) and training runs as ephemeral
EKS Jobs — see `40-pluggable-routing-and-mlops.md`.

---

## 5. The explicit framing

This re-architecture is **RouteIQ adopting vllm-sr-on-aws's deployment substrate while
keeping its superior framework capabilities** — synthesis path #2 from the gap doc
(`routeiq-vs-vllmsr-aws-gap.md` §6.2). The two projects are complementary, not competing:

- vllm-sr-on-aws built the **IaC / substrate / observability** RouteIQ never had (3 CDK
  stacks, EKS Auto Mode, Aurora/ElastiCache Serverless, data lake, 446 tests).
- RouteIQ built the **control plane + routing toolkit + API translation** vllm-sr-on-aws
  lacks (governance/OIDC/UI, ~9 strategies, the LiteLLM Responses bridge).

The target is **two layers of one platform**: vllm-sr-on-aws's CDK constructs provision
the AWS-native, HA substrate; RouteIQ runs as the front-layer intelligence + control
plane on it, with the VSR Thompson bandit available as one optional pluggable strategy
behind RouteIQ's registry. Neither effort was wasted — this stitches them together.

The phased plan to get from "Helm-only, needs-a-ton-of-work" to this target is
`30-migration-roadmap.md`.
