# 30 — Migration Roadmap: Helm-only → AWS-Native HA Target

> **Status**: Plan. **Date**: 2026-06-14.
> **From**: RouteIQ is framework-complete but **Helm-only on AWS, zero provisioning IaC**
> (`routeiq-vs-vllmsr-aws-gap.md` §1 — the entire AWS story is the 640-line manual
> `docs/deployment/aws.md` runbook + a chart that emits K8s objects only).
> **To**: the AWS-native, HA target in `10-aws-native-target-architecture.md`.
> **Strategy**: P0–P2 are **pure CDK ports** from vllm-sr-on-aws (low risk — RouteIQ
> gains tested constructs). P3 is **net-new design** (RouteIQ's differentiation:
> Kumaraswamy-Thompson + adapter framework + MLOps loop). P4 hardens the existing
> framework control plane onto the new substrate.

Construct citations are `vllm-sr-on-aws/cdk/lib/<file>` per `vllmsr-patterns.md`
and the gap-doc inventory (`routeiq-vs-vllmsr-aws-gap.md` §1.2). Eng-week estimates
assume one engineer fluent in CDK + the RouteIQ codebase.

---

## Risk classification at a glance

| Phase | Nature | Risk | Why |
|---|---|---|---|
| **P0 — CDK foundation** | Pure CDK port | **Low** | Constructs are tested + live in vllm-sr-on-aws; RouteIQ chart already targets this environment |
| **P1 — Externalized state** | Pure CDK port | **Low–Med** | Constructs ported; the *only* app work is wiring env vars the chart + ADR-0021 already expect. Aurora rollback is slow (≈30 min) |
| **P2 — Config + observability + data-lake** | Pure CDK port | **Low–Med** | AppConfig/AMP/AMG/Firehose ported; app emits OTel + reads AppConfig (small adapters) |
| **P3 — Routing + MLOps layer** | **Net-new design** | **Med–High** | Kumaraswamy-Thompson is new algorithm; adapter framework + closed-loop MLOps are RouteIQ-built, not ported |
| **P4 — Governance / tenant / UI hardening** | Framework-on-substrate | **Med** | Code exists (governance/OIDC/UI); work is wiring to Aurora/ESO/ALB + multi-tenant ops |

---

## P0 — CDK Foundation (VPC / EKS Auto Mode / ECR / IRSA)

**Goal.** Stand up a deployable AWS substrate: VPC, an EKS Auto Mode cluster, ECR repos,
and the IRSA factory — so `helm install routeiq-gateway` lands on real, CDK-provisioned
infra instead of hand-run CLI. Replaces `docs/deployment/aws.md`.

**BUILD-NEW vs PORT.** Almost entirely **PORT** from vllm-sr-on-aws:
- VPC + SGs + endpoints — port `network_construct.py` (VPC, 7 SGs, 11 interface endpoints
  incl. `BEDROCK_RUNTIME`, S3 gateway endpoint). EKS stack owns its VPC pattern
  (`vllm_sr_eks_stack.py`, 10.30/16).
- EKS Auto Mode cluster — port `eks_cluster_construct.py` (L1 `CfnCluster`,
  Compute/BlockStorage/ELB blocks `enabled`, OIDC provider, IRSA-via-CfnJson trust-key
  fix, access entries, container-insights add-on). See ADR-0030.
- IRSA factory — port from `vllm_sr_eks_stack.py` (the router/gateway/minter IRSA
  pattern); RouteIQ needs **one** pod role (Bedrock invoke + S3/Secrets/CW), exactly
  the policy already documented in `docs/deployment/aws.md:268-322`.
- ECR — port `ecr_construct.py` (immutable, scan-on-push, **GHCR pull-through cache** —
  RouteIQ publishes to `ghcr.io/baladithyab/routeiq` today, `values.yaml:33`, so the
  pull-through cache bridges the existing pipeline with zero image-pipeline rework).
- cdk-nag — port `nag_suppressions.py` + `AwsSolutionsChecks` (RouteIQ gains
  policy-as-code it never had).
- **BUILD-NEW (small)**: a `RouteIqStack` CDK app wiring these constructs + the chart's
  ServiceAccount IRSA annotation (`values.yaml:206-207`).

**Deliverables.** A `cdk/` tree with `RouteIqStack`; `cdk deploy` produces a private
multi-AZ Auto Mode cluster; `helm install` with the IRSA role-arn succeeds; CDK unit +
snapshot tests (mirror vllm-sr's test layout).

**Eng-weeks.** ~3–4.

**Dependencies.** None (foundation).

**Operator-gated.** AWS account bootstrap (`cdk bootstrap`), the actual `cdk deploy`,
EKS Auto Mode CIDR-lock decision (public-CIDR-locked vs private+bastion). Carry the
vllm-sr lessons: ASCII-only IAM descriptions at synth time; VPC-quota / retain-resources
on teardown.

---

## P1 — Externalized State (Aurora + ElastiCache Serverless)

**Goal.** Provision the durable + hot state backends RouteIQ's ADR-0021 (externalized
state) and ADR-0015 (leader election) already expect, and wire the chart's
`externalPostgresql` / `externalRedis` blocks (`values.yaml:316-338`) to them.

**BUILD-NEW vs PORT.** **PORT** the constructs; **wire** the app (no new app logic —
ADR-0021 already externalized `SessionCache` + `SharedCircuitBreakerState`, governance.py
already persists to Postgres):
- Aurora PostgreSQL Serverless v2 — port `replay_store_construct.py` (scale-to-zero,
  KMS, IAM-auth, 30-day secret rotation, schema-bootstrap Lambda + custom resource).
  Backs `governance.py` keys/spend/workspaces, `usage_policies.py` budgets, and (P3)
  bandit posteriors. See ADR-0028.
- ElastiCache Serverless (Valkey) — port `cache_construct.py` (IAM-auth user group, KMS).
  Backs `routeiq:session:*`, `routeiq:cb:*` (ADR-0021), `semantic_cache.py` L2, and
  rate-limit windows. See ADR-0029.
- **BUILD-NEW (small)**: External Secrets `ClusterSecretStore` wiring so the chart's
  `externalsecret.yaml` (already present, `external-secrets.io/v1beta1`) reads the
  Aurora/cache credentials from Secrets Manager; DB-URL rendered from the rotated secret
  (carry vllm-sr's "no env-expansion for postgres.password → boot-render" lesson).

**Deliverables.** Aurora + ElastiCache provisioned; chart points at both; HA leader
election (ADR-0015) uses the K8s Lease (no DB dependency); ADR-0021 state survives pod
churn; migrations init-container runs against Aurora (`deployment.yaml:44-64`).

**Eng-weeks.** ~2–3.

**Dependencies.** P0 (cluster + IRSA + secret store).

**Operator-gated.** The `cdk deploy` of Aurora; **deploy Aurora SEPARATELY from app CI**
— Aurora rollback is ≈30 min (vllm-sr lesson). Secret-rotation read-once race →
recycle-on-rotate.

---

## P2 — Config + Observability + Data Lake (AppConfig + AMP/AMG/CW + Firehose)

**Goal.** Hot-reloadable config + routing-strategy delivery (no redeploy), AWS-native
metrics/traces/logs, and the routing-telemetry data lake that feeds MLOps in P3.

**BUILD-NEW vs PORT.** **PORT** the constructs; **wire** the app:
- AppConfig — port `config_state_construct.py` (Application/Environment/Profile/
  DeploymentStrategy/HostedConfigurationVersion + validator Lambda + S3 + SNS). Replaces
  the chart's S3-ETag hot-reload (`values.yaml:233-248`) with AppConfig for config **and**
  for `strategy_registry.py`'s A/B + staged-rollout selection (the registry's
  `ExperimentConfig`/`StagedStrategy` become AppConfig-delivered — hot strategy swaps
  without redeploy). See ADR-0026. **BUILD-NEW (small)**: an AppConfig poll/extension
  adapter in the gateway.
- Observability — port `observability_construct.py` (AMP `CfnWorkspace`, AMG Grafana
  `CfnWorkspace`, CW dashboard, 9 alarms, MetricFilters, 2 log groups w/ PII
  DataProtectionPolicy, SNS). The chart already ships a `ServiceMonitor`
  (`servicemonitor.yaml`) + `PrometheusRule` + Grafana dashboard ConfigMap — point the
  ServiceMonitor at AMP remote-write. `telemetry_contracts.py` already emits OTel
  `gen_ai.*` spans. See ADR-0027.
- Data Lake — port `data_lake_construct.py` (routing_decision CW Logs → Firehose
  JSON→Parquet → S3 → Glue table w/ partition projection → Athena). **BUILD-NEW (small)**:
  ensure the gateway emits a structured `routing_decision` log line (selected_model,
  strategy, latency, cost) the subscription filter can pick up. Carry vllm-sr lessons:
  the **CW log group must be CDK-created** (CFN MetricFilter needs it to pre-exist);
  `response_body_mode: BUFFERED` if completions must be captured for the SFT corpus.

**Deliverables.** Config + strategy hot-reload via AppConfig; AMP/AMG dashboards live;
9 alarms → SNS; routing telemetry landing in S3/Athena as the MLOps training corpus.

**Eng-weeks.** ~3.

**Dependencies.** P0 (cluster); P1 (state, for strategy-state durability). P2 can run
largely in parallel with P1.

**Operator-gated.** AMG workspace + IdP wiring; AppConfig deployment-strategy approval;
data-lake retention/cost.

---

## P3 — Routing + MLOps Layer (Kumaraswamy-Thompson + adapter framework + closed loop)

**Goal.** RouteIQ's **differentiation** on the new substrate: ship the
Kumaraswamy-Thompson bandit as a first-class pluggable strategy, formalize the adapter
framework, and close the MLOps loop (telemetry → train → registry → rollout) on AWS.

**BUILD-NEW vs PORT.** This phase is **net-new design** — the routing intelligence is
what RouteIQ adds *over* the ported substrate (gap doc §3 reverse gap):
- **BUILD-NEW**: Kumaraswamy-Thompson router (`20-kumaraswamy-thompson-router.md`) as a
  `CustomRoutingStrategyBase` plugin (ADR-0002), posteriors persisted in Aurora (P1),
  hot EMA in ElastiCache (P1), arm-config delivered via AppConfig (P2). Registered in
  `strategy_registry.py` alongside the existing ~9 strategies.
- **BUILD-NEW**: the pluggable-adapter + MLOps framework (`40-pluggable-routing-and-mlops.md`)
  — the decoupled data-plane/control-plane loop from `docs/architecture/mlops-loop.md`,
  rehomed: telemetry sink = the P2 data lake (not Jaeger); training jobs = ephemeral EKS
  Jobs reading the Athena/Parquet corpus; artifact registry = S3 (+ optional MLflow);
  rollout = AppConfig + the chart's sidecar/`mtime` hot-reload (`mlops-loop.md` §2.3, §3A).
- **BUILD-NEW**: wire `eval_pipeline.py` (COLLECT/EVALUATE/AGGREGATE/FEEDBACK, 617 LOC)
  to consume the data-lake corpus and feed `ModelQualityTracker` scores back into the
  bandit + personalized router.
- **PORT (optional reference)**: the VSR `rl_driven` Thompson bandit can be exposed as
  *one* backend strategy behind the registry (the gap doc's "VSR bandit as optional
  routing backend"); gated by the `vsr-vs-routeiq-decision-v3.md` "measure-before-
  integrating" rule.

**Deliverables.** Kumaraswamy-Thompson live as a selectable strategy; closed MLOps loop
(corpus → train → S3 → hot-reload) on AWS; eval-driven feedback wired; A/B between
strategies via the registry + AppConfig.

**Eng-weeks.** ~4–6 (the new bandit + closing the loop is the bulk of net-new effort).

**Dependencies.** P1 (Aurora/cache for posteriors/EMA), P2 (AppConfig delivery + data lake).

**Operator-gated.** Whether to wire the VSR bandit as a backend (decision-v3 gate);
pickle-model opt-in (`LLMROUTER_ALLOW_PICKLE_MODELS`, `mlops-loop.md` §2.2); training
schedule/budget.

---

## P4 — Governance / Multi-Tenant / UI Hardening on the New Substrate

**Goal.** Run RouteIQ's **already-shipped** control plane (gap doc §5 — the layer
vllm-sr-on-aws lacks) durably and securely on the AWS-native estate.

**BUILD-NEW vs PORT.** Framework-on-substrate — the code exists; the work is wiring +
hardening:
- **WIRE**: `governance.py` (Org→Workspace→Key, 718 LOC) + `usage_policies.py` budgets →
  Aurora (P1) for durable multi-tenant rows; rate-limit windows → ElastiCache (P1).
- **WIRE**: `oidc.py` (1475 LOC, JWKS/token-exchange) + React admin UI (`/ui/`,
  `routes/admin_ui.py`) behind ALB/ACM (chart ingress) with the OIDC client-secret via
  External Secrets → Secrets Manager (chart `externalsecret.yaml`, `values.yaml:347-355`).
- **WIRE (optional)**: native Bedrock Guardrail — port `native_guardrail_construct.py`
  (`bedrock.CfnGuardrail`, flag-gated) as an additional layer behind the 14 in-app
  guardrail handlers (`guardrail_policies.py`, 1285 LOC).
- **PORT (optional)**: WAF — port `waf_construct.py` (3 managed groups + rate-limit, ALB
  association) in front of the ALB. `docs/deployment/aws.md:588` already lists WAF as a
  best practice; this makes it IaC.
- **HARDEN**: the chart's `networkPolicy` (`values.yaml:486-516`, default-OFF) → enable
  with explicit ingress selectors + egress to Aurora:5432 / cache:6379 (carry vllm-sr's
  NetworkPolicy-egress lesson); `readOnlyRootFilesystem` (already `true`,
  `values.yaml:188`) validated against the write-path probe.

**Deliverables.** Multi-tenant governance durable in Aurora; OIDC SSO + admin UI live
behind ALB/ACM; optional native Guardrail + WAF; NetworkPolicy enforced.

**Eng-weeks.** ~3–4.

**Dependencies.** P0 (ALB/ingress), P1 (Aurora/cache), P2 (observability for tenant
metrics).

**Operator-gated.** OIDC IdP registration (Keycloak/Auth0/Okta/Azure/Google); tenant
onboarding; WAF rule tuning; the `SR_TENANCY`-style header-strip prerequisite if
fronting other engines.

---

## Sequencing & critical path

```
P0 (foundation, ~3-4w)
   │
   ├──► P1 (state, ~2-3w) ───────────┐
   │                                  ├──► P3 (routing+MLOps, ~4-6w) ──► P4 (govern/UI, ~3-4w)
   └──► P2 (config/obs/lake, ~3w) ───┘
```

- **P1 and P2 can overlap** after P0 (independent constructs; P3 needs both).
- **Pure-CDK-port phases (P0–P2)** are the low-risk majority — RouteIQ banks
  vllm-sr-on-aws's tested constructs (gap doc: 446 tests, live across two substrates).
- **Net-new design is concentrated in P3** (Kumaraswamy-Thompson + closing the MLOps
  loop) — RouteIQ's actual differentiation.
- **Total: ~15–20 eng-weeks** to the full target; a deployable substrate running the
  existing framework (P0+P1+P2+P4-wiring) lands well before P3.

The end state is the synthesis in `10-aws-native-target-architecture.md` §5: RouteIQ's
framework on vllm-sr-on-aws's substrate, two layers of one platform.
