# vllm-sr-on-aws → RouteIQ: the deployment-pattern source map

> **Provenance note.** This file is the durable, in-repo record of the
> `vllm-sr-on-aws` CDK constructs that the AWS re-architecture docs
> (`10`/`30`, ADRs `0026`–`0030`) learn from. It supersedes the ephemeral
> scratch artifact (`/tmp/vllmsr-patterns.txt`) used during the design
> session of 2026-06-14 so that every `construct → pattern` citation in this
> directory resolves to a committed file, not a machine-local temp path.
>
> The constructs themselves live in the **separate** `vllm-sr-on-aws`
> repository under `cdk/lib/`. Citations of the form
> `vllm-sr-on-aws/cdk/lib/<file>.py:<lines>` are cross-repo references to
> that project's production CDK; the line numbers were accurate at the
> 2026-06-14 snapshot (the named symbols are stable even if offsets drift —
> see `99-review-findings.md`).

## The construct → deployment-pattern map (`vllm-sr-on-aws/cdk/lib/`)

| Construct | Deployment pattern RouteIQ learns from | Taught in |
|---|---|---|
| `eks_cluster_construct.py` | **EKS Auto Mode** — L1 `CfnCluster`, the 3 Auto-Mode blocks (ComputeConfig / StorageConfig / ElasticLoadBalancing), **IRSA via `CfnJson` trust-key** (ARN-derived OIDC issuer), OIDC provider, container-insights add-on, CW log groups / metric filters / alarms / dashboard / SNS. | ADR-0030, ADR-0027 |
| `vllm_sr_eks_stack.py` | Own VPC; per-workload IRSA roles (router→bedrock-mantle, EAIG gateway→native Bedrock, bearer-minter); multi-account capacity loop + `CfnOutput` per account; AWS Budget. | ADR-0030, doc 10 |
| `bedrock_capacity_member_stack.py` | Per-account cross-account IAM (OIDC web-identity **or** minter `ArnPrincipal` trust); a **separate app**, not a StackSet. | doc 10 (multi-account) |
| `replay_store_construct.py` | **Aurora PostgreSQL Serverless v2** — scale-to-zero, IAM auth, KMS, 30-day rotation, schema-bootstrap Lambda + custom resource. | ADR-0028 |
| `cache_construct.py` | **ElastiCache Serverless (Valkey)** — IAM-auth user group, KMS, always-on TLS. | ADR-0029 |
| `config_state_construct.py` | **AppConfig GitOps** — Application / Environment / ConfigurationProfile / DeploymentStrategy / HostedConfigurationVersion / Deployment + validator Lambda + S3 + SNS. | ADR-0026 |
| `observability_construct.py` | **AMP** (`CfnWorkspace`) + **AMG** (Grafana `CfnWorkspace`, flag-gated) + CW dashboard + 9 alarms + 3 metric filters + 2 log groups (PII `DataProtectionPolicy`) + SNS. | ADR-0027 |
| `data_lake_construct.py` | `routing_decision` CW Logs → Firehose (JSON→Parquet) → S3 → Glue table (partition projection, no crawler) → Athena. | doc 10, doc 40 (MLOps) |
| `security_construct.py` | KMS CMKs; `TaskExecutionRole` + `TaskRole` (Bedrock invoke + bearer-mint); Secrets. | ADR-0030 |
| `build_pipeline_construct.py` | 8 CodeBuild projects + 2 CodePipelines (model-promotion + deploy). | doc 30 (roadmap) |
| `network_construct.py` | VPC, 7 SGs, 11 interface endpoints incl. `BEDROCK_RUNTIME`, S3 gateway endpoint, Cloud Map. | doc 10 |
| `waf_construct.py` | WAFv2 WebACL (3 managed rule groups + rate limit) + ALB association. | doc 10 |
| `ecr_construct.py` | 7 immutable + scan-on-push repos + pull-through-cache (GHCR / DockerHub). | doc 30 |
| `jailbreak_canary_construct.py` / `opus_share_monitor_construct.py` / `policy_version_construct.py` | Scheduled Lambda / Fargate batch + EventBridge + alarms. | (reference) |
| `nag_suppressions.py` | Evidenced `cdk-nag` suppression pattern. | (reference) |
| `_rl_baselines.py` | RL baseline helpers. | doc 20 (contrast) |

## HA / ops lessons (hard-won, live)

These are the operational foot-guns the `vllm-sr-on-aws` team hit running
this stack live on EKS. They are the load-bearing "consequences" the ADRs
encode so RouteIQ inherits the scar tissue, not the scars:

1. **EKS Auto Mode CIDR-lock = `spec.loadBalancerSourceRanges`**, NOT the
   service annotation (the annotation is silently ignored under Auto Mode).
2. **IRSA trust-key must be the ARN-derived OIDC issuer**, never
   `.replace("https://", "")` on the issuer URL — the `.replace` is a silent
   no-op against a CFN token and yields `AccessDenied` at runtime.
3. **RWO-EBS + `RollingUpdate` = Multi-Attach deadlock** → use
   `strategy: Recreate` for any pod with a `ReadWriteOnce` EBS volume (or keep
   the pod stateless, which RouteIQ does — see ADR-0028/0029).
4. **APM webhook is PSA-restricted** → needs an explicit opt-out label on
   every injected pod.
5. **Bearer tokens ≤ 12 h rotation** via a CronJob (cross-account capacity).
6. **`kubectl apply -k` clobbers SA annotations** (writes the literal
   `${...}` placeholder) → resolve via `envsubst` in a `deploy-*.sh` wrapper,
   never raw `apply -k`.
7. **Aurora rollback ≈ 30 min** → deploy the database in a **separate** CI
   stage from the app so an app rollback isn't gated on a 30-min DB revert.
8. **CW metric filters need a CDK-created log group that pre-exists** — a
   runtime-created (Fluent Bit) log group makes the `CFN MetricFilter` fail
   on first deploy.

## See also

- `10-aws-native-target-architecture.md` — how these constructs compose into
  the RouteIQ-on-AWS target.
- `30-migration-roadmap.md` — the phased plan to port them.
- ADRs `0026`–`0030` — one decision record per substrate pattern.
- `99-review-findings.md` — the adversarial review that verified every
  citation in this set against the real `vllm-sr-on-aws` source.
- `vllm-sr-on-aws/docs/architecture/routeiq-vs-vllmsr-aws-gap.md` — the
  cross-repo gap analysis that motivated this re-architecture.
