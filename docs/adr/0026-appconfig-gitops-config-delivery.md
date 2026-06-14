# ADR-0026: AWS AppConfig as the GitOps Config-Delivery Plane

**Status**: Proposed
**Date**: 2026-06-14
**Decision Makers**: RouteIQ Core Team

## Context

RouteIQ already has a hot-reload config pipeline (`config_sync.py`, `hot_reload.py`),
but its only remote source is **S3 + ETag polling**:

- `ConfigSyncManager` downloads `s3://{bucket}/{key}` and detects change by ETag
  (`config_sync.py:5-6,120,166`), driven by `CONFIG_S3_BUCKET` / `CONFIG_S3_KEY`
  (`config_sync.py:130-142`); the `download_config_from_s3` helper is a raw
  `boto3` `download_file` (`config_loader.py:15-35`).
- `HotReloadManager.reload_config()` re-reads the file and re-instantiates the
  router (`hot_reload.py:108,455`). Routing-strategy edits and A/B weights also
  flow through this manager (`hot_reload.py:167,275,357`).

This works but has three structural weaknesses that a production AWS deployment
must close:

1. **No validation gate.** A malformed `config.yaml` — or one with an inline
   secret, or a `public: true` route with no guardrail — propagates to *every*
   replica the moment the ETag flips. The only safety net is RouteIQ's own
   in-process parse, which fails *after* the bad object is already authoritative.
2. **All-at-once blast radius.** ETag polling has no notion of a staged rollout
   or a bake period. A bad config reaches 100% of pods within one poll interval.
3. **No audit/approval trail.** S3 versioning records *that* the object changed;
   it does not gate *who* may change it, validate it, or require an approval step
   for prod.

vllm-sr-on-aws solved exactly this with **AWS AppConfig** as a validated,
staged, audited config-delivery plane (`cdk/lib/config_state_construct.py`). The
construct stands up the full AppConfig resource graph and a **Lambda validator**
that rejects bad configs *before* they deploy. RouteIQ should adopt the same
pattern as an optional `config_source` backend.

## Decision

Add an **AppConfig** config source alongside the existing S3/filesystem sources.
Provision the AppConfig control plane in IaC and have RouteIQ retrieve config via
the AppConfig polling contract instead of an ETag poll.

### Resource graph (mirrors `config_state_construct.py:551-709`)

- **`CfnApplication`** — one per deployment (`config_state_construct.py:571-576`).
- **`CfnEnvironment`** — one per stage (`dev`/`prod`) (`:578-584`).
- **`CfnConfigurationProfile`** — `location_uri="hosted"`, `type="AWS.Freeform"`
  for the YAML config (`:586-594`). The validator is attached as a
  `ValidatorsProperty(type="LAMBDA", content=<fn ARN>)` (`:631-636`) — a
  **Lambda-ARN validator, not a JSON_SCHEMA validator**, because RouteIQ's config
  has cross-field rules (a `public` route must enable a guardrail plugin; no
  inline secrets) that a flat JSON schema cannot express.
- **`CfnDeploymentStrategy`** — `growth_type="LINEAR"`, `growth_factor=20`,
  `deployment_duration_in_minutes=12`, `final_bake_time_in_minutes=5`
  (`:638-648`). Config rolls out linearly with a 5-minute bake, not all-at-once.
- **`CfnHostedConfigurationVersion`** — `content_type="application/x-yaml"`, seed
  content embedded at synth (`:650-658`). Chosen over an `AwsCustomResource`
  onCreate to avoid stale-attribute reads on stack update (`:566-570`).
- **`CfnDeployment`** — binds version→env→profile→strategy and is the resource
  that actually invokes the validator at deploy time (`:667-682`); it
  `add_dependency` on the validator's invoke permission (`:682`).

### Validator Lambda (`config_state_construct.py:1110-1239`)

A `lambda:InvokeFunction` `CfnPermission` grants principal
`appconfig.amazonaws.com`, scoped to the profile ARN (`:611-624`). For RouteIQ the
validator runs RouteIQ's own config parser (`config_loader` / settings validation)
in a Python 3.13 Lambda (`:1171`, 15s/256MB) and **rejects on any parse error,
inline secret, or guardrail-gating violation**. A bad config never becomes the
deployed version — the deployment fails closed.

### Day-2 GitOps path (`config_state_construct.py:929-1108`)

Operators commit `config.yaml` to the source bucket; a CodePipeline (Source →
[Approve, prod only] → Deploy) assumes a narrow deployer role and runs
`appconfig create-hosted-configuration-version` + `start-deployment`
(`:1022-1038`). The deployer role holds exactly five AppConfig actions and an
explicit **deny** of `Update/DeleteConfigurationProfile` so the validator can
never be stripped (`:711-819`). Mutation of the profile fires an EventBridge →
SNS alarm (`:880-927`).

### RouteIQ runtime retrieval

Add `config_source: appconfig` to `RouteIQConfigSync` settings (alongside
`s3`/`gcs`, `settings.py:606-621`). RouteIQ retrieves config from the
**env-scoped configuration ARN** the construct exports
(`config_state_construct.py:268-276`) via the AppConfig polling flow
(`GetLatestConfiguration` / the AppConfig agent sidecar). `ConfigSyncManager`'s
ETag-change detection (`config_sync.py:120,166`) is replaced by AppConfig's
own version token; `HotReloadManager.reload_config()` still fires the in-process
router rebuild (`hot_reload.py:108`) on a new version. The local-file source
remains the fallback when AppConfig is not configured.

## Consequences

### Positive

- **Validation before deploy.** The Lambda validator rejects bad/secret-bearing
  configs at deploy time, not at the next router parse. This is the single
  load-bearing win over ETag polling.
- **Staged rollout + bake.** Linear-20%-over-12-min + 5-min bake means a bad
  config never flips 100% of pods at once; AppConfig can auto-rollback on a
  CloudWatch alarm during the bake.
- **Audit + approval.** GitOps pipeline with a prod approval gate and an SNS
  alarm on any attempt to remove the validator.
- **Reuses RouteIQ's own machinery.** No new in-process reload path — the
  existing `HotReloadManager` callback fires on the new version; the validator
  Lambda reuses RouteIQ's config parser.

### Negative

- **New control plane to provision.** Requires the AppConfig resource graph in
  IaC (this is what RouteIQ lacks today; see ADR-0030 for the substrate).
- **Validator must track the schema.** The validator Lambda packages RouteIQ's
  config parser and must be redeployed when the config contract changes.
- **AppConfig agent dependency.** The lowest-latency retrieval path uses the
  AppConfig agent/extension sidecar; without it RouteIQ polls
  `GetLatestConfiguration` directly (slightly higher latency, still fine for a
  config-reload cadence).

## Alternatives Considered

### Alternative A: Keep S3 + ETag polling (status quo)

- **Pros**: Already implemented; zero new infra.
- **Cons**: No validation gate, no staged rollout, no approval/audit. A bad
  config reaches every pod in one poll interval.
- **Rejected**: The three weaknesses above are exactly what AppConfig closes.

### Alternative B: S3 + a separate validation Lambda on `s3:ObjectCreated`

- **Pros**: Keeps S3 as the store; adds a validation step.
- **Cons**: Validation is *after* the object is already the authoritative ETag
  target — RouteIQ would need a second "approved-objects" bucket and custom
  promotion logic. AppConfig's hosted store + LAMBDA validator + linear
  deployment is that machinery, managed.
- **Rejected**: Re-implements AppConfig poorly.

### Alternative C: Git-pull sidecar (ArgoCD/Flux-style) writing the ConfigMap

- **Pros**: Familiar GitOps; works for the Helm chart's `configmap.yaml`.
- **Cons**: No content validation, no bake/rollback semantics, and ConfigMap
  updates do not themselves trigger RouteIQ's in-process reload without a
  watcher. Orthogonal to *content* safety.
- **Rejected**: Solves delivery, not validation/rollout.

## References

- `cdk/lib/config_state_construct.py` (vllm-sr-on-aws) — AppConfig
  Application/Environment/ConfigurationProfile/DeploymentStrategy/
  HostedConfigurationVersion/Deployment (`:551-709`), LAMBDA validator
  (`:611-636,1110-1239`), GitOps pipeline + deployer role (`:711-819,929-1108`),
  runtime profile ARN (`:268-276`)
- `../architecture/aws-rearchitecture/vllmsr-patterns.md` — "config_state_construct.py: AppConfig GitOps
  (Application/Environment/ConfigurationProfile/DeploymentStrategy/
  HostedConfigurationVersion/Deployment + validator Lambda + S3 + SNS)"
- `src/litellm_llmrouter/config_sync.py` — current S3/ETag sync (to be extended)
- `src/litellm_llmrouter/hot_reload.py` — in-process reload callback (reused)
- `src/litellm_llmrouter/config_loader.py` — S3 download helper (validator reuse)
- [ADR-0030: EKS Auto Mode + IRSA Deployment Substrate](0030-eks-auto-mode-irsa-substrate.md)
