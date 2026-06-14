# Discover: VSR CDK scaffold → proposed RouteIQ `deploy/cdk/` tree + cred-free gate

> Scope: deep-read the `vllm-sr-on-aws` (VSR) CDK scaffold that the RouteIQ P0
> CDK foundation must MIRROR, then propose RouteIQ's `deploy/cdk/` tree and the
> cred-free test gate. The task's `undefined/` placeholders resolve to the VSR
> port-reference repo at **`/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/`**
> (named in the pivot handoff §11 as the PORT REFERENCE). Nothing is written under
> `reference/` (read-only) and no `aws`/`cdk`/`kubectl` was run.

---

## 1. What the VSR scaffold actually is (the 5 files, read in full)

### 1.1 `cdk/app.py` (323 lines) — the context-key reading pattern + aspect

Three helpers at module top are the **exact pattern RouteIQ must copy** (they
defuse a real CLI-string footgun):

- **`_ctx(app, key, default)`** (`app.py:20`) — `app.node.try_get_context(key)`,
  returns `default` only when the value is `None`. The simple read.
- **`_bool_ctx(app, key, default)`** (`app.py:26`) — the load-bearing one. A
  `cdk synth --context key=false` passes the **string** `"false"`, and
  `bool("false")` is `True` in Python, so a naive `bool(_ctx(...))` silently keeps
  the default. This helper returns native bools as-is and parses CLI strings via
  `str(value).strip().lower() in ("true","1","yes")`. Every boolean flag in
  `app.py` (`enable_amg`, `enable_https_listener`, `enable_eks`, …) goes through it.
- **`_split_csv_or_list(raw)`** (`app.py:42`) — normalises a value that may be a
  comma-separated string (`--context key=a,b`) OR a JSON list (from `cdk.json`)
  into a clean `list[str]`. Used for `capacity_account_ids` etc.

**Stack instantiation** (`app.py:54` `main()`): build one `cdk.App()`, read every
context key under the `vllm_sr:` prefix into locals, build a `cdk.Environment`
from `CDK_DEFAULT_ACCOUNT` / `CDK_DEFAULT_REGION` env (`app.py:182`), then
construct `VllmSrStack(app, f"VllmSrStack-{env_name}", env=env, …)` passing every
flag as a kwarg. A SECOND stack (`VllmSrEksStack-{env_name}`) is instantiated
**only inside `if _bool_ctx(app, "vllm_sr:enable_eks", False):`** — a flag-gated,
own-VPC parallel substrate that shares nothing at the CFN level (`app.py:231`).

**The AwsSolutionsChecks aspect** is added to the whole app **after** both stacks
are constructed and **before** `app.synth()`:

```python
cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
app.synth()
```
(`app.py:317-319`). This is what makes `cdk synth` itself the nag gate.

### 1.2 `cdk/cdk.json` (61 lines) — the `vllm_sr:` context-key convention

- `"app": "python3 app.py"` (`cdk.json:2`).
- A flat `"context": { … }` block where EVERY key is prefixed `vllm_sr:`
  (`vllm_sr:env`, `vllm_sr:vpc_cidr`, `vllm_sr:enable_eks`, `vllm_sr:eks_vpc_cidr`,
  `vllm_sr:sr_image_tag`, …). These are the committed DEFAULTS; `--context` and
  `cdk.context.json` override them. Values are real JSON (native bools/ints/lists),
  which is exactly why `_bool_ctx` is needed for the `--context` string path.

### 1.3 `cdk/requirements.txt` (15 lines) — the pins

```
aws-cdk-lib>=2.150.0,<3.0.0
cdk-monitoring-constructs>=10.0.0,<11.0.0
cdk-nag>=2.27.0,<3.0.0
constructs>=10.0.0,<11.0.0
pytest>=7.4.0,<9.0.0
cdklabs.ecs-codedeploy==0.0.441   # VSR-only blue/green; RouteIQ drops this
```
The task's required pins (`aws-cdk-lib>=2.150,<3`; `cdk-nag>=2.27,<3`; `constructs`;
`pytest`) all match. The `cdklabs.ecs-codedeploy==0.0.441` pin is ECS-blue/green
specific (VSR) and **does not** carry to RouteIQ (RouteIQ is EKS + Helm, no ECS
CodeDeploy). `cdk-monitoring-constructs` is optional for RouteIQ P0.

### 1.4 `cdk/lib/vllm_sr_eks_stack.py` (737 lines) — the IRSA factory CALL SITES + CfnOutputs

This is the **template RouteIqStack mirrors most closely** (own VPC + per-workload
IRSA + CfnOutputs + evidenced nag in one stack). Structure:

1. **Stack ctor** (`:42`) takes typed kwargs (`env_name`, `vpc_cidr` default
   `10.30.0.0/16`, `nat_gateways`, `admin_principal_arns`, flag bools, …) and
   applies `Tags.of(self).add("vllm-sr:env", …)` + conditional cost tags
   (`:78-84`, only emitted when supplied so default synth stays byte-stable).
2. **Own VPC** (`:92`) — `ec2.Vpc` with `max_azs=2`, NAT, `flow_logs={…ALL…}`
   (satisfies `AwsSolutions-VPC7`), and `public` + `PRIVATE_WITH_EGRESS` subnet
   tiers. Subnets tagged `kubernetes.io/role/elb=1` (public) and
   `kubernetes.io/role/internal-elb=1` (private) for the LB controller (`:122-125`).
3. **The cluster construct** (`:130`) — `EksClusterConstruct(self, "EksCluster", …)`.
4. **The IRSA factory CALL SITE** (`:146`, the one the task points at):
   ```python
   self.router_irsa_role = self.eks.irsa_role(
       "RouterIrsaRole",
       namespace="vllm-sr",
       service_account="vllm-sr-router",
       description=f"vLLM-SR {env_name} router pod IRSA - bedrock-mantle access",
   )
   self.router_irsa_role.add_to_policy(iam.PolicyStatement(sid="BedrockMantleInference", …))
   self.router_irsa_role.add_to_policy(iam.PolicyStatement(sid="BedrockMantleBearerToken",
       actions=["bedrock-mantle:CallWithBearerToken"], resources=["*"]))  # identity-level, * is the only valid form
   ```
   Two more IRSA call sites are flag-gated: `EaigGatewayRole` (`:199`, native
   Bedrock invoke) and `BearerMinterRole` (`:272`, cross-account `sts:AssumeRole`).
   **Pattern:** factory call → `.add_to_policy(...)` per grant → `CfnOutput` of the
   role ARN.
5. **CfnOutputs** — `RouterIrsaRoleArn` (`:170`, "Annotate the SA with
   `eks.amazonaws.com/role-arn`"), `EksVpcId` (`:322`), `NodeRoleName` (`:332`,
   for the NodeClass `role:` field), plus flag-gated outputs (`EaigGatewayRoleArn`,
   `BearerMinterRoleArn`, `CapacityRoleArn<acct>`, `ReplayStore*Endpoint`).
6. **`self._suppress_nag()`** called LAST in the ctor (`:541`).

The `irsa_role()` factory itself (`eks_cluster_construct.py:261`) is THE
load-bearing gotcha: it wraps the OIDC trust condition map in `CfnJson` (CFN map
keys must be literal at synth; an `Fn::GetAtt` token can't be a key), and uses
`self.oidc_provider_issuer` (ARN-derived, scheme-free) **NOT**
`oidc_issuer_url.replace("https://","")` (a silent no-op on an unresolved token →
`AssumeRoleWithWebIdentity` AccessDenied, root-caused live 2026-06-08). RouteIQ's
construct must reproduce this exactly (skill `cdk-gotchas`).

### 1.5 `cdk/lib/nag_suppressions.py` (2400 lines) — the evidenced-suppression SHAPE (don't copy all)

Two suppression idioms, both used:

- **Module-level `apply_nag_suppressions(stack)`** (`:29`) — one public entry that
  fans out to `_suppress_<construct>(stack)` helpers (`_suppress_network`,
  `_suppress_security`, `_suppress_replay_store`, …). Each helper calls
  `NagSuppressions.add_resource_suppressions_by_path(stack, f"/{stack_id}/<Construct>/<Resource>/Resource", [NagPackSuppression(id=…, reason=…)])`.
  Flag-gated families are guarded with `getattr(stack, "x", None)` because
  `add_..._by_path` **raises** on an absent path under cdk-nag >=2.27.
- **In-stack `self._suppress_nag()`** (the EKS stack, `:543`) — calls
  `NagSuppressions.add_resource_suppressions(self.eks.cluster_role, [{...}],
  apply_to_children=True)` against construct OBJECTS (not paths), and flag-gates
  each block with `if self.eaig_gateway_role is not None:` etc.

**The evidenced shape (the only thing to copy):** every suppression carries an
`id` (e.g. `AwsSolutions-IAM4`/`-IAM5`/`-EKS1`), an explicit `appliesTo` list
(specific managed-policy ARNs or `Resource::*`), and a `reason` that states (a)
WHY it's safe, (b) that it's least-privilege / the only valid form, and (c) an
`Owner:` line. Example (EKS stack `:553`): the 5 AWS-managed EKS Auto Mode
policies are suppressed for IAM4 with reason "Auto Mode REQUIRES these exact
AWS-published policies … not replaceable with a narrower custom policy … Owner:
EksClusterConstruct." **Do NOT copy the 2400-line file; copy the per-suppression
shape and write RouteIQ-specific reasons.**

### 1.6 The cred-free test layout (how all 3 test files synth WITHOUT creds)

The unifying trick: **`Template.from_stack(stack)` after passing an explicit
`env=cdk.Environment(account="123456789012", region="us-west-2")`** — a dummy but
concrete account/region. No AWS call, no creds, fully offline. The three files:

- **`tests/unit/test_eks_cluster.py`** (687 lines) — the per-construct unit test.
  A `@pytest.fixture template` builds `VllmSrEksStack(app, "…-dev",
  env=Environment(account="123456789012", region="us-west-2"), env_name="dev")`
  and returns `Template.from_stack(stack)`. Assertions use
  `template.has_resource_properties("AWS::EKS::Cluster", {...})`,
  `template.find_resources(...)`, `template.resource_count_is(...)`, and
  `Match.object_like/array_with` — NOT brittle full-count snapshots (mulch
  `cdk-resource-count-test-tripwire`). Flag-on variants build their own `app` +
  stack inside the test with the flag kwarg. Notable guards RouteIQ should clone:
  - `test_irsa_trust_keys_strip_https_scheme` (`:138`) — serialises the
    `Custom::AWSCDKCfnJson` TrustKeys value and asserts `":oidc-provider/"` IS
    present and `"https://"` is NOT (the `.replace` regression guard).
  - `test_iam_role_descriptions_are_ascii` (`:670`) — asserts every IAM role
    `Description` matches IAM's Latin-1 charset (an em-dash U+2014 passes synth
    but FAILS CREATE at the IAM API — a synth-green/deploy-red trap).
  - `test_eks_stack_has_no_unsuppressed_nag_errors` (`:512`) — builds the stack at
    its MAXIMAL flag surface, adds `Aspects.of(app).add(AwsSolutionsChecks())`,
    then `Annotations.from_stack(stack).find_error("*", Match.string_like_regexp(
    r"AwsSolutions-.*"))` and asserts empty.
- **`tests/unit/test_cdk_nag.py`** (104 lines) — the dedicated nag gate. Helper
  `_synth_with_nag(**kwargs)` builds the stack, adds the `AwsSolutionsChecks`
  aspect, and three tests assert NO surviving `AwsSolutions-*` errors (default +
  `enable_first_deploy_bake=True`) and no `CdkNagValidationFailure.*`. Failure
  messages render the offending `id` + `data[:200]` so they're actionable.
- **`tests/snapshot/test_template_snapshot.py`** (243 lines) — byte-stable template
  snapshots. `_synth_template(...)` → `Template.from_stack(stack).to_json()` →
  `_canonicalise` (`json.dumps(sort_keys=True, indent=2)`) → `_assert_snapshot`
  diffs against `__snapshots__/<name>.json`. Auto-write ONLY under
  `UPDATE_SNAPSHOTS=1`; a missing baseline is a LOUD failure (not a silent create
  — false-green hazard, seed 7d1e). An `autouse` fixture monkeypatches
  Docker-bundled-Lambda asset-path globals to a non-existent dir so synth takes
  the inline-placeholder branch and stays hermetic regardless of host Docker.

---

## 2. Proposed RouteIQ `deploy/cdk/` tree

Co-locate under the existing `deploy/` (next to `deploy/charts/routeiq-gateway/`),
per the handoff §10 recommendation. Naming convention: **`routeiq:` context keys**
(mirrors `vllm_sr:`), stack id **`RouteIqStack-<env>`**, construct files
`*_construct.py`, package import path `lib.*`.

```
deploy/cdk/
├── app.py                       # _ctx/_bool_ctx/_split_csv_or_list (verbatim shape) +
│                                #   RouteIqStack(app, f"RouteIqStack-{env}", env=…, **flags)
│                                #   + cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
│                                #   + app.synth()
├── cdk.json                     # "app": "python3 app.py" + "context": { "routeiq:*": … }
├── requirements.txt             # aws-cdk-lib>=2.150,<3; cdk-nag>=2.27,<3; constructs>=10,<11; pytest>=7.4,<9
├── pyproject.toml               # OPTIONAL — only if the cdk subtree wants its own ruff/mypy cfg
├── README.md                    # how to synth + the routeiq: context keys + operator deploy steps
├── lib/
│   ├── __init__.py
│   ├── routeiq_stack.py         # the composition root (mirrors vllm_sr_eks_stack.py):
│   │                            #   own VPC (routeiq:vpc_cidr, default 10.40.0.0/16 — distinct from
│   │                            #   VSR 10.20/10.30) + flow logs + elb subnet tags;
│   │                            #   EksClusterConstruct; the IRSA factory CALL SITE for ONE pod role
│   │                            #   (routeiq-gateway SA in the chart's namespace) with the P0 grant
│   │                            #   set (aps:RemoteWrite, rds-db:connect, elasticache:Connect,
│   │                            #   AppConfig read, bedrock:Invoke* + Converse); EcrConstruct (GHCR
│   │                            #   pull-through cache); CfnOutputs (GatewayIrsaRoleArn for the chart
│   │                            #   values.yaml:206-207 SA annotation, VpcId, NodeRoleName,
│   │                            #   EcrGhcrPrefix); self._suppress_nag() LAST.
│   ├── eks_cluster_construct.py # PORT of VSR's — L1 CfnCluster + 3 Auto-Mode blocks + OIDC provider
│   │                            #   + irsa_role() factory (CfnJson trust-key, ARN-derived scheme-free
│   │                            #   issuer — NOT .replace) + add_admin_access_entry() +
│   │                            #   enable_container_insights(). Strip VSR-specific bits.
│   ├── ecr_construct.py         # PORT — CfnPullThroughCacheRule for ghcr.io (RouteIQ publishes to
│   │                            #   ghcr.io/baladithyab/routeiq) + immutable scan-on-push repos +
│   │                            #   RepositoryCreationTemplate. enable_ghcr_ptc flag.
│   ├── network_construct.py     # OPTIONAL P0 split — VPC + SGs + interface endpoints (incl.
│   │                            #   BEDROCK_RUNTIME) + S3 gateway endpoint. May inline in routeiq_stack
│   │                            #   for the single-pod P0; split when P1 state lands.
│   └── nag_suppressions.py      # apply_nag_suppressions(stack) + _suppress_<construct>(stack) helpers,
│                                #   evidenced shape only (id + appliesTo + reason w/ Owner:), flag-gated
│                                #   getattr guards. RouteIQ-specific reasons, NOT a copy of VSR's 2400 lines.
└── tests/
    ├── __init__.py
    ├── unit/
    │   ├── __init__.py
    │   ├── test_eks_cluster.py  # per-construct: Template.from_stack with dummy env; Auto-Mode 3 blocks;
    │   │                        #   OIDC provider count; the IRSA trust-key https://-strip regression
    │   │                        #   guard; the IAM-role-description-ASCII guard; flag-on/off byte-stability.
    │   ├── test_ecr.py          # GHCR PullThroughCacheRule + immutable scan-on-push repos.
    │   ├── test_routeiq_stack.py# the gateway IRSA role has the P0 grant set; CfnOutputs exist
    │   │                        #   (GatewayIrsaRoleArn etc.); VPC CIDR distinct.
    │   └── test_cdk_nag.py      # _synth_with_nag(**flags) + AwsSolutionsChecks aspect; assert NO
    │                            #   AwsSolutions-* errors AND no CdkNagValidationFailure (default +
    │                            #   maximal flag surface).
    └── snapshot/
        ├── __init__.py
        ├── test_template_snapshot.py  # _synth_template → to_json → canonicalise → _assert_snapshot vs
        │                              #   __snapshots__/<name>.json; UPDATE_SNAPSHOTS=1 to refresh;
        │                              #   missing baseline = loud failure (no silent create).
        └── __snapshots__/
            └── dev.json               # committed flag-off baseline (generated with UPDATE_SNAPSHOTS=1).
```

### Notable adaptations from VSR → RouteIQ
- **Context prefix:** `vllm_sr:` → `routeiq:` everywhere (`app.py`, `cdk.json`,
  test contexts). Stack id `VllmSrEksStack-<env>` → `RouteIqStack-<env>`.
- **One pod, one IRSA role.** RouteIQ is a single stateless gateway pod, so P0
  needs ONE `irsa_role()` call site (SA = the chart's `serviceaccount.yaml`, wired
  to `values.yaml:206-207`'s `eks.amazonaws.com/role-arn`), with the P0 grant set
  (handoff §10.3): `aps:RemoteWrite` + `rds-db:connect` + `elasticache:Connect` +
  AppConfig read + `bedrock:InvokeModel`/`Converse`. The VSR multi-role
  (router/EAIG/minter) split is NOT needed at P0; defer cross-account capacity.
- **EKS not ECS.** Drop `cdklabs.ecs-codedeploy`, `compute_construct.py`,
  `vllm_sr_stack.py` (the ECS stack). RouteIQ's app layer ships via the existing
  Helm chart + kubectl, NOT CDK `KubernetesManifest`.
- **VPC CIDR** picks a distinct `10.40.0.0/16` (VSR ECS=10.20, VSR EKS=10.30) so a
  future peering stays overlap-free.
- **Carry the synth-time guards** the VSR tests encode: ASCII-only IAM
  descriptions (em-dash → IAM CREATE failure), the IRSA `https://`-strip guard,
  and `has_resource_properties`/`find_resources` over brittle counts.

---

## 3. The cred-free gate command

VSR runs its CDK tests through its own venv/pytest. RouteIQ uses **uv** (`uv run`),
floor Python 3.12, `pytest` already a dev dep. Because the tests call
`Template.from_stack(...)` with a dummy `env=Environment(account="123456789012",
region="us-west-2")`, they synth fully offline — no creds, no `cdk` CLI, no
network. The gate:

```bash
uv run pytest deploy/cdk/tests/
```

This runs the per-construct unit asserts + the `cdk-nag` AwsSolutions gate + the
template snapshot diff, all credential-free, suitable for the pre-push hook / CI.
(Operator-only, NOT part of the gate: `cdk synth` / `cdk deploy` / `kubectl` /
`helm` — those are deploy-time and creds-gated per the standing constraints.)

Notes for wiring:
- `aws-cdk-lib` + `cdk-nag` + `constructs` must be installed in the uv env. Either
  add them to a `cdk` extra in the root `pyproject.toml` and `uv sync --extra cdk`,
  or keep `deploy/cdk/requirements.txt` and `uv pip install -r` into the env. A
  root-`pyproject` extra is the more uv-native path and keeps one lockfile.
- The snapshot baseline `deploy/cdk/tests/snapshot/__snapshots__/dev.json` must be
  generated once with `UPDATE_SNAPSHOTS=1 uv run pytest deploy/cdk/tests/snapshot/`
  and committed; thereafter a missing baseline is a loud failure by design.

---

## 4. Source-file references (VSR port reference — symbol-stable, offsets as of 2026-06-14)

| Pattern | VSR file:symbol |
|---|---|
| Context helpers + stack instantiation + aspect | `cdk/app.py:20` `_ctx` / `:26` `_bool_ctx` / `:42` `_split_csv_or_list` / `:54` `main` / `:317` `AwsSolutionsChecks` |
| `vllm_sr:` context convention | `cdk/cdk.json:2-60` |
| Pins | `cdk/requirements.txt:1-5` |
| Own VPC + IRSA call sites + CfnOutputs + in-stack nag | `cdk/lib/vllm_sr_eks_stack.py:42` ctor / `:92` VPC / `:146` `irsa_role` call / `:170` `CfnOutput` / `:543` `_suppress_nag` |
| IRSA factory (CfnJson trust-key, scheme-free issuer) | `cdk/lib/eks_cluster_construct.py:261` `irsa_role` / `:151` `CfnCluster` / `:219` OIDC provider |
| Evidenced suppression shape | `cdk/lib/nag_suppressions.py:29` `apply_nag_suppressions` / `:86` `_suppress_network` (representative `add_resource_suppressions_by_path` + `NagPackSuppression(id,reason)`) |
| GHCR pull-through cache | `cdk/lib/ecr_construct.py:57` `EcrConstruct` / `:85` PTC rules |
| Cred-free unit synth | `cdk/tests/unit/test_eks_cluster.py:33` fixture / `:138` https-strip guard / `:512` nag-at-max-surface / `:670` ASCII-desc guard |
| Cred-free nag gate | `cdk/tests/unit/test_cdk_nag.py:23` `_synth_with_nag` / `:37` no-AwsSolutions-errors |
| Snapshot gate | `cdk/tests/snapshot/test_template_snapshot.py:113` `_synth_template` / `:140` `_assert_snapshot` (UPDATE_SNAPSHOTS, loud-on-missing) |
