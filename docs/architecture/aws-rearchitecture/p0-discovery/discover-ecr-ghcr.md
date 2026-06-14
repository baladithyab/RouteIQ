# Discovery: ECR construct + GHCR pull-through cache

> **Status**: Discovery / pre-build. **Date**: 2026-06-14.
> **Scope**: `ecr_construct.py` (P0 CDK port) and how an ECR **pull-through cache rule**
> bridges RouteIQ's existing `ghcr.io/baladithyab/routeiq` publish pipeline onto AWS with
> zero image-pipeline rework.

## TL;DR — the file does not exist yet

**There is no `ecr_construct.py` in this repo.** The task path
`undefined/lib/ecr_construct.py` resolves to nothing: RouteIQ is **Helm-only on AWS with
zero provisioning IaC** — the only `app.py`/Python "construct"-named file in the tree is
`src/litellm_llmrouter/gateway/app.py` (the FastAPI factory), not a CDK construct. There
is no `cdk.json`, no `cdk/` tree, no `*.tf`, no CDK app.

`ecr_construct.py` is a **planned P0 deliverable** — a **PORT** from
`vllm-sr-on-aws/cdk/lib/ecr_construct.py`. It is referenced only in the AWS
re-architecture docs:

- `docs/architecture/aws-rearchitecture/30-migration-roadmap.md:47-49` — P0 ECR port.
- `docs/architecture/aws-rearchitecture/10-aws-native-target-architecture.md:145` —
  "**7 ECR repos** via `ecr_construct.py` (immutable, scan-on-push, GHCR pull-through cache)."
- `docs/architecture/aws-rearchitecture/vllmsr-patterns.md:33` — source inventory row:
  "7 immutable + scan-on-push repos + pull-through-cache (GHCR / DockerHub)."
- `docs/handoffs/2026-06-14-1211-pivot-to-routeiq-on-aws.md:114,183,209,295`.

So the `__init__` signature, the repo-creation calls, and the pull-through wiring below
are the **target design the port must implement** (grounded in the vllm-sr pattern + the
authoritative AWS ECR docs), NOT a read of existing source. Every AWS-mechanism claim is
cited to AWS docs. The vllm-sr internals (exact param names) are NOT in this repo and are
therefore reconstructed from the doc inventory, not verbatim.

---

## 1. Target `EcrConstruct.__init__` signature (to be ported)

The repo does **not** contain the source, so the exact vllm-sr signature cannot be quoted.
Per the CDK construct convention + the doc inventory ("7 immutable + scan-on-push repos +
pull-through-cache"), the construct is a standard L3 wrapper:

```python
class EcrConstruct(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        repository_names: list[str],          # the 7 repo base names
        ghcr_credential_secret_arn: str | None = None,   # Secrets Manager ARN (ecr-pullthroughcache/*)
        dockerhub_credential_secret_arn: str | None = None,
        removal_policy: RemovalPolicy = RemovalPolicy.RETAIN,
        kms_key: kms.IKey | None = None,      # optional CMK for at-rest encryption of the repos
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        ...
```

**Caveat:** the parameter names (`repository_names`, `*_credential_secret_arn`, etc.) are a
faithful reconstruction, not a verbatim copy — the source lives in `vllm-sr-on-aws`, which
is not checked out here. RouteIQ only needs **one** image repo for the gateway
(`ghcr.io/baladithyab/routeiq`), so the "7 repos" count is a vllm-sr artifact the RouteIQ
port should parameterize down (handoff `:295`: "parameterized for RouteIQ's single …").

---

## 2. Immutable + scan-on-push repositories

In CDK these are two props on the `aws_ecr.Repository` L2 (verified against the
`aws-cdk-lib` aws_ecr README, v2.258.1, Python):

```python
import aws_cdk.aws_ecr as ecr

repo = ecr.Repository(
    self,
    "RouteIqGatewayRepo",
    repository_name=name,
    image_tag_mutability=ecr.TagMutability.IMMUTABLE,   # tags cannot be overwritten
    image_scan_on_push=True,                             # Basic scanning on every push
    encryption=ecr.RepositoryEncryption.KMS,             # if a CMK is supplied
    removal_policy=RemovalPolicy.RETAIN,
)
```

- **Immutable** — `image_tag_mutability=ecr.TagMutability.IMMUTABLE`. (CDK also now
  offers `IMMUTABLE_WITH_EXCLUSION` + `ImageTagMutabilityExclusionFilter.wildcard()` if a
  mutable `latest`-style tag is ever needed; not required for RouteIQ's pinned tag
  `1.0.0-rc1` / digest workflow.)
- **Scan-on-push** — `image_scan_on_push=True` (Basic scanning). Enhanced scanning is a
  registry-level setting, not a per-repo prop.
- Loop over `repository_names` to create N repos with identical settings.

**Interaction with RouteIQ's chart:** `values.yaml:32-39` already supports an immutable
**digest** reference (`image.digest`, "takes precedence over tag") — so immutable-tag ECR
repos are consistent with how the chart already wants to pin images.

---

## 3. The GHCR / DockerHub pull-through cache (the load-bearing part)

### 3.1 CDK / CloudFormation resource

There is no L2 for pull-through cache rules — you use the **L1**
`aws_cdk.aws_ecr.CfnPullThroughCacheRule`, which maps 1:1 to the CloudFormation resource
`AWS::ECR::PullThroughCacheRule`. Its properties (CFN TemplateReference, verified):

| Property | Type | Notes |
|---|---|---|
| `EcrRepositoryPrefix` | String | Destination namespace prefix in your private registry (e.g. `ghcr`). Pattern `^([a-z0-9]+(...))$`, len 2–30. |
| `UpstreamRegistry` | String | Enum. For GHCR: **`github-container-registry`**. For DockerHub: `docker-hub`. (Allowed: `ecr \| ecr-public \| quay \| k8s \| docker-hub \| github-container-registry \| azure-container-registry \| gitlab-container-registry \| chainguard`.) |
| `UpstreamRegistryUrl` | String | For GHCR: **`ghcr.io`**. For DockerHub: `registry-1.docker.io`. |
| `CredentialArn` | String | Secrets Manager secret ARN. Pattern **enforces** `…:secret:ecr-pullthroughcache/…`. Required only for registries that need auth (GHCR + DockerHub do). |
| `CustomRoleArn` | String | Optional IAM role for the rule. |
| `UpstreamRepositoryPrefix` | String | Optional upstream-side prefix. |

CDK (Python L1) for the **GHCR** rule:

```python
ecr.CfnPullThroughCacheRule(
    self,
    "GhcrPullThroughCache",
    ecr_repository_prefix="ghcr",                 # cached repos land under ghcr/*
    upstream_registry="github-container-registry",
    upstream_registry_url="ghcr.io",
    credential_arn=ghcr_credential_secret_arn,    # arn:aws:secretsmanager:<region>:<acct>:secret:ecr-pullthroughcache/<name>
)
```

Equivalent AWS CLI (authoritative, from the ECR userguide "For GitHub Container Registry"):

```bash
aws ecr create-pull-through-cache-rule \
    --ecr-repository-prefix github \
    --upstream-registry-url ghcr.io \
    --credential-arn arn:aws:secretsmanager:us-east-2:111122223333:secret:ecr-pullthroughcache/example1234 \
    --region us-east-2
```

(DockerHub differs ONLY in `--upstream-registry-url registry-1.docker.io` and
`--ecr-repository-prefix docker-hub`.)

**Per-Region:** pull-through cache rules are created **separately for each Region** — the
construct must be instantiated in (or its rule replicated to) every Region the cluster
runs in. `CredentialArn` updates **require replacement** of the rule.

### 3.2 The Secrets Manager secret the operator MUST provision (authenticated GHCR pulls)

This is the one piece the operator provisions out-of-band before the rule will work:

- **Name**: MUST be prefixed **`ecr-pullthroughcache/`** (e.g.
  `ecr-pullthroughcache/routeiq-ghcr`). The CFN `CredentialArn` pattern hard-enforces this
  prefix, and the ECR console only lists secrets with it.
- **Same account + Region** as the pull-through cache rule.
- **Encryption key**: MUST be the **default `aws/secretsmanager` AWS-managed key**. ECR
  **does not support a customer-managed CMK** for the pull-through credential secret. (This
  is a documented hard constraint — do not point the secret at the construct's `kms_key`.)
- **Contents** — two key/value pairs (GHCR):
  ```json
  { "username": "<github-username>", "accessToken": "<GitHub PAT>" }
  ```
  The PAT (classic personal access token) needs **`read:packages`** scope to pull from
  GHCR. (DockerHub secret uses the same `username` / `accessToken` keys.)

The construct takes the secret **ARN** as a parameter; it does **not** create the secret
(the credential is a real GitHub PAT — keep it out of CDK/source per the repo's no-secrets
rule). Operator-gated.

### 3.3 IAM `ecr:` permissions required

Two distinct permission sets (verified against the ECR "IAM permissions required to sync"
page):

**(a) For the principal that CREATES the rule** (CDK deploy / CloudFormation exec role):
- `ecr:CreatePullThroughCacheRule` — **must** be granted via an **identity-based** IAM
  policy (cannot come from a registry resource policy).
- Plus `secretsmanager:GetSecretValue` on the credential secret (ECR validates it at
  rule-create time) and `secretsmanager:DescribeSecret`.

**(b) For the principal/registry that PULLS through (imports upstream images)** — i.e. the
node/pod pull identity, or a registry permissions policy:
- `ecr:BatchImportUpstreamImage` — retrieves the upstream image and imports it into the
  private repo. Grantable via identity policy OR the **private registry permissions
  policy** OR a repository resource policy.
- `ecr:CreateRepository` — **only** needed if the destination cache repo (`ghcr/…`)
  doesn't already exist; ECR auto-creates the repo on first pull. Skip it if the construct
  pre-creates the repos or a repo-creation template handles it.
- Plus the normal pull set: `ecr:GetAuthorizationToken` (registry-wide),
  `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, `ecr:BatchCheckLayerAvailability`.

On EKS this maps onto the node role / the pod's IRSA role (RouteIQ's single pod role per
roadmap `:45`).

---

## 4. How this bridges `ghcr.io/baladithyab/routeiq` with ZERO image-pipeline rework

RouteIQ publishes to `ghcr.io/baladithyab/routeiq` today (`values.yaml:33`, built by
`docker/Dockerfile`, multi-stage, `INSTALL_LLMROUTER`/`BUILD_UI`/`ROUTEIQ_EXTRAS`
build-args, non-root UID 1000). The CI keeps pushing to GHCR **unchanged** — the pull-side
is rewritten, not the build-side:

1. CDK creates the GHCR pull-through cache rule (`§3.1`) + the `ecr-pullthroughcache/`
   credential secret is provisioned by the operator (`§3.2`).
2. The chart's image reference is repointed from
   `ghcr.io/baladithyab/routeiq` to the ECR pull-through path:
   ```
   <aws_account_id>.dkr.ecr.<region>.amazonaws.com/ghcr/baladithyab/routeiq:1.0.0-rc1
   ```
   (prefix = the rule's `EcrRepositoryPrefix`, then the upstream path `baladithyab/routeiq`).
   This is a `values.yaml` `image.repository` override at deploy time — **no Dockerfile,
   no GitHub Actions, no registry change.**
3. On first pull, ECR transparently fetches the image from GHCR (using the secret),
   imports it into a private repo `ghcr/baladithyab/routeiq` (auto-created), scans it on
   push, and serves it. Subsequent pulls are served from ECR (private, in-VPC via the ECR
   interface endpoints in `network_construct.py`, fast, no GHCR rate limits, scanned).

Net effect: the existing GHCR publish pipeline is the upstream of an AWS-side cache. The
only change is the chart's `image.repository` string + the IRSA/node pull permissions in
`§3.3`.

### Why NOT `imagePullSecrets` against GHCR directly

The chart already supports `global.imagePullSecrets` (`values.yaml:18`, wired via
`_helpers.tpl:123` → `deployment.yaml:33`), so a `dockerconfigjson` pull secret pointing at
GHCR is technically possible. The pull-through cache is preferred because it adds: private
in-VPC pulls (ECR interface endpoint), **scan-on-push on the cached copy**, immutable-tag
governance, and insulation from GHCR availability/rate limits — none of which a raw
`imagePullSecret` gives. (See the related failure note
`kms-pending-deletion-blocks-ecr-pull`: if a CMK encrypts the ECR repos and the key enters
pending-deletion, pulls fail — another reason the *credential* secret is locked to the
AWS-managed key per `§3.2`.)

---

## 5. Verification pointers (for the eventual port)

- `image_scan_on_push` / `image_tag_mutability` — `aws-cdk-lib` aws_ecr README (v2.258.1).
- `AWS::ECR::PullThroughCacheRule` props + `CredentialArn` `ecr-pullthroughcache/` pattern
  — CFN TemplateReference `aws-resource-ecr-pullthroughcacherule`.
- GHCR `ghcr.io` + `github-container-registry`, secret key names (`username`/`accessToken`),
  `aws/secretsmanager`-only encryption, `ecr-pullthroughcache/` prefix — ECR userguide
  `pull-through-cache-creating-rule` + `pull-through-cache-creating-secret`.
- `ecr:CreatePullThroughCacheRule` (identity-based), `ecr:BatchImportUpstreamImage`,
  `ecr:CreateRepository` — ECR userguide `pull-through-cache-iam`.
- RouteIQ chart facts — `deploy/charts/routeiq-gateway/values.yaml:18,32-39` and
  `templates/_helpers.tpl:123` / `templates/deployment.yaml:33`.
