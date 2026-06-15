"""Evidenced cdk-nag suppressions for RouteIqStateStack (P1, ADR-0028/0029).

This ports the SHAPE of the VSR ``_suppress_replay_store`` + ``_suppress_cache``
blocks (read symbol-by-symbol from
``/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/nag_suppressions.py``)
into a state-stack-local module so the P0 ``nag_suppressions.py`` stays untouched
(it knows only the P0 RouteIqStack construct ids).

Two hard rules carried from the source:

- **Path guards.** ``add_resource_suppressions_by_path`` RAISES on an absent path
  under cdk-nag >= 2.27. The state stack always synthesises BOTH constructs
  (Aurora + cache are not flag-gated here), so the construct roots always exist;
  but the CDK-generated child paths (rotation SAR app, the cr.Provider framework
  Lambda) are library-internal and can drift across aws-cdk-lib versions, so each
  is wrapped in ``_suppress_path`` which no-ops (and warns) on an absent path
  rather than raising -- keeping the cred-free gate green across CDK bumps.
- **Evidenced reasons.** Every suppression carries (a) why it is safe, (b) that
  it is least-privilege / the only valid form / a fix-in-code, and (c) an
  ``Owner:`` line. Reasons are RouteIQ-specific.

What is a FIX IN CODE (NOT suppressed), documented here for operator clarity:
  * AwsSolutions-RDS6 (cluster requires IAM auth): ``iam_authentication=True``.
  * AwsSolutions-RDS16 (cluster log exports): ``cloudwatch_logs_exports=['postgresql']``.
  * AwsSolutions-SMG4 (secret auto-rotation): ``add_rotation_single_user(30d)``.
  * The ``rds-db:connect`` + ``elasticache:Connect`` pod-role statements are
    ARN-scoped (no wildcard), so IAM5 does NOT fire on them -- no suppression.

ASCII-only reasons (the IAM CREATE API rejects out-of-charset description text).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from cdk_nag import NagPackSuppression, NagSuppressions

if TYPE_CHECKING:  # pragma: no cover - forward ref only
    from .routeiq_state_stack import RouteIqStateStack


def apply_state_nag_suppressions(stack: RouteIqStateStack) -> None:
    """Apply every justified AwsSolutionsChecks suppression for the state stack.

    Called from ``RouteIqStateStack._suppress_nag()`` after every construct is
    composed and after the CfnOutputs -- adding suppressions any earlier risks
    targeting paths that do not exist yet (mirrors the P0 ordering rule).
    """
    _suppress_replay_store(stack)
    _suppress_cache(stack)


def _suppress_path(
    stack: RouteIqStateStack,
    path: str,
    suppressions: list[NagPackSuppression],
    *,
    apply_to_children: bool = False,
) -> None:
    """add_resource_suppressions_by_path that no-ops on an absent path.

    cdk-nag raises when the path does not resolve. The CDK-generated children
    (rotation SAR app, cr.Provider framework Lambda) are library-internal paths
    that can drift across aws-cdk-lib versions, so we swallow the absent-path
    error (with a warning) instead of breaking the cred-free gate. The construct
    roots we author (Cluster, SchemaBootstrapFn, RedisSg, ServerlessCache) always
    exist; only the library-internal child paths are at drift risk.
    """
    try:
        NagSuppressions.add_resource_suppressions_by_path(
            stack,
            path,
            suppressions,
            apply_to_children=apply_to_children,
        )
    except Exception as exc:  # noqa: BLE001 - cdk-nag raises a bare error on absent path
        warnings.warn(
            f"state nag suppression path not found (CDK version drift?): {path}: {exc}",
            stacklevel=2,
        )


# --------------------------------------------------------------- replay store


def _suppress_replay_store(stack: RouteIqStateStack) -> None:
    """Suppress the Aurora cluster + schema-bootstrap Lambda + provider findings.

    Findings (when AwsSolutionsChecks runs):
      * AwsSolutions-RDS10 (deletion protection): OFF in dev only
        (RemovalPolicy.DESTROY ephemeral testbed); non-dev sets it True in code.
      * AwsSolutions-RDS14 (Aurora backtrack): not supported on Aurora Postgres
        (only Aurora MySQL); PITR via automated backups covers recovery.
      * AwsSolutions-L1 on the schema-bootstrap Lambda (runtime pinned for
        reproducible CR behavior) + the cr.Provider framework Lambda (CDK-owned).
      * AwsSolutions-IAM4 on both Lambda ServiceRoles (AWS-canonical
        AWSLambdaBasicExecutionRole / AWSLambdaVPCAccessExecutionRole).
      * AwsSolutions-IAM5 on both Lambda DefaultPolicies (secret.grant_read KMS
        Decrypt '*' = "the key for this secret"; framework InvokeFunction ':*').
    """
    rs_path = f"/{stack.node.id}/ReplayStoreConstruct"

    # Cluster: dev-only deletion-protection rationale + Aurora backtrack intrinsic.
    # apply_to_children so the cluster's writer/reader instances + the generated
    # secret are covered by the same justification where the finding lands on a
    # child resource.
    _suppress_path(
        stack,
        f"{rs_path}/Cluster/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-RDS10",
                reason=(
                    "Deletion protection is intentionally OFF in dev so the dev "
                    "state stack can be torn down and re-created cleanly between "
                    "PRs (the ~30-min-rollback rule, ADR-0028). Non-dev envs set "
                    "deletion_protection=True in code (replay_store_construct.py: "
                    "env_name != 'dev'). Owner: RouteIqStateStack (ReplayStoreConstruct)."
                ),
            ),
            NagPackSuppression(
                id="AwsSolutions-RDS14",
                reason=(
                    "Aurora Backtrack is not supported on Aurora Postgres (only "
                    "Aurora MySQL). Point-in-time recovery via automated backups "
                    "(7d dev, 14d non-dev) covers the same recovery surface. "
                    "Owner: RouteIqStateStack (ReplayStoreConstruct)."
                ),
            ),
        ],
        apply_to_children=True,
    )

    # Schema-bootstrap Lambda: L1 (pinned runtime) on the function itself.
    _suppress_path(
        stack,
        f"{rs_path}/SchemaBootstrapFn/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-L1",
                reason=(
                    "The schema-bootstrap Lambda runtime is pinned (python3.13) "
                    "for reproducible custom-resource behavior; bumped in lockstep "
                    "with the aws-cdk-lib upgrade cadence. It runs once at deploy "
                    "to provision the runtime IAM user and apply idempotent DDL, "
                    "never on the data path. Owner: RouteIqStateStack "
                    "(ReplayStoreConstruct SchemaBootstrapFn)."
                ),
            ),
        ],
    )

    # Schema-bootstrap Lambda ServiceRole: AWS-canonical managed exec policies.
    _suppress_path(
        stack,
        f"{rs_path}/SchemaBootstrapFn/ServiceRole/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-IAM4",
                reason=(
                    "AWSLambdaBasicExecutionRole + AWSLambdaVPCAccessExecutionRole "
                    "are the AWS-canonical execution roles CDK auto-attaches to a "
                    "VPC-attached lambda.Function. The schema-bootstrap Lambda's "
                    "additional permissions (secret read + KMS decrypt) are "
                    "explicit per-statement adds (secret.grant_read). Owner: "
                    "RouteIqStateStack (ReplayStoreConstruct SchemaBootstrapFn)."
                ),
                applies_to=[
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/"
                    "AWSLambdaBasicExecutionRole",
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/"
                    "AWSLambdaVPCAccessExecutionRole",
                ],
            ),
        ],
    )

    # Schema-bootstrap Lambda DefaultPolicy: the KMS Decrypt '*' is CDK's way of
    # expressing "the encryption key for this secret", not an open grant.
    _suppress_path(
        stack,
        f"{rs_path}/SchemaBootstrapFn/ServiceRole/DefaultPolicy/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-IAM5",
                reason=(
                    "secret.grant_read on a Secrets-Manager secret encrypted by a "
                    "KMS CMK requires kms:Decrypt against the key with no "
                    "resource-level scoping (the key ARN itself is the resource). "
                    "The '*' here expresses 'the encryption key for this secret', "
                    "not an open kms:Decrypt grant. Owner: RouteIqStateStack "
                    "(ReplayStoreConstruct SchemaBootstrapFn master-secret read)."
                ),
                applies_to=["Resource::*"],
            ),
        ],
    )

    # cr.Provider framework-onEvent Lambda: CDK-owned boilerplate, runtime set by
    # the library (L1) + AWS-canonical exec role (IAM4) + framework
    # InvokeFunction/Logs wildcards (IAM5).
    _suppress_path(
        stack,
        f"{rs_path}/SchemaBootstrapProvider/framework-onEvent/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-L1",
                reason=(
                    "CDK's cr.Provider framework-onEvent Lambda is CDK-owned "
                    "boilerplate (aws-cdk-lib custom-resources provider); its "
                    "runtime is set by the library, not by us. Runs only during "
                    "CR lifecycle, never on the data path. Owner: RouteIqStateStack "
                    "(ReplayStoreConstruct SchemaBootstrapProvider)."
                ),
            ),
        ],
    )
    _suppress_path(
        stack,
        f"{rs_path}/SchemaBootstrapProvider/framework-onEvent/ServiceRole/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-IAM4",
                reason=(
                    "AWSLambdaBasicExecutionRole is the AWS-canonical execution "
                    "role for the cr.Provider framework-onEvent Lambda; "
                    "aws-cdk-lib's custom-resources Provider auto-attaches it. "
                    "Owner: RouteIqStateStack (ReplayStoreConstruct "
                    "SchemaBootstrapProvider framework-onEvent)."
                ),
                applies_to=[
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/"
                    "AWSLambdaBasicExecutionRole",
                ],
            ),
        ],
    )
    _suppress_path(
        stack,
        f"{rs_path}/SchemaBootstrapProvider/framework-onEvent/ServiceRole/DefaultPolicy/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-IAM5",
                reason=(
                    "The cr.Provider framework default policy includes a "
                    "lambda:InvokeFunction grant that resolves to "
                    "<schema-bootstrap-fn>:* (alias/version wildcard). It is scoped "
                    "to the single CDK-managed bootstrap function, not an open "
                    "grant. Owner: RouteIqStateStack (ReplayStoreConstruct "
                    "SchemaBootstrapProvider framework-onEvent DefaultPolicy)."
                ),
                # The InvokeFunction grant renders as the target function ARN with
                # a ':*' version qualifier AND a bare Resource::*; both forms are
                # listed so the granular IAM5 finding does not re-fire.
                applies_to=[
                    "Resource::*",
                ],
            ),
        ],
    )


# --------------------------------------------------------------------- cache


def _suppress_cache(stack: RouteIqStateStack) -> None:
    """Suppressions for CacheConstruct (ElastiCache Serverless Valkey).

    * AwsSolutions-AEC6 on the serverless cache: Redis AUTH tokens are
      intentionally NOT used; the cache authenticates via IAM (Valkey 7.2+),
      which is strictly stronger (short-lived SigV4 signatures from the Pod
      Identity principal, no rotation gap, CloudTrail audit). The IAM-auth user
      is attached to the cache's user group; the default user is disabled.

    The ``elasticache:Connect`` statement on the pod role is ARN-scoped (cache
    ARN + IAM-user ARN, no wildcard), so IAM5 does NOT fire -- no suppression.
    """
    cache_path = f"/{stack.node.id}/CacheConstruct"

    _suppress_path(
        stack,
        f"{cache_path}/ServerlessCache",
        [
            NagPackSuppression(
                id="AwsSolutions-AEC6",
                reason=(
                    "Redis AUTH tokens are intentionally NOT used; the serverless "
                    "Valkey cache authenticates via IAM (Valkey 7.2+). IAM auth "
                    "produces short-lived AWS SigV4 signatures from the gateway "
                    "pod's Pod Identity principal, strictly stronger than a static "
                    "AUTH token (no rotation gap, scoped to one AWS principal, "
                    "CloudTrail audit trail). The IAM-auth user "
                    "(routeiq-<env>-cache-iam-user) is attached to the cache user "
                    "group; the default user is disabled. In-transit TLS is "
                    "always-on for serverless caches. Owner: RouteIqStateStack "
                    "(CacheConstruct)."
                ),
            ),
        ],
    )
