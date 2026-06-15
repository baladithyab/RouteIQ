"""Evidenced cdk-nag suppressions for RouteIqStack (proposal P0 doc 31 section 13).

This ports the SHAPE of vllm-sr-on-aws/cdk/lib/nag_suppressions.py - NOT its
2400-line body. ``apply_nag_suppressions(stack)`` fans out to
``_suppress_<construct>(stack)`` helpers, each calling
``NagSuppressions.add_resource_suppressions_by_path`` against an explicit path.

Two hard rules carried from the source:

- **Path guards.** ``add_resource_suppressions_by_path`` RAISES on an absent path
  under cdk-nag >= 2.27. Every suppression whose target is flag-gated or only
  conditionally synthesised is guarded with ``getattr(stack, ..., None)`` (or a
  ``find_resources``-style presence check) so a flag-off synth does not blow up.
- **Evidenced reasons.** Every suppression carries (a) why it is safe, (b) that
  it is least-privilege / the only valid form, and (c) an ``Owner:`` line. The
  reasons are RouteIQ-specific - NOT reused from VSR.

The construct paths are derived from the construct ids the RouteIqStack
composition root uses: ``NetworkConstruct``, ``EksClusterConstruct``,
``EcrConstruct``, and the stack-level ``PodRole``. The CfnAddon /
CfnPodIdentityAssociation / CfnPullThroughCacheRule resources are L1 and carry no
IAM, so cdk-nag does not flag them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cdk_nag import NagPackSuppression, NagSuppressions

from .naming import routing_log_group_name

if TYPE_CHECKING:  # pragma: no cover - forward ref only
    from .routeiq_stack import RouteIqStack


def apply_nag_suppressions(stack: RouteIqStack) -> None:
    """Apply every justified ``AwsSolutionsChecks`` suppression for the stack.

    Called from ``RouteIqStack._suppress_nag()`` after every construct is
    composed and after the CfnOutputs - adding suppressions any earlier risks
    targeting paths that do not exist yet (proposal section 10).
    """
    _suppress_network(stack)
    _suppress_eks_cluster(stack)
    _suppress_ecr(stack)
    _suppress_pod_role(stack)


# --------------------------------------------------------------------- network


def _suppress_network(stack: RouteIqStack) -> None:
    """Suppress the public ALB ingress finding (PREP-ONLY at P0).

    VPC flow logs are ENABLED in NetworkConstruct (proposal section 8.5), so
    AwsSolutions-VPC7 does NOT fire and needs no suppression. The only network
    finding is AwsSolutions-EC23 on the ALB SG's ``443 from 0.0.0.0/0`` ingress,
    which is PREP-ONLY: the chart default is ClusterIP / ingress.enabled=false,
    so the Auto-Mode managed LB does not render and the SG backs nothing live
    until an operator flips ``service.type`` to LoadBalancer (proposal section
    8.3 / 11.1a).
    """
    stack_id = stack.node.id
    network = getattr(stack, "network", None)
    if network is None:
        return

    NagSuppressions.add_resource_suppressions_by_path(
        stack,
        f"/{stack_id}/NetworkConstruct/AlbSg/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-EC23",
                reason=(
                    "The AlbSg 443-from-0.0.0.0/0 ingress is PREP-ONLY at P0: the "
                    "RouteIQ chart default is service.type=ClusterIP / "
                    "ingress.enabled=false, so the EKS Auto-Mode managed LB does "
                    "NOT render and this SG backs nothing live. The public edge is "
                    "a deliberate P-later decision; the ingress is narrowable to an "
                    "org allowlist via a future context flag plus the chart "
                    "loadBalancerSourceRanges allowlist (the two halves of the same "
                    "future edge-lock). It is the only valid form for a "
                    "yet-unattached internet-facing ALB SG. "
                    "Owner: NetworkConstruct (alb_sg PREP-ONLY)."
                ),
            ),
        ],
    )

    # VpceSg ingress (443 from pod_sg) references an SG-to-SG peer that resolves
    # to an Fn::GetAtt CidrBlock intrinsic at synth time. cdk-nag's EC23 rule
    # cannot evaluate that intrinsic and emits a CdkNagValidationFailure WARNING
    # (a tooling artifact, NOT a real EC23 violation): the peer is the
    # stack-internal pod_sg, never 0.0.0.0/0. Same artifact the VSR source
    # suppresses on its VpceSg.
    NagSuppressions.add_resource_suppressions_by_path(
        stack,
        f"/{stack_id}/NetworkConstruct/VpceSg/Resource",
        [
            NagPackSuppression(
                id="CdkNagValidationFailure",
                reason=(
                    "CdkNagValidationFailure[AwsSolutions-EC23]: the VpceSg ingress "
                    "peer is the stack-internal pod_sg (an Fn::GetAtt CidrBlock "
                    "intrinsic), never 0.0.0.0/0. cdk-nag cannot validate the "
                    "intrinsic, so the validation failure is a tooling artifact, "
                    "not a real EC23 violation. Owner: NetworkConstruct (vpce "
                    "ingress 443 from pod_sg)."
                ),
            ),
        ],
    )


# ----------------------------------------------------------------- eks cluster


def _suppress_eks_cluster(stack: RouteIqStack) -> None:
    """Suppress the AWS-managed Auto Mode policy findings (IAM4).

    EKS Auto Mode REQUIRES these exact AWS-published managed policies on the
    cluster + node roles, and the amazon-cloudwatch-observability add-on requires
    CloudWatchAgentServerPolicy on its agent role - none are replaceable with a
    narrower custom policy without breaking Auto Mode / the add-on (proposal
    section 4.1 / 7.2 / 7.5). The CloudWatch agent role is only present once
    enable_container_insights() has run, so it is getattr-guarded.
    """
    stack_id = stack.node.id
    eks_cluster = getattr(stack, "eks_cluster", None)
    if eks_cluster is None:
        return

    # The L1 CfnCluster (construct id "Cluster" - L1 CfnResource, so the nag path
    # is the construct itself, NOT a child /Resource). AwsSolutions-EKS2 fires
    # because control-plane log export (api/audit/authenticator/controllerManager/
    # scheduler) is not enabled. P0 ships POD/CONTAINER observability via the
    # amazon-cloudwatch-observability add-on + the CDK-created RoutingLogGroup
    # (proposal section 7.5); the cluster control-plane audit-log export is a
    # deferred P2 observability item (ADR-0027), not a P0 deliverable. Enabling it
    # adds per-log-stream CloudWatch ingestion cost that is best sized by the
    # operator once the deployment is live.
    NagSuppressions.add_resource_suppressions_by_path(
        stack,
        f"/{stack_id}/EksClusterConstruct/Cluster",
        [
            NagPackSuppression(
                id="AwsSolutions-EKS2",
                reason=(
                    "EKS control-plane log export is a deferred P2 observability "
                    "item (ADR-0027), not a P0 deliverable. P0 ships pod/container "
                    "observability via the amazon-cloudwatch-observability add-on "
                    "plus the CDK-created RoutingLogGroup; the control-plane audit "
                    "logs add per-stream CloudWatch ingestion cost best sized by "
                    "the operator once the deployment is live, and the EKS API "
                    "endpoint is PRIVATE-only at P0 so the attack surface is in-VPC "
                    "only. Owner: EksClusterConstruct (Cluster, control-plane "
                    "logging deferred to P2)."
                ),
            ),
        ],
    )

    # The Auto Mode cluster role - the 5 AWS-managed control-plane policies.
    NagSuppressions.add_resource_suppressions_by_path(
        stack,
        f"/{stack_id}/EksClusterConstruct/ClusterRole/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-IAM4",
                reason=(
                    "EKS Auto Mode REQUIRES these exact AWS-published control-plane "
                    "policies so the AWS-managed data plane can provision nodes, "
                    "EBS, and ELBs. They are AWS-managed, scoped to EKS service "
                    "actions, and not replaceable with a narrower custom policy "
                    "without breaking Auto Mode. Owner: EksClusterConstruct "
                    "(ClusterRole)."
                ),
                applies_to=[
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSClusterPolicy",
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSComputePolicy",
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSBlockStoragePolicy",
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSLoadBalancingPolicy",
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSNetworkingPolicy",
                ],
            ),
        ],
    )

    # The Auto Mode node role - the AWS-managed worker + ECR-pull-only policies.
    NagSuppressions.add_resource_suppressions_by_path(
        stack,
        f"/{stack_id}/EksClusterConstruct/NodeRole/Resource",
        [
            NagPackSuppression(
                id="AwsSolutions-IAM4",
                reason=(
                    "Auto Mode node identity needs the AWS-managed minimal "
                    "worker-node + ECR-pull-only policies. These are the "
                    "least-privilege AWS-published node policies; pod-level AWS "
                    "access (Bedrock/Secrets/S3) is via EKS Pod Identity on the "
                    "separate pod role, not this role. Owner: EksClusterConstruct "
                    "(NodeRole)."
                ),
                applies_to=[
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonEKSWorkerNodeMinimalPolicy",
                    "Policy::arn:<AWS::Partition>:iam::aws:policy/"
                    "AmazonEC2ContainerRegistryPullOnly",
                ],
            ),
        ],
    )

    # The CloudWatch observability agent role (only when container insights ran).
    # The role logical id is "ContainerInsightsAgentRole" (the helper builds it as
    # f"{construct_id}AgentRole" with construct_id="ContainerInsights").
    ci_role = getattr(eks_cluster, "container_insights_role", None)
    if ci_role is not None:
        NagSuppressions.add_resource_suppressions_by_path(
            stack,
            f"/{stack_id}/EksClusterConstruct/ContainerInsightsAgentRole/Resource",
            [
                NagPackSuppression(
                    id="AwsSolutions-IAM4",
                    reason=(
                        "The amazon-cloudwatch-observability add-on's CloudWatch "
                        "agent requires the AWS-managed CloudWatchAgentServerPolicy "
                        "(the AWS-published least-privilege policy for emitting "
                        "Container Insights metrics + logs); not replaceable with a "
                        "narrower custom policy without breaking the add-on. Owner: "
                        "EksClusterConstruct.enable_container_insights."
                    ),
                    applies_to=[
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/CloudWatchAgentServerPolicy",
                    ],
                ),
            ],
        )


# ------------------------------------------------------------------------- ecr


def _suppress_ecr(_stack: RouteIqStack) -> None:
    """No ECR suppressions at P0.

    The EcrConstruct emits only L1 resources - CfnPullThroughCacheRule,
    CfnRepositoryCreationTemplate, CfnRegistryScanningConfiguration - and NO
    standalone ecr.Repository, NO IAM role, NO bucket. cdk-nag's AwsSolutions
    pack does not flag those L1 ECR registry-governance resources, so there is
    nothing to suppress here. Kept as an explicit no-op fan-out target so the
    construct families stay symmetric and a future ECR finding has an obvious
    home. Owner: EcrConstruct (no findings at P0).
    """
    return


# -------------------------------------------------------------------- pod role


def _suppress_pod_role(stack: RouteIqStack) -> None:
    """Suppress the pod role's Bedrock + Logs resource wildcards (IAM5).

    The pod role carries exactly the P0 grant set (Bedrock invoke + Secrets +
    optional S3 + Logs). cdk-nag AwsSolutions-IAM5 fires on:

    - ``bedrock:InvokeModel*`` on ``foundation-model/*`` - the model set is
      deploy-time dynamic and the foundation-model wildcard is the only valid
      form for a router (proposal section 4.2 / 13). Suppressed only on the
      DEFAULT (wildcard) path; when ``bedrock_model_arns`` is supplied the
      resources are explicit ARNs and IAM5 does not fire.
    - the CloudWatch Logs ``:*`` suffix on the single routing log group - the
      ``:*`` suffix is required by CloudWatch Logs on log-stream-creating actions
      for the target group; it is not a cross-resource wildcard.
    - the ``routeiq/*`` Secrets-Manager name wildcard on the DEFAULT path - the
      exact secret ARNs are operator-pinned at deploy time; the wildcard is
      scoped to the account/region ``routeiq/`` secret namespace.

    The path is the pod role's CDK-generated DefaultPolicy
    (``/{stack_id}/PodRole/DefaultPolicy/Resource``). The role exists
    unconditionally, but the DefaultPolicy only materialises once at least one
    inline statement is attached (always true at P0: BedrockInvoke + SecretsRead
    are always added), so it is present whenever the pod role is.
    """
    stack_id = stack.node.id
    pod_role = getattr(stack, "pod_role", None)
    if pod_role is None:
        return

    suppressions = [
        NagPackSuppression(
            id="AwsSolutions-IAM5",
            reason=(
                "bedrock:InvokeModel / InvokeModelWithResponseStream / Converse / "
                "ConverseStream on foundation-model/* is scoped to the Bedrock "
                "service in the deploy region; the model set is deploy-time "
                "dynamic (the router selects a model per request), so the "
                "foundation-model wildcard is the only valid form for a router. "
                "Supplying routeiq:bedrock_model_arns pins explicit ARNs and this "
                "wildcard disappears. Owner: RouteIqStack (PodRole BedrockInvoke)."
            ),
            applies_to=[
                "Resource::arn:<AWS::Partition>:bedrock:<AWS::Region>::foundation-model/*",
            ],
        ),
        NagPackSuppression(
            id="AwsSolutions-IAM5",
            reason=(
                "secretsmanager:GetSecretValue / DescribeSecret on the routeiq/ "
                "secret-name wildcard is scoped to the deploy account/region "
                "routeiq/ Secrets Manager namespace (LITELLM_MASTER_KEY / "
                "ADMIN_API_KEYS / provider keys). Exact ARNs are operator-pinned "
                "at deploy time via routeiq:secret_arns, which removes the "
                "wildcard. Owner: RouteIqStack (PodRole SecretsRead)."
            ),
            applies_to=[
                "Resource::arn:<AWS::Partition>:secretsmanager:<AWS::Region>:"
                "<AWS::AccountId>:secret:routeiq/*",
            ],
        ),
        NagPackSuppression(
            id="AwsSolutions-IAM5",
            reason=(
                "logs:CreateLogStream / PutLogEvents target the single CDK-created "
                "routing log group with the canonical ':*' suffix that CloudWatch "
                "Logs requires on log-stream-creating actions for the target "
                "group. The suffix targets log streams under one stack-scoped log "
                "group, NOT arbitrary log groups - it is not a cross-resource "
                "wildcard. Owner: RouteIqStack (PodRole Logs)."
            ),
            applies_to=[
                "Resource::arn:<AWS::Partition>:logs:<AWS::Region>:<AWS::AccountId>:"
                f"log-group:{routing_log_group_name(stack.env_name)}:*",
            ],
        ),
    ]

    # ConfigS3Read is added ONLY when routeiq:config_s3_bucket is supplied
    # (otherwise the statement does not exist and the synth stays byte-stable).
    # When present, s3:GetObject / GetObjectAttributes target the <bucket>/* object
    # wildcard - the object key set is deploy-time dynamic (config files are
    # versioned + ETag-polled), and the wildcard is scoped to the single
    # operator-named config bucket. The applies_to is built from the bucket name so
    # it matches the literal finding resource (a fixed name, not a token).
    config_bucket = getattr(stack, "_config_s3_bucket", None)
    if config_bucket:
        suppressions.append(
            NagPackSuppression(
                id="AwsSolutions-IAM5",
                reason=(
                    "s3:GetObject / GetObjectAttributes on the config bucket "
                    "object wildcard is scoped to the single operator-named "
                    "RouteIQ config bucket; the object key set is deploy-time "
                    "dynamic (versioned config files + the ETag poll the "
                    "gateway.configSync.s3 path uses), so the <bucket>/* object "
                    "wildcard is the only valid form. It does not grant bucket "
                    "listing or cross-bucket access. Owner: RouteIqStack (PodRole "
                    "ConfigS3Read)."
                ),
                applies_to=[
                    f"Resource::arn:<AWS::Partition>:s3:::{config_bucket}/*",
                ],
            )
        )

    NagSuppressions.add_resource_suppressions_by_path(
        stack,
        f"/{stack_id}/PodRole/DefaultPolicy/Resource",
        suppressions,
    )
