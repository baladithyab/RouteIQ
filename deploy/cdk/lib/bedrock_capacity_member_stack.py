"""BedrockCapacityMemberStack - the cross-account Bedrock capacity onboarding unit.

WHAT THIS IS: a small, standalone CloudFormation stack deployed INTO each
*capacity* account (an AWS account, other than the RouteIQ home/deploy account,
whose Bedrock on-demand quotas you want to borrow). Bedrock TPM/RPM limits are
PER-ACCOUNT, so N accounts = N independent capacity pools. This maps cleanly onto
LiteLLM's native multi-deployment model_list: each capacity account becomes one
``model_list`` row sharing the same ``model_name``, and each row is an
independently cool-downable arm (RouteIQ-e677 / doc 50 section 2). A throttled
account's row trips LiteLLM's per-deployment cooldown and traffic shifts to the
healthy rows.

WHY A SEPARATE STACK (not a child of RouteIqStack, not a StackSet): the capacity
accounts are operated INDEPENDENTLY of the home account (a capacity account may be
a standalone account outside the home AWS Organization). Service-managed StackSets
target an OU and cannot reach a standalone account, so onboarding is an explicit
per-account ``cdk deploy`` with that account's credentials profile. This file just
codifies the onboarding so adding an account is reproducible, not click-ops.
DEPLOY is operator-gated; synth is fully credential-free.

WHAT IT CREATES: one role, ``RouteIqBedrockCapacity-<env>``, that the RouteIQ HOME
gateway pod role is allowed to assume (plain ``sts:AssumeRole``), with permissions
to invoke Bedrock IN THIS account.

DIVERGENCE FROM THE VSR SOURCE (vllm-sr-on-aws bedrock_capacity_member_stack.py),
deliberate for RouteIQ:

  - **Plain sts:AssumeRole ONLY - no web-identity.** RouteIQ's cluster uses EKS
    Pod Identity, NOT IRSA: there is NO OIDC issuer to register as an IAM identity
    provider, so the VSR ``home_oidc_issuer_url`` / ``OpenIdConnectProvider`` /
    ``OpenIdConnectPrincipal`` native (web-identity) trust branch is DROPPED
    entirely. The member role's only trust source is a plain
    ``iam.ArnPrincipal(home_pod_role_arn)`` - the RouteIQ home gateway pod role
    (the single Pod-Identity role minted by RouteIqStack) does
    ``sts:AssumeRole`` against it directly.
  - **No mantle.** The VSR ``BedrockMantleInference`` /
    ``BedrockMantleBearerToken`` statements (and the second
    ``home_minter_role_arn`` trust statement they served) are DROPPED - RouteIQ
    has no bedrock-mantle bearer path and no minter role. This also removes VSR's
    ``Resource::*`` nag suppression (no identity-level mantle action remains).
  - **Renamed** ``VllmSrBedrockCapacity-<env>`` -> ``RouteIqBedrockCapacity-<env>``
    (a STABLE role name => predictable ARN
    ``arn:<part>:iam::<acct>:role/RouteIqBedrockCapacity-<env>`` that the home
    grant computes and that LiteLLM references as ``aws_role_name``).

MAPS TO LiteLLM ``aws_role_name`` (doc 50 section 2c): each member-role ARN
(the ``CapacityRoleArn`` CfnOutput below) becomes one ``model_list`` entry's
``litellm_params.aws_role_name``. At call time LiteLLM's
``base_aws_llm.BaseAWSLLM`` STS-assumes that role (the ``aws_role_name`` ->
``_auth_with_aws_role`` path) and signs the Bedrock request with the resulting
THIS-account credentials, so the call draws on THIS account's Bedrock quota. The
home grant added inside RouteIqStack (flag-gated on ``capacity_account_ids``) is
what lets the home pod role perform that assume. C1 (this file) provisions the
roles those ``aws_role_name`` rows reference; C0 (doc/config) wires the rows.

Every IAM role/statement Description is ASCII / Latin-1 ONLY: an em-dash (U+2014)
passes ``cdk synth`` but FAILS the IAM CREATE API. The VSR source descriptions
used em-dashes; this port uses plain hyphens.
"""

from __future__ import annotations

from aws_cdk import Aws, CfnOutput, Stack, Tags
from aws_cdk import aws_iam as iam
from cdk_nag import NagSuppressions
from constructs import Construct


class BedrockCapacityMemberStack(Stack):
    """Cross-account Bedrock capacity role for ONE member (capacity) account.

    Deployed standalone INTO a capacity account. Trusts the RouteIQ home gateway
    pod role via plain ``sts:AssumeRole`` (NO web-identity, NO mantle).
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        # The ARN of the RouteIQ HOME gateway pod role (RouteIqStack's single
        # Pod-Identity pod role - its PodRoleArn CfnOutput). This is the ONLY trust
        # source: the member role trusts it for plain sts:AssumeRole. REQUIRED -
        # with no trust source the role is un-assumable.
        home_pod_role_arn: str,
        # Optional scope-down: explicit Bedrock model / inference-profile ARNs. When
        # supplied, the region-wildcard foundation-model + inference-profile ARNs
        # are replaced by exactly these, and the IAM5 wildcard suppression no longer
        # applies (the resources are explicit).
        bedrock_model_arns: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if not home_pod_role_arn:
            raise ValueError(
                "home_pod_role_arn is required: the member role must trust the "
                "RouteIQ home gateway pod role (plain sts:AssumeRole). With no "
                "trust source the capacity role is un-assumable. Pass the home "
                "RouteIqStack PodRoleArn CfnOutput value."
            )

        self.env_name = env_name
        self._bedrock_model_arns = list(bedrock_model_arns or [])

        Tags.of(self).add("routeiq:env", env_name)
        Tags.of(self).add("routeiq:stack", "bedrock-capacity-member")

        # -- The capacity role ------------------------------------------------
        # Trust is a single plain ArnPrincipal of the home pod role ARN. CDK's
        # ArnPrincipal emits the correct plain sts:AssumeRole assume-role action
        # with no conditions; there is NO OpenIdConnectPrincipal (no OIDC issuer on
        # the Pod-Identity home cluster) and NO second mantle/minter trust
        # statement. The role name is STABLE so the home grant can compute the ARN.
        self.capacity_role = iam.Role(
            self,
            "BedrockCapacityRole",
            role_name=f"RouteIqBedrockCapacity-{env_name}",  # stable => predictable ARN
            assumed_by=iam.ArnPrincipal(home_pod_role_arn),  # plain sts:AssumeRole
            description=(
                f"RouteIQ {env_name} cross-account Bedrock capacity role. Assumed by "
                "the RouteIQ home gateway pod role (plain sts:AssumeRole) to borrow "
                "THIS account Bedrock quota via LiteLLM aws_role_name. ASCII-only."
            ),
        )

        # -- Permissions: invoke Bedrock IN THIS account ----------------------
        # Same 4-action invoke contract as the home RouteIqStack pod role. Region
        # wildcard on foundation-model/* (the service evaluates a region-less
        # foundation-model ARN and fans out for global./us. inference profiles, and
        # a capacity account is explicitly a cross-region capacity source - so the
        # member port matches VSR's ``*`` region, unlike the home pod role which
        # pins its own deploy region). Supplying bedrock_model_arns pins explicit
        # ARNs and the wildcard disappears (the suppression no longer applies).
        bedrock_resources = self._bedrock_model_arns or [
            f"arn:{Aws.PARTITION}:bedrock:*::foundation-model/*",
            f"arn:{Aws.PARTITION}:bedrock:*:{Aws.ACCOUNT_ID}:inference-profile/*",
        ]
        self.capacity_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockInvoke",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:Converse",
                    "bedrock:ConverseStream",
                ],
                resources=bedrock_resources,
            )
        )

        CfnOutput(
            self,
            "CapacityRoleArn",
            value=self.capacity_role.role_arn,
            description=(
                "Cross-account Bedrock capacity role ARN. Set this as "
                "litellm_params.aws_role_name in a per-account model_list "
                "deployment (LiteLLM BaseAWSLLM aws_role_name / _auth_with_aws_role "
                "STS-assumes it to borrow THIS account Bedrock quota)."
            ),
        )

        self._suppress_nag()

    def _suppress_nag(self) -> None:
        """Evidenced suppression: ONE AwsSolutions-IAM5 on the Bedrock invoke
        wildcards (DROPPED relative to VSR: the ``Resource::*`` mantle suppression,
        because RouteIQ has no bedrock-mantle statement).

        Applied via ``add_resource_suppressions`` on the role itself (this is a
        standalone Stack, not a child of RouteIqStack, so the by-path form is not
        needed - the in-construct form is cleanest). When ``bedrock_model_arns`` is
        supplied the resources are explicit ARNs and IAM5 does not fire; the
        suppression is then inert (it only matches the wildcard appliesTo).
        """
        NagSuppressions.add_resource_suppressions(
            self.capacity_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "appliesTo": [
                        "Resource::arn:<AWS::Partition>:bedrock:*::foundation-model/*",
                        "Resource::arn:<AWS::Partition>:bedrock:*:<AWS::AccountId>:"
                        "inference-profile/*",
                    ],
                    "reason": (
                        "bedrock:InvokeModel / InvokeModelWithResponseStream / "
                        "Converse / ConverseStream on the region-wildcard "
                        "foundation-model + account inference-profile ARNs is the "
                        "only valid form for a router whose model set is deploy-time "
                        "dynamic and that borrows this account quota as a "
                        "cross-region capacity source (the service fans a region-less "
                        "foundation-model ARN out across regions for global./us. "
                        "inference profiles). Invoke-only, no admin surface. "
                        "Supplying bedrock_model_arns pins explicit ARNs and the "
                        "wildcard disappears. Owner: BedrockCapacityMemberStack "
                        "(BedrockInvoke)."
                    ),
                }
            ],
            apply_to_children=True,
        )
