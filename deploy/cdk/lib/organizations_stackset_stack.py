"""OrganizationsStackSetStack - service-managed StackSet onboarding for capacity
accounts (RouteIQ-ea99, the cred-free CDK half).

WHAT THIS IS: a standalone CloudFormation stack deployed INTO the RouteIQ AWS
Organization MANAGEMENT account (or a delegated CloudFormation StackSets admin).
It declares ONE service-managed ``AWS::CloudFormation::StackSet`` that fans the
cross-account Bedrock-capacity role (the SAME role
``BedrockCapacityMemberStack`` mints, expressed as an embedded CFN template
body) out to EVERY member account in one or more target Organizational Units --
and, with ``auto_deployment`` ON, to any account LATER moved into those OUs. This
is the org-native alternative to the per-account ``cdk deploy`` of
``BedrockCapacityMemberStack``: instead of running one deploy per capacity
account, an operator enrolls the OU once and new accounts onboard automatically.

WHY SERVICE-MANAGED (not self-managed): service-managed permissions let
CloudFormation use the org's trusted-access roles
(``AWSServiceRoleForCloudFormationStackSetsOrgAdmin`` +
``AWSControlTowerStackSetRole`` / ``stacksets.cloudformation.amazonaws.com``
trusted access) so NO per-account admin/execution role pair has to be
pre-created. It targets OUs (not account ids) and supports ``AutoDeployment``
(enroll-on-move). The trade-off vs the standalone ``BedrockCapacityMemberStack``:
service-managed StackSets can ONLY reach accounts INSIDE the home Organization,
so a standalone capacity account outside the org still uses the per-account
member stack (see that file's WHY-A-SEPARATE-STACK note). The two are
complementary onboarding paths, not duplicates.

DEFAULT OFF / OPERATOR-GATED: this stack is NOT instantiated by the default
``app.py`` synth, so the P0/P1/P2 template snapshots stay byte-stable. An
operator opts in by instantiating it in a deploy app with the target OU ids +
the home gateway pod-role ARN. The LIVE prerequisite -- enabling AWS
Organizations *trusted access* for CloudFormation StackSets (a one-time
``aws cloudformation activate-organizations-access`` / console toggle in the
management account) -- is an OPERATOR step that CANNOT be expressed in this
member-deploy stack; it is documented in
``docs/deployment/multi-account-multi-region.md``. Synth is credential-free.

WHAT IT CREATES: one ``AWS::CloudFormation::StackSet`` (service-managed,
auto-deploy on by default) whose embedded template body provisions, in each
targeted member account, the SAME ``RouteIqBedrockCapacity-<env>`` role that
``BedrockCapacityMemberStack`` mints standalone: a role trusting the home gateway
pod-role ARN via plain ``sts:AssumeRole`` (NO web-identity -- the Pod-Identity
analog; RouteIQ has no OIDC issuer), with the 4-action Bedrock invoke contract.
The STABLE role name keeps the member ARN predictable so the home grant
(RouteIqStack ``capacity_account_ids``) and LiteLLM ``aws_role_name`` reference
it deterministically.

Every IAM role/statement Description is ASCII / Latin-1 ONLY (an em-dash passes
``cdk synth`` but FAILS the IAM CREATE API). Mirrors
``BedrockCapacityMemberStack``.
"""

from __future__ import annotations

import json

from aws_cdk import CfnOutput, Stack, Tags
from aws_cdk import aws_cloudformation as cfn
from constructs import Construct

# The OU id pattern CloudFormation enforces (org root ``r-xxxx`` or an OU
# ``ou-xxxx-xxxxxxxx``). We validate up front so a typo'd target fails at synth
# with a clear message instead of at deploy with a CloudFormation error.
_OU_ID_PREFIXES = ("ou-", "r-")


def _member_capacity_template(
    *,
    env_name: str,
    home_pod_role_arn: str,
    bedrock_model_arns: list[str] | None,
) -> dict:
    """Build the embedded CFN template body the StackSet deploys per member account.

    Byte-for-byte the SAME capacity role contract as
    ``BedrockCapacityMemberStack``: a STABLE-named role trusting the home pod-role
    ARN via plain ``sts:AssumeRole`` with the 4-action Bedrock invoke policy. The
    template is a plain dict (not a CDK Stack synth) because a StackSet template
    body is just a CFN document string; keeping it explicit makes the byte content
    test-assertable without a nested-app synth.
    """
    if bedrock_model_arns:
        bedrock_resources: list = list(bedrock_model_arns)
    else:
        # Region-wildcard foundation-model + this-account inference-profile ARNs.
        # ``${AWS::Partition}`` / ``${AWS::AccountId}`` are resolved IN THE MEMBER
        # ACCOUNT at StackSet-instance deploy time (the right account/partition).
        bedrock_resources = [
            {"Fn::Sub": "arn:${AWS::Partition}:bedrock:*::foundation-model/*"},
            {"Fn::Sub": "arn:${AWS::Partition}:bedrock:*:${AWS::AccountId}:inference-profile/*"},
        ]

    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": (
            f"RouteIQ {env_name} service-managed StackSet member capacity role. "
            "Mints RouteIqBedrockCapacity-" + env_name + " (plain sts:AssumeRole "
            "trust on the home gateway pod role, 4-action Bedrock invoke). "
            "ASCII-only."
        ),
        "Resources": {
            "BedrockCapacityRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    # STABLE name => predictable ARN the home grant computes and
                    # LiteLLM references as aws_role_name (same as the standalone).
                    "RoleName": f"RouteIqBedrockCapacity-{env_name}",
                    "Description": (
                        f"RouteIQ {env_name} cross-account Bedrock capacity role "
                        "(StackSet-managed). Assumed by the RouteIQ home gateway pod "
                        "role (plain sts:AssumeRole) to borrow THIS account Bedrock "
                        "quota via LiteLLM aws_role_name. ASCII-only."
                    ),
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                # Plain sts:AssumeRole on the home pod role ARN. NO
                                # Federated principal / web-identity (Pod-Identity
                                # home cluster has no OIDC issuer).
                                "Principal": {"AWS": home_pod_role_arn},
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    },
                    "Policies": [
                        {
                            "PolicyName": "BedrockInvoke",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Sid": "BedrockInvoke",
                                        "Effect": "Allow",
                                        "Action": [
                                            "bedrock:InvokeModel",
                                            "bedrock:InvokeModelWithResponseStream",
                                            "bedrock:Converse",
                                            "bedrock:ConverseStream",
                                        ],
                                        "Resource": bedrock_resources,
                                    }
                                ],
                            },
                        }
                    ],
                },
            }
        },
        "Outputs": {
            "CapacityRoleArn": {
                "Description": (
                    "Cross-account Bedrock capacity role ARN. Set this as "
                    "litellm_params.aws_role_name in a per-account model_list "
                    "deployment."
                ),
                "Value": {"Fn::GetAtt": ["BedrockCapacityRole", "Arn"]},
            }
        },
    }


class OrganizationsStackSetStack(Stack):
    """Service-managed StackSet that onboards capacity accounts across OUs.

    Deployed standalone INTO the org management account. Default-OFF (not in the
    default ``app.py`` synth); operator opts in. Synth is credential-free.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        # The ARN of the RouteIQ HOME gateway pod role (RouteIqStack's PodRoleArn).
        # The SAME single trust source as the standalone member stack. REQUIRED -
        # with no trust source the minted member role is un-assumable.
        home_pod_role_arn: str,
        # Target Organizational Unit ids (or the org root r-xxxx). REQUIRED + non-
        # empty: a service-managed StackSet with no OU target deploys nowhere.
        organizational_unit_ids: list[str],
        # The AWS regions the StackSet provisions the (region-agnostic IAM) member
        # role in. IAM is global, so one region is enough; defaults to the home
        # deploy region. Multiple are accepted (harmless idempotent role create).
        target_regions: list[str] | None = None,
        # Auto-deploy to accounts LATER moved into the target OUs (enroll-on-move).
        # Default ON (the whole point of the OU-targeted onboarding path).
        auto_deploy: bool = True,
        # Retain the member stack instance (and its role) when an account leaves
        # the OU. Default False: removing an account from the OU removes the role.
        retain_stacks_on_account_removal: bool = False,
        # Optional Bedrock invoke scope-down (explicit ARNs replace the wildcards).
        bedrock_model_arns: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if not home_pod_role_arn:
            raise ValueError(
                "home_pod_role_arn is required: the StackSet-minted member role must "
                "trust the RouteIQ home gateway pod role (plain sts:AssumeRole). With "
                "no trust source the capacity role is un-assumable. Pass the home "
                "RouteIqStack PodRoleArn CfnOutput value."
            )
        ou_ids = [str(o).strip() for o in (organizational_unit_ids or []) if str(o).strip()]
        if not ou_ids:
            raise ValueError(
                "organizational_unit_ids is required and must be non-empty: a "
                "service-managed StackSet targets OUs (or the org root r-xxxx); with "
                "no target it deploys to no account."
            )
        for ou in ou_ids:
            if not ou.startswith(_OU_ID_PREFIXES):
                raise ValueError(
                    f"invalid organizational unit id {ou!r}: expected an OU id "
                    "(ou-xxxx-xxxxxxxx) or the org root id (r-xxxx)."
                )

        self.env_name = env_name
        regions = [str(r).strip() for r in (target_regions or [self.region]) if str(r).strip()]

        Tags.of(self).add("routeiq:env", env_name)
        Tags.of(self).add("routeiq:stack", "organizations-stackset")

        template_body = _member_capacity_template(
            env_name=env_name,
            home_pod_role_arn=home_pod_role_arn,
            bedrock_model_arns=bedrock_model_arns,
        )

        # The service-managed StackSet. permission_model=SERVICE_MANAGED uses the
        # org's trusted-access roles (no per-account admin/exec role to pre-make);
        # auto_deployment governs enroll-on-move; CAPABILITY_NAMED_IAM is required
        # because the member template mints a NAMED IAM role.
        self.stack_set = cfn.CfnStackSet(
            self,
            "BedrockCapacityStackSet",
            stack_set_name=f"RouteIqBedrockCapacity-{env_name}",
            permission_model="SERVICE_MANAGED",
            description=(
                f"RouteIQ {env_name} service-managed StackSet that onboards capacity "
                "accounts across the targeted OUs with the cross-account Bedrock "
                "capacity role. ASCII-only."
            ),
            capabilities=["CAPABILITY_NAMED_IAM"],
            auto_deployment=cfn.CfnStackSet.AutoDeploymentProperty(
                enabled=auto_deploy,
                retain_stacks_on_account_removal=(
                    retain_stacks_on_account_removal if auto_deploy else None
                ),
            ),
            # Acting from the org MANAGEMENT account (not a delegated admin).
            call_as="SELF",
            # Parallel rollout across the OU; tolerate zero failures by default.
            operation_preferences=cfn.CfnStackSet.OperationPreferencesProperty(
                failure_tolerance_count=0,
                max_concurrent_count=1,
                region_concurrency_type="SEQUENTIAL",
            ),
            stack_instances_group=[
                cfn.CfnStackSet.StackInstancesProperty(
                    deployment_targets=cfn.CfnStackSet.DeploymentTargetsProperty(
                        organizational_unit_ids=ou_ids,
                    ),
                    regions=regions,
                )
            ],
            # The template body is a JSON string (CFN accepts a JSON or YAML doc).
            template_body=json.dumps(template_body, sort_keys=True),
        )

        CfnOutput(
            self,
            "StackSetId",
            value=self.stack_set.attr_stack_set_id,
            description=(
                "Service-managed StackSet id. New accounts moved into the targeted "
                "OUs auto-onboard the RouteIqBedrockCapacity-" + env_name + " role "
                "(when auto_deploy is on)."
            ),
        )
        CfnOutput(
            self,
            "MemberCapacityRoleName",
            value=f"RouteIqBedrockCapacity-{env_name}",
            description=(
                "Stable member capacity role name. The home grant computes "
                "arn:<part>:iam::<member-acct>:role/<this> and LiteLLM references it "
                "as aws_role_name."
            ),
        )

        # No cdk-nag suppression block: AWS::CloudFormation::StackSet is an L1 with
        # no IAM identity of its own (the IAM role lives in the EMBEDDED member
        # template, which AwsSolutionsChecks does NOT descend into -- the role is
        # created in the member account, not this stack). So nothing here fires.
