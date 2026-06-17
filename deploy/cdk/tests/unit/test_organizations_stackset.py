"""Unit tests for RouteIQ-ea99 - service-managed StackSet capacity onboarding.

Credential-free (synth offline against the dummy env ``123456789012`` /
``us-west-2``). The ``OrganizationsStackSetStack`` is a STANDALONE stack an
operator deploys INTO the org management account; it is NOT in the default
``app.py`` synth, so the P0/P1/P2 snapshots are unaffected (asserted elsewhere by
the snapshot suite staying green).

Asserts:
  1. Exactly one ``AWS::CloudFormation::StackSet``, service-managed, named
     ``RouteIqBedrockCapacity-<env>``, CAPABILITY_NAMED_IAM, auto-deploy on.
  2. The DeploymentTargets carry the supplied OU ids (and ONLY those).
  3. The EMBEDDED member template mints the SAME stable-named capacity role with
     PLAIN sts:AssumeRole trust on the home pod-role ARN (NO web-identity / NO
     Federated principal) + the 4-action Bedrock invoke contract.
  4. No mantle action / no Resource:* in the embedded member policy.
  5. Validation: empty home_pod_role_arn, empty OU list, and a malformed OU id
     each raise.
  6. cdk-nag clean (the StackSet L1 has no IAM identity of its own; the role is
     created in the MEMBER account via the embedded template).

Mirrors the conventions of ``test_bedrock_capacity.py`` (standalone-stack synth,
the em-dash charset guard, the fresh-app + ``AwsSolutionsChecks`` aspect).
"""

from __future__ import annotations

import json
import re

import aws_cdk as cdk
import pytest
from aws_cdk import Aspects
from aws_cdk.assertions import Annotations, Match, Template
from cdk_nag import AwsSolutionsChecks

from lib.organizations_stackset_stack import OrganizationsStackSetStack

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"
_HOME_POD_ROLE_ARN = "arn:aws:iam::123456789012:role/RouteIqStack-dev-PodRole-AbCdEf123"
_OU_IDS = ["ou-abcd-12345678", "ou-abcd-87654321"]

# IAM's allowed Description charset (mirrors test_bedrock_capacity.py).
_IAM_DESCRIPTION_CHARSET = re.compile("^[" + "\t\n\r" + " -~" + "¡-ÿ" + "]*$")


def _dummy_env() -> cdk.Environment:
    return cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION)


def _stack(*, app: cdk.App | None = None, **kwargs):
    if app is None:
        app = cdk.App()
    kwargs.setdefault("env_name", "dev")
    kwargs.setdefault("home_pod_role_arn", _HOME_POD_ROLE_ARN)
    kwargs.setdefault("organizational_unit_ids", _OU_IDS)
    return OrganizationsStackSetStack(app, "OrgStackSet-dev", env=_dummy_env(), **kwargs)


def _template(**kwargs) -> Template:
    return Template.from_stack(_stack(**kwargs))


def _stackset_props(template: Template) -> dict:
    sets = template.find_resources("AWS::CloudFormation::StackSet")
    assert len(sets) == 1, sets
    return next(iter(sets.values()))["Properties"]


def _embedded_template(template: Template) -> dict:
    """Parse the embedded member CFN template body (a JSON string)."""
    props = _stackset_props(template)
    body = props["TemplateBody"]
    return json.loads(body) if isinstance(body, str) else body


# ---------------------------------------------------------------- StackSet shape


def test_one_service_managed_stackset() -> None:
    template = _template()
    template.resource_count_is("AWS::CloudFormation::StackSet", 1)
    props = _stackset_props(template)
    assert props["PermissionModel"] == "SERVICE_MANAGED", props
    assert props["StackSetName"] == "RouteIqBedrockCapacity-dev", props
    assert props["Capabilities"] == ["CAPABILITY_NAMED_IAM"], props
    assert props["AutoDeployment"]["Enabled"] is True, props["AutoDeployment"]


def test_deployment_targets_are_the_supplied_ous() -> None:
    template = _template()
    props = _stackset_props(template)
    groups = props["StackInstancesGroup"]
    assert len(groups) == 1, groups
    targets = groups[0]["DeploymentTargets"]
    assert sorted(targets["OrganizationalUnitIds"]) == sorted(_OU_IDS), targets
    assert groups[0]["Regions"] == [DUMMY_REGION], groups[0]


def test_target_regions_override() -> None:
    template = _template(target_regions=["us-east-1", "eu-west-1"])
    props = _stackset_props(template)
    assert props["StackInstancesGroup"][0]["Regions"] == ["us-east-1", "eu-west-1"]


def test_auto_deploy_off_disables_retain() -> None:
    template = _template(auto_deploy=False)
    props = _stackset_props(template)
    assert props["AutoDeployment"]["Enabled"] is False
    # retain is None when auto_deploy is off (not emitted).
    assert "RetainStacksOnAccountRemoval" not in props["AutoDeployment"]


# ---------------------------------------------------------- embedded member role


def test_embedded_role_is_stable_named() -> None:
    embedded = _embedded_template(_template())
    role = embedded["Resources"]["BedrockCapacityRole"]
    assert role["Type"] == "AWS::IAM::Role"
    assert role["Properties"]["RoleName"] == "RouteIqBedrockCapacity-dev"


def test_embedded_role_plain_assume_no_web_identity() -> None:
    """Trust is exactly one plain sts:AssumeRole on the home pod-role ARN.

    The StackSet analog of the standalone member's no-IRSA assertion: NO
    web-identity, NO Federated principal in the embedded role's trust document.
    """
    embedded = _embedded_template(_template())
    role = embedded["Resources"]["BedrockCapacityRole"]
    doc = role["Properties"]["AssumeRolePolicyDocument"]
    statements = doc["Statement"]
    assert len(statements) == 1, statements
    stmt = statements[0]
    assert stmt["Action"] == "sts:AssumeRole", stmt
    assert stmt["Effect"] == "Allow", stmt
    assert stmt["Principal"] == {"AWS": _HOME_POD_ROLE_ARN}, stmt["Principal"]

    blob = json.dumps(doc)
    assert "sts:AssumeRoleWithWebIdentity" not in blob, blob
    assert "Federated" not in blob, blob


def test_embedded_role_bedrock_invoke_actions() -> None:
    embedded = _embedded_template(_template())
    role = embedded["Resources"]["BedrockCapacityRole"]
    policy = role["Properties"]["Policies"][0]
    stmt = policy["PolicyDocument"]["Statement"][0]
    assert stmt["Action"] == [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:Converse",
        "bedrock:ConverseStream",
    ], stmt
    blob = json.dumps(stmt["Resource"])
    assert ":bedrock:*::foundation-model/*" in blob, blob
    assert ":inference-profile/*" in blob, blob


def test_embedded_role_explicit_arns_scope_down() -> None:
    explicit = [
        "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-v1:0",
    ]
    embedded = _embedded_template(_template(bedrock_model_arns=explicit))
    role = embedded["Resources"]["BedrockCapacityRole"]
    stmt = role["Properties"]["Policies"][0]["PolicyDocument"]["Statement"][0]
    assert stmt["Resource"] == explicit, stmt["Resource"]
    assert "foundation-model/*" not in json.dumps(stmt["Resource"]), stmt["Resource"]


def test_embedded_member_has_no_mantle_or_wildcard_resource() -> None:
    embedded = _embedded_template(_template())
    blob = json.dumps(embedded)
    assert "bedrock-mantle:" not in blob, blob
    # No Resource:"*" statement in the member policy.
    role = embedded["Resources"]["BedrockCapacityRole"]
    for policy in role["Properties"]["Policies"]:
        for stmt in policy["PolicyDocument"]["Statement"]:
            assert stmt.get("Resource") != "*", stmt


def test_embedded_descriptions_ascii() -> None:
    embedded = _embedded_template(_template())
    role = embedded["Resources"]["BedrockCapacityRole"]
    desc = role["Properties"].get("Description")
    assert isinstance(desc, str) and _IAM_DESCRIPTION_CHARSET.match(desc), desc
    top_desc = embedded.get("Description")
    assert isinstance(top_desc, str) and _IAM_DESCRIPTION_CHARSET.match(top_desc), top_desc


# ---------------------------------------------------------------- validation


def test_requires_home_pod_role_arn() -> None:
    app = cdk.App()
    with pytest.raises(ValueError, match="home_pod_role_arn is required"):
        OrganizationsStackSetStack(
            app,
            "Bad",
            env=_dummy_env(),
            env_name="dev",
            home_pod_role_arn="",
            organizational_unit_ids=_OU_IDS,
        )


def test_requires_non_empty_ou_list() -> None:
    app = cdk.App()
    with pytest.raises(ValueError, match="organizational_unit_ids is required"):
        OrganizationsStackSetStack(
            app,
            "Bad",
            env=_dummy_env(),
            env_name="dev",
            home_pod_role_arn=_HOME_POD_ROLE_ARN,
            organizational_unit_ids=[],
        )


def test_rejects_malformed_ou_id() -> None:
    app = cdk.App()
    with pytest.raises(ValueError, match="invalid organizational unit id"):
        OrganizationsStackSetStack(
            app,
            "Bad",
            env=_dummy_env(),
            env_name="dev",
            home_pod_role_arn=_HOME_POD_ROLE_ARN,
            organizational_unit_ids=["123456789012"],  # an account id, not an OU
        )


def test_accepts_org_root_id() -> None:
    """The org root id (r-xxxx) is a valid target."""
    template = _template(organizational_unit_ids=["r-abcd"])
    props = _stackset_props(template)
    assert props["StackInstancesGroup"][0]["DeploymentTargets"]["OrganizationalUnitIds"] == [
        "r-abcd"
    ]


# ---------------------------------------------------------------- cdk-nag


def test_stackset_cdk_nag_clean() -> None:
    """No AwsSolutions-* errors: the StackSet L1 has no IAM identity of its own.

    The IAM role lives in the EMBEDDED member template (created in the member
    account), which AwsSolutionsChecks does not descend into.
    """
    app = cdk.App()
    stack = _stack(app=app)
    Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        rendered = "\n".join(f"- {e.id}: {str(e.entry.data)[:200]}" for e in errors)
        raise AssertionError(f"{len(errors)} unsuppressed AwsSolutions-* error(s):\n{rendered}")
