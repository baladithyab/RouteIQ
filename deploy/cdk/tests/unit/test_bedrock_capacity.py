"""Unit tests for RouteIQ-6150 (C1) cross-account Bedrock capacity.

Two halves, both credential-free (synth offline against the dummy env
``123456789012`` / ``us-west-2``):

  1. The MEMBER side - ``BedrockCapacityMemberStack`` synthed standalone (an
     operator deploys it INTO each capacity account). It mints ONE role,
     ``RouteIqBedrockCapacity-<env>``, that trusts the RouteIQ home gateway pod
     role via PLAIN ``sts:AssumeRole`` (NO web-identity - the Pod-Identity analog
     of ``test_no_irsa.py``), with the 4-action Bedrock invoke contract, and NO
     mantle statements (the VSR mantle block is dropped). cdk-nag clean.

  2. The HOME side - ``RouteIqStack`` (via ``make_stack``). With no
     ``capacity_account_ids`` (the default) the pod role carries NO
     ``sts:AssumeRole`` statement and emits NO ``CapacityRoleArn*`` output
     (byte-stable OFF). With ids supplied, the pod role's DefaultPolicy gains a
     ``sts:AssumeRole`` statement scoped to EXACTLY the computed member ARNs (not
     a wildcard) and one ``CapacityRoleArn<acct>`` output per account. The home
     grant's no-unsuppressed-IAM5 guarantee is exercised by ``test_cdk_nag.py``'s
     maximal-flag fold (the ids are in ``_MAXIMAL_FLAGS``).

Mirrors the suite conventions: standalone-stack synth + the
``_IAM_DESCRIPTION_CHARSET`` em-dash guard from ``test_data_lake.py``; the
fresh-app + ``AwsSolutionsChecks`` aspect from ``test_cdk_nag.py``.
"""

from __future__ import annotations

import re

import aws_cdk as cdk
import pytest
from aws_cdk import Aspects
from aws_cdk.assertions import Annotations, Match, Template
from cdk_nag import AwsSolutionsChecks

from lib.bedrock_capacity_member_stack import BedrockCapacityMemberStack
from tests.conftest import make_stack

# The dummy account/region the cred-free gate pins (mirrors tests/conftest.py).
DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"

# A fake-but-concrete home gateway pod-role ARN (the RouteIqStack PodRoleArn the
# operator pastes in when deploying the member stack into a capacity account).
_HOME_POD_ROLE_ARN = "arn:aws:iam::123456789012:role/RouteIqStack-dev-PodRole-AbCdEf123"

# IAM's allowed Description charset: ASCII control trio + printable ASCII +
# Latin-1 supplement. An em-dash (U+2014) is OUTSIDE this set (passes synth but
# FAILS the IAM CREATE API). Mirrors test_data_lake.py.
_IAM_DESCRIPTION_CHARSET = re.compile("^[" + "\t\n\r" + " -~" + "¡-ÿ" + "]*$")


def _dummy_env() -> cdk.Environment:
    return cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION)


def _member_stack(*, app: cdk.App | None = None, **kwargs):
    """Synth a standalone ``BedrockCapacityMemberStack`` (offline)."""
    if app is None:
        app = cdk.App()
    kwargs.setdefault("env_name", "dev")
    kwargs.setdefault("home_pod_role_arn", _HOME_POD_ROLE_ARN)
    return BedrockCapacityMemberStack(
        app, "BedrockCapacityMemberStack-dev", env=_dummy_env(), **kwargs
    )


def _member_template(**kwargs) -> Template:
    return Template.from_stack(_member_stack(**kwargs))


# ---------------------------------------------------------------- member side


def test_member_role_name_is_stable() -> None:
    """The member role is named ``RouteIqBedrockCapacity-<env>`` (predictable ARN).

    The home grant computes ``arn:<part>:iam::<acct>:role/RouteIqBedrockCapacity-
    <env>`` and LiteLLM references that ARN as ``aws_role_name``, so the name MUST
    be stable / not CDK-generated.
    """
    template = _member_template()
    template.has_resource_properties(
        "AWS::IAM::Role",
        {"RoleName": "RouteIqBedrockCapacity-dev"},
    )


def test_member_role_trusts_only_home_pod_role_arn_plain_assume() -> None:
    """Trust is EXACTLY one plain sts:AssumeRole on the home pod-role ARN.

    The C1 analog of ``test_no_irsa.py``: assert NO sts:AssumeRoleWithWebIdentity
    and NO Federated principal anywhere in the assume-role document (RouteIQ's
    Pod-Identity home cluster has no OIDC issuer, so the VSR web-identity branch is
    dropped). The single trust statement's Principal.AWS is the home pod role ARN.
    """
    template = _member_template()
    roles = template.find_resources("AWS::IAM::Role")
    assert len(roles) == 1, roles
    role = next(iter(roles.values()))
    assume_doc = role["Properties"]["AssumeRolePolicyDocument"]
    statements = assume_doc["Statement"]
    assert len(statements) == 1, f"expected exactly one trust statement: {statements}"

    stmt = statements[0]
    assert stmt["Action"] == "sts:AssumeRole", stmt
    assert stmt["Effect"] == "Allow", stmt
    assert stmt["Principal"] == {"AWS": _HOME_POD_ROLE_ARN}, stmt["Principal"]

    # Hard negatives: no web-identity, no federated principal anywhere.
    blob = str(assume_doc)
    assert "sts:AssumeRoleWithWebIdentity" not in blob, blob
    assert "Federated" not in blob, blob


def test_member_role_bedrock_invoke_actions() -> None:
    """The role's policy carries the 4 Bedrock invoke actions on the wildcards."""
    template = _member_template()
    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": Match.object_like(
                {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Action": [
                                        "bedrock:InvokeModel",
                                        "bedrock:InvokeModelWithResponseStream",
                                        "bedrock:Converse",
                                        "bedrock:ConverseStream",
                                    ],
                                    "Effect": "Allow",
                                }
                            )
                        ]
                    )
                }
            )
        },
    )
    # The default resources are the region-wildcard foundation-model +
    # account inference-profile ARNs (an Fn::Join of the partition/account tokens).
    policies = template.find_resources("AWS::IAM::Policy")
    invoke_stmt = next(
        s
        for p in policies.values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
        if isinstance(s.get("Action"), list) and "bedrock:InvokeModel" in s["Action"]
    )
    blob = str(invoke_stmt["Resource"])
    assert ":bedrock:*::foundation-model/*" in blob, blob
    assert ":inference-profile/*" in blob, blob


def test_member_role_has_no_mantle_statements() -> None:
    """No bedrock-mantle action and no Resource ``*`` statement.

    RouteIQ dropped the VSR mantle path entirely; this guards against a copy-paste
    of the ``BedrockMantleInference`` / ``BedrockMantleBearerToken`` block (which
    carried the only ``Resource: "*"`` statement and the only mantle action).
    """
    template = _member_template()
    policies = template.find_resources("AWS::IAM::Policy")
    for policy in policies.values():
        for stmt in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = stmt.get("Action")
            action_list = actions if isinstance(actions, list) else [actions]
            for action in action_list:
                assert not str(action).startswith("bedrock-mantle:"), (
                    f"mantle action leaked into the member policy: {action}"
                )
            assert stmt.get("Resource") != "*", (
                f"a Resource:* statement leaked into the member policy: {stmt}"
            )


def test_member_role_description_is_ascii() -> None:
    """Every IAM role Description is ASCII / Latin-1 only (em-dash guard).

    The VSR source descriptions used em-dashes (U+2014) which pass ``cdk synth``
    but FAIL the IAM CREATE API; this port must use plain hyphens.
    """
    template = _member_template()
    roles = template.find_resources("AWS::IAM::Role")
    for logical, role in roles.items():
        desc = role["Properties"].get("Description")
        if isinstance(desc, str):
            assert _IAM_DESCRIPTION_CHARSET.match(desc), (
                f"IAM role {logical} Description has a char outside IAM's allowed "
                f"Latin-1 set (e.g. an em-dash): {desc!r}"
            )


def test_member_role_explicit_arns_scope_down() -> None:
    """Supplying bedrock_model_arns pins explicit ARNs (the wildcard disappears)."""
    explicit = [
        "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-v1:0",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude",
    ]
    template = _member_template(bedrock_model_arns=explicit)
    policies = template.find_resources("AWS::IAM::Policy")
    invoke_stmt = next(
        s
        for p in policies.values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
        if isinstance(s.get("Action"), list) and "bedrock:InvokeModel" in s["Action"]
    )
    assert invoke_stmt["Resource"] == explicit, invoke_stmt["Resource"]
    # And the region wildcard is gone.
    assert "foundation-model/*" not in str(invoke_stmt["Resource"]), invoke_stmt["Resource"]


def test_member_requires_home_pod_role_arn() -> None:
    """An empty home_pod_role_arn raises (the role would be un-assumable)."""
    app = cdk.App()
    with pytest.raises(ValueError, match="home_pod_role_arn is required"):
        BedrockCapacityMemberStack(
            app,
            "BadMemberStack",
            env=_dummy_env(),
            env_name="dev",
            home_pod_role_arn="",
        )


def test_member_stack_cdk_nag_clean() -> None:
    """No AwsSolutions-* errors survive on the standalone member stack.

    The single IAM5 suppression on the two Bedrock invoke wildcards covers the only
    finding; nothing else fires (plain-assume trust, invoke-only, no mantle).
    """
    app = cdk.App()
    stack = _member_stack(app=app)
    Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        rendered = "\n".join(f"- {e.id}: {str(e.entry.data)[:200]}" for e in errors)
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) on the member stack:\n{rendered}"
        )


# ------------------------------------------------------------------ home side


def test_no_capacity_grant_by_default() -> None:
    """Default make_stack(): no sts:AssumeRole stmt, no CapacityRoleArn output.

    Byte-stable OFF: the empty capacity_account_ids default emits zero home-grant
    statements and zero outputs, so the snapshot stays green without regen.
    """
    template = Template.from_stack(make_stack())

    # No CapacityRoleArn* output.
    outputs = template.find_outputs("*")
    capacity_outputs = [name for name in outputs if name.startswith("CapacityRoleArn")]
    assert capacity_outputs == [], f"unexpected capacity outputs: {capacity_outputs}"

    # No sts:AssumeRole statement anywhere in the pod role's inline policies.
    policies = template.find_resources("AWS::IAM::Policy")
    for policy in policies.values():
        for stmt in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = stmt.get("Action")
            action_list = actions if isinstance(actions, list) else [actions]
            assert "sts:AssumeRole" not in action_list, (
                f"an sts:AssumeRole statement leaked into the default surface: {stmt}"
            )


def test_capacity_grant_when_account_ids_supplied() -> None:
    """Supplying ids => pod-role sts:AssumeRole on the exact member ARNs + outputs.

    The Resource is EXACTLY the two computed RouteIqBedrockCapacity-dev ARNs (an
    explicit list, never a wildcard), and there is one CapacityRoleArn<acct> output
    per account.
    """
    accounts = ["111122223333", "444455556666"]
    # The ARN carries the partition as an {"Ref": "AWS::Partition"} intrinsic, so the
    # rendered Resource is an Fn::Join, not a literal string. Concatenating only the
    # LITERAL join segments (the partition Ref is dropped) yields
    # ``arn:`` + ``:iam::<acct>:role/RouteIqBedrockCapacity-dev``; assert on that.
    expected_literals = [f"arn::iam::{acct}:role/RouteIqBedrockCapacity-dev" for acct in accounts]
    template = Template.from_stack(make_stack(capacity_account_ids=accounts))

    def _flatten(arn) -> str:
        """Concatenate the string segments of an Fn::Join ARN (drop intrinsics)."""
        if isinstance(arn, str):
            return arn
        parts = arn["Fn::Join"][1]
        return "".join(p for p in parts if isinstance(p, str))

    # The home-grant statement lives in the pod role's DefaultPolicy; find the
    # sts:AssumeRole statement and assert its Resource is exactly the two member
    # ARNs (an explicit list, never a wildcard).
    policies = template.find_resources("AWS::IAM::Policy")
    assume_stmts = [
        stmt
        for policy in policies.values()
        for stmt in policy["Properties"]["PolicyDocument"]["Statement"]
        if "sts:AssumeRole"
        in (stmt["Action"] if isinstance(stmt["Action"], list) else [stmt["Action"]])
    ]
    assert len(assume_stmts) == 1, f"expected one sts:AssumeRole statement: {assume_stmts}"
    resources = assume_stmts[0]["Resource"]
    resource_list = resources if isinstance(resources, list) else [resources]
    assert len(resource_list) == 2, resource_list
    flattened = sorted(_flatten(r) for r in resource_list)
    assert flattened == sorted(expected_literals), flattened
    assert "*" not in str(resources), resources  # never a wildcard

    # One CapacityRoleArn<acct> output per account; its Value is the same member ARN.
    outputs = template.find_outputs("*")
    for acct, literal in zip(accounts, expected_literals, strict=True):
        out_name = f"CapacityRoleArn{acct}"
        assert out_name in outputs, f"missing output {out_name}: {list(outputs)}"
        assert _flatten(outputs[out_name]["Value"]) == literal, outputs[out_name]
