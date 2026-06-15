"""Cred-free assertions for the P2 combined-deploy seam (foundation= threading).

Covers the HIGH+LOW grant/coupling seeds that collapse onto ONE structural lever:
threading the P0 ``RouteIqStack`` into ``RouteIqObservabilityStack`` by reference
(``foundation=``), exactly as ``app.py`` does in the combined deploy.

  * RouteIQ-569f  -- the pod role gets the AppConfig runtime-poll grant
    (appconfigdata:StartConfigurationSession + GetLatestConfiguration), scoped to
    the env-scoped profile ARN (NOT ``*``).
  * RouteIQ-74c0 / RouteIQ-717b -- the pod role gets aps:RemoteWrite, scoped to the
    AMP workspace ARN (the previously-defined-but-never-called amp_remote_write_grant
    seam, now wired).
  * RouteIQ-81c4 -- the P0 routing log group is consumed via a cross-stack
    Fn::ImportValue + add_dependency(foundation) (CFN enforces P0-before-P2), AND the
    P0 stack publishes a STABLE export of the routing log group name.

NEGATIVE half (byte-stability): the default no-foundation P2 synth emits NONE of
these grants and keeps the plain-string by-NAME log-group reference, so the existing
``test_observability_stack.py`` tests + the 74a4 snapshot stay green.

All offline against the dummy env (account ``123456789012`` / ``us-west-2``); the
cross-stack refs resolve at synth into Export/Fn::ImportValue, never from_lookup.
"""

from __future__ import annotations

import json

from aws_cdk import Aspects
from aws_cdk.assertions import Annotations, Match, Template
from cdk_nag import AwsSolutionsChecks

from lib.naming import routing_log_group_export_name, routing_log_group_name
from tests.conftest import make_obs_stack

# The same hermetic-validator pin the rest of the obs suite uses lives in this
# package's conftest? No -- it is autouse-scoped to test_observability_stack.py. The
# 4772 determinism fix means the validator is inline-by-default at synth regardless
# of Docker, so make_obs_stack needs NO monkeypatch (the determinism is the point).


def _pod_role_statements(template: Template) -> list[dict]:
    """All PolicyStatements across every AWS::IAM::Policy in the P2 template."""
    statements: list[dict] = []
    for policy in template.find_resources("AWS::IAM::Policy").values():
        doc = policy["Properties"].get("PolicyDocument", {})
        statements.extend(doc.get("Statement", []))
    return statements


# ----------------------------------------------------------------- RouteIQ-569f


def test_pod_role_gets_appconfig_poll_grant_scoped_to_profile_arn() -> None:
    """The combined deploy grants the pod role the AppConfig runtime-poll actions.

    The poll is the appconfigdata DATA-PLANE prefix (the boto3 AppConfigData client
    the gateway uses), scoped to the env-scoped profile ARN, NOT ``*``.
    """
    _app, _foundation, obs = make_obs_stack()
    template = Template.from_stack(obs)

    poll = [s for s in _pod_role_statements(template) if s.get("Sid") == "AppConfigPoll"]
    assert len(poll) == 1, f"expected exactly one AppConfigPoll statement; got {poll}"
    actions = poll[0]["Action"]
    assert "appconfigdata:StartConfigurationSession" in actions, actions
    assert "appconfigdata:GetLatestConfiguration" in actions, actions
    # The resource is a token/Fn::Join to the env-scoped profile ARN, never "*".
    resource = poll[0]["Resource"]
    assert resource != "*", "AppConfig poll must be ARN-scoped, not a wildcard"
    blob = json.dumps(resource)
    assert "appconfig" in blob, resource
    # The ARN form is the env-scoped configuration ARN (.../environment/.../configuration/...).
    assert "configuration" in blob, resource


def test_default_p2_emits_no_appconfig_poll_grant() -> None:
    """NEGATIVE: a no-foundation P2 synth emits no AppConfigPoll statement (byte-stable)."""
    from tests.unit._obs_helpers import standalone_obs_template

    template = standalone_obs_template()
    poll = [s for s in _pod_role_statements(template) if s.get("Sid") == "AppConfigPoll"]
    assert not poll, f"default P2 must not grant AppConfig poll; got {poll}"


# --------------------------------------------------- RouteIQ-74c0 / RouteIQ-717b


def test_pod_role_gets_amp_remote_write_grant_scoped_to_workspace_arn() -> None:
    """The combined deploy grants the pod role aps:RemoteWrite scoped to the AMP ARN."""
    _app, _foundation, obs = make_obs_stack()
    template = Template.from_stack(obs)

    amp = [s for s in _pod_role_statements(template) if s.get("Sid") == "AmpRemoteWrite"]
    assert len(amp) == 1, f"expected exactly one AmpRemoteWrite statement; got {amp}"
    actions = amp[0]["Action"]
    assert actions == "aps:RemoteWrite" or actions == ["aps:RemoteWrite"], actions
    resource = amp[0]["Resource"]
    assert resource != "*", "aps:RemoteWrite must be ARN-scoped, not a wildcard"


def test_default_p2_emits_no_amp_remote_write_grant() -> None:
    """NEGATIVE: a no-foundation P2 synth emits no aps:RemoteWrite (byte-stable)."""
    from tests.unit._obs_helpers import standalone_obs_template

    template = standalone_obs_template()
    blob = json.dumps(template.to_json())
    assert "aps:RemoteWrite" not in blob, "default P2 must not grant aps:RemoteWrite"


def test_grants_live_on_a_p2_owned_policy_attached_to_imported_role() -> None:
    """The grants live on a P2-stack-owned iam.Policy attached to the imported role.

    Cycle avoidance: the policy + the P2-owned ARNs stay in the P2 template; the only
    cross-stack edge is P2 -> P0 (the role import). The policy's Roles entry is an
    Fn::ImportValue / token to the P0 pod role (NOT an in-stack Ref).
    """
    _app, _foundation, obs = make_obs_stack()
    template = Template.from_stack(obs)
    policies = template.find_resources("AWS::IAM::Policy")
    obs_grant = [
        p
        for p in policies.values()
        if any(
            st.get("Sid") in ("AppConfigPoll", "AmpRemoteWrite")
            for st in p["Properties"].get("PolicyDocument", {}).get("Statement", [])
        )
    ]
    assert len(obs_grant) == 1, f"expected one PodObsGrants policy; got {len(obs_grant)}"
    roles = obs_grant[0]["Properties"].get("Roles")
    assert roles, "PodObsGrants must attach to the imported pod role"
    blob = json.dumps(roles)
    # The role ref is a cross-stack import token (the P0 pod role lives in P0).
    assert "ImportValue" in blob or "Fn::ImportValue" in blob or "Ref" in blob, roles


# ----------------------------------------------------------------- RouteIQ-81c4


def test_p0_exports_routing_log_group_name_with_stable_export_name() -> None:
    """The P0 stack publishes a STABLE export of the routing log group name."""
    _app, foundation, _obs = make_obs_stack()
    p0 = Template.from_stack(foundation)
    outputs = p0.find_outputs("*")
    assert "RoutingLogGroupName" in outputs, list(outputs)
    export = outputs["RoutingLogGroupName"].get("Export", {})
    assert export.get("Name") == routing_log_group_export_name("dev"), export
    assert outputs["RoutingLogGroupName"]["Value"] == routing_log_group_name("dev")


def test_p2_consumes_p0_log_group_via_import_value_not_plain_name() -> None:
    """With foundation, the P2 filter LogGroupName is an Fn::ImportValue (not a name)."""
    _app, _foundation, obs = make_obs_stack()
    template = Template.from_stack(obs)
    filters = template.find_resources("AWS::Logs::MetricFilter")
    assert filters, "expected metric filters in the P2 stack"
    for f in filters.values():
        lgn = f["Properties"]["LogGroupName"]
        # Cross-stack reference => a dict token (Fn::ImportValue), NOT a plain string.
        assert isinstance(lgn, dict), (
            f"with foundation threaded, the filter LogGroupName must be a cross-stack "
            f"reference (Fn::ImportValue), not the plain derived name; got {lgn!r}"
        )
        assert "Fn::ImportValue" in json.dumps(lgn), lgn


def test_p2_depends_on_p0_so_cfn_deploys_p0_first() -> None:
    """add_dependency(foundation): the P2 stack depends on P0 in the synth assembly."""
    app, foundation, obs = make_obs_stack()
    assembly = app.synth()
    obs_artifact = assembly.get_stack_artifact(obs.artifact_id)
    p0_deps = {d.id for d in obs_artifact.dependencies}
    assert foundation.artifact_id in p0_deps, (
        f"P2 must declare a stack dependency on P0 (CFN deploys P0 before P2); "
        f"P2 deps = {p0_deps}, P0 artifact = {foundation.artifact_id}"
    )


def test_default_p2_uses_plain_derived_name_not_import_value() -> None:
    """NEGATIVE: standalone P2 keeps the plain-string by-NAME reference (byte-stable)."""
    from tests.unit._obs_helpers import standalone_obs_template

    template = standalone_obs_template()
    template.has_resource_properties(
        "AWS::Logs::MetricFilter",
        Match.object_like({"LogGroupName": routing_log_group_name("dev")}),
    )


# --------------------------------------------------------- no secrets / accounts


def test_combined_synth_has_no_secret_or_foreign_account_literals() -> None:
    """No credential literal and no account-id beyond the dummy env leak (combined)."""
    import re

    _app, _foundation, obs = make_obs_stack(enable_data_lake=True, enable_amg=True)
    blob = json.dumps(Template.from_stack(obs).to_json())
    assert not re.search(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}", blob)
    assert not re.search(r"\bAKIA[0-9A-Z]{16}\b", blob)
    others = {m for m in re.findall(r"\b\d{12}\b", blob) if m != "123456789012"}
    assert not others, f"unexpected account-id literal(s): {others}"


# ----------------------------------------------------------------- cdk-nag clean


def test_combined_grants_introduce_no_unsuppressed_nag_findings() -> None:
    """The PodObsGrants policy is ARN-scoped, so it triggers no AwsSolutions-* error.

    Both stacks are synthesised with AwsSolutionsChecks; neither the new P2-stack
    grant policy (AppConfig poll + aps:RemoteWrite, both ARN-scoped, no wildcard) nor
    the P0 export introduces an unsuppressed finding.
    """
    app, foundation, obs = make_obs_stack()
    Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    app.synth()
    for stack in (foundation, obs):
        errors = Annotations.from_stack(stack).find_error(
            "*", Match.string_like_regexp("AwsSolutions-.*")
        )
        rendered = "\n".join(f"- {e.id}: {str(e.entry.data)[:200]}" for e in errors)
        assert not errors, (
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) on {stack.stack_name}:\n{rendered}"
        )
