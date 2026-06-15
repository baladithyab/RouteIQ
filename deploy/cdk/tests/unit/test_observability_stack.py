"""Unit tests for RouteIqObservabilityStack (the RouteIQ P2 composition root).

The P2 stack wires ConfigStateConstruct + ObservabilityConstruct + (flag-gated)
DataLakeConstruct + (flag-gated) the Athena workgroup, references the P0 routing
log group props-only (NEVER from_lookup), and emits the operator-visible
CfnOutputs (AmpWorkspaceId / AmpRemoteWriteUrl / AppConfigArn / DataLakeBucket /
AthenaWorkgroup).

Synthesised offline against the dummy env (account ``123456789012`` /
``us-west-2``), credential-free: the routing log group is referenced by NAME (the
P0 output or the P0 naming convention), so no AWS API call is made at synth.

Per the suite convention (cdk-resource-count-test-tripwire) property assertions
use ``find_resources`` / ``has_output_value`` / ``Match`` over brittle full counts
on shared resources, except where a count IS the assertion (lake off => 0
Firehose streams).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk import Aspects
from aws_cdk.assertions import Annotations, Match, Template
from cdk_nag import AwsSolutionsChecks


@pytest.fixture(autouse=True)
def _force_hermetic_validator_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the AppConfig validator Lambda's INLINE fallback (mx-86079a).

    The ConfigStateConstruct's validator Lambda takes the Docker ``from_asset``
    bundling path when ``shutil.which("docker")`` finds a docker binary - but the
    cred-free gate must NOT depend on a reachable/healthy Docker daemon. Pinning
    ``_VALIDATOR_ASSET_PATH`` to a non-existent dir makes the construct's
    dual-resolution take the deterministic ``Code.from_inline`` placeholder path
    (the same hermetic contract the snapshot test's ``_DOCKER_ASSET_PATH_GLOBALS``
    mechanism enforces), so synth proceeds with no Docker.
    """
    import lib.config_state_construct as cfg

    missing = str(Path(__file__).parent / "_does_not_exist")
    monkeypatch.setattr(cfg, "_VALIDATOR_ASSET_PATH", missing)


DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"

# The P0 naming convention the stack derives when no name is supplied (matches
# EksClusterConstruct.routing_log_group_name for env "dev").
_DERIVED_LOG_GROUP_NAME = "/aws/containerinsights/routeiq-dev/routeiq-routing"

# IAM's allowed Description charset (ASCII control trio + printable ASCII +
# Latin-1 supplement). An em-dash (U+2014) is OUTSIDE this set.
_IAM_DESCRIPTION_CHARSET = re.compile("^[" + "\t\n\r" + " -~" + "¡-ÿ" + "]*$")


def _stack(**flags: Any):
    from lib.routeiq_observability_stack import RouteIqObservabilityStack

    app = cdk.App()
    env_name = flags.pop("env_name", "dev")
    return RouteIqObservabilityStack(
        app,
        f"RouteIqObservabilityStack-{env_name}",
        env=cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION),
        env_name=env_name,
        **flags,
    )


def _template(**flags: Any) -> Template:
    return Template.from_stack(_stack(**flags))


# ----------------------------------------------------------------- composition


def test_wires_appconfig_and_observability_by_default() -> None:
    """Default surface: AppConfig + AMP + SNS + metric filters, no lake, no AMG."""
    template = _template()
    # AppConfig application/environment/profile.
    template.resource_count_is("AWS::AppConfig::Application", 1)
    template.has_resource_properties("AWS::AppConfig::Application", {"Name": "routeiq"})
    # AMP workspace + TLS SNS topic.
    template.resource_count_is("AWS::APS::Workspace", 1)
    template.has_resource_properties("AWS::SNS::Topic", {"TopicName": "routeiq-dev-oncall"})
    # Per-model dimensioned metric filter exists (the load-bearing P2 surface).
    filters = template.find_resources("AWS::Logs::MetricFilter")
    by_model = [
        f
        for f in filters.values()
        if f["Properties"]["MetricTransformations"][0].get("MetricName")
        == "routing_latency_ms_by_model"
    ]
    assert len(by_model) == 1, "expected the per-model dimensioned filter"
    # AMG + data lake off by default.
    template.resource_count_is("AWS::Grafana::Workspace", 0)
    template.resource_count_is("AWS::KinesisFirehose::DeliveryStream", 0)
    template.resource_count_is("AWS::Athena::WorkGroup", 0)


def test_references_p0_log_group_by_derived_name_not_from_lookup() -> None:
    """The metric filters attach to the P0 group referenced by the derived name.

    The routing log group is IMPORTED by NAME (cred-free), not created here, so no
    ``AWS::Logs::LogGroup`` resource carries the routing-group name (the only
    CDK-owned log group in this stack is the validator Lambda's own group, which is
    unrelated). The per-model filter's LogGroupName is the plain P0 convention name.
    """
    template = _template()
    # No CDK-owned log group carries the imported routing-group name: importing by
    # name does NOT emit a LogGroup resource (the validator Lambda's group is the
    # only log group, and it is a /aws/lambda/* group).
    log_groups = template.find_resources("AWS::Logs::LogGroup")
    routing_groups = [
        lid
        for lid, lg in log_groups.items()
        if lg["Properties"].get("LogGroupName") == _DERIVED_LOG_GROUP_NAME
    ]
    assert not routing_groups, (
        f"routing log group must be IMPORTED (not created); found {routing_groups}"
    )
    # The metric filter targets the derived P0 name (a plain string, not a Ref to a
    # CDK-owned group -> proves the import path).
    template.has_resource_properties(
        "AWS::Logs::MetricFilter",
        Match.object_like({"LogGroupName": _DERIVED_LOG_GROUP_NAME}),
    )


def test_explicit_routing_log_group_name_is_used() -> None:
    """An operator-supplied routing_log_group_name overrides the derived name."""
    explicit = "/aws/containerinsights/routeiq-prod-cluster/routeiq-routing"
    template = _template(env_name="prod", routing_log_group_name=explicit)
    template.has_resource_properties(
        "AWS::Logs::MetricFilter",
        Match.object_like({"LogGroupName": explicit}),
    )


# --------------------------------------------------------------------- outputs


def test_core_outputs_present() -> None:
    """AmpWorkspaceId / AmpRemoteWriteUrl / AppConfigArn / AlarmTopicArn outputs."""
    template = _template()
    outputs = template.find_outputs("*")
    for required in (
        "AmpWorkspaceId",
        "AmpRemoteWriteUrl",
        "AppConfigArn",
        "AppConfigProfileArn",
        "AppConfigApplicationId",
        "AlarmTopicArn",
    ):
        assert required in outputs, f"missing CfnOutput {required!r}; got {list(outputs)}"


def test_lake_and_athena_outputs_only_when_lake_enabled() -> None:
    """DataLakeBucket / AthenaWorkgroup outputs appear only with the lake on."""
    off = _template().find_outputs("*")
    assert "DataLakeBucket" not in off
    assert "AthenaWorkgroup" not in off

    on = _template(enable_data_lake=True).find_outputs("*")
    assert "DataLakeBucket" in on
    assert "AthenaWorkgroup" in on
    assert "AthenaDatabase" in on


# ----------------------------------------------------------------- data lake on


def test_data_lake_enabled_emits_firehose_glue_and_athena_workgroup() -> None:
    """enable_data_lake=true synthesises the Firehose/Glue lake + Athena workgroup."""
    template = _template(enable_data_lake=True)
    template.resource_count_is("AWS::KinesisFirehose::DeliveryStream", 1)
    template.resource_count_is("AWS::Glue::Table", 1)
    template.resource_count_is("AWS::Glue::Database", 1)
    template.resource_count_is("AWS::Logs::SubscriptionFilter", 1)
    template.resource_count_is("AWS::Athena::WorkGroup", 1)
    # The lake subscription reads the SAME P0 group the metric filters do.
    template.has_resource_properties(
        "AWS::Logs::SubscriptionFilter",
        Match.object_like({"LogGroupName": _DERIVED_LOG_GROUP_NAME}),
    )


def test_athena_workgroup_results_are_kms_encrypted_and_enforced() -> None:
    """The workgroup enforces config + SSE-KMS query results (AwsSolutions-ATH1)."""
    template = _template(enable_data_lake=True)
    template.has_resource_properties(
        "AWS::Athena::WorkGroup",
        {
            "Name": "routeiq-dev-routing-decisions",
            "WorkGroupConfiguration": Match.object_like(
                {
                    "EnforceWorkGroupConfiguration": True,
                    "ResultConfiguration": Match.object_like(
                        {
                            "EncryptionConfiguration": Match.object_like(
                                {"EncryptionOption": "SSE_KMS"}
                            )
                        }
                    ),
                }
            ),
        },
    )


# ----------------------------------------------------------------------- AMG on


def test_amg_enabled_emits_grafana_workspace_and_endpoint_output() -> None:
    """enable_amg=true synthesises the Grafana workspace + the endpoint output."""
    template = _template(enable_amg=True)
    template.resource_count_is("AWS::Grafana::Workspace", 1)
    assert "AmgWorkspaceEndpoint" in template.find_outputs("*")


# ------------------------------------------------------- AppConfig at the root


def test_appconfig_full_graph_wired_at_composition_root() -> None:
    """The composition root wires the full AppConfig graph the P2 mandate names.

    Application + Environment + ConfigurationProfile(LAMBDA validator) +
    DeploymentStrategy(LINEAR 20/12/5) + HostedConfigurationVersion + Deployment,
    plus the validator Lambda. Exactly one application / strategy / deployment.
    """
    template = _template()
    template.resource_count_is("AWS::AppConfig::Application", 1)
    template.resource_count_is("AWS::AppConfig::Environment", 1)
    template.resource_count_is("AWS::AppConfig::ConfigurationProfile", 1)
    template.resource_count_is("AWS::AppConfig::DeploymentStrategy", 1)
    template.resource_count_is("AWS::AppConfig::HostedConfigurationVersion", 1)
    template.resource_count_is("AWS::AppConfig::Deployment", 1)
    template.has_resource_properties("AWS::AppConfig::Application", {"Name": "routeiq"})
    template.has_resource_properties("AWS::AppConfig::Environment", {"Name": "dev"})


def test_appconfig_profile_lambda_validator_at_root() -> None:
    """The profile at the root carries the Type=LAMBDA validator (Content=ARN)."""
    template = _template()
    template.has_resource_properties(
        "AWS::AppConfig::ConfigurationProfile",
        {
            "Name": "router-yaml",
            "Type": "AWS.Freeform",
            "LocationUri": "hosted",
            "Validators": Match.array_with(
                [
                    Match.object_like(
                        {
                            "Type": "LAMBDA",
                            "Content": Match.object_like({"Fn::GetAtt": Match.array_with(["Arn"])}),
                        }
                    )
                ]
            ),
        },
    )


def test_appconfig_strategy_linear_20_12_5_at_root() -> None:
    """The deployment strategy at the root is LINEAR 20 / 12min / 5min final bake."""
    template = _template()
    template.has_resource_properties(
        "AWS::AppConfig::DeploymentStrategy",
        {
            "Name": "Linear20Pct3Min",
            "DeploymentDurationInMinutes": 12,
            "GrowthFactor": 20,
            "GrowthType": "LINEAR",
            "FinalBakeTimeInMinutes": 5,
            "ReplicateTo": "NONE",
        },
    )


def test_validator_invoke_permission_for_appconfig_at_root() -> None:
    """The validator lambda:InvokeFunction permission for appconfig.amazonaws.com.

    Exactly one appconfig-principled Lambda::Permission, SourceArn-scoped (an
    Fn::Join token to the profile-only ARN form).
    """
    template = _template()
    template.has_resource_properties(
        "AWS::Lambda::Permission",
        {"Action": "lambda:InvokeFunction", "Principal": "appconfig.amazonaws.com"},
    )
    perms = template.find_resources("AWS::Lambda::Permission")
    appconfig_perms = [
        p for p in perms.values() if p["Properties"].get("Principal") == "appconfig.amazonaws.com"
    ]
    assert len(appconfig_perms) == 1, appconfig_perms
    src_arn = appconfig_perms[0]["Properties"].get("SourceArn")
    assert isinstance(src_arn, dict) and "Fn::Join" in src_arn, src_arn


# --------------------------------------------- per-model dimension key (the fix)


def test_per_model_filter_keyed_on_gen_ai_response_model_not_selected_model() -> None:
    """The dimensioned filter keys on $.["gen_ai.response.model"], NOT $.selected_model.

    The load-bearing telemetry-key fix (telemetry_contracts.py:673): RouteIQ emits
    the OTel ``gen_ai.response.model`` field, never a flat ``selected_model``. A
    filter keyed on ``$.selected_model`` would match ZERO RouteIQ events. The key
    contains dots so it MUST be bracket-quoted. AWS forbids DefaultValue on a
    dimensioned filter, so the dimensioned filter omits it.
    """
    template = _template()
    filters = template.find_resources("AWS::Logs::MetricFilter")
    by_model = [
        f
        for f in filters.values()
        if f["Properties"]["MetricTransformations"][0].get("MetricName")
        == "routing_latency_ms_by_model"
    ]
    assert len(by_model) == 1, "expected exactly one per-model dimensioned filter"
    transform = by_model[0]["Properties"]["MetricTransformations"][0]
    assert transform.get("Dimensions") == [
        {"Key": "model", "Value": '$.["gen_ai.response.model"]'}
    ], transform.get("Dimensions")
    # The negative assertion: the dimension value is NOT the VSR $.selected_model.
    assert transform["Dimensions"][0]["Value"] != "$.selected_model"
    # AWS forbids DefaultValue on a dimensioned filter.
    assert "DefaultValue" not in transform, transform
    # And the WHOLE stack template carries no $.selected_model dimension anywhere.
    blob = json.dumps(template.to_json())
    assert "$.selected_model" not in blob, "stack must not key any CW dimension on selected_model"


def test_aggregate_filters_keep_default_value_zero() -> None:
    """The aggregate (no-dimension) filters keep DefaultValue=0 (the alarm series)."""
    template = _template()
    filters = template.find_resources("AWS::Logs::MetricFilter")
    for metric_name in ("routing_latency_ms", "router_error_log_count"):
        matches = [
            f
            for f in filters.values()
            if f["Properties"]["MetricTransformations"][0].get("MetricName") == metric_name
        ]
        assert len(matches) == 1, (metric_name, matches)
        transform = matches[0]["Properties"]["MetricTransformations"][0]
        assert "Dimensions" not in transform, (metric_name, transform)
        assert transform.get("DefaultValue") == 0, (metric_name, transform)


def test_router_error_filter_selects_on_level_error_the_emitted_field() -> None:
    """The RouterErrorFilter pattern selects on ``$.level = "error"`` (RouteIQ-731c).

    The load-bearing P2-hardening assertion: the gateway error path now emits a
    structured JSON line carrying a top-level LOWERCASED ``level == "error"``
    (observability.py ``emit_error_log``), so the alarm can finally fire. This
    test pins the CDK filter pattern to the field that emitter produces. The two
    halves (emitter emits ``level``, filter scans ``level``) together close the
    seed: before, the filter scanned a field no emitter wrote and the alarm was
    inert.
    """
    template = _template()
    filters = template.find_resources("AWS::Logs::MetricFilter")
    error_filters = [
        f
        for f in filters.values()
        if f["Properties"]["MetricTransformations"][0].get("MetricName") == "router_error_log_count"
    ]
    assert len(error_filters) == 1, "expected exactly one router-error filter"
    pattern = error_filters[0]["Properties"]["FilterPattern"]
    # The CloudWatch JSON metric-filter pattern for a string match is
    # ``{ $.level = "error" }``; assert it keys on the lowercased ``level`` field.
    assert "$.level" in pattern, pattern
    assert '"error"' in pattern, pattern
    # Negative: the lowercased literal the emitter writes, NOT the Python uppercase
    # ``levelname`` (the false-comment trap the construct doc fixed).
    assert '"ERROR"' not in pattern, pattern


# ------------------------------------------------ alarms -> the single SNS topic


def test_all_alarms_wire_to_the_single_sns_topic() -> None:
    """Every CW alarm pages exactly ONE topic - the single on-call SNS topic.

    The RouteIQ port builds 3 RouteIQ-relevant alarms (routing-latency ceiling,
    router-error-count, routing-latency anomaly); the VSR ECS/blue-green alarm
    family (p99/jailbreak/hallucination/cache/throttle/replay/opus-share) is
    DROPPED by design (observability_construct docstring). Whatever the count, all
    alarm + OK actions resolve to the SAME single Topic Ref, and there is exactly
    one SNS topic in the stack.
    """
    template = _template()
    template.resource_count_is("AWS::SNS::Topic", 1)
    topic_ids = list(template.find_resources("AWS::SNS::Topic").keys())
    assert len(topic_ids) == 1
    topic_id = topic_ids[0]

    alarms = template.find_resources("AWS::CloudWatch::Alarm")
    assert alarms, "expected at least one CloudWatch alarm"
    referenced_topics: set[str] = set()
    for lid, alarm in alarms.items():
        props = alarm["Properties"]
        actions = (props.get("AlarmActions") or []) + (props.get("OKActions") or [])
        assert props.get("AlarmActions"), f"{lid} missing AlarmActions"
        assert props.get("OKActions"), f"{lid} missing OKActions"
        for action in actions:
            # Each action is a {"Ref": "<topic logical id>"} to the single topic.
            if isinstance(action, dict) and "Ref" in action:
                referenced_topics.add(action["Ref"])
    assert referenced_topics == {topic_id}, (
        f"every alarm must page the single on-call topic {topic_id}; got refs {referenced_topics}"
    )


def test_three_routeiq_alarms_present() -> None:
    """The RouteIQ port emits its 3 routing alarms (the VSR 9-alarm family trimmed).

    Documents the intentional divergence: the VSR ObservabilityConstruct has 9
    alarms; RouteIQ keeps the 3 that read CW-Logs-native routing metrics (latency
    ceiling + error-count + latency anomaly) and drops the ECS/blue-green family.
    """
    template = _template()
    alarms = template.find_resources("AWS::CloudWatch::Alarm")
    assert len(alarms) == 3, (
        f"expected the 3 RouteIQ routing alarms (latency/error/anomaly); got {len(alarms)}"
    )


# ------------------------------------- routing log group is CDK-created at P0


def test_routing_log_group_is_imported_not_created_here() -> None:
    """The dimensioned filter attaches to the P0 CDK-created group, referenced by
    NAME (it pre-exists the filter at deploy time).

    A CFN MetricFilter requires its target group to ALREADY EXIST. P0 CDK-creates
    the RoutingLogGroup; this stack imports it by name (cred-free) and does NOT
    re-create it. So no CDK-owned LogGroup in this stack carries the routing-group
    name (the only owned group is the validator Lambda's /aws/lambda/* group), and
    the filter's LogGroupName is the plain P0 convention string (not a Ref to a
    group this stack owns).
    """
    template = _template()
    log_groups = template.find_resources("AWS::Logs::LogGroup")
    routing_owned = [
        lid
        for lid, lg in log_groups.items()
        if lg["Properties"].get("LogGroupName") == _DERIVED_LOG_GROUP_NAME
    ]
    assert not routing_owned, (
        f"routing log group must be the P0-CDK-created group imported by NAME, not "
        f"re-created here; found owned {routing_owned}"
    )
    # The per-model filter's LogGroupName is the literal P0 name (proves the import,
    # and proves the filter targets a group that pre-exists it).
    template.has_resource_properties(
        "AWS::Logs::MetricFilter",
        Match.object_like(
            {
                "LogGroupName": _DERIVED_LOG_GROUP_NAME,
                "MetricTransformations": Match.array_with(
                    [Match.object_like({"MetricName": "routing_latency_ms_by_model"})]
                ),
            }
        ),
    )


# --------------------------------------------------------- no secrets / accounts


def test_no_hardcoded_secrets_or_account_ids_maximal() -> None:
    """No literal secrets and no real account-ids leak into the maximal template.

    The only 12-digit account literal is the dummy env account (123456789012); no
    sk-.../AKIA.../ASIA... credential literal and no other account-id appears.
    """
    template = _template(enable_amg=True, enable_data_lake=True, notify_emails=["x@y.example"])
    blob = json.dumps(template.to_json())
    # No credential literals.
    assert not re.search(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}", blob), "OpenAI/Anthropic key literal"
    assert not re.search(r"\bAKIA[0-9A-Z]{16}\b", blob), "AWS access-key-id literal"
    assert not re.search(r"\bASIA[0-9A-Z]{16}\b", blob), "AWS temp access-key-id literal"
    # No account-id other than the dummy env account.
    others = {m for m in re.findall(r"\b\d{12}\b", blob) if m != DUMMY_ACCOUNT}
    assert not others, f"unexpected 12-digit account-id literal(s): {others}"


# ----------------------------------------- SNS topic TLS-enforced at the root


def test_sns_topic_tls_enforced() -> None:
    """The single on-call topic denies non-TLS publish/subscribe (DenyInsecureTransport)."""
    template = _template()
    template.has_resource_properties(
        "AWS::SNS::TopicPolicy",
        {
            "PolicyDocument": Match.object_like(
                {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Sid": "DenyInsecureTransport",
                                    "Effect": "Deny",
                                    "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                                }
                            )
                        ]
                    )
                }
            )
        },
    )


# ------------------------------------------------------------------- cdk-nag


def _synth_with_nag(**flags: Any) -> cdk.Stack:
    stack = _stack(**flags)
    Aspects.of(stack.node.root).add(AwsSolutionsChecks(verbose=True))
    return stack


def _render(errors: Any) -> str:
    return "\n".join(f"- {e.id}: {str(e.entry.data)[:200]}" for e in errors)


def test_no_unsuppressed_aws_solutions_errors_maximal() -> None:
    """No AwsSolutions-* errors survive on the maximal surface (AMG + lake + emails)."""
    stack = _synth_with_nag(
        enable_amg=True,
        enable_data_lake=True,
        notify_emails=["oncall@example.com"],
    )
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) (maximal surface):\n"
            f"{_render(errors)}"
        )


def test_no_validation_failures_maximal() -> None:
    """No CdkNagValidationFailure errors survive on the maximal surface."""
    stack = _synth_with_nag(
        enable_amg=True,
        enable_data_lake=True,
        notify_emails=["oncall@example.com"],
    )
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("CdkNagValidationFailure.*")
    )
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed CdkNagValidationFailure error(s) (maximal surface):\n"
            f"{_render(errors)}"
        )


# ------------------------------------------------------------------------ ASCII


def test_iam_role_descriptions_are_ascii() -> None:
    """Every IAM role Description is ASCII / Latin-1 only (P0 section 4.5)."""
    template = _template(enable_amg=True, enable_data_lake=True)
    roles = template.find_resources("AWS::IAM::Role")
    for logical, role in roles.items():
        desc = role["Properties"].get("Description")
        if isinstance(desc, str):
            assert _IAM_DESCRIPTION_CHARSET.match(desc), (
                f"IAM role {logical} Description has a char outside IAM's allowed "
                f"Latin-1 set: {desc!r}"
            )
