"""Unit tests for ObservabilityConstruct (RouteIQ P2, ADR-0027).

Asserts the P2 observability surface: the AMP workspace, the FLAG-GATED-OFF AMG
workspace + its data-source role, the TLS-enforced SNS on-call topic, and - the
load-bearing assertion - the PER-MODEL dimensioned CloudWatch metric filter keyed
on the OTel ``$.["gen_ai.response.model"]`` field (NOT ``$.selected_model``, which
RouteIQ never emits), with NO ``DefaultValue`` (AWS forbids it on a dimensioned
filter), alongside the aggregate filters that DO keep ``DefaultValue=0``.

The construct is synthesised offline into a throwaway ``Stack`` against the dummy
env (account ``123456789012`` / ``us-west-2``), with the P0 routing log group
re-imported by NAME via ``logs.LogGroup.from_log_group_name`` - the cred-free,
props-only path the composition root uses (D4), NOT ``from_lookup``.

Per mulch ``cdk-resource-count-test-tripwire`` the property assertions use
``find_resources`` / ``has_resource_properties`` / ``Match.object_like`` over
brittle full counts on shared resources, except where a count IS the assertion
(AMG off => 0 Grafana workspaces).
"""

from __future__ import annotations

import re
from typing import Any

import aws_cdk as cdk
from aws_cdk import aws_logs as logs
from aws_cdk.assertions import Match, Template

# The P0-shaped routing log group name the obs construct imports (matches
# EksClusterConstruct.routing_log_group_name for env "dev").
_ROUTING_LOG_GROUP_NAME = "/aws/containerinsights/routeiq-dev/routeiq-routing"

# The OTel telemetry-contract dimension key the per-model filter MUST use.
_GEN_AI_RESPONSE_MODEL = '$.["gen_ai.response.model"]'

# IAM's allowed Description charset (ASCII control trio + printable ASCII +
# Latin-1 supplement). An em-dash (U+2014) is OUTSIDE this set - the guarded
# failure mode. Built from \u escapes so this source stays plain ASCII.
_IAM_DESCRIPTION_CHARSET = re.compile("^[" + "\t\n\r" + " -~" + "\u00a1-\u00ff" + "]*$")

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"


def _obs_template(**flags: Any) -> Template:
    """Synthesise a throwaway Stack hosting an ObservabilityConstruct (offline).

    Imports the P0 routing log group by NAME (cred-free) and passes it in as the
    ``routing_log_group`` prop. Deferred import so this module loads cleanly while
    the construct is authored.
    """
    from lib.observability_construct import ObservabilityConstruct

    app = cdk.App()
    env_name = flags.pop("env_name", "dev")
    stack = cdk.Stack(
        app,
        f"RouteIqObsTest-{env_name}",
        env=cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION),
    )
    log_group = logs.LogGroup.from_log_group_name(
        stack, "ImportedRoutingLogGroup", _ROUTING_LOG_GROUP_NAME
    )
    ObservabilityConstruct(
        stack,
        "ObservabilityConstruct",
        env_name=env_name,
        routing_log_group=log_group,
        **flags,
    )
    return Template.from_stack(stack)


# --------------------------------------------------------------------------- AMP


def test_amp_workspace_with_alias() -> None:
    """An AMP workspace exists with the env-scoped alias."""
    template = _obs_template()
    template.has_resource_properties(
        "AWS::APS::Workspace",
        {"Alias": "routeiq-dev"},
    )


# --------------------------------------------------------------------------- AMG


def test_amg_off_by_default() -> None:
    """AMG is flag-gated OFF: no Grafana workspace + no data-source role on default."""
    template = _obs_template()
    template.resource_count_is("AWS::Grafana::Workspace", 0)
    # The AmgWorkspaceRole is only built when AMG is on; assert no grafana role.
    roles = template.find_resources("AWS::IAM::Role")
    grafana_assumers = [
        lid
        for lid, r in roles.items()
        if "grafana.amazonaws.com" in str(r["Properties"].get("AssumeRolePolicyDocument", {}))
    ]
    assert not grafana_assumers, f"unexpected Grafana data-source role: {grafana_assumers}"


def test_amg_on_emits_workspace_and_role() -> None:
    """AMG on: exactly one Grafana workspace with AWS_SSO + SERVICE_MANAGED + 3 sources."""
    template = _obs_template(enable_amg=True)
    template.resource_count_is("AWS::Grafana::Workspace", 1)
    template.has_resource_properties(
        "AWS::Grafana::Workspace",
        {
            "AccountAccessType": "CURRENT_ACCOUNT",
            "AuthenticationProviders": ["AWS_SSO"],
            "PermissionType": "SERVICE_MANAGED",
            "DataSources": Match.array_with(["PROMETHEUS", "XRAY", "CLOUDWATCH"]),
            "Name": "routeiq-dev",
        },
    )


def test_amg_role_has_three_sids() -> None:
    """The AMG data-source role carries the AmpQuery / CloudWatchRead / XRayRead sids."""
    template = _obs_template(enable_amg=True)
    policies = template.find_resources("AWS::IAM::Policy")
    sids = set()
    for p in policies.values():
        for stmt in p["Properties"]["PolicyDocument"].get("Statement", []):
            if "Sid" in stmt:
                sids.add(stmt["Sid"])
    for required in ("AmpQuery", "CloudWatchRead", "XRayRead"):
        assert required in sids, f"AMG role missing sid {required!r}; got {sids}"


# ---------------------------------------------------------------- metric filters


def _metric_filters(template: Template) -> dict[str, Any]:
    return template.find_resources("AWS::Logs::MetricFilter")


def _by_metric_name(template: Template, metric_name: str) -> dict[str, Any]:
    """Return the single metric filter whose transformation MetricName matches."""
    matches = []
    for r in _metric_filters(template).values():
        transforms = r["Properties"].get("MetricTransformations", [])
        if transforms and transforms[0].get("MetricName") == metric_name:
            matches.append(r)
    assert len(matches) == 1, (
        f"expected exactly one metric filter with MetricName={metric_name!r}, got {len(matches)}"
    )
    return matches[0]


def test_per_model_filter_keyed_on_gen_ai_response_model() -> None:
    """The per-model dimensioned filter keys on the OTel gen_ai.response.model field.

    The load-bearing P2 assertion: the routing_latency_ms_by_model filter's
    Dimensions maps ``model`` -> ``$.["gen_ai.response.model"]`` (NOT
    ``$.selected_model``), and carries NO DefaultValue (AWS forbids it on a
    dimensioned filter).
    """
    template = _obs_template()
    by_model = _by_metric_name(template, "routing_latency_ms_by_model")
    transform = by_model["Properties"]["MetricTransformations"][0]
    assert transform.get("Dimensions") == [{"Key": "model", "Value": _GEN_AI_RESPONSE_MODEL}], (
        transform.get("Dimensions")
    )
    # AWS forbids DefaultValue on a dimensioned filter.
    assert "DefaultValue" not in transform, (
        f"dimensioned filter must NOT carry DefaultValue (AWS forbids it); got {transform}"
    )


def test_aggregate_latency_filter_has_no_dimensions_and_default_zero() -> None:
    """The aggregate routing_latency_ms filter has NO Dimensions and DefaultValue=0."""
    template = _obs_template()
    agg = _by_metric_name(template, "routing_latency_ms")
    transform = agg["Properties"]["MetricTransformations"][0]
    assert "Dimensions" not in transform, transform
    assert transform.get("DefaultValue") == 0, transform


def test_error_filter_has_default_zero() -> None:
    """The aggregate router_error_log_count filter keeps DefaultValue=0."""
    template = _obs_template()
    err = _by_metric_name(template, "router_error_log_count")
    transform = err["Properties"]["MetricTransformations"][0]
    assert transform.get("DefaultValue") == 0, transform


def test_filters_select_routing_decision_event() -> None:
    """The latency filters select the structured $.event = "routing_decision" line."""
    template = _obs_template()
    by_model = _by_metric_name(template, "routing_latency_ms_by_model")
    pattern = by_model["Properties"]["FilterPattern"]
    assert "routing_decision" in pattern, pattern
    assert "$.event" in pattern, pattern


# ------------------------------------------------------------------------- SNS


def test_sns_topic_denies_insecure_transport() -> None:
    """The on-call topic's resource policy denies non-TLS publish/subscribe."""
    template = _obs_template()
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
                                    "Action": Match.array_with(["sns:Publish", "sns:Subscribe"]),
                                    "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                                }
                            )
                        ]
                    )
                }
            )
        },
    )


def test_sns_topic_name() -> None:
    """The on-call topic is named routeiq-<env>-oncall."""
    template = _obs_template()
    template.has_resource_properties(
        "AWS::SNS::Topic",
        {"TopicName": "routeiq-dev-oncall"},
    )


def test_notify_emails_subscribe() -> None:
    """Supplied notify_emails become email subscriptions on the topic."""
    template = _obs_template(notify_emails=["oncall@example.com"])
    template.has_resource_properties(
        "AWS::SNS::Subscription",
        {"Protocol": "email", "Endpoint": "oncall@example.com"},
    )


# ----------------------------------------------------------------------- alarms


def test_every_alarm_has_alarm_and_ok_actions() -> None:
    """Every CW alarm wires BOTH an alarm action and an OK action to the topic.

    The VSR regression-fix lesson: an alarm with no action sits in ALARM state and
    pages no one. Assert every AWS::CloudWatch::Alarm carries non-empty AlarmActions
    AND OKActions.
    """
    template = _obs_template()
    alarms = template.find_resources("AWS::CloudWatch::Alarm")
    assert alarms, "expected at least one CloudWatch alarm"
    for lid, alarm in alarms.items():
        props = alarm["Properties"]
        assert props.get("AlarmActions"), f"{lid} missing AlarmActions"
        assert props.get("OKActions"), f"{lid} missing OKActions"


def test_routing_latency_alarm_threshold() -> None:
    """The routing-latency ceiling alarm reads routing_latency_ms at 30000ms."""
    template = _obs_template()
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        Match.object_like(
            {
                "MetricName": "routing_latency_ms",
                "Statistic": "Maximum",
                "Threshold": 30000,
                "ComparisonOperator": "GreaterThanThreshold",
            }
        ),
    )


def test_router_error_alarm_threshold() -> None:
    """The router-error alarm reads router_error_log_count Sum at threshold 10."""
    template = _obs_template()
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        Match.object_like(
            {
                "MetricName": "router_error_log_count",
                "Statistic": "Sum",
                "Threshold": 10,
            }
        ),
    )


def test_anomaly_alarm_present() -> None:
    """An anomaly-detection alarm exists (GreaterThanUpperThreshold over the band)."""
    template = _obs_template()
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        Match.object_like(
            {
                "ComparisonOperator": "GreaterThanUpperThreshold",
                "ThresholdMetricId": Match.any_value(),
            }
        ),
    )


# -------------------------------------------------------------------- dashboard


def test_dashboard_exists_with_search_expansion() -> None:
    """A routing dashboard exists whose widgets use the SEARCH() per-model expansion."""
    template = _obs_template()
    dashboards = template.find_resources("AWS::CloudWatch::Dashboard")
    assert len(dashboards) == 1, dashboards
    body = next(iter(dashboards.values()))["Properties"]["DashboardBody"]
    body_str = str(body)
    assert "SEARCH(" in body_str, "dashboard body should use a SEARCH() expansion"
    assert "routing_latency_ms_by_model" in body_str, body_str[:300]


# ------------------------------------------------------------------------ ASCII


def test_iam_role_descriptions_are_ascii() -> None:
    """Every IAM role Description is ASCII / Latin-1 only (P0 section 4.5).

    Exercised on the AMG-on surface so the AmgWorkspaceRole description is checked
    (an em-dash passes synth but fails the IAM CREATE API).
    """
    template = _obs_template(enable_amg=True)
    roles = template.find_resources("AWS::IAM::Role")
    for logical, role in roles.items():
        desc = role["Properties"].get("Description")
        if isinstance(desc, str):
            assert _IAM_DESCRIPTION_CHARSET.match(desc), (
                f"IAM role {logical} Description has a char outside IAM's allowed "
                f"Latin-1 set: {desc!r}"
            )
