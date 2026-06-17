"""Unit tests for the AppConfig CloudWatch-alarm rollback Monitor (RouteIQ-b056).

ADR-0026 deploys RouteIQ config via the Linear20Pct3Min strategy with a final
bake. A Monitor on the AppConfig environment makes that bake self-healing: when a
watched CloudWatch alarm goes ALARM during the deployment, AppConfig auto-rolls
the config back to the previous version. This is flag-gated
(``ConfigStateConstruct(enable_rollback_monitor=...)``), DEFAULT OFF, so the
byte-stable guarantee is that the default environment carries NO Monitors and no
alarm / monitor-role resources are emitted.

Synthesised offline against the dummy env (account ``123456789012`` /
``us-west-2``), credential-free. The validator Lambda is forced onto its hermetic
inline-fallback path by the autouse fixture pinning ``_VALIDATOR_ASSET_PATH`` to a
missing dir (mx-86079a), mirroring test_config_state.py / test_config_audit.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk import Aspects
from aws_cdk.assertions import Annotations, Match, Template
from cdk_nag import AwsSolutionsChecks

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"


@pytest.fixture(autouse=True)
def _force_hermetic_validator_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the validator asset path to a missing dir -> deterministic inline path."""
    import lib.config_state_construct as cfg

    missing = str(Path(__file__).parent / "_does_not_exist_monitor")
    monkeypatch.setattr(cfg, "_VALIDATOR_ASSET_PATH", missing)


def _dummy_env() -> cdk.Environment:
    return cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION)


def _monitor_template(*, with_aspect: bool = False, **construct_kwargs: Any):
    """Synthesise a Stack hosting ONE ConfigStateConstruct (offline)."""
    from lib.config_state_construct import ConfigStateConstruct

    app = cdk.App()
    stack = cdk.Stack(app, "ConfigMonitorTestStack", env=_dummy_env())
    ConfigStateConstruct(
        stack,
        "ConfigStateConstruct",
        env_name=construct_kwargs.pop("env_name", "dev"),
        **construct_kwargs,
    )
    if with_aspect:
        Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
        return stack, Template.from_stack(stack)
    return Template.from_stack(stack)


# -------------------------------------------------------------- flag-off default


def test_monitor_off_by_default_emits_no_alarm_or_monitor_role() -> None:
    """The default construct (no enable_rollback_monitor) emits zero monitor resources."""
    template = _monitor_template()
    template.resource_count_is("AWS::CloudWatch::Alarm", 0)
    # The environment carries NO Monitors property when the flag is off.
    envs = template.find_resources("AWS::AppConfig::Environment")
    assert len(envs) == 1
    [(_lid, env)] = envs.items()
    assert "Monitors" not in env["Properties"], env["Properties"]


# ----------------------------------------------------------------- flag-on graph


def test_monitor_emits_alarm_when_enabled() -> None:
    """enable_rollback_monitor=True adds exactly one CloudWatch alarm."""
    template = _monitor_template(enable_rollback_monitor=True)
    template.resource_count_is("AWS::CloudWatch::Alarm", 1)


def test_alarm_watches_routeiq_router_error_metric() -> None:
    """The rollback alarm watches the RouteIQ/RouterErrorCount metric."""
    template = _monitor_template(enable_rollback_monitor=True)
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        Match.object_like(
            {
                "Namespace": "RouteIQ",
                "MetricName": "RouterErrorCount",
                "ComparisonOperator": "GreaterThanOrEqualToThreshold",
            }
        ),
    )


def test_environment_carries_monitor_with_alarm_and_role_arn() -> None:
    """The AppConfig environment's Monitors[] carries alarmArn + alarmRoleArn.

    Both are load-bearing: AppConfig needs the alarm to watch AND a role it can
    assume to read the alarm state during the deployment bake. Without the
    monitor wiring there is no auto-rollback on a bad config.
    """
    template = _monitor_template(enable_rollback_monitor=True)
    template.has_resource_properties(
        "AWS::AppConfig::Environment",
        Match.object_like(
            {
                "Monitors": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "AlarmArn": Match.any_value(),
                                "AlarmRoleArn": Match.any_value(),
                            }
                        )
                    ]
                )
            }
        ),
    )


def test_monitor_role_assumed_by_appconfig() -> None:
    """The monitor role is assumed by appconfig.amazonaws.com (not a wildcard)."""
    template = _monitor_template(enable_rollback_monitor=True)
    template.has_resource_properties(
        "AWS::IAM::Role",
        Match.object_like(
            {
                "AssumeRolePolicyDocument": Match.object_like(
                    {
                        "Statement": Match.array_with(
                            [
                                Match.object_like(
                                    {
                                        "Action": "sts:AssumeRole",
                                        "Principal": {"Service": "appconfig.amazonaws.com"},
                                    }
                                )
                            ]
                        )
                    }
                )
            }
        ),
    )


def test_monitor_role_grants_describe_alarms() -> None:
    """The monitor role can read alarm state via cloudwatch:DescribeAlarms."""
    template = _monitor_template(enable_rollback_monitor=True)
    template.has_resource_properties(
        "AWS::IAM::Policy",
        Match.object_like(
            {
                "PolicyDocument": Match.object_like(
                    {
                        "Statement": Match.array_with(
                            [
                                Match.object_like(
                                    {
                                        "Action": "cloudwatch:DescribeAlarms",
                                        "Effect": "Allow",
                                    }
                                )
                            ]
                        )
                    }
                )
            }
        ),
    )


def test_monitor_emits_alarm_arn_output() -> None:
    """The construct emits a ConfigRollbackAlarmArn output when the monitor is on."""
    template = _monitor_template(enable_rollback_monitor=True)
    outputs = template.find_outputs("*")
    assert any("ConfigRollbackAlarmArn" in k for k in outputs), list(outputs)


# -------------------------------------------------------------------- cdk-nag


def test_monitor_construct_cdk_nag_clean() -> None:
    """No unsuppressed AwsSolutions-* errors over the monitor-on construct stack."""
    stack, _template = _monitor_template(with_aspect=True, enable_rollback_monitor=True)
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        rendered = "\n".join(f"- {entry.id}: {str(entry.entry.data)[:200]}" for entry in errors)
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) over the "
            f"rollback-monitor construct stack:\n{rendered}"
        )
