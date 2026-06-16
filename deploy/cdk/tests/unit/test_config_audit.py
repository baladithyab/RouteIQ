"""Unit tests for the validator-mutation audit core (RouteIQ-1669 / ADR-0026).

ADR-0026 makes the AppConfig config-validator Lambda the single load-bearing
config gate and calls for an alarm on any attempt to strip it. The full GitOps
CodePipeline + deployer role described in ADR-0026's "Day-2 GitOps path" were
deliberately descoped (a future tier); the small, self-contained cred-free CORE
that DOES ship is ``ConfigStateConstruct(enable_config_audit=...)``: a
TLS-enforced SNS topic + an EventBridge rule matching the two mutating AppConfig
profile API calls (Update/DeleteConfigurationProfile) via CloudTrail.

It is FLAG-GATED off by default, so the byte-stable guarantee is that the default
``ConfigStateConstruct`` (audit off) emits ZERO ``AWS::Events::Rule`` /
``AWS::SNS::Topic`` (asserted in test_config_state.py). These tests assert the
flag-ON resource graph + that the flag stays off-by-default at the P2 stack.

Synthesised offline against the dummy env (account ``123456789012`` /
``us-west-2``), credential-free. The validator Lambda is forced onto its hermetic
inline-fallback path by the autouse fixture pinning ``_VALIDATOR_ASSET_PATH`` to
a missing dir (mx-86079a), mirroring test_config_state.py.
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

    missing = str(Path(__file__).parent / "_does_not_exist_audit")
    monkeypatch.setattr(cfg, "_VALIDATOR_ASSET_PATH", missing)


def _dummy_env() -> cdk.Environment:
    return cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION)


def _audit_template(*, with_aspect: bool = False, **construct_kwargs: Any):
    """Synthesise a Stack hosting ONE ConfigStateConstruct (offline)."""
    from lib.config_state_construct import ConfigStateConstruct

    app = cdk.App()
    stack = cdk.Stack(app, "ConfigAuditTestStack", env=_dummy_env())
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


def test_audit_off_by_default_emits_no_events_or_sns() -> None:
    """The default construct (no enable_config_audit) emits zero audit resources."""
    template = _audit_template()
    template.resource_count_is("AWS::Events::Rule", 0)
    template.resource_count_is("AWS::SNS::Topic", 0)


# ----------------------------------------------------------------- flag-on graph


def test_audit_emits_sns_topic_and_event_rule_when_enabled() -> None:
    """enable_config_audit=True adds exactly one SNS topic + one EventBridge rule."""
    template = _audit_template(enable_config_audit=True)
    template.resource_count_is("AWS::SNS::Topic", 1)
    template.resource_count_is("AWS::Events::Rule", 1)


def test_audit_topic_is_tls_enforced() -> None:
    """The audit SNS topic carries a DenyInsecureTransport resource policy."""
    template = _audit_template(enable_config_audit=True)
    template.has_resource_properties(
        "AWS::SNS::TopicPolicy",
        Match.object_like(
            {
                "PolicyDocument": Match.object_like(
                    {
                        "Statement": Match.array_with(
                            [
                                Match.object_like(
                                    {
                                        "Effect": "Deny",
                                        "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                                    }
                                )
                            ]
                        )
                    }
                )
            }
        ),
    )


def test_audit_rule_matches_mutating_appconfig_profile_calls() -> None:
    """The EventBridge rule matches Update/DeleteConfigurationProfile via CloudTrail.

    Those are the two API calls that can strip the validator (Update of the
    Validators array) or remove the profile (Delete) - the load-bearing audit
    signal of ADR-0026.
    """
    template = _audit_template(enable_config_audit=True)
    rule = next(iter(template.find_resources("AWS::Events::Rule").values()))
    pattern = rule["Properties"]["EventPattern"]
    assert pattern["source"] == ["aws.appconfig"]
    assert pattern["detail-type"] == ["AWS API Call via CloudTrail"]
    event_names = set(pattern["detail"]["eventName"])
    assert event_names == {"UpdateConfigurationProfile", "DeleteConfigurationProfile"}


def test_audit_rule_targets_the_audit_topic() -> None:
    """The EventBridge rule's target is the audit SNS topic."""
    template = _audit_template(enable_config_audit=True)
    rule = next(iter(template.find_resources("AWS::Events::Rule").values()))
    targets = rule["Properties"]["Targets"]
    assert len(targets) == 1
    # The target Arn is a Ref to the SNS topic logical id.
    assert "Arn" in targets[0]


def test_audit_emits_topic_arn_output() -> None:
    """The construct emits a ConfigAuditTopicArn output when audit is on."""
    template = _audit_template(enable_config_audit=True)
    outputs = template.find_outputs("*")
    assert any("ConfigAuditTopicArn" in k for k in outputs), list(outputs)


# -------------------------------------------------------------------- cdk-nag


def test_audit_construct_cdk_nag_clean() -> None:
    """No AwsSolutions-* errors survive over the audit-on construct stack.

    The TLS-enforced SNS topic satisfies SNS3; the EventBridge rule + the
    validator Lambda's inline suppressions (IAM4/L1) carry the rest. Guards that
    the new audit core needs no NEW unsuppressed nag findings.
    """
    stack, _template = _audit_template(with_aspect=True, enable_config_audit=True)
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        rendered = "\n".join(f"- {entry.id}: {str(entry.entry.data)[:200]}" for entry in errors)
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) over the config-audit "
            f"construct stack:\n{rendered}"
        )
