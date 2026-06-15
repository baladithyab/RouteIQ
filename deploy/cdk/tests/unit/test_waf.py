"""Unit tests for WafConstruct (RouteIQ-4f59 - WAFv2 edge layer port).

Asserts the WAFv2 web ACL the construct provisions: REGIONAL scope, default
Allow, the 5 rules at their fixed priorities (IpReputation@10 BLOCK,
KnownBadInputs@20 BLOCK, CommonRuleSet@30 COUNT-by-default with SizeRestrictions_BODY
ALWAYS Count, RateLimitPerIp@40 scoped to /v1/* COUNT-by-default, OversizedBodyCount@50
8 KB COUNT-only), the crs_block / rate_block flag flips, the rate_limit floor-10
clamp, the CfnLoggingConfiguration to an aws-waf-logs-routeiq-<env> LogGroup with
Authorization+Cookie redaction, the association to a supplied ALB ARN, and the
ALB-ARN-required guard. The WCU-budget-under-ceiling assertion is a pure module
constant check (no synth).

The construct is FLAG-GATED off by default at the composition root
(``RouteIqStack`` only instantiates it when ``routeiq:enable_waf=true`` AND a
``waf_alb_arn`` is supplied - no ALB renders at P0 / ClusterIP), so the "default
OFF" assertion is that ``make_stack()`` (the P0 default surface) emits zero
``AWS::WAFv2::*`` resources. This duplicates the ``test_p4_edge_no_cdk_alb.py``
guarantee locally so the WAF test is self-contained; that p4 test MUST stay green
untouched (it is the byte-stable wall).

Synthesised offline against the dummy env (account ``123456789012`` /
``us-west-2``), credential-free: the construct-in-isolation synth holds ONE
WafConstruct in a bare ``cdk.Stack`` with a dummy ALB ARN (mirroring
``test_data_lake.py``); the flag-off synth is ``make_stack()``. The cdk-nag mode
adds the same ``AwsSolutionsChecks`` aspect ``app.py`` wires and asserts no
``AwsSolutions-*`` errors survive (proving the WAF needs ZERO suppressions - the
association closes ALB2, the logging config satisfies WAFv2LoggingEnabled).
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk import Aspects
from aws_cdk.assertions import Annotations, Match, Template
from cdk_nag import AwsSolutionsChecks

from lib.waf_construct import (
    _WCU_CRS,
    _WCU_IP_REPUTATION,
    _WCU_KNOWN_BAD_INPUTS,
    WafConstruct,
)
from tests.conftest import template_for

# The dummy account/region the cred-free gate pins (mirrors tests/conftest.py).
DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"

# A dummy ALB ARN to associate the web ACL to (cred-free; never resolved against
# AWS - the construct only stamps it into the CfnWebACLAssociation.ResourceArn).
_DUMMY_ALB_ARN = "arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/routeiq/abc"


def _dummy_env() -> cdk.Environment:
    return cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION)


def _waf_template(*, with_aspect: bool = False, **construct_kwargs: Any):
    """Synthesise a minimal stack holding ONE WafConstruct (offline).

    Per-test overrides flow through ``construct_kwargs`` (rate_limit / crs_block /
    rate_block / env_name). When ``with_aspect`` is True the AwsSolutionsChecks
    aspect is added to the app (for the cdk-nag mode) and the built stack is
    returned alongside the Template so the test can read its annotations.
    """
    app = cdk.App()
    stack = cdk.Stack(app, "WafTestStack", env=_dummy_env())
    WafConstruct(
        stack,
        "WafConstruct",
        env_name=construct_kwargs.pop("env_name", "dev"),
        alb_arn=construct_kwargs.pop("alb_arn", _DUMMY_ALB_ARN),
        **construct_kwargs,
    )
    if with_aspect:
        Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
        return stack, Template.from_stack(stack)
    return Template.from_stack(stack)


def _rules_by_name(template: Template) -> dict[str, dict[str, Any]]:
    """Return the WebACL rules keyed by their Name for ergonomic assertions."""
    acl = next(iter(template.find_resources("AWS::WAFv2::WebACL").values()))
    return {rule["Name"]: rule for rule in acl["Properties"]["Rules"]}


# -------------------------------------------------------------- flag-off default


def test_waf_off_by_default_emits_no_wafv2() -> None:
    """The P0 default surface (make_stack) emits zero WAFv2 resources.

    The composition root builds WafConstruct only when routeiq:enable_waf=true AND
    a waf_alb_arn is supplied, so the default surface carries no web ACL,
    association, or logging config. This is the same guarantee
    test_p4_edge_no_cdk_alb.py makes - duplicated here so the WAF test is
    self-contained. DO NOT delete or weaken the p4 test.
    """
    template = template_for()
    template.resource_count_is("AWS::WAFv2::WebACL", 0)
    template.resource_count_is("AWS::WAFv2::WebACLAssociation", 0)
    template.resource_count_is("AWS::WAFv2::LoggingConfiguration", 0)


# --------------------------------------------------------------------- web ACL


def test_web_acl_regional_default_allow() -> None:
    """The web ACL is REGIONAL (ALB, not CloudFront) with DefaultAction Allow."""
    template = _waf_template()
    template.has_resource_properties(
        "AWS::WAFv2::WebACL",
        Match.object_like(
            {
                "Scope": "REGIONAL",
                "DefaultAction": {"Allow": {}},
            }
        ),
    )


def test_web_acl_name_is_routeiq_prefixed() -> None:
    """The web ACL + its visibility metric are renamed routeiq-<env>-alb."""
    template = _waf_template(env_name="dev")
    template.has_resource_properties(
        "AWS::WAFv2::WebACL",
        Match.object_like(
            {
                "Name": "routeiq-dev-alb",
                "VisibilityConfig": Match.object_like({"MetricName": "routeiq-dev-alb"}),
            }
        ),
    )


def test_web_acl_has_five_rules_at_expected_priorities() -> None:
    """Exactly 5 rules, with the canonical name->priority mapping."""
    template = _waf_template()
    rules = _rules_by_name(template)
    assert set(rules) == {
        "IpReputation",
        "KnownBadInputs",
        "CommonRuleSet",
        "RateLimitPerIp",
        "OversizedBodyCount",
    }
    assert rules["IpReputation"]["Priority"] == 10
    assert rules["KnownBadInputs"]["Priority"] == 20
    assert rules["CommonRuleSet"]["Priority"] == 30
    assert rules["RateLimitPerIp"]["Priority"] == 40
    assert rules["OversizedBodyCount"]["Priority"] == 50


def test_ip_reputation_and_known_bad_inputs_block_day_one() -> None:
    """Both AWS managed groups enforce (OverrideAction None) day one."""
    rules = _rules_by_name(_waf_template())
    assert rules["IpReputation"]["OverrideAction"] == {"None": {}}
    assert rules["KnownBadInputs"]["OverrideAction"] == {"None": {}}
    # And they reference the correct AWS-vendored managed groups.
    assert (
        rules["IpReputation"]["Statement"]["ManagedRuleGroupStatement"]["Name"]
        == "AWSManagedRulesAmazonIpReputationList"
    )
    assert (
        rules["KnownBadInputs"]["Statement"]["ManagedRuleGroupStatement"]["Name"]
        == "AWSManagedRulesKnownBadInputsRuleSet"
    )


def test_crs_count_by_default_and_size_body_override_count() -> None:
    """CRS group is Count by default; SizeRestrictions_BODY is ALWAYS Count.

    With crs_block=True the GROUP flips to enforce (OverrideAction None) but the
    SizeRestrictions_BODY override STAYS Count (legit /v1 bodies exceed the 8 KB
    WAF window and would otherwise be blocked).
    """

    def _crs(template: Template) -> dict[str, Any]:
        return _rules_by_name(template)["CommonRuleSet"]

    # Default: group Count.
    crs_default = _crs(_waf_template())
    assert crs_default["OverrideAction"] == {"Count": {}}
    body_override = crs_default["Statement"]["ManagedRuleGroupStatement"]["RuleActionOverrides"]
    assert body_override == [{"Name": "SizeRestrictions_BODY", "ActionToUse": {"Count": {}}}]

    # crs_block=True: group enforces, body override STILL Count.
    crs_blocked = _crs(_waf_template(crs_block=True))
    assert crs_blocked["OverrideAction"] == {"None": {}}
    assert crs_blocked["Statement"]["ManagedRuleGroupStatement"]["RuleActionOverrides"] == [
        {"Name": "SizeRestrictions_BODY", "ActionToUse": {"Count": {}}}
    ]


def test_rate_limit_scoped_to_v1_count_by_default() -> None:
    """RateLimitPerIp: limit 2000 / IP / 300s window, scoped to /v1/ STARTS_WITH.

    Count by default; Block when rate_block=True; honors a custom rate_limit; and
    clamps below-floor values up to the WAF minimum of 10.
    """
    rule = _rules_by_name(_waf_template())["RateLimitPerIp"]
    assert rule["Action"] == {"Count": {}}
    rbs = rule["Statement"]["RateBasedStatement"]
    assert rbs["Limit"] == 2000
    assert rbs["AggregateKeyType"] == "IP"
    assert rbs["EvaluationWindowSec"] == 300
    scope = rbs["ScopeDownStatement"]["ByteMatchStatement"]
    assert scope["SearchString"] == "/v1/"
    assert scope["PositionalConstraint"] == "STARTS_WITH"
    assert scope["FieldToMatch"] == {"UriPath": {}}

    # rate_block=True -> Block.
    blocked = _rules_by_name(_waf_template(rate_block=True))["RateLimitPerIp"]
    assert blocked["Action"] == {"Block": {}}

    # Custom limit honored.
    custom = _rules_by_name(_waf_template(rate_limit=500))["RateLimitPerIp"]
    assert custom["Statement"]["RateBasedStatement"]["Limit"] == 500

    # Below-floor limit clamps up to the WAF minimum of 10.
    clamped = _rules_by_name(_waf_template(rate_limit=1))["RateLimitPerIp"]
    assert clamped["Statement"]["RateBasedStatement"]["Limit"] == 10


def test_oversized_body_count_only_at_8kb() -> None:
    """OversizedBodyCount: SizeConstraint on the body, GT 8192, MATCH, Count only."""
    rule = _rules_by_name(_waf_template())["OversizedBodyCount"]
    assert rule["Action"] == {"Count": {}}
    scs = rule["Statement"]["SizeConstraintStatement"]
    assert scs["ComparisonOperator"] == "GT"
    assert scs["Size"] == 8192
    assert scs["FieldToMatch"] == {"Body": {"OversizeHandling": "MATCH"}}


# --------------------------------------------------------------------- logging


def test_logging_to_aws_waf_logs_prefixed_group_with_redaction() -> None:
    """LogGroup is aws-waf-logs-routeiq-<env>; filter DROP-default + KEEP BLOCK/COUNT;
    Authorization + Cookie single-headers are redacted."""
    template = _waf_template(env_name="dev")

    # The aws-waf-logs- prefix is REQUIRED by WAF; ONE_MONTH retention.
    template.has_resource_properties(
        "AWS::Logs::LogGroup",
        Match.object_like(
            {
                "LogGroupName": "aws-waf-logs-routeiq-dev",
                "RetentionInDays": 30,
            }
        ),
    )

    template.has_resource_properties(
        "AWS::WAFv2::LoggingConfiguration",
        Match.object_like(
            {
                "LoggingFilter": {
                    "DefaultBehavior": "DROP",
                    "Filters": [
                        {
                            "Behavior": "KEEP",
                            "Requirement": "MEETS_ANY",
                            "Conditions": [
                                {"ActionCondition": {"Action": "BLOCK"}},
                                {"ActionCondition": {"Action": "COUNT"}},
                            ],
                        }
                    ],
                },
                "RedactedFields": [
                    {"SingleHeader": {"Name": "authorization"}},
                    {"SingleHeader": {"Name": "cookie"}},
                ],
            }
        ),
    )


# ----------------------------------------------------------------- association


def test_association_targets_supplied_alb_arn() -> None:
    """The CfnWebACLAssociation.ResourceArn is exactly the supplied ALB ARN."""
    template = _waf_template()
    template.has_resource_properties(
        "AWS::WAFv2::WebACLAssociation",
        Match.object_like({"ResourceArn": _DUMMY_ALB_ARN}),
    )


def test_waf_requires_alb_arn() -> None:
    """An empty alb_arn raises ValueError (the web ACL must associate to an ALB)."""
    app = cdk.App()
    stack = cdk.Stack(app, "WafTestStack", env=_dummy_env())
    with pytest.raises(ValueError, match="alb_arn is required"):
        WafConstruct(stack, "WafConstruct", env_name="dev", alb_arn="")


# ------------------------------------------------------------------ WCU budget


def test_wcu_budget_under_ceiling() -> None:
    """The managed-group WCU sum stays well under the 1500/web-ACL ceiling.

    Pure module-constant assertion (no synth): the two custom rules add ~60 WCU
    on top, so the headroom is real.
    """
    assert _WCU_IP_REPUTATION + _WCU_KNOWN_BAD_INPUTS + _WCU_CRS < 1500


# -------------------------------------------------------------------- cdk-nag


def test_waf_construct_isolation_cdk_nag_clean() -> None:
    """No AwsSolutions-* errors survive over the construct-in-isolation stack.

    Proves the WAF needs ZERO suppressions: the CfnWebACLAssociation closes
    AwsSolutions-ALB2 and the CfnLoggingConfiguration satisfies
    AwsSolutions-WAFv2LoggingEnabled. The construct emits no IAM role, so there is
    no IAM4/IAM5 surface either.
    """
    stack, _template = _waf_template(with_aspect=True)
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        rendered = "\n".join(f"- {entry.id}: {str(entry.entry.data)[:200]}" for entry in errors)
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) over the WAF "
            f"construct-isolation stack:\n{rendered}"
        )
