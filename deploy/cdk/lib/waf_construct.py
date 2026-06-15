"""WafConstruct - AWS WAFv2 edge layer for the internet-facing ALB (RouteIQ-4f59).

Port of the vllm-sr-on-aws ``waf_construct.py`` (read-only source) into RouteIQ,
re-derived symbol-by-symbol. The 5-rule WAFv2 posture is sound as-is for RouteIQ:
the gateway exposes the same ``/v1/chat/completions`` surface and carries the same
bearer-on-Authorization, so the managed rule groups, the ``/v1/*`` rate scope-down,
the ``SizeRestrictions_BODY`` always-Count override, the 8 KB oversized-body COUNT,
and the Authorization+Cookie log redaction all transfer unchanged. Only the names
(``vllm-sr-`` -> ``routeiq-``) and the flag prefix (``vllm_sr:`` -> ``routeiq:``)
are renamed.

This is a flag-gated construct (``routeiq:enable_waf``, DEFAULT FALSE). It is
instantiated by ``RouteIqStack`` ONLY when ``enable_waf is True`` AND an operator
supplies a ``waf_alb_arn`` (see "Deploy-gated live attach" below). With the flag
OFF (the default surface) ``RouteIqStack`` emits ZERO ``AWS::WAFv2::*`` resources,
which keeps ``test_p4_edge_no_cdk_alb.py`` (foundation has no ``AWS::WAFv2``
substring) and the snapshot ``__snapshots__/dev.json`` byte-stable.

Deploy-gated live attach (the operator half, NOT synth-provable)
================================================================
The web ACL associates to a LIVE ALB by ARN. At P0 the RouteIQ chart default is
``service.type=ClusterIP`` / ``ingress.enabled=false``, so EKS Auto Mode renders
NO ALB - there is no in-stack ALB ARN to wire (this is exactly why the AlbSg
443-from-0.0.0.0/0 ingress is suppressed PREP-ONLY in ``nag_suppressions.py``).
The CONSTRUCT + its synth test (association to a SUPPLIED ARN) are the cred-free
deliverables here; the LIVE attach is OPERATOR-GATED: the operator flips
``ingress.enabled=true`` -> Auto Mode renders an ALB -> the operator passes that
ALB ARN as ``routeiq:waf_alb_arn`` and re-deploys. RouteIQ-4f59 therefore closes
PARTIAL (construct shipped; live attach deploy-gated), mirroring the existing
AlbSg PREP-ONLY edge documentation.

Posture (v1, default-OFF behind ``routeiq:enable_waf``)
=======================================================
* Scope ``REGIONAL`` (ALB, not CloudFront). DefaultAction ``Allow`` - rules block.
* Managed rule groups (all referenced via ManagedRuleGroupStatement, VendorName
  ``AWS``):
    1. ``AWSManagedRulesAmazonIpReputationList`` - BLOCK day 1 (IP-only, ~0 FP).
    2. ``AWSManagedRulesKnownBadInputsRuleSet``  - BLOCK day 1 (signature, low FP).
    3. ``AWSManagedRulesCommonRuleSet`` (CRS)    - group in COUNT for the initial
       bake window (operator flips to block via ``routeiq:waf_crs_block=true``),
       and ``SizeRestrictions_BODY`` PERMANENTLY overridden to Count because
       legitimate ``/v1/chat/completions`` bodies routinely exceed the WAF 8 KB
       body-inspection window (see below) and would otherwise be blocked.
* Custom rules:
    1. ``RateLimitPerIp`` - RateBasedStatement, per-IP, scoped to ``/v1/*``,
       limit ``routeiq:waf_rate_limit`` (default 2000 / 5 min). COUNT by default
       (``routeiq:waf_rate_block=true`` to block) - a rate ceiling is workload-
       specific, so we observe before enforcing.
    2. ``OversizedBodyCount`` - SizeConstraintStatement on the body with
       ``oversize_handling=MATCH``, COUNT only. This is an OBSERVABILITY signal
       for the 8 KB-smuggling concern (a base64 payload padded past the 8 KB WAF
       inspection window so the classifier-bypassing bytes are never seen by
       WAF). We deliberately do NOT block: on ALB the body-inspection limit is a
       FIXED 8 KB (the AssociationConfig 16-64 KB override is CloudFront/APIGW-
       only, a no-op for ALB), and legit LLM payloads are routinely larger, so
       blocking >8 KB would break normal traffic. Counting surfaces how often a
       request even *could* carry past-8 KB-smuggled content for the operator.
* Logging: CfnLoggingConfiguration -> a CloudWatch Logs group named
  ``aws-waf-logs-routeiq-{env}`` (the ``aws-waf-logs-`` prefix is REQUIRED by
  WAF). LoggingFilter logs BLOCK + COUNT, drops plain ALLOW. Authorization +
  Cookie headers are redacted (the bearer token rides Authorization).

WCU budget stays well under the 1500/web-ACL ceiling (managed groups ~925 +
two custom rules ~60).

cdk-nag
-------
Attaching the web ACL CLOSES ``AwsSolutions-ALB2`` ("ALB not associated with a
WAFv2 web ACL"). ``AwsSolutions-WAFv2LoggingEnabled`` is satisfied by the
CfnLoggingConfiguration wired here. Any residual suppressions are applied in
``nag_suppressions.py`` (guarded on ``getattr(stack, "waf", None) is not None``
so the OFF path never references a non-existent resource path).
"""

from __future__ import annotations

from aws_cdk import RemovalPolicy
from aws_cdk import aws_logs as logs
from aws_cdk import aws_wafv2 as wafv2
from constructs import Construct

# Managed rule group WCU costs (AWS-published) - summed only to assert the
# total stays under the 1500 ceiling in tests; not used at synth.
_WCU_IP_REPUTATION = 25
_WCU_KNOWN_BAD_INPUTS = 200
_WCU_CRS = 700

_DEFAULT_RATE_LIMIT = 2000
"""Per-IP requests per 5-min window. WAF RateBasedStatement floor is 10, ceiling
2e9. 2000/5min ~= 6.7 rps sustained per IP - generous for a single client, tight
enough to blunt a refusal-probe / scraping flood. Override routeiq:waf_rate_limit."""

# On ALB, WAF inspects only the first 8 KiB of the body (FIXED - the
# AssociationConfig override that raises it to 16-64 KiB is CloudFront/APIGW
# only). We COUNT bodies larger than this as a smuggling-surface signal.
_BODY_INSPECTION_LIMIT_BYTES = 8192


class WafConstruct(Construct):
    """WAFv2 web ACL + logging + association for the ALB (RouteIQ-4f59)."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        alb_arn: str,
        rate_limit: int | None = None,
        crs_block: bool = False,
        rate_block: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if not alb_arn:
            raise ValueError("alb_arn is required - the web ACL associates to the ALB by ARN.")

        self.env_name = env_name
        self._alb_arn = alb_arn
        self._rate_limit = max(10, int(rate_limit) if rate_limit else _DEFAULT_RATE_LIMIT)
        self._crs_block = bool(crs_block)
        self._rate_block = bool(rate_block)

        self.web_acl = self._build_web_acl()
        self.log_group = self._build_logging(self.web_acl)
        self.association = wafv2.CfnWebACLAssociation(
            self,
            "Association",
            resource_arn=alb_arn,
            web_acl_arn=self.web_acl.attr_arn,
        )

    # ------------------------------------------------------------------
    # web ACL
    # ------------------------------------------------------------------

    def _vis(self, metric: str) -> wafv2.CfnWebACL.VisibilityConfigProperty:
        return wafv2.CfnWebACL.VisibilityConfigProperty(
            cloud_watch_metrics_enabled=True,
            metric_name=metric,
            sampled_requests_enabled=True,
        )

    def _build_web_acl(self) -> wafv2.CfnWebACL:
        rules: list[wafv2.CfnWebACL.RuleProperty] = []

        # Priority 10 - Amazon IP reputation (BLOCK day 1; IP-only, ~0 FP).
        rules.append(
            wafv2.CfnWebACL.RuleProperty(
                name="IpReputation",
                priority=10,
                override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                statement=wafv2.CfnWebACL.StatementProperty(
                    managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                        vendor_name="AWS",
                        name="AWSManagedRulesAmazonIpReputationList",
                    )
                ),
                visibility_config=self._vis("IpReputation"),
            )
        )

        # Priority 20 - known-bad inputs (BLOCK day 1; exploit signatures).
        rules.append(
            wafv2.CfnWebACL.RuleProperty(
                name="KnownBadInputs",
                priority=20,
                override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                statement=wafv2.CfnWebACL.StatementProperty(
                    managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                        vendor_name="AWS",
                        name="AWSManagedRulesKnownBadInputsRuleSet",
                    )
                ),
                visibility_config=self._vis("KnownBadInputs"),
            )
        )

        # Priority 30 - Core Rule Set. Group COUNT during bake (operator flips to
        # block) BUT SizeRestrictions_BODY is ALWAYS counted (legit LLM bodies
        # exceed the 8 KB WAF window and would otherwise be blocked).
        crs_override = (
            wafv2.CfnWebACL.OverrideActionProperty(none={})
            if self._crs_block
            else wafv2.CfnWebACL.OverrideActionProperty(count={})
        )
        rules.append(
            wafv2.CfnWebACL.RuleProperty(
                name="CommonRuleSet",
                priority=30,
                override_action=crs_override,
                statement=wafv2.CfnWebACL.StatementProperty(
                    managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                        vendor_name="AWS",
                        name="AWSManagedRulesCommonRuleSet",
                        rule_action_overrides=[
                            wafv2.CfnWebACL.RuleActionOverrideProperty(
                                name="SizeRestrictions_BODY",
                                action_to_use=wafv2.CfnWebACL.RuleActionProperty(count={}),
                            ),
                        ],
                    )
                ),
                visibility_config=self._vis("CommonRuleSet"),
            )
        )

        # Priority 40 - per-IP rate limit, scoped to /v1/* (the inference API).
        # COUNT by default; flip with routeiq:waf_rate_block.
        v1_scope = wafv2.CfnWebACL.StatementProperty(
            byte_match_statement=wafv2.CfnWebACL.ByteMatchStatementProperty(
                search_string="/v1/",
                field_to_match=wafv2.CfnWebACL.FieldToMatchProperty(uri_path={}),
                positional_constraint="STARTS_WITH",
                text_transformations=[
                    wafv2.CfnWebACL.TextTransformationProperty(priority=0, type="NONE")
                ],
            )
        )
        rate_action = (
            wafv2.CfnWebACL.RuleActionProperty(block={})
            if self._rate_block
            else wafv2.CfnWebACL.RuleActionProperty(count={})
        )
        rules.append(
            wafv2.CfnWebACL.RuleProperty(
                name="RateLimitPerIp",
                priority=40,
                action=rate_action,
                statement=wafv2.CfnWebACL.StatementProperty(
                    rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                        limit=self._rate_limit,
                        aggregate_key_type="IP",
                        evaluation_window_sec=300,
                        scope_down_statement=v1_scope,
                    )
                ),
                visibility_config=self._vis("RateLimitPerIp"),
            )
        )

        # Priority 50 - oversized body COUNT (8 KB-smuggling surface signal,
        # NEVER block: ALB body inspection is a fixed 8 KB and legit LLM bodies
        # exceed it; blocking would break normal traffic).
        rules.append(
            wafv2.CfnWebACL.RuleProperty(
                name="OversizedBodyCount",
                priority=50,
                action=wafv2.CfnWebACL.RuleActionProperty(count={}),
                statement=wafv2.CfnWebACL.StatementProperty(
                    size_constraint_statement=wafv2.CfnWebACL.SizeConstraintStatementProperty(
                        field_to_match=wafv2.CfnWebACL.FieldToMatchProperty(
                            body=wafv2.CfnWebACL.BodyProperty(oversize_handling="MATCH")
                        ),
                        comparison_operator="GT",
                        size=_BODY_INSPECTION_LIMIT_BYTES,
                        text_transformations=[
                            wafv2.CfnWebACL.TextTransformationProperty(priority=0, type="NONE")
                        ],
                    )
                ),
                visibility_config=self._vis("OversizedBodyCount"),
            )
        )

        return wafv2.CfnWebACL(
            self,
            "WebAcl",
            name=f"routeiq-{self.env_name}-alb",
            scope="REGIONAL",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            visibility_config=self._vis(f"routeiq-{self.env_name}-alb"),
            rules=rules,
        )

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    def _build_logging(self, web_acl: wafv2.CfnWebACL) -> logs.LogGroup:
        # WAF logging destinations MUST be named with the ``aws-waf-logs-`` prefix.
        log_group = logs.LogGroup(
            self,
            "WafLogGroup",
            log_group_name=f"aws-waf-logs-routeiq-{self.env_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        wafv2.CfnLoggingConfiguration(
            self,
            "LoggingConfig",
            resource_arn=web_acl.attr_arn,
            log_destination_configs=[log_group.log_group_arn],
            # Log BLOCK + COUNT actions; drop plain ALLOW to keep the log lean
            # and focused on security-relevant events.
            logging_filter={
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
            # The bearer token rides Authorization; redact it + Cookie from logs.
            redacted_fields=[
                wafv2.CfnLoggingConfiguration.FieldToMatchProperty(
                    single_header={"Name": "authorization"}
                ),
                wafv2.CfnLoggingConfiguration.FieldToMatchProperty(
                    single_header={"Name": "cookie"}
                ),
            ],
        )
        return log_group
