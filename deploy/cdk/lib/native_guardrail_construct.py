"""NativeGuardrailConstruct - a flag-gated Amazon Bedrock Guardrail (RouteIQ-c0be).

Authors a native ``AWS::Bedrock::Guardrail`` + a PINNED
``AWS::Bedrock::GuardrailVersion`` so RouteIQ has an IaC-owned, content-safety
guardrail it can attach on the Bedrock data path. This is the IAC STAGE only -
the data-path activation (wiring the guardrail id/version into the
``bedrock_guardrails`` plugin via ``BEDROCK_GUARDRAIL_ID`` /
``BEDROCK_GUARDRAIL_VERSION``) is RouteIQ-9f14, a SEPARATE wave; this construct
does NOT touch the plugin. Its CfnOutputs (the guardrail id + the pinned version
string) are exactly the two values that plugin consumes.

FLAG-GATED, DEFAULT OFF (``routeiq:enable_native_guardrail``). When the flag is
off the composition root never instantiates this construct, so the stack emits
ZERO ``AWS::Bedrock::Guardrail`` / ``AWS::Bedrock::GuardrailVersion`` resources
and the default synth/snapshot is byte-stable. There is ZERO live consumer this
wave.

PINNED VERSION (not DRAFT): the construct mints a ``CfnGuardrailVersion`` so the
data path can reference an IMMUTABLE numbered version rather than the mutable
DRAFT - an edit to the guardrail does not silently change behaviour under a
pinned consumer. The version DependsOn the guardrail (CFN does not infer it).

The policy mirrors the RouteIQ guardrail-policy intent (ADR-0023): the six
managed content-filter categories at HIGH strength on input + output, plus a PII
sensitive-information policy over the common identifier set (the
``pii_guard``/``content_filter`` plugin analogue). All Description text is
ASCII / Latin-1 only (an em-dash U+2014 passes ``cdk synth`` but FAILS the
underlying create API - the P0 section-4.5 convention applied here too).
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Tags
from aws_cdk import aws_bedrock as bedrock
from constructs import Construct

# The six managed Bedrock content-filter categories. HARM categories are filtered
# at HIGH strength on both input and output; PROMPT_ATTACK is input-only (an
# output-side prompt-attack filter is not a valid Bedrock config).
_HARM_CATEGORIES = ["HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT"]

# The common PII identifier set the guardrail anonymizes (ANONYMIZE = mask, not
# hard BLOCK, so legit traffic mentioning an email is not denied outright - the
# RouteIQ pii_guard intent). Names are the Bedrock PiiEntity ``type`` enum values.
_PII_ANONYMIZE_TYPES = [
    "EMAIL",
    "PHONE",
    "NAME",
    "ADDRESS",
    "CREDIT_DEBIT_CARD_NUMBER",
    "US_SOCIAL_SECURITY_NUMBER",
]

# Default operator-facing block messages (ASCII-only).
_DEFAULT_BLOCKED_INPUT = "This request was blocked by the RouteIQ content guardrail."
_DEFAULT_BLOCKED_OUTPUT = "This response was blocked by the RouteIQ content guardrail."


class NativeGuardrailConstruct(Construct):
    """A native Bedrock Guardrail + a pinned GuardrailVersion (flag-gated, off by default).

    Public attributes for the composition root / outputs:
    ``guardrail`` (the ``CfnGuardrail``), ``guardrail_version``
    (the pinned ``CfnGuardrailVersion``), ``guardrail_id``, ``guardrail_arn``,
    ``guardrail_version_number``.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        blocked_input_messaging: str = _DEFAULT_BLOCKED_INPUT,
        blocked_outputs_messaging: str = _DEFAULT_BLOCKED_OUTPUT,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # -- Content policy: the six managed categories at HIGH ------------------
        # HARM categories filter input AND output; PROMPT_ATTACK is input-only
        # (Bedrock rejects an output-side PROMPT_ATTACK filter).
        filters: list[bedrock.CfnGuardrail.ContentFilterConfigProperty] = [
            bedrock.CfnGuardrail.ContentFilterConfigProperty(
                type=category,
                input_strength="HIGH",
                output_strength="HIGH",
            )
            for category in _HARM_CATEGORIES
        ]
        filters.append(
            bedrock.CfnGuardrail.ContentFilterConfigProperty(
                type="PROMPT_ATTACK",
                input_strength="HIGH",
                output_strength="NONE",
            )
        )
        content_policy = bedrock.CfnGuardrail.ContentPolicyConfigProperty(
            filters_config=filters,
        )

        # -- Sensitive-information policy: anonymize common PII ------------------
        sensitive_policy = bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
            pii_entities_config=[
                bedrock.CfnGuardrail.PiiEntityConfigProperty(
                    type=pii_type,
                    action="ANONYMIZE",
                )
                for pii_type in _PII_ANONYMIZE_TYPES
            ],
        )

        # -- The guardrail (ASCII-only name/description) ------------------------
        self.guardrail = bedrock.CfnGuardrail(
            self,
            "Guardrail",
            name=f"routeiq-{env_name}-content-guardrail",
            description=(
                f"RouteIQ {env_name} native content-safety guardrail "
                "(content filters + PII). IaC-owned; attached via the "
                "bedrock_guardrails plugin (RouteIQ-9f14)."
            ),
            blocked_input_messaging=blocked_input_messaging,
            blocked_outputs_messaging=blocked_outputs_messaging,
            content_policy_config=content_policy,
            sensitive_information_policy_config=sensitive_policy,
        )
        Tags.of(self.guardrail).add("routeiq:env", env_name)

        # -- A PINNED, immutable numbered version -------------------------------
        # The data path references THIS numbered version (not the mutable DRAFT),
        # so a later guardrail edit does not silently change behaviour under a
        # pinned consumer. CFN does not infer the dependency from the identifier
        # attribute reference, so order it explicitly after the guardrail exists.
        self.guardrail_version = bedrock.CfnGuardrailVersion(
            self,
            "GuardrailVersion",
            guardrail_identifier=self.guardrail.attr_guardrail_id,
            description="Pinned RouteIQ content-guardrail version for the data path",
        )
        self.guardrail_version.add_dependency(self.guardrail)

        # -- public attrs + operator-visible outputs ---------------------------
        self.guardrail_id: str = self.guardrail.attr_guardrail_id
        self.guardrail_arn: str = self.guardrail.attr_guardrail_arn
        self.guardrail_version_number: str = self.guardrail_version.attr_version

        # The two values RouteIQ-9f14 feeds the bedrock_guardrails plugin
        # (BEDROCK_GUARDRAIL_ID / BEDROCK_GUARDRAIL_VERSION).
        CfnOutput(
            self,
            "GuardrailId",
            value=self.guardrail_id,
            description=(
                "Bedrock guardrail id. Set as BEDROCK_GUARDRAIL_ID for the "
                "bedrock_guardrails plugin (RouteIQ-9f14)."
            ),
        )
        CfnOutput(
            self,
            "GuardrailVersionNumber",
            value=self.guardrail_version_number,
            description=(
                "Pinned guardrail version. Set as BEDROCK_GUARDRAIL_VERSION for "
                "the bedrock_guardrails plugin (RouteIQ-9f14)."
            ),
        )
