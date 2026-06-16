"""Unit tests for NativeGuardrailConstruct (RouteIQ-c0be - native Bedrock Guardrail).

Asserts the flag-gated native Amazon Bedrock guardrail RouteIQ authors: a
``CfnGuardrail`` carrying the six managed content-filter categories (HARM at HIGH
input+output, PROMPT_ATTACK input-only) + a PII anonymize policy, plus a PINNED
``CfnGuardrailVersion`` that DependsOn the guardrail.

The construct is FLAG-GATED off at the composition root (``RouteIqStack`` only
instantiates it when ``routeiq:enable_native_guardrail=true``), so the byte-stable
guarantee is that ``make_stack()`` (the default P0 surface) emits ZERO
``AWS::Bedrock::Guardrail`` / ``AWS::Bedrock::GuardrailVersion`` resources. There
is ZERO live consumer this wave - the data-path activation (wiring the
guardrail id/version into the ``bedrock_guardrails`` plugin) is RouteIQ-9f14, a
SEPARATE wave - so these tests assert ONLY the authored resource graph.

Synthesised offline against the dummy env (account ``123456789012`` /
``us-west-2``), credential-free. The construct-in-isolation synth holds one
``NativeGuardrailConstruct`` in a bare ``cdk.Stack`` (mirroring ``test_waf.py``);
the flag-off synth is ``make_stack()``. The cdk-nag mode adds the same
``AwsSolutionsChecks`` aspect ``app.py`` wires and asserts no ``AwsSolutions-*``
errors survive (the guardrail authors no IAM role, so there is no IAM4/IAM5
surface to suppress).
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
from aws_cdk import Aspects
from aws_cdk.assertions import Annotations, Match, Template
from cdk_nag import AwsSolutionsChecks

from lib.native_guardrail_construct import (
    _HARM_CATEGORIES,
    _PII_ANONYMIZE_TYPES,
    NativeGuardrailConstruct,
)
from tests.conftest import template_for

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"


def _dummy_env() -> cdk.Environment:
    return cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION)


def _guardrail_template(*, with_aspect: bool = False, **construct_kwargs: Any):
    """Synthesise a minimal stack holding ONE NativeGuardrailConstruct (offline)."""
    app = cdk.App()
    stack = cdk.Stack(app, "GuardrailTestStack", env=_dummy_env())
    NativeGuardrailConstruct(
        stack,
        "NativeGuardrailConstruct",
        env_name=construct_kwargs.pop("env_name", "dev"),
        **construct_kwargs,
    )
    if with_aspect:
        Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
        return stack, Template.from_stack(stack)
    return Template.from_stack(stack)


def _guardrail(template: Template) -> dict[str, Any]:
    return next(iter(template.find_resources("AWS::Bedrock::Guardrail").values()))


# -------------------------------------------------------------- flag-off default


def test_guardrail_off_by_default_emits_no_bedrock_guardrail() -> None:
    """The P0 default surface (make_stack) emits zero Bedrock guardrail resources.

    The composition root builds NativeGuardrailConstruct only when
    routeiq:enable_native_guardrail=true, so the default surface carries no
    guardrail or version. This is the byte-stable guarantee the dev snapshot
    relies on.
    """
    template = template_for()
    template.resource_count_is("AWS::Bedrock::Guardrail", 0)
    template.resource_count_is("AWS::Bedrock::GuardrailVersion", 0)


# ----------------------------------------------------------------- guardrail


def test_guardrail_named_routeiq_prefixed_with_block_messages() -> None:
    """The guardrail is routeiq-<env>-prefixed and carries both block messages."""
    template = _guardrail_template(env_name="dev")
    template.has_resource_properties(
        "AWS::Bedrock::Guardrail",
        Match.object_like(
            {
                "Name": "routeiq-dev-content-guardrail",
                "BlockedInputMessaging": Match.string_like_regexp(".*RouteIQ.*"),
                "BlockedOutputsMessaging": Match.string_like_regexp(".*RouteIQ.*"),
            }
        ),
    )


def test_guardrail_content_filters_harm_high_and_prompt_attack_input_only() -> None:
    """All HARM categories filter at HIGH in+out; PROMPT_ATTACK is input-only."""
    guardrail = _guardrail(_guardrail_template())
    filters = guardrail["Properties"]["ContentPolicyConfig"]["FiltersConfig"]
    by_type = {f["Type"]: f for f in filters}
    # All six managed categories present.
    assert set(by_type) == set(_HARM_CATEGORIES) | {"PROMPT_ATTACK"}
    for harm in _HARM_CATEGORIES:
        assert by_type[harm]["InputStrength"] == "HIGH"
        assert by_type[harm]["OutputStrength"] == "HIGH"
    # PROMPT_ATTACK cannot filter output (Bedrock rejects it) - input-only.
    assert by_type["PROMPT_ATTACK"]["InputStrength"] == "HIGH"
    assert by_type["PROMPT_ATTACK"]["OutputStrength"] == "NONE"


def test_guardrail_pii_anonymize_policy() -> None:
    """The sensitive-information policy ANONYMIZEs the common PII identifier set."""
    guardrail = _guardrail(_guardrail_template())
    pii = guardrail["Properties"]["SensitiveInformationPolicyConfig"]["PiiEntitiesConfig"]
    by_type = {p["Type"]: p for p in pii}
    assert set(by_type) == set(_PII_ANONYMIZE_TYPES)
    for entry in pii:
        assert entry["Action"] == "ANONYMIZE"


# ----------------------------------------------------------------- version pin


def test_guardrail_version_is_pinned_and_depends_on_guardrail() -> None:
    """A CfnGuardrailVersion is minted and DependsOn the guardrail.

    The pinned numbered version (not the mutable DRAFT) is what the data path
    references, so a later guardrail edit cannot silently change behaviour under
    a pinned consumer. CFN does not infer the dependency, so it must be explicit.
    """
    template = _guardrail_template()
    template.resource_count_is("AWS::Bedrock::GuardrailVersion", 1)

    res = template.to_json()["Resources"]
    guardrail_ids = [lid for lid, r in res.items() if r["Type"] == "AWS::Bedrock::Guardrail"]
    version_ids = [lid for lid, r in res.items() if r["Type"] == "AWS::Bedrock::GuardrailVersion"]
    assert len(guardrail_ids) == 1 and len(version_ids) == 1
    depends_on = res[version_ids[0]].get("DependsOn") or []
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    assert guardrail_ids[0] in depends_on, (
        f"GuardrailVersion must DependsOn the guardrail ({guardrail_ids[0]}); "
        f"DependsOn={depends_on}"
    )


def test_guardrail_emits_id_and_version_outputs() -> None:
    """The construct emits GuardrailId + GuardrailVersionNumber operator outputs.

    These are exactly the two values RouteIQ-9f14 feeds the bedrock_guardrails
    plugin (BEDROCK_GUARDRAIL_ID / BEDROCK_GUARDRAIL_VERSION).
    """
    template = _guardrail_template()
    outputs = template.find_outputs("*")
    keys = list(outputs)
    assert any("GuardrailId" in k for k in keys), keys
    assert any("GuardrailVersionNumber" in k for k in keys), keys


# -------------------------------------------------------------------- cdk-nag


def test_guardrail_construct_isolation_cdk_nag_clean() -> None:
    """No AwsSolutions-* errors survive over the construct-in-isolation stack.

    The guardrail authors no IAM role / no Lambda, so there is no IAM4/IAM5/L1
    surface; it needs ZERO suppressions.
    """
    stack, _template = _guardrail_template(with_aspect=True)
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        rendered = "\n".join(f"- {entry.id}: {str(entry.entry.data)[:200]}" for entry in errors)
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) over the guardrail "
            f"construct-isolation stack:\n{rendered}"
        )
