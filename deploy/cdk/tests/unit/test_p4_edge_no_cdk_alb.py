"""P4 edge-architecture guard: the public edge is NOT a CDK construct.

P4 ships the public edge (ALB + ACM TLS) as an OPERATOR-FLIPPED Helm Ingress
(``ingress.enabled=true`` + ALB IngressClass + an operator-supplied ACM
``certificate-arn`` annotation), rendered into an AWS-managed ALB by EKS Auto
Mode. There is deliberately NO ``elbv2`` / ``acm`` / ``route53`` / ``wafv2`` /
``bedrock.CfnGuardrail`` construct in ``deploy/cdk/`` (the optional WAF + native
Bedrock Guardrail were deferred to seeds, flag-gated OFF). The chart values.yaml
documents this decision and the helm render tests
(``tests/test_render.py::test_ingress_*``) cover the operator-flip path.

This module is the CDK-side complement: it pins the decision by asserting the
synthesized foundation + state stacks contain NONE of those AWS resource types,
so a regression that smuggles a CDK-provisioned ALB/ACM/WAF/Guardrail in (which
would invalidate the operator-flipped edge contract AND ship an un-flag-gated
public surface) fails the cred-free gate.

Synthesised offline via the shared helpers (dummy env account ``123456789012`` /
``us-west-2``) -- no AWS creds, no ``from_lookup``, no network.
"""

from __future__ import annotations

import pytest

from tests.conftest import state_template_for, template_for

# Resource types the P4 design says must NOT be CDK-provisioned. The edge is
# operator-flipped (Helm Ingress -> AWS-managed Auto Mode ALB); WAF + native
# Bedrock Guardrail are deferred-to-seed, flag-gated OFF constructs.
_FORBIDDEN_EDGE_TYPES = (
    "AWS::ElasticLoadBalancingV2::LoadBalancer",
    "AWS::ElasticLoadBalancingV2::Listener",
    "AWS::ElasticLoadBalancingV2::TargetGroup",
    "AWS::CertificateManager::Certificate",
    "AWS::Route53::RecordSet",
    "AWS::Route53::HostedZone",
    "AWS::WAFv2::WebACL",
    "AWS::WAFv2::WebACLAssociation",
    "AWS::Bedrock::Guardrail",
)


@pytest.mark.parametrize("forbidden", _FORBIDDEN_EDGE_TYPES)
def test_foundation_stack_has_no_cdk_edge_resource(forbidden: str) -> None:
    template = template_for()
    found = template.find_resources(forbidden)
    assert found == {}, (
        f"{forbidden} must NOT be CDK-provisioned in the foundation stack -- the "
        f"P4 public edge is an operator-flipped Helm Ingress (AWS-managed Auto "
        f"Mode ALB); WAF/Guardrail are deferred-to-seed flag-gated constructs. "
        f"Found: {list(found)}"
    )


@pytest.mark.parametrize("forbidden", _FORBIDDEN_EDGE_TYPES)
def test_state_stack_has_no_cdk_edge_resource(forbidden: str) -> None:
    template = state_template_for()
    found = template.find_resources(forbidden)
    assert found == {}, (
        f"{forbidden} must NOT be CDK-provisioned in the state stack. Found: {list(found)}"
    )


def test_no_acm_or_waf_string_leaks_into_foundation_template() -> None:
    """Belt-and-suspenders: no ACM cert ARN / WAF / Guardrail service string is
    embedded anywhere in the synthesized foundation template JSON.

    A cert ARN belongs on the operator-supplied Helm Ingress annotation, never
    baked into the stack -- this keeps the substrate account-agnostic.
    """
    import json

    blob = json.dumps(template_for().to_json())
    # ':acm:' would appear in any embedded ACM certificate ARN.
    assert ":acm:" not in blob, "ACM ARN must not be baked into the CDK stack"
    assert "AWS::WAFv2" not in blob, "WAF must not be CDK-provisioned (deferred seed)"
    assert "AWS::Bedrock::Guardrail" not in blob, (
        "native Bedrock Guardrail must not be CDK-provisioned (deferred seed)"
    )
