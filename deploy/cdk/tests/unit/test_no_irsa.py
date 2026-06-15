"""Pod-Identity-over-IRSA regression guard (proposal P0 doc 31 section 12.2).

This is the NEW negative regression guard that REPLACES VSR's
``test_irsa_trust_keys_strip_https_scheme``. RouteIQ took the Pod Identity path
(``eks.CfnPodIdentityAssociation`` over a static ``pods.eks.amazonaws.com``
trust), so there is NO ``CfnJson`` TrustKeys resource and NO OIDC provider to
test. The whole IRSA token-keyed-trust surface is DELETED.

The guard proves that:

  - NO ``Custom::AWSCDKCfnJson`` resource exists (CDK emits one of these for the
    IRSA trust-condition map; its absence proves no IRSA trust map was built).
  - NO OIDC provider exists in either CFN form: the native
    ``AWS::IAM::OIDCProvider`` (proposal wording) OR CDK's custom-resource
    ``Custom::AWSCDKOpenIdConnectProvider`` (the form an L2
    ``OpenIdConnectProvider`` actually synthesises to). Both forms are asserted
    absent so an accidental IRSA regression cannot creep back in via either path.

If either resource ever reappears, an IRSA regression has crept in and this test
fails LOUD. The stack synthesises offline via ``template_for`` (dummy env
account ``123456789012`` / ``us-west-2``).
"""

from __future__ import annotations

from tests.conftest import template_for


def test_no_cfn_json_trust_map() -> None:
    """NO Custom::AWSCDKCfnJson: the IRSA trust-condition map was never built."""
    template = template_for()
    cfn_json = template.find_resources("Custom::AWSCDKCfnJson")
    assert cfn_json == {}, (
        "An IRSA regression crept in: a Custom::AWSCDKCfnJson resource exists. "
        "RouteIQ uses EKS Pod Identity (a static pods.eks.amazonaws.com trust), "
        f"which needs NO CfnJson trust map. Found: {list(cfn_json)}"
    )


def test_no_oidc_provider() -> None:
    """NO OIDC provider in either CFN form (native or CDK custom resource)."""
    template = template_for()
    native = template.find_resources("AWS::IAM::OIDCProvider")
    assert native == {}, (
        "An IRSA regression crept in: an AWS::IAM::OIDCProvider exists. RouteIQ "
        "uses EKS Pod Identity and provisions NO OIDC provider. "
        f"Found: {list(native)}"
    )
    # The L2 OpenIdConnectProvider synthesises to a CDK custom resource, not the
    # native type, so assert that form is absent too.
    custom = template.find_resources("Custom::AWSCDKOpenIdConnectProvider")
    assert custom == {}, (
        "An IRSA regression crept in: a Custom::AWSCDKOpenIdConnectProvider "
        "(CDK's L2 OIDC provider) exists. RouteIQ uses EKS Pod Identity and "
        f"provisions NO OIDC provider. Found: {list(custom)}"
    )


def test_no_irsa_under_ghcr_disabled_surface() -> None:
    """The Pod Identity path holds with the GHCR PTC flag off too (byte-stable).

    Flipping enable_ghcr_ptc=False removes the ECR PTC surface but must not
    reintroduce any IRSA surface.
    """
    template = template_for(enable_ghcr_ptc=False)
    assert template.find_resources("Custom::AWSCDKCfnJson") == {}
    assert template.find_resources("AWS::IAM::OIDCProvider") == {}
    assert template.find_resources("Custom::AWSCDKOpenIdConnectProvider") == {}
