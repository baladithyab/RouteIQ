"""Unit tests for the RouteIqStack composition root (section 12.4).

Asserts the stack-level wiring that the constructs do not own on their own: the
ONE gateway pod IAM role's exact P0 grant set and its static Pod Identity trust,
the six operator-visible CfnOutputs, and the VPC CIDR.

  - The pod role carries EXACTLY the P0 grant set on the default surface
    (Bedrock invoke + Secrets + Logs; S3 only when a config bucket is supplied -
    proposal section 4.2). No extra statements leak in.
  - The trust is a STATIC ``pods.eks.amazonaws.com`` service-principal grant of
    ``sts:AssumeRole`` + ``sts:TagSession`` - and crucially NO
    ``sts:AssumeRoleWithWebIdentity`` (Pod Identity does not use it; the IRSA
    web-identity trust is absent - the complement of test_no_irsa.py).
  - The six stack-level CfnOutputs exist (the IRSA-only OidcProvider* outputs are
    dropped; PodAssociationId is the new binding output).
  - VPC CIDR is 10.40.0.0/16 (distinct from VSR's 10.20 / 10.30 so a future
    peering stays overlap-free).

Synthesised offline via the shared helpers (dummy env account ``123456789012`` /
``us-west-2``).
"""

from __future__ import annotations

import json

from tests.conftest import template_for

# The exact P0 default grant set (proposal section 4.2). ConfigS3Read is added
# ONLY when a config bucket is supplied, so it is NOT in the default set.
_DEFAULT_POD_ROLE_SIDS = {"BedrockInvoke", "SecretsRead", "Logs"}

# The six operator-visible stack-level CfnOutputs (proposal section 7.6 / 10).
# The constructs also emit their own construct-prefixed outputs; these are the
# stack-root ones the chart / CI consume by exact logical id.
_EXPECTED_OUTPUTS = {
    "ClusterName",
    "ClusterEndpoint",
    "PodRoleArn",
    "PodAssociationId",
    "NodeRoleName",
    "EcrGhcrPrefix",
}


def _pod_role_policy(template) -> dict:
    """Return the single PodRole DefaultPolicy resource's Properties."""
    policies = template.find_resources("AWS::IAM::Policy")
    pod_policies = [p for lid, p in policies.items() if "PodRole" in lid]
    assert len(pod_policies) == 1, f"expected exactly one PodRole policy, got {len(pod_policies)}"
    return pod_policies[0]["Properties"]


def _pod_role(template) -> dict:
    """Return the single PodRole AWS::IAM::Role resource's Properties."""
    roles = template.find_resources("AWS::IAM::Role")
    pod_roles = [r for lid, r in roles.items() if lid.startswith("PodRole")]
    assert len(pod_roles) == 1, f"expected exactly one PodRole, got {len(pod_roles)}"
    return pod_roles[0]["Properties"]


def test_pod_role_default_grant_set_exact() -> None:
    """The default pod role has EXACTLY {BedrockInvoke, SecretsRead, Logs}."""
    template = template_for()
    props = _pod_role_policy(template)
    sids = {s.get("Sid") for s in props["PolicyDocument"]["Statement"]}
    assert sids == _DEFAULT_POD_ROLE_SIDS, (
        f"default pod role grant set must be exactly {_DEFAULT_POD_ROLE_SIDS}; got {sids}"
    )


def test_pod_role_bedrock_actions() -> None:
    """BedrockInvoke grants the four Bedrock invoke/converse actions."""
    template = template_for()
    props = _pod_role_policy(template)
    bedrock = next(
        s for s in props["PolicyDocument"]["Statement"] if s.get("Sid") == "BedrockInvoke"
    )
    actions = set(bedrock["Action"])
    assert actions == {
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:Converse",
        "bedrock:ConverseStream",
    }, actions


def test_pod_role_secrets_and_logs_actions() -> None:
    """SecretsRead + Logs grant the expected least-privilege actions."""
    template = template_for()
    props = _pod_role_policy(template)
    stmts = {s["Sid"]: s for s in props["PolicyDocument"]["Statement"]}
    assert set(stmts["SecretsRead"]["Action"]) == {
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret",
    }, stmts["SecretsRead"]
    assert set(stmts["Logs"]["Action"]) == {
        "logs:CreateLogStream",
        "logs:PutLogEvents",
    }, stmts["Logs"]


def test_pod_role_config_s3_added_only_when_bucket_supplied() -> None:
    """ConfigS3Read appears ONLY when routeiq:config_s3_bucket is supplied."""
    # Default: absent.
    default = _pod_role_policy(template_for())
    default_sids = {s.get("Sid") for s in default["PolicyDocument"]["Statement"]}
    assert "ConfigS3Read" not in default_sids, default_sids

    # With a bucket: present and scoped to the bucket/* object wildcard.
    with_bucket = _pod_role_policy(template_for(config_s3_bucket="routeiq-config-dev"))
    stmts = {s.get("Sid"): s for s in with_bucket["PolicyDocument"]["Statement"]}
    assert "ConfigS3Read" in stmts, list(stmts)
    s3 = stmts["ConfigS3Read"]
    assert set(s3["Action"]) == {"s3:GetObject", "s3:GetObjectAttributes"}, s3
    assert "routeiq-config-dev/*" in json.dumps(s3["Resource"]), s3["Resource"]


def test_pod_role_trust_is_static_pod_identity() -> None:
    """Trust = static pods.eks.amazonaws.com grant of AssumeRole + TagSession."""
    template = template_for()
    props = _pod_role(template)
    arpd = props["AssumeRolePolicyDocument"]
    actions_by_principal: dict[str, set[str]] = {}
    for s in arpd["Statement"]:
        principal = s.get("Principal", {})
        service = principal.get("Service")
        action = s.get("Action")
        acts = {action} if isinstance(action, str) else set(action)
        if isinstance(service, str):
            actions_by_principal.setdefault(service, set()).update(acts)
    assert "pods.eks.amazonaws.com" in actions_by_principal, actions_by_principal
    pod_acts = actions_by_principal["pods.eks.amazonaws.com"]
    assert "sts:AssumeRole" in pod_acts, pod_acts
    assert "sts:TagSession" in pod_acts, pod_acts


def test_pod_role_has_no_web_identity_trust() -> None:
    """NO sts:AssumeRoleWithWebIdentity anywhere (the IRSA trust is absent).

    The complement of test_no_irsa.py: Pod Identity creds come from the
    pod-identity agent, NOT an STS web-identity exchange, so the action is never
    granted on any role in the stack.
    """
    template = template_for()
    blob = json.dumps(template.to_json())
    assert "AssumeRoleWithWebIdentity" not in blob, (
        "sts:AssumeRoleWithWebIdentity must NOT appear anywhere - RouteIQ uses "
        "EKS Pod Identity, not IRSA web-identity"
    )


def test_pod_role_trust_has_no_web_identity_action() -> None:
    """Specifically, the pod role's trust document has no web-identity action."""
    template = template_for()
    props = _pod_role(template)
    actions: set[str] = set()
    for s in props["AssumeRolePolicyDocument"]["Statement"]:
        action = s.get("Action")
        if isinstance(action, str):
            actions.add(action)
        elif isinstance(action, list):
            actions.update(action)
    assert "sts:AssumeRoleWithWebIdentity" not in actions, actions


def test_six_stack_outputs_exist() -> None:
    """The six operator-visible stack-level CfnOutputs are present."""
    template = template_for()
    found = set(template.to_json().get("Outputs", {}))
    missing = _EXPECTED_OUTPUTS - found
    assert not missing, f"missing stack outputs {missing}; found {sorted(found)}"


def test_vpc_cidr_is_routeiq_block() -> None:
    """The VPC CIDR is 10.40.0.0/16 (proposal section 8.1)."""
    template = template_for()
    template.has_resource_properties("AWS::EC2::VPC", {"CidrBlock": "10.40.0.0/16"})


def test_pod_role_descriptions_ascii() -> None:
    """The pod role Description is ASCII / Latin-1 only (proposal section 4.5)."""
    template = template_for()
    desc = _pod_role(template).get("Description")
    assert isinstance(desc, str)
    # An em-dash (U+2014) would fail the IAM CREATE API; assert plain ASCII here
    # since the P0 pod-role description is pure ASCII.
    assert desc.isascii(), f"pod role Description must be ASCII; got {desc!r}"
