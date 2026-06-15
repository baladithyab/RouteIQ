"""Unit tests for EksClusterConstruct (proposal P0 doc 31 section 12.1).

Asserts the EKS Auto Mode cluster surface RouteIQ provisions through
``RouteIqStack``: the three Auto Mode blocks all ``enabled``, the node
``CfnAccessEntry`` of type ``EC2``, the Pod Identity wiring (the
``CfnPodIdentityAssociation`` over the right ``(namespace, serviceAccount)`` plus
the defensive ``eks-pod-identity-agent`` ``CfnAddon`` it DependsOn), and the
IAM-role-Description-is-ASCII synth-time guard.

This is the Pod-Identity analogue of VSR's IRSA trust-key test
(``test_irsa_trust_keys_strip_https_scheme``): RouteIQ has NO ``CfnJson`` /
no OIDC provider to test (see ``test_no_irsa.py`` for the negative guard), so the
positive assertion is that the static Pod Identity association exists and is
ordered after the agent add-on.

Per mulch ``cdk-resource-count-test-tripwire`` the property assertions use
``has_resource_properties`` / ``find_resources`` / ``Match.object_like`` over
brittle full counts on shared resources. The stack is synthesised offline via the
shared ``make_stack`` / ``template_for`` helpers in ``tests/conftest.py`` (dummy
env account ``123456789012`` / ``us-west-2``).
"""

from __future__ import annotations

import re

from aws_cdk.assertions import Match, Template

from tests.conftest import make_stack, template_for

# The Pod Identity binding RouteIQ wires for the gateway pod (matches cdk.json's
# routeiq:sa_namespace / routeiq:sa_name defaults, mirrored in DEFAULT_STACK_FLAGS).
_GATEWAY_NAMESPACE = "routeiq"
_GATEWAY_SERVICE_ACCOUNT = "routeiq-gateway"
_POD_IDENTITY_ADDON_NAME = "eks-pod-identity-agent"

# IAM's allowed Description charset: the ASCII control trio (tab / LF / CR), the
# printable ASCII range, and the Latin-1 supplement. Built from explicit \u
# escapes so the test source itself stays plain ASCII. An em-dash (U+2014) is
# OUTSIDE this set, which is the failure mode being guarded.
_IAM_DESCRIPTION_CHARSET = re.compile("^[" + "\t\n\r" + " -~" + "\u00a1-\u00ff" + "]*$")


def test_auto_mode_three_blocks_enabled() -> None:
    """Auto Mode = compute_config + storage block_storage + ELB all enabled.

    These are the three L1 ``CfnCluster`` blocks that, toggled together, put the
    data plane fully under AWS management (proposal section 7.1). ComputeConfig
    also pins the two AWS-managed node pools.
    """
    template = template_for()
    template.has_resource_properties(
        "AWS::EKS::Cluster",
        {
            "ComputeConfig": Match.object_like(
                {
                    "Enabled": True,
                    "NodePools": ["general-purpose", "system"],
                }
            ),
            "StorageConfig": {"BlockStorage": {"Enabled": True}},
            "KubernetesNetworkConfig": {"ElasticLoadBalancing": {"Enabled": True}},
        },
    )


def test_private_endpoint_only() -> None:
    """The control plane is private-only at P0 (proposal section 7.1 / 11.1a)."""
    template = template_for()
    template.has_resource_properties(
        "AWS::EKS::Cluster",
        {
            "ResourcesVpcConfig": Match.object_like(
                {
                    "EndpointPrivateAccess": True,
                    "EndpointPublicAccess": False,
                }
            )
        },
    )


def test_api_auth_mode_with_creator_admin() -> None:
    """API access-entry auth mode + creator-admin so kubectl works post-deploy."""
    template = template_for()
    template.has_resource_properties(
        "AWS::EKS::Cluster",
        {
            "AccessConfig": Match.object_like(
                {
                    "AuthenticationMode": "API",
                    "BootstrapClusterCreatorAdminPermissions": True,
                }
            )
        },
    )


def test_node_access_entry_is_type_ec2() -> None:
    """The Auto Mode node role's CfnAccessEntry is type ``EC2`` (proposal 7.3a).

    An EKS API constraint forbids access policies on EC2-type entries, so the
    node entry carries NO AccessPolicies. Assert there is an EC2-type entry for
    the node role.
    """
    template = template_for()
    entries = template.find_resources("AWS::EKS::AccessEntry")
    ec2_types = [e["Properties"].get("Type") for e in entries.values()]
    assert "EC2" in ec2_types, f"expected an EC2-type node access entry, got {ec2_types}"
    # The EC2 entry must not carry access policies (AWS rejects them on EC2 type).
    for e in entries.values():
        if e["Properties"].get("Type") == "EC2":
            assert "AccessPolicies" not in e["Properties"], (
                "EC2-type access entry must not carry AccessPolicies"
            )


def test_pod_identity_association_for_gateway_sa() -> None:
    """A CfnPodIdentityAssociation binds the gateway (namespace, serviceAccount).

    The Pod Identity analogue of VSR's IRSA trust-key test: the pod->role
    binding is keyed on ``(routeiq, routeiq-gateway)`` (proposal section 7.4 /
    11.2). There are TWO associations on the default surface (the gateway pod +
    the cloudwatch-agent for Container Insights), so select by the namespace/SA
    pair rather than asserting a single resource.
    """
    template = template_for()
    associations = template.find_resources("AWS::EKS::PodIdentityAssociation")
    pairs = {
        (a["Properties"].get("Namespace"), a["Properties"].get("ServiceAccount"))
        for a in associations.values()
    }
    assert (_GATEWAY_NAMESPACE, _GATEWAY_SERVICE_ACCOUNT) in pairs, (
        f"gateway Pod Identity association {(_GATEWAY_NAMESPACE, _GATEWAY_SERVICE_ACCOUNT)} "
        f"not found; got {pairs}"
    )


def test_pod_identity_agent_addon_overwrite() -> None:
    """The defensive eks-pod-identity-agent add-on exists with OVERWRITE.

    AWS docs claim the agent is built into Auto Mode, but the production VSR
    construct installs it by hand; RouteIQ adds it idempotently
    (``resolve_conflicts=OVERWRITE``) so the association is guaranteed to resolve
    (proposal section 7.4a). Select the add-on by name via find_resources.
    """
    template = template_for()
    addons = template.find_resources("AWS::EKS::Addon")
    agent = [
        a for a in addons.values() if a["Properties"].get("AddonName") == _POD_IDENTITY_ADDON_NAME
    ]
    assert agent, (
        f"{_POD_IDENTITY_ADDON_NAME} add-on not found; "
        f"got {[a['Properties'].get('AddonName') for a in addons.values()]}"
    )
    assert agent[0]["Properties"].get("ResolveConflicts") == "OVERWRITE", (
        "eks-pod-identity-agent add-on must set ResolveConflicts=OVERWRITE so "
        "re-applying a built-in add-on is a harmless no-op"
    )


def test_gateway_association_depends_on_pod_identity_agent_addon() -> None:
    """The gateway association DependsOn the eks-pod-identity-agent add-on.

    Resolves the proposal section 3.1 #2 contradiction: the association must not
    race the agent, so it carries a CFN ``DependsOn`` on the add-on's logical id
    regardless of whether Auto Mode pre-installs the agent.
    """
    template = Template.from_stack(make_stack())
    res = template.to_json()["Resources"]

    # Find the eks-pod-identity-agent add-on logical id.
    agent_ids = [
        lid
        for lid, r in res.items()
        if r["Type"] == "AWS::EKS::Addon"
        and r["Properties"].get("AddonName") == _POD_IDENTITY_ADDON_NAME
    ]
    assert len(agent_ids) == 1, f"expected exactly one pod-identity-agent add-on, got {agent_ids}"
    agent_id = agent_ids[0]

    # Find the gateway association logical id (by its namespace/SA).
    gateway_ids = [
        lid
        for lid, r in res.items()
        if r["Type"] == "AWS::EKS::PodIdentityAssociation"
        and r["Properties"].get("Namespace") == _GATEWAY_NAMESPACE
        and r["Properties"].get("ServiceAccount") == _GATEWAY_SERVICE_ACCOUNT
    ]
    assert len(gateway_ids) == 1, f"expected one gateway association, got {gateway_ids}"
    depends_on = res[gateway_ids[0]].get("DependsOn") or []
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    assert agent_id in depends_on, (
        f"gateway Pod Identity association must DependsOn the {_POD_IDENTITY_ADDON_NAME} "
        f"add-on ({agent_id}); DependsOn={depends_on}"
    )


def test_pod_identity_agent_addon_depends_on_cluster() -> None:
    """The agent add-on DependsOn the cluster (it cannot install before it exists)."""
    template = Template.from_stack(make_stack())
    res = template.to_json()["Resources"]
    cluster_ids = [lid for lid, r in res.items() if r["Type"] == "AWS::EKS::Cluster"]
    assert len(cluster_ids) == 1, cluster_ids
    cluster_id = cluster_ids[0]
    agent = next(
        r
        for r in res.values()
        if r["Type"] == "AWS::EKS::Addon"
        and r["Properties"].get("AddonName") == _POD_IDENTITY_ADDON_NAME
    )
    depends_on = agent.get("DependsOn") or []
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    assert cluster_id in depends_on, (
        f"eks-pod-identity-agent add-on must DependsOn the cluster ({cluster_id}); "
        f"DependsOn={depends_on}"
    )


def test_container_insights_addon_and_log_group() -> None:
    """P0 observability: amazon-cloudwatch-observability add-on + routing log group.

    enable_container_insights() runs in the stack at P0 (proposal section 7.5).
    Assert the add-on is present and the CDK-created routing log group exists so
    the deferred metric filters have a group to attach to.
    """
    template = template_for()
    addons = template.find_resources("AWS::EKS::Addon")
    names = {a["Properties"].get("AddonName") for a in addons.values()}
    assert "amazon-cloudwatch-observability" in names, names

    log_groups = template.find_resources("AWS::Logs::LogGroup")
    names_seen = {
        lg["Properties"].get("LogGroupName")
        for lg in log_groups.values()
        if isinstance(lg["Properties"].get("LogGroupName"), str)
    }
    assert "/aws/containerinsights/routeiq-dev/routeiq-routing" in names_seen, names_seen


def test_routing_latency_by_model_metric_filter_deferred() -> None:
    """RoutingLatencyByModel MetricFilter is PREP-ONLY / deferred at P0.

    It is data-source-blocked until P2 and flag-gated OFF in the construct
    (proposal section 7.5 / ADR-0027), so NO ``AWS::Logs::MetricFilter`` is
    emitted at the default P0 surface. (When it is later enabled it must
    dimension on ``$.["gen_ai.response.model"]``, not ``$.selected_model``.)
    """
    template = template_for()
    template.resource_count_is("AWS::Logs::MetricFilter", 0)


def test_admin_access_entries_when_configured() -> None:
    """Extra admin principals each get a STANDARD CfnAccessEntry + cluster-admin.

    Driven by routeiq:admin_principal_arns (proposal section 7.3b). The node
    EC2-type entry is always present, so assert the supplied admin ARNs each
    appear as a STANDARD entry carrying AmazonEKSClusterAdminPolicy.
    """
    arns = [
        "arn:aws:iam::123456789012:role/Admin",
        "arn:aws:iam::123456789012:role/CiCd",
    ]
    template = template_for(admin_principal_arns=arns)
    entries = template.find_resources("AWS::EKS::AccessEntry")
    standard = {
        e["Properties"]["PrincipalArn"]
        for e in entries.values()
        if e["Properties"].get("Type") == "STANDARD"
    }
    assert set(arns) <= standard, f"admin entries missing; got {standard}"
    for e in entries.values():
        if e["Properties"].get("Type") != "STANDARD":
            continue
        policies = e["Properties"].get("AccessPolicies", [])
        assert "AmazonEKSClusterAdminPolicy" in str(policies), policies


def test_no_admin_access_entries_by_default() -> None:
    """Default (no admin_principal_arns) emits ONLY the EC2 node entry, no STANDARD."""
    template = template_for()
    entries = template.find_resources("AWS::EKS::AccessEntry")
    types = [e["Properties"].get("Type") for e in entries.values()]
    assert "STANDARD" not in types, f"unexpected STANDARD access entry on default surface: {types}"


def test_iam_role_descriptions_are_ascii() -> None:
    """Every IAM role Description is ASCII / Latin-1 only (proposal section 4.5).

    An em-dash (U+2014) passes ``cdk synth`` but FAILS the IAM CREATE API. Guard
    so a stray non-Latin-1 char in any role description is caught at synth, not at
    deploy. IAM's allowed charset is the ASCII control trio + printable ASCII +
    Latin-1 supplement (see ``_IAM_DESCRIPTION_CHARSET``).
    """
    template = template_for()
    roles = template.find_resources("AWS::IAM::Role")
    for logical, role in roles.items():
        desc = role["Properties"].get("Description")
        if isinstance(desc, str):
            assert _IAM_DESCRIPTION_CHARSET.match(desc), (
                f"IAM role {logical} Description has a char outside IAM's allowed "
                f"Latin-1 set (e.g. an em-dash): {desc!r}"
            )
