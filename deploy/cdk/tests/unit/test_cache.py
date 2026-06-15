"""Unit tests for the CacheConstruct ElastiCache Serverless Valkey surface (P1).

Asserts the resources the construct emits per the P1 build outline section 4 /
section 8, synthesised offline (Template.from_stack + dummy env account
``123456789012`` / ``us-west-2``; no AWS creds, no cdk CLI, no network):

  * ONE AWS::ElastiCache::ServerlessCache (engine valkey, MajorEngineVersion "8"
    -- the MAJOR-only split derivation, NOT "8.0"; name routeiq-<env>-cache-sl;
    KMS at rest; snapshot 7d @ 03:00; user-group wired).
  * The IAM-auth CfnUser (UserId == UserName, AuthenticationMode iam,
    AccessString "on ~* +@all").
  * The disabled default CfnUser (UserName "default", password mode, inert pw,
    AccessString "off ~keys* -@all").
  * The CfnUserGroup with BOTH users and an explicit DependsOn each.
  * NO vestigial AWS::ElastiCache::SubnetGroup (serverless wires subnet_ids flat).
  * Always-on TLS: no TransitEncryptionEnabled toggle on the serverless cache.
  * Dedicated SG with the late-bound 6379 ingress from the pod SG after
    attach_dependencies; idempotent on a double call.
  * removal policy: dev -> Delete, non-dev -> Retain.
  * grant_iam_connect: a no-wildcard elasticache:Connect statement on cache +
    user ARNs.

CacheConstruct takes the P0 VPC / private-data subnets / KMS CMK directly (it
does not own a network), so these tests wrap it in a tiny throwaway stack with a
dummy VPC, a CMK, and a peer SG -- the isolated-construct snapshot shape the VSR
source uses. The state-stack-level wiring (cross-stack pod-role grants, outputs)
is asserted separately in the state-stack test once RouteIqStateStack lands.
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk.assertions import Template

from lib.cache_construct import CacheConstruct

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"


def _build(*, env_name: str = "dev", attach: bool = True) -> tuple[cdk.Stack, CacheConstruct]:
    """Synthesise a throwaway stack wrapping just the CacheConstruct.

    A dummy VPC supplies private-isolated subnets; a state-owned CMK supplies the
    at-rest key; a peer SG stands in for the P0 pod_sg. Offline, no creds.
    """
    app = cdk.App()
    stack = cdk.Stack(
        app,
        f"CacheTestStack-{env_name}",
        env=cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION),
    )
    vpc = ec2.Vpc(
        stack,
        "Vpc",
        max_azs=2,
        subnet_configuration=[
            ec2.SubnetConfiguration(
                name="private-data",
                subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                cidr_mask=24,
            ),
        ],
    )
    cmk = kms.Key(stack, "StateCmk", enable_key_rotation=True)
    pod_sg = ec2.SecurityGroup(stack, "PodSg", vpc=vpc, allow_all_outbound=True)

    cache = CacheConstruct(
        stack,
        "CacheConstruct",
        env_name=env_name,
        vpc=vpc,
        private_data_subnets=vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
        ).subnets,
        kms_key=cmk,
    )
    if attach:
        cache.attach_dependencies(app_sg=pod_sg)
    return stack, cache


def _template(*, env_name: str = "dev", attach: bool = True) -> Template:
    stack, _ = _build(env_name=env_name, attach=attach)
    return Template.from_stack(stack)


def _props(template: Template, resource_type: str) -> dict:
    resources = template.find_resources(resource_type)
    assert len(resources) == 1, f"expected exactly one {resource_type}, got {len(resources)}"
    return next(iter(resources.values()))["Properties"]


def test_serverless_cache_core_properties() -> None:
    """ONE valkey serverless cache: MajorEngineVersion '8', -sl name, KMS, snapshots."""
    template = _template()
    template.resource_count_is("AWS::ElastiCache::ServerlessCache", 1)
    props = _props(template, "AWS::ElastiCache::ServerlessCache")
    assert props["Engine"] == "valkey", props
    # MAJOR ONLY -- the split("8.0")[0] derivation, NOT "8.0".
    assert props["MajorEngineVersion"] == "8", props["MajorEngineVersion"]
    assert props["ServerlessCacheName"] == "routeiq-dev-cache-sl", props
    assert props["SnapshotRetentionLimit"] == 7, props
    assert props["DailySnapshotTime"] == "03:00", props
    # KMS at-rest key present (a Ref / GetAtt token to the CMK, not absent).
    assert "KmsKeyId" in props and props["KmsKeyId"], props


def test_major_engine_version_is_major_only() -> None:
    """MajorEngineVersion is exactly '8' (serverless takes the major, not '8.0')."""
    props = _props(_template(), "AWS::ElastiCache::ServerlessCache")
    assert props["MajorEngineVersion"] == "8", props["MajorEngineVersion"]
    assert props["MajorEngineVersion"] != "8.0"


def test_iam_auth_user() -> None:
    """IAM-auth user: UserId == UserName, AuthenticationMode iam, all keys/commands."""
    template = _template()
    users = template.find_resources("AWS::ElastiCache::User")
    iam_users = [
        u["Properties"]
        for u in users.values()
        if u["Properties"].get("UserName") == "routeiq-dev-cache-iam-user"
    ]
    assert len(iam_users) == 1, f"expected one IAM user, got {len(iam_users)}"
    props = iam_users[0]
    # IAM auth REQUIRES UserId == UserName.
    assert props["UserId"] == props["UserName"] == "routeiq-dev-cache-iam-user", props
    assert props["Engine"] == "valkey", props
    assert props["AuthenticationMode"] == {"Type": "iam"}, props["AuthenticationMode"]
    assert props["AccessString"] == "on ~* +@all", props


def test_disabled_default_user() -> None:
    """Default user disabled: UserName 'default', password mode, 'off' access string."""
    template = _template()
    users = template.find_resources("AWS::ElastiCache::User")
    default_users = [
        u["Properties"] for u in users.values() if u["Properties"].get("UserName") == "default"
    ]
    assert len(default_users) == 1, f"expected one default user, got {len(default_users)}"
    props = default_users[0]
    # The reserved default user keeps user_name="default" but a unique user_id.
    assert props["UserId"] == "routeiq-dev-cache-default", props
    assert props["UserName"] == "default", props
    # IAM is impossible on default (id != name); PASSWORD with an inert pw is the
    # only mode Valkey accepts. The access string disables all commands.
    assert props["AuthenticationMode"]["Type"] == "password", props["AuthenticationMode"]
    assert props["AccessString"] == "off ~keys* -@all", props


def test_user_group_depends_on_both_users() -> None:
    """User group references both users and DependsOn each (string list needs it)."""
    template = _template()
    template.resource_count_is("AWS::ElastiCache::User", 2)
    groups = template.find_resources("AWS::ElastiCache::UserGroup")
    assert len(groups) == 1, f"expected one user group, got {len(groups)}"
    group = next(iter(groups.values()))
    props = group["Properties"]
    assert props["UserGroupId"] == "routeiq-dev-cache-ug", props
    # Both user ids must be members.
    user_ids = props["UserIds"]
    assert len(user_ids) == 2, user_ids
    # CDK cannot infer the dep from a string list, so both add_dependency calls
    # MUST render an explicit DependsOn on the group.
    depends_on = group.get("DependsOn") or []
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    assert len(depends_on) >= 2, (
        f"user group must DependsOn both CfnUser resources; got {depends_on}"
    )


def test_no_subnet_group() -> None:
    """NO vestigial AWS::ElastiCache::SubnetGroup (serverless wires subnet_ids flat)."""
    template = _template()
    subnet_groups = template.find_resources("AWS::ElastiCache::SubnetGroup")
    assert subnet_groups == {}, (
        "CacheConstruct must emit NO CfnSubnetGroup (serverless takes a flat "
        f"subnet_ids list); found {list(subnet_groups)}"
    )


def test_always_on_tls_has_no_toggle() -> None:
    """Serverless TLS is always-on: no TransitEncryptionEnabled property to set."""
    props = _props(_template(), "AWS::ElastiCache::ServerlessCache")
    assert "TransitEncryptionEnabled" not in props, (
        "serverless caches have no TLS toggle (always-on); "
        f"unexpected TransitEncryptionEnabled in {props}"
    )


def test_security_group_ingress_6379_from_pod_sg() -> None:
    """After attach_dependencies the redis SG has tcp/6379 ingress from the peer SG."""
    template = _template(attach=True)
    # The ingress rule renders as a standalone SecurityGroupIngress (peer is an
    # SG-to-SG ref) referencing port 6379.
    ingress = template.find_resources(
        "AWS::EC2::SecurityGroupIngress",
        {"Properties": {"FromPort": 6379, "ToPort": 6379, "IpProtocol": "tcp"}},
    )
    assert len(ingress) == 1, f"expected one 6379 ingress rule, got {len(ingress)}"


def test_no_ingress_before_attach() -> None:
    """Without attach_dependencies the construct emits no 6379 ingress (late-bound)."""
    template = _template(attach=False)
    ingress = template.find_resources(
        "AWS::EC2::SecurityGroupIngress",
        {"Properties": {"FromPort": 6379}},
    )
    assert ingress == {}, f"6379 ingress must be late-bound; found {list(ingress)}"


def test_attach_dependencies_is_idempotent() -> None:
    """A double attach_dependencies adds no duplicate ingress rule (guard works)."""
    stack, cache = _build(attach=True)
    pod_sg = ec2.SecurityGroup(stack, "SecondPodSg", vpc=cache._vpc, allow_all_outbound=True)
    cache.attach_dependencies(app_sg=pod_sg)  # second call -- must be a no-op
    template = Template.from_stack(stack)
    ingress = template.find_resources(
        "AWS::EC2::SecurityGroupIngress",
        {"Properties": {"FromPort": 6379}},
    )
    assert len(ingress) == 1, (
        f"second attach must not add a duplicate ingress rule; got {len(ingress)}"
    )


def test_security_group_description_is_ascii() -> None:
    """The redis SG description stays within the EC2 ASCII charset allowlist."""
    template = _template()
    sgs = template.find_resources("AWS::EC2::SecurityGroup")
    redis_sgs = [
        s["Properties"].get("GroupDescription", "")
        for s in sgs.values()
        if "ElastiCache" in s["Properties"].get("GroupDescription", "")
    ]
    assert redis_sgs, "expected a redis SG with an ElastiCache description"
    for desc in redis_sgs:
        assert desc.isascii(), f"SG description must be ASCII; got {desc!r}"


def test_removal_policy_dev_destroy() -> None:
    """dev -> the serverless cache has DeletionPolicy Delete (rollback unwinds clean)."""
    template = _template(env_name="dev")
    caches = template.find_resources("AWS::ElastiCache::ServerlessCache")
    cache = next(iter(caches.values()))
    assert cache.get("DeletionPolicy") == "Delete", cache.get("DeletionPolicy")


def test_removal_policy_non_dev_retain() -> None:
    """non-dev -> the serverless cache has DeletionPolicy Retain (protect the store)."""
    template = _template(env_name="prod")
    caches = template.find_resources("AWS::ElastiCache::ServerlessCache")
    cache = next(iter(caches.values()))
    assert cache.get("DeletionPolicy") == "Retain", cache.get("DeletionPolicy")
    # The name carries the env, too.
    assert cache["Properties"]["ServerlessCacheName"] == "routeiq-prod-cache-sl"


def test_grant_iam_connect_no_wildcard() -> None:
    """grant_iam_connect emits a no-wildcard elasticache:Connect on cache + user."""
    stack, cache = _build()
    role = iam.Role(
        stack,
        "GrantTestRole",
        assumed_by=iam.ServicePrincipal("pods.eks.amazonaws.com"),
    )
    cache.grant_iam_connect(role)
    template = Template.from_stack(stack)
    policies = template.find_resources("AWS::IAM::Policy")
    connect_statements = [
        stmt
        for pol in policies.values()
        for stmt in pol["Properties"]["PolicyDocument"]["Statement"]
        if stmt.get("Action") == "elasticache:Connect"
        or (isinstance(stmt.get("Action"), list) and "elasticache:Connect" in stmt["Action"])
    ]
    assert connect_statements, "expected an elasticache:Connect statement on the role"
    stmt = connect_statements[0]
    assert stmt["Sid"] == "ElastiCacheConnectRouteIqState", stmt
    resources = stmt["Resource"]
    if not isinstance(resources, list):
        resources = [resources]
    # Two fully-specified resources (cache ARN + user ARN), no bare wildcard.
    assert len(resources) == 2, f"expected cache + user ARNs; got {resources}"
    for res in resources:
        # Each resource is an ARN token (str with the hand-built user ARN, or a
        # GetAtt for the cache). A bare "*" is forbidden.
        assert res != "*", f"elasticache:Connect must not target *; got {resources}"


def test_public_attrs_are_tokens() -> None:
    """endpoint_address / endpoint_port / cache_arn / iam_user_arn are exposed."""
    _, cache = _build()
    assert cache.endpoint_address is not None
    assert cache.endpoint_port is not None
    assert cache.cache_arn is not None
    assert cache.iam_user_name == "routeiq-dev-cache-iam-user"
    # iam_user_arn is hand-built and ends with the user name.
    assert ":user:routeiq-dev-cache-iam-user" in cache.iam_user_arn
