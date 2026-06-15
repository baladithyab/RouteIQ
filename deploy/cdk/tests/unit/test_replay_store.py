"""Unit tests for the ReplayStoreConstruct Aurora Postgres Serverless v2 surface (P1).

Asserts the resources the construct emits per the P1 build outline section 3 /
section 8, synthesised offline (Template.from_stack + dummy env account
``123456789012`` / ``us-west-2``; no AWS creds, no cdk CLI, no network):

  * ONE AWS::RDS::DBCluster: aurora-postgresql, EngineVersion 16.13,
    StorageEncrypted, IAM DB auth, port 5432, ServerlessV2 scaling 0.5-2.0 (dev),
    cloudwatch log export ["postgresql"], NO EngineMode (serverless v2 uses
    provisioned mode under the hood).
  * Two AWS::RDS::DBInstance with DBInstanceClass "db.serverless",
    PubliclyAccessible false.
  * 30-day secret rotation (a RotationSchedule with 30-day rules) + exactly ONE
    generated master secret.
  * The dedicated SG with ingress 5432 from the pod SG AND an SG self-reference
    (the bootstrap Lambda).
  * The schema-bootstrap Lambda (name routeiq-<env>-schema-bootstrap), a
    cr.Provider framework function, and a Custom:: resource (the CustomResource).
  * scale-to-zero branch: min_acu=0 -> MinCapacity 0 + auto-pause property
    present; default (0.5) -> auto-pause property ABSENT (byte-stability proof).
  * removal policy + deletion protection: dev -> Delete + protection off;
    non-dev -> Snapshot + protection on.
  * grant_iam_db_connect: a no-wildcard rds-db:connect statement scoped to the
    RUNTIME user routeiq.

ReplayStoreConstruct takes the P0 KMS CMK / VPC / app SG directly (it does not
own a network), so these tests wrap it in a tiny throwaway stack with a dummy VPC
(both a private-app PRIVATE_WITH_EGRESS tier for the bootstrap Lambda + a
private-data PRIVATE_ISOLATED tier for the cluster), a CMK, and a peer SG -- the
isolated-construct snapshot shape the cache test (and the VSR source) use. The
state-stack-level wiring (cross-stack pod-role grants, outputs) is asserted
separately in the state-stack test once RouteIqStateStack lands.

The schema-bootstrap Lambda's asset dir is pinned MISSING via monkeypatch so the
deterministic inline-fallback path (handler "index.lambda_handler") is exercised
-- this keeps the synth hermetic regardless of Docker / whether the asset dir was
built (the from_asset bundling path is Docker/pip-dependent).
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk.assertions import Match, Template

from lib import replay_store_construct as rs_module
from lib.replay_store_construct import ReplayStoreConstruct

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"


@pytest.fixture(autouse=True)
def _force_inline_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the bootstrap-Lambda asset path MISSING so synth is hermetic.

    The construct branches on os.path.isdir(asset)/isfile(handler.py): present ->
    from_asset bundling (Docker/local-pip dependent, NON-hermetic); absent ->
    Code.from_inline placeholder (handler "index.lambda_handler", deterministic).
    Pointing the global at a non-existent dir forces the deterministic path
    regardless of Docker or whether the real asset dir exists on this host.
    """
    monkeypatch.setattr(
        rs_module,
        "_SCHEMA_BOOTSTRAP_ASSET_PATH",
        "/tmp/routeiq-replay-store-test-does-not-exist",
        raising=True,
    )


def _build(
    *,
    env_name: str = "dev",
    min_acu: float | None = None,
    max_acu: float | None = None,
) -> tuple[cdk.Stack, ReplayStoreConstruct]:
    """Synthesise a throwaway stack wrapping just the ReplayStoreConstruct."""
    app = cdk.App()
    stack = cdk.Stack(
        app,
        f"ReplayStoreTestStack-{env_name}",
        env=cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION),
    )
    # A public tier is required because the private-app tier is
    # PRIVATE_WITH_EGRESS (the bootstrap Lambda's tier), which needs a NAT
    # gateway placed in a public subnet. This mirrors the P0 NetworkConstruct
    # three-tier shape (public / private-app / private-data).
    vpc = ec2.Vpc(
        stack,
        "Vpc",
        max_azs=2,
        nat_gateways=1,
        subnet_configuration=[
            ec2.SubnetConfiguration(
                name="public",
                subnet_type=ec2.SubnetType.PUBLIC,
                cidr_mask=24,
            ),
            ec2.SubnetConfiguration(
                name="private-app",
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                cidr_mask=24,
            ),
            ec2.SubnetConfiguration(
                name="private-data",
                subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                cidr_mask=24,
            ),
        ],
    )
    cmk = kms.Key(stack, "StateCmk", enable_key_rotation=True)
    pod_sg = ec2.SecurityGroup(stack, "PodSg", vpc=vpc, allow_all_outbound=True)

    replay = ReplayStoreConstruct(
        stack,
        "ReplayStoreConstruct",
        env_name=env_name,
        kms_key=cmk,
        vpc=vpc,
        app_sg=pod_sg,
        private_data_subnets=vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
        ).subnets,
        min_acu=min_acu,
        max_acu=max_acu,
    )
    return stack, replay


def _template(**kwargs: Any) -> Template:
    stack, _ = _build(**kwargs)
    return Template.from_stack(stack)


def _props(template: Template, resource_type: str) -> dict:
    resources = template.find_resources(resource_type)
    assert len(resources) == 1, f"expected exactly one {resource_type}, got {len(resources)}"
    return next(iter(resources.values()))["Properties"]


def test_cluster_core_properties() -> None:
    """ONE aurora-postgresql cluster: 16.13, encrypted, IAM auth, 5432, logs."""
    template = _template()
    template.resource_count_is("AWS::RDS::DBCluster", 1)
    props = _props(template, "AWS::RDS::DBCluster")
    assert props["Engine"] == "aurora-postgresql", props
    assert props["EngineVersion"] == "16.13", props["EngineVersion"]
    assert props["StorageEncrypted"] is True, props
    # CFN key for iam_authentication=True.
    assert props.get("EnableIAMDatabaseAuthentication") is True, props
    assert props["Port"] == 5432, props
    assert props["DatabaseName"] == "litellm", props
    assert props["EnableCloudwatchLogsExports"] == ["postgresql"], props
    # KMS CMK wired at rest.
    assert "KmsKeyId" in props and props["KmsKeyId"], props


def test_no_engine_mode() -> None:
    """Serverless v2 uses provisioned engine_mode under the hood -- no EngineMode."""
    props = _props(_template(), "AWS::RDS::DBCluster")
    assert "EngineMode" not in props, (
        f"serverless v2 emits no EngineMode property; found {props.get('EngineMode')!r}"
    )
    # Express the same as a template-level absence match for robustness.
    _template().has_resource_properties("AWS::RDS::DBCluster", {"EngineMode": Match.absent()})


def test_serverless_v2_scaling_default_dev() -> None:
    """dev default ACU window is 0.5-2.0 with NO auto-pause (warm floor)."""
    props = _props(_template(env_name="dev"), "AWS::RDS::DBCluster")
    scaling = props["ServerlessV2ScalingConfiguration"]
    assert scaling["MinCapacity"] == 0.5, scaling
    assert scaling["MaxCapacity"] == 2, scaling
    # Default (min_acu=0.5) => NO auto-pause property (byte-stability proof).
    assert "SecondsUntilAutoPause" not in scaling, scaling


def test_serverless_v2_scale_to_zero_adds_auto_pause() -> None:
    """min_acu=0 -> MinCapacity 0 AND the auto-pause property is present."""
    props = _props(_template(min_acu=0), "AWS::RDS::DBCluster")
    scaling = props["ServerlessV2ScalingConfiguration"]
    assert scaling["MinCapacity"] == 0, scaling
    # CDK renders serverless_v2_auto_pause_duration as SecondsUntilAutoPause
    # (24h = 86400s) inside the scaling block.
    assert scaling.get("SecondsUntilAutoPause") == 86400, scaling


def test_two_serverless_db_instances_private() -> None:
    """Writer + reader are db.serverless instances, neither publicly accessible."""
    template = _template()
    template.resource_count_is("AWS::RDS::DBInstance", 2)
    instances = template.find_resources("AWS::RDS::DBInstance")
    for inst in instances.values():
        props = inst["Properties"]
        assert props["DBInstanceClass"] == "db.serverless", props
        assert props.get("PubliclyAccessible") is False, props


def test_secret_rotation_30_days() -> None:
    """A 30-day single-user rotation schedule + exactly one generated master secret."""
    template = _template()
    # Exactly one generated master secret (CDK owns it; no literal ARN).
    secrets = template.find_resources("AWS::SecretsManager::Secret")
    assert len(secrets) == 1, f"expected one generated master secret, got {len(secrets)}"
    # A rotation schedule with a 30-day cadence.
    schedules = template.find_resources("AWS::SecretsManager::RotationSchedule")
    assert len(schedules) == 1, f"expected one rotation schedule, got {len(schedules)}"
    rules = next(iter(schedules.values()))["Properties"]["RotationRules"]
    assert rules.get("AutomaticallyAfterDays") == 30 or rules.get("ScheduleExpression"), rules


def test_security_group_ingress_5432_from_pod_and_self() -> None:
    """The replay SG has tcp/5432 ingress from the pod SG AND an SG self-reference."""
    template = _template()
    # Both ingress rules render as standalone SecurityGroupIngress (SG-to-SG
    # peers): one from the pod SG, one self-referencing the replay SG.
    ingress = template.find_resources(
        "AWS::EC2::SecurityGroupIngress",
        {"Properties": {"FromPort": 5432, "ToPort": 5432, "IpProtocol": "tcp"}},
    )
    assert len(ingress) == 2, (
        f"expected two 5432 ingress rules (pod SG + self-ref), got {len(ingress)}"
    )


def test_security_group_description_is_ascii() -> None:
    """The replay SG description stays within the EC2 ASCII charset allowlist."""
    template = _template()
    sgs = template.find_resources("AWS::EC2::SecurityGroup")
    aurora_sgs = [
        s["Properties"].get("GroupDescription", "")
        for s in sgs.values()
        if "Aurora" in s["Properties"].get("GroupDescription", "")
    ]
    assert aurora_sgs, "expected an Aurora SG with a control-plane description"
    for desc in aurora_sgs:
        assert desc.isascii(), f"SG description must be ASCII; got {desc!r}"


def test_schema_bootstrap_lambda_and_custom_resource() -> None:
    """The bootstrap Lambda, the cr.Provider framework fn, and the CR all render."""
    template = _template()
    # The named bootstrap Lambda exists.
    functions = template.find_resources("AWS::Lambda::Function")
    names = [
        f["Properties"].get("FunctionName")
        for f in functions.values()
        if f["Properties"].get("FunctionName")
    ]
    assert "routeiq-dev-schema-bootstrap" in names, names
    # The inline-fallback handler name (asset pinned missing in this test).
    bootstrap = next(
        f["Properties"]
        for f in functions.values()
        if f["Properties"].get("FunctionName") == "routeiq-dev-schema-bootstrap"
    )
    assert bootstrap["Handler"] == "index.lambda_handler", bootstrap["Handler"]
    assert bootstrap["Runtime"] == "python3.13", bootstrap["Runtime"]
    # A custom resource backs the schema bootstrap (Custom::* type token).
    customs = {
        rtype: res
        for rtype, res in template.to_json()["Resources"].items()
        if res["Type"].startswith("Custom::")
        or res["Type"] == "AWS::CloudFormation::CustomResource"
    }
    assert customs, "expected a schema-bootstrap CustomResource"


def test_bootstrap_lambda_env_targets_runtime_user_and_litellm_db() -> None:
    """The bootstrap Lambda env names the litellm DB + the routeiq runtime user."""
    template = _template()
    functions = template.find_resources("AWS::Lambda::Function")
    bootstrap = next(
        f["Properties"]
        for f in functions.values()
        if f["Properties"].get("FunctionName") == "routeiq-dev-schema-bootstrap"
    )
    env = bootstrap["Environment"]["Variables"]
    assert env["DB_NAME"] == "litellm", env
    assert env["DB_RUNTIME_USER"] == "routeiq", env
    assert env["DB_PORT"] == "5432", env
    # DB_SECRET_ARN is a token Ref to the generated secret (never a literal).
    assert "DB_SECRET_ARN" in env, env


def test_deletion_protection_and_removal_dev() -> None:
    """dev -> deletion protection OFF + DeletionPolicy Delete (rollback unwinds)."""
    template = _template(env_name="dev")
    clusters = template.find_resources("AWS::RDS::DBCluster")
    cluster = next(iter(clusters.values()))
    props = cluster["Properties"]
    assert props.get("DeletionProtection") in (False, None), props.get("DeletionProtection")
    assert cluster.get("DeletionPolicy") == "Delete", cluster.get("DeletionPolicy")


def test_deletion_protection_and_removal_non_dev() -> None:
    """non-dev -> deletion protection ON + DeletionPolicy Snapshot."""
    template = _template(env_name="prod")
    clusters = template.find_resources("AWS::RDS::DBCluster")
    cluster = next(iter(clusters.values()))
    props = cluster["Properties"]
    assert props.get("DeletionProtection") is True, props.get("DeletionProtection")
    assert cluster.get("DeletionPolicy") == "Snapshot", cluster.get("DeletionPolicy")
    # non-dev default ACU window is 0.5-8.0.
    scaling = props["ServerlessV2ScalingConfiguration"]
    assert scaling["MaxCapacity"] == 8, scaling


def test_grant_iam_db_connect_runtime_user_no_wildcard() -> None:
    """grant_iam_db_connect emits rds-db:connect scoped to dbuser .../routeiq."""
    stack, replay = _build()
    role = iam.Role(
        stack,
        "GrantTestRole",
        assumed_by=iam.ServicePrincipal("pods.eks.amazonaws.com"),
    )
    replay.grant_iam_db_connect(role)
    template = Template.from_stack(stack)
    policies = template.find_resources("AWS::IAM::Policy")
    connect_statements = [
        stmt
        for pol in policies.values()
        for stmt in pol["Properties"]["PolicyDocument"]["Statement"]
        if stmt.get("Action") == "rds-db:connect"
        or (isinstance(stmt.get("Action"), list) and "rds-db:connect" in stmt["Action"])
    ]
    assert connect_statements, "expected an rds-db:connect statement on the role"
    stmt = connect_statements[0]
    assert stmt["Sid"] == "RdsIamDbConnectRouteIqState", stmt
    resource = stmt["Resource"]
    # The resource is a Fn::Join token (the dbuser ARN with the cluster resource
    # identifier interpolated). It must NOT be a bare wildcard, and the runtime
    # user routeiq (NOT the master postgres) must be the dbuser suffix.
    assert resource != "*", resource
    joined = repr(resource)
    assert "/routeiq" in joined, f"grant must target the runtime user routeiq; got {joined}"
    assert "/postgres" not in joined, f"grant must NOT target the master postgres; got {joined}"
    assert ":rds-db:" in joined, joined


def test_public_attrs_exposed() -> None:
    """endpoint / reader_endpoint / port / secret_arn / cluster_resource_identifier."""
    _, replay = _build()
    assert replay.endpoint is not None
    assert replay.reader_endpoint is not None
    assert replay.port == 5432
    assert replay.secret is not None
    assert replay.secret_arn is not None
    assert replay.cluster_resource_identifier is not None
    assert replay.replay_store_sg is not None
