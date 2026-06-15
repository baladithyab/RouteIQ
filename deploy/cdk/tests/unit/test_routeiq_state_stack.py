"""Unit tests for RouteIqStateStack - the SEPARATE P1 state-plane stack (ADR-0028/0029).

These assert the STACK-LEVEL wiring (cross-stack inputs, the two stores composed
together, the cross-stack pod-role grants, the chart-facing CfnOutputs, and the
cred-free synth) rather than the construct internals (those are covered by
``test_replay_store.py`` / ``test_cache.py``).

All synth is offline: one ``cdk.App`` holds the P0 ``RouteIqStack`` + the P1
``RouteIqStateStack`` wired by reference (``foundation=``), dummy env account
``123456789012`` / ``us-west-2``, no AWS creds, no cdk CLI, no network. The
schema-bootstrap Lambda asset is pinned MISSING so the deterministic
inline-fallback path is used (the from_asset bundling path is Docker/pip
dependent).

Cross-stack design under test: the state stack reads the P0 VPC / private-data
subnets / pod_sg / pod_role off the foundation instance; CDK resolves these into
``Export`` / ``Fn::ImportValue`` at synth (NOT ``from_lookup``), which is what
keeps the whole suite cred-free.
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from lib import replay_store_construct as rs_module
from tests.conftest import DUMMY_ACCOUNT, DUMMY_REGION, dummy_env, make_state_stack


@pytest.fixture(autouse=True)
def _force_inline_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the bootstrap-Lambda asset path MISSING so synth is hermetic.

    Same shape as test_replay_store.py: present -> from_asset bundling
    (Docker/local-pip dependent, NON-hermetic); absent -> Code.from_inline
    placeholder (deterministic). Forces the deterministic path.
    """
    monkeypatch.setattr(
        rs_module,
        "_SCHEMA_BOOTSTRAP_ASSET_PATH",
        "/tmp/routeiq-state-stack-test-does-not-exist",
        raising=True,
    )


def _state_template(**flags: Any) -> Template:
    _app, _foundation, state = make_state_stack(**flags)
    return Template.from_stack(state)


# ----------------------------------------------------------- separation of stacks


def test_state_stack_is_separate_from_p0() -> None:
    """The state stack carries the stores; the P0 stack carries NEITHER (rollback rule)."""
    _app, foundation, state = make_state_stack()
    state_t = Template.from_stack(state)
    p0_t = Template.from_stack(foundation)

    # The state stack owns exactly one Aurora cluster + one serverless cache.
    state_t.resource_count_is("AWS::RDS::DBCluster", 1)
    state_t.resource_count_is("AWS::ElastiCache::ServerlessCache", 1)

    # The P0 stack owns NEITHER -- they live in their own blast radius so the
    # state plane rolls back independently (ADR-0028 ~30-min-rollback rule).
    p0_t.resource_count_is("AWS::RDS::DBCluster", 0)
    p0_t.resource_count_is("AWS::ElastiCache::ServerlessCache", 0)


def test_state_stack_owns_its_own_cmk() -> None:
    """The state stack mints its OWN KMS CMK (rotation on) for both stores."""
    template = _state_template()
    keys = template.find_resources("AWS::KMS::Key")
    assert len(keys) >= 1, "state stack must own a CMK"
    # At least one key has rotation enabled.
    rotating = [k for k in keys.values() if k["Properties"].get("EnableKeyRotation") is True]
    assert rotating, "state CMK must have key rotation enabled"


# ------------------------------------------------------- both stores composed


def test_aurora_and_cache_both_present() -> None:
    """The state stack composes BOTH ReplayStoreConstruct and CacheConstruct."""
    template = _state_template()
    # Aurora: cluster + 2 instances (writer + reader).
    template.resource_count_is("AWS::RDS::DBCluster", 1)
    template.resource_count_is("AWS::RDS::DBInstance", 2)
    # Cache: serverless + user group + 2 users (iam + disabled default).
    template.resource_count_is("AWS::ElastiCache::ServerlessCache", 1)
    template.resource_count_is("AWS::ElastiCache::UserGroup", 1)
    template.resource_count_is("AWS::ElastiCache::User", 2)


def test_cache_ingress_6379_from_pod_sg_wired() -> None:
    """attach_dependencies wired the cache SG ingress 6379 from the (imported) pod SG."""
    template = _state_template()
    ingress = template.find_resources(
        "AWS::EC2::SecurityGroupIngress",
        {"Properties": {"FromPort": 6379, "ToPort": 6379, "IpProtocol": "tcp"}},
    )
    assert len(ingress) == 1, f"expected one 6379 ingress (pod SG -> cache), got {len(ingress)}"


def test_aurora_ingress_5432_from_pod_and_self() -> None:
    """The Aurora SG admits 5432 from the imported pod SG AND self (bootstrap Lambda)."""
    template = _state_template()
    ingress = template.find_resources(
        "AWS::EC2::SecurityGroupIngress",
        {"Properties": {"FromPort": 5432, "ToPort": 5432, "IpProtocol": "tcp"}},
    )
    assert len(ingress) == 2, (
        f"expected two 5432 ingress rules (pod SG + self-ref), got {len(ingress)}"
    )


# ----------------------------------------------- cross-stack pod-role grants


def test_pod_role_grants_land_in_state_stack_no_cycle() -> None:
    """rds-db:connect + elasticache:Connect render IN THE STATE stack, attached to P0 role.

    The grants are a SEPARATE iam.Policy owned by the state stack (NOT a mutation
    of the P0 role's default policy), so the state-owned ARNs stay in the state
    stack and no P0 -> State cycle forms. The policy attaches to the imported P0
    role via a Roles ref (a cross-stack Fn::ImportValue of the role name).
    """
    _app, _foundation, state = make_state_stack()
    state_t = Template.from_stack(state)
    policies = state_t.find_resources("AWS::IAM::Policy")
    all_statements = [
        stmt
        for pol in policies.values()
        for stmt in pol["Properties"]["PolicyDocument"]["Statement"]
    ]

    def _has_action(stmt: dict, action: str) -> bool:
        act = stmt.get("Action")
        return act == action or (isinstance(act, list) and action in act)

    rds_stmts = [s for s in all_statements if _has_action(s, "rds-db:connect")]
    ec_stmts = [s for s in all_statements if _has_action(s, "elasticache:Connect")]
    assert rds_stmts, "expected rds-db:connect in the state stack (PodStateGrants policy)"
    assert ec_stmts, "expected elasticache:Connect in the state stack (PodStateGrants policy)"

    # The grant policy attaches to the imported P0 role (Roles ref present).
    grant_pols = [
        pol
        for pol in policies.values()
        if any(
            _has_action(s, "rds-db:connect")
            for s in pol["Properties"]["PolicyDocument"]["Statement"]
        )
    ]
    assert grant_pols, "expected a PodStateGrants policy"
    assert "Roles" in grant_pols[0]["Properties"], grant_pols[0]["Properties"]

    # rds-db:connect targets the RUNTIME user routeiq, NOT the master postgres,
    # and is NOT a bare wildcard.
    rds = rds_stmts[0]
    assert rds["Sid"] == "RdsIamDbConnectRouteIqState", rds
    assert rds["Resource"] != "*", rds
    rds_repr = repr(rds["Resource"])
    assert "/routeiq" in rds_repr, rds_repr
    assert "/postgres" not in rds_repr, rds_repr

    # elasticache:Connect targets two ARNs (cache + IAM user), no wildcard.
    ec = ec_stmts[0]
    assert ec["Sid"] == "ElastiCacheConnectRouteIqState", ec
    assert ec["Resource"] != "*", ec


def test_grants_skipped_when_no_pod_role() -> None:
    """Without a reachable pod role the grants are skipped (separate-pipeline mode)."""
    from lib.routeiq_stack import RouteIqStack
    from lib.routeiq_state_stack import RouteIqStateStack

    app = cdk.App()
    # Build a foundation just to lift its network handles, but wire the state
    # stack with explicit vpc/subnets/pod_sg and NO pod_role.
    foundation = RouteIqStack(app, "RouteIqStack-dev", env=dummy_env(), env_name="dev")
    state = RouteIqStateStack(
        app,
        "RouteIqStateStack-dev",
        env=dummy_env(),
        env_name="dev",
        vpc=foundation.network.vpc,
        private_data_subnets=foundation.network.private_data_subnets,
        pod_sg=foundation.network.pod_sg,
        pod_role=None,
    )
    template = Template.from_stack(state)
    # No IAM policy in the state stack carries rds-db:connect when no role was given.
    policies = template.find_resources("AWS::IAM::Policy")
    for pol in policies.values():
        for stmt in pol["Properties"]["PolicyDocument"]["Statement"]:
            act = stmt.get("Action")
            assert act != "rds-db:connect"
            assert not (isinstance(act, list) and "rds-db:connect" in act)


def test_missing_inputs_raises() -> None:
    """No foundation AND incomplete explicit handles -> ValueError (ASCII message)."""
    from lib.routeiq_state_stack import RouteIqStateStack

    app = cdk.App()
    with pytest.raises(ValueError, match="foundation"):
        RouteIqStateStack(
            app,
            "RouteIqStateStack-bad",
            env=dummy_env(),
            env_name="dev",
            # no foundation, no vpc/subnets/pod_sg
        )


# ------------------------------------------------------------- CfnOutputs


def test_chart_facing_outputs_present() -> None:
    """The four chart-mapping outputs + operator confirmations are emitted."""
    template = _state_template()
    outputs = template.to_json().get("Outputs", {})
    keys = set(outputs.keys())
    for required in (
        "DbClusterEndpoint",
        "DbSecretArn",
        "CacheEndpoint",
        "CachePort",
    ):
        assert required in keys, f"missing CfnOutput {required}; have {sorted(keys)}"
    # Operator confirmations also surfaced.
    for extra in (
        "DbClusterReaderEndpoint",
        "DbRuntimeUser",
        "DbClusterResourceId",
        "CacheIamUserName",
    ):
        assert extra in keys, f"missing CfnOutput {extra}; have {sorted(keys)}"


def test_db_runtime_user_output_value() -> None:
    """DbRuntimeUser is the chart externalPostgresql.username = routeiq."""
    template = _state_template()
    outputs = template.to_json()["Outputs"]
    assert outputs["DbRuntimeUser"]["Value"] == "routeiq", outputs["DbRuntimeUser"]


# --------------------------------------------------------- cred-free synth


def test_cred_free_synth_dummy_env() -> None:
    """The stack synthesises offline against the dummy fake account/region."""
    _app, _foundation, state = make_state_stack()
    assert state.account == DUMMY_ACCOUNT
    assert state.region == DUMMY_REGION
    # Template.from_stack forces the synth; reaching here means no creds were needed.
    Template.from_stack(state)


# ------------------------------------------------------ ACU / env variants


def test_scale_to_zero_min_acu_zero() -> None:
    """min_acu=0 propagates to the Aurora cluster MinCapacity 0 + auto-pause."""
    template = _state_template(min_acu=0)
    clusters = template.find_resources("AWS::RDS::DBCluster")
    scaling = next(iter(clusters.values()))["Properties"]["ServerlessV2ScalingConfiguration"]
    assert scaling["MinCapacity"] == 0, scaling
    assert scaling.get("SecondsUntilAutoPause") == 86400, scaling


def test_prod_env_deletion_protection_and_retain() -> None:
    """A non-dev state stack protects both stores (Aurora protect + cache RETAIN)."""
    template = _state_template(env_name="prod")
    clusters = template.find_resources("AWS::RDS::DBCluster")
    cluster = next(iter(clusters.values()))
    assert cluster["Properties"].get("DeletionProtection") is True
    assert cluster.get("DeletionPolicy") == "Snapshot"
    caches = template.find_resources("AWS::ElastiCache::ServerlessCache")
    cache = next(iter(caches.values()))
    assert cache.get("DeletionPolicy") == "Retain", cache.get("DeletionPolicy")


def test_all_sg_descriptions_ascii() -> None:
    """Every SG in the state stack has an ASCII-only description (EC2 charset rule)."""
    template = _state_template()
    sgs = template.find_resources("AWS::EC2::SecurityGroup")
    for sg in sgs.values():
        desc = sg["Properties"].get("GroupDescription", "")
        assert desc.isascii(), f"SG description must be ASCII; got {desc!r}"
