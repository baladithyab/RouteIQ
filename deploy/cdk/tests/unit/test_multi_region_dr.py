"""Unit tests for RouteIQ-6fd3 - multi-region active-active / DR primitives.

Credential-free (synth offline against the dummy env ``123456789012`` /
``us-west-2``). The ``MultiRegionDrStack`` is a STANDALONE stack; it is NOT in
the default ``app.py`` synth, so the P0/P1/P2 snapshots stay byte-stable.

Asserts:
  1. DEFAULT OFF: an all-defaults synth emits ZERO DR resources (a near-empty
     stack -- the byte-stable-off guarantee).
  2. Aurora Global Database (AWS::RDS::GlobalCluster) only when enabled; attaches
     the source cluster; storage-encrypted + deletion-protected; requires a
     source cluster id.
  3. ElastiCache Global Datastore (AWS::ElastiCache::GlobalReplicationGroup) only
     when enabled; node-based primary member; requires a primary rg id.
  4. Edge: route53 mode emits a health-checked PRIMARY + a SECONDARY failover
     record (and validates inputs); global_accelerator mode emits an Accelerator
     + a TCP/443 Listener; none emits neither.
  5. Each primitive is INDEPENDENT (enabling one does not pull in another).
  6. cdk-nag clean across the maximal flag fold.

Mirrors ``test_bedrock_capacity.py`` conventions (standalone synth + the
fresh-app + ``AwsSolutionsChecks`` aspect).
"""

from __future__ import annotations

import aws_cdk as cdk
import pytest
from aws_cdk import Aspects
from aws_cdk.assertions import Annotations, Match, Template
from cdk_nag import AwsSolutionsChecks

from lib.multi_region_dr_stack import MultiRegionDrStack

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"

# Full route53 edge inputs reused across tests.
_R53 = dict(
    edge_mode="route53",
    hosted_zone_id="Z0123456789ABCDEFGHIJ",
    record_name="gw.routeiq.example.com",
    primary_endpoint="primary.us-west-2.routeiq.example.com",
    secondary_endpoint="secondary.us-east-1.routeiq.example.com",
    primary_health_check_fqdn="primary.us-west-2.routeiq.example.com",
)


def _dummy_env() -> cdk.Environment:
    return cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION)


def _stack(*, app: cdk.App | None = None, **kwargs):
    if app is None:
        app = cdk.App()
    kwargs.setdefault("env_name", "dev")
    return MultiRegionDrStack(app, "DrStack-dev", env=_dummy_env(), **kwargs)


def _template(**kwargs) -> Template:
    return Template.from_stack(_stack(**kwargs))


def _resource_types(template: Template) -> set[str]:
    return {r["Type"] for r in template.to_json().get("Resources", {}).values()}


# ---------------------------------------------------------------- default OFF


def test_default_off_emits_zero_dr_resources() -> None:
    """The byte-stable-off guarantee: an all-defaults synth has NO DR resources."""
    template = _template()
    types = _resource_types(template)
    for cfn_type in (
        "AWS::RDS::GlobalCluster",
        "AWS::ElastiCache::GlobalReplicationGroup",
        "AWS::Route53::HealthCheck",
        "AWS::Route53::RecordSet",
        "AWS::GlobalAccelerator::Accelerator",
        "AWS::GlobalAccelerator::Listener",
    ):
        assert cfn_type not in types, f"{cfn_type} leaked into the default-off synth"
    # Near-empty stack (no DR resources at all).
    assert types == set() or types <= set(), types


def test_invalid_edge_mode_raises() -> None:
    app = cdk.App()
    with pytest.raises(ValueError, match="invalid edge_mode"):
        MultiRegionDrStack(
            app,
            "Bad",
            env=_dummy_env(),
            env_name="dev",
            edge_mode="dns",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------- Aurora global


def test_aurora_global_only_when_enabled() -> None:
    template = _template(
        enable_aurora_global=True, source_db_cluster_identifier="routeiq-dev-aurora"
    )
    template.resource_count_is("AWS::RDS::GlobalCluster", 1)
    template.has_resource_properties(
        "AWS::RDS::GlobalCluster",
        {
            "GlobalClusterIdentifier": "routeiq-dev-global",
            "SourceDBClusterIdentifier": "routeiq-dev-aurora",
            "StorageEncrypted": True,
            "DeletionProtection": True,
        },
    )


def test_aurora_global_requires_source_cluster() -> None:
    app = cdk.App()
    with pytest.raises(ValueError, match="source_db_cluster_identifier is required"):
        MultiRegionDrStack(
            app,
            "Bad",
            env=_dummy_env(),
            env_name="dev",
            enable_aurora_global=True,
        )


def test_aurora_global_independent_of_cache_and_edge() -> None:
    """Enabling only Aurora does NOT pull in cache/edge resources."""
    template = _template(enable_aurora_global=True, source_db_cluster_identifier="c")
    types = _resource_types(template)
    assert "AWS::RDS::GlobalCluster" in types
    assert "AWS::ElastiCache::GlobalReplicationGroup" not in types
    assert "AWS::GlobalAccelerator::Accelerator" not in types
    assert "AWS::Route53::RecordSet" not in types


# ---------------------------------------------------------------- cache global


def test_cache_global_only_when_enabled() -> None:
    template = _template(enable_cache_global=True, primary_replication_group_id="routeiq-dev-rg")
    template.resource_count_is("AWS::ElastiCache::GlobalReplicationGroup", 1)
    grg = next(iter(template.find_resources("AWS::ElastiCache::GlobalReplicationGroup").values()))[
        "Properties"
    ]
    members = grg["Members"]
    assert len(members) == 1, members
    assert members[0]["ReplicationGroupId"] == "routeiq-dev-rg"
    assert members[0]["Role"] == "PRIMARY"
    assert members[0]["ReplicationGroupRegion"] == DUMMY_REGION


def test_cache_global_requires_primary_rg() -> None:
    app = cdk.App()
    with pytest.raises(ValueError, match="primary_replication_group_id is required"):
        MultiRegionDrStack(
            app,
            "Bad",
            env=_dummy_env(),
            env_name="dev",
            enable_cache_global=True,
        )


# ---------------------------------------------------------------- edge: route53


def test_route53_failover_records() -> None:
    template = _template(**_R53)
    template.resource_count_is("AWS::Route53::HealthCheck", 1)
    template.resource_count_is("AWS::Route53::RecordSet", 2)

    records = template.find_resources("AWS::Route53::RecordSet")
    by_failover = {r["Properties"]["Failover"]: r["Properties"] for r in records.values()}
    assert set(by_failover) == {"PRIMARY", "SECONDARY"}, by_failover

    primary = by_failover["PRIMARY"]
    secondary = by_failover["SECONDARY"]
    # PRIMARY is health-checked; SECONDARY is the fallback (no health check).
    assert "HealthCheckId" in primary, primary
    assert "HealthCheckId" not in secondary, secondary
    assert primary["ResourceRecords"] == [_R53["primary_endpoint"]]
    assert secondary["ResourceRecords"] == [_R53["secondary_endpoint"]]


def test_route53_health_check_targets_readiness() -> None:
    template = _template(**_R53)
    template.has_resource_properties(
        "AWS::Route53::HealthCheck",
        {
            "HealthCheckConfig": Match.object_like(
                {
                    "Type": "HTTPS",
                    "ResourcePath": "/_health/ready",
                    "Port": 443,
                }
            )
        },
    )


def test_route53_requires_all_inputs() -> None:
    app = cdk.App()
    with pytest.raises(ValueError, match="edge_mode='route53' requires"):
        MultiRegionDrStack(
            app,
            "Bad",
            env=_dummy_env(),
            env_name="dev",
            edge_mode="route53",
            hosted_zone_id="Z123",
            # missing record_name / endpoints / health-check fqdn
        )


# ---------------------------------------------------------- edge: accelerator


def test_global_accelerator_edge() -> None:
    template = _template(edge_mode="global_accelerator")
    template.resource_count_is("AWS::GlobalAccelerator::Accelerator", 1)
    template.resource_count_is("AWS::GlobalAccelerator::Listener", 1)
    template.has_resource_properties(
        "AWS::GlobalAccelerator::Accelerator",
        {"Name": "routeiq-dev-dr", "Enabled": True},
    )
    template.has_resource_properties(
        "AWS::GlobalAccelerator::Listener",
        {
            "Protocol": "TCP",
            "PortRanges": [{"FromPort": 443, "ToPort": 443}],
        },
    )
    # No route53 records in GA mode.
    assert "AWS::Route53::RecordSet" not in _resource_types(template)


def test_edge_none_emits_no_edge_resources() -> None:
    template = _template(edge_mode="none")
    types = _resource_types(template)
    assert "AWS::Route53::RecordSet" not in types
    assert "AWS::GlobalAccelerator::Accelerator" not in types


# ---------------------------------------------------------------- cdk-nag


def test_dr_cdk_nag_clean_maximal() -> None:
    """No AwsSolutions-* errors across the maximal flag fold.

    Aurora Global Database + Cache Global Datastore + Global Accelerator together;
    none of these L1s carry an IAM identity AwsSolutionsChecks flags.
    """
    app = cdk.App()
    stack = _stack(
        app=app,
        enable_aurora_global=True,
        source_db_cluster_identifier="routeiq-dev-aurora",
        enable_cache_global=True,
        primary_replication_group_id="routeiq-dev-rg",
        edge_mode="global_accelerator",
    )
    Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        rendered = "\n".join(f"- {e.id}: {str(e.entry.data)[:200]}" for e in errors)
        raise AssertionError(f"{len(errors)} unsuppressed AwsSolutions-* error(s):\n{rendered}")
