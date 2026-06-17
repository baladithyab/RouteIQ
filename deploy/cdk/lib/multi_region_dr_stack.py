"""MultiRegionDrStack - the multi-region active-active / DR topology primitives
(RouteIQ-6fd3, the cred-free CDK half).

WHAT THIS IS: a standalone CloudFormation stack that declares the AWS PRIMITIVES
a multi-region active-active / disaster-recovery RouteIQ deployment needs, on top
of the single-region HA primitives already shipped (the P1 ``RouteIqStateStack``
Aurora reader + ElastiCache Serverless cache, ADR-0028/0029). It authors three
independent DR primitives, each individually flag-gated:

  1. **Aurora Global Database** (``AWS::RDS::GlobalCluster``) - a global cluster
     spanning the primary region's Aurora cluster + a secondary-region read
     replica cluster. Sub-second cross-region replication; managed
     unplanned-failover promotes the secondary to a standalone writer.
  2. **ElastiCache Global Datastore** (``AWS::ElastiCache::GlobalReplicationGroup``)
     - cross-region replication of the rate-limiter / governance-counter cache so
     the secondary region serves warm. NOTE the divergence below: Global
     Datastore requires NODE-BASED replication groups; the P1 state cache is
     SERVERLESS (no Global Datastore support), so this primitive is authored
     against a node-based primary the operator supplies, NOT the P1 serverless
     cache. It is OFF by default and documented as a node-based-only path.
  3. **Edge failover** - either a Route53 health-checked failover DNS record set
     (PRIMARY/SECONDARY) OR an AWS Global Accelerator with two regional endpoint
     groups. Exactly one edge mode is chosen via ``edge_mode``; both are
     individually flag-gated and OFF by default.

WHY A SEPARATE, DEFAULT-OFF STACK: a multi-region topology is a deliberate,
operator-driven posture change with a long blast radius (a Global Database
secondary, cross-region replication, a new edge entry point). Per the
~30-minute-rollback rule it is its OWN stack, NOT a child of the P0/P1/P2
stacks, so it deploys / rolls back independently and so the default ``app.py``
synth (which does NOT instantiate it) keeps the P0/P1/P2 template snapshots
byte-stable. Every primitive defaults OFF: an all-defaults synth of THIS stack
emits ZERO DR resources (a near-empty stack), and turning one on adds only that
primitive's resources.

OPERATOR-GATED LIVE STEPS (documented in
``docs/deployment/multi-account-multi-region.md``, NOT expressible here):
  - Creating the secondary-region Aurora read-replica CLUSTER + instance that
    joins the global cluster (a per-region deploy in the secondary region).
  - The actual cross-region failover drill / promotion runbook.
  - Pre-warming / cutover of the secondary region's EKS data plane.
This stack provisions the GLOBAL/edge primitives; the per-region secondary data
plane is a separate secondary-region deploy.

Charset discipline: every Description here stays ASCII / Latin-1 (an em-dash
passes ``cdk synth`` but FAILS the IAM/RDS CREATE API). Mirrors the suite.
"""

from __future__ import annotations

from typing import Literal

from aws_cdk import CfnOutput, Stack, Tags
from aws_cdk import aws_elasticache as elasticache
from aws_cdk import aws_globalaccelerator as globalaccelerator
from aws_cdk import aws_rds as rds
from aws_cdk import aws_route53 as route53
from constructs import Construct

EdgeMode = Literal["none", "route53", "global_accelerator"]
_VALID_EDGE_MODES = ("none", "route53", "global_accelerator")


class MultiRegionDrStack(Stack):
    """Multi-region active-active / DR primitives. Standalone, default-OFF.

    Each of the three primitives (Aurora Global Database, ElastiCache Global
    Datastore, edge failover) is independently flag-gated. An all-defaults synth
    emits zero DR resources. Synth is credential-free.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        # --- Aurora Global Database (AWS::RDS::GlobalCluster) ------------------
        enable_aurora_global: bool = False,
        # The PRIMARY-region Aurora cluster identifier to attach as the global
        # cluster's source writer. Required when enable_aurora_global is True.
        # This is the P1 RouteIqStateStack Aurora cluster id (cross-stack input).
        source_db_cluster_identifier: str | None = None,
        aurora_engine: str = "aurora-postgresql",
        aurora_engine_version: str | None = None,
        # --- ElastiCache Global Datastore (node-based ONLY) -------------------
        enable_cache_global: bool = False,
        # The PRIMARY-region NODE-BASED replication group id (NOT the P1 serverless
        # cache - Global Datastore is unsupported on serverless). Required when
        # enable_cache_global is True.
        primary_replication_group_id: str | None = None,
        # --- Edge failover ----------------------------------------------------
        edge_mode: EdgeMode = "none",
        # Route53 failover inputs (edge_mode="route53").
        hosted_zone_id: str | None = None,
        record_name: str | None = None,
        primary_endpoint: str | None = None,
        secondary_endpoint: str | None = None,
        primary_health_check_fqdn: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if edge_mode not in _VALID_EDGE_MODES:
            raise ValueError(
                f"invalid edge_mode {edge_mode!r}: expected one of {_VALID_EDGE_MODES}."
            )

        self.env_name = env_name
        Tags.of(self).add("routeiq:env", env_name)
        Tags.of(self).add("routeiq:stack", "multi-region-dr")

        if enable_aurora_global:
            self._add_aurora_global(
                source_db_cluster_identifier=source_db_cluster_identifier,
                engine=aurora_engine,
                engine_version=aurora_engine_version,
            )

        if enable_cache_global:
            self._add_cache_global(
                primary_replication_group_id=primary_replication_group_id,
            )

        if edge_mode == "route53":
            self._add_route53_failover(
                hosted_zone_id=hosted_zone_id,
                record_name=record_name,
                primary_endpoint=primary_endpoint,
                secondary_endpoint=secondary_endpoint,
                primary_health_check_fqdn=primary_health_check_fqdn,
            )
        elif edge_mode == "global_accelerator":
            self._add_global_accelerator()

    # ------------------------------------------------------------------ Aurora

    def _add_aurora_global(
        self,
        *,
        source_db_cluster_identifier: str | None,
        engine: str,
        engine_version: str | None,
    ) -> None:
        """Aurora Global Database global cluster (AWS::RDS::GlobalCluster).

        Attaches the primary-region Aurora cluster as the global source writer.
        The secondary-region read-replica cluster joining this global cluster is
        an OPERATOR per-region deploy (documented), not authored here.
        """
        if not source_db_cluster_identifier:
            raise ValueError(
                "source_db_cluster_identifier is required when enable_aurora_global "
                "is True: the global cluster attaches the primary-region Aurora "
                "cluster as its source writer (the P1 RouteIqStateStack cluster id)."
            )

        self.global_cluster = rds.CfnGlobalCluster(
            self,
            "AuroraGlobalCluster",
            global_cluster_identifier=f"routeiq-{self.env_name}-global",
            source_db_cluster_identifier=source_db_cluster_identifier,
            # Engine/EngineVersion are mutually exclusive with
            # SourceDBClusterIdentifier in the live API (the engine is inherited
            # from the source). We pass them ONLY when no source is given; with a
            # source they stay unset. Here a source is always given, so both unset.
            engine=None,
            engine_version=None,
            storage_encrypted=True,
            deletion_protection=True,
        )
        # Keep the engine knobs referenced for the (documented) no-source variant
        # without emitting them when a source is attached. (no-op binding)
        _ = (engine, engine_version)

        CfnOutput(
            self,
            "GlobalClusterIdentifier",
            value=self.global_cluster.ref,
            description=(
                "Aurora Global Database global cluster identifier. The secondary-"
                "region read-replica cluster joins this id (operator per-region "
                "deploy)."
            ),
        )

    # ------------------------------------------------------------------- Cache

    def _add_cache_global(self, *, primary_replication_group_id: str | None) -> None:
        """ElastiCache Global Datastore (AWS::ElastiCache::GlobalReplicationGroup).

        Global Datastore is NODE-BASED only -- the P1 state cache is SERVERLESS
        and CANNOT join one, so this attaches a node-based primary replication
        group the operator supplies. Documented divergence.
        """
        if not primary_replication_group_id:
            raise ValueError(
                "primary_replication_group_id is required when enable_cache_global "
                "is True: ElastiCache Global Datastore is node-based only (the P1 "
                "serverless cache cannot join one). Supply a node-based primary "
                "replication group id."
            )

        self.global_replication_group = elasticache.CfnGlobalReplicationGroup(
            self,
            "CacheGlobalReplicationGroup",
            global_replication_group_id_suffix=f"routeiq-{self.env_name}-global",
            members=[
                elasticache.CfnGlobalReplicationGroup.GlobalReplicationGroupMemberProperty(
                    replication_group_id=primary_replication_group_id,
                    replication_group_region=self.region,
                    role="PRIMARY",
                )
            ],
        )

        CfnOutput(
            self,
            "GlobalReplicationGroupId",
            value=self.global_replication_group.ref,
            description=(
                "ElastiCache Global Datastore id. The secondary-region SECONDARY "
                "member replication group joins this id (operator per-region deploy)."
            ),
        )

    # -------------------------------------------------------------------- Edge

    def _add_route53_failover(
        self,
        *,
        hosted_zone_id: str | None,
        record_name: str | None,
        primary_endpoint: str | None,
        secondary_endpoint: str | None,
        primary_health_check_fqdn: str | None,
    ) -> None:
        """Route53 health-checked PRIMARY/SECONDARY failover record set.

        A health check on the primary endpoint drives a failover routing policy:
        the PRIMARY record serves while healthy, the SECONDARY takes over when the
        health check fails. The primary record is the only one with a health-check
        association (the SECONDARY is the fallback).
        """
        missing = [
            name
            for name, val in (
                ("hosted_zone_id", hosted_zone_id),
                ("record_name", record_name),
                ("primary_endpoint", primary_endpoint),
                ("secondary_endpoint", secondary_endpoint),
                ("primary_health_check_fqdn", primary_health_check_fqdn),
            )
            if not val
        ]
        if missing:
            raise ValueError("edge_mode='route53' requires: " + ", ".join(missing) + ".")

        health_check = route53.CfnHealthCheck(
            self,
            "PrimaryHealthCheck",
            health_check_config=route53.CfnHealthCheck.HealthCheckConfigProperty(
                type="HTTPS",
                fully_qualified_domain_name=primary_health_check_fqdn,
                port=443,
                resource_path="/_health/ready",
                request_interval=30,
                failure_threshold=3,
            ),
        )

        # PRIMARY failover record (health-checked).
        self.primary_record = route53.CfnRecordSet(
            self,
            "PrimaryFailoverRecord",
            hosted_zone_id=hosted_zone_id,
            name=record_name,
            type="CNAME",
            ttl="60",
            set_identifier=f"routeiq-{self.env_name}-primary",
            failover="PRIMARY",
            health_check_id=health_check.attr_health_check_id,
            resource_records=[primary_endpoint],
        )
        # SECONDARY failover record (fallback, no health check).
        self.secondary_record = route53.CfnRecordSet(
            self,
            "SecondaryFailoverRecord",
            hosted_zone_id=hosted_zone_id,
            name=record_name,
            type="CNAME",
            ttl="60",
            set_identifier=f"routeiq-{self.env_name}-secondary",
            failover="SECONDARY",
            resource_records=[secondary_endpoint],
        )

        CfnOutput(
            self,
            "FailoverRecordName",
            value=record_name,  # type: ignore[arg-type]
            description=(
                "Route53 failover record name (PRIMARY health-checked, SECONDARY "
                "fallback). Point clients at this name for cross-region failover."
            ),
        )

    def _add_global_accelerator(self) -> None:
        """AWS Global Accelerator with one listener (the edge entry point).

        The accelerator + a TCP/443 listener are the static-anycast-IP edge entry
        point; the regional endpoint GROUPS (one per region, weighted for
        active-active or PRIMARY/SECONDARY) are attached per-region by the operator
        (they reference per-region ALB/NLB ARNs that do not exist at this synth).
        """
        self.accelerator = globalaccelerator.CfnAccelerator(
            self,
            "DrAccelerator",
            name=f"routeiq-{self.env_name}-dr",
            enabled=True,
            ip_address_type="IPV4",
        )
        self.listener = globalaccelerator.CfnListener(
            self,
            "DrListener",
            accelerator_arn=self.accelerator.attr_accelerator_arn,
            protocol="TCP",
            port_ranges=[
                globalaccelerator.CfnListener.PortRangeProperty(from_port=443, to_port=443)
            ],
            client_affinity="SOURCE_IP",
        )

        CfnOutput(
            self,
            "AcceleratorArn",
            value=self.accelerator.attr_accelerator_arn,
            description=(
                "Global Accelerator ARN. Operator attaches per-region endpoint "
                "groups (weighted active-active or PRIMARY/SECONDARY) referencing "
                "per-region ALB/NLB ARNs."
            ),
        )
        CfnOutput(
            self,
            "AcceleratorDnsName",
            value=self.accelerator.attr_dns_name,
            description=(
                "Global Accelerator static anycast DNS name. Point clients here for "
                "cross-region active-active routing."
            ),
        )
