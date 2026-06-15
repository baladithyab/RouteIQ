"""RouteIqStateStack - the SEPARATE P1 state-plane composition root (ADR-0028/0029).

The RouteIQ control-plane state (Aurora Postgres Serverless v2 + ElastiCache
Serverless Valkey) is deliberately its OWN stack, NOT folded into the P0
``RouteIqStack``. Rationale: the ADR-0028 ~30-minute-rollback rule. The state
plane must roll back INDEPENDENTLY of the EKS / network / ECR foundation: an
Aurora minor that gets retired at deploy (the
``cdk-rds-aurora-engine-version-retired-at-deploy`` gotcha), a schema-bootstrap
regression, or a cache mis-provision must be reversible by rolling back ONLY
this stack, leaving the P0 cluster + the running gateway pods untouched. One
stack = one blast radius = one rollback unit.

This stack consumes the P0 foundation by REFERENCE (cross-stack, same CDK app):

  * the P0 VPC + the ``private-data`` (PRIVATE_ISOLATED) subnet tier,
  * the P0 ``pod_sg`` (the EKS Pod Identity gateway pod ENI SG) as the ingress
    peer for both data-tier SGs,
  * the P0 ``pod_role`` (the single Pod-Identity-bound gateway pod role) as the
    grantee for the two P1 IAM additions (``rds-db:connect`` + ``elasticache:Connect``).

CRED-FREE cross-stack wiring (NO ``from_lookup``): the state stack takes the P0
``RouteIqStack`` instance directly (``foundation=``) and reads its construct
handles. Same-app cross-stack references are resolved by CDK at SYNTH time into
auto-generated ``Export`` / ``Fn::ImportValue`` pairs -- no environment lookup,
no AWS API call, no creds. ``from_lookup`` (which DOES hit AWS) is deliberately
avoided so ``Template.from_stack`` and the cdk-nag gate run fully offline against
the dummy env account ``123456789012`` / ``us-west-2`` (build-outline cred-free
gate). Every credential / secret value is a CDK token or a CfnParameter, NEVER a
literal.

Resources owned here:

  * a state-stack KMS CMK (rotation on) -- the customer key for BOTH stores, so a
    state-stack rollback drops its own key (the P0 stack mints no shared CMK),
  * ``ReplayStoreConstruct`` (Aurora) + ``CacheConstruct`` (ElastiCache),
  * the cross-stack grants on the P0 pod role,
  * CfnOutputs the deploy step maps to chart ``--set`` flags
    (``DbClusterEndpoint``, ``DbSecretArn``, ``CacheEndpoint``, ``CachePort``,
    plus the runtime-user / IAM-user / cluster-resource-id operators need),
  * ``_suppress_nag()`` LAST (so every resource exists before suppression by path).

POD-ROLE GRANTS ARE CROSS-STACK (the stack-composition note): the P0 ``pod_role``
lives in the P0 stack; the P1 grants (``rds-db:connect`` on the Aurora dbuser ARN,
``elasticache:Connect`` on the cache + IAM-user ARNs) attach to it from HERE via
``role.add_to_principal_policy`` (which mutates the role's default policy in its
OWN stack, with the resource ARNs flowing back as cross-stack imports). This is
clean and synth-stable because the grant statements are ARN-scoped (no wildcard,
no new nag suppression needed on the P0 role). When the operator does NOT pass
``foundation`` (e.g. the two stacks are deployed from separate apps / pipelines),
the grants are SKIPPED and must be applied out-of-band -- see the chart README
stack-composition note.

ENGINE-VERSION GOTCHA (load-bearing): the Aurora minor (``VER_16_13``, pinned in
``ReplayStoreConstruct``, overridable via the ``routeiq:state_pg_version``
context key) MUST be verified LIVE in us-west-2 with ``aws rds
describe-db-engine-versions --engine aurora-postgresql --region us-west-2``
before deploy. AWS RETIRES old Aurora Postgres minors; ``cdk synth`` / cdk-nag /
the unit tests CANNOT catch a retired minor -- only a real ``CreateDBCluster``
does (mulch cdk-rds-aurora-engine-version-retired-at-deploy).

Every IAM / SG / resource description is ASCII / Latin-1 only: an em-dash
(U+2014) or arrow passes ``cdk synth`` but FAILS the EC2 / IAM CREATE API
(mulch ec2-sg-description-charset).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from aws_cdk import Aws, CfnOutput, RemovalPolicy, Stack, Tags
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from constructs import Construct

from .cache_construct import CacheConstruct
from .replay_store_construct import ReplayStoreConstruct
from .state_nag_suppressions import apply_state_nag_suppressions

if TYPE_CHECKING:  # pragma: no cover - forward ref only
    from .routeiq_stack import RouteIqStack


class RouteIqStateStack(Stack):
    """The SEPARATE P1 state stack: Aurora + ElastiCache + cross-stack pod-role grants."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        # The P0 stack, passed by REFERENCE for cred-free cross-stack wiring. When
        # given, the state stack reads foundation.network.{vpc,private_data_subnets,
        # pod_sg} and foundation.pod_role directly. Optional so the construct-only
        # paths (and a separate-pipeline deploy) stay constructible.
        foundation: RouteIqStack | None = None,
        # Explicit cross-stack inputs, used when foundation is None (e.g. the two
        # stacks are deployed from separate apps). All four must be supplied
        # together in that mode; the helper validates.
        vpc: ec2.IVpc | None = None,
        private_data_subnets: Sequence[ec2.ISubnet] | None = None,
        pod_sg: ec2.ISecurityGroup | None = None,
        pod_role: iam.IRole | None = None,
        # The P0 shared interface-endpoint SG (network.vpce_sg). Used to admit the
        # schema-bootstrap Lambda's SG to the Secrets Manager endpoint on 443
        # (RouteIQ-8374). OPTIONAL even in separate-pipeline mode: when absent the
        # 443 ingress rule is skipped and applied out-of-band (same posture as the
        # pod-role grants). On the foundation path it is read from foundation.network.
        vpce_sg: ec2.ISecurityGroup | None = None,
        # Aurora ACU window overrides (default 0.5 warm floor; max 2.0 dev / 8.0
        # otherwise inside the construct). Pass min_acu=0 (with the construct's
        # 24h auto-pause) for the cost-sensitive dev scale-to-zero path.
        min_acu: float | None = None,
        max_acu: float | None = None,
        # ElastiCache engine version. major is derived inside the construct.
        cache_engine_version: str = "8.0",
        cost_center: str | None = None,
        team: str | None = None,
        tenant: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # Resolve the cross-stack inputs. foundation (by-reference) wins; otherwise
        # the explicit handles must all be present. ASCII-only error text.
        (
            resolved_vpc,
            resolved_data_subnets,
            resolved_pod_sg,
            resolved_pod_role,
            resolved_vpce_sg,
        ) = self._resolve_foundation(
            foundation=foundation,
            vpc=vpc,
            private_data_subnets=private_data_subnets,
            pod_sg=pod_sg,
            pod_role=pod_role,
            vpce_sg=vpce_sg,
        )
        self._pod_role = resolved_pod_role
        self._vpce_sg = resolved_vpce_sg

        # Stack tags. Cost tags emitted ONLY when supplied so the default synth
        # stays byte-stable for the snapshot test (mirrors the P0 RouteIqStack).
        Tags.of(self).add("routeiq:env", env_name)
        Tags.of(self).add("routeiq:plane", "state")
        for _key, _value in (
            ("routeiq:cost-center", cost_center),
            ("routeiq:team", team),
            ("routeiq:tenant", tenant),
        ):
            if _value:
                Tags.of(self).add(_key, _value)

        # -- 1. State-stack KMS CMK (owned here) -------------------------------
        # The customer CMK for BOTH stores. Owned by THIS stack so a state-stack
        # rollback drops its own key (the P0 stack mints no shared CMK). Rotation
        # on. RETAIN for non-dev (a dropped key would orphan encrypted snapshots);
        # DESTROY for dev so a rolled-back dev deploy unwinds cleanly. ASCII desc.
        self.kms_key = kms.Key(
            self,
            "StateCmk",
            description=(
                f"RouteIQ {env_name} state-plane CMK (Aurora at-rest + ElastiCache at-rest)"
            ),
            enable_key_rotation=True,
            removal_policy=(RemovalPolicy.DESTROY if env_name == "dev" else RemovalPolicy.RETAIN),
        )

        # -- 2. Aurora Postgres Serverless v2 (ADR-0028) -----------------------
        # The cluster sits in the P0 private-data (PRIVATE_ISOLATED) tier; its SG
        # admits 5432 from the P0 pod_sg. The bootstrap Lambda lives in the
        # private-app tier (for SecretsManager/KMS endpoint reachability) and
        # reaches 5432 via the SG self-ref. Engine VER_16_13 is pinned in the
        # construct (verify live before deploy).
        self.replay_store = ReplayStoreConstruct(
            self,
            "ReplayStoreConstruct",
            env_name=env_name,
            kms_key=self.kms_key,
            vpc=resolved_vpc,
            app_sg=resolved_pod_sg,
            private_data_subnets=resolved_data_subnets,
            min_acu=min_acu,
            max_acu=max_acu,
        )

        # -- 2b. Cross-stack: bootstrap Lambda -> Secrets Manager endpoint (RouteIQ-8374)
        # The schema-bootstrap Lambda runs in replay_store_sg in the private-app tier
        # and must reach the P0 Secrets Manager interface endpoint (private_dns_enabled,
        # so the SM DNS resolves to the endpoint ENI) on 443 to read the master secret.
        # The P0 vpce_sg admits 443 from pod_sg only; the Lambda is NOT in pod_sg, so
        # without this rule GetSecretValue hangs and the custom resource times out at
        # deploy.
        #
        # CYCLE AVOIDANCE: we do NOT call resolved_vpce_sg.add_ingress_rule(...). That
        # L2 helper attaches the rendered AWS::EC2::SecurityGroupIngress to the SG's
        # OWN stack (P0, since vpce_sg is a P0 construct), and that ingress references
        # replay_store_sg.GroupId (a State-stack SG) -- a P0 -> State edge, while State
        # already imports P0 (State -> P0). That closes a DependencyCycle at synth
        # (exactly the pod-role-grant hazard). Instead we instantiate a STANDALONE
        # ec2.CfnSecurityGroupIngress scoped to THIS (State) stack: the ingress resource
        # lives in the State template, group_id is the imported P0 vpce_sg id (a
        # State -> P0 import, which already exists), and source_security_group_id is the
        # in-stack replay_store_sg. Only cross-stack edge stays State -> P0; no cycle.
        # ASCII-only description. Skipped (apply out-of-band) when vpce_sg is not
        # reachable (separate-pipeline mode), same posture as the pod-role grants.
        #
        # POSTURE DECISION (RouteIQ-9ad6): the P0 vpce_sg is a SHARED interface-endpoint
        # SG fronting all FIVE endpoints (ECR, ECR_DOCKER, CloudWatch-Logs,
        # SecretsManager, Bedrock-Runtime -- see network_construct.py). Admitting
        # replay_store_sg on 443 to vpce_sg therefore lets the bootstrap Lambda reach
        # ALL FIVE endpoints, not just Secrets Manager (which is all it actually needs).
        # A strict least-privilege split would carve a dedicated secrets-only endpoint
        # SG and admit the Lambda only to that. We deliberately ACCEPT the shared-vpce
        # posture instead, with this justification:
        #   - the bootstrap Lambda is a transient, deploy-time-only break-glass leader
        #     (runs on CFN Create/Update, then idles), not a long-lived data-plane role;
        #   - the over-reach is endpoint REACHABILITY only -- it confers no IAM rights:
        #     the Lambda's execution-role policy is still scoped to GetSecretValue on the
        #     one master secret ARN (+ the secret's KMS key), so it cannot actually use
        #     ECR/Logs/Bedrock even though their ENIs are network-reachable;
        #   - splitting a per-endpoint SG (a second InterfaceVpcEndpoint set + SG + the
        #     same cross-stack-cycle dance) is materially heavier P0 surface for a
        #     reachability-only delta on a deploy-time actor.
        # If the bootstrap is ever promoted to a standing/long-lived role, revisit and
        # carve the dedicated secrets-only endpoint SG.
        self.vpce_bootstrap_ingress: ec2.CfnSecurityGroupIngress | None = None
        if resolved_vpce_sg is not None:
            self.vpce_bootstrap_ingress = ec2.CfnSecurityGroupIngress(
                self,
                "VpceBootstrapSecretsIngress443",
                ip_protocol="tcp",
                from_port=443,
                to_port=443,
                group_id=resolved_vpce_sg.security_group_id,
                source_security_group_id=self.replay_store.replay_store_sg.security_group_id,
                description="HTTPS from schema-bootstrap Lambda to Secrets Manager endpoint",
            )

        # -- 3. ElastiCache Serverless Valkey (ADR-0029) -----------------------
        # Serverless cache in the SAME private-data subnets; always-on TLS; IAM-auth
        # user group (user_id == user_name); default user disabled. The peer-SG
        # ingress (6379 from pod_sg) is wired via attach_dependencies so the
        # construct stays constructible without the peer for an isolated snapshot.
        self.cache = CacheConstruct(
            self,
            "CacheConstruct",
            env_name=env_name,
            vpc=resolved_vpc,
            private_data_subnets=resolved_data_subnets,
            kms_key=self.kms_key,
            engine_version=cache_engine_version,
        )
        self.cache.attach_dependencies(app_sg=resolved_pod_sg)

        # -- 4. Cross-stack pod-role grants (the stack-composition seam) -------
        # The two P1 IAM additions the P0 forward-compat comment named
        # (routeiq_stack.py: "P1 (ADR-0028): rds-db:connect" / "P1 (ADR-0029):
        # elasticache:Connect"). Both ARN-scoped (no wildcard), so no new nag
        # suppression on the P0 role. Applied ONLY when the pod role is reachable
        # (foundation given, or pod_role passed). When NOT reachable the operator
        # must apply them out-of-band (chart README stack-composition note).
        #
        # CRITICAL CROSS-STACK CYCLE AVOIDANCE: we do NOT call the construct
        # grant_* helpers here. Those use role.add_to_principal_policy, which
        # MUTATES the imported P0 role's DEFAULT policy -- and that default policy
        # is synthesised IN THE P0 STACK. Pushing this state stack's resource ARNs
        # (the Aurora cluster resource id, the cache ARN, the IAM-user ARN) into a
        # P0-stack resource creates a P0 -> State stack dependency, while State
        # already depends on P0 (it imports the P0 VPC subnets). That closes a
        # cyclic stack reference (DependencyCycle at synth). Instead we own a
        # SEPARATE iam.Policy IN THIS (State) stack and attach it to the imported
        # role: the policy resource + the state-owned ARNs stay in the State stack,
        # the only cross-stack edge is State -> P0 (the role import), which already
        # exists. Equivalent runtime grant, no cycle.
        self.pod_grant_policy: iam.Policy | None = None
        if self._pod_role is not None:
            self.pod_grant_policy = iam.Policy(
                self,
                "PodStateGrants",
                statements=[
                    # rds-db:connect on the RUNTIME user routeiq (NOT the master
                    # postgres): the gateway pod IAM-auths as routeiq. ARN-scoped
                    # to a single dbuser (no wildcard).
                    iam.PolicyStatement(
                        sid="RdsIamDbConnectRouteIqState",
                        effect=iam.Effect.ALLOW,
                        actions=["rds-db:connect"],
                        resources=[
                            (
                                f"arn:{Aws.PARTITION}:rds-db:{Aws.REGION}:{Aws.ACCOUNT_ID}"
                                f":dbuser:{self.replay_store.cluster_resource_identifier}"
                                f"/routeiq"
                            )
                        ],
                    ),
                    # elasticache:Connect on BOTH the cache ARN and the IAM-user
                    # ARN (SigV4 IAM auth needs both to resolve the signing
                    # principal onto the cache user). Both fully-specified ARNs.
                    iam.PolicyStatement(
                        sid="ElastiCacheConnectRouteIqState",
                        effect=iam.Effect.ALLOW,
                        actions=["elasticache:Connect"],
                        resources=[self.cache.cache_arn, self.cache.iam_user_arn],
                    ),
                ],
            )
            self.pod_grant_policy.attach_to_role(self._pod_role)

        # -- 5. CfnOutputs the deploy step maps to chart --set flags -----------
        self._add_outputs()

        # -- 6. cdk-nag suppressions (LAST, by path) ---------------------------
        self._suppress_nag()

    # --------------------------------------------------------------- resolution
    def _resolve_foundation(
        self,
        *,
        foundation: RouteIqStack | None,
        vpc: ec2.IVpc | None,
        private_data_subnets: Sequence[ec2.ISubnet] | None,
        pod_sg: ec2.ISecurityGroup | None,
        pod_role: iam.IRole | None,
        vpce_sg: ec2.ISecurityGroup | None,
    ) -> tuple[
        ec2.IVpc,
        Sequence[ec2.ISubnet],
        ec2.ISecurityGroup,
        iam.IRole | None,
        ec2.ISecurityGroup | None,
    ]:
        """Resolve the cross-stack network/role inputs from foundation or explicit args.

        foundation (the P0 RouteIqStack, by reference) is the primary path: it
        carries network.vpc / network.private_data_subnets / network.pod_sg /
        network.vpce_sg and pod_role. Reading these construct handles in the SAME
        CDK app makes CDK emit Export / Fn::ImportValue at synth time (cred-free,
        no from_lookup).

        When foundation is None the explicit handles take over (the
        separate-pipeline deploy mode). vpc + private_data_subnets + pod_sg are
        MANDATORY in that mode; pod_role AND vpce_sg are optional (the grants /
        the Secrets Manager 443 ingress are skipped without them, applied
        out-of-band). ASCII-only error text.
        """
        if foundation is not None:
            network = foundation.network
            return (
                network.vpc,
                list(network.private_data_subnets),
                network.pod_sg,
                getattr(foundation, "pod_role", None) or pod_role,
                getattr(network, "vpce_sg", None) or vpce_sg,
            )
        if vpc is None or private_data_subnets is None or pod_sg is None:
            raise ValueError(
                "RouteIqStateStack needs either foundation=<RouteIqStack> "
                "(cross-stack by reference) OR all of vpc + private_data_subnets "
                "+ pod_sg (the separate-pipeline mode). pod_role and vpce_sg are "
                "optional; without pod_role the rds-db:connect / elasticache:Connect "
                "grants are skipped, and without vpce_sg the bootstrap-Lambda 443 "
                "ingress is skipped -- both applied out-of-band."
            )
        return vpc, list(private_data_subnets), pod_sg, pod_role, vpce_sg

    # --------------------------------------------------------------- outputs
    def _add_outputs(self) -> None:
        """Emit the operator-visible CfnOutputs the deploy step maps to chart values.

        The deploy step (CI helm upgrade) maps these to chart --set flags, mirroring
        the P0 EcrGhcrPrefix / ClusterName pattern (CfnOutput string -> --set value,
        NOT a CDK-side HelmChart resource):

            --set externalPostgresql.host=<DbClusterEndpoint>
            --set externalRedis.host=<CacheEndpoint>
            --set externalRedis.port=<CachePort>
            --set-string externalRedis.ssl=true   (serverless Valkey = TLS mandatory)

        DbSecretArn is the Aurora MASTER secret (break-glass + schema-bootstrap
        only; the runtime pod IAM-auths, it does NOT read this). The runtime user /
        IAM cache user / cluster-resource-id are surfaced so the operator can
        confirm the dbuser ARN + the cache IAM principal.
        """
        CfnOutput(
            self,
            "DbClusterEndpoint",
            value=self.replay_store.endpoint,
            description=(
                "Aurora WRITER endpoint. Map to chart externalPostgresql.host "
                "(--set externalPostgresql.host=...). The gateway WRITES, so it "
                "targets the writer."
            ),
        )
        CfnOutput(
            self,
            "DbClusterReaderEndpoint",
            value=self.replay_store.reader_endpoint,
            description="Aurora READER endpoint (read-replica reads; optional chart wiring)",
        )
        CfnOutput(
            self,
            "DbSecretArn",
            value=self.replay_store.secret_arn,
            description=(
                "Aurora MASTER secret ARN (break-glass + schema-bootstrap only). "
                "The runtime pod IAM-auths as routeiq and does NOT read this."
            ),
        )
        CfnOutput(
            self,
            "DbRuntimeUser",
            value="routeiq",
            description=(
                "Aurora runtime IAM-auth DB user (= chart externalPostgresql.username). "
                "The pod role holds rds-db:connect on dbuser .../routeiq."
            ),
        )
        CfnOutput(
            self,
            "DbClusterResourceId",
            value=self.replay_store.cluster_resource_identifier,
            description=(
                "Aurora cluster RESOURCE id (DbiResourceId) used to build the "
                "rds-db:connect dbuser ARN. Confirm the grant targets this id."
            ),
        )
        CfnOutput(
            self,
            "CacheEndpoint",
            value=self.cache.endpoint_address,
            description=(
                "ElastiCache serverless endpoint address. Map to chart "
                "externalRedis.host (--set externalRedis.host=...). Always-on TLS: "
                "also set externalRedis.ssl=true (REDIS_SSL=true)."
            ),
        )
        CfnOutput(
            self,
            "CachePort",
            value=self.cache.endpoint_port,
            description="ElastiCache serverless endpoint port. Map to chart externalRedis.port.",
        )
        CfnOutput(
            self,
            "CacheIamUserName",
            value=self.cache.iam_user_name,
            description=(
                "ElastiCache IAM-auth user name (user_id == user_name). The pod "
                "must present THIS name as the cache user on the SigV4 connect."
            ),
        )

    def _suppress_nag(self) -> None:
        """Apply the evidenced cdk-nag suppressions for this state stack (LAST)."""
        apply_state_nag_suppressions(self)
