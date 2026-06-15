"""ElastiCache **Serverless** for Valkey for RouteIQ on AWS (P1, ADR-0029).

Ported from ``/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/cache_construct.py``
(read symbol-by-symbol from the real source; the source wins over any offset).
RouteIQ is greenfield on AWS, so this is a ``CfnServerlessCache`` from day one
(pay-per-use: ~$0.084/GB-hr storage + $0.0023/M ECPU, 100MB Valkey floor approx
$7/mo idle). The cache load is small and bursty (hot rate-limiter + governance
counters), so serverless is the right shape: auto-scaling, Multi-AZ by default,
no node sizing / replica / failover management.

    * **Valkey 8.x serverless cache.** Engine auto-scales; no node type, no
      num_node_groups / replicas_per_node_group.
    * At-rest encryption with the state-stack-owned customer KMS CMK; in-transit
      TLS is ALWAYS required on serverless caches (not a toggle). Consumers MUST
      connect with TLS (the chart sets REDIS_SSL=true).
    * IAM authentication (Valkey 7.2+) via a dedicated CfnUser + CfnUserGroup;
      Redis AUTH tokens are intentionally NOT used. The reserved default user is
      disabled so unauthenticated connects fail closed.
    * Daily snapshots retained 7 days.
    * A dedicated security group (``redis_sg``) attached to the serverless cache;
      ingress on tcp/6379 from the RouteIQ pod SG is wired late in
      ``attach_dependencies`` (idempotency-guarded), so the construct stays
      constructible without the peer SG for an isolated unit-test snapshot.

RouteIQ divergences from the VSR source:

    * Takes ``vpc`` + ``private_data_subnets`` DIRECTLY instead of a
      NetworkConstruct handle. The RouteIQ state stack does not own a
      NetworkConstruct; it receives the P0 VPC + private-data subnet list as
      cross-stack inputs (see ``routeiq_state_stack``).
    * The vestigial ``CfnSubnetGroup`` is DROPPED. Serverless caches wire a flat
      ``subnet_ids`` list directly; the subnet group was kept in VSR only for a
      sibling that referenced its name, and RouteIQ has no such sibling.
    * ``node_type`` is DROPPED (it was vestigial for serverless in VSR too).
    * The peer SG in ``attach_dependencies`` is the P0 ``pod_sg`` (an EKS Pod
      Identity gateway pod ENI SG), not a Fargate task SG.
    * ``grant_iam_connect(role)`` lives on this construct (VSR put the
      ``elasticache:Connect`` statement in a SecurityConstruct). The state stack
      calls it with the cross-stack pod role, symmetric with the Aurora
      construct's ``grant_iam_db_connect``.

The construct exposes the public attributes downstream wiring consumes:
    endpoint_address (str token), endpoint_port (str token), cache_arn,
    iam_user_arn, iam_user_name, redis_sg.

Resource naming (all ``routeiq-<env>-cache*``):
    ServerlessCacheName: ``routeiq-<env>-cache-sl`` (the ``-sl`` suffix is kept
        per the cross-cache-type name-uniqueness lesson: ElastiCache enforces
        name uniqueness ACROSS cache types, so a future RG / serverless coexist
        window does not collide; harmless on greenfield, correct convention --
        mulch elasticache-rg-to-serverless-name-collision).
    IAM UserId / UserName: ``routeiq-<env>-cache-iam-user`` (MUST be equal for
        IAM auth, and MUST equal the AWS principal that signs the connect).
    Disabled default UserId: ``routeiq-<env>-cache-default``.
    UserGroupId: ``routeiq-<env>-cache-ug``.

Every SG description here stays within the EC2 description charset allowlist
(a-zA-Z0-9 . _ - : / ( ) # , @ [ ] + = & ; { } ! $ *): no arrows, em-dashes,
backticks, pipes, or angle brackets. An out-of-charset description passes cdk
synth but FAILS the EC2 CREATE API (mulch ec2-sg-description-charset).
"""

from __future__ import annotations

from collections.abc import Sequence

from aws_cdk import Aws, RemovalPolicy
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_elasticache as elasticache
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from constructs import Construct

_REDIS_PORT = 6379

# Inert placeholder password for the disabled ``default`` user. It never grants
# access because the access-string starts with ``off`` (no commands permitted);
# it exists ONLY to satisfy Valkey's "default user must have an auth mode"
# requirement. IAM auth is impossible on the default user (its user_id can never
# equal its reserved user_name "default"), and Valkey rejects no_password_required
# on the default user at deploy time -- so PASSWORD with an inert pw is the only
# mode Valkey accepts here (mulch valkey-default-user-rejects-no-password).
#
# NOTE: this is NOT a real secret. It grants nothing (access_string is "off ...");
# the use of a literal here is the deploy-required Valkey workaround, not a
# credential. It is also overwritten by no real connection path.
_DISABLED_DEFAULT_USER_PASSWORD = "RouteIqDefaultUserDisabledPlaceholderPw01"  # noqa: S105


class CacheConstruct(Construct):
    """ElastiCache Serverless for Valkey + IAM-auth user group (RouteIQ P1)."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        vpc: ec2.IVpc,
        private_data_subnets: Sequence[ec2.ISubnet],
        kms_key: kms.IKey,
        engine_version: str = "8.0",
        snapshot_retention_days: int = 7,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        self._vpc = vpc
        self._kms_key = kms_key
        # Idempotency guard for attach_dependencies: a double call must not add a
        # duplicate ingress rule.
        self._app_attached = False

        # Dedicated SG for the serverless cache's network interfaces. Owned here
        # (not lifted from the network) so the cache feature is self-contained.
        # ASCII-only description (EC2 charset rule).
        self.redis_sg = ec2.SecurityGroup(
            self,
            "RedisSg",
            vpc=vpc,
            description=(
                "ElastiCache Valkey serverless cache ENIs (ingress 6379 from pod_sg only)"
            ),
            allow_all_outbound=True,
        )

        # IAM-auth user (Valkey 7.2+). For IAM authentication the UserName MUST
        # equal the UserId AND must equal the AWS principal that signs the connect
        # (the RouteIQ gateway pod presents its Pod Identity session as this
        # user-name to ElastiCache). ElastiCache rejects a mismatch with
        # "User Id and User name must be same for authentication type: iam".
        # access_string "on ~* +@all": user enabled, all keys, all commands.
        self.iam_user_name = f"routeiq-{env_name}-cache-iam-user"
        self.iam_user = elasticache.CfnUser(
            self,
            "IamUser",
            user_id=self.iam_user_name,
            user_name=self.iam_user_name,
            engine="valkey",
            authentication_mode={"Type": "iam"},
            access_string="on ~* +@all",
        )

        # The ElastiCache "default" user always exists; override it to be disabled
        # so unauthenticated connects fail closed. Auth mode is PASSWORD (not IAM,
        # not no-password) because:
        #   * IAM auth requires user_id == user_name, but the reserved default
        #     user MUST keep user_name="default" while user_id must be the unique
        #     routeiq-<env>-cache-default -- they can never match, so ElastiCache
        #     rejects IAM on the default user.
        #   * Valkey also rejects no_password_required=True for the default user.
        # The inert placeholder pw grants nothing because access_string starts
        # with "off" (mulch valkey-default-user-rejects-no-password).
        self.default_user = elasticache.CfnUser(
            self,
            "DefaultUser",
            user_id=f"routeiq-{env_name}-cache-default",
            user_name="default",
            engine="valkey",
            authentication_mode={
                "Type": "password",
                "Passwords": [_DISABLED_DEFAULT_USER_PASSWORD],
            },
            access_string="off ~keys* -@all",
        )

        self.user_group = elasticache.CfnUserGroup(
            self,
            "UserGroup",
            user_group_id=f"routeiq-{env_name}-cache-ug",
            engine="valkey",
            user_ids=[self.default_user.user_id, self.iam_user.user_id],
        )
        # CfnUserGroup must wait for BOTH CfnUser resources at deploy time. CDK
        # does NOT infer this from the user_ids string list (they are plain
        # strings), so both add_dependency calls are MANDATORY or the group can be
        # created before the users exist.
        self.user_group.add_dependency(self.default_user)
        self.user_group.add_dependency(self.iam_user)

        # Serverless caches take a flat subnet_ids list (NOT a CfnSubnetGroup ref,
        # which is a node-based-cluster concept). RouteIQ drops the vestigial
        # subnet group entirely; the subnets are the P0 private-data
        # (PRIVATE_ISOLATED) tier, two AZs for Multi-AZ.
        subnet_ids = [s.subnet_id for s in private_data_subnets]

        # NAME: routeiq-<env>-cache-sl -- the -sl suffix is kept by convention
        # because ElastiCache enforces name uniqueness ACROSS cache types (a
        # node-based RG id and a serverless-cache name share one uniqueness
        # constraint). Harmless on greenfield, but it lets a future RG / serverless
        # coexist during any create-then-delete window
        # (mulch elasticache-rg-to-serverless-name-collision).
        self.serverless_cache_name = f"routeiq-{env_name}-cache-sl"
        # major_engine_version is the MAJOR ONLY ("8") for serverless, vs a
        # node-based engine_version ("8.0"). Derive it so callers can keep passing
        # engine_version="8.0". This is the single most common copy error.
        _major = str(engine_version).split(".")[0]
        self.serverless_cache = elasticache.CfnServerlessCache(
            self,
            "ServerlessCache",
            serverless_cache_name=self.serverless_cache_name,
            engine="valkey",
            major_engine_version=_major,
            description=(
                f"RouteIQ {env_name} hot rate-limiter and governance counter cache (serverless)"
            ),
            security_group_ids=[self.redis_sg.security_group_id],
            subnet_ids=subnet_ids,
            kms_key_id=kms_key.key_id,
            user_group_id=self.user_group.user_group_id,
            snapshot_retention_limit=snapshot_retention_days,
            daily_snapshot_time="03:00",
        )
        # Explicit dependency so the cache waits for the user group at deploy.
        self.serverless_cache.add_dependency(self.user_group)
        # Removal policy: DESTROY for dev, RETAIN otherwise. Serverless avoids the
        # node-based RG teardown deadlock (no subnet-group / user-group ordering
        # trap on serverless delete), but the dev=DESTROY posture is kept so a
        # rolled-back dev deploy unwinds cleanly and non-dev protects the store.
        self.serverless_cache.apply_removal_policy(
            RemovalPolicy.DESTROY if env_name == "dev" else RemovalPolicy.RETAIN
        )

        # ------------------------------------------------------- public attrs
        # Endpoint.Address / .Port are runtime attributes synthesised by
        # ElastiCache Serverless; expose as tokens for downstream env-var wiring
        # (REDIS_HOST / REDIS_PORT in the chart).
        self.endpoint_address: str = self.serverless_cache.attr_endpoint_address
        self.endpoint_port: str = self.serverless_cache.attr_endpoint_port

        # ARNs for the elasticache:Connect grant. Prefer the synthesised attr_arn
        # (exact) for the cache; the user ARN is hand-built (no attr exposed for
        # it). Serverless ARN forms:
        #   arn:<partition>:elasticache:<region>:<account>:serverlesscache:<name>
        #   arn:<partition>:elasticache:<region>:<account>:user:<userId>
        self.cache_arn: str = self.serverless_cache.attr_arn
        self.iam_user_arn: str = (
            f"arn:{Aws.PARTITION}:elasticache:{Aws.REGION}:{Aws.ACCOUNT_ID}"
            f":user:{self.iam_user_name}"
        )

    def attach_dependencies(
        self,
        *,
        app_sg: ec2.ISecurityGroup | None = None,
    ) -> None:
        """Wire the late-bound peer-SG ingress (tcp/6379 from the pod SG).

        The peer SG (P0 ``pod_sg``) is owned by another stack/construct; passing
        it here -- rather than requiring it in ``__init__`` -- keeps the cache
        constructible WITHOUT the peer for an isolated unit-test snapshot. The
        ``self._app_attached`` guard makes a double call idempotent (no duplicate
        ingress rule).
        """
        if app_sg is not None and not self._app_attached:
            self.redis_sg.add_ingress_rule(
                peer=app_sg,
                connection=ec2.Port.tcp(_REDIS_PORT),
                description="Valkey 6379 from RouteIQ gateway pod to ElastiCache serverless cache",
            )
            self._app_attached = True

    def grant_iam_connect(self, role: iam.IRole) -> None:
        """Grant ``elasticache:Connect`` on the cache AND the IAM-auth user.

        ElastiCache IAM auth requires the connecting principal to hold
        ``elasticache:Connect`` against BOTH the cache resource and the user
        resource it authenticates AS. Both are fully-specified ARNs (no wildcard),
        so this needs no nag suppression. The RouteIQ state stack calls this with
        the cross-stack P0 pod role, symmetric with the Aurora construct's
        ``grant_iam_db_connect``.
        """
        role.add_to_principal_policy(
            iam.PolicyStatement(
                sid="ElastiCacheConnectRouteIqState",
                actions=["elasticache:Connect"],
                resources=[self.cache_arn, self.iam_user_arn],
            )
        )
