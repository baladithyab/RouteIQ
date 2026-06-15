"""ReplayStoreConstruct - Aurora Postgres Serverless v2 for RouteIQ control plane (P1, ADR-0028).

Ported from
``/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/replay_store_construct.py``
(read symbol-by-symbol from the real source; the source wins over any offset).
RouteIQ reuses the VSR cluster TOPOLOGY almost verbatim -- Aurora Serverless v2 +
IAM auth + caller KMS CMK + 30-day rotation + writer/reader + private subnets +
snapshot-on-teardown -- and diverges only where RouteIQ's identity, schema, and
network shape differ.

Provisions the RouteIQ control-plane Postgres (governance / usage / prompts
state + the LiteLLM Prisma DB) as its OWN store inside the SEPARATE
``RouteIqStateStack`` (NOT folded into the P0 ``RouteIqStack``), per the
ADR-0028 ~30-min-rollback rule: the state plane must roll back independently of
the EKS/network foundation.

Resources owned by the construct:

  * **Aurora Postgres Serverless v2 cluster** with one writer + one reader
    ClusterInstance (Multi-AZ via 2 AZs), serverless v2 capacity 0.5-2.0 ACU for
    ``env_name == "dev"`` and 0.5-8.0 otherwise. A 0.5-ACU warm floor (NOT
    scale-to-zero) is the default: RouteIQ governance + quota fail-OPEN on a slow
    DB init, so a warm floor keeps the spend ledger reachable. ``min_acu=0``
    (scale-to-zero, paired with a 24h auto-pause) stays available behind the ctx
    flag for cost-sensitive dev (build-outline D2).
  * **IAM-DB authentication** (``iam_authentication=True``): no static DB password
    is handed to the runtime gateway pod, which IAM-auths as the RUNTIME user
    ``routeiq`` with short-lived tokens. The Secrets Manager secret is the MASTER
    (``postgres``) credential for break-glass + the schema-bootstrap leader only.
  * **KMS CMK from caller** (state-stack-owned): ``storage_encryption_key`` is the
    customer CMK so a state-stack rollback drops its own key.
  * **Secrets Manager master secret** with 30-day single-user rotation
    (``rds.Credentials.from_generated_secret('postgres')``).
  * **DB subnet group** over the PRIVATE_ISOLATED ``private-data`` tier (no
    internet egress; Aurora is a target, not an egress source).
  * **Dedicated SG** (``replay_store_sg``): ingress tcp/5432 from the app SG (the
    P0 ``pod_sg``) + an SG self-reference so the schema-bootstrap Lambda (whose
    ENI is in the same SG) can reach the cluster.
  * **Schema-bootstrap Lambda + cr.Provider + CustomResource** (Create AND
    Update, idempotent ``CREATE TABLE IF NOT EXISTS``). The Lambda authenticates
    with the MASTER secret (break-glass) to provision the runtime IAM user and
    apply RouteIQ's control-plane DDL. The ``schema_version`` ResourceProperty is
    the re-run lever.

RouteIQ divergences from the VSR source:

  * ``sr_task_sg`` -> ``app_sg`` (the P0 ``pod_sg``, an EKS Pod Identity gateway
    pod ENI SG, not a Fargate task SG).
  * ``_DB_NAME = "litellm"`` (the chart ``externalPostgresql.database`` default);
    ``_DB_MASTER_USER = "postgres"`` kept (Aurora master, break-glass);
    ``_DB_RUNTIME_USER = "routeiq"`` (the chart ``externalPostgresql.username``,
    the IAM-auth runtime identity).
  * ``grant_iam_db_connect`` grants the RUNTIME user ``routeiq``, NOT the master
    ``postgres`` (VSR granted ``postgres`` because its runtime used the master;
    RouteIQ's runtime IAM-auths as ``routeiq``).
  * The bootstrap Lambda sits in a SEPARATE subnet selection
    (``bootstrap_subnet_type``, default PRIVATE_WITH_EGRESS / ``private-app``)
    while the cluster stays in ``private-data`` (PRIVATE_ISOLATED). RouteIQ's
    SecretsManager + KMS interface endpoints live in the private-app tier, so a
    Lambda placed in private-data (isolated, no endpoints) could not reach
    Secrets Manager / KMS to read the master secret. The Lambda's ENI is still in
    ``replay_store_sg``, so the SG self-ref ingress lets it reach 5432 on the
    cluster regardless of the cluster's own (isolated) tier (build-outline
    section 3 / section 10).
  * Context key ``routeiq:state_pg_version`` (was ``vllm_sr:replay_store_pg_version``).
  * The schema-bootstrap Lambda runs RouteIQ's vendored idempotent DDL (A2A / MCP
    / audit tables) + provisions the runtime IAM user; Prisma stays the
    app-startup leader (``ROUTEIQ_LEADER_MIGRATIONS``), per build-outline D3.

Public attributes consumed by the state stack:
    ``cluster``, ``endpoint``, ``reader_endpoint``, ``port``, ``secret``,
    ``secret_arn``, ``replay_store_sg``, ``cluster_resource_identifier``.

ENGINE-VERSION GOTCHA (load-bearing, build-outline D1 /
mulch cdk-rds-aurora-engine-version-retired-at-deploy): AWS RETIRES old Aurora
Postgres minor versions. ``VER_16_13`` is pinned here AND must be verified LIVE
in us-west-2 with ``aws rds describe-db-engine-versions --engine
aurora-postgresql --region us-west-2`` before deploy. synth / cdk-nag /
unit-tests CANNOT catch a retired minor -- only a real ``CreateDBCluster`` does.
Overridable via the ``routeiq:state_pg_version`` context key for the next
retirement.

ASCII-only descriptions everywhere (mulch ec2-sg-description-charset + the IAM
em-dash trap): an out-of-charset description passes ``cdk synth`` but FAILS the
EC2 / IAM CREATE API.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import jsii
from aws_cdk import (
    Aws,
    BundlingOptions,
    CustomResource,
    DockerImage,
    Duration,
    ILocalBundling,
    RemovalPolicy,
)
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_rds as rds
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import custom_resources as cr
from constructs import Construct

# Lambda target for the docker-free local bundle. asyncpg ships a manylinux
# wheel, so pip can fetch the Linux/x86_64/py3.13 build from any host (incl.
# macOS) with --platform/--only-binary, no docker required.
_LAMBDA_PY_VERSION = "3.13"
_LAMBDA_PLATFORM = "manylinux2014_x86_64"

_REPLAY_PORT = 5432
_DB_NAME = "litellm"
_DB_MASTER_USER = "postgres"
_DB_RUNTIME_USER = "routeiq"

_SCHEMA_BOOTSTRAP_ASSET_PATH = str(
    Path(__file__).resolve().parents[2] / "lambda" / "routeiq-schema-bootstrap"
)

# Inline placeholder used when the asset directory is missing on disk (isolated
# unit-test invocation, docker absent + no local pip). Fails CLOSED at runtime so
# a placeholder deploy does not silently regress to a no-op schema bootstrap.
_SCHEMA_BOOTSTRAP_INLINE_PLACEHOLDER = (
    "def lambda_handler(event, context):\n"
    "    raise RuntimeError(\n"
    "        'routeiq-schema-bootstrap Lambda asset is unavailable; the \\n'\n"
    "        'stack was deployed with a placeholder. Re-run cdk deploy \\n'\n"
    "        'after lambda/routeiq-schema-bootstrap/handler.py is present.'\n"
    "    )\n"
)


@jsii.implements(ILocalBundling)
class _LocalPipBundler:
    """Docker-free local bundler for the schema-bootstrap Lambda.

    CDK tries ``try_bundle`` (this local path) BEFORE spinning up the docker
    bundling image; returning True means "I produced the asset, skip docker". We
    pip-install the requirements as the manylinux/py3.13 wheel into the output dir
    + copy the handler. This makes the asset deployable from a host WITHOUT docker
    (the prior inline-placeholder fallback set handler="handler.lambda_handler"
    but Code.from_inline creates index.py -> the Lambda failed at runtime with
    "No module named 'handler'"; mulch cdk live-deploy failure).
    """

    def __init__(self, asset_path: str) -> None:
        self._asset_path = asset_path

    def try_bundle(self, output_dir: str, *, image=None, **_kwargs) -> bool:  # noqa: ARG002
        req = os.path.join(self._asset_path, "requirements.txt")
        # 1) platform-targeted wheel install (no compile, no docker). Only run pip
        #    when requirements.txt has a real (non-comment, non-blank) line, so a
        #    boto3-only/stdlib handler does not error on an empty requirement set;
        #    a real dependency runs with check=True so a wheel-resolution failure
        #    PROPAGATES (fail-loud at synth) rather than shipping a deps-missing
        #    asset.
        real_reqs = False
        if os.path.isfile(req):
            with open(req, encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        real_reqs = True
                        break
        if real_reqs:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--platform",
                    _LAMBDA_PLATFORM,
                    "--implementation",
                    "cp",
                    "--python-version",
                    _LAMBDA_PY_VERSION,
                    "--only-binary=:all:",
                    "--target",
                    output_dir,
                    "-r",
                    req,
                ],
                check=True,
            )
        # 2) copy the handler + any sibling .py (NOT tests/caches).
        for entry in os.listdir(self._asset_path):
            if entry.endswith(".py"):
                shutil.copy2(
                    os.path.join(self._asset_path, entry),
                    os.path.join(output_dir, entry),
                )
        return True


class ReplayStoreConstruct(Construct):
    """Aurora Postgres Serverless v2 cluster + schema-bootstrap CR for RouteIQ."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        kms_key: kms.IKey,
        vpc: ec2.IVpc,
        app_sg: ec2.ISecurityGroup,
        private_data_subnets: Sequence[ec2.ISubnet] | None = None,
        min_acu: float | None = None,
        max_acu: float | None = None,
        subnet_type: ec2.SubnetType = ec2.SubnetType.PRIVATE_ISOLATED,
        bootstrap_subnet_type: ec2.SubnetType = ec2.SubnetType.PRIVATE_WITH_EGRESS,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        self._kms_key = kms_key
        self._vpc = vpc
        self._subnet_type = subnet_type
        self._bootstrap_subnet_type = bootstrap_subnet_type

        # Default ACU bounds: dev = 0.5-2.0, other envs = 0.5-8.0. The 0.5 warm
        # floor (NOT 0/scale-to-zero) is the default per build-outline D2 -- the
        # min_acu=0 scale-to-zero path is opt-in via the state-stack ctx flag and
        # pairs a 24h auto-pause below.
        if min_acu is None:
            min_acu = 0.5
        if max_acu is None:
            max_acu = 2.0 if env_name == "dev" else 8.0
        self.min_acu = float(min_acu)
        self.max_acu = float(max_acu)

        # ------------------------------------------------------------- SG (own)
        # Self-owned SG. Ingress 5432 from app_sg (the P0 pod_sg) only -- no CIDR
        # ingress, no peer SGs lifted from the network. ASCII-only description.
        self.replay_store_sg = ec2.SecurityGroup(
            self,
            "ReplayStoreSg",
            vpc=vpc,
            description=(
                "Aurora Postgres RouteIQ control-plane cluster ENIs (ingress 5432 from pod_sg only)"
            ),
            allow_all_outbound=True,
        )
        self.replay_store_sg.add_ingress_rule(
            peer=app_sg,
            connection=ec2.Port.tcp(_REPLAY_PORT),
            description="Postgres 5432 from RouteIQ gateway pod to Aurora",
        )
        # Schema-bootstrap Lambda runs inside the same SG, so the cluster accepts
        # its connections via SG self-reference. The Lambda's ENI lives in the
        # private-app tier (for SecretsManager/KMS endpoint reachability) but is
        # in THIS SG, so the self-ref ingress lets it reach 5432 on the cluster
        # regardless of the cluster's (isolated) tier.
        self.replay_store_sg.add_ingress_rule(
            peer=self.replay_store_sg,
            connection=ec2.Port.tcp(_REPLAY_PORT),
            description="Postgres 5432 from schema-bootstrap Lambda (same SG)",
        )

        # --------------------------------------------------------- subnet group
        # Across 2 AZs. PRIVATE_ISOLATED (the private-data tier) -- no internet
        # egress; Aurora is a target. When the caller passes private_data_subnets
        # explicitly we pin those subnets (the RouteIQ state stack does this with
        # the P0 network's private-data list); otherwise the subnet group selects
        # by subnet_type from the VPC.
        if private_data_subnets is not None:
            vpc_subnets = ec2.SubnetSelection(subnets=list(private_data_subnets))
        else:
            vpc_subnets = ec2.SubnetSelection(subnet_type=subnet_type)
        self.subnet_group = rds.SubnetGroup(
            self,
            "SubnetGroup",
            description=(
                f"RouteIQ {env_name} Aurora control-plane subnet group "
                f"({subnet_type.value}, Multi-AZ)"
            ),
            vpc=vpc,
            vpc_subnets=vpc_subnets,
        )

        # ---------------------------------------------------- credentials secret
        # Aurora-generated MASTER secret. The runtime gateway pod does NOT read
        # this (it IAM-auths as routeiq); the schema-bootstrap Lambda reads it as
        # the break-glass leader. CDK owns the secret -- NO literal ARN anywhere.
        self._credentials = rds.Credentials.from_generated_secret(
            _DB_MASTER_USER, encryption_key=kms_key
        )

        # --------------------------------------------------------------- cluster
        # Serverless v2 uses provisioned engine_mode under the hood; no EngineMode
        # property in the synthesized template (Match.absent() in the unit test).
        #
        # ENGINE VERSION: VER_16_13 is pinned (build-outline D1). AWS RETIRES old
        # minors; this MUST be verified live in us-west-2 with
        # `aws rds describe-db-engine-versions --engine aurora-postgresql
        # --region us-west-2` -- synth/nag/unit-tests CANNOT catch a retired
        # version, only a real CreateDBCluster does
        # (mulch cdk-rds-aurora-engine-version-retired-at-deploy). Overridable via
        # the routeiq:state_pg_version context key for the next retirement.
        pg_version = self.node.try_get_context("routeiq:state_pg_version")
        if pg_version:
            engine_version = rds.AuroraPostgresEngineVersion.of(
                str(pg_version), str(pg_version).split(".", 1)[0]
            )
        else:
            engine_version = rds.AuroraPostgresEngineVersion.VER_16_13

        self.cluster = rds.DatabaseCluster(
            self,
            "Cluster",
            engine=rds.DatabaseClusterEngine.aurora_postgres(version=engine_version),
            credentials=self._credentials,
            default_database_name=_DB_NAME,
            iam_authentication=True,
            storage_encryption_key=kms_key,
            storage_encrypted=True,
            vpc=vpc,
            subnet_group=self.subnet_group,
            security_groups=[self.replay_store_sg],
            port=_REPLAY_PORT,
            serverless_v2_min_capacity=self.min_acu,
            serverless_v2_max_capacity=self.max_acu,
            # Scale-to-zero: Aurora Serverless v2 permits min_capacity=0 ONLY when
            # paired with an auto-pause duration. Set the longest delay (24h) when
            # an operator opts into the $0 idle floor (min_acu=0) so the cluster
            # pauses only after a full idle day -- minimising cold-resume hits on
            # the DB-init connect path (RouteIQ governance/quota fail-OPEN on a
            # slow init, so a fast warm connect matters). Omitted entirely when
            # min_acu>0 so the default (warm-floor) CFN output + snapshots stay
            # byte-identical (build-outline D2).
            serverless_v2_auto_pause_duration=(Duration.hours(24) if self.min_acu == 0 else None),
            writer=rds.ClusterInstance.serverless_v2(
                "writer",
                publicly_accessible=False,
            ),
            readers=[
                rds.ClusterInstance.serverless_v2(
                    "reader1",
                    scale_with_writer=True,
                    publicly_accessible=False,
                ),
            ],
            backup=rds.BackupProps(
                retention=Duration.days(7 if env_name == "dev" else 14),
                preferred_window="03:00-04:00",
            ),
            preferred_maintenance_window="sun:04:30-sun:05:30",
            deletion_protection=env_name != "dev",
            removal_policy=(RemovalPolicy.DESTROY if env_name == "dev" else RemovalPolicy.SNAPSHOT),
            cloudwatch_logs_exports=["postgresql"],
        )

        # 30-day single-user rotation. The L2 provisions the rotation Lambda; no
        # SecretsManager nag fires when this is wired.
        self.cluster.add_rotation_single_user(
            automatically_after=Duration.days(30),
        )

        # Invariant guard: the L2 sets cluster.secret to the generated secret.
        secret = self.cluster.secret
        if secret is None:
            raise RuntimeError(
                "rds.DatabaseCluster.secret is None despite "
                "Credentials.from_generated_secret() - aws-cdk-lib invariant "
                "violation"
            )
        self.secret: secretsmanager.ISecret = secret
        self.secret_arn: str = self.secret.secret_arn

        # ---------------------------------------------------- public attributes
        self.endpoint: str = self.cluster.cluster_endpoint.hostname
        self.reader_endpoint: str = self.cluster.cluster_read_endpoint.hostname
        self.port: int = _REPLAY_PORT
        # The cluster resource identifier is needed to build the rds-db:connect
        # dbuser ARN (grant_iam_db_connect) and is surfaced as a state-stack
        # output so the operator can confirm the dbuser ARN.
        self.cluster_resource_identifier: str = self.cluster.cluster_resource_identifier

        # ----------------------------------------------- schema-bootstrap CR
        self._schema_bootstrap_lambda = self._build_schema_bootstrap_lambda()
        self._schema_bootstrap_provider = self._build_schema_bootstrap_provider(
            self._schema_bootstrap_lambda
        )
        self.schema_bootstrap_cr = self._build_schema_bootstrap_cr(self._schema_bootstrap_provider)

    # ------------------------------------------------------------------ helpers
    def _build_schema_bootstrap_lambda(self) -> lambda_.Function:
        asset_ok = os.path.isdir(_SCHEMA_BOOTSTRAP_ASSET_PATH) and os.path.isfile(
            os.path.join(_SCHEMA_BOOTSTRAP_ASSET_PATH, "handler.py")
        )
        if asset_ok:
            # Bundle the asset with a DOCKER-FREE local bundler first (CDK calls
            # try_bundle before the docker image): pip-installs the manylinux
            # asyncpg wheel + copies handler.py. The docker image is kept as the
            # fallback for hosts where the local pip path cannot run, so bundling
            # succeeds whether or not docker is present.
            code: lambda_.Code = lambda_.Code.from_asset(
                _SCHEMA_BOOTSTRAP_ASSET_PATH,
                bundling=BundlingOptions(
                    image=DockerImage.from_registry("public.ecr.aws/sam/build-python3.13"),
                    local=_LocalPipBundler(_SCHEMA_BOOTSTRAP_ASSET_PATH),
                    command=[
                        "bash",
                        "-c",
                        (
                            "pip install -r requirements.txt "
                            "-t /asset-output && cp -au . /asset-output"
                        ),
                    ],
                ),
            )
        else:
            # The asset genuinely is not on disk (isolated unit test). The inline
            # placeholder RAISES at runtime so a placeholder deploy fails loudly
            # rather than silently no-opping the schema. A real deploy always has
            # the asset + local bundler.
            code = lambda_.Code.from_inline(_SCHEMA_BOOTSTRAP_INLINE_PLACEHOLDER)

        fn = lambda_.Function(
            self,
            "SchemaBootstrapFn",
            function_name=f"routeiq-{self.env_name}-schema-bootstrap",
            description=(
                "Bootstraps the RouteIQ control-plane schema in the Aurora "
                "cluster. Provisions the runtime IAM user routeiq, then applies "
                "idempotent CREATE TABLE IF NOT EXISTS DDL. Invoked by the "
                "schema-bootstrap CFN custom resource on Create AND Update."
            ),
            runtime=lambda_.Runtime.PYTHON_3_13,
            # Asset bundle exposes handler.py -> handler.lambda_handler. The inline
            # placeholder fallback (asset missing, tests only) becomes index.py via
            # Code.from_inline, so its handler is index.lambda_handler -- using the
            # wrong name is what produced the "No module named 'handler'" runtime
            # failure when the old inline path deployed.
            handler="handler.lambda_handler" if asset_ok else "index.lambda_handler",
            code=code,
            timeout=Duration.minutes(5),
            memory_size=512,
            vpc=self._vpc,
            # The bootstrap Lambda lives in the private-app (PRIVATE_WITH_EGRESS)
            # tier so it can reach the SecretsManager + KMS interface endpoints
            # (which live in private-app, NOT the isolated private-data tier the
            # cluster sits in). It still reaches 5432 on the cluster via the
            # replay_store_sg self-ref ingress (build-outline section 3 / 10).
            vpc_subnets=ec2.SubnetSelection(subnet_type=self._bootstrap_subnet_type),
            security_groups=[self.replay_store_sg],
            environment={
                "DB_HOST": self.endpoint,
                "DB_PORT": str(_REPLAY_PORT),
                "DB_NAME": _DB_NAME,
                "DB_SECRET_ARN": self.secret_arn,
                "DB_RUNTIME_USER": _DB_RUNTIME_USER,
                "ENV_NAME": self.env_name,
                "LOG_LEVEL": "INFO",
            },
        )

        # Grant the bootstrap Lambda read access to the master credentials.
        # secret.grant_read wires BOTH secretsmanager:GetSecretValue AND a
        # kms:Decrypt grant on the data CMK.
        self.secret.grant_read(fn)
        # NOTE: do NOT add fn.node.add_dependency(self.cluster). secret.grant_read
        # writes a kms:Decrypt key-policy grant naming this fn's role, which makes
        # Cmk depend on the role. Cluster references Cmk via KmsKeyId, so
        # role -> cluster -> Cmk -> role would close a cycle. The CR resource below
        # picks up the cluster dependency instead so CFN orders cluster creation
        # before the schema-bootstrap invocation.

        logs.LogGroup(
            self,
            "SchemaBootstrapLogGroup",
            log_group_name=f"/aws/lambda/{fn.function_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        return fn

    def _build_schema_bootstrap_provider(self, on_event_handler: lambda_.IFunction) -> cr.Provider:
        return cr.Provider(
            self,
            "SchemaBootstrapProvider",
            on_event_handler=on_event_handler,
        )

    def _build_schema_bootstrap_cr(self, provider: cr.Provider) -> CustomResource:
        # The Provider's framework Lambda only invokes the on_event_handler when
        # CFN issues Create/Update/Delete on the custom resource. CustomResource
        # passes ``properties`` as the ResourceProperties of every request, so the
        # static ``schema_version`` is the re-run lever: bumping it (e.g. "1" ->
        # "2") flips the property hash -> CFN issues an Update -> the handler
        # re-runs the idempotent DDL.
        custom = CustomResource(
            self,
            "SchemaBootstrapCr",
            service_token=provider.service_token,
            properties={
                "schema_version": "1",
                "db_host": self.endpoint,
                "db_port": str(_REPLAY_PORT),
                "db_name": _DB_NAME,
            },
        )
        # Order the CR after the cluster so the writer endpoint is reachable.
        # Putting the dependency on the CR (NOT the Lambda role) avoids the
        # Cmk <-> grant_read <-> role <-> cluster cycle (see the secret.grant_read
        # note above).
        custom.node.add_dependency(self.cluster)
        return custom

    # ------------------------------------------------------- public API: grants
    def grant_iam_db_connect(self, role: iam.IRole, db_user: str = _DB_RUNTIME_USER) -> None:
        """Grant ``rds-db:connect`` on the RUNTIME user to ``role``.

        CRITICAL: grants the RUNTIME user (default ``routeiq``), NOT the master
        ``postgres``. The RouteIQ gateway pod IAM-auths as ``routeiq`` (the user
        the schema-bootstrap Lambda provisions with ``GRANT rds_iam``); VSR granted
        ``postgres`` only because its runtime used the master, which RouteIQ does
        not. The resource is a fully-specified dbuser ARN (no wildcard), so this
        needs no nag suppression. The RouteIQ state stack calls this with the
        cross-stack P0 pod role.
        """
        role.add_to_principal_policy(
            iam.PolicyStatement(
                sid="RdsIamDbConnectRouteIqState",
                effect=iam.Effect.ALLOW,
                actions=["rds-db:connect"],
                resources=[
                    (
                        f"arn:{Aws.PARTITION}:rds-db:{Aws.REGION}:{Aws.ACCOUNT_ID}"
                        f":dbuser:{self.cluster.cluster_resource_identifier}"
                        f"/{db_user}"
                    )
                ],
            )
        )
