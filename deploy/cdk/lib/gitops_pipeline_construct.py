"""GitOpsPipelineConstruct - the ADR-0026 Day-2 GitOps deploy tier (RouteIQ-1669).

The flag-gated, DEFAULT-OFF *remaining* half of ADR-0026's "Day-2 GitOps path".
The validator-mutation audit CORE already shipped (commit 761a4ee:
``ConfigStateConstruct(enable_config_audit=...)``); this construct adds the
CodePipeline + the narrow deployer IAM role ADR-0026 describes, so RouteIQ-1669
goes from PARTIAL to FULL on the cred-free axis. The LIVE deploy (real config
commits, the prod approval click, an actual AppConfig deployment) is the operator
half and stays operator-gated.

What this construct AUTHORS (cred-free, byte-stable when the flag is off):

    Source bucket (versioned, KMS-SSE, BPA, enforce_ssl) - operators commit
       ``config.yaml`` here; an S3 source action triggers the pipeline.
    CodePipeline:
       Source[config bucket]  ->  [Approve, PROD ONLY]  ->  Deploy
       (the Approve stage is added ONLY when ``env_name == "prod"`` - dev/stage
        flow straight to Deploy; prod requires a manual approval click).
    Deployer CodeBuild project - runs the deploy buildspec
       (``appconfig create-hosted-configuration-version`` + ``start-deployment``).
       Its service role IS the NARROW DEPLOYER ROLE:
         * EXACTLY five AppConfig actions:
             - CreateHostedConfigurationVersion  (push the new config version)
             - StartDeployment                   (roll it out via the strategy)
             - GetConfigurationProfile           (resolve the profile)
             - GetEnvironment                    (resolve the target environment)
             - GetDeploymentStrategy             (resolve the rollout strategy)
         * an explicit DENY of UpdateConfigurationProfile + DeleteConfigurationProfile
           so the deployer CANNOT strip the load-bearing config VALIDATOR
           (ADR-0026's single-config-gate invariant; the
           ``enable_config_audit`` rule alarms on it, this DENY makes the deployer
           structurally unable to do it).

THE VALIDATOR INVARIANT (why the explicit DENY matters): the AppConfig
configuration profile carries the LAMBDA validator (ConfigStateConstruct). The
validator is the single gate that rejects bad/secret-bearing config at deploy
time. A deployer that could ``UpdateConfigurationProfile`` could strip the
``Validators`` array; a ``DeleteConfigurationProfile`` removes the profile + its
validator outright. The deployer's whole job is to PUSH versions + START
deployments - never to mutate the profile - so an explicit IAM DENY on those two
actions enforces the invariant at the principal level (deny always wins over any
allow), complementing the audit alarm.

FLAG-GATED, DEFAULT OFF (``routeiq:enable_gitops_pipeline``): the composition root
never instantiates this construct unless the flag is true, so a default synth
emits ZERO ``AWS::CodePipeline::Pipeline`` / ``AWS::CodeBuild::Project`` /
deployer-role resources and the snapshot stays byte-stable. There is ZERO live
consumer this wave; the live deploy is operator-gated.

DETERMINISTIC SYNTH (no Docker asset): the deployer CodeBuild project uses an
INLINE buildspec (``BuildSpec.from_object``), NOT a ``from_asset`` bundle, so the
synthesised template is byte-identical on a Docker-equipped host and a Docker-less
one (the cred-free gate + the snapshot). Mirrors the ConfigStateConstruct's
"deterministic synth, no implicit Docker probe" discipline (RouteIQ-4772).

ASCII / Latin-1-only Descriptions (P0 section 4.5): an em-dash (U+2014) passes
``cdk synth`` but FAILS the IAM/CFN CREATE API. Every Description stays ASCII.

cdk-nag suppressions are INLINE on this construct (matching the
ConfigStateConstruct / DataLakeConstruct ownership split), NOT in
nag_suppressions.py.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack, Tags
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_codepipeline as codepipeline
from aws_cdk import aws_codepipeline_actions as cp_actions
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from cdk_nag import NagPackSuppression, NagSuppressions
from constructs import Construct

# The EXACTLY-FIVE AppConfig actions the GitOps deployer needs to push a new
# config version and roll it out (ADR-0026 "the deployer role holds exactly five
# AppConfig actions"). These are the deploy-path actions for
# ``create-hosted-configuration-version`` + ``start-deployment`` + the three GETs
# that resolve the profile / environment / strategy the deployment binds.
_DEPLOYER_APPCONFIG_ACTIONS = [
    "appconfig:CreateHostedConfigurationVersion",
    "appconfig:StartDeployment",
    "appconfig:GetConfigurationProfile",
    "appconfig:GetEnvironment",
    "appconfig:GetDeploymentStrategy",
]

# The two MUTATING profile API calls the deployer is explicitly DENIED. An
# UpdateConfigurationProfile can strip the Validators array (removing the
# load-bearing config validator); a DeleteConfigurationProfile removes the profile
# + validator outright. The deployer must never do either - so deny them at the
# principal level (an explicit Deny beats any Allow), enforcing ADR-0026's
# single-config-gate invariant structurally, not just via the audit alarm.
_DEPLOYER_DENIED_ACTIONS = [
    "appconfig:UpdateConfigurationProfile",
    "appconfig:DeleteConfigurationProfile",
]

# The object key in the source bucket the pipeline watches + the deployer reads.
_SOURCE_CONFIG_KEY = "config.zip"


class GitOpsPipelineConstruct(Construct):
    """The ADR-0026 Day-2 GitOps CodePipeline + narrow deployer role (flag-gated).

    Public attributes for the composition root / outputs:
    ``source_bucket`` (the config source ``s3.Bucket``), ``pipeline`` (the
    ``codepipeline.Pipeline``), ``deployer_role`` (the narrow CodeBuild project
    service role carrying the 5 AppConfig actions + the explicit deny),
    ``deploy_project`` (the deployer ``codebuild.Project``).
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        appconfig_application_id: str,
        appconfig_environment_id: str,
        appconfig_profile_id: str,
        appconfig_strategy_id: str,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        stack = Stack.of(self)
        # prod is the only env that requires a manual approval click; dev/stage
        # flow straight Source -> Deploy ([Approve, prod only], per the seed).
        self._is_prod = env_name == "prod"
        _retain = env_name != "dev"

        # -- A dedicated CMK for the pipeline + source bucket (rotated, RETAIN in
        # non-dev so a teardown never orphans the key into PendingDeletion -- the
        # documented ECR-pull footgun). ASCII description.
        self.kms_key = kms.Key(
            self,
            "GitOpsKey",
            description=f"RouteIQ {env_name} GitOps config-pipeline CMK",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN if _retain else RemovalPolicy.DESTROY,
        )

        # -- 1. The config SOURCE bucket --------------------------------------
        # Operators commit config.zip (a zipped config.yaml) here; the S3 source
        # action triggers the pipeline on a new object version. Standard posture:
        # KMS-SSE (own CMK), versioned (S3Trigger=EVENTS needs versioning), BPA,
        # enforce_ssl. Versioned so the pipeline can pin an exact source revision.
        self.source_bucket = s3.Bucket(
            self,
            "ConfigSourceBucket",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.kms_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN if _retain else RemovalPolicy.DESTROY,
            auto_delete_objects=not _retain,
        )

        # -- 2. The NARROW DEPLOYER ROLE (the load-bearing security surface) ---
        # This is the CodeBuild deploy project's service role: it is what actually
        # runs the AppConfig deploy, so the 5 actions + the explicit deny live on
        # IT (the deploy actor), not on a passive pipeline role. ASCII description.
        self.deployer_role = iam.Role(
            self,
            "DeployerRole",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            description=(
                f"RouteIQ {env_name} GitOps AppConfig deployer (exactly 5 "
                "AppConfig actions + explicit deny of profile mutation)"
            ),
        )
        # The env-scoped + profile-scoped AppConfig ARNs the deploy actions touch.
        # CreateHostedConfigurationVersion + GetConfigurationProfile act on the
        # configurationprofile ARN; StartDeployment + GetEnvironment act on the
        # environment ARN; GetDeploymentStrategy acts on the strategy ARN. We scope
        # to the application's resources (NEVER ``*``): the three child ARNs under
        # this application's profile / environment / strategy.
        app_arn = stack.format_arn(
            service="appconfig",
            resource="application",
            resource_name=appconfig_application_id,
        )
        profile_arn = stack.format_arn(
            service="appconfig",
            resource="application",
            resource_name=(
                f"{appconfig_application_id}/configurationprofile/{appconfig_profile_id}"
            ),
        )
        environment_arn = stack.format_arn(
            service="appconfig",
            resource="application",
            resource_name=f"{appconfig_application_id}/environment/{appconfig_environment_id}",
        )
        strategy_arn = stack.format_arn(
            service="appconfig",
            resource="deploymentstrategy",
            resource_name=appconfig_strategy_id,
        )
        # ALLOW: exactly the five deploy actions, scoped to this application's
        # profile + environment + strategy ARNs (+ the application ARN for the
        # CreateHostedConfigurationVersion call form). NEVER a wildcard resource.
        self.deployer_role.add_to_policy(
            iam.PolicyStatement(
                sid="AppConfigDeploy",
                effect=iam.Effect.ALLOW,
                actions=list(_DEPLOYER_APPCONFIG_ACTIONS),
                resources=[app_arn, profile_arn, environment_arn, strategy_arn],
            )
        )
        # DENY: the two profile-mutation actions, so the deployer can never strip
        # the validator (ADR-0026's invariant). An explicit Deny beats any Allow,
        # so even a future broadening of the allow set cannot re-enable these.
        self.deployer_role.add_to_policy(
            iam.PolicyStatement(
                sid="DenyValidatorStrip",
                effect=iam.Effect.DENY,
                actions=list(_DEPLOYER_DENIED_ACTIONS),
                resources=[app_arn, profile_arn],
            )
        )

        # -- 3. The deployer CodeBuild project (INLINE buildspec, no Docker asset)
        # Runs the AppConfig deploy: create-hosted-configuration-version from the
        # committed config.yaml, then start-deployment via the construct's strategy.
        # The inline buildspec keeps synth byte-stable + host-Docker-independent
        # (RouteIQ-4772 discipline). The project's service role IS the narrow
        # deployer role above.
        self.deploy_project = codebuild.PipelineProject(
            self,
            "DeployProject",
            role=self.deployer_role,
            encryption_key=self.kms_key,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
            ),
            environment_variables={
                "APPCONFIG_APPLICATION_ID": codebuild.BuildEnvironmentVariable(
                    value=appconfig_application_id
                ),
                "APPCONFIG_ENVIRONMENT_ID": codebuild.BuildEnvironmentVariable(
                    value=appconfig_environment_id
                ),
                "APPCONFIG_PROFILE_ID": codebuild.BuildEnvironmentVariable(
                    value=appconfig_profile_id
                ),
                "APPCONFIG_STRATEGY_ID": codebuild.BuildEnvironmentVariable(
                    value=appconfig_strategy_id
                ),
            },
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "build": {
                            "commands": [
                                "set -euo pipefail",
                                # Push the new config version (AppConfig runs the
                                # LAMBDA validator before it becomes deployable).
                                (
                                    "VERSION=$(aws appconfig "
                                    "create-hosted-configuration-version "
                                    "--application-id $APPCONFIG_APPLICATION_ID "
                                    "--configuration-profile-id $APPCONFIG_PROFILE_ID "
                                    "--content-type application/x-yaml "
                                    "--content fileb://config.yaml "
                                    "--query VersionNumber --output text)"
                                ),
                                # Roll it out via the construct's linear strategy.
                                (
                                    "aws appconfig start-deployment "
                                    "--application-id $APPCONFIG_APPLICATION_ID "
                                    "--environment-id $APPCONFIG_ENVIRONMENT_ID "
                                    "--configuration-profile-id $APPCONFIG_PROFILE_ID "
                                    "--configuration-version $VERSION "
                                    "--deployment-strategy-id $APPCONFIG_STRATEGY_ID"
                                ),
                            ],
                        },
                    },
                }
            ),
        )

        # -- 4. The pipeline: Source -> [Approve, prod only] -> Deploy ---------
        source_output = codepipeline.Artifact("ConfigSource")
        source_action = cp_actions.S3SourceAction(
            action_name="Source",
            bucket=self.source_bucket,
            bucket_key=_SOURCE_CONFIG_KEY,
            output=source_output,
            trigger=cp_actions.S3Trigger.EVENTS,
        )
        deploy_action = cp_actions.CodeBuildAction(
            action_name="Deploy",
            project=self.deploy_project,
            input=source_output,
        )

        stages: list[codepipeline.StageProps] = [
            codepipeline.StageProps(stage_name="Source", actions=[source_action]),
        ]
        # [Approve, PROD ONLY]: a manual approval gate inserted only for prod so a
        # human must click before a prod config rolls. dev/stage skip it.
        self.approval_action: cp_actions.ManualApprovalAction | None = None
        if self._is_prod:
            self.approval_action = cp_actions.ManualApprovalAction(
                action_name="Approve",
                additional_information=(
                    "Approve the RouteIQ prod config deployment. The AppConfig "
                    "LAMBDA validator gates content; this gates intent."
                ),
            )
            stages.append(
                codepipeline.StageProps(stage_name="Approve", actions=[self.approval_action])
            )
        stages.append(codepipeline.StageProps(stage_name="Deploy", actions=[deploy_action]))

        self.pipeline = codepipeline.Pipeline(
            self,
            "ConfigPipeline",
            pipeline_name=f"routeiq-{env_name}-config-gitops",
            artifact_bucket=self.source_bucket,
            stages=stages,
        )
        Tags.of(self.pipeline).add("routeiq:env", env_name)

        # -- 5. Operator-visible outputs --------------------------------------
        CfnOutput(
            self,
            "ConfigSourceBucketName",
            value=self.source_bucket.bucket_name,
            description=(
                "GitOps config source bucket. Commit a zipped config.yaml as "
                f"{_SOURCE_CONFIG_KEY} to trigger the deploy pipeline."
            ),
        )
        CfnOutput(
            self,
            "ConfigPipelineName",
            value=self.pipeline.pipeline_name,
            description="GitOps CodePipeline (Source -> [Approve, prod] -> Deploy)",
        )
        CfnOutput(
            self,
            "DeployerRoleArn",
            value=self.deployer_role.role_arn,
            description=("Narrow AppConfig deployer role (5 actions + deny profile mutation)"),
        )

        self._suppress_nag()

    def _suppress_nag(self) -> None:
        """Inline cdk-nag suppressions for the GitOps pipeline surface.

        Every suppression is evidenced + ASCII + carries an Owner line, matching
        the ConfigStateConstruct / DataLakeConstruct inline-ownership split. The
        CodePipeline + CodeBuild + their auto-generated roles carry the canonical
        CDK findings (wildcard report-group ARNs, the pipeline's own KMS/S3 grants);
        the deployer role's AppConfig statement is ARN-scoped and needs none.
        """
        # IAM5 on the pipeline + the deploy project's CDK-managed policies. The
        # CodeBuild project's CloudWatch Logs + report-group permissions use the
        # ``*`` report-group / log-stream forms CDK emits (a build creates streams
        # under its own log group at runtime; there is no pre-known stream ARN).
        # The deployer role's AppConfig allow is ARN-scoped (never ``*``); these
        # wildcards are CDK-canonical build infra, not the deploy grant.
        NagSuppressions.add_resource_suppressions(
            self,
            [
                NagPackSuppression(
                    id="AwsSolutions-IAM5",
                    reason=(
                        "The CodePipeline + CodeBuild CDK-managed policies use "
                        "Resource=* for CloudWatch Logs stream creation and "
                        "CodeBuild report groups (no pre-known stream/report ARN "
                        "exists at synth) and for the pipeline's own artifact "
                        "bucket object operations under its KMS'd source bucket. "
                        "The NARROW DEPLOYER role's AppConfig statement is "
                        "ARN-scoped to this application's profile/environment/"
                        "strategy (never *) and explicitly DENIES "
                        "Update/DeleteConfigurationProfile. Owner: RouteIQ P2 "
                        "GitOpsPipelineConstruct."
                    ),
                    applies_to=["Resource::*"],
                ),
            ],
            apply_to_children=True,
        )


__all__ = [
    "GitOpsPipelineConstruct",
    "_DEPLOYER_APPCONFIG_ACTIONS",
    "_DEPLOYER_DENIED_ACTIONS",
]
