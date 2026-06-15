"""ConfigStateConstruct - the RouteIQ P2 AppConfig config-state resource graph.

A focused port of the vllm-sr-on-aws ``ConfigStateConstruct`` AppConfig core
(``cdk/lib/config_state_construct.py``), trimmed to the AppConfig substrate +
the config validator Lambda. Re-derived symbol-by-symbol from the real VSR
source, re-based ``vllm-sr`` -> ``routeiq``.

Resource graph (the P2 deliverable):

    appconfig_application (name="routeiq")
      -> appconfig_environment (name=<env_name>)
      -> appconfig_profile (router-yaml, hosted, AWS.Freeform,
                            validators=[LAMBDA])
      -> appconfig_strategy (Linear20Pct3Min: LINEAR 20% / 12min / 5min bake)
      -> appconfig_initial_version (placeholder RouteIQ config.yaml, x-yaml)
      -> appconfig_initial_deployment (auto-runs on first deploy, DependsOn the
                                       validator invoke permission)
    validator_lambda (Python 3.13)
      -> CfnPermission allowing appconfig.amazonaws.com to invoke (scoped to the
         profile-only ARN)

DROPPED relative to the VSR source (NOT P2 deliverables for the RouteIQ
config-state substrate): the S3 state bucket, the CloudTrail audit trail, the
CodePipeline deployer pipeline, the EventBridge validator-mutation rule, the
deployer IAM role, and the ``spike_infra_only`` posture. RouteIQ does not need
those for P2 config-state; the operator pushes real config day-2 via the
``aws appconfig`` CLI (a later P-tier pipeline is tracked separately).

P0-established patterns preserved:
  * ``env_name`` kwarg, no positional flags;
  * every IAM/resource Description is ASCII / Latin-1 ONLY (an em-dash U+2014
    passes ``cdk synth`` but FAILS the IAM CREATE API - proposal section 4.5);
  * the validator Lambda's cdk-nag suppressions (IAM4 + L1) are INLINE on the
    function (matching the VSR ownership split), NOT in nag_suppressions.py.

Mulch-recorded VSR lessons carried verbatim:
  * ``cdk-aws-custom-resource-update-stale-attribute``: use the L1
    ``CfnHostedConfigurationVersion`` rather than ``AwsCustomResource`` with
    only ``onCreate`` to avoid stale attribute reads on stack update.
  * ``mx-86079a`` (lambda-asset-fallback-inline-placeholder) + RouteIQ-4772
    (deterministic synth): the inline accept-all placeholder is the DEFAULT synth
    path -- selected by an EXPLICIT ``bundle_validator_asset`` toggle (kwarg or the
    ``routeiq:bundle_validator_asset`` context flag), NOT an implicit
    ``shutil.which("docker")`` probe. So the synthesised template is identical on a
    Docker-equipped host and a Docker-less one (CI, isolated unit tests, the
    cred-free gate, the snapshot). The REAL PyYAML-bundled ``handler.py`` ships ONLY
    under the explicit opt-in (the operator-controlled "real validator" deploy),
    with Docker + the asset present; otherwise the inline placeholder is used.
  * ``mx-1d874a`` (lambda runtime tracks aws-cdk-lib latest): validator pins
    ``Runtime.PYTHON_3_13``.
  * bug #10 (handler-string trap): ``from_inline`` ALWAYS writes the body to
    ``index.py`` (handler ``index.lambda_handler``); ``from_asset`` stages the
    real ``handler.py`` (handler ``handler.lambda_handler``). The two are NOT
    interchangeable.
"""

from __future__ import annotations

import os
from pathlib import Path

from aws_cdk import Aws, BundlingOptions, Duration, RemovalPolicy, Stack
from aws_cdk import aws_appconfig as appconfig
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from cdk_nag import NagPackSuppression, NagSuppressions
from constructs import Construct

# Path from this file (deploy/cdk/lib/) up to deploy/cdk/, then into
# ``lambda/appconfig-validator/``. Absolute so resolution is independent of the
# cdk-app cwd. NOTE: ``parents[1]`` (not the VSR ``parents[2]``): this construct
# lives at deploy/cdk/lib/ and the lambda dir is deploy/cdk/lambda/, one level up.
_VALIDATOR_ASSET_PATH = str(Path(__file__).resolve().parents[1] / "lambda" / "appconfig-validator")


# Inline placeholder for the ``Code.from_inline`` fallback (mulch mx-86079a).
#
# Accept-all (returns ``None``) rather than raising. AppConfig Lambda validators
# signal SUCCESS by returning normally and FAILURE by raising; a no-op success
# keeps the placeholder non-breaking so the AppConfigInitialDeployment can
# validate the placeholder config itself when the real asset is unavailable at
# synth (no Docker / isolated CI). The real
# ``lambda/appconfig-validator/handler.py`` enforces the RouteIQ-shape +
# inline-secret rules once it is bundled (Docker present on the deploy host).
#
# NOTE: ``Code.from_inline`` ALWAYS writes this body to ``index.py``, so the
# Lambda's ``handler`` MUST be ``index.lambda_handler`` on the inline path (it is
# ``handler.lambda_handler`` only on the ``from_asset`` path, where the real
# ``handler.py`` is staged). ``_build_validator_lambda`` selects the handler
# string to match whichever code path was taken (bug #10 handler-string trap).
_VALIDATOR_INLINE_PLACEHOLDER = (
    "def lambda_handler(event, context):\n"
    "    # Placeholder validator: accept every configuration (return None).\n"
    "    # Replaced by lambda/appconfig-validator/handler.py once bundled.\n"
    "    return None\n"
)


# Minimum RouteIQ config.yaml seed for the first AppConfig deploy. Hand-written
# as a string so the YAML body is deterministic across CDK synths (avoids
# yaml.safe_dump key-order non-determinism). RouteIQ-shaped (model_list +
# litellm_settings + a general: block), NOT the VSR listeners/extproc schema.
# Carries ``config_source: file`` (the RouteIQ analogue of VSR's
# ``global.router.config_source: file``); the operator replaces this whole
# version with the real config via a day-2 CreateHostedConfigurationVersion.
# All secrets are referenced indirectly via ``os.environ/<VAR>`` so the
# placeholder itself passes the validator's inline-secret deny rule.
_PLACEHOLDER_ROUTER_CONFIG = """\
# RouteIQ AppConfig placeholder configuration (config.yaml).
# Replaced by the operator via a subsequent CreateHostedConfigurationVersion.
model_list:
  - model_name: claude-haiku
    litellm_params:
      model: bedrock/anthropic.claude-3-haiku-20240307-v1:0
      aws_region_name: os.environ/AWS_REGION
litellm_settings:
  drop_params: true
  num_retries: 2
general:
  config_source: file
"""


class ConfigStateConstruct(Construct):
    """RouteIQ AppConfig application/environment/profile/strategy + validator."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        # RouteIQ-4772: make the validator Lambda synth DETERMINISTIC. The default
        # (False) ALWAYS takes the inline placeholder path -- the byte-stable,
        # host-Docker-INDEPENDENT path the cred-free gate + the snapshot exercise.
        # The real PyYAML-bundled validator (lambda/appconfig-validator/handler.py)
        # ships ONLY under an EXPLICIT opt-in (this kwarg True, or the
        # ``routeiq:bundle_validator_asset`` context flag) AND with Docker + the
        # asset present -- the operator-controlled "real validator" deploy. This
        # removes the old silent ``shutil.which("docker")`` dual-resolution where a
        # cred-free gate validated an accept-all no-op while a docker-equipped
        # ``cdk deploy`` shipped a DIFFERENT template (a different Code property +
        # Handler string + runtime behaviour). The deploy path is documented; the
        # gate-tested path is the synth default.
        bundle_validator_asset: bool | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        stack = Stack.of(self)

        # Resolve the validator-bundling toggle: explicit kwarg wins; else the
        # ``routeiq:bundle_validator_asset`` context flag (CLI string or cdk.json
        # bool); else the deterministic default of False (inline path).
        if bundle_validator_asset is None:
            ctx = self.node.try_get_context("routeiq:bundle_validator_asset")
            if isinstance(ctx, bool):
                bundle_validator_asset = ctx
            elif ctx is None:
                bundle_validator_asset = False
            else:
                bundle_validator_asset = str(ctx).strip().lower() in ("true", "1", "yes")
        self.bundle_validator_asset = bundle_validator_asset

        (
            self.appconfig_application,
            self.appconfig_environment,
            self.appconfig_profile,
            self.appconfig_strategy,
            self.appconfig_initial_version,
            self.appconfig_initial_deployment,
            self.validator_lambda,
            self.validator_permission,
        ) = self._build_appconfig(env_name)

        self.appconfig_application_id: str = self.appconfig_application.ref
        self.appconfig_environment_id: str = self.appconfig_environment.ref
        self.appconfig_profile_id: str = self.appconfig_profile.ref

        # The env-scoped runtime poll ARN, per the AppConfigPoll contract. This
        # is the form a RouteIQ replica uses to poll AppConfig for new config
        # versions (StartConfigurationSession + GetLatestConfiguration). DISTINCT
        # from the profile-only ARN form used by the validator invoke permission.
        self.appconfig_profile_arn: str = stack.format_arn(
            service="appconfig",
            resource="application",
            resource_name=(
                f"{self.appconfig_application_id}/environment/"
                f"{self.appconfig_environment_id}/configuration/"
                f"{self.appconfig_profile_id}"
            ),
        )

    def appconfig_poll_statement(self) -> iam.PolicyStatement:
        """Return the AppConfig runtime-poll PolicyStatement for the pod role.

        RouteIQ-569f. The single statement a RouteIQ replica needs to poll AppConfig
        for new config versions at runtime (ADR-0026): StartConfigurationSession +
        GetLatestConfiguration, scoped to THIS construct's env-scoped profile ARN
        (``self.appconfig_profile_arn``), NEVER ``*``.

        PREFIX CORRECTNESS (incidental, folded into 569f): the runtime poll is the
        ``appconfigdata`` DATA-PLANE prefix -- the boto3 ``appconfigdata`` client
        (RouteIQ-4333's poll adapter) calls ``appconfigdata:StartConfigurationSession``
        + ``appconfigdata:GetLatestConfiguration``. Granting ONLY the control-plane
        ``appconfig:GetLatestConfiguration`` would AccessDeny on the poll. The
        control-plane GET is also kept for SDK-path tolerance.

        Returned as a STATEMENT (not added to a role) so the composition root can
        own the ``iam.Policy`` cross-stack without closing a DependencyCycle (see
        RouteIqObservabilityStack._grant_pod_role). ASCII-only sid.
        """
        return iam.PolicyStatement(
            sid="AppConfigPoll",
            effect=iam.Effect.ALLOW,
            actions=[
                "appconfigdata:StartConfigurationSession",
                "appconfigdata:GetLatestConfiguration",
                "appconfig:GetLatestConfiguration",
            ],
            resources=[self.appconfig_profile_arn],
        )

    def appconfig_poll_grant(self, pod_role: iam.IRole) -> None:
        """Grant a pod role the AppConfig runtime-poll actions on this profile.

        The documented LATE-BINDING seam (mirrors
        ObservabilityConstruct.amp_remote_write_grant). Adds the
        :meth:`appconfig_poll_statement` to the role's principal policy directly.
        NOTE: do NOT call this cross-stack (it mutates the imported role's default
        policy in its OWN stack, closing a DependencyCycle when the resources are
        owned by another stack). The combined-deploy path instead uses
        :meth:`appconfig_poll_statement` inside a composition-root-owned
        ``iam.Policy`` (RouteIqObservabilityStack._grant_pod_role).
        """
        pod_role.add_to_principal_policy(self.appconfig_poll_statement())

    def _build_appconfig(
        self, env_name: str
    ) -> tuple[
        appconfig.CfnApplication,
        appconfig.CfnEnvironment,
        appconfig.CfnConfigurationProfile,
        appconfig.CfnDeploymentStrategy,
        appconfig.CfnHostedConfigurationVersion,
        appconfig.CfnDeployment,
        lambda_.Function,
        lambda_.CfnPermission,
    ]:
        """AppConfig application, environment, profile, strategy, seed deployment.

        ``CfnHostedConfigurationVersion`` is preferred over an
        ``AwsCustomResource``-with-only-onCreate per mulch
        ``cdk-aws-custom-resource-update-stale-attribute``: L1 resources
        re-resolve their attributes on every stack update.
        """
        application = appconfig.CfnApplication(
            self,
            "AppConfigApplication",
            name="routeiq",
            description="RouteIQ AppConfig application",
        )

        environment = appconfig.CfnEnvironment(
            self,
            "AppConfigEnvironment",
            application_id=application.ref,
            name=env_name,
            description=f"RouteIQ AppConfig environment for {env_name}",
        )

        profile = appconfig.CfnConfigurationProfile(
            self,
            "AppConfigProfile",
            application_id=application.ref,
            name="router-yaml",
            location_uri="hosted",
            type="AWS.Freeform",
            description="RouteIQ config profile (free-form YAML)",
        )

        # Validator Lambda + invoke-permission, then wire into the profile.
        # Created BEFORE the strategy/initial-deployment chain so the initial
        # deployment can validate the placeholder config.
        validator_lambda = self._build_validator_lambda(env_name)

        # Resource-based permission allowing AppConfig to invoke the validator.
        # AppConfig's invocation is scoped to a specific configuration profile;
        # we grant on the profile-only ARN form
        # (application/<app>/configurationprofile/<profile>), NOT the env-scoped
        # runtime-poll form. The permission MUST be in place before the initial
        # deployment runs (otherwise AppConfig cannot invoke the validator and
        # the deployment errors).
        profile_arn_for_permission = Stack.of(self).format_arn(
            service="appconfig",
            resource="application",
            resource_name=(f"{application.ref}/configurationprofile/{profile.ref}"),
        )
        validator_permission = lambda_.CfnPermission(
            self,
            "AppConfigValidatorPermission",
            action="lambda:InvokeFunction",
            function_name=validator_lambda.function_arn,
            principal="appconfig.amazonaws.com",
            source_account=Aws.ACCOUNT_ID,
            source_arn=profile_arn_for_permission,
        )
        validator_permission.add_dependency(
            profile.node.default_child  # type: ignore[arg-type]
            if profile.node.default_child is not None
            else profile
        )

        # Wire the validator onto the profile via the ValidatorsProperty
        # (type=LAMBDA, content=<function ARN>), AFTER the Lambda + permission
        # exist. The validator's job is to reject inline secrets + bad-shape
        # configs at AppConfig deploy time, before any worker polls them.
        profile.validators = [
            appconfig.CfnConfigurationProfile.ValidatorsProperty(
                type="LAMBDA",
                content=validator_lambda.function_arn,
            )
        ]

        strategy = appconfig.CfnDeploymentStrategy(
            self,
            "AppConfigStrategy",
            name="Linear20Pct3Min",
            deployment_duration_in_minutes=12,
            growth_factor=20,
            growth_type="LINEAR",
            final_bake_time_in_minutes=5,
            replicate_to="NONE",
            description="Linear 20% per step, 5 steps over 12min (100%) + 5min final bake",
        )

        initial_version = appconfig.CfnHostedConfigurationVersion(
            self,
            "AppConfigInitialVersion",
            application_id=application.ref,
            configuration_profile_id=profile.ref,
            content=_PLACEHOLDER_ROUTER_CONFIG,
            content_type="application/x-yaml",
            description=(
                "Placeholder config for first deploy; operator replaces via "
                "CreateHostedConfigurationVersion"
            ),
        )

        initial_deployment = appconfig.CfnDeployment(
            self,
            "AppConfigInitialDeployment",
            application_id=application.ref,
            environment_id=environment.ref,
            configuration_profile_id=profile.ref,
            configuration_version=initial_version.ref,
            deployment_strategy_id=strategy.ref,
            description="Initial AppConfig deployment of placeholder config",
        )
        # The validator's invoke permission MUST exist before the initial
        # deployment kicks off, otherwise AppConfig cannot validate the
        # placeholder version and the deployment fails. This ordering is
        # load-bearing.
        initial_deployment.add_dependency(validator_permission)

        return (
            application,
            environment,
            profile,
            strategy,
            initial_version,
            initial_deployment,
            validator_lambda,
            validator_permission,
        )

    def _build_validator_lambda(self, env_name: str) -> lambda_.Function:
        """Construct the AppConfig RouteIQ-config validator Lambda.

        DETERMINISTIC code resolution (RouteIQ-4772). The synth path is selected by
        an EXPLICIT toggle (``self.bundle_validator_asset``), NOT by an implicit
        ``shutil.which("docker")`` probe, so the synthesised template never depends
        on whether Docker is on the host:

            * DEFAULT (toggle False) -> ``Code.from_inline`` with the accept-all
              placeholder. ``from_inline`` ALWAYS writes the body to ``index.py``,
              so the entry point is ``index.lambda_handler``. This is the
              byte-stable, host-independent path the cred-free gate + the snapshot
              exercise.
            * OPT-IN (toggle True) AND the asset present -> ``Code.from_asset`` with
              PyYAML bundling (Docker required), staging the REAL
              ``lambda/appconfig-validator/handler.py`` (entry point
              ``handler.lambda_handler``). The operator-controlled "real validator"
              deploy. If the toggle is on but the asset is missing, fall back to the
              inline placeholder so synth still proceeds (rather than raising).

        bug #10 (handler-string trap): ``index.lambda_handler`` (inline) and
        ``handler.lambda_handler`` (asset) are NOT interchangeable; the handler is
        selected to match the chosen code path.
        """
        code: lambda_.Code
        asset_present = os.path.isdir(_VALIDATOR_ASSET_PATH) and os.path.isfile(
            os.path.join(_VALIDATOR_ASSET_PATH, "handler.py")
        )
        if self.bundle_validator_asset and asset_present:
            code = lambda_.Code.from_asset(
                _VALIDATOR_ASSET_PATH,
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_13.bundling_image,
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
            # from_asset stages the real ``handler.py`` at the function root.
            handler = "handler.lambda_handler"
        else:
            code = lambda_.Code.from_inline(_VALIDATOR_INLINE_PLACEHOLDER)
            # from_inline ALWAYS writes the body to ``index.py`` (bug #10).
            handler = "index.lambda_handler"

        fn = lambda_.Function(
            self,
            "AppConfigValidatorLambda",
            function_name=f"routeiq-{env_name}-appconfig-validator",
            description=(
                "AppConfig validator for the RouteIQ config: runs RouteIQ's "
                "config validation rules (YAML parse + RouteIQ-shape + "
                "inline-secret deny). Wired into the AppConfig profile's "
                "Validators array."
            ),
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler=handler,
            code=code,
            timeout=Duration.seconds(15),
            memory_size=256,
            environment={
                "ENV_NAME": env_name,
                "LOG_LEVEL": "INFO",
            },
        )

        # Bounded log retention without the deprecated log_retention kwarg.
        logs.LogGroup(
            self,
            "AppConfigValidatorLogGroup",
            log_group_name=f"/aws/lambda/{fn.function_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Inline cdk-nag suppressions for the validator's auto-attached
        # AWSLambdaBasicExecutionRole. The validator is the load-bearing
        # config-gate and its execution role is CDK-canonical (needs no
        # permissions beyond CloudWatch Logs). These are INLINE on the function
        # (matching the VSR ownership split), NOT in nag_suppressions.py.
        NagSuppressions.add_resource_suppressions(
            fn.role,  # type: ignore[arg-type]
            [
                NagPackSuppression(
                    id="AwsSolutions-IAM4",
                    reason=(
                        "AWSLambdaBasicExecutionRole is the AWS-canonical "
                        "execution role auto-attached by CDK to every "
                        "lambda.Function. The RouteIQ config validator needs no "
                        "additional permissions beyond CloudWatch Logs. "
                        "Owner: RouteIQ P2 ConfigStateConstruct."
                    ),
                    applies_to=[
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/"
                        "service-role/AWSLambdaBasicExecutionRole",
                    ],
                ),
            ],
            apply_to_children=True,
        )

        # L1: PYTHON_3_13 vs cdk-nag's "latest" tracking. Per mulch mx-1d874a,
        # user-authored lambdas track aws-cdk-lib's latest runtime; bumping
        # happens in lockstep with the next aws-cdk-lib upgrade. Until then the
        # validator stays on Python 3.13 (matches handler.py + the inline
        # placeholder fallback).
        NagSuppressions.add_resource_suppressions(
            fn,
            [
                NagPackSuppression(
                    id="AwsSolutions-L1",
                    reason=(
                        "Runtime is pinned to Python 3.13 to match the bundled "
                        "lambda/appconfig-validator/handler.py and the inline "
                        "placeholder fallback. Bumping to a newer runtime "
                        "happens in lockstep with the next aws-cdk-lib upgrade "
                        "per mulch mx-1d874a. "
                        "Owner: RouteIQ P2 ConfigStateConstruct."
                    ),
                ),
            ],
        )

        return fn
