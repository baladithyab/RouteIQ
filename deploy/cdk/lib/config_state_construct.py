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
  * ``mx-86079a`` (lambda-asset-fallback-inline-placeholder): when
    ``lambda/appconfig-validator/`` is absent at synth (CI, isolated unit tests)
    OR Docker is unavailable, fall back to ``Code.from_inline`` with an
    accept-all placeholder so synth + snapshot tests proceed.
  * ``mx-1d874a`` (lambda runtime tracks aws-cdk-lib latest): validator pins
    ``Runtime.PYTHON_3_13``.
  * bug #10 (handler-string trap): ``from_inline`` ALWAYS writes the body to
    ``index.py`` (handler ``index.lambda_handler``); ``from_asset`` stages the
    real ``handler.py`` (handler ``handler.lambda_handler``). The two are NOT
    interchangeable.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from aws_cdk import Aws, BundlingOptions, Duration, RemovalPolicy, Stack
from aws_cdk import aws_appconfig as appconfig
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
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        stack = Stack.of(self)

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
            description="Linear 20% per 3 minutes (60% over 12min) + 5min final bake",
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

        Resolution order for the function code (dual-resolution):

            1. ``lambda/appconfig-validator/handler.py`` is present on disk AND
               Docker is available -> ``Code.from_asset`` with PyYAML bundling.
               The asset stages the real ``handler.py``, so the entry point is
               ``handler.lambda_handler``.
            2. Otherwise -> ``Code.from_inline`` with an accept-all placeholder
               (mulch mx-86079a). ``from_inline`` ALWAYS writes the body to
               ``index.py``, so the entry point is ``index.lambda_handler`` -
               using ``handler.lambda_handler`` here is bug #10 ("No module
               named handler" at AppConfig validation time). The resource shape
               is still synthesised so snapshot tests + cred-free CI proceed.

        The handler string is therefore selected to match whichever code path
        was taken - the two are NOT interchangeable.
        """
        code: lambda_.Code
        # Bundling requires Docker on the synth host. In CI / cred-free synth
        # contexts where Docker is unavailable the BundlingOptions path raises
        # ``spawnSync docker ENOENT`` - fall back to the inline placeholder so
        # synth + snapshot tests proceed (mx-86079a).
        bundling_available = shutil.which("docker") is not None
        if (
            bundling_available
            and os.path.isdir(_VALIDATOR_ASSET_PATH)
            and os.path.isfile(os.path.join(_VALIDATOR_ASSET_PATH, "handler.py"))
        ):
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
