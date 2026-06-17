"""SageMakerRetrainingConstruct - scheduled ML model retraining (RouteIQ-8a24).

A flag-gated, DEFAULT-OFF construct that authors the AWS-native side of an
automated/scheduled retraining pipeline for RouteIQ's routing ML models:

    EventBridge schedule (rate/cron)
      -> EventBridge rule target = a SageMaker CreateTrainingJob (via an
         AWS API call target, executed by a narrow invoker role)
      -> S3 artifact bucket (the trained model artifact landing zone)

The in-process side of the retraining loop is the EXISTING model-artifact loader
(``model_artifacts.py``) which verifies + hot-loads a new artifact; this
construct only provisions the recurring trigger + the artifact bucket the
training job writes to. The LIVE training (a real SageMaker Training Job with a
real algorithm image + dataset) is operator-gated -- this construct authors the
SCHEDULE + the BUCKET + the narrow IAM, cred-free and byte-stable when off.

What this construct AUTHORS (cred-free, byte-stable when the flag is off):

  * Artifact bucket (versioned, KMS-SSE, BPA, enforce_ssl) - the trained model
    artifact landing zone the SageMaker job's ``OutputDataConfig`` writes to and
    the model-artifact loader reads from.
  * SageMaker execution role - the role the training job ASSUMES (sagemaker
    service principal); scoped to read the dataset + write the artifact bucket.
  * EventBridge invoker role - the role EventBridge assumes to call
    ``sagemaker:CreateTrainingJob`` (and ``iam:PassRole`` of the execution role,
    scoped to exactly that role).
  * EventBridge rule on the schedule (rate/cron) with a SageMaker training-job
    target (AwsApi target) carrying the job request as input.

FLAG-GATED, DEFAULT OFF (wired by the composition root only when
``enable_sagemaker_retraining=True``): a default synth emits ZERO
``AWS::Events::Rule`` / ``AWS::SageMaker`` / artifact-bucket resources so the
snapshot stays byte-stable.

DETERMINISTIC SYNTH (no Docker asset, no from_lookup): every ARN is
``stack.format_arn`` / a token, so the synth is credential-free.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from cdk_nag import NagSuppressions
from constructs import Construct

# Default recurring cadence: weekly retrain (operators override via schedule_*).
_DEFAULT_SCHEDULE_RATE_DAYS = 7

# The exact SageMaker actions the EventBridge invoker may call (never wildcard).
_INVOKER_SAGEMAKER_ACTIONS = ["sagemaker:CreateTrainingJob"]


class SageMakerRetrainingConstruct(Construct):
    """Scheduled SageMaker retraining trigger + artifact bucket (flag-gated).

    Public attributes:
      ``artifact_bucket`` - the trained-model artifact landing-zone bucket.
      ``execution_role``  - the SageMaker training-job execution role.
      ``invoker_role``    - the EventBridge -> SageMaker invoker role.
      ``schedule_rule``   - the EventBridge rule firing the retrain.
      ``kms_key``         - the CMK encrypting the artifact bucket.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        schedule_rate_days: int = _DEFAULT_SCHEDULE_RATE_DAYS,
        schedule_expression: str | None = None,
        training_image_uri: str | None = None,
        dataset_bucket_arn: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        stack = Stack.of(self)
        _retain = env_name != "dev"
        Tags.of(self).add("routeiq:env", env_name)

        # -- CMK for the artifact bucket (rotated, RETAIN in non-dev). ASCII desc.
        self.kms_key = kms.Key(
            self,
            "RetrainKey",
            description=f"RouteIQ {env_name} SageMaker retraining artifact CMK",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN if _retain else RemovalPolicy.DESTROY,
        )

        # -- 1. Artifact bucket (the model artifact landing zone) -------------
        self.artifact_bucket = s3.Bucket(
            self,
            "ArtifactBucket",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.kms_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN if _retain else RemovalPolicy.DESTROY,
            auto_delete_objects=not _retain,
        )

        # -- 2. SageMaker training-job execution role -------------------------
        # The role the training job assumes; it reads the dataset + writes the
        # artifact bucket. ASCII description (em-dash fails the IAM CREATE API).
        self.execution_role = iam.Role(
            self,
            "ExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description=(
                f"RouteIQ {env_name} SageMaker retraining job execution role "
                "(reads dataset, writes artifact bucket)"
            ),
        )
        self.artifact_bucket.grant_read_write(self.execution_role)
        self.kms_key.grant_encrypt_decrypt(self.execution_role)
        if dataset_bucket_arn:
            self.execution_role.add_to_principal_policy(
                iam.PolicyStatement(
                    sid="ReadDataset",
                    effect=iam.Effect.ALLOW,
                    actions=["s3:GetObject", "s3:ListBucket"],
                    resources=[dataset_bucket_arn, f"{dataset_bucket_arn}/*"],
                )
            )

        # -- 3. EventBridge invoker role --------------------------------------
        # The role EventBridge assumes to call CreateTrainingJob + PassRole the
        # execution role (scoped to EXACTLY that role -- never a wildcard).
        self.invoker_role = iam.Role(
            self,
            "InvokerRole",
            assumed_by=iam.ServicePrincipal("events.amazonaws.com"),
            description=(f"RouteIQ {env_name} EventBridge -> SageMaker retraining invoker"),
        )
        self.invoker_role.add_to_principal_policy(
            iam.PolicyStatement(
                sid="CreateTrainingJob",
                effect=iam.Effect.ALLOW,
                actions=_INVOKER_SAGEMAKER_ACTIONS,
                # SageMaker training-job ARNs are minted at create time; scope to
                # this account+region's training-job namespace (never global ``*``).
                resources=[
                    stack.format_arn(
                        service="sagemaker",
                        resource="training-job",
                        resource_name="routeiq-retrain-*",
                    )
                ],
            )
        )
        self.invoker_role.add_to_principal_policy(
            iam.PolicyStatement(
                sid="PassExecutionRole",
                effect=iam.Effect.ALLOW,
                actions=["iam:PassRole"],
                resources=[self.execution_role.role_arn],
                conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
            )
        )

        # -- 4. EventBridge schedule rule + SageMaker target ------------------
        if schedule_expression:
            schedule = events.Schedule.expression(schedule_expression)
        else:
            schedule = events.Schedule.rate(Duration.days(schedule_rate_days))

        self.schedule_rule = events.Rule(
            self,
            "RetrainSchedule",
            schedule=schedule,
            description=(f"RouteIQ {env_name} scheduled routing-model retraining trigger"),
        )
        # The AwsApi target calls sagemaker:CreateTrainingJob on the schedule.
        # The job request shape (algorithm image, input/output data, resource
        # config) is operator-supplied via the training_image_uri + the artifact
        # bucket; we wire the minimal deterministic skeleton so the rule has a
        # concrete target (the live job spec is operator-tuned).
        self.schedule_rule.add_target(
            targets.AwsApi(
                service="SageMaker",
                action="createTrainingJob",
                parameters={
                    "TrainingJobName": f"routeiq-retrain-{env_name}",
                    "AlgorithmSpecification": {
                        "TrainingImage": training_image_uri or "OPERATOR_SET_TRAINING_IMAGE_URI",
                        "TrainingInputMode": "File",
                    },
                    "RoleArn": self.execution_role.role_arn,
                    "OutputDataConfig": {
                        "S3OutputPath": self.artifact_bucket.s3_url_for_object("models/")
                    },
                    "ResourceConfig": {
                        "InstanceType": "ml.m5.xlarge",
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 30,
                    },
                    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
                },
                policy_statement=iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=_INVOKER_SAGEMAKER_ACTIONS,
                    resources=[
                        stack.format_arn(
                            service="sagemaker",
                            resource="training-job",
                            resource_name="routeiq-retrain-*",
                        )
                    ],
                ),
            )
        )

        # -- 5. Operator outputs ----------------------------------------------
        CfnOutput(
            self,
            "RetrainArtifactBucketName",
            value=self.artifact_bucket.bucket_name,
            description="Trained model artifact landing-zone bucket.",
        )
        CfnOutput(
            self,
            "RetrainExecutionRoleArn",
            value=self.execution_role.role_arn,
            description="SageMaker retraining job execution role.",
        )

        self._apply_nag_suppressions()

    def _apply_nag_suppressions(self) -> None:
        """Suppress the CDK-managed wildcard IAM5 findings on the grants.

        ``grant_read_write`` / ``grant_encrypt_decrypt`` emit object-ARN
        ``/*`` wildcards (the standard CDK grant pattern); the artifact-bucket
        read/write is inherently object-level so the ``/*`` is correct + scoped
        to THIS bucket (never a bare ``*``).
        """
        NagSuppressions.add_resource_suppressions(
            self.execution_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "Object-level (/*) wildcards on the artifact bucket + KMS "
                        "grant are scoped to THIS construct's bucket/key, never a "
                        "bare *; required for the training job to write artifacts."
                    ),
                }
            ],
            apply_to_children=True,
        )


__all__ = ["SageMakerRetrainingConstruct"]
