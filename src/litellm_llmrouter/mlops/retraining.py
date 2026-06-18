"""
SageMaker Scheduled-Retraining Orchestration Adapter (RouteIQ-8a24)
===================================================================

Today routing-model training is a manual MLflow example script -- there is NO
scheduled retraining.  This module adds the CONTROL-PLANE orchestration that
turns that manual step into an operator-gated, scheduled loop:

    EventBridge (cron/rate) -> start SageMaker Training Job / Pipeline
      -> poll status -> on success hand the produced S3 model artifact to the
         EXISTING SageMaker Model Registry adapter
         (``sagemaker_registry.register_model_package``)

so the routing-model lifecycle (train -> register -> approve -> promote) stays
ONE lifecycle.  The registered package then flows through the existing
approve/promote path and the model-artifact init/sidecar reads the S3 artifact.

Cred-free + mockable by construction (mirrors ``sagemaker_registry.py``): the
adapter takes INJECTED boto3 ``sagemaker`` + ``events`` clients
(``RetrainingPipelineAdapter(sagemaker_client=mock, events_client=mock)``).
Unit tests pass ``MagicMock``s and never touch AWS, never construct a real
client, never need credentials.  The lazy ``_sagemaker_client()`` /
``_events_client()`` only build a real ``boto3.client(...)`` when no client was
injected AND the adapter is enabled (live path), which is ``# pragma: no cover``.

Settings-gated under ``settings.mlops.retraining`` and DEFAULT OFF: with
``enabled=False`` every method short-circuits to a safe non-run result carrying a
``reason`` and never constructs a client.  Live SageMaker / EventBridge calls are
therefore operator-gated.

The control-plane operations used (all cred-free-mockable):
  * ``create_training_job`` (training-job mode) OR ``start_pipeline_execution``
    (pipeline mode) -- kicks off a retraining run that writes the model artifact
    to the configured S3 ``s3://<bucket>/<prefix>/...`` location.
  * ``describe_training_job`` / ``describe_pipeline_execution`` -- polls status.
  * ``events.put_rule`` + ``events.put_targets`` -- (re)creates the schedule rule
    that triggers the retraining run on the configured cron/rate expression.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "RetrainingPipelineAdapter",
    "RetrainingRunResult",
    "RetrainingStatusResult",
    "ScheduleResult",
    "RetrainRegisterResult",
    "get_retraining_adapter",
    "reset_retraining_adapter",
]

# SageMaker training-job terminal/active statuses.
_TRAINING_SUCCESS = "Completed"
_TRAINING_TERMINAL = {"Completed", "Failed", "Stopped"}
# SageMaker pipeline-execution terminal/active statuses.
_PIPELINE_SUCCESS = "Succeeded"
_PIPELINE_TERMINAL = {"Succeeded", "Failed", "Stopped"}


@dataclass
class RetrainingRunResult:
    """Outcome of starting a retraining run (training job or pipeline)."""

    started: bool = False
    mode: str = ""
    """``training_job`` or ``pipeline``."""

    job_name: str = ""
    """Training job name (training-job mode)."""

    job_arn: str = ""
    """Training job / pipeline-execution ARN."""

    pipeline_execution_arn: str = ""
    """Pipeline execution ARN (pipeline mode)."""

    output_s3_uri: str = ""
    """Where the run is configured to write the model artifact."""

    reason: str = ""


@dataclass
class RetrainingStatusResult:
    """Outcome of polling a retraining run's status."""

    polled: bool = False
    mode: str = ""
    status: str = ""
    """Raw SageMaker status string (e.g. ``InProgress``/``Completed``)."""

    succeeded: bool = False
    terminal: bool = False
    model_artifact_uri: str = ""
    """S3 URI of the produced model artifact (training-job mode, on success)."""

    reason: str = ""


@dataclass
class ScheduleResult:
    """Outcome of (re)creating the EventBridge retraining schedule rule."""

    scheduled: bool = False
    rule_name: str = ""
    schedule_expression: str = ""
    rule_arn: str = ""
    target_count: int = 0
    reason: str = ""


@dataclass
class RetrainRegisterResult:
    """Outcome of handing a successful run's artifact to the registry adapter."""

    registered: bool = False
    model_package_arn: str = ""
    model_artifact_uri: str = ""
    approval_status: str = ""
    reason: str = ""
    registry_reason: str = ""
    """The registry adapter's own reason string (passthrough on failure)."""


class RetrainingPipelineAdapter:
    """SageMaker scheduled-retraining orchestration adapter (RouteIQ-8a24).

    Control-plane only: starts a SageMaker Training Job / Pipeline execution for
    a routing model, polls its status, and on success hands the produced S3
    artifact to the EXISTING registry adapter so train -> register -> approve ->
    promote stays one lifecycle.  Also builds (and, on the live path, can put) an
    EventBridge schedule rule that triggers the retraining run on a cron/rate
    expression.

    Args:
        mode: ``training_job`` (``create_training_job``) or ``pipeline``
            (``start_pipeline_execution``).
        region: AWS region for the lazily-built boto3 clients (live path only).
        training_image: ECR image URI for the training container (training-job
            mode).
        role_arn: IAM role ARN SageMaker assumes for the run.
        instance_type / instance_count / volume_size_gb: training resource config.
        s3_artifact_bucket / s3_artifact_prefix: where the run writes the model
            artifact (the same location the model-artifact init/sidecar reads).
        pipeline_name: SageMaker Pipeline name (pipeline mode).
        schedule_expression: EventBridge cron/rate expression for the schedule.
        schedule_rule_name: EventBridge rule name.
        enabled: Whether live SageMaker / EventBridge calls are permitted.
        sagemaker_client: Injected boto3 sagemaker client (cred-free tests pass a
            mock); when None a real client is built lazily on first live use.
        events_client: Injected boto3 events client (same contract).
    """

    def __init__(
        self,
        *,
        mode: str = "training_job",
        region: str = "",
        training_image: str = "",
        role_arn: str = "",
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        volume_size_gb: int = 30,
        max_runtime_seconds: int = 86400,
        s3_artifact_bucket: str = "",
        s3_artifact_prefix: str = "routeiq/routing-models",
        s3_input_uri: str = "",
        pipeline_name: str = "routeiq-routing-retrain",
        schedule_expression: str = "rate(7 days)",
        schedule_rule_name: str = "routeiq-routing-retrain",
        job_name_prefix: str = "routeiq-routing-retrain",
        enabled: bool = True,
        sagemaker_client: Any = None,
        events_client: Any = None,
    ) -> None:
        self.mode = mode
        self.region = region
        self.training_image = training_image
        self.role_arn = role_arn
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.volume_size_gb = volume_size_gb
        self.max_runtime_seconds = max_runtime_seconds
        self.s3_artifact_bucket = s3_artifact_bucket
        self.s3_artifact_prefix = s3_artifact_prefix.strip("/")
        self.s3_input_uri = s3_input_uri
        self.pipeline_name = pipeline_name
        self.schedule_expression = schedule_expression
        self.schedule_rule_name = schedule_rule_name
        self.job_name_prefix = job_name_prefix
        self.enabled = enabled
        self._sm_client = sagemaker_client
        self._ev_client = events_client

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------

    def _sagemaker_client(self) -> Optional[Any]:
        """Return the injected sagemaker client, or lazily build a real one.

        Cred-free: tests inject a mock so this never builds a real client. The
        real-client branch is only reached when no client was injected -- it is
        ``# pragma: no cover`` because exercising it would require boto3 + AWS.
        """
        if self._sm_client is not None:
            return self._sm_client
        try:  # pragma: no cover - live path requires boto3 + AWS creds
            import boto3

            kwargs: Dict[str, Any] = {}
            if self.region:
                kwargs["region_name"] = self.region
            self._sm_client = boto3.client("sagemaker", **kwargs)
            return self._sm_client
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("SageMaker client construction failed: %s", e)
            return None

    def _events_client(self) -> Optional[Any]:
        """Return the injected events client, or lazily build a real one.

        Same cred-free contract as :meth:`_sagemaker_client`.
        """
        if self._ev_client is not None:
            return self._ev_client
        try:  # pragma: no cover - live path requires boto3 + AWS creds
            import boto3

            kwargs: Dict[str, Any] = {}
            if self.region:
                kwargs["region_name"] = self.region
            self._ev_client = boto3.client("events", **kwargs)
            return self._ev_client
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("EventBridge client construction failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _output_s3_uri(self) -> str:
        """The ``s3://bucket/prefix`` the run writes the model artifact to."""
        if not self.s3_artifact_bucket:
            return ""
        prefix = self.s3_artifact_prefix
        return (
            f"s3://{self.s3_artifact_bucket}/{prefix}"
            if prefix
            else (f"s3://{self.s3_artifact_bucket}")
        )

    def _new_job_name(self) -> str:
        return f"{self.job_name_prefix}-{int(time.time())}"

    # ------------------------------------------------------------------
    # Start a retraining run
    # ------------------------------------------------------------------

    def start_retraining(
        self,
        *,
        job_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        pipeline_parameters: Optional[Dict[str, str]] = None,
    ) -> RetrainingRunResult:
        """Start a SageMaker Training Job (or Pipeline execution).

        In ``training_job`` mode issues ``create_training_job`` with the
        configured image/role/instance and an S3 output path under
        ``s3_artifact_bucket/s3_artifact_prefix``.  In ``pipeline`` mode issues
        ``start_pipeline_execution`` on ``pipeline_name``.

        Returns a :class:`RetrainingRunResult`.  A safe, non-started result
        (never raises) is returned when the adapter is disabled or the client is
        unavailable; on a live error the reason carries the error string.
        """
        result = RetrainingRunResult(
            mode=self.mode, output_s3_uri=self._output_s3_uri()
        )
        if not self.enabled:
            result.reason = "disabled"
            return result

        client = self._sagemaker_client()
        if client is None:
            result.reason = "no_client"
            return result

        try:
            if self.mode == "pipeline":
                return self._start_pipeline(client, result, pipeline_parameters)
            return self._start_training_job(client, result, job_name, hyperparameters)
        except Exception as e:
            result.started = False
            result.reason = str(e)
            logger.warning("SageMaker retraining start failed: %s", e)
            return result

    def _start_training_job(
        self,
        client: Any,
        result: RetrainingRunResult,
        job_name: Optional[str],
        hyperparameters: Optional[Dict[str, Any]],
    ) -> RetrainingRunResult:
        name = job_name or self._new_job_name()
        algorithm: Dict[str, Any] = {"TrainingInputMode": "File"}
        if self.training_image:
            algorithm["TrainingImage"] = self.training_image

        kwargs: Dict[str, Any] = {
            "TrainingJobName": name,
            "AlgorithmSpecification": algorithm,
            "OutputDataConfig": {"S3OutputPath": self._output_s3_uri()},
            "ResourceConfig": {
                "InstanceType": self.instance_type,
                "InstanceCount": int(self.instance_count),
                "VolumeSizeInGB": int(self.volume_size_gb),
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": int(self.max_runtime_seconds)},
            "HyperParameters": {
                str(k): str(v) for k, v in (hyperparameters or {}).items()
            },
            "Tags": [{"Key": "routeiq:role", "Value": "routing-model-retrain"}],
        }
        if self.role_arn:
            kwargs["RoleArn"] = self.role_arn
        if self.s3_input_uri:
            kwargs["InputDataConfig"] = [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": self.s3_input_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ]

        response = client.create_training_job(**kwargs)
        result.started = True
        result.job_name = name
        result.job_arn = (
            response.get("TrainingJobArn", "") if isinstance(response, dict) else ""
        )
        logger.info("Started SageMaker training job %s", name)
        return result

    def _start_pipeline(
        self,
        client: Any,
        result: RetrainingRunResult,
        pipeline_parameters: Optional[Dict[str, str]],
    ) -> RetrainingRunResult:
        params = [
            {"Name": str(k), "Value": str(v)}
            for k, v in (pipeline_parameters or {}).items()
        ]
        kwargs: Dict[str, Any] = {"PipelineName": self.pipeline_name}
        if params:
            kwargs["PipelineParameters"] = params
        response = client.start_pipeline_execution(**kwargs)
        result.started = True
        result.pipeline_execution_arn = (
            response.get("PipelineExecutionArn", "")
            if isinstance(response, dict)
            else ""
        )
        result.job_arn = result.pipeline_execution_arn
        logger.info("Started SageMaker pipeline execution for %s", self.pipeline_name)
        return result

    # ------------------------------------------------------------------
    # Poll status
    # ------------------------------------------------------------------

    def get_status(
        self,
        *,
        job_name: Optional[str] = None,
        pipeline_execution_arn: Optional[str] = None,
    ) -> RetrainingStatusResult:
        """Describe/poll a retraining run's status.

        In ``training_job`` mode calls ``describe_training_job`` and, on
        ``Completed``, extracts the produced model artifact's S3 URI from
        ``ModelArtifacts.S3ModelArtifacts``.  In ``pipeline`` mode calls
        ``describe_pipeline_execution``.  Safe no-op (never raises) when disabled
        or the client is unavailable.
        """
        result = RetrainingStatusResult(mode=self.mode)
        if not self.enabled:
            result.reason = "disabled"
            return result

        client = self._sagemaker_client()
        if client is None:
            result.reason = "no_client"
            return result

        try:
            if self.mode == "pipeline":
                arn = pipeline_execution_arn or ""
                response = client.describe_pipeline_execution(PipelineExecutionArn=arn)
                status = (
                    response.get("PipelineExecutionStatus", "")
                    if isinstance(response, dict)
                    else ""
                )
                result.status = status
                result.succeeded = status == _PIPELINE_SUCCESS
                result.terminal = status in _PIPELINE_TERMINAL
            else:
                name = job_name or ""
                response = client.describe_training_job(TrainingJobName=name)
                status = (
                    response.get("TrainingJobStatus", "")
                    if isinstance(response, dict)
                    else ""
                )
                result.status = status
                result.succeeded = status == _TRAINING_SUCCESS
                result.terminal = status in _TRAINING_TERMINAL
                if result.succeeded and isinstance(response, dict):
                    artifacts = response.get("ModelArtifacts") or {}
                    result.model_artifact_uri = str(
                        artifacts.get("S3ModelArtifacts", "")
                    )
            result.polled = True
        except Exception as e:
            result.polled = False
            result.reason = str(e)
            logger.warning("SageMaker retraining status poll failed: %s", e)
        return result

    # ------------------------------------------------------------------
    # Hand the artifact to the registry adapter (one lifecycle)
    # ------------------------------------------------------------------

    def register_on_success(
        self,
        *,
        status: RetrainingStatusResult,
        registry_adapter: Any = None,
        approval_status: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        framework: str = "",
    ) -> RetrainRegisterResult:
        """Hand a successful run's S3 artifact to the registry adapter.

        Keeps train -> register -> approve -> promote ONE lifecycle by calling
        the EXISTING ``sagemaker_registry.register_model_package`` with the
        produced ``model_artifact_uri``.  ``registry_adapter`` is injectable
        (tests pass a mock); when None the module's settings-gated registry
        singleton is used.

        Safe no-op (never raises) when disabled, when the run did not succeed, or
        when no registry adapter is available.
        """
        result = RetrainRegisterResult(
            approval_status=approval_status or "",
            model_artifact_uri=status.model_artifact_uri,
        )
        if not self.enabled:
            result.reason = "disabled"
            return result
        if not status.succeeded:
            result.reason = "run_not_succeeded"
            return result
        if not status.model_artifact_uri:
            result.reason = "no_artifact"
            return result

        adapter = registry_adapter
        if adapter is None:
            try:
                from litellm_llmrouter.mlops.sagemaker_registry import (
                    get_sagemaker_registry_adapter,
                )

                adapter = get_sagemaker_registry_adapter()
            except Exception as e:  # pragma: no cover - defensive
                result.reason = f"registry_import_failed: {e}"
                return result
        if adapter is None:
            # RouteIQ-fe8e: a SUCCESSFUL retrain produced an artifact but the
            # SageMaker Model Registry adapter is disabled, so the train ->
            # register -> approve -> promote lifecycle silently stops here.
            # Surface a clear WARNING (this used to be a silent no-op) so the
            # operator knows the produced artifact at ``model_artifact_uri`` was
            # NOT registered and must be enabled / registered out of band. The
            # ``no_registry_adapter`` reason already carries the no-registry
            # signal on the result; ``registry_reason`` mirrors it for callers
            # that inspect the registry-side reason field.
            result.reason = "no_registry_adapter"
            result.registry_reason = "no_registry"
            logger.warning(
                "Retrain succeeded (artifact=%s) but the SageMaker Model Registry "
                "adapter is disabled; the model package was NOT registered. Enable "
                "settings.mlops.sagemaker to close the train->register->promote "
                "lifecycle.",
                status.model_artifact_uri,
            )
            return result

        artifact: Dict[str, Any] = {
            "model_data_url": status.model_artifact_uri,
            "description": "RouteIQ routing model (scheduled retrain)",
        }
        if self.training_image:
            artifact["image"] = self.training_image
        if framework:
            artifact["framework"] = framework

        reg = adapter.register_model_package(
            artifact=artifact,
            approval_status=approval_status,
            metadata=metadata,
        )
        result.registered = bool(getattr(reg, "registered", False))
        result.model_package_arn = getattr(reg, "model_package_arn", "")
        result.approval_status = getattr(reg, "approval_status", "") or (
            approval_status or ""
        )
        result.registry_reason = getattr(reg, "reason", "")
        if not result.registered and not result.reason:
            result.reason = result.registry_reason or "registry_register_failed"
        return result

    # ------------------------------------------------------------------
    # EventBridge schedule
    # ------------------------------------------------------------------

    def build_schedule_descriptor(
        self,
        *,
        target_arn: str = "",
        target_id: str = "routeiq-routing-retrain",
        input_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build (without calling AWS) the EventBridge schedule rule descriptor.

        Returns a dict with the ``put_rule`` + ``put_targets`` payloads the
        :meth:`put_schedule` live path would issue.  Cred-free and side-effect
        free -- handy for tests, a dry-run, or IaC export.
        """
        import json

        rule: Dict[str, Any] = {
            "Name": self.schedule_rule_name,
            "ScheduleExpression": self.schedule_expression,
            "State": "ENABLED",
            "Description": "RouteIQ scheduled routing-model retraining trigger.",
        }
        target: Dict[str, Any] = {"Id": target_id}
        if target_arn:
            target["Arn"] = target_arn
        payload = (
            input_payload
            if input_payload is not None
            else {
                "mode": self.mode,
                "pipeline_name": self.pipeline_name,
                "output_s3_uri": self._output_s3_uri(),
            }
        )
        target["Input"] = json.dumps(payload, sort_keys=True)
        return {"rule": rule, "targets": [target]}

    def put_schedule(
        self,
        *,
        target_arn: str = "",
        target_id: str = "routeiq-routing-retrain",
        input_payload: Optional[Dict[str, Any]] = None,
    ) -> ScheduleResult:
        """(Re)create the EventBridge rule that triggers the retraining run.

        Issues ``events.put_rule`` (cron/rate ``schedule_expression``) then
        ``events.put_targets`` so the schedule fires the retraining trigger.
        Safe no-op (never raises) when disabled or the events client is
        unavailable; on a live error the reason carries the error string.
        """
        result = ScheduleResult(
            rule_name=self.schedule_rule_name,
            schedule_expression=self.schedule_expression,
        )
        if not self.enabled:
            result.reason = "disabled"
            return result

        client = self._events_client()
        if client is None:
            result.reason = "no_client"
            return result

        descriptor = self.build_schedule_descriptor(
            target_arn=target_arn,
            target_id=target_id,
            input_payload=input_payload,
        )
        try:
            rule_response = client.put_rule(**descriptor["rule"])
            result.rule_arn = (
                rule_response.get("RuleArn", "")
                if isinstance(rule_response, dict)
                else ""
            )
            client.put_targets(
                Rule=self.schedule_rule_name,
                Targets=descriptor["targets"],
            )
            result.scheduled = True
            result.target_count = len(descriptor["targets"])
            logger.info(
                "Put EventBridge retraining schedule %s (%s)",
                self.schedule_rule_name,
                self.schedule_expression,
            )
        except Exception as e:
            result.scheduled = False
            result.reason = str(e)
            logger.warning("EventBridge schedule put failed: %s", e)
        return result


# ---------------------------------------------------------------------------
# Settings-gated singleton
# ---------------------------------------------------------------------------

_adapter: Optional[RetrainingPipelineAdapter] = None


def get_retraining_adapter() -> Optional[RetrainingPipelineAdapter]:
    """Get the retraining adapter singleton, or None when disabled.

    Returns None unless ``settings.mlops.retraining.enabled`` is true. Settings
    read failures degrade to disabled (None). The returned adapter builds its
    boto3 clients lazily on first live use, so getting it costs nothing.
    """
    global _adapter
    if _adapter is not None:
        return _adapter
    cfg = _retraining_settings()
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    _adapter = RetrainingPipelineAdapter(
        mode=getattr(cfg, "mode", "training_job") or "training_job",
        region=getattr(cfg, "region", "") or "",
        training_image=getattr(cfg, "training_image", "") or "",
        role_arn=getattr(cfg, "role_arn", "") or "",
        instance_type=getattr(cfg, "instance_type", "ml.m5.xlarge") or "ml.m5.xlarge",
        instance_count=int(getattr(cfg, "instance_count", 1) or 1),
        volume_size_gb=int(getattr(cfg, "volume_size_gb", 30) or 30),
        max_runtime_seconds=int(getattr(cfg, "max_runtime_seconds", 86400) or 86400),
        s3_artifact_bucket=getattr(cfg, "s3_artifact_bucket", "") or "",
        s3_artifact_prefix=getattr(cfg, "s3_artifact_prefix", "routeiq/routing-models")
        or "routeiq/routing-models",
        s3_input_uri=getattr(cfg, "s3_input_uri", "") or "",
        pipeline_name=getattr(cfg, "pipeline_name", "routeiq-routing-retrain")
        or "routeiq-routing-retrain",
        schedule_expression=getattr(cfg, "schedule_expression", "rate(7 days)")
        or "rate(7 days)",
        schedule_rule_name=getattr(cfg, "schedule_rule_name", "routeiq-routing-retrain")
        or "routeiq-routing-retrain",
        job_name_prefix=getattr(cfg, "job_name_prefix", "routeiq-routing-retrain")
        or "routeiq-routing-retrain",
        enabled=True,
    )
    return _adapter


def _retraining_settings():  # type: ignore[no-untyped-def]
    """Read ``settings.mlops.retraining`` (None on any failure)."""
    try:
        from litellm_llmrouter.settings import get_settings

        mlops = getattr(get_settings(), "mlops", None)
        return getattr(mlops, "retraining", None) if mlops is not None else None
    except Exception:  # pragma: no cover - defensive
        return None


def reset_retraining_adapter() -> None:
    """Reset the singleton (MUST be called in the autouse test fixture)."""
    global _adapter
    _adapter = None
