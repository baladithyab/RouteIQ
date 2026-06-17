"""Unit tests for the SageMaker scheduled-retraining orchestration adapter
(RouteIQ-8a24).

Cred-free: ``MagicMock`` boto3 ``sagemaker`` + ``events`` clients are INJECTED
into the adapter, so no real client is constructed, no credentials are needed,
and AWS is never contacted. Verifies:

- start_retraining (training_job mode) -> ``create_training_job`` with the right
  image/role/instance + S3 output path; (pipeline mode) ->
  ``start_pipeline_execution``.
- get_status -> ``describe_training_job`` / ``describe_pipeline_execution`` poll;
  success extracts the produced model artifact's S3 URI.
- register_on_success -> hands the artifact to the EXISTING registry adapter
  (``register_model_package``) so train->register stays one lifecycle.
- put_schedule -> EventBridge ``put_rule`` + ``put_targets`` with the cron/rate
  expression and target payload.
- default-off settings gate: ``get_retraining_adapter()`` is None until
  ``settings.mlops.retraining.enabled`` is set, and a disabled adapter never
  touches either client.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.mlops.retraining import (
    RetrainingPipelineAdapter,
    RetrainingStatusResult,
    get_retraining_adapter,
    reset_retraining_adapter,
)
from litellm_llmrouter.settings import reset_settings


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    reset_retraining_adapter()
    yield
    reset_settings()
    reset_retraining_adapter()


def _sm_client(*, job_arn: str = "arn:aws:sagemaker:::training-job/j"):
    client = MagicMock()
    client.create_training_job.return_value = {"TrainingJobArn": job_arn}
    client.start_pipeline_execution.return_value = {
        "PipelineExecutionArn": "arn:aws:sagemaker:::pipeline/p/execution/e"
    }
    return client


def _ev_client(*, rule_arn: str = "arn:aws:events:::rule/routeiq-routing-retrain"):
    client = MagicMock()
    client.put_rule.return_value = {"RuleArn": rule_arn}
    return client


def _adapter(**kwargs):
    defaults = dict(
        enabled=True,
        training_image="123.dkr.ecr.us-east-1.amazonaws.com/routeiq-train:latest",
        role_arn="arn:aws:iam::123:role/RouteIQTrain",
        s3_artifact_bucket="routeiq-models",
        s3_artifact_prefix="routeiq/routing-models",
        sagemaker_client=_sm_client(),
        events_client=_ev_client(),
    )
    defaults.update(kwargs)
    return RetrainingPipelineAdapter(**defaults)


# ---------------------------------------------------------------------------
# Start a retraining run -- training-job mode
# ---------------------------------------------------------------------------


def test_start_training_job_issues_create_training_job():
    client = _sm_client()
    adapter = _adapter(sagemaker_client=client)

    result = adapter.start_retraining(
        job_name="rq-train-001",
        hyperparameters={"router_type": "knn", "k": 5},
    )

    assert result.started is True
    assert result.mode == "training_job"
    assert result.job_name == "rq-train-001"
    assert result.job_arn.endswith("training-job/j")
    assert result.output_s3_uri == "s3://routeiq-models/routeiq/routing-models"

    client.create_training_job.assert_called_once()
    call = client.create_training_job.call_args.kwargs
    assert call["TrainingJobName"] == "rq-train-001"
    assert call["AlgorithmSpecification"]["TrainingImage"].endswith(
        "routeiq-train:latest"
    )
    assert call["RoleArn"] == "arn:aws:iam::123:role/RouteIQTrain"
    assert call["OutputDataConfig"]["S3OutputPath"] == (
        "s3://routeiq-models/routeiq/routing-models"
    )
    assert call["ResourceConfig"]["InstanceType"] == "ml.m5.xlarge"
    # Hyperparameters are stringified per the SageMaker contract.
    assert call["HyperParameters"]["router_type"] == "knn"
    assert call["HyperParameters"]["k"] == "5"


def test_start_training_job_auto_name_and_input_channel():
    client = _sm_client()
    adapter = _adapter(
        sagemaker_client=client,
        s3_input_uri="s3://routeiq-data/train/",
        job_name_prefix="rq-retrain",
    )
    result = adapter.start_retraining()
    assert result.started is True
    assert result.job_name.startswith("rq-retrain-")
    call = client.create_training_job.call_args.kwargs
    channel = call["InputDataConfig"][0]
    assert channel["ChannelName"] == "training"
    assert channel["DataSource"]["S3DataSource"]["S3Uri"] == "s3://routeiq-data/train/"


def test_start_training_job_records_reason_on_failure():
    client = _sm_client()
    client.create_training_job.side_effect = Exception("ResourceLimitExceeded")
    adapter = _adapter(sagemaker_client=client)
    result = adapter.start_retraining()
    assert result.started is False
    assert "ResourceLimitExceeded" in result.reason


# ---------------------------------------------------------------------------
# Start a retraining run -- pipeline mode
# ---------------------------------------------------------------------------


def test_start_pipeline_issues_start_pipeline_execution():
    client = _sm_client()
    adapter = _adapter(
        sagemaker_client=client,
        mode="pipeline",
        pipeline_name="routeiq-routing-retrain",
    )
    result = adapter.start_retraining(pipeline_parameters={"DataVersion": "v3"})

    assert result.started is True
    assert result.mode == "pipeline"
    assert result.pipeline_execution_arn.endswith("execution/e")
    client.start_pipeline_execution.assert_called_once()
    call = client.start_pipeline_execution.call_args.kwargs
    assert call["PipelineName"] == "routeiq-routing-retrain"
    assert call["PipelineParameters"] == [{"Name": "DataVersion", "Value": "v3"}]
    client.create_training_job.assert_not_called()


# ---------------------------------------------------------------------------
# Poll status
# ---------------------------------------------------------------------------


def test_status_training_job_in_progress():
    client = _sm_client()
    client.describe_training_job.return_value = {"TrainingJobStatus": "InProgress"}
    adapter = _adapter(sagemaker_client=client)
    result = adapter.get_status(job_name="rq-train-001")
    assert result.polled is True
    assert result.status == "InProgress"
    assert result.succeeded is False
    assert result.terminal is False
    assert result.model_artifact_uri == ""


def test_status_training_job_completed_extracts_artifact():
    client = _sm_client()
    client.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {
            "S3ModelArtifacts": "s3://routeiq-models/routeiq/routing-models/out/model.tar.gz"
        },
    }
    adapter = _adapter(sagemaker_client=client)
    result = adapter.get_status(job_name="rq-train-001")
    assert result.polled is True
    assert result.succeeded is True
    assert result.terminal is True
    assert result.model_artifact_uri.endswith("model.tar.gz")


def test_status_pipeline_succeeded():
    client = _sm_client()
    client.describe_pipeline_execution.return_value = {
        "PipelineExecutionStatus": "Succeeded"
    }
    adapter = _adapter(sagemaker_client=client, mode="pipeline")
    result = adapter.get_status(
        pipeline_execution_arn="arn:aws:sagemaker:::pipeline/p/execution/e"
    )
    assert result.polled is True
    assert result.status == "Succeeded"
    assert result.succeeded is True
    assert result.terminal is True


def test_status_records_reason_on_failure():
    client = _sm_client()
    client.describe_training_job.side_effect = Exception("ValidationException")
    adapter = _adapter(sagemaker_client=client)
    result = adapter.get_status(job_name="missing")
    assert result.polled is False
    assert "ValidationException" in result.reason


# ---------------------------------------------------------------------------
# register_on_success -> hands artifact to the EXISTING registry adapter
# ---------------------------------------------------------------------------


def test_register_on_success_calls_registry_adapter():
    adapter = _adapter()
    registry = MagicMock()
    registry.register_model_package.return_value = MagicMock(
        registered=True,
        model_package_arn="arn:aws:sagemaker:::model-package/g/1",
        approval_status="PendingManualApproval",
        reason="",
    )
    status = RetrainingStatusResult(
        polled=True,
        mode="training_job",
        status="Completed",
        succeeded=True,
        terminal=True,
        model_artifact_uri="s3://routeiq-models/out/model.tar.gz",
    )

    result = adapter.register_on_success(
        status=status,
        registry_adapter=registry,
        metadata={"strategy": "llmrouter-knn"},
        framework="sklearn",
    )

    assert result.registered is True
    assert result.model_package_arn.endswith("model-package/g/1")
    assert result.model_artifact_uri.endswith("model.tar.gz")

    registry.register_model_package.assert_called_once()
    call = registry.register_model_package.call_args.kwargs
    assert call["artifact"]["model_data_url"] == (
        "s3://routeiq-models/out/model.tar.gz"
    )
    assert call["artifact"]["framework"] == "sklearn"
    assert call["artifact"]["image"].endswith("routeiq-train:latest")
    assert call["metadata"] == {"strategy": "llmrouter-knn"}


def test_register_on_success_skips_when_run_not_succeeded():
    adapter = _adapter()
    registry = MagicMock()
    status = RetrainingStatusResult(
        polled=True, mode="training_job", status="Failed", succeeded=False
    )
    result = adapter.register_on_success(status=status, registry_adapter=registry)
    assert result.registered is False
    assert result.reason == "run_not_succeeded"
    registry.register_model_package.assert_not_called()


def test_register_on_success_skips_when_no_artifact():
    adapter = _adapter()
    registry = MagicMock()
    status = RetrainingStatusResult(
        polled=True, status="Completed", succeeded=True, model_artifact_uri=""
    )
    result = adapter.register_on_success(status=status, registry_adapter=registry)
    assert result.registered is False
    assert result.reason == "no_artifact"
    registry.register_model_package.assert_not_called()


def test_register_on_success_passes_through_registry_failure():
    adapter = _adapter()
    registry = MagicMock()
    registry.register_model_package.return_value = MagicMock(
        registered=False,
        model_package_arn="",
        approval_status="",
        reason="AccessDenied",
    )
    status = RetrainingStatusResult(
        polled=True,
        succeeded=True,
        model_artifact_uri="s3://b/model.tar.gz",
    )
    result = adapter.register_on_success(status=status, registry_adapter=registry)
    assert result.registered is False
    assert result.registry_reason == "AccessDenied"
    assert result.reason == "AccessDenied"


# ---------------------------------------------------------------------------
# EventBridge schedule
# ---------------------------------------------------------------------------


def test_build_schedule_descriptor_is_side_effect_free():
    ev = _ev_client()
    adapter = _adapter(
        events_client=ev,
        schedule_expression="cron(0 3 ? * SUN *)",
        schedule_rule_name="rq-weekly-retrain",
    )
    desc = adapter.build_schedule_descriptor(
        target_arn="arn:aws:lambda:::function/trigger"
    )
    assert desc["rule"]["Name"] == "rq-weekly-retrain"
    assert desc["rule"]["ScheduleExpression"] == "cron(0 3 ? * SUN *)"
    assert desc["rule"]["State"] == "ENABLED"
    target = desc["targets"][0]
    assert target["Arn"] == "arn:aws:lambda:::function/trigger"
    payload = json.loads(target["Input"])
    assert payload["mode"] == "training_job"
    assert payload["output_s3_uri"] == "s3://routeiq-models/routeiq/routing-models"
    # Pure builder: never touches the events client.
    ev.put_rule.assert_not_called()
    ev.put_targets.assert_not_called()


def test_put_schedule_issues_put_rule_and_put_targets():
    ev = _ev_client()
    adapter = _adapter(
        events_client=ev,
        schedule_expression="rate(7 days)",
        schedule_rule_name="routeiq-routing-retrain",
    )
    result = adapter.put_schedule(target_arn="arn:aws:lambda:::function/trigger")

    assert result.scheduled is True
    assert result.rule_name == "routeiq-routing-retrain"
    assert result.schedule_expression == "rate(7 days)"
    assert result.rule_arn.endswith("rule/routeiq-routing-retrain")
    assert result.target_count == 1

    ev.put_rule.assert_called_once()
    rule_call = ev.put_rule.call_args.kwargs
    assert rule_call["Name"] == "routeiq-routing-retrain"
    assert rule_call["ScheduleExpression"] == "rate(7 days)"

    ev.put_targets.assert_called_once()
    tgt_call = ev.put_targets.call_args.kwargs
    assert tgt_call["Rule"] == "routeiq-routing-retrain"
    assert tgt_call["Targets"][0]["Arn"] == "arn:aws:lambda:::function/trigger"


def test_put_schedule_records_reason_on_failure():
    ev = _ev_client()
    ev.put_rule.side_effect = Exception("AccessDeniedException")
    adapter = _adapter(events_client=ev)
    result = adapter.put_schedule(target_arn="arn:aws:lambda:::function/trigger")
    assert result.scheduled is False
    assert "AccessDeniedException" in result.reason


# ---------------------------------------------------------------------------
# Disabled / gating (cred-free, byte-stable default off)
# ---------------------------------------------------------------------------


def test_disabled_adapter_never_touches_clients():
    sm = MagicMock()
    ev = MagicMock()
    adapter = RetrainingPipelineAdapter(
        enabled=False, sagemaker_client=sm, events_client=ev
    )

    run = adapter.start_retraining()
    status = adapter.get_status(job_name="x")
    sched = adapter.put_schedule()
    reg = adapter.register_on_success(
        status=RetrainingStatusResult(succeeded=True, model_artifact_uri="s3://b/m")
    )

    assert run.started is False and run.reason == "disabled"
    assert status.polled is False and status.reason == "disabled"
    assert sched.scheduled is False and sched.reason == "disabled"
    assert reg.registered is False and reg.reason == "disabled"

    sm.create_training_job.assert_not_called()
    sm.start_pipeline_execution.assert_not_called()
    sm.describe_training_job.assert_not_called()
    ev.put_rule.assert_not_called()
    ev.put_targets.assert_not_called()


def test_no_client_results_are_safe():
    # Enabled but no injected client and (in tests) no boto3/AWS => safe no-run.
    adapter = RetrainingPipelineAdapter(enabled=True)
    # Force the lazy builders to return None so we never construct a real client.
    adapter._sagemaker_client = lambda: None  # type: ignore[method-assign]
    adapter._events_client = lambda: None  # type: ignore[method-assign]
    run = adapter.start_retraining()
    status = adapter.get_status(job_name="x")
    sched = adapter.put_schedule()
    assert run.reason == "no_client"
    assert status.reason == "no_client"
    assert sched.reason == "no_client"


def test_register_on_success_no_registry_adapter():
    # Enabled, succeeded run, artifact present, but no registry adapter available
    # (singleton disabled by default) => safe no-op.
    adapter = RetrainingPipelineAdapter(enabled=True)
    status = RetrainingStatusResult(
        succeeded=True, model_artifact_uri="s3://b/model.tar.gz"
    )
    result = adapter.register_on_success(status=status)
    assert result.registered is False
    assert result.reason == "no_registry_adapter"


# ---------------------------------------------------------------------------
# Settings-gated singleton
# ---------------------------------------------------------------------------


def test_singleton_disabled_by_default():
    assert get_retraining_adapter() is None


def test_singleton_enabled_via_settings(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_MLOPS__RETRAINING__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_MLOPS__RETRAINING__MODE", "pipeline")
    monkeypatch.setenv("ROUTEIQ_MLOPS__RETRAINING__S3_ARTIFACT_BUCKET", "my-bucket")
    monkeypatch.setenv("ROUTEIQ_MLOPS__RETRAINING__SCHEDULE_EXPRESSION", "rate(1 day)")
    monkeypatch.setenv("ROUTEIQ_MLOPS__RETRAINING__PIPELINE_NAME", "custom-pipeline")
    reset_settings()
    reset_retraining_adapter()
    adapter = get_retraining_adapter()
    assert adapter is not None
    assert adapter.enabled is True
    assert adapter.mode == "pipeline"
    assert adapter.s3_artifact_bucket == "my-bucket"
    assert adapter.schedule_expression == "rate(1 day)"
    assert adapter.pipeline_name == "custom-pipeline"


def test_singleton_is_stable():
    monkeypatch_env = {"ROUTEIQ_MLOPS__RETRAINING__ENABLED": "true"}
    import os

    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    try:
        reset_settings()
        reset_retraining_adapter()
        a = get_retraining_adapter()
        b = get_retraining_adapter()
        assert a is b
    finally:
        for k in monkeypatch_env:
            os.environ.pop(k, None)
