"""Unit tests for the SageMaker Model Registry + Experiments adapter
(RouteIQ-93e9).

Cred-free: a ``MagicMock`` boto3 sagemaker client is INJECTED into the adapter,
so no real client is constructed, no credentials are needed, and AWS is never
contacted. Verifies:

- register_model_package -> a model package is created in the group (group
  ensured first, idempotent on already-exists) with the configured approval
  status.
- log_experiment_run -> a trial component + metrics are logged.
- default-off settings gate: ``get_sagemaker_registry_adapter()`` is None until
  ``settings.mlops.sagemaker.enabled`` is set, and a disabled adapter never
  touches the client.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.mlops.sagemaker_registry import (
    SageMakerRegistryAdapter,
    get_sagemaker_registry_adapter,
    reset_sagemaker_registry_adapter,
)
from litellm_llmrouter.settings import reset_settings


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    reset_sagemaker_registry_adapter()
    yield
    reset_settings()
    reset_sagemaker_registry_adapter()


def _mock_client(*, package_arn: str = "arn:aws:sagemaker:::model-package/p/1"):
    client = MagicMock()
    client.create_model_package.return_value = {"ModelPackageArn": package_arn}
    return client


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


def test_register_creates_model_package_and_ensures_group():
    client = _mock_client()
    adapter = SageMakerRegistryAdapter(
        model_package_group_name="routeiq-routing-models",
        approval_status="PendingManualApproval",
        enabled=True,
        client=client,
    )

    result = adapter.register_model_package(
        artifact={
            "image": "123.dkr.ecr.us-east-1.amazonaws.com/routeiq:latest",
            "model_data_url": "s3://bucket/model.tar.gz",
            "framework": "sklearn",
            "sha256": "abc123",
        },
        metadata={"strategy": "llmrouter-knn"},
    )

    assert result.registered is True
    assert result.model_package_arn.endswith("/p/1")
    assert result.approval_status == "PendingManualApproval"
    assert result.group_created is True

    client.create_model_package_group.assert_called_once()
    client.create_model_package.assert_called_once()
    call = client.create_model_package.call_args.kwargs
    assert call["ModelPackageGroupName"] == "routeiq-routing-models"
    assert call["ModelApprovalStatus"] == "PendingManualApproval"
    # Customer metadata carries strategy + framework + sha256.
    md = call["CustomerMetadataProperties"]
    assert md["strategy"] == "llmrouter-knn"
    assert md["framework"] == "sklearn"
    assert md["sha256"] == "abc123"
    # The container references the injected image + data url.
    container = call["InferenceSpecification"]["Containers"][0]
    assert container["Image"].endswith("routeiq:latest")
    assert container["ModelDataUrl"] == "s3://bucket/model.tar.gz"


def test_register_idempotent_when_group_exists():
    client = _mock_client()
    client.create_model_package_group.side_effect = Exception(
        "ValidationException: Model Package Group already exists"
    )
    adapter = SageMakerRegistryAdapter(enabled=True, client=client)

    result = adapter.register_model_package(artifact={"image": "img"})

    assert result.registered is True
    assert result.group_created is False  # group already existed
    client.create_model_package.assert_called_once()


def test_register_records_reason_on_failure():
    client = _mock_client()
    client.create_model_package.side_effect = Exception("AccessDenied")
    adapter = SageMakerRegistryAdapter(enabled=True, client=client)

    result = adapter.register_model_package(artifact={"image": "img"})

    assert result.registered is False
    assert "AccessDenied" in result.reason


def test_register_override_approval_status():
    client = _mock_client()
    adapter = SageMakerRegistryAdapter(enabled=True, client=client)
    result = adapter.register_model_package(
        artifact={"image": "img"}, approval_status="Approved"
    )
    assert result.approval_status == "Approved"
    assert client.create_model_package.call_args.kwargs["ModelApprovalStatus"] == (
        "Approved"
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def test_log_experiment_run_creates_trial_component_and_metrics():
    client = _mock_client()
    adapter = SageMakerRegistryAdapter(
        experiment_name="routeiq-routing-eval", enabled=True, client=client
    )

    result = adapter.log_experiment_run(
        metrics={"mean_quality": 0.91, "n_cases": 6},
        run_name="run-001",
        parameters={"strategy": "llmrouter-knn", "tolerance": 0.1},
    )

    assert result.logged is True
    assert result.trial_component_name == "run-001"
    assert set(result.metrics_logged) == {"mean_quality", "n_cases"}

    client.create_trial_component.assert_called_once()
    tc_call = client.create_trial_component.call_args.kwargs
    assert tc_call["TrialComponentName"] == "run-001"
    # String + numeric params both encode correctly.
    assert tc_call["Parameters"]["strategy"] == {"StringValue": "llmrouter-knn"}
    assert tc_call["Parameters"]["tolerance"] == {"NumberValue": 0.1}

    client.batch_put_metrics.assert_called_once()
    metric_names = {
        m["MetricName"] for m in client.batch_put_metrics.call_args.kwargs["MetricData"]
    }
    assert metric_names == {"mean_quality", "n_cases"}


def test_log_experiment_run_default_run_name():
    client = _mock_client()
    adapter = SageMakerRegistryAdapter(enabled=True, client=client)
    result = adapter.log_experiment_run(metrics={"q": 0.5})
    assert result.logged is True
    assert result.trial_component_name.startswith("routeiq-run-")


# ---------------------------------------------------------------------------
# Disabled / gating (cred-free, byte-stable default off)
# ---------------------------------------------------------------------------


def test_disabled_adapter_never_touches_client():
    client = MagicMock()
    adapter = SageMakerRegistryAdapter(enabled=False, client=client)

    reg = adapter.register_model_package(artifact={"image": "img"})
    run = adapter.log_experiment_run(metrics={"q": 0.5})

    assert reg.registered is False and reg.reason == "disabled"
    assert run.logged is False and run.reason == "disabled"
    client.create_model_package_group.assert_not_called()
    client.create_model_package.assert_not_called()
    client.create_trial_component.assert_not_called()


def test_singleton_disabled_by_default():
    assert get_sagemaker_registry_adapter() is None


def test_singleton_enabled_via_settings(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_MLOPS__SAGEMAKER__ENABLED", "true")
    monkeypatch.setenv(
        "ROUTEIQ_MLOPS__SAGEMAKER__MODEL_PACKAGE_GROUP_NAME", "custom-group"
    )
    monkeypatch.setenv("ROUTEIQ_MLOPS__SAGEMAKER__APPROVAL_STATUS", "Approved")
    reset_settings()
    reset_sagemaker_registry_adapter()
    adapter = get_sagemaker_registry_adapter()
    assert adapter is not None
    assert adapter.model_package_group_name == "custom-group"
    assert adapter.approval_status == "Approved"
    assert adapter.enabled is True
