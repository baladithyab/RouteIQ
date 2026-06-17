"""
SageMaker Model Registry + Experiments Adapter (RouteIQ-93e9)
=============================================================

Registers a trained ROUTING-MODEL artifact into a SageMaker Model Package Group
(with an approval status) and logs run metrics to SageMaker Experiments, so the
routing model lifecycle (train -> register -> approve -> promote) and offline
eval metrics live in the same place as the rest of an org's ML governance.

Cred-free + mockable by construction: the adapter takes an INJECTED boto3
``sagemaker`` client (``SageMakerRegistryAdapter(client=mock)``); unit tests pass
a ``MagicMock`` and never touch AWS, never construct a real client, never need
credentials. The lazy ``_client()`` only builds a real ``boto3.client('sagemaker')``
when no client was injected AND the adapter is enabled (live path).

Settings-gated under ``settings.mlops.sagemaker`` and DEFAULT OFF: with
``enabled=False`` ``register_model_package`` / ``log_experiment_run`` short-circuit
and never construct a client. Live SageMaker calls are therefore operator-gated.

The two SageMaker control-plane operations used:
  * ``create_model_package`` -- registers a versioned model package into a Model
    Package Group with ``ModelApprovalStatus``. The group is ensured first via
    ``create_model_package_group`` (idempotent: an already-exists error is
    treated as success).
  * ``put_trial_component`` / ``associate_trial_component`` would be the live
    SDK calls for Experiments; the SageMaker SDK's Experiments helper is logged
    here through ``create_trial_component`` + ``batch_put_metrics`` on the
    sagemaker client, which is the cred-free control-plane surface a mock models.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "SageMakerRegistryAdapter",
    "ModelPackageResult",
    "ExperimentRunResult",
    "get_sagemaker_registry_adapter",
    "reset_sagemaker_registry_adapter",
]


@dataclass
class ModelPackageResult:
    """Outcome of registering a routing-model artifact as a model package."""

    registered: bool = False
    model_package_arn: str = ""
    model_package_group_name: str = ""
    approval_status: str = ""
    group_created: bool = False
    reason: str = ""


@dataclass
class ExperimentRunResult:
    """Outcome of logging one experiment run's metrics."""

    logged: bool = False
    experiment_name: str = ""
    trial_component_name: str = ""
    metrics_logged: List[str] = field(default_factory=list)
    reason: str = ""


class SageMakerRegistryAdapter:
    """SageMaker Model Registry + Experiments adapter (RouteIQ-93e9).

    Args:
        model_package_group_name: Target Model Package Group.
        approval_status: ModelApprovalStatus for registered packages.
        experiment_name: SageMaker Experiment runs are logged under.
        region: AWS region for the lazily-built boto3 client (live path only).
        enabled: Whether live SageMaker calls are permitted.
        client: Injected boto3 sagemaker client (cred-free tests pass a mock);
            when None a real client is built lazily on first live use.
    """

    def __init__(
        self,
        *,
        model_package_group_name: str = "routeiq-routing-models",
        approval_status: str = "PendingManualApproval",
        experiment_name: str = "routeiq-routing-eval",
        region: str = "",
        enabled: bool = True,
        client: Any = None,
    ) -> None:
        self.model_package_group_name = model_package_group_name
        self.approval_status = approval_status
        self.experiment_name = experiment_name
        self.region = region
        self.enabled = enabled
        self._client = client

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _sagemaker_client(self) -> Optional[Any]:
        """Return the injected client, or lazily build a real one (live path).

        Cred-free: tests inject a mock so this never builds a real client. The
        real-client branch is only reached when no client was injected -- it is
        ``# pragma: no cover`` because exercising it would require boto3 + AWS.
        """
        if self._client is not None:
            return self._client
        try:  # pragma: no cover - live path requires boto3 + AWS creds
            import boto3

            kwargs: Dict[str, Any] = {}
            if self.region:
                kwargs["region_name"] = self.region
            self._client = boto3.client("sagemaker", **kwargs)
            return self._client
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("SageMaker client construction failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Model Registry
    # ------------------------------------------------------------------

    def _ensure_model_package_group(self, client: Any) -> bool:
        """Create the Model Package Group if absent (idempotent).

        Returns True iff a create was issued and succeeded. An already-exists
        error (``ValidationException`` carrying "already exists") is treated as
        success-without-create (returns False, no error).
        """
        try:
            client.create_model_package_group(
                ModelPackageGroupName=self.model_package_group_name,
                ModelPackageGroupDescription=(
                    "RouteIQ routing-model artifacts (registered by the MLOps loop)."
                ),
            )
            return True
        except Exception as e:
            msg = str(e).lower()
            if "already exist" in msg or "alreadyexists" in msg:
                return False
            # Re-raise other errors so register_model_package records the failure.
            raise

    def register_model_package(
        self,
        *,
        artifact: Dict[str, Any],
        approval_status: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ModelPackageResult:
        """Register a routing-model artifact into the Model Package Group.

        ``artifact`` describes the model image/data (e.g.
        ``{"image": "...", "model_data_url": "s3://...", "framework": "..."}``).
        Ensures the group exists, then ``create_model_package`` with the
        configured (or overridden) ``ModelApprovalStatus``.

        Returns a :class:`ModelPackageResult`. A safe, non-registered result
        (never raises) is returned when the adapter is disabled or the client is
        unavailable; on a live error the reason carries the error string.
        """
        result = ModelPackageResult(
            model_package_group_name=self.model_package_group_name,
            approval_status=approval_status or self.approval_status,
        )
        if not self.enabled:
            result.reason = "disabled"
            return result

        client = self._sagemaker_client()
        if client is None:
            result.reason = "no_client"
            return result

        try:
            result.group_created = self._ensure_model_package_group(client)

            container: Dict[str, Any] = {}
            if artifact.get("image"):
                container["Image"] = artifact["image"]
            if artifact.get("model_data_url"):
                container["ModelDataUrl"] = artifact["model_data_url"]

            customer_metadata: Dict[str, str] = {}
            if metadata:
                customer_metadata.update({str(k): str(v) for k, v in metadata.items()})
            if artifact.get("framework"):
                customer_metadata.setdefault("framework", str(artifact["framework"]))
            if artifact.get("sha256"):
                customer_metadata.setdefault("sha256", str(artifact["sha256"]))

            response = client.create_model_package(
                ModelPackageGroupName=self.model_package_group_name,
                ModelPackageDescription=str(
                    artifact.get("description", "RouteIQ routing model")
                ),
                InferenceSpecification={
                    "Containers": [container] if container else [],
                    "SupportedContentTypes": ["application/json"],
                    "SupportedResponseMIMETypes": ["application/json"],
                },
                ModelApprovalStatus=result.approval_status,
                CustomerMetadataProperties=customer_metadata or {},
            )
            result.registered = True
            result.model_package_arn = (
                response.get("ModelPackageArn", "")
                if isinstance(response, dict)
                else ""
            )
            logger.info(
                "Registered routing model package into %s (status=%s)",
                self.model_package_group_name,
                result.approval_status,
            )
        except Exception as e:
            result.registered = False
            result.reason = str(e)
            logger.warning("SageMaker model-package registration failed: %s", e)
        return result

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def log_experiment_run(
        self,
        *,
        metrics: Dict[str, float],
        run_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRunResult:
        """Log a run's metrics to a SageMaker Experiments trial component.

        Creates a trial component named for the run (or a timestamped default)
        under the configured experiment and records each scalar metric. Returns
        an :class:`ExperimentRunResult`. Safe no-op (never raises) when disabled
        or the client is unavailable.
        """
        tc_name = run_name or f"routeiq-run-{int(time.time())}"
        result = ExperimentRunResult(
            experiment_name=self.experiment_name, trial_component_name=tc_name
        )
        if not self.enabled:
            result.reason = "disabled"
            return result

        client = self._sagemaker_client()
        if client is None:
            result.reason = "no_client"
            return result

        try:
            params: dict[str, dict[str, Any]] = {
                str(k): {"NumberValue": float(v)}
                if isinstance(v, (int, float))
                else {"StringValue": str(v)}
                for k, v in (parameters or {}).items()
            }
            client.create_trial_component(
                TrialComponentName=tc_name,
                DisplayName=tc_name,
                Parameters=params or {},
                Tags=[
                    {"Key": "routeiq:experiment", "Value": self.experiment_name},
                ],
            )

            metric_data = [
                {
                    "MetricName": str(name),
                    "Value": float(value),
                    "Timestamp": time.time(),
                }
                for name, value in metrics.items()
                if isinstance(value, (int, float))
            ]
            if metric_data:
                client.batch_put_metrics(
                    TrialComponentName=tc_name,
                    MetricData=metric_data,
                )
            result.logged = True
            result.metrics_logged = [str(m["MetricName"]) for m in metric_data]
            logger.info(
                "Logged %d metrics to SageMaker experiment %s (run=%s)",
                len(metric_data),
                self.experiment_name,
                tc_name,
            )
        except Exception as e:
            result.logged = False
            result.reason = str(e)
            logger.warning("SageMaker experiment logging failed: %s", e)
        return result


# ---------------------------------------------------------------------------
# Settings-gated singleton
# ---------------------------------------------------------------------------

_adapter: Optional[SageMakerRegistryAdapter] = None


def get_sagemaker_registry_adapter() -> Optional[SageMakerRegistryAdapter]:
    """Get the SageMaker adapter singleton, or None when disabled.

    Returns None unless ``settings.mlops.sagemaker.enabled`` is true. Settings
    read failures degrade to disabled (None). The returned adapter builds its
    boto3 client lazily on first live use, so getting it costs nothing.
    """
    global _adapter
    if _adapter is not None:
        return _adapter
    cfg = _sagemaker_settings()
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    _adapter = SageMakerRegistryAdapter(
        model_package_group_name=getattr(
            cfg, "model_package_group_name", "routeiq-routing-models"
        ),
        approval_status=getattr(cfg, "approval_status", "PendingManualApproval"),
        experiment_name=getattr(cfg, "experiment_name", "routeiq-routing-eval"),
        region=getattr(cfg, "region", "") or "",
        enabled=True,
    )
    return _adapter


def _sagemaker_settings():  # type: ignore[no-untyped-def]
    """Read ``settings.mlops.sagemaker`` (None on any failure)."""
    try:
        from litellm_llmrouter.settings import get_settings

        mlops = getattr(get_settings(), "mlops", None)
        return getattr(mlops, "sagemaker", None) if mlops is not None else None
    except Exception:  # pragma: no cover - defensive
        return None


def reset_sagemaker_registry_adapter() -> None:
    """Reset the singleton (MUST be called in the autouse test fixture)."""
    global _adapter
    _adapter = None
