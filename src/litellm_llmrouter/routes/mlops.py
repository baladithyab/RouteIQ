"""Admin MLOps control-plane routes (RouteIQ-035c).

Exposes an in-process trigger for the SageMaker scheduled-retraining pipeline so
an operator can kick a retrain (or build/put the EventBridge schedule) without
waiting for the cron rule. The route delegates to the settings-gated
``get_retraining_adapter()`` singleton:

* DEFAULT-OFF: when ``settings.mlops.retraining.enabled`` is false the adapter
  singleton is ``None`` and the route returns a ``disabled`` result -- it builds
  no boto3 client and never contacts AWS (byte-stable; the router is registered
  but inert).
* When enabled, the route calls the adapter's ``start_retraining`` /
  ``put_schedule`` (which themselves remain cred-free-mockable and fail-safe).

Admin-auth gated: the router carries ``admin_api_key_auth`` so only the
control-plane key can trigger a retrain. Registered by the gateway app factory
in ``_register_routes`` (the LIVE callsite).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..auth import admin_api_key_auth

logger = logging.getLogger(__name__)


class RetrainTriggerRequest(BaseModel):
    """Optional body for a manual retrain trigger."""

    job_name: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    pipeline_parameters: Optional[Dict[str, str]] = None


def create_mlops_router() -> APIRouter:
    """Create the admin MLOps router (RouteIQ-035c).

    The router is always creatable (so the surface is documented + middleware-
    wrapped), but every handler is inert when the retraining adapter is disabled:
    it returns ``{"triggered": False, "reason": "disabled"}`` without touching
    AWS. This is the byte-stable default.
    """
    router = APIRouter(
        prefix="/api/v1/routeiq/mlops",
        tags=["mlops"],
        dependencies=[Depends(admin_api_key_auth)],
    )

    @router.get("/retraining/status", summary="MLOps retraining adapter status")
    async def retraining_status() -> Dict[str, Any]:
        from ..mlops.retraining import get_retraining_adapter

        adapter = get_retraining_adapter()
        if adapter is None:
            return {"enabled": False, "reason": "disabled"}
        return {
            "enabled": True,
            "mode": adapter.mode,
            "schedule_expression": adapter.schedule_expression,
            "schedule_rule_name": adapter.schedule_rule_name,
            "output_s3_uri": adapter._output_s3_uri(),
        }

    @router.post("/retraining/trigger", summary="Trigger a routing-model retrain")
    async def trigger_retraining(
        body: Optional[RetrainTriggerRequest] = None,
    ) -> Dict[str, Any]:
        from ..mlops.retraining import get_retraining_adapter

        adapter = get_retraining_adapter()
        if adapter is None:
            # Default-off: adapter disabled => never contacts AWS.
            return {"triggered": False, "reason": "disabled"}

        req = body or RetrainTriggerRequest()
        result = adapter.start_retraining(
            job_name=req.job_name,
            hyperparameters=req.hyperparameters,
            pipeline_parameters=req.pipeline_parameters,
        )
        logger.info(
            "MLOps retrain trigger: started=%s mode=%s reason=%s",
            result.started,
            result.mode,
            result.reason,
        )
        return {
            "triggered": bool(result.started),
            "mode": result.mode,
            "job_name": result.job_name,
            "job_arn": result.job_arn,
            "pipeline_execution_arn": result.pipeline_execution_arn,
            "output_s3_uri": result.output_s3_uri,
            "reason": result.reason,
        }

    @router.post("/retraining/schedule", summary="(Re)create the retrain schedule")
    async def put_schedule() -> Dict[str, Any]:
        from ..mlops.retraining import get_retraining_adapter

        adapter = get_retraining_adapter()
        if adapter is None:
            return {"scheduled": False, "reason": "disabled"}
        result = adapter.put_schedule()
        return {
            "scheduled": bool(result.scheduled),
            "rule_name": result.rule_name,
            "schedule_expression": result.schedule_expression,
            "rule_arn": result.rule_arn,
            "target_count": result.target_count,
            "reason": result.reason,
        }

    return router


__all__ = ["create_mlops_router", "RetrainTriggerRequest"]
