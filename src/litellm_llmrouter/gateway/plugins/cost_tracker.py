"""
Cost Tracker Plugin
====================

Tracks per-request LLM costs using actual token usage from API responses
and emits OTel span attributes and metrics.

The existing quota system (quota.py) over-estimates spend by 2-10x because it
reserves max_tokens with no post-call reconciliation. This plugin calculates
actual cost from response ``usage`` data and emits accurate cost telemetry.

Hooks:
    on_llm_pre_call  -- record request start time and estimated cost
    on_llm_success   -- calculate actual cost, emit OTel attributes/metrics
    on_llm_failure   -- record zero cost, increment error counter

OTel span attributes emitted on success:
    llm.cost.input_tokens, llm.cost.output_tokens, llm.cost.total_tokens
    llm.cost.input_cost_usd, llm.cost.output_cost_usd, llm.cost.total_cost_usd
    llm.cost.model, llm.cost.estimated_cost_usd, llm.cost.estimation_error_pct

OTel metrics:
    llm_cost_total          Counter  (USD, by model/team/user)
    llm_tokens_total        Counter  (tokens, by model/team/direction)
    llm_cost_per_request    Histogram (USD, by model)
    llm_active_requests     UpDownCounter (gauge-like, by model)
    llm_cost_errors_total   Counter  (errors, by model)

Configuration:
    COST_TRACKER_ENABLED: Enable/disable this plugin (default: true)
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# OTel imports -- optional at runtime
try:
    from opentelemetry import trace
    from opentelemetry.metrics import get_meter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]
    get_meter = None  # type: ignore[assignment]

# Span attribute keys
ATTR_INPUT_TOKENS = "llm.cost.input_tokens"
ATTR_OUTPUT_TOKENS = "llm.cost.output_tokens"
ATTR_TOTAL_TOKENS = "llm.cost.total_tokens"
ATTR_INPUT_COST = "llm.cost.input_cost_usd"
ATTR_OUTPUT_COST = "llm.cost.output_cost_usd"
ATTR_TOTAL_COST = "llm.cost.total_cost_usd"
ATTR_MODEL = "llm.cost.model"
ATTR_ESTIMATED_COST = "llm.cost.estimated_cost_usd"
ATTR_ESTIMATION_ERROR = "llm.cost.estimation_error_pct"

# Internal metadata key stored in kwargs["litellm_params"]["metadata"]
_META_START_TIME = "_cost_tracker_start_time"
_META_ESTIMATED_COST = "_estimated_cost"
_META_RESERVED_SPEND = "_reserved_spend"
_META_QUOTA_SUBJECT = "_quota_subject"


def _is_enabled() -> bool:
    """Check if cost tracker is enabled via environment variable."""
    return os.getenv("COST_TRACKER_ENABLED", "true").lower() != "false"


def _extract_usage(response: Any) -> tuple[int, int]:
    """
    Extract token usage from an LLM response object.

    Returns:
        (input_tokens, output_tokens) -- defaults to (0, 0) when unavailable.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        # Some responses carry usage as a dict
        if isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return 0, 0

    if isinstance(usage, dict):
        return (
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
        )

    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _calculate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> tuple[float, float, float]:
    """
    Calculate cost using ``litellm.completion_cost``.

    Returns:
        (input_cost, output_cost, total_cost) in USD.
        Falls back to (0.0, 0.0, 0.0) if litellm is unavailable.
    """
    try:
        import litellm

        total_cost = litellm.completion_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        # Approximate split using per-token rates from model_cost
        cost_info = getattr(litellm, "model_cost", {}).get(model, {})
        input_rate = cost_info.get("input_cost_per_token", 0)
        output_rate = cost_info.get("output_cost_per_token", 0)

        if input_rate or output_rate:
            input_cost = input_tokens * input_rate
            output_cost = output_tokens * output_rate
        else:
            # Cannot split -- attribute entirely to total
            input_cost = 0.0
            output_cost = 0.0

        return input_cost, output_cost, float(total_cost)
    except Exception:
        return 0.0, 0.0, 0.0


def _get_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Safely retrieve the metadata dict from kwargs."""
    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return {}
    metadata = litellm_params.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    return metadata


class CostTrackerPlugin(GatewayPlugin):
    """
    Plugin that tracks per-request LLM costs and emits OTel telemetry.

    Hooks into the LLM lifecycle via PluginCallbackBridge:
    - on_llm_pre_call:  record pre-call state (start time, estimated cost)
    - on_llm_success:   calculate actual cost, emit metrics and span attributes
    - on_llm_failure:   handle failed requests (zero cost, error counter)
    """

    def __init__(self) -> None:
        self._tracer: Any = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="cost-tracker",
            version="1.0.0",
            capabilities={
                PluginCapability.EVALUATOR,
                PluginCapability.OBSERVABILITY_EXPORTER,
            },
            priority=50,
            description="Tracks per-request LLM costs and emits OTel telemetry",
        )

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        if not _is_enabled():
            logger.info("CostTrackerPlugin disabled via COST_TRACKER_ENABLED")
            return

        if OTEL_AVAILABLE:
            self._tracer = trace.get_tracer("routeiq.cost_tracker")

        logger.info("CostTrackerPlugin started (using centralized GatewayMetrics)")

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        logger.info("CostTrackerPlugin shut down")

    # --------------------------------------------------------------------- #
    # LLM lifecycle hooks
    # --------------------------------------------------------------------- #

    async def on_llm_pre_call(
        self, model: str, messages: list[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        from litellm_llmrouter.metrics import get_gateway_metrics

        metadata = _get_metadata(kwargs)
        metadata[_META_START_TIME] = time.monotonic()

        metrics = get_gateway_metrics()
        if metrics is not None:
            metrics.request_active.add(1, {"model": model})

        return None  # don't modify kwargs

    async def on_llm_success(
        self, model: str, response: Any, kwargs: dict[str, Any]
    ) -> None:
        metadata = _get_metadata(kwargs)

        # -- Tokens --------------------------------------------------------
        input_tokens, output_tokens = _extract_usage(response)
        total_tokens = input_tokens + output_tokens

        # -- Cost ----------------------------------------------------------
        input_cost, output_cost, total_cost = _calculate_cost(
            model, input_tokens, output_tokens
        )

        # -- Estimation error ----------------------------------------------
        estimated_cost = float(metadata.get(_META_ESTIMATED_COST, 0) or 0)
        estimation_error_pct = 0.0
        if estimated_cost > 0 and total_cost > 0:
            estimation_error_pct = ((estimated_cost - total_cost) / total_cost) * 100.0

        # -- OTel span attributes ------------------------------------------
        self._set_span_attributes(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            estimated_cost=estimated_cost,
            estimation_error_pct=estimation_error_pct,
        )

        # -- OTel metrics (via centralized GatewayMetrics) --------------------
        from litellm_llmrouter.metrics import get_gateway_metrics

        team = metadata.get("team_id", "")
        user = metadata.get("user_id", "")
        metrics = get_gateway_metrics()

        if metrics is not None:
            if total_cost > 0:
                metrics.cost_total.add(
                    total_cost,
                    {"model": model, "team": team, "user": user},
                )

            if total_tokens > 0:
                metrics.tokens_total.add(
                    input_tokens,
                    {"model": model, "team": team, "direction": "input"},
                )
                metrics.tokens_total.add(
                    output_tokens,
                    {"model": model, "team": team, "direction": "output"},
                )

            metrics.cost_per_request.record(total_cost, {"model": model})
            metrics.request_active.add(-1, {"model": model})

        # -- Post-call cost reconciliation ---------------------------------
        await self._reconcile_quota(metadata, total_cost)

        logger.debug(
            "CostTracker: model=%s tokens=%d cost=$%.6f estimated=$%.6f error=%.1f%%",
            model,
            total_tokens,
            total_cost,
            estimated_cost,
            estimation_error_pct,
        )

    async def on_llm_failure(
        self, model: str, exception: Exception, kwargs: dict[str, Any]
    ) -> None:
        # Zero cost on failure (no tokens consumed)
        self._set_span_attributes(
            model=model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            estimated_cost=0.0,
            estimation_error_pct=0.0,
        )

        from litellm_llmrouter.metrics import get_gateway_metrics

        metrics = get_gateway_metrics()
        if metrics is not None:
            metrics.cost_errors.add(1, {"model": model})
            metrics.request_active.add(-1, {"model": model})

        logger.debug(
            "CostTracker: model=%s failed error=%s",
            model,
            str(exception)[:200],
        )

    # --------------------------------------------------------------------- #
    # Cost reconciliation
    # --------------------------------------------------------------------- #

    @staticmethod
    async def _reconcile_quota(metadata: dict[str, Any], actual_cost: float) -> None:
        """Reconcile quota spend if reserved_spend was recorded pre-call.

        Extracts the reserved spend and quota subject from request metadata
        (stored during ``quota_guard`` pre-call reservation) and calls
        ``QuotaEnforcer.reconcile_spend()`` to adjust the quota counter.
        """
        reserved_spend = float(metadata.get(_META_RESERVED_SPEND, 0) or 0)
        subject = metadata.get(_META_QUOTA_SUBJECT, "")

        if not reserved_spend or not subject:
            return

        try:
            from litellm_llmrouter.quota import get_quota_enforcer

            enforcer = get_quota_enforcer()
            await enforcer.reconcile_spend(
                subject=subject,
                actual_cost=actual_cost,
                reserved_cost=reserved_spend,
            )
        except Exception:
            logger.debug("CostTracker: reconciliation failed for subject=%s", subject)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _set_span_attributes(
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        input_cost: float,
        output_cost: float,
        total_cost: float,
        estimated_cost: float,
        estimation_error_pct: float,
    ) -> None:
        """Set cost-related OTel span attributes on the current span."""
        if not OTEL_AVAILABLE:
            return

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return

        span.set_attribute(ATTR_MODEL, model)
        span.set_attribute(ATTR_INPUT_TOKENS, input_tokens)
        span.set_attribute(ATTR_OUTPUT_TOKENS, output_tokens)
        span.set_attribute(ATTR_TOTAL_TOKENS, total_tokens)
        span.set_attribute(ATTR_INPUT_COST, input_cost)
        span.set_attribute(ATTR_OUTPUT_COST, output_cost)
        span.set_attribute(ATTR_TOTAL_COST, total_cost)
        span.set_attribute(ATTR_ESTIMATED_COST, estimated_cost)
        span.set_attribute(ATTR_ESTIMATION_ERROR, estimation_error_pct)
