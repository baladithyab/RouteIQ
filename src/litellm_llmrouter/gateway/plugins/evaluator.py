"""
Evaluator Plugin Framework
===========================

This module provides the base class and contracts for evaluator plugins that
perform post-execution scoring and emit OTEL metrics.

Evaluator plugins can hook into:
- Post-MCP tool invocation (after tool call completes)
- Post-A2A agent invocation (after agent call completes)

OTEL Attributes emitted:
- `eval.plugin`: Name of the evaluator plugin
- `eval.score`: Numeric score (0.0-1.0)
- `eval.status`: Status string (success, error, skipped)
- `eval.duration_ms`: Evaluation duration in milliseconds

Usage:
    class MyEvaluator(EvaluatorPlugin):
        async def evaluate_mcp_result(self, context) -> EvaluationResult:
            # Perform evaluation
            return EvaluationResult(score=0.95, status="success")

Configuration:
    - ROUTEIQ_EVALUATOR_ENABLED: Enable/disable evaluator hooks (default: false)
    - ROUTEIQ_EVALUATOR_PLUGINS: Comma-separated list of evaluator plugin paths
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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

# OTEL attribute names for evaluator spans
ATTR_EVAL_PLUGIN = "eval.plugin"
ATTR_EVAL_SCORE = "eval.score"
ATTR_EVAL_STATUS = "eval.status"
ATTR_EVAL_DURATION_MS = "eval.duration_ms"
ATTR_EVAL_ERROR = "eval.error"
ATTR_EVAL_INVOCATION_TYPE = "eval.invocation_type"

# OTel imports - optional
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Status = None
    StatusCode = None


@dataclass
class EvaluationResult:
    """
    Result of an evaluation operation.

    Attributes:
        score: Numeric score between 0.0 and 1.0 (None if not applicable)
        status: Status string (success, error, skipped)
        metadata: Additional metadata from the evaluation
        error: Error message if status is "error"
    """

    score: float | None = None
    status: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class MCPInvocationContext:
    """
    Context for MCP tool invocation evaluation.

    Contains information about the tool call that was executed.
    """

    tool_name: str
    """Name of the MCP tool that was called."""

    server_id: str
    """ID of the MCP server that handled the call."""

    server_name: str
    """Name of the MCP server."""

    arguments: dict[str, Any]
    """Arguments passed to the tool."""

    result: Any
    """Result returned from the tool invocation."""

    success: bool
    """Whether the tool invocation succeeded."""

    error: str | None
    """Error message if invocation failed."""

    duration_ms: float
    """Duration of the tool invocation in milliseconds."""

    span: Any = None
    """Current OTEL span (if tracing is enabled)."""


@dataclass
class A2AInvocationContext:
    """
    Context for A2A agent invocation evaluation.

    Contains information about the agent call that was executed.
    """

    agent_id: str
    """ID of the A2A agent that was called."""

    agent_name: str
    """Name of the A2A agent."""

    agent_url: str
    """URL of the A2A agent endpoint."""

    method: str
    """JSON-RPC method that was called."""

    request: Any
    """Request that was sent to the agent."""

    result: Any
    """Result returned from the agent invocation."""

    success: bool
    """Whether the agent invocation succeeded."""

    error: str | None
    """Error message if invocation failed."""

    duration_ms: float
    """Duration of the agent invocation in milliseconds."""

    span: Any = None
    """Current OTEL span (if tracing is enabled)."""


class EvaluatorPlugin(GatewayPlugin, ABC):
    """
    Abstract base class for evaluator plugins.

    Evaluator plugins hook into post-invocation flows to score
    and evaluate MCP tool and A2A agent calls.

    Subclasses must implement:
    - evaluate_mcp_result: Called after MCP tool invocations
    - evaluate_a2a_result: Called after A2A agent invocations
    """

    @property
    def metadata(self) -> PluginMetadata:
        """
        Return plugin metadata.

        Override to customize metadata. By default declares EVALUATOR capability.
        """
        return PluginMetadata(
            name=self.__class__.__name__,
            version="0.0.0",
            capabilities={PluginCapability.EVALUATOR},
            priority=2000,  # Evaluators load after core plugins
            description="Evaluator plugin",
        )

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """
        Called during application startup.

        Evaluator plugins typically don't need to do much here,
        but can initialize resources if needed.
        """
        if context and context.logger:
            context.logger.info(f"Evaluator plugin {self.name} starting")

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """
        Called during application shutdown.

        Clean up any resources initialized in startup.
        """
        pass

    @abstractmethod
    async def evaluate_mcp_result(
        self, context: MCPInvocationContext
    ) -> EvaluationResult:
        """
        Evaluate an MCP tool invocation result.

        Called after each MCP tool call completes. Implementations should:
        - Analyze the result for quality/correctness
        - Return a score and status
        - Optionally emit additional OTEL events

        Args:
            context: Information about the tool invocation

        Returns:
            EvaluationResult with score and status
        """
        pass

    @abstractmethod
    async def evaluate_a2a_result(
        self, context: A2AInvocationContext
    ) -> EvaluationResult:
        """
        Evaluate an A2A agent invocation result.

        Called after each A2A agent call completes. Implementations should:
        - Analyze the result for quality/correctness
        - Return a score and status
        - Optionally emit additional OTEL events

        Args:
            context: Information about the agent invocation

        Returns:
            EvaluationResult with score and status
        """
        pass


def add_evaluation_attributes(
    span: Any,
    plugin_name: str,
    result: EvaluationResult,
    duration_ms: float,
    invocation_type: str = "mcp",
) -> None:
    """
    Add evaluation attributes to an OTEL span.

    Args:
        span: The OTEL span to add attributes to
        plugin_name: Name of the evaluator plugin
        result: The evaluation result
        duration_ms: Duration of the evaluation in milliseconds
        invocation_type: Type of invocation ("mcp" or "a2a")
    """
    if span is None or not hasattr(span, "set_attribute"):
        return

    try:
        span.set_attribute(ATTR_EVAL_PLUGIN, plugin_name)
        span.set_attribute(ATTR_EVAL_STATUS, result.status)
        span.set_attribute(ATTR_EVAL_DURATION_MS, round(duration_ms, 2))
        span.set_attribute(ATTR_EVAL_INVOCATION_TYPE, invocation_type)

        if result.score is not None:
            span.set_attribute(ATTR_EVAL_SCORE, result.score)

        if result.error:
            span.set_attribute(ATTR_EVAL_ERROR, result.error)

        # Add evaluation event
        if OTEL_AVAILABLE and hasattr(span, "add_event"):
            span.add_event(
                "evaluation",
                attributes={
                    "plugin": plugin_name,
                    "score": result.score if result.score is not None else -1,
                    "status": result.status,
                },
            )
    except Exception as e:
        logger.debug(f"Failed to add evaluation attributes to span: {e}")


# Global evaluator registry (protected by lock for thread safety)
_evaluator_plugins: list[EvaluatorPlugin] = []
_evaluator_lock = threading.Lock()


def register_evaluator(plugin: EvaluatorPlugin) -> None:
    """Register an evaluator plugin for hook invocation."""
    with _evaluator_lock:
        _evaluator_plugins.append(plugin)
    logger.info(f"Registered evaluator plugin: {plugin.name}")


def get_evaluator_plugins() -> list[EvaluatorPlugin]:
    """Get all registered evaluator plugins."""
    with _evaluator_lock:
        return list(_evaluator_plugins)


def clear_evaluator_plugins() -> None:
    """Clear all registered evaluator plugins. Primarily for testing."""
    with _evaluator_lock:
        _evaluator_plugins.clear()


def is_evaluator_enabled() -> bool:
    """Check if evaluator hooks are enabled."""
    return os.getenv("ROUTEIQ_EVALUATOR_ENABLED", "false").lower() == "true"


async def run_mcp_evaluators(context: MCPInvocationContext) -> list[EvaluationResult]:
    """
    Run all registered evaluator plugins for an MCP invocation.

    Args:
        context: The MCP invocation context

    Returns:
        List of evaluation results from all plugins
    """
    if not is_evaluator_enabled():
        return []

    plugins = get_evaluator_plugins()
    results: list[EvaluationResult] = []
    for plugin in plugins:
        start_time = time.perf_counter()
        try:
            result = await plugin.evaluate_mcp_result(context)
            duration_ms = (time.perf_counter() - start_time) * 1000
            add_evaluation_attributes(
                context.span, plugin.name, result, duration_ms, "mcp"
            )
            results.append(result)
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_result = EvaluationResult(status="error", error=str(e))
            add_evaluation_attributes(
                context.span, plugin.name, error_result, duration_ms, "mcp"
            )
            results.append(error_result)
            logger.warning(f"Evaluator {plugin.name} failed for MCP: {e}")

    return results


async def run_a2a_evaluators(context: A2AInvocationContext) -> list[EvaluationResult]:
    """
    Run all registered evaluator plugins for an A2A invocation.

    Args:
        context: The A2A invocation context

    Returns:
        List of evaluation results from all plugins
    """
    if not is_evaluator_enabled():
        return []

    plugins = get_evaluator_plugins()
    results: list[EvaluationResult] = []
    for plugin in plugins:
        start_time = time.perf_counter()
        try:
            result = await plugin.evaluate_a2a_result(context)
            duration_ms = (time.perf_counter() - start_time) * 1000
            add_evaluation_attributes(
                context.span, plugin.name, result, duration_ms, "a2a"
            )
            results.append(result)
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_result = EvaluationResult(status="error", error=str(e))
            add_evaluation_attributes(
                context.span, plugin.name, error_result, duration_ms, "a2a"
            )
            results.append(error_result)
            logger.warning(f"Evaluator {plugin.name} failed for A2A: {e}")

    return results
