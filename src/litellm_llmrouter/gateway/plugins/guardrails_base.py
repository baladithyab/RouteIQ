"""
Content Security Guardrails Base Framework
============================================

Provides the base class, decision model, and exception types for all guardrail
plugins.  Guardrails inspect and optionally transform LLM request/response
content to enforce content-security policies (OWASP LLM01-LLM06).

Guardrail plugins hook into the LiteLLM callback lifecycle via
``PluginCallbackBridge``:

    on_llm_pre_call   -> input guardrails  (injection, PII, jailbreak)
    on_llm_success    -> output guardrails (toxicity, PII leakage)

Blocking mechanism:
    A guardrail raises ``GuardrailBlockError`` from ``on_llm_pre_call``.
    The callback bridge lets it propagate so LiteLLM returns HTTP 400.

OTel attributes (``guardrail.*`` namespace):
    guardrail.name           - Plugin name
    guardrail.action         - Action taken (pass/block/redact/warn/log)
    guardrail.category       - Detection category (injection/pii/toxicity/...)
    guardrail.score          - Confidence score 0.0-1.0
    guardrail.evaluation_ms  - Evaluation latency

Configuration:
    Each guardrail plugin reads its own ``GUARDRAIL_<NAME>_*`` env vars.
"""

from __future__ import annotations

import logging
import os
import time
from abc import abstractmethod
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

# OTel attribute names for guardrail spans
ATTR_GUARDRAIL_NAME = "guardrail.name"
ATTR_GUARDRAIL_ACTION = "guardrail.action"
ATTR_GUARDRAIL_CATEGORY = "guardrail.category"
ATTR_GUARDRAIL_SCORE = "guardrail.score"
ATTR_GUARDRAIL_EVALUATION_MS = "guardrail.evaluation_ms"
ATTR_GUARDRAIL_DETAILS = "guardrail.details"

# OTel imports - optional
try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]


def record_guardrail_check_metric(check_type: str, action: str) -> None:
    """Emit the guardrail.check metric (best-effort, shared by all guardrails).

    Both ``GuardrailPlugin`` subclasses and the standalone guardrail plugins
    (Bedrock, content filter, LlamaGuard) record through this single helper so
    the ``check_type`` / ``action`` vocabulary stays consistent.

    Telemetry must never raise: a missing meter (OTel disabled) is a no-op and
    any recording error is swallowed.

    Args:
        check_type: Detection category / check type (low cardinality).
        action: ``pass`` / ``deny`` / ``redact`` / ``warn`` / ``log``.
    """
    try:
        from litellm_llmrouter.metrics import get_gateway_metrics

        m = get_gateway_metrics()
        if m is not None:
            m.record_guardrail_check(check_type, action)
    except Exception:  # pragma: no cover - telemetry must not break flow
        pass


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class GuardrailBlockError(Exception):
    """Raised when a guardrail blocks a request.

    This exception is deliberately *not* caught by the generic error handler in
    ``PluginCallbackBridge.async_log_pre_api_call`` so that it propagates to
    LiteLLM and results in an HTTP 400 response to the caller.
    """

    def __init__(
        self,
        guardrail_name: str,
        category: str,
        message: str,
        score: float = 1.0,
    ) -> None:
        self.guardrail_name = guardrail_name
        self.category = category
        self.score = score
        super().__init__(message)


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardrailDecision:
    """Immutable record of a single guardrail evaluation."""

    allowed: bool
    action_taken: str  # "pass", "block", "redact", "warn", "log"
    guardrail_name: str
    category: str  # "injection", "pii", "toxicity", "jailbreak", "custom"
    score: float  # 0.0-1.0 confidence
    details: dict[str, Any] = field(default_factory=dict)
    modified_content: str | None = None
    evaluation_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class GuardrailPlugin(GatewayPlugin):
    """Base class for guardrail plugins.

    Subclasses implement ``evaluate_input`` and/or ``evaluate_output``.
    The base class provides:

    * Standardised OTel span attribute emission
    * Action dispatching (block / redact / warn / log)
    * Feature-flag checking via env var
    """

    # Subclasses set these in __init__
    _enabled: bool = False
    _action: str = "block"

    # ----- metadata (subclasses override) ----

    @property
    def metadata(self) -> PluginMetadata:  # pragma: no cover
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            capabilities={PluginCapability.GUARDRAIL},
            priority=50,
            description="Guardrail plugin",
        )

    # ----- lifecycle ----

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        if context and context.logger:
            context.logger.info(f"Guardrail plugin {self.name} starting")

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        pass

    # ----- abstract evaluation hooks ----

    @abstractmethod
    async def evaluate_input(
        self,
        model: str,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> GuardrailDecision:
        """Evaluate inbound messages. Called from ``on_llm_pre_call``."""
        ...

    async def evaluate_output(
        self,
        model: str,
        response: Any,
        kwargs: dict[str, Any],
    ) -> GuardrailDecision | None:
        """Evaluate outbound response. Called from ``on_llm_success``.

        Default implementation is a no-op; override to add output scanning.
        """
        return None

    # ----- LLM lifecycle hooks (final, delegate to evaluate_*) ----

    async def on_llm_pre_call(
        self,
        model: str,
        messages: list[Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self._enabled:
            return None

        start = time.perf_counter()
        decision = await self.evaluate_input(model, messages, kwargs)
        decision = GuardrailDecision(
            allowed=decision.allowed,
            action_taken=decision.action_taken,
            guardrail_name=decision.guardrail_name,
            category=decision.category,
            score=decision.score,
            details=decision.details,
            modified_content=decision.modified_content,
            evaluation_time_ms=(time.perf_counter() - start) * 1000,
        )
        self._emit_otel_attributes(decision)
        self._record_check_metric(decision)
        self._log_decision(decision)

        if not decision.allowed and decision.action_taken == "block":
            raise GuardrailBlockError(
                guardrail_name=decision.guardrail_name,
                category=decision.category,
                message=decision.details.get("reason", "Blocked by guardrail"),
                score=decision.score,
            )

        # If the subclass modified messages (e.g. PII redaction), propagate
        if decision.modified_content is not None:
            return {"messages": messages}
        return None

    async def on_llm_success(
        self,
        model: str,
        response: Any,
        kwargs: dict[str, Any],
    ) -> None:
        if not self._enabled:
            return

        start = time.perf_counter()
        decision = await self.evaluate_output(model, response, kwargs)
        if decision is None:
            return

        decision = GuardrailDecision(
            allowed=decision.allowed,
            action_taken=decision.action_taken,
            guardrail_name=decision.guardrail_name,
            category=decision.category,
            score=decision.score,
            details=decision.details,
            modified_content=decision.modified_content,
            evaluation_time_ms=(time.perf_counter() - start) * 1000,
        )
        self._emit_otel_attributes(decision)
        self._record_check_metric(decision)
        self._log_decision(decision)

    # ----- helpers ----

    @staticmethod
    def _env_bool(key: str, default: bool = False) -> bool:
        return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

    @staticmethod
    def _env_str(key: str, default: str = "") -> str:
        return os.getenv(key, default)

    def _emit_otel_attributes(self, decision: GuardrailDecision) -> None:
        if not OTEL_AVAILABLE or trace is None:
            return
        span = trace.get_current_span()
        if span is None or not hasattr(span, "set_attribute"):
            return
        try:
            span.set_attribute(ATTR_GUARDRAIL_NAME, decision.guardrail_name)
            span.set_attribute(ATTR_GUARDRAIL_ACTION, decision.action_taken)
            span.set_attribute(ATTR_GUARDRAIL_CATEGORY, decision.category)
            span.set_attribute(ATTR_GUARDRAIL_SCORE, decision.score)
            span.set_attribute(
                ATTR_GUARDRAIL_EVALUATION_MS, decision.evaluation_time_ms
            )
        except Exception:
            pass

    def _record_check_metric(self, decision: GuardrailDecision) -> None:
        """Emit the guardrail.check metric for this decision (best-effort).

        Uses the detection ``category`` as the (low-cardinality) check type and
        normalises ``block`` -> ``deny`` so the action vocabulary matches the
        config-driven guardrail-policy engine. Telemetry must never raise: a
        missing meter (OTel disabled) is a no-op and errors are swallowed.
        """
        action = "deny" if decision.action_taken == "block" else decision.action_taken
        record_guardrail_check_metric(decision.category, action)

    def _log_decision(self, decision: GuardrailDecision) -> None:
        if decision.action_taken == "pass":
            return
        level = logging.WARNING if decision.action_taken == "block" else logging.INFO
        logger.log(
            level,
            "Guardrail %s: action=%s category=%s score=%.2f details=%s",
            decision.guardrail_name,
            decision.action_taken,
            decision.category,
            decision.score,
            decision.details,
        )
