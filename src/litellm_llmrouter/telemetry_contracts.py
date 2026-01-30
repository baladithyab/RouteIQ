"""
Versioned Telemetry Contracts for LLMRouter
=============================================

This module defines versioned telemetry contracts for routing decisions,
ensuring a stable, documented schema for MLOps extraction and observability.

Contract: routeiq.router_decision.v1
------------------------------------
Emitted as an OTEL span event with a single JSON payload attribute.
This provides a stable interface for telemetry consumers (Jaeger, MLflow, etc.)
while allowing internal implementation to evolve.

Security:
- No PII (prompts, responses) are included in the contract
- Only query_length is logged to protect user privacy
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# Contract version - increment on breaking changes
CONTRACT_VERSION = "v1"
CONTRACT_NAME = "routeiq.router_decision"
CONTRACT_FULL_NAME = f"{CONTRACT_NAME}.{CONTRACT_VERSION}"


class RoutingOutcome(str, Enum):
    """Outcome of a routing decision."""

    SUCCESS = "success"
    FAILURE = "failure"
    FALLBACK = "fallback"
    ERROR = "error"
    NO_CANDIDATES = "no_candidates"
    TIMEOUT = "timeout"


@dataclass
class RouterDecisionInput:
    """Input context for a routing decision (PII-safe)."""

    requested_model: Optional[str] = None
    """The model name requested by the caller (if any)."""

    query_length: int = 0
    """Length of the query in characters (no content logged for PII safety)."""

    user_id: Optional[str] = None
    """User identifier (hashed/anonymized recommended)."""

    team_id: Optional[str] = None
    """Team identifier for multi-tenant scenarios."""

    request_metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional safe metadata (no PII)."""


@dataclass
class CandidateDeployment:
    """A candidate deployment considered during routing."""

    model_name: str
    """Name/identifier of the model deployment."""

    provider: Optional[str] = None
    """Provider (openai, anthropic, bedrock, etc.)."""

    score: Optional[float] = None
    """Routing score assigned by the strategy (if applicable)."""

    available: bool = True
    """Whether the deployment was available at decision time."""


@dataclass
class RoutingTimings:
    """Latency breakdown for routing decision."""

    total_ms: float = 0.0
    """Total routing decision time in milliseconds."""

    strategy_ms: Optional[float] = None
    """Time spent in routing strategy (ML inference, etc.)."""

    embedding_ms: Optional[float] = None
    """Time spent generating embeddings (if applicable)."""

    candidate_filter_ms: Optional[float] = None
    """Time spent filtering candidates."""


@dataclass
class RoutingOutcomeData:
    """Outcome details after routing completion."""

    status: RoutingOutcome = RoutingOutcome.SUCCESS
    """Final outcome status."""

    error_message: Optional[str] = None
    """Error message if status is error/failure."""

    error_type: Optional[str] = None
    """Error type/class name if applicable."""

    input_tokens: Optional[int] = None
    """Input tokens used (if available after completion)."""

    output_tokens: Optional[int] = None
    """Output tokens used (if available after completion)."""

    total_tokens: Optional[int] = None
    """Total tokens used (if available after completion)."""


@dataclass
class FallbackInfo:
    """Information about fallback behavior."""

    fallback_triggered: bool = False
    """Whether fallback to another model was triggered."""

    original_model: Optional[str] = None
    """Original model that failed (if fallback triggered)."""

    fallback_reason: Optional[str] = None
    """Reason for fallback (rate_limit, error, timeout, etc.)."""

    fallback_attempt: int = 0
    """Which fallback attempt this is (0 = primary, 1+ = fallback)."""


@dataclass
class RouterDecisionEvent:
    """
    Complete versioned routing telemetry event.

    This is the main contract class that should be emitted as a span event.
    All fields are PII-safe and designed for MLOps consumption.
    """

    # === Identity & Versioning ===
    contract_version: str = CONTRACT_VERSION
    """Contract version for schema evolution."""

    contract_name: str = CONTRACT_FULL_NAME
    """Full contract name (routeiq.router_decision.v1)."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this event."""

    # === Trace Context ===
    trace_id: Optional[str] = None
    """OpenTelemetry trace ID."""

    span_id: Optional[str] = None
    """OpenTelemetry span ID."""

    parent_span_id: Optional[str] = None
    """Parent span ID if this is a child span."""

    # === Timestamps ===
    timestamp_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    )
    """ISO 8601 UTC timestamp."""

    timestamp_unix_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    """Unix timestamp in milliseconds."""

    # === Input Context ===
    input: RouterDecisionInput = field(default_factory=RouterDecisionInput)
    """Input context for the routing decision."""

    # === Routing Decision ===
    strategy_name: str = ""
    """Name of the routing strategy used (e.g., llmrouter-knn)."""

    strategy_version: Optional[str] = None
    """Version of the routing strategy/model."""

    candidate_deployments: List[CandidateDeployment] = field(default_factory=list)
    """List of candidate deployments considered."""

    selected_deployment: Optional[str] = None
    """Model/deployment that was selected."""

    selection_reason: Optional[str] = None
    """Human-readable reason for selection."""

    # === Performance Metrics ===
    timings: RoutingTimings = field(default_factory=RoutingTimings)
    """Latency breakdown."""

    # === Outcome ===
    outcome: RoutingOutcomeData = field(default_factory=RoutingOutcomeData)
    """Final outcome after routing/completion."""

    # === Fallback Information ===
    fallback: FallbackInfo = field(default_factory=FallbackInfo)
    """Fallback behavior details."""

    # === Custom Metadata ===
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    """Custom attributes for extension."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string for span event attribute."""
        return json.dumps(self.to_dict(), default=str)


class RouterDecisionEventBuilder:
    """
    Builder for constructing RouterDecisionEvent instances.

    Provides a fluent API for building routing telemetry events
    with proper defaults and validation.

    Usage:
        event = (RouterDecisionEventBuilder()
            .with_strategy("llmrouter-knn", version="1.0")
            .with_input(query_length=100)
            .with_selection(selected="gpt-4")
            .with_timing(total_ms=10.5)
            .with_outcome(RoutingOutcome.SUCCESS)
            .build())
    """

    def __init__(self):
        self._event = RouterDecisionEvent()

    def with_strategy(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> "RouterDecisionEventBuilder":
        """Set strategy name and version."""
        self._event.strategy_name = name
        self._event.strategy_version = version
        return self

    def with_trace_context(
        self,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> "RouterDecisionEventBuilder":
        """Set trace context from OTEL."""
        self._event.trace_id = trace_id
        self._event.span_id = span_id
        self._event.parent_span_id = parent_span_id
        return self

    def with_input(
        self,
        query_length: int = 0,
        requested_model: Optional[str] = None,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RouterDecisionEventBuilder":
        """Set input context (PII-safe)."""
        self._event.input = RouterDecisionInput(
            requested_model=requested_model,
            query_length=query_length,
            user_id=user_id,
            team_id=team_id,
            request_metadata=metadata or {},
        )
        return self

    def with_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> "RouterDecisionEventBuilder":
        """
        Set candidate deployments.

        Args:
            candidates: List of dicts with model_name, provider, score, available
        """
        self._event.candidate_deployments = [
            CandidateDeployment(
                model_name=c.get("model_name", c.get("name", "")),
                provider=c.get("provider"),
                score=c.get("score"),
                available=c.get("available", True),
            )
            for c in candidates
        ]
        return self

    def with_selection(
        self,
        selected: Optional[str],
        reason: Optional[str] = None,
    ) -> "RouterDecisionEventBuilder":
        """Set the selected deployment."""
        self._event.selected_deployment = selected
        self._event.selection_reason = reason
        return self

    def with_timing(
        self,
        total_ms: float,
        strategy_ms: Optional[float] = None,
        embedding_ms: Optional[float] = None,
        candidate_filter_ms: Optional[float] = None,
    ) -> "RouterDecisionEventBuilder":
        """Set timing breakdown."""
        self._event.timings = RoutingTimings(
            total_ms=total_ms,
            strategy_ms=strategy_ms,
            embedding_ms=embedding_ms,
            candidate_filter_ms=candidate_filter_ms,
        )
        return self

    def with_outcome(
        self,
        status: RoutingOutcome,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> "RouterDecisionEventBuilder":
        """Set outcome data."""
        self._event.outcome = RoutingOutcomeData(
            status=status,
            error_message=error_message,
            error_type=error_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        return self

    def with_fallback(
        self,
        triggered: bool = False,
        original_model: Optional[str] = None,
        reason: Optional[str] = None,
        attempt: int = 0,
    ) -> "RouterDecisionEventBuilder":
        """Set fallback information."""
        self._event.fallback = FallbackInfo(
            fallback_triggered=triggered,
            original_model=original_model,
            fallback_reason=reason,
            fallback_attempt=attempt,
        )
        return self

    def with_custom_attributes(
        self,
        attributes: Dict[str, Any],
    ) -> "RouterDecisionEventBuilder":
        """Add custom attributes."""
        self._event.custom_attributes.update(attributes)
        return self

    def build(self) -> RouterDecisionEvent:
        """Build the final event."""
        return self._event


def extract_router_decision_from_span_event(
    event_attributes: Dict[str, Any],
) -> Optional[RouterDecisionEvent]:
    """
    Extract a RouterDecisionEvent from span event attributes.

    This is the reverse operation used by MLOps extraction scripts
    to parse the emitted telemetry.

    Args:
        event_attributes: Dictionary of span event attributes

    Returns:
        RouterDecisionEvent if valid, None otherwise
    """
    # Check for the JSON payload attribute
    payload = event_attributes.get("routeiq.router_decision.payload")
    if not payload:
        return None

    try:
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload

        # Validate contract version
        if data.get("contract_name") != CONTRACT_FULL_NAME:
            # Future: handle version migration
            return None

        # Reconstruct the event
        event = RouterDecisionEvent(
            contract_version=data.get("contract_version", CONTRACT_VERSION),
            contract_name=data.get("contract_name", CONTRACT_FULL_NAME),
            event_id=data.get("event_id", ""),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            parent_span_id=data.get("parent_span_id"),
            timestamp_utc=data.get("timestamp_utc", ""),
            timestamp_unix_ms=data.get("timestamp_unix_ms", 0),
            strategy_name=data.get("strategy_name", ""),
            strategy_version=data.get("strategy_version"),
            selected_deployment=data.get("selected_deployment"),
            selection_reason=data.get("selection_reason"),
            custom_attributes=data.get("custom_attributes", {}),
        )

        # Parse nested objects
        if "input" in data:
            inp = data["input"]
            event.input = RouterDecisionInput(
                requested_model=inp.get("requested_model"),
                query_length=inp.get("query_length", 0),
                user_id=inp.get("user_id"),
                team_id=inp.get("team_id"),
                request_metadata=inp.get("request_metadata", {}),
            )

        if "candidate_deployments" in data:
            event.candidate_deployments = [
                CandidateDeployment(
                    model_name=c.get("model_name", ""),
                    provider=c.get("provider"),
                    score=c.get("score"),
                    available=c.get("available", True),
                )
                for c in data["candidate_deployments"]
            ]

        if "timings" in data:
            t = data["timings"]
            event.timings = RoutingTimings(
                total_ms=t.get("total_ms", 0.0),
                strategy_ms=t.get("strategy_ms"),
                embedding_ms=t.get("embedding_ms"),
                candidate_filter_ms=t.get("candidate_filter_ms"),
            )

        if "outcome" in data:
            o = data["outcome"]
            event.outcome = RoutingOutcomeData(
                status=RoutingOutcome(o.get("status", "success")),
                error_message=o.get("error_message"),
                error_type=o.get("error_type"),
                input_tokens=o.get("input_tokens"),
                output_tokens=o.get("output_tokens"),
                total_tokens=o.get("total_tokens"),
            )

        if "fallback" in data:
            f = data["fallback"]
            event.fallback = FallbackInfo(
                fallback_triggered=f.get("fallback_triggered", False),
                original_model=f.get("original_model"),
                fallback_reason=f.get("fallback_reason"),
                fallback_attempt=f.get("fallback_attempt", 0),
            )

        return event

    except (json.JSONDecodeError, KeyError, TypeError):
        # Invalid payload, return None for backward compatibility
        return None


# Span event name constant for consumers
ROUTER_DECISION_EVENT_NAME = CONTRACT_FULL_NAME
ROUTER_DECISION_PAYLOAD_KEY = f"{CONTRACT_NAME}.payload"
