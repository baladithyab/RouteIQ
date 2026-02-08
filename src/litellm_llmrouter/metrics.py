"""
OTel Metrics Instrument Registry
=================================

Central registry for all OpenTelemetry metric instruments used by RouteIQ Gateway.

Instruments are created once during initialization and recorded at runtime via
the LiteLLM callback interface (RouterDecisionCallback) and middleware.

Follows OpenTelemetry GenAI Semantic Conventions for gen_ai.* metrics and
adds gateway-specific operational metrics under the gateway.* namespace.

Usage:
    from litellm_llmrouter.metrics import get_gateway_metrics, init_gateway_metrics

    # During startup (called by ObservabilityManager._init_metrics):
    meter = metrics.get_meter(__name__, version)
    init_gateway_metrics(meter)

    # At runtime:
    m = get_gateway_metrics()
    if m:
        m.request_duration.record(1.23, {"gen_ai.request.model": "gpt-4"})
"""

import logging
from typing import Optional

from opentelemetry.metrics import (
    Counter,
    Histogram,
    Meter,
    UpDownCounter,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Histogram Bucket Boundaries (OpenTelemetry GenAI Semantic Conventions)
# =============================================================================

# Duration metrics (seconds): covers 10ms to ~82s
DURATION_BUCKETS = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)

# Token usage: covers 1 to ~4M tokens
TOKEN_BUCKETS = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
)

# TTFT (seconds): covers 10ms to ~20s
TTFT_BUCKETS = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
)

# Tokens per second (throughput): covers 1 to ~500 tok/s
TOKENS_PER_SECOND_BUCKETS = (
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
)

# Routing decision duration (seconds): covers 100us to ~1s
ROUTING_DURATION_BUCKETS = (
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
)

# Cost buckets (USD): covers $0.0001 to $10
COST_BUCKETS = (
    0.0001,
    0.001,
    0.01,
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
)


class GatewayMetrics:
    """
    Central registry of all OTel metric instruments for RouteIQ Gateway.

    All instruments are created at init time from a provided Meter instance.
    Callers record observations at runtime by accessing the instrument attributes.

    Instruments follow two namespaces:
    - gen_ai.*: OpenTelemetry GenAI Semantic Conventions
    - gateway.*: RouteIQ-specific operational metrics
    """

    def __init__(self, meter: Meter) -> None:
        """
        Create all metric instruments from the given Meter.

        Args:
            meter: An OpenTelemetry Meter instance from the MeterProvider.
        """
        self._meter = meter

        # =================================================================
        # GenAI Semantic Convention Metrics
        # =================================================================

        self.request_duration: Histogram = meter.create_histogram(
            name="gen_ai.client.operation.duration",
            description="Duration of GenAI client operations",
            unit="s",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
        )

        self.token_usage: Histogram = meter.create_histogram(
            name="gen_ai.client.token.usage",
            description="Token usage per GenAI request",
            unit="{token}",
            explicit_bucket_boundaries_advisory=TOKEN_BUCKETS,
        )

        self.time_to_first_token: Histogram = meter.create_histogram(
            name="gen_ai.server.time_to_first_token",
            description="Time to first token for streaming responses",
            unit="s",
            explicit_bucket_boundaries_advisory=TTFT_BUCKETS,
        )

        self.tokens_per_second: Histogram = meter.create_histogram(
            name="gen_ai.client.tokens_per_second",
            description="Token generation throughput",
            unit="{token}/s",
            explicit_bucket_boundaries_advisory=TOKENS_PER_SECOND_BUCKETS,
        )

        # =================================================================
        # Gateway Operational Metrics
        # =================================================================

        self.request_total: Counter = meter.create_counter(
            name="gateway.request.total",
            description="Total gateway requests",
            unit="{request}",
        )

        self.request_error: Counter = meter.create_counter(
            name="gateway.request.error",
            description="Total gateway request errors",
            unit="{request}",
        )

        self.request_active: UpDownCounter = meter.create_up_down_counter(
            name="gateway.request.active",
            description="Number of currently active requests",
            unit="{request}",
        )

        # =================================================================
        # Routing Metrics
        # =================================================================

        self.routing_decision_duration: Histogram = meter.create_histogram(
            name="gateway.routing.decision.duration",
            description="Duration of routing decisions",
            unit="s",
            explicit_bucket_boundaries_advisory=ROUTING_DURATION_BUCKETS,
        )

        self.routing_strategy_usage: Counter = meter.create_counter(
            name="gateway.routing.strategy.usage",
            description="Routing strategy usage count",
            unit="{decision}",
        )

        # =================================================================
        # Cost Tracking
        # =================================================================

        self.cost_total: Counter = meter.create_counter(
            name="gateway.cost.total",
            description="Total estimated cost in USD",
            unit="USD",
        )

        self.cost_per_request: Histogram = meter.create_histogram(
            name="gateway.cost.per_request",
            description="Cost per LLM request in USD",
            unit="USD",
        )

        self.tokens_total: Counter = meter.create_counter(
            name="gateway.tokens.total",
            description="Total tokens consumed across all requests",
            unit="{token}",
        )

        self.cost_errors: Counter = meter.create_counter(
            name="gateway.cost.errors",
            description="Errors during cost calculation",
            unit="{error}",
        )

        # =================================================================
        # Cost Reconciliation
        # =================================================================

        self.reconciliation_savings: Histogram = meter.create_histogram(
            name="quota.reconciliation.savings",
            description="Amount credited back by post-call cost reconciliation",
            unit="USD",
            explicit_bucket_boundaries_advisory=COST_BUCKETS,
        )

        self.reconciliation_count: Counter = meter.create_counter(
            name="quota.reconciliation.count",
            description="Number of post-call cost reconciliations performed",
            unit="{reconciliation}",
        )

        # =================================================================
        # Resilience Metrics
        # =================================================================

        self.circuit_breaker_transitions: Counter = meter.create_counter(
            name="gateway.circuit_breaker.transitions",
            description="Circuit breaker state transitions",
            unit="{transition}",
        )

        logger.info("GatewayMetrics: all instruments created")

    # Aliases for backward compatibility
    @property
    def strategy_usage(self) -> Counter:
        """Alias for routing_strategy_usage (used by RouterDecisionMiddleware)."""
        return self.routing_strategy_usage

    # =================================================================
    # Convenience recording methods
    # =================================================================

    def record_routing_decision(
        self,
        strategy: str,
        model: str,
        duration_s: float,
        outcome: str = "success",
    ) -> None:
        """Record a routing decision with all related metrics."""
        self.routing_decision_duration.record(
            duration_s, {"strategy": strategy, "model": model}
        )
        self.routing_strategy_usage.add(1, {"strategy": strategy, "outcome": outcome})

    def record_circuit_breaker_transition(
        self, breaker_name: str, from_state: str, to_state: str
    ) -> None:
        """Record a circuit breaker state transition."""
        self.circuit_breaker_transitions.add(
            1,
            {
                "breaker": breaker_name,
                "from_state": from_state,
                "to_state": to_state,
            },
        )

    def record_streaming_metrics(
        self,
        ttft_ms: float,
        tps: float,
        total_tokens: int,
        attrs: dict | None = None,
    ) -> None:
        """
        Record streaming-specific metrics (TTFT, throughput, token usage).

        Args:
            ttft_ms: Time to first token in milliseconds.
            tps: Tokens per second throughput.
            total_tokens: Total tokens generated in the streaming response.
            attrs: Optional base attributes (e.g., model, provider).
        """
        base_attrs = attrs or {}
        self.time_to_first_token.record(
            ttft_ms / 1000.0, base_attrs
        )  # Convert ms to seconds
        if tps > 0:
            self.tokens_per_second.record(tps, base_attrs)
        if total_tokens > 0:
            self.token_usage.record(total_tokens, base_attrs)


# =============================================================================
# Module-level Singleton
# =============================================================================

_gateway_metrics: Optional[GatewayMetrics] = None


def init_gateway_metrics(meter: Meter) -> GatewayMetrics:
    """
    Initialize the global GatewayMetrics singleton.

    Called by ObservabilityManager._init_metrics() during startup.

    Args:
        meter: An OpenTelemetry Meter instance.

    Returns:
        The initialized GatewayMetrics instance.
    """
    global _gateway_metrics
    _gateway_metrics = GatewayMetrics(meter)
    return _gateway_metrics


def get_gateway_metrics() -> Optional[GatewayMetrics]:
    """
    Get the global GatewayMetrics singleton.

    Returns:
        The GatewayMetrics instance, or None if not initialized.
    """
    return _gateway_metrics


def reset_gateway_metrics() -> None:
    """
    Reset the global GatewayMetrics singleton.

    Must be called in test fixtures to avoid singleton leaks between tests.
    """
    global _gateway_metrics
    _gateway_metrics = None
