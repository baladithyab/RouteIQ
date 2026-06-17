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

        # Current circuit-breaker state as a gauge-like UpDownCounter
        # (per-breaker). Value is 1 for the breaker's current state and 0 for
        # the others, keyed by the ``state`` label (closed/open/half_open) so a
        # PromQL ``max by (breaker) (gateway_circuit_breaker_state * on(state) ...)``
        # / KEDA scaler can read the live state without scraping logs.
        self.circuit_breaker_state: UpDownCounter = meter.create_up_down_counter(
            name="gateway.circuit_breaker.state",
            description=(
                "Current circuit breaker state (1=active state, 0=inactive) "
                "labelled by breaker + state"
            ),
            unit="{breaker}",
        )

        # In-flight (active) requests tracked by the backpressure middleware /
        # drain manager. KEDA scales on this gauge-like UpDownCounter.
        self.backpressure_active_requests: UpDownCounter = meter.create_up_down_counter(
            name="gateway.backpressure.active_requests",
            description="Number of in-flight requests tracked by backpressure",
            unit="{request}",
        )

        # Backpressure rejections (503 over-capacity / drain). Monotonic counter
        # so the Prometheus name gets the ``_total`` suffix.
        self.backpressure_rejections: Counter = meter.create_counter(
            name="gateway.backpressure.rejections",
            description="Requests rejected by backpressure (over capacity / draining)",
            unit="{request}",
        )

        # =================================================================
        # Dark-subsystem Instruments (metrics-2)
        # =================================================================
        #
        # Low-cardinality instruments for subsystems that previously emitted
        # no metrics. Labels are deliberately coarse (strategy / model name,
        # enforcement reason, check type, hit/miss) -- never request ids or
        # raw user text.

        self.routing_selection: Counter = meter.create_counter(
            name="gateway.routing.selection",
            description="Model selections made by a routing strategy",
            unit="{selection}",
        )

        self.governance_denial: Counter = meter.create_counter(
            name="gateway.governance.enforcement.denial",
            description="Governance enforcement denials by reason",
            unit="{denial}",
        )

        self.guardrail_check: Counter = meter.create_counter(
            name="gateway.guardrail.check",
            description="Guardrail checks by check type and action",
            unit="{check}",
        )

        self.semantic_cache_lookup: Counter = meter.create_counter(
            name="gateway.semantic_cache.lookup",
            description="Semantic cache lookups by result (hit/miss)",
            unit="{lookup}",
        )

        self.context_optimizer_tokens_saved: Histogram = meter.create_histogram(
            name="gateway.context_optimizer.tokens_saved",
            description="Estimated tokens saved per context-optimizer pass",
            unit="{token}",
            explicit_bucket_boundaries_advisory=TOKEN_BUCKETS,
        )

        self.mcp_tool_invocation: Counter = meter.create_counter(
            name="gateway.mcp.tool.invocation",
            description="MCP tool invocations by result (success/error)",
            unit="{invocation}",
        )

        self.a2a_invocation: Counter = meter.create_counter(
            name="gateway.a2a.invocation",
            description="A2A agent invocations by result (success/error)",
            unit="{invocation}",
        )

        self.eval_sample: Counter = meter.create_counter(
            name="gateway.eval.sample",
            description="Evaluation samples scored by verdict (pass/fail)",
            unit="{sample}",
        )

        # =================================================================
        # MLOps Drift Instruments (Cluster H, RouteIQ-6dce)
        # =================================================================
        #
        # Gauge-like UpDownCounters for the drift detector. Because this
        # codebase's no-op OTel adapter does not implement observable gauges,
        # we model each gauge with an UpDownCounter delta: the detector tracks
        # the last-emitted value per key and pushes the signed delta so a
        # Prometheus/CloudWatch query reads the live value. Labels are coarse
        # (signal kind only) -- never request ids or raw user text.

        self.mlops_input_drift_score: UpDownCounter = meter.create_up_down_counter(
            name="gateway.mlops.input_drift.score",
            description=(
                "Current input-distribution drift score (population stability "
                "index) vs the captured baseline"
            ),
            unit="1",
        )

        self.mlops_quality_regression: UpDownCounter = meter.create_up_down_counter(
            name="gateway.mlops.quality_regression.delta",
            description=(
                "Current routing-quality regression (baseline - current "
                "aggregated quality, in [0,1]); positive means quality dropped"
            ),
            unit="1",
        )

        self.mlops_drift_signal: Counter = meter.create_counter(
            name="gateway.mlops.drift.signal",
            description=(
                "MLOps drift signals fired by kind (input_drift / quality_regression)"
            ),
            unit="{signal}",
        )

        # =================================================================
        # MLOps Promotion Instruments (Cluster H, RouteIQ-2a1c)
        # =================================================================

        self.mlops_promotion: Counter = meter.create_counter(
            name="gateway.mlops.promotion",
            description=(
                "Champion/challenger promotion-loop actions by action "
                "(promote / rollback / hold)"
            ),
            unit="{action}",
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

    # All three possible circuit-breaker states, kept module-coupled-free so we
    # don't import the enum from resilience.py (avoids an import cycle).
    _CB_STATES = ("closed", "open", "half_open")

    def record_circuit_breaker_state(
        self, breaker_name: str, from_state: str, to_state: str
    ) -> None:
        """Set the per-breaker state gauge.

        Emits the breaker's current state as a 0/1 gauge-like signal: the
        ``to_state`` series is driven to 1 and the previous ``from_state``
        series back to 0. Because OTel has no push-style synchronous gauge in
        this codebase (the no-op adapter doesn't implement observable gauges),
        we model the gauge with an UpDownCounter delta so a Prometheus / KEDA
        query can ``max by (breaker)`` over the ``state`` label.

        No-op when ``from_state == to_state``.
        """
        if from_state == to_state:
            return
        # Drop the previous state's series to 0 (only if it was a known state).
        if from_state in self._CB_STATES:
            self.circuit_breaker_state.add(
                -1, {"breaker": breaker_name, "state": from_state}
            )
        # Drive the new state's series to 1.
        if to_state in self._CB_STATES:
            self.circuit_breaker_state.add(
                1, {"breaker": breaker_name, "state": to_state}
            )

    def inc_backpressure_active(self, delta: int = 1) -> None:
        """Adjust the in-flight (active) request gauge by ``delta``.

        Called with ``+1`` when a request enters the backpressure-tracked path
        and ``-1`` when it completes (including streamed responses).
        """
        if delta:
            self.backpressure_active_requests.add(delta)

    def record_backpressure_rejection(self, reason: str = "over_capacity") -> None:
        """Record a request rejected by backpressure.

        Args:
            reason: ``over_capacity`` (semaphore exhausted) or ``draining``.
        """
        self.backpressure_rejections.add(1, {"reason": reason})

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

    # ----- dark-subsystem record helpers (metrics-2) -----

    def record_routing_selection(self, strategy: str, model: str) -> None:
        """Record a model selection by a routing strategy (any family)."""
        self.routing_selection.add(1, {"strategy": strategy, "model": model})

    def record_governance_denial(self, reason: str) -> None:
        """Record a governance enforcement denial.

        Args:
            reason: One of ``budget`` / ``rate_limit`` / ``model_access``.
        """
        self.governance_denial.add(1, {"reason": reason})

    def record_guardrail_check(self, check_type: str, action: str) -> None:
        """Record a guardrail check outcome.

        Args:
            check_type: The guardrail category/check type (low cardinality).
            action: The action taken (``pass`` / ``deny`` / ``redact`` / ...).
        """
        self.guardrail_check.add(1, {"check_type": check_type, "action": action})

    def record_semantic_cache_lookup(self, result: str) -> None:
        """Record a semantic cache lookup.

        Args:
            result: ``hit`` or ``miss``.
        """
        self.semantic_cache_lookup.add(1, {"result": result})

    def record_context_optimizer_tokens_saved(self, tokens_saved: int) -> None:
        """Record the estimated tokens saved by one context-optimizer pass."""
        self.context_optimizer_tokens_saved.record(tokens_saved)

    def record_mcp_tool_invocation(self, result: str) -> None:
        """Record an MCP tool invocation.

        Args:
            result: ``success`` or ``error``.
        """
        self.mcp_tool_invocation.add(1, {"result": result})

    def record_a2a_invocation(self, result: str) -> None:
        """Record an A2A agent invocation.

        Args:
            result: ``success`` or ``error``.
        """
        self.a2a_invocation.add(1, {"result": result})

    def record_eval_sample(self, verdict: str) -> None:
        """Record an evaluation sample scored by the judge.

        Args:
            verdict: ``pass`` or ``fail``.
        """
        self.eval_sample.add(1, {"verdict": verdict})

    # ----- MLOps drift / promotion record helpers (Cluster H) -----

    def set_input_drift_score(self, current: float, previous: float = 0.0) -> None:
        """Set the input-drift score gauge.

        Modeled as an UpDownCounter delta (``current - previous``) because the
        no-op OTel adapter lacks an observable gauge. Callers track the
        last-emitted value and pass it as ``previous``.
        """
        delta = current - previous
        if delta:
            self.mlops_input_drift_score.add(delta)

    def set_quality_regression(self, current: float, previous: float = 0.0) -> None:
        """Set the routing-quality-regression gauge (delta-modeled)."""
        delta = current - previous
        if delta:
            self.mlops_quality_regression.add(delta)

    def record_drift_signal(self, kind: str) -> None:
        """Record a fired drift signal.

        Args:
            kind: ``input_drift`` or ``quality_regression``.
        """
        self.mlops_drift_signal.add(1, {"kind": kind})

    def record_promotion_action(self, action: str) -> None:
        """Record a champion/challenger promotion-loop action.

        Args:
            action: ``promote`` / ``rollback`` / ``hold``.
        """
        self.mlops_promotion.add(1, {"action": action})


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
