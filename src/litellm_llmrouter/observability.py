"""
OpenTelemetry Observability Integration
========================================

This module provides unified observability via OpenTelemetry:
- Distributed tracing for request flow
- Structured logging with trace correlation
- Metrics collection (via Prometheus)

The module integrates with LiteLLM's existing observability while adding
LLMRouter-specific instrumentation for routing decisions.

IMPORTANT: This module is designed to REUSE existing TracerProvider/MeterProvider
if one is already configured (e.g., by LiteLLM or FastAPI instrumentation).
This prevents "provider mismatch" issues where custom spans are exported to
a different provider than the one used by auto-instrumentation.
"""

import logging
import os
from typing import Any, Dict, Optional, cast

from opentelemetry import trace, metrics
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    Sampler,
    ALWAYS_ON,
    ALWAYS_OFF,
    TraceIdRatioBased,
    ParentBased,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

logger = logging.getLogger(__name__)

# ==============================================================================
# TG4.1: Router Decision Span Attributes
# ==============================================================================
# These span attribute keys align with the TG4.1 acceptance criteria for
# routing decision visibility in traces.
#
# Span Attribute Naming Convention:
# - Use 'router.' prefix for all routing-related attributes
# - Use snake_case for attribute names
# ==============================================================================

ROUTER_STRATEGY_ATTR = "router.strategy"
"""Strategy name used for routing (e.g., 'knn', 'mlp', 'random')."""

ROUTER_MODEL_SELECTED_ATTR = "router.model_selected"
"""Model/deployment that was selected by the router."""

ROUTER_SCORE_ATTR = "router.score"
"""Routing score for ML-based strategies (e.g., confidence score)."""

ROUTER_CANDIDATES_EVALUATED_ATTR = "router.candidates_evaluated"
"""Number of candidate models evaluated during routing."""

ROUTER_DECISION_OUTCOME_ATTR = "router.decision_outcome"
"""Outcome of the routing decision (success, failure, error, fallback, no_candidates)."""

ROUTER_DECISION_REASON_ATTR = "router.decision_reason"
"""Human-readable reason for the routing decision."""

ROUTER_LATENCY_MS_ATTR = "router.latency_ms"
"""Routing decision latency in milliseconds."""

ROUTER_ERROR_TYPE_ATTR = "router.error_type"
"""Error type if routing failed (exception class name)."""

ROUTER_ERROR_MESSAGE_ATTR = "router.error_message"
"""Error message if routing failed."""

ROUTER_VERSION_ATTR = "router.strategy_version"
"""Version of the routing strategy/model (e.g., model SHA256 prefix)."""

ROUTER_FALLBACK_TRIGGERED_ATTR = "router.fallback_triggered"
"""Whether fallback to another model was triggered."""


# ==============================================================================
# v0.2.0: Management Operation Span Attributes
# ==============================================================================

MGMT_OPERATION_ATTR = "routeiq.management.operation"
"""Classified management operation name (e.g., 'key.generate')."""

MGMT_RESOURCE_TYPE_ATTR = "routeiq.management.resource_type"
"""Resource category (e.g., 'key', 'team', 'model')."""

MGMT_SENSITIVITY_ATTR = "routeiq.management.sensitivity"
"""Operation sensitivity level ('read' or 'write')."""


def set_router_decision_attributes(
    span: trace.Span,
    *,
    strategy: Optional[str] = None,
    model_selected: Optional[str] = None,
    score: Optional[float] = None,
    candidates_evaluated: Optional[int] = None,
    outcome: Optional[str] = None,
    reason: Optional[str] = None,
    latency_ms: Optional[float] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    strategy_version: Optional[str] = None,
    fallback_triggered: Optional[bool] = None,
) -> None:
    """
    Set TG4.1 router decision span attributes on the given span.

    This function provides a centralized way to emit routing decision
    telemetry as first-class span attributes, enabling analysis of routing
    decisions in tracing backends (Jaeger, Tempo, etc.).

    Args:
        span: The OpenTelemetry span to add attributes to
        strategy: Routing strategy name (e.g., 'knn', 'mlp', 'random')
        model_selected: Model/deployment that was selected
        score: Routing score for ML-based strategies
        candidates_evaluated: Number of candidates evaluated
        outcome: Routing outcome (success, failure, error, fallback, no_candidates)
        reason: Human-readable reason for the decision
        latency_ms: Routing decision latency in milliseconds
        error_type: Error type if routing failed
        error_message: Error message if routing failed
        strategy_version: Version of the routing strategy/model
        fallback_triggered: Whether fallback was triggered

    Example:
        with tracer.start_as_current_span("routing.decision") as span:
            # ... perform routing ...
            set_router_decision_attributes(
                span,
                strategy="knn",
                model_selected="gpt-4",
                candidates_evaluated=5,
                outcome="success",
            )
    """
    if not span or not span.is_recording():
        return

    if strategy is not None:
        span.set_attribute(ROUTER_STRATEGY_ATTR, strategy)

    if model_selected is not None:
        span.set_attribute(ROUTER_MODEL_SELECTED_ATTR, model_selected)

    if score is not None:
        span.set_attribute(ROUTER_SCORE_ATTR, score)

    if candidates_evaluated is not None:
        span.set_attribute(ROUTER_CANDIDATES_EVALUATED_ATTR, candidates_evaluated)

    if outcome is not None:
        span.set_attribute(ROUTER_DECISION_OUTCOME_ATTR, outcome)

    if reason is not None:
        span.set_attribute(ROUTER_DECISION_REASON_ATTR, reason)

    if latency_ms is not None:
        span.set_attribute(ROUTER_LATENCY_MS_ATTR, latency_ms)

    if error_type is not None:
        span.set_attribute(ROUTER_ERROR_TYPE_ATTR, error_type)

    if error_message is not None:
        span.set_attribute(ROUTER_ERROR_MESSAGE_ATTR, error_message)

    if strategy_version is not None:
        span.set_attribute(ROUTER_VERSION_ATTR, strategy_version)

    if fallback_triggered is not None:
        span.set_attribute(ROUTER_FALLBACK_TRIGGERED_ATTR, fallback_triggered)


def set_genai_attributes(
    span: trace.Span,
    *,
    system: Optional[str] = None,
    request_model: Optional[str] = None,
    response_model: Optional[str] = None,
    operation_name: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    finish_reasons: Optional[list[str]] = None,
) -> None:
    """
    Set GenAI semantic convention span attributes on the given span.

    These attributes follow the OpenTelemetry GenAI Semantic Conventions
    (https://opentelemetry.io/docs/specs/semconv/gen-ai/).

    Args:
        span: The OpenTelemetry span to add attributes to.
        system: GenAI provider system (e.g., 'openai', 'anthropic').
        request_model: Model requested by the caller.
        response_model: Model actually used in the response.
        operation_name: Operation type (e.g., 'chat_completion', 'embedding').
        input_tokens: Number of input/prompt tokens used.
        output_tokens: Number of output/completion tokens used.
        finish_reasons: List of finish reasons from the response.
    """
    if not span or not span.is_recording():
        return

    if system is not None:
        span.set_attribute("gen_ai.system", system)

    if request_model is not None:
        span.set_attribute("gen_ai.request.model", request_model)

    if response_model is not None:
        span.set_attribute("gen_ai.response.model", response_model)

    if operation_name is not None:
        span.set_attribute("gen_ai.operation.name", operation_name)

    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)

    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    if finish_reasons is not None:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)


def _is_sdk_tracer_provider(provider: Any) -> bool:
    """
    Check if the provider is an SDK TracerProvider that can accept span processors.

    We check for the actual SDK TracerProvider type because the ProxyTracerProvider
    returned by trace.get_tracer_provider() when no SDK is configured cannot accept
    span processors.

    Args:
        provider: The tracer provider to check

    Returns:
        True if it's an SDK TracerProvider with add_span_processor capability
    """
    # Check if it's the actual SDK TracerProvider class
    if isinstance(provider, TracerProvider):
        return True
    # Also check by attribute in case of wrapped providers
    return hasattr(provider, "add_span_processor") and hasattr(
        provider, "_active_span_processor"
    )


def _get_sampler_from_env() -> Sampler:
    """
    Build a Sampler based on environment variables.

    This function honors multiple env var sources with the following priority:

    1. OTEL Standard (highest priority):
       If OTEL_TRACES_SAMPLER is set, use the standard OTEL sampler configuration:
       - "always_on": Sample all traces
       - "always_off": Sample no traces
       - "traceidratio": Sample based on OTEL_TRACES_SAMPLER_ARG (ratio 0.0-1.0)
       - "parentbased_always_on": Parent-based with always_on root
       - "parentbased_always_off": Parent-based with always_off root
       - "parentbased_traceidratio": Parent-based with ratio-based root

    2. RouteIQ-specific (recommended for production):
       If ROUTEIQ_OTEL_TRACES_SAMPLER is set (or defaults apply), use the sampler type
       with ROUTEIQ_OTEL_TRACES_SAMPLER_ARG for ratio-based samplers.
       Defaults: sampler=parentbased_traceidratio, arg=0.1 (10% sampling)

    3. Legacy LLMROUTER (deprecated, for backwards compatibility):
       If LLMROUTER_OTEL_SAMPLE_RATE is set (0.0-1.0), use a parent-based ratio sampler.
       This is a convenience for simple percentage-based sampling.

    4. Default: Use RouteIQ defaults (parentbased_traceidratio with 10% sampling)

    Returns:
        Configured Sampler instance
    """
    # Check OTEL standard env var first (highest priority)
    otel_sampler = os.getenv("OTEL_TRACES_SAMPLER", "").lower()

    if otel_sampler:
        sampler_arg = os.getenv("OTEL_TRACES_SAMPLER_ARG", "")

        if otel_sampler == "always_on":
            logger.info("Using OTEL sampler: always_on")
            return ALWAYS_ON

        elif otel_sampler == "always_off":
            logger.info("Using OTEL sampler: always_off")
            return ALWAYS_OFF

        elif otel_sampler == "traceidratio":
            try:
                ratio = float(sampler_arg) if sampler_arg else 1.0
                ratio = max(0.0, min(1.0, ratio))  # Clamp to valid range
            except ValueError:
                logger.warning(
                    f"Invalid OTEL_TRACES_SAMPLER_ARG '{sampler_arg}', using 1.0"
                )
                ratio = 1.0
            logger.info(f"Using OTEL sampler: traceidratio ({ratio})")
            return TraceIdRatioBased(ratio)

        elif otel_sampler == "parentbased_always_on":
            logger.info("Using OTEL sampler: parentbased_always_on")
            return ParentBased(root=ALWAYS_ON)

        elif otel_sampler == "parentbased_always_off":
            logger.info("Using OTEL sampler: parentbased_always_off")
            return ParentBased(root=ALWAYS_OFF)

        elif otel_sampler == "parentbased_traceidratio":
            try:
                ratio = float(sampler_arg) if sampler_arg else 1.0
                ratio = max(0.0, min(1.0, ratio))
            except ValueError:
                logger.warning(
                    f"Invalid OTEL_TRACES_SAMPLER_ARG '{sampler_arg}', using 1.0"
                )
                ratio = 1.0
            logger.info(f"Using OTEL sampler: parentbased_traceidratio ({ratio})")
            return ParentBased(root=TraceIdRatioBased(ratio))

        else:
            logger.warning(
                f"Unknown OTEL_TRACES_SAMPLER '{otel_sampler}', falling back to RouteIQ defaults"
            )

    # Check RouteIQ-specific env vars (recommended for production)
    # Default: parentbased_traceidratio with 0.1 (10% sampling)
    routeiq_sampler = os.getenv("ROUTEIQ_OTEL_TRACES_SAMPLER", "").lower()
    routeiq_sampler_arg = os.getenv("ROUTEIQ_OTEL_TRACES_SAMPLER_ARG", "")

    # Check legacy LLMROUTER env var (deprecated, for backwards compatibility)
    llmrouter_sample_rate = os.getenv("LLMROUTER_OTEL_SAMPLE_RATE", "")

    if routeiq_sampler or routeiq_sampler_arg:
        # RouteIQ env vars are explicitly set - use them
        sampler_type = routeiq_sampler or "parentbased_traceidratio"
        sampler_arg = routeiq_sampler_arg or "0.1"

        return _build_sampler_from_type(
            sampler_type, sampler_arg, prefix="ROUTEIQ_OTEL_TRACES_SAMPLER"
        )

    if llmrouter_sample_rate:
        # Legacy env var is set - use it
        try:
            ratio = float(llmrouter_sample_rate)
            ratio = max(0.0, min(1.0, ratio))  # Clamp to valid range
            logger.info(
                f"Using LLMROUTER_OTEL_SAMPLE_RATE: {ratio} (deprecated, use ROUTEIQ_OTEL_TRACES_SAMPLER_ARG)"
            )
            # Use ParentBased to respect incoming trace decisions
            return ParentBased(root=TraceIdRatioBased(ratio))
        except ValueError:
            logger.warning(
                f"Invalid LLMROUTER_OTEL_SAMPLE_RATE '{llmrouter_sample_rate}', "
                "using RouteIQ defaults"
            )

    # Default: Use RouteIQ production defaults (10% sampling with parentbased_traceidratio)
    logger.info("Using RouteIQ default sampler: parentbased_traceidratio (0.1)")
    return ParentBased(root=TraceIdRatioBased(0.1))


def _build_sampler_from_type(
    sampler_type: str, sampler_arg: str, prefix: str
) -> Sampler:
    """
    Build a Sampler from a sampler type string and argument.

    Args:
        sampler_type: The sampler type (e.g., "always_on", "parentbased_traceidratio")
        sampler_arg: The sampler argument (e.g., "0.1" for ratio-based samplers)
        prefix: The env var prefix for logging (e.g., "ROUTEIQ_OTEL_TRACES_SAMPLER")

    Returns:
        Configured Sampler instance
    """
    if sampler_type == "always_on":
        logger.info(f"Using {prefix}: always_on")
        return ALWAYS_ON

    elif sampler_type == "always_off":
        logger.info(f"Using {prefix}: always_off")
        return ALWAYS_OFF

    elif sampler_type == "traceidratio":
        try:
            ratio = float(sampler_arg) if sampler_arg else 0.1
            ratio = max(0.0, min(1.0, ratio))
        except ValueError:
            logger.warning(f"Invalid {prefix}_ARG '{sampler_arg}', using 0.1")
            ratio = 0.1
        logger.info(f"Using {prefix}: traceidratio ({ratio})")
        return TraceIdRatioBased(ratio)

    elif sampler_type == "parentbased_always_on":
        logger.info(f"Using {prefix}: parentbased_always_on")
        return ParentBased(root=ALWAYS_ON)

    elif sampler_type == "parentbased_always_off":
        logger.info(f"Using {prefix}: parentbased_always_off")
        return ParentBased(root=ALWAYS_OFF)

    elif sampler_type == "parentbased_traceidratio":
        try:
            ratio = float(sampler_arg) if sampler_arg else 0.1
            ratio = max(0.0, min(1.0, ratio))
        except ValueError:
            logger.warning(f"Invalid {prefix}_ARG '{sampler_arg}', using 0.1")
            ratio = 0.1
        logger.info(f"Using {prefix}: parentbased_traceidratio ({ratio})")
        return ParentBased(root=TraceIdRatioBased(ratio))

    else:
        logger.warning(
            f"Unknown {prefix} '{sampler_type}', using parentbased_traceidratio (0.1)"
        )
        return ParentBased(root=TraceIdRatioBased(0.1))


def get_routeiq_resource_attributes() -> dict[str, str]:
    """Return OTel resource attributes from RouteIQ env vars."""
    return {
        "service.name": os.getenv("ROUTEIQ_SERVICE_NAME", "routeiq"),
        "deployment.environment": os.getenv("ROUTEIQ_DEPLOYMENT_ENV", "default"),
        "routeiq.metrics.namespace": os.getenv("ROUTEIQ_METRICS_NAMESPACE", "RouteIQ"),
    }


class ObservabilityManager:
    """
    Manages OpenTelemetry observability for the LiteLLM + LLMRouter Gateway.

    This class provides:
    - Tracer initialization with OTLP exporters
    - Logger setup with trace correlation
    - Meter setup for custom metrics
    - Integration with LiteLLM's existing observability

    IMPORTANT: This manager is designed to REUSE existing SDK providers rather
    than creating new ones. This ensures all spans (from auto-instrumentation,
    LiteLLM, and our custom code) are exported through the same provider.
    """

    def __init__(
        self,
        service_name: str = "litellm-gateway",
        service_version: str = "1.0.0",
        deployment_environment: str = "production",
        otlp_endpoint: Optional[str] = None,
        enable_traces: bool = True,
        enable_logs: bool = True,
        enable_metrics: bool = True,
        sampler: Optional[Sampler] = None,
    ):
        """
        Initialize the observability manager.

        Args:
            service_name: Name of the service for telemetry
            service_version: Version of the service
            deployment_environment: Deployment environment (production, staging, etc.)
            otlp_endpoint: OTLP collector endpoint (e.g., "http://otel-collector:4317")
            enable_traces: Whether to enable distributed tracing
            enable_logs: Whether to enable structured logging
            enable_metrics: Whether to enable metrics collection
            sampler: Optional custom Sampler. If None, the sampler is configured from
                     environment variables (OTEL_TRACES_SAMPLER, LLMROUTER_OTEL_SAMPLE_RATE).
        """
        self.service_name = service_name
        self.service_version = service_version
        self.deployment_environment = deployment_environment
        self.otlp_endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        self.enable_traces = enable_traces
        self.enable_logs = enable_logs
        self.enable_metrics = enable_metrics
        # Resolve sampler: explicit > env-based > default
        self._sampler = sampler if sampler is not None else _get_sampler_from_env()

        # Create resource with service identification
        self.resource = Resource.create(
            {
                SERVICE_NAME: self.service_name,
                SERVICE_VERSION: self.service_version,
                "deployment.environment": self.deployment_environment,
                "service.namespace": "ai-gateway",
            }
        )

        self._tracer_provider: Optional[TracerProvider] = None
        self._logger_provider: Optional[LoggerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._tracer: Optional[trace.Tracer] = None
        self._meter: Optional[metrics.Meter] = None
        self._span_processor_added: bool = False

    def initialize(self) -> None:
        """
        Initialize all OpenTelemetry providers and exporters.

        This method should be called during application startup.
        """
        if self.enable_traces:
            self._init_tracing()

        if self.enable_logs:
            self._init_logging()

        if self.enable_metrics:
            self._init_metrics()

        logger.info(
            f"OpenTelemetry initialized for {self.service_name} "
            f"(endpoint: {self.otlp_endpoint})"
        )

    def _init_tracing(self) -> None:
        """
        Initialize distributed tracing with OTLP exporter.

        IMPORTANT: This method is designed to REUSE an existing SDK TracerProvider
        if one is already configured. This ensures that all spans from all sources
        (auto-instrumentation, LiteLLM, our custom code) go through the same provider
        and are exported together.

        The logic is:
        1. Check if an SDK TracerProvider already exists (from LiteLLM or auto-instrumentation)
        2. If yes, reuse it and just add our OTLP BatchSpanProcessor
        3. If no, create a new SDK TracerProvider with our resource and sampler

        Note: When reusing an existing provider, we cannot change its sampler.
        The sampler is only applied when creating a new provider.
        """
        existing_provider = trace.get_tracer_provider()

        # Check if we have an actual SDK TracerProvider we can reuse
        if _is_sdk_tracer_provider(existing_provider):
            # Reuse existing SDK provider - this is the preferred path
            # It ensures all spans go to the same exporter
            self._tracer_provider = cast(TracerProvider, existing_provider)
            logger.info("Reusing existing SDK TracerProvider - attaching OTLP exporter")
        else:
            # No SDK provider exists yet - create one with our resource and sampler
            # This happens when our code runs before any auto-instrumentation
            self._tracer_provider = TracerProvider(
                resource=self.resource,
                sampler=self._sampler,
            )
            trace.set_tracer_provider(self._tracer_provider)
            logger.info(
                "Created new SDK TracerProvider with resource: %s, sampler: %s",
                self.service_name,
                type(self._sampler).__name__,
            )

        # Add our OTLP exporter as a BatchSpanProcessor
        # This ensures spans are exported even if LiteLLM didn't configure OTLP
        if not self._span_processor_added and self._tracer_provider is not None:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint, insecure=True
                )
                span_processor = BatchSpanProcessor(otlp_exporter)
                self._tracer_provider.add_span_processor(span_processor)
                self._span_processor_added = True
                logger.info(
                    "Added OTLP BatchSpanProcessor to TracerProvider "
                    f"(endpoint: {self.otlp_endpoint})"
                )
            except Exception as e:
                logger.error(f"Failed to add OTLP span processor: {e}", exc_info=True)

        # Get tracer for this module
        self._tracer = trace.get_tracer(__name__, self.service_version)

        logger.info(f"Tracing initialized with OTLP endpoint: {self.otlp_endpoint}")

    def _init_logging(self) -> None:
        """Initialize structured logging with trace correlation."""
        # Check if a logger provider already exists
        try:
            from opentelemetry._logs import get_logger_provider

            existing_provider = get_logger_provider()
            if hasattr(existing_provider, "add_log_record_processor"):
                self._logger_provider = cast(LoggerProvider, existing_provider)
                logger.info("Using existing LoggerProvider")
            else:
                # Create new provider
                self._logger_provider = LoggerProvider(resource=self.resource)
                set_logger_provider(self._logger_provider)
                logger.info("Created new LoggerProvider")
        except Exception:
            # Create new provider
            self._logger_provider = LoggerProvider(resource=self.resource)
            set_logger_provider(self._logger_provider)
            logger.info("Created new LoggerProvider")

        # Add OTLP exporter for logs
        assert self._logger_provider is not None  # Set in all branches above
        otlp_log_exporter = OTLPLogExporter(endpoint=self.otlp_endpoint, insecure=True)
        log_processor = BatchLogRecordProcessor(otlp_log_exporter)
        self._logger_provider.add_log_record_processor(log_processor)

        # Instrument Python logging to add trace context
        LoggingInstrumentor().instrument(set_logging_format=True)

        # Add OTLP handler to root logger
        handler = LoggingHandler(
            level=logging.INFO, logger_provider=self._logger_provider
        )
        logging.getLogger().addHandler(handler)

        logger.info(f"Logging initialized with OTLP endpoint: {self.otlp_endpoint}")

    def _init_metrics(self) -> None:
        """Initialize metrics collection with OTLP exporter and instrument registry."""
        # Check if a meter provider already exists
        existing_provider = metrics.get_meter_provider()
        if hasattr(existing_provider, "register_metric_reader"):
            self._meter_provider = cast(MeterProvider, existing_provider)
            logger.info("Using existing MeterProvider")
        else:
            # Create OTLP metric exporter
            otlp_metric_exporter = OTLPMetricExporter(
                endpoint=self.otlp_endpoint, insecure=True
            )
            metric_reader = PeriodicExportingMetricReader(
                otlp_metric_exporter, export_interval_millis=60000
            )

            # Create new provider
            self._meter_provider = MeterProvider(
                resource=self.resource, metric_readers=[metric_reader]
            )
            metrics.set_meter_provider(self._meter_provider)
            logger.info("Created new MeterProvider")

        # Get meter for this module
        self._meter = metrics.get_meter(__name__, self.service_version)

        # Initialize the central metrics instrument registry
        try:
            from litellm_llmrouter.metrics import init_gateway_metrics

            init_gateway_metrics(self._meter)
            logger.info("GatewayMetrics instrument registry initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize GatewayMetrics: {e}")

        logger.info(f"Metrics initialized with OTLP endpoint: {self.otlp_endpoint}")

    def get_tracer(self) -> trace.Tracer:
        """Get the tracer instance for creating spans."""
        if self._tracer is None:
            raise RuntimeError("Tracing not initialized. Call initialize() first.")
        return self._tracer

    def get_meter(self) -> metrics.Meter:
        """Get the meter instance for creating metrics."""
        if self._meter is None:
            raise RuntimeError("Metrics not initialized. Call initialize() first.")
        return self._meter

    def create_routing_span(self, strategy_name: str, model_count: int) -> trace.Span:
        """
        Create a span for a routing decision.

        Args:
            strategy_name: Name of the routing strategy
            model_count: Number of models being considered

        Returns:
            OpenTelemetry span for the routing operation
        """
        tracer = self.get_tracer()
        span = tracer.start_span("llm.routing.decision")
        span.set_attribute("llm.routing.strategy", strategy_name)
        span.set_attribute("llm.routing.model_count", model_count)
        return span

    def create_cache_span(self, operation: str, cache_key: str) -> trace.Span:
        """
        Create a span for a cache operation.

        Args:
            operation: Cache operation (lookup, set, delete)
            cache_key: Cache key (truncated for privacy)

        Returns:
            OpenTelemetry span for the cache operation
        """
        tracer = self.get_tracer()
        span = tracer.start_span(f"cache.{operation}")
        # Truncate cache key for privacy
        span.set_attribute("cache.key", cache_key[:50] if cache_key else "")
        return span

    def log_routing_decision(
        self,
        strategy: str,
        selected_model: str,
        query: Optional[str] = None,
        latency_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a routing decision with trace correlation.

        Args:
            strategy: Routing strategy used
            selected_model: Model that was selected
            query: User query (optional, for privacy)
            latency_ms: Routing decision latency
            extra: Additional context to log
        """
        log_data: Dict[str, Any] = {
            "event": "routing.decision",
            "strategy": strategy,
            "selected_model": selected_model,
        }

        if latency_ms is not None:
            log_data["latency_ms"] = latency_ms

        if extra:
            log_data.update(extra)

        # Don't log query content by default for privacy
        if query and os.getenv("LOG_QUERIES", "false").lower() == "true":
            log_data["query_length"] = len(query)

        logger.info("Routing decision made", extra=log_data)

    def log_error_with_trace(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an error with trace correlation and stack trace.

        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        log_data = {
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if context:
            log_data.update(context)

        logger.error(
            f"Error occurred: {error}",
            extra=log_data,
            exc_info=True,
        )

    @property
    def sampler(self) -> Sampler:
        """Get the configured sampler for tracing."""
        return self._sampler


# Global observability manager instance
_observability_manager: Optional[ObservabilityManager] = None


def init_observability(
    service_name: Optional[str] = None,
    service_version: Optional[str] = None,
    deployment_environment: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    enable_traces: bool = True,
    enable_logs: bool = True,
    enable_metrics: bool = True,
) -> ObservabilityManager:
    """
    Initialize the global observability manager.

    This function should be called once during application startup.

    Args:
        service_name: Name of the service (default: from env or "litellm-gateway")
        service_version: Version of the service (default: from env or "1.0.0")
        deployment_environment: Environment (default: from env or "production")
        otlp_endpoint: OTLP collector endpoint
        enable_traces: Whether to enable tracing
        enable_logs: Whether to enable logging
        enable_metrics: Whether to enable metrics

    Returns:
        Initialized ObservabilityManager instance
    """
    global _observability_manager

    # Get values from environment if not provided
    resolved_service_name: str = (
        service_name
        or os.getenv("OTEL_SERVICE_NAME", "litellm-gateway")
        or "litellm-gateway"
    )
    resolved_service_version: str = (
        service_version or os.getenv("OTEL_SERVICE_VERSION", "1.0.0") or "1.0.0"
    )
    resolved_deployment_env: str = (
        deployment_environment
        or os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "production")
        or "production"
    )

    _observability_manager = ObservabilityManager(
        service_name=resolved_service_name,
        service_version=resolved_service_version,
        deployment_environment=resolved_deployment_env,
        otlp_endpoint=otlp_endpoint,
        enable_traces=enable_traces,
        enable_logs=enable_logs,
        enable_metrics=enable_metrics,
    )

    _observability_manager.initialize()

    return _observability_manager


def get_observability_manager() -> Optional[ObservabilityManager]:
    """
    Get the global observability manager instance.

    Returns:
        ObservabilityManager instance or None if not initialized
    """
    return _observability_manager


def get_tracer() -> trace.Tracer:
    """
    Get the global tracer instance.

    Returns:
        OpenTelemetry Tracer

    Raises:
        RuntimeError: If observability is not initialized
    """
    if _observability_manager is None:
        raise RuntimeError(
            "Observability not initialized. Call init_observability() first."
        )
    return _observability_manager.get_tracer()


def get_meter() -> metrics.Meter:
    """
    Get the global meter instance.

    Returns:
        OpenTelemetry Meter

    Raises:
        RuntimeError: If observability is not initialized
    """
    if _observability_manager is None:
        raise RuntimeError(
            "Observability not initialized. Call init_observability() first."
        )
    return _observability_manager.get_meter()


def reset_observability_manager() -> None:
    """
    Reset the global observability manager singleton.

    Must be called in test fixtures to avoid singleton leaks between tests.
    Also resets the dependent GatewayMetrics singleton.
    """
    global _observability_manager
    _observability_manager = None

    # Also reset the metrics singleton which depends on the meter
    from litellm_llmrouter.metrics import reset_gateway_metrics

    reset_gateway_metrics()


def record_ttft(model: str, duration_s: float) -> None:
    """
    Record time-to-first-token for a streaming response.

    Args:
        model: The model that generated the streaming response.
        duration_s: Time in seconds from request start to first token.
    """
    from litellm_llmrouter.metrics import get_gateway_metrics

    gw_metrics = get_gateway_metrics()
    if gw_metrics is not None:
        gw_metrics.time_to_first_token.record(duration_s, {"model": model})
