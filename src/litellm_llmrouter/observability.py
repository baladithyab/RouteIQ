"""
OpenTelemetry Observability Integration
========================================

This module provides unified observability via OpenTelemetry:
- Distributed tracing for request flow
- Structured logging with trace correlation
- Metrics collection (via Prometheus)

The module integrates with LiteLLM's existing observability while adding
LLMRouter-specific instrumentation for routing decisions.
"""

import logging
import os
from typing import Any, Dict, Optional

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
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

logger = logging.getLogger(__name__)


class ObservabilityManager:
    """
    Manages OpenTelemetry observability for the LiteLLM + LLMRouter Gateway.
    
    This class provides:
    - Tracer initialization with OTLP exporters
    - Logger setup with trace correlation
    - Meter setup for custom metrics
    - Integration with LiteLLM's existing observability
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
        """Initialize distributed tracing with OTLP exporter."""
        # Check if a tracer provider already exists (from LiteLLM)
        existing_provider = trace.get_tracer_provider()
        if hasattr(existing_provider, "add_span_processor"):
            # Use existing provider
            self._tracer_provider = existing_provider
            logger.info("Using existing TracerProvider from LiteLLM")
        else:
            # Create new provider
            self._tracer_provider = TracerProvider(resource=self.resource)
            trace.set_tracer_provider(self._tracer_provider)
            logger.info("Created new TracerProvider")

        # Add OTLP exporter
        otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(otlp_exporter)
        self._tracer_provider.add_span_processor(span_processor)

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
                self._logger_provider = existing_provider
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
        """Initialize metrics collection with OTLP exporter."""
        # Check if a meter provider already exists
        existing_provider = metrics.get_meter_provider()
        if hasattr(existing_provider, "register_metric_reader"):
            self._meter_provider = existing_provider
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

    def create_routing_span(
        self, strategy_name: str, model_count: int
    ) -> trace.Span:
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
        log_data = {
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
    service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "litellm-gateway")
    service_version = service_version or os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
    deployment_environment = deployment_environment or os.getenv(
        "OTEL_DEPLOYMENT_ENVIRONMENT", "production"
    )

    _observability_manager = ObservabilityManager(
        service_name=service_name,
        service_version=service_version,
        deployment_environment=deployment_environment,
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
        raise RuntimeError("Observability not initialized. Call init_observability() first.")
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
        raise RuntimeError("Observability not initialized. Call init_observability() first.")
    return _observability_manager.get_meter()
