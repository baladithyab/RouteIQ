"""
Unit Tests for OpenTelemetry Observability Integration
=======================================================

Tests for the observability module including:
- Tracer initialization and configuration
- Span creation for key operations
- Log correlation with trace context
- OTLP exporter configuration
- TracerProvider reuse logic (avoid competing providers)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock all external dependencies before importing
sys.modules["litellm"] = MagicMock()
sys.modules["litellm._logging"] = MagicMock()
sys.modules["litellm.proxy"] = MagicMock()
sys.modules["litellm.proxy.proxy_server"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["pydantic"] = MagicMock()

# Mock OTLP exporter modules (not available in test environment)
sys.modules["opentelemetry.exporter"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp.proto"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp.proto.grpc"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp.proto.grpc._log_exporter"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp.proto.grpc.metric_exporter"] = MagicMock()
sys.modules["opentelemetry.instrumentation"] = MagicMock()
sys.modules["opentelemetry.instrumentation.logging"] = MagicMock()

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402


# Import the module under test directly (not through __init__.py)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "observability", "src/litellm_llmrouter/observability.py"
)
observability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observability)

ObservabilityManager = observability.ObservabilityManager
init_observability = observability.init_observability
get_observability_manager = observability.get_observability_manager
get_tracer = observability.get_tracer
get_meter = observability.get_meter
_is_sdk_tracer_provider = observability._is_sdk_tracer_provider


class TestIsSdkTracerProvider:
    """Test suite for _is_sdk_tracer_provider helper function."""

    def test_returns_true_for_sdk_tracer_provider(self):
        """Test that _is_sdk_tracer_provider returns True for SDK TracerProvider."""
        provider = TracerProvider()
        assert _is_sdk_tracer_provider(provider) is True

    def test_returns_false_for_none(self):
        """Test that _is_sdk_tracer_provider returns False for None."""
        assert _is_sdk_tracer_provider(None) is False

    def test_returns_false_for_proxy_provider(self):
        """Test that _is_sdk_tracer_provider returns False for ProxyTracerProvider."""
        # The default provider (before any SDK is set) is a ProxyTracerProvider
        # which doesn't have add_span_processor or _active_span_processor
        mock_proxy = MagicMock(spec=[])  # Empty spec = no attributes
        assert _is_sdk_tracer_provider(mock_proxy) is False

    def test_returns_true_for_provider_with_span_processor_methods(self):
        """Test detection of providers with span processor capabilities."""
        # A provider-like object with the required methods
        mock_provider = MagicMock()
        mock_provider.add_span_processor = MagicMock()
        mock_provider._active_span_processor = MagicMock()
        assert _is_sdk_tracer_provider(mock_provider) is True


class TestTracerProviderReuse:
    """Test suite for TracerProvider reuse logic."""

    def teardown_method(self):
        """Reset global state after each test."""
        observability._observability_manager = None

    def test_init_tracing_reuses_existing_sdk_provider(self):
        """Test that _init_tracing reuses existing SDK TracerProvider."""
        # Create and set an SDK TracerProvider
        existing_provider = TracerProvider()

        with patch.object(trace, "get_tracer_provider", return_value=existing_provider):
            manager = ObservabilityManager(
                service_name="test",
                enable_traces=True,
                enable_logs=False,
                enable_metrics=False,
            )

            # Mock the OTLP exporter to avoid network calls
            with patch(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
            ):
                with patch.object(existing_provider, "add_span_processor") as mock_add:
                    manager._init_tracing()

                    # Should have added span processor to existing provider
                    mock_add.assert_called_once()

                    # Should be using the existing provider
                    assert manager._tracer_provider is existing_provider

    def test_init_tracing_creates_new_provider_when_no_sdk_exists(self):
        """Test that _init_tracing creates new provider when no SDK exists."""
        # Return a mock ProxyTracerProvider (no add_span_processor)
        mock_proxy = MagicMock(spec=[])

        with patch.object(trace, "get_tracer_provider", return_value=mock_proxy):
            with patch.object(trace, "set_tracer_provider") as mock_set:
                with patch.object(trace, "get_tracer") as _:  # noqa: F841
                    manager = ObservabilityManager(
                        service_name="test",
                        enable_traces=True,
                        enable_logs=False,
                        enable_metrics=False,
                    )

                    # Mock the OTLP exporter
                    with patch(
                        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
                    ):
                        manager._init_tracing()

                        # Should have created and set a new provider
                        mock_set.assert_called_once()
                        assert isinstance(manager._tracer_provider, TracerProvider)

    def test_span_processor_only_added_once(self):
        """Test that OTLP span processor is only added once."""
        existing_provider = TracerProvider()

        with patch.object(trace, "get_tracer_provider", return_value=existing_provider):
            with patch(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
            ):
                with patch.object(existing_provider, "add_span_processor") as mock_add:
                    manager = ObservabilityManager(
                        enable_traces=True,
                        enable_logs=False,
                        enable_metrics=False,
                    )

                    # Initialize twice
                    manager._init_tracing()
                    manager._init_tracing()

                    # Should only add processor once (due to _span_processor_added flag)
                    assert mock_add.call_count == 1


class TestObservabilityManager:
    """Test suite for ObservabilityManager class."""

    def test_initialization_with_defaults(self):
        """Test that ObservabilityManager initializes with default values."""
        manager = ObservabilityManager()

        assert manager.service_name == "litellm-gateway"
        assert manager.service_version == "1.0.0"
        assert manager.deployment_environment == "production"
        assert manager.enable_traces is True
        assert manager.enable_logs is True
        assert manager.enable_metrics is True

    def test_initialization_with_custom_values(self):
        """Test that ObservabilityManager accepts custom configuration."""
        manager = ObservabilityManager(
            service_name="custom-service",
            service_version="2.0.0",
            deployment_environment="staging",
            otlp_endpoint="http://custom:4317",
            enable_traces=False,
            enable_logs=True,
            enable_metrics=False,
        )

        assert manager.service_name == "custom-service"
        assert manager.service_version == "2.0.0"
        assert manager.deployment_environment == "staging"
        assert manager.otlp_endpoint == "http://custom:4317"
        assert manager.enable_traces is False
        assert manager.enable_logs is True
        assert manager.enable_metrics is False

    def test_resource_creation(self):
        """Test that resource is created with correct attributes."""
        manager = ObservabilityManager(
            service_name="test-service",
            service_version="1.2.3",
            deployment_environment="dev",
        )

        resource_attrs = manager.resource.attributes
        assert resource_attrs["service.name"] == "test-service"
        assert resource_attrs["service.version"] == "1.2.3"
        assert resource_attrs["deployment.environment"] == "dev"
        assert resource_attrs["service.namespace"] == "ai-gateway"

    def test_get_tracer_before_init_raises_error(self):
        """Test that getting tracer before initialization raises error."""
        manager = ObservabilityManager(enable_traces=True)

        with pytest.raises(RuntimeError, match="Tracing not initialized"):
            manager.get_tracer()

    def test_get_meter_before_init_raises_error(self):
        """Test that getting meter before initialization raises error."""
        manager = ObservabilityManager(enable_metrics=True)

        with pytest.raises(RuntimeError, match="Metrics not initialized"):
            manager.get_meter()

    def test_create_routing_span_requires_initialization(self):
        """Test that creating routing span requires initialization."""
        manager = ObservabilityManager(enable_traces=True)

        with pytest.raises(RuntimeError):
            manager.create_routing_span("llmrouter-knn", 5)

    def test_create_cache_span_requires_initialization(self):
        """Test that creating cache span requires initialization."""
        manager = ObservabilityManager(enable_traces=True)

        with pytest.raises(RuntimeError):
            manager.create_cache_span("lookup", "test-key")

    def test_log_routing_decision_without_init(self):
        """Test that logging routing decision works without initialization."""
        manager = ObservabilityManager()

        # Should not raise an error
        manager.log_routing_decision(
            strategy="llmrouter-knn",
            selected_model="gpt-4",
            latency_ms=123.45,
        )

    def test_log_error_with_trace_without_init(self):
        """Test that logging errors works without initialization."""
        manager = ObservabilityManager()

        error = ValueError("Test error")
        context = {"request_id": "req-123"}

        # Should not raise an error
        manager.log_error_with_trace(error, context)

    def test_otlp_endpoint_from_env(self):
        """Test that OTLP endpoint can be set from environment."""
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://env:4317"}):
            manager = ObservabilityManager()
            assert manager.otlp_endpoint == "http://env:4317"

    def test_otlp_endpoint_default(self):
        """Test that OTLP endpoint has a default value."""
        with patch.dict(os.environ, {}, clear=True):
            manager = ObservabilityManager()
            assert manager.otlp_endpoint == "http://localhost:4317"


class TestGlobalFunctions:
    """Test suite for global observability functions."""

    def teardown_method(self):
        """Reset global state after each test."""
        observability._observability_manager = None

    def test_get_observability_manager_before_init(self):
        """Test that get_observability_manager returns None before init."""
        manager = get_observability_manager()
        assert manager is None

    def test_get_tracer_before_init_raises_error(self):
        """Test that get_tracer raises error before initialization."""
        with pytest.raises(RuntimeError, match="Observability not initialized"):
            get_tracer()

    def test_get_meter_before_init_raises_error(self):
        """Test that get_meter raises error before initialization."""
        with pytest.raises(RuntimeError, match="Observability not initialized"):
            get_meter()

    def test_init_observability_returns_manager(self):
        """Test that init_observability returns a manager instance."""
        with patch.dict(os.environ, {}, clear=True):
            # Mock the initialize method to avoid actual OTLP connections
            with patch.object(ObservabilityManager, "initialize"):
                manager = init_observability(
                    service_name="test",
                    enable_traces=False,
                    enable_logs=False,
                    enable_metrics=False,
                )

                assert manager is not None
                assert isinstance(manager, ObservabilityManager)
                assert manager.service_name == "test"

    def test_init_observability_with_env_vars(self):
        """Test that init_observability uses environment variables."""
        with patch.dict(
            os.environ,
            {
                "OTEL_SERVICE_NAME": "env-service",
                "OTEL_SERVICE_VERSION": "2.0.0",
                "OTEL_DEPLOYMENT_ENVIRONMENT": "staging",
            },
        ):
            with patch.object(ObservabilityManager, "initialize"):
                manager = init_observability(
                    enable_traces=False,
                    enable_logs=False,
                    enable_metrics=False,
                )

                assert manager.service_name == "env-service"
                assert manager.service_version == "2.0.0"
                assert manager.deployment_environment == "staging"

    def test_get_observability_manager_after_init(self):
        """Test that get_observability_manager returns the initialized manager."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(ObservabilityManager, "initialize"):
                init_manager = init_observability(
                    service_name="test",
                    enable_traces=False,
                    enable_logs=False,
                    enable_metrics=False,
                )

                get_manager = get_observability_manager()

                assert get_manager is init_manager

    def test_init_observability_calls_initialize(self):
        """Test that init_observability calls the initialize method."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(ObservabilityManager, "initialize") as mock_init:
                init_observability(
                    service_name="test",
                    enable_traces=False,
                    enable_logs=False,
                    enable_metrics=False,
                )

                mock_init.assert_called_once()


class TestObservabilityConfiguration:
    """Test suite for observability configuration."""

    def test_service_name_from_parameter(self):
        """Test that service name can be set via parameter."""
        manager = ObservabilityManager(service_name="param-service")
        assert manager.service_name == "param-service"

    def test_service_version_from_parameter(self):
        """Test that service version can be set via parameter."""
        manager = ObservabilityManager(service_version="3.0.0")
        assert manager.service_version == "3.0.0"

    def test_deployment_environment_from_parameter(self):
        """Test that deployment environment can be set via parameter."""
        manager = ObservabilityManager(deployment_environment="production")
        assert manager.deployment_environment == "production"

    def test_enable_flags_control_initialization(self):
        """Test that enable flags control what gets initialized."""
        manager = ObservabilityManager(
            enable_traces=True,
            enable_logs=False,
            enable_metrics=False,
        )

        assert manager.enable_traces is True
        assert manager.enable_logs is False
        assert manager.enable_metrics is False

    def test_resource_attributes_are_set(self):
        """Test that resource attributes are properly set."""
        manager = ObservabilityManager(
            service_name="test-service",
            service_version="1.0.0",
            deployment_environment="dev",
        )

        attrs = manager.resource.attributes
        assert "service.name" in attrs
        assert "service.version" in attrs
        assert "deployment.environment" in attrs
        assert "service.namespace" in attrs
