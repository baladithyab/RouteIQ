"""
Unit Tests for OpenTelemetry Observability Integration
=======================================================

Tests for the observability module including:
- Tracer initialization and configuration
- Span creation for key operations
- Log correlation with trace context
- OTLP exporter configuration
"""

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock all external dependencies before importing
sys.modules['litellm'] = MagicMock()
sys.modules['litellm._logging'] = MagicMock()
sys.modules['litellm.proxy'] = MagicMock()
sys.modules['litellm.proxy.proxy_server'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['pydantic'] = MagicMock()

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk._logs import LoggerProvider


# Import the module under test directly (not through __init__.py)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "observability",
    "src/litellm_llmrouter/observability.py"
)
observability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observability)

ObservabilityManager = observability.ObservabilityManager
init_observability = observability.init_observability
get_observability_manager = observability.get_observability_manager
get_tracer = observability.get_tracer
get_meter = observability.get_meter


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
            with patch.object(ObservabilityManager, 'initialize'):
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
        with patch.dict(os.environ, {
            "OTEL_SERVICE_NAME": "env-service",
            "OTEL_SERVICE_VERSION": "2.0.0",
            "OTEL_DEPLOYMENT_ENVIRONMENT": "staging",
        }):
            with patch.object(ObservabilityManager, 'initialize'):
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
            with patch.object(ObservabilityManager, 'initialize'):
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
            with patch.object(ObservabilityManager, 'initialize') as mock_init:
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
