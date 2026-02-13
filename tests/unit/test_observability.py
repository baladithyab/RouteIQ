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
from unittest.mock import MagicMock, patch

import pytest

# NOTE: We don't mock modules at the top level because it pollutes sys.modules
# and corrupts other tests that run later in the suite.
# The observability module is loaded directly via importlib.util to avoid
# importing the full litellm_llmrouter package which has heavy dependencies.

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    ALWAYS_OFF,
    TraceIdRatioBased,
    ParentBased,
)


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


# Import the _get_sampler_from_env function
_get_sampler_from_env = observability._get_sampler_from_env

# Import the _build_sampler_from_type function
_build_sampler_from_type = observability._build_sampler_from_type


class TestBuildSamplerFromType:
    """Test suite for _build_sampler_from_type helper function."""

    def test_always_on(self):
        """Test building always_on sampler."""
        sampler = _build_sampler_from_type("always_on", "", "TEST")
        assert sampler is ALWAYS_ON

    def test_always_off(self):
        """Test building always_off sampler."""
        sampler = _build_sampler_from_type("always_off", "", "TEST")
        assert sampler is ALWAYS_OFF

    def test_traceidratio_with_arg(self):
        """Test building traceidratio sampler with valid arg."""
        sampler = _build_sampler_from_type("traceidratio", "0.5", "TEST")
        assert isinstance(sampler, TraceIdRatioBased)
        assert sampler._rate == 0.5

    def test_traceidratio_default_arg(self):
        """Test building traceidratio sampler defaults to 0.1."""
        sampler = _build_sampler_from_type("traceidratio", "", "TEST")
        assert isinstance(sampler, TraceIdRatioBased)
        assert sampler._rate == 0.1

    def test_traceidratio_invalid_arg(self):
        """Test building traceidratio sampler with invalid arg defaults to 0.1."""
        sampler = _build_sampler_from_type("traceidratio", "invalid", "TEST")
        assert isinstance(sampler, TraceIdRatioBased)
        assert sampler._rate == 0.1

    def test_parentbased_always_on(self):
        """Test building parentbased_always_on sampler."""
        sampler = _build_sampler_from_type("parentbased_always_on", "", "TEST")
        assert isinstance(sampler, ParentBased)

    def test_parentbased_always_off(self):
        """Test building parentbased_always_off sampler."""
        sampler = _build_sampler_from_type("parentbased_always_off", "", "TEST")
        assert isinstance(sampler, ParentBased)

    def test_parentbased_traceidratio_with_arg(self):
        """Test building parentbased_traceidratio sampler with valid arg."""
        sampler = _build_sampler_from_type("parentbased_traceidratio", "0.25", "TEST")
        assert isinstance(sampler, ParentBased)

    def test_parentbased_traceidratio_default_arg(self):
        """Test building parentbased_traceidratio sampler defaults to 0.1."""
        sampler = _build_sampler_from_type("parentbased_traceidratio", "", "TEST")
        assert isinstance(sampler, ParentBased)

    def test_unknown_sampler_type(self):
        """Test unknown sampler type falls back to parentbased_traceidratio(0.1)."""
        sampler = _build_sampler_from_type("unknown", "", "TEST")
        assert isinstance(sampler, ParentBased)


class TestSamplerConfiguration:
    """Test suite for trace sampler configuration."""

    def test_default_sampler_is_parentbased_always_on(self):
        """Test that default sampler is ParentBased(always_on) for backwards compatibility."""
        with patch.dict(os.environ, {}, clear=True):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, ParentBased)

    def test_otel_sampler_always_on(self):
        """Test OTEL_TRACES_SAMPLER=always_on returns ALWAYS_ON."""
        with patch.dict(os.environ, {"OTEL_TRACES_SAMPLER": "always_on"}, clear=True):
            sampler = _get_sampler_from_env()
            assert sampler is ALWAYS_ON

    def test_otel_sampler_always_off(self):
        """Test OTEL_TRACES_SAMPLER=always_off returns ALWAYS_OFF."""
        with patch.dict(os.environ, {"OTEL_TRACES_SAMPLER": "always_off"}, clear=True):
            sampler = _get_sampler_from_env()
            assert sampler is ALWAYS_OFF

    def test_otel_sampler_traceidratio(self):
        """Test OTEL_TRACES_SAMPLER=traceidratio with arg."""
        with patch.dict(
            os.environ,
            {"OTEL_TRACES_SAMPLER": "traceidratio", "OTEL_TRACES_SAMPLER_ARG": "0.5"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, TraceIdRatioBased)
            # Access internal ratio (implementation detail, but useful for testing)
            assert sampler._rate == 0.5

    def test_otel_sampler_traceidratio_default_arg(self):
        """Test OTEL_TRACES_SAMPLER=traceidratio without arg defaults to 1.0."""
        with patch.dict(
            os.environ,
            {"OTEL_TRACES_SAMPLER": "traceidratio"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, TraceIdRatioBased)
            assert sampler._rate == 1.0

    def test_otel_sampler_traceidratio_invalid_arg(self):
        """Test OTEL_TRACES_SAMPLER=traceidratio with invalid arg defaults to 1.0."""
        with patch.dict(
            os.environ,
            {
                "OTEL_TRACES_SAMPLER": "traceidratio",
                "OTEL_TRACES_SAMPLER_ARG": "invalid",
            },
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, TraceIdRatioBased)
            assert sampler._rate == 1.0

    def test_otel_sampler_parentbased_always_on(self):
        """Test OTEL_TRACES_SAMPLER=parentbased_always_on."""
        with patch.dict(
            os.environ,
            {"OTEL_TRACES_SAMPLER": "parentbased_always_on"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, ParentBased)

    def test_otel_sampler_parentbased_always_off(self):
        """Test OTEL_TRACES_SAMPLER=parentbased_always_off."""
        with patch.dict(
            os.environ,
            {"OTEL_TRACES_SAMPLER": "parentbased_always_off"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, ParentBased)

    def test_otel_sampler_parentbased_traceidratio(self):
        """Test OTEL_TRACES_SAMPLER=parentbased_traceidratio with arg."""
        with patch.dict(
            os.environ,
            {
                "OTEL_TRACES_SAMPLER": "parentbased_traceidratio",
                "OTEL_TRACES_SAMPLER_ARG": "0.25",
            },
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, ParentBased)

    def test_otel_sampler_unknown_falls_back_to_default(self):
        """Test unknown OTEL_TRACES_SAMPLER falls back to default."""
        with patch.dict(
            os.environ,
            {"OTEL_TRACES_SAMPLER": "unknown_sampler"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            # Falls back to default (ParentBased)
            assert isinstance(sampler, ParentBased)

    def test_llmrouter_sample_rate(self):
        """Test LLMROUTER_OTEL_SAMPLE_RATE for simple ratio sampling."""
        with patch.dict(
            os.environ,
            {"LLMROUTER_OTEL_SAMPLE_RATE": "0.1"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, ParentBased)

    def test_llmrouter_sample_rate_clamps_values(self):
        """Test LLMROUTER_OTEL_SAMPLE_RATE clamps to 0.0-1.0 range."""
        # Value > 1.0 should be clamped to 1.0
        with patch.dict(
            os.environ,
            {"LLMROUTER_OTEL_SAMPLE_RATE": "2.0"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, ParentBased)

        # Value < 0.0 should be clamped to 0.0
        with patch.dict(
            os.environ,
            {"LLMROUTER_OTEL_SAMPLE_RATE": "-0.5"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, ParentBased)

    def test_llmrouter_sample_rate_invalid_value(self):
        """Test LLMROUTER_OTEL_SAMPLE_RATE with invalid value falls back to default."""
        with patch.dict(
            os.environ,
            {"LLMROUTER_OTEL_SAMPLE_RATE": "not_a_number"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            # Falls back to default (ParentBased with ALWAYS_ON)
            assert isinstance(sampler, ParentBased)

    def test_otel_sampler_takes_precedence_over_llmrouter(self):
        """Test OTEL_TRACES_SAMPLER takes precedence over LLMROUTER_OTEL_SAMPLE_RATE."""
        with patch.dict(
            os.environ,
            {
                "OTEL_TRACES_SAMPLER": "always_off",
                "LLMROUTER_OTEL_SAMPLE_RATE": "1.0",
            },
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert sampler is ALWAYS_OFF

    def test_otel_sampler_case_insensitive(self):
        """Test OTEL_TRACES_SAMPLER is case-insensitive."""
        with patch.dict(
            os.environ,
            {"OTEL_TRACES_SAMPLER": "ALWAYS_ON"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert sampler is ALWAYS_ON

    # --- ROUTEIQ_OTEL_TRACES_SAMPLER tests ---

    def test_routeiq_sampler_always_on(self):
        """Test ROUTEIQ_OTEL_TRACES_SAMPLER=always_on."""
        with patch.dict(
            os.environ,
            {"ROUTEIQ_OTEL_TRACES_SAMPLER": "always_on"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert sampler is ALWAYS_ON

    def test_routeiq_sampler_always_off(self):
        """Test ROUTEIQ_OTEL_TRACES_SAMPLER=always_off."""
        with patch.dict(
            os.environ,
            {"ROUTEIQ_OTEL_TRACES_SAMPLER": "always_off"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert sampler is ALWAYS_OFF

    def test_routeiq_sampler_parentbased_traceidratio_with_arg(self):
        """Test ROUTEIQ_OTEL_TRACES_SAMPLER=parentbased_traceidratio with arg."""
        with patch.dict(
            os.environ,
            {
                "ROUTEIQ_OTEL_TRACES_SAMPLER": "parentbased_traceidratio",
                "ROUTEIQ_OTEL_TRACES_SAMPLER_ARG": "0.25",
            },
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, ParentBased)

    def test_routeiq_sampler_arg_only_uses_default_sampler(self):
        """Test ROUTEIQ_OTEL_TRACES_SAMPLER_ARG alone uses default sampler type."""
        with patch.dict(
            os.environ,
            {"ROUTEIQ_OTEL_TRACES_SAMPLER_ARG": "0.05"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            # Should use default sampler type (parentbased_traceidratio) with the arg
            assert isinstance(sampler, ParentBased)

    def test_routeiq_sampler_traceidratio(self):
        """Test ROUTEIQ_OTEL_TRACES_SAMPLER=traceidratio."""
        with patch.dict(
            os.environ,
            {
                "ROUTEIQ_OTEL_TRACES_SAMPLER": "traceidratio",
                "ROUTEIQ_OTEL_TRACES_SAMPLER_ARG": "0.3",
            },
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert isinstance(sampler, TraceIdRatioBased)
            assert sampler._rate == 0.3

    def test_routeiq_sampler_unknown_falls_back(self):
        """Test ROUTEIQ_OTEL_TRACES_SAMPLER with unknown value falls back."""
        with patch.dict(
            os.environ,
            {"ROUTEIQ_OTEL_TRACES_SAMPLER": "unknown_sampler"},
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            # Falls back to parentbased_traceidratio(0.1)
            assert isinstance(sampler, ParentBased)

    def test_otel_takes_precedence_over_routeiq(self):
        """Test OTEL_TRACES_SAMPLER takes precedence over ROUTEIQ_OTEL_TRACES_SAMPLER."""
        with patch.dict(
            os.environ,
            {
                "OTEL_TRACES_SAMPLER": "always_on",
                "ROUTEIQ_OTEL_TRACES_SAMPLER": "always_off",
            },
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert sampler is ALWAYS_ON

    def test_routeiq_takes_precedence_over_llmrouter(self):
        """Test ROUTEIQ_OTEL_TRACES_SAMPLER takes precedence over LLMROUTER_OTEL_SAMPLE_RATE."""
        with patch.dict(
            os.environ,
            {
                "ROUTEIQ_OTEL_TRACES_SAMPLER": "always_off",
                "LLMROUTER_OTEL_SAMPLE_RATE": "1.0",
            },
            clear=True,
        ):
            sampler = _get_sampler_from_env()
            assert sampler is ALWAYS_OFF

    def test_default_is_parentbased_traceidratio_10_percent(self):
        """Test default sampler is parentbased_traceidratio with 0.1 (10%)."""
        with patch.dict(os.environ, {}, clear=True):
            sampler = _get_sampler_from_env()
            # Default is ParentBased(TraceIdRatioBased(0.1))
            assert isinstance(sampler, ParentBased)


class TestObservabilityManagerSampler:
    """Test suite for ObservabilityManager sampler integration."""

    def test_manager_has_sampler_property(self):
        """Test that ObservabilityManager exposes sampler via property."""
        manager = ObservabilityManager()
        assert hasattr(manager, "sampler")
        assert manager.sampler is not None

    def test_manager_uses_env_sampler_by_default(self):
        """Test that manager uses env-based sampler when no explicit sampler provided."""
        with patch.dict(os.environ, {"OTEL_TRACES_SAMPLER": "always_off"}, clear=True):
            manager = ObservabilityManager()
            assert manager.sampler is ALWAYS_OFF

    def test_manager_accepts_custom_sampler(self):
        """Test that manager accepts a custom sampler parameter."""
        custom_sampler = TraceIdRatioBased(0.42)
        manager = ObservabilityManager(sampler=custom_sampler)
        assert manager.sampler is custom_sampler

    def test_manager_custom_sampler_overrides_env(self):
        """Test that explicit sampler parameter overrides env vars."""
        custom_sampler = ALWAYS_ON
        with patch.dict(os.environ, {"OTEL_TRACES_SAMPLER": "always_off"}, clear=True):
            manager = ObservabilityManager(sampler=custom_sampler)
            assert manager.sampler is ALWAYS_ON

    def test_manager_default_sampler_is_parentbased(self):
        """Test that default sampler is ParentBased for backwards compatibility."""
        with patch.dict(os.environ, {}, clear=True):
            manager = ObservabilityManager()
            assert isinstance(manager.sampler, ParentBased)


# Import the get_routeiq_resource_attributes function
get_routeiq_resource_attributes = observability.get_routeiq_resource_attributes


def test_custom_metrics_namespace(monkeypatch):
    """ROUTEIQ_METRICS_NAMESPACE configures OTel resource."""
    monkeypatch.setenv("ROUTEIQ_METRICS_NAMESPACE", "RouteIQ/prod")
    monkeypatch.setenv("ROUTEIQ_SERVICE_NAME", "routeiq-prod")
    monkeypatch.setenv("ROUTEIQ_DEPLOYMENT_ENV", "production")

    attrs = get_routeiq_resource_attributes()
    assert attrs["service.name"] == "routeiq-prod"
    assert attrs["deployment.environment"] == "production"
    assert attrs["routeiq.metrics.namespace"] == "RouteIQ/prod"


def test_default_metrics_namespace(monkeypatch):
    """Defaults work when env vars are not set."""
    monkeypatch.delenv("ROUTEIQ_SERVICE_NAME", raising=False)
    monkeypatch.delenv("ROUTEIQ_DEPLOYMENT_ENV", raising=False)
    monkeypatch.delenv("ROUTEIQ_METRICS_NAMESPACE", raising=False)

    attrs = get_routeiq_resource_attributes()
    assert attrs["service.name"] == "routeiq"
    assert attrs["deployment.environment"] == "default"
    assert attrs["routeiq.metrics.namespace"] == "RouteIQ"
