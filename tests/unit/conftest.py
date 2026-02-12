"""
Pytest configuration for unit tests.

This conftest provides a shared OpenTelemetry tracer provider for tracing tests,
ensuring all unit tests can properly export and verify spans.
"""

import pytest

# ============================================================================
# Shared OpenTelemetry configuration - set up once for all tracing tests
# ============================================================================

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Shared tracer provider and exporter for all tracing tests
_shared_exporter = InMemorySpanExporter()
_shared_provider = TracerProvider()
_shared_provider.add_span_processor(SimpleSpanProcessor(_shared_exporter))

# Set the global tracer provider once
trace.set_tracer_provider(_shared_provider)


@pytest.fixture
def shared_span_exporter():
    """
    Provides the shared span exporter for tests that need to verify spans.

    Clears spans before and after each test.
    """
    _shared_exporter.clear()
    yield _shared_exporter
    _shared_exporter.clear()


@pytest.fixture(autouse=True)
def _reset_observability_singleton():
    """Reset observability and metrics singletons between tests."""
    yield
    from litellm_llmrouter.observability import reset_observability_manager
    from litellm_llmrouter.config_sync import reset_config_sync_manager
    from litellm_llmrouter.hot_reload import reset_hot_reload_manager
    from litellm_llmrouter.metrics import reset_gateway_metrics

    reset_observability_manager()
    reset_config_sync_manager()
    reset_hot_reload_manager()
    reset_gateway_metrics()

    from litellm_llmrouter.migrations import reset_migration_state

    reset_migration_state()
