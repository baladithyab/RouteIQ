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
def _reset_all_singletons():
    """Reset all module-level singletons between tests to prevent cross-test contamination."""
    yield

    # --- Original 5 resets ---
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

    # --- Additional singleton resets ---
    from litellm_llmrouter.strategy_registry import reset_routing_singletons
    from litellm_llmrouter.mcp_gateway import reset_mcp_gateway
    from litellm_llmrouter.a2a_gateway import reset_a2a_gateway
    from litellm_llmrouter.resilience import (
        reset_drain_manager,
        reset_circuit_breaker_manager,
    )
    from litellm_llmrouter.policy_engine import reset_policy_engine
    from litellm_llmrouter.quota import reset_quota_enforcer
    from litellm_llmrouter.audit import reset_audit_repository
    from litellm_llmrouter.mcp_jsonrpc import reset_sessions
    from litellm_llmrouter.conversation_affinity import reset_affinity_tracker
    from litellm_llmrouter.gateway.plugin_manager import reset_plugin_manager
    from litellm_llmrouter.gateway.plugin_middleware import reset_plugin_middleware
    from litellm_llmrouter.gateway.plugin_callback_bridge import reset_callback_bridge
    from litellm_llmrouter.database import reset_database_singletons
    from litellm_llmrouter.semantic_cache import reset_cache_manager
    from litellm_llmrouter.model_artifacts import (
        reset_activation_manager,
        reset_artifact_verifier,
    )
    from litellm_llmrouter.leader_election import reset_leader_election
    from litellm_llmrouter.mcp_sse_transport import reset_sse_sessions

    reset_routing_singletons()
    reset_mcp_gateway()
    reset_a2a_gateway()
    reset_drain_manager()
    reset_circuit_breaker_manager()
    reset_policy_engine()
    reset_quota_enforcer()
    reset_audit_repository()
    reset_sessions()
    reset_affinity_tracker()
    reset_plugin_manager()
    reset_plugin_middleware()
    reset_callback_bridge()
    reset_database_singletons()
    reset_cache_manager()
    reset_activation_manager()
    reset_artifact_verifier()
    reset_leader_election()
    reset_sse_sessions()

    # NOTE: reset_http_client_pool() is async and cannot be called from
    # this sync fixture. The http_client_pool module has a fallback for
    # when the pool is not initialized, so skipping it is safe.

    # NOTE: _reset_embedder() from cache_plugin.py is private (underscore
    # prefix) and low-risk, so it's intentionally excluded.
