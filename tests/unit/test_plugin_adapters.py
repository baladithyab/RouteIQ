"""Tests for plugin concrete adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from litellm_llmrouter.gateway.plugin_adapters import (
    MCPGatewayAdapter,
    A2AGatewayAdapter,
    ConfigSyncAdapter,
    RoutingAdapter,
    ResilienceAdapter,
    ModelsAdapter,
    MetricsAdapter,
    create_all_adapters,
    _NoOpMeter,
    _NoOpInstrument,
)
from litellm_llmrouter.gateway.plugin_protocols import (
    MCPGatewayAccessor,
    A2AGatewayAccessor,
    ConfigSyncAccessor,
    RoutingAccessor,
    ResilienceAccessor,
    ModelsAccessor,
    MetricsAccessor,
)


# =============================================================================
# Protocol conformance
# =============================================================================


class TestAdaptersConformToProtocols:
    """Verify adapters satisfy their Protocol interfaces at runtime."""

    def test_mcp_adapter_is_mcp_accessor(self):
        assert isinstance(MCPGatewayAdapter(), MCPGatewayAccessor)

    def test_a2a_adapter_is_a2a_accessor(self):
        assert isinstance(A2AGatewayAdapter(), A2AGatewayAccessor)

    def test_config_sync_adapter_is_config_sync_accessor(self):
        assert isinstance(ConfigSyncAdapter(), ConfigSyncAccessor)

    def test_routing_adapter_is_routing_accessor(self):
        assert isinstance(RoutingAdapter(), RoutingAccessor)

    def test_resilience_adapter_is_resilience_accessor(self):
        assert isinstance(ResilienceAdapter(), ResilienceAccessor)

    def test_models_adapter_is_models_accessor(self):
        assert isinstance(ModelsAdapter(), ModelsAccessor)

    def test_metrics_adapter_is_metrics_accessor(self):
        assert isinstance(MetricsAdapter(), MetricsAccessor)


# =============================================================================
# Factory
# =============================================================================


class TestCreateAllAdapters:
    """Test the adapter factory function."""

    def test_returns_all_seven_adapters(self):
        adapters = create_all_adapters()
        assert set(adapters.keys()) == {
            "mcp",
            "a2a",
            "config_sync",
            "routing",
            "resilience",
            "models",
            "metrics",
        }

    def test_adapters_are_correct_types(self):
        adapters = create_all_adapters()
        assert isinstance(adapters["mcp"], MCPGatewayAdapter)
        assert isinstance(adapters["a2a"], A2AGatewayAdapter)
        assert isinstance(adapters["config_sync"], ConfigSyncAdapter)
        assert isinstance(adapters["routing"], RoutingAdapter)
        assert isinstance(adapters["resilience"], ResilienceAdapter)
        assert isinstance(adapters["models"], ModelsAdapter)
        assert isinstance(adapters["metrics"], MetricsAdapter)


# =============================================================================
# MCP Adapter
# =============================================================================


class TestMCPGatewayAdapter:
    def test_is_enabled_defaults_false(self):
        adapter = MCPGatewayAdapter()
        assert adapter.is_enabled() is False

    def test_is_enabled_true(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_ENABLED", "true")
        adapter = MCPGatewayAdapter()
        assert adapter.is_enabled() is True


# =============================================================================
# A2A Adapter
# =============================================================================


class TestA2AGatewayAdapter:
    def test_is_enabled_defaults_false(self):
        adapter = A2AGatewayAdapter()
        assert adapter.is_enabled() is False

    def test_is_enabled_true(self, monkeypatch):
        monkeypatch.setenv("A2A_GATEWAY_ENABLED", "true")
        adapter = A2AGatewayAdapter()
        assert adapter.is_enabled() is True


# =============================================================================
# Routing Adapter
# =============================================================================


class TestRoutingAdapter:
    @patch("litellm_llmrouter.strategy_registry.get_routing_registry")
    def test_list_strategies(self, mock_get_registry):
        mock_registry = MagicMock()
        mock_registry.list_strategies.return_value = ["knn", "svm"]
        mock_get_registry.return_value = mock_registry

        adapter = RoutingAdapter()
        result = adapter.list_strategies()
        assert result == ["knn", "svm"]

    @patch("litellm_llmrouter.strategy_registry.get_routing_registry")
    def test_get_active_strategy(self, mock_get_registry):
        mock_registry = MagicMock()
        mock_registry.get_active.return_value = "knn"
        mock_get_registry.return_value = mock_registry

        adapter = RoutingAdapter()
        assert adapter.get_active_strategy() == "knn"

    @patch("litellm_llmrouter.strategy_registry.get_routing_registry")
    def test_set_active_strategy(self, mock_get_registry):
        mock_registry = MagicMock()
        mock_registry.set_active.return_value = True
        mock_get_registry.return_value = mock_registry

        adapter = RoutingAdapter()
        assert adapter.set_active_strategy("knn") is True
        mock_registry.set_active.assert_called_once_with("knn")


# =============================================================================
# Resilience Adapter
# =============================================================================


class TestResilienceAdapter:
    @patch("litellm_llmrouter.resilience.get_circuit_breaker_manager")
    def test_is_degraded(self, mock_get_cbm):
        mock_cbm = MagicMock()
        mock_cbm.is_degraded.return_value = False
        mock_get_cbm.return_value = mock_cbm

        adapter = ResilienceAdapter()
        assert adapter.is_degraded() is False

    @patch("litellm_llmrouter.resilience.get_circuit_breaker_manager")
    def test_get_circuit_breaker_status(self, mock_get_cbm):
        mock_cbm = MagicMock()
        mock_cbm.get_status.return_value = {
            "breakers": {"database": {"state": "closed"}},
            "is_degraded": False,
        }
        mock_get_cbm.return_value = mock_cbm

        adapter = ResilienceAdapter()
        result = adapter.get_circuit_breaker_status()
        assert "database" in result


# =============================================================================
# Models Adapter
# =============================================================================


class TestModelsAdapter:
    def test_list_models_returns_empty_when_no_router(self):
        adapter = ModelsAdapter()
        # Will fail to import llm_router or it will be None
        result = adapter.list_models()
        assert isinstance(result, list)

    def test_get_model_returns_none_when_no_router(self):
        adapter = ModelsAdapter()
        result = adapter.get_model("nonexistent")
        assert result is None


# =============================================================================
# Metrics Adapter
# =============================================================================


class TestMetricsAdapter:
    def test_get_meter_returns_something(self):
        adapter = MetricsAdapter()
        meter = adapter.get_meter("test-plugin")
        # Should return either an OTel meter or NoOpMeter
        assert meter is not None

    def test_create_counter(self):
        adapter = MetricsAdapter()
        counter = adapter.create_counter("test.counter", unit="1", description="test")
        assert counter is not None

    def test_create_histogram(self):
        adapter = MetricsAdapter()
        hist = adapter.create_histogram("test.hist", unit="ms", description="test")
        assert hist is not None


# =============================================================================
# No-op stubs
# =============================================================================


class TestNoOpStubs:
    def test_noop_meter_creates_instruments(self):
        meter = _NoOpMeter()
        counter = meter.create_counter("x")
        assert isinstance(counter, _NoOpInstrument)

    def test_noop_instrument_record_does_nothing(self):
        inst = _NoOpInstrument()
        inst.record(1.0, {"key": "value"})  # Should not raise

    def test_noop_instrument_add_does_nothing(self):
        inst = _NoOpInstrument()
        inst.add(5, {"key": "value"})  # Should not raise
