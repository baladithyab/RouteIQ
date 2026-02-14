"""Tests for plugin Protocol interfaces (structural typing)."""

from __future__ import annotations

from typing import Any

import pytest

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
# Dummy implementations that satisfy each Protocol structurally
# =============================================================================


class DummyMCP:
    def is_enabled(self) -> bool:
        return True

    async def register_server(self, server_id, name, url, **kw) -> dict[str, Any]:
        return {"server_id": server_id, "status": "registered"}

    async def unregister_server(self, server_id) -> bool:
        return True

    async def list_servers(self) -> list[dict[str, Any]]:
        return [{"server_id": "s1"}]

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "tool1"}]

    async def invoke_tool(self, tool_name, arguments) -> dict[str, Any]:
        return {"success": True, "result": "ok"}


class DummyA2A:
    def is_enabled(self) -> bool:
        return False

    async def register_agent(self, agent_id, name, url, **kw) -> dict[str, Any]:
        return {"agent_id": agent_id}

    async def unregister_agent(self, agent_id) -> bool:
        return False

    async def list_agents(self, *, capability=None) -> list[dict[str, Any]]:
        return []

    async def invoke_agent(self, agent_id, request) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": 1, "result": {}}


class DummyConfigSync:
    async def force_sync(self) -> bool:
        return True

    def get_status(self) -> dict[str, Any]:
        return {"sync_enabled": True}

    def get_current_config(self) -> dict[str, Any]:
        return {"models": []}


class DummyRouting:
    def list_strategies(self) -> list[str]:
        return ["knn"]

    def get_active_strategy(self) -> str | None:
        return "knn"

    def set_active_strategy(self, name) -> bool:
        return True

    def register_strategy(self, name, strategy, **kw) -> bool:
        return True

    def get_weights(self) -> dict[str, int]:
        return {}


class DummyResilience:
    def get_circuit_breaker_status(self) -> dict[str, dict[str, Any]]:
        return {}

    def force_open(self, name) -> bool:
        return True

    def force_close(self, name) -> bool:
        return True

    def is_degraded(self) -> bool:
        return False


class DummyModels:
    def list_models(self) -> list[dict[str, Any]]:
        return []

    def get_model(self, name) -> dict[str, Any] | None:
        return None


class DummyMetrics:
    def get_meter(self, name, version=""):
        return None

    def create_counter(self, name, **kw):
        return None

    def create_histogram(self, name, **kw):
        return None


# =============================================================================
# Tests
# =============================================================================


class TestProtocolStructuralTyping:
    """Verify that dummy implementations satisfy the runtime_checkable Protocols."""

    def test_mcp_accessor(self):
        assert isinstance(DummyMCP(), MCPGatewayAccessor)

    def test_a2a_accessor(self):
        assert isinstance(DummyA2A(), A2AGatewayAccessor)

    def test_config_sync_accessor(self):
        assert isinstance(DummyConfigSync(), ConfigSyncAccessor)

    def test_routing_accessor(self):
        assert isinstance(DummyRouting(), RoutingAccessor)

    def test_resilience_accessor(self):
        assert isinstance(DummyResilience(), ResilienceAccessor)

    def test_models_accessor(self):
        assert isinstance(DummyModels(), ModelsAccessor)

    def test_metrics_accessor(self):
        assert isinstance(DummyMetrics(), MetricsAccessor)


class TestProtocolRejectsNonConforming:
    """Verify that objects missing methods are NOT instances of the Protocol."""

    def test_empty_object_is_not_mcp_accessor(self):
        assert not isinstance(object(), MCPGatewayAccessor)

    def test_partial_impl_is_not_routing_accessor(self):
        class Partial:
            def list_strategies(self):
                return []

        assert not isinstance(Partial(), RoutingAccessor)


class TestDummyMethodBehavior:
    """Sanity-check that dummy implementations return expected values."""

    @pytest.mark.asyncio
    async def test_mcp_register_returns_dict(self):
        mcp = DummyMCP()
        result = await mcp.register_server("id", "name", "http://localhost")
        assert result["server_id"] == "id"

    @pytest.mark.asyncio
    async def test_a2a_is_disabled(self):
        a2a = DummyA2A()
        assert not a2a.is_enabled()

    @pytest.mark.asyncio
    async def test_config_sync_force_sync(self):
        cs = DummyConfigSync()
        assert await cs.force_sync() is True

    def test_routing_list_strategies(self):
        r = DummyRouting()
        assert "knn" in r.list_strategies()

    def test_resilience_not_degraded(self):
        res = DummyResilience()
        assert not res.is_degraded()
