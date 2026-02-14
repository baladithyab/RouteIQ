"""
Plugin Subsystem Adapters (Concrete Implementations)
=====================================================

Concrete implementations of the Protocol interfaces defined in
``plugin_protocols.py``. Each adapter wraps the corresponding gateway
singleton and translates between the typed Protocol surface and the
internal implementation.

Adapters use lazy imports to avoid import-time coupling and to keep
the plugin system lightweight when subsystems are disabled.

Design principles:
- Lazy singleton access: adapters acquire the singleton on every call
  to tolerate reset_*() in tests and late initialisation
- Defensive: methods return safe defaults when the subsystem is disabled
  or not yet initialised (never raise on missing singleton)
- Thread-safe: delegates to the underlying singleton's locking
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# MCP Gateway Adapter
# =============================================================================


class MCPGatewayAdapter:
    """Adapts the MCPGateway singleton to the MCPGatewayAccessor protocol."""

    def is_enabled(self) -> bool:
        return os.getenv("MCP_GATEWAY_ENABLED", "false").lower() == "true"

    async def register_server(
        self,
        server_id: str,
        name: str,
        url: str,
        *,
        transport: str = "sse",
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from litellm_llmrouter.mcp_gateway import (
            get_mcp_gateway,
            MCPServer,
            MCPTransport,
        )

        transport_enum = MCPTransport(transport) if transport else MCPTransport.SSE
        server = MCPServer(
            server_id=server_id,
            name=name,
            url=url,
            transport=transport_enum,
        )
        gateway = get_mcp_gateway()
        gateway.register_server(server)

        # Register pre-declared tools if provided
        if tools:
            from litellm_llmrouter.mcp_gateway import MCPToolDefinition

            for tool_dict in tools:
                tool_def = MCPToolDefinition(
                    name=tool_dict.get("name", ""),
                    description=tool_dict.get("description", ""),
                    input_schema=tool_dict.get("input_schema", {}),
                    server_id=server_id,
                )
                gateway.register_tool_definition(server_id, tool_def)

        return {"server_id": server_id, "status": "registered"}

    async def unregister_server(self, server_id: str) -> bool:
        from litellm_llmrouter.mcp_gateway import get_mcp_gateway

        gateway = get_mcp_gateway()
        return gateway.unregister_server(server_id)

    async def list_servers(self) -> list[dict[str, Any]]:
        from litellm_llmrouter.mcp_gateway import get_mcp_gateway

        gateway = get_mcp_gateway()
        servers = gateway.list_servers()
        return [
            {
                "server_id": s.server_id,
                "name": s.name,
                "url": s.url,
                "transport": s.transport.value
                if hasattr(s.transport, "value")
                else str(s.transport),
            }
            for s in servers
        ]

    async def list_tools(self) -> list[dict[str, Any]]:
        from litellm_llmrouter.mcp_gateway import get_mcp_gateway

        gateway = get_mcp_gateway()
        return gateway.list_tools()

    async def invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        from litellm_llmrouter.mcp_gateway import get_mcp_gateway

        gateway = get_mcp_gateway()
        result = await gateway.invoke_tool(tool_name, arguments)
        return {
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "tool_name": result.tool_name,
            "server_id": result.server_id,
        }


# =============================================================================
# A2A Gateway Adapter
# =============================================================================


class A2AGatewayAdapter:
    """Adapts the A2AGateway singleton to the A2AGatewayAccessor protocol."""

    def is_enabled(self) -> bool:
        return os.getenv("A2A_GATEWAY_ENABLED", "false").lower() == "true"

    async def register_agent(
        self,
        agent_id: str,
        name: str,
        url: str,
        *,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from litellm_llmrouter.a2a_gateway import get_a2a_gateway, A2AAgent

        agent = A2AAgent(
            agent_id=agent_id,
            name=name,
            url=url,
            capabilities=capabilities or [],
        )
        gateway = get_a2a_gateway()
        gateway.register_agent(agent)
        return {"agent_id": agent_id, "status": "registered"}

    async def unregister_agent(self, agent_id: str) -> bool:
        from litellm_llmrouter.a2a_gateway import get_a2a_gateway

        gateway = get_a2a_gateway()
        return gateway.unregister_agent(agent_id)

    async def list_agents(
        self, *, capability: str | None = None
    ) -> list[dict[str, Any]]:
        from litellm_llmrouter.a2a_gateway import get_a2a_gateway

        gateway = get_a2a_gateway()
        agents = gateway.list_agents(capability=capability)
        return [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "url": a.url,
                "capabilities": list(a.capabilities) if a.capabilities else [],
            }
            for a in agents
        ]

    async def invoke_agent(
        self,
        agent_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        from litellm_llmrouter.a2a_gateway import get_a2a_gateway

        gateway = get_a2a_gateway()
        response = await gateway.invoke_agent(agent_id, request)
        return response


# =============================================================================
# Config Sync Adapter
# =============================================================================


class ConfigSyncAdapter:
    """Adapts config_sync singletons to the ConfigSyncAccessor protocol."""

    async def force_sync(self) -> bool:
        from litellm_llmrouter.config_sync import get_sync_manager

        manager = get_sync_manager()
        return manager.force_sync()

    def get_status(self) -> dict[str, Any]:
        from litellm_llmrouter.config_sync import get_config_sync_status

        status = get_config_sync_status()
        return {
            "config_source": status.config_source,
            "sync_enabled": status.sync_enabled,
            "sync_interval_seconds": status.sync_interval_seconds,
            "last_sync_attempt": status.last_sync_attempt,
            "last_sync_success": status.last_sync_success,
            "last_sync_error": status.last_sync_error,
            "config_version_hash": status.config_version_hash,
            "model_count": status.model_count,
            "next_sync_at": status.next_sync_at,
        }

    def get_current_config(self) -> dict[str, Any]:
        try:
            from litellm_llmrouter.config_sync import get_sync_manager

            manager = get_sync_manager()
            return getattr(manager, "current_config", {}) or {}
        except Exception:
            return {}


# =============================================================================
# Routing Adapter
# =============================================================================


class RoutingAdapter:
    """Adapts the RoutingStrategyRegistry to the RoutingAccessor protocol."""

    def list_strategies(self) -> list[str]:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        return registry.list_strategies()

    def get_active_strategy(self) -> str | None:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        return registry.get_active()

    def set_active_strategy(self, name: str) -> bool:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        return registry.set_active(name)

    def register_strategy(
        self,
        name: str,
        strategy: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        try:
            registry.register(name, strategy, metadata=metadata)
            return True
        except Exception as e:
            logger.warning(f"Failed to register strategy '{name}': {e}")
            return False

    def get_weights(self) -> dict[str, int]:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        snapshot = registry.snapshot()
        return dict(snapshot.weights) if snapshot.weights else {}


# =============================================================================
# Resilience Adapter
# =============================================================================


class ResilienceAdapter:
    """Adapts the CircuitBreakerManager to the ResilienceAccessor protocol."""

    def get_circuit_breaker_status(self) -> dict[str, dict[str, Any]]:
        from litellm_llmrouter.resilience import get_circuit_breaker_manager

        manager = get_circuit_breaker_manager()
        status = manager.get_status()
        return status.get("breakers", {})

    def force_open(self, breaker_name: str) -> bool:
        from litellm_llmrouter.resilience import get_circuit_breaker_manager

        manager = get_circuit_breaker_manager()
        try:
            breaker = manager.get_breaker(breaker_name)
            # force_open is async on CircuitBreaker, but we need sync here.
            # Use the internal state directly for the adapter.
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as a task; caller should await if needed
                asyncio.ensure_future(breaker.force_open())
            else:
                loop.run_until_complete(breaker.force_open())
            return True
        except Exception as e:
            logger.warning(f"Failed to force open breaker '{breaker_name}': {e}")
            return False

    def force_close(self, breaker_name: str) -> bool:
        from litellm_llmrouter.resilience import get_circuit_breaker_manager

        manager = get_circuit_breaker_manager()
        try:
            breaker = manager.get_breaker(breaker_name)
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(breaker.force_closed())
            else:
                loop.run_until_complete(breaker.force_closed())
            return True
        except Exception as e:
            logger.warning(f"Failed to force close breaker '{breaker_name}': {e}")
            return False

    def is_degraded(self) -> bool:
        from litellm_llmrouter.resilience import get_circuit_breaker_manager

        manager = get_circuit_breaker_manager()
        return manager.is_degraded()


# =============================================================================
# Models Adapter
# =============================================================================


class ModelsAdapter:
    """Adapts the LiteLLM Router model list to the ModelsAccessor protocol."""

    def list_models(self) -> list[dict[str, Any]]:
        try:
            from litellm.proxy.proxy_server import llm_router

            if llm_router is None:
                return []
            model_list = getattr(llm_router, "model_list", []) or []
            return [
                {
                    "model_name": m.get("model_name", ""),
                    "litellm_model": m.get("litellm_params", {}).get("model", ""),
                    "model_info": m.get("model_info", {}),
                }
                for m in model_list
            ]
        except Exception:
            return []

    def get_model(self, model_name: str) -> dict[str, Any] | None:
        try:
            from litellm.proxy.proxy_server import llm_router

            if llm_router is None:
                return None
            model_list = getattr(llm_router, "model_list", []) or []
            for m in model_list:
                if m.get("model_name") == model_name:
                    return {
                        "model_name": m.get("model_name", ""),
                        "litellm_model": m.get("litellm_params", {}).get("model", ""),
                        "model_info": m.get("model_info", {}),
                    }
            return None
        except Exception:
            return None


# =============================================================================
# Metrics Adapter
# =============================================================================


class MetricsAdapter:
    """Adapts OTel metrics to the MetricsAccessor protocol."""

    def get_meter(self, name: str, version: str = "") -> Any:
        try:
            from opentelemetry import metrics

            return metrics.get_meter(name, version)
        except Exception:
            return _NoOpMeter()

    def create_counter(
        self,
        name: str,
        *,
        unit: str = "",
        description: str = "",
    ) -> Any:
        try:
            from opentelemetry import metrics

            meter = metrics.get_meter("routeiq.plugin")
            return meter.create_counter(name, unit=unit, description=description)
        except Exception:
            return _NoOpInstrument()

    def create_histogram(
        self,
        name: str,
        *,
        unit: str = "",
        description: str = "",
    ) -> Any:
        try:
            from opentelemetry import metrics

            meter = metrics.get_meter("routeiq.plugin")
            return meter.create_histogram(name, unit=unit, description=description)
        except Exception:
            return _NoOpInstrument()


# =============================================================================
# No-op stubs (returned when OTel is not available)
# =============================================================================


class _NoOpMeter:
    """No-op meter returned when OTel is not initialized."""

    def create_counter(self, *args: Any, **kwargs: Any) -> "_NoOpInstrument":
        return _NoOpInstrument()

    def create_histogram(self, *args: Any, **kwargs: Any) -> "_NoOpInstrument":
        return _NoOpInstrument()

    def create_up_down_counter(self, *args: Any, **kwargs: Any) -> "_NoOpInstrument":
        return _NoOpInstrument()


class _NoOpInstrument:
    """No-op instrument that silently accepts record/add calls."""

    def record(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add(self, *args: Any, **kwargs: Any) -> None:
        pass


# =============================================================================
# Factory
# =============================================================================


def create_all_adapters() -> dict[str, Any]:
    """
    Create all subsystem adapters.

    Returns:
        Dict mapping accessor name to adapter instance.
    """
    return {
        "mcp": MCPGatewayAdapter(),
        "a2a": A2AGatewayAdapter(),
        "config_sync": ConfigSyncAdapter(),
        "routing": RoutingAdapter(),
        "resilience": ResilienceAdapter(),
        "models": ModelsAdapter(),
        "metrics": MetricsAdapter(),
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MCPGatewayAdapter",
    "A2AGatewayAdapter",
    "ConfigSyncAdapter",
    "RoutingAdapter",
    "ResilienceAdapter",
    "ModelsAdapter",
    "MetricsAdapter",
    "create_all_adapters",
]
