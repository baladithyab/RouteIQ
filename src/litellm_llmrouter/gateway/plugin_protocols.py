"""
Plugin Subsystem Protocols (Typed Interfaces)
==============================================

Defines Protocol-based interfaces that expose gateway subsystems to plugins
via PluginContext. Each Protocol is a thin, typed facade over an existing
singleton (MCP Gateway, A2A Gateway, Config Sync, etc.).

Design principles:
- Protocols, not ABCs: plugins only depend on structural typing
- All methods return simple data types (no internal implementation details leak)
- Default ``None`` on PluginContext fields: plugins opt-in to what they need
- Adapters (plugin_adapters.py) do the heavy lifting of bridging singletons
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# =============================================================================
# MCP Gateway Accessor
# =============================================================================


@runtime_checkable
class MCPGatewayAccessor(Protocol):
    """Typed interface for MCP Gateway operations available to plugins."""

    def is_enabled(self) -> bool:
        """Return whether the MCP gateway is enabled."""
        ...

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
        """
        Register an MCP server with the gateway.

        Args:
            server_id: Unique server identifier.
            name: Human-readable server name.
            url: Server endpoint URL (validated against SSRF).
            transport: Transport type (sse, stdio, streamable_http).
            tools: Optional pre-declared tool definitions.
            metadata: Optional metadata dict.

        Returns:
            Dict with registration details (server_id, status).

        Raises:
            ValueError: If the URL fails SSRF validation.
        """
        ...

    async def unregister_server(self, server_id: str) -> bool:
        """
        Unregister an MCP server.

        Returns:
            True if the server was removed, False if not found.
        """
        ...

    async def list_servers(self) -> list[dict[str, Any]]:
        """Return a list of registered MCP server dicts."""
        ...

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the aggregated tool list across all servers."""
        ...

    async def invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Invoke an MCP tool by name.

        Args:
            tool_name: Fully-qualified tool name.
            arguments: Tool input arguments.

        Returns:
            Dict with ``success`` bool and ``result`` or ``error``.
        """
        ...


# =============================================================================
# A2A Gateway Accessor
# =============================================================================


@runtime_checkable
class A2AGatewayAccessor(Protocol):
    """Typed interface for A2A Gateway operations available to plugins."""

    def is_enabled(self) -> bool:
        """Return whether the A2A gateway is enabled."""
        ...

    async def register_agent(
        self,
        agent_id: str,
        name: str,
        url: str,
        *,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Register an A2A agent.

        Returns:
            Dict with registration details (agent_id, status).
        """
        ...

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an A2A agent.

        Returns:
            True if agent was removed, False if not found.
        """
        ...

    async def list_agents(
        self, *, capability: str | None = None
    ) -> list[dict[str, Any]]:
        """List registered agents, optionally filtered by capability."""
        ...

    async def invoke_agent(
        self,
        agent_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Invoke an A2A agent synchronously (non-streaming).

        Args:
            agent_id: Target agent identifier.
            request: JSON-RPC 2.0 request payload.

        Returns:
            JSON-RPC 2.0 response dict.
        """
        ...


# =============================================================================
# Config Sync Accessor
# =============================================================================


@runtime_checkable
class ConfigSyncAccessor(Protocol):
    """Typed interface for configuration sync status and control."""

    async def force_sync(self) -> bool:
        """
        Force an immediate configuration sync from remote storage.

        Returns:
            True if sync succeeded, False otherwise.
        """
        ...

    def get_status(self) -> dict[str, Any]:
        """
        Return current sync status as a plain dict.

        Keys include: sync_enabled, last_sync_success, config_version_hash,
        model_count, config_source.
        """
        ...

    def get_current_config(self) -> dict[str, Any]:
        """Return a snapshot of the current gateway configuration."""
        ...


# =============================================================================
# Routing Accessor
# =============================================================================


@runtime_checkable
class RoutingAccessor(Protocol):
    """Typed interface for routing strategy inspection and control."""

    def list_strategies(self) -> list[str]:
        """Return names of all registered routing strategies."""
        ...

    def get_active_strategy(self) -> str | None:
        """Return the name of the currently active strategy, or None."""
        ...

    def set_active_strategy(self, name: str) -> bool:
        """
        Set the active routing strategy by name.

        Returns:
            True if the strategy was activated, False if not found.
        """
        ...

    def register_strategy(
        self,
        name: str,
        strategy: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Register a new routing strategy at runtime.

        Args:
            name: Unique strategy name.
            strategy: Strategy object (must conform to RoutingStrategy interface).
            metadata: Optional metadata.

        Returns:
            True if registered successfully.
        """
        ...

    def get_weights(self) -> dict[str, int]:
        """Return current A/B testing weights (strategy_name -> weight)."""
        ...


# =============================================================================
# Resilience Accessor
# =============================================================================


@runtime_checkable
class ResilienceAccessor(Protocol):
    """Typed interface for circuit breaker and resilience inspection."""

    def get_circuit_breaker_status(self) -> dict[str, dict[str, Any]]:
        """
        Return status of all circuit breakers.

        Returns:
            Dict mapping breaker name to status dict with keys:
            state, failure_count, success_count, last_failure_time.
        """
        ...

    def force_open(self, breaker_name: str) -> bool:
        """
        Force a circuit breaker open (for maintenance/testing).

        Returns:
            True if breaker was found and opened.
        """
        ...

    def force_close(self, breaker_name: str) -> bool:
        """
        Force a circuit breaker closed (recover from forced open).

        Returns:
            True if breaker was found and closed.
        """
        ...

    def is_degraded(self) -> bool:
        """Return True if any circuit breaker is open (degraded mode)."""
        ...


# =============================================================================
# Models Accessor
# =============================================================================


@runtime_checkable
class ModelsAccessor(Protocol):
    """Typed interface for querying available LLM model deployments."""

    def list_models(self) -> list[dict[str, Any]]:
        """
        Return list of model deployment dicts.

        Each dict has at least: model_name, litellm_params.model.
        """
        ...

    def get_model(self, model_name: str) -> dict[str, Any] | None:
        """
        Get a single model deployment by name.

        Returns:
            Model deployment dict, or None if not found.
        """
        ...


# =============================================================================
# Metrics Accessor
# =============================================================================


@runtime_checkable
class MetricsAccessor(Protocol):
    """Typed interface for creating and recording OTel metrics."""

    def get_meter(self, name: str, version: str = "") -> Any:
        """
        Get an OTel Meter scoped to the given name.

        Args:
            name: Meter name (typically plugin name).
            version: Optional version string.

        Returns:
            An ``opentelemetry.metrics.Meter`` instance (or no-op stub).
        """
        ...

    def create_counter(
        self,
        name: str,
        *,
        unit: str = "",
        description: str = "",
    ) -> Any:
        """Create and return an OTel Counter instrument."""
        ...

    def create_histogram(
        self,
        name: str,
        *,
        unit: str = "",
        description: str = "",
    ) -> Any:
        """Create and return an OTel Histogram instrument."""
        ...


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MCPGatewayAccessor",
    "A2AGatewayAccessor",
    "ConfigSyncAccessor",
    "RoutingAccessor",
    "ResilienceAccessor",
    "ModelsAccessor",
    "MetricsAccessor",
]
