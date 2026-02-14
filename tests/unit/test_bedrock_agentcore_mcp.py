"""Tests for the Bedrock AgentCore MCP reference plugin."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm_llmrouter.gateway.plugins.bedrock_agentcore_mcp import (
    BedrockAgentCoreMCPPlugin,
)
from litellm_llmrouter.gateway.plugin_manager import PluginCapability, PluginContext


class TestBedrockAgentCoreMCPPluginMetadata:
    def test_name(self):
        plugin = BedrockAgentCoreMCPPlugin()
        assert plugin.name == "bedrock-agentcore-mcp"

    def test_version(self):
        plugin = BedrockAgentCoreMCPPlugin()
        assert plugin.metadata.version == "0.2.0"

    def test_capabilities(self):
        plugin = BedrockAgentCoreMCPPlugin()
        assert PluginCapability.TOOL_RUNTIME in plugin.metadata.capabilities
        assert PluginCapability.OBSERVABILITY_EXPORTER in plugin.metadata.capabilities

    def test_priority(self):
        plugin = BedrockAgentCoreMCPPlugin()
        assert plugin.metadata.priority == 500


class TestBedrockAgentCoreMCPPluginStartup:
    @pytest.mark.asyncio
    async def test_startup_without_context(self):
        """Plugin should handle missing context gracefully."""
        plugin = BedrockAgentCoreMCPPlugin()
        await plugin.startup(MagicMock(), context=None)
        assert len(plugin._registered_server_ids) == 0

    @pytest.mark.asyncio
    async def test_startup_with_empty_agent_ids(self):
        """Plugin should not register any servers without agent IDs."""
        plugin = BedrockAgentCoreMCPPlugin()
        plugin._agent_ids = []

        mock_mcp = MagicMock()
        mock_mcp.is_enabled.return_value = True
        mock_mcp.register_server = AsyncMock()

        context = PluginContext(mcp=mock_mcp)
        await plugin.startup(MagicMock(), context=context)
        mock_mcp.register_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_startup_registers_agents_as_mcp_servers(self):
        """Plugin should register each agent as an MCP server."""
        plugin = BedrockAgentCoreMCPPlugin()
        plugin._agent_ids = ["agent-1", "agent-2"]

        mock_mcp = MagicMock()
        mock_mcp.is_enabled.return_value = True
        mock_mcp.register_server = AsyncMock(
            return_value={"server_id": "test", "status": "registered"}
        )

        mock_metrics = MagicMock()
        mock_counter = MagicMock()
        mock_metrics.create_counter.return_value = mock_counter

        context = PluginContext(mcp=mock_mcp, metrics=mock_metrics)
        await plugin.startup(MagicMock(), context=context)

        assert mock_mcp.register_server.call_count == 2
        assert len(plugin._registered_server_ids) == 2
        assert "bedrock-agentcore-agent-1" in plugin._registered_server_ids
        assert "bedrock-agentcore-agent-2" in plugin._registered_server_ids

    @pytest.mark.asyncio
    async def test_startup_skips_when_mcp_disabled(self):
        """Plugin should skip registration when MCP is disabled."""
        plugin = BedrockAgentCoreMCPPlugin()
        plugin._agent_ids = ["agent-1"]

        mock_mcp = MagicMock()
        mock_mcp.is_enabled.return_value = False

        context = PluginContext(mcp=mock_mcp)
        await plugin.startup(MagicMock(), context=context)
        mock_mcp.register_server.assert_not_called()


class TestBedrockAgentCoreMCPPluginShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_unregisters_servers(self):
        """Plugin should unregister all MCP servers on shutdown."""
        plugin = BedrockAgentCoreMCPPlugin()
        plugin._registered_server_ids = ["server-1", "server-2"]

        mock_mcp = MagicMock()
        mock_mcp.unregister_server = AsyncMock(return_value=True)

        context = PluginContext(mcp=mock_mcp)
        plugin._context = context
        await plugin.shutdown(MagicMock(), context=context)

        assert mock_mcp.unregister_server.call_count == 2
        assert len(plugin._registered_server_ids) == 0


class TestBedrockAgentCoreMCPPluginHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_ok(self):
        """Health check reports ok when all servers are registered."""
        plugin = BedrockAgentCoreMCPPlugin()
        plugin._agent_ids = ["agent-1"]
        plugin._registered_server_ids = ["bedrock-agentcore-agent-1"]

        mock_mcp = MagicMock()
        mock_mcp.list_servers = AsyncMock(
            return_value=[{"server_id": "bedrock-agentcore-agent-1"}]
        )

        plugin._context = PluginContext(mcp=mock_mcp)

        result = await plugin.health_check()
        assert result["status"] == "ok"
        assert result["agents_configured"] == 1

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_server_missing(self):
        """Health check reports degraded when MCP server is missing."""
        plugin = BedrockAgentCoreMCPPlugin()
        plugin._agent_ids = ["agent-1"]
        plugin._registered_server_ids = ["bedrock-agentcore-agent-1"]
        plugin._last_health_check = 0  # Force fresh check

        mock_mcp = MagicMock()
        mock_mcp.list_servers = AsyncMock(return_value=[])

        plugin._context = PluginContext(mcp=mock_mcp)

        result = await plugin.health_check()
        assert result["status"] == "degraded"


class TestBedrockAgentCoreMCPPluginConfigReload:
    @pytest.mark.asyncio
    async def test_config_reload_updates_agents(self):
        """Config reload re-registers servers when agent IDs change."""
        plugin = BedrockAgentCoreMCPPlugin()
        plugin._agent_ids = ["agent-1"]
        plugin._registered_server_ids = ["bedrock-agentcore-agent-1"]

        mock_mcp = MagicMock()
        mock_mcp.is_enabled.return_value = True
        mock_mcp.unregister_server = AsyncMock(return_value=True)
        mock_mcp.register_server = AsyncMock(
            return_value={"server_id": "test", "status": "registered"}
        )

        plugin._context = PluginContext(mcp=mock_mcp)

        # New config with different agent IDs
        await plugin.on_config_reload(
            old_config={},
            new_config={"bedrock_agentcore_agent_ids": ["agent-2", "agent-3"]},
        )

        # Old server should be unregistered
        mock_mcp.unregister_server.assert_called_once_with("bedrock-agentcore-agent-1")
        # New servers should be registered
        assert mock_mcp.register_server.call_count == 2
        assert plugin._agent_ids == ["agent-2", "agent-3"]

    @pytest.mark.asyncio
    async def test_config_reload_noop_when_no_change(self):
        """Config reload is a no-op when agent IDs haven't changed."""
        plugin = BedrockAgentCoreMCPPlugin()
        plugin._agent_ids = ["agent-1"]

        mock_mcp = MagicMock()
        plugin._context = PluginContext(mcp=mock_mcp)

        await plugin.on_config_reload(
            old_config={},
            new_config={"bedrock_agentcore_agent_ids": ["agent-1"]},
        )

        mock_mcp.unregister_server.assert_not_called()


class TestBedrockAgentCoreMCPPluginManagementOps:
    @pytest.mark.asyncio
    async def test_on_management_operation_logs_key_generate(self):
        """on_management_operation logs key generation events."""
        plugin = BedrockAgentCoreMCPPlugin()

        # Should not raise
        await plugin.on_management_operation(
            operation="key.generate",
            resource_type="key",
            method="POST",
            path="/key/generate",
            metadata={"status_code": 200, "outcome": "success"},
        )

    @pytest.mark.asyncio
    async def test_on_management_operation_ignores_irrelevant_ops(self):
        """on_management_operation is a no-op for non-key operations."""
        plugin = BedrockAgentCoreMCPPlugin()

        # Should not raise
        await plugin.on_management_operation(
            operation="team.create",
            resource_type="team",
            method="POST",
            path="/team/new",
        )
