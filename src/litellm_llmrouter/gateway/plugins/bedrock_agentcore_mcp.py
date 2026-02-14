"""
Bedrock AgentCore MCP Connector Plugin
=======================================

Reference plugin that demonstrates the v0.2.0 Universal PluginContext by
connecting Amazon Bedrock AgentCore tool-use capabilities to the MCP gateway.

This plugin:
- Uses ``context.mcp`` to register/unregister MCP servers
- Uses ``context.metrics`` to emit custom counters
- Uses ``context.validate_outbound_url`` for SSRF protection
- Implements ``on_config_reload`` to refresh server list
- Implements ``health_check`` to report AgentCore connectivity
- Implements ``on_management_operation`` to observe key/model changes

Configuration via environment variables:
- ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_ENABLED: Enable this plugin (default: false)
- ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_REGION: AWS region (default: us-east-1)
- ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_AGENT_IDS: Comma-separated agent IDs to expose
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, TYPE_CHECKING

from ..plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


class BedrockAgentCoreMCPPlugin(GatewayPlugin):
    """
    Exposes Amazon Bedrock AgentCore agents as MCP tool servers.

    Each configured AgentCore agent is registered as an MCP server,
    making its tools discoverable and invokable through the MCP gateway.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="bedrock-agentcore-mcp",
            version="0.2.0",
            capabilities={
                PluginCapability.TOOL_RUNTIME,
                PluginCapability.OBSERVABILITY_EXPORTER,
            },
            priority=500,
            description="Connects Bedrock AgentCore agents as MCP tool servers",
        )

    def __init__(self) -> None:
        self._region = os.getenv("ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_REGION", "us-east-1")
        self._agent_ids: list[str] = []
        raw_ids = os.getenv("ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_AGENT_IDS", "")
        if raw_ids.strip():
            self._agent_ids = [a.strip() for a in raw_ids.split(",") if a.strip()]

        self._registered_server_ids: list[str] = []
        self._context: PluginContext | None = None
        self._tools_registered_counter: Any = None
        self._invocations_counter: Any = None
        self._last_health_check: float = 0.0
        self._healthy = True

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Register AgentCore agents as MCP servers."""
        if context is None:
            logger.warning("BedrockAgentCoreMCPPlugin: No context provided, skipping")
            return

        self._context = context

        # Set up metrics
        if context.metrics:
            self._tools_registered_counter = context.metrics.create_counter(
                "routeiq.plugin.bedrock_agentcore.tools_registered",
                unit="{tool}",
                description="Number of AgentCore tools registered as MCP servers",
            )
            self._invocations_counter = context.metrics.create_counter(
                "routeiq.plugin.bedrock_agentcore.invocations",
                unit="{invocation}",
                description="Number of AgentCore tool invocations",
            )

        # Register agents as MCP servers
        await self._register_agents(context)

        logger.info(
            f"BedrockAgentCoreMCPPlugin started: "
            f"region={self._region}, "
            f"agents={len(self._agent_ids)}, "
            f"servers_registered={len(self._registered_server_ids)}"
        )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Unregister all MCP servers."""
        ctx = context or self._context
        if ctx and ctx.mcp:
            for server_id in self._registered_server_ids:
                try:
                    await ctx.mcp.unregister_server(server_id)
                except Exception as e:
                    logger.warning(f"Failed to unregister MCP server {server_id}: {e}")

        self._registered_server_ids.clear()
        logger.info("BedrockAgentCoreMCPPlugin shut down")

    async def health_check(self) -> dict[str, Any]:
        """Report plugin health based on AgentCore connectivity."""
        now = time.monotonic()

        # Cache health check for 30 seconds
        if now - self._last_health_check < 30:
            return {
                "status": "ok" if self._healthy else "degraded",
                "region": self._region,
                "agents_configured": len(self._agent_ids),
                "servers_registered": len(self._registered_server_ids),
            }

        self._last_health_check = now

        # Basic connectivity check
        try:
            # We don't actually call AgentCore here (too expensive for health checks).
            # Instead, verify our MCP servers are still registered.
            if self._context and self._context.mcp:
                servers = await self._context.mcp.list_servers()
                registered = {s.get("server_id") for s in servers}
                missing = [
                    sid for sid in self._registered_server_ids if sid not in registered
                ]
                if missing:
                    self._healthy = False
                    return {
                        "status": "degraded",
                        "reason": f"MCP servers missing: {missing}",
                        "region": self._region,
                    }
            self._healthy = True
        except Exception as e:
            self._healthy = False
            return {
                "status": "degraded",
                "reason": str(e),
                "region": self._region,
            }

        return {
            "status": "ok",
            "region": self._region,
            "agents_configured": len(self._agent_ids),
            "servers_registered": len(self._registered_server_ids),
        }

    async def on_config_reload(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        """Re-read agent IDs from config and re-register MCP servers."""
        # Check if agent IDs changed in the new config
        new_agent_ids = new_config.get("bedrock_agentcore_agent_ids", [])
        if isinstance(new_agent_ids, str):
            new_agent_ids = [a.strip() for a in new_agent_ids.split(",") if a.strip()]

        if not new_agent_ids:
            return  # No change — keep using env var config

        if set(new_agent_ids) != set(self._agent_ids):
            logger.info(
                f"BedrockAgentCoreMCPPlugin: Agent IDs changed, re-registering. "
                f"Old: {self._agent_ids}, New: {new_agent_ids}"
            )
            self._agent_ids = new_agent_ids

            if self._context:
                # Unregister old servers
                if self._context.mcp:
                    for server_id in self._registered_server_ids:
                        try:
                            await self._context.mcp.unregister_server(server_id)
                        except Exception:
                            pass
                self._registered_server_ids.clear()

                # Register new servers
                await self._register_agents(self._context)

    async def on_management_operation(
        self,
        operation: str,
        resource_type: str,
        method: str,
        path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Observe management operations for metrics and logging."""
        # Track key generation events (example of observing management ops)
        if operation == "key.generate":
            logger.debug(
                f"BedrockAgentCoreMCPPlugin: Key generated "
                f"(status={metadata.get('status_code') if metadata else 'unknown'})"
            )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    async def _register_agents(self, context: PluginContext) -> None:
        """Register configured AgentCore agents as MCP servers."""
        if not context.mcp:
            logger.debug("MCP accessor not available, skipping agent registration")
            return

        if not context.mcp.is_enabled():
            logger.debug("MCP gateway disabled, skipping agent registration")
            return

        for agent_id in self._agent_ids:
            server_id = f"bedrock-agentcore-{agent_id}"
            # Build the AgentCore endpoint URL
            url = (
                f"https://bedrock-agent-runtime.{self._region}.amazonaws.com"
                f"/agents/{agent_id}/agentAliases/TSTALIASID/sessions/test/text"
            )

            # Validate URL against SSRF if available
            if context.validate_outbound_url:
                try:
                    context.validate_outbound_url(url)
                except Exception as e:
                    logger.warning(f"SSRF validation failed for AgentCore URL: {e}")
                    continue

            try:
                await context.mcp.register_server(
                    server_id=server_id,
                    name=f"Bedrock AgentCore ({agent_id})",
                    url=url,
                    transport="streamable_http",
                    metadata={
                        "provider": "bedrock-agentcore",
                        "region": self._region,
                        "agent_id": agent_id,
                    },
                )
                self._registered_server_ids.append(server_id)

                # Emit metric
                if self._tools_registered_counter:
                    self._tools_registered_counter.add(
                        1,
                        {"agent_id": agent_id, "region": self._region},
                    )

                logger.info(f"Registered AgentCore agent as MCP server: {server_id}")

            except Exception as e:
                logger.warning(
                    f"Failed to register AgentCore agent {agent_id} as MCP server: {e}"
                )
