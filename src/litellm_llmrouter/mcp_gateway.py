"""
MCP Gateway - Model Context Protocol Support
=============================================

Provides MCP (Model Context Protocol) gateway functionality for LiteLLM.
MCP is a protocol for connecting AI models to external tools and data sources.

See: https://modelcontextprotocol.io/
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from litellm._logging import verbose_proxy_logger


class MCPTransport(str, Enum):
    """MCP transport types."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class MCPServer:
    """Represents an MCP server registration."""

    server_id: str
    name: str
    url: str
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP
    tools: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    auth_type: str = "none"  # none, api_key, bearer_token, oauth2
    metadata: dict[str, Any] = field(default_factory=dict)


class MCPGateway:
    """
    MCP Gateway for managing MCP server connections.

    This gateway allows:
    - Registering MCP servers
    - Discovering available tools and resources
    - Proxying MCP requests
    """

    def __init__(self):
        self.servers: dict[str, MCPServer] = {}
        self.enabled = os.getenv("MCP_GATEWAY_ENABLED", "false").lower() == "true"

    def is_enabled(self) -> bool:
        """Check if MCP gateway is enabled."""
        return self.enabled

    def register_server(self, server: MCPServer) -> None:
        """Register an MCP server with the gateway."""
        if not self.enabled:
            verbose_proxy_logger.warning("MCP Gateway is not enabled")
            return

        self.servers[server.server_id] = server
        verbose_proxy_logger.info(
            f"MCP: Registered server {server.name} ({server.server_id})"
        )

    def unregister_server(self, server_id: str) -> bool:
        """Unregister an MCP server from the gateway."""
        if server_id in self.servers:
            del self.servers[server_id]
            verbose_proxy_logger.info(f"MCP: Unregistered server {server_id}")
            return True
        return False

    def get_server(self, server_id: str) -> MCPServer | None:
        """Get an MCP server by ID."""
        return self.servers.get(server_id)

    def list_servers(self) -> list[MCPServer]:
        """List all registered MCP servers."""
        return list(self.servers.values())

    def list_tools(self) -> list[dict[str, Any]]:
        """List all tools from all registered servers."""
        tools = []
        for server in self.servers.values():
            for tool in server.tools:
                tools.append(
                    {
                        "server_id": server.server_id,
                        "server_name": server.name,
                        "tool": tool,
                    }
                )
        return tools

    def list_resources(self) -> list[dict[str, Any]]:
        """List all resources from all registered servers."""
        resources = []
        for server in self.servers.values():
            for resource in server.resources:
                resources.append(
                    {
                        "server_id": server.server_id,
                        "server_name": server.name,
                        "resource": resource,
                    }
                )
        return resources


# Singleton instance
_mcp_gateway: MCPGateway | None = None


def get_mcp_gateway() -> MCPGateway:
    """Get the global MCP gateway instance."""
    global _mcp_gateway
    if _mcp_gateway is None:
        _mcp_gateway = MCPGateway()
    return _mcp_gateway
