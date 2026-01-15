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
class MCPToolDefinition:
    """Represents an MCP tool definition with input schema."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_id: str = ""


@dataclass
class MCPToolResult:
    """Represents the result of an MCP tool invocation."""

    success: bool
    result: Any = None
    error: str | None = None
    tool_name: str = ""
    server_id: str = ""


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
    tool_definitions: dict[str, MCPToolDefinition] = field(default_factory=dict)


class MCPGateway:
    """
    MCP Gateway for managing MCP server connections.

    This gateway allows:
    - Registering MCP servers
    - Discovering available tools and resources
    - Invoking MCP tools
    - Proxying MCP requests
    """

    def __init__(self):
        self.servers: dict[str, MCPServer] = {}
        self.enabled = os.getenv("MCP_GATEWAY_ENABLED", "false").lower() == "true"
        # Map tool names to server IDs for quick lookup
        self._tool_to_server: dict[str, str] = {}

    def is_enabled(self) -> bool:
        """Check if MCP gateway is enabled."""
        return self.enabled

    def register_server(self, server: MCPServer) -> None:
        """Register an MCP server with the gateway."""
        if not self.enabled:
            verbose_proxy_logger.warning("MCP Gateway is not enabled")
            return

        self.servers[server.server_id] = server
        
        # Update tool-to-server mapping
        for tool_name in server.tools:
            self._tool_to_server[tool_name] = server.server_id
        
        verbose_proxy_logger.info(
            f"MCP: Registered server {server.name} ({server.server_id})"
        )

    def unregister_server(self, server_id: str) -> bool:
        """Unregister an MCP server from the gateway."""
        if server_id in self.servers:
            server = self.servers[server_id]
            # Remove tool mappings
            for tool_name in server.tools:
                if self._tool_to_server.get(tool_name) == server_id:
                    del self._tool_to_server[tool_name]
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

    def get_tool(self, tool_name: str) -> MCPToolDefinition | None:
        """
        Get a tool definition by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool definition if found, None otherwise
        """
        server_id = self._tool_to_server.get(tool_name)
        if not server_id:
            return None
        
        server = self.servers.get(server_id)
        if not server:
            return None
        
        # Check if we have a full definition
        if tool_name in server.tool_definitions:
            return server.tool_definitions[tool_name]
        
        # Return basic definition if tool exists in list
        if tool_name in server.tools:
            return MCPToolDefinition(
                name=tool_name,
                description=f"Tool from {server.name}",
                server_id=server_id,
            )
        
        return None

    def find_server_for_tool(self, tool_name: str) -> MCPServer | None:
        """
        Find the server that provides a given tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Server that provides the tool, or None if not found
        """
        server_id = self._tool_to_server.get(tool_name)
        if server_id:
            return self.servers.get(server_id)
        return None

    def register_tool_definition(
        self,
        server_id: str,
        tool: MCPToolDefinition,
    ) -> bool:
        """
        Register a detailed tool definition for a server.
        
        Args:
            server_id: ID of the server providing the tool
            tool: Tool definition with schema
            
        Returns:
            True if registered successfully, False otherwise
        """
        server = self.servers.get(server_id)
        if not server:
            return False
        
        tool.server_id = server_id
        server.tool_definitions[tool.name] = tool
        
        # Add to tools list if not already there
        if tool.name not in server.tools:
            server.tools.append(tool.name)
            self._tool_to_server[tool.name] = server_id
        
        verbose_proxy_logger.info(
            f"MCP: Registered tool definition {tool.name} for server {server_id}"
        )
        return True

    async def invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """
        Invoke an MCP tool.
        
        Args:
            tool_name: Name of the tool to invoke
            arguments: Arguments to pass to the tool
            
        Returns:
            Result of the tool invocation
        """
        server = self.find_server_for_tool(tool_name)
        if not server:
            return MCPToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
            )
        
        # Validate arguments against schema if available
        tool_def = server.tool_definitions.get(tool_name)
        if tool_def and tool_def.input_schema:
            validation_error = self._validate_arguments(
                arguments, tool_def.input_schema
            )
            if validation_error:
                return MCPToolResult(
                    success=False,
                    error=validation_error,
                    tool_name=tool_name,
                    server_id=server.server_id,
                )
        
        # In a real implementation, this would make an HTTP/SSE/stdio call
        # to the MCP server. For now, we return a placeholder result.
        verbose_proxy_logger.info(
            f"MCP: Invoking tool {tool_name} on server {server.server_id}"
        )
        
        # Placeholder - actual implementation would call the MCP server
        return MCPToolResult(
            success=True,
            result={"message": f"Tool {tool_name} invoked successfully"},
            tool_name=tool_name,
            server_id=server.server_id,
        )

    def _validate_arguments(
        self,
        arguments: dict[str, Any],
        schema: dict[str, Any],
    ) -> str | None:
        """
        Validate arguments against a JSON schema.
        
        Args:
            arguments: Arguments to validate
            schema: JSON schema to validate against
            
        Returns:
            Error message if validation fails, None if valid
        """
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in arguments:
                return f"Missing required argument: {field}"
        
        # Check property types (basic validation)
        properties = schema.get("properties", {})
        for arg_name, arg_value in arguments.items():
            if arg_name in properties:
                prop_schema = properties[arg_name]
                expected_type = prop_schema.get("type")
                if expected_type:
                    if not self._check_type(arg_value, expected_type):
                        return f"Invalid type for '{arg_name}': expected {expected_type}"
        
        return None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected JSON schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True  # Unknown type, allow
        return isinstance(value, expected)

    async def check_server_health(self, server_id: str) -> dict[str, Any]:
        """
        Check the health of an MCP server.
        
        Args:
            server_id: ID of the server to check
            
        Returns:
            Health status dict with status, latency, and error info
        """
        import time
        
        server = self.servers.get(server_id)
        if not server:
            return {
                "server_id": server_id,
                "status": "not_found",
                "error": f"Server {server_id} not found",
            }
        
        start_time = time.time()
        
        # In a real implementation, this would make an HTTP request to the server
        # For now, we simulate a health check based on URL validity
        try:
            # Simulate connectivity check
            if server.url and server.url.startswith(("http://", "https://")):
                latency_ms = int((time.time() - start_time) * 1000)
                return {
                    "server_id": server_id,
                    "name": server.name,
                    "url": server.url,
                    "status": "healthy",
                    "latency_ms": latency_ms,
                    "transport": server.transport.value,
                    "tool_count": len(server.tools),
                    "resource_count": len(server.resources),
                }
            else:
                return {
                    "server_id": server_id,
                    "name": server.name,
                    "status": "unhealthy",
                    "error": "Invalid URL",
                }
        except Exception as e:
            return {
                "server_id": server_id,
                "name": server.name,
                "status": "unhealthy",
                "error": str(e),
            }

    async def check_all_servers_health(self) -> list[dict[str, Any]]:
        """
        Check the health of all registered MCP servers.
        
        Returns:
            List of health status dicts for all servers
        """
        results = []
        for server_id in self.servers:
            health = await self.check_server_health(server_id)
            results.append(health)
        return results

    def get_registry(self, access_groups: list[str] | None = None) -> dict[str, Any]:
        """
        Generate an MCP registry document for discovery.
        
        Args:
            access_groups: Optional list of access groups to filter by
            
        Returns:
            MCP registry document with all servers and capabilities
        """
        servers_list = []
        for server in self.servers.values():
            # Filter by access groups if specified
            if access_groups:
                server_groups = server.metadata.get("access_groups", [])
                if not any(g in server_groups for g in access_groups):
                    continue
            
            servers_list.append({
                "id": server.server_id,
                "name": server.name,
                "url": server.url,
                "transport": server.transport.value,
                "tools": server.tools,
                "resources": server.resources,
                "auth_type": server.auth_type,
            })
        
        return {
            "version": "1.0",
            "servers": servers_list,
            "server_count": len(servers_list),
        }

    def list_access_groups(self) -> list[str]:
        """
        List all unique access groups across all servers.
        
        Returns:
            List of unique access group names
        """
        groups = set()
        for server in self.servers.values():
            server_groups = server.metadata.get("access_groups", [])
            groups.update(server_groups)
        return sorted(groups)


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
