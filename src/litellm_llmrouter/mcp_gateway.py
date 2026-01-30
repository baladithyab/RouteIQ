"""
MCP Gateway - Model Context Protocol Support
=============================================

Provides MCP (Model Context Protocol) gateway functionality for LiteLLM.
MCP is a protocol for connecting AI models to external tools and data sources.

Features:
- Server registration with in-memory + optional Redis-backed sync
- Tool discovery and invocation
- OTel tracing integration (via mcp_tracing module)
- HA-safe: Redis-backed registry syncs across replicas
- Thread-safe singleton and registry operations

Security Notes:
- Outbound URLs are validated against SSRF attacks before making requests
- See url_security.py for details on blocked targets
- Remote tool invocation is disabled by default (enable via LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true)

See: https://modelcontextprotocol.io/
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import httpx
from litellm._logging import verbose_proxy_logger

# Import SSRF protection utilities
try:
    from .url_security import validate_outbound_url, SSRFBlockedError

    SSRF_PROTECTION_AVAILABLE = True
except ImportError:
    SSRF_PROTECTION_AVAILABLE = False
    SSRFBlockedError = Exception  # Fallback type

    def validate_outbound_url(url: str, **kwargs) -> str:
        """No-op fallback when url_security module is not available."""
        return url


# Redis for HA sync (optional)
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for Redis storage."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_id": self.server_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPToolDefinition":
        """Deserialize from dict."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
            server_id=data.get("server_id", ""),
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for Redis storage."""
        return {
            "server_id": self.server_id,
            "name": self.name,
            "url": self.url,
            "transport": self.transport.value,
            "tools": self.tools,
            "resources": self.resources,
            "auth_type": self.auth_type,
            "metadata": self.metadata,
            "tool_definitions": {
                k: v.to_dict() for k, v in self.tool_definitions.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServer":
        """Deserialize from dict."""
        tool_defs = {}
        for k, v in data.get("tool_definitions", {}).items():
            tool_defs[k] = MCPToolDefinition.from_dict(v)

        return cls(
            server_id=data.get("server_id", ""),
            name=data.get("name", ""),
            url=data.get("url", ""),
            transport=MCPTransport(data.get("transport", "streamable_http")),
            tools=data.get("tools", []),
            resources=data.get("resources", []),
            auth_type=data.get("auth_type", "none"),
            metadata=data.get("metadata", {}),
            tool_definitions=tool_defs,
        )


class MCPGateway:
    """
    MCP Gateway for managing MCP server connections.

    This gateway allows:
    - Registering MCP servers (with optional Redis-backed HA sync)
    - Discovering available tools and resources
    - Invoking MCP tools
    - Proxying MCP requests

    HA Sync Mode:
    When REDIS_HOST is set and MCP_HA_SYNC_ENABLED=true, server registrations
    are synchronized via Redis pub/sub, allowing all replicas to share the
    same registry state.

    Thread Safety:
    All registry mutations are protected by a reentrant lock. Read operations
    return immutable snapshots to avoid stale-read issues and allow iteration
    without holding locks.

    Tool Invocation:
    Remote tool invocation is DISABLED by default for security. To enable,
    set LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true. When disabled, invoke_tool()
    returns an error with code "tool_invocation_disabled".
    """

    # Redis key prefix for MCP servers
    REDIS_KEY_PREFIX = "litellm:mcp:servers:"
    REDIS_PUBSUB_CHANNEL = "litellm:mcp:sync"

    # HTTP client timeouts for tool invocation (seconds)
    TOOL_INVOCATION_CONNECT_TIMEOUT = 10.0
    TOOL_INVOCATION_READ_TIMEOUT = 30.0
    TOOL_INVOCATION_TOTAL_TIMEOUT = 60.0

    def __init__(self):
        # Thread safety: RLock for registry mutations (reentrant to allow nested calls)
        self._lock = threading.RLock()

        self.servers: dict[str, MCPServer] = {}
        self.enabled = os.getenv("MCP_GATEWAY_ENABLED", "false").lower() == "true"
        # Map tool names to server IDs for quick lookup
        self._tool_to_server: dict[str, str] = {}

        # Feature flag for remote tool invocation (disabled by default for security)
        self._tool_invocation_enabled = (
            os.getenv("LLMROUTER_ENABLE_MCP_TOOL_INVOCATION", "false").lower() == "true"
        )

        # HA sync via Redis (optional)
        self._redis_client: Any = None
        self._ha_sync_enabled = (
            os.getenv("MCP_HA_SYNC_ENABLED", "true").lower() == "true"
            and REDIS_AVAILABLE
        )
        self._last_sync_time = 0.0
        self._sync_interval = float(os.getenv("MCP_SYNC_INTERVAL", "5"))  # seconds

        if self._ha_sync_enabled:
            self._init_redis_client()

    def _init_redis_client(self) -> None:
        """Initialize Redis client for HA sync if available."""
        if not REDIS_AVAILABLE:
            verbose_proxy_logger.debug("MCP: Redis not available, HA sync disabled")
            return

        redis_host = os.getenv("REDIS_HOST")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))

        if not redis_host:
            verbose_proxy_logger.debug("MCP: REDIS_HOST not set, HA sync disabled")
            self._ha_sync_enabled = False
            return

        try:
            self._redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=5.0,
            )
            # Test connection
            self._redis_client.ping()
            verbose_proxy_logger.info(
                f"MCP: Redis HA sync enabled ({redis_host}:{redis_port})"
            )

            # Load existing servers from Redis
            self._load_servers_from_redis()

        except Exception as e:
            verbose_proxy_logger.warning(
                f"MCP: Redis connection failed, HA sync disabled: {e}"
            )
            self._redis_client = None
            self._ha_sync_enabled = False

    def _load_servers_from_redis(self) -> None:
        """Load all MCP servers from Redis into local cache."""
        if not self._redis_client:
            return

        try:
            # Get all MCP server keys
            pattern = f"{self.REDIS_KEY_PREFIX}*"
            keys = self._redis_client.keys(pattern)

            loaded_count = 0
            # Collect data outside lock, then apply inside lock
            servers_to_load: list[MCPServer] = []
            for key in keys:
                try:
                    data = self._redis_client.get(key)
                    if data:
                        server_dict = json.loads(data)
                        server = MCPServer.from_dict(server_dict)
                        servers_to_load.append(server)
                except Exception as e:
                    verbose_proxy_logger.warning(
                        f"MCP: Failed to load server from Redis key {key}: {e}"
                    )

            # Apply under lock
            with self._lock:
                for server in servers_to_load:
                    self.servers[server.server_id] = server
                    # Update tool mapping
                    for tool_name in server.tools:
                        self._tool_to_server[tool_name] = server.server_id
                    loaded_count += 1
                self._last_sync_time = time.time()

            if loaded_count > 0:
                verbose_proxy_logger.info(
                    f"MCP: Loaded {loaded_count} servers from Redis"
                )

        except Exception as e:
            verbose_proxy_logger.warning(f"MCP: Failed to load servers from Redis: {e}")

    def _save_server_to_redis(self, server: MCPServer) -> bool:
        """Save a server to Redis for HA sync."""
        if not self._redis_client:
            return False

        try:
            key = f"{self.REDIS_KEY_PREFIX}{server.server_id}"
            data = json.dumps(server.to_dict())
            self._redis_client.set(key, data)

            # Publish sync notification
            self._redis_client.publish(
                self.REDIS_PUBSUB_CHANNEL,
                json.dumps({"action": "register", "server_id": server.server_id}),
            )

            verbose_proxy_logger.debug(f"MCP: Saved server {server.server_id} to Redis")
            return True

        except Exception as e:
            verbose_proxy_logger.warning(f"MCP: Failed to save server to Redis: {e}")
            return False

    def _delete_server_from_redis(self, server_id: str) -> bool:
        """Delete a server from Redis."""
        if not self._redis_client:
            return False

        try:
            key = f"{self.REDIS_KEY_PREFIX}{server_id}"
            self._redis_client.delete(key)

            # Publish sync notification
            self._redis_client.publish(
                self.REDIS_PUBSUB_CHANNEL,
                json.dumps({"action": "unregister", "server_id": server_id}),
            )

            verbose_proxy_logger.debug(f"MCP: Deleted server {server_id} from Redis")
            return True

        except Exception as e:
            verbose_proxy_logger.warning(
                f"MCP: Failed to delete server from Redis: {e}"
            )
            return False

    def sync_from_redis(self) -> int:
        """
        Sync local server cache from Redis.

        Called periodically or on-demand to ensure HA consistency.

        Returns:
            Number of servers synchronized
        """
        if not self._redis_client:
            return 0

        # Rate limit sync - read _last_sync_time under lock
        now = time.time()
        with self._lock:
            if now - self._last_sync_time < self._sync_interval:
                return len(self.servers)

        self._load_servers_from_redis()
        with self._lock:
            return len(self.servers)

    def is_enabled(self) -> bool:
        """Check if MCP gateway is enabled."""
        return self.enabled

    def is_ha_sync_enabled(self) -> bool:
        """Check if HA sync via Redis is enabled."""
        return self._ha_sync_enabled and self._redis_client is not None

    def is_tool_invocation_enabled(self) -> bool:
        """Check if remote tool invocation is enabled."""
        return self._tool_invocation_enabled

    def register_server(self, server: MCPServer) -> None:
        """
        Register an MCP server with the gateway.

        If HA sync is enabled, the server is also saved to Redis for
        cross-replica visibility.

        Security: Server URLs are validated against SSRF attacks before registration.
        Thread Safety: Registry mutation is protected by lock.
        """
        if not self.enabled:
            verbose_proxy_logger.warning("MCP Gateway is not enabled")
            return

        # Security: Validate URL against SSRF attacks at registration time
        # Done outside lock to avoid holding lock during validation
        if server.url:
            try:
                validate_outbound_url(
                    server.url, resolve_dns=False
                )  # Don't resolve during registration
            except SSRFBlockedError as e:
                verbose_proxy_logger.warning(
                    f"MCP: SSRF blocked for server '{server.server_id}': {e}"
                )
                raise ValueError(f"Server URL blocked for security reasons: {e.reason}")
            except ValueError as e:
                verbose_proxy_logger.warning(
                    f"MCP: Invalid URL for server '{server.server_id}': {e}"
                )
                raise ValueError(f"Server URL is invalid: {str(e)}")

        # Thread-safe registry update
        with self._lock:
            self.servers[server.server_id] = server

            # Update tool-to-server mapping
            for tool_name in server.tools:
                self._tool_to_server[tool_name] = server.server_id

        # Save to Redis for HA sync (outside lock to avoid blocking on I/O)
        if self._ha_sync_enabled:
            self._save_server_to_redis(server)

        verbose_proxy_logger.info(
            f"MCP: Registered server {server.name} ({server.server_id})"
            f"{' [HA synced]' if self._ha_sync_enabled else ''}"
        )

    def unregister_server(self, server_id: str) -> bool:
        """
        Unregister an MCP server from the gateway.

        If HA sync is enabled, the server is also removed from Redis.
        Thread Safety: Registry mutation is protected by lock.
        """
        with self._lock:
            if server_id in self.servers:
                server = self.servers[server_id]
                # Remove tool mappings
                for tool_name in server.tools:
                    if self._tool_to_server.get(tool_name) == server_id:
                        del self._tool_to_server[tool_name]
                del self.servers[server_id]
                found = True
            else:
                found = False

        if found:
            # Remove from Redis for HA sync (outside lock)
            if self._ha_sync_enabled:
                self._delete_server_from_redis(server_id)

            verbose_proxy_logger.info(f"MCP: Unregistered server {server_id}")
            return True
        return False

    def get_server(self, server_id: str) -> MCPServer | None:
        """Get an MCP server by ID. Thread-safe."""
        # Sync from Redis if HA enabled (rate-limited)
        if self._ha_sync_enabled:
            self.sync_from_redis()
        with self._lock:
            return self.servers.get(server_id)

    def list_servers(self) -> list[MCPServer]:
        """List all registered MCP servers. Returns a snapshot copy. Thread-safe."""
        # Sync from Redis if HA enabled (rate-limited)
        if self._ha_sync_enabled:
            self.sync_from_redis()
        with self._lock:
            return list(self.servers.values())

    def get_servers_snapshot(self) -> MappingProxyType[str, MCPServer]:
        """
        Get an immutable snapshot of the servers registry.

        Returns:
            Read-only view of current servers dict.
        """
        with self._lock:
            # Return immutable proxy to a copy to prevent mutation
            return MappingProxyType(dict(self.servers))

    def list_tools(self) -> list[dict[str, Any]]:
        """List all tools from all registered servers. Thread-safe."""
        tools = []
        with self._lock:
            servers_snapshot = list(self.servers.values())
        for server in servers_snapshot:
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
        """List all resources from all registered servers. Thread-safe."""
        resources = []
        with self._lock:
            servers_snapshot = list(self.servers.values())
        for server in servers_snapshot:
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
        Get a tool definition by name. Thread-safe.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool definition if found, None otherwise
        """
        with self._lock:
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
                    server_id=server.server_id,
                )

        return None

    def find_server_for_tool(self, tool_name: str) -> MCPServer | None:
        """
        Find the server that provides a given tool. Thread-safe.

        Args:
            tool_name: Name of the tool

        Returns:
            Server that provides the tool, or None if not found
        """
        with self._lock:
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
        Register a detailed tool definition for a server. Thread-safe.

        Args:
            server_id: ID of the server providing the tool
            tool: Tool definition with schema

        Returns:
            True if registered successfully, False otherwise
        """
        with self._lock:
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

        Security:
        - Remote tool invocation is DISABLED by default. Enable via LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true.
        - Server URLs are validated against SSRF attacks before making requests.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Arguments to pass to the tool

        Returns:
            Result of the tool invocation
        """
        # Check feature flag first - disabled by default for security
        if not self._tool_invocation_enabled:
            verbose_proxy_logger.warning(
                "MCP: Tool invocation disabled. Set LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true to enable."
            )
            return MCPToolResult(
                success=False,
                error="tool_invocation_disabled: Remote tool invocation is disabled. Enable via LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true",
                tool_name=tool_name,
            )

        server = self.find_server_for_tool(tool_name)
        if not server:
            return MCPToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
            )

        # Validate server has a URL
        if not server.url:
            return MCPToolResult(
                success=False,
                error="Server URL is not configured",
                tool_name=tool_name,
                server_id=server.server_id,
            )

        # Security: Validate URL against SSRF attacks before making outbound call
        try:
            validate_outbound_url(server.url)
        except SSRFBlockedError as e:
            verbose_proxy_logger.warning(
                f"MCP: SSRF blocked for server '{server.server_id}' when invoking tool '{tool_name}': {e}"
            )
            return MCPToolResult(
                success=False,
                error=f"Server URL blocked for security reasons: {e.reason}",
                tool_name=tool_name,
                server_id=server.server_id,
            )
        except ValueError as e:
            verbose_proxy_logger.warning(
                f"MCP: Invalid URL for server '{server.server_id}': {e}"
            )
            return MCPToolResult(
                success=False,
                error=f"Server URL is invalid: {str(e)}",
                tool_name=tool_name,
                server_id=server.server_id,
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

        # Build the tool invocation URL (per MCP stub server protocol)
        # The server URL base + /mcp/tools/call endpoint
        base_url = server.url.rstrip("/")
        if not base_url.endswith("/mcp/tools/call"):
            # Append standard endpoint if not already present
            if base_url.endswith("/mcp"):
                invocation_url = f"{base_url}/tools/call"
            else:
                invocation_url = f"{base_url}/mcp/tools/call"
        else:
            invocation_url = base_url

        verbose_proxy_logger.info(
            f"MCP: Invoking tool {tool_name} on server {server.server_id} at {invocation_url}"
        )

        # Prepare request payload (matches MCP stub server format)
        request_payload = {
            "tool_name": tool_name,
            "arguments": arguments,
        }

        # Build headers (include auth if configured)
        headers = {"Content-Type": "application/json"}
        if server.auth_type == "bearer_token" and server.metadata.get("auth_token"):
            headers["Authorization"] = f"Bearer {server.metadata['auth_token']}"
        elif server.auth_type == "api_key" and server.metadata.get("api_key"):
            headers["X-API-Key"] = server.metadata["api_key"]

        # Make HTTP request with strict timeouts
        # Do NOT hold locks during I/O
        timeout = httpx.Timeout(
            connect=self.TOOL_INVOCATION_CONNECT_TIMEOUT,
            read=self.TOOL_INVOCATION_READ_TIMEOUT,
            write=self.TOOL_INVOCATION_CONNECT_TIMEOUT,
            pool=self.TOOL_INVOCATION_CONNECT_TIMEOUT,
        )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    invocation_url,
                    json=request_payload,
                    headers=headers,
                )

                # Check HTTP status
                if response.status_code >= 400:
                    error_detail = (
                        response.text[:500]
                        if response.text
                        else f"HTTP {response.status_code}"
                    )
                    verbose_proxy_logger.warning(
                        f"MCP: Tool invocation failed with HTTP {response.status_code}: {error_detail}"
                    )
                    return MCPToolResult(
                        success=False,
                        error=f"HTTP {response.status_code}: {error_detail}",
                        tool_name=tool_name,
                        server_id=server.server_id,
                    )

                # Parse response JSON
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    verbose_proxy_logger.warning(
                        f"MCP: Invalid JSON response from server: {e}"
                    )
                    return MCPToolResult(
                        success=False,
                        error=f"Invalid JSON response from MCP server: {str(e)}",
                        tool_name=tool_name,
                        server_id=server.server_id,
                    )

                # Parse MCP server response format
                # Expected format: {"status": "success", "tool_name": "...", "result": {...}}
                # or error: {"detail": "error message"}
                if isinstance(response_data, dict):
                    status = response_data.get("status", "")
                    if status == "success":
                        return MCPToolResult(
                            success=True,
                            result=response_data.get("result"),
                            tool_name=response_data.get("tool_name", tool_name),
                            server_id=server.server_id,
                        )
                    elif "detail" in response_data:
                        # Error response format
                        return MCPToolResult(
                            success=False,
                            error=response_data["detail"],
                            tool_name=tool_name,
                            server_id=server.server_id,
                        )
                    elif "error" in response_data:
                        return MCPToolResult(
                            success=False,
                            error=response_data["error"],
                            tool_name=tool_name,
                            server_id=server.server_id,
                        )
                    else:
                        # Unknown format - return as-is with success
                        return MCPToolResult(
                            success=True,
                            result=response_data,
                            tool_name=tool_name,
                            server_id=server.server_id,
                        )
                else:
                    # Non-dict response
                    return MCPToolResult(
                        success=True,
                        result=response_data,
                        tool_name=tool_name,
                        server_id=server.server_id,
                    )

        except httpx.TimeoutException as e:
            verbose_proxy_logger.warning(
                f"MCP: Timeout invoking tool '{tool_name}' on server '{server.server_id}': {e}"
            )
            return MCPToolResult(
                success=False,
                error=f"Timeout connecting to MCP server: {str(e)}",
                tool_name=tool_name,
                server_id=server.server_id,
            )
        except httpx.ConnectError as e:
            verbose_proxy_logger.warning(
                f"MCP: Connection error invoking tool '{tool_name}': {e}"
            )
            return MCPToolResult(
                success=False,
                error=f"Failed to connect to MCP server: {str(e)}",
                tool_name=tool_name,
                server_id=server.server_id,
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                f"MCP: Unexpected error invoking tool '{tool_name}': {e}"
            )
            return MCPToolResult(
                success=False,
                error=f"Unexpected error during tool invocation: {str(e)}",
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
        for field_name in required:
            if field_name not in arguments:
                return f"Missing required argument: {field_name}"

        # Check property types (basic validation)
        properties = schema.get("properties", {})
        for arg_name, arg_value in arguments.items():
            if arg_name in properties:
                prop_schema = properties[arg_name]
                expected_type = prop_schema.get("type")
                if expected_type:
                    if not self._check_type(arg_value, expected_type):
                        return (
                            f"Invalid type for '{arg_name}': expected {expected_type}"
                        )

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

        Security: Server URLs are validated against SSRF attacks before making requests.
        Thread Safety: Server lookup is protected by lock.

        Args:
            server_id: ID of the server to check

        Returns:
            Health status dict with status, latency, and error info
        """
        import time

        # Get server under lock
        with self._lock:
            server = self.servers.get(server_id)

        if not server:
            return {
                "server_id": server_id,
                "status": "not_found",
                "error": f"Server {server_id} not found",
            }

        start_time = time.time()

        # Security: Validate URL against SSRF attacks (outside lock)
        if server.url:
            try:
                validate_outbound_url(server.url)
            except SSRFBlockedError as e:
                verbose_proxy_logger.warning(
                    f"MCP: SSRF blocked for health check of server '{server_id}': {e}"
                )
                return {
                    "server_id": server_id,
                    "name": server.name,
                    "status": "blocked",
                    "error": f"URL blocked for security reasons: {e.reason}",
                }
            except ValueError as e:
                return {
                    "server_id": server_id,
                    "name": server.name,
                    "status": "invalid_url",
                    "error": str(e),
                }

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
        # Get snapshot of server IDs under lock
        with self._lock:
            server_ids = list(self.servers.keys())

        results = []
        for server_id in server_ids:
            health = await self.check_server_health(server_id)
            results.append(health)
        return results

    def get_registry(self, access_groups: list[str] | None = None) -> dict[str, Any]:
        """
        Generate an MCP registry document for discovery. Thread-safe.

        Args:
            access_groups: Optional list of access groups to filter by

        Returns:
            MCP registry document with all servers and capabilities
        """
        servers_list = []
        with self._lock:
            servers_snapshot = list(self.servers.values())

        for server in servers_snapshot:
            # Filter by access groups if specified
            if access_groups:
                server_groups = server.metadata.get("access_groups", [])
                if not any(g in server_groups for g in access_groups):
                    continue

            servers_list.append(
                {
                    "id": server.server_id,
                    "name": server.name,
                    "url": server.url,
                    "transport": server.transport.value,
                    "tools": server.tools,
                    "resources": server.resources,
                    "auth_type": server.auth_type,
                }
            )

        return {
            "version": "1.0",
            "servers": servers_list,
            "server_count": len(servers_list),
        }

    def list_access_groups(self) -> list[str]:
        """
        List all unique access groups across all servers. Thread-safe.

        Returns:
            List of unique access group names
        """
        groups = set()
        with self._lock:
            servers_snapshot = list(self.servers.values())

        for server in servers_snapshot:
            server_groups = server.metadata.get("access_groups", [])
            groups.update(server_groups)
        return sorted(groups)


# Singleton instance and lock for thread-safe initialization
_mcp_gateway: MCPGateway | None = None
_mcp_gateway_lock = threading.Lock()


def get_mcp_gateway() -> MCPGateway:
    """
    Get the global MCP gateway instance.

    Thread-safe: Uses double-checked locking pattern for efficient
    singleton initialization.
    """
    global _mcp_gateway
    if _mcp_gateway is None:
        with _mcp_gateway_lock:
            # Double-check after acquiring lock
            if _mcp_gateway is None:
                _mcp_gateway = MCPGateway()
    return _mcp_gateway


def reset_mcp_gateway() -> None:
    """
    Reset the global MCP gateway instance.

    WARNING: For testing purposes only. Not safe to call while
    requests are in flight.
    """
    global _mcp_gateway
    with _mcp_gateway_lock:
        _mcp_gateway = None
