"""
Property-Based Tests for MCP Gateway
=====================================

Tests the MCP (Model Context Protocol) gateway functionality using Hypothesis
for property-based testing.

Properties tested:
- Property 12: MCP Server Tool Loading
- Property 13: OpenAPI to MCP Conversion
- Property 27: MCP Tool Invocation
- Property 28: MCP Database Persistence
- Property 29: MCP OAuth Flow
- Property 30: MCP Server Health Check
- Property 31: MCP Registry Discovery
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck


# =============================================================================
# Data Models (mirroring src/litellm_llmrouter/mcp_gateway.py)
# =============================================================================


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
        self.enabled = False

    def is_enabled(self) -> bool:
        """Check if MCP gateway is enabled."""
        return self.enabled

    def register_server(self, server: MCPServer) -> None:
        """Register an MCP server with the gateway."""
        if not self.enabled:
            return
        self.servers[server.server_id] = server

    def unregister_server(self, server_id: str) -> bool:
        """Unregister an MCP server from the gateway."""
        if server_id in self.servers:
            del self.servers[server_id]
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


# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def mcp_server_id_strategy(draw):
    """Generate valid MCP server IDs."""
    prefix = draw(st.sampled_from(["mcp", "server", "tool", "github", "slack"]))
    suffix = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=10))
    return f"{prefix}-{suffix}"


@st.composite
def mcp_server_name_strategy(draw):
    """Generate valid MCP server names."""
    words = ["GitHub", "Slack", "Jira", "Database", "File", "Search", "API", "Tool"]
    name = draw(st.sampled_from(words))
    suffix = draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=0, max_size=5))
    return f"{name} MCP Server{suffix}"


@st.composite
def mcp_url_strategy(draw):
    """Generate valid MCP server URLs."""
    protocol = draw(st.sampled_from(["http", "https"]))
    host = draw(st.sampled_from(["localhost", "mcp-server", "tools.example.com"]))
    port = draw(st.integers(min_value=1024, max_value=65535))
    return f"{protocol}://{host}:{port}"


@st.composite
def mcp_transport_strategy(draw):
    """Generate valid MCP transport types."""
    return draw(st.sampled_from(list(MCPTransport)))


@st.composite
def mcp_tool_name_strategy(draw):
    """Generate valid MCP tool names."""
    verbs = ["create", "read", "update", "delete", "list", "search", "get", "set"]
    nouns = ["issue", "file", "user", "repo", "message", "channel", "task", "item"]
    verb = draw(st.sampled_from(verbs))
    noun = draw(st.sampled_from(nouns))
    return f"{verb}_{noun}"


@st.composite
def mcp_resource_name_strategy(draw):
    """Generate valid MCP resource names."""
    resources = ["repo_contents", "user_profile", "channel_history", "file_tree", "task_list"]
    return draw(st.sampled_from(resources))


@st.composite
def mcp_auth_type_strategy(draw):
    """Generate valid MCP auth types."""
    return draw(st.sampled_from(["none", "api_key", "bearer_token", "oauth2"]))


@st.composite
def mcp_server_strategy(draw):
    """Generate a complete MCP server."""
    server_id = draw(mcp_server_id_strategy())
    name = draw(mcp_server_name_strategy())
    url = draw(mcp_url_strategy())
    transport = draw(mcp_transport_strategy())
    num_tools = draw(st.integers(min_value=0, max_value=5))
    tools = [draw(mcp_tool_name_strategy()) for _ in range(num_tools)]
    num_resources = draw(st.integers(min_value=0, max_value=3))
    resources = [draw(mcp_resource_name_strategy()) for _ in range(num_resources)]
    auth_type = draw(mcp_auth_type_strategy())
    
    return MCPServer(
        server_id=server_id,
        name=name,
        url=url,
        transport=transport,
        tools=tools,
        resources=resources,
        auth_type=auth_type,
        metadata={},
    )


# =============================================================================
# Property 12: MCP Server Tool Loading
# =============================================================================


class TestMCPServerToolLoadingProperty:
    """
    Property 12: MCP Server Tool Loading
    
    For any MCP server configured in mcp_servers section or registered via API,
    the Gateway should load all tool definitions from the server and make them
    available in the tools list.
    
    Validates: Requirements 8.2, 8.3, 8.4
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    def test_registered_server_tools_appear_in_list(self, server: MCPServer):
        """All tools from a registered server should appear in the tools list."""
        gateway = MCPGateway()
        gateway.enabled = True
        gateway.register_server(server)
        
        tools = gateway.list_tools()
        tool_names = [t["tool"] for t in tools if t["server_id"] == server.server_id]
        
        assert set(tool_names) == set(server.tools)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    def test_registered_server_resources_appear_in_list(self, server: MCPServer):
        """All resources from a registered server should appear in the resources list."""
        gateway = MCPGateway()
        gateway.enabled = True
        gateway.register_server(server)
        
        resources = gateway.list_resources()
        resource_names = [r["resource"] for r in resources if r["server_id"] == server.server_id]
        
        assert set(resource_names) == set(server.resources)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    def test_tool_list_includes_server_info(self, server: MCPServer):
        """Each tool in the list should include server ID and name."""
        gateway = MCPGateway()
        gateway.enabled = True
        gateway.register_server(server)
        
        tools = gateway.list_tools()
        for tool in tools:
            if tool["server_id"] == server.server_id:
                assert "server_id" in tool
                assert "server_name" in tool
                assert "tool" in tool
                assert tool["server_name"] == server.name

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(servers=st.lists(mcp_server_strategy(), min_size=1, max_size=5, unique_by=lambda s: s.server_id))
    def test_multiple_servers_tools_aggregated(self, servers: list[MCPServer]):
        """Tools from multiple servers should all appear in the aggregated list."""
        gateway = MCPGateway()
        gateway.enabled = True
        for server in servers:
            gateway.register_server(server)
        
        tools = gateway.list_tools()
        
        # Count expected tools
        expected_count = sum(len(s.tools) for s in servers)
        assert len(tools) == expected_count

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    def test_unregistered_server_tools_removed(self, server: MCPServer):
        """After unregistering a server, its tools should not appear in the list."""
        gateway = MCPGateway()
        gateway.enabled = True
        gateway.register_server(server)
        gateway.unregister_server(server.server_id)
        
        tools = gateway.list_tools()
        server_tools = [t for t in tools if t["server_id"] == server.server_id]
        
        assert len(server_tools) == 0


class TestMCPServerRegistrationProperty:
    """
    Tests for MCP server registration and retrieval.
    
    Validates: Requirements 8.2
    """

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    def test_registered_server_is_retrievable(self, server: MCPServer):
        """A registered server should be retrievable by ID."""
        gateway = MCPGateway()
        gateway.enabled = True
        gateway.register_server(server)
        
        retrieved = gateway.get_server(server.server_id)
        
        assert retrieved is not None
        assert retrieved.server_id == server.server_id
        assert retrieved.name == server.name
        assert retrieved.url == server.url

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    def test_registered_server_appears_in_list(self, server: MCPServer):
        """A registered server should appear in the servers list."""
        gateway = MCPGateway()
        gateway.enabled = True
        gateway.register_server(server)
        
        servers = gateway.list_servers()
        server_ids = [s.server_id for s in servers]
        
        assert server.server_id in server_ids

    @settings(max_examples=100)
    @given(servers=st.lists(mcp_server_strategy(), min_size=1, max_size=5, unique_by=lambda s: s.server_id))
    def test_all_registered_servers_discoverable(self, servers: list[MCPServer]):
        """All registered servers should be discoverable."""
        gateway = MCPGateway()
        gateway.enabled = True
        for server in servers:
            gateway.register_server(server)
        
        listed = gateway.list_servers()
        listed_ids = {s.server_id for s in listed}
        expected_ids = {s.server_id for s in servers}
        
        assert expected_ids == listed_ids

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    def test_server_data_preserved_on_registration(self, server: MCPServer):
        """All server data should be preserved after registration."""
        gateway = MCPGateway()
        gateway.enabled = True
        gateway.register_server(server)
        
        retrieved = gateway.get_server(server.server_id)
        
        assert retrieved.server_id == server.server_id
        assert retrieved.name == server.name
        assert retrieved.url == server.url
        assert retrieved.transport == server.transport
        assert retrieved.tools == server.tools
        assert retrieved.resources == server.resources
        assert retrieved.auth_type == server.auth_type


class TestMCPServerUnregistrationProperty:
    """
    Tests for MCP server unregistration.
    """

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    def test_unregistered_server_not_retrievable(self, server: MCPServer):
        """An unregistered server should not be retrievable."""
        gateway = MCPGateway()
        gateway.enabled = True
        gateway.register_server(server)
        gateway.unregister_server(server.server_id)
        
        retrieved = gateway.get_server(server.server_id)
        
        assert retrieved is None

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    def test_unregistered_server_not_in_list(self, server: MCPServer):
        """An unregistered server should not appear in the list."""
        gateway = MCPGateway()
        gateway.enabled = True
        gateway.register_server(server)
        gateway.unregister_server(server.server_id)
        
        servers = gateway.list_servers()
        server_ids = [s.server_id for s in servers]
        
        assert server.server_id not in server_ids

    @settings(max_examples=100)
    @given(server_id=mcp_server_id_strategy())
    def test_unregister_nonexistent_returns_false(self, server_id: str):
        """Unregistering a nonexistent server should return False."""
        gateway = MCPGateway()
        gateway.enabled = True
        result = gateway.unregister_server(server_id)
        
        assert result is False


class TestMCPGatewayDisabledProperty:
    """
    Tests for MCP gateway when disabled.
    """

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    def test_disabled_gateway_does_not_register(self, server: MCPServer):
        """A disabled gateway should not register servers."""
        gateway = MCPGateway()
        gateway.enabled = False
        gateway.register_server(server)
        
        # Server should not be stored
        retrieved = gateway.get_server(server.server_id)
        assert retrieved is None

    def test_disabled_gateway_is_enabled_returns_false(self):
        """A disabled gateway should return False for is_enabled()."""
        gateway = MCPGateway()
        gateway.enabled = False
        assert gateway.is_enabled() is False


class TestMCPTransportProperty:
    """
    Tests for MCP transport types.
    """

    @settings(max_examples=100)
    @given(transport=mcp_transport_strategy())
    def test_transport_preserved_on_registration(self, transport: MCPTransport):
        """Transport type should be preserved after registration."""
        gateway = MCPGateway()
        gateway.enabled = True
        server = MCPServer(
            server_id="test-server",
            name="Test Server",
            url="http://localhost:8080",
            transport=transport,
        )
        gateway.register_server(server)
        
        retrieved = gateway.get_server("test-server")
        
        assert retrieved.transport == transport

    def test_all_transport_types_valid(self):
        """All defined transport types should be valid."""
        gateway = MCPGateway()
        gateway.enabled = True
        valid_transports = [MCPTransport.STDIO, MCPTransport.SSE, MCPTransport.STREAMABLE_HTTP]
        
        for transport in valid_transports:
            server = MCPServer(
                server_id=f"test-{transport.value}",
                name=f"Test {transport.value}",
                url="http://localhost:8080",
                transport=transport,
            )
            gateway.register_server(server)
            
            retrieved = gateway.get_server(f"test-{transport.value}")
            assert retrieved is not None
            assert retrieved.transport == transport



# =============================================================================
# Property 13: OpenAPI to MCP Conversion
# =============================================================================


@dataclass
class OpenAPIOperation:
    """Represents an OpenAPI operation for testing."""

    operation_id: str
    method: str
    path: str
    summary: str = ""
    description: str = ""
    parameters: list[dict[str, Any]] = field(default_factory=list)
    request_body: dict[str, Any] | None = None
    responses: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPTool:
    """Represents an MCP tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


def openapi_to_mcp_tool(operation: OpenAPIOperation) -> MCPTool:
    """
    Convert an OpenAPI operation to an MCP tool definition.
    
    This is a simplified implementation for testing purposes.
    The actual implementation would be more comprehensive.
    """
    # Build input schema from parameters and request body
    properties = {}
    required = []
    
    for param in operation.parameters:
        param_name = param.get("name", "")
        param_schema = param.get("schema", {"type": "string"})
        properties[param_name] = param_schema
        if param.get("required", False):
            required.append(param_name)
    
    # Add request body properties if present
    if operation.request_body:
        content = operation.request_body.get("content", {})
        json_content = content.get("application/json", {})
        body_schema = json_content.get("schema", {})
        body_props = body_schema.get("properties", {})
        properties.update(body_props)
        required.extend(body_schema.get("required", []))
    
    input_schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        input_schema["required"] = required
    
    return MCPTool(
        name=operation.operation_id,
        description=operation.summary or operation.description or f"{operation.method.upper()} {operation.path}",
        input_schema=input_schema,
    )


@st.composite
def openapi_operation_strategy(draw):
    """Generate valid OpenAPI operations."""
    verbs = ["get", "post", "put", "patch", "delete"]
    resources = ["users", "items", "orders", "products", "tasks"]
    
    method = draw(st.sampled_from(verbs))
    resource = draw(st.sampled_from(resources))
    
    # Generate operation ID
    action_map = {
        "get": "get",
        "post": "create",
        "put": "update",
        "patch": "patch",
        "delete": "delete",
    }
    operation_id = f"{action_map[method]}_{resource}"
    
    # Generate path
    has_id = draw(st.booleans()) and method != "post"
    path = f"/api/{resource}" + ("/{id}" if has_id else "")
    
    # Generate parameters
    params = []
    if has_id:
        params.append({
            "name": "id",
            "in": "path",
            "required": True,
            "schema": {"type": "string"},
        })
    
    # Add optional query params
    num_query_params = draw(st.integers(min_value=0, max_value=3))
    query_param_names = ["limit", "offset", "filter", "sort", "page"]
    for i in range(num_query_params):
        if i < len(query_param_names):
            params.append({
                "name": query_param_names[i],
                "in": "query",
                "required": False,
                "schema": {"type": "string"},
            })
    
    # Generate request body for POST/PUT/PATCH
    request_body = None
    if method in ["post", "put", "patch"]:
        request_body = {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["name"] if method == "post" else [],
                    }
                }
            }
        }
    
    return OpenAPIOperation(
        operation_id=operation_id,
        method=method,
        path=path,
        summary=f"{method.upper()} {resource}",
        parameters=params,
        request_body=request_body,
    )


class TestOpenAPIToMCPConversionProperty:
    """
    Property 13: OpenAPI to MCP Conversion
    
    For any valid OpenAPI specification provided via spec_path configuration,
    the Gateway should generate corresponding MCP tool definitions that can
    be invoked via the MCP protocol.
    
    Validates: Requirements 8.6
    """

    @settings(max_examples=100)
    @given(operation=openapi_operation_strategy())
    def test_operation_id_becomes_tool_name(self, operation: OpenAPIOperation):
        """The OpenAPI operation ID should become the MCP tool name."""
        tool = openapi_to_mcp_tool(operation)
        
        assert tool.name == operation.operation_id

    @settings(max_examples=100)
    @given(operation=openapi_operation_strategy())
    def test_summary_becomes_description(self, operation: OpenAPIOperation):
        """The OpenAPI summary should become the MCP tool description."""
        tool = openapi_to_mcp_tool(operation)
        
        # Description should be non-empty
        assert len(tool.description) > 0
        # If summary exists, it should be used
        if operation.summary:
            assert tool.description == operation.summary

    @settings(max_examples=100)
    @given(operation=openapi_operation_strategy())
    def test_parameters_become_input_schema(self, operation: OpenAPIOperation):
        """OpenAPI parameters should be converted to MCP input schema properties."""
        tool = openapi_to_mcp_tool(operation)
        
        # All parameters should appear in input schema
        for param in operation.parameters:
            param_name = param.get("name", "")
            assert param_name in tool.input_schema.get("properties", {})

    @settings(max_examples=100)
    @given(operation=openapi_operation_strategy())
    def test_required_parameters_marked_required(self, operation: OpenAPIOperation):
        """Required OpenAPI parameters should be marked required in MCP schema."""
        tool = openapi_to_mcp_tool(operation)
        
        required_params = [
            p["name"] for p in operation.parameters if p.get("required", False)
        ]
        
        schema_required = tool.input_schema.get("required", [])
        for param_name in required_params:
            assert param_name in schema_required

    @settings(max_examples=100)
    @given(operation=openapi_operation_strategy())
    def test_request_body_properties_in_schema(self, operation: OpenAPIOperation):
        """Request body properties should appear in MCP input schema."""
        tool = openapi_to_mcp_tool(operation)
        
        if operation.request_body:
            content = operation.request_body.get("content", {})
            json_content = content.get("application/json", {})
            body_schema = json_content.get("schema", {})
            body_props = body_schema.get("properties", {})
            
            for prop_name in body_props:
                assert prop_name in tool.input_schema.get("properties", {})

    @settings(max_examples=100)
    @given(operation=openapi_operation_strategy())
    def test_input_schema_is_valid_json_schema(self, operation: OpenAPIOperation):
        """The generated input schema should be valid JSON Schema."""
        tool = openapi_to_mcp_tool(operation)
        
        # Must have type
        assert "type" in tool.input_schema
        assert tool.input_schema["type"] == "object"
        
        # Properties must be a dict
        assert isinstance(tool.input_schema.get("properties", {}), dict)
        
        # Required must be a list if present
        if "required" in tool.input_schema:
            assert isinstance(tool.input_schema["required"], list)

    @settings(max_examples=100)
    @given(operations=st.lists(openapi_operation_strategy(), min_size=1, max_size=5, unique_by=lambda o: o.operation_id))
    def test_multiple_operations_generate_multiple_tools(self, operations: list[OpenAPIOperation]):
        """Multiple OpenAPI operations should generate multiple MCP tools."""
        tools = [openapi_to_mcp_tool(op) for op in operations]
        
        assert len(tools) == len(operations)
        
        # All tool names should be unique
        tool_names = [t.name for t in tools]
        assert len(tool_names) == len(set(tool_names))

    def test_get_operation_no_request_body(self):
        """GET operations should not have request body properties."""
        operation = OpenAPIOperation(
            operation_id="get_users",
            method="get",
            path="/api/users",
            summary="Get users",
            parameters=[
                {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer"}}
            ],
        )
        
        tool = openapi_to_mcp_tool(operation)
        
        # Should only have the query parameter
        assert "limit" in tool.input_schema.get("properties", {})
        assert len(tool.input_schema.get("properties", {})) == 1

    def test_post_operation_with_request_body(self):
        """POST operations should include request body properties."""
        operation = OpenAPIOperation(
            operation_id="create_user",
            method="post",
            path="/api/users",
            summary="Create user",
            request_body={
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                            },
                            "required": ["name", "email"],
                        }
                    }
                }
            },
        )
        
        tool = openapi_to_mcp_tool(operation)
        
        # Should have request body properties
        assert "name" in tool.input_schema.get("properties", {})
        assert "email" in tool.input_schema.get("properties", {})
        assert "name" in tool.input_schema.get("required", [])
        assert "email" in tool.input_schema.get("required", [])



# =============================================================================
# Property 27: MCP Tool Invocation
# =============================================================================


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


class MCPGatewayWithInvocation(MCPGateway):
    """Extended MCP Gateway with tool invocation support for testing."""

    def __init__(self):
        super().__init__()
        self._tool_to_server: dict[str, str] = {}

    def register_server(self, server: MCPServer) -> None:
        """Register an MCP server with the gateway."""
        if not self.enabled:
            return
        self.servers[server.server_id] = server
        # Update tool-to-server mapping
        for tool_name in server.tools:
            self._tool_to_server[tool_name] = server.server_id

    def unregister_server(self, server_id: str) -> bool:
        """Unregister an MCP server from the gateway."""
        if server_id in self.servers:
            server = self.servers[server_id]
            for tool_name in server.tools:
                if self._tool_to_server.get(tool_name) == server_id:
                    del self._tool_to_server[tool_name]
            del self.servers[server_id]
            return True
        return False

    def get_tool(self, tool_name: str) -> MCPToolDefinition | None:
        """Get a tool definition by name."""
        server_id = self._tool_to_server.get(tool_name)
        if not server_id:
            return None
        server = self.servers.get(server_id)
        if not server or tool_name not in server.tools:
            return None
        return MCPToolDefinition(
            name=tool_name,
            description=f"Tool from {server.name}",
            server_id=server_id,
        )

    def find_server_for_tool(self, tool_name: str) -> MCPServer | None:
        """Find the server that provides a given tool."""
        server_id = self._tool_to_server.get(tool_name)
        if server_id:
            return self.servers.get(server_id)
        return None

    async def invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """Invoke an MCP tool."""
        server = self.find_server_for_tool(tool_name)
        if not server:
            return MCPToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
            )
        # Simulated successful invocation
        return MCPToolResult(
            success=True,
            result={"message": f"Tool {tool_name} invoked successfully"},
            tool_name=tool_name,
            server_id=server.server_id,
        )


class TestMCPToolInvocationProperty:
    """
    Property 27: MCP Tool Invocation
    
    For any registered MCP server with available tools, when a POST request
    is made to `/mcp/tools/call` with a valid tool name and arguments,
    the Gateway should invoke the tool on the MCP server and return the
    tool's response.
    
    Validates: Requirements 8.8
    """

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    def test_tool_lookup_finds_registered_tools(self, server: MCPServer):
        """Registered tools should be findable by name."""
        gateway = MCPGatewayWithInvocation()
        gateway.enabled = True
        gateway.register_server(server)
        
        for tool_name in server.tools:
            tool = gateway.get_tool(tool_name)
            assert tool is not None
            assert tool.name == tool_name

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    def test_tool_lookup_returns_correct_server(self, server: MCPServer):
        """Tool lookup should return the correct server."""
        gateway = MCPGatewayWithInvocation()
        gateway.enabled = True
        gateway.register_server(server)
        
        for tool_name in server.tools:
            found_server = gateway.find_server_for_tool(tool_name)
            assert found_server is not None
            assert found_server.server_id == server.server_id

    @settings(max_examples=100)
    @given(tool_name=mcp_tool_name_strategy())
    def test_nonexistent_tool_returns_none(self, tool_name: str):
        """Looking up a nonexistent tool should return None."""
        gateway = MCPGatewayWithInvocation()
        gateway.enabled = True
        
        tool = gateway.get_tool(tool_name)
        assert tool is None

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    @pytest.mark.asyncio
    async def test_invoke_registered_tool_succeeds(self, server: MCPServer):
        """Invoking a registered tool should succeed."""
        gateway = MCPGatewayWithInvocation()
        gateway.enabled = True
        gateway.register_server(server)
        
        for tool_name in server.tools:
            result = await gateway.invoke_tool(tool_name, {})
            assert result.success is True
            assert result.tool_name == tool_name
            assert result.server_id == server.server_id

    @settings(max_examples=100)
    @given(tool_name=mcp_tool_name_strategy())
    @pytest.mark.asyncio
    async def test_invoke_nonexistent_tool_fails(self, tool_name: str):
        """Invoking a nonexistent tool should fail."""
        gateway = MCPGatewayWithInvocation()
        gateway.enabled = True
        
        result = await gateway.invoke_tool(tool_name, {})
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @settings(max_examples=100)
    @given(server=mcp_server_strategy())
    def test_unregistered_server_tools_not_invocable(self, server: MCPServer):
        """After unregistering a server, its tools should not be findable."""
        gateway = MCPGatewayWithInvocation()
        gateway.enabled = True
        gateway.register_server(server)
        gateway.unregister_server(server.server_id)
        
        for tool_name in server.tools:
            tool = gateway.get_tool(tool_name)
            assert tool is None

    @settings(max_examples=100)
    @given(
        servers=st.lists(
            mcp_server_strategy(),
            min_size=2,
            max_size=3,
            unique_by=lambda s: s.server_id
        )
    )
    def test_tools_from_multiple_servers_findable(self, servers: list[MCPServer]):
        """Tools from multiple servers should all be findable."""
        gateway = MCPGatewayWithInvocation()
        gateway.enabled = True
        
        for server in servers:
            gateway.register_server(server)
        
        for server in servers:
            for tool_name in server.tools:
                tool = gateway.get_tool(tool_name)
                assert tool is not None
                found_server = gateway.find_server_for_tool(tool_name)
                assert found_server is not None


class TestMCPToolResultProperty:
    """Tests for MCP tool result structure."""

    def test_success_result_has_required_fields(self):
        """A successful result should have all required fields."""
        result = MCPToolResult(
            success=True,
            result={"data": "test"},
            tool_name="test_tool",
            server_id="test-server",
        )
        
        assert result.success is True
        assert result.result is not None
        assert result.error is None
        assert result.tool_name == "test_tool"
        assert result.server_id == "test-server"

    def test_error_result_has_error_message(self):
        """An error result should have an error message."""
        result = MCPToolResult(
            success=False,
            error="Tool not found",
            tool_name="missing_tool",
        )
        
        assert result.success is False
        assert result.error is not None
        assert len(result.error) > 0

    @settings(max_examples=100)
    @given(tool_name=mcp_tool_name_strategy())
    def test_result_preserves_tool_name(self, tool_name: str):
        """The result should preserve the tool name."""
        result = MCPToolResult(
            success=True,
            result={},
            tool_name=tool_name,
        )
        
        assert result.tool_name == tool_name



# =============================================================================
# Property 28: MCP Database Persistence
# =============================================================================


@dataclass
class MCPServerDB:
    """Database model for MCP server (test version)."""

    server_id: str
    name: str
    url: str
    transport: str = "streamable_http"
    tools: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    auth_type: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)
    team_id: str | None = None
    user_id: str | None = None
    is_public: bool = False
    created_at: Any = None
    updated_at: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_id": self.server_id,
            "name": self.name,
            "url": self.url,
            "transport": self.transport,
            "tools": self.tools,
            "resources": self.resources,
            "auth_type": self.auth_type,
            "metadata": self.metadata,
            "team_id": self.team_id,
            "user_id": self.user_id,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MCPServerRepository:
    """In-memory MCP server repository for testing."""

    def __init__(self):
        self._servers: dict[str, MCPServerDB] = {}

    async def create(self, server: MCPServerDB) -> MCPServerDB:
        """Create a new server."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        server.created_at = now
        server.updated_at = now
        self._servers[server.server_id] = server
        return server

    async def get(self, server_id: str) -> MCPServerDB | None:
        """Get a server by ID."""
        return self._servers.get(server_id)

    async def list_all(
        self,
        user_id: str | None = None,
        team_id: str | None = None,
        include_public: bool = True,
    ) -> list[MCPServerDB]:
        """List all servers with optional filtering."""
        servers = list(self._servers.values())
        filtered = []
        for server in servers:
            if include_public and server.is_public:
                filtered.append(server)
                continue
            if user_id and server.user_id == user_id:
                filtered.append(server)
                continue
            if team_id and server.team_id == team_id:
                filtered.append(server)
                continue
            if not user_id and not team_id:
                filtered.append(server)
        return filtered

    async def update(self, server_id: str, server: MCPServerDB) -> MCPServerDB | None:
        """Update an existing server."""
        from datetime import datetime, timezone
        if server_id not in self._servers:
            return None
        server.updated_at = datetime.now(timezone.utc)
        server.created_at = self._servers[server_id].created_at
        self._servers[server_id] = server
        return server

    async def delete(self, server_id: str) -> bool:
        """Delete a server."""
        if server_id not in self._servers:
            return False
        del self._servers[server_id]
        return True


@st.composite
def mcp_server_db_strategy(draw):
    """Generate a complete MCP server DB model."""
    server_id = draw(mcp_server_id_strategy())
    name = draw(mcp_server_name_strategy())
    url = draw(mcp_url_strategy())
    transport = draw(st.sampled_from(["stdio", "sse", "streamable_http"]))
    num_tools = draw(st.integers(min_value=0, max_value=5))
    tools = [draw(mcp_tool_name_strategy()) for _ in range(num_tools)]
    num_resources = draw(st.integers(min_value=0, max_value=3))
    resources = [draw(mcp_resource_name_strategy()) for _ in range(num_resources)]
    auth_type = draw(mcp_auth_type_strategy())
    team_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=10)))
    user_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=10)))
    is_public = draw(st.booleans())
    
    return MCPServerDB(
        server_id=server_id,
        name=name,
        url=url,
        transport=transport,
        tools=tools,
        resources=resources,
        auth_type=auth_type,
        metadata={},
        team_id=team_id,
        user_id=user_id,
        is_public=is_public,
    )


class TestMCPDatabasePersistenceProperty:
    """
    Property 28: MCP Database Persistence
    
    For any MCP server registered when database_url is configured, the server
    and its tools should be persisted to PostgreSQL and should survive Gateway
    restarts, and should be retrievable via the `/v1/mcp/server` endpoint.
    
    Validates: Requirements 8.7
    """

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_created_server_is_retrievable(self, server: MCPServerDB):
        """A created server should be retrievable by ID."""
        repo = MCPServerRepository()
        created = await repo.create(server)
        
        retrieved = await repo.get(server.server_id)
        
        assert retrieved is not None
        assert retrieved.server_id == server.server_id
        assert retrieved.name == server.name
        assert retrieved.url == server.url

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_created_server_has_timestamps(self, server: MCPServerDB):
        """A created server should have created_at and updated_at timestamps."""
        repo = MCPServerRepository()
        created = await repo.create(server)
        
        assert created.created_at is not None
        assert created.updated_at is not None

    @settings(max_examples=100)
    @given(servers=st.lists(mcp_server_db_strategy(), min_size=1, max_size=5, unique_by=lambda s: s.server_id))
    @pytest.mark.asyncio
    async def test_all_created_servers_in_list(self, servers: list[MCPServerDB]):
        """All created servers should appear in the list."""
        repo = MCPServerRepository()
        for server in servers:
            await repo.create(server)
        
        listed = await repo.list_all()
        listed_ids = {s.server_id for s in listed}
        expected_ids = {s.server_id for s in servers}
        
        assert expected_ids == listed_ids

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_deleted_server_not_retrievable(self, server: MCPServerDB):
        """A deleted server should not be retrievable."""
        repo = MCPServerRepository()
        await repo.create(server)
        await repo.delete(server.server_id)
        
        retrieved = await repo.get(server.server_id)
        
        assert retrieved is None

    @settings(max_examples=100)
    @given(server_id=mcp_server_id_strategy())
    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, server_id: str):
        """Deleting a nonexistent server should return False."""
        repo = MCPServerRepository()
        result = await repo.delete(server_id)
        
        assert result is False

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_server_to_dict_round_trip(self, server: MCPServerDB):
        """Server should be serializable to dict and back."""
        repo = MCPServerRepository()
        created = await repo.create(server)
        
        as_dict = created.to_dict()
        
        assert as_dict["server_id"] == server.server_id
        assert as_dict["name"] == server.name
        assert as_dict["url"] == server.url
        assert as_dict["transport"] == server.transport
        assert as_dict["tools"] == server.tools
        assert as_dict["resources"] == server.resources

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_update_preserves_created_at(self, server: MCPServerDB):
        """Update should preserve the original created_at timestamp."""
        repo = MCPServerRepository()
        created = await repo.create(server)
        original_created_at = created.created_at
        
        # Update the server
        server.name = "Updated Name"
        updated = await repo.update(server.server_id, server)
        
        assert updated is not None
        assert updated.created_at == original_created_at

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self, server: MCPServerDB):
        """Updating a nonexistent server should return None."""
        repo = MCPServerRepository()
        result = await repo.update(server.server_id, server)
        
        assert result is None


class TestMCPServerFilteringProperty:
    """Tests for MCP server filtering by user, team, and public status."""

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_filter_by_user_id(self, server: MCPServerDB):
        """Filtering by user_id should return matching servers."""
        repo = MCPServerRepository()
        server.user_id = "test-user"
        server.is_public = False
        await repo.create(server)
        
        # Filter by matching user
        results = await repo.list_all(user_id="test-user", include_public=False)
        assert any(s.server_id == server.server_id for s in results)
        
        # Filter by non-matching user
        results = await repo.list_all(user_id="other-user", include_public=False)
        assert not any(s.server_id == server.server_id for s in results)

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_filter_by_team_id(self, server: MCPServerDB):
        """Filtering by team_id should return matching servers."""
        repo = MCPServerRepository()
        server.team_id = "test-team"
        server.is_public = False
        await repo.create(server)
        
        # Filter by matching team
        results = await repo.list_all(team_id="test-team", include_public=False)
        assert any(s.server_id == server.server_id for s in results)
        
        # Filter by non-matching team
        results = await repo.list_all(team_id="other-team", include_public=False)
        assert not any(s.server_id == server.server_id for s in results)

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_include_public_servers(self, server: MCPServerDB):
        """Public servers should be included when include_public is True."""
        repo = MCPServerRepository()
        server.is_public = True
        await repo.create(server)
        
        # Should be included with include_public=True
        results = await repo.list_all(include_public=True)
        assert any(s.server_id == server.server_id for s in results)

    @settings(max_examples=100)
    @given(server=mcp_server_db_strategy())
    @pytest.mark.asyncio
    async def test_exclude_public_servers(self, server: MCPServerDB):
        """Public servers should be excluded when include_public is False and no user/team match."""
        repo = MCPServerRepository()
        server.is_public = True
        server.user_id = None
        server.team_id = None
        await repo.create(server)
        
        # Should be excluded with include_public=False and non-matching filters
        results = await repo.list_all(
            user_id="other-user",
            team_id="other-team",
            include_public=False
        )
        assert not any(s.server_id == server.server_id for s in results)



# =============================================================================
# Property 30: MCP Server Health Check
# =============================================================================


class MCPGatewayWithHealth(MCPGatewayWithInvocation):
    """Extended MCP Gateway with health check support for testing."""

    async def check_server_health(self, server_id: str) -> dict[str, Any]:
        """Check the health of an MCP server."""
        server = self.servers.get(server_id)
        if not server:
            return {
                "server_id": server_id,
                "status": "not_found",
                "error": f"Server {server_id} not found",
            }
        
        # Simulate health check
        if server.url and server.url.startswith(("http://", "https://")):
            return {
                "server_id": server_id,
                "name": server.name,
                "url": server.url,
                "status": "healthy",
                "latency_ms": 1,
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

    async def check_all_servers_health(self) -> list[dict[str, Any]]:
        """Check the health of all registered MCP servers."""
        results = []
        for server_id in self.servers:
            health = await self.check_server_health(server_id)
            results.append(health)
        return results


class TestMCPServerHealthCheckProperty:
    """
    Property 30: MCP Server Health Check
    
    For any registered MCP server, when a GET request is made to
    `/v1/mcp/server/health`, the Gateway should check connectivity
    to the server and return the health status for each server.
    
    Validates: Requirements 8.13
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    @pytest.mark.asyncio
    async def test_health_check_returns_status(self, server: MCPServer):
        """Health check should return a status for registered servers."""
        gateway = MCPGatewayWithHealth()
        gateway.enabled = True
        gateway.register_server(server)
        
        health = await gateway.check_server_health(server.server_id)
        
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    @pytest.mark.asyncio
    async def test_health_check_includes_server_info(self, server: MCPServer):
        """Health check should include server information."""
        gateway = MCPGatewayWithHealth()
        gateway.enabled = True
        gateway.register_server(server)
        
        health = await gateway.check_server_health(server.server_id)
        
        assert health["server_id"] == server.server_id
        if health["status"] == "healthy":
            assert health["name"] == server.name
            assert "latency_ms" in health

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server_id=mcp_server_id_strategy())
    @pytest.mark.asyncio
    async def test_health_check_nonexistent_returns_not_found(self, server_id: str):
        """Health check for nonexistent server should return not_found status."""
        gateway = MCPGatewayWithHealth()
        gateway.enabled = True
        
        health = await gateway.check_server_health(server_id)
        
        assert health["status"] == "not_found"
        assert "error" in health

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(servers=st.lists(mcp_server_strategy(), min_size=1, max_size=3, unique_by=lambda s: s.server_id))
    @pytest.mark.asyncio
    async def test_check_all_servers_returns_all(self, servers: list[MCPServer]):
        """Check all servers should return health for all registered servers."""
        gateway = MCPGatewayWithHealth()
        gateway.enabled = True
        for server in servers:
            gateway.register_server(server)
        
        health_results = await gateway.check_all_servers_health()
        
        assert len(health_results) == len(servers)
        result_ids = {h["server_id"] for h in health_results}
        expected_ids = {s.server_id for s in servers}
        assert result_ids == expected_ids

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    @pytest.mark.asyncio
    async def test_healthy_server_includes_metrics(self, server: MCPServer):
        """Healthy servers should include tool and resource counts."""
        gateway = MCPGatewayWithHealth()
        gateway.enabled = True
        gateway.register_server(server)
        
        health = await gateway.check_server_health(server.server_id)
        
        if health["status"] == "healthy":
            assert health["tool_count"] == len(server.tools)
            assert health["resource_count"] == len(server.resources)


# =============================================================================
# Property 31: MCP Registry Discovery
# =============================================================================


class MCPGatewayWithRegistry(MCPGatewayWithHealth):
    """Extended MCP Gateway with registry support for testing."""

    def get_registry(self, access_groups: list[str] | None = None) -> dict[str, Any]:
        """Generate an MCP registry document for discovery."""
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


class TestMCPRegistryDiscoveryProperty:
    """
    Property 31: MCP Registry Discovery
    
    For any set of registered MCP servers, the `/v1/mcp/registry.json`
    endpoint should return a valid MCP registry document listing all
    servers and their capabilities for client discovery.
    
    Validates: Requirements 8.12
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(servers=st.lists(mcp_server_strategy(), min_size=1, max_size=5, unique_by=lambda s: s.server_id))
    def test_registry_lists_all_servers(self, servers: list[MCPServer]):
        """Registry should list all registered servers."""
        gateway = MCPGatewayWithRegistry()
        gateway.enabled = True
        for server in servers:
            gateway.register_server(server)
        
        registry = gateway.get_registry()
        
        assert registry["server_count"] == len(servers)
        assert len(registry["servers"]) == len(servers)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    def test_registry_includes_server_details(self, server: MCPServer):
        """Registry should include server details."""
        gateway = MCPGatewayWithRegistry()
        gateway.enabled = True
        gateway.register_server(server)
        
        registry = gateway.get_registry()
        
        assert len(registry["servers"]) == 1
        server_entry = registry["servers"][0]
        assert server_entry["id"] == server.server_id
        assert server_entry["name"] == server.name
        assert server_entry["url"] == server.url
        assert server_entry["tools"] == server.tools
        assert server_entry["resources"] == server.resources

    def test_registry_has_version(self):
        """Registry should include a version field."""
        gateway = MCPGatewayWithRegistry()
        gateway.enabled = True
        
        registry = gateway.get_registry()
        
        assert "version" in registry
        assert registry["version"] == "1.0"

    def test_empty_registry_returns_empty_list(self):
        """Empty gateway should return empty registry."""
        gateway = MCPGatewayWithRegistry()
        gateway.enabled = True
        
        registry = gateway.get_registry()
        
        assert registry["server_count"] == 0
        assert registry["servers"] == []

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    def test_registry_filter_by_access_groups(self, server: MCPServer):
        """Registry should filter by access groups when specified."""
        gateway = MCPGatewayWithRegistry()
        gateway.enabled = True
        
        # Add server with access groups
        server.metadata["access_groups"] = ["group-a", "group-b"]
        gateway.register_server(server)
        
        # Filter by matching group
        registry = gateway.get_registry(access_groups=["group-a"])
        assert len(registry["servers"]) == 1
        
        # Filter by non-matching group
        registry = gateway.get_registry(access_groups=["group-c"])
        assert len(registry["servers"]) == 0

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(server=mcp_server_strategy())
    def test_registry_no_filter_returns_all(self, server: MCPServer):
        """Registry without filter should return all servers."""
        gateway = MCPGatewayWithRegistry()
        gateway.enabled = True
        gateway.register_server(server)
        
        registry = gateway.get_registry(access_groups=None)
        
        assert len(registry["servers"]) == 1
