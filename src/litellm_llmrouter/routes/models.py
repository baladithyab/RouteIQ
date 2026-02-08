"""
Pydantic request/response models for route endpoints.
"""

from typing import Any

from pydantic import BaseModel


class AgentRegistration(BaseModel):
    """Request model for A2A agent registration (compatibility layer)."""

    agent_name: str
    description: str = ""
    url: str
    capabilities: list[str] = []
    agent_card_params: dict[str, Any] = {}
    litellm_params: dict[str, Any] = {}


class ServerRegistration(BaseModel):
    """Request model for MCP server registration."""

    server_id: str
    name: str
    url: str
    transport: str = "streamable_http"
    tools: list[str] = []
    resources: list[str] = []
    auth_type: str = "none"
    metadata: dict[str, Any] = {}


class ReloadRequest(BaseModel):
    """Request model for reload operations."""

    strategy: str | None = None
    force_sync: bool = False


class MCPToolCall(BaseModel):
    """Request model for MCP tool invocation."""

    tool_name: str
    arguments: dict[str, Any] = {}


class MCPToolRegister(BaseModel):
    """Request model for registering an MCP tool definition."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = {}
