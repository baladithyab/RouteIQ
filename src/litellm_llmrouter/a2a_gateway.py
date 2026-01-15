"""
A2A Gateway - Agent-to-Agent Protocol Support
==============================================

Provides A2A (Agent-to-Agent) protocol gateway functionality for LiteLLM.
A2A is a protocol for agent-to-agent communication, allowing AI agents
to discover and communicate with each other.

See: https://google.github.io/A2A/
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from litellm._logging import verbose_proxy_logger


@dataclass
class A2AAgent:
    """Represents an A2A agent registration."""

    agent_id: str
    name: str
    description: str
    url: str
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request."""

    method: str
    params: dict[str, Any]
    id: str | int | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response."""

    id: str | int | None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        response = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            response["error"] = self.error
        else:
            response["result"] = self.result
        return response

    @classmethod
    def error_response(
        cls, request_id: str | int | None, code: int, message: str
    ) -> "JSONRPCResponse":
        """Create an error response."""
        return cls(id=request_id, error={"code": code, "message": message})

    @classmethod
    def success_response(
        cls, request_id: str | int | None, result: dict[str, Any]
    ) -> "JSONRPCResponse":
        """Create a success response."""
        return cls(id=request_id, result=result)


class A2AGateway:
    """
    A2A Gateway for managing agent registrations and discovery.

    This gateway allows:
    - Registering AI agents with their capabilities
    - Discovering available agents
    - Routing requests to appropriate agents
    """

    def __init__(self):
        self.agents: dict[str, A2AAgent] = {}
        self.enabled = os.getenv("A2A_GATEWAY_ENABLED", "false").lower() == "true"

    def is_enabled(self) -> bool:
        """Check if A2A gateway is enabled."""
        return self.enabled

    def register_agent(self, agent: A2AAgent) -> None:
        """Register an agent with the gateway."""
        if not self.enabled:
            verbose_proxy_logger.warning("A2A Gateway is not enabled")
            return

        self.agents[agent.agent_id] = agent
        verbose_proxy_logger.info(
            f"A2A: Registered agent {agent.name} ({agent.agent_id})"
        )

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the gateway."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            verbose_proxy_logger.info(f"A2A: Unregistered agent {agent_id}")
            return True
        return False

    def get_agent(self, agent_id: str) -> A2AAgent | None:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def list_agents(self) -> list[A2AAgent]:
        """List all registered agents."""
        return list(self.agents.values())

    def discover_agents(self, capability: str | None = None) -> list[A2AAgent]:
        """Discover agents, optionally filtered by capability."""
        if capability is None:
            return self.list_agents()
        return [a for a in self.agents.values() if capability in a.capabilities]

    def get_agent_card(self, agent_id: str) -> dict[str, Any] | None:
        """Get the A2A agent card for an agent."""
        agent = self.get_agent(agent_id)
        if not agent:
            return None

        return {
            "name": agent.name,
            "description": agent.description,
            "url": agent.url,
            "capabilities": {
                "streaming": "streaming" in agent.capabilities,
                "pushNotifications": "push_notifications" in agent.capabilities,
                "stateTransitionHistory": "state_history" in agent.capabilities,
            },
            "skills": [
                {"id": cap, "name": cap.replace("_", " ").title()}
                for cap in agent.capabilities
            ],
        }

    async def invoke_agent(
        self, agent_id: str, request: JSONRPCRequest
    ) -> JSONRPCResponse:
        """
        Invoke an agent using JSON-RPC 2.0 protocol.

        Supports methods:
        - message/send: Send a message and get a response
        - message/stream: Send a message and stream the response (returns first chunk)

        Args:
            agent_id: The ID of the agent to invoke
            request: The JSON-RPC 2.0 request

        Returns:
            JSONRPCResponse with the result or error
        """
        if not self.enabled:
            return JSONRPCResponse.error_response(
                request.id, -32000, "A2A Gateway is not enabled"
            )

        agent = self.get_agent(agent_id)
        if not agent:
            return JSONRPCResponse.error_response(
                request.id, -32000, f"Agent '{agent_id}' not found"
            )

        if not agent.url:
            return JSONRPCResponse.error_response(
                request.id, -32000, f"Agent '{agent_id}' has no URL configured"
            )

        # Validate JSON-RPC format
        if request.jsonrpc != "2.0":
            return JSONRPCResponse.error_response(
                request.id, -32600, "Invalid Request: jsonrpc must be '2.0'"
            )

        method = request.method
        if method not in ("message/send", "message/stream"):
            return JSONRPCResponse.error_response(
                request.id, -32601, f"Method '{method}' not found"
            )

        verbose_proxy_logger.info(
            f"A2A: Invoking agent '{agent_id}' with method '{method}'"
        )

        try:
            # Forward the request to the agent backend
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    agent.url,
                    json=request.to_dict(),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                result = response.json()

                # Return the response from the agent
                if "error" in result:
                    return JSONRPCResponse(
                        id=request.id,
                        error=result["error"],
                    )
                return JSONRPCResponse(
                    id=request.id,
                    result=result.get("result", result),
                )

        except httpx.TimeoutException:
            verbose_proxy_logger.error(f"A2A: Timeout invoking agent '{agent_id}'")
            return JSONRPCResponse.error_response(
                request.id, -32000, f"Timeout invoking agent '{agent_id}'"
            )
        except httpx.HTTPStatusError as e:
            verbose_proxy_logger.error(
                f"A2A: HTTP error invoking agent '{agent_id}': {e}"
            )
            return JSONRPCResponse.error_response(
                request.id, -32000, f"HTTP error: {e.response.status_code}"
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"A2A: Error invoking agent '{agent_id}': {e}")
            return JSONRPCResponse.error_response(
                request.id, -32603, f"Internal error: {str(e)}"
            )

    async def stream_agent_response(
        self, agent_id: str, request: JSONRPCRequest
    ) -> AsyncIterator[str]:
        """
        Stream response from an agent using Server-Sent Events.

        Args:
            agent_id: The ID of the agent to invoke
            request: The JSON-RPC 2.0 request with method 'message/stream'

        Yields:
            JSON-encoded response chunks as newline-delimited JSON
        """
        if not self.enabled:
            yield json.dumps(
                JSONRPCResponse.error_response(
                    request.id, -32000, "A2A Gateway is not enabled"
                ).to_dict()
            ) + "\n"
            return

        agent = self.get_agent(agent_id)
        if not agent:
            yield json.dumps(
                JSONRPCResponse.error_response(
                    request.id, -32000, f"Agent '{agent_id}' not found"
                ).to_dict()
            ) + "\n"
            return

        if not agent.url:
            yield json.dumps(
                JSONRPCResponse.error_response(
                    request.id, -32000, f"Agent '{agent_id}' has no URL configured"
                ).to_dict()
            ) + "\n"
            return

        verbose_proxy_logger.info(
            f"A2A: Streaming from agent '{agent_id}'"
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    agent.url,
                    json=request.to_dict(),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            yield line + "\n"

        except httpx.TimeoutException:
            verbose_proxy_logger.error(f"A2A: Timeout streaming from agent '{agent_id}'")
            yield json.dumps(
                JSONRPCResponse.error_response(
                    request.id, -32000, f"Timeout streaming from agent '{agent_id}'"
                ).to_dict()
            ) + "\n"
        except httpx.HTTPStatusError as e:
            verbose_proxy_logger.error(
                f"A2A: HTTP error streaming from agent '{agent_id}': {e}"
            )
            yield json.dumps(
                JSONRPCResponse.error_response(
                    request.id, -32000, f"HTTP error: {e.response.status_code}"
                ).to_dict()
            ) + "\n"
        except Exception as e:
            verbose_proxy_logger.exception(
                f"A2A: Error streaming from agent '{agent_id}': {e}"
            )
            yield json.dumps(
                JSONRPCResponse.error_response(
                    request.id, -32603, f"Streaming error: {str(e)}"
                ).to_dict()
            ) + "\n"


# Singleton instance
_a2a_gateway: A2AGateway | None = None


def get_a2a_gateway() -> A2AGateway:
    """Get the global A2A gateway instance."""
    global _a2a_gateway
    if _a2a_gateway is None:
        _a2a_gateway = A2AGateway()
    return _a2a_gateway
