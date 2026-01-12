"""
A2A Gateway - Agent-to-Agent Protocol Support
==============================================

Provides A2A (Agent-to-Agent) protocol gateway functionality for LiteLLM.
A2A is a protocol for agent-to-agent communication, allowing AI agents
to discover and communicate with each other.

See: https://google.github.io/A2A/
"""

import os
from dataclasses import dataclass, field
from typing import Any

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


# Singleton instance
_a2a_gateway: A2AGateway | None = None


def get_a2a_gateway() -> A2AGateway:
    """Get the global A2A gateway instance."""
    global _a2a_gateway
    if _a2a_gateway is None:
        _a2a_gateway = A2AGateway()
    return _a2a_gateway
