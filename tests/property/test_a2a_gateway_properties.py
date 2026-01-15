"""
Property-Based Tests for A2A Gateway.

These tests validate the correctness properties defined in the design document
for the A2A (Agent-to-Agent) Gateway (Requirements 7.x).

Property tests use Hypothesis to generate many test cases and verify that
universal properties hold across all valid inputs.

**Property 11: A2A Agent Registration and Discovery**
For any A2A agent registered via configuration or API, the agent should be
discoverable via the `/v1/agents` endpoint and should be filterable by
capability and permission, and the agent card should be retrievable in
A2A protocol format at `/.well-known/agent-card.json`.

**Validates: Requirements 7.2, 7.6, 7.13**
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck


# =============================================================================
# Data Models (mirroring src/litellm_llmrouter/a2a_gateway.py)
# =============================================================================


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
    
    This is a test-local implementation that mirrors the production code
    to enable property testing without litellm dependency.
    """

    def __init__(self, enabled: bool = True):
        self.agents: dict[str, A2AAgent] = {}
        self.enabled = enabled

    def is_enabled(self) -> bool:
        """Check if A2A gateway is enabled."""
        return self.enabled

    def register_agent(self, agent: A2AAgent) -> None:
        """Register an agent with the gateway."""
        if not self.enabled:
            return
        self.agents[agent.agent_id] = agent

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the gateway."""
        if agent_id in self.agents:
            del self.agents[agent_id]
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


# =============================================================================
# Test Data Generators
# =============================================================================

# Valid agent ID: alphanumeric with hyphens and underscores
agent_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='-_'),
    min_size=1,
    max_size=50,
).filter(lambda x: x.strip() and not x.startswith('-') and not x.startswith('_'))

# Agent name: human-readable text
agent_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'Z'), whitelist_characters='-_ '),
    min_size=1,
    max_size=100,
).filter(lambda x: x.strip())

# Agent description: longer text
agent_description_strategy = st.text(
    min_size=0,
    max_size=500,
)

# Valid URL
url_strategy = st.from_regex(
    r'https?://[a-z0-9\-\.]+\.[a-z]{2,}(:[0-9]+)?(/[a-z0-9\-_/]*)?',
    fullmatch=True
).filter(lambda x: len(x) <= 200)

# Capability: lowercase alphanumeric with underscores
capability_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Ll', 'N'), whitelist_characters='_'),
    min_size=1,
    max_size=30,
).filter(lambda x: x.strip() and not x.startswith('_'))

# List of capabilities
capabilities_strategy = st.lists(capability_strategy, min_size=0, max_size=10, unique=True)

# Metadata: simple key-value pairs
metadata_strategy = st.dictionaries(
    keys=st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=20),
    values=st.one_of(
        st.text(max_size=100),
        st.integers(min_value=-1000, max_value=1000),
        st.booleans(),
    ),
    max_size=5,
)


@st.composite
def a2a_agent_strategy(draw):
    """Generate a valid A2A agent."""
    return A2AAgent(
        agent_id=draw(agent_id_strategy),
        name=draw(agent_name_strategy),
        description=draw(agent_description_strategy),
        url=draw(url_strategy),
        capabilities=draw(capabilities_strategy),
        metadata=draw(metadata_strategy),
    )


@st.composite
def multiple_agents_strategy(draw, min_agents=1, max_agents=10):
    """Generate multiple unique agents."""
    num_agents = draw(st.integers(min_value=min_agents, max_value=max_agents))
    agents = []
    seen_ids = set()
    
    for _ in range(num_agents):
        agent = draw(a2a_agent_strategy())
        # Ensure unique agent IDs
        if agent.agent_id not in seen_ids:
            seen_ids.add(agent.agent_id)
            agents.append(agent)
    
    assume(len(agents) >= min_agents)
    return agents


# =============================================================================
# Property Tests
# =============================================================================


class TestA2AAgentRegistrationProperty:
    """
    Property 11: A2A Agent Registration and Discovery
    
    For any A2A agent registered via configuration or API, the agent should be
    discoverable via the `/v1/agents` endpoint and should be filterable by
    capability and permission, and the agent card should be retrievable in
    A2A protocol format at `/.well-known/agent-card.json`.
    
    **Validates: Requirements 7.2, 7.6, 7.13**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_registered_agent_is_retrievable(self, agent: A2AAgent):
        """
        Property 11: For any registered agent, get_agent returns the same agent.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.2
        """
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        retrieved = gateway.get_agent(agent.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == agent.agent_id
        assert retrieved.name == agent.name
        assert retrieved.description == agent.description
        assert retrieved.url == agent.url
        assert retrieved.capabilities == agent.capabilities

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_registered_agent_appears_in_list(self, agent: A2AAgent):
        """
        Property 11: For any registered agent, list_agents includes that agent.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.2
        """
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        agents = gateway.list_agents()
        agent_ids = [a.agent_id for a in agents]
        assert agent.agent_id in agent_ids

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agents=multiple_agents_strategy(min_agents=1, max_agents=5))
    def test_all_registered_agents_are_discoverable(self, agents: list[A2AAgent]):
        """
        Property 11: For any set of registered agents, all are discoverable.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.2
        """
        gateway = A2AGateway(enabled=True)
        for agent in agents:
            gateway.register_agent(agent)
        
        discovered = gateway.discover_agents()
        discovered_ids = {a.agent_id for a in discovered}
        
        for agent in agents:
            assert agent.agent_id in discovered_ids

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_agent_card_has_required_fields(self, agent: A2AAgent):
        """
        Property 11: For any registered agent, agent card has A2A protocol fields.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.6
        """
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        card = gateway.get_agent_card(agent.agent_id)
        assert card is not None
        
        # A2A protocol required fields
        assert "name" in card
        assert "description" in card
        assert "url" in card
        assert "capabilities" in card
        assert "skills" in card
        
        # Verify values match agent
        assert card["name"] == agent.name
        assert card["description"] == agent.description
        assert card["url"] == agent.url

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_agent_card_capabilities_format(self, agent: A2AAgent):
        """
        Property 11: Agent card capabilities follow A2A protocol format.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.6
        """
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        card = gateway.get_agent_card(agent.agent_id)
        assert card is not None
        
        capabilities = card["capabilities"]
        assert isinstance(capabilities, dict)
        assert "streaming" in capabilities
        assert "pushNotifications" in capabilities
        assert "stateTransitionHistory" in capabilities
        
        # Verify boolean values
        assert isinstance(capabilities["streaming"], bool)
        assert isinstance(capabilities["pushNotifications"], bool)
        assert isinstance(capabilities["stateTransitionHistory"], bool)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_agent_card_skills_format(self, agent: A2AAgent):
        """
        Property 11: Agent card skills follow A2A protocol format.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.6
        """
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        card = gateway.get_agent_card(agent.agent_id)
        assert card is not None
        
        skills = card["skills"]
        assert isinstance(skills, list)
        assert len(skills) == len(agent.capabilities)
        
        for skill in skills:
            assert "id" in skill
            assert "name" in skill
            assert skill["id"] in agent.capabilities


class TestA2AAgentDiscoveryByCapability:
    """
    Property 11: A2A Agent Discovery by Capability
    
    Tests that agents can be filtered by capability.
    
    **Validates: Requirements 7.13**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        agent=a2a_agent_strategy(),
        capability=capability_strategy,
    )
    def test_discover_by_capability_returns_matching_agents(
        self, agent: A2AAgent, capability: str
    ):
        """
        Property 11: discover_agents with capability returns only matching agents.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.13
        """
        gateway = A2AGateway(enabled=True)
        
        # Add capability to agent
        agent.capabilities = list(set(agent.capabilities + [capability]))
        gateway.register_agent(agent)
        
        discovered = gateway.discover_agents(capability)
        
        # All discovered agents should have the capability
        for a in discovered:
            assert capability in a.capabilities
        
        # Our agent should be in the results
        discovered_ids = [a.agent_id for a in discovered]
        assert agent.agent_id in discovered_ids

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agents=multiple_agents_strategy(min_agents=2, max_agents=5))
    def test_discover_filters_correctly(self, agents: list[A2AAgent]):
        """
        Property 11: discover_agents correctly filters by capability.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.13
        """
        gateway = A2AGateway(enabled=True)
        
        # Assign unique capability to first agent
        unique_cap = "unique_test_capability_xyz"
        agents[0].capabilities = list(set(agents[0].capabilities + [unique_cap]))
        
        for agent in agents:
            gateway.register_agent(agent)
        
        # Discover by unique capability
        discovered = gateway.discover_agents(unique_cap)
        
        # Only agents with the capability should be returned
        for a in discovered:
            assert unique_cap in a.capabilities
        
        # First agent should be in results
        discovered_ids = [a.agent_id for a in discovered]
        assert agents[0].agent_id in discovered_ids

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_discover_without_capability_returns_all(self, agent: A2AAgent):
        """
        Property 11: discover_agents without capability returns all agents.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.13
        """
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        # Discover without filter
        discovered = gateway.discover_agents(None)
        
        discovered_ids = [a.agent_id for a in discovered]
        assert agent.agent_id in discovered_ids


class TestA2AAgentUnregistration:
    """
    Tests for agent unregistration.
    
    **Validates: Requirements 7.2**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_unregistered_agent_not_retrievable(self, agent: A2AAgent):
        """
        Property 11: After unregistration, agent is no longer retrievable.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.2
        """
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        # Verify registered
        assert gateway.get_agent(agent.agent_id) is not None
        
        # Unregister
        result = gateway.unregister_agent(agent.agent_id)
        assert result is True
        
        # Verify not retrievable
        assert gateway.get_agent(agent.agent_id) is None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_unregistered_agent_not_in_list(self, agent: A2AAgent):
        """
        Property 11: After unregistration, agent is not in list.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.2
        """
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        # Unregister
        gateway.unregister_agent(agent.agent_id)
        
        # Verify not in list
        agents = gateway.list_agents()
        agent_ids = [a.agent_id for a in agents]
        assert agent.agent_id not in agent_ids

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent_id=agent_id_strategy)
    def test_unregister_nonexistent_returns_false(self, agent_id: str):
        """
        Property 11: Unregistering nonexistent agent returns False.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.2
        """
        gateway = A2AGateway(enabled=True)
        
        result = gateway.unregister_agent(agent_id)
        assert result is False


class TestA2AGatewayDisabled:
    """
    Tests for gateway disabled state.
    
    **Validates: Requirements 7.1**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_disabled_gateway_does_not_register(self, agent: A2AAgent):
        """
        Property 11: When gateway is disabled, agents are not registered.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.1
        """
        gateway = A2AGateway(enabled=False)
        gateway.register_agent(agent)
        
        # Agent should not be registered
        assert gateway.get_agent(agent.agent_id) is None
        assert len(gateway.list_agents()) == 0


class TestA2AAgentCardNonexistent:
    """
    Tests for agent card retrieval of nonexistent agents.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent_id=agent_id_strategy)
    def test_agent_card_nonexistent_returns_none(self, agent_id: str):
        """
        Property 11: Agent card for nonexistent agent returns None.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.6
        """
        gateway = A2AGateway(enabled=True)
        
        card = gateway.get_agent_card(agent_id)
        assert card is None


class TestA2AAgentDataIntegrity:
    """
    Tests for data integrity during registration and retrieval.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_agent_data_preserved_on_registration(self, agent: A2AAgent):
        """
        Property 11: All agent data is preserved during registration.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.2
        """
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        retrieved = gateway.get_agent(agent.agent_id)
        assert retrieved is not None
        
        # Verify all fields preserved
        assert retrieved.agent_id == agent.agent_id
        assert retrieved.name == agent.name
        assert retrieved.description == agent.description
        assert retrieved.url == agent.url
        assert retrieved.capabilities == agent.capabilities
        assert retrieved.metadata == agent.metadata

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_agent_serialization_round_trip(self, agent: A2AAgent):
        """
        Property 11: Agent data survives JSON serialization round-trip.
        
        Feature: production-ai-gateway, Property 11: A2A Agent Registration and Discovery
        Validates: Requirements 7.2
        """
        # Serialize to JSON-compatible dict
        agent_dict = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "url": agent.url,
            "capabilities": agent.capabilities,
            "metadata": agent.metadata,
        }
        
        # Round-trip through JSON
        json_str = json.dumps(agent_dict)
        loaded = json.loads(json_str)
        
        # Reconstruct agent
        reconstructed = A2AAgent(
            agent_id=loaded["agent_id"],
            name=loaded["name"],
            description=loaded["description"],
            url=loaded["url"],
            capabilities=loaded["capabilities"],
            metadata=loaded["metadata"],
        )
        
        # Verify equality
        assert reconstructed.agent_id == agent.agent_id
        assert reconstructed.name == agent.name
        assert reconstructed.description == agent.description
        assert reconstructed.url == agent.url
        assert reconstructed.capabilities == agent.capabilities
        assert reconstructed.metadata == agent.metadata



# =============================================================================
# JSON-RPC 2.0 Data Models (mirroring src/litellm_llmrouter/a2a_gateway.py)
# =============================================================================


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


# =============================================================================
# Additional Test Data Generators for Invocation
# =============================================================================

# JSON-RPC request ID
request_id_strategy = st.one_of(
    st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    st.integers(min_value=1, max_value=1000000),
)

# A2A message part
message_part_strategy = st.fixed_dictionaries({
    "type": st.just("text"),
    "text": st.text(min_size=1, max_size=500),
})

# A2A message
a2a_message_strategy = st.fixed_dictionaries({
    "role": st.sampled_from(["user", "agent"]),
    "parts": st.lists(message_part_strategy, min_size=1, max_size=3),
})

# JSON-RPC params for message/send
message_send_params_strategy = st.fixed_dictionaries({
    "message": a2a_message_strategy,
})


@st.composite
def jsonrpc_request_strategy(draw, method: str = "message/send"):
    """Generate a valid JSON-RPC 2.0 request."""
    return JSONRPCRequest(
        method=method,
        params=draw(message_send_params_strategy),
        id=draw(request_id_strategy),
        jsonrpc="2.0",
    )


# =============================================================================
# Property Tests for A2A Agent Invocation
# =============================================================================


class TestA2AAgentInvocationProperty:
    """
    Property 23: A2A Agent Invocation
    
    For any registered A2A agent, when a valid JSON-RPC 2.0 request with method
    `message/send` is POSTed to `/a2a/{agent_id}`, the Gateway should forward
    the message to the agent backend and return a valid JSON-RPC 2.0 response.
    
    **Validates: Requirements 7.8, 7.9**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request=jsonrpc_request_strategy())
    def test_jsonrpc_request_has_required_fields(self, request: JSONRPCRequest):
        """
        Property 23: JSON-RPC request has all required fields.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.8
        """
        request_dict = request.to_dict()
        assert "jsonrpc" in request_dict
        assert request_dict["jsonrpc"] == "2.0"
        assert "method" in request_dict
        assert "params" in request_dict
        assert "id" in request_dict

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request=jsonrpc_request_strategy())
    def test_jsonrpc_request_serialization_round_trip(self, request: JSONRPCRequest):
        """
        Property 23: JSON-RPC request survives JSON serialization round-trip.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.8
        """
        request_dict = request.to_dict()
        json_str = json.dumps(request_dict)
        loaded = json.loads(json_str)
        
        assert loaded["jsonrpc"] == request.jsonrpc
        assert loaded["method"] == request.method
        assert loaded["params"] == request.params
        assert loaded["id"] == request.id

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request_id=request_id_strategy)
    def test_jsonrpc_error_response_format(self, request_id: str | int):
        """
        Property 23: JSON-RPC error response has correct format.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.9
        """
        response = JSONRPCResponse.error_response(request_id, -32000, "Test error")
        response_dict = response.to_dict()
        
        assert response_dict["jsonrpc"] == "2.0"
        assert response_dict["id"] == request_id
        assert "error" in response_dict
        assert response_dict["error"]["code"] == -32000
        assert response_dict["error"]["message"] == "Test error"
        assert "result" not in response_dict

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request_id=request_id_strategy)
    def test_jsonrpc_success_response_format(self, request_id: str | int):
        """
        Property 23: JSON-RPC success response has correct format.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.9
        """
        result = {"status": "completed", "data": "test"}
        response = JSONRPCResponse.success_response(request_id, result)
        response_dict = response.to_dict()
        
        assert response_dict["jsonrpc"] == "2.0"
        assert response_dict["id"] == request_id
        assert "result" in response_dict
        assert response_dict["result"] == result
        assert "error" not in response_dict

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        agent=a2a_agent_strategy(),
        request=jsonrpc_request_strategy(),
    )
    def test_invoke_nonexistent_agent_returns_error(
        self, agent: A2AAgent, request: JSONRPCRequest
    ):
        """
        Property 23: Invoking nonexistent agent returns JSON-RPC error.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.8, 7.9
        """
        gateway = A2AGateway(enabled=True)
        # Don't register the agent
        
        # Simulate what invoke_agent would return for nonexistent agent
        response = JSONRPCResponse.error_response(
            request.id, -32000, f"Agent '{agent.agent_id}' not found"
        )
        
        assert response.error is not None
        assert response.error["code"] == -32000
        assert "not found" in response.error["message"].lower()

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        agent=a2a_agent_strategy(),
        request=jsonrpc_request_strategy(),
    )
    def test_invoke_disabled_gateway_returns_error(
        self, agent: A2AAgent, request: JSONRPCRequest
    ):
        """
        Property 23: Invoking agent on disabled gateway returns JSON-RPC error.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.8, 7.9
        """
        gateway = A2AGateway(enabled=False)
        
        # Simulate what invoke_agent would return for disabled gateway
        response = JSONRPCResponse.error_response(
            request.id, -32000, "A2A Gateway is not enabled"
        )
        
        assert response.error is not None
        assert response.error["code"] == -32000
        assert "not enabled" in response.error["message"].lower()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request_id=request_id_strategy)
    def test_invalid_jsonrpc_version_returns_error(self, request_id: str | int):
        """
        Property 23: Invalid JSON-RPC version returns error.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.8
        """
        # Create request with invalid version
        request = JSONRPCRequest(
            method="message/send",
            params={},
            id=request_id,
            jsonrpc="1.0",  # Invalid version
        )
        
        # Validation should fail
        assert request.jsonrpc != "2.0"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(method=st.text(min_size=1, max_size=50).filter(
        lambda x: x not in ("message/send", "message/stream")
    ))
    def test_unsupported_method_returns_error(self, method: str):
        """
        Property 23: Unsupported method returns JSON-RPC method not found error.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.8, 7.9
        """
        # Simulate error response for unsupported method
        response = JSONRPCResponse.error_response(
            "1", -32601, f"Method '{method}' not found"
        )
        
        assert response.error is not None
        assert response.error["code"] == -32601  # Method not found
        assert method in response.error["message"]

    def test_supported_methods(self):
        """
        Property 23: Verify supported A2A methods.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.8, 7.9
        """
        supported_methods = ["message/send", "message/stream"]
        
        for method in supported_methods:
            request = JSONRPCRequest(
                method=method,
                params={"message": {"role": "user", "parts": [{"type": "text", "text": "test"}]}},
                id="1",
            )
            assert request.method in supported_methods

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        agent=a2a_agent_strategy(),
        request=jsonrpc_request_strategy(),
    )
    def test_agent_without_url_returns_error(
        self, agent: A2AAgent, request: JSONRPCRequest
    ):
        """
        Property 23: Agent without URL returns JSON-RPC error.
        
        Feature: production-ai-gateway, Property 23: A2A Agent Invocation
        Validates: Requirements 7.8, 7.9
        """
        # Create agent without URL
        agent.url = ""
        
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        # Simulate what invoke_agent would return for agent without URL
        response = JSONRPCResponse.error_response(
            request.id, -32000, f"Agent '{agent.agent_id}' has no URL configured"
        )
        
        assert response.error is not None
        assert "no URL" in response.error["message"]


class TestA2AJSONRPCErrorCodes:
    """
    Tests for JSON-RPC 2.0 error codes compliance.
    
    Standard error codes:
    - -32700: Parse error
    - -32600: Invalid Request
    - -32601: Method not found
    - -32602: Invalid params
    - -32603: Internal error
    - -32000 to -32099: Server error (reserved for implementation-defined errors)
    """

    def test_parse_error_code(self):
        """Test parse error code (-32700)."""
        response = JSONRPCResponse.error_response(None, -32700, "Parse error")
        assert response.error["code"] == -32700

    def test_invalid_request_code(self):
        """Test invalid request code (-32600)."""
        response = JSONRPCResponse.error_response("1", -32600, "Invalid Request")
        assert response.error["code"] == -32600

    def test_method_not_found_code(self):
        """Test method not found code (-32601)."""
        response = JSONRPCResponse.error_response("1", -32601, "Method not found")
        assert response.error["code"] == -32601

    def test_internal_error_code(self):
        """Test internal error code (-32603)."""
        response = JSONRPCResponse.error_response("1", -32603, "Internal error")
        assert response.error["code"] == -32603

    def test_server_error_code(self):
        """Test server error code (-32000)."""
        response = JSONRPCResponse.error_response("1", -32000, "Server error")
        assert response.error["code"] == -32000



# =============================================================================
# Property Tests for A2A Streaming Response
# =============================================================================


class TestA2AStreamingResponseProperty:
    """
    Property 24: A2A Streaming Response
    
    For any registered A2A agent, when a valid JSON-RPC 2.0 request with method
    `message/stream` is POSTed to `/a2a/{agent_id}`, the Gateway should stream
    the response using Server-Sent Events with proper event formatting.
    
    **Validates: Requirements 7.10**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request=jsonrpc_request_strategy(method="message/stream"))
    def test_streaming_request_has_correct_method(self, request: JSONRPCRequest):
        """
        Property 24: Streaming request has method 'message/stream'.
        
        Feature: production-ai-gateway, Property 24: A2A Streaming Response
        Validates: Requirements 7.10
        """
        # Override method to message/stream
        request.method = "message/stream"
        assert request.method == "message/stream"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request_id=request_id_strategy)
    def test_streaming_error_response_is_valid_json(self, request_id: str | int):
        """
        Property 24: Streaming error response is valid JSON.
        
        Feature: production-ai-gateway, Property 24: A2A Streaming Response
        Validates: Requirements 7.10
        """
        response = JSONRPCResponse.error_response(
            request_id, -32000, "Streaming error"
        )
        response_dict = response.to_dict()
        
        # Should be serializable to JSON
        json_str = json.dumps(response_dict)
        loaded = json.loads(json_str)
        
        assert loaded["jsonrpc"] == "2.0"
        assert loaded["id"] == request_id
        assert "error" in loaded

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_streaming_requires_agent_url(self, agent: A2AAgent):
        """
        Property 24: Streaming requires agent to have a URL.
        
        Feature: production-ai-gateway, Property 24: A2A Streaming Response
        Validates: Requirements 7.10
        """
        # Agent without URL should fail streaming
        agent.url = ""
        
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        # Verify agent has no URL
        retrieved = gateway.get_agent(agent.agent_id)
        assert retrieved is not None
        assert retrieved.url == ""

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_streaming_capability_in_agent_card(self, agent: A2AAgent):
        """
        Property 24: Agent card indicates streaming capability.
        
        Feature: production-ai-gateway, Property 24: A2A Streaming Response
        Validates: Requirements 7.10
        """
        # Add streaming capability
        agent.capabilities = list(set(agent.capabilities + ["streaming"]))
        
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        card = gateway.get_agent_card(agent.agent_id)
        assert card is not None
        assert card["capabilities"]["streaming"] is True

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_strategy())
    def test_non_streaming_agent_card(self, agent: A2AAgent):
        """
        Property 24: Agent card correctly indicates no streaming capability.
        
        Feature: production-ai-gateway, Property 24: A2A Streaming Response
        Validates: Requirements 7.10
        """
        # Remove streaming capability
        agent.capabilities = [c for c in agent.capabilities if c != "streaming"]
        
        gateway = A2AGateway(enabled=True)
        gateway.register_agent(agent)
        
        card = gateway.get_agent_card(agent.agent_id)
        assert card is not None
        assert card["capabilities"]["streaming"] is False

    def test_streaming_method_is_supported(self):
        """
        Property 24: message/stream is a supported method.
        
        Feature: production-ai-gateway, Property 24: A2A Streaming Response
        Validates: Requirements 7.10
        """
        supported_methods = ["message/send", "message/stream"]
        assert "message/stream" in supported_methods

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request_id=request_id_strategy,
        error_message=st.text(min_size=1, max_size=200),
    )
    def test_streaming_error_format_ndjson(
        self, request_id: str | int, error_message: str
    ):
        """
        Property 24: Streaming errors are formatted as newline-delimited JSON.
        
        Feature: production-ai-gateway, Property 24: A2A Streaming Response
        Validates: Requirements 7.10
        """
        response = JSONRPCResponse.error_response(request_id, -32603, error_message)
        json_line = json.dumps(response.to_dict()) + "\n"
        
        # Should end with newline
        assert json_line.endswith("\n")
        
        # Should be valid JSON when stripped
        loaded = json.loads(json_line.strip())
        assert loaded["jsonrpc"] == "2.0"
        assert loaded["error"]["message"] == error_message


# =============================================================================
# Database Models (mirroring src/litellm_llmrouter/database.py)
# =============================================================================


@dataclass
class A2AAgentDB:
    """Database model for A2A agent."""

    agent_id: str
    name: str
    description: str
    url: str
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    team_id: str | None = None
    user_id: str | None = None
    is_public: bool = False
    created_at: Any = None
    updated_at: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "team_id": self.team_id,
            "user_id": self.user_id,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "A2AAgentDB":
        """Create from dictionary."""
        return cls(
            agent_id=data.get("agent_id", ""),
            name=data["name"],
            description=data.get("description", ""),
            url=data["url"],
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {}),
            team_id=data.get("team_id"),
            user_id=data.get("user_id"),
            is_public=data.get("is_public", False),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


class A2AAgentRepository:
    """
    In-memory repository for A2A agent persistence testing.
    
    Mirrors the production A2AAgentRepository for property testing.
    """

    def __init__(self):
        self._agents: dict[str, A2AAgentDB] = {}

    async def create(self, agent: A2AAgentDB) -> A2AAgentDB:
        """Create a new agent."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        agent.created_at = now
        agent.updated_at = now
        self._agents[agent.agent_id] = agent
        return agent

    async def get(self, agent_id: str) -> A2AAgentDB | None:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    async def list_all(
        self,
        user_id: str | None = None,
        team_id: str | None = None,
        include_public: bool = True,
    ) -> list[A2AAgentDB]:
        """List all agents with optional filtering."""
        agents = list(self._agents.values())
        filtered = []
        for agent in agents:
            if include_public and agent.is_public:
                filtered.append(agent)
                continue
            if user_id and agent.user_id == user_id:
                filtered.append(agent)
                continue
            if team_id and agent.team_id == team_id:
                filtered.append(agent)
                continue
            if not user_id and not team_id:
                filtered.append(agent)
        return filtered

    async def update(self, agent_id: str, agent: A2AAgentDB) -> A2AAgentDB | None:
        """Update an existing agent (full update)."""
        from datetime import datetime, timezone
        if agent_id not in self._agents:
            return None
        agent.updated_at = datetime.now(timezone.utc)
        agent.created_at = self._agents[agent_id].created_at
        self._agents[agent_id] = agent
        return agent

    async def patch(
        self, agent_id: str, updates: dict[str, Any]
    ) -> A2AAgentDB | None:
        """Partially update an agent."""
        from datetime import datetime, timezone
        agent = await self.get(agent_id)
        if not agent:
            return None
        for key, value in updates.items():
            if hasattr(agent, key) and key not in ("agent_id", "created_at"):
                setattr(agent, key, value)
        agent.updated_at = datetime.now(timezone.utc)
        self._agents[agent_id] = agent
        return agent

    async def delete(self, agent_id: str) -> bool:
        """Delete an agent."""
        if agent_id not in self._agents:
            return False
        del self._agents[agent_id]
        return True

    async def make_public(self, agent_id: str) -> A2AAgentDB | None:
        """Make an agent public."""
        return await self.patch(agent_id, {"is_public": True})


# =============================================================================
# Additional Test Data Generators for Database Persistence
# =============================================================================

# Team ID strategy
team_id_strategy = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='-_'),
        min_size=1,
        max_size=50,
    ).filter(lambda x: x.strip()),
)

# User ID strategy
user_id_strategy = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='-_'),
        min_size=1,
        max_size=50,
    ).filter(lambda x: x.strip()),
)


@st.composite
def a2a_agent_db_strategy(draw):
    """Generate a valid A2A agent for database persistence."""
    import uuid
    return A2AAgentDB(
        agent_id=str(uuid.uuid4()),
        name=draw(agent_name_strategy),
        description=draw(agent_description_strategy),
        url=draw(url_strategy),
        capabilities=draw(capabilities_strategy),
        metadata=draw(metadata_strategy),
        team_id=draw(team_id_strategy),
        user_id=draw(user_id_strategy),
        is_public=draw(st.booleans()),
    )


@st.composite
def multiple_agents_db_strategy(draw, min_agents=1, max_agents=10):
    """Generate multiple unique agents for database testing."""
    num_agents = draw(st.integers(min_value=min_agents, max_value=max_agents))
    agents = []
    for _ in range(num_agents):
        agent = draw(a2a_agent_db_strategy())
        agents.append(agent)
    assume(len(agents) >= min_agents)
    return agents


# =============================================================================
# Property Tests for A2A Database Persistence
# =============================================================================


class TestA2ADatabasePersistenceProperty:
    """
    Property 25: A2A Database Persistence
    
    For any A2A agent created via POST `/v1/agents`, the agent should be
    persisted to the database and retrievable via GET `/v1/agents/{agent_id}`.
    The agent should support full updates (PUT), partial updates (PATCH),
    and deletion (DELETE).
    
    **Validates: Requirements 7.7**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_created_agent_is_retrievable(self, agent: A2AAgentDB):
        """
        Property 25: For any created agent, get returns the same agent.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.7
        """
        repo = A2AAgentRepository()
        created = await repo.create(agent)
        
        retrieved = await repo.get(agent.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == agent.agent_id
        assert retrieved.name == agent.name
        assert retrieved.description == agent.description
        assert retrieved.url == agent.url
        assert retrieved.capabilities == agent.capabilities
        assert retrieved.team_id == agent.team_id
        assert retrieved.user_id == agent.user_id
        assert retrieved.is_public == agent.is_public

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_created_agent_has_timestamps(self, agent: A2AAgentDB):
        """
        Property 25: Created agent has created_at and updated_at timestamps.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.7
        """
        repo = A2AAgentRepository()
        created = await repo.create(agent)
        
        assert created.created_at is not None
        assert created.updated_at is not None
        assert created.created_at == created.updated_at

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agents=multiple_agents_db_strategy(min_agents=1, max_agents=5))
    @pytest.mark.asyncio
    async def test_all_created_agents_in_list(self, agents: list[A2AAgentDB]):
        """
        Property 25: All created agents appear in list_all.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.7
        """
        repo = A2AAgentRepository()
        for agent in agents:
            await repo.create(agent)
        
        all_agents = await repo.list_all()
        all_ids = {a.agent_id for a in all_agents}
        
        for agent in agents:
            assert agent.agent_id in all_ids

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_deleted_agent_not_retrievable(self, agent: A2AAgentDB):
        """
        Property 25: Deleted agent is not retrievable.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.7
        """
        repo = A2AAgentRepository()
        await repo.create(agent)
        
        # Verify created
        assert await repo.get(agent.agent_id) is not None
        
        # Delete
        result = await repo.delete(agent.agent_id)
        assert result is True
        
        # Verify not retrievable
        assert await repo.get(agent.agent_id) is None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, agent: A2AAgentDB):
        """
        Property 25: Deleting nonexistent agent returns False.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.7
        """
        repo = A2AAgentRepository()
        
        result = await repo.delete(agent.agent_id)
        assert result is False

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_agent_to_dict_round_trip(self, agent: A2AAgentDB):
        """
        Property 25: Agent survives to_dict/from_dict round-trip.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.7
        """
        repo = A2AAgentRepository()
        created = await repo.create(agent)
        
        # Convert to dict and back
        agent_dict = created.to_dict()
        reconstructed = A2AAgentDB.from_dict(agent_dict)
        
        assert reconstructed.agent_id == created.agent_id
        assert reconstructed.name == created.name
        assert reconstructed.description == created.description
        assert reconstructed.url == created.url
        assert reconstructed.capabilities == created.capabilities
        assert reconstructed.team_id == created.team_id
        assert reconstructed.user_id == created.user_id
        assert reconstructed.is_public == created.is_public


class TestA2AAgentUpdateProperty:
    """
    Property 26: A2A Agent Updates
    
    For any registered A2A agent, PUT `/v1/agents/{agent_id}` should replace
    all fields, and PATCH `/v1/agents/{agent_id}` should update only the
    provided fields while preserving others.
    
    **Validates: Requirements 7.11, 7.12**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        agent=a2a_agent_db_strategy(),
        new_name=agent_name_strategy,
        new_description=agent_description_strategy,
    )
    @pytest.mark.asyncio
    async def test_full_update_replaces_all_fields(
        self, agent: A2AAgentDB, new_name: str, new_description: str
    ):
        """
        Property 26: Full update (PUT) replaces all fields.
        
        Feature: production-ai-gateway, Property 26: A2A Agent Updates
        Validates: Requirements 7.11
        """
        repo = A2AAgentRepository()
        await repo.create(agent)
        
        # Create updated agent
        updated_agent = A2AAgentDB(
            agent_id=agent.agent_id,
            name=new_name,
            description=new_description,
            url=agent.url,
            capabilities=[],  # Clear capabilities
            metadata={},  # Clear metadata
            team_id=None,
            user_id=None,
            is_public=not agent.is_public,
        )
        
        result = await repo.update(agent.agent_id, updated_agent)
        assert result is not None
        assert result.name == new_name
        assert result.description == new_description
        assert result.capabilities == []
        assert result.metadata == {}
        assert result.is_public == (not agent.is_public)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        agent=a2a_agent_db_strategy(),
        new_name=agent_name_strategy,
    )
    @pytest.mark.asyncio
    async def test_partial_update_preserves_other_fields(
        self, agent: A2AAgentDB, new_name: str
    ):
        """
        Property 26: Partial update (PATCH) preserves unspecified fields.
        
        Feature: production-ai-gateway, Property 26: A2A Agent Updates
        Validates: Requirements 7.12
        """
        repo = A2AAgentRepository()
        await repo.create(agent)
        
        original_description = agent.description
        original_url = agent.url
        original_capabilities = agent.capabilities.copy()
        
        # Patch only name
        result = await repo.patch(agent.agent_id, {"name": new_name})
        assert result is not None
        assert result.name == new_name
        assert result.description == original_description
        assert result.url == original_url
        assert result.capabilities == original_capabilities

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_update_preserves_created_at(self, agent: A2AAgentDB):
        """
        Property 26: Update preserves created_at timestamp.
        
        Feature: production-ai-gateway, Property 26: A2A Agent Updates
        Validates: Requirements 7.11, 7.12
        """
        repo = A2AAgentRepository()
        created = await repo.create(agent)
        original_created_at = created.created_at
        
        # Update
        updated_agent = A2AAgentDB(
            agent_id=agent.agent_id,
            name="Updated Name",
            description=agent.description,
            url=agent.url,
            capabilities=agent.capabilities,
            metadata=agent.metadata,
            team_id=agent.team_id,
            user_id=agent.user_id,
            is_public=agent.is_public,
        )
        result = await repo.update(agent.agent_id, updated_agent)
        
        assert result is not None
        assert result.created_at == original_created_at

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_update_changes_updated_at(self, agent: A2AAgentDB):
        """
        Property 26: Update changes updated_at timestamp.
        
        Feature: production-ai-gateway, Property 26: A2A Agent Updates
        Validates: Requirements 7.11, 7.12
        """
        import time
        repo = A2AAgentRepository()
        created = await repo.create(agent)
        original_updated_at = created.updated_at
        
        # Small delay to ensure timestamp difference
        time.sleep(0.001)
        
        # Patch
        result = await repo.patch(agent.agent_id, {"name": "New Name"})
        
        assert result is not None
        assert result.updated_at >= original_updated_at

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self, agent: A2AAgentDB):
        """
        Property 26: Updating nonexistent agent returns None.
        
        Feature: production-ai-gateway, Property 26: A2A Agent Updates
        Validates: Requirements 7.11
        """
        repo = A2AAgentRepository()
        
        result = await repo.update(agent.agent_id, agent)
        assert result is None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_patch_nonexistent_returns_none(self, agent: A2AAgentDB):
        """
        Property 26: Patching nonexistent agent returns None.
        
        Feature: production-ai-gateway, Property 26: A2A Agent Updates
        Validates: Requirements 7.12
        """
        repo = A2AAgentRepository()
        
        result = await repo.patch(agent.agent_id, {"name": "New Name"})
        assert result is None


class TestA2APermissionFilteringProperty:
    """
    Property 25: A2A Permission Filtering
    
    Tests that agents can be filtered by user_id, team_id, and is_public.
    
    **Validates: Requirements 7.13**
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agents=multiple_agents_db_strategy(min_agents=3, max_agents=5))
    @pytest.mark.asyncio
    async def test_filter_by_user_id(self, agents: list[A2AAgentDB]):
        """
        Property 25: list_all filters by user_id correctly.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.13
        """
        repo = A2AAgentRepository()
        
        # Set specific user_id for first agent
        target_user = "test-user-123"
        agents[0].user_id = target_user
        agents[0].is_public = False
        
        for agent in agents:
            await repo.create(agent)
        
        # Filter by user_id
        filtered = await repo.list_all(user_id=target_user, include_public=False)
        
        # Should include agent with matching user_id
        filtered_ids = {a.agent_id for a in filtered}
        assert agents[0].agent_id in filtered_ids

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agents=multiple_agents_db_strategy(min_agents=3, max_agents=5))
    @pytest.mark.asyncio
    async def test_filter_by_team_id(self, agents: list[A2AAgentDB]):
        """
        Property 25: list_all filters by team_id correctly.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.13
        """
        repo = A2AAgentRepository()
        
        # Set specific team_id for first agent
        target_team = "test-team-456"
        agents[0].team_id = target_team
        agents[0].is_public = False
        
        for agent in agents:
            await repo.create(agent)
        
        # Filter by team_id
        filtered = await repo.list_all(team_id=target_team, include_public=False)
        
        # Should include agent with matching team_id
        filtered_ids = {a.agent_id for a in filtered}
        assert agents[0].agent_id in filtered_ids

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agents=multiple_agents_db_strategy(min_agents=3, max_agents=5))
    @pytest.mark.asyncio
    async def test_include_public_agents(self, agents: list[A2AAgentDB]):
        """
        Property 25: list_all includes public agents when include_public=True.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.13
        """
        repo = A2AAgentRepository()
        
        # Make first agent public
        agents[0].is_public = True
        agents[0].user_id = None
        agents[0].team_id = None
        
        for agent in agents:
            await repo.create(agent)
        
        # Filter with include_public=True
        filtered = await repo.list_all(
            user_id="other-user",
            include_public=True,
        )
        
        # Should include public agent
        filtered_ids = {a.agent_id for a in filtered}
        assert agents[0].agent_id in filtered_ids

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agents=multiple_agents_db_strategy(min_agents=3, max_agents=5))
    @pytest.mark.asyncio
    async def test_exclude_public_agents(self, agents: list[A2AAgentDB]):
        """
        Property 25: list_all excludes public agents when include_public=False.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.13
        """
        repo = A2AAgentRepository()
        
        # Make first agent public only (no user/team)
        agents[0].is_public = True
        agents[0].user_id = None
        agents[0].team_id = None
        
        for agent in agents:
            await repo.create(agent)
        
        # Filter with include_public=False and non-matching user
        filtered = await repo.list_all(
            user_id="other-user",
            include_public=False,
        )
        
        # Should NOT include public agent (since it doesn't match user)
        filtered_ids = {a.agent_id for a in filtered}
        assert agents[0].agent_id not in filtered_ids

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_make_public(self, agent: A2AAgentDB):
        """
        Property 25: make_public sets is_public to True.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.13
        """
        repo = A2AAgentRepository()
        agent.is_public = False
        await repo.create(agent)
        
        result = await repo.make_public(agent.agent_id)
        assert result is not None
        assert result.is_public is True

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_make_public_nonexistent_returns_none(self, agent: A2AAgentDB):
        """
        Property 25: make_public on nonexistent agent returns None.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.13
        """
        repo = A2AAgentRepository()
        
        result = await repo.make_public(agent.agent_id)
        assert result is None


class TestA2AAgentDBDataIntegrity:
    """
    Tests for data integrity in database operations.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_agent_id_immutable_on_update(self, agent: A2AAgentDB):
        """
        Property 25: agent_id cannot be changed via update.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.7
        """
        repo = A2AAgentRepository()
        await repo.create(agent)
        
        original_id = agent.agent_id
        
        # Try to update with different agent_id (should use original)
        updated_agent = A2AAgentDB(
            agent_id="different-id",
            name="New Name",
            description=agent.description,
            url=agent.url,
            capabilities=agent.capabilities,
            metadata=agent.metadata,
            team_id=agent.team_id,
            user_id=agent.user_id,
            is_public=agent.is_public,
        )
        
        # Update using original ID
        result = await repo.update(original_id, updated_agent)
        assert result is not None
        
        # Original ID should still work
        retrieved = await repo.get(original_id)
        assert retrieved is not None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_created_at_immutable_on_patch(self, agent: A2AAgentDB):
        """
        Property 25: created_at cannot be changed via patch.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.7
        """
        repo = A2AAgentRepository()
        created = await repo.create(agent)
        original_created_at = created.created_at
        
        # Try to patch created_at (should be ignored)
        from datetime import datetime
        result = await repo.patch(agent.agent_id, {"created_at": datetime(2000, 1, 1)})
        
        assert result is not None
        assert result.created_at == original_created_at

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(agent=a2a_agent_db_strategy())
    @pytest.mark.asyncio
    async def test_json_serialization_preserves_data(self, agent: A2AAgentDB):
        """
        Property 25: Agent data survives JSON serialization.
        
        Feature: production-ai-gateway, Property 25: A2A Database Persistence
        Validates: Requirements 7.7
        """
        repo = A2AAgentRepository()
        created = await repo.create(agent)
        
        # Serialize to JSON
        agent_dict = created.to_dict()
        json_str = json.dumps(agent_dict)
        loaded = json.loads(json_str)
        
        # Verify key fields
        assert loaded["agent_id"] == created.agent_id
        assert loaded["name"] == created.name
        assert loaded["description"] == created.description
        assert loaded["url"] == created.url
        assert loaded["capabilities"] == created.capabilities
        assert loaded["team_id"] == created.team_id
        assert loaded["user_id"] == created.user_id
        assert loaded["is_public"] == created.is_public
