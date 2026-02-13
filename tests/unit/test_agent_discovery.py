"""
Tests for Cloudflare Agent Skills Discovery (/.well-known/agent.json).

The A2A Agent Card endpoint at /.well-known/agent.json also satisfies the
Cloudflare Agent Skills Discovery RFC, which expects a machine-readable
description of agent capabilities at this well-known path.

These tests verify:
- The endpoint returns valid JSON with the required discovery fields
- The response includes name, description, version, and capabilities
- The skills array maps correctly from registered A2A agents
- The endpoint works with no agents registered (empty skills)
- Required A2A fields are present (name, url, version, capabilities, skills)
"""

from unittest.mock import patch

import pytest

from litellm_llmrouter.a2a_gateway import (
    A2AAgent,
    reset_a2a_gateway,
)
from litellm_llmrouter.routes.health import get_gateway_agent_card

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_gateway():
    """Reset the global A2A gateway singleton before/after each test."""
    reset_a2a_gateway()
    yield
    reset_a2a_gateway()


@pytest.fixture
def sample_agent():
    """Create a sample A2A agent for testing."""
    return A2AAgent(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent for discovery",
        url="https://agent.example.com/a2a",
        capabilities=["streaming"],
    )


async def test_agent_json_returns_name():
    """/.well-known/agent.json response includes a name field."""
    result = await get_gateway_agent_card()
    assert "name" in result
    assert isinstance(result["name"], str)
    assert len(result["name"]) > 0


async def test_agent_json_returns_description():
    """/.well-known/agent.json response includes a description field."""
    result = await get_gateway_agent_card()
    assert "description" in result
    assert isinstance(result["description"], str)


async def test_agent_json_returns_version():
    """/.well-known/agent.json response includes a version field."""
    result = await get_gateway_agent_card()
    assert "version" in result
    assert isinstance(result["version"], str)


async def test_agent_json_returns_capabilities():
    """/.well-known/agent.json response includes capabilities."""
    result = await get_gateway_agent_card()
    assert "capabilities" in result
    assert isinstance(result["capabilities"], dict)


async def test_agent_json_returns_skills():
    """/.well-known/agent.json response includes skills array."""
    result = await get_gateway_agent_card()
    assert "skills" in result
    assert isinstance(result["skills"], list)


async def test_agent_json_empty_skills_when_no_agents():
    """Skills array is empty when no agents are registered."""
    result = await get_gateway_agent_card()
    assert result["skills"] == []


async def test_agent_json_skills_populated_after_registration(sample_agent):
    """Skills array includes registered agents."""
    from litellm_llmrouter.a2a_gateway import get_a2a_gateway

    gateway = get_a2a_gateway()
    # Enable the gateway and register directly (bypassing SSRF check
    # which is separately tested in test_a2a_compliance.py)
    gateway.enabled = True
    with gateway._lock:
        gateway.agents[sample_agent.agent_id] = sample_agent

    result = await get_gateway_agent_card()
    assert len(result["skills"]) == 1
    assert result["skills"][0]["id"] == "test-agent"
    assert result["skills"][0]["name"] == "Test Agent"


async def test_agent_json_has_authentication():
    """/.well-known/agent.json response includes authentication info."""
    result = await get_gateway_agent_card()
    assert "authentication" in result


async def test_agent_json_has_url():
    """/.well-known/agent.json response includes a url field."""
    result = await get_gateway_agent_card()
    assert "url" in result


async def test_agent_json_has_default_modes():
    """/.well-known/agent.json response includes default I/O modes."""
    result = await get_gateway_agent_card()
    assert "defaultInputModes" in result
    assert "defaultOutputModes" in result
    assert isinstance(result["defaultInputModes"], list)
    assert isinstance(result["defaultOutputModes"], list)


async def test_agent_json_cloudflare_rfc_required_fields():
    """Verify all fields expected by Cloudflare Agent Skills Discovery RFC.

    The Cloudflare RFC expects: name, description, version, capabilities, skills.
    The A2A Agent Card includes all of these plus authentication and modes.
    """
    result = await get_gateway_agent_card()

    # Cloudflare RFC required fields
    cloudflare_required = ["name", "description", "version", "capabilities"]
    for field in cloudflare_required:
        assert field in result, f"Missing Cloudflare RFC field: {field}"

    # Must have skills (or an empty array)
    assert "skills" in result

    # Capabilities must be a dict (both A2A and Cloudflare expect this)
    assert isinstance(result["capabilities"], dict)


async def test_agent_json_gateway_identity():
    """The gateway agent card identifies itself as RouteIQ."""
    result = await get_gateway_agent_card()
    assert "RouteIQ" in result["name"]


async def test_agent_json_with_custom_base_url():
    """Agent card url field reflects the configured base URL."""
    with patch.dict("os.environ", {"A2A_BASE_URL": "https://gateway.example.com"}):
        result = await get_gateway_agent_card()
        assert "gateway.example.com" in result["url"]
