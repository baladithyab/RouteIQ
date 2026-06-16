"""
Unit tests for the A2A native invocation route + unified registry (RouteIQ-a33c).

Before: a2a_gateway had JSON-RPC/SSE types but NO HTTP invocation route, and
agent CRUD wrapped LiteLLM's in-memory registry => two registries (CRUD wrote
LiteLLM, invocation read the native gateway). These tests verify:
- POST /a2a/{agent_id} resolves an agent and dispatches a JSON-RPC invocation
- the streaming alias returns SSE
- a single unified registry: an agent registered in LiteLLM's registry is
  resolvable by the native gateway (resolve_agent adopts it), and vice versa
- feature-flag gating (A2A_GATEWAY_ENABLED)

HTTP to the backend agent is mocked — no live agent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm_llmrouter import a2a_gateway as a2a_mod
from litellm_llmrouter.a2a_gateway import (
    A2AAgent,
    A2AGateway,
    JSONRPCRequest,
    get_a2a_gateway,
    reset_a2a_gateway,
)
from litellm_llmrouter.gateway.app import create_standalone_app


# ---------------------------------------------------------------------------
# Registry unification (gateway level)
# ---------------------------------------------------------------------------


class TestUnifiedRegistry:
    def test_resolve_native_agent(self):
        gw = A2AGateway()
        gw.enabled = True
        gw.register_agent(
            A2AAgent(agent_id="a1", name="A1", description="", url="https://x/a")
        )
        assert gw.resolve_agent("a1") is not None

    def test_resolve_adopts_from_litellm_registry(self):
        """A miss in the native registry falls back to LiteLLM + adopts it."""
        gw = A2AGateway()
        gw.enabled = True
        assert gw.get_agent("ll-agent") is None

        ll_agent = MagicMock()
        ll_agent.agent_id = "ll-agent"
        ll_agent.agent_name = "LL Agent"
        ll_agent.agent_card_params = {
            "url": "https://x/ll",
            "description": "from litellm",
            "capabilities": {"streaming": True},
        }
        registry = MagicMock()
        registry.get_agent_by_id.return_value = ll_agent

        with patch.object(
            a2a_mod.A2AGateway,
            "_adopt_from_litellm_registry",
            wraps=gw._adopt_from_litellm_registry,
        ):
            with patch.dict(
                "sys.modules",
                {
                    "litellm.proxy.agent_endpoints.agent_registry": MagicMock(
                        global_agent_registry=registry
                    )
                },
            ):
                resolved = gw.resolve_agent("ll-agent")

        assert resolved is not None
        assert resolved.name == "LL Agent"
        assert resolved.url == "https://x/ll"
        assert "streaming" in resolved.capabilities
        # Adopted into the native registry => single source of truth.
        assert gw.get_agent("ll-agent") is not None

    def test_resolve_returns_none_when_unknown_everywhere(self):
        gw = A2AGateway()
        gw.enabled = True
        registry = MagicMock()
        registry.get_agent_by_id.return_value = None
        registry.get_agent_by_name.return_value = None
        with patch.dict(
            "sys.modules",
            {
                "litellm.proxy.agent_endpoints.agent_registry": MagicMock(
                    global_agent_registry=registry
                )
            },
        ):
            assert gw.resolve_agent("ghost") is None


class TestInvokeUsesResolvedAgent:
    @pytest.mark.asyncio
    async def test_invoke_dispatches_to_resolved_agent(self):
        gw = A2AGateway()
        gw.enabled = True
        gw.register_agent(
            A2AAgent(agent_id="a1", name="A1", description="", url="https://x/a")
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={"jsonrpc": "2.0", "id": 1, "result": {"text": "hi"}}
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(a2a_mod, "get_client_for_request", return_value=mock_client),
            patch.object(
                a2a_mod, "validate_outbound_url_async", new_callable=AsyncMock
            ),
        ):
            req = JSONRPCRequest(
                method="message/send", params={"message": {"text": "hi"}}, id=1
            )
            resp = await gw.invoke_agent("a1", req)

        assert resp.error is None
        assert mock_client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_invoke_unknown_agent_returns_not_found(self):
        gw = A2AGateway()
        gw.enabled = True
        registry = MagicMock()
        registry.get_agent_by_id.return_value = None
        registry.get_agent_by_name.return_value = None
        with patch.dict(
            "sys.modules",
            {
                "litellm.proxy.agent_endpoints.agent_registry": MagicMock(
                    global_agent_registry=registry
                )
            },
        ):
            req = JSONRPCRequest(method="message/send", params={}, id=1)
            resp = await gw.invoke_agent("ghost", req)
        assert resp.error is not None
        assert resp.error["code"] == -32002


# ---------------------------------------------------------------------------
# HTTP route level
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("A2A_GATEWAY_ENABLED", "true")
    reset_a2a_gateway()
    app = create_standalone_app(enable_plugins=False, enable_resilience=False)
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: {"api_key": "test-user-key"}
    yield TestClient(app, raise_server_exceptions=False)
    reset_a2a_gateway()


class TestInvocationRoute:
    def test_invocation_route_exists_and_dispatches(self, client):
        gw = get_a2a_gateway()
        gw.enabled = True
        gw.register_agent(
            A2AAgent(agent_id="a1", name="A1", description="", url="https://x/a")
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={"jsonrpc": "2.0", "id": 1, "result": {"text": "pong"}}
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(a2a_mod, "get_client_for_request", return_value=mock_client),
            patch.object(
                a2a_mod, "validate_outbound_url_async", new_callable=AsyncMock
            ),
        ):
            r = client.post(
                "/a2a/a1",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "message/send",
                    "params": {"message": {"text": "ping"}},
                },
            )

        assert r.status_code == 200
        data = r.json()
        assert data["jsonrpc"] == "2.0"
        assert "result" in data

    def test_invocation_missing_method_is_400(self, client):
        r = client.post("/a2a/a1", json={"jsonrpc": "2.0", "id": 1})
        assert r.status_code == 400

    def test_streaming_alias_returns_sse(self, client):
        gw = get_a2a_gateway()
        gw.enabled = True
        gw.register_agent(
            A2AAgent(
                agent_id="a1",
                name="A1",
                description="",
                url="https://x/a",
                capabilities=["streaming"],
            )
        )

        async def _fake_stream(agent_id, request):
            yield "event: task-status-update\ndata: {}\n\n"

        with patch.object(
            A2AGateway, "stream_agent_response_sse", side_effect=_fake_stream
        ):
            r = client.post(
                "/a2a/a1/message/stream",
                json={"jsonrpc": "2.0", "id": 1, "method": "message/stream"},
            )

        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]

    def test_disabled_gateway_returns_404(self, client):
        gw = get_a2a_gateway()
        gw.enabled = False
        r = client.post(
            "/a2a/a1",
            json={"jsonrpc": "2.0", "id": 1, "method": "message/send"},
        )
        assert r.status_code == 404
