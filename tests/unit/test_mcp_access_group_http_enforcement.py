"""HTTP-entrypoint enforcement test for MCP access groups (RouteIQ-2fa1).

The access-group enforcement seam lives on ``MCPGateway.invoke_tool`` (unit-tested
in ``test_mcp_access_group_enforcement.py``), but it only gates a LIVE invocation
if the REST route at ``POST /llmrouter/mcp/tools/call`` actually plumbs the authed
caller's access groups into it. The route is admin-gated at the router level, so
the caller's access groups are resolved from the key's governance record
(``KeyGovernance.metadata['mcp_access_groups']``). These tests drive the real
FastAPI route via ``TestClient`` and prove:

* a key whose governance record LACKS the tool's access group is REJECTED (403)
  at the HTTP entrypoint when enforcement is ON;
* a key whose governance record HAS the matching group is allowed (200);
* enforcement OFF (default) is byte-stable -- the lacking-group key still invokes.

The outbound MCP call is mocked at ``_invoke_tool_impl`` so no network I/O happens.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import litellm_llmrouter.mcp_gateway as mcp_gw_mod
from litellm_llmrouter.gateway.app import create_standalone_app
from litellm_llmrouter.governance import (
    KeyGovernance,
    get_governance_engine,
    reset_governance_engine,
)
from litellm_llmrouter.mcp_gateway import (
    MCPGateway,
    MCPServer,
    MCPToolResult,
    reset_mcp_gateway,
)


_TEST_ADMIN_KEY = "sk-riq-test-admin-mcp-ag"


@pytest.fixture(autouse=True)
def _reset():
    reset_mcp_gateway()
    reset_governance_engine()
    yield
    reset_mcp_gateway()
    reset_governance_engine()
    mcp_gw_mod._mcp_gateway = None


def _install_gateway(*, enforce: bool, access_groups=None) -> MCPGateway:
    """Install a configured MCP gateway (one access-group-gated tool) as singleton."""
    gw = MCPGateway()
    gw.enabled = True
    gw._tool_invocation_enabled = True
    gw._access_group_enforcement = enforce

    meta: dict = {}
    if access_groups is not None:
        meta["access_groups"] = access_groups
    gw.register_server(
        MCPServer(
            server_id="srv-secure",
            name="secure",
            url="https://example.com/rpc",
            tools=["secret_tool"],
            metadata=meta,
        )
    )

    async def _ok(tool_name, arguments):
        return MCPToolResult(
            success=True, result="ok", tool_name=tool_name, server_id="srv-secure"
        )

    gw._invoke_tool_impl = AsyncMock(side_effect=_ok)  # type: ignore[method-assign]
    mcp_gw_mod._mcp_gateway = gw
    return gw


def _register_key_groups(groups):
    """Attach ``mcp_access_groups`` to the admin key's governance record."""
    if groups is None:
        return
    get_governance_engine().register_key_governance(
        KeyGovernance(
            key_id=_TEST_ADMIN_KEY,
            metadata={"mcp_access_groups": groups},
        )
    )


def _admin_client(monkeypatch):
    """TestClient that authenticates as admin via X-Admin-API-Key."""
    monkeypatch.setenv("ADMIN_AUTH_ENABLED", "true")
    monkeypatch.setenv("ADMIN_API_KEYS", _TEST_ADMIN_KEY)
    monkeypatch.setenv("ROUTEIQ_ENV", "test")

    app = create_standalone_app(enable_plugins=False, enable_resilience=False)
    c = TestClient(app, raise_server_exceptions=False)
    _orig = c.request

    def _authed(*args, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("X-Admin-API-Key", _TEST_ADMIN_KEY)
        return _orig(*args, headers=headers, **kwargs)

    c.request = _authed
    return c


_CALL_PATH = "/llmrouter/mcp/tools/call"
_BODY = {"tool_name": "secret_tool", "arguments": {"x": 1}}


class TestHTTPEnforcementOn:
    def test_key_lacking_group_rejected_403(self, monkeypatch):
        """The core SECURITY assertion: key without the group -> 403."""
        _install_gateway(enforce=True, access_groups=["team-a"])
        _register_key_groups(["team-other"])
        client = _admin_client(monkeypatch)
        resp = client.post(_CALL_PATH, json=_BODY)
        assert resp.status_code == 403, resp.text
        assert resp.json()["detail"]["error"] == "access_denied"

    def test_key_with_no_governance_groups_rejected_403(self, monkeypatch):
        """No governance groups declared -> non-member -> 403."""
        _install_gateway(enforce=True, access_groups=["team-a"])
        # No KeyGovernance record at all for the admin key.
        client = _admin_client(monkeypatch)
        resp = client.post(_CALL_PATH, json=_BODY)
        assert resp.status_code == 403, resp.text
        assert resp.json()["detail"]["error"] == "access_denied"

    def test_key_with_matching_group_allowed_200(self, monkeypatch):
        """A member key invokes successfully (200)."""
        gw = _install_gateway(enforce=True, access_groups=["team-a", "team-b"])
        _register_key_groups(["team-b"])
        client = _admin_client(monkeypatch)
        resp = client.post(_CALL_PATH, json=_BODY)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "success"
        gw._invoke_tool_impl.assert_awaited_once()

    def test_unrestricted_tool_allowed_even_without_groups(self, monkeypatch):
        """Server with NO access_groups is open even under enforcement."""
        _install_gateway(enforce=True, access_groups=None)
        client = _admin_client(monkeypatch)
        resp = client.post(_CALL_PATH, json=_BODY)
        assert resp.status_code == 200, resp.text


class TestHTTPEnforcementOff:
    def test_off_is_byte_stable_lacking_group_still_invokes(self, monkeypatch):
        """Enforcement OFF (default) -> the lacking-group key still invokes."""
        _install_gateway(enforce=False, access_groups=["team-a"])
        _register_key_groups(["team-other"])
        client = _admin_client(monkeypatch)
        resp = client.post(_CALL_PATH, json=_BODY)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "success"


def test_access_group_enforced_through_tracing_wrapper(monkeypatch):
    """RouteIQ-2fa1 regression: with MCP tracing instrumented (production default)
    AND access-group enforcement ON, the traced_invoke_tool wrapper MUST forward
    caller_access_groups so the enforcement seam runs (and denies a non-member)
    BEFORE any network — instead of raising TypeError (the original blocker)."""
    import asyncio
    import litellm_llmrouter.mcp_gateway as mg
    from litellm_llmrouter.mcp_gateway import MCPServer, MCPTransport, reset_mcp_gateway
    from litellm_llmrouter.settings import reset_settings
    from litellm_llmrouter import mcp_tracing

    monkeypatch.setenv("MCP_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("LLMROUTER_ENABLE_MCP_TOOL_INVOCATION", "true")
    monkeypatch.setenv("ROUTEIQ_MCP__ACCESS_GROUP_ENFORCEMENT", "true")
    reset_settings()
    reset_mcp_gateway()
    gw = mg.get_mcp_gateway()
    gw.register_server(
        MCPServer(
            server_id="s1",
            name="restricted",
            url="https://example.test/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
            tools=["t1"],
            metadata={"access_groups": ["team-a"]},
        )
    )
    mcp_tracing.instrument_mcp_gateway()  # production default: wraps invoke_tool

    # Non-member: denied by the seam BEFORE network (no TypeError, no DNS attempt).
    res = asyncio.run(gw.invoke_tool("t1", {}, caller_access_groups=["team-b"]))
    assert res.success is False
    assert res.error and "access" in res.error.lower(), res.error
    # Member: passes the access seam (then fails on DNS, proving the kwarg flowed
    # through the wrapper and the seam ALLOWED the member).
    res2 = asyncio.run(gw.invoke_tool("t1", {}, caller_access_groups=["team-a"]))
    assert res2.success is False
    assert "access" not in (res2.error or "").lower()
    reset_settings()
    reset_mcp_gateway()
