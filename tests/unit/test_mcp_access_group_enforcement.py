"""Unit tests for per-key/per-tool MCP access-group enforcement (RouteIQ-2fa1).

An access_groups listing endpoint already existed, but there was NO enforcement
at INVOKE time -- any key could invoke any tool regardless of access group. These
tests prove the enforcement on the LIVE invocation path (``invoke_tool``, the real
entrypoint, incl. the terminal path), gated default-OFF so the prior behaviour is
byte-stable:

* enforcement OFF (default) -> access_groups is advisory; any caller invokes.
* enforcement ON + server has access_groups -> a caller presenting a matching
  group invokes; a caller NOT in the group is REJECTED before any outbound call.
* enforcement ON + server has NO access_groups -> unrestricted (open by default).

All invocation is MOCKED at ``_invoke_tool_impl`` so no live MCP/HTTP call is made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from litellm_llmrouter.mcp_gateway import (
    MCPGateway,
    MCPServer,
    MCPToolResult,
    reset_mcp_gateway,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_mcp_gateway()
    yield
    reset_mcp_gateway()


def _make_gateway(*, enforce: bool, access_groups=None) -> MCPGateway:
    """Build a gateway with invocation enabled + one tool whose server is gated.

    ``_invoke_tool_impl`` is replaced with a stub that returns success so the
    allowed path never makes a real outbound call.
    """
    gw = MCPGateway()
    gw.enabled = True
    gw._tool_invocation_enabled = True
    gw._access_group_enforcement = enforce

    meta = {}
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
        return MCPToolResult(success=True, result="ok", tool_name=tool_name)

    gw._invoke_tool_impl = AsyncMock(side_effect=_ok)  # type: ignore[method-assign]
    return gw


class TestEnforcementOff:
    @pytest.mark.asyncio
    async def test_default_off_is_advisory_any_caller_invokes(self):
        """Default OFF: access_groups gated server still invokable by anyone."""
        gw = _make_gateway(enforce=False, access_groups=["team-a"])
        result = await gw.invoke_tool(
            "secret_tool", {}, caller_access_groups=["team-zzz"]
        )
        assert result.success is True
        gw._invoke_tool_impl.assert_awaited_once()


class TestEnforcementOn:
    @pytest.mark.asyncio
    async def test_allowed_group_invokes(self):
        """ON: a caller in the server's access group invokes successfully."""
        gw = _make_gateway(enforce=True, access_groups=["team-a", "team-b"])
        result = await gw.invoke_tool(
            "secret_tool", {"x": 1}, caller_access_groups=["team-b"]
        )
        assert result.success is True
        gw._invoke_tool_impl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disallowed_group_rejected_before_invoke(self):
        """ON: a caller NOT in the group is rejected at the terminal path."""
        gw = _make_gateway(enforce=True, access_groups=["team-a"])
        result = await gw.invoke_tool(
            "secret_tool", {}, caller_access_groups=["team-other"]
        )
        assert result.success is False
        assert result.error is not None and result.error.startswith("access_denied:")
        assert result.server_id == "srv-secure"
        # Rejected BEFORE any outbound MCP call.
        gw._invoke_tool_impl.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_caller_groups_rejected(self):
        """ON: a caller with no access groups is rejected for a gated tool."""
        gw = _make_gateway(enforce=True, access_groups=["team-a"])
        result = await gw.invoke_tool("secret_tool", {}, caller_access_groups=None)
        assert result.success is False
        assert result.error.startswith("access_denied:")
        gw._invoke_tool_impl.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unrestricted_server_open_by_default(self):
        """ON but server declares NO access_groups -> unrestricted (open)."""
        gw = _make_gateway(enforce=True, access_groups=None)
        result = await gw.invoke_tool("secret_tool", {}, caller_access_groups=[])
        assert result.success is True
        gw._invoke_tool_impl.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_csv_access_groups_metadata_parsed(self):
        """access_groups may be a CSV string (not just a list)."""
        gw = _make_gateway(enforce=True, access_groups="team-a, team-b")
        ok = await gw.invoke_tool("secret_tool", {}, caller_access_groups=["team-a"])
        assert ok.success is True

    @pytest.mark.asyncio
    async def test_disabled_invocation_short_circuits_before_access_check(self):
        """The disabled-invocation gate still wins over the access-group gate."""
        gw = _make_gateway(enforce=True, access_groups=["team-a"])
        gw._tool_invocation_enabled = False
        result = await gw.invoke_tool(
            "secret_tool", {}, caller_access_groups=["team-a"]
        )
        assert result.success is False
        assert result.error.startswith("tool_invocation_disabled:")


class TestSettingsResolution:
    def test_resolver_reads_typed_setting(self, monkeypatch):
        """The enforcement flag resolves from MCPSettings.access_group_enforcement."""
        from litellm_llmrouter import settings as settings_mod

        monkeypatch.setenv("ROUTEIQ_MCP__ACCESS_GROUP_ENFORCEMENT", "true")
        settings_mod.reset_settings()
        try:
            gw = MCPGateway()
            assert gw.is_access_group_enforcement_enabled() is True
        finally:
            settings_mod.reset_settings()
