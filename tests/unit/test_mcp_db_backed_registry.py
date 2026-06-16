"""
Unit tests for the DB-backed MCP server registry (RouteIQ-635d).

MCPServerDB / MCPServerRepository existed but mcp_gateway never used them, so
registrations were in-memory per-worker (non-HA). These tests verify the gateway
now backs its registry with the repository:
- register persists to the repository (MCPServerDB)
- a FRESH gateway rehydrates persisted servers via load_persisted_servers()
- graceful fallback to in-memory when no DB is configured
- unregister removes from the repository

The repository is mocked (no live DB).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer, MCPTransport
from litellm_llmrouter.database import MCPServerDB


def _make_gateway(db_enabled: bool) -> MCPGateway:
    gw = MCPGateway()
    gw.enabled = True
    gw._ha_sync_enabled = False
    gw._db_persistence_enabled = db_enabled
    return gw


class TestDBPersistenceToggle:
    def test_disabled_without_database(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("MCP_DB_PERSISTENCE_ENABLED", raising=False)
        gw = MCPGateway()
        assert gw.is_db_persistence_enabled() is False

    def test_enabled_when_database_configured(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
        monkeypatch.delenv("MCP_DB_PERSISTENCE_ENABLED", raising=False)
        gw = MCPGateway()
        assert gw.is_db_persistence_enabled() is True

    def test_env_force_disable(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
        monkeypatch.setenv("MCP_DB_PERSISTENCE_ENABLED", "false")
        gw = MCPGateway()
        assert gw.is_db_persistence_enabled() is False


class TestInMemoryFallback:
    @pytest.mark.asyncio
    async def test_register_works_without_db(self):
        gw = _make_gateway(db_enabled=False)
        gw.register_server(
            MCPServer(server_id="s1", name="srv", url="https://x/rpc", tools=["t"])
        )
        assert gw.get_server("s1") is not None
        # No DB load => no-op returns 0.
        loaded = await gw.load_persisted_servers()
        assert loaded == 0


class TestRegisterPersists:
    @pytest.mark.asyncio
    async def test_register_persists_to_repository(self):
        gw = _make_gateway(db_enabled=True)

        repo = MagicMock()
        repo.get = AsyncMock(return_value=None)
        repo.create = AsyncMock()
        repo.update = AsyncMock()

        with patch("litellm_llmrouter.database.get_mcp_repository", return_value=repo):
            await gw._persist_server_to_db(
                MCPServer(
                    server_id="s1",
                    name="srv",
                    url="https://x/rpc",
                    transport=MCPTransport.SSE,
                    tools=["t1", "t2"],
                )
            )

        repo.create.assert_awaited_once()
        created: MCPServerDB = repo.create.call_args.args[0]
        assert isinstance(created, MCPServerDB)
        assert created.server_id == "s1"
        assert created.transport == "sse"
        assert created.tools == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_register_updates_existing(self):
        gw = _make_gateway(db_enabled=True)
        repo = MagicMock()
        repo.get = AsyncMock(
            return_value=MCPServerDB(server_id="s1", name="o", url="u")
        )
        repo.create = AsyncMock()
        repo.update = AsyncMock()

        with patch("litellm_llmrouter.database.get_mcp_repository", return_value=repo):
            await gw._persist_server_to_db(
                MCPServer(server_id="s1", name="srv", url="https://x/rpc")
            )

        repo.update.assert_awaited_once()
        repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_persist_failure_is_swallowed(self):
        gw = _make_gateway(db_enabled=True)
        repo = MagicMock()
        repo.get = AsyncMock(side_effect=RuntimeError("db down"))
        with patch("litellm_llmrouter.database.get_mcp_repository", return_value=repo):
            # Must not raise — degrades to in-memory.
            await gw._persist_server_to_db(
                MCPServer(server_id="s1", name="srv", url="https://x/rpc")
            )


class TestFreshGatewayRehydrates:
    @pytest.mark.asyncio
    async def test_fresh_gateway_loads_persisted_servers(self):
        # Simulate: worker A registered s1; a FRESH gateway (worker B) loads it.
        persisted = [
            MCPServerDB(
                server_id="s1",
                name="persisted-srv",
                url="https://persisted/rpc",
                transport="sse",
                tools=["alpha", "beta"],
            )
        ]
        repo = MagicMock()
        repo.list_all = AsyncMock(return_value=persisted)

        fresh = _make_gateway(db_enabled=True)
        assert fresh.get_server("s1") is None  # nothing in-memory yet

        with patch("litellm_llmrouter.database.get_mcp_repository", return_value=repo):
            loaded = await fresh.load_persisted_servers()

        assert loaded == 1
        server = fresh.get_server("s1")
        assert server is not None
        assert server.name == "persisted-srv"
        assert server.transport == MCPTransport.SSE
        # Tool->server mapping rehydrated so invocation routing works.
        assert fresh.find_server_for_tool("alpha") is server
        assert fresh.find_server_for_tool("beta") is server

    @pytest.mark.asyncio
    async def test_load_failure_returns_zero(self):
        repo = MagicMock()
        repo.list_all = AsyncMock(side_effect=RuntimeError("db down"))
        fresh = _make_gateway(db_enabled=True)
        with patch("litellm_llmrouter.database.get_mcp_repository", return_value=repo):
            loaded = await fresh.load_persisted_servers()
        assert loaded == 0


class TestUnregisterDeletes:
    @pytest.mark.asyncio
    async def test_unregister_deletes_from_repository(self):
        gw = _make_gateway(db_enabled=True)
        repo = MagicMock()
        repo.delete = AsyncMock(return_value=True)
        with patch("litellm_llmrouter.database.get_mcp_repository", return_value=repo):
            await gw._delete_server_from_db("s1")
        repo.delete.assert_awaited_once_with("s1")
