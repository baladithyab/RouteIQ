"""Unit tests for routing read-only queries through the READER pool (RouteIQ-6972).

``get_read_db_pool()`` was built but no read query used it. The repository
read-only SELECTs (``A2AAgentRepository._load_from_db`` /
``MCPServerRepository._load_from_db``) are now routed through it:

* with ``postgres.reader_host`` SET, a read method acquires from the SEPARATE
  reader pool (built against the host-swapped URL), while writes stay on the
  writer pool;
* with ``reader_host`` unset (the default), the read transparently falls back to
  the writer pool -- byte-stable single-pool behaviour.

Credential-free: ``_create_pool`` (the only asyncpg touch) is mocked, so no real
asyncpg / Postgres / credentials are required.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm_llmrouter.database as db
from litellm_llmrouter.database import (
    A2AAgentRepository,
    MCPServerRepository,
    get_db_pool,
    get_read_db_pool,
    reset_db_pool,
)
from litellm_llmrouter.settings import get_settings, reset_settings


_DB_URL = "postgresql://user:pw@writer.example.com:5432/routeiq"


class _FakePool:
    """A pool sentinel that records acquire() and yields a fetchrow=None conn."""

    def __init__(self, role: str):
        self.role = role
        self.acquire_count = 0

    def acquire(self):
        self.acquire_count += 1

        @asynccontextmanager
        async def _cm():
            conn = MagicMock()
            conn.fetchrow = AsyncMock(return_value=None)
            conn.execute = AsyncMock(return_value=None)
            yield conn

        return _cm()


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_db_pool()
    reset_settings()
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    yield
    reset_db_pool()
    reset_settings()


def _install_pool_factory(monkeypatch):
    """Patch _create_pool to hand back a distinct sentinel per role/URL."""
    created: dict[str, _FakePool] = {}

    async def _fake_create_pool(url, *, role="writer"):
        pool = _FakePool(role)
        pool.url = url
        created[role] = pool
        return pool

    monkeypatch.setattr(db, "_create_pool", _fake_create_pool)
    return created


@pytest.mark.asyncio
async def test_read_acquires_from_reader_pool_when_reader_host_set(monkeypatch):
    """reader_host SET -> _load_from_db reads via the reader pool, not the writer."""
    get_settings(postgres={"reader_host": "reader.example.com"})
    created = _install_pool_factory(monkeypatch)

    repo = A2AAgentRepository()
    assert repo._db_url == _DB_URL  # DB path active

    await repo.get("agent-123")  # cache miss -> _load_from_db

    # A separate reader pool was built and acquired from.
    assert "reader" in created
    reader = created["reader"]
    assert reader.acquire_count == 1
    # Built against the host-swapped reader endpoint (creds/port preserved).
    assert "reader.example.com" in reader.url
    # The writer pool was NOT used for the read.
    assert "writer" not in created or created["writer"].acquire_count == 0


@pytest.mark.asyncio
async def test_write_stays_on_writer_pool(monkeypatch):
    """reader_host SET -> a WRITE (_persist) still uses the writer pool."""
    get_settings(postgres={"reader_host": "reader.example.com"})
    created = _install_pool_factory(monkeypatch)

    A2AAgentRepository()
    pool = await get_db_pool(_DB_URL)  # writer pool
    async with pool.acquire() as conn:
        await conn.execute("INSERT ...")

    assert created["writer"].acquire_count == 1
    # No reader pool created by a writer-only path.
    assert "reader" not in created


@pytest.mark.asyncio
async def test_read_falls_back_to_writer_when_no_reader_host(monkeypatch):
    """reader_host UNSET (default) -> reads share the writer pool (byte-stable)."""
    get_settings()  # reader_host defaults to None
    created = _install_pool_factory(monkeypatch)

    repo = MCPServerRepository()
    await repo.get("server-abc")  # cache miss -> _load_from_db

    # No separate reader pool; the read went through the writer pool.
    assert "reader" not in created
    assert "writer" in created
    assert created["writer"].acquire_count == 1
    # The reader getter returns the very same writer pool object.
    assert await get_read_db_pool(_DB_URL) is created["writer"]
