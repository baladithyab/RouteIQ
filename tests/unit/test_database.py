"""
Unit Tests for database.py
===========================

Tests for A2A and MCP persistence layer:
- A2AAgentDB dataclass (from_dict, to_dict, defaults)
- A2AAgentRepository (CRUD, filtering, patch, make_public)
- A2AAgentActivity dataclass (avg_latency, to_dict)
- A2AActivityTracker (record_invocation, daily/aggregated activity, date filtering)
- MCPServerDB dataclass (from_dict, to_dict, defaults)
- MCPServerRepository (CRUD, filtering)
- Singleton accessors (get_a2a_repository, get_mcp_repository, get_a2a_activity_tracker)
- Database config helpers (get_database_url, is_database_configured)

All tests run against in-memory storage (no DATABASE_URL set).
"""

import os
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest

# Ensure no DATABASE_URL leaks into tests
os.environ.pop("DATABASE_URL", None)

from litellm_llmrouter.database import (
    A2AAgentActivity,
    A2AAgentDB,
    A2AAgentRepository,
    A2AActivityTracker,
    MCPServerDB,
    MCPServerRepository,
    get_a2a_activity_tracker,
    get_a2a_repository,
    get_database_url,
    get_mcp_repository,
    is_database_configured,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all database singletons between tests."""
    import litellm_llmrouter.database as db_mod

    db_mod._a2a_repository = None
    db_mod._a2a_activity_tracker = None
    db_mod._mcp_repository = None
    yield
    db_mod._a2a_repository = None
    db_mod._a2a_activity_tracker = None
    db_mod._mcp_repository = None


@pytest.fixture
def a2a_repo():
    """Fresh A2AAgentRepository instance."""
    return A2AAgentRepository()


@pytest.fixture
def mcp_repo():
    """Fresh MCPServerRepository instance."""
    return MCPServerRepository()


@pytest.fixture
def activity_tracker():
    """Fresh A2AActivityTracker instance."""
    return A2AActivityTracker()


# =============================================================================
# Database Configuration
# =============================================================================


class TestDatabaseConfig:
    def test_get_database_url_returns_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            # DATABASE_URL may or may not be set; override explicitly
            os.environ.pop("DATABASE_URL", None)
            assert get_database_url() is None

    def test_get_database_url_returns_value(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            assert get_database_url() == "postgresql://localhost/test"

    def test_is_database_configured_false(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DATABASE_URL", None)
            assert is_database_configured() is False

    def test_is_database_configured_true(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            assert is_database_configured() is True


# =============================================================================
# A2AAgentDB Dataclass
# =============================================================================


class TestA2AAgentDB:
    def test_from_dict_minimal(self):
        data = {"name": "test-agent", "url": "http://localhost:8080"}
        agent = A2AAgentDB.from_dict(data)
        assert agent.name == "test-agent"
        assert agent.url == "http://localhost:8080"
        assert agent.description == ""
        assert agent.capabilities == []
        assert agent.metadata == {}
        assert agent.team_id is None
        assert agent.user_id is None
        assert agent.is_public is False
        # agent_id should be auto-generated UUID
        assert len(agent.agent_id) > 0

    def test_from_dict_full(self):
        now = datetime.now(timezone.utc)
        data = {
            "agent_id": "agent-123",
            "name": "full-agent",
            "description": "A full agent",
            "url": "http://agent.example.com",
            "capabilities": ["chat", "code"],
            "metadata": {"version": "1.0"},
            "team_id": "team-1",
            "user_id": "user-1",
            "is_public": True,
            "created_at": now,
            "updated_at": now,
        }
        agent = A2AAgentDB.from_dict(data)
        assert agent.agent_id == "agent-123"
        assert agent.name == "full-agent"
        assert agent.capabilities == ["chat", "code"]
        assert agent.is_public is True
        assert agent.created_at == now

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        agent = A2AAgentDB(
            agent_id="a1",
            name="test",
            description="desc",
            url="http://localhost",
            capabilities=["cap1"],
            metadata={"k": "v"},
            team_id="t1",
            user_id="u1",
            is_public=True,
            created_at=now,
            updated_at=now,
        )
        d = agent.to_dict()
        assert d["agent_id"] == "a1"
        assert d["name"] == "test"
        assert d["capabilities"] == ["cap1"]
        assert d["is_public"] is True
        assert d["created_at"] == now.isoformat()
        assert d["updated_at"] == now.isoformat()

    def test_to_dict_none_timestamps(self):
        agent = A2AAgentDB(
            agent_id="a1", name="test", description="", url="http://localhost"
        )
        d = agent.to_dict()
        assert d["created_at"] is None
        assert d["updated_at"] is None

    def test_default_field_factories(self):
        agent = A2AAgentDB(
            agent_id="a1", name="test", description="", url="http://localhost"
        )
        assert agent.capabilities == []
        assert agent.metadata == {}
        # Verify they're independent instances
        agent2 = A2AAgentDB(
            agent_id="a2", name="test2", description="", url="http://localhost"
        )
        agent.capabilities.append("x")
        assert agent2.capabilities == []


# =============================================================================
# A2AAgentRepository
# =============================================================================


class TestA2AAgentRepository:
    async def test_create_and_get(self, a2a_repo):
        agent = A2AAgentDB(
            agent_id="agent-1",
            name="Test Agent",
            description="A test agent",
            url="http://localhost:8080",
        )
        created = await a2a_repo.create(agent)
        assert created.agent_id == "agent-1"
        assert created.created_at is not None
        assert created.updated_at is not None

        fetched = await a2a_repo.get("agent-1")
        assert fetched is not None
        assert fetched.name == "Test Agent"

    async def test_get_nonexistent(self, a2a_repo):
        result = await a2a_repo.get("nonexistent")
        assert result is None

    async def test_list_all_no_filters(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(agent_id="a1", name="Agent1", description="", url="http://a1")
        )
        await a2a_repo.create(
            A2AAgentDB(agent_id="a2", name="Agent2", description="", url="http://a2")
        )
        agents = await a2a_repo.list_all()
        assert len(agents) == 2

    async def test_list_all_filter_by_user_id(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(
                agent_id="a1",
                name="Agent1",
                description="",
                url="http://a1",
                user_id="user-1",
            )
        )
        await a2a_repo.create(
            A2AAgentDB(
                agent_id="a2",
                name="Agent2",
                description="",
                url="http://a2",
                user_id="user-2",
            )
        )
        agents = await a2a_repo.list_all(user_id="user-1", include_public=False)
        assert len(agents) == 1
        assert agents[0].agent_id == "a1"

    async def test_list_all_filter_by_team_id(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(
                agent_id="a1",
                name="Agent1",
                description="",
                url="http://a1",
                team_id="team-1",
            )
        )
        await a2a_repo.create(
            A2AAgentDB(
                agent_id="a2",
                name="Agent2",
                description="",
                url="http://a2",
                team_id="team-2",
            )
        )
        agents = await a2a_repo.list_all(team_id="team-1", include_public=False)
        assert len(agents) == 1
        assert agents[0].agent_id == "a1"

    async def test_list_all_includes_public(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(
                agent_id="a1",
                name="Public",
                description="",
                url="http://a1",
                is_public=True,
            )
        )
        await a2a_repo.create(
            A2AAgentDB(
                agent_id="a2",
                name="Private",
                description="",
                url="http://a2",
                is_public=False,
                user_id="user-x",
            )
        )
        # Filter by user "user-y" but include_public=True
        agents = await a2a_repo.list_all(user_id="user-y", include_public=True)
        # Should include the public agent
        ids = [a.agent_id for a in agents]
        assert "a1" in ids
        assert "a2" not in ids

    async def test_list_all_excludes_public(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(
                agent_id="a1",
                name="Public",
                description="",
                url="http://a1",
                is_public=True,
            )
        )
        agents = await a2a_repo.list_all(user_id="someone", include_public=False)
        assert len(agents) == 0

    async def test_update_existing(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(agent_id="a1", name="Original", description="", url="http://a1")
        )
        updated_agent = A2AAgentDB(
            agent_id="a1", name="Updated", description="new desc", url="http://a1-new"
        )
        result = await a2a_repo.update("a1", updated_agent)
        assert result is not None
        assert result.name == "Updated"
        assert result.url == "http://a1-new"
        # created_at should be preserved from original
        assert result.created_at is not None

    async def test_update_nonexistent(self, a2a_repo):
        updated_agent = A2AAgentDB(
            agent_id="nonexistent", name="X", description="", url="http://x"
        )
        result = await a2a_repo.update("nonexistent", updated_agent)
        assert result is None

    async def test_patch_existing(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(
                agent_id="a1", name="Original", description="old", url="http://a1"
            )
        )
        result = await a2a_repo.patch("a1", {"name": "Patched", "description": "new"})
        assert result is not None
        assert result.name == "Patched"
        assert result.description == "new"
        assert result.url == "http://a1"  # Unchanged

    async def test_patch_ignores_protected_fields(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(agent_id="a1", name="Original", description="", url="http://a1")
        )
        original_created = (await a2a_repo.get("a1")).created_at
        result = await a2a_repo.patch("a1", {"agent_id": "hacked", "created_at": None})
        assert result is not None
        assert result.agent_id == "a1"  # Not changed
        assert result.created_at == original_created  # Not changed

    async def test_patch_nonexistent(self, a2a_repo):
        result = await a2a_repo.patch("nonexistent", {"name": "X"})
        assert result is None

    async def test_delete_existing(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(agent_id="a1", name="Agent", description="", url="http://a1")
        )
        assert await a2a_repo.delete("a1") is True
        assert await a2a_repo.get("a1") is None

    async def test_delete_nonexistent(self, a2a_repo):
        assert await a2a_repo.delete("nonexistent") is False

    async def test_make_public(self, a2a_repo):
        await a2a_repo.create(
            A2AAgentDB(
                agent_id="a1",
                name="Agent",
                description="",
                url="http://a1",
                is_public=False,
            )
        )
        result = await a2a_repo.make_public("a1")
        assert result is not None
        assert result.is_public is True


# =============================================================================
# A2AAgentActivity Dataclass
# =============================================================================


class TestA2AAgentActivity:
    def test_avg_latency_zero_invocations(self):
        activity = A2AAgentActivity(agent_id="a1", invocation_date=date.today())
        assert activity.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        activity = A2AAgentActivity(
            agent_id="a1",
            invocation_date=date.today(),
            invocation_count=4,
            total_latency_ms=1000,
        )
        assert activity.avg_latency_ms == 250.0

    def test_to_dict(self):
        today = date.today()
        activity = A2AAgentActivity(
            agent_id="a1",
            invocation_date=today,
            invocation_count=10,
            total_latency_ms=5000,
            success_count=8,
            error_count=2,
        )
        d = activity.to_dict()
        assert d["agent_id"] == "a1"
        assert d["date"] == today.isoformat()
        assert d["invocation_count"] == 10
        assert d["avg_latency_ms"] == 500.0
        assert d["success_count"] == 8
        assert d["error_count"] == 2


# =============================================================================
# A2AActivityTracker
# =============================================================================


class TestA2AActivityTracker:
    async def test_record_invocation_success(self, activity_tracker):
        await activity_tracker.record_invocation("a1", latency_ms=100, success=True)
        activities = await activity_tracker.get_daily_activity(agent_id="a1")
        assert len(activities) == 1
        assert activities[0].invocation_count == 1
        assert activities[0].success_count == 1
        assert activities[0].error_count == 0
        assert activities[0].total_latency_ms == 100

    async def test_record_invocation_failure(self, activity_tracker):
        await activity_tracker.record_invocation("a1", latency_ms=50, success=False)
        activities = await activity_tracker.get_daily_activity(agent_id="a1")
        assert len(activities) == 1
        assert activities[0].error_count == 1
        assert activities[0].success_count == 0

    async def test_record_multiple_invocations_aggregate(self, activity_tracker):
        await activity_tracker.record_invocation("a1", latency_ms=100, success=True)
        await activity_tracker.record_invocation("a1", latency_ms=200, success=True)
        await activity_tracker.record_invocation("a1", latency_ms=300, success=False)
        activities = await activity_tracker.get_daily_activity(agent_id="a1")
        assert len(activities) == 1
        assert activities[0].invocation_count == 3
        assert activities[0].total_latency_ms == 600
        assert activities[0].success_count == 2
        assert activities[0].error_count == 1

    async def test_get_daily_activity_filter_by_agent(self, activity_tracker):
        await activity_tracker.record_invocation("a1", latency_ms=100)
        await activity_tracker.record_invocation("a2", latency_ms=200)
        a1_activities = await activity_tracker.get_daily_activity(agent_id="a1")
        assert len(a1_activities) == 1
        assert a1_activities[0].agent_id == "a1"

    async def test_get_daily_activity_filter_by_date_range(self, activity_tracker):
        # Manually insert activity for a specific past date
        past_date = date(2024, 1, 15)
        today = date.today()
        activity_tracker._activity[("a1", past_date)] = A2AAgentActivity(
            agent_id="a1",
            invocation_date=past_date,
            invocation_count=5,
            total_latency_ms=500,
            success_count=5,
        )
        await activity_tracker.record_invocation("a1", latency_ms=100)

        # Filter to only today
        activities = await activity_tracker.get_daily_activity(
            start_date=today, end_date=today
        )
        assert len(activities) == 1
        assert activities[0].invocation_date == today

        # Filter to only past date
        activities = await activity_tracker.get_daily_activity(
            start_date=past_date, end_date=past_date
        )
        assert len(activities) == 1
        assert activities[0].invocation_date == past_date

    async def test_get_daily_activity_sorted_descending(self, activity_tracker):
        d1 = date(2024, 1, 10)
        d2 = date(2024, 1, 20)
        activity_tracker._activity[("a1", d1)] = A2AAgentActivity(
            agent_id="a1", invocation_date=d1, invocation_count=1
        )
        activity_tracker._activity[("a1", d2)] = A2AAgentActivity(
            agent_id="a1", invocation_date=d2, invocation_count=2
        )
        activities = await activity_tracker.get_daily_activity()
        assert activities[0].invocation_date == d2  # Most recent first

    async def test_get_aggregated_activity_empty(self, activity_tracker):
        result = await activity_tracker.get_aggregated_activity()
        assert result["total_invocations"] == 0
        assert result["total_success"] == 0
        assert result["total_errors"] == 0
        assert result["avg_latency_ms"] == 0.0
        assert result["unique_agents"] == 0

    async def test_get_aggregated_activity_with_data(self, activity_tracker):
        await activity_tracker.record_invocation("a1", latency_ms=100, success=True)
        await activity_tracker.record_invocation("a1", latency_ms=200, success=True)
        await activity_tracker.record_invocation("a2", latency_ms=300, success=False)

        result = await activity_tracker.get_aggregated_activity()
        assert result["total_invocations"] == 3
        assert result["total_success"] == 2
        assert result["total_errors"] == 1
        assert result["avg_latency_ms"] == 200.0  # 600 / 3
        assert result["unique_agents"] == 2

    async def test_get_aggregated_activity_date_range(self, activity_tracker):
        today = date.today()
        result = await activity_tracker.get_aggregated_activity(
            start_date=today, end_date=today
        )
        # No data for today yet
        assert result["total_invocations"] == 0
        assert result["date_range"]["start"] == today.isoformat()


# =============================================================================
# MCPServerDB Dataclass
# =============================================================================


class TestMCPServerDB:
    def test_from_dict_minimal(self):
        data = {"name": "test-server", "url": "http://mcp.example.com"}
        server = MCPServerDB.from_dict(data)
        assert server.name == "test-server"
        assert server.url == "http://mcp.example.com"
        assert server.transport == "streamable_http"
        assert server.tools == []
        assert server.resources == []
        assert server.auth_type == "none"
        assert server.metadata == {}
        assert server.is_public is False

    def test_from_dict_full(self):
        data = {
            "server_id": "srv-1",
            "name": "full-server",
            "url": "http://mcp.example.com",
            "transport": "sse",
            "tools": ["tool1", "tool2"],
            "resources": ["res1"],
            "auth_type": "bearer",
            "metadata": {"env": "prod"},
            "team_id": "t1",
            "user_id": "u1",
            "is_public": True,
        }
        server = MCPServerDB.from_dict(data)
        assert server.server_id == "srv-1"
        assert server.transport == "sse"
        assert server.tools == ["tool1", "tool2"]
        assert server.auth_type == "bearer"
        assert server.is_public is True

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        server = MCPServerDB(
            server_id="s1",
            name="test",
            url="http://localhost",
            transport="sse",
            tools=["t1"],
            resources=["r1"],
            auth_type="bearer",
            metadata={"k": "v"},
            team_id="t1",
            user_id="u1",
            is_public=True,
            created_at=now,
            updated_at=now,
        )
        d = server.to_dict()
        assert d["server_id"] == "s1"
        assert d["transport"] == "sse"
        assert d["tools"] == ["t1"]
        assert d["auth_type"] == "bearer"
        assert d["created_at"] == now.isoformat()

    def test_to_dict_none_timestamps(self):
        server = MCPServerDB(server_id="s1", name="test", url="http://localhost")
        d = server.to_dict()
        assert d["created_at"] is None
        assert d["updated_at"] is None


# =============================================================================
# MCPServerRepository
# =============================================================================


class TestMCPServerRepository:
    async def test_create_and_get(self, mcp_repo):
        server = MCPServerDB(
            server_id="srv-1",
            name="Test Server",
            url="http://localhost:9000",
        )
        created = await mcp_repo.create(server)
        assert created.server_id == "srv-1"
        assert created.created_at is not None
        assert created.updated_at is not None

        fetched = await mcp_repo.get("srv-1")
        assert fetched is not None
        assert fetched.name == "Test Server"

    async def test_get_nonexistent(self, mcp_repo):
        result = await mcp_repo.get("nonexistent")
        assert result is None

    async def test_list_all_no_filters(self, mcp_repo):
        await mcp_repo.create(
            MCPServerDB(server_id="s1", name="Server1", url="http://s1")
        )
        await mcp_repo.create(
            MCPServerDB(server_id="s2", name="Server2", url="http://s2")
        )
        servers = await mcp_repo.list_all()
        assert len(servers) == 2

    async def test_list_all_filter_by_user_id(self, mcp_repo):
        await mcp_repo.create(
            MCPServerDB(server_id="s1", name="S1", url="http://s1", user_id="user-1")
        )
        await mcp_repo.create(
            MCPServerDB(server_id="s2", name="S2", url="http://s2", user_id="user-2")
        )
        servers = await mcp_repo.list_all(user_id="user-1", include_public=False)
        assert len(servers) == 1
        assert servers[0].server_id == "s1"

    async def test_list_all_filter_by_team_id(self, mcp_repo):
        await mcp_repo.create(
            MCPServerDB(server_id="s1", name="S1", url="http://s1", team_id="team-a")
        )
        await mcp_repo.create(
            MCPServerDB(server_id="s2", name="S2", url="http://s2", team_id="team-b")
        )
        servers = await mcp_repo.list_all(team_id="team-a", include_public=False)
        assert len(servers) == 1
        assert servers[0].server_id == "s1"

    async def test_list_all_includes_public(self, mcp_repo):
        await mcp_repo.create(
            MCPServerDB(server_id="s1", name="Public", url="http://s1", is_public=True)
        )
        await mcp_repo.create(
            MCPServerDB(
                server_id="s2",
                name="Private",
                url="http://s2",
                is_public=False,
                user_id="user-x",
            )
        )
        servers = await mcp_repo.list_all(user_id="other", include_public=True)
        ids = [s.server_id for s in servers]
        assert "s1" in ids
        assert "s2" not in ids

    async def test_update_existing(self, mcp_repo):
        await mcp_repo.create(
            MCPServerDB(server_id="s1", name="Original", url="http://s1")
        )
        updated = MCPServerDB(server_id="s1", name="Updated", url="http://s1-new")
        result = await mcp_repo.update("s1", updated)
        assert result is not None
        assert result.name == "Updated"
        assert result.url == "http://s1-new"
        # created_at preserved
        assert result.created_at is not None

    async def test_update_nonexistent(self, mcp_repo):
        server = MCPServerDB(server_id="none", name="X", url="http://x")
        result = await mcp_repo.update("none", server)
        assert result is None

    async def test_delete_existing(self, mcp_repo):
        await mcp_repo.create(
            MCPServerDB(server_id="s1", name="Server", url="http://s1")
        )
        assert await mcp_repo.delete("s1") is True
        assert await mcp_repo.get("s1") is None

    async def test_delete_nonexistent(self, mcp_repo):
        assert await mcp_repo.delete("nonexistent") is False


# =============================================================================
# Singleton Accessors
# =============================================================================


class TestSingletons:
    def test_get_a2a_repository_returns_same_instance(self):
        repo1 = get_a2a_repository()
        repo2 = get_a2a_repository()
        assert repo1 is repo2

    def test_get_mcp_repository_returns_same_instance(self):
        repo1 = get_mcp_repository()
        repo2 = get_mcp_repository()
        assert repo1 is repo2

    def test_get_a2a_activity_tracker_returns_same_instance(self):
        tracker1 = get_a2a_activity_tracker()
        tracker2 = get_a2a_activity_tracker()
        assert tracker1 is tracker2

    def test_a2a_repository_is_correct_type(self):
        assert isinstance(get_a2a_repository(), A2AAgentRepository)

    def test_mcp_repository_is_correct_type(self):
        assert isinstance(get_mcp_repository(), MCPServerRepository)

    def test_activity_tracker_is_correct_type(self):
        assert isinstance(get_a2a_activity_tracker(), A2AActivityTracker)


# =============================================================================
# IAM-auth runtime (ADR-0028) -- rds-db:connect token as a callable password
# =============================================================================
#
# boto3 + asyncpg are NOT importable in the dev test env, so both are mocked:
# asyncpg via sys.modules (the in-function `import asyncpg`), and _mint_db_token
# is monkeypatch-stubbed (so boto3 is never imported). All cred-free. The pool
# singleton (_pool) is reset by the global autouse conftest fixture
# (reset_database_singletons -> reset_db_pool) + reset_settings.

import sys  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from litellm_llmrouter.database import (  # noqa: E402
    IamRegionUnresolvedError,
    _region_from_rds_host,
    _resolve_db_iam_region,
    get_db_pool,
    run_migrations,
)
from litellm_llmrouter.settings import reset_settings  # noqa: E402

# Placeholder only -- NEVER a real token value (RouteIQ-d3a4 discipline).
_FAKE_DB_TOKEN = "TOKEN-15MIN"  # noqa: S105 - test placeholder, not a secret
_RDS_URL = "postgresql://routeiq@db.cluster-xyz.us-east-1.rds.amazonaws.com:5432/litellm?sslmode=require"
# Serverless Aurora endpoint whose host carries NO parseable region label.
_SERVERLESS_RDS_URL = (
    "postgresql://routeiq@my-aurora-sl.cluster-abc.serverless:5432/litellm"
)


def _mock_asyncpg():
    """A mock asyncpg module whose create_pool is an AsyncMock."""
    mod = MagicMock()
    mod.create_pool = AsyncMock(return_value=MagicMock(name="pool"))
    return mod


class TestRegionFromRdsHost:
    def test_parses_region_from_rds_host(self):
        host = "db.cluster-xyz.us-east-1.rds.amazonaws.com"
        assert _region_from_rds_host(host) == "us-east-1"

    def test_non_rds_host_returns_none(self):
        assert _region_from_rds_host("localhost") is None
        assert _region_from_rds_host(None) is None


class TestGetDbPoolIamAuth:
    async def test_iam_auth_passes_callable_password(self, monkeypatch):
        """ROUTEIQ_DB_IAM_AUTH=true -> create_pool gets a callable password."""
        import litellm_llmrouter.database as db_mod

        monkeypatch.setattr(db_mod, "_mint_db_token", lambda **kw: _FAKE_DB_TOKEN)
        mock_asyncpg = _mock_asyncpg()
        env = {
            "DATABASE_URL": _RDS_URL,
            "ROUTEIQ_DB_IAM_AUTH": "true",
            "AWS_REGION": "us-east-1",
        }
        with patch.dict(os.environ, env, clear=True):
            # Rebuild settings under THIS env (the conftest only resets in
            # teardown, so the first test in a run would otherwise see a leaked
            # iam_auth=False singleton).
            reset_settings()
            with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
                await get_db_pool()

        kwargs = mock_asyncpg.create_pool.call_args.kwargs
        assert "password" in kwargs, kwargs
        pw = kwargs["password"]
        assert callable(pw)
        # asyncpg invokes it per connection -> proves 15-min refresh-per-call.
        assert pw() == _FAKE_DB_TOKEN
        assert pw() == _FAKE_DB_TOKEN

    async def test_iam_auth_mint_args(self, monkeypatch):
        """The callable invokes _mint_db_token with host/port/user/region parsed."""
        import litellm_llmrouter.database as db_mod

        spy = MagicMock(return_value=_FAKE_DB_TOKEN)
        monkeypatch.setattr(db_mod, "_mint_db_token", spy)
        mock_asyncpg = _mock_asyncpg()
        env = {
            "DATABASE_URL": _RDS_URL,
            "ROUTEIQ_DB_IAM_AUTH": "true",
            "AWS_REGION": "us-east-1",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_settings()
            with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
                await get_db_pool()

        pw = mock_asyncpg.create_pool.call_args.kwargs["password"]
        pw()  # trigger the mint
        spy.assert_called_with(
            host="db.cluster-xyz.us-east-1.rds.amazonaws.com",
            port=5432,
            user="routeiq",
            region="us-east-1",
        )

    async def test_static_path_no_token_when_flag_off(self, monkeypatch):
        """Flag OFF (default) -> NO password kwarg; _mint_db_token never called."""
        import litellm_llmrouter.database as db_mod

        called = {"mint": False}

        def _should_not_mint(**kw):
            called["mint"] = True
            return _FAKE_DB_TOKEN

        monkeypatch.setattr(db_mod, "_mint_db_token", _should_not_mint)
        mock_asyncpg = _mock_asyncpg()
        env = {"DATABASE_URL": _RDS_URL}  # ROUTEIQ_DB_IAM_AUTH unset -> OFF
        with patch.dict(os.environ, env, clear=True):
            reset_settings()
            with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
                await get_db_pool()

        kwargs = mock_asyncpg.create_pool.call_args.kwargs
        assert "password" not in kwargs, kwargs
        assert called["mint"] is False

    async def test_iam_setup_failure_falls_back_to_static(self, monkeypatch):
        """A settings-read failure degrades softly: pool built, no password kwarg."""
        import litellm_llmrouter.database as db_mod

        def _boom():
            raise RuntimeError("settings exploded")

        # Force the IAM-setup block to raise BEFORE the callable is built.
        monkeypatch.setattr(db_mod, "get_database_url", get_database_url)
        import litellm_llmrouter.settings as settings_mod

        monkeypatch.setattr(settings_mod, "get_settings", _boom)
        mock_asyncpg = _mock_asyncpg()
        env = {"DATABASE_URL": _RDS_URL, "ROUTEIQ_DB_IAM_AUTH": "true"}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
                pool = await get_db_pool()

        assert pool is not None  # still built (fail-soft)
        kwargs = mock_asyncpg.create_pool.call_args.kwargs
        assert "password" not in kwargs, kwargs


# =============================================================================
# RouteIQ-89a6 / RouteIQ-6829 -- DB region resolution + fail-loud on unresolved
# =============================================================================
#
# When neither settings.postgres.iam_region nor AWS_REGION is set and the RDS
# host carries no parseable region, generate_db_auth_token(Region=None) would
# sign an invalid SigV4 token that fails AUTH at connect time. The rds-db mint
# path must FAIL LOUD instead (raise naming the missing env).


class TestResolveDbIamRegion:
    """RouteIQ-89a6/6829: region resolution order + fail-loud on unresolved."""

    def test_unresolved_region_raises_fail_loud(self):
        """No iam_region arg, no parseable host region, no AWS_REGION -> raise."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(IamRegionUnresolvedError) as exc_info:
                _resolve_db_iam_region("my-aurora-sl.cluster-abc.serverless", None)
        msg = str(exc_info.value)
        assert "ROUTEIQ_POSTGRES__IAM_REGION" in msg
        assert "AWS_REGION" in msg

    def test_resolves_from_iam_region_arg(self):
        """The explicit iam_region arg takes precedence over host + AWS_REGION."""
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}, clear=True):
            assert (
                _resolve_db_iam_region(
                    "db.cluster-xyz.eu-west-1.rds.amazonaws.com", "ca-central-1"
                )
                == "ca-central-1"
            )

    def test_resolves_from_rds_host_label(self):
        """The region parsed from the RDS host is used when no arg/AWS_REGION."""
        with patch.dict(os.environ, {}, clear=True):
            assert (
                _resolve_db_iam_region(
                    "db.cluster-xyz.us-east-1.rds.amazonaws.com", None
                )
                == "us-east-1"
            )

    def test_resolves_from_aws_region_env(self):
        """AWS_REGION supplies the region when arg + host cannot."""
        with patch.dict(os.environ, {"AWS_REGION": "ap-southeast-2"}, clear=True):
            assert (
                _resolve_db_iam_region("my-aurora-sl.cluster-abc.serverless", None)
                == "ap-southeast-2"
            )

    def test_resolves_from_aws_default_region_env(self):
        """RouteIQ-86ff: AWS_DEFAULT_REGION-only deployments resolve (used to raise)."""
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "eu-central-1"}, clear=True):
            assert (
                _resolve_db_iam_region("my-aurora-sl.cluster-abc.serverless", None)
                == "eu-central-1"
            )

    def test_aws_region_precedes_aws_default_region(self):
        """AWS_REGION wins over AWS_DEFAULT_REGION when both are set."""
        with patch.dict(
            os.environ,
            {"AWS_REGION": "us-east-1", "AWS_DEFAULT_REGION": "eu-central-1"},
            clear=True,
        ):
            assert (
                _resolve_db_iam_region("my-aurora-sl.cluster-abc.serverless", None)
                == "us-east-1"
            )

    def test_resolves_from_boto3_session_when_env_empty(self, monkeypatch):
        """RouteIQ-86ff: boto3 Session().region_name (profile/IMDS) is the final
        fallback before raising -- mocked so no real boto3/network is needed."""
        import litellm_llmrouter.database as db_mod

        monkeypatch.setattr(
            db_mod, "_region_from_boto3_session", lambda: "ca-central-1"
        )
        with patch.dict(os.environ, {}, clear=True):
            assert (
                _resolve_db_iam_region("my-aurora-sl.cluster-abc.serverless", None)
                == "ca-central-1"
            )

    def test_raises_when_boto3_session_also_empty(self, monkeypatch):
        """Still fails loud when NOTHING resolves (boto3 session yields None too)."""
        import litellm_llmrouter.database as db_mod

        monkeypatch.setattr(db_mod, "_region_from_boto3_session", lambda: None)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(IamRegionUnresolvedError) as exc_info:
                _resolve_db_iam_region("my-aurora-sl.cluster-abc.serverless", None)
        msg = str(exc_info.value)
        assert "ROUTEIQ_POSTGRES__IAM_REGION" in msg
        assert "AWS_DEFAULT_REGION" in msg

    def test_boto3_session_helper_fails_soft_without_boto3(self):
        """_region_from_boto3_session returns None (never raises) if boto3 missing."""
        import litellm_llmrouter.database as db_mod

        with patch.dict(sys.modules, {"boto3": None}):
            # boto3=None makes `import boto3` raise ImportError -> caught -> None.
            assert db_mod._region_from_boto3_session() is None


class TestGetDbPoolIamFailLoud:
    """get_db_pool fails loud when IAM is on but the region is unresolvable."""

    async def test_unresolved_region_raises_in_pool_build(self, monkeypatch):
        """IAM on + serverless host + no region env -> raise (no Region=None mint)."""
        import litellm_llmrouter.database as db_mod

        def _should_not_mint(**kw):
            raise AssertionError("mint reached despite unresolved region")

        monkeypatch.setattr(db_mod, "_mint_db_token", _should_not_mint)
        mock_asyncpg = _mock_asyncpg()
        # IAM on, serverless URL, and NO AWS_REGION / ROUTEIQ_POSTGRES__IAM_REGION.
        env = {"DATABASE_URL": _SERVERLESS_RDS_URL, "ROUTEIQ_DB_IAM_AUTH": "true"}
        with patch.dict(os.environ, env, clear=True):
            reset_settings()
            with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
                with pytest.raises(IamRegionUnresolvedError):
                    await get_db_pool()
        # Pool build aborted before create_pool -> never reached.
        mock_asyncpg.create_pool.assert_not_called()

    async def test_resolved_region_mints_ok(self, monkeypatch):
        """A serverless host + AWS_REGION resolves -> callable password minted."""
        import litellm_llmrouter.database as db_mod

        spy = MagicMock(return_value=_FAKE_DB_TOKEN)
        monkeypatch.setattr(db_mod, "_mint_db_token", spy)
        mock_asyncpg = _mock_asyncpg()
        env = {
            "DATABASE_URL": _SERVERLESS_RDS_URL,
            "ROUTEIQ_DB_IAM_AUTH": "true",
            "AWS_REGION": "ap-southeast-2",  # supplies the otherwise-missing region
        }
        with patch.dict(os.environ, env, clear=True):
            reset_settings()
            with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
                await get_db_pool()

        pw = mock_asyncpg.create_pool.call_args.kwargs["password"]
        assert callable(pw)
        pw()  # trigger the mint
        spy.assert_called_with(
            host="my-aurora-sl.cluster-abc.serverless",
            port=5432,
            user="routeiq",
            region="ap-southeast-2",
        )


# =============================================================================
# RouteIQ-0921 -- run_migrations() is a BOOT-CRITICAL DB caller: fail loud
# =============================================================================
#
# run_migrations() (invoked from the gateway lifespan) calls get_db_pool() inside
# a broad ``except Exception``. get_db_pool already re-raises IamRegionUnresolvedError
# past ITS broad handler; run_migrations must NOT re-swallow it -- an unresolved
# region under RDS/Aurora IAM auth is a fail-loud startup misconfig, not a soft
# "skip migrations" degradation. A generic DB error still soft-degrades (logged,
# swallowed) so a transient outage never blocks boot.


class TestRunMigrationsBootFailLoud:
    async def test_run_migrations_reraises_iam_region_error(self, monkeypatch):
        """run_migrations re-raises IamRegionUnresolvedError (fail loud at boot)."""
        import litellm_llmrouter.database as db_mod

        monkeypatch.setattr(db_mod, "get_database_url", lambda: _SERVERLESS_RDS_URL)
        monkeypatch.setattr(
            db_mod,
            "get_db_pool",
            AsyncMock(side_effect=IamRegionUnresolvedError("no region")),
        )
        with pytest.raises(IamRegionUnresolvedError):
            await run_migrations()

    async def test_run_migrations_soft_degrades_on_generic_error(self, monkeypatch):
        """A non-IAM DB error is swallowed (boot continues, migrations skipped)."""
        import litellm_llmrouter.database as db_mod

        monkeypatch.setattr(db_mod, "get_database_url", lambda: _RDS_URL)
        monkeypatch.setattr(
            db_mod,
            "get_db_pool",
            AsyncMock(side_effect=RuntimeError("transient DB outage")),
        )
        # No raise: generic errors keep the soft-degrade behaviour.
        await run_migrations()
