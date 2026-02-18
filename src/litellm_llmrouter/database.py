"""
Database Module for A2A and MCP Persistence
============================================

Provides database persistence for A2A agents and MCP servers using PostgreSQL.
Uses asyncpg for async database operations.

Database Schema:
- a2a_agents: Stores A2A agent registrations
- mcp_servers: Stores MCP server registrations
- mcp_tools: Stores MCP tool definitions
- mcp_resources: Stores MCP resource definitions
- audit_logs: Stores control-plane audit logs
"""

import os
import uuid
from datetime import date, datetime, timezone
from typing import Any
from dataclasses import dataclass, field

from litellm._logging import verbose_proxy_logger


# =============================================================================
# Database Configuration
# =============================================================================


def get_database_url() -> str | None:
    """Get the database URL from environment."""
    return os.getenv("DATABASE_URL")


def is_database_configured() -> bool:
    """Check if database is configured."""
    return get_database_url() is not None


# =============================================================================
# A2A Agent Database Model
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
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "A2AAgentDB":
        """Create from dictionary."""
        return cls(
            agent_id=data.get("agent_id", str(uuid.uuid4())),
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


# =============================================================================
# A2A Agent Repository (In-Memory with Optional DB Persistence)
# =============================================================================


class A2AAgentRepository:
    """
    Repository for A2A agent persistence.

    Uses in-memory storage with optional PostgreSQL persistence.
    When database_url is configured, agents are persisted to PostgreSQL.
    """

    def __init__(self):
        self._agents: dict[str, A2AAgentDB] = {}
        self._db_url = get_database_url()

    async def create(self, agent: A2AAgentDB) -> A2AAgentDB:
        """Create a new agent."""
        now = datetime.now(timezone.utc)
        agent.created_at = now
        agent.updated_at = now

        # Store in memory
        self._agents[agent.agent_id] = agent

        # Persist to database if configured
        if self._db_url:
            await self._persist_to_db(agent)

        verbose_proxy_logger.info(f"A2A DB: Created agent {agent.agent_id}")
        return agent

    async def get(self, agent_id: str) -> A2AAgentDB | None:
        """Get an agent by ID."""
        # Try memory first
        if agent_id in self._agents:
            return self._agents[agent_id]

        # Try database if configured
        if self._db_url:
            agent = await self._load_from_db(agent_id)
            if agent:
                self._agents[agent_id] = agent
                return agent

        return None

    async def list_all(
        self,
        user_id: str | None = None,
        team_id: str | None = None,
        include_public: bool = True,
    ) -> list[A2AAgentDB]:
        """
        List all agents with optional filtering.

        Args:
            user_id: Filter by user ID
            team_id: Filter by team ID
            include_public: Include public agents
        """
        agents = list(self._agents.values())

        # Apply filters
        filtered = []
        for agent in agents:
            # Include if public and include_public is True
            if include_public and agent.is_public:
                filtered.append(agent)
                continue

            # Include if user matches
            if user_id and agent.user_id == user_id:
                filtered.append(agent)
                continue

            # Include if team matches
            if team_id and agent.team_id == team_id:
                filtered.append(agent)
                continue

            # Include if no filters specified
            if not user_id and not team_id:
                filtered.append(agent)

        return filtered

    async def update(self, agent_id: str, agent: A2AAgentDB) -> A2AAgentDB | None:
        """Update an existing agent (full update)."""
        if agent_id not in self._agents:
            return None

        agent.updated_at = datetime.utcnow()
        agent.created_at = self._agents[agent_id].created_at
        self._agents[agent_id] = agent

        if self._db_url:
            await self._persist_to_db(agent)

        verbose_proxy_logger.info(f"A2A DB: Updated agent {agent_id}")
        return agent

    async def patch(self, agent_id: str, updates: dict[str, Any]) -> A2AAgentDB | None:
        """Partially update an agent."""
        agent = await self.get(agent_id)
        if not agent:
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(agent, key) and key not in ("agent_id", "created_at"):
                setattr(agent, key, value)

        agent.updated_at = datetime.utcnow()
        self._agents[agent_id] = agent

        if self._db_url:
            await self._persist_to_db(agent)

        verbose_proxy_logger.info(f"A2A DB: Patched agent {agent_id}")
        return agent

    async def delete(self, agent_id: str) -> bool:
        """Delete an agent."""
        if agent_id not in self._agents:
            return False

        del self._agents[agent_id]

        if self._db_url:
            await self._delete_from_db(agent_id)

        verbose_proxy_logger.info(f"A2A DB: Deleted agent {agent_id}")
        return True

    async def make_public(self, agent_id: str) -> A2AAgentDB | None:
        """Make an agent public."""
        return await self.patch(agent_id, {"is_public": True})

    # =========================================================================
    # Database Operations (PostgreSQL)
    # =========================================================================

    async def _persist_to_db(self, agent: A2AAgentDB) -> None:
        """Persist agent to PostgreSQL database."""
        try:
            import asyncpg
            import json

            conn = await asyncpg.connect(self._db_url)
            try:
                await conn.execute(
                    """
                    INSERT INTO a2a_agents (
                        agent_id, name, description, url, capabilities, metadata,
                        team_id, user_id, is_public, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (agent_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        url = EXCLUDED.url,
                        capabilities = EXCLUDED.capabilities,
                        metadata = EXCLUDED.metadata,
                        team_id = EXCLUDED.team_id,
                        user_id = EXCLUDED.user_id,
                        is_public = EXCLUDED.is_public,
                        updated_at = EXCLUDED.updated_at
                    """,
                    agent.agent_id,
                    agent.name,
                    agent.description,
                    agent.url,
                    json.dumps(agent.capabilities),
                    json.dumps(agent.metadata),
                    agent.team_id,
                    agent.user_id,
                    agent.is_public,
                    agent.created_at,
                    agent.updated_at,
                )
            finally:
                await conn.close()
        except ImportError:
            verbose_proxy_logger.warning("asyncpg not installed, skipping DB persist")
        except Exception as e:
            verbose_proxy_logger.error(f"A2A DB: Error persisting agent: {e}")

    async def _load_from_db(self, agent_id: str) -> A2AAgentDB | None:
        """Load agent from PostgreSQL database."""
        try:
            import asyncpg
            import json

            conn = await asyncpg.connect(self._db_url)
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM a2a_agents WHERE agent_id = $1",
                    agent_id,
                )
                if row:
                    return A2AAgentDB(
                        agent_id=row["agent_id"],
                        name=row["name"],
                        description=row["description"],
                        url=row["url"],
                        capabilities=json.loads(row["capabilities"])
                        if row["capabilities"]
                        else [],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        team_id=row["team_id"],
                        user_id=row["user_id"],
                        is_public=row["is_public"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
            finally:
                await conn.close()
        except ImportError:
            verbose_proxy_logger.warning("asyncpg not installed, skipping DB load")
        except Exception as e:
            verbose_proxy_logger.error(f"A2A DB: Error loading agent: {e}")
        return None

    async def _delete_from_db(self, agent_id: str) -> None:
        """Delete agent from PostgreSQL database."""
        try:
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                await conn.execute(
                    "DELETE FROM a2a_agents WHERE agent_id = $1",
                    agent_id,
                )
            finally:
                await conn.close()
        except ImportError:
            verbose_proxy_logger.warning("asyncpg not installed, skipping DB delete")
        except Exception as e:
            verbose_proxy_logger.error(f"A2A DB: Error deleting agent: {e}")


# =============================================================================
# SQL Migration Scripts
# =============================================================================


A2A_AGENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS a2a_agents (
    agent_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    url VARCHAR(1024) NOT NULL,
    capabilities JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    team_id VARCHAR(255),
    user_id VARCHAR(255),
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_a2a_agents_team ON a2a_agents(team_id);
CREATE INDEX IF NOT EXISTS idx_a2a_agents_user ON a2a_agents(user_id);
CREATE INDEX IF NOT EXISTS idx_a2a_agents_public ON a2a_agents(is_public);
"""


async def run_migrations() -> None:
    """Run database migrations."""
    db_url = get_database_url()
    if not db_url:
        verbose_proxy_logger.info("No DATABASE_URL configured, skipping migrations")
        return

    try:
        import asyncpg

        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(A2A_AGENTS_TABLE_SQL)
            await conn.execute(A2A_ACTIVITY_TABLE_SQL)
            await conn.execute(MCP_SERVERS_TABLE_SQL)
            verbose_proxy_logger.info("A2A DB: Migrations completed successfully")
        finally:
            await conn.close()

        # Run audit log migrations
        from .audit import run_audit_migrations

        await run_audit_migrations()

    except ImportError:
        verbose_proxy_logger.warning("asyncpg not installed, skipping migrations")
    except Exception as e:
        verbose_proxy_logger.error(f"A2A DB: Error running migrations: {e}")


# =============================================================================
# Singleton Repository Instance
# =============================================================================


_a2a_repository: A2AAgentRepository | None = None


def get_a2a_repository() -> A2AAgentRepository:
    """Get the global A2A agent repository instance."""
    global _a2a_repository
    if _a2a_repository is None:
        _a2a_repository = A2AAgentRepository()
    return _a2a_repository


# =============================================================================
# A2A Agent Activity Tracking
# =============================================================================


@dataclass
class A2AAgentActivity:
    """Represents a single agent invocation activity record."""

    agent_id: str
    invocation_date: date
    invocation_count: int = 0
    total_latency_ms: int = 0
    success_count: int = 0
    error_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.invocation_count == 0:
            return 0.0
        return self.total_latency_ms / self.invocation_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "date": self.invocation_date.isoformat(),
            "invocation_count": self.invocation_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "success_count": self.success_count,
            "error_count": self.error_count,
        }


class A2AActivityTracker:
    """
    Tracks A2A agent invocation activity for analytics.

    Uses in-memory storage with optional PostgreSQL persistence.
    Activity is aggregated by agent_id and date.
    """

    def __init__(self):
        # Key: (agent_id, date) -> A2AAgentActivity
        self._activity: dict[tuple[str, date], A2AAgentActivity] = {}
        self._db_url = get_database_url()

    async def record_invocation(
        self,
        agent_id: str,
        latency_ms: int,
        success: bool = True,
    ) -> None:
        """
        Record an agent invocation.

        Args:
            agent_id: The ID of the agent that was invoked
            latency_ms: The latency of the invocation in milliseconds
            success: Whether the invocation was successful
        """
        today = date.today()
        key = (agent_id, today)

        if key not in self._activity:
            self._activity[key] = A2AAgentActivity(
                agent_id=agent_id,
                invocation_date=today,
            )

        activity = self._activity[key]
        activity.invocation_count += 1
        activity.total_latency_ms += latency_ms
        if success:
            activity.success_count += 1
        else:
            activity.error_count += 1

        # Persist to database if configured
        if self._db_url:
            await self._persist_activity_to_db(activity)

    async def get_daily_activity(
        self,
        agent_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[A2AAgentActivity]:
        """
        Get daily activity records with optional filtering.

        Args:
            agent_id: Filter by agent ID (None for all agents)
            start_date: Start date for the range (inclusive)
            end_date: End date for the range (inclusive)

        Returns:
            List of activity records matching the filters
        """
        results = []

        for (aid, activity_date), activity in self._activity.items():
            # Filter by agent_id
            if agent_id and aid != agent_id:
                continue

            # Filter by date range
            if start_date and activity_date < start_date:
                continue
            if end_date and activity_date > end_date:
                continue

            results.append(activity)

        # Sort by date descending
        results.sort(key=lambda x: x.invocation_date, reverse=True)
        return results

    async def get_aggregated_activity(
        self,
        agent_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        """
        Get aggregated activity statistics.

        Args:
            agent_id: Filter by agent ID (None for all agents)
            start_date: Start date for the range (inclusive)
            end_date: End date for the range (inclusive)

        Returns:
            Aggregated statistics including total invocations, avg latency, etc.
        """
        activities = await self.get_daily_activity(agent_id, start_date, end_date)

        if not activities:
            return {
                "total_invocations": 0,
                "total_success": 0,
                "total_errors": 0,
                "avg_latency_ms": 0.0,
                "unique_agents": 0,
                "date_range": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None,
                },
            }

        total_invocations = sum(a.invocation_count for a in activities)
        total_latency = sum(a.total_latency_ms for a in activities)
        total_success = sum(a.success_count for a in activities)
        total_errors = sum(a.error_count for a in activities)
        unique_agents = len(set(a.agent_id for a in activities))

        return {
            "total_invocations": total_invocations,
            "total_success": total_success,
            "total_errors": total_errors,
            "avg_latency_ms": round(total_latency / total_invocations, 2)
            if total_invocations > 0
            else 0.0,
            "unique_agents": unique_agents,
            "date_range": {
                "start": min(a.invocation_date for a in activities).isoformat(),
                "end": max(a.invocation_date for a in activities).isoformat(),
            },
        }

    async def _persist_activity_to_db(self, activity: A2AAgentActivity) -> None:
        """Persist activity to PostgreSQL database."""
        try:
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                await conn.execute(
                    """
                    INSERT INTO a2a_agent_activity (
                        agent_id, invocation_date, invocation_count,
                        total_latency_ms, success_count, error_count
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (agent_id, invocation_date) DO UPDATE SET
                        invocation_count = EXCLUDED.invocation_count,
                        total_latency_ms = EXCLUDED.total_latency_ms,
                        success_count = EXCLUDED.success_count,
                        error_count = EXCLUDED.error_count
                    """,
                    activity.agent_id,
                    activity.invocation_date,
                    activity.invocation_count,
                    activity.total_latency_ms,
                    activity.success_count,
                    activity.error_count,
                )
            finally:
                await conn.close()
        except ImportError:
            verbose_proxy_logger.warning(
                "asyncpg not installed, skipping activity persist"
            )
        except Exception as e:
            verbose_proxy_logger.error(f"A2A DB: Error persisting activity: {e}")


# SQL for activity table
A2A_ACTIVITY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS a2a_agent_activity (
    agent_id VARCHAR(255) NOT NULL,
    invocation_date DATE NOT NULL,
    invocation_count INTEGER DEFAULT 0,
    total_latency_ms BIGINT DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    PRIMARY KEY (agent_id, invocation_date)
);

CREATE INDEX IF NOT EXISTS idx_a2a_activity_agent ON a2a_agent_activity(agent_id);
CREATE INDEX IF NOT EXISTS idx_a2a_activity_date ON a2a_agent_activity(invocation_date);
"""


# Singleton activity tracker instance
_a2a_activity_tracker: A2AActivityTracker | None = None


def get_a2a_activity_tracker() -> A2AActivityTracker:
    """Get the global A2A activity tracker instance."""
    global _a2a_activity_tracker
    if _a2a_activity_tracker is None:
        _a2a_activity_tracker = A2AActivityTracker()
    return _a2a_activity_tracker


# =============================================================================
# MCP Server Database Model
# =============================================================================


@dataclass
class MCPServerDB:
    """Database model for MCP server."""

    server_id: str
    name: str
    url: str
    transport: str = "streamable_http"
    tools: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    auth_type: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)
    team_id: str | None = None
    user_id: str | None = None
    is_public: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerDB":
        """Create from dictionary."""
        return cls(
            server_id=data.get("server_id", str(uuid.uuid4())),
            name=data["name"],
            url=data["url"],
            transport=data.get("transport", "streamable_http"),
            tools=data.get("tools", []),
            resources=data.get("resources", []),
            auth_type=data.get("auth_type", "none"),
            metadata=data.get("metadata", {}),
            team_id=data.get("team_id"),
            user_id=data.get("user_id"),
            is_public=data.get("is_public", False),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_id": self.server_id,
            "name": self.name,
            "url": self.url,
            "transport": self.transport,
            "tools": self.tools,
            "resources": self.resources,
            "auth_type": self.auth_type,
            "metadata": self.metadata,
            "team_id": self.team_id,
            "user_id": self.user_id,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# =============================================================================
# MCP Server Repository (In-Memory with Optional DB Persistence)
# =============================================================================


class MCPServerRepository:
    """
    Repository for MCP server persistence.

    Uses in-memory storage with optional PostgreSQL persistence.
    When database_url is configured, servers are persisted to PostgreSQL.
    """

    def __init__(self):
        self._servers: dict[str, MCPServerDB] = {}
        self._db_url = get_database_url()

    async def create(self, server: MCPServerDB) -> MCPServerDB:
        """Create a new server."""
        now = datetime.now(timezone.utc)
        server.created_at = now
        server.updated_at = now

        # Store in memory
        self._servers[server.server_id] = server

        # Persist to database if configured
        if self._db_url:
            await self._persist_to_db(server)

        verbose_proxy_logger.info(f"MCP DB: Created server {server.server_id}")
        return server

    async def get(self, server_id: str) -> MCPServerDB | None:
        """Get a server by ID."""
        # Try memory first
        if server_id in self._servers:
            return self._servers[server_id]

        # Try database if configured
        if self._db_url:
            server = await self._load_from_db(server_id)
            if server:
                self._servers[server_id] = server
                return server

        return None

    async def list_all(
        self,
        user_id: str | None = None,
        team_id: str | None = None,
        include_public: bool = True,
    ) -> list[MCPServerDB]:
        """
        List all servers with optional filtering.

        Args:
            user_id: Filter by user ID
            team_id: Filter by team ID
            include_public: Include public servers
        """
        servers = list(self._servers.values())

        # Apply filters
        filtered = []
        for server in servers:
            # Include if public and include_public is True
            if include_public and server.is_public:
                filtered.append(server)
                continue

            # Include if user matches
            if user_id and server.user_id == user_id:
                filtered.append(server)
                continue

            # Include if team matches
            if team_id and server.team_id == team_id:
                filtered.append(server)
                continue

            # Include if no filters specified
            if not user_id and not team_id:
                filtered.append(server)

        return filtered

    async def update(self, server_id: str, server: MCPServerDB) -> MCPServerDB | None:
        """Update an existing server (full update)."""
        if server_id not in self._servers:
            return None

        server.updated_at = datetime.now(timezone.utc)
        server.created_at = self._servers[server_id].created_at
        self._servers[server_id] = server

        if self._db_url:
            await self._persist_to_db(server)

        verbose_proxy_logger.info(f"MCP DB: Updated server {server_id}")
        return server

    async def delete(self, server_id: str) -> bool:
        """Delete a server."""
        if server_id not in self._servers:
            return False

        del self._servers[server_id]

        if self._db_url:
            await self._delete_from_db(server_id)

        verbose_proxy_logger.info(f"MCP DB: Deleted server {server_id}")
        return True

    # =========================================================================
    # Database Operations (PostgreSQL)
    # =========================================================================

    async def _persist_to_db(self, server: MCPServerDB) -> None:
        """Persist server to PostgreSQL database."""
        try:
            import asyncpg
            import json

            conn = await asyncpg.connect(self._db_url)
            try:
                await conn.execute(
                    """
                    INSERT INTO mcp_servers (
                        server_id, name, url, transport, tools, resources,
                        auth_type, metadata, team_id, user_id, is_public,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (server_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        url = EXCLUDED.url,
                        transport = EXCLUDED.transport,
                        tools = EXCLUDED.tools,
                        resources = EXCLUDED.resources,
                        auth_type = EXCLUDED.auth_type,
                        metadata = EXCLUDED.metadata,
                        team_id = EXCLUDED.team_id,
                        user_id = EXCLUDED.user_id,
                        is_public = EXCLUDED.is_public,
                        updated_at = EXCLUDED.updated_at
                    """,
                    server.server_id,
                    server.name,
                    server.url,
                    server.transport,
                    json.dumps(server.tools),
                    json.dumps(server.resources),
                    server.auth_type,
                    json.dumps(server.metadata),
                    server.team_id,
                    server.user_id,
                    server.is_public,
                    server.created_at,
                    server.updated_at,
                )
            finally:
                await conn.close()
        except ImportError:
            verbose_proxy_logger.warning("asyncpg not installed, skipping DB persist")
        except Exception as e:
            verbose_proxy_logger.error(f"MCP DB: Error persisting server: {e}")

    async def _load_from_db(self, server_id: str) -> MCPServerDB | None:
        """Load server from PostgreSQL database."""
        try:
            import asyncpg
            import json

            conn = await asyncpg.connect(self._db_url)
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM mcp_servers WHERE server_id = $1",
                    server_id,
                )
                if row:
                    return MCPServerDB(
                        server_id=row["server_id"],
                        name=row["name"],
                        url=row["url"],
                        transport=row["transport"],
                        tools=json.loads(row["tools"]) if row["tools"] else [],
                        resources=json.loads(row["resources"])
                        if row["resources"]
                        else [],
                        auth_type=row["auth_type"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        team_id=row["team_id"],
                        user_id=row["user_id"],
                        is_public=row["is_public"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
            finally:
                await conn.close()
        except ImportError:
            verbose_proxy_logger.warning("asyncpg not installed, skipping DB load")
        except Exception as e:
            verbose_proxy_logger.error(f"MCP DB: Error loading server: {e}")
        return None

    async def _delete_from_db(self, server_id: str) -> None:
        """Delete server from PostgreSQL database."""
        try:
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                await conn.execute(
                    "DELETE FROM mcp_servers WHERE server_id = $1",
                    server_id,
                )
            finally:
                await conn.close()
        except ImportError:
            verbose_proxy_logger.warning("asyncpg not installed, skipping DB delete")
        except Exception as e:
            verbose_proxy_logger.error(f"MCP DB: Error deleting server: {e}")


# SQL Migration for MCP tables
MCP_SERVERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mcp_servers (
    server_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    url VARCHAR(1024) NOT NULL,
    transport VARCHAR(50) DEFAULT 'streamable_http',
    tools JSONB DEFAULT '[]',
    resources JSONB DEFAULT '[]',
    auth_type VARCHAR(50) DEFAULT 'none',
    metadata JSONB DEFAULT '{}',
    team_id VARCHAR(255),
    user_id VARCHAR(255),
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mcp_servers_team ON mcp_servers(team_id);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_user ON mcp_servers(user_id);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_public ON mcp_servers(is_public);
"""


# Singleton MCP repository instance
_mcp_repository: MCPServerRepository | None = None


def get_mcp_repository() -> MCPServerRepository:
    """Get the global MCP server repository instance."""
    global _mcp_repository
    if _mcp_repository is None:
        _mcp_repository = MCPServerRepository()
    return _mcp_repository


def reset_database_singletons() -> None:
    """Reset all database singleton instances. For testing only."""
    global _a2a_repository, _a2a_activity_tracker, _mcp_repository
    _a2a_repository = None
    _a2a_activity_tracker = None
    _mcp_repository = None
