"""RouteIQ schema-bootstrap Lambda (P1, ADR-0028).

Invoked by the ``ReplayStoreConstruct`` CFN custom resource on Create AND Update
(NOT Delete). Authenticates to the Aurora cluster with the ROTATED master secret
(NOT IAM) per build-outline D3: the runtime data plane (the gateway pod) uses
short-lived IAM tokens, but the bootstrap is the break-glass leader that runs
BEFORE any IAM DB user exists -- and only the master can issue the
``GRANT rds_iam`` that makes the IAM-auth path work (the rds_iam chicken-and-egg).

The handler does two phase-split jobs, both idempotent:

  1. Provision the runtime IAM user (``CREATE ROLE routeiq`` IF NOT EXISTS via a
     ``DO $$ ... $$`` block) + ``GRANT rds_iam TO routeiq`` + database/schema
     privileges so the pod can IAM-auth as ``routeiq``.
  2. Run RouteIQ's idempotent control-plane DDL: the A2A agents/activity tables,
     the MCP servers table, and the audit_logs table (all
     ``CREATE TABLE / INDEX IF NOT EXISTS``). These DDL strings are vendored
     here verbatim from ``src/litellm_llmrouter/{database,audit}.py`` -- the
     Lambda cannot import the whole RouteIQ package, and the build-outline (D3 /
     section 3a) decided to inline the DDL rather than ship the package. LiteLLM's
     Prisma schema is NOT run here; Prisma stays the app-startup leader
     (``ROUTEIQ_LEADER_MIGRATIONS``), matching build-outline D3.

The ``schema_version`` ResourceProperty is the re-run lever: bumping it on the
CustomResource flips the property hash, CFN issues an Update, and this handler
re-applies the idempotent DDL.

The handler FAILS CLOSED on any error (raises) so a bad bootstrap surfaces as a
CloudFormation failure rather than a silent no-op.

Connect resilience: a freshly created (or auto-paused, min_acu=0) Aurora
Serverless v2 cluster can take 15-30s+ to accept connections (cold resume). The
connect path uses a generous per-attempt timeout plus bounded retry/backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import asyncpg
import boto3

_LOG = logging.getLogger()
_LOG.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Cold-resume tolerance: a new / auto-paused Serverless v2 cluster may need
# 15-30s+ before it accepts connections. Generous per-attempt timeout + bounded
# retries with linear backoff.
_CONNECT_TIMEOUT_SECONDS = 30
_CONNECT_MAX_ATTEMPTS = 5
_CONNECT_BACKOFF_SECONDS = 10

# --------------------------------------------------------------------------- DDL
# Vendored VERBATIM from src/litellm_llmrouter/database.py (A2A + MCP) and
# src/litellm_llmrouter/audit.py (audit_logs). All idempotent
# (CREATE TABLE / INDEX IF NOT EXISTS), so the CustomResource may re-run them on
# any schema_version bump without harm. Keep in sync with the source DDL.

_A2A_AGENTS_TABLE_SQL = """
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

_A2A_ACTIVITY_TABLE_SQL = """
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

_MCP_SERVERS_TABLE_SQL = """
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

_AUDIT_LOGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    request_id VARCHAR(255),
    actor_type VARCHAR(50) NOT NULL DEFAULT 'unknown',
    actor_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    outcome VARCHAR(20) NOT NULL DEFAULT 'success',
    outcome_reason TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor ON audit_logs(actor_type, actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_outcome ON audit_logs(outcome);
CREATE INDEX IF NOT EXISTS idx_audit_logs_request_id ON audit_logs(request_id);
"""

_DDL_STATEMENTS = (
    _A2A_AGENTS_TABLE_SQL,
    _A2A_ACTIVITY_TABLE_SQL,
    _MCP_SERVERS_TABLE_SQL,
    _AUDIT_LOGS_TABLE_SQL,
)


def _fetch_master_credentials(secret_arn: str) -> dict:
    """Read the rotated Aurora master secret (username/password/host/port/dbname)."""
    client = boto3.client("secretsmanager")
    resp = client.get_secret_value(SecretId=secret_arn)
    return json.loads(resp["SecretString"])


async def _connect_with_retry(
    *, host: str, port: int, user: str, password: str, database: str
) -> asyncpg.Connection:
    """Connect over TLS, retrying through a cold Serverless v2 resume."""
    last_exc: Exception | None = None
    for attempt in range(1, _CONNECT_MAX_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(
                asyncpg.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=database,
                    # RDS IAM auth and the runtime data plane both require TLS;
                    # the bootstrap connects over TLS too. ssl="require" lets
                    # asyncpg negotiate TLS without pinning a CA bundle (the
                    # cluster is private-subnet only, reached via the SG).
                    ssl="require",
                ),
                timeout=_CONNECT_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - retry every connect failure
            last_exc = exc
            _LOG.warning(
                "connect attempt %d/%d failed (cluster may be resuming): %s",
                attempt,
                _CONNECT_MAX_ATTEMPTS,
                exc,
            )
            if attempt < _CONNECT_MAX_ATTEMPTS:
                time.sleep(_CONNECT_BACKOFF_SECONDS)
    raise RuntimeError(
        f"could not connect to Aurora at {host}:{port} after {_CONNECT_MAX_ATTEMPTS} attempts"
    ) from last_exc


async def _bootstrap(
    *, host: str, port: int, db_name: str, secret_arn: str, runtime_user: str
) -> None:
    """Provision the runtime IAM user, then apply the idempotent control-plane DDL."""
    creds = _fetch_master_credentials(secret_arn)
    conn = await _connect_with_retry(
        host=host,
        port=port,
        user=creds["username"],
        password=creds["password"],
        database=db_name,
    )
    try:
        # 1) Runtime IAM user (idempotent). asyncpg cannot parameterise an
        #    identifier, but runtime_user is a fixed deploy-time construct value
        #    (never request-derived), so the f-string is not an injection vector.
        #    CREATE ROLE is not IF NOT EXISTS in PostgreSQL, so guard with a DO
        #    block; GRANT rds_iam is what wires AWS IAM authentication onto the
        #    role; the GRANTs make the role usable for the control-plane tables.
        await conn.execute(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{runtime_user}') THEN
                    CREATE ROLE {runtime_user} WITH LOGIN;
                END IF;
            END
            $$;
            """
        )
        await conn.execute(f"GRANT rds_iam TO {runtime_user};")
        await conn.execute(f'GRANT ALL PRIVILEGES ON DATABASE "{db_name}" TO {runtime_user};')
        await conn.execute(f"GRANT ALL ON SCHEMA public TO {runtime_user};")
        await conn.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {runtime_user};"
        )
        await conn.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {runtime_user};"
        )
        _LOG.info("runtime IAM user %s provisioned (rds_iam granted)", runtime_user)

        # 2) Idempotent control-plane DDL.
        for ddl in _DDL_STATEMENTS:
            await conn.execute(ddl)
        _LOG.info("control-plane schema DDL applied (idempotent)")
    finally:
        await conn.close()


def lambda_handler(event: dict, context: object) -> dict:  # noqa: ARG001
    """CFN custom-resource entry point. Create/Update bootstrap; Delete no-op.

    Fails CLOSED on error (raises) so a bad bootstrap surfaces in CloudFormation.
    """
    request_type = event.get("RequestType", "Create")
    _LOG.info("schema-bootstrap RequestType=%s", request_type)

    if request_type == "Delete":
        # The cluster removal_policy handles teardown; nothing to undo here.
        _LOG.info("Delete is a no-op (cluster removal_policy owns teardown)")
        return {"PhysicalResourceId": "routeiq-schema-bootstrap"}

    host = os.environ["DB_HOST"]
    port = int(os.environ.get("DB_PORT", "5432"))
    db_name = os.environ["DB_NAME"]
    secret_arn = os.environ["DB_SECRET_ARN"]
    runtime_user = os.environ.get("DB_RUNTIME_USER", "routeiq")

    asyncio.run(
        _bootstrap(
            host=host,
            port=port,
            db_name=db_name,
            secret_arn=secret_arn,
            runtime_user=runtime_user,
        )
    )
    return {"PhysicalResourceId": "routeiq-schema-bootstrap"}
