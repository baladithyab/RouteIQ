"""Unit tests for the governance engine (workspace isolation & key governance)."""

import pytest

from litellm_llmrouter.governance import (
    GovernanceEngine,
    GovernanceContext,
    WorkspaceConfig,
    KeyGovernance,
    OrgRole,
    WorkspaceRole,
    get_governance_engine,
    reset_governance_engine,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clean_engine():
    """Ensure a fresh engine for each test."""
    reset_governance_engine()
    yield
    reset_governance_engine()


@pytest.fixture
def engine() -> GovernanceEngine:
    return get_governance_engine()


@pytest.fixture
def sample_workspace() -> WorkspaceConfig:
    return WorkspaceConfig(
        workspace_id="ws-acme",
        name="Acme Corp",
        org_id="org-1",
        allowed_models=["gpt-4o*", "@anthropic/*"],
        blocked_models=["gpt-4o-mini"],
        max_budget_usd=1000.0,
        max_rpm=100,
        max_tpm=500_000,
        enforced_guardrails=["content-filter", "pii-guard"],
        default_routing_profile="eco",
    )


@pytest.fixture
def sample_key_gov() -> KeyGovernance:
    return KeyGovernance(
        key_id="key-001",
        workspace_id="ws-acme",
        scopes={"completions.write", "embeddings.write"},
        max_budget_usd=200.0,
        max_rpm=50,
        allowed_models=["gpt-4o", "@anthropic/*"],
    )


# ============================================================================
# Enums
# ============================================================================


def test_org_roles():
    assert OrgRole.OWNER == "owner"
    assert OrgRole.ADMIN == "admin"
    assert OrgRole.MEMBER == "member"


def test_workspace_roles():
    assert WorkspaceRole.ADMIN == "ws_admin"
    assert WorkspaceRole.MANAGER == "ws_manager"
    assert WorkspaceRole.MEMBER == "ws_member"


# ============================================================================
# Workspace CRUD
# ============================================================================


def test_register_and_get_workspace(engine, sample_workspace):
    engine.register_workspace(sample_workspace)
    ws = engine.get_workspace("ws-acme")
    assert ws is not None
    assert ws.name == "Acme Corp"
    assert ws.created_at is not None
    assert ws.updated_at is not None


def test_list_workspaces(engine, sample_workspace):
    engine.register_workspace(sample_workspace)
    engine.register_workspace(
        WorkspaceConfig(workspace_id="ws-other", name="Other", org_id="org-2")
    )
    assert len(engine.list_workspaces()) == 2
    assert len(engine.list_workspaces(org_id="org-1")) == 1
    assert len(engine.list_workspaces(org_id="org-99")) == 0


def test_delete_workspace(engine, sample_workspace):
    engine.register_workspace(sample_workspace)
    assert engine.delete_workspace("ws-acme") is True
    assert engine.get_workspace("ws-acme") is None
    assert engine.delete_workspace("ws-acme") is False


def test_update_workspace_preserves_created_at(engine, sample_workspace):
    engine.register_workspace(sample_workspace)
    original_created = engine.get_workspace("ws-acme").created_at

    updated = WorkspaceConfig(
        workspace_id="ws-acme",
        name="Acme Corp v2",
        created_at=original_created,
    )
    engine.register_workspace(updated)
    ws = engine.get_workspace("ws-acme")
    assert ws.name == "Acme Corp v2"
    assert ws.created_at == original_created


# ============================================================================
# Key Governance CRUD
# ============================================================================


def test_register_and_get_key_governance(engine, sample_key_gov):
    engine.register_key_governance(sample_key_gov)
    kg = engine.get_key_governance("key-001")
    assert kg is not None
    assert kg.workspace_id == "ws-acme"
    assert "completions.write" in kg.scopes


def test_delete_key_governance(engine, sample_key_gov):
    engine.register_key_governance(sample_key_gov)
    assert engine.delete_key_governance("key-001") is True
    assert engine.get_key_governance("key-001") is None
    assert engine.delete_key_governance("key-001") is False


# ============================================================================
# Context Resolution
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_context_no_governance(engine):
    """Key with no governance rules gets a default context."""
    ctx = await engine.resolve_context("unknown-key")
    assert ctx.key_id == "unknown-key"
    assert ctx.workspace_id is None
    assert ctx.effective_allowed_models == set()
    assert ctx.effective_max_rpm is None


@pytest.mark.asyncio
async def test_resolve_context_workspace_only(engine, sample_workspace):
    """Key with workspace but no per-key governance inherits workspace rules."""
    engine.register_workspace(sample_workspace)
    engine.register_key_governance(
        KeyGovernance(key_id="key-ws-only", workspace_id="ws-acme")
    )
    ctx = await engine.resolve_context("key-ws-only")
    assert ctx.workspace_id == "ws-acme"
    assert ctx.workspace_name == "Acme Corp"
    assert ctx.effective_max_rpm == 100
    assert ctx.effective_max_budget_usd == 1000.0
    assert set(ctx.effective_guardrails) == {"content-filter", "pii-guard"}
    assert ctx.effective_routing_profile == "eco"


@pytest.mark.asyncio
async def test_resolve_context_intersection(engine, sample_workspace, sample_key_gov):
    """Key + workspace: most restrictive wins for limits, intersection for models."""
    engine.register_workspace(sample_workspace)
    engine.register_key_governance(sample_key_gov)

    ctx = await engine.resolve_context("key-001")
    assert ctx.workspace_id == "ws-acme"
    # Rate limits: min(workspace=100, key=50) = 50
    assert ctx.effective_max_rpm == 50
    # Budget: min(workspace=1000, key=200) = 200
    assert ctx.effective_max_budget_usd == 200.0
    # Model intersection: key has [gpt-4o, @anthropic/*]
    # workspace has [gpt-4o*, @anthropic/*]
    # gpt-4o matches gpt-4o*, @anthropic/* matches @anthropic/*
    assert "gpt-4o" in ctx.effective_allowed_models
    assert "@anthropic/*" in ctx.effective_allowed_models


@pytest.mark.asyncio
async def test_resolve_context_caching(engine, sample_workspace, sample_key_gov):
    """Resolved contexts are cached."""
    engine.register_workspace(sample_workspace)
    engine.register_key_governance(sample_key_gov)

    ctx1 = await engine.resolve_context("key-001")
    ctx2 = await engine.resolve_context("key-001")
    assert ctx1 is ctx2  # same object from cache


@pytest.mark.asyncio
async def test_cache_invalidation_on_workspace_update(
    engine, sample_workspace, sample_key_gov
):
    """Updating a workspace invalidates cached contexts for keys in it."""
    engine.register_workspace(sample_workspace)
    engine.register_key_governance(sample_key_gov)

    ctx1 = await engine.resolve_context("key-001")

    # Update workspace
    updated = sample_workspace.model_copy()
    updated.max_rpm = 200
    engine.register_workspace(updated)

    ctx2 = await engine.resolve_context("key-001")
    assert ctx2 is not ctx1
    assert ctx2.effective_max_rpm == 50  # still min(200, key=50)


# ============================================================================
# Model Access Checks
# ============================================================================


@pytest.mark.asyncio
async def test_model_access_no_restriction(engine):
    """No allowlist means all models are allowed."""
    ctx = GovernanceContext()
    assert await engine.check_model_access(ctx, "gpt-4o") is True
    assert await engine.check_model_access(ctx, "anything") is True


@pytest.mark.asyncio
async def test_model_access_allowlist(engine):
    """Allowlist restricts to matching patterns."""
    ctx = GovernanceContext(effective_allowed_models={"gpt-4o*", "@anthropic/*"})
    assert await engine.check_model_access(ctx, "gpt-4o") is True
    assert await engine.check_model_access(ctx, "gpt-4o-mini") is True
    assert await engine.check_model_access(ctx, "anthropic/claude-3-opus") is True
    assert await engine.check_model_access(ctx, "mistral-large") is False


@pytest.mark.asyncio
async def test_model_access_blocklist_overrides_allowlist(engine):
    """Blocked models take precedence over allowlist."""
    ctx = GovernanceContext(
        effective_allowed_models={"gpt-4o*"},
        effective_blocked_models={"gpt-4o-mini"},
    )
    assert await engine.check_model_access(ctx, "gpt-4o") is True
    assert await engine.check_model_access(ctx, "gpt-4o-mini") is False


@pytest.mark.asyncio
async def test_model_access_wildcard_all(engine):
    """Wildcard '*' allows everything."""
    ctx = GovernanceContext(effective_allowed_models={"*"})
    assert await engine.check_model_access(ctx, "anything") is True


# ============================================================================
# Model Pattern Matching
# ============================================================================


def test_model_matches_exact(engine):
    assert engine._model_matches("gpt-4o", "gpt-4o") is True
    assert engine._model_matches("gpt-4o", "gpt-4") is False


def test_model_matches_glob(engine):
    assert engine._model_matches("gpt-4o-mini", "gpt-4o*") is True
    assert engine._model_matches("gpt-4o", "gpt-4o*") is True
    assert engine._model_matches("gpt-3.5-turbo", "gpt-4o*") is False


def test_model_matches_provider_prefix(engine):
    assert engine._model_matches("anthropic/claude-3", "@anthropic/*") is True
    assert engine._model_matches("Anthropic/claude-3", "@anthropic/*") is True
    assert engine._model_matches("openai/gpt-4", "@anthropic/*") is False


def test_model_matches_wildcard_all(engine):
    assert engine._model_matches("anything", "*") is True


def test_model_matches_case_insensitive(engine):
    assert engine._model_matches("GPT-4O", "gpt-4o") is True
    assert engine._model_matches("gpt-4o", "GPT-4O") is True


# ============================================================================
# Budget Checks
# ============================================================================


@pytest.mark.asyncio
async def test_budget_check_no_limit(engine):
    """No budget limit always passes."""
    ctx = GovernanceContext()
    assert await engine.check_budget(ctx) is True


@pytest.mark.asyncio
async def test_budget_check_no_redis(engine):
    """Budget check with limit but no Redis fails open."""
    ctx = GovernanceContext(effective_max_budget_usd=100.0)
    # No Redis configured, should fail open
    assert await engine.check_budget(ctx) is True


# ============================================================================
# Rate Limit Checks
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limit_no_limit(engine):
    """No rate limit always passes."""
    ctx = GovernanceContext()
    assert await engine.check_rate_limit(ctx) is True


@pytest.mark.asyncio
async def test_rate_limit_no_redis(engine):
    """Rate limit with limit but no Redis fails open."""
    ctx = GovernanceContext(effective_max_rpm=100)
    assert await engine.check_rate_limit(ctx) is True


# ============================================================================
# Full Enforcement
# ============================================================================


@pytest.mark.asyncio
async def test_enforce_model_denied(engine, sample_workspace, sample_key_gov):
    """Enforce raises 403 on model access denial."""
    engine.register_workspace(sample_workspace)
    engine.register_key_governance(sample_key_gov)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await engine.enforce("key-001", "mistral-large")
    assert exc_info.value.status_code == 403
    assert "model_access_denied" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_enforce_allowed_model(engine, sample_workspace, sample_key_gov):
    """Enforce returns context on allowed model."""
    engine.register_workspace(sample_workspace)
    engine.register_key_governance(sample_key_gov)

    ctx = await engine.enforce("key-001", "gpt-4o")
    assert ctx.workspace_id == "ws-acme"
    assert ctx.key_id == "key-001"


@pytest.mark.asyncio
async def test_enforce_blocked_model(engine, sample_workspace, sample_key_gov):
    """Enforce raises 403 on blocked model (even if it matches allowlist)."""
    engine.register_workspace(sample_workspace)
    engine.register_key_governance(sample_key_gov)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await engine.enforce("key-001", "gpt-4o-mini")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_enforce_no_governance_allows_everything(engine):
    """Key with no governance rules can access any model."""
    ctx = await engine.enforce("ungoverned-key", "any-model")
    assert ctx.key_id == "ungoverned-key"


# ============================================================================
# Singleton
# ============================================================================


def test_singleton():
    e1 = get_governance_engine()
    e2 = get_governance_engine()
    assert e1 is e2

    reset_governance_engine()
    e3 = get_governance_engine()
    assert e3 is not e1


# ============================================================================
# Helper: _min_optional
# ============================================================================


def test_min_optional(engine):
    assert engine._min_optional(None, None) is None
    assert engine._min_optional(10, None) == 10
    assert engine._min_optional(None, 20) == 20
    assert engine._min_optional(10, 20) == 10
    assert engine._min_optional(20, 10) == 10


# ============================================================================
# Status
# ============================================================================


def test_status(engine, sample_workspace, sample_key_gov):
    engine.register_workspace(sample_workspace)
    engine.register_key_governance(sample_key_gov)
    status = engine.get_status()
    assert status["workspace_count"] == 1
    assert status["governed_key_count"] == 1
    assert status["cache_entries"] == 0
