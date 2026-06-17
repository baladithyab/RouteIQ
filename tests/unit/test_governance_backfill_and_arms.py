"""Tests for legacy self-service key backfill (RouteIQ-a433) + cross-account arm
synthesis (RouteIQ-c6e9)."""

from __future__ import annotations

import hashlib

import pytest

from litellm_llmrouter.governance import (
    KeyGovernance,
    WorkspaceConfig,
    reset_governance_engine,
    synthesize_workspace_arm,
    synthesize_workspace_arms,
)
from litellm_llmrouter.governance_store import backfill_self_service_key_metadata


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_governance_engine()
    yield
    reset_governance_engine()


# -- RouteIQ-a433: backfill ------------------------------------------------


def test_backfill_stamps_missing_fields() -> None:
    from litellm_llmrouter.governance import get_governance_engine

    engine = get_governance_engine()
    raw = "sk-legacy-rawsecret"
    engine.register_key_governance(
        KeyGovernance(key_id=raw, metadata={"self_service": True})
    )
    count = backfill_self_service_key_metadata(engine)
    assert count == 1
    kg = engine.get_key_governance(raw)
    assert kg.metadata["public_id"].startswith("kid_")
    assert kg.metadata["secret_hash"] == hashlib.sha256(raw.encode()).hexdigest()
    assert kg.metadata["masked"] == "sk-rq-...cret"


def test_backfill_idempotent() -> None:
    from litellm_llmrouter.governance import get_governance_engine

    engine = get_governance_engine()
    engine.register_key_governance(
        KeyGovernance(key_id="sk-x", metadata={"self_service": True})
    )
    assert backfill_self_service_key_metadata(engine) == 1
    # second run: row already has public_id -> no churn
    assert backfill_self_service_key_metadata(engine) == 0


def test_backfill_skips_non_self_service_and_migrated() -> None:
    from litellm_llmrouter.governance import get_governance_engine

    engine = get_governance_engine()
    engine.register_key_governance(KeyGovernance(key_id="sk-admin", metadata={}))
    engine.register_key_governance(
        KeyGovernance(
            key_id="sk-new",
            metadata={"self_service": True, "public_id": "kid_existing"},
        )
    )
    assert backfill_self_service_key_metadata(engine) == 0
    assert engine.get_key_governance("sk-new").metadata["public_id"] == "kid_existing"


# -- RouteIQ-c6e9: cross-account arm synthesis -----------------------------


def test_no_synthesis_without_role() -> None:
    ws = WorkspaceConfig(workspace_id="w1", name="WS", allowed_models=["m"])
    assert synthesize_workspace_arm(ws, "m") is None
    assert synthesize_workspace_arms(ws) == []


def test_synthesize_single_arm() -> None:
    ws = WorkspaceConfig(
        workspace_id="w1",
        name="WS",
        aws_role_arn="arn:aws:iam::222:role/cross",
        aws_region="us-west-2",
    )
    arm = synthesize_workspace_arm(ws, "anthropic.claude-3")
    assert arm == {
        "model": "bedrock/anthropic.claude-3",
        "aws_role_name": "arn:aws:iam::222:role/cross",
        "aws_session_name": "routeiq-ws-w1",
        "aws_region_name": "us-west-2",
    }


def test_synthesize_uses_default_region_when_unset() -> None:
    ws = WorkspaceConfig(
        workspace_id="w2", name="WS", aws_role_arn="arn:aws:iam::333:role/x"
    )
    arm = synthesize_workspace_arm(ws, "m", default_region="eu-west-1")
    assert arm["aws_region_name"] == "eu-west-1"
    assert arm["model"] == "bedrock/m"


def test_synthesize_arms_skips_wildcards() -> None:
    ws = WorkspaceConfig(
        workspace_id="w3",
        name="WS",
        aws_role_arn="arn:aws:iam::444:role/y",
        allowed_models=["bedrock/concrete-model", "gpt-4o*", "claude?"],
    )
    arms = synthesize_workspace_arms(ws)
    assert [a["model"] for a in arms] == ["bedrock/concrete-model"]
