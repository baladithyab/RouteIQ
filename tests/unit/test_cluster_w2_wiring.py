"""Cluster-W2 wiring tests: built-not-wired mechanisms -> live callsites.

Covers the four cluster-W2 seeds where the mechanism was already built but had
no live caller / a persistence bug:

* RouteIQ-a865 -- ``get_governance_store()`` honours ``ROUTEIQ_GOVERNANCE_BACKEND``
  (file default vs DynamoDB).
* RouteIQ-a433 -- the legacy self-service key backfill is reachable + persists
  through a durable store.
* RouteIQ-c6e9 (a) -- the Aurora workspace round-trip no longer DROPS the
  per-workspace ``aws_*`` fields (metadata-fold).
* RouteIQ-c6e9 (b) -- ``merge_workspace_arm_synthesis()`` synthesizes arms into
  the live model_list when the flag is on; byte-stable no-op when off.

Cred-free: no boto3, no DB, no AWS. The Aurora round-trip is asserted through the
pure ``_workspace_columns`` / ``_row_to_workspace`` serialization helpers (the DB
layer is bypassed -- they ARE the persistence contract).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.governance import (
    KeyGovernance,
    WorkspaceConfig,
    reset_governance_engine,
)
from litellm_llmrouter.governance_store import (
    GovernanceStore,
    _row_to_workspace,
    _workspace_columns,
    backfill_self_service_key_metadata,
    get_governance_store,
    reset_governance_store,
)
from litellm_llmrouter.governance_store_dynamodb import (
    DynamoDBGovernanceStore,
    reset_dynamodb_governance_store,
)
from litellm_llmrouter.settings import reset_settings


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_governance_engine()
    reset_governance_store()
    reset_dynamodb_governance_store()
    reset_settings()
    yield
    reset_governance_engine()
    reset_governance_store()
    reset_dynamodb_governance_store()
    reset_settings()


# ===========================================================================
# RouteIQ-a865: get_governance_store() backend selection
# ===========================================================================


def test_default_backend_returns_aurora_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_GOVERNANCE_BACKEND", raising=False)
    store = get_governance_store()
    assert isinstance(store, GovernanceStore)
    assert not isinstance(store, DynamoDBGovernanceStore)


def test_dynamodb_backend_returns_ddb_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    store = get_governance_store()
    assert isinstance(store, DynamoDBGovernanceStore)
    assert store.enabled is True


def test_reset_clears_both_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    first = get_governance_store()
    reset_governance_store()
    monkeypatch.delenv("ROUTEIQ_GOVERNANCE_BACKEND", raising=False)
    second = get_governance_store()
    # After reset + backend switch, a fresh (different-class) store is built.
    assert isinstance(first, DynamoDBGovernanceStore)
    assert isinstance(second, GovernanceStore)


def test_settings_surfaces_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm_llmrouter.settings import get_settings

    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    assert get_settings().governance.backend == "dynamodb"


# ===========================================================================
# RouteIQ-c6e9 (a): Aurora workspace round-trip preserves aws_* fields
# ===========================================================================


def _round_trip(ws: WorkspaceConfig) -> WorkspaceConfig:
    """Serialize a workspace to its column dict and rebuild it (Aurora contract).

    ``_workspace_columns`` writes the JSON-encoded ``metadata`` column; a stored
    row mirrors that, so feeding the same dict back through ``_row_to_workspace``
    reproduces the on-disk -> in-memory hydration.
    """
    cols = _workspace_columns(ws)
    return _row_to_workspace(cols)


def test_aws_fields_survive_round_trip() -> None:
    ws = WorkspaceConfig(
        workspace_id="w1",
        name="WS",
        aws_account_id="222222222222",
        aws_role_arn="arn:aws:iam::222222222222:role/cross",
        aws_region="us-west-2",
        metadata={"tier": "gold"},
    )
    out = _round_trip(ws)
    assert out.aws_account_id == "222222222222"
    assert out.aws_role_arn == "arn:aws:iam::222222222222:role/cross"
    assert out.aws_region == "us-west-2"
    # User-facing metadata round-trips unchanged (no reserved key leaks back).
    assert out.metadata == {"tier": "gold"}


def test_no_aws_fields_metadata_byte_stable() -> None:
    ws = WorkspaceConfig(workspace_id="w2", name="WS", metadata={"k": "v"})
    cols = _workspace_columns(ws)
    # No reserved fold key written when all three aws_* are unset.
    assert cols["metadata"] == '{"k": "v"}'
    out = _row_to_workspace(cols)
    assert out.aws_role_arn is None
    assert out.aws_account_id is None
    assert out.aws_region is None
    assert out.metadata == {"k": "v"}


def test_partial_aws_fields_survive() -> None:
    ws = WorkspaceConfig(
        workspace_id="w3",
        name="WS",
        aws_role_arn="arn:aws:iam::333:role/x",
    )
    out = _round_trip(ws)
    assert out.aws_role_arn == "arn:aws:iam::333:role/x"
    assert out.aws_account_id is None
    assert out.aws_region is None


# ===========================================================================
# RouteIQ-a433: backfill is reachable + persists (mechanism + persistence)
# ===========================================================================


def test_backfill_persists_through_store() -> None:
    from litellm_llmrouter.governance import get_governance_engine

    gov = get_governance_engine()
    raw = "sk-legacy-rawsecret"
    gov.register_key_governance(
        KeyGovernance(key_id=raw, metadata={"self_service": True})
    )
    count = backfill_self_service_key_metadata(gov)
    assert count == 1
    kg = gov.get_key_governance(raw)
    assert kg.metadata["public_id"].startswith("kid_")
    assert "secret_hash" in kg.metadata
    assert "masked" in kg.metadata
    # idempotent: re-run is a no-op.
    assert backfill_self_service_key_metadata(gov) == 0


# ===========================================================================
# RouteIQ-c6e9 (b): merge_workspace_arm_synthesis() live caller
# ===========================================================================


class _FakeRouter:
    def __init__(self) -> None:
        self.model_list: list = []

    def set_model_list(self, entries: list) -> None:
        self.model_list = entries


def _install_fake_router(monkeypatch: pytest.MonkeyPatch, router) -> None:
    """Install a fake ``litellm.proxy.proxy_server`` module exposing llm_router.

    Mirrors ``test_bedrock_discovery_wiring.fake_proxy_server`` -- patching the
    attribute directly fails because the submodule is not imported as an
    attribute of ``litellm.proxy``; injecting into ``sys.modules`` is the
    contract the merge functions' ``from litellm.proxy.proxy_server import
    llm_router`` honours.
    """
    mod = types.ModuleType("litellm.proxy.proxy_server")
    mod.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", mod)


def _register_ws_with_role() -> None:
    from litellm_llmrouter.governance import get_governance_engine

    gov = get_governance_engine()
    gov.register_workspace(
        WorkspaceConfig(
            workspace_id="w1",
            name="WS",
            aws_role_arn="arn:aws:iam::222:role/cross",
            aws_region="us-west-2",
            allowed_models=["anthropic.claude-3"],
        )
    )


def test_synthesis_disabled_is_byte_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm_llmrouter import startup

    monkeypatch.delenv("ROUTEIQ_WORKSPACE_ARM_SYNTHESIS_ENABLED", raising=False)
    _register_ws_with_role()
    router = _FakeRouter()
    _install_fake_router(monkeypatch, router)
    merged = startup.merge_workspace_arm_synthesis()
    assert merged == 0
    assert router.model_list == []  # untouched -> byte-stable


def test_synthesis_enabled_merges_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm_llmrouter import startup

    monkeypatch.setenv("ROUTEIQ_WORKSPACE_ARM_SYNTHESIS_ENABLED", "true")
    _register_ws_with_role()
    router = _FakeRouter()
    _install_fake_router(monkeypatch, router)
    merged = startup.merge_workspace_arm_synthesis()
    assert merged == 1
    assert len(router.model_list) == 1
    entry = router.model_list[0]
    params = entry["litellm_params"]
    assert params["model"] == "bedrock/anthropic.claude-3"
    assert params["aws_role_name"] == "arn:aws:iam::222:role/cross"
    assert params["aws_session_name"] == "routeiq-ws-w1"
    assert params["aws_region_name"] == "us-west-2"


def test_synthesis_enabled_no_role_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm_llmrouter import startup
    from litellm_llmrouter.governance import get_governance_engine

    monkeypatch.setenv("ROUTEIQ_WORKSPACE_ARM_SYNTHESIS_ENABLED", "true")
    gov = get_governance_engine()
    gov.register_workspace(
        WorkspaceConfig(workspace_id="w0", name="WS", allowed_models=["m"])
    )
    router = _FakeRouter()
    _install_fake_router(monkeypatch, router)
    merged = startup.merge_workspace_arm_synthesis()
    assert merged == 0
    assert router.model_list == []


def test_synthesis_non_leader_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm_llmrouter import startup

    monkeypatch.setenv("ROUTEIQ_WORKSPACE_ARM_SYNTHESIS_ENABLED", "true")
    _register_ws_with_role()
    router = _FakeRouter()
    _install_fake_router(monkeypatch, router)
    monkeypatch.setattr(startup, "_discovery_is_leader", lambda: False)
    merged = startup.merge_workspace_arm_synthesis()
    assert merged == 0
    assert router.model_list == []


def test_synthesis_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm_llmrouter import startup

    monkeypatch.setenv("ROUTEIQ_WORKSPACE_ARM_SYNTHESIS_ENABLED", "true")
    _register_ws_with_role()
    boom = MagicMock()
    boom.set_model_list.side_effect = RuntimeError("router boom")
    boom.model_list = []
    _install_fake_router(monkeypatch, boom)
    # Must swallow the error and report 0, not raise.
    assert startup.merge_workspace_arm_synthesis() == 0


# ===========================================================================
# Settings: flat-env alias mapping (RouteIQ-a865 / c6e9 / 3d33)
# ===========================================================================


def test_flat_env_aliases_map_to_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm_llmrouter.settings import get_settings

    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    monkeypatch.setenv("ROUTEIQ_WORKSPACE_ARM_SYNTHESIS_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    s = get_settings()
    assert s.governance.backend == "dynamodb"
    assert s.workspace_arm_synthesis.enabled is True
    assert s.secrets_vault.enabled is True


def test_flat_env_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm_llmrouter.settings import get_settings

    for var in (
        "ROUTEIQ_GOVERNANCE_BACKEND",
        "ROUTEIQ_WORKSPACE_ARM_SYNTHESIS_ENABLED",
        "ROUTEIQ_SECRETS_VAULT_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    s = get_settings()
    assert s.governance.backend == "file"
    assert s.workspace_arm_synthesis.enabled is False
    assert s.secrets_vault.enabled is False


def test_settings_is_simplenamespace_safe() -> None:
    # Defensive: merge_workspace_arm_synthesis tolerates a settings shim that
    # lacks the field (getattr default False) -- never raises.
    from litellm_llmrouter import startup

    shim = SimpleNamespace(workspace_arm_synthesis=SimpleNamespace())
    with patch("litellm_llmrouter.settings.get_settings", return_value=shim):
        assert startup.merge_workspace_arm_synthesis() == 0
