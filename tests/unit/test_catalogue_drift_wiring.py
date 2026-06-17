"""Unit tests for the periodic catalogue-drift WIRING (RouteIQ-9a42).

The drift mechanism (``compute_catalogue_drift`` + ``record_catalogue_drift_metric``)
is unit-tested elsewhere; these tests prove the LIVE wiring:

* ``startup.run_catalogue_drift_check`` re-runs discovery, diffs vs the live
  ``model_list``, and emits the drift metric -- but only when drift detection is
  enabled AND this replica is the leader;
* it is a byte-stable no-op when disabled (default) and on a non-leader replica;
* ``gateway.app._maybe_start_catalogue_drift_task`` creates a periodic task only
  when enabled (so the default boot creates no task).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

import litellm_llmrouter.startup as startup
from litellm_llmrouter.settings import BedrockDiscoverySettings, reset_settings
from litellm_llmrouter import bedrock_discovery as bd
from litellm_llmrouter.bedrock_discovery import reset_drift_metric


_FM_ARN = "arn:aws:bedrock:::foundation-model/amazon.nova-pro-v1:0"
_FM_ID = "amazon.nova-pro-v1:0"


def _fm(model_id, provider, inference_types, arn):
    return {
        "modelId": model_id,
        "modelArn": arn,
        "providerName": provider,
        "inferenceTypesSupported": inference_types,
        "outputModalities": ["TEXT"],
        "modelLifecycle": {"status": "ACTIVE"},
    }


def _on_demand_client(_region):
    """A mocked bedrock control client offering one ON_DEMAND FM."""
    client = MagicMock()
    client.list_foundation_models.return_value = {
        "modelSummaries": [_fm(_FM_ID, "Amazon", ["ON_DEMAND"], _FM_ARN)]
    }
    client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}
    return client


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    reset_drift_metric()
    yield
    reset_settings()
    reset_drift_metric()


@pytest.fixture
def empty_proxy_server(monkeypatch):
    """Install a fake proxy_server whose llm_router has an EMPTY model_list."""

    class _Router:
        model_list: list = []

    mod = types.ModuleType("litellm.proxy.proxy_server")
    mod.llm_router = _Router()
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", mod)
    return mod.llm_router


def _leader(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(startup, "_discovery_is_leader", lambda: value)


# ---------------------------------------------------------------------------
# run_catalogue_drift_check
# ---------------------------------------------------------------------------


def test_disabled_drift_metric_is_noop(monkeypatch, empty_proxy_server):
    """drift_metric_enabled=False (default) -> None, no scan."""
    _leader(monkeypatch, True)
    factory = MagicMock(side_effect=_on_demand_client)
    settings = BedrockDiscoverySettings(
        enabled=True, source_regions=["us-east-1"], drift_metric_enabled=False
    )
    result = startup.run_catalogue_drift_check(
        settings=settings, client_factory=factory
    )
    assert result is None
    factory.assert_not_called()


def test_non_leader_skips_drift_check(monkeypatch, empty_proxy_server):
    """A non-leader replica never scans for drift."""
    _leader(monkeypatch, False)
    factory = MagicMock(side_effect=_on_demand_client)
    settings = BedrockDiscoverySettings(
        enabled=True, source_regions=["us-east-1"], drift_metric_enabled=True
    )
    result = startup.run_catalogue_drift_check(
        settings=settings, client_factory=factory
    )
    assert result is None
    factory.assert_not_called()


def test_enabled_leader_detects_and_records_drift(monkeypatch, empty_proxy_server):
    """enabled + leader + empty model_list -> drift detected + metric recorded."""
    _leader(monkeypatch, True)
    recorded = {}
    monkeypatch.setattr(
        bd,
        "record_catalogue_drift_metric",
        lambda drift: recorded.update(
            missing=len(drift.missing), extra=len(drift.extra)
        ),
    )
    settings = BedrockDiscoverySettings(
        enabled=True, source_regions=["us-east-1"], drift_metric_enabled=True
    )
    result = startup.run_catalogue_drift_check(
        settings=settings, client_factory=_on_demand_client
    )
    # One model discovered, none configured -> drift_count == 1 (missing).
    assert result == 1
    assert recorded == {"missing": 1, "extra": 0}


def test_never_raises_on_discovery_failure(monkeypatch, empty_proxy_server):
    """A discovery error is swallowed (returns None), never disturbs the gateway."""
    _leader(monkeypatch, True)

    def _boom(_region):
        raise RuntimeError("boom")

    settings = BedrockDiscoverySettings(
        enabled=True, source_regions=["us-east-1"], drift_metric_enabled=True
    )
    # discover_models itself swallows per-region errors, so this still returns an
    # int (0 drift) rather than raising -- the contract is "never raises".
    result = startup.run_catalogue_drift_check(settings=settings, client_factory=_boom)
    assert result is None or isinstance(result, int)


# ---------------------------------------------------------------------------
# _maybe_start_catalogue_drift_task (gateway lifespan wiring)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_not_started_when_disabled(monkeypatch):
    """Default OFF -> no periodic task is created (byte-stable)."""
    from litellm_llmrouter.gateway import app as gateway_app

    monkeypatch.setenv("ROUTEIQ_BEDROCK_DISCOVERY__ENABLED", "false")
    monkeypatch.setenv("ROUTEIQ_BEDROCK_DISCOVERY__DRIFT_METRIC_ENABLED", "false")
    reset_settings()
    task = gateway_app._maybe_start_catalogue_drift_task()
    assert task is None


@pytest.mark.asyncio
async def test_task_started_when_enabled(monkeypatch):
    """enabled + drift_metric_enabled -> a periodic asyncio.Task is created."""
    import asyncio

    from litellm_llmrouter.gateway import app as gateway_app

    monkeypatch.setenv("ROUTEIQ_BEDROCK_DISCOVERY__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_BEDROCK_DISCOVERY__DRIFT_METRIC_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_BEDROCK_DISCOVERY__DRIFT_CHECK_INTERVAL_SECONDS", "30")
    reset_settings()
    task = gateway_app._maybe_start_catalogue_drift_task()
    try:
        assert task is not None
        assert isinstance(task, asyncio.Task)
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    reset_settings()
