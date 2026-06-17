"""Unit tests for the Bedrock auto-discovery STARTUP WIRING (RouteIQ-c417).

Proves the previously-dark ``merge_bedrock_discovered_models()`` hook in
``startup.py`` has a LIVE callsite (wired into the gateway lifespan before the
routing strategy is installed) and:

* activates when ``bedrock_discovery.enabled`` is True: the synthesized
  mixed-provider auto-group is APPENDED to the live ``llm_router.model_list``;
* is a byte-stable no-op when disabled (model_list untouched);
* is leader-gated (a non-leader replica skips the scan);
* stays credential-free via an injected ``client_factory`` (the boto3 control
  client is fully mocked, no live AWS call / no AWS credentials).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.settings import (
    BedrockDiscoverySettings,
    reset_settings,
)
from litellm_llmrouter.startup import merge_bedrock_discovered_models


# --- documented control-plane fixtures (mirror test_bedrock_discovery.py) ---

_CLAUDE_ARN = (
    "arn:aws:bedrock:::foundation-model/anthropic.claude-sonnet-4-5-20250101-v1:0"
)
_CLAUDE_ID = "anthropic.claude-sonnet-4-5-20250101-v1:0"
_NOVA_ARN = "arn:aws:bedrock:::foundation-model/amazon.nova-pro-v1:0"
_NOVA_ID = "amazon.nova-pro-v1:0"


def _fm(model_id, provider, inference_types, arn):
    return {
        "modelId": model_id,
        "modelArn": arn,
        "providerName": provider,
        "inferenceTypesSupported": inference_types,
        "outputModalities": ["TEXT"],
        "modelLifecycle": {"status": "ACTIVE"},
    }


def _profile(profile_id, model_arns):
    return {
        "inferenceProfileId": profile_id,
        "inferenceProfileArn": (
            f"arn:aws:bedrock:us-east-1::inference-profile/{profile_id}"
        ),
        "type": "SYSTEM_DEFINED",
        "models": [{"modelArn": a} for a in model_arns],
    }


def _mixed_provider_client(_region):
    """A mocked bedrock control client returning a Claude + Nova mix."""
    client = MagicMock()
    client.list_foundation_models.return_value = {
        "modelSummaries": [
            _fm(_CLAUDE_ID, "Anthropic", ["INFERENCE_PROFILE"], _CLAUDE_ARN),
            _fm(_NOVA_ID, "Amazon", ["ON_DEMAND", "INFERENCE_PROFILE"], _NOVA_ARN),
        ]
    }

    def _list_profiles(**kwargs):
        if kwargs.get("typeEquals") == "SYSTEM_DEFINED":
            return {
                "inferenceProfileSummaries": [
                    _profile(f"global.{_CLAUDE_ID}", [_CLAUDE_ARN]),
                    _profile(f"global.{_NOVA_ID}", [_NOVA_ARN]),
                ]
            }
        return {"inferenceProfileSummaries": []}

    client.list_inference_profiles.side_effect = _list_profiles
    return client


class _FakeRouter:
    """Minimal stand-in for litellm's Router exposing the merge seam."""

    def __init__(self, model_list):
        self.model_list = list(model_list)
        self.set_calls = 0

    def set_model_list(self, model_list):
        self.set_calls += 1
        self.model_list = list(model_list)


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def fake_proxy_server(monkeypatch):
    """Install a fake ``litellm.proxy.proxy_server`` module with an llm_router."""
    base = [
        {
            "model_name": "gpt-4o",
            "litellm_params": {"model": "openai/gpt-4o"},
        }
    ]
    router = _FakeRouter(base)
    mod = types.ModuleType("litellm.proxy.proxy_server")
    mod.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", mod)
    return router


def _leader(monkeypatch, value: bool) -> None:
    import litellm_llmrouter.startup as startup

    monkeypatch.setattr(startup, "_discovery_is_leader", lambda: value)


def test_enabled_auto_group_merges_mixed_provider_group(monkeypatch, fake_proxy_server):
    """enabled + auto_group -> live model_list gains the mixed-provider group."""
    _leader(monkeypatch, True)
    settings = BedrockDiscoverySettings(
        enabled=True,
        source_regions=["us-east-1"],
        auto_group=True,
        auto_group_name="claude-auto",
    )

    merged = merge_bedrock_discovered_models(
        settings=settings, client_factory=_mixed_provider_client
    )

    assert merged == 2  # one arm each for Claude + Nova
    router = fake_proxy_server
    assert router.set_calls == 1
    names = [m["model_name"] for m in router.model_list]
    # Original operator-authored entry preserved.
    assert "gpt-4o" in names
    # All discovered arms collapsed under the single auto-group name.
    auto_arms = [m for m in router.model_list if m["model_name"] == "claude-auto"]
    assert len(auto_arms) == 2
    providers = {m["model_info"]["provider"] for m in auto_arms}
    assert providers == {"Anthropic", "Amazon"}  # mixed-provider group


def test_disabled_is_byte_stable_no_op(monkeypatch, fake_proxy_server):
    """Disabled (default) -> model_list unchanged, set_model_list never called."""
    _leader(monkeypatch, True)
    settings = BedrockDiscoverySettings(enabled=False)
    before = list(fake_proxy_server.model_list)

    merged = merge_bedrock_discovered_models(
        settings=settings, client_factory=_mixed_provider_client
    )

    assert merged == 0
    assert fake_proxy_server.set_calls == 0
    assert fake_proxy_server.model_list == before


def test_non_leader_skips_scan(monkeypatch, fake_proxy_server):
    """A non-leader replica never scans even when enabled (no-op)."""
    _leader(monkeypatch, False)
    factory = MagicMock(side_effect=_mixed_provider_client)
    settings = BedrockDiscoverySettings(
        enabled=True, source_regions=["us-east-1"], auto_group=True
    )

    merged = merge_bedrock_discovered_models(settings=settings, client_factory=factory)

    assert merged == 0
    factory.assert_not_called()
    assert fake_proxy_server.set_calls == 0
