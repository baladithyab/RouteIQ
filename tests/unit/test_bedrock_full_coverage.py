"""Credential-free unit tests for the Bedrock FULL-COVERAGE routing mode.

RouteIQ-77e8 (absorbs RouteIQ-1c9d). Three layers:

1. ``BedrockDiscoverySettings`` rollup: ``full_bedrock_coverage=True`` is ONE
   flag that implies discovery + marketplace + ``logical_groups`` synthesis with
   the global>geo>regional profile preference left on (residency off). Resolved
   through ``effective_*`` helpers WITHOUT mutating stored config.
2. ``DiscoveryResult.to_litellm_model_list(synthesis_mode=...)`` dispatch:
   ``distinct`` (default/byte-stable) | ``auto_group`` | ``logical_groups``.
   ``logical_groups`` produces one group per LOGICAL model fanned across its
   arms (closes RouteIQ-1c9d's built-not-wired gap).
3. ``merge_bedrock_discovered_models`` wires the rollup end-to-end into the live
   ``model_list``.

All boto3 is mocked from documented response shapes; NO live AWS call, NO
credentials. Default OFF / byte-stable is asserted explicitly.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.bedrock_discovery import (
    DiscoveredModel,
    DiscoveryResult,
    InferenceProfileRef,
    MarketplaceEndpoint,
    ProfileTier,
    ServerlessClass,
)
from litellm_llmrouter.settings import (
    BedrockDiscoverySettings,
    BedrockSynthesisMode,
    reset_settings,
)
from litellm_llmrouter.startup import merge_bedrock_discovered_models


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    yield
    reset_settings()


# ---------------------------------------------------------------------------
# Builders (mirror the shared discovery fixtures)
# ---------------------------------------------------------------------------


def _claude_global(region: str = "us-east-1") -> DiscoveredModel:
    arn = "arn:aws:bedrock:::foundation-model/anthropic.claude-sonnet-4-v1:0"
    return DiscoveredModel(
        model_id="anthropic.claude-sonnet-4-v1:0",
        model_arn=arn,
        provider_name="Anthropic",
        region=region,
        inference_types=("INFERENCE_PROFILE",),
        output_modalities=("TEXT",),
        serverless_class=ServerlessClass.INFERENCE_PROFILE,
        profiles=[
            InferenceProfileRef(
                profile_id="global.anthropic.claude-sonnet-4-v1:0",
                arn=(
                    f"arn:aws:bedrock:{region}::inference-profile/"
                    "global.anthropic.claude-sonnet-4-v1:0"
                ),
                profile_type="SYSTEM_DEFINED",
                tier=ProfileTier.GLOBAL,
            )
        ],
    )


def _claude_geo(region: str = "eu-west-1") -> DiscoveredModel:
    """SAME logical Claude, in-geo eu. profile only (different region arm)."""
    arn = "arn:aws:bedrock:::foundation-model/anthropic.claude-sonnet-4-v1:0"
    return DiscoveredModel(
        model_id="anthropic.claude-sonnet-4-v1:0",
        model_arn=arn,
        provider_name="Anthropic",
        region=region,
        inference_types=("INFERENCE_PROFILE",),
        output_modalities=("TEXT",),
        serverless_class=ServerlessClass.INFERENCE_PROFILE,
        profiles=[
            InferenceProfileRef(
                profile_id="eu.anthropic.claude-sonnet-4-v1:0",
                arn=(
                    f"arn:aws:bedrock:{region}::inference-profile/"
                    "eu.anthropic.claude-sonnet-4-v1:0"
                ),
                profile_type="SYSTEM_DEFINED",
                tier=ProfileTier.GEOGRAPHIC,
                geo_prefix="eu.",
            )
        ],
    )


def _nova(region: str = "us-east-1") -> DiscoveredModel:
    arn = "arn:aws:bedrock:::foundation-model/amazon.nova-pro-v1:0"
    return DiscoveredModel(
        model_id="amazon.nova-pro-v1:0",
        model_arn=arn,
        provider_name="Amazon",
        region=region,
        inference_types=("ON_DEMAND",),
        output_modalities=("TEXT",),
        serverless_class=ServerlessClass.RAW_ON_DEMAND,
        profiles=[],
    )


# ===========================================================================
# Layer 1: settings rollup (effective_* resolution, no field mutation)
# ===========================================================================


class TestFullCoverageRollup:
    def test_defaults_are_byte_stable(self):
        """Untouched settings: discovery off, marketplace off, distinct mode."""
        s = BedrockDiscoverySettings()
        assert s.full_bedrock_coverage is False
        assert s.synthesis_mode is BedrockSynthesisMode.DISTINCT
        assert s.effective_enabled is False
        assert s.effective_include_marketplace_endpoints is False
        assert s.effective_synthesis_mode is BedrockSynthesisMode.DISTINCT

    def test_full_coverage_implies_enabled_marketplace_logical(self):
        """ONE flag turns on discovery + marketplace + logical_groups."""
        s = BedrockDiscoverySettings(full_bedrock_coverage=True)
        assert s.effective_enabled is True
        assert s.effective_include_marketplace_endpoints is True
        assert s.effective_synthesis_mode is BedrockSynthesisMode.LOGICAL_GROUPS
        # Stored fields are NOT mutated -- only the effective view changes.
        assert s.enabled is False
        assert s.include_marketplace_endpoints is False
        assert s.synthesis_mode is BedrockSynthesisMode.DISTINCT

    def test_full_coverage_global_preference_default(self):
        """Full-coverage leaves residency off so global>geo>regional applies."""
        s = BedrockDiscoverySettings(full_bedrock_coverage=True)
        assert s.residency_constraint is False

    def test_explicit_synthesis_mode_overrides_full_coverage(self):
        """An explicit non-default synthesis_mode wins over the rollup default."""
        s = BedrockDiscoverySettings(
            full_bedrock_coverage=True,
            synthesis_mode=BedrockSynthesisMode.AUTO_GROUP,
        )
        assert s.effective_synthesis_mode is BedrockSynthesisMode.AUTO_GROUP

    def test_legacy_auto_group_bool_maps_to_auto_group_mode(self):
        """Back-compat: auto_group=True with the default mode -> auto_group."""
        s = BedrockDiscoverySettings(enabled=True, auto_group=True)
        assert s.effective_synthesis_mode is BedrockSynthesisMode.AUTO_GROUP

    def test_explicit_mode_wins_over_legacy_auto_group_bool(self):
        s = BedrockDiscoverySettings(
            enabled=True,
            auto_group=True,
            synthesis_mode=BedrockSynthesisMode.LOGICAL_GROUPS,
        )
        assert s.effective_synthesis_mode is BedrockSynthesisMode.LOGICAL_GROUPS

    def test_enabled_without_rollup_keeps_distinct(self):
        s = BedrockDiscoverySettings(enabled=True)
        assert s.effective_enabled is True
        assert s.effective_synthesis_mode is BedrockSynthesisMode.DISTINCT


# ===========================================================================
# Layer 2: to_litellm_model_list(synthesis_mode=...) dispatch
# ===========================================================================


class TestSynthesisModeDispatch:
    def test_distinct_is_default_and_one_name_per_arm(self):
        """Default (no synthesis_mode, auto_group off): distinct names."""
        result = DiscoveryResult(models=[_claude_global(), _nova()])
        entries = result.to_litellm_model_list()
        names = {e["model_name"] for e in entries}
        # two distinct per-model names (tier.provider.modelId form).
        assert len(names) == 2
        assert all("claude-auto" not in n for n in names)

    def test_auto_group_mode_collapses_to_one_name(self):
        result = DiscoveryResult(models=[_claude_global(), _nova()])
        entries = result.to_litellm_model_list(
            synthesis_mode="auto_group", auto_group_name="claude-auto"
        )
        assert {e["model_name"] for e in entries} == {"claude-auto"}
        assert len(entries) == 2

    def test_logical_groups_mode_one_group_per_logical_model(self):
        """logical_groups: SAME logical model fans across arms, distinct models
        stay distinct groups (closes RouteIQ-1c9d)."""
        result = DiscoveryResult(
            models=[
                _claude_global("us-east-1"),
                _claude_geo("eu-west-1"),
                _nova("us-east-1"),
            ]
        )
        entries = result.to_litellm_model_list(synthesis_mode="logical_groups")
        names = {e["model_name"] for e in entries}
        # one group per logical model.
        assert names == {"anthropic.claude-sonnet-4", "amazon.nova-pro"}
        # the claude logical group fans across two region arms.
        claude_arms = [
            e for e in entries if e["model_name"] == "anthropic.claude-sonnet-4"
        ]
        assert len(claude_arms) == 2
        assert {a["model_info"]["region"] for a in claude_arms} == {
            "us-east-1",
            "eu-west-1",
        }

    def test_logical_groups_binds_marketplace_arm(self):
        """A mantle endpoint of the same logical model joins its group."""
        ep = MarketplaceEndpoint(
            endpoint_arn=(
                "arn:aws:bedrock:us-east-1:123456789012:"
                "marketplace-model-endpoint/anthropic.claude-sonnet-4-v1:0/ep-1"
            ),
            model_source=(
                "arn:aws:bedrock:::marketplace-model/anthropic.claude-sonnet-4-v1:0"
            ),
            region="us-east-1",
            status="REGISTERED",
        )
        result = DiscoveryResult(
            models=[_claude_global("us-east-1")], marketplace_endpoints=[ep]
        )
        entries = result.to_litellm_model_list(synthesis_mode="logical_groups")
        assert {e["model_name"] for e in entries} == {"anthropic.claude-sonnet-4"}
        models = {e["litellm_params"]["model"] for e in entries}
        assert any(m.startswith("bedrock/converse/global.") for m in models)
        assert any("marketplace-model-endpoint" in m for m in models)

    def test_explicit_mode_supersedes_legacy_auto_group_arg(self):
        """When synthesis_mode is given it wins over the auto_group bool arg."""
        result = DiscoveryResult(models=[_claude_global(), _nova()])
        entries = result.to_litellm_model_list(
            auto_group=True, synthesis_mode="logical_groups"
        )
        names = {e["model_name"] for e in entries}
        assert names == {"anthropic.claude-sonnet-4", "amazon.nova-pro"}

    def test_global_preferred_over_geo_in_logical_groups(self):
        """global>geo>regional: the global arm carries the global. profile id."""
        result = DiscoveryResult(models=[_claude_global("us-east-1")])
        entries = result.to_litellm_model_list(synthesis_mode="logical_groups")
        assert entries[0]["litellm_params"]["model"].startswith(
            "bedrock/converse/global."
        )

    def test_empty_result_yields_empty_list_for_all_modes(self):
        for mode in ("distinct", "auto_group", "logical_groups"):
            assert DiscoveryResult().to_litellm_model_list(synthesis_mode=mode) == []


# ===========================================================================
# Layer 3: end-to-end wiring through merge_bedrock_discovered_models
# ===========================================================================


_CLAUDE_ARN = (
    "arn:aws:bedrock:::foundation-model/anthropic.claude-sonnet-4-20250101-v1:0"
)
_CLAUDE_ID = "anthropic.claude-sonnet-4-20250101-v1:0"
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


def _full_coverage_client(_region):
    """Mocked control client: Claude (global profile) + Nova (raw) + a
    REGISTERED marketplace endpoint."""
    client = MagicMock()
    client.list_foundation_models.return_value = {
        "modelSummaries": [
            _fm(_CLAUDE_ID, "Anthropic", ["INFERENCE_PROFILE"], _CLAUDE_ARN),
            _fm(_NOVA_ID, "Amazon", ["ON_DEMAND"], _NOVA_ARN),
        ]
    }

    def _list_profiles(**kwargs):
        if kwargs.get("typeEquals") == "SYSTEM_DEFINED":
            return {
                "inferenceProfileSummaries": [
                    _profile(f"global.{_CLAUDE_ID}", [_CLAUDE_ARN]),
                ]
            }
        return {"inferenceProfileSummaries": []}

    client.list_inference_profiles.side_effect = _list_profiles
    client.list_marketplace_model_endpoints.return_value = {
        "marketplaceModelEndpoints": [
            {
                "endpointArn": (
                    "arn:aws:bedrock:us-east-1:123456789012:"
                    "marketplace-model-endpoint/deepseek.r1/ep-1"
                ),
                "modelSourceIdentifier": (
                    "arn:aws:bedrock:::marketplace-model/deepseek.r1"
                ),
                "status": "REGISTERED",
            }
        ]
    }
    return client


class _FakeRouter:
    def __init__(self, model_list):
        self.model_list = list(model_list)
        self.set_calls = 0

    def set_model_list(self, model_list):
        self.set_calls += 1
        self.model_list = list(model_list)


@pytest.fixture
def fake_proxy_server(monkeypatch):
    router = _FakeRouter(
        [{"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}}]
    )
    mod = types.ModuleType("litellm.proxy.proxy_server")
    mod.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", mod)
    return router


def _leader(monkeypatch, value: bool) -> None:
    import litellm_llmrouter.startup as startup

    monkeypatch.setattr(startup, "_discovery_is_leader", lambda: value)


class TestFullCoverageWiring:
    def test_full_coverage_merges_logical_groups_plus_marketplace(
        self, monkeypatch, fake_proxy_server
    ):
        """full_bedrock_coverage on (enabled NOT explicitly set): discovery runs,
        marketplace is scanned, and arms merge as logical groups."""
        _leader(monkeypatch, True)
        settings = BedrockDiscoverySettings(
            full_bedrock_coverage=True,
            source_regions=["us-east-1"],
        )

        merged = merge_bedrock_discovered_models(
            settings=settings, client_factory=_full_coverage_client
        )

        # Claude (1 arm) + Nova (1 arm) + marketplace deepseek (1 arm) = 3.
        assert merged == 3
        router = fake_proxy_server
        assert router.set_calls == 1
        names = {m["model_name"] for m in router.model_list}
        # operator-authored entry preserved.
        assert "gpt-4o" in names
        # logical-group names per logical model (NOT one collapsed name); the
        # version suffix is stripped, the date is kept.
        assert "anthropic.claude-sonnet-4-20250101" in names
        assert "amazon.nova-pro" in names
        # marketplace mantle arm bound into a logical group too.
        assert any("deepseek" in n for n in names)
        # global profile preferred for Claude.
        claude = [
            m
            for m in router.model_list
            if m["model_name"] == "anthropic.claude-sonnet-4-20250101"
        ][0]
        assert claude["litellm_params"]["model"].startswith("bedrock/converse/global.")

    def test_full_coverage_disabled_is_byte_stable(
        self, monkeypatch, fake_proxy_server
    ):
        """Default off: no scan, model_list untouched."""
        _leader(monkeypatch, True)
        before = list(fake_proxy_server.model_list)
        settings = BedrockDiscoverySettings()  # all defaults

        merged = merge_bedrock_discovered_models(
            settings=settings, client_factory=_full_coverage_client
        )

        assert merged == 0
        assert fake_proxy_server.set_calls == 0
        assert fake_proxy_server.model_list == before

    def test_full_coverage_non_leader_skips(self, monkeypatch, fake_proxy_server):
        _leader(monkeypatch, False)
        factory = MagicMock(side_effect=_full_coverage_client)
        settings = BedrockDiscoverySettings(
            full_bedrock_coverage=True, source_regions=["us-east-1"]
        )

        merged = merge_bedrock_discovered_models(
            settings=settings, client_factory=factory
        )

        assert merged == 0
        factory.assert_not_called()
        assert fake_proxy_server.set_calls == 0
