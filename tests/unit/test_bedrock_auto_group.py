"""Credential-free unit tests for the Bedrock auto-group synthesis (RouteIQ-a416).

``DiscoveryResult.to_litellm_model_list(auto_group=...)`` /
``DiscoveryResult.to_auto_group_model_list()`` collapse every discovered
serverless arm onto a SINGLE configurable ``model_name`` so the entries form one
LiteLLM routing GROUP spanning multiple providers/tiers (Claude + Nova +
gpt-oss). Gated: ``auto_group`` defaults OFF so the per-model distinct-name
mapping stays byte-stable.

All boto3 calls are mocked from documented response shapes; NO live AWS call and
NO credentials are required.
"""

from __future__ import annotations

from litellm_llmrouter.bedrock_discovery import (
    DiscoveredModel,
    DiscoveryResult,
    InferenceProfileRef,
    ProfileTier,
    ServerlessClass,
)


# ---------------------------------------------------------------------------
# Builders: a mixed multi-provider, multi-tier discovery result
# ---------------------------------------------------------------------------


def _claude_global() -> DiscoveredModel:
    """Claude 4.x: INFERENCE_PROFILE-gated, has a global.* profile."""
    arn = "arn:aws:bedrock:::foundation-model/anthropic.claude-sonnet-4-v1:0"
    return DiscoveredModel(
        model_id="anthropic.claude-sonnet-4-v1:0",
        model_arn=arn,
        provider_name="Anthropic",
        region="us-east-1",
        inference_types=("INFERENCE_PROFILE",),
        output_modalities=("TEXT",),
        serverless_class=ServerlessClass.INFERENCE_PROFILE,
        profiles=[
            InferenceProfileRef(
                profile_id="global.anthropic.claude-sonnet-4-v1:0",
                arn="arn:aws:bedrock:us-east-1::inference-profile/global.anthropic.claude-sonnet-4-v1:0",
                profile_type="SYSTEM_DEFINED",
                tier=ProfileTier.GLOBAL,
            )
        ],
    )


def _nova_ondemand() -> DiscoveredModel:
    """Amazon Nova: ON_DEMAND raw + a us. geo profile."""
    arn = "arn:aws:bedrock:::foundation-model/amazon.nova-pro-v1:0"
    return DiscoveredModel(
        model_id="amazon.nova-pro-v1:0",
        model_arn=arn,
        provider_name="Amazon",
        region="us-east-1",
        inference_types=("ON_DEMAND", "INFERENCE_PROFILE"),
        output_modalities=("TEXT",),
        serverless_class=ServerlessClass.RAW_ON_DEMAND,
        profiles=[
            InferenceProfileRef(
                profile_id="us.amazon.nova-pro-v1:0",
                arn="arn:aws:bedrock:us-east-1::inference-profile/us.amazon.nova-pro-v1:0",
                profile_type="SYSTEM_DEFINED",
                tier=ProfileTier.GEOGRAPHIC,
                geo_prefix="us.",
            )
        ],
    )


def _gptoss_regional() -> DiscoveredModel:
    """OpenAI gpt-oss: ON_DEMAND, NO global profile -> raw regional path."""
    arn = "arn:aws:bedrock:::foundation-model/openai.gpt-oss-120b-1:0"
    return DiscoveredModel(
        model_id="openai.gpt-oss-120b-1:0",
        model_arn=arn,
        provider_name="OpenAI",
        region="us-west-2",
        inference_types=("ON_DEMAND",),
        output_modalities=("TEXT",),
        serverless_class=ServerlessClass.RAW_ON_DEMAND,
        profiles=[],
    )


def _mixed_result() -> DiscoveryResult:
    return DiscoveryResult(
        models=[_claude_global(), _nova_ondemand(), _gptoss_regional()]
    )


# ---------------------------------------------------------------------------
# Default OFF: byte-stable per-model distinct names
# ---------------------------------------------------------------------------


def test_default_off_keeps_distinct_per_model_names():
    """auto_group defaults OFF: each arm keeps a distinct model_name."""
    result = _mixed_result()
    entries = result.to_litellm_model_list()  # auto_group not passed => off
    names = [e["model_name"] for e in entries]
    # 3 models -> 3 distinct names (no collapsing).
    assert len(entries) == 3
    assert len(set(names)) == 3
    # None of them is the auto-group name.
    assert "claude-auto" not in names


def test_explicit_auto_group_false_is_identity():
    result = _mixed_result()
    assert result.to_litellm_model_list(auto_group=False) == (
        result.to_litellm_model_list()
    )


# ---------------------------------------------------------------------------
# Enabled: a SINGLE group spanning >1 provider
# ---------------------------------------------------------------------------


def test_auto_group_collapses_to_single_model_name():
    """Enabled: all arms share the one configurable group name."""
    result = _mixed_result()
    entries = result.to_litellm_model_list(auto_group=True)
    assert entries, "expected a non-empty group"
    names = {e["model_name"] for e in entries}
    # The acceptance criterion: a single group name spanning all arms.
    assert names == {"claude-auto"}
    # One arm per discovered model (all yielded a serverless path).
    assert len(entries) == 3


def test_auto_group_spans_more_than_one_provider():
    """The synthesized group spans >1 provider (Claude + Nova + gpt-oss)."""
    result = _mixed_result()
    entries = result.to_litellm_model_list(auto_group=True)
    providers = {e["model_info"]["provider"] for e in entries}
    assert len(providers) > 1
    assert {"Anthropic", "Amazon", "OpenAI"} <= providers


def test_auto_group_custom_name():
    result = _mixed_result()
    entries = result.to_litellm_model_list(auto_group=True, auto_group_name="my-mix")
    assert {e["model_name"] for e in entries} == {"my-mix"}


def test_auto_group_arms_carry_distinguishing_arm_id():
    """Even sharing model_name, each arm keeps a distinct arm_id for telemetry."""
    result = _mixed_result()
    entries = result.to_litellm_model_list(auto_group=True)
    arm_ids = [e["model_info"]["arm_id"] for e in entries]
    assert len(arm_ids) == len(set(arm_ids))  # all distinct
    # arm_id encodes region/invocation_id.
    assert any("global.anthropic.claude-sonnet-4-v1:0" in a for a in arm_ids)
    assert any("openai.gpt-oss-120b-1:0" in a for a in arm_ids)


def test_auto_group_dedups_identical_arms_across_regions():
    """The same invocation path discovered twice yields ONE arm."""
    m = _nova_ondemand()
    result = DiscoveryResult(models=[m, _nova_ondemand()])  # identical
    entries = result.to_auto_group_model_list()
    assert len(entries) == 1


def test_auto_group_absent_when_no_models():
    """Empty discovery -> empty group (caller leaves model_list untouched)."""
    assert DiscoveryResult().to_litellm_model_list(auto_group=True) == []


def test_auto_group_residency_skips_global_but_keeps_geo_and_raw():
    """Residency constraint drops the global tier; geo/raw arms still group."""
    result = _mixed_result()
    entries = result.to_litellm_model_list(auto_group=True, residency_constraint=True)
    # Nova (geo us.) and gpt-oss (raw) still resolve; Claude (global-only) drops.
    arm_ids = {e["model_info"]["arm_id"] for e in entries}
    assert any("us.amazon.nova-pro-v1:0" in a for a in arm_ids)
    assert any("openai.gpt-oss-120b-1:0" in a for a in arm_ids)
    # Claude had ONLY a global profile -> no serverless path under residency.
    assert not any("claude-sonnet-4" in a for a in arm_ids)
    # Still a single group.
    assert {e["model_name"] for e in entries} == {"claude-auto"}


# ---------------------------------------------------------------------------
# Settings wiring
# ---------------------------------------------------------------------------


def test_settings_auto_group_defaults():
    from litellm_llmrouter.settings import BedrockDiscoverySettings

    s = BedrockDiscoverySettings()
    assert s.auto_group is False
    assert s.auto_group_name == "claude-auto"
