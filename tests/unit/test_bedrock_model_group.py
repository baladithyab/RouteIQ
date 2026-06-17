"""Credential-free unit tests for the Bedrock model-group synthesizer (RouteIQ-1c9d).

``DiscoveryResult.synthesize_model_groups()`` binds each LOGICAL model into ONE
group fanned out across its region / account / mantle arms automatically: the
same logical model offered as a ``global.`` profile in one region, a ``us.`` geo
profile in another, a raw ON_DEMAND modelId in a third, AND a marketplace /
mantle custom-deployment endpoint all collapse under one shared ``model_name``
keyed on the model's logical identity (region-varying parts stripped).

Unlike ``to_auto_group_model_list`` (which collapses EVERY discovered model onto
one shared group), this keeps DISTINCT logical models as DISTINCT groups while
fanning each one out over its physical arms. All boto3 is mocked; NO live AWS
call and NO credentials are required.
"""

from __future__ import annotations

from litellm_llmrouter.bedrock_discovery import (
    DiscoveredModel,
    DiscoveryResult,
    InferenceProfileRef,
    MarketplaceEndpoint,
    ProfileTier,
    ServerlessClass,
    _logical_model_name,
)


# ---------------------------------------------------------------------------
# Builders: the SAME logical model offered across several region/account arms
# ---------------------------------------------------------------------------


def _claude_global(region: str = "us-east-1") -> DiscoveredModel:
    """Claude Sonnet 4 via the global.* profile in ``region``."""
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
                arn=f"arn:aws:bedrock:{region}::inference-profile/global.anthropic.claude-sonnet-4-v1:0",
                profile_type="SYSTEM_DEFINED",
                tier=ProfileTier.GLOBAL,
            )
        ],
    )


def _claude_geo_residency(region: str = "eu-west-1") -> DiscoveredModel:
    """The SAME Claude Sonnet 4, but only an in-geo ``eu.`` profile (a different
    region/account arm) — its logical identity is identical."""
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
                arn=f"arn:aws:bedrock:{region}::inference-profile/eu.anthropic.claude-sonnet-4-v1:0",
                profile_type="SYSTEM_DEFINED",
                tier=ProfileTier.GEOGRAPHIC,
                geo_prefix="eu.",
            )
        ],
    )


def _nova(region: str = "us-east-1") -> DiscoveredModel:
    """A DIFFERENT logical model (Nova) — must form its OWN group."""
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


# ---------------------------------------------------------------------------
# Logical-name collapse
# ---------------------------------------------------------------------------


def test_logical_name_strips_tier_geo_and_version():
    """global./geo prefixes and the version suffix all collapse to one identity."""
    assert (
        _logical_model_name("global.anthropic.claude-sonnet-4-v1:0", "Anthropic")
        == "anthropic.claude-sonnet-4"
    )
    assert (
        _logical_model_name("us.anthropic.claude-sonnet-4-v1:0", "Anthropic")
        == "anthropic.claude-sonnet-4"
    )
    assert (
        _logical_model_name("eu.anthropic.claude-sonnet-4", "Anthropic")
        == "anthropic.claude-sonnet-4"
    )
    assert (
        _logical_model_name("anthropic.claude-sonnet-4-v1:0", "Anthropic")
        == "anthropic.claude-sonnet-4"
    )


# ---------------------------------------------------------------------------
# THE acceptance: one logical model_name fans out across region arms
# ---------------------------------------------------------------------------


def test_one_logical_name_fans_out_across_region_arms():
    """global (us-east-1) + geo (eu-west-1) arms of the SAME logical model bind
    under ONE model_name with TWO arms."""
    result = DiscoveryResult(
        models=[_claude_global("us-east-1"), _claude_geo_residency("eu-west-1")]
    )
    entries = result.synthesize_model_groups()
    # both arms share the single logical model_name.
    names = {e["model_name"] for e in entries}
    assert names == {"anthropic.claude-sonnet-4"}
    assert len(entries) == 2
    # the two arms span two regions.
    regions = {e["model_info"]["region"] for e in entries}
    assert regions == {"us-east-1", "eu-west-1"}
    # distinct arm_ids (region/invocation_id) for telemetry attribution.
    arm_ids = {e["model_info"]["arm_id"] for e in entries}
    assert len(arm_ids) == 2


def test_distinct_logical_models_form_distinct_groups():
    """Two DIFFERENT logical models do NOT collapse together — Claude and Nova
    each get their own group."""
    result = DiscoveryResult(models=[_claude_global("us-east-1"), _nova("us-east-1")])
    entries = result.synthesize_model_groups()
    names = {e["model_name"] for e in entries}
    assert names == {"anthropic.claude-sonnet-4", "amazon.nova-pro"}
    # one arm each here.
    assert len(entries) == 2


def test_marketplace_mantle_arm_joins_logical_group():
    """A marketplace / mantle custom-deployment endpoint of the same logical
    model joins that model's group as an additional arm (RouteIQ-1c9d)."""
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
    entries = result.synthesize_model_groups(bind_marketplace=True)
    # both the serverless arm and the mantle endpoint land under one group.
    assert {e["model_name"] for e in entries} == {"anthropic.claude-sonnet-4"}
    assert len(entries) == 2
    models = {e["litellm_params"]["model"] for e in entries}
    # one serverless bedrock/converse arm + one raw bedrock/<endpointArn> arm.
    assert any(m.startswith("bedrock/converse/global.") for m in models)
    assert any("marketplace-model-endpoint" in m for m in models)


def test_marketplace_not_bound_when_disabled():
    ep = MarketplaceEndpoint(
        endpoint_arn="arn:aws:bedrock:us-east-1:1:marketplace-model-endpoint/x/ep-1",
        model_source="arn:aws:bedrock:::marketplace-model/x",
        region="us-east-1",
    )
    result = DiscoveryResult(models=[_claude_global()], marketplace_endpoints=[ep])
    entries = result.synthesize_model_groups(bind_marketplace=False)
    assert len(entries) == 1  # only the serverless arm.


def test_dedups_identical_arms_discovered_twice():
    """The same (region, invocation_id) arm discovered from two source regions
    yields ONE arm, not two."""
    result = DiscoveryResult(
        models=[_claude_global("us-east-1"), _claude_global("us-east-1")]
    )
    entries = result.synthesize_model_groups()
    assert len(entries) == 1


def test_name_prefix_applied_to_every_group():
    result = DiscoveryResult(models=[_claude_global(), _nova()])
    entries = result.synthesize_model_groups(name_prefix="auto/")
    names = {e["model_name"] for e in entries}
    assert names == {"auto/anthropic.claude-sonnet-4", "auto/amazon.nova-pro"}


def test_residency_constraint_drops_global_only_arm():
    """Under residency, the global-only Claude arm has no serverless path and is
    skipped; the geo arm of the SAME logical model still binds the group."""
    result = DiscoveryResult(
        models=[_claude_global("us-east-1"), _claude_geo_residency("eu-west-1")]
    )
    entries = result.synthesize_model_groups(residency_constraint=True)
    # only the eu. geo arm survives.
    assert len(entries) == 1
    assert entries[0]["model_name"] == "anthropic.claude-sonnet-4"
    assert entries[0]["model_info"]["region"] == "eu-west-1"


def test_empty_discovery_yields_no_groups():
    assert DiscoveryResult().synthesize_model_groups() == []


def test_groups_emitted_in_deterministic_order():
    """Group order is stable (sorted by logical name) so the synthesized list is
    reproducible."""
    result = DiscoveryResult(models=[_nova(), _claude_global()])
    entries = result.synthesize_model_groups()
    names_in_order = [e["model_name"] for e in entries]
    # amazon.* sorts before anthropic.*
    assert names_in_order == ["amazon.nova-pro", "anthropic.claude-sonnet-4"]
