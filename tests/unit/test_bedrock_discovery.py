"""Credential-free unit tests for the Bedrock model discovery subsystem.

Covers three sequential seeds in one module:

* RouteIQ-6ae6 -- in-process control-plane discovery ALGORITHM: unfiltered
  ListFoundationModels + paginated ListInferenceProfiles (BOTH types),
  three-way serverless detection, ACTIVE-lifecycle filter, per-(model,region)
  join.
* RouteIQ-f86e -- provider-agnostic: keyed by providerName, NO hard-coded
  allow-list. Fixtures include OpenAI ``gpt-oss`` (verbatim ids) and a
  clearly-labeled SPECULATIVE xAI ``grok-*`` to prove no provider is dropped.
* RouteIQ-9ea5 -- inference-profile preference hierarchy global > geographic >
  raw regional, with the two guards (residency skips global; global only when a
  global profile exists). gpt-oss is the no-global counterexample.

ALL boto3 calls are mocked -- the documented response shapes are reconstructed
as fixtures. NO live AWS call and NO AWS credentials are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.bedrock_discovery import (
    DiscoveredModel,
    DiscoveryResult,
    InferenceProfileRef,
    ModelSelection,
    NoServerlessPathInRegion,
    ProfileTier,
    ServerlessClass,
    classify_serverless,
    discover_models,
    discover_region,
    resolve_source_regions,
    select_invocation_path,
    to_litellm_entry,
)
from litellm_llmrouter.settings import (
    BedrockDiscoverySettings,
    GatewaySettings,
    get_settings,
    reset_settings,
)


# ============================================================================
# Fixtures: documented control-plane response shapes
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_settings_singleton():
    """Discovery reads settings via the gateway singleton -- reset around each test."""
    reset_settings()
    yield
    reset_settings()


def _fm(
    model_id: str,
    provider: str,
    inference_types: list[str],
    *,
    status: str = "ACTIVE",
    modalities: list[str] | None = None,
    arn: str | None = None,
) -> dict:
    """Build one ListFoundationModels modelSummaries[] entry (documented shape)."""
    return {
        "modelId": model_id,
        "modelArn": arn or f"arn:aws:bedrock:::foundation-model/{model_id}",
        "providerName": provider,
        "inferenceTypesSupported": inference_types,
        "outputModalities": modalities or ["TEXT"],
        "modelLifecycle": {"status": status},
    }


def _profile(
    profile_id: str,
    model_arns: list[str],
    *,
    ptype: str = "SYSTEM_DEFINED",
    arn: str | None = None,
) -> dict:
    """Build one ListInferenceProfiles inferenceProfileSummaries[] entry."""
    return {
        "inferenceProfileId": profile_id,
        "inferenceProfileArn": arn
        or f"arn:aws:bedrock:us-east-1::inference-profile/{profile_id}",
        "type": ptype,
        "models": [{"modelArn": a} for a in model_arns],
    }


# Anthropic Claude 4.x: INFERENCE_PROFILE-gated frontier model.
_CLAUDE_ARN = (
    "arn:aws:bedrock:::foundation-model/anthropic.claude-sonnet-4-5-20250101-v1:0"
)
_CLAUDE_ID = "anthropic.claude-sonnet-4-5-20250101-v1:0"

# Amazon Nova: ON_DEMAND + INFERENCE_PROFILE.
_NOVA_ARN = "arn:aws:bedrock:::foundation-model/amazon.nova-pro-v1:0"
_NOVA_ID = "amazon.nova-pro-v1:0"

# OpenAI gpt-oss (spec #7, VERBATIM ids). NO global profile.
_GPTOSS120_ARN = "arn:aws:bedrock:::foundation-model/openai.gpt-oss-120b-1:0"
_GPTOSS120_ID = "openai.gpt-oss-120b-1:0"

# Speculative xAI Grok (spec #8) -- hypothetical id to prove provider-agnosticism.
# NOT a real Bedrock modelId; included ONLY to assert no provider is dropped.
_GROK_ARN = "arn:aws:bedrock:::foundation-model/xai.grok-4-SPECULATIVE-v1:0"
_GROK_ID = "xai.grok-4-SPECULATIVE-v1:0"

# A PROVISIONED-only model that must be filtered out.
_PROV_ID = "vendor.provisioned-only-v1:0"


def _make_client(
    *,
    summaries: list[dict],
    system_profiles: list[dict] | None = None,
    application_profiles: list[dict] | None = None,
    availability: dict | None = None,
) -> MagicMock:
    """Mock a boto3 ``bedrock`` control-plane client from documented shapes.

    ``list_inference_profiles`` honours the ``typeEquals`` kwarg and supports a
    two-page SYSTEM_DEFINED response via ``nextToken`` to exercise pagination.
    """
    client = MagicMock()
    client.list_foundation_models.return_value = {"modelSummaries": summaries}

    sys_profiles = system_profiles or []
    app_profiles = application_profiles or []

    def _list_profiles(**kwargs):
        te = kwargs.get("typeEquals")
        token = kwargs.get("nextToken")
        if te == "SYSTEM_DEFINED":
            # Paginate: first call returns page 1 + nextToken, second returns rest.
            if len(sys_profiles) > 1 and token is None:
                return {
                    "inferenceProfileSummaries": sys_profiles[:1],
                    "nextToken": "PAGE2",
                }
            if token == "PAGE2":
                return {"inferenceProfileSummaries": sys_profiles[1:]}
            return {"inferenceProfileSummaries": sys_profiles}
        if te == "APPLICATION":
            return {"inferenceProfileSummaries": app_profiles}
        return {"inferenceProfileSummaries": []}

    client.list_inference_profiles.side_effect = _list_profiles

    if availability is not None:
        client.get_foundation_model_availability.return_value = availability
    else:
        # Default: no availability method behaviour invoked unless check enabled.
        client.get_foundation_model_availability.return_value = {
            "authorizationStatus": "AUTHORIZED",
            "regionAvailability": "AVAILABLE",
            "agreementAvailability": {"status": "AVAILABLE"},
        }
    return client


# ============================================================================
# SEED 1 (RouteIQ-6ae6): the discovery ALGORITHM
# ============================================================================


class TestThreeWayServerlessDetection:
    def test_on_demand_is_raw(self):
        assert classify_serverless(["ON_DEMAND"]) is ServerlessClass.RAW_ON_DEMAND

    def test_inference_profile_is_profile_class(self):
        assert (
            classify_serverless(["INFERENCE_PROFILE"])
            is ServerlessClass.INFERENCE_PROFILE
        )

    def test_both_prefers_raw_on_demand(self):
        assert (
            classify_serverless(["ON_DEMAND", "INFERENCE_PROFILE"])
            is ServerlessClass.RAW_ON_DEMAND
        )

    def test_provisioned_only_not_serverless(self):
        assert classify_serverless(["PROVISIONED"]) is ServerlessClass.NOT_SERVERLESS

    def test_empty_not_serverless(self):
        assert classify_serverless([]) is ServerlessClass.NOT_SERVERLESS

    def test_case_insensitive(self):
        assert classify_serverless(["on_demand"]) is ServerlessClass.RAW_ON_DEMAND


class TestDiscoverRegion:
    def test_unfiltered_list_called_without_byinferencetype(self):
        client = _make_client(summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])])
        discover_region(client, "us-east-1")
        # ListFoundationModels MUST be called unfiltered (spec #4): no kwargs.
        client.list_foundation_models.assert_called_once_with()

    def test_active_lifecycle_filter(self):
        client = _make_client(
            summaries=[
                _fm(_NOVA_ID, "Amazon", ["ON_DEMAND"], status="ACTIVE"),
                _fm("amazon.legacy-v1:0", "Amazon", ["ON_DEMAND"], status="LEGACY"),
            ]
        )
        models = discover_region(client, "us-east-1")
        ids = {m.model_id for m in models}
        assert _NOVA_ID in ids
        assert "amazon.legacy-v1:0" not in ids

    def test_provisioned_only_dropped(self):
        client = _make_client(
            summaries=[
                _fm(_NOVA_ID, "Amazon", ["ON_DEMAND"]),
                _fm(_PROV_ID, "Vendor", ["PROVISIONED"]),
            ]
        )
        ids = {m.model_id for m in discover_region(client, "us-east-1")}
        assert _NOVA_ID in ids
        assert _PROV_ID not in ids

    def test_inference_profile_gated_model_kept(self):
        # Claude 4.x exposes ONLY INFERENCE_PROFILE -- must NOT be dropped.
        client = _make_client(
            summaries=[_fm(_CLAUDE_ID, "Anthropic", ["INFERENCE_PROFILE"])]
        )
        models = discover_region(client, "us-east-1")
        assert len(models) == 1
        assert models[0].serverless_class is ServerlessClass.INFERENCE_PROFILE

    def test_both_profile_types_queried(self):
        client = _make_client(summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])])
        discover_region(client, "us-east-1")
        called_types = {
            c.kwargs.get("typeEquals")
            for c in client.list_inference_profiles.call_args_list
        }
        # GOTCHA spec #3: BOTH SYSTEM_DEFINED and APPLICATION must be queried.
        assert called_types == {"SYSTEM_DEFINED", "APPLICATION"}

    def test_pagination_followed(self):
        sys_profiles = [
            _profile("global.amazon.nova-pro-v1:0", [_NOVA_ARN]),
            _profile("us.amazon.nova-pro-v1:0", [_NOVA_ARN]),
        ]
        client = _make_client(
            summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])],
            system_profiles=sys_profiles,
        )
        models = discover_region(client, "us-east-1")
        nova = models[0]
        profile_ids = {p.profile_id for p in nova.profiles}
        # Both pages must be joined.
        assert profile_ids == {
            "global.amazon.nova-pro-v1:0",
            "us.amazon.nova-pro-v1:0",
        }

    def test_join_by_model_arn(self):
        client = _make_client(
            summaries=[_fm(_CLAUDE_ID, "Anthropic", ["INFERENCE_PROFILE"])],
            system_profiles=[
                _profile("us.anthropic.claude-sonnet-4-5-20250101-v1:0", [_CLAUDE_ARN]),
                _profile("eu.amazon.nova-pro-v1:0", [_NOVA_ARN]),  # different arn
            ],
        )
        models = discover_region(client, "us-east-1")
        claude = models[0]
        # Only the matching-arn profile is joined.
        assert len(claude.profiles) == 1
        assert claude.profiles[0].profile_id.startswith("us.anthropic")

    def test_record_carries_region_and_provider(self):
        client = _make_client(summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])])
        m = discover_region(client, "eu-west-1")[0]
        assert m.region == "eu-west-1"
        assert m.provider_name == "Amazon"


class TestAvailabilityGate:
    def test_unauthorized_model_dropped(self):
        client = _make_client(
            summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])],
            availability={
                "authorizationStatus": "NOT_AUTHORIZED",
                "regionAvailability": "AVAILABLE",
                "agreementAvailability": {"status": "AVAILABLE"},
            },
        )
        models = discover_region(client, "us-east-1", check_availability=True)
        assert models == []

    def test_authorized_model_kept_and_marked(self):
        client = _make_client(summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])])
        models = discover_region(client, "us-east-1", check_availability=True)
        assert len(models) == 1
        assert models[0].entitled is True

    def test_availability_not_checked_when_disabled(self):
        client = _make_client(summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])])
        models = discover_region(client, "us-east-1", check_availability=False)
        assert models[0].entitled is None
        client.get_foundation_model_availability.assert_not_called()

    def test_unentitled_model_dropped(self):
        # entitlementAvailability is the MOST DIRECT entitlement signal
        # (AVAILABLE | NOT_AVAILABLE); NOT_AVAILABLE must drop the model even
        # when authorization/region/agreement all read clean.
        client = _make_client(
            summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])],
            availability={
                "entitlementAvailability": "NOT_AVAILABLE",
                "authorizationStatus": "AUTHORIZED",
                "regionAvailability": "AVAILABLE",
                "agreementAvailability": {"status": "AVAILABLE"},
            },
        )
        models = discover_region(client, "us-east-1", check_availability=True)
        assert models == []

    def test_entitled_model_kept(self):
        client = _make_client(
            summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])],
            availability={
                "entitlementAvailability": "AVAILABLE",
                "authorizationStatus": "AUTHORIZED",
                "regionAvailability": "AVAILABLE",
                "agreementAvailability": {"status": "AVAILABLE"},
            },
        )
        models = discover_region(client, "us-east-1", check_availability=True)
        assert len(models) == 1
        assert models[0].entitled is True

    def test_missing_entitlement_field_treated_as_unknown(self):
        # Older boto3 / partial responses omit entitlementAvailability entirely;
        # absence must NOT drop an otherwise-clean model (None == not gated).
        client = _make_client(
            summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])],
            availability={
                "authorizationStatus": "AUTHORIZED",
                "regionAvailability": "AVAILABLE",
                "agreementAvailability": {"status": "AVAILABLE"},
            },
        )
        models = discover_region(client, "us-east-1", check_availability=True)
        assert len(models) == 1
        assert models[0].entitled is True


# ============================================================================
# SEED 2 (RouteIQ-f86e): provider-agnostic
# ============================================================================


class TestProviderAgnostic:
    def test_openai_gptoss_onboarded_verbatim_ids(self):
        client = _make_client(
            summaries=[
                _fm(
                    _GPTOSS120_ID,
                    "OpenAI",
                    ["ON_DEMAND"],
                    arn=_GPTOSS120_ARN,
                ),
                _fm(
                    "openai.gpt-oss-20b-1:0",
                    "OpenAI",
                    ["ON_DEMAND"],
                ),
            ]
        )
        models = discover_region(client, "us-east-1")
        ids = {m.model_id for m in models}
        # Verbatim ids (spec #7) must appear untouched.
        assert _GPTOSS120_ID in ids
        assert "openai.gpt-oss-20b-1:0" in ids
        assert {m.provider_name for m in models} == {"OpenAI"}

    def test_speculative_grok_onboarded_when_offered(self):
        # SPECULATIVE: xAI Grok is NOT GA on Bedrock mid-2026 (spec #8). This
        # hypothetical fixture proves the scan auto-onboards ANY provider when
        # Bedrock returns it -- no hard-coded allow-list drops it.
        client = _make_client(
            summaries=[
                _fm(_GROK_ID, "xAI", ["ON_DEMAND"], arn=_GROK_ARN),
            ]
        )
        models = discover_region(client, "us-east-1")
        assert len(models) == 1
        assert models[0].provider_name == "xAI"
        assert models[0].model_id == _GROK_ID

    def test_mixed_providers_all_present_in_model_list(self):
        client = _make_client(
            summaries=[
                _fm(_CLAUDE_ID, "Anthropic", ["INFERENCE_PROFILE"], arn=_CLAUDE_ARN),
                _fm(_NOVA_ID, "Amazon", ["ON_DEMAND"], arn=_NOVA_ARN),
                _fm(_GPTOSS120_ID, "OpenAI", ["ON_DEMAND"], arn=_GPTOSS120_ARN),
                _fm(_GROK_ID, "xAI", ["ON_DEMAND"], arn=_GROK_ARN),  # speculative
            ],
            system_profiles=[
                _profile("us.anthropic.claude-sonnet-4-5-20250101-v1:0", [_CLAUDE_ARN]),
            ],
        )
        result = DiscoveryResult(models=discover_region(client, "us-east-1"))
        # BOTH OpenAI and the hypothetical Grok must appear -- no provider dropped.
        assert {"Anthropic", "Amazon", "OpenAI", "xAI"} <= result.providers()
        entries = result.to_litellm_model_list()
        names = " ".join(e["model_name"] for e in entries)
        assert "openai.gpt-oss-120b-1:0" in names
        assert "xai.grok-4-SPECULATIVE-v1:0" in names


# ============================================================================
# SEED 3 (RouteIQ-9ea5): preference hierarchy + guards
# ============================================================================


def _model_with_profiles(
    *,
    region: str,
    serverless: ServerlessClass,
    profiles: list[InferenceProfileRef],
    model_id: str = _CLAUDE_ID,
    provider: str = "Anthropic",
) -> DiscoveredModel:
    return DiscoveredModel(
        model_id=model_id,
        model_arn=_CLAUDE_ARN,
        provider_name=provider,
        region=region,
        inference_types=("INFERENCE_PROFILE",),
        output_modalities=("TEXT",),
        serverless_class=serverless,
        profiles=profiles,
    )


def _global_ref(model_id: str = _CLAUDE_ID) -> InferenceProfileRef:
    return InferenceProfileRef(
        profile_id=f"global.{model_id}",
        arn=f"arn:aws:bedrock:us-east-1::inference-profile/global.{model_id}",
        profile_type="SYSTEM_DEFINED",
        tier=ProfileTier.GLOBAL,
    )


def _geo_ref(prefix: str, model_id: str = _CLAUDE_ID) -> InferenceProfileRef:
    return InferenceProfileRef(
        profile_id=f"{prefix}{model_id}",
        arn=f"arn:aws:bedrock:us-east-1::inference-profile/{prefix}{model_id}",
        profile_type="SYSTEM_DEFINED",
        tier=ProfileTier.GEOGRAPHIC,
        geo_prefix=prefix,
    )


class TestSelectionHierarchy:
    def test_global_geo_regional_picks_global(self):
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.RAW_ON_DEMAND,
            profiles=[_global_ref(), _geo_ref("us.")],
        )
        sel = select_invocation_path(m)
        assert sel.tier is ProfileTier.GLOBAL
        assert sel.invocation_id == f"global.{_CLAUDE_ID}"

    def test_geo_and_regional_picks_geo(self):
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.RAW_ON_DEMAND,
            profiles=[_geo_ref("us.")],
        )
        sel = select_invocation_path(m)
        assert sel.tier is ProfileTier.GEOGRAPHIC
        assert sel.invocation_id == f"us.{_CLAUDE_ID}"

    def test_only_regional_picks_raw_modelid(self):
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.RAW_ON_DEMAND,
            profiles=[],
        )
        sel = select_invocation_path(m)
        assert sel.tier is ProfileTier.REGIONAL
        assert sel.invocation_id == _CLAUDE_ID

    def test_geo_matches_region_family(self):
        # eu-west-1 should pick the eu. geo, not the us. one.
        m = _model_with_profiles(
            region="eu-west-1",
            serverless=ServerlessClass.RAW_ON_DEMAND,
            profiles=[_geo_ref("us."), _geo_ref("eu.")],
        )
        sel = select_invocation_path(m)
        assert sel.invocation_id == f"eu.{_CLAUDE_ID}"

    def test_foreign_geo_only_falls_through_to_raw(self):
        # Only a eu. geo while we are in us-east-1: no in-geo profile -> raw.
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.RAW_ON_DEMAND,
            profiles=[_geo_ref("eu.")],
        )
        sel = select_invocation_path(m)
        assert sel.tier is ProfileTier.REGIONAL
        assert sel.invocation_id == _CLAUDE_ID


class TestSelectionGuards:
    def test_residency_constraint_skips_global(self):
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.RAW_ON_DEMAND,
            profiles=[_global_ref(), _geo_ref("us.")],
        )
        sel = select_invocation_path(m, residency_constraint=True)
        # Guard (a): global skipped -> in-geo geographic wins.
        assert sel.tier is ProfileTier.GEOGRAPHIC
        assert sel.invocation_id == f"us.{_CLAUDE_ID}"

    def test_global_only_when_global_profile_exists(self):
        # Guard (b): no global profile -> never pick global even commercial.
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.RAW_ON_DEMAND,
            profiles=[_geo_ref("us.")],
        )
        sel = select_invocation_path(m)
        assert sel.tier is not ProfileTier.GLOBAL

    def test_inference_profile_only_no_path_raises(self):
        # PROVISIONED-only-style: INFERENCE_PROFILE class but residency blocks
        # global and there is no in-geo geo profile and no raw ON_DEMAND.
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.INFERENCE_PROFILE,
            profiles=[_global_ref()],
        )
        with pytest.raises(NoServerlessPathInRegion):
            select_invocation_path(m, residency_constraint=True)

    def test_not_serverless_raises(self):
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.NOT_SERVERLESS,
            profiles=[],
        )
        with pytest.raises(NoServerlessPathInRegion):
            select_invocation_path(m)


class TestGptOssCounterexample:
    """gpt-oss has NO global profile (spec #7,#12): raw on commercial, us-gov. in GovCloud."""

    def test_gptoss_commercial_picks_raw_modelid(self):
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.RAW_ON_DEMAND,
            profiles=[],  # no global, no geo on commercial serverless path
            model_id=_GPTOSS120_ID,
            provider="OpenAI",
        )
        sel = select_invocation_path(m)
        assert sel.tier is ProfileTier.REGIONAL
        assert sel.invocation_id == _GPTOSS120_ID

    def test_gptoss_govcloud_picks_usgov_geo(self):
        govgeo = InferenceProfileRef(
            profile_id=f"us-gov.{_GPTOSS120_ID}",
            arn=f"arn:aws-us-gov:bedrock:us-gov-west-1::inference-profile/us-gov.{_GPTOSS120_ID}",
            profile_type="SYSTEM_DEFINED",
            tier=ProfileTier.GEOGRAPHIC,
            geo_prefix="us-gov.",
        )
        m = _model_with_profiles(
            region="us-gov-west-1",
            serverless=ServerlessClass.INFERENCE_PROFILE,
            profiles=[govgeo],
            model_id=_GPTOSS120_ID,
            provider="OpenAI",
        )
        sel = select_invocation_path(m)
        assert sel.tier is ProfileTier.GEOGRAPHIC
        assert sel.invocation_id == f"us-gov.{_GPTOSS120_ID}"

    def test_usgov_geo_not_matched_by_commercial_us_region(self):
        # us-east-1 must NOT match a us-gov. profile (longest-prefix discipline).
        govgeo = InferenceProfileRef(
            profile_id=f"us-gov.{_GPTOSS120_ID}",
            arn="arn:x",
            profile_type="SYSTEM_DEFINED",
            tier=ProfileTier.GEOGRAPHIC,
            geo_prefix="us-gov.",
        )
        m = _model_with_profiles(
            region="us-east-1",
            serverless=ServerlessClass.RAW_ON_DEMAND,
            profiles=[govgeo],
            model_id=_GPTOSS120_ID,
            provider="OpenAI",
        )
        sel = select_invocation_path(m)
        assert sel.tier is ProfileTier.REGIONAL  # falls through to raw


# ============================================================================
# LiteLLM mapping (spec #13-14)
# ============================================================================


class TestLiteLLMMapping:
    def test_global_inline_with_bedrock_prefix_and_cost(self):
        sel = ModelSelection(
            model_id=_CLAUDE_ID,
            region="us-east-1",
            provider_name="Anthropic",
            tier=ProfileTier.GLOBAL,
            serverless_class=ServerlessClass.INFERENCE_PROFILE,
            invocation_id=f"global.{_CLAUDE_ID}",
        )
        entry = to_litellm_entry(sel)
        assert (
            entry["litellm_params"]["model"] == f"bedrock/converse/global.{_CLAUDE_ID}"
        )
        assert entry["litellm_params"]["aws_region_name"] == "us-east-1"
        # Cost stub registered for global.* (issue #17286).
        assert "model_info" in entry
        assert entry["model_info"]["input_cost_per_token"] == 0.0

    def test_geo_inline_no_cost_stub(self):
        sel = ModelSelection(
            model_id=_CLAUDE_ID,
            region="us-east-1",
            provider_name="Anthropic",
            tier=ProfileTier.GEOGRAPHIC,
            serverless_class=ServerlessClass.INFERENCE_PROFILE,
            invocation_id=f"us.{_CLAUDE_ID}",
        )
        entry = to_litellm_entry(sel)
        assert entry["litellm_params"]["model"] == f"bedrock/converse/us.{_CLAUDE_ID}"
        assert "model_info" not in entry  # cost stub only for global.*

    def test_raw_regional_keeps_bedrock_prefix(self):
        sel = ModelSelection(
            model_id=_GPTOSS120_ID,
            region="us-east-1",
            provider_name="OpenAI",
            tier=ProfileTier.REGIONAL,
            serverless_class=ServerlessClass.RAW_ON_DEMAND,
            invocation_id=_GPTOSS120_ID,
        )
        entry = to_litellm_entry(sel)
        assert entry["litellm_params"]["model"] == f"bedrock/converse/{_GPTOSS120_ID}"

    def test_application_profile_arn_in_model_id_field(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc"
        sel = ModelSelection(
            model_id=_NOVA_ID,
            region="us-east-1",
            provider_name="Amazon",
            tier=ProfileTier.GEOGRAPHIC,
            serverless_class=ServerlessClass.INFERENCE_PROFILE,
            invocation_id=f"us.{_NOVA_ID}",
            application_profile_arn=arn,
        )
        entry = to_litellm_entry(sel)
        # Application Inference Profile ARN goes in a separate model_id field.
        assert entry["litellm_params"]["model_id"] == arn

    def test_register_cost_can_be_disabled(self):
        sel = ModelSelection(
            model_id=_CLAUDE_ID,
            region="us-east-1",
            provider_name="Anthropic",
            tier=ProfileTier.GLOBAL,
            serverless_class=ServerlessClass.INFERENCE_PROFILE,
            invocation_id=f"global.{_CLAUDE_ID}",
        )
        entry = to_litellm_entry(sel, register_cost=False)
        assert "model_info" not in entry


# ============================================================================
# Settings gating + orchestrator
# ============================================================================


class TestSettingsGating:
    def test_default_disabled(self):
        s = get_settings()
        assert s.bedrock_discovery.enabled is False

    def test_disabled_is_noop(self):
        result = discover_models(settings=BedrockDiscoverySettings(enabled=False))
        assert result.models == []
        assert result.region_errors == {}

    def test_csv_source_regions_direct_construction(self):
        s = BedrockDiscoverySettings(source_regions="us-east-1, eu-west-1")
        assert s.source_regions == ["us-east-1", "eu-west-1"]

    def test_documented_csv_env_form_does_not_crash_gateway(self, monkeypatch):
        # REGRESSION (the test the original suite was missing): the DOCUMENTED
        # comma-separated env form used to raise SettingsError out of the
        # complex-type JSON decode and abort ALL of GatewaySettings(), not just
        # bedrock_discovery. Construct via the FULL env path, not Python kwargs.
        monkeypatch.setenv(
            "ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS", "us-east-1,eu-west-1"
        )
        reset_settings()
        s = GatewaySettings()  # must not raise
        assert s.bedrock_discovery.source_regions == ["us-east-1", "eu-west-1"]

    def test_documented_json_list_env_form_parses(self, monkeypatch):
        monkeypatch.setenv(
            "ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS",
            '["us-east-1","eu-west-1"]',
        )
        reset_settings()
        s = GatewaySettings()
        assert s.bedrock_discovery.source_regions == ["us-east-1", "eu-west-1"]

    def test_register_cost_default_on(self):
        assert BedrockDiscoverySettings().register_cost is True


class TestResolveSourceRegions:
    def test_intersect_with_available(self):
        out = resolve_source_regions(
            ["us-east-1", "mars-1"], available=["us-east-1", "eu-west-1"]
        )
        assert out == ["us-east-1"]

    def test_empty_available_passes_through(self):
        # When boto3 is absent (no available set), configured list passes through.
        out = resolve_source_regions(["us-east-1", "eu-west-1"], available=[])
        assert out == ["us-east-1", "eu-west-1"]

    def test_blank_entries_stripped(self):
        out = resolve_source_regions([" us-east-1 ", "", "  "], available=["us-east-1"])
        assert out == ["us-east-1"]


class TestDiscoverModelsOrchestrator:
    def test_multi_region_scan_with_injected_factory(self):
        def factory(region):
            return _make_client(
                summaries=[
                    _fm(_NOVA_ID, "Amazon", ["ON_DEMAND"], arn=_NOVA_ARN),
                    _fm(_GPTOSS120_ID, "OpenAI", ["ON_DEMAND"], arn=_GPTOSS120_ARN),
                ]
            )

        settings = BedrockDiscoverySettings(
            enabled=True, source_regions=["us-east-1", "eu-west-1"]
        )
        result = discover_models(settings=settings, client_factory=factory)
        # Two regions x two models = four per-(model,region) records.
        assert len(result.models) == 4
        assert {"Amazon", "OpenAI"} <= result.providers()

    def test_per_region_failure_isolated(self):
        # Use two REAL Bedrock control-plane regions so resolve_source_regions
        # (which intersects with get_available_regions('bedrock')) keeps both;
        # the factory then fails for one to exercise per-region try/except.
        def factory(region):
            if region == "eu-central-1":
                raise RuntimeError("AccessDenied")
            return _make_client(summaries=[_fm(_NOVA_ID, "Amazon", ["ON_DEMAND"])])

        settings = BedrockDiscoverySettings(
            enabled=True, source_regions=["us-east-1", "eu-central-1"]
        )
        result = discover_models(settings=settings, client_factory=factory)
        # Good region still produced a record; bad region recorded an error.
        assert len(result.models) == 1
        assert "eu-central-1" in result.region_errors
        assert "AccessDenied" in result.region_errors["eu-central-1"]

    def test_enabled_but_no_regions_resolved(self):
        settings = BedrockDiscoverySettings(enabled=True, source_regions=[])
        # No source regions and no resolvable default -> empty, no crash.
        result = discover_models(
            settings=settings, client_factory=lambda r: _make_client(summaries=[])
        )
        # source_regions empty -> resolve falls back to boto3 default which may be
        # None in a cred-free env; either way no live call and no models.
        assert isinstance(result, DiscoveryResult)
        assert result.models == []


class TestEndToEndModelList:
    def test_full_scan_to_litellm_entries(self):
        def factory(region):
            return _make_client(
                summaries=[
                    _fm(
                        _CLAUDE_ID, "Anthropic", ["INFERENCE_PROFILE"], arn=_CLAUDE_ARN
                    ),
                    _fm(_GPTOSS120_ID, "OpenAI", ["ON_DEMAND"], arn=_GPTOSS120_ARN),
                ],
                system_profiles=[
                    _profile(f"global.{_CLAUDE_ID}", [_CLAUDE_ARN]),
                    _profile(f"us.{_CLAUDE_ID}", [_CLAUDE_ARN]),
                ],
            )

        settings = BedrockDiscoverySettings(enabled=True, source_regions=["us-east-1"])
        result = discover_models(settings=settings, client_factory=factory)
        entries = result.to_litellm_model_list()
        by_model = {e["litellm_params"]["model"]: e for e in entries}
        # Claude has global -> global profile inline.
        assert f"bedrock/converse/global.{_CLAUDE_ID}" in by_model
        # gpt-oss has no global -> raw modelId, bedrock/ prefix preserved.
        assert f"bedrock/converse/{_GPTOSS120_ID}" in by_model

    def test_residency_constraint_changes_claude_to_geo(self):
        def factory(region):
            return _make_client(
                summaries=[
                    _fm(
                        _CLAUDE_ID, "Anthropic", ["INFERENCE_PROFILE"], arn=_CLAUDE_ARN
                    ),
                ],
                system_profiles=[
                    _profile(f"global.{_CLAUDE_ID}", [_CLAUDE_ARN]),
                    _profile(f"us.{_CLAUDE_ID}", [_CLAUDE_ARN]),
                ],
            )

        settings = BedrockDiscoverySettings(enabled=True, source_regions=["us-east-1"])
        result = discover_models(settings=settings, client_factory=factory)
        entries = result.to_litellm_model_list(residency_constraint=True)
        models = {e["litellm_params"]["model"] for e in entries}
        # Residency guard: global skipped, us. geo chosen.
        assert f"bedrock/converse/us.{_CLAUDE_ID}" in models
        assert f"bedrock/converse/global.{_CLAUDE_ID}" not in models
