"""Unit tests for Bedrock Marketplace/mantle endpoint discovery (RouteIQ-7105)
and model-catalogue drift detection (RouteIQ-9a42).

7105: ``ListMarketplaceModelEndpoints`` custom-deployment endpoint ARNs are
registered as ``model_list`` arms (``bedrock/<endpointArn>``), opt-in via
``include_marketplace_endpoints``; default OFF is byte-stable (no extra control
call, no extra arms).

9a42: a periodic diff of discovered (models_available) vs configured
(models_configured) fires a ``gateway.bedrock.catalogue_drift`` metric, gated.

All boto3 is MOCKED via an injected ``client_factory`` -- cred-free, no live AWS.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from litellm_llmrouter import bedrock_discovery as bd
from litellm_llmrouter.bedrock_discovery import (
    CatalogueDrift,
    DiscoveryResult,
    MarketplaceEndpoint,
    compute_catalogue_drift,
    discover_marketplace_endpoints,
    discover_models,
    record_catalogue_drift_metric,
    reset_drift_metric,
)
from litellm_llmrouter.settings import BedrockDiscoverySettings, reset_settings

# --- documented control-plane fixtures (mirror test_bedrock_discovery.py) ---

_FM_ARN = "arn:aws:bedrock:::foundation-model/amazon.nova-pro-v1:0"
_FM_ID = "amazon.nova-pro-v1:0"
_MP_ENDPOINT_ARN = (
    "arn:aws:bedrock:us-east-1:123456789012:marketplace-model-endpoint/my-llama-ep"
)
_MP_MODEL_SOURCE = (
    "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/meta-llama-3"
)


def _fm(model_id, provider, inference_types, arn):
    return {
        "modelId": model_id,
        "modelArn": arn,
        "providerName": provider,
        "inferenceTypesSupported": inference_types,
        "outputModalities": ["TEXT"],
        "modelLifecycle": {"status": "ACTIVE"},
    }


def _mp_summary(endpoint_arn, model_source, status="REGISTERED"):
    return {
        "endpointArn": endpoint_arn,
        "modelSourceIdentifier": model_source,
        "status": status,
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
    }


def _client_with_marketplace(endpoints):
    """A mocked bedrock control client: one FM + the given marketplace endpoints."""
    client = MagicMock()
    client.list_foundation_models.return_value = {
        "modelSummaries": [_fm(_FM_ID, "Amazon", ["ON_DEMAND"], _FM_ARN)]
    }
    client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}
    client.list_marketplace_model_endpoints.return_value = {
        "marketplaceModelEndpoints": endpoints
    }
    return client


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    reset_drift_metric()
    yield
    reset_settings()
    reset_drift_metric()


# ===========================================================================
# RouteIQ-7105: Marketplace / mantle endpoint discovery
# ===========================================================================


class TestMarketplaceDiscovery:
    def test_endpoint_enumerated_and_routable(self):
        client = _client_with_marketplace(
            [_mp_summary(_MP_ENDPOINT_ARN, _MP_MODEL_SOURCE)]
        )
        eps = discover_marketplace_endpoints(client, "us-east-1")
        assert len(eps) == 1
        assert eps[0].endpoint_arn == _MP_ENDPOINT_ARN
        assert eps[0].model_source == _MP_MODEL_SOURCE
        assert eps[0].is_routable is True

    def test_incompatible_endpoint_not_routable(self):
        ep = MarketplaceEndpoint(
            endpoint_arn=_MP_ENDPOINT_ARN,
            model_source=_MP_MODEL_SOURCE,
            region="us-east-1",
            status="INCOMPATIBLE_ENDPOINT",
        )
        assert ep.is_routable is False

    def test_missing_api_attr_is_empty(self):
        """A boto3 client predating the API yields no endpoints (no crash)."""
        client = MagicMock(spec=[])  # no list_marketplace_model_endpoints attr
        assert discover_marketplace_endpoints(client, "us-east-1") == []

    def test_paginated(self):
        client = MagicMock()
        page1 = {
            "marketplaceModelEndpoints": [
                _mp_summary(_MP_ENDPOINT_ARN + "-1", _MP_MODEL_SOURCE)
            ],
            "nextToken": "tok",
        }
        page2 = {
            "marketplaceModelEndpoints": [
                _mp_summary(_MP_ENDPOINT_ARN + "-2", _MP_MODEL_SOURCE)
            ]
        }
        client.list_marketplace_model_endpoints.side_effect = [page1, page2]
        eps = discover_marketplace_endpoints(client, "us-east-1")
        assert len(eps) == 2

    def test_disabled_default_no_marketplace_scan(self):
        """Default OFF: discover_models does NOT call the marketplace API."""
        client = _client_with_marketplace(
            [_mp_summary(_MP_ENDPOINT_ARN, _MP_MODEL_SOURCE)]
        )
        settings = BedrockDiscoverySettings(enabled=True, source_regions=["us-east-1"])
        result = discover_models(settings=settings, client_factory=lambda r: client)
        # FM still discovered; marketplace skipped (byte-stable).
        assert len(result.models) == 1
        assert result.marketplace_endpoints == []
        client.list_marketplace_model_endpoints.assert_not_called()

    def test_enabled_marketplace_arm_registered(self):
        """Opt-in: a mantle endpoint fixture -> registered model_list arm."""
        client = _client_with_marketplace(
            [_mp_summary(_MP_ENDPOINT_ARN, _MP_MODEL_SOURCE)]
        )
        settings = BedrockDiscoverySettings(
            enabled=True,
            source_regions=["us-east-1"],
            include_marketplace_endpoints=True,
        )
        result = discover_models(settings=settings, client_factory=lambda r: client)
        assert len(result.marketplace_endpoints) == 1

        entries = result.to_litellm_model_list()
        mp_entries = [
            e
            for e in entries
            if e["litellm_params"]["model"] == f"bedrock/{_MP_ENDPOINT_ARN}"
        ]
        assert len(mp_entries) == 1
        entry = mp_entries[0]
        assert entry["model_name"].startswith("marketplace.")
        assert entry["litellm_params"]["aws_region_name"] == "us-east-1"
        assert entry["model_info"]["marketplace_model_source"] == _MP_MODEL_SOURCE

    def test_marketplace_joins_auto_group(self):
        """Auto-group: the marketplace arm joins the shared group name."""
        client = _client_with_marketplace(
            [_mp_summary(_MP_ENDPOINT_ARN, _MP_MODEL_SOURCE)]
        )
        settings = BedrockDiscoverySettings(
            enabled=True,
            source_regions=["us-east-1"],
            include_marketplace_endpoints=True,
            auto_group=True,
            auto_group_name="claude-auto",
        )
        result = discover_models(settings=settings, client_factory=lambda r: client)
        entries = result.to_litellm_model_list(
            auto_group=True, auto_group_name="claude-auto"
        )
        names = {e["model_name"] for e in entries}
        assert names == {"claude-auto"}  # FM + marketplace arms share the group
        models = {e["litellm_params"]["model"] for e in entries}
        assert f"bedrock/{_MP_ENDPOINT_ARN}" in models


# ===========================================================================
# RouteIQ-9a42: model-catalogue drift detection
# ===========================================================================


class _RecordingMeter:
    """A minimal OTel-meter stand-in capturing counter .add() calls."""

    def __init__(self):
        self.added: list[tuple[float, dict]] = []

    def create_counter(self, **kwargs):
        meter = self

        class _Counter:
            def add(self, amount, attributes=None):
                meter.added.append((amount, attributes or {}))

        return _Counter()


class TestCatalogueDrift:
    def test_no_drift_when_aligned(self):
        client = _client_with_marketplace([])
        settings = BedrockDiscoverySettings(enabled=True, source_regions=["us-east-1"])
        result = discover_models(settings=settings, client_factory=lambda r: client)
        # The FM's selected raw on-demand id is the model_id (no profiles here).
        configured = [
            {
                "model_name": "nova",
                "litellm_params": {"model": f"bedrock/converse/{_FM_ID}"},
            }
        ]
        drift = compute_catalogue_drift(result, configured)
        assert drift.has_drift is False
        assert drift.drift_count == 0

    def test_missing_model_is_drift(self):
        """A discovered model absent from the configured list is 'missing' drift."""
        client = _client_with_marketplace([])
        settings = BedrockDiscoverySettings(enabled=True, source_regions=["us-east-1"])
        result = discover_models(settings=settings, client_factory=lambda r: client)
        drift = compute_catalogue_drift(result, configured_model_list=[])
        assert drift.has_drift is True
        assert _FM_ID in drift.missing
        assert drift.extra == frozenset()

    def test_extra_configured_is_drift(self):
        """A configured bedrock arm not in the discovered set is 'extra' drift."""
        result = DiscoveryResult()  # nothing discovered
        configured = [
            {
                "model_name": "stale",
                "litellm_params": {"model": "bedrock/converse/some.stale.model"},
            }
        ]
        drift = compute_catalogue_drift(result, configured)
        assert "some.stale.model" in drift.extra
        assert drift.missing == frozenset()

    def test_non_bedrock_arms_ignored(self):
        """Non-bedrock model strings are not part of the Bedrock catalogue."""
        result = DiscoveryResult()
        configured = [
            {"model_name": "gpt", "litellm_params": {"model": "openai/gpt-4o"}}
        ]
        drift = compute_catalogue_drift(result, configured)
        assert drift.drift_count == 0

    def test_metric_fires_on_drift(self, monkeypatch):
        """An injected drift -> the catalogue_drift counter fires by direction."""
        meter = _RecordingMeter()
        monkeypatch.setattr(bd, "get_meter", lambda: meter, raising=False)
        # Patch the lazy import target so _get_drift_counter uses our meter.
        import litellm_llmrouter.observability as obs

        monkeypatch.setattr(obs, "get_meter", lambda: meter, raising=False)

        drift = CatalogueDrift(
            available=frozenset({"a", "b"}),
            configured=frozenset({"a", "c"}),
        )
        assert "b" in drift.missing
        assert "c" in drift.extra
        record_catalogue_drift_metric(drift)

        directions = {attrs.get("direction") for _amt, attrs in meter.added}
        assert directions == {"missing", "extra"}
        # one missing (b) + one extra (c)
        total = sum(amt for amt, _ in meter.added)
        assert total == 2

    def test_metric_noop_when_no_meter(self, monkeypatch):
        """No OTel meter -> record is a clean no-op (never raises)."""
        import litellm_llmrouter.observability as obs

        def _raise():
            raise RuntimeError("not initialized")

        monkeypatch.setattr(obs, "get_meter", _raise, raising=False)
        drift = CatalogueDrift(available=frozenset({"x"}), configured=frozenset())
        # Must not raise.
        record_catalogue_drift_metric(drift)

    def test_metric_noop_when_no_drift(self, monkeypatch):
        """No drift -> counter not touched."""
        meter = _RecordingMeter()
        import litellm_llmrouter.observability as obs

        monkeypatch.setattr(obs, "get_meter", lambda: meter, raising=False)
        drift = CatalogueDrift(available=frozenset({"a"}), configured=frozenset({"a"}))
        record_catalogue_drift_metric(drift)
        assert meter.added == []
