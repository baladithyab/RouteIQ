"""Test that routeiq-routing works without the full gateway."""

import pytest


def test_standalone_import():
    """StandaloneCentroidRouter can be instantiated without the full gateway."""
    from routeiq_routing._centroid_standalone import StandaloneCentroidRouter

    router = StandaloneCentroidRouter()
    assert router is not None


def test_model_costs():
    """MODEL_COSTS contains expected top models."""
    from routeiq_routing._centroid_standalone import MODEL_COSTS

    assert "gpt-4o" in MODEL_COSTS
    assert "gpt-4o-mini" in MODEL_COSTS
    assert "claude-3-5-sonnet-20241022" in MODEL_COSTS
    assert "gemini-2.0-flash" in MODEL_COSTS
    assert "deepseek-chat" in MODEL_COSTS
    # Verify cost structure
    assert "input" in MODEL_COSTS["gpt-4o"]
    assert "output" in MODEL_COSTS["gpt-4o"]


def test_model_context_windows():
    """MODEL_CONTEXT_WINDOWS contains expected models."""
    from routeiq_routing._centroid_standalone import MODEL_CONTEXT_WINDOWS

    assert "gpt-4o" in MODEL_CONTEXT_WINDOWS
    assert MODEL_CONTEXT_WINDOWS["gpt-4o"] == 128_000
    assert MODEL_CONTEXT_WINDOWS["gpt-4.1"] == 1_050_000


def test_routing_profile():
    """RoutingProfile enum has expected values."""
    from routeiq_routing._centroid_standalone import RoutingProfile

    assert RoutingProfile.ECO == "eco"
    assert RoutingProfile.PREMIUM == "premium"
    assert RoutingProfile.AUTO == "auto"


def test_estimate_token_count():
    """Token estimation uses ~4 chars per token heuristic."""
    from routeiq_routing._centroid_standalone import estimate_token_count

    messages = [{"role": "user", "content": "a" * 400}]
    assert estimate_token_count(messages) == 100


def test_check_context_window():
    """Context window check works with known models."""
    from routeiq_routing._centroid_standalone import check_context_window

    short_messages = [{"role": "user", "content": "hello"}]
    assert check_context_window("gpt-4o", short_messages) is True

    # Unknown model uses 128k default
    assert check_context_window("unknown-model", short_messages) is True


def test_select_deployment_eco_prefers_cheapest():
    """Eco profile should prefer the cheapest model."""
    from routeiq_routing._centroid_standalone import StandaloneCentroidRouter

    router = StandaloneCentroidRouter(profile="eco")
    deployments = [
        {"model_name": "my-model", "litellm_params": {"model": "gpt-4o"}},
        {"model_name": "my-model", "litellm_params": {"model": "gpt-4o-mini"}},
    ]
    result = router.select_deployment(
        deployments, messages=[{"role": "user", "content": "hi"}]
    )
    assert result is not None
    # Eco should prefer the mini (cheaper) model
    assert result["litellm_params"]["model"] == "gpt-4o-mini"


def test_select_deployment_premium_prefers_capable():
    """Premium profile should prefer the more capable model."""
    from routeiq_routing._centroid_standalone import StandaloneCentroidRouter

    router = StandaloneCentroidRouter(profile="premium")
    deployments = [
        {"model_name": "my-model", "litellm_params": {"model": "gpt-4o-mini"}},
        {"model_name": "my-model", "litellm_params": {"model": "gpt-4o"}},
    ]
    result = router.select_deployment(
        deployments, messages=[{"role": "user", "content": "hi"}]
    )
    assert result is not None
    # Premium should prefer the full model (complex tier)
    assert result["litellm_params"]["model"] == "gpt-4o"


def test_select_deployment_auto_short_prompt():
    """Auto profile with a short prompt should prefer simple tier."""
    from routeiq_routing._centroid_standalone import StandaloneCentroidRouter

    router = StandaloneCentroidRouter(profile="auto")
    deployments = [
        {"model_name": "my-model", "litellm_params": {"model": "gpt-4o"}},
        {"model_name": "my-model", "litellm_params": {"model": "gpt-4o-mini"}},
    ]
    result = router.select_deployment(
        deployments, messages=[{"role": "user", "content": "hi"}]
    )
    assert result is not None
    # Short prompt -> simple tier -> mini
    assert result["litellm_params"]["model"] == "gpt-4o-mini"


def test_select_deployment_auto_long_prompt():
    """Auto profile with a long prompt should prefer complex tier."""
    from routeiq_routing._centroid_standalone import StandaloneCentroidRouter

    router = StandaloneCentroidRouter(profile="auto")
    deployments = [
        {"model_name": "my-model", "litellm_params": {"model": "gpt-4o"}},
        {"model_name": "my-model", "litellm_params": {"model": "gpt-4o-mini"}},
    ]
    # >2000 tokens = >8000 chars
    long_content = "x" * 10_000
    result = router.select_deployment(
        deployments, messages=[{"role": "user", "content": long_content}]
    )
    assert result is not None
    # Long prompt -> complex tier -> full model
    assert result["litellm_params"]["model"] == "gpt-4o"


def test_select_deployment_empty_returns_none():
    """Empty deployment list returns None."""
    from routeiq_routing._centroid_standalone import StandaloneCentroidRouter

    router = StandaloneCentroidRouter()
    result = router.select_deployment([], messages=[])
    assert result is None


def test_select_deployment_with_tier_mapping():
    """Explicit tier_mapping overrides name heuristics."""
    from routeiq_routing._centroid_standalone import StandaloneCentroidRouter

    router = StandaloneCentroidRouter(
        profile="eco",
        tier_mapping={"simple": ["deepseek"], "complex": ["gpt-4o"]},
    )
    deployments = [
        {"model_name": "my-model", "litellm_params": {"model": "gpt-4o"}},
        {"model_name": "my-model", "litellm_params": {"model": "deepseek-chat"}},
    ]
    result = router.select_deployment(
        deployments, messages=[{"role": "user", "content": "hi"}]
    )
    assert result is not None
    assert result["litellm_params"]["model"] == "deepseek-chat"


def test_invalid_profile_defaults_to_auto():
    """Invalid profile string falls back to auto."""
    from routeiq_routing._centroid_standalone import (
        RoutingProfile,
        StandaloneCentroidRouter,
    )

    router = StandaloneCentroidRouter(profile="invalid")
    assert router._profile == RoutingProfile.AUTO


def test_init_import_without_full_gateway():
    """The package __init__ exposes StandaloneCentroidRouter."""
    from routeiq_routing import StandaloneCentroidRouter

    router = StandaloneCentroidRouter()
    assert router is not None
