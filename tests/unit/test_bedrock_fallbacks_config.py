"""RouteIQ-767f — per-model-group fallbacks survive the custom-strategy install.

RouteIQ installs its own ``CustomRoutingStrategyBase`` via
``router.set_custom_routing_strategy()``. That call ONLY rebinds the Router's
per-attempt deployment selectors (``get_available_deployment`` /
``async_get_available_deployment``). LiteLLM's MODEL-GROUP fallback machinery
(``fallbacks`` / ``context_window_fallbacks`` / ``content_policy_fallbacks``)
lives on a different layer — the completion wrapper
(``Router.async_function_with_fallbacks``) reads ``self.fallbacks`` etc., which
are set in ``Router.__init__`` from ``router_settings`` and are NEVER touched
by ``set_custom_routing_strategy``.

These tests prove, against a REAL ``litellm.Router`` (not a MagicMock):

  1. ``config/config.bedrock-fallbacks.yaml`` parses and declares the full
     triad (``fallbacks`` + ``context_window_fallbacks`` +
     ``content_policy_fallbacks``) with cross-region inference-profile arms, in
     LiteLLM's required shape (each fallback dict has exactly one key).
  2. ``router_settings`` flows those params into ``Router.__init__`` exactly the
     way ``litellm.proxy.proxy_server`` does (``get_valid_args`` filter), and
     they land on ``self.fallbacks`` / ``self.context_window_fallbacks`` /
     ``self.content_policy_fallbacks``.
  3. Installing RouteIQ's custom strategy does NOT drop those attributes — the
     deployment selectors are rebound to OUR strategy, yet the fallback config
     is byte-for-byte unchanged. This is the acceptance: ML routing is ADDITIVE
     to model-group fallbacks; RouteIQ does not strip upstream fallbacks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# litellm is a hard dependency; importorskip keeps collection honest if the env
# is somehow missing it, rather than erroring at import time.
litellm = pytest.importorskip("litellm")
yaml = pytest.importorskip("yaml")

from litellm import Router  # noqa: E402

from litellm_llmrouter.custom_routing_strategy import (  # noqa: E402
    RouteIQRoutingStrategy,
    install_routeiq_strategy,
)

# Repo root resolved from this test file: tests/unit/<file> -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "config.bedrock-fallbacks.yaml"


@pytest.fixture(scope="module")
def bedrock_fallbacks_config() -> dict:
    """Parsed ``config/config.bedrock-fallbacks.yaml``."""
    assert _CONFIG_PATH.is_file(), f"missing config: {_CONFIG_PATH}"
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_router_like_proxy(config: dict) -> Router:
    """Build a Router from ``router_settings`` the way the LiteLLM proxy does.

    Mirrors ``litellm.proxy.proxy_server`` (``get_valid_args`` filter): only keys
    that are valid ``Router.__init__`` args are forwarded. This is exactly the
    path RouteIQ relies on, so a regression where a fallback key got dropped /
    renamed upstream would surface here.
    """
    router_settings = config["router_settings"]
    valid_args = set(Router.get_valid_args()) - {"model_list", "search_tools"}
    router_params = {k: v for k, v in router_settings.items() if k in valid_args}
    return Router(
        model_list=[dict(d) for d in config["model_list"]],
        **router_params,
    )


# ---------------------------------------------------------------------------
# 1. Config parses + declares the full fallback triad in the required shape.
# ---------------------------------------------------------------------------


def test_config_declares_fallback_triad(bedrock_fallbacks_config: dict) -> None:
    rs = bedrock_fallbacks_config["router_settings"]

    for key in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks"):
        assert key in rs, f"{key} missing from router_settings"
        assert isinstance(rs[key], list) and rs[key], f"{key} must be a non-empty list"
        # LiteLLM's Router.validate_fallbacks: each entry is a 1-key dict.
        for entry in rs[key]:
            assert isinstance(entry, dict)
            assert len(entry) == 1, f"{key} entry must have exactly one key: {entry}"
            (targets,) = entry.values()
            assert isinstance(targets, list) and targets


def test_config_has_cross_region_arms(bedrock_fallbacks_config: dict) -> None:
    """The Sonnet primary fails over to a DIFFERENT region + an x-region profile."""
    rs = bedrock_fallbacks_config["router_settings"]
    sonnet_fb = next(
        targets
        for entry in rs["fallbacks"]
        for grp, targets in entry.items()
        if grp == "claude-sonnet-us-east-1"
    )
    assert "claude-sonnet-us-west-2" in sonnet_fb  # cross-region failover arm
    assert "claude-sonnet-us-xregion" in sonnet_fb  # x-region inference profile

    # The model_list backs every group referenced by the fallbacks, and the
    # cross-region arm uses a system-defined inference profile (us. prefix) in a
    # distinct region from the primary.
    groups = {
        m["model_name"]: m["litellm_params"]
        for m in bedrock_fallbacks_config["model_list"]
    }
    assert groups["claude-sonnet-us-east-1"]["aws_region_name"] == "us-east-1"
    assert groups["claude-sonnet-us-west-2"]["aws_region_name"] == "us-west-2"
    assert groups["claude-sonnet-us-xregion"]["model"].startswith(
        "bedrock/us.anthropic"
    )


# ---------------------------------------------------------------------------
# 2. router_settings -> Router.__init__ lands the params on the Router.
# ---------------------------------------------------------------------------


def test_router_loads_fallbacks_from_config(bedrock_fallbacks_config: dict) -> None:
    router = _build_router_like_proxy(bedrock_fallbacks_config)

    assert router.fallbacks == bedrock_fallbacks_config["router_settings"]["fallbacks"]
    assert (
        router.context_window_fallbacks
        == bedrock_fallbacks_config["router_settings"]["context_window_fallbacks"]
    )
    assert (
        router.content_policy_fallbacks
        == bedrock_fallbacks_config["router_settings"]["content_policy_fallbacks"]
    )


# ---------------------------------------------------------------------------
# 3. ACCEPTANCE: installing RouteIQ's custom strategy does NOT drop fallbacks.
# ---------------------------------------------------------------------------


def test_custom_strategy_install_preserves_fallbacks(
    bedrock_fallbacks_config: dict,
) -> None:
    router = _build_router_like_proxy(bedrock_fallbacks_config)

    # Snapshot the fallback config BEFORE the custom-strategy install.
    fb_before = list(router.fallbacks)
    cwf_before = list(router.context_window_fallbacks)
    cpf_before = list(router.content_policy_fallbacks)

    strategy = install_routeiq_strategy(router, strategy_name="llmrouter-knn")
    assert isinstance(strategy, RouteIQRoutingStrategy)

    # The deployment selectors WERE rebound to our strategy (custom ML routing
    # is active) ...
    assert (
        router.get_available_deployment.__func__
        is RouteIQRoutingStrategy.get_available_deployment
    )
    assert (
        router.async_get_available_deployment.__func__
        is RouteIQRoutingStrategy.async_get_available_deployment
    )

    # ... yet the model-group fallback config is byte-for-byte UNCHANGED. The
    # two layers compose: ML routing selects a deployment inside a group, while
    # these fallbacks reroute to another GROUP on 429 / context-overflow /
    # content-block.
    assert router.fallbacks == fb_before
    assert router.context_window_fallbacks == cwf_before
    assert router.content_policy_fallbacks == cpf_before
    assert router.fallbacks, "fallbacks must remain non-empty after install"


def test_fallback_lookup_resolves_after_install(
    bedrock_fallbacks_config: dict,
) -> None:
    """The Router can still resolve a model-group fallback target post-install.

    Uses LiteLLM's own fallback-lookup helper so this asserts the params remain
    *usable* by the fallback wrapper, not merely present as an attribute.
    """
    from litellm.router_utils.fallback_event_handlers import (
        get_fallback_model_group,
    )

    router = _build_router_like_proxy(bedrock_fallbacks_config)
    install_routeiq_strategy(router, strategy_name="llmrouter-knn")

    fallback_group, _idx = get_fallback_model_group(
        fallbacks=router.fallbacks,
        model_group="claude-sonnet-us-east-1",
    )
    assert fallback_group == [
        "claude-sonnet-us-west-2",
        "claude-sonnet-us-xregion",
    ]
