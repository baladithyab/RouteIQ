"""Unit tests for wiring ``AdapterLoaderPlugin`` into the gateway plugin loader.

RouteIQ-a089: ``adapters/loader.py`` was implemented + unit-tested but never
registered into ``gateway/app.py``'s ``_load_plugins_before_routes()``, so the
``routeiq.routing_adapters`` entry-point discovery never ran at startup and
``AdapterFrameworkSettings.entrypoint_discovery`` was inert.

These tests prove:
1. The loader plugin is registered (and an out-of-tree entry-point adapter is
   discovered + staged on startup) when ``entrypoint_discovery`` is ON.
2. The loader plugin is NOT registered and nothing changes when the flag is OFF
   (the default boot stays byte-stable).

The conftest ``autouse`` fixture resets the plugin manager, routing registry,
settings, and MLOps coordinator between tests, so each test starts clean.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from litellm_llmrouter.adapters.contract import (
    ADAPTER_API_VERSION,
    AdapterManifest,
)
from litellm_llmrouter.gateway.app import (
    _load_plugins_before_routes,
    _register_adapter_loader_plugin,
)
from litellm_llmrouter.gateway.plugin_manager import (
    PluginCapability,
    get_plugin_manager,
)
from litellm_llmrouter.strategy_registry import RoutingContext, get_routing_registry


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _OutOfTreeAdapter:
    """An out-of-tree routing adapter exposed via an entry point."""

    def declare_capabilities(self) -> AdapterManifest:
        return AdapterManifest(
            name="ext-adapter",
            version="1.0",
            adapter_api_version=ADAPTER_API_VERSION,
            capabilities={PluginCapability.ROUTING_STRATEGY},
        )

    def route(self, ctx: RoutingContext) -> Optional[dict]:
        deps = getattr(ctx.router, "model_list", [])
        return deps[0] if deps else None

    select_deployment = route

    def validate(self):
        return True, None


class _FakeEntryPoint:
    def __init__(self, name: str, factory: Any):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


class _FakeEntryPoints:
    """Mimics the Python 3.10+ ``EntryPoints`` object with ``.select``."""

    def __init__(self, mapping: dict[str, list[_FakeEntryPoint]]):
        self._mapping = mapping

    def select(self, group: str) -> list[_FakeEntryPoint]:
        return list(self._mapping.get(group, []))


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the entry-points lookup used inside ``adapters.loader``."""
    import litellm_llmrouter.adapters.loader as loader_mod

    eps = _FakeEntryPoints(
        {
            "routeiq.routing_adapters": [
                _FakeEntryPoint("ext", lambda: _OutOfTreeAdapter()),
            ]
        }
    )
    monkeypatch.setattr(loader_mod.importlib.metadata, "entry_points", lambda: eps)


# ---------------------------------------------------------------------------
# Flag ON: loader registered + entry-point adapter discovered at startup
# ---------------------------------------------------------------------------


def test_loader_registered_when_flag_on(monkeypatch: pytest.MonkeyPatch):
    from litellm_llmrouter.settings import reset_settings

    monkeypatch.setenv("ROUTEIQ_ADAPTER_FRAMEWORK__ENTRYPOINT_DISCOVERY", "true")
    reset_settings()

    loaded = _load_plugins_before_routes()

    manager = get_plugin_manager()
    names = [p.name for p in manager.plugins]
    assert "adapter-loader" in names
    assert loaded >= 1


@pytest.mark.asyncio
async def test_entry_point_adapter_discovered_on_startup(
    monkeypatch: pytest.MonkeyPatch,
):
    from litellm_llmrouter.settings import reset_settings

    monkeypatch.setenv("ROUTEIQ_ADAPTER_FRAMEWORK__ENTRYPOINT_DISCOVERY", "true")
    reset_settings()
    _patch_entry_points(monkeypatch)

    _load_plugins_before_routes()
    manager = get_plugin_manager()

    # Run the plugin lifecycle (startup triggers entry-point discovery).
    await manager.startup(app=None)

    # The out-of-tree adapter was staged + promoted into the routing registry.
    registry = get_routing_registry()
    assert registry.get("ext-adapter") is not None

    await manager.shutdown(app=None)


def test_capability_negotiation_restricts_allowlist(
    monkeypatch: pytest.MonkeyPatch,
):
    from litellm_llmrouter.settings import reset_settings

    monkeypatch.setenv("ROUTEIQ_ADAPTER_FRAMEWORK__ENTRYPOINT_DISCOVERY", "true")
    monkeypatch.setenv("ROUTEIQ_ADAPTER_FRAMEWORK__CAPABILITY_NEGOTIATION", "true")
    reset_settings()

    manager = get_plugin_manager()
    assert _register_adapter_loader_plugin(manager) is True

    loader = next(p for p in manager.plugins if p.name == "adapter-loader")
    # With negotiation on, the loader only allows routing-strategy adapters.
    assert loader._allowed_capabilities == {PluginCapability.ROUTING_STRATEGY}


# ---------------------------------------------------------------------------
# Flag OFF (default): loader NOT registered, nothing changes
# ---------------------------------------------------------------------------


def test_loader_not_registered_when_flag_off(monkeypatch: pytest.MonkeyPatch):
    from litellm_llmrouter.settings import reset_settings

    monkeypatch.setenv("ROUTEIQ_ADAPTER_FRAMEWORK__ENTRYPOINT_DISCOVERY", "false")
    reset_settings()

    loaded = _load_plugins_before_routes()

    manager = get_plugin_manager()
    names = [p.name for p in manager.plugins]
    assert "adapter-loader" not in names
    # No env-configured plugins either => nothing loaded.
    assert loaded == 0


def test_loader_default_off_no_entry_point_enumeration(
    monkeypatch: pytest.MonkeyPatch,
):
    from litellm_llmrouter.settings import reset_settings

    # Default settings (flag absent) => discovery off => loader is a no-op and
    # the entry-point lookup must never run.
    reset_settings()

    called = {"n": 0}

    def _boom() -> Any:
        called["n"] += 1
        raise AssertionError("entry_points() must not be called when flag is off")

    import litellm_llmrouter.adapters.loader as loader_mod

    monkeypatch.setattr(loader_mod.importlib.metadata, "entry_points", _boom)

    manager = get_plugin_manager()
    assert _register_adapter_loader_plugin(manager) is False
    names = [p.name for p in manager.plugins]
    assert "adapter-loader" not in names
    assert called["n"] == 0
