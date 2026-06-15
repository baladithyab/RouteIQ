"""
Entry-Point Adapter Loader
==========================

``AdapterLoaderPlugin`` discovers out-of-tree routing adapters via the
``routeiq.routing_adapters`` Python entry-point group and stages them into the
routing registry (validate-then-promote). It reuses the ``GatewayPlugin``
lifecycle, the registry's staged-loading gate, and the capability allowlist.

Each discovery step is wrapped in try/except and skips on failure — the loader
never raises out of ``startup`` (a malformed third-party package must not take
down the gateway).

Design reference:
``docs/architecture/aws-rearchitecture/40-pluggable-routing-and-mlops.md`` §2.4.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

from litellm_llmrouter.adapters.contract import (
    ADAPTER_API_VERSION,
    AdapterManifest,
    _abi_compatible,
    attach_route_alias,
)
from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "routeiq.routing_adapters"


class AdapterLoaderPlugin(GatewayPlugin):
    """A ``GatewayPlugin`` that discovers + stages out-of-tree routing adapters."""

    def __init__(self, allowed_capabilities: Optional[set[PluginCapability]] = None):
        # None => allow all (matches the "no declared capabilities = all checks
        # pass" plugin-manager convention).
        self._allowed_capabilities = allowed_capabilities
        self._staged_names: list[str] = []

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="adapter-loader",
            version="1.0.0",
            capabilities={PluginCapability.ROUTING_STRATEGY},
            description="Discovers routing adapters via entry-points and stages them.",
        )

    def _capabilities_allowed(self, caps: set[PluginCapability]) -> bool:
        if self._allowed_capabilities is None:
            return True
        return caps.issubset(self._allowed_capabilities)

    def _iter_entry_points(self) -> Any:
        """Enumerate entry points for the adapter group (version-portable)."""
        try:
            eps: Any = importlib.metadata.entry_points()
            select = getattr(eps, "select", None)
            if callable(select):  # Python 3.10+ EntryPoints API
                return list(select(group=ENTRY_POINT_GROUP))
            # Legacy dict API (Python <3.10)
            return list(eps.get(ENTRY_POINT_GROUP, []))
        except Exception as e:
            logger.debug("Adapter entry-point enumeration failed: %s", e)
            return []

    def _stage_one(self, ep: Any, context: Optional[PluginContext]) -> bool:
        """Stage a single discovered adapter. Never raises."""
        try:
            adapter = ep.load()()
        except Exception as e:
            logger.warning("Adapter %r failed to load: %s", getattr(ep, "name", "?"), e)
            return False

        declare = getattr(adapter, "declare_capabilities", None)
        if not callable(declare):
            logger.warning(
                "Adapter %r has no declare_capabilities; skipping",
                getattr(ep, "name", "?"),
            )
            return False
        try:
            manifest: AdapterManifest = declare()
        except Exception as e:
            logger.warning(
                "Adapter %r manifest failed: %s", getattr(ep, "name", "?"), e
            )
            return False

        # 1. ABI negotiation
        if not _abi_compatible(manifest.adapter_api_version, ADAPTER_API_VERSION):
            logger.warning(
                "Skip adapter %r: incompatible ABI %s (gateway %s)",
                manifest.name,
                manifest.adapter_api_version,
                ADAPTER_API_VERSION,
            )
            return False

        # 2. Capability allowlist
        if not self._capabilities_allowed(manifest.capabilities):
            logger.warning(
                "Skip adapter %r: capabilities %s not allowed",
                manifest.name,
                manifest.capabilities,
            )
            return False

        # 3. Ensure the route() alias so the strategy rides the pipeline.
        attach_route_alias(adapter)

        # 4. Stage -> validate -> promote via the registry's staged-loading gate.
        try:
            from litellm_llmrouter.strategy_registry import get_routing_registry

            registry = get_routing_registry()
            ok, error = registry.stage_strategy(
                manifest.name,
                adapter,
                version=manifest.version,
                metadata={"manifest": asdict(manifest)},
            )
            if not ok:
                logger.warning("Adapter %r staging failed: %s", manifest.name, error)
                return False
        except Exception as e:
            logger.warning("Adapter %r staging error: %s", manifest.name, e)
            return False

        # 5. If it learns, register it into the MLOps coordinator.
        if manifest.learns:
            try:
                from litellm_llmrouter.adapters.mlops import get_mlops_coordinator

                get_mlops_coordinator().register_learning_adapter(
                    manifest.name, adapter
                )
            except Exception as e:
                logger.debug("MLOps registration for %r skipped: %s", manifest.name, e)

        self._staged_names.append(manifest.name)
        logger.info("Staged out-of-tree adapter %r", manifest.name)
        return True

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        for ep in self._iter_entry_points():
            self._stage_one(ep, context)

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        # Staged adapters are owned by the registry, which is reset on its own
        # lifecycle; nothing to tear down here.
        self._staged_names.clear()
