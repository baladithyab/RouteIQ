"""
Hot Reload API Endpoints
========================

Provides API endpoints for triggering config and router model reloads.
These endpoints integrate with LiteLLM's proxy server.

Usage:
    POST /router/reload - Reload a specific routing strategy
    POST /config/reload - Reload the entire configuration
    GET /config/sync/status - Get config sync status
"""

import os
import signal
from typing import Any

from litellm._logging import verbose_proxy_logger

from .config_sync import get_sync_manager


class HotReloadManager:
    """Manages hot reload operations for config and routing models."""

    def __init__(self):
        self._router_reload_callbacks: dict[str, Any] = {}

    def register_router_reload_callback(
        self, strategy_name: str, callback: Any
    ) -> None:
        """Register a callback for reloading a specific strategy."""
        self._router_reload_callbacks[strategy_name] = callback
        verbose_proxy_logger.debug(
            f"Registered reload callback for strategy: {strategy_name}"
        )

    def reload_router(self, strategy: str | None = None) -> dict[str, Any]:
        """Reload a routing strategy or all strategies.

        Args:
            strategy: Specific strategy to reload, or None for all.

        Returns:
            Dict with reload status and details.
        """
        reloaded = []
        errors = []

        if strategy:
            # Reload specific strategy
            if strategy in self._router_reload_callbacks:
                try:
                    self._router_reload_callbacks[strategy]()
                    reloaded.append(strategy)
                    verbose_proxy_logger.info(f"Reloaded router strategy: {strategy}")
                except Exception as e:
                    errors.append({"strategy": strategy, "error": str(e)})
                    verbose_proxy_logger.error(
                        f"Failed to reload strategy {strategy}: {e}"
                    )
            else:
                errors.append(
                    {
                        "strategy": strategy,
                        "error": "Strategy not found or not hot-reloadable",
                    }
                )
        else:
            # Reload all strategies
            for name, callback in self._router_reload_callbacks.items():
                try:
                    callback()
                    reloaded.append(name)
                    verbose_proxy_logger.info(f"Reloaded router strategy: {name}")
                except Exception as e:
                    errors.append({"strategy": name, "error": str(e)})
                    verbose_proxy_logger.error(f"Failed to reload strategy {name}: {e}")

        return {
            "status": "success" if not errors else "partial" if reloaded else "failed",
            "reloaded": reloaded,
            "errors": errors,
        }

    def reload_config(self, force_sync: bool = False) -> dict[str, Any]:
        """Reload the configuration.

        Args:
            force_sync: If True, force sync from S3/GCS before reload.

        Returns:
            Dict with reload status.
        """
        try:
            sync_manager = get_sync_manager()

            # Force sync if requested
            if force_sync:
                synced = sync_manager.force_sync()
                if synced:
                    verbose_proxy_logger.info("Config synced from remote storage")

            # Send SIGHUP to trigger LiteLLM's config reload
            os.kill(os.getpid(), signal.SIGHUP)

            verbose_proxy_logger.info("Config reload triggered via SIGHUP")

            return {
                "status": "success",
                "message": "Config reload triggered",
                "synced_from_remote": force_sync,
            }
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to reload config: {e}")
            return {"status": "failed", "error": str(e)}

    def get_config_sync_status(self) -> dict[str, Any]:
        """Get the current config sync status."""
        sync_manager = get_sync_manager()
        return sync_manager.get_status()


# Singleton instance
_hot_reload_manager: HotReloadManager | None = None


def get_hot_reload_manager() -> HotReloadManager:
    """Get the global hot reload manager instance."""
    global _hot_reload_manager
    if _hot_reload_manager is None:
        _hot_reload_manager = HotReloadManager()
    return _hot_reload_manager
