"""
Hot Reload API Endpoints
========================

Provides API endpoints for triggering config and router model reloads.
These endpoints integrate with LiteLLM's proxy server and the strategy registry.

Usage:
    POST /router/reload - Reload a specific routing strategy
    POST /config/reload - Reload the entire configuration
    POST /strategy/weights - Update A/B strategy weights at runtime
    POST /strategy/stage - Stage a new strategy for validation
    POST /strategy/promote - Promote a staged strategy to active
    GET /config/sync/status - Get config sync status
    GET /strategy/status - Get current strategy registry status
"""

import os
import signal
from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger

from .config_sync import get_sync_manager


class HotReloadManager:
    """Manages hot reload operations for config and routing models."""

    def __init__(self):
        self._router_reload_callbacks: dict[str, Any] = {}
        self._strategy_registry = None
        self._initialized = False

    def _ensure_registry(self):
        """Lazily initialize strategy registry reference."""
        if not self._initialized:
            from .strategy_registry import get_routing_registry

            self._strategy_registry = get_routing_registry()

            # Register ourselves as update callback
            self._strategy_registry.add_update_callback(self._on_registry_update)
            self._initialized = True
        return self._strategy_registry

    def _on_registry_update(self):
        """Callback when strategy registry is updated."""
        verbose_proxy_logger.info("Strategy registry updated via hot reload")

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
            # Log full error server-side but return sanitized response
            verbose_proxy_logger.error(
                f"Failed to reload config: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return {"status": "failed", "error": "config_reload_failed"}

    def get_config_sync_status(self) -> dict[str, Any]:
        """Get the current config sync status."""
        sync_manager = get_sync_manager()
        return sync_manager.get_status()

    def get_router_info(self) -> dict[str, Any]:
        """Get information about the current routing configuration."""
        strategies = list(self._router_reload_callbacks.keys())

        # Include strategy registry status
        registry_status = {}
        if self._ensure_registry():
            registry_status = self._strategy_registry.get_status()

        return {
            "registered_strategies": strategies,
            "strategy_count": len(strategies),
            "hot_reload_enabled": len(strategies) > 0,
            "registry": registry_status,
        }

    # === A/B Testing / Experiment Management ===

    def update_strategy_weights(
        self,
        weights: Dict[str, int],
        experiment_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Update A/B strategy weights at runtime.

        This allows changing traffic split between strategies without
        reloading models or restarting the service.

        Args:
            weights: Dict mapping strategy names to relative weights.
                     Example: {"baseline": 90, "candidate": 10}
            experiment_id: Optional experiment identifier for telemetry.

        Returns:
            Dict with update status.
        """
        registry = self._ensure_registry()

        if not registry:
            return {"status": "failed", "error": "Strategy registry not available"}

        try:
            success = registry.set_weights(weights, experiment_id)

            if success:
                verbose_proxy_logger.info(
                    f"Updated A/B weights: {weights} (experiment={experiment_id})"
                )
                return {
                    "status": "success",
                    "weights": weights,
                    "experiment_id": experiment_id,
                    "current_status": registry.get_status(),
                }
            else:
                return {
                    "status": "failed",
                    "error": "One or more strategies not registered",
                    "registered_strategies": registry.list_strategies(),
                }
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to update weights: {e}")
            return {"status": "failed", "error": str(e)}

    def set_active_strategy(self, strategy_name: str) -> dict[str, Any]:
        """
        Set a single active routing strategy (disable A/B testing).

        Args:
            strategy_name: Name of strategy to activate.

        Returns:
            Dict with status.
        """
        registry = self._ensure_registry()

        if not registry:
            return {"status": "failed", "error": "Strategy registry not available"}

        try:
            success = registry.set_active(strategy_name)

            if success:
                verbose_proxy_logger.info(f"Set active strategy: {strategy_name}")
                return {
                    "status": "success",
                    "active_strategy": strategy_name,
                    "ab_disabled": True,
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Strategy '{strategy_name}' not registered",
                    "registered_strategies": registry.list_strategies(),
                }
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to set active strategy: {e}")
            return {"status": "failed", "error": str(e)}

    def clear_ab_weights(self) -> dict[str, Any]:
        """
        Clear A/B weights and revert to single active strategy.

        Returns:
            Dict with status.
        """
        registry = self._ensure_registry()

        if not registry:
            return {"status": "failed", "error": "Strategy registry not available"}

        try:
            registry.clear_weights()
            verbose_proxy_logger.info("Cleared A/B weights")
            return {
                "status": "success",
                "ab_disabled": True,
                "current_status": registry.get_status(),
            }
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to clear weights: {e}")
            return {"status": "failed", "error": str(e)}

    # === Staged Strategy Management ===

    def stage_strategy_config(
        self,
        config: Dict[str, Any],
        auto_promote: bool = False,
    ) -> dict[str, Any]:
        """
        Stage a new strategy configuration for validation.

        The strategy will be validated but not activated until
        explicitly promoted via promote_staged_strategy().

        Args:
            config: Strategy configuration dict with required keys:
                    - name: Strategy identifier
                    - strategy_type: Type of strategy (e.g., 'llmrouter-knn')
                    - version: Optional version string
                    - model_path: Path to model file (if applicable)
                    - Additional strategy-specific config
            auto_promote: Whether to auto-promote after successful validation.

        Returns:
            Dict with staging status and validation result.
        """
        registry = self._ensure_registry()

        if not registry:
            return {"status": "failed", "error": "Strategy registry not available"}

        name = config.get("name")
        if not name:
            return {"status": "failed", "error": "Strategy name required"}

        strategy_type = config.get("strategy_type", "llmrouter-knn")
        version = config.get("version")

        try:
            # Create strategy instance based on type
            from .strategies import LLMRouterStrategyFamily

            strategy_args = {
                k: v
                for k, v in config.items()
                if k not in ("name", "strategy_type", "version")
            }

            strategy = LLMRouterStrategyFamily(
                strategy_name=strategy_type,
                **strategy_args,
            )

            # Stage the strategy
            success, error = registry.stage_strategy(
                name=name,
                strategy=strategy,
                version=version,
                auto_promote=auto_promote,
                metadata={"config": config},
            )

            if success:
                verbose_proxy_logger.info(
                    f"Staged strategy: {name} (version={version})"
                )
                return {
                    "status": "success",
                    "staged": name,
                    "version": version,
                    "validation_passed": True,
                    "auto_promoted": auto_promote,
                }
            else:
                verbose_proxy_logger.warning(f"Strategy staging failed: {error}")
                return {
                    "status": "validation_failed",
                    "staged": name,
                    "error": error,
                }

        except Exception as e:
            verbose_proxy_logger.error(f"Failed to stage strategy: {e}")
            return {"status": "failed", "error": str(e)}

    def promote_staged_strategy(self, name: str) -> dict[str, Any]:
        """
        Promote a staged strategy to active.

        Args:
            name: Name of the staged strategy to promote.

        Returns:
            Dict with promotion status.
        """
        registry = self._ensure_registry()

        if not registry:
            return {"status": "failed", "error": "Strategy registry not available"}

        try:
            success, error = registry.promote_staged(name)

            if success:
                verbose_proxy_logger.info(f"Promoted staged strategy: {name}")
                return {
                    "status": "success",
                    "promoted": name,
                    "current_status": registry.get_status(),
                }
            else:
                return {
                    "status": "failed",
                    "error": error,
                }
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to promote strategy: {e}")
            return {"status": "failed", "error": str(e)}

    def rollback_staged_strategy(self, name: str) -> dict[str, Any]:
        """
        Rollback (discard) a staged strategy.

        Args:
            name: Name of the staged strategy to discard.

        Returns:
            Dict with rollback status.
        """
        registry = self._ensure_registry()

        if not registry:
            return {"status": "failed", "error": "Strategy registry not available"}

        try:
            success = registry.rollback_staged(name)

            if success:
                verbose_proxy_logger.info(f"Rolled back staged strategy: {name}")
                return {"status": "success", "rolled_back": name}
            else:
                return {
                    "status": "failed",
                    "error": f"No staged strategy named '{name}'",
                }
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to rollback strategy: {e}")
            return {"status": "failed", "error": str(e)}

    def get_strategy_status(self) -> dict[str, Any]:
        """
        Get current strategy registry status including A/B config and staged strategies.

        Returns:
            Dict with complete strategy status.
        """
        registry = self._ensure_registry()

        if not registry:
            return {"status": "failed", "error": "Strategy registry not available"}

        try:
            status: dict[str, Any] = registry.get_status()

            # Add staged strategy details
            staged_details = {}
            for name in registry.list_staged():
                staged = registry.get_staged(name)
                if staged:
                    staged_details[name] = {
                        "staged_at": staged.staged_at,
                        "validation_passed": staged.validation_passed,
                        "validation_error": staged.validation_error,
                        "auto_promote": staged.auto_promote,
                    }

            status["staged_details"] = staged_details
            return status

        except Exception as e:
            verbose_proxy_logger.error(f"Failed to get strategy status: {e}")
            return {"status": "failed", "error": str(e)}

    def reload_strategy_config(self, config: Dict[str, Any]) -> dict[str, Any]:
        """
        Reload strategy registry configuration from a config dict.

        Used by config hot reload to update weights/experiment without restart.

        Args:
            config: Dict with optional keys: weights, experiment, active_strategy

        Returns:
            Dict with reload status.
        """
        registry = self._ensure_registry()

        if not registry:
            return {"status": "failed", "error": "Strategy registry not available"}

        try:
            success, errors = registry.reload_from_config(config)

            if success:
                verbose_proxy_logger.info("Strategy config reloaded successfully")
                return {
                    "status": "success",
                    "current_status": registry.get_status(),
                }
            else:
                verbose_proxy_logger.warning(
                    f"Strategy config reload had errors: {errors}"
                )
                return {
                    "status": "partial" if registry.get_status() else "failed",
                    "errors": errors,
                    "current_status": registry.get_status(),
                }
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to reload strategy config: {e}")
            return {"status": "failed", "error": str(e)}


# Singleton instance
_hot_reload_manager: HotReloadManager | None = None


def get_hot_reload_manager() -> HotReloadManager:
    """Get the global hot reload manager instance."""
    global _hot_reload_manager
    if _hot_reload_manager is None:
        _hot_reload_manager = HotReloadManager()
    return _hot_reload_manager


def reset_hot_reload_manager() -> None:
    """Reset the global hot reload manager (for testing)."""
    global _hot_reload_manager
    _hot_reload_manager = None
