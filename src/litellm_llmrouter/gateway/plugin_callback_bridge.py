"""
Plugin Callback Bridge: LiteLLM Callback → Plugin Hook Integration
===================================================================

Bridges LiteLLM's callback system to GatewayPlugin LLM lifecycle hooks.

LiteLLM calls methods on objects in ``litellm.callbacks`` at various points
in the LLM call lifecycle. This bridge translates those callbacks into
GatewayPlugin hook invocations:

    litellm.log_pre_api_call  →  plugin.on_llm_pre_call
    litellm.log_success_event →  plugin.on_llm_success
    litellm.log_failure_event →  plugin.on_llm_failure

In addition to plugin hooks, the bridge integrates the guardrail policy
engine (``guardrail_policies.py``) at two points:

    async_log_pre_api_call    →  input guardrails  (deny/log/alert)
    async_log_success_event   →  output guardrails (log/alert only)

Registration:
    Called during plugin startup in app.py. The bridge is appended to
    ``litellm.callbacks`` alongside the existing RouterDecisionCallback.

Design:
    - Uses duck-typing (no CustomLogger subclass) for minimal coupling
    - Plugin hook failures are caught and logged, never crash the LLM call
    - Plugins are called in priority order
    - Guardrail policy evaluation runs after plugin hooks (plugins may modify kwargs)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from litellm_llmrouter.gateway.plugins.guardrails_base import GuardrailBlockError

logger = logging.getLogger(__name__)


class PluginCallbackBridge:
    """
    LiteLLM callback that bridges to GatewayPlugin LLM lifecycle hooks.

    Implements the LiteLLM callback interface via duck-typing (same pattern
    as RouterDecisionCallback). Delegates to plugins that override
    on_llm_pre_call, on_llm_success, or on_llm_failure.
    """

    def __init__(self, plugins: list[Any] | None = None) -> None:
        """
        Args:
            plugins: List of GatewayPlugin instances with LLM lifecycle hooks.
                     Can be set later via set_plugins().
        """
        self._plugins: list[Any] = plugins or []

    def set_plugins(self, plugins: list[Any]) -> None:
        """Update the list of callback-capable plugins."""
        self._plugins = plugins
        if plugins:
            names = [p.name for p in plugins]
            logger.info(
                f"PluginCallbackBridge active with {len(plugins)} plugins: {names}"
            )

    # =========================================================================
    # Synchronous hooks (LiteLLM calls these on the main thread)
    # =========================================================================

    def log_pre_api_call(
        self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> None:
        """Called before each LLM API call (sync path)."""
        # Sync hooks delegate to async via the event loop if available,
        # but most LiteLLM proxy paths use the async variants below.
        pass

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after successful LLM API call (sync path)."""
        pass

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after failed LLM API call (sync path)."""
        pass

    # =========================================================================
    # Async hooks (LiteLLM proxy calls these in the async path)
    # =========================================================================

    async def async_log_pre_api_call(
        self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> None:
        """Called before each LLM API call (async path).

        Runs plugin hooks first, then evaluates input guardrail policies.
        If any guardrail with action=DENY fails, raises HTTPException(446).
        """
        # --- Plugin hooks ---
        for plugin in self._plugins:
            try:
                result = await plugin.on_llm_pre_call(model, messages, kwargs)
                if isinstance(result, dict):
                    # Merge overrides into kwargs
                    kwargs.update(result)
            except GuardrailBlockError:
                raise  # Let guardrail blocks propagate as request failures
            except Exception as e:
                logger.error(
                    f"Plugin '{plugin.name}' on_llm_pre_call failed: {e}",
                    exc_info=True,
                )

        # --- Input guardrail policy evaluation ---
        await self._evaluate_input_guardrails(model, messages, kwargs)

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after successful LLM API call (async path).

        Runs plugin hooks first, then evaluates output guardrail policies.
        Output guardrails log violations but do NOT block (response is already sent).
        """
        model = kwargs.get("model", "unknown")
        for plugin in self._plugins:
            try:
                await plugin.on_llm_success(model, response_obj, kwargs)
            except Exception as e:
                logger.error(
                    f"Plugin '{plugin.name}' on_llm_success failed: {e}",
                    exc_info=True,
                )

        # --- Output guardrail policy evaluation ---
        await self._evaluate_output_guardrails(model, response_obj, kwargs)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after failed LLM API call (async path)."""
        if not self._plugins:
            return

        model = kwargs.get("model", "unknown")
        # LiteLLM passes the exception as response_obj for failures
        exception = (
            response_obj
            if isinstance(response_obj, Exception)
            else Exception(str(response_obj))
        )
        for plugin in self._plugins:
            try:
                await plugin.on_llm_failure(model, exception, kwargs)
            except Exception as e:
                logger.error(
                    f"Plugin '{plugin.name}' on_llm_failure failed: {e}",
                    exc_info=True,
                )

    # =========================================================================
    # Guardrail policy evaluation
    # =========================================================================

    async def _evaluate_input_guardrails(
        self,
        model: str,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> None:
        """Evaluate input guardrail policies before the LLM call.

        Builds a request_data dict from the kwargs and messages, then runs
        all input-phase guardrail policies.  If any policy with action=DENY
        fails, raises ``HTTPException(446)`` to block the request.

        Fail-open: if the guardrail engine is unavailable or evaluation
        raises an unexpected error, the request passes through.
        """
        try:
            from litellm_llmrouter.guardrail_policies import (
                get_guardrail_policy_engine,
                GuardrailPhase,
                HTTP_446_GUARDRAIL_DENIED,
            )

            engine = get_guardrail_policy_engine()

            # Quick check: are there any enabled input policies?
            input_policies = engine.list_policies(phase=GuardrailPhase.INPUT)
            if not input_policies:
                return

            # Resolve workspace_id from governance context (if governance ran)
            workspace_id = None
            metadata = kwargs.get("litellm_params", {}).get("metadata")
            if isinstance(metadata, dict):
                gov_ctx = metadata.get("_governance_ctx")
                if isinstance(gov_ctx, dict):
                    workspace_id = gov_ctx.get("workspace_id")
            # Also check top-level metadata
            if workspace_id is None:
                metadata = kwargs.get("metadata")
                if isinstance(metadata, dict):
                    gov_ctx = metadata.get("_governance_ctx")
                    if isinstance(gov_ctx, dict):
                        workspace_id = gov_ctx.get("workspace_id")

            # Build request_data for guardrail evaluation
            request_data: dict[str, Any] = {
                "messages": messages or [],
                "model": model,
                "metadata": kwargs.get("metadata", {}),
                "max_tokens": kwargs.get("max_tokens"),
            }

            results = await engine.evaluate_input(
                request_data, workspace_id=workspace_id
            )

            # Check for DENY results
            deny_results = engine.get_deny_results(results)
            if deny_results:
                raise HTTPException(
                    status_code=HTTP_446_GUARDRAIL_DENIED,
                    detail={
                        "error": "guardrail_denied",
                        "guardrails": [
                            {
                                "id": r.guardrail_id,
                                "name": r.guardrail_name,
                                "message": r.message,
                            }
                            for r in deny_results
                        ],
                    },
                )

            # Log warning-level results (LOG/ALERT actions)
            warning_results = engine.get_warning_results(results)
            for r in warning_results:
                logger.info(
                    "Input guardrail warning '%s': %s (action=%s, latency=%.1fms)",
                    r.guardrail_name,
                    r.message,
                    r.action.value,
                    r.latency_ms,
                )

        except HTTPException:
            raise  # Propagate guardrail deny
        except Exception as exc:
            logger.debug("Input guardrail evaluation skipped: %s", exc)

    async def _evaluate_output_guardrails(
        self,
        model: str,
        response_obj: Any,
        kwargs: dict[str, Any],
    ) -> None:
        """Evaluate output guardrail policies after the LLM call.

        Output guardrails log violations but do NOT block — the response
        has already been sent to the client.

        Extracts response content from the response object (supports
        multiple response formats used by LiteLLM).
        """
        try:
            from litellm_llmrouter.guardrail_policies import (
                get_guardrail_policy_engine,
                GuardrailPhase,
            )

            engine = get_guardrail_policy_engine()

            # Quick check: are there any enabled output policies?
            output_policies = engine.list_policies(phase=GuardrailPhase.OUTPUT)
            if not output_policies:
                return

            # Resolve workspace_id from governance context
            workspace_id = None
            metadata = kwargs.get("litellm_params", {}).get("metadata")
            if isinstance(metadata, dict):
                gov_ctx = metadata.get("_governance_ctx")
                if isinstance(gov_ctx, dict):
                    workspace_id = gov_ctx.get("workspace_id")
            if workspace_id is None:
                metadata = kwargs.get("metadata")
                if isinstance(metadata, dict):
                    gov_ctx = metadata.get("_governance_ctx")
                    if isinstance(gov_ctx, dict):
                        workspace_id = gov_ctx.get("workspace_id")

            # Extract response content from the response object
            content = self._extract_response_content(response_obj)

            response_data: dict[str, Any] = {
                "content": content,
                "model": model,
                "metadata": kwargs.get("metadata", {}),
            }

            results = await engine.evaluate_output(
                response_data, workspace_id=workspace_id
            )

            # Log all failures (output guardrails never block)
            for r in results:
                if not r.passed:
                    logger.warning(
                        "Output guardrail '%s' failed: %s (action=%s, latency=%.1fms)",
                        r.guardrail_name,
                        r.message,
                        r.action.value,
                        r.latency_ms,
                    )

        except Exception as exc:
            logger.debug("Output guardrail evaluation skipped: %s", exc)

    @staticmethod
    def _extract_response_content(response_obj: Any) -> str:
        """Extract text content from a LiteLLM response object.

        Handles multiple response formats:
        - ModelResponse with choices[0].message.content
        - Dict with choices[0].message.content
        - Dict with content key
        - Fallback to str(response_obj)
        """
        if response_obj is None:
            return ""

        # ModelResponse / dict with choices
        try:
            choices = getattr(response_obj, "choices", None)
            if choices is None and isinstance(response_obj, dict):
                choices = response_obj.get("choices")
            if choices and len(choices) > 0:
                choice = choices[0]
                message = getattr(choice, "message", None)
                if message is None and isinstance(choice, dict):
                    message = choice.get("message", {})
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                if isinstance(content, str):
                    return content
        except (AttributeError, IndexError, TypeError):
            pass

        # Direct content key
        if isinstance(response_obj, dict) and "content" in response_obj:
            content = response_obj["content"]
            if isinstance(content, str):
                return content

        return str(response_obj)[:10000]  # Truncate for safety

    # =========================================================================
    # Additional LiteLLM proxy hooks (no-ops for now)
    # =========================================================================

    async def async_post_call_success_hook(
        self, data: dict[str, Any], user_api_key_dict: Any, response: Any
    ) -> Any:
        """Post-call success hook (proxy-specific)."""
        pass

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, Any],
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> Any:
        """Post-call failure hook (proxy-specific)."""
        pass


# Module-level singleton
_callback_bridge: PluginCallbackBridge | None = None


def get_callback_bridge() -> PluginCallbackBridge | None:
    """Get the global callback bridge instance."""
    return _callback_bridge


def register_callback_bridge(plugins: list[Any]) -> PluginCallbackBridge | None:
    """
    Register the plugin callback bridge with LiteLLM.

    Args:
        plugins: List of GatewayPlugin instances with LLM lifecycle hooks

    Returns:
        The registered bridge, or None if no callback plugins or LiteLLM unavailable
    """
    global _callback_bridge

    if not plugins:
        logger.debug("No callback-capable plugins, skipping bridge registration")
        return None

    try:
        import litellm

        bridge = PluginCallbackBridge(plugins)

        if not hasattr(litellm, "callbacks"):
            litellm.callbacks = []

        # Avoid duplicate registration
        for existing in litellm.callbacks:
            if isinstance(existing, PluginCallbackBridge):
                logger.debug(
                    "PluginCallbackBridge already registered, updating plugins"
                )
                existing.set_plugins(plugins)
                _callback_bridge = existing
                return existing

        litellm.callbacks.append(bridge)
        _callback_bridge = bridge
        logger.info(
            f"Registered PluginCallbackBridge with LiteLLM "
            f"({len(plugins)} callback plugins)"
        )
        return bridge

    except ImportError:
        logger.warning("LiteLLM not available, cannot register callback bridge")
        return None
    except Exception as e:
        logger.error(f"Failed to register callback bridge: {e}")
        return None


def reset_callback_bridge() -> None:
    """Reset the global callback bridge (for testing)."""
    global _callback_bridge
    _callback_bridge = None
