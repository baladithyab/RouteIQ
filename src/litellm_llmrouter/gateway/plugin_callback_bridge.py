"""
Plugin Callback Bridge: LiteLLM Callback → Plugin Hook Integration
===================================================================

Bridges LiteLLM's callback system to GatewayPlugin LLM lifecycle hooks.

LiteLLM calls methods on objects in ``litellm.callbacks`` at various points
in the LLM call lifecycle. This bridge translates those callbacks into
GatewayPlugin hook invocations:

    CustomLogger.async_pre_call_deployment_hook →  plugin.on_llm_pre_call  (MUTATION)
    litellm.log_success_event                   →  plugin.on_llm_success
    litellm.log_failure_event                   →  plugin.on_llm_failure

Request MUTATION seam (RouteIQ-60e3)
------------------------------------
Plugin request mutations (``on_llm_pre_call`` returning a kwargs override, or
mutating ``messages`` in place) ride ``CustomLogger.async_pre_call_deployment_hook``
— NOT the logging hook ``async_log_pre_api_call``. Two reasons, both verified
against litellm 1.89.0 source:

  1. ``async_log_pre_api_call`` is a *logging* hook. litellm dispatches it
     only when the callback ``isinstance(callback, CustomLogger)``
     (``litellm_logging.py`` ``pre_call``), and it fires AFTER the provider
     request body is serialized. A duck-typed (non-CustomLogger) callback is
     never dispatched, so mutation via that seam silently no-ops.
  2. ``async_pre_call_deployment_hook`` (``custom_logger.py:264``) is the
     documented seam to "modify the request AFTER a deployment is selected,
     but BEFORE the request is sent". litellm calls it from the ``@client``
     ``wrapper_async`` (``utils.py:1815``) on the live ``kwargs`` and REPLACES
     ``kwargs`` with the returned dict before the completion call runs.

Because that dispatch ALSO gates on ``isinstance(callback, CustomLogger)``
(``utils.py:1274-1280``), this bridge subclasses ``CustomLogger`` so litellm
actually invokes it.

Guardrail integration
----------------------
The guardrail policy engine (``guardrail_policies.py``) is evaluated in the
deployment hook (input guardrails: deny/log/alert) and in the success event
(output guardrails: log/alert only). Plugin guardrails that block by raising
``GuardrailBlockError`` propagate out of the deployment hook, failing the
request before it reaches the provider.

Registration:
    Called during plugin startup in app.py. The bridge is appended to
    ``litellm.callbacks`` alongside the existing RouterDecisionCallback.

Design:
    - Subclasses ``litellm.integrations.custom_logger.CustomLogger`` so litellm
      dispatches the mutation + logging hooks (the isinstance gate).
    - Plugin hook failures are caught and logged, never crash the LLM call
      (except guardrail blocks, which intentionally propagate).
    - Plugins are called in priority order.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from litellm_llmrouter.gateway.plugins.guardrails_base import GuardrailBlockError

logger = logging.getLogger(__name__)


def _custom_logger_base() -> type:
    """Resolve the litellm ``CustomLogger`` base class lazily.

    Imported lazily (not at module import time) so this module has no import
    side effects and stays importable when litellm is absent. Falls back to
    ``object`` so the bridge remains usable (duck-typed) in environments
    without litellm — e.g. isolated unit tests that exercise the hooks directly.
    """
    try:
        from litellm.integrations.custom_logger import CustomLogger

        return CustomLogger
    except Exception:  # pragma: no cover - litellm always present in prod
        return object


class PluginCallbackBridge(_custom_logger_base()):  # type: ignore[misc]
    """
    LiteLLM ``CustomLogger`` that bridges to GatewayPlugin LLM lifecycle hooks.

    Subclasses ``CustomLogger`` so litellm dispatches both the request-mutation
    seam (``async_pre_call_deployment_hook``) and the logging seams, which both
    gate on ``isinstance(callback, CustomLogger)``. Delegates to plugins that
    override on_llm_pre_call, on_llm_success, or on_llm_failure.
    """

    def __init__(self, plugins: list[Any] | None = None) -> None:
        """
        Args:
            plugins: List of GatewayPlugin instances with LLM lifecycle hooks.
                     Can be set later via set_plugins().
        """
        # CustomLogger.__init__ may register the instance in litellm's
        # in-memory logger list; call it when the base is the real CustomLogger.
        try:
            super().__init__()
        except Exception:  # pragma: no cover - object.__init__ never raises
            pass
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

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, Any], call_type: Any
    ) -> dict[str, Any] | None:
        """Request-MUTATION seam (litellm ``CustomLogger`` hook).

        Called by litellm from the ``@client`` ``wrapper_async`` after a
        deployment is selected but BEFORE the request is sent
        (``utils.py:1815``), on the live ``kwargs``. The returned dict REPLACES
        ``kwargs`` for the completion call, so this is the correct seam for
        plugin request mutations (cachePoint, requestMetadata, per-team
        callbacks, guardrailConfig, context optimization).

        Runs plugin ``on_llm_pre_call`` mutations first (only when plugins are
        registered), then ALWAYS evaluates input guardrail policies. Guardrail
        blocks (``GuardrailBlockError`` or ``HTTPException``) propagate to fail
        the request before it is sent.

        SECURITY: input guardrail evaluation is UNCONDITIONAL — it must run even
        when no callback plugins are loaded, otherwise an operator with input
        guardrail policies but zero callback-capable plugins would have input
        guardrails silently bypassed (fail-open). ``_evaluate_input_guardrails``
        has its own internal "are there policies?" check, so it is a cheap
        no-op when no input policies are configured.

        Returns the mutated ``kwargs`` so litellm picks up plugin mutations.
        Returns ``None`` when there are no plugins (so the no-plugins path stays
        byte-stable for litellm; guardrails, if any, still ran and either denied
        the request or passed through without mutating ``kwargs``).
        """
        model = kwargs.get("model", "unknown")
        # ``messages`` is the live list inside kwargs — passing it (not a copy)
        # lets plugins like context_optimizer mutate it in place AND lets
        # litellm see the change because we return kwargs below.
        messages = kwargs.get("messages") or []

        # Plugin mutations only run when plugins are registered.
        if self._plugins:
            await self._dispatch_pre_call_mutations(model, messages, kwargs)

        # --- Input guardrail policy evaluation (authoritative deny seam) ---
        # ALWAYS runs, regardless of self._plugins. Internal policy check makes
        # this a cheap no-op when no input guardrail policies are configured.
        await self._evaluate_input_guardrails(model, messages, kwargs)

        # Preserve byte-stable behavior for the no-plugins path: only signal a
        # kwargs replacement to litellm when plugins may have mutated it.
        return kwargs if self._plugins else None

    async def _dispatch_pre_call_mutations(
        self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> None:
        """Run each plugin's ``on_llm_pre_call`` for request mutation.

        Dict returns are merged into ``kwargs``; ``messages`` may also be
        mutated in place by a plugin. ``GuardrailBlockError`` propagates so a
        blocking guardrail fails the request; other exceptions are isolated.
        """
        for plugin in self._plugins:
            try:
                result = await plugin.on_llm_pre_call(model, messages, kwargs)
                if isinstance(result, dict):
                    # Merge overrides into kwargs (in place — kwargs is the
                    # object litellm will send / that we return upstream).
                    kwargs.update(result)
            except GuardrailBlockError:
                raise  # Let guardrail blocks propagate as request failures
            except HTTPException:
                raise  # Guardrail-policy deny / explicit block propagates
            except Exception as e:
                logger.error(
                    f"Plugin '{plugin.name}' on_llm_pre_call failed: {e}",
                    exc_info=True,
                )

    async def async_log_pre_api_call(
        self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> None:
        """LOGGING seam (async path).

        Plugin request *mutations* ride ``async_pre_call_deployment_hook`` (the
        correct pre-send seam). This logging hook fires AFTER the provider body
        is serialized, so it is kept for backward compatibility and as the
        observability dispatch point; it still calls ``on_llm_pre_call`` so
        plugins that only inspect/log the request keep working. Mutations that
        land here would be too late to reach the wire — callers needing to
        mutate the request must rely on the deployment hook.

        SECURITY: this seam ALSO evaluates input guardrail policies
        unconditionally. It is the path the governance/guardrail contract tests
        assert against, and — like the deployment hook — must fail-closed when
        guardrail policies exist but no callback plugins are loaded. Guardrail
        denies raise ``HTTPException`` here too. ``_evaluate_input_guardrails``
        has its own internal "are there policies?" check, so it is a cheap
        no-op when no input policies are configured.
        """
        if self._plugins:
            await self._dispatch_pre_call_mutations(model, messages, kwargs)

        # --- Input guardrail policy evaluation (unconditional, fail-closed) ---
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

    _INPUT_GUARDRAIL_SENTINEL = "_routeiq_input_guardrails_evaluated"

    @classmethod
    def _input_guardrails_already_evaluated(cls, kwargs: dict[str, Any]) -> bool:
        """True if input guardrails ran for this request on the other seam.

        The sentinel lives in ``kwargs['metadata']`` (mirroring the governance
        ``_governance_ctx`` convention) so it never leaks into the top-level
        completion kwargs litellm sends to the provider.
        """
        if not isinstance(kwargs, dict):
            return False
        metadata = kwargs.get("metadata")
        return bool(
            isinstance(metadata, dict) and metadata.get(cls._INPUT_GUARDRAIL_SENTINEL)
        )

    @classmethod
    def _mark_input_guardrails_evaluated(cls, kwargs: dict[str, Any]) -> None:
        """Record that input guardrails ran so the paired seam short-circuits."""
        if not isinstance(kwargs, dict):
            return
        metadata = kwargs.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            kwargs["metadata"] = metadata
        metadata[cls._INPUT_GUARDRAIL_SENTINEL] = True

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

        Fail-open ONLY on infrastructure errors: if the guardrail engine is
        unavailable or evaluation raises an unexpected error, the request passes
        through. A configured DENY policy that matches is ALWAYS fail-closed
        (raises ``HTTPException``), regardless of whether any callback plugins
        are loaded.

        De-duplication: this runs on two seams for the live request path
        (``async_pre_call_deployment_hook`` then ``async_log_pre_api_call``).
        The first invocation marks ``kwargs`` with a sentinel so the second
        seam skips re-evaluation (avoids double latency / double warning logs)
        without changing behavior — a DENY on the first seam already raised, and
        a pass on the first seam guarantees a pass on the second. The contract
        tests call a single seam with fresh ``kwargs`` (no sentinel), so they
        always evaluate.
        """
        # Skip if this exact request was already guardrail-evaluated on the
        # other seam (deployment hook is authoritative for the live path).
        # The sentinel lives in metadata (not top-level kwargs) to keep the
        # completion kwargs byte-stable for litellm.
        if self._input_guardrails_already_evaluated(kwargs):
            return

        try:
            from litellm_llmrouter.guardrail_policies import (
                get_guardrail_policy_engine,
                GuardrailPhase,
                HTTP_446_GUARDRAIL_DENIED,
            )

            engine = get_guardrail_policy_engine()

            # Quick check: are there any enabled input policies? Cheap no-op
            # when none are configured (the common no-guardrail path).
            input_policies = engine.list_policies(phase=GuardrailPhase.INPUT)
            if not input_policies:
                # No policies: nothing to dedupe. Do NOT mark kwargs — the
                # paired seam will hit this same cheap check, and mutating
                # kwargs here would inject a spurious ``metadata`` key and break
                # byte-stability on the common no-guardrail path.
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

            # Passed (no DENY): mark so the paired seam short-circuits.
            self._mark_input_guardrails_evaluated(kwargs)

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


def register_callback_bridge(
    plugins: list[Any], *, force: bool = False
) -> PluginCallbackBridge | None:
    """
    Register the plugin callback bridge with LiteLLM.

    Args:
        plugins: List of GatewayPlugin instances with LLM lifecycle hooks
        force: Register the bridge even when ``plugins`` is empty. Required so
            the input-guardrail deny seam runs when guardrail policies are
            configured but no callback-capable plugins are loaded — otherwise
            input guardrails would be silently bypassed (fail-open). The bridge
            evaluates guardrail policies unconditionally on its pre-call seams.

    Returns:
        The registered bridge, or None if (no plugins AND not forced) or
        LiteLLM is unavailable.
    """
    global _callback_bridge

    if not plugins and not force:
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
