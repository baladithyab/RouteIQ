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
        # MUST run before the (non-blocking) affinity injection so a DENY policy
        # fails the request and never reaches the engine — affinity logic does
        # NOT short-circuit this fail-closed seam.
        await self._evaluate_input_guardrails(model, messages, kwargs)

        # --- Multinode engine-affinity passthrough (RouteIQ-bdd0 + 3316) ---
        # Default-OFF + byte-stable: a no-op unless ROUTEIQ_MULTINODE_AFFINITY_ENABLED
        # is set AND an affinity hint / disaggregation signal is present. Returns
        # True only when it actually mutated kwargs, so we know to signal a kwargs
        # replacement to litellm even on the no-plugins path.
        affinity_mutated = await self._apply_engine_affinity(kwargs)

        # Preserve byte-stable behavior for the no-plugins path: only signal a
        # kwargs replacement to litellm when plugins OR affinity mutated kwargs.
        return kwargs if (self._plugins or affinity_mutated) else None

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

        # --- Multinode affinity recording (RouteIQ-bdd0) ---
        # Default-OFF + best-effort: record the response_id -> deployment mapping
        # so the NEXT turn (with previous_response_id) can be made sticky. No-op
        # unless ROUTEIQ_MULTINODE_AFFINITY_ENABLED and a tracker is initialized.
        await self._record_affinity(model, response_obj, kwargs)

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
    # Multinode engine-affinity passthrough (RouteIQ-bdd0 + RouteIQ-3316)
    # =========================================================================

    #: Sentinel in ``kwargs['metadata']`` marking that affinity was already
    #: applied for this request, so the deployment hook + logging hook (both of
    #: which can fire for a CustomLogger) don't double-inject headers/params.
    _AFFINITY_SENTINEL = "_routeiq_affinity_applied"

    @classmethod
    def _affinity_already_applied(cls, kwargs: dict[str, Any]) -> bool:
        """True if affinity was already injected for this request (idempotency)."""
        if not isinstance(kwargs, dict):
            return False
        metadata = kwargs.get("metadata")
        return bool(isinstance(metadata, dict) and metadata.get(cls._AFFINITY_SENTINEL))

    @classmethod
    def _mark_affinity_applied(cls, kwargs: dict[str, Any]) -> None:
        """Record that affinity ran so a paired seam / re-fire short-circuits."""
        if not isinstance(kwargs, dict):
            return
        metadata = kwargs.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            kwargs["metadata"] = metadata
        metadata[cls._AFFINITY_SENTINEL] = True

    @staticmethod
    def _request_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge top-level + litellm_params metadata into a single read-only view.

        Affinity hints (conversation/session ids, a per-request disagg flag) may
        ride either ``kwargs['metadata']`` or ``kwargs['litellm_params']['metadata']``
        depending on the entry path. Reads only; never mutates either dict.
        """
        merged: dict[str, Any] = {}
        lp = kwargs.get("litellm_params")
        if isinstance(lp, dict):
            lp_md = lp.get("metadata")
            if isinstance(lp_md, dict):
                merged.update(lp_md)
        top_md = kwargs.get("metadata")
        if isinstance(top_md, dict):
            merged.update(top_md)
        return merged

    @classmethod
    def _derive_affinity_key(cls, kwargs: dict[str, Any]) -> str | None:
        """Derive a stable session/conversation affinity key from the request.

        Tries, in priority order: the Responses-API ``previous_response_id``
        (the next-turn stickiness signal recorded on the prior success), then a
        conversation id, then a stable session id — checked at the top level of
        ``kwargs`` and in the merged metadata view. Returns ``None`` when no
        stable key is present (so a stateless request stays byte-stable).
        """
        metadata = cls._request_metadata(kwargs)

        for source in (kwargs, metadata):
            for field in (
                "previous_response_id",
                "conversation_id",
                "session_id",
                "litellm_session_id",
            ):
                value = source.get(field)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _resolve_disagg_signals(
        metadata: dict[str, Any],
    ) -> tuple[bool, bool, dict[str, Any] | None]:
        """Resolve disaggregation passthrough signals (RouteIQ-3316).

        A per-request metadata flag (``do_remote_prefill`` / ``do_remote_decode``
        / ``kv_transfer_params``) takes precedence; otherwise falls back to the
        settings-level disagg defaults. RouteIQ makes NO disaggregation decision
        — it only carries the signals. Returns ``(prefill, decode, params)`` with
        only-truthy semantics so a non-disagg request resolves to all-falsy.
        """
        # Settings-level defaults (cheap; failure => no defaults).
        default_prefill = False
        default_decode = False
        try:
            from litellm_llmrouter.settings import get_settings

            ma = get_settings().multinode_affinity
            default_prefill = bool(ma.disagg_default_remote_prefill)
            default_decode = bool(ma.disagg_default_remote_decode)
        except Exception:  # pragma: no cover - settings always importable
            pass

        def _flag(name: str, default: bool) -> bool:
            value = metadata.get(name)
            if isinstance(value, bool):
                return value
            return default

        prefill = _flag("do_remote_prefill", default_prefill)
        decode = _flag("do_remote_decode", default_decode)

        params = metadata.get("kv_transfer_params")
        if not isinstance(params, dict) or not params:
            params = None

        return prefill, decode, params

    async def _apply_engine_affinity(self, kwargs: dict[str, Any]) -> bool:
        """Inject engine-affinity headers + disagg params into outbound kwargs.

        Gated by ``ROUTEIQ_MULTINODE_AFFINITY_ENABLED`` (default OFF, byte-stable
        no-op). When ON:

          * Derives a session/conversation affinity key from the request and,
            if a sticky deployment is known (via the conversation-affinity
            tracker), injects ``x-worker-instance-id`` + ``x-routeiq-affinity-key``
            headers so the engine front-end routes to the same decode worker
            (RouteIQ-bdd0).
          * Carries disaggregation-coordination signals (``do_remote_prefill`` /
            ``do_remote_decode`` / ``kv_transfer_params``) through to the engine
            when the request signals disagg intent (RouteIQ-3316).

        NEVER clobbers a caller-supplied header/param (``apply_engine_affinity``
        uses ``setdefault`` semantics). Idempotent: a metadata sentinel prevents
        double-application across the deployment + logging seams.

        Returns ``True`` only when it actually merged something into ``kwargs``.
        """
        try:
            from litellm_llmrouter.engine_affinity import (
                apply_engine_affinity,
                multinode_affinity_enabled,
            )

            if not multinode_affinity_enabled():
                return False
            if self._affinity_already_applied(kwargs):
                return False

            affinity_key = self._derive_affinity_key(kwargs)

            # Look up a sticky decode-worker hint for this affinity key.
            worker_instance_id: str | None = None
            if affinity_key:
                try:
                    from litellm_llmrouter.conversation_affinity import (
                        get_affinity_tracker,
                    )

                    tracker = get_affinity_tracker()
                    if tracker is not None:
                        record = await tracker.get_affinity(affinity_key)
                        if record is not None:
                            # The recorded provider deployment is the sticky
                            # decode-worker hint for the engine front-end.
                            worker_instance_id = record.provider_deployment
                except Exception as exc:  # tracker failure must never block
                    logger.debug("Affinity tracker lookup skipped: %s", exc)

            metadata = self._request_metadata(kwargs)
            prefill, decode, kv_params = self._resolve_disagg_signals(metadata)

            # Nothing to carry: no affinity key AND no disagg signal => no-op.
            if not affinity_key and not (prefill or decode or kv_params):
                return False

            merged = apply_engine_affinity(
                kwargs,
                affinity_key=affinity_key,
                worker_instance_id=worker_instance_id,
                do_remote_prefill=prefill,
                do_remote_decode=decode,
                kv_transfer_params=kv_params,
            )

            # ``apply_engine_affinity`` returns a NEW dict; merge only the keys it
            # added back into the LIVE kwargs (the object litellm sends), without
            # clobbering anything already set. ``extra_headers`` is merged
            # key-by-key so caller headers survive.
            mutated = False
            for key, value in merged.items():
                if key == "extra_headers" and isinstance(value, dict):
                    existing = kwargs.get("extra_headers")
                    if not isinstance(existing, dict):
                        existing = {}
                    before = dict(existing)
                    for hk, hv in value.items():
                        existing.setdefault(hk, hv)
                    if existing != before or "extra_headers" not in kwargs:
                        kwargs["extra_headers"] = existing
                        if existing != before:
                            mutated = True
                elif key not in kwargs:
                    kwargs[key] = value
                    mutated = True

            if mutated:
                self._mark_affinity_applied(kwargs)
            return mutated

        except Exception as exc:  # affinity must never break the request
            logger.debug("Engine-affinity passthrough skipped: %s", exc)
            return False

    async def _record_affinity(
        self, model: str, response_obj: Any, kwargs: dict[str, Any]
    ) -> None:
        """Record the ``response_id -> deployment`` mapping for the next turn.

        Gated by ``ROUTEIQ_MULTINODE_AFFINITY_ENABLED`` (default OFF). Best-effort:
        extracts the response id from the response object and the selected
        provider/deployment from kwargs, then awaits ``record_response`` so the
        NEXT request carrying that id as ``previous_response_id`` can be made
        sticky. Never raises into the success path.
        """
        try:
            from litellm_llmrouter.engine_affinity import multinode_affinity_enabled

            if not multinode_affinity_enabled():
                return

            from litellm_llmrouter.conversation_affinity import get_affinity_tracker

            tracker = get_affinity_tracker()
            if tracker is None:
                return

            response_id = self._extract_response_id(response_obj)
            if not response_id:
                return

            provider_deployment = self._extract_selected_deployment(kwargs, model)
            await tracker.record_response(response_id, provider_deployment, model)
        except Exception as exc:  # recording must never break the success path
            logger.debug("Affinity recording skipped: %s", exc)

    @staticmethod
    def _extract_response_id(response_obj: Any) -> str | None:
        """Extract the response id from a LiteLLM response object.

        Handles ModelResponse-like objects (``.id``) and dict responses
        (``{"id": ...}``). Returns ``None`` when no id is present.
        """
        if response_obj is None:
            return None
        rid = getattr(response_obj, "id", None)
        if rid is None and isinstance(response_obj, dict):
            rid = response_obj.get("id")
        if isinstance(rid, str) and rid:
            return rid
        return None

    @staticmethod
    def _extract_selected_deployment(kwargs: dict[str, Any], model: str) -> str:
        """Resolve the selected provider/deployment string for the affinity record.

        Prefers the concrete provider model in ``litellm_params['model']`` (e.g.
        ``openai/gpt-4``); falls back to the public ``model`` alias.
        """
        lp = kwargs.get("litellm_params")
        if isinstance(lp, dict):
            provider_model = lp.get("model")
            if isinstance(provider_model, str) and provider_model:
                return provider_model
        return model

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
