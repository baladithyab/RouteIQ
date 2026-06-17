"""
Integration-style unit tests for the request-mutation seam fix and the three
reopened Bedrock levers.

These tests exercise the REAL litellm dispatch path
(``litellm.utils.async_pre_call_deployment_hook``) so that a green test proves
the mutation actually reaches the outbound completion call args -- the #1 thing
the RouteIQ-60e3 reject demanded. They would FAIL on the pre-fix code (a
duck-typed bridge dispatching mutation from the logging hook
``async_log_pre_api_call``), because:

  * litellm's deployment-hook dispatch gates on ``isinstance(callback,
    CustomLogger)`` (utils.py:1274). A non-CustomLogger bridge is skipped, so
    the kwargs are returned unmutated.
  * The logging hook ``async_log_pre_api_call`` is likewise gated and fires
    after the body is serialized.

Coverage:
  * RouteIQ-60e3 -- the bridge is a CustomLogger and litellm's real dispatch
    invokes it + applies the returned kwargs; guardrail blocks propagate;
    context-optimizer in-place messages mutation reaches the wire.
  * RouteIQ-294a -- requestMetadata tenant tags reach the converse body
    (asserted on the actual transformed Bedrock RequestObject, not just kwargs).
  * RouteIQ-9cd8 -- per-team override sets TOP-LEVEL kwargs
    callbacks/success_callback/failure_callback (not metadata).
  * RouteIQ-b9ee -- cachePoint via cache_control_injection_points reaches the
    converse body for the system prefix AND the tool_config.
"""

import pytest

import litellm
from litellm.types.utils import CallTypes
from litellm.utils import async_pre_call_deployment_hook

from litellm_llmrouter.gateway.plugin_callback_bridge import (
    PluginCallbackBridge,
    reset_callback_bridge,
)
from litellm_llmrouter.gateway.plugins.bedrock_request_levers import (
    BedrockRequestLeversPlugin,
)
from litellm_llmrouter.gateway.plugins.guardrails_base import GuardrailBlockError
from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin, PluginMetadata


@pytest.fixture(autouse=True)
def _reset():
    reset_callback_bridge()
    original = list(getattr(litellm, "callbacks", []))
    yield
    litellm.callbacks = original
    reset_callback_bridge()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_levers_plugin(**flags) -> BedrockRequestLeversPlugin:
    """Build a levers plugin with explicit flags (bypasses settings/startup)."""
    p = BedrockRequestLeversPlugin()
    p._request_metadata = flags.get("request_metadata", False)
    p._metadata_prefix = flags.get("metadata_prefix", "routeiq_")
    p._team_callbacks = flags.get("team_callbacks", False)
    p._team_callback_map = flags.get("team_callback_map", {})
    p._cache_point = flags.get("cache_point", False)
    p._cache_system = flags.get("cache_system", True)
    p._cache_tools = flags.get("cache_tools", True)
    return p


async def _dispatch_via_litellm(bridge: PluginCallbackBridge, kwargs: dict) -> dict:
    """Run the EXACT litellm dispatch the @client wrapper uses (utils.py:1815).

    Registers the bridge in ``litellm.callbacks`` and calls litellm's own
    ``async_pre_call_deployment_hook`` so the test passes only if litellm
    actually invokes our hook (the isinstance(CustomLogger) gate) and threads
    the returned kwargs through.
    """
    litellm.callbacks = [bridge]
    return await async_pre_call_deployment_hook(kwargs, CallTypes.acompletion.value)


# ===========================================================================
# RouteIQ-60e3: the seam itself
# ===========================================================================


class TestSeamFix:
    def test_bridge_is_custom_logger(self):
        """The whole fix hinges on the isinstance(CustomLogger) gate."""
        from litellm.integrations.custom_logger import CustomLogger

        bridge = PluginCallbackBridge([])
        assert isinstance(bridge, CustomLogger)

    @pytest.mark.asyncio
    async def test_mutation_reaches_completion_kwargs_via_real_litellm_dispatch(self):
        """A plugin kwargs override reaches the args litellm sends.

        This is the acceptance test for RouteIQ-60e3: it routes through
        litellm's OWN deployment-hook dispatch, which gates on CustomLogger.
        On the pre-fix duck-typed bridge the kwargs come back UNMUTATED.
        """

        class InjectingPlugin(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="injector")

            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

            async def on_llm_pre_call(self, model, messages, kwargs):
                return {"requestMetadata": {"injected": "yes"}, "temperature": 0.1}

        bridge = PluginCallbackBridge([InjectingPlugin()])
        kwargs = {"model": "bedrock/anthropic.claude", "messages": []}

        result = await _dispatch_via_litellm(bridge, kwargs)

        assert result["requestMetadata"] == {"injected": "yes"}
        assert result["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_guardrail_block_propagates_from_deployment_hook(self):
        """A blocking guardrail raised from on_llm_pre_call fails the request
        on the real seam (before the request is sent)."""

        class BlockingPlugin(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="blocker")

            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

            async def on_llm_pre_call(self, model, messages, kwargs):
                raise GuardrailBlockError(
                    guardrail_name="x", category="c", message="blocked", score=1.0
                )

        bridge = PluginCallbackBridge([BlockingPlugin()])
        with pytest.raises(GuardrailBlockError):
            await _dispatch_via_litellm(bridge, {"model": "m", "messages": []})

    @pytest.mark.asyncio
    async def test_in_place_messages_mutation_reaches_wire(self):
        """A plugin that mutates kwargs['messages'] in place (context_optimizer
        pattern) is visible in the returned kwargs."""

        class TrimPlugin(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="trimmer")

            async def startup(self, app, context=None):
                pass

            async def shutdown(self, app, context=None):
                pass

            async def on_llm_pre_call(self, model, messages, kwargs):
                messages.clear()
                messages.append({"role": "user", "content": "trimmed"})
                return None

        bridge = PluginCallbackBridge([TrimPlugin()])
        kwargs = {"model": "m", "messages": [{"role": "user", "content": "long"}]}

        result = await _dispatch_via_litellm(bridge, kwargs)
        assert result["messages"] == [{"role": "user", "content": "trimmed"}]

    @pytest.mark.asyncio
    async def test_no_plugins_is_byte_stable_noop(self):
        """No plugins -> hook returns the kwargs unchanged (no spurious keys)."""
        bridge = PluginCallbackBridge([])
        kwargs = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        result = await _dispatch_via_litellm(bridge, kwargs)
        # litellm copies kwargs; our hook returns None so the copy is unchanged.
        assert result == kwargs

    @pytest.mark.asyncio
    async def test_audit_bedrock_guardrails_guardrailConfig_reaches_wire(self):
        """AUDIT (RouteIQ-60e3): bedrock_guardrails' native guardrailConfig
        mutation was equally no-op under the old logging seam. The same
        CustomLogger fix carries it to the wire -- no plugin-code change. This
        locks that in so the shared mechanism can't silently regress."""
        from litellm_llmrouter.gateway.plugins.bedrock_guardrails import (
            BedrockGuardrailsPlugin,
        )

        guard = BedrockGuardrailsPlugin()
        guard._enabled = True
        guard._guardrail_id = "gr-123"
        guard._guardrail_version = "DRAFT"
        guard._native_activation = True
        guard._trace = "disabled"

        bridge = PluginCallbackBridge([guard])
        # Empty messages -> no ApplyGuardrail call, only the native override runs.
        result = await _dispatch_via_litellm(bridge, {"model": "m", "messages": []})
        assert result["guardrailConfig"]["guardrailIdentifier"] == "gr-123"


# ===========================================================================
# RouteIQ-294a: requestMetadata tenant tags
# ===========================================================================


class TestRequestMetadataLever:
    @pytest.mark.asyncio
    async def test_off_is_noop(self):
        plugin = _make_levers_plugin(request_metadata=False)
        kwargs = {
            "model": "m",
            "messages": [],
            "metadata": {"_governance_ctx": {"workspace_id": "ws1", "key_id": "k1"}},
        }
        assert await plugin.on_llm_pre_call("m", [], kwargs) is None

    @pytest.mark.asyncio
    async def test_tenant_tags_land_in_top_level_kwarg(self):
        plugin = _make_levers_plugin(request_metadata=True)
        kwargs = {
            "model": "m",
            "messages": [],
            "metadata": {"_governance_ctx": {"workspace_id": "ws1", "key_id": "k1"}},
        }
        override = await plugin.on_llm_pre_call("m", [], kwargs)
        assert override is not None
        rm = override["requestMetadata"]
        assert rm["routeiq_workspace"] == "ws1"
        assert rm["routeiq_key"] == "k1"

    @pytest.mark.asyncio
    async def test_request_metadata_reaches_converse_body(self):
        """End-to-end: the lever output lands in the actual transformed Bedrock
        converse RequestObject (via map_openai_params + _transform_request)."""
        from litellm.llms.bedrock.chat.converse_transformation import (
            AmazonConverseConfig,
        )

        plugin = _make_levers_plugin(request_metadata=True)
        kwargs = {
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": [],
            "metadata": {"_governance_ctx": {"workspace_id": "ws1", "key_id": "k1"}},
        }
        override = await plugin.on_llm_pre_call("m", [], kwargs)

        cfg = AmazonConverseConfig()
        model = "anthropic.claude-3-sonnet-20240229-v1:0"
        optional_params = cfg.map_openai_params(
            non_default_params={"requestMetadata": override["requestMetadata"]},
            optional_params={},
            model=model,
            drop_params=False,
        )
        assert optional_params["requestMetadata"]["routeiq_workspace"] == "ws1"

        body = cfg._transform_request(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
        )
        assert body["requestMetadata"]["routeiq_workspace"] == "ws1"
        assert body["requestMetadata"]["routeiq_key"] == "k1"

    @pytest.mark.asyncio
    async def test_caller_supplied_request_metadata_not_clobbered(self):
        plugin = _make_levers_plugin(request_metadata=True)
        kwargs = {
            "model": "m",
            "messages": [],
            "requestMetadata": {"caller": "value"},
            "metadata": {"_governance_ctx": {"workspace_id": "ws1"}},
        }
        override = await plugin.on_llm_pre_call("m", [], kwargs)
        assert override is None

    @pytest.mark.asyncio
    async def test_invalid_chars_sanitized(self):
        """Bedrock requestMetadata charset is restricted; a stray char is
        sanitized so it can never 400 the request."""
        plugin = _make_levers_plugin(request_metadata=True)
        kwargs = {
            "model": "m",
            "messages": [],
            "metadata": {"_governance_ctx": {"workspace_id": "ws!*1", "key_id": "k1"}},
        }
        override = await plugin.on_llm_pre_call("m", [], kwargs)
        assert override["requestMetadata"]["routeiq_workspace"] == "ws__1"


# ===========================================================================
# RouteIQ-9cd8: per-team callback override
# ===========================================================================


class TestTeamCallbackLever:
    @pytest.mark.asyncio
    async def test_off_is_noop(self):
        plugin = _make_levers_plugin(
            team_callbacks=False,
            team_callback_map={"team-a": {"success_callback": ["s3"]}},
        )
        kwargs = {"model": "m", "messages": [], "metadata": {"team_id": "team-a"}}
        assert await plugin.on_llm_pre_call("m", [], kwargs) is None

    @pytest.mark.asyncio
    async def test_team_sinks_set_as_top_level_kwargs(self):
        """The override sets TOP-LEVEL callbacks/success/failure -- the keys
        litellm reads dynamic per-request callbacks from (utils.py:791/903/923).
        It must NOT bury them under metadata."""
        plugin = _make_levers_plugin(
            team_callbacks=True,
            team_callback_map={
                "team-a": {
                    "success_callback": ["s3"],
                    "failure_callback": ["langfuse"],
                    "callbacks": ["datadog"],
                }
            },
        )
        kwargs = {"model": "m", "messages": [], "metadata": {"team_id": "team-a"}}
        override = await plugin.on_llm_pre_call("m", [], kwargs)

        assert override["success_callback"] == ["s3"]
        assert override["failure_callback"] == ["langfuse"]
        assert override["callbacks"] == ["datadog"]
        # Must NOT be hidden under metadata.
        assert "callbacks" not in kwargs.get("metadata", {})

    @pytest.mark.asyncio
    async def test_team_sinks_reach_completion_kwargs_via_real_dispatch(self):
        """The team sinks survive litellm's own deployment-hook dispatch as
        top-level kwargs (proving they reach the request layer)."""
        plugin = _make_levers_plugin(
            team_callbacks=True,
            team_callback_map={"team-a": {"success_callback": ["s3"]}},
        )
        bridge = PluginCallbackBridge([plugin])
        kwargs = {"model": "m", "messages": [], "metadata": {"team_id": "team-a"}}
        result = await _dispatch_via_litellm(bridge, kwargs)
        assert result["success_callback"] == ["s3"]

    @pytest.mark.asyncio
    async def test_existing_callbacks_merged_not_replaced(self):
        plugin = _make_levers_plugin(
            team_callbacks=True,
            team_callback_map={"team-a": {"success_callback": ["s3"]}},
        )
        kwargs = {
            "model": "m",
            "messages": [],
            "success_callback": ["existing"],
            "metadata": {"team_id": "team-a"},
        }
        override = await plugin.on_llm_pre_call("m", [], kwargs)
        assert override["success_callback"] == ["existing", "s3"]

    @pytest.mark.asyncio
    async def test_unmapped_team_is_noop(self):
        plugin = _make_levers_plugin(
            team_callbacks=True,
            team_callback_map={"team-a": {"success_callback": ["s3"]}},
        )
        kwargs = {"model": "m", "messages": [], "metadata": {"team_id": "other"}}
        assert await plugin.on_llm_pre_call("m", [], kwargs) is None

    @pytest.mark.asyncio
    async def test_falls_back_to_workspace_id_as_team(self):
        plugin = _make_levers_plugin(
            team_callbacks=True,
            team_callback_map={"ws-x": {"success_callback": ["s3"]}},
        )
        kwargs = {
            "model": "m",
            "messages": [],
            "metadata": {"_governance_ctx": {"workspace_id": "ws-x"}},
        }
        override = await plugin.on_llm_pre_call("m", [], kwargs)
        assert override["success_callback"] == ["s3"]


# ===========================================================================
# RouteIQ-b9ee: Bedrock prompt-caching cachePoint
# ===========================================================================


class TestCachePointLever:
    @pytest.mark.asyncio
    async def test_off_is_noop(self):
        plugin = _make_levers_plugin(cache_point=False)
        kwargs = {"model": "m", "messages": [{"role": "system", "content": "sys"}]}
        assert await plugin.on_llm_pre_call("m", kwargs["messages"], kwargs) is None

    @pytest.mark.asyncio
    async def test_injection_points_built_for_system_and_tools(self):
        plugin = _make_levers_plugin(cache_point=True)
        messages = [
            {"role": "system", "content": "long system prefix"},
            {"role": "user", "content": "hi"},
        ]
        kwargs = {
            "model": "m",
            "messages": messages,
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }
        override = await plugin.on_llm_pre_call("m", messages, kwargs)
        points = override["cache_control_injection_points"]
        locations = {p["location"] for p in points}
        assert "message" in locations  # system prefix
        assert "tool_config" in locations  # tool schema prefix

    @pytest.mark.asyncio
    async def test_system_cachepoint_reaches_converse_body(self):
        """The system-prefix injection point drives a system cachePoint in the
        actual transformed converse body (via anthropic_cache_control_hook +
        converse transform)."""
        from litellm.integrations.anthropic_cache_control_hook import (
            AnthropicCacheControlHook,
        )
        from litellm.llms.bedrock.chat.converse_transformation import (
            AmazonConverseConfig,
        )
        from litellm.types.utils import StandardCallbackDynamicParams

        plugin = _make_levers_plugin(cache_point=True, cache_tools=False)
        messages = [
            {"role": "system", "content": "long system prefix"},
            {"role": "user", "content": "hi"},
        ]
        kwargs = {"model": "m", "messages": messages}
        override = await plugin.on_llm_pre_call("m", messages, kwargs)

        # litellm applies message-location points via this hook before transform.
        hook = AnthropicCacheControlHook()
        _, processed_messages, _ = hook.get_chat_completion_prompt(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages,
            non_default_params={
                "cache_control_injection_points": override[
                    "cache_control_injection_points"
                ]
            },
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params=StandardCallbackDynamicParams(),
        )

        cfg = AmazonConverseConfig()
        body = cfg._transform_request(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=processed_messages,
            optional_params={},
            litellm_params={},
        )
        system_blocks = body.get("system", [])
        assert any("cachePoint" in block for block in system_blocks), (
            f"expected a system cachePoint block, got: {system_blocks}"
        )

    @pytest.mark.asyncio
    async def test_tool_cachepoint_reaches_converse_body(self):
        """The tool_config injection point drives a tool cachePoint entry in the
        actual transformed converse toolConfig (converse_transformation.py:1647)."""
        from litellm.llms.bedrock.chat.converse_transformation import (
            AmazonConverseConfig,
        )

        plugin = _make_levers_plugin(cache_point=True, cache_system=False)
        messages = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        kwargs = {"model": "m", "messages": messages, "tools": tools}
        override = await plugin.on_llm_pre_call("m", messages, kwargs)

        cfg = AmazonConverseConfig()
        optional_params = {
            "tools": tools,
            "cache_control_injection_points": override[
                "cache_control_injection_points"
            ],
        }
        body = cfg._transform_request(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
        )
        tool_config = body.get("toolConfig", {})
        bedrock_tools = tool_config.get("tools", [])
        assert any("cachePoint" in t for t in bedrock_tools), (
            f"expected a tool cachePoint entry, got: {bedrock_tools}"
        )

    @pytest.mark.asyncio
    async def test_tool_point_skipped_when_no_tools(self):
        plugin = _make_levers_plugin(cache_point=True)
        messages = [{"role": "system", "content": "sys"}]
        kwargs = {"model": "m", "messages": messages}  # no tools
        override = await plugin.on_llm_pre_call("m", messages, kwargs)
        points = override["cache_control_injection_points"]
        assert all(p["location"] != "tool_config" for p in points)

    @pytest.mark.asyncio
    async def test_caller_supplied_injection_points_not_clobbered(self):
        plugin = _make_levers_plugin(cache_point=True)
        messages = [{"role": "system", "content": "sys"}]
        kwargs = {
            "model": "m",
            "messages": messages,
            "cache_control_injection_points": [{"location": "message", "index": 0}],
        }
        override = await plugin.on_llm_pre_call("m", messages, kwargs)
        assert override is None
