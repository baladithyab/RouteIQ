"""
Tests for Content/Toxicity Filter Plugin and related plugin_manager additions.

Tests cover:
1. Content filter detects each category (violence, hate_speech, sexual, self_harm, illegal_activity)
2. Content filter handles clean content (no false positives)
3. Score threshold behavior (below threshold = pass, above = action)
4. Action modes (block raises GuardrailBlockError, warn logs, log is silent)
5. Input filtering (on_llm_pre_call)
6. Output filtering (on_llm_success)
7. Feature flag disabled is no-op
8. Category configuration (only scan configured categories)
9. Plugin metadata and capabilities
10. New PluginCapability enum values exist
11. New GatewayPlugin hook methods exist as no-ops
12. Empty/None message handling
"""

import os

import pytest
from unittest.mock import MagicMock


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove content filter env vars before each test."""
    for key in list(os.environ.keys()):
        if key.startswith("CONTENT_FILTER_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def plugin():
    """Create a fresh ContentFilterPlugin instance."""
    from litellm_llmrouter.gateway.plugins.content_filter import ContentFilterPlugin

    return ContentFilterPlugin()


@pytest.fixture
def enabled_plugin(monkeypatch):
    """Create a ContentFilterPlugin with all categories enabled."""
    monkeypatch.setenv("CONTENT_FILTER_ENABLED", "true")
    monkeypatch.setenv("CONTENT_FILTER_THRESHOLD", "0.3")
    monkeypatch.setenv("CONTENT_FILTER_ACTION", "block")

    from litellm_llmrouter.gateway.plugins.content_filter import ContentFilterPlugin

    p = ContentFilterPlugin()
    return p


@pytest.fixture
def mock_app():
    """Create a mock FastAPI app."""
    return MagicMock()


# ============================================================================
# Plugin Metadata and Capabilities
# ============================================================================


class TestPluginMetadata:
    """Test plugin metadata and capability declarations."""

    def test_metadata_name(self, plugin):
        assert plugin.metadata.name == "content-filter"

    def test_metadata_version(self, plugin):
        assert plugin.metadata.version == "1.0.0"

    def test_metadata_capabilities(self, plugin):
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        assert PluginCapability.GUARDRAIL in plugin.metadata.capabilities

    def test_metadata_priority(self, plugin):
        assert plugin.metadata.priority == 70

    def test_metadata_description(self, plugin):
        assert "content" in plugin.metadata.description.lower()

    def test_plugin_name_property(self, plugin):
        assert plugin.name == "content-filter"


# ============================================================================
# New PluginCapability Enum Values
# ============================================================================


class TestNewPluginCapabilities:
    """Test that new PluginCapability enum values exist."""

    def test_guardrail_capability_exists(self):
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        assert PluginCapability.GUARDRAIL.value == "guardrail"

    def test_cache_capability_exists(self):
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        assert PluginCapability.CACHE.value == "cache"

    def test_cost_tracker_capability_exists(self):
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        assert PluginCapability.COST_TRACKER.value == "cost_tracker"

    def test_existing_capabilities_unchanged(self):
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        assert PluginCapability.ROUTES.value == "routes"
        assert PluginCapability.EVALUATOR.value == "evaluator"
        assert PluginCapability.MIDDLEWARE.value == "middleware"


# ============================================================================
# New GatewayPlugin Hook Methods
# ============================================================================


class TestNewGatewayPluginHooks:
    """Test that new GatewayPlugin hook methods exist and are no-ops."""

    @pytest.fixture
    def noop_plugin(self):
        from litellm_llmrouter.gateway.plugin_manager import NoOpPlugin

        return NoOpPlugin()

    @pytest.mark.asyncio
    async def test_on_config_reload_exists(self, noop_plugin):
        result = await noop_plugin.on_config_reload(
            {"old": "config"}, {"new": "config"}
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_on_route_register_exists(self, noop_plugin):
        result = await noop_plugin.on_route_register("/api/test", ["GET", "POST"])
        assert result is None

    @pytest.mark.asyncio
    async def test_on_model_health_change_exists(self, noop_plugin):
        result = await noop_plugin.on_model_health_change("gpt-4", True, "recovered")
        assert result is None

    @pytest.mark.asyncio
    async def test_on_config_reload_on_base_class(self):
        """Verify the method is defined on GatewayPlugin base class."""
        from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin

        assert hasattr(GatewayPlugin, "on_config_reload")

    @pytest.mark.asyncio
    async def test_on_route_register_on_base_class(self):
        from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin

        assert hasattr(GatewayPlugin, "on_route_register")

    @pytest.mark.asyncio
    async def test_on_model_health_change_on_base_class(self):
        from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin

        assert hasattr(GatewayPlugin, "on_model_health_change")


# ============================================================================
# PluginManager.get_guardrail_plugins
# ============================================================================


class TestGetGuardrailPlugins:
    """Test the PluginManager.get_guardrail_plugins method."""

    def test_returns_guardrail_plugins(self, plugin, mock_app):
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            reset_plugin_manager,
        )

        reset_plugin_manager()
        pm = PluginManager()
        pm.register(plugin)
        result = pm.get_guardrail_plugins()
        assert len(result) == 1
        assert result[0].name == "content-filter"

    def test_excludes_non_guardrail_plugins(self):
        from litellm_llmrouter.gateway.plugin_manager import (
            NoOpPlugin,
            PluginManager,
            reset_plugin_manager,
        )

        reset_plugin_manager()
        pm = PluginManager()
        pm.register(NoOpPlugin())
        result = pm.get_guardrail_plugins()
        assert len(result) == 0

    def test_excludes_quarantined_plugins(self, plugin):
        from litellm_llmrouter.gateway.plugin_manager import (
            PluginManager,
            reset_plugin_manager,
        )

        reset_plugin_manager()
        pm = PluginManager()
        pm.register(plugin)
        pm._quarantined.add("content-filter")
        result = pm.get_guardrail_plugins()
        assert len(result) == 0


# ============================================================================
# Startup and Configuration
# ============================================================================


class TestStartup:
    """Test plugin startup and configuration loading."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self, plugin, mock_app):
        await plugin.startup(mock_app)
        assert not plugin._enabled
        assert plugin._categories == {}

    @pytest.mark.asyncio
    async def test_enabled_via_env(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        assert enabled_plugin._enabled
        assert len(enabled_plugin._categories) == 5

    @pytest.mark.asyncio
    async def test_custom_threshold(self, monkeypatch, mock_app):
        monkeypatch.setenv("CONTENT_FILTER_ENABLED", "true")
        monkeypatch.setenv("CONTENT_FILTER_THRESHOLD", "0.9")

        from litellm_llmrouter.gateway.plugins.content_filter import (
            ContentFilterPlugin,
        )

        p = ContentFilterPlugin()
        await p.startup(mock_app)
        for config in p._categories.values():
            assert config.threshold == 0.9

    @pytest.mark.asyncio
    async def test_custom_action(self, monkeypatch, mock_app):
        monkeypatch.setenv("CONTENT_FILTER_ENABLED", "true")
        monkeypatch.setenv("CONTENT_FILTER_ACTION", "warn")

        from litellm_llmrouter.gateway.plugins.content_filter import (
            ContentFilterPlugin,
        )

        p = ContentFilterPlugin()
        await p.startup(mock_app)
        for config in p._categories.values():
            assert config.action == "warn"

    @pytest.mark.asyncio
    async def test_subset_of_categories(self, monkeypatch, mock_app):
        monkeypatch.setenv("CONTENT_FILTER_ENABLED", "true")
        monkeypatch.setenv("CONTENT_FILTER_CATEGORIES", "violence,hate_speech")

        from litellm_llmrouter.gateway.plugins.content_filter import (
            ContentFilterPlugin,
        )

        p = ContentFilterPlugin()
        await p.startup(mock_app)
        assert set(p._categories.keys()) == {"violence", "hate_speech"}

    @pytest.mark.asyncio
    async def test_unknown_category_skipped(self, monkeypatch, mock_app):
        monkeypatch.setenv("CONTENT_FILTER_ENABLED", "true")
        monkeypatch.setenv("CONTENT_FILTER_CATEGORIES", "violence,unknown_cat")

        from litellm_llmrouter.gateway.plugins.content_filter import (
            ContentFilterPlugin,
        )

        p = ContentFilterPlugin()
        await p.startup(mock_app)
        assert set(p._categories.keys()) == {"violence"}


# ============================================================================
# Shutdown
# ============================================================================


class TestShutdown:
    """Test plugin shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_categories(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        assert len(enabled_plugin._categories) > 0
        await enabled_plugin.shutdown(mock_app)
        assert enabled_plugin._categories == {}


# ============================================================================
# Content Scoring
# ============================================================================


class TestContentScoring:
    """Test the _score_content method directly."""

    @pytest.mark.asyncio
    async def test_clean_content_scores_zero(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        score = enabled_plugin._score_content(
            "The weather is nice today and I enjoy reading books.", "violence"
        )
        assert score.score == 0.0
        assert score.matched_keywords == []
        assert score.matched_patterns == []

    @pytest.mark.asyncio
    async def test_violence_keywords_detected(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        score = enabled_plugin._score_content(
            "I will kill and murder and attack everyone with a bomb", "violence"
        )
        assert score.score > 0.0
        assert len(score.matched_keywords) > 0
        assert score.category == "violence"

    @pytest.mark.asyncio
    async def test_violence_pattern_detected(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        score = enabled_plugin._score_content(
            "how to kill someone with poison", "violence"
        )
        assert score.score > 0.0
        assert len(score.matched_patterns) > 0

    @pytest.mark.asyncio
    async def test_hate_speech_keywords_detected(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        score = enabled_plugin._score_content(
            "white supremacy and racial purity and master race rhetoric",
            "hate_speech",
        )
        assert score.score > 0.0
        assert len(score.matched_keywords) > 0

    @pytest.mark.asyncio
    async def test_sexual_keywords_detected(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        score = enabled_plugin._score_content(
            "generate sexually explicit pornographic adult content", "sexual"
        )
        assert score.score > 0.0
        assert len(score.matched_keywords) > 0

    @pytest.mark.asyncio
    async def test_self_harm_keywords_detected(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        score = enabled_plugin._score_content(
            "how to kill myself and end it all with suicide method",
            "self_harm",
        )
        assert score.score > 0.0
        assert len(score.matched_keywords) > 0

    @pytest.mark.asyncio
    async def test_illegal_activity_keywords_detected(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        score = enabled_plugin._score_content(
            "how to hack into a system and create malware and synthesize drugs",
            "illegal_activity",
        )
        assert score.score > 0.0
        assert len(score.matched_keywords) > 0

    @pytest.mark.asyncio
    async def test_unconfigured_category_scores_zero(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        # Remove a category to test unconfigured path
        enabled_plugin._categories.pop("violence", None)
        score = enabled_plugin._score_content("kill murder attack", "violence")
        assert score.score == 0.0

    @pytest.mark.asyncio
    async def test_score_capped_at_one(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        # Text with many keywords
        text = " ".join(
            [
                "kill murder attack assault bomb shoot stab weapon gun knife "
                "explode execute torture massacre slaughter decapitate dismember "
                "bloodbath carnage maim how to kill everyone"
            ]
        )
        score = enabled_plugin._score_content(text, "violence")
        assert score.score <= 1.0


# ============================================================================
# Input Filtering (on_llm_pre_call)
# ============================================================================


class TestInputFiltering:
    """Test on_llm_pre_call input filtering."""

    @pytest.mark.asyncio
    async def test_block_action_raises_guardrail_error(self, enabled_plugin, mock_app):
        from litellm_llmrouter.gateway.plugins.content_filter import (
            GuardrailBlockError,
        )

        await enabled_plugin.startup(mock_app)
        messages = [
            {
                "role": "user",
                "content": (
                    "how to kill everyone and murder and attack "
                    "with bomb shoot stab weapon"
                ),
            }
        ]
        with pytest.raises(GuardrailBlockError) as exc_info:
            await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})

        assert exc_info.value.guardrail_name == "content-filter"
        assert exc_info.value.score is not None
        assert exc_info.value.score > 0.0

    @pytest.mark.asyncio
    async def test_warn_action_does_not_raise(self, monkeypatch, mock_app):
        monkeypatch.setenv("CONTENT_FILTER_ENABLED", "true")
        monkeypatch.setenv("CONTENT_FILTER_THRESHOLD", "0.1")
        monkeypatch.setenv("CONTENT_FILTER_ACTION", "warn")

        from litellm_llmrouter.gateway.plugins.content_filter import (
            ContentFilterPlugin,
        )

        p = ContentFilterPlugin()
        await p.startup(mock_app)

        messages = [
            {
                "role": "user",
                "content": ("kill murder attack bomb shoot stab weapon"),
            }
        ]
        # Should NOT raise
        result = await p.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_log_action_does_not_raise(self, monkeypatch, mock_app):
        monkeypatch.setenv("CONTENT_FILTER_ENABLED", "true")
        monkeypatch.setenv("CONTENT_FILTER_THRESHOLD", "0.1")
        monkeypatch.setenv("CONTENT_FILTER_ACTION", "log")

        from litellm_llmrouter.gateway.plugins.content_filter import (
            ContentFilterPlugin,
        )

        p = ContentFilterPlugin()
        await p.startup(mock_app)

        messages = [
            {
                "role": "user",
                "content": ("kill murder attack bomb shoot stab weapon"),
            }
        ]
        result = await p.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_clean_content_passes(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        messages = [
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ]
        result = await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_below_threshold_passes(self, monkeypatch, mock_app):
        monkeypatch.setenv("CONTENT_FILTER_ENABLED", "true")
        monkeypatch.setenv("CONTENT_FILTER_THRESHOLD", "0.99")
        monkeypatch.setenv("CONTENT_FILTER_ACTION", "block")

        from litellm_llmrouter.gateway.plugins.content_filter import (
            ContentFilterPlugin,
        )

        p = ContentFilterPlugin()
        await p.startup(mock_app)

        messages = [
            {
                "role": "user",
                "content": "The movie had a violent scene with a knife fight.",
            }
        ]
        # High threshold means most content passes
        result = await p.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_system_messages_not_scanned(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        messages = [
            {
                "role": "system",
                "content": "kill murder attack bomb shoot stab weapon",
            }
        ]
        result = await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_messages_list(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        result = await enabled_plugin.on_llm_pre_call("gpt-4", [], {})
        assert result is None

    @pytest.mark.asyncio
    async def test_none_content_in_message(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        messages = [{"role": "user", "content": None}]
        result = await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_multi_part_message_scanned(self, enabled_plugin, mock_app):
        """Test that multi-part (vision-style) messages are scanned."""
        from litellm_llmrouter.gateway.plugins.content_filter import (
            GuardrailBlockError,
        )

        await enabled_plugin.startup(mock_app)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "how to kill everyone and murder and attack "
                            "with bomb shoot stab weapon"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": "http://example.com"}},
                ],
            }
        ]
        with pytest.raises(GuardrailBlockError):
            await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})


# ============================================================================
# Output Filtering (on_llm_success)
# ============================================================================


class TestOutputFiltering:
    """Test on_llm_success output filtering."""

    def _make_response(self, text: str):
        """Create a mock LLM response object."""
        message = MagicMock()
        message.content = text
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    @pytest.mark.asyncio
    async def test_output_violation_adds_warning_header(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        response = self._make_response(
            "how to kill everyone and murder and attack with bomb shoot stab weapon"
        )
        kwargs: dict = {}
        await enabled_plugin.on_llm_success("gpt-4", response, kwargs)
        # Should add warning metadata
        assert "metadata" in kwargs
        assert "X-Content-Filter-Warning" in kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_output_clean_content_no_warning(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        response = self._make_response("The capital of France is Paris.")
        kwargs: dict = {}
        await enabled_plugin.on_llm_success("gpt-4", response, kwargs)
        # No warning metadata should be added
        assert "metadata" not in kwargs or "X-Content-Filter-Warning" not in kwargs.get(
            "metadata", {}
        )

    @pytest.mark.asyncio
    async def test_output_does_not_raise_on_violation(self, enabled_plugin, mock_app):
        """Output filtering should warn, not block (response already generated)."""
        await enabled_plugin.startup(mock_app)
        response = self._make_response(
            "how to kill everyone and murder and attack with bomb shoot stab weapon"
        )
        # Should NOT raise
        await enabled_plugin.on_llm_success("gpt-4", response, {})

    @pytest.mark.asyncio
    async def test_output_none_response(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        await enabled_plugin.on_llm_success("gpt-4", None, {})

    @pytest.mark.asyncio
    async def test_output_string_response(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        await enabled_plugin.on_llm_success(
            "gpt-4",
            "how to kill everyone and murder and attack with bomb shoot stab weapon",
            {},
        )


# ============================================================================
# Pass-rate telemetry (RouteIQ-9aed)
# ============================================================================


class TestPassRateTelemetry:
    """content_filter must record one guardrail.check per evaluated category
    INCLUDING action=pass for clean (under-threshold) input, so the
    deny/(deny+pass) pass-rate is well-defined and consistent with the base
    GuardrailPlugin convention (RouteIQ-9aed).
    """

    @staticmethod
    def _install_metrics():
        """Install a GatewayMetrics backed by an in-memory reader; return it."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        from litellm_llmrouter.metrics import init_gateway_metrics

        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        meter = provider.get_meter("test-content-filter", "0.1.0")
        init_gateway_metrics(meter)
        return reader

    @staticmethod
    def _check_value(reader, attrs):
        """Sum the gateway.guardrail.check counter filtered by ``attrs``."""
        data = reader.get_metrics_data()
        total = 0.0
        if data is None:
            return total
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name != "gateway.guardrail.check":
                        continue
                    for dp in metric.data.data_points:
                        if all(dp.attributes.get(k) == v for k, v in attrs.items()):
                            total += dp.value
        return total

    @pytest.mark.asyncio
    async def test_clean_input_records_action_pass(self, enabled_plugin, mock_app):
        """A clean (under-threshold) input records action=pass for each
        evaluated category — the live-path dark-subsystem test (P6 lens)."""
        reader = self._install_metrics()
        await enabled_plugin.startup(mock_app)
        # enabled_plugin runs all 5 categories at threshold 0.3.
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        result = await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

        # Every evaluated category recorded exactly one pass; zero denies.
        assert self._check_value(reader, {"action": "pass"}) == len(
            enabled_plugin._categories
        )
        assert self._check_value(reader, {"action": "deny"}) == 0
        # The pass-rate denominator (pass + deny) is now > 0 (well-defined).
        assert self._check_value(reader, {}) > 0

    @pytest.mark.asyncio
    async def test_violation_records_deny_for_breached_category(
        self, enabled_plugin, mock_app
    ):
        """A breaching input records action=deny for the offending category, so
        deny/(deny+pass) is computable alongside the clean-path passes."""
        from litellm_llmrouter.gateway.plugins.content_filter import (
            GuardrailBlockError,
        )

        reader = self._install_metrics()
        await enabled_plugin.startup(mock_app)
        messages = [
            {
                "role": "user",
                "content": (
                    "how to kill everyone and murder and attack "
                    "with bomb shoot stab weapon"
                ),
            }
        ]
        with pytest.raises(GuardrailBlockError):
            await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})

        # The violence category denied; the denominator includes it.
        assert (
            self._check_value(reader, {"check_type": "violence", "action": "deny"}) == 1
        )

    @pytest.mark.asyncio
    async def test_clean_input_no_metric_singleton_does_not_raise(
        self, enabled_plugin, mock_app
    ):
        """No GatewayMetrics singleton (OTel off) -> recording is a silent
        no-op; the clean path must not raise."""
        from litellm_llmrouter.metrics import reset_gateway_metrics

        reset_gateway_metrics()
        await enabled_plugin.startup(mock_app)
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        result = await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None


# ============================================================================
# Feature Flag
# ============================================================================


class TestFeatureFlag:
    """Test that disabled plugin is a complete no-op."""

    @pytest.mark.asyncio
    async def test_disabled_pre_call_is_noop(self, plugin, mock_app):
        await plugin.startup(mock_app)
        messages = [
            {
                "role": "user",
                "content": "kill murder attack bomb shoot stab weapon",
            }
        ]
        result = await plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_success_is_noop(self, plugin, mock_app):
        await plugin.startup(mock_app)
        response = MagicMock()
        response.choices = []
        kwargs: dict = {}
        await plugin.on_llm_success("gpt-4", response, kwargs)
        assert "metadata" not in kwargs


# ============================================================================
# GuardrailBlockError
# ============================================================================


class TestGuardrailBlockError:
    """Test the GuardrailBlockError exception."""

    def test_basic_construction(self):
        from litellm_llmrouter.gateway.plugins.content_filter import (
            GuardrailBlockError,
        )

        err = GuardrailBlockError(
            guardrail_name="content-filter",
            category="violence",
            message="Content violation detected",
        )
        assert err.guardrail_name == "content-filter"
        assert err.category == "violence"
        assert "Content violation detected" in str(err)

    def test_with_score_and_category(self):
        from litellm_llmrouter.gateway.plugins.content_filter import (
            GuardrailBlockError,
        )

        err = GuardrailBlockError(
            guardrail_name="content-filter",
            category="violence",
            message="Violence detected",
            score=0.85,
        )
        assert err.score == 0.85
        assert err.category == "violence"

    def test_is_exception(self):
        from litellm_llmrouter.gateway.plugins.content_filter import (
            GuardrailBlockError,
        )

        err = GuardrailBlockError(
            guardrail_name="test", category="test", message="test"
        )
        assert isinstance(err, Exception)


# ============================================================================
# Health Check
# ============================================================================


class TestHealthCheck:
    """Test the health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_disabled(self, plugin, mock_app):
        await plugin.startup(mock_app)
        health = await plugin.health_check()
        assert health["status"] == "disabled"
        assert health["enabled"] is False
        assert health["categories"] == []

    @pytest.mark.asyncio
    async def test_health_check_enabled(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        health = await enabled_plugin.health_check()
        assert health["status"] == "ok"
        assert health["enabled"] is True
        assert len(health["categories"]) == 5


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_string_message(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        messages = [{"role": "user", "content": ""}]
        result = await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_non_dict_message_in_list(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        messages = ["not a dict", {"role": "user", "content": "hello"}]
        result = await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_message_without_role(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        messages = [{"content": "kill murder attack bomb"}]
        result = await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_assistant_messages_not_scanned(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        messages = [
            {
                "role": "assistant",
                "content": "kill murder attack bomb shoot stab weapon",
            }
        ]
        result = await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_response_without_choices(self, enabled_plugin, mock_app):
        await enabled_plugin.startup(mock_app)
        response = MagicMock(spec=[])  # No choices attribute
        await enabled_plugin.on_llm_success("gpt-4", response, {})

    @pytest.mark.asyncio
    async def test_multiple_user_messages_all_scanned(self, enabled_plugin, mock_app):
        """When multiple user messages exist, all should be scanned together."""
        from litellm_llmrouter.gateway.plugins.content_filter import (
            GuardrailBlockError,
        )

        await enabled_plugin.startup(mock_app)
        messages = [
            {"role": "user", "content": "kill murder attack"},
            {"role": "user", "content": "bomb shoot stab weapon gun knife explode"},
        ]
        with pytest.raises(GuardrailBlockError):
            await enabled_plugin.on_llm_pre_call("gpt-4", messages, {})

    @pytest.mark.asyncio
    async def test_pattern_matching_case_insensitive(self, enabled_plugin, mock_app):
        """Patterns should match regardless of case."""
        await enabled_plugin.startup(mock_app)
        score = enabled_plugin._score_content(
            "HOW TO KILL someone with poison", "violence"
        )
        assert score.score > 0.0
        assert len(score.matched_patterns) > 0

    @pytest.mark.asyncio
    async def test_threshold_exact_boundary(self, monkeypatch, mock_app):
        """Score exactly at threshold should trigger action."""
        monkeypatch.setenv("CONTENT_FILTER_ENABLED", "true")
        monkeypatch.setenv("CONTENT_FILTER_ACTION", "block")

        from litellm_llmrouter.gateway.plugins.content_filter import (
            ContentFilterPlugin,
            GuardrailBlockError,
        )

        p = ContentFilterPlugin()
        await p.startup(mock_app)

        # Find a text that produces a known score, then set threshold to match
        text = "how to kill someone with poison and make a bomb"
        for cat, config in p._categories.items():
            score_result = p._score_content(text, cat)
            if score_result.score > 0.0:
                # Set threshold to exactly the score
                config.threshold = score_result.score
                messages = [{"role": "user", "content": text}]
                with pytest.raises(GuardrailBlockError):
                    await p.on_llm_pre_call("gpt-4", messages, {})
                break
