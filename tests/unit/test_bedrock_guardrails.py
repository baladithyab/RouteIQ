"""
Tests for native Bedrock Guardrail activation in the bedrock_guardrails plugin.

Covers RouteIQ-9f14: wiring guardrailIdentifier/guardrailVersion into the
native Bedrock arm request path. When native activation is configured, the
plugin attaches a ``guardrailConfig`` block to the outbound request kwargs;
when it is not, the outbound request is left byte-for-byte unchanged.

The broader Bedrock plugin behaviour (ApplyGuardrail input filter, metadata,
health) is covered in ``test_guardrails.py``.
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.gateway.plugins.bedrock_guardrails import (
    BedrockGuardrailsPlugin,
)


def _make_plugin(
    *,
    enabled: bool = True,
    guardrail_id: str = "test-guardrail-123",
    guardrail_version: str = "DRAFT",
    native_activation: bool = False,
    trace: str = "disabled",
) -> BedrockGuardrailsPlugin:
    """Create a configured BedrockGuardrailsPlugin without touching env."""
    plugin = BedrockGuardrailsPlugin()
    plugin._enabled = enabled
    plugin._guardrail_id = guardrail_id
    plugin._guardrail_version = guardrail_version
    plugin._native_activation = native_activation
    plugin._trace = trace
    return plugin


# ---------------------------------------------------------------------------
# Defaults: native activation is OFF unless explicitly configured
# ---------------------------------------------------------------------------


def test_bedrock_guardrail_native_activation_off_by_default():
    plugin = BedrockGuardrailsPlugin()
    assert plugin._native_activation is False
    assert plugin._trace == "disabled"


@pytest.mark.asyncio
async def test_bedrock_guardrail_startup_native_activation_off_by_default(monkeypatch):
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ENABLED", "true")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ID", "gr-abc123")
    monkeypatch.delenv("BEDROCK_GUARDRAIL_NATIVE_ACTIVATION", raising=False)
    monkeypatch.delenv("BEDROCK_GUARDRAIL_TRACE", raising=False)
    plugin = BedrockGuardrailsPlugin()
    await plugin.startup(MagicMock())
    assert plugin._enabled is True
    assert plugin._native_activation is False
    assert plugin._trace == "disabled"


@pytest.mark.asyncio
async def test_bedrock_guardrail_startup_native_activation_on(monkeypatch):
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ENABLED", "true")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ID", "gr-abc123")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_VERSION", "7")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_NATIVE_ACTIVATION", "true")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_TRACE", "enabled")
    plugin = BedrockGuardrailsPlugin()
    await plugin.startup(MagicMock())
    assert plugin._native_activation is True
    assert plugin._trace == "enabled"
    assert plugin._guardrail_version == "7"


@pytest.mark.asyncio
async def test_bedrock_guardrail_startup_trace_defaults_disabled_on_garbage(
    monkeypatch,
):
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ENABLED", "true")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ID", "gr-abc123")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_NATIVE_ACTIVATION", "true")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_TRACE", "not-a-mode")
    plugin = BedrockGuardrailsPlugin()
    await plugin.startup(MagicMock())
    assert plugin._trace == "disabled"


# ---------------------------------------------------------------------------
# Off path is byte-stable: no guardrailConfig injected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bedrock_guardrail_no_config_when_native_activation_off():
    """With native activation off, the outbound request gets no guardrailConfig."""
    plugin = _make_plugin(native_activation=False)
    mock_client = MagicMock()
    mock_client.apply_guardrail.return_value = {
        "action": "NONE",
        "outputs": [],
        "assessments": [],
    }
    plugin._client = mock_client

    kwargs: dict = {}
    messages = [{"role": "user", "content": "Hello"}]
    mock_boto3 = MagicMock()
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = await plugin.on_llm_pre_call("bedrock/model", messages, kwargs)

    assert result is None
    assert "guardrailConfig" not in kwargs


@pytest.mark.asyncio
async def test_bedrock_guardrail_no_config_when_no_guardrail_id():
    """Native activation on but no id configured -> no guardrailConfig."""
    plugin = _make_plugin(native_activation=True, guardrail_id="")
    assert plugin._native_guardrail_override({}) is None


@pytest.mark.asyncio
async def test_bedrock_guardrail_no_config_when_plugin_disabled():
    plugin = _make_plugin(enabled=False, native_activation=True)
    kwargs: dict = {}
    result = await plugin.on_llm_pre_call("bedrock/model", [], kwargs)
    assert result is None
    assert "guardrailConfig" not in kwargs


# ---------------------------------------------------------------------------
# On path: guardrailConfig injected into the outbound request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bedrock_guardrail_config_injected_when_native_activation_on():
    """With native activation on, the outbound request carries guardrailConfig."""
    plugin = _make_plugin(
        native_activation=True,
        guardrail_id="gr-xyz789",
        guardrail_version="3",
        trace="enabled",
    )
    mock_client = MagicMock()
    mock_client.apply_guardrail.return_value = {
        "action": "NONE",
        "outputs": [],
        "assessments": [],
    }
    plugin._client = mock_client

    kwargs: dict = {}
    messages = [{"role": "user", "content": "Hello"}]
    mock_boto3 = MagicMock()
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = await plugin.on_llm_pre_call("bedrock/model", messages, kwargs)

    assert result == {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-xyz789",
            "guardrailVersion": "3",
            "trace": "enabled",
        }
    }


@pytest.mark.asyncio
async def test_bedrock_guardrail_config_injected_with_empty_messages():
    """Native activation injects config even when there is no user message."""
    plugin = _make_plugin(native_activation=True, guardrail_id="gr-1", trace="disabled")
    kwargs: dict = {}
    result = await plugin.on_llm_pre_call("bedrock/model", [], kwargs)
    assert result == {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-1",
            "guardrailVersion": "DRAFT",
            "trace": "disabled",
        }
    }


@pytest.mark.asyncio
async def test_bedrock_guardrail_does_not_clobber_caller_config():
    """A caller-supplied guardrailConfig is respected, not overwritten."""
    plugin = _make_plugin(native_activation=True, guardrail_id="gr-plugin")
    caller_config = {"guardrailIdentifier": "gr-caller", "guardrailVersion": "1"}
    kwargs = {"guardrailConfig": caller_config}
    result = plugin._native_guardrail_override(kwargs)
    assert result is None
    assert kwargs["guardrailConfig"] is caller_config


@pytest.mark.asyncio
async def test_bedrock_guardrail_native_override_helper_off():
    plugin = _make_plugin(native_activation=False)
    assert plugin._native_guardrail_override({}) is None


@pytest.mark.asyncio
async def test_bedrock_guardrail_health_reports_native_activation():
    plugin = _make_plugin(native_activation=True)
    health = await plugin.health_check()
    assert health["native_activation"] is True
