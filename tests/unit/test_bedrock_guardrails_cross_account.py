"""Tests for bedrock_guardrails cross-account/cross-region per-arm (RouteIQ-3024)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.gateway.plugins.bedrock_guardrails import BedrockGuardrailsPlugin


@pytest.fixture
def _plugin() -> BedrockGuardrailsPlugin:
    return BedrockGuardrailsPlugin()


def _fake_boto3(client: MagicMock, sts: MagicMock | None = None) -> MagicMock:
    boto3 = MagicMock()

    def _client(service: str, **kwargs):
        if service == "sts":
            return sts
        return client

    boto3.client.side_effect = _client
    return boto3


def test_default_same_account_same_region(_plugin: BedrockGuardrailsPlugin) -> None:
    _plugin._region = "us-east-1"
    client = MagicMock()
    boto3 = _fake_boto3(client)
    resolved = _plugin._resolve_guardrail_client(boto3)
    assert resolved is client
    # built in the gateway region, no STS
    boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")
    # cached on the legacy attribute for backward compat
    assert _plugin._client is client


def test_cross_region_uses_guardrail_region(_plugin: BedrockGuardrailsPlugin) -> None:
    _plugin._region = "us-east-1"
    _plugin._guardrail_region = "eu-west-1"
    client = MagicMock()
    boto3 = _fake_boto3(client)
    _plugin._resolve_guardrail_client(boto3)
    boto3.client.assert_called_once_with("bedrock-runtime", region_name="eu-west-1")


def test_cross_account_assumes_role(_plugin: BedrockGuardrailsPlugin) -> None:
    _plugin._region = "us-east-1"
    _plugin._guardrail_account_role_arn = "arn:aws:iam::999:role/guardrail"
    client = MagicMock()
    sts = MagicMock()
    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AK",
            "SecretAccessKey": "SK",
            "SessionToken": "TOK",
        }
    }
    boto3 = _fake_boto3(client, sts)
    resolved = _plugin._resolve_guardrail_client(boto3)
    assert resolved is client
    sts.assume_role.assert_called_once()
    # bedrock-runtime client built with the temporary credentials
    bedrock_call = [
        c for c in boto3.client.call_args_list if c.args[0] == "bedrock-runtime"
    ][0]
    assert bedrock_call.kwargs["aws_access_key_id"] == "AK"
    assert bedrock_call.kwargs["aws_session_token"] == "TOK"


def test_per_arm_client_cached(_plugin: BedrockGuardrailsPlugin) -> None:
    _plugin._region = "us-east-1"
    _plugin._guardrail_region = "ap-south-1"
    client = MagicMock()
    boto3 = _fake_boto3(client)
    first = _plugin._resolve_guardrail_client(boto3)
    second = _plugin._resolve_guardrail_client(boto3)
    assert first is second
    # only one bedrock-runtime client built (cached by (region, role))
    assert (
        sum(1 for c in boto3.client.call_args_list if c.args[0] == "bedrock-runtime")
        == 1
    )
