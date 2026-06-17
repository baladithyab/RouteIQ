"""Tests for the DURING_CALL (streaming) guardrail phase (RouteIQ-0ae5).

The during_call phase lets an output-style guardrail (e.g. ``regex_deny`` of a
leaked secret) fire MID-STREAM on a rolling buffer instead of only after the
full response is assembled. These tests cover:

  * the new ``GuardrailPhase.DURING_CALL`` enum member,
  * ``evaluate_during_call`` runs DURING_CALL policies on the buffer and is a
    byte-stable no-op (empty list) when no during_call policy is registered,
  * a during_call DENY policy fires on the buffered content,
  * the ``StreamingGuardrailBuffer`` cadence (feed/flush/reset),
  * ``get_status`` counts during_call policies.
"""

from __future__ import annotations

import pytest

from litellm_llmrouter.guardrail_policies import (
    GuardrailAction,
    GuardrailPhase,
    GuardrailPolicy,
    GuardrailPolicyEngine,
    GuardrailType,
    StreamingGuardrailBuffer,
    reset_guardrail_policy_engine,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_guardrail_policy_engine()
    yield
    reset_guardrail_policy_engine()


def _engine_with_during_call_deny() -> GuardrailPolicyEngine:
    engine = GuardrailPolicyEngine()
    engine.add_policy(
        GuardrailPolicy(
            guardrail_id="leak-mid-stream",
            name="block leaked secret mid-stream",
            phase=GuardrailPhase.DURING_CALL,
            check_type=GuardrailType.REGEX_DENY,
            parameters={"patterns": ["(?i)sk-secret-[a-z0-9]+"]},
            action=GuardrailAction.DENY,
        )
    )
    return engine


def test_during_call_phase_enum_exists() -> None:
    assert GuardrailPhase.DURING_CALL.value == "during_call"


async def test_evaluate_during_call_noop_when_no_policy() -> None:
    """No during_call policy -> empty list (byte-stable hot-path no-op)."""
    engine = GuardrailPolicyEngine()
    results = await engine.evaluate_during_call("anything streamed so far")
    assert results == []


async def test_evaluate_during_call_passes_clean_buffer() -> None:
    engine = _engine_with_during_call_deny()
    results = await engine.evaluate_during_call("a perfectly clean response")
    assert len(results) == 1
    assert results[0].passed is True
    assert not engine.has_deny_result(results)


async def test_evaluate_during_call_denies_on_leaked_secret() -> None:
    engine = _engine_with_during_call_deny()
    results = await engine.evaluate_during_call("oops here is sk-secret-abc123 leaked")
    assert engine.has_deny_result(results)


async def test_input_output_policies_not_run_during_call() -> None:
    """An INPUT/OUTPUT policy must not be evaluated in the during_call phase."""
    engine = GuardrailPolicyEngine()
    engine.add_policy(
        GuardrailPolicy(
            guardrail_id="out-only",
            name="output only",
            phase=GuardrailPhase.OUTPUT,
            check_type=GuardrailType.REGEX_DENY,
            parameters={"patterns": ["secret"]},
            action=GuardrailAction.DENY,
        )
    )
    results = await engine.evaluate_during_call("this has a secret in it")
    assert results == []  # output policy is not a during_call policy


def test_streaming_buffer_cadence() -> None:
    buf = StreamingGuardrailBuffer(flush_bytes=10)
    assert buf.feed("12345") is False  # 5 bytes, below threshold
    assert buf.text == "12345"
    assert buf.feed("67890") is True  # now 10 -> cadence reached, counter resets
    assert buf.text == "1234567890"
    assert buf.feed("a") is False  # counter reset, 1 byte
    buf.reset()
    assert buf.text == ""


def test_streaming_buffer_ignores_empty_chunk() -> None:
    buf = StreamingGuardrailBuffer(flush_bytes=4)
    assert buf.feed("") is False
    assert buf.text == ""


def test_get_status_counts_during_call() -> None:
    engine = _engine_with_during_call_deny()
    status = engine.get_status()
    assert status["during_call_policies"] == 1
    assert status["input_policies"] == 0
    assert status["output_policies"] == 0
