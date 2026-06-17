"""Tests for circuit-breaker state-metric re-baseline at startup (RouteIQ-c714)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.resilience import (
    CircuitBreakerManager,
    reset_circuit_breaker_manager,
    reset_circuit_breaker_state_metrics,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_circuit_breaker_manager()
    yield
    reset_circuit_breaker_manager()


def test_noop_when_metrics_unavailable() -> None:
    with patch("litellm_llmrouter.metrics.get_gateway_metrics", return_value=None):
        assert reset_circuit_breaker_state_metrics() == 0


def test_rebaselines_known_infra_breakers() -> None:
    metrics = MagicMock()
    mgr = CircuitBreakerManager()
    with patch("litellm_llmrouter.metrics.get_gateway_metrics", return_value=metrics):
        count = reset_circuit_breaker_state_metrics(mgr)
    # database + redis + leader_election
    assert count == 3
    # each breaker drives its 'closed' series up via record_circuit_breaker_state
    calls = [c.args for c in metrics.record_circuit_breaker_state.call_args_list]
    names = {c[0] for c in calls}
    assert names == {"database", "redis", "leader_election"}
    for _name, from_state, to_state in calls:
        assert from_state == "open"
        assert to_state == "closed"


def test_rebaselines_existing_provider_breakers() -> None:
    metrics = MagicMock()
    mgr = CircuitBreakerManager()
    # touch a provider breaker so its series exists on the manager
    mgr.get_or_create_provider_breaker("openai")
    with patch("litellm_llmrouter.metrics.get_gateway_metrics", return_value=metrics):
        count = reset_circuit_breaker_state_metrics(mgr)
    names = {c.args[0] for c in metrics.record_circuit_breaker_state.call_args_list}
    assert "provider:openai" in names
    assert count == 4  # 3 infra + 1 provider


def test_resets_open_series_to_zero() -> None:
    """Each non-closed series is explicitly driven to 0 (clears a stale open)."""
    metrics = MagicMock()
    mgr = CircuitBreakerManager()
    with patch("litellm_llmrouter.metrics.get_gateway_metrics", return_value=metrics):
        reset_circuit_breaker_state_metrics(mgr)
    # circuit_breaker_state.add(0, {state: open|half_open}) called per breaker
    add_calls = metrics.circuit_breaker_state.add.call_args_list
    open_zeroed = [
        c for c in add_calls if c.args[0] == 0 and c.args[1].get("state") == "open"
    ]
    assert len(open_zeroed) == 3
