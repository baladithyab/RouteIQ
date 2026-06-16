"""Unit tests for Model Overview real health status (RouteIQ-c8d5).

Proves the Status column reflects REAL deployment health derived from the live
per-provider circuit-breaker state, not a cosmetic always-active value:

  * closed breaker  -> active
  * half_open       -> degraded
  * open            -> unavailable
  * no breaker      -> active (no failures observed)

Exercises the pure mapping helpers AND the live ``CircuitBreakerManager`` so a
healthy vs open-breaker deployment yields different statuses end-to-end.
"""

from litellm_llmrouter.resilience import (
    CircuitBreakerState,
    get_circuit_breaker_manager,
)
from litellm_llmrouter.routes.admin_ui import (
    _breaker_state_to_status,
    _build_provider_status_map,
    _model_entry_to_info,
    _resolve_model_status,
)


class TestBreakerStateMapping:
    def test_closed_is_active(self):
        assert _breaker_state_to_status(CircuitBreakerState.CLOSED) == "active"
        assert _breaker_state_to_status("closed") == "active"

    def test_open_is_unavailable(self):
        assert _breaker_state_to_status(CircuitBreakerState.OPEN) == "unavailable"
        assert _breaker_state_to_status("open") == "unavailable"

    def test_half_open_is_degraded(self):
        assert _breaker_state_to_status(CircuitBreakerState.HALF_OPEN) == "degraded"
        assert _breaker_state_to_status("half_open") == "degraded"

    def test_unknown_defaults_active(self):
        assert _breaker_state_to_status("mystery") == "active"


class TestProviderStatusMap:
    def test_open_provider_breaker_reports_unavailable(self):
        """An open provider breaker drives the model status to unavailable."""
        manager = get_circuit_breaker_manager()
        breaker = manager.get_or_create_provider_breaker("openai")
        # Force the breaker OPEN directly (private state — same as resilience tests).
        breaker._state = CircuitBreakerState.OPEN

        statuses = _build_provider_status_map()
        assert statuses["openai"] == "unavailable"

    def test_closed_provider_breaker_reports_active(self):
        manager = get_circuit_breaker_manager()
        manager.get_or_create_provider_breaker("anthropic")  # closed by default

        statuses = _build_provider_status_map()
        assert statuses["anthropic"] == "active"

    def test_non_provider_breakers_excluded(self):
        manager = get_circuit_breaker_manager()
        # A non-provider breaker (e.g. database) must NOT appear in the map.
        manager.get_breaker("database")
        manager.get_or_create_provider_breaker("bedrock")

        statuses = _build_provider_status_map()
        assert "bedrock" in statuses
        assert "database" not in statuses

    def test_healthy_vs_open_breaker_differ(self):
        """Two providers, one healthy one open, get DIFFERENT statuses."""
        manager = get_circuit_breaker_manager()
        manager.get_or_create_provider_breaker("anthropic")  # closed
        open_breaker = manager.get_or_create_provider_breaker("openai")
        open_breaker._state = CircuitBreakerState.OPEN

        statuses = _build_provider_status_map()
        assert statuses["anthropic"] == "active"
        assert statuses["openai"] == "unavailable"


class TestResolveModelStatus:
    def test_defaults_active_when_provider_absent(self):
        assert _resolve_model_status("cohere", {}) == "active"

    def test_uses_provider_status(self):
        provider_status = {"openai": "unavailable"}
        assert _resolve_model_status("openai", provider_status) == "unavailable"


class TestModelEntryToInfo:
    def test_status_from_open_breaker(self):
        entry = {
            "model_name": "gpt-4o",
            "litellm_params": {"model": "openai/gpt-4o"},
        }
        info = _model_entry_to_info(entry, {"openai": "unavailable"})
        assert info["provider"] == "openai"
        assert info["status"] == "unavailable"

    def test_explicit_provider_override_wins(self):
        entry = {
            "model_name": "custom",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "custom_llm_provider": "azure",
            },
        }
        info = _model_entry_to_info(entry, {"azure": "degraded"})
        assert info["provider"] == "azure"
        assert info["status"] == "degraded"

    def test_healthy_default(self):
        entry = {
            "model_name": "gpt-4o",
            "litellm_params": {"model": "openai/gpt-4o"},
        }
        info = _model_entry_to_info(entry, {})
        assert info["status"] == "active"
