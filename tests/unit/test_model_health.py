"""Tests for per-model health endpoint."""

from unittest.mock import MagicMock

from litellm_llmrouter.resilience import compute_model_health_summary


def test_model_health_summary():
    """Model health summary counts healthy/degraded/unhealthy."""
    breakers = {
        "model-a": MagicMock(state="closed"),
        "model-b": MagicMock(state="half_open"),
        "model-c": MagicMock(state="open"),
    }
    summary = compute_model_health_summary(breakers)
    assert summary["healthy"] == 1
    assert summary["degraded"] == 1
    assert summary["unhealthy"] == 1
    assert summary["total"] == 3


def test_model_health_empty():
    """No breakers means all-zero summary."""
    summary = compute_model_health_summary({})
    assert summary == {"healthy": 0, "degraded": 0, "unhealthy": 0, "total": 0}


def test_model_health_all_healthy():
    """All closed breakers means all healthy."""
    breakers = {
        "m1": MagicMock(state="closed"),
        "m2": MagicMock(state="closed"),
    }
    summary = compute_model_health_summary(breakers)
    assert summary["healthy"] == 2
    assert summary["total"] == 2
    assert summary["degraded"] == 0
    assert summary["unhealthy"] == 0
