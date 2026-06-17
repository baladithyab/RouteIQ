"""Tests for leader-aware drain / leadership hand-off (RouteIQ-8387)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.resilience import (
    leader_aware_drain,
    release_leadership_for_handoff,
    reset_drain_manager,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_drain_manager()
    yield
    reset_drain_manager()


async def test_handoff_noop_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_LEADER_DRAIN_HANDOFF", raising=False)
    # No leader election lookup happens; returns True (nothing to do).
    with patch("litellm_llmrouter.leader_election.get_leader_election") as get_le:
        assert await release_leadership_for_handoff() is True
        get_le.assert_not_called()


async def test_handoff_noop_when_not_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_LEADER_DRAIN_HANDOFF", "true")
    election = MagicMock()
    election.is_leader = False
    with patch(
        "litellm_llmrouter.leader_election.get_leader_election", return_value=election
    ):
        assert await release_leadership_for_handoff() is True
    election.stop_renewal.assert_not_called()


async def test_handoff_releases_when_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_LEADER_DRAIN_HANDOFF", "true")
    election = MagicMock()
    election.is_leader = True
    election.release = AsyncMock(return_value=True)
    with patch(
        "litellm_llmrouter.leader_election.get_leader_election", return_value=election
    ):
        assert await release_leadership_for_handoff() is True
    election.stop_renewal.assert_called_once()
    election.release.assert_awaited_once()


async def test_handoff_failsafe_on_release_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTEIQ_LEADER_DRAIN_HANDOFF", "true")
    election = MagicMock()
    election.is_leader = True
    election.release = AsyncMock(side_effect=RuntimeError("db down"))
    with patch(
        "litellm_llmrouter.leader_election.get_leader_election", return_value=election
    ):
        # Never raises; returns False on a real release failure.
        assert await release_leadership_for_handoff() is False


async def test_leader_aware_drain_runs_full_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTEIQ_LEADER_DRAIN_HANDOFF", "true")
    election = MagicMock()
    election.is_leader = True
    election.release = AsyncMock(return_value=True)
    with patch(
        "litellm_llmrouter.leader_election.get_leader_election", return_value=election
    ):
        # No active requests -> drain completes immediately (True).
        assert await leader_aware_drain() is True
    election.release.assert_awaited_once()
    from litellm_llmrouter.resilience import get_drain_manager

    assert get_drain_manager().is_draining is True
