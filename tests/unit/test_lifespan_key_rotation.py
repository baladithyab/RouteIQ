"""End-to-end lifespan wiring test for the key auto-rotation startup block
(RouteIQ-fa1b, proving the RouteIQ-b8a2 wiring).

The existing test (``test_tc_wire_callsites.py::test_rotation_invoked_under_gate``)
RE-IMPLEMENTS the gate condition::

    if key_rotation_enabled() and _discovery_is_leader():
        rotate_stale_keys(get_governance_engine())

instead of driving the real startup. That can pass even if the *actual* lifespan
block (``gateway/app.py`` step 3e, inside ``_routeiq_lifespan``) is deleted or
its gate is broken.

These tests drive the REAL ``_routeiq_lifespan`` startup -- entered via
``app.router.lifespan_context(app)`` on a ``create_gateway_app(mount_litellm=False)``
app (the same entrypoint ``test_shutdown_invokes_leader_handoff`` uses). LiteLLM
is NOT booted (no ``LITELLM_CONFIG_PATH``; its init is try/except-wrapped). The
durable store + leadership are mocked so the only thing under test is whether the
production gate actually invokes ``scim.rotate_stale_keys``.

Asserted contract:
* rotate_stale_keys IS called when flag ON + leader;
* NOT called when the flag is off;
* NOT called when not the leader (flag on).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.gateway.app import create_gateway_app
from litellm_llmrouter.gateway.plugin_manager import reset_plugin_manager


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_plugin_manager()
    yield
    reset_plugin_manager()


@contextmanager
def _patched_rotation_block(*, is_leader: bool):
    """Patch the collaborators the step-3e block imports at call time, where they
    are DEFINED (the block does ``from ..scim import ...`` etc. inside the
    coroutine, so source-module patching is what intercepts them).

    Yields the ``rotate_stale_keys`` mock so the test can assert on its calls.
    The shutdown leader hand-off is also stubbed so lifespan teardown is inert.
    """
    fake_store = MagicMock()
    fake_store.enabled = False  # rotated-row persistence branch is skipped

    rotate = MagicMock(return_value=[])
    with (
        patch("litellm_llmrouter.scim.rotate_stale_keys", rotate),
        patch(
            "litellm_llmrouter.startup._discovery_is_leader",
            return_value=is_leader,
        ),
        patch(
            "litellm_llmrouter.governance_store.get_governance_store",
            return_value=fake_store,
        ),
        patch(
            "litellm_llmrouter.resilience.release_leadership_for_handoff",
            new=AsyncMock(return_value=False),
        ),
    ):
        yield rotate


async def _run_lifespan() -> None:
    """Enter + exit the REAL gateway lifespan once (no LiteLLM boot)."""
    app = create_gateway_app(mount_litellm=False, enable_plugins=False)
    async with app.router.lifespan_context(app):
        pass


async def test_real_lifespan_rotates_when_gated_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag ON + leader => the REAL step-3e block calls rotate_stale_keys."""
    monkeypatch.setenv("ROUTEIQ_KEY_ROTATION_ENABLED", "true")
    with _patched_rotation_block(is_leader=True) as rotate:
        await _run_lifespan()
    rotate.assert_called_once()


async def test_real_lifespan_does_not_rotate_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag OFF (default) => the REAL lifespan short-circuits before rotation."""
    monkeypatch.delenv("ROUTEIQ_KEY_ROTATION_ENABLED", raising=False)
    with _patched_rotation_block(is_leader=True) as rotate:
        await _run_lifespan()
    rotate.assert_not_called()


async def test_real_lifespan_does_not_rotate_when_not_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag ON but NOT leader => the leader gate suppresses rotation."""
    monkeypatch.setenv("ROUTEIQ_KEY_ROTATION_ENABLED", "true")
    with _patched_rotation_block(is_leader=False) as rotate:
        await _run_lifespan()
    rotate.assert_not_called()
