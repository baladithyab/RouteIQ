"""Observability-stack template snapshot test (P2, ADR-0026/0027) -- RouteIQ-74a4.

Extends the P0 (``test_template_snapshot.py``) + P1
(``test_state_template_snapshot.py``) snapshot gates to the SEPARATE
``RouteIqObservabilityStack`` (AppConfig config-state + AMP/CW observability). Synth
the flag-default ``dev`` P2 stack -- wired cross-stack to a P0 ``RouteIqStack`` by
reference, exactly as ``app.py`` does in the combined deploy (so the snapshot
exercises the 569f/74c0/717b grants + the 81c4 cross-stack log-group import) -- and
either writes the baseline (ONLY under an explicit ``UPDATE_SNAPSHOTS=1``) or diffs
the canonicalised template against the committed ``__snapshots__/observability-dev.json``.

Same discipline as the P0/P1 snapshot tests:

  - Canonical form: ``Template.from_stack(obs).to_json()`` ->
    ``json.dumps(sort_keys=True, indent=2)`` (sorted keys => diff-stable).
  - Auto-write ONLY under ``UPDATE_SNAPSHOTS=1``. A MISSING baseline WITHOUT the flag
    is a LOUD failure, NOT a silent create (false-green hazard).
  - Hermetic: the AppConfig validator Lambda asset-path global is pinned to a
    non-existent dir as a DEFENSIVE no-op. After RouteIQ-4772 the validator is
    inline-by-default at synth (the ``bundle_validator_asset`` toggle defaults False)
    regardless of host Docker, so the template is ALREADY host-independent -- the pin
    is belt-and-braces so a future re-introduction of an implicit Docker probe cannot
    silently make the snapshot non-hermetic.

Only the P2 observability artifact is snapshotted here; the P0 foundation keeps its
own ``dev.json`` baseline.

Regenerate intentionally with:
``UPDATE_SNAPSHOTS=1 uv run pytest deploy/cdk/tests/snapshot/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from aws_cdk.assertions import Template

from lib import config_state_construct as cfg_module
from tests.conftest import make_obs_stack

_SNAPSHOT_DIR = Path(__file__).parent / "__snapshots__"
_SNAPSHOT_PATH = _SNAPSHOT_DIR / "observability-dev.json"

# Module globals naming a Docker-bundled-asset directory, as (module, attribute).
# The P2 stack's only such global is the AppConfig validator Lambda asset path. After
# RouteIQ-4772 the default synth is inline regardless of this path (the explicit
# bundle_validator_asset toggle, not a Docker probe, drives the choice), so pinning
# it missing is a DEFENSIVE no-op that future-proofs the snapshot's hermeticity.
_DOCKER_ASSET_PATH_GLOBALS: list[tuple[object, str]] = [
    (cfg_module, "_VALIDATOR_ASSET_PATH"),
]


@pytest.fixture(autouse=True)
def _force_hermetic_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the Docker-bundled-asset path globals to a non-existent dir (defensive)."""
    missing = str(Path(__file__).parent / "_obs_does_not_exist")
    for module, attr in _DOCKER_ASSET_PATH_GLOBALS:
        monkeypatch.setattr(module, attr, missing)


def _canonicalise(template: dict) -> str:
    return json.dumps(template, sort_keys=True, indent=2)


def _synth_obs_template() -> dict:
    """Synthesise the flag-default dev P2 stack (wired to P0) and return its template."""
    _app, _foundation, obs = make_obs_stack(env_name="dev")
    return Template.from_stack(obs).to_json()


def _assert_snapshot(canonical: str, snapshot_path: Path) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-write ONLY under an explicit UPDATE_SNAPSHOTS=1. An absent baseline WITHOUT
    # the flag is a loud failure, never a silent create (false-green).
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        snapshot_path.write_text(canonical + "\n", encoding="utf-8")
        return

    if not snapshot_path.exists():
        raise AssertionError(
            f"Observability-stack snapshot baseline {snapshot_path.name} is MISSING. "
            "This is NOT auto-created during a normal test run (that would mask real "
            "divergence). The Gate stage generates it once green; to (re)generate "
            "intentionally, run UPDATE_SNAPSHOTS=1 uv run pytest "
            "deploy/cdk/tests/snapshot/ and commit the result."
        )

    expected = snapshot_path.read_text(encoding="utf-8").rstrip("\n")
    assert canonical == expected, (
        f"Synthesised observability-stack template diverged from committed snapshot "
        f"{snapshot_path.name}. If the change is intentional, re-run with "
        f"UPDATE_SNAPSHOTS=1 (or delete {snapshot_path}) and commit the refreshed "
        "baseline."
    )


def test_observability_dev_template_snapshot() -> None:
    """The flag-default dev P2 template matches __snapshots__/observability-dev.json."""
    canonical = _canonicalise(_synth_obs_template())
    _assert_snapshot(canonical, _SNAPSHOT_PATH)
