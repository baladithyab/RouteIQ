"""State-stack template snapshot test (P1, ADR-0028/0029).

Extends the P0 snapshot gate (``test_template_snapshot.py``) to the SEPARATE
``RouteIqStateStack`` (Aurora + ElastiCache). Synthesises the flag-default ``dev``
state stack -- wired cross-stack to a P0 ``RouteIqStack`` by reference, exactly as
``app.py`` does -- and either writes the baseline (ONLY under an explicit
``UPDATE_SNAPSHOTS=1``) or diffs the canonicalised template against the committed
``__snapshots__/state-dev.json``.

Same discipline as the P0 snapshot test:

  - Canonical form: ``Template.from_stack(state).to_json()`` ->
    ``json.dumps(sort_keys=True, indent=2)`` (sorted keys => diff-stable).
  - Auto-write ONLY under ``UPDATE_SNAPSHOTS=1``. A MISSING baseline WITHOUT the
    flag is a LOUD failure, NOT a silent create (silent creation is a false-green
    hazard). The Gate stage generates the baseline once green.
  - Hermetic: the schema-bootstrap Lambda asset path global is pinned to a
    non-existent dir so the synth drives the deterministic ``Code.from_inline``
    fallback regardless of whether Docker / the asset dir is present -- without
    this the state stack synthesises differently on a Docker-equipped host than on
    a Docker-less one (the ``from_asset`` bundling path), breaking the snapshot.

Only the STATE-stack artifact is snapshotted here; the P0 foundation keeps its own
``dev.json`` baseline in ``test_template_snapshot.py``.

Regenerate intentionally with:
``UPDATE_SNAPSHOTS=1 uv run pytest deploy/cdk/tests/snapshot/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from aws_cdk.assertions import Template

from lib import replay_store_construct as rs_module
from tests.conftest import make_state_stack

_SNAPSHOT_DIR = Path(__file__).parent / "__snapshots__"
_SNAPSHOT_PATH = _SNAPSHOT_DIR / "state-dev.json"

# Module globals naming a Docker-bundled-asset directory, as (module, attribute).
# The state stack's only such global is the schema-bootstrap Lambda asset path:
# present -> from_asset bundling (Docker/local-pip dependent, NON-hermetic);
# absent -> Code.from_inline placeholder (deterministic). Pinning it missing keeps
# the snapshot byte-stable across Docker-equipped and Docker-less hosts.
_DOCKER_ASSET_PATH_GLOBALS: list[tuple[object, str]] = [
    (rs_module, "_SCHEMA_BOOTSTRAP_ASSET_PATH"),
]


@pytest.fixture(autouse=True)
def _force_hermetic_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the Docker-bundled-asset path globals to a non-existent dir.

    Drives the deterministic inline-fallback path so the same source yields the
    same template on a Docker-equipped host as on a Docker-less one. Without this
    the state-stack snapshot would be non-hermetic (the schema-bootstrap Lambda's
    from_asset bundling path differs by host).
    """
    missing = str(Path(__file__).parent / "_state_does_not_exist")
    for module, attr in _DOCKER_ASSET_PATH_GLOBALS:
        monkeypatch.setattr(module, attr, missing)


def _canonicalise(template: dict) -> str:
    return json.dumps(template, sort_keys=True, indent=2)


def _synth_state_template() -> dict:
    """Synthesise the flag-default dev state stack and return its template dict."""
    _app, _foundation, state = make_state_stack(env_name="dev")
    return Template.from_stack(state).to_json()


def _assert_snapshot(canonical: str, snapshot_path: Path) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-write ONLY under an explicit UPDATE_SNAPSHOTS=1. An absent baseline
    # WITHOUT the flag is a loud failure, never a silent create (false-green).
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        snapshot_path.write_text(canonical + "\n", encoding="utf-8")
        return

    if not snapshot_path.exists():
        raise AssertionError(
            f"State-stack snapshot baseline {snapshot_path.name} is MISSING. This "
            "is NOT auto-created during a normal test run (that would mask real "
            "divergence). The Gate stage generates it once green; to (re)generate "
            "intentionally, run UPDATE_SNAPSHOTS=1 uv run pytest "
            "deploy/cdk/tests/snapshot/ and commit the result."
        )

    expected = snapshot_path.read_text(encoding="utf-8").rstrip("\n")
    assert canonical == expected, (
        f"Synthesised state-stack template diverged from committed snapshot "
        f"{snapshot_path.name}. If the change is intentional, re-run with "
        f"UPDATE_SNAPSHOTS=1 (or delete {snapshot_path}) and commit the refreshed "
        "baseline."
    )


def test_state_dev_template_snapshot() -> None:
    """The flag-default dev state-stack template matches __snapshots__/state-dev.json."""
    canonical = _canonicalise(_synth_state_template())
    _assert_snapshot(canonical, _SNAPSHOT_PATH)
