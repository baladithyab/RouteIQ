"""Stack template snapshot test (proposal P0 doc 31 section 12.6).

Synthesises the flag-off ``dev`` ``RouteIqStack`` and either writes the baseline
(only under an explicit ``UPDATE_SNAPSHOTS=1``) or diffs the canonicalised
template against the committed ``__snapshots__/dev.json``.

  - Canonical form: ``Template.from_stack(stack).to_json()`` ->
    ``json.dumps(sort_keys=True, indent=2)``. Sorting keys makes the baseline
    diff-stable regardless of CDK's emission order.
  - Auto-write ONLY under ``UPDATE_SNAPSHOTS=1``. A MISSING baseline WITHOUT the
    flag is a LOUD failure, NOT a silent create - silent creation is a
    false-green hazard (a transiently-absent baseline could be regenerated and
    then "match," turning a real divergence green). The Gate stage generates the
    baseline once it is green; this test never creates it during a normal run.
  - An autouse fixture pins any Docker-bundled-asset path globals to a
    non-existent dir so synth stays hermetic regardless of whether Docker is
    present on the host. RouteIQ's P0 stack is pure L1/L2 today (no from_asset
    bundling), so the pin is currently a defensive no-op, but it is wired so that
    if a future construct adds a Docker-bundled Lambda the snapshot stays
    hermetic without a test change.

Regenerate intentionally with:
``UPDATE_SNAPSHOTS=1 uv run pytest deploy/cdk/tests/snapshot/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from aws_cdk.assertions import Template

from tests.conftest import make_stack

_SNAPSHOT_DIR = Path(__file__).parent / "__snapshots__"
_SNAPSHOT_PATH = _SNAPSHOT_DIR / "dev.json"

# Module globals that name a Docker-bundled-asset directory. Each entry is
# (module, attribute). RouteIQ's P0 stack has none today (pure L1/L2), so this
# list is empty and the hermetic fixture is a no-op. When a future construct adds
# a Docker-bundled Lambda, append its (module, "_X_ASSET_PATH") here and the
# fixture pins it to a non-existent dir to drive the inline fallback.
_DOCKER_ASSET_PATH_GLOBALS: list[tuple[object, str]] = []


@pytest.fixture(autouse=True)
def _force_hermetic_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin any Docker-bundled-asset path globals to a non-existent dir.

    A construct that synthesises differently when Docker is present (``from_asset``
    bundling) vs absent (``Code.from_inline`` fallback) makes the snapshot
    non-hermetic: the same source yields a different template on a Docker-equipped
    host than on a Docker-less one. Pointing each such global at a missing dir
    drives the deterministic inline-fallback path. RouteIQ has no such globals at
    P0, so this is currently a defensive no-op - but it is the contract that keeps
    the snapshot hermetic if one is added later.
    """
    missing = str(Path(__file__).parent / "_does_not_exist")
    for module, attr in _DOCKER_ASSET_PATH_GLOBALS:
        monkeypatch.setattr(module, attr, missing)


def _canonicalise(template: dict) -> str:
    return json.dumps(template, sort_keys=True, indent=2)


def _synth_template() -> dict:
    """Synthesise the flag-off dev stack and return its template dict."""
    stack = make_stack(env_name="dev")
    return Template.from_stack(stack).to_json()


def _assert_snapshot(canonical: str, snapshot_path: Path) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-write ONLY under an explicit UPDATE_SNAPSHOTS=1. The previous
    # "create if missing" pattern is a false-green hazard, so an absent baseline
    # WITHOUT the flag is a loud failure, never a silent create.
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        snapshot_path.write_text(canonical + "\n", encoding="utf-8")
        return

    if not snapshot_path.exists():
        raise AssertionError(
            f"Snapshot baseline {snapshot_path.name} is MISSING. This is NOT "
            "auto-created during a normal test run (that would mask real "
            "divergence). The Gate stage generates it once green; to (re)generate "
            "intentionally, run UPDATE_SNAPSHOTS=1 uv run pytest "
            "deploy/cdk/tests/snapshot/ and commit the result."
        )

    expected = snapshot_path.read_text(encoding="utf-8").rstrip("\n")
    assert canonical == expected, (
        f"Synthesised template diverged from committed snapshot {snapshot_path.name}. "
        "If the change is intentional, re-run with UPDATE_SNAPSHOTS=1 "
        f"(or delete {snapshot_path}) and commit the refreshed baseline."
    )


def test_dev_template_snapshot() -> None:
    """The flag-off dev template matches the committed __snapshots__/dev.json."""
    canonical = _canonicalise(_synth_template())
    _assert_snapshot(canonical, _SNAPSHOT_PATH)
