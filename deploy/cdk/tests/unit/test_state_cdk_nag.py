"""cdk-nag final-gate test for RouteIqStateStack (P1, ADR-0028/0029).

Synthesises the P0 ``RouteIqStack`` + the P1 ``RouteIqStateStack`` (wired by
reference) under the same ``AwsSolutionsChecks`` aspect ``app.py`` wires, then
asserts NO ``aws:cdk:error`` metadata entries (the cdk-nag finding channel)
survive ON THE STATE STACK. The production ``apply_state_nag_suppressions`` call
lives inside ``RouteIqStateStack.__init__`` (``_suppress_nag()``), so this test
exercises the production wiring rather than re-applying suppressions in test code.

The schema-bootstrap Lambda asset is pinned MISSING (inline-fallback path) so the
synth is hermetic regardless of Docker / whether the asset dir was built.

Why manifest-reading instead of ``Annotations.from_stack(...).find_error(...)``:
on this aws-cdk-lib / jsii / Python combo ``find_error`` crashes with
``KeyError: 'aws-cdk-lib.cloud_assembly_schema.MetadataEntry'`` when the stack
carries cross-stack-reference metadata entries (the State stack imports the P0
VPC / role, so it always does). That crash is a jsii reference-map proxy bug, NOT
a real nag finding. Reading the synthesised ``manifest.json`` for
``aws:cdk:error`` entries is the equivalent assertion via a path that does not
touch the broken jsii proxy. Still fully offline (the dummy env account
``123456789012`` / ``us-west-2``; no AWS creds, no network).
"""

from __future__ import annotations

import json

import aws_cdk as cdk
import pytest
from aws_cdk import Aspects
from cdk_nag import AwsSolutionsChecks

from lib import replay_store_construct as rs_module
from lib.routeiq_stack import RouteIqStack
from lib.routeiq_state_stack import RouteIqStateStack
from tests.conftest import dummy_env


@pytest.fixture(autouse=True)
def _force_inline_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the bootstrap-Lambda asset MISSING so the nag synth is hermetic."""
    monkeypatch.setattr(
        rs_module,
        "_SCHEMA_BOOTSTRAP_ASSET_PATH",
        "/tmp/routeiq-state-nag-test-does-not-exist",
        raising=True,
    )


def _state_nag_errors(*, env_name: str = "dev", min_acu=None) -> list[tuple[str, str]]:
    """Synth P0 + state stack with the nag aspect; return state-stack nag errors.

    Returns a list of (construct_path, message) for every ``aws:cdk:error`` the
    AwsSolutionsChecks aspect emitted on the RouteIqStateStack artifact. An empty
    list means every finding was either fixed-in-code or suppressed-with-reason.
    """
    app = cdk.App()
    foundation = RouteIqStack(app, f"RouteIqStack-{env_name}", env=dummy_env(), env_name=env_name)
    RouteIqStateStack(
        app,
        f"RouteIqStateStack-{env_name}",
        env=dummy_env(),
        env_name=env_name,
        foundation=foundation,
        min_acu=min_acu,
    )
    Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    asm = app.synth()
    with open(f"{asm.directory}/manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    artifact = manifest["artifacts"][f"RouteIqStateStack-{env_name}"]
    errors: list[tuple[str, str]] = []
    for path, entries in artifact.get("metadata", {}).items():
        for entry in entries:
            if entry.get("type") == "aws:cdk:error":
                errors.append((path, str(entry.get("data"))))
    return errors


def _render(errors: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {path}: {data[:200]}" for path, data in errors)


def test_no_unsuppressed_findings_default() -> None:
    """No cdk-nag errors survive on the state stack (default dev surface)."""
    errors = _state_nag_errors()
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed cdk-nag error(s) on RouteIqStateStack "
            f"(dev surface):\n{_render(errors)}"
        )


def test_no_unsuppressed_findings_prod() -> None:
    """No cdk-nag errors survive on the state stack at the non-dev surface.

    Exercises the deletion-protection-ON + RETAIN posture (the RDS10 suppression
    is dev-rationale; non-dev sets protection in code, so the finding does NOT
    fire there). Proves the suppression set is correct across env variants.
    """
    errors = _state_nag_errors(env_name="prod")
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed cdk-nag error(s) on RouteIqStateStack "
            f"(prod surface):\n{_render(errors)}"
        )


def test_no_unsuppressed_findings_scale_to_zero() -> None:
    """No cdk-nag errors survive with the scale-to-zero (min_acu=0) path."""
    errors = _state_nag_errors(min_acu=0)
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed cdk-nag error(s) on RouteIqStateStack "
            f"(scale-to-zero):\n{_render(errors)}"
        )
