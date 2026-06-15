"""Cred-free assertion: ``app.py`` threads ``foundation=`` in the combined deploy.

The grant/coupling fixes (RouteIQ-569f / 74c0 / 717b / 81c4) only fire when the
combined-deploy ``app.py`` passes the P0 ``RouteIqStack`` into the P2
``RouteIqObservabilityStack`` by reference (``foundation=foundation``). The
``make_obs_stack`` test helper threads it directly, so the rest of the suite would
stay GREEN even if ``app.py`` itself dropped the arg -- and every grant would
silently vanish (the exact "missing arg -> swallowed -> silent fallback" class the
project guards against for the routing-strategy seam). This test drives the REAL
``app.main()`` wiring and asserts the seam, so a regression in ``app.py`` is caught.

Fully offline. ``CDK_DEFAULT_ACCOUNT`` / ``CDK_DEFAULT_REGION`` are pinned to the
dummy env so the synth needs no AWS creds. Context is injected by patching the
``cdk.App`` factory the app module uses to seed ``context=`` directly (the JSII
kernel reads ``CDK_CONTEXT_JSON`` ONCE at import time, BEFORE a test can set it, so
the env-var route does not work post-import -- the constructor ``context=`` arg
does). The state stack is flagged OFF to keep the synth minimal.
"""

from __future__ import annotations

import json

import pytest

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"

# The combined-deploy context: P2 observability ON (threaded to P0), state stack
# OFF (minimal synth), nested P2 flags OFF (byte-stable default surface).
_COMBINED_CONTEXT: dict[str, object] = {
    "routeiq:env": "dev",
    "routeiq:enable_observability_stack": True,
    "routeiq:enable_state_stack": False,
    "routeiq:enable_amg": False,
    "routeiq:enable_data_lake": False,
}


@pytest.fixture
def _combined_deploy_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Pin the dummy env + seed the combined-deploy context + outdir into ``cdk.App``.

    Patches ``app.cdk.App`` so every ``cdk.App()`` the app module constructs is
    pre-seeded with ``context=_COMBINED_CONTEXT`` AND ``outdir=<tmp>`` (the env-var
    ``CDK_CONTEXT_JSON`` route is read by the JSII kernel at import time and cannot
    be set late; ``CDK_OUTDIR`` is likewise not honored by the App). Yields the
    cloud-assembly output dir so the synth tests can read the emitted templates
    deterministically with no churn in the tree.
    """
    import app as app_module

    monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", DUMMY_ACCOUNT)
    monkeypatch.setenv("CDK_DEFAULT_REGION", DUMMY_REGION)

    outdir = tmp_path / "cdk.out"
    real_app_cls = app_module.cdk.App

    def _seeded_app(*args: object, **kwargs: object):
        merged_ctx = {**_COMBINED_CONTEXT, **(kwargs.pop("context", None) or {})}
        kwargs.setdefault("outdir", str(outdir))
        return real_app_cls(*args, context=merged_ctx, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(app_module.cdk, "App", _seeded_app)
    yield outdir


def test_app_main_threads_foundation_into_observability_stack(
    _combined_deploy_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``app.main()`` passes the P0 foundation into the P2 stack by reference.

    Records the ``foundation`` kwarg the combined-deploy path hands to
    ``RouteIqObservabilityStack``; it must be the SAME ``RouteIqStack`` instance
    ``app.main()`` built (the load-bearing seam). If ``app.py`` dropped the arg,
    ``foundation`` would be ``None`` and the grants would never apply.
    """
    import app as app_module
    from lib.routeiq_observability_stack import RouteIqObservabilityStack
    from lib.routeiq_stack import RouteIqStack

    captured: dict[str, object] = {}

    class _RecordingObsStack(RouteIqObservabilityStack):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["foundation"] = kwargs.get("foundation")
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(app_module, "RouteIqObservabilityStack", _RecordingObsStack)

    app_module.main()

    foundation = captured.get("foundation")
    assert foundation is not None, (
        "app.py must thread foundation= into RouteIqObservabilityStack in the "
        "combined deploy; a dropped arg silently skips ALL pod-role grants"
    )
    assert isinstance(foundation, RouteIqStack), type(foundation)


def test_app_main_combined_synth_emits_the_pod_role_grants(
    _combined_deploy_env,
) -> None:
    """End-to-end: the synthesized P2 template carries the ARN-scoped grants.

    Drives the real ``app.main()`` synth (offline) and inspects the
    ``RouteIqObservabilityStack-dev`` artifact for the AppConfig-poll +
    aps:RemoteWrite statements -- proving the combined-deploy path actually applies
    them (not just that the helper can).
    """
    import app as app_module

    assembly_dir = _combined_deploy_env
    app_module.main()

    obs_template = _load_template(assembly_dir, "RouteIqObservabilityStack-dev")
    statements = _all_policy_statements(obs_template)
    sids = {s.get("Sid") for s in statements}
    assert "AppConfigPoll" in sids, f"AppConfig poll grant missing; sids={sids}"
    assert "AmpRemoteWrite" in sids, f"aps:RemoteWrite grant missing; sids={sids}"

    # Both grants are ARN-scoped (never a wildcard).
    for sid in ("AppConfigPoll", "AmpRemoteWrite"):
        st = next(s for s in statements if s.get("Sid") == sid)
        assert st["Resource"] != "*", f"{sid} must be ARN-scoped, not '*'"


def test_app_main_obs_artifact_depends_on_p0(
    _combined_deploy_env,
) -> None:
    """The synthesized P2 artifact declares a CFN dependency on the P0 stack (81c4).

    ``add_dependency(foundation)`` so CFN deploys P0 (the routing log group) before
    P2 (the metric filters), asserted from the synthesized manifest.
    """
    import app as app_module

    assembly_dir = _combined_deploy_env
    app_module.main()

    manifest = json.loads((assembly_dir / "manifest.json").read_text(encoding="utf-8"))
    obs = manifest["artifacts"]["RouteIqObservabilityStack-dev"]
    deps = set(obs.get("dependencies", []))
    assert "RouteIqStack-dev" in deps, (
        f"P2 must depend on P0 (CFN deploys P0 first); P2 deps = {deps}"
    )


# --------------------------------------------------------------------- helpers


def _load_template(assembly_dir, stack_name: str) -> dict:
    """Load a synthesized stack template by name from the cloud assembly."""
    path = assembly_dir / f"{stack_name}.template.json"
    assert path.exists(), f"missing synthesized template {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _all_policy_statements(template: dict) -> list[dict]:
    """Every PolicyStatement across all AWS::IAM::Policy resources in a template."""
    statements: list[dict] = []
    for res in template.get("Resources", {}).values():
        if res.get("Type") != "AWS::IAM::Policy":
            continue
        doc = res.get("Properties", {}).get("PolicyDocument", {})
        statements.extend(doc.get("Statement", []))
    return statements
