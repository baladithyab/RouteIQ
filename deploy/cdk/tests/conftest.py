"""Shared, credential-free test harness for the RouteIQ CDK suite.

Every test synthesizes ``RouteIqStack`` with an explicit dummy
``env=cdk.Environment(account="123456789012", region="us-west-2")`` — a
concrete-but-fake account/region — so ``Template.from_stack(stack)`` runs fully
offline: no AWS creds, no ``cdk`` CLI, no network (proposal §12).

``make_stack(**flags)`` builds the stack with the P0 default flag surface and
applies any per-test overrides; ``template_for(**flags)`` returns the
``Template`` directly. The actual test files (``tests/unit/*``,
``tests/snapshot/*``) land in the Chart+Tests stage and import these helpers.

The ``RouteIqStack`` import is deferred to call time so this module imports
cleanly while ``lib.routeiq_stack`` is still being authored in the construct
stage — a bare ``import tests.conftest`` will not fail before the stack exists.
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

# The dummy account/region the cred-free gate pins. Concrete but fake: a synth
# against these needs no real account and makes no AWS API call.
DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"

# The P0 default flag surface (mirrors cdk.json's routeiq:* defaults). Per-test
# overrides are merged on top in make_stack(**flags).
DEFAULT_STACK_FLAGS: dict[str, Any] = {
    "env_name": "dev",
    "vpc_cidr": "10.40.0.0/16",
    "nat_gateways": 1,
    "k8s_version": "1.33",
    "enable_ghcr_ptc": True,
    "sa_namespace": "routeiq",
    "sa_name": "routeiq-gateway",
    "image_tag": "1.0.0-rc1",
    "admin_principal_arns": [],
    "bedrock_model_arns": [],
    "config_s3_bucket": None,
    "secret_arns": [],
}


def dummy_env() -> cdk.Environment:
    """The fake-but-concrete CDK environment the offline synth uses."""
    return cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION)


def make_stack(*, app: cdk.App | None = None, construct_id: str | None = None, **flags: Any):
    """Build a ``RouteIqStack`` with the dummy env and the P0 default flags.

    Any keyword in ``flags`` overrides the matching ``DEFAULT_STACK_FLAGS``
    entry, so a test can flip a single flag (e.g. ``enable_ghcr_ptc=False``)
    without restating the whole surface. Pass an ``app`` to share a single
    ``cdk.App`` across stacks, or let the helper mint one.
    """
    # Imported here (not at module top) so this harness imports cleanly while
    # lib.routeiq_stack is still being authored in the construct stage.
    from lib.routeiq_stack import RouteIqStack

    if app is None:
        app = cdk.App()
    merged = {**DEFAULT_STACK_FLAGS, **flags}
    env_name = merged["env_name"]
    sid = construct_id or f"RouteIqStack-{env_name}"
    return RouteIqStack(app, sid, env=dummy_env(), **merged)


def template_for(**flags: Any) -> Template:
    """Synthesize a ``RouteIqStack`` (offline) and return its ``Template``."""
    return Template.from_stack(make_stack(**flags))


def make_state_stack(
    *,
    env_name: str = "dev",
    min_acu: float | None = None,
    max_acu: float | None = None,
    cache_engine_version: str = "8.0",
    construct_id: str | None = None,
    foundation_id: str | None = None,
):
    """Build a P0 ``RouteIqStack`` + the P1 ``RouteIqStateStack`` wired to it.

    Both stacks share ONE ``cdk.App`` with the dummy env, so the cross-stack
    references the state stack reads off the P0 foundation resolve at synth time
    into ``Export`` / ``Fn::ImportValue`` -- cred-free, no ``from_lookup``, no
    AWS API call. Returns ``(app, foundation, state_stack)``.

    Imported here (not at module top) so this harness imports cleanly while the
    stacks are still being authored in the construct stage.
    """
    from lib.routeiq_stack import RouteIqStack
    from lib.routeiq_state_stack import RouteIqStateStack

    app = cdk.App()
    foundation = RouteIqStack(
        app,
        foundation_id or f"RouteIqStack-{env_name}",
        env=dummy_env(),
        env_name=env_name,
    )
    state = RouteIqStateStack(
        app,
        construct_id or f"RouteIqStateStack-{env_name}",
        env=dummy_env(),
        env_name=env_name,
        foundation=foundation,
        min_acu=min_acu,
        max_acu=max_acu,
        cache_engine_version=cache_engine_version,
    )
    return app, foundation, state


def state_template_for(**flags: Any) -> Template:
    """Synthesize the ``RouteIqStateStack`` (offline) and return its ``Template``."""
    _app, _foundation, state = make_state_stack(**flags)
    return Template.from_stack(state)


def make_obs_stack(
    *,
    env_name: str = "dev",
    enable_amg: bool = False,
    enable_data_lake: bool = False,
    notify_emails: list[str] | None = None,
    construct_id: str | None = None,
    foundation_id: str | None = None,
):
    """Build a P0 ``RouteIqStack`` + the P2 ``RouteIqObservabilityStack`` wired to it.

    The combined-deploy shape ``app.py`` uses: both stacks share ONE ``cdk.App``
    with the dummy env, and the P2 stack takes ``foundation=`` (the P0 stack, by
    reference). CDK resolves the cross-stack references (the pod-role ARN, the
    routing log-group name) at synth into ``Export`` / ``Fn::ImportValue`` --
    cred-free, no ``from_lookup``, no AWS API call. Returns
    ``(app, foundation, obs_stack)``.

    Mirrors ``make_state_stack``. Imported here (not at module top) so this harness
    imports cleanly while the stacks are still being authored.
    """
    from lib.routeiq_observability_stack import RouteIqObservabilityStack
    from lib.routeiq_stack import RouteIqStack

    app = cdk.App()
    foundation = RouteIqStack(
        app,
        foundation_id or f"RouteIqStack-{env_name}",
        env=dummy_env(),
        env_name=env_name,
    )
    obs = RouteIqObservabilityStack(
        app,
        construct_id or f"RouteIqObservabilityStack-{env_name}",
        env=dummy_env(),
        env_name=env_name,
        foundation=foundation,
        enable_amg=enable_amg,
        enable_data_lake=enable_data_lake,
        notify_emails=notify_emails,
    )
    return app, foundation, obs


@pytest.fixture
def stack_template() -> Template:
    """Default flag-surface ``Template`` for assertion-style unit tests."""
    return template_for()
