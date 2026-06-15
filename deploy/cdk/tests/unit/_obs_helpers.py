"""Shared helpers for the P2 observability unit tests.

``standalone_obs_template()`` synthesises the DEFAULT (no-foundation) P2
``RouteIqObservabilityStack`` -- the standalone / separate-pipeline path that must
stay byte-identical (no grants, plain-string by-NAME log-group reference). Used by
the negative halves of the combined-grants assertions.

The validator Lambda is deterministic-inline-by-default after RouteIQ-4772, so this
needs no Docker / monkeypatch; the construct's ``bundle_validator_asset`` toggle
defaults False (inline path) regardless of host Docker state.
"""

from __future__ import annotations

from typing import Any

import aws_cdk as cdk
from aws_cdk.assertions import Template

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"


def standalone_obs_stack(**flags: Any):
    """Build a no-foundation ``RouteIqObservabilityStack`` (the standalone path)."""
    from lib.routeiq_observability_stack import RouteIqObservabilityStack

    app = cdk.App()
    env_name = flags.pop("env_name", "dev")
    return RouteIqObservabilityStack(
        app,
        f"RouteIqObservabilityStack-{env_name}",
        env=cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION),
        env_name=env_name,
        **flags,
    )


def standalone_obs_template(**flags: Any) -> Template:
    """The default (no-foundation) P2 template."""
    return Template.from_stack(standalone_obs_stack(**flags))
