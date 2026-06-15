"""cdk-nag final-gate test for the ObservabilityConstruct (RouteIQ P2, ADR-0027).

Mirrors ``test_cdk_nag.py`` for the P2 observability surface: synthesises a
throwaway stack hosting an ``ObservabilityConstruct`` with the same
``AwsSolutionsChecks`` aspect ``app.py`` wires, then asserts NO ``AwsSolutions-*``
errors AND NO ``CdkNagValidationFailure.*`` errors survive - at BOTH the default
flag surface (AMG off) AND the maximal surface (``enable_amg=True`` +
``notify_emails`` supplied, so the AMG data-source role's ``*``-resource read
statements are exercised).

The construct applies its own evidenced suppressions via
``apply_observability_nag_suppressions`` (called from the test harness here in the
same shape the composition root will call it), so this test exercises the
production suppression wiring rather than re-applying suppressions in test code.

The stacks synthesise offline against the dummy env (account ``123456789012`` /
``us-west-2``), importing the P0 routing log group by NAME (cred-free).
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import Aspects
from aws_cdk import aws_logs as logs
from aws_cdk.assertions import Annotations, Match
from cdk_nag import AwsSolutionsChecks

_ROUTING_LOG_GROUP_NAME = "/aws/containerinsights/routeiq-dev/routeiq-routing"

# The maximal flag surface: AMG on (the data-source role + its * read scopes) +
# an operator email subscription.
_MAXIMAL_FLAGS = {
    "enable_amg": True,
    "notify_emails": ["oncall@example.com"],
}


def _synth_with_nag(**flags) -> cdk.Stack:
    """Build a stack hosting an ObservabilityConstruct with the nag aspect added."""
    from lib.obs_nag_suppressions import apply_observability_nag_suppressions
    from lib.observability_construct import ObservabilityConstruct

    app = cdk.App()
    env_name = flags.pop("env_name", "dev")
    stack = cdk.Stack(
        app,
        f"RouteIqObsStack-{env_name}",
        env=cdk.Environment(account="123456789012", region="us-west-2"),
    )
    log_group = logs.LogGroup.from_log_group_name(
        stack, "ImportedRoutingLogGroup", _ROUTING_LOG_GROUP_NAME
    )
    stack.observability = ObservabilityConstruct(  # type: ignore[attr-defined]
        stack,
        "ObservabilityConstruct",
        env_name=env_name,
        routing_log_group=log_group,
        **flags,
    )
    apply_observability_nag_suppressions(stack)
    Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    return stack


def _render(errors) -> str:
    return "\n".join(f"- {entry.id}: {str(entry.entry.data)[:200]}" for entry in errors)


def test_no_unsuppressed_aws_solutions_errors_default() -> None:
    """No AwsSolutions-* errors survive on the default (AMG-off) surface."""
    stack = _synth_with_nag()
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) (default surface):\n"
            f"{_render(errors)}"
        )


def test_no_validation_failures_default() -> None:
    """No CdkNagValidationFailure errors survive on the default surface."""
    stack = _synth_with_nag()
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("CdkNagValidationFailure.*")
    )
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed CdkNagValidationFailure error(s) (default surface):\n"
            f"{_render(errors)}"
        )


def test_no_unsuppressed_aws_solutions_errors_maximal() -> None:
    """No AwsSolutions-* errors survive at the maximal (AMG-on) surface."""
    stack = _synth_with_nag(**_MAXIMAL_FLAGS)
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("AwsSolutions-.*")
    )
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) (maximal surface):\n"
            f"{_render(errors)}"
        )


def test_no_validation_failures_maximal() -> None:
    """No CdkNagValidationFailure errors survive at the maximal surface."""
    stack = _synth_with_nag(**_MAXIMAL_FLAGS)
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("CdkNagValidationFailure.*")
    )
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed CdkNagValidationFailure error(s) (maximal surface):\n"
            f"{_render(errors)}"
        )
