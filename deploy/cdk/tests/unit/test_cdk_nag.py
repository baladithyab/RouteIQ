"""cdk-nag final-gate test (proposal P0 doc 31 section 12.5).

Synthesises ``RouteIqStack`` with the same ``AwsSolutionsChecks`` aspect that
``app.py`` wires for ``cdk synth``, then asserts that NO ``AwsSolutions-*``
errors AND NO ``CdkNagValidationFailure.*`` errors survive - at BOTH the default
flag surface AND a maximal flag surface (every optional input supplied). The
production ``apply_nag_suppressions`` call lives inside ``RouteIqStack.__init__``
(``_suppress_nag()``), so this test exercises the production wiring rather than
re-applying suppressions in test code.

If any finding survives, the failure message renders the offending ``id`` +
``data[:200]`` so reviewers know exactly which path needs attention.

The stacks synthesise offline against the dummy env (account ``123456789012`` /
``us-west-2``) via the shared ``make_stack`` helper, which adds the aspect to a
freshly-minted app per call.
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import Aspects
from aws_cdk.assertions import Annotations, Match
from cdk_nag import AwsSolutionsChecks

from tests.conftest import make_stack

# The maximal flag surface: every optional input supplied so each conditionally
# emitted resource + suppression is exercised (admin access entries, explicit
# Bedrock + secret ARNs, the config S3 statement, cost-allocation tags).
_MAXIMAL_FLAGS = {
    "admin_principal_arns": [
        "arn:aws:iam::123456789012:role/Admin",
        "arn:aws:iam::123456789012:role/CiCd",
    ],
    "bedrock_model_arns": [
        "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-v1:0",
    ],
    # RouteIQ-6150 (C1): two cross-account capacity ids so the home pod-role
    # sts:AssumeRole grant on the computed RouteIqBedrockCapacity-<env> member ARNs
    # is exercised by the default+maximal nag gate. The grant targets EXPLICIT ARNs
    # (not a wildcard), so it must draw NO unsuppressed AwsSolutions-IAM5.
    "capacity_account_ids": ["111122223333", "444455556666"],
    "config_s3_bucket": "routeiq-config-dev",
    "secret_arns": [
        "arn:aws:secretsmanager:us-west-2:123456789012:secret:routeiq/master-AbCdEf",
    ],
    "cost_center": "ml-platform",
    "team": "routeiq",
    "tenant": "acme",
}


def _synth_with_nag(**flags) -> cdk.Stack:
    """Build a RouteIqStack with the AwsSolutionsChecks aspect added to its app."""
    app = cdk.App()
    stack = make_stack(app=app, **flags)
    Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    return stack


def _render(errors) -> str:
    return "\n".join(f"- {entry.id}: {str(entry.entry.data)[:200]}" for entry in errors)


def test_no_unsuppressed_aws_solutions_errors_default() -> None:
    """No AwsSolutions-* errors survive on the default flag surface."""
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
    """No CdkNagValidationFailure errors survive on the default flag surface.

    cdk-nag emits these when a rule input cannot be statically resolved (an
    Fn::GetAtt intrinsic). They must be suppressed-with-reason or fixed; an
    unsuppressed one means hidden coverage gaps.
    """
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
    """No AwsSolutions-* errors survive at the maximal flag surface.

    Every optional input supplied: admin access entries, explicit Bedrock +
    secret ARNs (which flip the IAM5 wildcard suppressions to explicit ARNs), the
    config S3 statement, and cost-allocation tags.
    """
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
    """No CdkNagValidationFailure errors survive at the maximal flag surface."""
    stack = _synth_with_nag(**_MAXIMAL_FLAGS)
    errors = Annotations.from_stack(stack).find_error(
        "*", Match.string_like_regexp("CdkNagValidationFailure.*")
    )
    if errors:
        raise AssertionError(
            f"{len(errors)} unsuppressed CdkNagValidationFailure error(s) (maximal surface):\n"
            f"{_render(errors)}"
        )
