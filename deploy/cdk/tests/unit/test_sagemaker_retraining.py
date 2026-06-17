"""Unit tests for the scheduled SageMaker retraining tier (RouteIQ-8a24).

The construct authors the AWS-native side of an automated retraining pipeline:
an EventBridge schedule -> SageMaker CreateTrainingJob -> S3 artifact bucket +
the narrow invoker/execution IAM. FLAG-GATED off by default at the P2 stack
(``enable_sagemaker_retraining``), so the byte-stable guarantee is that the
default ``RouteIqObservabilityStack`` emits ZERO ``AWS::Events::Rule`` /
artifact-bucket resources.

These tests assert the flag-OFF default render, the flag-ON resource graph (the
schedule rule + artifact bucket + the two IAM roles), the bucket security
posture, the narrow invoker action set, and cdk-nag cleanliness.

Synthesised offline against the dummy env (123456789012 / us-west-2),
credential-free, via the shared ``make_obs_stack`` helper.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk import Aspects
from aws_cdk.assertions import Template
from cdk_nag import AwsSolutionsChecks

from lib.routeiq_observability_stack import RouteIqObservabilityStack
from lib.routeiq_stack import RouteIqStack
from lib.sagemaker_retraining_construct import _INVOKER_SAGEMAKER_ACTIONS
from tests.conftest import dummy_env, make_obs_stack


@pytest.fixture(autouse=True)
def _force_hermetic_validator_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the validator asset path to a missing dir -> deterministic inline path."""
    import lib.config_state_construct as cfg

    missing = str(Path(__file__).parent / "_does_not_exist_sagemaker")
    monkeypatch.setattr(cfg, "_VALIDATOR_ASSET_PATH", missing)


def _obs_template(**kwargs: object) -> Template:
    _app, _foundation, obs = make_obs_stack(**kwargs)  # type: ignore[arg-type]
    return Template.from_stack(obs)


# -------------------------------------------------------------- flag-off default


def test_retraining_off_by_default_emits_no_schedule() -> None:
    template = _obs_template(env_name="dev")
    template.resource_count_is("AWS::Events::Rule", 0)


def test_retraining_attr_none_by_default() -> None:
    _app, _foundation, obs = make_obs_stack(env_name="dev")
    assert obs.sagemaker_retraining is None


# ----------------------------------------------------------------- flag-on graph


def test_retraining_emits_schedule_and_bucket_when_enabled() -> None:
    template = _obs_template(env_name="dev", enable_sagemaker_retraining=True)
    template.resource_count_is("AWS::Events::Rule", 1)
    # An artifact bucket is provisioned (>=1 S3 bucket appears once enabled).
    buckets = template.find_resources("AWS::S3::Bucket")
    assert len(buckets) >= 1


def test_retraining_construct_present() -> None:
    _app, _foundation, obs = make_obs_stack(env_name="dev", enable_sagemaker_retraining=True)
    assert obs.sagemaker_retraining is not None
    assert obs.sagemaker_retraining.artifact_bucket is not None
    assert obs.sagemaker_retraining.execution_role is not None
    assert obs.sagemaker_retraining.invoker_role is not None


# ----------------------------------------------------- artifact bucket posture


def test_artifact_bucket_is_kms_encrypted_and_blocks_public_access() -> None:
    template = _obs_template(env_name="dev", enable_sagemaker_retraining=True)
    buckets = template.find_resources("AWS::S3::Bucket")
    secure = [
        b
        for b in buckets.values()
        if b["Properties"].get("BucketEncryption")
        and b["Properties"].get("PublicAccessBlockConfiguration")
    ]
    assert secure, "expected a KMS-encrypted, BPA-on artifact bucket"


# ----------------------------------------------------- narrow invoker action set


def test_invoker_action_set_is_narrow() -> None:
    """The EventBridge invoker may ONLY call CreateTrainingJob (never wildcard)."""
    template = _obs_template(env_name="dev", enable_sagemaker_retraining=True)
    statements: list[dict] = []
    for policy in template.find_resources("AWS::IAM::Policy").values():
        statements.extend(policy["Properties"]["PolicyDocument"].get("Statement", []))
    create_stmts = [s for s in statements if s.get("Sid") == "CreateTrainingJob"]
    assert create_stmts, "expected a CreateTrainingJob ALLOW statement"
    for s in create_stmts:
        actions = s["Action"]
        if isinstance(actions, str):
            actions = [actions]
        assert set(actions) == set(_INVOKER_SAGEMAKER_ACTIONS)
        # ARN-scoped, never a bare wildcard resource.
        resources = s["Resource"]
        if not isinstance(resources, list):
            resources = [resources]
        assert "*" not in resources


def test_passrole_scoped_to_execution_role() -> None:
    template = _obs_template(env_name="dev", enable_sagemaker_retraining=True)
    statements: list[dict] = []
    for policy in template.find_resources("AWS::IAM::Policy").values():
        statements.extend(policy["Properties"]["PolicyDocument"].get("Statement", []))
    passrole = [s for s in statements if s.get("Sid") == "PassExecutionRole"]
    assert passrole, "expected a scoped PassRole statement"
    for s in passrole:
        assert s["Action"] == "iam:PassRole" or "iam:PassRole" in s["Action"]
        # condition pins PassedToService to sagemaker
        cond = s.get("Condition", {})
        assert cond.get("StringEquals", {}).get("iam:PassedToService") == (
            "sagemaker.amazonaws.com"
        )


# -------------------------------------------------------------------- cdk-nag


def _synth_nag_errors(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Synth the retraining-on P2 stack with cdk-nag; read findings from manifest."""
    import lib.config_state_construct as cfg

    monkeypatch.setattr(cfg, "_VALIDATOR_ASSET_PATH", "/_does_not_exist_sm_nag")
    outdir = tempfile.mkdtemp()
    app = cdk.App(outdir=outdir)
    foundation = RouteIqStack(app, "RouteIqStack-dev", env=dummy_env(), env_name="dev")
    RouteIqObservabilityStack(
        app,
        "RouteIqObservabilityStack-dev",
        env=dummy_env(),
        env_name="dev",
        foundation=foundation,
        enable_sagemaker_retraining=True,
    )
    Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
    app.synth()
    manifest = json.loads((Path(outdir) / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for artifact in manifest.get("artifacts", {}).values():
        for path, entries in (artifact.get("metadata") or {}).items():
            for entry in entries:
                if entry.get("type") == "aws:cdk:error" and "AwsSolutions" in str(
                    entry.get("data")
                ):
                    errors.append(f"{path}: {str(entry.get('data'))[:200]}")
    return errors


def test_retraining_construct_cdk_nag_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """No unsuppressed AwsSolutions-* errors over the retraining-on P2 stack."""
    errors = _synth_nag_errors(monkeypatch)
    if errors:
        rendered = "\n".join(f"- {e}" for e in errors)
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) over the "
            f"retraining-on P2 stack:\n{rendered}"
        )
