"""Unit tests for the GitOps deploy tier (RouteIQ-1669 / ADR-0026).

ADR-0026's "Day-2 GitOps path" has two halves. The validator-mutation audit CORE
shipped in 761a4ee (``ConfigStateConstruct(enable_config_audit=...)``). This is the
REMAINING cred-free half: ``GitOpsPipelineConstruct`` - a CodePipeline
(Source[config bucket] -> [Approve, prod only] -> Deploy) + a NARROW deployer IAM
role carrying EXACTLY the 5 AppConfig deploy actions + an explicit DENY of
Update/DeleteConfigurationProfile so the deployer cannot strip the load-bearing
config validator.

It is FLAG-GATED off by default at the P2 stack (``enable_gitops_pipeline``), so
the byte-stable guarantee is that the default ``RouteIqObservabilityStack`` emits
ZERO ``AWS::CodePipeline::Pipeline`` / ``AWS::CodeBuild::Project`` /
deployer-role resources. These tests assert the flag-ON resource graph + the
deployer-role action set + the prod-only approval stage + that the flag stays
off-by-default + cdk-nag clean.

CRED-FREE / OPERATOR-GATED SPLIT: the CodePipeline + deployer role authored here
is cred-free. The LIVE deploy (real config commits, the prod approval click, an
actual AppConfig deployment) is operator-gated.

Synthesised offline against the dummy env (account ``123456789012`` /
``us-west-2``), credential-free, via the shared ``make_obs_stack`` helper (the P2
stack wired to a P0 foundation, exactly as ``app.py`` does). The validator Lambda
is forced onto its hermetic inline-fallback path by the autouse fixture pinning
``_VALIDATOR_ASSET_PATH`` to a missing dir (mx-86079a), mirroring test_config_audit.py.
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

from lib.gitops_pipeline_construct import (
    _DEPLOYER_APPCONFIG_ACTIONS,
    _DEPLOYER_DENIED_ACTIONS,
)
from lib.routeiq_observability_stack import RouteIqObservabilityStack
from lib.routeiq_stack import RouteIqStack
from tests.conftest import dummy_env, make_obs_stack


@pytest.fixture(autouse=True)
def _force_hermetic_validator_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the validator asset path to a missing dir -> deterministic inline path."""
    import lib.config_state_construct as cfg

    missing = str(Path(__file__).parent / "_does_not_exist_gitops")
    monkeypatch.setattr(cfg, "_VALIDATOR_ASSET_PATH", missing)


def _obs_template(**kwargs: object) -> Template:
    _app, _foundation, obs = make_obs_stack(**kwargs)  # type: ignore[arg-type]
    return Template.from_stack(obs)


# -------------------------------------------------------------- flag-off default


def test_gitops_off_by_default_emits_no_pipeline_or_codebuild() -> None:
    """The default P2 stack (no enable_gitops_pipeline) emits zero GitOps resources."""
    template = _obs_template(env_name="dev")
    template.resource_count_is("AWS::CodePipeline::Pipeline", 0)
    template.resource_count_is("AWS::CodeBuild::Project", 0)


# ----------------------------------------------------------------- flag-on graph


def test_gitops_emits_pipeline_and_codebuild_when_enabled() -> None:
    """enable_gitops_pipeline=True adds a CodePipeline + a CodeBuild deploy project."""
    template = _obs_template(env_name="dev", enable_gitops_pipeline=True)
    template.resource_count_is("AWS::CodePipeline::Pipeline", 1)
    template.resource_count_is("AWS::CodeBuild::Project", 1)


def test_gitops_source_bucket_present() -> None:
    """The GitOps tier provisions a config source bucket (the pipeline Source)."""
    _app, _foundation, obs = make_obs_stack(env_name="dev", enable_gitops_pipeline=True)
    assert obs.gitops_pipeline is not None
    assert obs.gitops_pipeline.source_bucket is not None


# ----------------------------------------------------- narrow deployer role


def _deployer_policy_statements(template: Template) -> list[dict]:
    """Return all IAM policy statements in the template (across all policies)."""
    statements: list[dict] = []
    for policy in template.find_resources("AWS::IAM::Policy").values():
        doc = policy["Properties"]["PolicyDocument"]
        statements.extend(doc.get("Statement", []))
    return statements


def test_deployer_role_has_exactly_five_appconfig_allow_actions() -> None:
    """The deployer role's AppConfig ALLOW carries EXACTLY the 5 deploy actions.

    ADR-0026: the deployer holds exactly five AppConfig actions
    (CreateHostedConfigurationVersion + StartDeployment + the three resolve GETs).
    """
    template = _obs_template(env_name="dev", enable_gitops_pipeline=True)
    statements = _deployer_policy_statements(template)
    allow = [
        s for s in statements if s.get("Sid") == "AppConfigDeploy" and s.get("Effect") == "Allow"
    ]
    assert len(allow) == 1, f"expected one AppConfigDeploy ALLOW statement; got {allow}"
    actions = allow[0]["Action"]
    if isinstance(actions, str):
        actions = [actions]
    assert set(actions) == set(_DEPLOYER_APPCONFIG_ACTIONS)
    assert len(actions) == 5


def test_deployer_role_explicitly_denies_profile_mutation() -> None:
    """The deployer role explicitly DENIES Update/DeleteConfigurationProfile.

    The validator invariant: a deployer that could mutate the profile could strip
    the load-bearing config VALIDATOR. An explicit Deny beats any Allow, so the
    deployer is structurally unable to do it (complementing the audit alarm).
    """
    template = _obs_template(env_name="dev", enable_gitops_pipeline=True)
    statements = _deployer_policy_statements(template)
    deny = [
        s for s in statements if s.get("Sid") == "DenyValidatorStrip" and s.get("Effect") == "Deny"
    ]
    assert len(deny) == 1, f"expected one DenyValidatorStrip DENY statement; got {deny}"
    actions = deny[0]["Action"]
    if isinstance(actions, str):
        actions = [actions]
    assert set(actions) == set(_DEPLOYER_DENIED_ACTIONS)


def test_deployer_appconfig_allow_is_arn_scoped_never_wildcard() -> None:
    """The deployer's AppConfig ALLOW is ARN-scoped (never Resource=*)."""
    template = _obs_template(env_name="dev", enable_gitops_pipeline=True)
    statements = _deployer_policy_statements(template)
    allow = next(
        s for s in statements if s.get("Sid") == "AppConfigDeploy" and s.get("Effect") == "Allow"
    )
    resources = allow["Resource"]
    if not isinstance(resources, list):
        resources = [resources]
    # No bare "*" resource; each ARN is a Fn::Join / string scoped to the app.
    assert "*" not in resources


# ----------------------------------------------------- prod-only approval stage


def test_dev_pipeline_has_no_approval_stage() -> None:
    """dev/stage flow straight Source -> Deploy (no manual approval gate)."""
    _app, _foundation, obs = make_obs_stack(env_name="dev", enable_gitops_pipeline=True)
    assert obs.gitops_pipeline is not None
    stage_names = [s.stage_name for s in obs.gitops_pipeline.pipeline.stages]
    assert "Approve" not in stage_names
    assert stage_names == ["Source", "Deploy"]


def test_prod_pipeline_inserts_approval_stage() -> None:
    """[Approve, PROD ONLY]: prod inserts a manual approval gate before Deploy."""
    _app, _foundation, obs = make_obs_stack(env_name="prod", enable_gitops_pipeline=True)
    assert obs.gitops_pipeline is not None
    stage_names = [s.stage_name for s in obs.gitops_pipeline.pipeline.stages]
    assert stage_names == ["Source", "Approve", "Deploy"]
    assert obs.gitops_pipeline.approval_action is not None


def test_prod_pipeline_emits_manual_approval_in_template() -> None:
    """The prod pipeline's Approve stage renders a Manual approval action."""
    template = _obs_template(env_name="prod", enable_gitops_pipeline=True)
    pipeline = next(iter(template.find_resources("AWS::CodePipeline::Pipeline").values()))
    stage_names = [s["Name"] for s in pipeline["Properties"]["Stages"]]
    assert "Approve" in stage_names


# -------------------------------------------------------------- outputs


def test_gitops_emits_operator_outputs() -> None:
    """The construct emits the source bucket / pipeline / deployer-role outputs."""
    template = _obs_template(env_name="dev", enable_gitops_pipeline=True)
    outputs = template.find_outputs("*")
    keys = list(outputs)
    assert any("ConfigSourceBucketName" in k for k in keys), keys
    assert any("ConfigPipelineName" in k for k in keys), keys
    assert any("DeployerRoleArn" in k for k in keys), keys


# -------------------------------------------------------------------- cdk-nag


def _synth_appconfig_errors(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Synth the GitOps-on P2 stack with cdk-nag and return its AwsSolutions errors.

    Reads cdk-nag findings from the synthesised cloud-assembly ``manifest.json`` on
    disk rather than via ``Annotations.from_stack`` / ``cx_api.messages``: the
    CodePipeline's trace metadata trips a known jsii reference-map deref bug
    (``KeyError: aws-cdk-lib.cloud_assembly_schema.MetadataEntry``) when the Python
    layer dereferences the message list. The on-disk manifest carries the same
    ``aws:cdk:error`` metadata entries cdk-nag emits, so reading them as plain JSON
    is an equivalent, jsii-bug-immune assertion of nag-cleanliness.
    """
    import lib.config_state_construct as cfg

    monkeypatch.setattr(cfg, "_VALIDATOR_ASSET_PATH", "/_does_not_exist_gitops_nag")
    outdir = tempfile.mkdtemp()
    app = cdk.App(outdir=outdir)
    foundation = RouteIqStack(app, "RouteIqStack-dev", env=dummy_env(), env_name="dev")
    RouteIqObservabilityStack(
        app,
        "RouteIqObservabilityStack-dev",
        env=dummy_env(),
        env_name="dev",
        foundation=foundation,
        enable_gitops_pipeline=True,
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


def test_gitops_construct_cdk_nag_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """No AwsSolutions-* errors survive over the GitOps-on P2 stack.

    The CodePipeline/CodeBuild CDK-managed wildcards carry inline IAM5 suppressions
    on the construct; the deployer role's AppConfig allow is ARN-scoped + the deny
    needs none; the source/artifact bucket is KMS-SSE + enforce_ssl + BPA. Guards
    that the GitOps tier needs no NEW unsuppressed nag findings.
    """
    errors = _synth_appconfig_errors(monkeypatch)
    if errors:
        rendered = "\n".join(f"- {e}" for e in errors)
        raise AssertionError(
            f"{len(errors)} unsuppressed AwsSolutions-* error(s) over the GitOps-on "
            f"P2 stack:\n{rendered}"
        )
