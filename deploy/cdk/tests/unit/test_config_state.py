"""Unit tests for ConfigStateConstruct (RouteIQ P2 AppConfig config-state).

Asserts the AppConfig resource graph the P2 config-state substrate provisions
(re-derived symbol-by-symbol from the VSR ``config_state_construct.py`` AppConfig
core, re-based ``vllm-sr`` -> ``routeiq``, with the VSR S3-state-bucket /
CloudTrail / CodePipeline / EventBridge extras DROPPED - they are NOT P2
config-state deliverables for RouteIQ):

    AWS::AppConfig::Application            name="routeiq"
    AWS::AppConfig::Environment            name=<env_name>
    AWS::AppConfig::ConfigurationProfile   router-yaml / hosted / AWS.Freeform,
                                           Validators=[{Type:LAMBDA, Content:<arn>}]
    AWS::AppConfig::DeploymentStrategy     Linear20Pct3Min: LINEAR / 20 / 12min /
                                           5min final bake / ReplicateTo=NONE
    AWS::AppConfig::HostedConfigurationVersion  application/x-yaml placeholder
    AWS::AppConfig::Deployment             initial deploy, DependsOn the validator
                                           invoke permission
    AWS::Lambda::Function                  the config-validator (python3.13)
    AWS::Lambda::Permission                lambda:InvokeFunction for
                                           appconfig.amazonaws.com, SourceArn-scoped
                                           to the profile

Synthesised offline against the dummy env (account ``123456789012`` /
``us-west-2``); the validator Lambda is forced onto its hermetic INLINE-fallback
path (no Docker dependency) by an autouse fixture pinning ``_VALIDATOR_ASSET_PATH``
to a missing dir (mx-86079a). NO secrets, NO real account-ids.

Per the suite convention (cdk-resource-count-test-tripwire) property assertions
use ``has_resource_properties`` / ``find_resources`` / ``Match`` over brittle full
counts on shared resources, except where a count IS the assertion (exactly one
AppConfig application / deployment / appconfig-principled permission).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

# The dummy account/region the cred-free gate pins (mirrors tests/conftest.py).
DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"

# IAM's allowed Description charset: ASCII control trio + printable ASCII +
# Latin-1 supplement. An em-dash (U+2014) is OUTSIDE this set (passes synth, FAILS
# the IAM/CFN CREATE API).
_IAM_DESCRIPTION_CHARSET = re.compile("^[" + "\t\n\r" + " -~" + "¡-ÿ" + "]*$")


@pytest.fixture(autouse=True)
def _force_hermetic_validator_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the AppConfig validator Lambda's INLINE fallback (mx-86079a).

    The construct takes the Docker ``from_asset`` bundling path when
    ``shutil.which("docker")`` finds a docker binary - but the cred-free gate must
    NOT depend on a reachable/healthy Docker daemon. Pinning
    ``_VALIDATOR_ASSET_PATH`` to a non-existent dir makes the dual-resolution take
    the deterministic ``Code.from_inline`` placeholder path, so synth proceeds with
    no Docker.
    """
    import lib.config_state_construct as cfg

    missing = str(Path(__file__).parent / "_does_not_exist")
    monkeypatch.setattr(cfg, "_VALIDATOR_ASSET_PATH", missing)


def _config_state_template(**construct_kwargs: Any) -> Template:
    """Synthesise a throwaway Stack hosting ONE ConfigStateConstruct (offline)."""
    from lib.config_state_construct import ConfigStateConstruct

    app = cdk.App()
    env_name = construct_kwargs.pop("env_name", "dev")
    stack = cdk.Stack(
        app,
        f"RouteIqConfigStateTest-{env_name}",
        env=cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION),
    )
    ConfigStateConstruct(stack, "ConfigStateConstruct", env_name=env_name, **construct_kwargs)
    return Template.from_stack(stack)


# ------------------------------------------------------- application/env/profile


def test_appconfig_application_named_routeiq() -> None:
    """Exactly one AppConfig application, named ``routeiq`` (NOT vllm-sr)."""
    template = _config_state_template()
    template.resource_count_is("AWS::AppConfig::Application", 1)
    template.has_resource_properties("AWS::AppConfig::Application", {"Name": "routeiq"})


def test_appconfig_environment_named_for_env() -> None:
    """The AppConfig environment is named for the deploy env."""
    template = _config_state_template(env_name="prod")
    template.has_resource_properties("AWS::AppConfig::Environment", {"Name": "prod"})


def test_appconfig_profile_router_yaml_hosted_freeform() -> None:
    """The configuration profile is router-yaml / hosted / AWS.Freeform."""
    template = _config_state_template()
    template.has_resource_properties(
        "AWS::AppConfig::ConfigurationProfile",
        {
            "Name": "router-yaml",
            "LocationUri": "hosted",
            "Type": "AWS.Freeform",
        },
    )


# --------------------------------------------------------------- LAMBDA validator


def test_appconfig_profile_has_lambda_validator() -> None:
    """The profile carries a Validators[] entry of Type=LAMBDA whose Content is
    the validator Lambda's ARN.

    Load-bearing differentiator vs an unvalidated profile (VSR CONTRACT 4.1):
    without the validator wiring, ad-hoc CreateHostedConfigurationVersion calls
    bypass the YAML-parse / RouteIQ-shape / inline-secret-deny server-side checks.
    """
    template = _config_state_template()
    template.has_resource_properties(
        "AWS::AppConfig::ConfigurationProfile",
        {
            "Name": "router-yaml",
            "Validators": Match.array_with(
                [
                    Match.object_like(
                        {
                            "Type": "LAMBDA",
                            # Content is an Fn::GetAtt token to the validator Arn.
                            "Content": Match.object_like({"Fn::GetAtt": Match.array_with(["Arn"])}),
                        }
                    )
                ]
            ),
        },
    )


def test_validator_lambda_python_3_13_runtime() -> None:
    """The validator Lambda runs on python3.13 (mx-1d874a + the bundled handler)."""
    template = _config_state_template()
    fns = template.find_resources(
        "AWS::Lambda::Function", {"Properties": {"Runtime": "python3.13"}}
    )
    assert len(fns) >= 1, "expected the config-validator Lambda on python3.13"


def test_validator_lambda_named_for_env() -> None:
    """The validator Lambda is named routeiq-<env>-appconfig-validator."""
    template = _config_state_template()
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {"FunctionName": "routeiq-dev-appconfig-validator"},
    )


# ------------------------------------------------------------- deployment strategy


def test_appconfig_strategy_linear_20_12_5() -> None:
    """The deployment strategy is Linear20Pct3Min: LINEAR / 20 / 12min / 5min bake.

    The exact mandated knobs (LINEAR 20/12/5): growth_factor=20, growth_type=LINEAR,
    deployment_duration_in_minutes=12, final_bake_time_in_minutes=5, ReplicateTo=NONE.
    """
    template = _config_state_template()
    template.has_resource_properties(
        "AWS::AppConfig::DeploymentStrategy",
        {
            "Name": "Linear20Pct3Min",
            "DeploymentDurationInMinutes": 12,
            "GrowthFactor": 20,
            "GrowthType": "LINEAR",
            "FinalBakeTimeInMinutes": 5,
            "ReplicateTo": "NONE",
        },
    )


# ------------------------------------------------------ hosted version + deployment


def test_appconfig_initial_hosted_version_is_yaml() -> None:
    """The initial hosted configuration version is application/x-yaml."""
    template = _config_state_template()
    template.has_resource_properties(
        "AWS::AppConfig::HostedConfigurationVersion",
        {"ContentType": "application/x-yaml"},
    )


def test_appconfig_initial_deployment_present() -> None:
    """Exactly one AppConfig deployment (the initial placeholder deploy)."""
    template = _config_state_template()
    deployments = template.find_resources("AWS::AppConfig::Deployment")
    assert len(deployments) == 1, deployments


# --------------------------------------------------------- validator invoke perm


def test_validator_permission_grants_appconfig_invoke() -> None:
    """A Lambda::Permission allows appconfig.amazonaws.com to InvokeFunction.

    Without this, AppConfig cannot call the validator and the initial deployment
    fails at CREATE time.
    """
    template = _config_state_template()
    template.has_resource_properties(
        "AWS::Lambda::Permission",
        {
            "Action": "lambda:InvokeFunction",
            "Principal": "appconfig.amazonaws.com",
        },
    )


def test_exactly_one_appconfig_invoke_permission_source_arn_scoped() -> None:
    """Exactly one appconfig-principled permission, SourceArn-scoped to the profile.

    SourceArn scopes the invoke to a specific AppConfig configuration profile so no
    other AppConfig profile in the account can abuse this validator. The ARN is
    built via Stack.format_arn so the synthesised property is an Fn::Join token -
    assert structurally.
    """
    template = _config_state_template()
    perms = template.find_resources("AWS::Lambda::Permission")
    appconfig_perms = [
        p for p in perms.values() if p["Properties"].get("Principal") == "appconfig.amazonaws.com"
    ]
    assert len(appconfig_perms) == 1, (
        f"expected exactly one Lambda::Permission for appconfig.amazonaws.com, "
        f"got {len(appconfig_perms)}"
    )
    src_arn = appconfig_perms[0]["Properties"].get("SourceArn")
    assert src_arn is not None, "SourceArn must scope the permission to the profile"
    assert isinstance(src_arn, dict) and "Fn::Join" in src_arn, src_arn
    # The source ARN is the profile-only form (configurationprofile/<id>), NOT the
    # env-scoped runtime-poll form. Assert the joined parts mention the profile.
    joined = "".join(part for part in src_arn["Fn::Join"][1] if isinstance(part, str))
    assert "configurationprofile/" in joined, joined


def test_initial_deployment_depends_on_validator_permission() -> None:
    """The initial deployment DependsOn the validator invoke permission.

    Load-bearing ordering: without the dependency, CFN may schedule the deployment
    before the permission exists, so AppConfig cannot validate the placeholder
    version and the deployment fails.
    """
    template = _config_state_template()
    deployments = template.find_resources("AWS::AppConfig::Deployment")
    assert len(deployments) == 1
    [(_logical_id, deployment)] = deployments.items()
    deps = deployment.get("DependsOn") or []
    if isinstance(deps, str):
        deps = [deps]
    perms = template.find_resources("AWS::Lambda::Permission")
    appconfig_perm_ids = [
        lid
        for lid, p in perms.items()
        if p["Properties"].get("Principal") == "appconfig.amazonaws.com"
    ]
    assert len(appconfig_perm_ids) == 1
    assert appconfig_perm_ids[0] in deps, (
        f"expected {appconfig_perm_ids[0]} in deployment DependsOn={deps}"
    )


# ------------------------------------------------- VSR extras are NOT ported (P2 trim)


def test_vsr_state_bucket_and_pipeline_extras_are_dropped() -> None:
    """The VSR S3-state-bucket / CloudTrail / CodePipeline extras are NOT part of
    the RouteIQ P2 config-state substrate.

    The RouteIQ port trims the AppConfig core out of the VSR construct: there is NO
    S3 state bucket, NO CloudTrail audit trail, NO CodePipeline deployer pipeline
    (the full GitOps pipeline + deployer role remain a future tier - RouteIQ-1669).
    The EventBridge validator-mutation rule is now an OPT-IN audit core
    (enable_config_audit, DEFAULT OFF), so the DEFAULT surface still emits ZERO
    AWS::Events::Rule / AWS::SNS::Topic (asserted here); see test_config_audit.py
    for the flag-on assertions.
    """
    template = _config_state_template()
    template.resource_count_is("AWS::S3::Bucket", 0)
    template.resource_count_is("AWS::CloudTrail::Trail", 0)
    template.resource_count_is("AWS::CodePipeline::Pipeline", 0)
    # DEFAULT (audit off): zero EventBridge rule / SNS topic.
    template.resource_count_is("AWS::Events::Rule", 0)
    template.resource_count_is("AWS::SNS::Topic", 0)


# ------------------------------------------------------ placeholder config + secrets


def test_placeholder_config_parses_and_is_routeiq_shaped() -> None:
    """The placeholder config.yaml parses as YAML and carries RouteIQ-shape keys."""
    yaml = pytest.importorskip("yaml")
    from lib.config_state_construct import _PLACEHOLDER_ROUTER_CONFIG

    parsed = yaml.safe_load(_PLACEHOLDER_ROUTER_CONFIG)
    assert isinstance(parsed, dict)
    # RouteIQ-shaped (model_list + litellm_settings + general:), NOT the VSR
    # listeners/extproc schema.
    assert "model_list" in parsed
    assert isinstance(parsed["model_list"], list) and parsed["model_list"]
    assert parsed["model_list"][0].get("model_name")
    assert parsed["general"]["config_source"] == "file"


def test_placeholder_config_has_no_inline_secrets() -> None:
    """The seed config references secrets indirectly (os.environ/<VAR>), no literals.

    A literal sk-.../AKIA... in the seed would be rejected by the validator at
    deploy time AND is the inline-secret hazard the validator exists to block; the
    seed itself must pass its own deny rules.
    """
    from lib.config_state_construct import _PLACEHOLDER_ROUTER_CONFIG

    body = _PLACEHOLDER_ROUTER_CONFIG
    # No OpenAI/Anthropic-style key, no AWS access-key id, no bearer literal.
    assert not re.search(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}", body), body
    assert not re.search(r"\bAKIA[0-9A-Z]{16}\b", body), body
    assert not re.search(r"\bASIA[0-9A-Z]{16}\b", body), body
    # The one secret-adjacent value is an indirect os.environ/ reference.
    assert "os.environ/AWS_REGION" in body, body


def test_no_hardcoded_account_id_in_synth() -> None:
    """No real account-id leaks into the synthesised template.

    The only account-id present is the dummy 123456789012 the cred-free env pins;
    no other 12-digit account literal appears. (AppConfig/Lambda ARNs reference
    Aws.ACCOUNT_ID as a Ref token, not a literal.)
    """
    import json

    template = _config_state_template()
    blob = json.dumps(template.to_json())
    others = {m for m in re.findall(r"\b\d{12}\b", blob) if m != DUMMY_ACCOUNT}
    assert not others, f"unexpected 12-digit account-id literal(s) in template: {others}"


# ------------------------------------------------------------------------ ASCII


def test_iam_role_and_lambda_descriptions_are_ascii() -> None:
    """Every IAM role + Lambda Description is ASCII / Latin-1 only (P0 4.5).

    An em-dash (U+2014) passes ``cdk synth`` but FAILS the IAM/Lambda CREATE API.
    """
    template = _config_state_template()
    for res_type in ("AWS::IAM::Role", "AWS::Lambda::Function"):
        for logical, res in template.find_resources(res_type).items():
            desc = res["Properties"].get("Description")
            if isinstance(desc, str):
                assert _IAM_DESCRIPTION_CHARSET.match(desc), (
                    f"{res_type} {logical} Description has a char outside IAM's "
                    f"allowed Latin-1 set: {desc!r}"
                )
