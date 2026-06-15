"""Cred-free assertion: the AppConfig validator Lambda synth is DETERMINISTIC.

RouteIQ-4772. The validator code path is selected by an EXPLICIT toggle
(``bundle_validator_asset`` kwarg / ``routeiq:bundle_validator_asset`` context flag),
NOT an implicit ``shutil.which("docker")`` probe, so the synthesised template never
depends on whether Docker is on the host.

  * DEFAULT (toggle unset / False) -> the inline ``ZipFile`` placeholder + the
    ``index.lambda_handler`` entry point -- regardless of Docker. This is the
    gate-tested + snapshot path.
  * The toggle is the SINGLE lever that flips to the ``from_asset`` (real validator)
    deploy path; without it the default is host-independent.

This test does NOT monkeypatch ``_VALIDATOR_ASSET_PATH`` -- the whole point of the
fix is that the default synth is inline regardless of the asset/Docker state. So it
asserts the inline shape WITHOUT the old hermetic pin (proving the determinism).
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"


def _obs_template(**flags: object) -> Template:
    from lib.routeiq_observability_stack import RouteIqObservabilityStack

    app = cdk.App()
    stack = RouteIqObservabilityStack(
        app,
        "RouteIqObservabilityStack-dev",
        env=cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION),
        env_name="dev",
        **flags,
    )
    return Template.from_stack(stack)


def _validator_function(template: Template) -> dict:
    """The single AppConfig validator Lambda::Function in the P2 template."""
    fns = template.find_resources("AWS::Lambda::Function")
    validators = [
        f
        for f in fns.values()
        if isinstance(f["Properties"].get("FunctionName"), str)
        and "appconfig-validator" in f["Properties"]["FunctionName"]
    ]
    assert len(validators) == 1, f"expected one validator Lambda; got {validators}"
    return validators[0]


def test_default_synth_is_the_inline_path_regardless_of_docker() -> None:
    """The DEFAULT synth ships the inline placeholder (ZipFile) + index.lambda_handler.

    No ``_VALIDATOR_ASSET_PATH`` monkeypatch and no bundle flag: the determinism fix
    means the template is the inline path whether or not Docker / the asset exist.
    """
    template = _obs_template()
    fn = _validator_function(template)
    assert fn["Properties"]["Handler"] == "index.lambda_handler", fn["Properties"]
    code = fn["Properties"]["Code"]
    # Inline => a ZipFile body, NOT an S3 asset (S3Bucket/S3Key).
    assert "ZipFile" in code, f"default validator must be inline ZipFile, got {code}"
    assert "S3Bucket" not in code and "S3Key" not in code, (
        f"default validator must NOT be a from_asset S3 asset; got {code}"
    )


def test_inline_path_is_the_gate_tested_shape() -> None:
    """The validator profile still wires a Type=LAMBDA validator (the gate shape)."""
    template = _obs_template()
    template.has_resource_properties(
        "AWS::AppConfig::ConfigurationProfile",
        {"Validators": Match.array_with([Match.object_like({"Type": "LAMBDA"})])},
    )


def test_bundle_flag_default_false_keeps_inline() -> None:
    """The construct toggle defaults to inline even when the asset dir exists.

    Constructs the ConfigStateConstruct directly with the real asset path intact
    (no monkeypatch). Default ``bundle_validator_asset`` is False, so the synth stays
    inline regardless of whether the on-disk asset + Docker are present.
    """
    from lib.config_state_construct import ConfigStateConstruct

    app = cdk.App()
    stack = cdk.Stack(
        app,
        "S",
        env=cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION),
    )
    cs = ConfigStateConstruct(stack, "Cfg", env_name="dev")
    assert cs.bundle_validator_asset is False
    template = Template.from_stack(stack)
    fn = _validator_function(template)
    assert fn["Properties"]["Handler"] == "index.lambda_handler"
    assert "ZipFile" in fn["Properties"]["Code"]
