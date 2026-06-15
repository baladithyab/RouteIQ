#!/usr/bin/env python3
"""CDK app entrypoint for the RouteIQ Gateway AWS substrate.

Reads context keys from cdk.json (or cdk.context.json overrides) under the
``routeiq:`` prefix and instantiates a single :class:`RouteIqStack` named
``RouteIqStack-<env>`` (private multi-AZ EKS Auto Mode cluster + ECR GHCR
pull-through cache + one least-privilege pod IAM role via EKS Pod Identity).

This is the P0 foundation per
``docs/architecture/aws-rearchitecture/31-p0-cdk-foundation-proposal.md``. The
helper SHAPE (``_ctx`` / ``_bool_ctx`` / ``_split_csv_or_list``) is ported
verbatim from the vllm-sr-on-aws CDK app so the CLI-string footgun stays
defused; the stack/context naming is RouteIQ-specific (``routeiq:`` not
``vllm_sr:``).
"""

from __future__ import annotations

import os

import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

from lib.routeiq_stack import RouteIqStack


def _ctx(app: cdk.App, key: str, default):
    """Read a context value, falling back to ``default`` when unset."""
    value = app.node.try_get_context(key)
    return default if value is None else value


def _bool_ctx(app: cdk.App, key: str, default: bool) -> bool:
    """Read a boolean context value, parsing CLI string forms correctly.

    ``cdk synth --context key=false`` returns the string ``"false"``,
    and ``bool("false")`` is ``True`` in Python — so a naive
    ``bool(_ctx(...))`` silently keeps the default. This helper accepts
    both native bools (from cdk.json) and CLI strings (from --context).
    """
    value = app.node.try_get_context(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _split_csv_or_list(raw) -> list[str]:
    """Normalize a context value that may be a comma-separated string (from
    ``--context key=a,b``) or a JSON list (from cdk.json) into a clean
    ``list[str]``. Every list-shaped context reads the same way."""
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def main() -> None:
    app = cdk.App()

    env_name = _ctx(app, "routeiq:env", "dev")
    vpc_cidr = _ctx(app, "routeiq:vpc_cidr", "10.40.0.0/16")
    nat_gateways = int(_ctx(app, "routeiq:nat_gateways", 1))
    k8s_version = str(_ctx(app, "routeiq:k8s_version", "1.33"))
    enable_ghcr_ptc = _bool_ctx(app, "routeiq:enable_ghcr_ptc", True)
    # The Pod Identity association binding. These MUST match the chart's
    # rendered ServiceAccount (namespace, name) — the CDK CfnPodIdentityAssociation
    # is keyed on (namespace, serviceAccount), so a drift here silently breaks the
    # pod->role binding. See proposal §11.2.
    sa_namespace = _ctx(app, "routeiq:sa_namespace", "routeiq")
    sa_name = _ctx(app, "routeiq:sa_name", "routeiq-gateway")
    image_tag = _ctx(app, "routeiq:image_tag", "1.0.0-rc1")
    # CfnAccessEntry kubectl identities for the CI/operator role(s). A
    # comma-separated string (--context) or a JSON list (cdk.json); empty default
    # keeps the synth byte-stable.
    admin_principal_arns = _split_csv_or_list(_ctx(app, "routeiq:admin_principal_arns", []))
    # Optional resource scopes for the least-privilege pod-role statements
    # (BedrockInvoke / SecretsRead / ConfigS3Read). Empty defaults keep the
    # default synth byte-stable; supply at deploy time to narrow.
    bedrock_model_arns = _split_csv_or_list(_ctx(app, "routeiq:bedrock_model_arns", []))
    config_s3_bucket = _ctx(app, "routeiq:config_s3_bucket", None) or None
    secret_arns = _split_csv_or_list(_ctx(app, "routeiq:secret_arns", []))

    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION"),
    )

    RouteIqStack(
        app,
        f"RouteIqStack-{env_name}",
        env=env,
        env_name=env_name,
        vpc_cidr=vpc_cidr,
        nat_gateways=nat_gateways,
        k8s_version=k8s_version,
        enable_ghcr_ptc=enable_ghcr_ptc,
        sa_namespace=sa_namespace,
        sa_name=sa_name,
        image_tag=image_tag,
        admin_principal_arns=admin_principal_arns,
        bedrock_model_arns=bedrock_model_arns,
        config_s3_bucket=config_s3_bucket,
        secret_arns=secret_arns,
    )

    cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))

    app.synth()


if __name__ == "__main__":
    main()
