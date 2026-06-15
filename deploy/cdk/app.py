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

from lib.routeiq_observability_stack import RouteIqObservabilityStack
from lib.routeiq_stack import RouteIqStack
from lib.routeiq_state_stack import RouteIqStateStack


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

    # P1 state-plane context keys (ADR-0028 Aurora + ADR-0029 ElastiCache). The
    # state stack is a SEPARATE stack/CI-stage per the ~30-min-rollback rule, so
    # it is gated on enable_state_stack (default true) and is byte-stable off.
    # routeiq:state_pg_version is read INSIDE ReplayStoreConstruct via
    # node.try_get_context (the context propagates to the child stack), so it is
    # NOT threaded as a constructor arg here. state_min_acu / state_max_acu /
    # cache_engine_version ARE constructor args.
    enable_state_stack = _bool_ctx(app, "routeiq:enable_state_stack", True)
    _state_min = _ctx(app, "routeiq:state_min_acu", None)
    _state_max = _ctx(app, "routeiq:state_max_acu", None)
    state_min_acu = float(_state_min) if _state_min not in (None, "") else None
    state_max_acu = float(_state_max) if _state_max not in (None, "") else None
    cache_engine_version = str(_ctx(app, "routeiq:cache_engine_version", "8.0"))

    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION"),
    )

    foundation = RouteIqStack(
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

    # P1 state stack (Aurora + ElastiCache), wired cross-stack to the P0
    # foundation BY REFERENCE (cred-free; CDK emits Export/Fn::ImportValue at
    # synth, never from_lookup). Separate stack = independent ~30-min rollback.
    if enable_state_stack:
        RouteIqStateStack(
            app,
            f"RouteIqStateStack-{env_name}",
            env=env,
            env_name=env_name,
            foundation=foundation,
            min_acu=state_min_acu,
            max_acu=state_max_acu,
            cache_engine_version=cache_engine_version,
        )

    # P2 (ADR-0026/0027): the SEPARATE config-state + observability + data-lake
    # stack. Flag-gated off by default so the default synth carries only the P0
    # stack (the ~30-minute-rollback rule keeps P2 independently deployable). When
    # routeiq:enable_observability_stack=true it synths alongside P0. AMG and the
    # data lake are each their own nested flag (both default off). The routing log
    # group is referenced by NAME (the P0 output, or the P0 naming convention) -
    # props-only, never from_lookup, to keep synth credential-free.
    if _bool_ctx(app, "routeiq:enable_observability_stack", False):
        RouteIqObservabilityStack(
            app,
            f"RouteIqObservabilityStack-{env_name}",
            env=env,
            env_name=env_name,
            routing_log_group_name=(_ctx(app, "routeiq:routing_log_group_name", None) or None),
            enable_amg=_bool_ctx(app, "routeiq:enable_amg", False),
            enable_data_lake=_bool_ctx(app, "routeiq:enable_data_lake", False),
            notify_emails=_split_csv_or_list(_ctx(app, "routeiq:notify_emails", [])),
        )

    cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))

    app.synth()


if __name__ == "__main__":
    main()
