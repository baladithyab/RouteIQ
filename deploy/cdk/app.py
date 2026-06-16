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
    # RouteIQ-6150 (C1): cross-account Bedrock capacity account ids. For each id the
    # home pod role is granted sts:AssumeRole on the computed
    # RouteIqBedrockCapacity-<env> member role ARN (which BedrockCapacityMemberStack
    # mints in that account). Empty default keeps the synth byte-stable (no grant,
    # no output).
    capacity_account_ids = _split_csv_or_list(_ctx(app, "routeiq:capacity_account_ids", []))
    # RouteIQ-4f59 (WAF): the WAFv2 edge layer. DEFAULT OFF; the construct is built
    # only when routeiq:enable_waf is true AND an operator supplies a non-empty
    # routeiq:waf_alb_arn (no ALB renders at P0 / ClusterIP, so the live attach is
    # deploy-gated). Empty/false defaults keep the default synth byte-stable (zero
    # AWS::WAFv2::* resources). The waf_rate_limit honors a CLI string or null.
    enable_waf = _bool_ctx(app, "routeiq:enable_waf", False)
    waf_alb_arn = _ctx(app, "routeiq:waf_alb_arn", None) or None
    _waf_rate_limit_raw = _ctx(app, "routeiq:waf_rate_limit", None)
    waf_rate_limit = int(_waf_rate_limit_raw) if _waf_rate_limit_raw not in (None, "") else None
    waf_crs_block = _bool_ctx(app, "routeiq:waf_crs_block", False)
    waf_rate_block = _bool_ctx(app, "routeiq:waf_rate_block", False)

    # RouteIQ-acdc (GPU NodePool): DEFAULT OFF. The two AWS-managed Auto Mode node
    # pools are CPU-only, so a pod requesting nvidia.com/gpu sits Pending forever.
    # When true the foundation emits a GpuNodePoolManifest CfnOutput carrying the
    # custom GPU NodePool + EC2 NodeClass YAML the operator/GitOps applies
    # out-of-band. Off keeps the default synth byte-stable (zero GPU surface).
    enable_gpu_nodepool = _bool_ctx(app, "routeiq:enable_gpu_nodepool", False)

    # RouteIQ-c0be (Native Bedrock Guardrail): DEFAULT OFF. AUTHORING STAGE - when
    # true the foundation mints an IaC-owned CfnGuardrail + a pinned
    # CfnGuardrailVersion + GuardrailId/VersionNumber CfnOutputs. The data-path
    # activation (feeding those into the bedrock_guardrails plugin) is a separate
    # wave (RouteIQ-9f14). Off keeps the default synth byte-stable (zero
    # AWS::Bedrock::Guardrail resources).
    enable_native_guardrail = _bool_ctx(app, "routeiq:enable_native_guardrail", False)

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
        capacity_account_ids=capacity_account_ids,
        config_s3_bucket=config_s3_bucket,
        secret_arns=secret_arns,
        enable_waf=enable_waf,
        waf_alb_arn=waf_alb_arn,
        waf_rate_limit=waf_rate_limit,
        waf_crs_block=waf_crs_block,
        waf_rate_block=waf_rate_block,
        enable_gpu_nodepool=enable_gpu_nodepool,
        enable_native_guardrail=enable_native_guardrail,
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
    # routeiq:enable_observability_stack=true it synths alongside P0.
    #
    # COMBINED-DEPLOY WIRING (RouteIQ-569f / 81c4 / 74c0 / 717b): the P0 foundation
    # is threaded by REFERENCE (foundation=foundation), exactly as the P1 state
    # stack is. CDK resolves the cross-stack references (the pod-role ARN, the
    # routing log-group name) at synth into Export / Fn::ImportValue -- cred-free,
    # never from_lookup. With the foundation present the P2 stack: references the
    # real P0 routing ILogGroup + add_dependency(P0) so CFN deploys the group before
    # the filters (81c4); and owns a P2-stack iam.Policy attached to the P0 pod role
    # granting the AppConfig runtime poll (569f) + aps:RemoteWrite (74c0/717b). AMG
    # and the data lake are each their own nested flag (both default off).
    if _bool_ctx(app, "routeiq:enable_observability_stack", False):
        RouteIqObservabilityStack(
            app,
            f"RouteIqObservabilityStack-{env_name}",
            env=env,
            env_name=env_name,
            foundation=foundation,
            routing_log_group_name=(_ctx(app, "routeiq:routing_log_group_name", None) or None),
            enable_amg=_bool_ctx(app, "routeiq:enable_amg", False),
            enable_data_lake=_bool_ctx(app, "routeiq:enable_data_lake", False),
            # RouteIQ-1669 (ADR-0026 audit core): DEFAULT OFF. When true the
            # config-state construct adds a TLS SNS topic + an EventBridge rule on
            # mutating AppConfig profile API calls (the validator-mutation audit).
            # The full GitOps CodePipeline + deployer role remain a future tier.
            enable_config_audit=_bool_ctx(app, "routeiq:enable_config_audit", False),
            notify_emails=_split_csv_or_list(_ctx(app, "routeiq:notify_emails", [])),
        )

    cdk.Aspects.of(app).add(AwsSolutionsChecks(verbose=True))

    app.synth()


if __name__ == "__main__":
    main()
