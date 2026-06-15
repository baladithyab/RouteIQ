"""Shared RouteIQ AWS-resource naming helpers (single source of truth).

The routing-log-group name and the EKS cluster name are derived from the same
``routeiq-<env>`` stem in MULTIPLE places (the P0 producer in
``eks_cluster_construct``, the P0 IAM nag-suppression literal in
``nag_suppressions``, and the P2 default fallback in
``routeiq_observability_stack``). When that literal is duplicated, a drift in any
one copy silently de-couples the P2 metric filters / lake subscription from the P0
log group they must attach to (RouteIQ-45fa). These helpers collapse the literal to
ONE definition so every call site reads the same string.

PURE / byte-neutral: the returned strings are byte-identical to the literals they
replace, so no synthesised template changes (the P0/P1 snapshots stay green).

Also home for the P0->P2 cross-stack export name for the routing log group
(RouteIQ-81c4): a STABLE export name so once P2 ``Fn::ImportValue``s it, the P0
export can be referenced deterministically (and not renamed without first removing
the P2 consumer -- the cfn-cross-stack-export-revisioned-ref-deadlock hazard).
"""

from __future__ import annotations


def cluster_name(env_name: str) -> str:
    """The EKS Auto Mode cluster name for an env (``routeiq-<env>``)."""
    return f"routeiq-{env_name}"


def routing_log_group_name(env_name: str) -> str:
    """The dedicated routing-decision CloudWatch log group name for an env.

    ``/aws/containerinsights/routeiq-<env>/routeiq-routing`` -- the P0
    EksClusterConstruct CDK-creates this group; the P2 metric filters + the data
    lake subscription attach to it. ONE source of truth so a P0/P2 drift cannot
    silently de-couple the filters from the group.
    """
    return f"/aws/containerinsights/{cluster_name(env_name)}/routeiq-routing"


def routing_log_group_export_name(env_name: str) -> str:
    """The STABLE CFN export name the P0 stack publishes for the routing group name.

    Consumed by the P2 ``RouteIqObservabilityStack`` via ``Fn::ImportValue`` when a
    combined-deploy app threads the P0 foundation. STABLE on purpose
    (cfn-cross-stack-export-revisioned-ref-deadlock): once a P2 stack imports it, the
    export cannot be renamed/removed without first removing the P2 consumer.
    """
    return f"RouteIqStack-{env_name}-RoutingLogGroupName"
