"""Evidenced cdk-nag suppressions for the RouteIQ P2 ObservabilityConstruct.

Ports the SHAPE of the P0 ``nag_suppressions.py``:
``apply_observability_nag_suppressions(stack)`` fans out to per-concern helpers,
each calling ``NagSuppressions.add_resource_suppressions`` (or ``_by_path``)
against an explicit resource, with two hard rules carried from the P0 source:

- **Flag guards.** Suppressions whose target is flag-gated (the AMG data-source
  role only exists when ``enable_amg=True``) are guarded with
  ``getattr(..., None)`` so a flag-off synth does not target an absent resource.
- **Evidenced reasons.** Every suppression carries (a) why it is safe, (b) that it
  is the least-privilege / only valid form, and (c) an ``Owner:`` line - RouteIQ
  specific, NOT reused from VSR.

The stack passed in is the composition root that owns the
``ObservabilityConstruct`` on ``stack.observability``. The AMP workspace, the SNS
topic + its resource policy, the metric filters, the CW-Logs-native alarms, and
the dashboard carry NO cdk-nag AwsSolutions findings on the default surface (the
metric filters + alarms + dashboard are L2 over an imported log group with no IAM;
the SNS topic uses the SNS-managed key and an enforce_ssl resource policy). The
ONLY finding is the AMG data-source role's read-scope ``*`` resources, which only
exist when AMG is enabled - hence the single getattr-guarded helper below.
"""

from __future__ import annotations

from typing import Any

from cdk_nag import NagPackSuppression, NagSuppressions


def apply_observability_nag_suppressions(stack: Any) -> None:
    """Apply every justified AwsSolutionsChecks suppression for the obs surface.

    Called AFTER the ObservabilityConstruct is composed (so every resource exists
    before a suppression targets it by path). On the default (AMG-off) surface
    this is effectively a no-op - the construct produces no unsuppressed findings.
    """
    _suppress_amg_role(stack)


def _suppress_amg_role(stack: Any) -> None:
    """Suppress the AMG data-source role's read-scope wildcards (IAM5).

    Only present when ``enable_amg=True``. Amazon Managed Grafana's data-source
    role needs account-wide CloudWatch + X-Ray READ access to populate the
    PROMETHEUS/CLOUDWATCH/XRAY panels: ``cloudwatch:GetMetricData`` /
    ``ListMetrics`` / ``GetMetricStatistics`` / ``DescribeAlarms`` and
    ``xray:GetTraceSummaries`` / ``BatchGetTraces`` / ``GetServiceGraph`` do not
    accept a narrower resource than ``*`` (there is no per-metric or per-trace
    ARN to scope to). This is the AWS-documented read scope for a Grafana
    data-source role; the AMP query statement IS scoped to the single workspace
    ARN. getattr-guarded so a flag-off synth does not target an absent role.
    """
    observability = getattr(stack, "observability", None)
    if observability is None:
        return
    amg_role = getattr(observability, "amg_role", None)
    if amg_role is None:
        # AMG off: the data-source role does not exist, nothing to suppress.
        return

    NagSuppressions.add_resource_suppressions(
        amg_role,
        [
            NagPackSuppression(
                id="AwsSolutions-IAM5",
                reason=(
                    "The Amazon Managed Grafana data-source role's CloudWatchRead "
                    "and XRayRead statements use Resource=* because the CloudWatch "
                    "GetMetricData/ListMetrics/GetMetricStatistics/DescribeAlarms "
                    "and X-Ray GetTraceSummaries/BatchGetTraces/GetServiceGraph "
                    "read APIs do not accept a per-metric or per-trace ARN - * is "
                    "the AWS-documented, only-valid read scope for a Grafana "
                    "data-source role. The AMP query statement is scoped to the "
                    "single workspace ARN. The role is read-only and only exists "
                    "when routeiq enable_amg=True. Owner: ObservabilityConstruct "
                    "(AmgWorkspaceRole CloudWatchRead/XRayRead)."
                ),
                applies_to=["Resource::*"],
            ),
        ],
        apply_to_children=True,
    )
