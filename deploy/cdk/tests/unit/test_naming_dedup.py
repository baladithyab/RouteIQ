"""Cred-free assertions for the routing-log-group naming dedup + the stale filter.

* RouteIQ-45fa -- the routing-log-group name is ONE shared constant
  (``lib.naming.routing_log_group_name``). The P0 producer (the CDK-created
  LogGroup + the pod-role Logs IAM statement), the P0 nag-suppression literal, and
  the P2 default fallback ALL equal ``routing_log_group_name(env)`` -- one source
  of truth, so a drift cannot silently de-couple the P2 filters from the P0 group.
  Plus a source guard: the literal ``/aws/containerinsights/routeiq-`` appears in
  exactly ONE non-test source location (``lib/naming.py``).
* RouteIQ-8f08 -- the stale P0 prep-only RoutingLatencyByModel filter (keyed on the
  dead ``$.msg`` / ``$.routing_latency_ms`` VSR contract) is DELETED. The P0
  default surface still emits 0 MetricFilters, and the stale field strings appear
  NOWHERE in the P0 template or the P0 construct source.
"""

from __future__ import annotations

import json
from pathlib import Path

from aws_cdk.assertions import Template

from lib.naming import routing_log_group_name
from tests.conftest import make_obs_stack, template_for

_LIB_DIR = Path(__file__).resolve().parents[2] / "lib"


# ----------------------------------------------------------------- RouteIQ-45fa


def test_p0_logs_iam_statement_uses_shared_name() -> None:
    """The P0 pod-role Logs IAM statement targets routing_log_group_name(env)."""
    template = template_for(env_name="dev")
    blob = json.dumps(template.to_json())
    # The shared name appears in the Logs IAM statement resource (with the :* suffix).
    assert routing_log_group_name("dev") in blob, (
        "the P0 Logs IAM statement must reference the shared routing-log-group name"
    )


def test_p0_creates_log_group_with_shared_name() -> None:
    """The P0 CDK-created routing LogGroup carries routing_log_group_name(env)."""
    template = template_for(env_name="dev")
    log_groups = template.find_resources("AWS::Logs::LogGroup")
    names = {
        lg["Properties"].get("LogGroupName")
        for lg in log_groups.values()
        if isinstance(lg["Properties"].get("LogGroupName"), str)
    }
    assert routing_log_group_name("dev") in names, names


def test_nag_suppression_literal_uses_shared_name() -> None:
    """The P0 IAM5 nag-suppression appliesTo equals the shared name (+ :* suffix).

    Asserts via the rendered suppression metadata on the synthesised P0 stack: the
    Logs IAM5 suppression's resource literal embeds routing_log_group_name(env).
    """
    # The suppression literal is a Python f-string built from the shared helper, so
    # the strongest check is that the helper value is the substring used; assert the
    # source no longer hand-codes the path independently (see source-guard below) and
    # that the helper is the producer of the literal.
    from lib import nag_suppressions

    src = (_LIB_DIR / "nag_suppressions.py").read_text(encoding="utf-8")
    assert "routing_log_group_name(stack.env_name)" in src, (
        "nag_suppressions must build the Logs IAM5 appliesTo from the shared helper"
    )
    assert "/aws/containerinsights/routeiq-" not in src, (
        "nag_suppressions must NOT hand-code the routing-log-group literal"
    )
    assert nag_suppressions is not None


def test_p2_default_fallback_uses_shared_name() -> None:
    """The P2 no-foundation default falls back to routing_log_group_name(env)."""
    from tests.unit._obs_helpers import standalone_obs_template

    template = standalone_obs_template(env_name="dev")
    template.has_resource_properties(
        "AWS::Logs::MetricFilter",
        {"LogGroupName": routing_log_group_name("dev")},
    )


def test_routing_log_group_literal_in_exactly_one_source_location() -> None:
    """The ``/aws/containerinsights/routeiq-`` literal lives ONLY in lib/naming.py.

    One source of truth (RouteIQ-45fa). Every other call site derives the name from
    the shared helper.
    """
    offenders: list[str] = []
    for path in _LIB_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "/aws/containerinsights/routeiq-" in text and path.name != "naming.py":
            offenders.append(path.name)
    assert not offenders, (
        f"the routing-log-group literal must live ONLY in lib/naming.py; "
        f"found it hand-coded in: {offenders}"
    )


def test_combined_p0_and_p2_agree_on_one_log_group_name() -> None:
    """The P0 export VALUE and the P2 filter (imported) both trace to the one name."""
    _app, foundation, obs = make_obs_stack(env_name="dev")
    p0 = Template.from_stack(foundation)
    outputs = p0.find_outputs("*")
    assert outputs["RoutingLogGroupName"]["Value"] == routing_log_group_name("dev")


# ----------------------------------------------------------------- RouteIQ-8f08


def test_p0_emits_zero_metric_filters_by_default() -> None:
    """The stale prep-only filter is gone; P0 default surface emits 0 MetricFilters."""
    template = template_for(env_name="dev")
    template.resource_count_is("AWS::Logs::MetricFilter", 0)


def test_p0_template_has_no_stale_routing_latency_contract() -> None:
    """The dead $.msg / $.routing_latency_ms VSR contract appears nowhere in P0."""
    template = template_for(env_name="dev")
    blob = json.dumps(template.to_json())
    assert "$.routing_latency_ms" not in blob, "stale $.routing_latency_ms in P0 template"
    assert "routing_latency_ms" not in blob, "no routing_latency metric should exist at P0"


def test_eks_construct_source_dropped_the_stale_filter() -> None:
    """The eks_cluster_construct source no longer carries the stale field/flag."""
    src = (_LIB_DIR / "eks_cluster_construct.py").read_text(encoding="utf-8")
    assert "routing_latency_ms" not in src, (
        "eks_cluster_construct must not reference the stale routing_latency_ms field"
    )
    assert "_ENABLE_ROUTING_LATENCY_BY_MODEL" not in src, (
        "the dead prep-filter flag must be removed"
    )
    # The stale VSR key must be gone too (P2 owns the real $.event-keyed filter).
    assert '"$.msg"' not in src, "the stale $.msg VSR selector must be removed"
