"""Helm-template render assertions for the KEDA ScaledObject (RouteIQ-9c4c).

KEDA event-driven autoscaling scales the gateway on a LOAD signal (the
backpressure in-flight gauge / RPS via the Prometheus adapter, or a CloudWatch
metric) instead of CPU/mem only. It is flag-gated (autoscaling.keda.enabled),
DEFAULT OFF, and MUTUALLY EXCLUSIVE with the chart's CPU/mem HPA: when KEDA is on
the chart HPA is suppressed (KEDA creates + owns its own HPA, so two would fight).

These tests shell out to ``helm template`` and assert on the rendered YAML -- no
cluster, no AWS creds. If ``helm`` is not on PATH the whole module skips (mirrors
the integration-skip ethos).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

# deploy/charts/routeiq-gateway/tests/test_render_keda.py -> deploy/charts/routeiq-gateway
CHART_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm not on PATH; render assertions auto-skip (matches integration-skip ethos)",
)


def _helm_template(*set_args: str) -> str:
    """Render the chart with the given ``--set k=v`` overrides; return raw YAML."""
    cmd = ["helm", "template", "rq", str(CHART_DIR)]
    for kv in set_args:
        cmd += ["--set", kv]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"helm template failed (rc={proc.returncode}):\n"
            f"argv={cmd}\nstderr={proc.stderr}"
        )
    return proc.stdout


def _docs(rendered: str) -> Iterator[dict]:
    for doc in yaml.safe_load_all(rendered):
        if doc:
            yield doc


def _by_kind(rendered: str, kind: str) -> list[dict]:
    return [d for d in _docs(rendered) if d.get("kind") == kind]


# ------------------------------------------------------------ default (off)


def test_no_scaledobject_when_keda_disabled() -> None:
    """Default render emits NO KEDA ScaledObject (byte-stable default)."""
    rendered = _helm_template()
    assert _by_kind(rendered, "ScaledObject") == []


def test_hpa_renders_when_autoscaling_on_keda_off() -> None:
    """With autoscaling on but KEDA off, the chart HPA renders (the default scaler)."""
    rendered = _helm_template("autoscaling.enabled=true")
    assert len(_by_kind(rendered, "HorizontalPodAutoscaler")) == 1
    assert _by_kind(rendered, "ScaledObject") == []


# ------------------------------------------------------------ flag-on graph


def test_scaledobject_renders_when_keda_enabled() -> None:
    """autoscaling.keda.enabled=true renders exactly one KEDA ScaledObject."""
    rendered = _helm_template("autoscaling.keda.enabled=true")
    objs = _by_kind(rendered, "ScaledObject")
    assert len(objs) == 1
    so = objs[0]
    assert so["apiVersion"] == "keda.sh/v1alpha1"
    # Targets the gateway Deployment by name.
    target = so["spec"]["scaleTargetRef"]
    assert target["kind"] == "Deployment"
    assert target["name"] == "rq-routeiq-gateway"


def test_keda_mutually_exclusive_with_hpa() -> None:
    """When KEDA is on, the chart HPA is suppressed (KEDA owns scaling).

    Both flags on at once: the ScaledObject renders and the chart HPA does NOT,
    so two HPAs never target one Deployment.
    """
    rendered = _helm_template(
        "autoscaling.enabled=true", "autoscaling.keda.enabled=true"
    )
    assert len(_by_kind(rendered, "ScaledObject")) == 1
    assert _by_kind(rendered, "HorizontalPodAutoscaler") == []


def test_keda_default_trigger_is_prometheus_backpressure() -> None:
    """The default trigger is a Prometheus query over the backpressure gauge."""
    rendered = _helm_template("autoscaling.keda.enabled=true")
    so = _by_kind(rendered, "ScaledObject")[0]
    triggers = so["spec"]["triggers"]
    assert len(triggers) == 1
    trig = triggers[0]
    assert trig["type"] == "prometheus"
    # The default PromQL scales on the backpressure in-flight gauge (RouteIQ-6cd5).
    assert "gateway_backpressure_active_requests" in trig["metadata"]["query"]
    assert "serverAddress" in trig["metadata"]
    assert "threshold" in trig["metadata"]


def test_keda_cloudwatch_trigger() -> None:
    """type=cloudwatch renders an aws-cloudwatch trigger with the AWS fields."""
    rendered = _helm_template(
        "autoscaling.keda.enabled=true",
        "autoscaling.keda.trigger.type=cloudwatch",
        "autoscaling.keda.trigger.awsRegion=us-east-1",
        "autoscaling.keda.trigger.namespace=RouteIQ",
        "autoscaling.keda.trigger.metricName=RequestsPerSecond",
    )
    so = _by_kind(rendered, "ScaledObject")[0]
    trig = so["spec"]["triggers"][0]
    assert trig["type"] == "aws-cloudwatch"
    assert trig["metadata"]["awsRegion"] == "us-east-1"
    assert trig["metadata"]["namespace"] == "RouteIQ"
    assert trig["metadata"]["metricName"] == "RequestsPerSecond"


def test_keda_replica_bounds_default_to_autoscaling_bounds() -> None:
    """min/maxReplicaCount fall back to autoscaling.min/maxReplicas when unset."""
    rendered = _helm_template(
        "autoscaling.keda.enabled=true",
        "autoscaling.minReplicas=3",
        "autoscaling.maxReplicas=12",
    )
    so = _by_kind(rendered, "ScaledObject")[0]
    assert so["spec"]["minReplicaCount"] == 3
    assert so["spec"]["maxReplicaCount"] == 12


def test_keda_authentication_ref_emitted_when_set() -> None:
    """An authenticationRef is emitted on the trigger only when configured."""
    rendered = _helm_template(
        "autoscaling.keda.enabled=true",
        "autoscaling.keda.trigger.authenticationRef=amp-sigv4",
    )
    so = _by_kind(rendered, "ScaledObject")[0]
    trig = so["spec"]["triggers"][0]
    assert trig["authenticationRef"]["name"] == "amp-sigv4"
