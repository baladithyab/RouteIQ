"""Helm-template render assertions for the bundled ADOT collector (RouteIQ-85e3).

These tests shell out to ``helm template`` and assert on the rendered Kubernetes
YAML -- no cluster, no AWS creds, CI-cheap (mirrors ``test_render.py``). They pin
the P1 contract: with defaults the chart renders an ADOT collector (Deployment +
ConfigMap with awsemf/awsxray exporters + Service) AND the gateway exports OTLP at
that collector; the ``otel.collector.enabled=false`` flag removes the collector
(and the gateway's auto-targeted OTLP endpoint). Live CloudWatch/X-Ray delivery
needs pod IAM and is an operator concern (documented), so it is NOT asserted here.

If ``helm`` is not on PATH the whole module is skipped (matches the
integration-test auto-skip ethos).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

# deploy/charts/routeiq-gateway/tests/test_render_adot_collector.py
#   -> deploy/charts/routeiq-gateway
CHART_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm not on PATH; render assertions auto-skip (matches integration-skip ethos)",
)


def _helm_template(*set_args: str, show_only: str | None = None) -> str:
    cmd = ["helm", "template", "rq", str(CHART_DIR)]
    for kv in set_args:
        cmd += ["--set", kv]
    if show_only is not None:
        cmd += ["--show-only", show_only]
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


def _kinds(rendered: str) -> list[str]:
    return [d.get("kind") for d in _docs(rendered)]


def _by_kind_name(rendered: str, kind: str, name_suffix: str) -> dict | None:
    for d in _docs(rendered):
        if d.get("kind") == kind and str(
            d.get("metadata", {}).get("name", "")
        ).endswith(name_suffix):
            return d
    return None


def _deployment(rendered: str) -> dict:
    for d in _docs(rendered):
        if d.get("kind") == "Deployment" and not str(
            d.get("metadata", {}).get("name", "")
        ).endswith("-adot-collector"):
            return d
    raise AssertionError("no gateway Deployment in render")


def _main_container_env(rendered: str) -> list[dict]:
    spec = _deployment(rendered)["spec"]["template"]["spec"]
    return spec["containers"][0].get("env", [])


def _env_value(env: list[dict], name: str) -> str | None:
    for e in env:
        if e.get("name") == name:
            return e.get("value")
    return None


# ---------------------------------------------------------------------------
# Default render: collector present + gateway points OTLP at it
# ---------------------------------------------------------------------------


def test_default_render_emits_adot_collector_deployment() -> None:
    rendered = _helm_template()
    dep = _by_kind_name(rendered, "Deployment", "-adot-collector")
    assert dep is not None, "default render must emit an ADOT collector Deployment"

    container = dep["spec"]["template"]["spec"]["containers"][0]
    # The aws-otel-collector image (ADOT distro), not a vanilla otel/collector.
    assert "aws-otel-collector" in container["image"], (
        f"collector image must be the ADOT distro: {container['image']}"
    )
    # Config delivery is load-bearing: --config must point at the mounted file so
    # the stock image default config is NOT used silently.
    args = " ".join(container.get("command", []) + container.get("args", []))
    assert "--config=/etc/otel/collector.yaml" in args, (
        f"collector must load the mounted config explicitly: {args}"
    )


def test_default_render_collector_configmap_has_awsemf_and_awsxray() -> None:
    rendered = _helm_template()
    cm = _by_kind_name(rendered, "ConfigMap", "-adot-collector")
    assert cm is not None, "default render must emit an ADOT collector ConfigMap"

    # The pipeline config is itself YAML embedded under data.collector.yaml
    # (keyed `collector.yaml`, not `config.yaml`, to avoid colliding with the
    # gateway ConfigMap's LiteLLM config.yaml).
    pipeline = yaml.safe_load(cm["data"]["collector.yaml"])
    exporters = pipeline["exporters"]
    assert "awsemf" in exporters, "metrics->CloudWatch EMF exporter (awsemf) missing"
    assert "awsxray" in exporters, "traces->X-Ray exporter (awsxray) missing"

    # OTLP receiver is what the gateway dials.
    assert "otlp" in pipeline["receivers"], "collector must receive OTLP"

    # The exporters are actually wired into the service pipelines (not dangling).
    pipelines = pipeline["service"]["pipelines"]
    assert "awsxray" in pipelines["traces"]["exporters"]
    assert "awsemf" in pipelines["metrics"]["exporters"]


def test_default_render_collector_service_exposes_otlp_ports() -> None:
    rendered = _helm_template()
    svc = _by_kind_name(rendered, "Service", "-adot-collector")
    assert svc is not None, "default render must emit an ADOT collector Service"
    ports = {p["port"] for p in svc["spec"]["ports"]}
    assert 4317 in ports, "OTLP gRPC port 4317 missing on collector Service"
    assert 4318 in ports, "OTLP HTTP port 4318 missing on collector Service"


def test_default_render_gateway_points_otlp_at_collector() -> None:
    rendered = _helm_template()
    env = _main_container_env(rendered)

    # P1: OTLP is the default exporter for all three signals.
    assert _env_value(env, "OTEL_TRACES_EXPORTER") == "otlp"
    assert _env_value(env, "OTEL_METRICS_EXPORTER") == "otlp"
    assert _env_value(env, "OTEL_LOGS_EXPORTER") == "otlp"

    # And the endpoint is auto-targeted at the in-cluster collector Service DNS.
    endpoint = _env_value(env, "OTEL_EXPORTER_OTLP_ENDPOINT")
    assert endpoint is not None, "gateway must export OTLP to the bundled collector"
    assert endpoint.endswith(":4317"), f"expected gRPC :4317 endpoint, got {endpoint}"
    assert "adot-collector" in endpoint, (
        f"endpoint must target the bundled collector Service: {endpoint}"
    )
    assert "svc.cluster.local" in endpoint, (
        f"endpoint must be an in-cluster Service DNS: {endpoint}"
    )

    # Exactly ONE endpoint env entry (no duplicate from the externalOtel block).
    n = sum(1 for e in env if e.get("name") == "OTEL_EXPORTER_OTLP_ENDPOINT")
    assert n == 1, f"expected exactly one OTEL_EXPORTER_OTLP_ENDPOINT, got {n}"


def test_region_threads_into_collector_exporters() -> None:
    # aws.region is LOAD-BEARING for live delivery (awsemf/awsxray need a region).
    rendered = _helm_template("aws.region=us-west-2")
    cm = _by_kind_name(rendered, "ConfigMap", "-adot-collector")
    pipeline = yaml.safe_load(cm["data"]["collector.yaml"])
    assert pipeline["exporters"]["awsemf"]["region"] == "us-west-2"
    assert pipeline["exporters"]["awsxray"]["region"] == "us-west-2"


# ---------------------------------------------------------------------------
# Disable flag: collector + auto-targeted gateway endpoint both vanish
# ---------------------------------------------------------------------------


def test_collector_disabled_removes_collector_and_gateway_endpoint() -> None:
    rendered = _helm_template("gateway.otel.collector.enabled=false")

    # No collector resources of any kind.
    for d in _docs(rendered):
        name = str(d.get("metadata", {}).get("name", ""))
        assert not name.endswith("-adot-collector"), (
            f"collector resource {d.get('kind')}/{name} must be absent when disabled"
        )

    # And the gateway no longer auto-targets the (now-absent) collector: no OTLP
    # endpoint env entry (the operator must supply externalOtel.endpoint instead).
    env = _main_container_env(rendered)
    assert _env_value(env, "OTEL_EXPORTER_OTLP_ENDPOINT") is None, (
        "disabling the collector must drop the auto-targeted OTLP endpoint"
    )


def test_explicit_endpoint_overrides_bundled_collector_target() -> None:
    # An explicit gateway.otel.endpoint takes precedence over the bundled DNS.
    rendered = _helm_template("gateway.otel.endpoint=http://my-col:4317")
    env = _main_container_env(rendered)
    assert _env_value(env, "OTEL_EXPORTER_OTLP_ENDPOINT") == "http://my-col:4317"
    # Still exactly one entry (no duplicate).
    n = sum(1 for e in env if e.get("name") == "OTEL_EXPORTER_OTLP_ENDPOINT")
    assert n == 1, f"expected exactly one OTEL_EXPORTER_OTLP_ENDPOINT, got {n}"


def test_networkpolicy_allows_egress_to_bundled_collector() -> None:
    """RouteIQ-85e3 fix-forward: with NetworkPolicy + the bundled collector both
    on, the gateway's egress MUST permit OTLP to the collector pods, else an
    enabled NetworkPolicy silently drops all telemetry."""
    rendered = _helm_template(
        "networkPolicy.enabled=true", "gateway.otel.collector.enabled=true"
    )
    npol = [
        d
        for d in yaml.safe_load_all(rendered)
        if d and d.get("kind") == "NetworkPolicy"
    ]
    assert len(npol) == 1, "expected exactly one NetworkPolicy"
    egress = npol[0]["spec"]["egress"]
    # An egress rule must target the collector's OTLP ports (4317 gRPC + 4318 HTTP).
    otlp_ports = {
        p["port"]
        for rule in egress
        for p in rule.get("ports", [])
        if p.get("port") in (4317, 4318)
    }
    assert {4317, 4318} <= otlp_ports, (
        f"NetworkPolicy egress must allow OTLP ports 4317+4318 to the collector; got {otlp_ports}"
    )


def test_networkpolicy_omits_collector_egress_when_collector_disabled() -> None:
    """No collector -> no collector-egress rule (no dangling 4317/4318 allowance)."""
    rendered = _helm_template(
        "networkPolicy.enabled=true", "gateway.otel.collector.enabled=false"
    )
    npol = [
        d
        for d in yaml.safe_load_all(rendered)
        if d and d.get("kind") == "NetworkPolicy"
    ]
    assert len(npol) == 1
    egress = npol[0]["spec"]["egress"]
    otlp_ports = {
        p["port"]
        for rule in egress
        for p in rule.get("ports", [])
        if p.get("port") in (4317, 4318)
    }
    assert not otlp_ports, (
        f"collector disabled must omit the OTLP egress rule; got ports {otlp_ports}"
    )
