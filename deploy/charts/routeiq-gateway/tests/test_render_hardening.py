"""Helm-template render assertions for the RouteIQ Gateway hardening (P4 C5/C6).

These tests shell out to ``helm template`` with ``--set`` overrides and assert on
the rendered Kubernetes YAML -- no cluster, no AWS creds, CI-cheap. They guard the
P4 hardening work:

  * C5 -- NetworkPolicy in-VPC egress (``networkPolicy.egress.inVpc``): the
    correctness landmine where enabling the shipped NetworkPolicy on the AWS
    substrate is an outage (the ``allowHttpsExternal`` RFC-1918 carve-out blocks
    Aurora 5432 / ElastiCache 6379 / in-VPC Bedrock 443, all inside the CDK VPC
    CIDR 10.40.0.0/16). The inVpc block re-opens those in-VPC targets.

  * C6 -- readOnlyRootFilesystem cache redirect: HF/XDG/matplotlib caches are
    redirected into the writable ``/app/data`` emptyDir so transformers-based
    routing strategies do not EROFS under read-only root.

If ``helm`` is not on PATH the whole module is skipped (mirrors the
integration-test auto-skip ethos -- the unit gate must not hard-depend on helm).

This module is intentionally self-contained (its own helm-render helpers) so it
does not import from the sibling ``test_render.py`` (C4 OIDC/ESO/ingress) and the
two modules cannot collide.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

# deploy/charts/routeiq-gateway/tests/test_render_hardening.py
#   -> deploy/charts/routeiq-gateway
CHART_DIR = Path(__file__).resolve().parents[1]

# The CDK VPC CIDR default (network_construct.py:53). The chart default for
# networkPolicy.egress.inVpc.cidr MUST track this.
VPC_CIDR = "10.40.0.0/16"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm not on PATH; render assertions auto-skip (matches integration-skip ethos)",
)


def _helm_template(*set_args: str, show_only: str | None = None) -> str:
    """Render the chart with the given ``--set k=v`` overrides; return raw YAML.

    Uses a list argv (no shell) so ``--set foo[0].bar=baz`` index syntax is
    passed verbatim and never glob-expanded by a shell.
    """
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
    """Yield each non-empty YAML document from a multi-doc helm render."""
    for doc in yaml.safe_load_all(rendered):
        if doc:
            yield doc


def _kinds(rendered: str) -> list[str]:
    return [d.get("kind") for d in _docs(rendered)]


def _network_policy(rendered: str) -> dict:
    for d in _docs(rendered):
        if d.get("kind") == "NetworkPolicy":
            return d
    raise AssertionError("no NetworkPolicy in render")


def _egress_rules(rendered: str) -> list[dict]:
    return _network_policy(rendered)["spec"]["egress"]


def _deployment(rendered: str) -> dict:
    # Select the GATEWAY Deployment specifically. RouteIQ-85e3 added a bundled
    # ADOT collector Deployment (<fullname>-adot-collector) that renders first
    # (alphabetical template name), so "first Deployment" would pick the collector.
    for d in _docs(rendered):
        if d.get("kind") == "Deployment" and not str(
            d.get("metadata", {}).get("name", "")
        ).endswith("-adot-collector"):
            return d
    raise AssertionError("no gateway Deployment in render")


def _main_container(rendered: str) -> dict:
    spec = _deployment(rendered)["spec"]["template"]["spec"]
    # the gateway is the first (and only) primary container
    return spec["containers"][0]


def _main_container_env(rendered: str) -> list[dict]:
    return _main_container(rendered).get("env", [])


def _env_value(env: list[dict], name: str) -> str | None:
    for e in env:
        if e.get("name") == name:
            return e.get("value")
    return None


def _ipblock_rules_for_cidr(egress: list[dict], cidr: str) -> list[dict]:
    """Return egress rules whose ``to`` contains an ipBlock with the given cidr.

    Asserts on the parsed YAML object (not raw text) so a comment that merely
    mentions a CIDR string can never produce a false positive.
    """
    matches: list[dict] = []
    for rule in egress:
        for to in rule.get("to", []) or []:
            ipblock = to.get("ipBlock") or {}
            if ipblock.get("cidr") == cidr:
                matches.append(rule)
    return matches


def _ports_of(rule: dict) -> set[int]:
    return {p.get("port") for p in rule.get("ports", []) or []}


# ---------------------------------------------------------------------------
# C5 -- NetworkPolicy in-VPC egress hardening
# ---------------------------------------------------------------------------


def test_networkpolicy_disabled_by_default_no_object() -> None:
    # Default render: networkPolicy.enabled=false -> NO NetworkPolicy object.
    rendered = _helm_template()
    assert "NetworkPolicy" not in _kinds(rendered), (
        "default render must NOT emit a NetworkPolicy"
    )


def test_networkpolicy_enabled_without_invpc_has_the_gap() -> None:
    # Enabling the shipped NetworkPolicy WITHOUT inVpc is the outage state:
    # the only app egress is the allowHttpsExternal 0.0.0.0/0 rule which EXCEPTS
    # the three RFC-1918 ranges -> the in-VPC VPC CIDR is NOT reachable.
    rendered = _helm_template("networkPolicy.enabled=true")
    egress = _egress_rules(rendered)

    public = _ipblock_rules_for_cidr(egress, "0.0.0.0/0")
    assert public, "allowHttpsExternal 0.0.0.0/0 egress rule missing"
    # The public rule carves out every RFC-1918 range (this is what blocks the
    # in-VPC DB/cache/Bedrock on AWS).
    excepts = set()
    for r in public:
        for to in r["to"]:
            excepts.update(to["ipBlock"].get("except", []))
    assert {"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"} <= excepts

    # Gap proof: NO in-VPC CIDR allow exists by default.
    assert _ipblock_rules_for_cidr(egress, VPC_CIDR) == [], (
        "in-VPC egress must be ABSENT when inVpc.enabled is false (proves the gap)"
    )


def test_networkpolicy_invpc_opens_db_cache_bedrock() -> None:
    # The fix: inVpc.enabled=true adds a 10.40.0.0/16 allow for 5432/6379/443.
    rendered = _helm_template(
        "networkPolicy.enabled=true",
        "networkPolicy.egress.inVpc.enabled=true",
    )
    egress = _egress_rules(rendered)

    invpc = _ipblock_rules_for_cidr(egress, VPC_CIDR)
    assert len(invpc) == 1, f"expected exactly one in-VPC egress rule, got {invpc}"
    # Aurora 5432, ElastiCache 6379, interface VPC endpoints (Bedrock/Secrets/
    # ECR/Logs) 443 -- all three in one CIDR rule.
    assert _ports_of(invpc[0]) == {5432, 6379, 443}

    # allowHttpsExternal coexists: S3 + public LLM providers still reachable.
    assert _ipblock_rules_for_cidr(egress, "0.0.0.0/0"), (
        "allowHttpsExternal 0.0.0.0/0 rule must STILL be present alongside inVpc"
    )


def test_networkpolicy_invpc_cidr_override_tracks_vpc_cidr() -> None:
    # The CIDR coupling to the CDK vpc_cidr is operator-overridable; prove the
    # rendered ipBlock tracks an override (so a non-default VPC CIDR is honored).
    rendered = _helm_template(
        "networkPolicy.enabled=true",
        "networkPolicy.egress.inVpc.enabled=true",
        "networkPolicy.egress.inVpc.cidr=10.99.0.0/16",
    )
    egress = _egress_rules(rendered)
    assert _ipblock_rules_for_cidr(egress, "10.99.0.0/16"), (
        "in-VPC ipBlock must track the overridden cidr"
    )
    # The old default CIDR must NOT linger when overridden.
    assert _ipblock_rules_for_cidr(egress, VPC_CIDR) == []


def test_networkpolicy_invpc_ports_override_tracks() -> None:
    # An operator may restrict the in-VPC ports; prove the rendered rule tracks.
    rendered = _helm_template(
        "networkPolicy.enabled=true",
        "networkPolicy.egress.inVpc.enabled=true",
        "networkPolicy.egress.inVpc.ports={5432,6379}",
    )
    invpc = _ipblock_rules_for_cidr(_egress_rules(rendered), VPC_CIDR)
    assert len(invpc) == 1
    assert _ports_of(invpc[0]) == {5432, 6379}


# ---------------------------------------------------------------------------
# C6 -- readOnlyRootFilesystem cache redirect + writable-mount backing
# ---------------------------------------------------------------------------

# Every cache env path must be prefixed by a writable mount so a regression that
# reintroduces an unmounted cache dir (e.g. /app/.cache on read-only root) fails.
WRITABLE_MOUNTS = ("/tmp", "/app/data", "/app/models")

EXPECTED_CACHE_ENV = {
    "HF_HOME": "/app/data/.cache/huggingface",
    "HF_HUB_CACHE": "/app/data/.cache/huggingface/hub",
    "TRANSFORMERS_CACHE": "/app/data/.cache/huggingface",
    "XDG_CACHE_HOME": "/app/data/.cache",
    "MPLCONFIGDIR": "/app/data/.cache/matplotlib",
}


def test_cache_redirect_env_present_by_default() -> None:
    # The cache redirect is emitted unconditionally (harmless when root writable).
    env = _main_container_env(_helm_template())
    for name, value in EXPECTED_CACHE_ENV.items():
        assert _env_value(env, name) == value, (
            f"{name} should redirect to {value}; got {_env_value(env, name)}"
        )


def test_readonly_root_fs_with_writable_mounts() -> None:
    rendered = _helm_template()
    container = _main_container(rendered)

    # readOnlyRootFilesystem is ON by default.
    assert container["securityContext"]["readOnlyRootFilesystem"] is True

    # The three writable emptyDir mounts that back the read-only root are present.
    mount_paths = {m["mountPath"] for m in container.get("volumeMounts", [])}
    for path in WRITABLE_MOUNTS:
        assert path in mount_paths, f"writable mount {path} missing: {mount_paths}"


def test_cache_env_targets_a_writable_mount() -> None:
    # Regression guard: every cache env path must live under a mounted writable
    # path. If someone points a cache back at an unmounted dir (e.g. /app/.cache),
    # it would EROFS under read-only root -- this test fails first.
    env = _main_container_env(_helm_template())
    for name in EXPECTED_CACHE_ENV:
        value = _env_value(env, name)
        assert value is not None, f"{name} not emitted"
        assert any(value == m or value.startswith(m + "/") for m in WRITABLE_MOUNTS), (
            f"{name}={value} is NOT under a writable mount {WRITABLE_MOUNTS}; "
            "would EROFS under readOnlyRootFilesystem"
        )
