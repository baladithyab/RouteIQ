"""Helm-template render assertions for the RouteIQ Gateway chart (P4 C4).

These tests shell out to ``helm template`` with ``--set`` overrides and assert on
the rendered Kubernetes YAML -- no cluster, no AWS creds, CI-cheap. They guard the
P4 OIDC + ESO + ALB/ACM ingress wiring (and the byte-stable default render that
keeps the public edge inert).

If ``helm`` is not on PATH the whole module is skipped (mirrors the
integration-test auto-skip ethos -- the unit gate must not hard-depend on helm).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

# deploy/charts/routeiq-gateway/tests/test_render.py -> deploy/charts/routeiq-gateway
CHART_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm not on PATH; render assertions auto-skip (matches integration-skip ethos)",
)


def _helm_template(*set_args: str, show_only: str | None = None) -> str:
    """Render the chart with the given ``--set k=v`` overrides; return raw YAML.

    Uses a list argv (no shell) so ``--set foo[0].bar=baz`` index syntax is passed
    verbatim and never glob-expanded by a shell.
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


def _deployment(rendered: str) -> dict:
    for d in _docs(rendered):
        if d.get("kind") == "Deployment":
            return d
    raise AssertionError("no Deployment in render")


def _main_container_env(rendered: str) -> list[dict]:
    """Return the env list of the primary (non-init) gateway container."""
    spec = _deployment(rendered)["spec"]["template"]["spec"]
    containers = spec["containers"]
    # the gateway is the first (and only) primary container
    return containers[0].get("env", [])


def _env_value(env: list[dict], name: str) -> str | None:
    for e in env:
        if e.get("name") == name:
            return e.get("value")
    return None


def _env_secret_ref(env: list[dict], name: str) -> dict | None:
    for e in env:
        if e.get("name") == name and "valueFrom" in e:
            return e["valueFrom"].get("secretKeyRef")
    return None


# ---------------------------------------------------------------------------
# OIDC env wiring (oidc.enabled -> ROUTEIQ_OIDC_* + client-secret secretKeyRef)
# ---------------------------------------------------------------------------


def test_oidc_enabled_emits_env_and_client_secret_ref() -> None:
    rendered = _helm_template(
        "oidc.enabled=true",
        "oidc.issuerUrl=https://idp.example/",
        "oidc.clientId=cid",
        "oidc.existingSecret=routeiq-gateway-secrets",
    )
    env = _main_container_env(rendered)

    assert _env_value(env, "ROUTEIQ_OIDC_ENABLED") == "true"
    assert _env_value(env, "ROUTEIQ_OIDC_ISSUER_URL") == "https://idp.example/"
    assert _env_value(env, "ROUTEIQ_OIDC_CLIENT_ID") == "cid"

    ref = _env_secret_ref(env, "ROUTEIQ_OIDC_CLIENT_SECRET")
    assert ref is not None, "ROUTEIQ_OIDC_CLIENT_SECRET secretKeyRef missing"
    # The secretKeyRef must point at the ESO target secret + the synced key.
    assert ref["name"] == "routeiq-gateway-secrets"
    assert ref["key"] == "oidc-client-secret"


def test_oidc_enabled_without_existing_secret_omits_client_secret() -> None:
    # Public-client mode: no existingSecret -> no client-secret env (degrade path).
    rendered = _helm_template(
        "oidc.enabled=true",
        "oidc.issuerUrl=https://idp.example/",
        "oidc.clientId=cid",
    )
    env = _main_container_env(rendered)
    assert _env_value(env, "ROUTEIQ_OIDC_ENABLED") == "true"
    assert _env_secret_ref(env, "ROUTEIQ_OIDC_CLIENT_SECRET") is None


def test_oidc_existing_secret_key_override_is_honored() -> None:
    # The secretKey<->existingSecretKey alignment is load-bearing; prove the
    # override threads through to the rendered secretKeyRef.
    rendered = _helm_template(
        "oidc.enabled=true",
        "oidc.issuerUrl=https://idp.example/",
        "oidc.clientId=cid",
        "oidc.existingSecret=my-secrets",
        "oidc.existingSecretKey=custom-oidc-key",
    )
    ref = _env_secret_ref(_main_container_env(rendered), "ROUTEIQ_OIDC_CLIENT_SECRET")
    assert ref == {"name": "my-secrets", "key": "custom-oidc-key"}


# ---------------------------------------------------------------------------
# ESO ExternalSecret for the OIDC client secret (Secrets Manager via ESO)
# ---------------------------------------------------------------------------


def test_external_secret_oidc_client_secret_entry() -> None:
    rendered = _helm_template(
        "externalSecrets.enabled=true",
        "externalSecrets.data[0].secretKey=oidc-client-secret",
        "externalSecrets.data[0].remoteRef.key=routeiq/oidc-client-secret",
        show_only="templates/externalsecret.yaml",
    )
    docs = list(_docs(rendered))
    assert len(docs) == 1
    es = docs[0]
    assert es["kind"] == "ExternalSecret"
    assert es["apiVersion"] == "external-secrets.io/v1beta1"

    # ClusterSecretStore name is load-bearing + string-matched.
    assert es["spec"]["secretStoreRef"]["name"] == "aws-secrets-manager"
    assert es["spec"]["secretStoreRef"]["kind"] == "ClusterSecretStore"
    assert es["spec"]["target"]["creationPolicy"] == "Owner"

    data = es["spec"]["data"]
    assert any(
        d["secretKey"] == "oidc-client-secret"
        and d["remoteRef"]["key"] == "routeiq/oidc-client-secret"
        for d in data
    ), f"oidc-client-secret entry not found in {data}"


# ---------------------------------------------------------------------------
# ALB/ACM ingress (ingress.enabled + className=alb + ACM annotations)
# ---------------------------------------------------------------------------


def test_ingress_alb_scheme_and_classname() -> None:
    rendered = _helm_template(
        "ingress.enabled=true",
        "ingress.className=alb",
        r"ingress.annotations.alb\.ingress\.kubernetes\.io/scheme=internet-facing",
        show_only="templates/ingress.yaml",
    )
    docs = list(_docs(rendered))
    assert len(docs) == 1
    ing = docs[0]
    assert ing["kind"] == "Ingress"
    assert ing["spec"]["ingressClassName"] == "alb"
    annotations = ing["metadata"].get("annotations", {})
    assert annotations.get("alb.ingress.kubernetes.io/scheme") == "internet-facing", (
        f"scheme annotation missing: {annotations}"
    )


def test_ingress_acm_certificate_arn_annotation_threads_through() -> None:
    # ACM cert ARN is operator-supplied (account-agnostic placeholder here); prove
    # the chart passes an ALB certificate-arn annotation through verbatim.
    cert_arn = "arn:aws:acm:us-west-2:000000000000:certificate/abc-123"
    rendered = _helm_template(
        "ingress.enabled=true",
        "ingress.className=alb",
        rf"ingress.annotations.alb\.ingress\.kubernetes\.io/certificate-arn={cert_arn}",
        show_only="templates/ingress.yaml",
    )
    ing = next(_docs(rendered))
    annotations = ing["metadata"].get("annotations", {})
    assert annotations.get("alb.ingress.kubernetes.io/certificate-arn") == cert_arn, (
        f"certificate-arn annotation missing/wrong: {annotations}"
    )


# ---------------------------------------------------------------------------
# Byte-stable default render: the public edge + OIDC + ESO ship INERT
# ---------------------------------------------------------------------------


def test_default_render_has_no_ingress_externalsecret_or_oidc_env() -> None:
    rendered = _helm_template()
    kinds = _kinds(rendered)

    # Public edge inert: no Ingress object by default.
    assert "Ingress" not in kinds, "default render must NOT emit an Ingress"
    # ESO inert: no ExternalSecret by default.
    assert "ExternalSecret" not in kinds, (
        "default render must NOT emit an ExternalSecret"
    )

    # OIDC inert: no ROUTEIQ_OIDC_* env by default.
    env = _main_container_env(rendered)
    oidc_env = [e["name"] for e in env if e.get("name", "").startswith("ROUTEIQ_OIDC")]
    assert oidc_env == [], f"default render leaked OIDC env: {oidc_env}"
