"""Helm-template render assertions for the GitOps Application manifest (RouteIQ-eb77).

The chart can render an ArgoCD Application (or Flux Kustomization) so a GitOps
controller reconciles the chart from a git repo as the source of truth. DEFAULT
OFF (gitops.enabled=false) -> the manifest is NOT rendered and the default
render stays byte-stable.

These tests shell out to ``helm template`` (no cluster, no AWS) and assert:
  * default render emits NO Application / Kustomization,
  * gitops.enabled=true + provider=argocd renders an argoproj.io Application,
  * provider=flux renders a Flux Kustomization,
  * the source (repoURL/path/targetRevision) + destination are wired,
  * automated sync is omitted unless syncPolicy.automated=true.

Auto-skips when helm is not on PATH (matches the chart test ethos).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

CHART_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm not on PATH; render assertions auto-skip",
)


def _helm_template(*set_args: str) -> str:
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


def _kinds(rendered: str) -> list[str]:
    return [d.get("kind") for d in _docs(rendered)]


def test_gitops_off_by_default_renders_no_application() -> None:
    rendered = _helm_template()
    kinds = _kinds(rendered)
    assert "Application" not in kinds
    assert "Kustomization" not in kinds


def test_argocd_application_rendered_when_enabled() -> None:
    rendered = _helm_template(
        "gitops.enabled=true",
        "gitops.provider=argocd",
        "gitops.repoURL=https://example.com/repo",
        "gitops.targetRevision=v1.2.3",
    )
    app = next(d for d in _docs(rendered) if d.get("kind") == "Application")
    assert app["apiVersion"] == "argoproj.io/v1alpha1"
    src = app["spec"]["source"]
    assert src["repoURL"] == "https://example.com/repo"
    assert src["targetRevision"] == "v1.2.3"
    assert src["path"] == "deploy/charts/routeiq-gateway"
    assert app["spec"]["destination"]["namespace"] == "routeiq"


def test_argocd_automated_sync_omitted_by_default() -> None:
    rendered = _helm_template("gitops.enabled=true")
    app = next(d for d in _docs(rendered) if d.get("kind") == "Application")
    # syncPolicy.automated defaults false -> no automated block
    assert "automated" not in app["spec"].get("syncPolicy", {})


def test_argocd_automated_sync_when_enabled() -> None:
    rendered = _helm_template("gitops.enabled=true", "gitops.syncPolicy.automated=true")
    app = next(d for d in _docs(rendered) if d.get("kind") == "Application")
    assert app["spec"]["syncPolicy"]["automated"]["prune"] is True


def test_flux_kustomization_rendered() -> None:
    rendered = _helm_template("gitops.enabled=true", "gitops.provider=flux")
    kust = next(d for d in _docs(rendered) if d.get("kind") == "Kustomization")
    assert kust["apiVersion"] == "kustomize.toolkit.fluxcd.io/v1"
    assert kust["spec"]["sourceRef"]["kind"] == "GitRepository"
    assert kust["spec"]["interval"] == "5m"
    assert kust["spec"]["targetNamespace"] == "routeiq"
