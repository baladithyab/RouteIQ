"""Helm-template render assertions for C3b self-hosted engine + KVBM (the
cred-free half of RouteIQ-91fe + RouteIQ-28be).

A self-hosted inference engine (vLLM Production Stack / AIBrix / a Dynamo
serving group) is registered as ONE LiteLLM deployment whose ``api_base`` is the
engine's in-cluster FRONTEND/gateway Service -- NOT individual workers/replicas.
The engine's own KV-aware router (Production Stack router / AIBrix Router /
Dynamo Smart Router + FlashIndexer) and -- for Dynamo -- the KVBM cache tiers
(G1 HBM -> G2 DRAM -> G3 NVMe -> G4 S3) are a **Layer-2-and-below** concern that
lives entirely BELOW that one ``api_base``, opaque to RouteIQ. RouteIQ does
Layer-1 model SELECTION only.

These tests prove the cred-free **seam**, not the live deploy (which needs GPU
hardware + the Dynamo operator + a real S3 bucket -- the operator-gated half):

  * a self-hosted ``hosted_vllm`` row with ``api_base`` + ``fake-api-key`` passes
    through the chart values -> gateway ConfigMap ``config.yaml`` UNCHANGED;
  * the LAYERING INVARIANT holds even with a KVBM-enabled Dynamo frontend
    registered: exactly ONE ``model_list`` row points INTO any given engine
    frontend ``api_base`` (never a per-worker/per-replica row) -- the one config
    error to avoid (``51-...`` §3.2);
  * the chart's DEFAULT render ships ZERO self-hosted engine arms (byte-stable
    when off; C3b is operator-supplied config, not a new chart default).

They shell out to ``helm template`` -- no cluster, no AWS creds, CI-cheap -- and
auto-skip if ``helm`` is not on PATH (mirrors the integration-skip ethos and the
sibling ``test_render*.py`` modules).

Authorities (not re-adjudicated here):
``docs/architecture/aws-rearchitecture/50-litellm-universal-surface.md`` §4 and
``docs/architecture/aws-rearchitecture/51-multinode-large-model-serving.md``
Part 2/3, plus the implementation note ``52-self-hosted-engine-kvbm.md``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

# deploy/charts/routeiq-gateway/tests/test_render_self_hosted_engine.py
#   -> deploy/charts/routeiq-gateway
CHART_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm not on PATH; render assertions auto-skip (matches integration-skip ethos)",
)


def _helm_template(values: dict | None = None) -> str:
    """Render the chart; return raw YAML.

    ``values`` (when given) is written to a temp values file passed via ``-f`` --
    the only clean way to inject the multi-line ``config.gateway`` engine YAML
    (``--set`` cannot express a multi-line block scalar). Mirrors the helper in
    ``test_render_capacity_pools.py``.
    """
    cmd = ["helm", "template", "rq", str(CHART_DIR)]
    tmp_path: str | None = None
    try:
        if values is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
                yaml.safe_dump(values, fh)
                tmp_path = fh.name
            cmd += ["-f", tmp_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise AssertionError(
                f"helm template failed (rc={proc.returncode}):\n"
                f"argv={cmd}\nstderr={proc.stderr}"
            )
        return proc.stdout
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def _docs(rendered: str) -> Iterator[dict]:
    for doc in yaml.safe_load_all(rendered):
        if doc:
            yield doc


def _configmap_gateway_config(rendered: str) -> dict:
    """Return the parsed ``config.yaml`` from the gateway ConfigMap.

    Proves the rendered string is valid YAML AND round-trips to the same
    structure the operator supplied (helm did not mangle the engine row).
    """
    for d in _docs(rendered):
        if d.get("kind") == "ConfigMap" and "config.yaml" in d.get("data", {}):
            return yaml.safe_load(d["data"]["config.yaml"])
    raise AssertionError("no ConfigMap with config.yaml in render")


def _rows_for(config: dict, model_name: str) -> list[dict]:
    return [
        row
        for row in config.get("model_list", [])
        if row.get("model_name") == model_name
    ]


# A single self-hosted engine arm registered as ONE api_base (C3b deliverable).
# The api_base is the engine FRONTEND Service; KVBM/KV-aware routing is BELOW it.
SELF_HOSTED_ENGINE = """\
model_list:
  - model_name: oss-70b
    litellm_params:
      model: hosted_vllm/meta-llama/Llama-3.1-70B-Instruct
      api_base: http://aibrix-gateway.aibrix-system.svc.cluster.local:8000/v1
      api_key: fake-api-key

general_settings:
  master_key: env/LITELLM_MASTER_KEY
"""


def _render_engine() -> dict:
    rendered = _helm_template(values={"config": {"gateway": SELF_HOSTED_ENGINE}})
    return _configmap_gateway_config(rendered)


# ---------------------------------------------------------------------------
# RouteIQ-91fe -- the self-hosted engine row passes through the seam unchanged
# ---------------------------------------------------------------------------


def test_chart_carries_self_hosted_engine_row_unchanged() -> None:
    config = _render_engine()
    rows = _rows_for(config, "oss-70b")

    # Exactly ONE row -- the engine frontend is ONE LiteLLM deployment.
    assert len(rows) == 1, f"expected ONE self-hosted engine arm, got {len(rows)}"

    params = rows[0]["litellm_params"]
    # The three load-bearing fields survive verbatim through the chart seam.
    assert params["model"] == "hosted_vllm/meta-llama/Llama-3.1-70B-Instruct"
    assert (
        params["api_base"]
        == "http://aibrix-gateway.aibrix-system.svc.cluster.local:8000/v1"
    )
    # hosted_vllm needs a non-empty api_key sentinel even for an unauthenticated
    # in-cluster engine.
    assert params["api_key"] == "fake-api-key"


def test_self_hosted_provider_is_hosted_vllm_not_a_replica_scheme() -> None:
    # The provider prefix must be hosted_vllm (or openai_like) -- an OpenAI-
    # compatible /v1 consumed directly via api_base, no translation shim.
    config = _render_engine()
    model = _rows_for(config, "oss-70b")[0]["litellm_params"]["model"]
    assert model.startswith("hosted_vllm/"), model
    # api_base targets a Service DNS, not a pod IP (a replica). Cheap heuristic:
    # the in-cluster Service hostname, never a bare 10.x / pod-ip endpoint.
    api_base = _rows_for(config, "oss-70b")[0]["litellm_params"]["api_base"]
    assert "svc.cluster.local" in api_base, (
        f"api_base must be the engine FRONTEND Service, not a replica: {api_base}"
    )


# ---------------------------------------------------------------------------
# RouteIQ-28be -- the layering invariant: ONE api_base per (KVBM-enabled) engine
# ---------------------------------------------------------------------------

# A KVBM-enabled Dynamo serving group. KVBM (G1->G2->G3->G4) is opaque BELOW the
# ONE frontend api_base -- so RouteIQ config is UNCHANGED vs a plain engine. The
# invariant under test: exactly ONE model_list row points INTO the Dynamo
# frontend (never the prefill/decode workers as separate rows -- the one error).
KVBM_DYNAMO_FRONTEND = """\
model_list:
  # ONE row -> the Dynamo FRONTEND. KVBM tiers + disagg prefill/decode are BELOW
  # this api_base, invisible to RouteIQ. A legitimate SECOND capacity arm (a
  # Bedrock source under the same alias) is fine -- it is a different SOURCE, not
  # a replica inside the Dynamo group.
  - model_name: big-reasoning-405b
    litellm_params:
      model: hosted_vllm/big-reasoning-405b
      api_base: http://dynamo-frontend.dynamo-system.svc.cluster.local:8000/v1
      api_key: fake-api-key
  - model_name: big-reasoning-405b
    litellm_params:
      model: bedrock/meta.llama3-1-405b-instruct-v1:0
      aws_region_name: us-west-2

general_settings:
  master_key: env/LITELLM_MASTER_KEY
"""


def _engine_frontend_rows(config: dict) -> list[dict]:
    """Rows whose api_base points INTO a self-hosted engine frontend Service.

    The layering invariant counts rows targeting an in-cluster engine api_base;
    a Bedrock arm (no api_base) is a separate CAPACITY SOURCE, not part of the
    engine group, so it is excluded from the per-frontend count.
    """
    return [
        row
        for row in config.get("model_list", [])
        if "svc.cluster.local" in row.get("litellm_params", {}).get("api_base", "")
    ]


def test_layering_invariant_one_api_base_per_dynamo_frontend() -> None:
    rendered = _helm_template(values={"config": {"gateway": KVBM_DYNAMO_FRONTEND}})
    config = _configmap_gateway_config(rendered)

    frontend_rows = _engine_frontend_rows(config)
    # Exactly ONE row targets the engine frontend -- the KVBM/Dynamo group is ONE
    # api_base. Registering individual workers/replicas would show up as >1 here.
    assert len(frontend_rows) == 1, (
        "LAYERING INVARIANT VIOLATED: more than one model_list row points INTO an "
        f"engine frontend Service (register the frontend as ONE deployment): "
        f"{frontend_rows}"
    )

    # Each distinct in-cluster engine api_base appears at most once across the
    # whole model_list (no replica fan-out under any alias).
    api_bases = [r["litellm_params"]["api_base"] for r in frontend_rows]
    assert len(api_bases) == len(set(api_bases)), (
        f"an engine api_base is registered more than once (replica fan-out): {api_bases}"
    )

    # The sibling Bedrock arm IS allowed (separate capacity source) -- the alias
    # still has 2 rows total, but only the ONE engine-frontend row is the engine.
    assert len(_rows_for(config, "big-reasoning-405b")) == 2


def test_shipped_self_hosted_engine_example_config_is_well_formed() -> None:
    # The shipped config example (config/config.self-hosted-engine.yaml) must
    # parse and obey the SAME layering invariant: ONE engine-frontend api_base.
    # config/ is repo-root/config -> 4 parents up from this test file.
    example = CHART_DIR.parents[2] / "config" / "config.self-hosted-engine.yaml"
    assert example.is_file(), f"missing shipped example config: {example}"
    config = yaml.safe_load(example.read_text())

    frontend_rows = _engine_frontend_rows(config)
    assert len(frontend_rows) == 1, (
        f"example must register exactly ONE engine frontend api_base: {frontend_rows}"
    )
    params = frontend_rows[0]["litellm_params"]
    assert params["model"].startswith(("hosted_vllm/", "openai_like/"))
    assert params["api_key"] == "fake-api-key"
    assert config["general_settings"]["master_key"]  # load-bearing key present


# ---------------------------------------------------------------------------
# Byte-stable default: the chart DEFAULT ships ZERO self-hosted engine arms.
# C3b is operator-supplied config, not a new chart default.
# ---------------------------------------------------------------------------


def test_default_render_has_no_self_hosted_engine_arm() -> None:
    config = _configmap_gateway_config(_helm_template())
    # No hosted_vllm/openai_like row and no in-cluster api_base by default.
    for row in config.get("model_list", []):
        params = row.get("litellm_params", {})
        assert "svc.cluster.local" not in params.get("api_base", ""), (
            f"default config leaked a self-hosted engine arm: {row}"
        )
        assert not params.get("model", "").startswith(
            ("hosted_vllm/", "openai_like/")
        ), f"default config leaked a self-hosted engine arm: {row}"
