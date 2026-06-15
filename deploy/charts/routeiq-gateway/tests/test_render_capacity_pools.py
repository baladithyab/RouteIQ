"""Helm-template render assertions for C0 capacity pools (RouteIQ-e677).

C0 is the lowest-effort capacity-federation phase from
``docs/architecture/aws-rearchitecture/50-litellm-universal-surface.md`` (§2a +
§2b, the C0 row of the §5 C0-C3 roadmap). Multi-key and cross-region capacity
pools are a **native LiteLLM** primitive: N ``model_list`` rows that share one
``model_name`` are N independently cool-downable arms (each gets a distinct
``model_info.id`` = sha256 over ``model_group`` + every ``litellm_param`` ->
``reference/litellm/litellm/router.py:5592-5621``). Stacking rows that differ
only in ``api_key`` (a key pool, §2a) or ``aws_region_name`` (cross-region, §2b)
multiplies quota with automatic 429-driven failover -- with ZERO new routing
code.

So C0 ships DOCS + CONFIG, not code. The chart already accepts an arbitrary
``config.gateway`` YAML string that lands verbatim in the gateway ConfigMap
(templates/configmap.yaml -> ``.Values.config.gateway | nindent``); LiteLLM
builds its ``model_list`` from it at startup and RouteIQ's
``CustomRoutingStrategyBase._get_model_list`` groups candidates purely by
``model_name`` (``src/litellm_llmrouter/custom_routing_strategy.py`` -- a
``model_name`` group is the candidate set; nothing dedups within a group).

These tests therefore PROVE THE SEAM, not new behaviour: that the chart accepts
and renders N ``model_list`` rows under one ``model_name`` (key pool +
cross-region + heterogeneous group) without collapsing/deduping them, and that
the rendered ConfigMap ``config.yaml`` parses back to exactly those N arms. They
shell out to ``helm template`` -- no cluster, no AWS creds, CI-cheap -- and
auto-skip if ``helm`` is not on PATH (mirrors the integration-skip ethos).

Self-contained (own helm-render helpers via a temp values file, since the C0
pool config is multi-line YAML that ``--set`` cannot express cleanly) so it does
not collide with the sibling ``test_render.py`` / ``test_render_hardening.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

# deploy/charts/routeiq-gateway/tests/test_render_capacity_pools.py
#   -> deploy/charts/routeiq-gateway
CHART_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm not on PATH; render assertions auto-skip (matches integration-skip ethos)",
)


def _helm_template(
    *set_args: str,
    values: dict | None = None,
    show_only: str | None = None,
) -> str:
    """Render the chart; return raw YAML.

    ``values`` (when given) is written to a temp values file passed via ``-f`` --
    the only clean way to inject the multi-line ``config.gateway`` pool YAML
    (``--set`` cannot express a multi-line block scalar). ``--set`` overrides are
    still supported and apply AFTER the values file.
    """
    cmd = ["helm", "template", "rq", str(CHART_DIR)]
    tmp_path: str | None = None
    try:
        if values is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
                yaml.safe_dump(values, fh)
                tmp_path = fh.name
            cmd += ["-f", tmp_path]
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
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def _docs(rendered: str) -> Iterator[dict]:
    for doc in yaml.safe_load_all(rendered):
        if doc:
            yield doc


def _configmap_gateway_config(rendered: str) -> dict:
    """Return the parsed ``config.yaml`` from the gateway ConfigMap.

    Proves the rendered string is valid YAML AND round-trips to the same N-arm
    structure the operator supplied (helm did not collapse the duplicate
    ``model_name`` keys -- they are LIST ITEMS, not map keys, so they survive).
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


# A multi-key pool (§2a) + cross-region Bedrock (§2b) under ONE alias. This is the
# C0 deliverable config shape: the operator stacks rows; the only varied field is
# api_key (key pool) or aws_region_name (cross-region).
C0_POOL_CONFIG = """\
model_list:
  # --- gpt-4o: a two-key API capacity pool (§2a) -> 2x rpm, 429 failover ---
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_KEY_A
      rpm: 10000
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_KEY_B
      rpm: 10000

  # --- claude-sonnet: cross-region Bedrock pool (§2b) -> per-region throttle ---
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-west-2

litellm_settings:
  drop_params: true

general_settings:
  master_key: env/LITELLM_MASTER_KEY
"""


def _render_with_pool() -> dict:
    rendered = _helm_template(values={"config": {"gateway": C0_POOL_CONFIG}})
    return _configmap_gateway_config(rendered)


# ---------------------------------------------------------------------------
# §2a -- multi-key API capacity pool: N rows, one model_name, distinct api_keys
# ---------------------------------------------------------------------------


def test_chart_accepts_two_key_pool_under_one_model_name() -> None:
    config = _render_with_pool()
    rows = _rows_for(config, "gpt-4o")

    # The two rows survive as TWO list items (helm did not dedup the repeated
    # model_name -- it is a list, not a map). This is the C0 seam.
    assert len(rows) == 2, f"expected 2 gpt-4o pool arms, got {len(rows)}: {rows}"

    # They differ ONLY in api_key -> distinct LiteLLM model_info.id -> two arms.
    keys = sorted(r["litellm_params"]["api_key"] for r in rows)
    assert keys == ["os.environ/OPENAI_KEY_A", "os.environ/OPENAI_KEY_B"]

    # Same physical model + per-arm rpm preserved (per-key rpm semaphores sum
    # into the pool -- §2a).
    assert {r["litellm_params"]["model"] for r in rows} == {"openai/gpt-4o"}
    assert all(r["litellm_params"]["rpm"] == 10000 for r in rows)


# ---------------------------------------------------------------------------
# §2b -- cross-region Bedrock pool: N rows, one model_name, distinct regions
# ---------------------------------------------------------------------------


def test_chart_accepts_cross_region_bedrock_pool() -> None:
    config = _render_with_pool()
    rows = _rows_for(config, "claude-sonnet")

    assert len(rows) == 2, f"expected 2 claude-sonnet region arms, got {len(rows)}"

    # Differ ONLY in aws_region_name -> distinct arms, each region independently
    # cool-downable (a regional throttle cools just that region -- §2b).
    regions = sorted(r["litellm_params"]["aws_region_name"] for r in rows)
    assert regions == ["us-east-1", "us-west-2"]
    assert {r["litellm_params"]["model"] for r in rows} == {
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    }


def test_pool_rows_are_not_deduped_or_reordered() -> None:
    # Whole-group invariant: 4 rows in, 4 rows out, model_names preserved. Proves
    # the chart is a pure passthrough for the model_list (no helm-side collapse).
    config = _render_with_pool()
    names = [r["model_name"] for r in config["model_list"]]
    assert names == ["gpt-4o", "gpt-4o", "claude-sonnet", "claude-sonnet"]


def test_pool_render_round_trips_to_valid_litellm_config() -> None:
    # The rendered config.yaml must be valid YAML with the load-bearing top-level
    # keys LiteLLM consumes at startup (model_list + general_settings.master_key).
    config = _render_with_pool()
    assert isinstance(config.get("model_list"), list)
    assert config["general_settings"]["master_key"] == "env/LITELLM_MASTER_KEY"
    # Every arm carries the two load-bearing keys (model_name + litellm_params.model).
    for row in config["model_list"]:
        assert "model_name" in row
        assert "model" in row["litellm_params"]


# ---------------------------------------------------------------------------
# §1 sketch -- a heterogeneous group (key + x-region + x-account + self-hosted)
# all under ONE alias still renders as N arms (the universal-surface example).
# ---------------------------------------------------------------------------

HETEROGENEOUS_GROUP = """\
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_KEY_A
      rpm: 10000
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-west-2
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1
      aws_role_name: arn:aws:iam::222222222222:role/RouteIqBedrockCapacity-prod
      aws_session_name: routeiq
  - model_name: claude-sonnet
    litellm_params:
      model: hosted_vllm/anthropic/claude-equivalent-oss-70b
      api_base: http://aibrix-gateway.aibrix-system.svc.cluster.local:8000/v1
      api_key: fake-api-key

general_settings:
  master_key: env/LITELLM_MASTER_KEY
"""


def test_chart_accepts_heterogeneous_four_arm_group() -> None:
    rendered = _helm_template(values={"config": {"gateway": HETEROGENEOUS_GROUP}})
    config = _configmap_gateway_config(rendered)
    rows = _rows_for(config, "claude-sonnet")

    # Four heterogeneous backends under one alias all survive as four arms.
    assert len(rows) == 4, f"expected 4 heterogeneous arms, got {len(rows)}"

    # The four varied fields that make them distinct LiteLLM arms are all present
    # somewhere in the group (api_key / aws_region_name / aws_role_name / api_base).
    params = [r["litellm_params"] for r in rows]
    assert any("api_key" in p and "api_base" not in p for p in params)  # key pool arm
    assert any(
        p.get("aws_region_name") == "us-west-2" and "aws_role_name" not in p
        for p in params
    )  # cross-region arm
    assert any("aws_role_name" in p for p in params)  # cross-account arm (C1)
    assert any("api_base" in p for p in params)  # self-hosted arm (C3)


# ---------------------------------------------------------------------------
# Byte-stable default: the chart's DEFAULT config is unchanged (no pool); C0 is
# operator-supplied config, not a new chart default.
# ---------------------------------------------------------------------------


def test_default_config_has_no_capacity_pool() -> None:
    config = _configmap_gateway_config(_helm_template())
    names = [r["model_name"] for r in config["model_list"]]
    # Default ships exactly the two single-arm aliases; no model_name repeats.
    assert names == ["gpt-4o", "claude-3-5-sonnet"]
    assert len(names) == len(set(names)), "default config must not pool (no dup names)"
