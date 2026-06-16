"""Static verification of the Fluent Bit routing-JSON promotion contract.

WHY THIS TEST EXISTS (RouteIQ-7069)
-----------------------------------
The P2 CloudWatch metric filters and the Glue/Athena data lake scan **top-level**
JSON keys on the routing log group: ``$.event``, ``$.latency_ms``,
``$.["gen_ai.response.model"]`` and ``$.level``. But the EKS Container Insights
add-on's Fluent Bit WRAPS every container stdout line as a stringified ``log``
field, so those top-level paths resolve to ``null`` unless the Fluent Bit
pipeline LIFTS the inner router JSON to the record root.

The chart's ``fluent-bit-config.yaml`` is the thing that does that lift. Field
population can only be *fully* proven on a live cluster (the skill
``eks-container-insights-fluentbit-wraps-stdout``), but the promotion **contract**
- which filters, in which order, with which options - is statically verifiable
and is exactly the part a careless config edit silently breaks. This test pins
that contract so such an edit fails CI:

  1. it renders the chart template with ``helm template`` (the real rendered
     config, not a hand-copied fixture);
  2. it parses the embedded ``parsers.conf`` + ``routeiq-routing.conf`` Fluent
     Bit INI sections;
  3. it asserts the parser is ``Format json`` and the filter chain is wired so
     the inner keys are promoted to the TOP LEVEL (kubernetes filter does not
     pre-consume ``log``; the parser filter reads ``log`` and reserves data);
  4. it then SIMULATES the filter chain's documented semantics against a REAL
     ``build_routing_decision_record`` line (and a ``build_error_log_record``
     line) wrapped in the docker/Container-Insights envelope, and asserts the
     load-bearing keys (``event``, ``latency_ms``, ``gen_ai.response.model``,
     ``level``) actually appear at the record top level.

The historical bug this guards against: the kubernetes filter with
``Merge_Log On`` + ``Merge_Log_Key`` NESTS the parsed JSON under a sub-key (not
the root the metric filters need) and ``Keep_Log Off`` DELETES ``log`` before the
downstream parser filter can read it - leaving the promotion a silent no-op.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

from litellm_llmrouter.observability import (
    build_error_log_record,
    build_routing_decision_record,
)

# Repo-root-relative chart path (this file lives at <root>/tests/unit/).
_CHART_DIR = (
    Path(__file__).resolve().parents[2] / "deploy" / "charts" / "routeiq-gateway"
)
_TEMPLATE = "templates/fluent-bit-config.yaml"

# The four top-level JSON keys the P2 CloudWatch metric filters + the data lake
# scan. ``level`` comes from the structured ERROR line; the other three from the
# routing_decision line. If any does not resolve at the record top level after
# the Fluent Bit chain, the metric filter for it stays INSUFFICIENT_DATA forever.
_REQUIRED_TOP_LEVEL_KEYS = ("event", "latency_ms", "gen_ai.response.model", "level")


# --------------------------------------------------------------------------- helm


def _render_fluent_bit_configmap() -> Dict[str, Any]:
    """Render the chart's fluent-bit ConfigMap with promotion enabled.

    Skips (not fails) when ``helm`` is unavailable so the suite still runs in a
    helm-less environment; the parse/simulation assertions below are the
    load-bearing part and they need the real rendered config.
    """
    helm = shutil.which("helm")
    if helm is None:  # pragma: no cover - environment-dependent
        pytest.skip("helm not on PATH; cannot render the chart template")

    proc = subprocess.run(
        [
            helm,
            "template",
            "routeiq",
            str(_CHART_DIR),
            "--set",
            "fluentBit.routingPromotion.enabled=true",
            "--set",
            "fluentBit.routingPromotion.clusterName=routeiq-dev",
            "--set",
            "aws.region=us-west-2",
            "--show-only",
            _TEMPLATE,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"helm template failed:\n{proc.stderr}"

    import yaml

    docs = [d for d in yaml.safe_load_all(proc.stdout) if d]
    assert len(docs) == 1, f"expected one ConfigMap, got {len(docs)}"
    cm = docs[0]
    assert cm["kind"] == "ConfigMap"
    return cm["data"]


# ----------------------------------------------------- tiny Fluent Bit INI parser


def _parse_fluent_bit_sections(text: str) -> List[Dict[str, str]]:
    """Parse Fluent Bit INI-style ``[SECTION]`` blocks into a list of dicts.

    Each dict carries ``__name__`` (the section header, e.g. ``FILTER``) plus its
    ``Key  Value`` options (first whitespace splits key from value). This mirrors
    Fluent Bit's own classic-config tokenisation closely enough to assert wiring.
    """
    sections: List[Dict[str, str]] = []
    current: Dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            if current is not None:
                sections.append(current)
            current = {"__name__": line[1:-1].strip()}
            continue
        if current is None:
            continue
        parts = line.split(None, 1)
        key = parts[0]
        value = parts[1].strip() if len(parts) > 1 else ""
        current[key] = value
    if current is not None:
        sections.append(current)
    return sections


def _section(sections: List[Dict[str, str]], name: str, **match: str) -> Dict[str, str]:
    """Return the single section of ``name`` whose options match ``match``."""
    found = [
        s
        for s in sections
        if s["__name__"] == name and all(s.get(k) == v for k, v in match.items())
    ]
    assert len(found) == 1, (
        f"expected exactly one [{name}] matching {match}, got {len(found)}"
    )
    return found[0]


# -------------------------------------------------- documented filter semantics


def _apply_kubernetes_filter(
    record: Dict[str, Any], cfg: Dict[str, str]
) -> Dict[str, Any]:
    """Simulate the kubernetes [FILTER]'s effect on the wrapped ``log`` field.

    Only the behaviours that bear on top-level promotion are modelled (per the
    Fluent Bit docs):
      * ``Merge_Log On`` JSON-parses ``log`` and merges the result either at the
        top level (no ``Merge_Log_Key``) or NESTED under ``Merge_Log_Key``;
      * ``Keep_Log Off`` deletes the original ``log`` field after merging.
    With ``Merge_Log Off`` + ``Keep_Log On`` the filter leaves ``log`` untouched
    (it only attaches kubernetes metadata, already present in the envelope).
    """
    out = dict(record)
    if cfg.get("Merge_Log", "Off") == "On":
        try:
            parsed = json.loads(out["log"])
        except (KeyError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            merge_key = cfg.get("Merge_Log_Key")
            if merge_key:
                out[merge_key] = parsed  # NESTED - not the top level
            else:
                out.update(parsed)  # top level
        if cfg.get("Keep_Log", "On") == "Off":
            out.pop("log", None)
    return out


def _apply_parser_filter(
    record: Dict[str, Any], cfg: Dict[str, str], parser_format: str
) -> Dict[str, Any]:
    """Simulate the parser [FILTER] for a ``Format json`` parser.

    Per the docs: a JSON parser promotes the decoded object's keys to the record
    TOP LEVEL. ``Reserve_Data On`` keeps the other original fields; ``Preserve_Key
    Off`` drops the ``Key_Name`` field after parsing. If ``Key_Name`` is absent the
    filter is a no-op (record passes through unmodified).
    """
    key_name = cfg["Key_Name"]
    if key_name not in record:
        return dict(record)  # no-op: nothing to parse

    assert parser_format == "json", (
        f"simulator only models Format=json parsers; got {parser_format!r}"
    )
    promoted = json.loads(record[key_name])
    assert isinstance(promoted, dict)

    reserve = cfg.get("Reserve_Data", "Off") == "On"
    out: Dict[str, Any] = dict(record) if reserve else {}
    out.update(promoted)  # JSON parser -> keys at the TOP LEVEL
    if cfg.get("Preserve_Key", "Off") != "On":
        out.pop(key_name, None)
    return out


def _wrap_in_ci_envelope(app_line: str) -> Dict[str, Any]:
    """Wrap an app stdout line as the docker/Container-Insights envelope.

    Matches the shape the skill documents: top-level keys are
    ``{_p, kubernetes, log, stream, time}`` with the app's JSON as a STRING under
    ``log``.
    """
    return {
        "time": "2026-06-16T00:00:00.000000000Z",
        "stream": "stdout",
        "_p": "F",
        "log": app_line,
        "kubernetes": {
            "pod_name": "routeiq-gateway-abc",
            "namespace_name": "routeiq",
            "container_name": "gateway",
        },
    }


# ------------------------------------------------------------------------- tests


def test_configmap_carries_both_config_keys() -> None:
    """The ConfigMap ships ``parsers.conf`` + the full ``routeiq-routing.conf``."""
    data = _render_fluent_bit_configmap()
    assert "parsers.conf" in data
    assert "routeiq-routing.conf" in data


def test_parser_is_json_format() -> None:
    """The ``routeiq_router_json`` parser MUST be Format=json (auto-promotes).

    A Format=json parser merges the decoded object at the record root, which is
    what the CW metric filters' top-level ``$.field`` paths require. A regex or
    other format would NOT auto-promote and would silently break the contract.
    """
    data = _render_fluent_bit_configmap()
    parsers = _parse_fluent_bit_sections(data["parsers.conf"])
    parser = _section(parsers, "PARSER", Name="routeiq_router_json")
    assert parser.get("Format") == "json", (
        "routeiq_router_json must be Format=json to promote inner keys to the "
        f"top level; got Format={parser.get('Format')!r}"
    )


def test_kubernetes_filter_does_not_consume_log_field() -> None:
    """The kubernetes filter must NOT nest or delete ``log`` before the parser.

    Guards the historical bug: ``Merge_Log On`` + ``Merge_Log_Key`` nests the
    parsed JSON under a sub-key (not the top level) and ``Keep_Log Off`` deletes
    ``log`` before the parser filter can read it - a silent no-op. The parser
    filter below owns the top-level promotion, so the kubernetes filter must
    leave ``log`` intact: Merge_Log Off + Keep_Log On.
    """
    data = _render_fluent_bit_configmap()
    sections = _parse_fluent_bit_sections(data["routeiq-routing.conf"])
    kube = _section(sections, "FILTER", Name="kubernetes")
    assert kube.get("Merge_Log", "Off") == "Off", (
        "kubernetes Merge_Log must be Off so it does not nest the router JSON "
        "under a sub-key instead of the top level the metric filters scan"
    )
    assert kube.get("Keep_Log", "On") == "On", (
        "kubernetes Keep_Log must be On so the `log` field survives for the "
        "downstream [FILTER] parser (Key_Name log) to promote"
    )


def test_parser_filter_wired_for_top_level_promotion() -> None:
    """The parser [FILTER] reads ``log`` with the JSON parser and reserves data."""
    data = _render_fluent_bit_configmap()
    sections = _parse_fluent_bit_sections(data["routeiq-routing.conf"])
    parser_filter = _section(sections, "FILTER", Name="parser")
    assert parser_filter.get("Key_Name") == "log"
    assert parser_filter.get("Parser") == "routeiq_router_json"
    assert parser_filter.get("Reserve_Data") == "On", (
        "Reserve_Data On keeps the kubernetes metadata alongside the promoted keys"
    )


def test_filter_order_kubernetes_then_parser() -> None:
    """The kubernetes filter must precede the parser filter in the chain."""
    data = _render_fluent_bit_configmap()
    sections = _parse_fluent_bit_sections(data["routeiq-routing.conf"])
    filter_names = [s.get("Name") for s in sections if s["__name__"] == "FILTER"]
    assert "kubernetes" in filter_names and "parser" in filter_names
    assert filter_names.index("kubernetes") < filter_names.index("parser")


def _run_chain(app_line: str, data: Dict[str, str]) -> Dict[str, Any]:
    """Run a wrapped app line through the rendered kubernetes->parser chain."""
    parsers = _parse_fluent_bit_sections(data["parsers.conf"])
    parser_def = _section(parsers, "PARSER", Name="routeiq_router_json")
    parser_format = parser_def.get("Format", "")

    sections = _parse_fluent_bit_sections(data["routeiq-routing.conf"])
    kube_cfg = _section(sections, "FILTER", Name="kubernetes")
    parser_cfg = _section(sections, "FILTER", Name="parser")

    record = _wrap_in_ci_envelope(app_line)
    record = _apply_kubernetes_filter(record, kube_cfg)
    record = _apply_parser_filter(record, parser_cfg, parser_format)
    return record


def test_routing_decision_keys_promoted_to_top_level() -> None:
    """A REAL routing_decision line ends with its keys at the record top level.

    This is the promotion contract the CW metric filters + the data lake depend
    on. We feed the actual app emitter's output (build_routing_decision_record)
    through the rendered Fluent Bit chain and assert the load-bearing top-level
    keys resolve.
    """
    data = _render_fluent_bit_configmap()
    record = build_routing_decision_record(
        selected_model="gpt-4o",
        decision="route",
        reason_code="ml_routed",
        category="code",
        latency_ms=250.0,
        prompt_tokens=120,
        completion_tokens=40,
    )
    app_line = json.dumps(record, sort_keys=True, separators=(",", ":"))

    promoted = _run_chain(app_line, data)

    # the three routing_decision top-level keys the metric filters + lake scan
    assert promoted.get("event") == "routing_decision"
    assert promoted.get("latency_ms") == 250
    assert promoted.get("gen_ai.response.model") == "gpt-4o"
    # the wrapped `log` string is consumed (Preserve_Key Off), kubernetes
    # metadata is reserved (Reserve_Data On)
    assert "log" not in promoted
    assert "kubernetes" in promoted


def test_error_level_key_promoted_to_top_level() -> None:
    """A REAL error line lands ``$.level == "error"`` at the record top level.

    The RouterErrorFilter selects on a top-level ``$.level == "error"`` key; this
    asserts the same chain promotes it (the 4th scanned key).
    """
    data = _render_fluent_bit_configmap()
    record = build_error_log_record(
        error_type="RuntimeError",
        error_message="boom",
    )
    app_line = json.dumps(record, sort_keys=True, separators=(",", ":"))

    promoted = _run_chain(app_line, data)

    assert promoted.get("level") == "error"
    assert promoted.get("event") == "error"


def test_all_required_metric_filter_keys_resolve() -> None:
    """Every key the P2 metric filters scan resolves at the top level.

    A single assertion over the union of the routing + error lines so a future
    edit that drops any one of the four scanned paths fails loudly.
    """
    data = _render_fluent_bit_configmap()
    routing_line = json.dumps(
        build_routing_decision_record(selected_model="claude-3-5", latency_ms=10.0),
        separators=(",", ":"),
    )
    error_line = json.dumps(
        build_error_log_record(error_type="E", error_message="x"),
        separators=(",", ":"),
    )
    promoted_keys = set(_run_chain(routing_line, data)) | set(
        _run_chain(error_line, data)
    )
    missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in promoted_keys]
    assert not missing, f"keys not promoted to top level: {missing}"


def test_simulator_catches_the_historical_nest_and_delete_bug() -> None:
    """Negative control: the simulator FAILS on the old broken kubernetes config.

    If the kubernetes filter is reverted to ``Merge_Log On`` + ``Merge_Log_Key``
    + ``Keep_Log Off``, the inner keys end up NESTED under the merge key and
    ``log`` is deleted before the parser filter runs - so the top-level keys do
    NOT resolve. This proves the simulator actually models the bug (so the
    positive tests above are meaningful, not vacuously green).
    """
    broken_kube = {
        "__name__": "FILTER",
        "Name": "kubernetes",
        "Merge_Log": "On",
        "Merge_Log_Key": "log_processed",
        "Keep_Log": "Off",
    }
    parser_cfg = {
        "__name__": "FILTER",
        "Name": "parser",
        "Key_Name": "log",
        "Parser": "routeiq_router_json",
        "Reserve_Data": "On",
        "Preserve_Key": "Off",
    }
    app_line = json.dumps(
        build_routing_decision_record(selected_model="gpt-4o", latency_ms=250.0),
        separators=(",", ":"),
    )
    record = _wrap_in_ci_envelope(app_line)
    record = _apply_kubernetes_filter(record, broken_kube)
    record = _apply_parser_filter(record, parser_cfg, "json")

    # the broken chain leaves event nested, NOT at the top level
    assert "event" not in record
    assert record.get("log_processed", {}).get("event") == "routing_decision"
