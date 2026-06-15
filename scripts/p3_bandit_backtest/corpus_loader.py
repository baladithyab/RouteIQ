"""Offline corpus loader for the Kumaraswamy-Thompson bandit backtest.

Stdlib only (``csv``, ``json``, ``dataclasses``). No DB, no RouteIQ import — this
file de-risks the bandit math against the preserved corpus BEFORE any live wiring.

The preserved corpus lives at::

    vllm-sr-on-aws/mlops-corpus-export/sft_corpus.csv          (241 rows / 31 arms)
    vllm-sr-on-aws/mlops-corpus-export/router_replay_records.sql  (Postgres pg_dump COPY)

The corpus is **reward-poor**: there is no recorded quality / win-loss / judge / latency
signal anywhere. The only reward-bearing fields are cost/token proxies plus a served-success
flag. A bandit backtest must therefore use a *constructed* reward, bounded to ``[0, 1]``:

    served            = response_status == 200 AND finish_reason == 'stop' AND content non-null
    reward_costmin    = served * (1 - actual_cost / cmax)          # CSV-only
    reward_savings    = served * (cost_savings / baseline_cost)    # SQL-only, == 1 - actual/baseline

State these caveats loudly in any result: the reward is a CONSTRUCTED cost-efficiency proxy,
not a quality reward; the log is single-action with no exploration and no counterfactuals;
241 rows / 31 arms is small and long-tailed (bucket the arms).

See ``research/p3/discover-corpus.md`` for the full field-level discovery.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Optional

# Default corpus locations (relative to this file's grandparent's sibling repo).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVBOX = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
DEFAULT_CSV_PATH = os.path.join(
    _DEVBOX, "vllm-sr-on-aws", "mlops-corpus-export", "sft_corpus.csv"
)
DEFAULT_SQL_PATH = os.path.join(
    _DEVBOX, "vllm-sr-on-aws", "mlops-corpus-export", "router_replay_records.sql"
)


@dataclass
class Row:
    """One replayed routing decision from the preserved corpus."""

    id: str
    prompt: str
    selected_model: str  # the logged action (arm key in the backtest)
    baseline_model: str  # 'gpt-5-4' or '' (the cost reference / oracle)
    prompt_tokens: float
    completion_tokens: float
    actual_cost: float
    served: int  # 1 if the call answered cleanly, else 0
    finish_reason: str
    reward_costmin: float  # in [0,1]: served * (1 - actual_cost/cmax)  (CSV-derivable)
    # SQL-only fields (None when loaded from CSV alone):
    baseline_cost: Optional[float] = None
    cost_savings: Optional[float] = None
    reward_savings: Optional[float] = (
        None  # in [0,1]: served * (cost_savings/baseline_cost)
    )


def _f(x: Optional[str], default: float = 0.0) -> float:
    """Parse a possibly-empty/whitespace string to float."""
    s = (x or "").strip()
    if not s or s == r"\N":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _parse_served(response_body: str) -> tuple[int, str]:
    """Derive (served, finish_reason) from a stored chat.completion JSON body."""
    rb = (response_body or "").strip()
    if not rb or rb == r"\N":
        return 0, ""
    try:
        j = json.loads(rb)
        ch = j["choices"][0]
        fr = ch.get("finish_reason", "") or ""
        content = ch.get("message", {}).get("content")
        served = 1 if (content and fr == "stop") else 0
        return served, fr
    except Exception:
        return 0, ""


def load_csv(path: str = DEFAULT_CSV_PATH) -> list[Row]:
    """Load the CSV corpus into a list of :class:`Row`.

    Supports the pure cost-min reward (``reward_costmin``) which only needs
    ``actual_cost`` and the served flag. The savings-ratio reward requires the
    SQL file; use :func:`load_sql` for that.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))

    costs = [
        _f(r.get("actual_cost")) for r in raw if (r.get("actual_cost") or "").strip()
    ]
    cmax = max(costs) if costs else 1.0

    rows: list[Row] = []
    for r in raw:
        served, fr = _parse_served(r.get("response_body", ""))
        ac = _f(r.get("actual_cost"))
        # cheaper => higher reward; gated on served
        reward_costmin = served * (1.0 - (ac / cmax if cmax else 0.0))
        rows.append(
            Row(
                id=r["id"],
                prompt=r.get("prompt", ""),
                selected_model=r.get("selected_model", ""),
                baseline_model=r.get("baseline_model", ""),
                prompt_tokens=_f(r.get("prompt_tokens")),
                completion_tokens=_f(r.get("completion_tokens")),
                actual_cost=ac,
                served=served,
                finish_reason=fr,
                reward_costmin=reward_costmin,
            )
        )
    return rows


def _split_copy_block(sql_text: str, table: str = "router_replay_records") -> list[str]:
    """Return the tab-delimited data lines of a pg_dump ``COPY ... FROM stdin;`` block."""
    lines = sql_text.splitlines()
    out: list[str] = []
    in_block = False
    for line in lines:
        if not in_block:
            if line.startswith("COPY ") and table in line and "FROM stdin" in line:
                in_block = True
            continue
        if line == r"\.":
            break
        out.append(line)
    return out


# The 54-column COPY header order (verbatim from the pg_dump file).
_SQL_COLS = [
    "id",
    "timestamp",
    "request_id",
    "decision",
    "decision_tier",
    "decision_priority",
    "category",
    "original_model",
    "selected_model",
    "reasoning_mode",
    "signals",
    "projections",
    "projection_scores",
    "signal_confidences",
    "signal_values",
    "tool_trace",
    "projection_trace",
    "session_policy",
    "request_body",
    "response_body",
    "response_status",
    "from_cache",
    "streaming",
    "request_body_truncated",
    "response_body_truncated",
    "guardrails_enabled",
    "jailbreak_enabled",
    "pii_enabled",
    "prompt",
    "prompt_truncated",
    "tool_definitions",
    "tool_definitions_truncated",
    "rag_enabled",
    "rag_backend",
    "rag_context_length",
    "rag_similarity_score",
    "hallucination_enabled",
    "hallucination_detected",
    "hallucination_confidence",
    "hallucination_spans",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "actual_cost",
    "baseline_cost",
    "cost_savings",
    "currency",
    "baseline_model",
    "session_id",
    "turn_index",
    "previous_response_id",
    "conversation_id",
    "created_at",
    "cached_prompt_tokens",
]


def _unescape_copy_field(v: str) -> str:
    """Decode pg COPY field escapes (\\t, \\n, \\\\). ``\\N`` -> empty handled by caller."""
    if v == r"\N":
        return ""
    return (
        v.replace(r"\t", "\t")
        .replace(r"\n", "\n")
        .replace(r"\r", "\r")
        .replace(r"\\", "\\")
    )


def load_sql(path: str = DEFAULT_SQL_PATH) -> list[Row]:
    """Load the SQL corpus, computing the richer savings-ratio reward.

    ``reward_savings = served * (cost_savings / baseline_cost)`` in ``[0, 1]``. Rows with
    no baseline/cost (the 5 jailbreak-block rows and a few no-cost rows) get reward 0.
    """
    with open(path, encoding="utf-8") as fh:
        sql_text = fh.read()
    data_lines = _split_copy_block(sql_text)

    rows: list[Row] = []
    for line in data_lines:
        fields = line.split("\t")
        if len(fields) != len(_SQL_COLS):
            continue
        rec = {col: _unescape_copy_field(v) for col, v in zip(_SQL_COLS, fields)}
        served, fr = _parse_served(rec.get("response_body", ""))
        # response_status is a stronger served gate than the body alone
        status = _f(rec.get("response_status"))
        if status != 200.0:
            served = 0
        ac = _f(rec.get("actual_cost"))
        bc = _f(rec.get("baseline_cost"))
        cs = _f(rec.get("cost_savings"))
        savings_ratio = (cs / bc) if bc > 0 else 0.0
        savings_ratio = min(max(savings_ratio, 0.0), 1.0)
        reward_savings = served * savings_ratio
        rows.append(
            Row(
                id=rec["id"],
                prompt=rec.get("prompt", ""),
                selected_model=rec.get("selected_model", ""),
                baseline_model=rec.get("baseline_model", ""),
                prompt_tokens=_f(rec.get("prompt_tokens")),
                completion_tokens=_f(rec.get("completion_tokens")),
                actual_cost=ac,
                served=served,
                finish_reason=fr,
                reward_costmin=0.0,  # not the SQL reward of interest; recompute via load_csv
                baseline_cost=bc,
                cost_savings=cs,
                reward_savings=reward_savings,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Arm bucketing (collapse 31 sparse arms into ~5-8 size/family tiers)
# ---------------------------------------------------------------------------


def size_bucket(model: str) -> str:
    """Bucket a model name by approximate parameter size tier.

    31 arms over 241 pulls is too sparse for stable per-arm posteriors. Collapsing
    into small/mid/large tiers gives ~30-80 pulls per bucket.
    """
    m = model.lower()
    # extract a trailing parameter-size hint like '4b', '120b', '480b'
    import re

    sizes = re.findall(r"(\d+)b\b", m)
    if sizes:
        n = max(int(s) for s in sizes)
        if n <= 9:
            return "small"
        if n <= 40:
            return "mid"
        return "large"
    # named tiers without an explicit size
    if any(t in m for t in ("nano", "mini", "flash", "small", "3b", "4b")):
        return "small"
    if any(t in m for t in ("super", "thinking", "next", "480", "235", "120")):
        return "large"
    return "mid"
