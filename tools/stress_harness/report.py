"""Assemble a markdown report + JSON sidecar from an ``AnalysisResult``.

The report is STRATEGY-AGNOSTIC (RouteIQ-4f19): it NAMES the active strategy,
always emits the generic distributions (model / strategy / per-category), and
renders the per-strategy verdict the dispatch produced — whether that's the
bandit fan-out check, the learned-router consistency check, the cost-aware
cheap-on-easy check, the personalized per-user drift check, the iterative
latency/cost report, or the generic "no strategy-specific verdict for <name>"
fallback.

``build_json`` produces the machine-readable sidecar; ``build_markdown`` renders
the human report. ``write_report`` writes both and returns their paths.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import AnalysisResult
from .verdicts import family_for, registered_families


def build_json(result: AnalysisResult) -> dict[str, Any]:
    """Project the ``AnalysisResult`` to a JSON-able dict."""
    data = asdict(result)
    data["verdict_family"] = family_for(result.active_strategy)
    data["registered_verdict_families"] = registered_families()
    return data


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _verdict_health(healthy: bool | None) -> str:
    if healthy is True:
        return "HEALTHY"
    if healthy is False:
        return "UNHEALTHY"
    return "INFO"


def build_markdown(result: AnalysisResult) -> str:
    """Render the human-facing markdown report."""
    lines: list[str] = []
    add = lines.append

    add("# RouteIQ router stress-test + routing-validation report")
    add("")
    strategy = result.active_strategy or "<unknown>"
    add(f"- **Active routing strategy**: `{strategy}`")
    add(f"- Verdict family dispatched: `{family_for(result.active_strategy)}`")
    if result.available_strategies:
        add(
            f"- Available strategies ({len(result.available_strategies)}): "
            + ", ".join(f"`{s}`" for s in result.available_strategies)
        )
    add(
        f"- Total requests fired: **{result.total_requests}** "
        f"(successful: {result.successful_requests}, "
        f"decision-line-enriched: {result.enriched_requests})"
    )
    add(
        f"- Conversations: **{result.total_conversations}** "
        f"(multi-turn: **{result.multi_turn_conversations}**)"
    )
    add("")

    # --- the strategy-agnostic verdict (the headline finding) ---
    add("## Routing verdict (strategy-agnostic dispatch)")
    add("")
    if result.verdict is not None:
        v = result.verdict
        add(
            f"**[{_verdict_health(v.healthy)}]** strategy `{v.strategy}` "
            f"-> family `{v.family}`"
        )
        add("")
        for msg in v.messages or [v.summary]:
            add(f"- {msg}")
        add("")
        if v.findings:
            add("<details><summary>verdict findings</summary>")
            add("")
            add("```json")
            add(json.dumps(v.findings, indent=2, default=str))
            add("```")
            add("")
            add("</details>")
            add("")
    else:
        add("_No verdict computed._")
        add("")

    # --- model distribution (ALWAYS available — the headline distribution) ---
    add("## Model distribution — requests per model")
    add("")
    add(
        f"Concrete model RouteIQ actually routed to, over all "
        f"**{result.successful_requests}** successful `model:auto` requests "
        f"(**{result.distinct_models}** distinct models). Always available — "
        "needs no decision-line enrichment. This is the primary routing-spread "
        "signal for ANY strategy."
    )
    add("")
    if result.model_distribution:
        add("| model | requests | share |")
        add("| --- | --- | --- |")
        dist = result.model_distribution
        for model in sorted(dist, key=lambda m: -dist[m]["count"]):
            row_stats = dist[model]
            add(
                f"| {model} | {int(row_stats['count'])} | "
                f"{_fmt_pct(row_stats['share'])} |"
            )
        add("")
    else:
        add("_No successful requests to attribute._")
        add("")

    # --- strategy distribution ---
    add("## Strategy distribution")
    add("")
    if result.strategy_distribution:
        add("| strategy | decisions |")
        add("| --- | --- |")
        sd = result.strategy_distribution
        for strat in sorted(sd, key=lambda s: -sd[s]):
            add(f"| {strat} | {sd[strat]} |")
        add("")
    else:
        add(
            "_No per-request strategy distribution (no decision-line enrichment "
            "and no server rollup). The active strategy named above governs the "
            "run._"
        )
        add("")

    # --- per-category routing table (ALWAYS available) ---
    add("## Per-category routing — which category routed to which model")
    add("")
    if result.category_model_counts:
        for category in sorted(result.category_model_counts):
            row = result.category_model_counts[category]
            add(
                f"- **{category}**: "
                + ", ".join(
                    f"`{m}`×{row[m]}" for m in sorted(row, key=lambda m: -row[m])
                )
            )
        add("")
    else:
        add("_No successful requests to attribute._")
        add("")

    if result.category_strategy_counts:
        add("**Per-category strategy counts** (from decision-line enrichment)")
        add("")
        for category in sorted(result.category_strategy_counts):
            row = result.category_strategy_counts[category]
            add(
                f"- **{category}**: "
                + ", ".join(
                    f"`{s}`×{row[s]}" for s in sorted(row, key=lambda s: -row[s])
                )
            )
        add("")

    # --- multi-turn views ---
    add("## Multi-turn routing")
    add("")
    switching = result.conversation_model_switching or {}
    if switching.get("conversations"):
        add(
            f"- Conversations with >=2 successful turns: "
            f"**{switching['conversations']}**; model CHANGED across turns in "
            f"**{switching['with_switch']}** "
            f"({_fmt_pct(switching.get('switch_rate'))})."
        )
        add("")
    if result.turn_position_distribution:
        add(
            "**Routing by turn position** (does the chosen model drift as context grows?)"
        )
        add("")
        models = sorted(
            {m for row in result.turn_position_distribution.values() for m in row}
        )
        add("| turn | " + " | ".join(models) + " |")
        add("| --- |" + " --- |" * len(models))
        for pos in sorted(result.turn_position_distribution):
            row = result.turn_position_distribution[pos]
            add(f"| {pos} | " + " | ".join(str(row.get(m, 0)) for m in models) + " |")
        add("")

    # --- control-plane provenance ---
    add("## Control-plane stats (RouteIQ surfaces)")
    add("")
    stats = result.server_stats
    if stats is not None:
        add(
            f"- active_strategy (from /routing/config): `{stats.active_strategy or 'n/a'}`"
        )
        add(f"- server total_decisions: {stats.total_decisions}")
        if stats.model_distribution:
            add("- server model_distribution (/stats/global):")
            for model in sorted(
                stats.model_distribution, key=lambda m: -stats.model_distribution[m]
            ):
                add(f"  - `{model}`: {stats.model_distribution[model]}")
        if stats.notes:
            add("- surface notes:")
            for note in stats.notes:
                add(f"  - {note}")
        add("")
    else:
        add("_Control-plane stats not fetched (no --stats-url / --admin-key)._")
        add("")

    # --- degradation / provenance notes ---
    if result.notes:
        add("## Notes")
        add("")
        for note in result.notes:
            add(f"- {note}")
        add("")

    add("---")
    families = ", ".join(f"`{f}`" for f in registered_families())
    add(
        f"_Verdict registry families: {families} "
        "(+ generic fallback for any unregistered strategy)._"
    )
    add("")
    return "\n".join(lines)


def write_report(result: AnalysisResult, out_dir: str | Path) -> dict[str, str]:
    """Write ``report.md`` + ``report.json`` to ``out_dir``; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "report.md"
    json_path = out / "report.json"
    md_path.write_text(build_markdown(result))
    json_path.write_text(json.dumps(build_json(result), indent=2, default=str))
    return {"markdown": str(md_path), "json": str(json_path)}
