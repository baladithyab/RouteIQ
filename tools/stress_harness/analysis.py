"""Strategy-agnostic routing analysis (RouteIQ-4f19).

Computes the ALWAYS-available generic distributions for ANY strategy
(RouteIQ-4f19 (b)) and then dispatches the per-strategy verdict by the active
strategy name (RouteIQ-4f19 (c) + RouteIQ-833c). The distributions never depend
on the strategy or on the optional decision-line enrichment — they are computed
straight from the client-observed records — so the harness produces a coherent
report whether RouteIQ is running KNN, a Thompson bandit, cost-aware, or a
strategy this harness has never heard of.

Three always-available distributions:
  1. model_distribution      — how many requests went to each model (the primary
                               routing-spread signal; body / decision-line model).
  2. strategy_distribution    — per-request strategy from the decision lines (when
                               enriched) PLUS the server-side rollup if fetched.
  3. category_model_counts    — category -> model -> count (the per-category
                               routing table) + category_strategy_counts.

Plus multi-turn views (routing by turn position, within-conversation switching)
and the strategy-agnostic verdict.

This module NEVER raises on absent decision lines / stats / models — it degrades
and annotates ``AnalysisResult.note(...)``.
"""

from __future__ import annotations

from collections import defaultdict

from .models import (
    AnalysisResult,
    EnrichedRecord,
    RouteIQStats,
)
from .verdicts import dispatch_verdict


def _shares(counts: dict[str, int]) -> dict[str, dict[str, float]]:
    """Turn a model -> count map into model -> {count, share}."""
    total = sum(counts.values())
    out: dict[str, dict[str, float]] = {}
    for model, count in counts.items():
        out[model] = {
            "count": float(count),
            "share": (count / total) if total else 0.0,
        }
    return out


def _model_of(rec: EnrichedRecord) -> str:
    """Best concrete model label for a request: decision-line model wins
    (authoritative), else the response-body model, else ``<none>`` (a 2xx with
    no model field — rare)."""
    return rec.effective_model or "<none>"


def model_distribution(records: list[EnrichedRecord], result: AnalysisResult) -> None:
    """ALWAYS-available 'how many requests went to each model?' histogram, plus
    the per-category-by-model counts.

    Built straight from successful requests — no strategy, no decision line, no
    catalog. This is the load-bearing answer for validating routing spread and
    is what the per-category routing table reports.
    """
    overall: dict[str, int] = defaultdict(int)
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        if not rec.request.ok:
            continue
        model = _model_of(rec)
        overall[model] += 1
        by_cat[rec.request.my_category_tag][model] += 1

    result.model_distribution = _shares(dict(overall))
    # distinct_models must NOT count the placeholder bucket.
    result.distinct_models = sum(
        1 for m in overall if m not in ("<none>", "<unmapped>")
    )
    result.category_model_counts = {cat: dict(m) for cat, m in by_cat.items()}


def strategy_distribution(
    records: list[EnrichedRecord], result: AnalysisResult
) -> None:
    """Per-request strategy distribution from the decision lines + category x
    strategy counts. Empty when no decision lines were enriched (the active
    strategy still names the run); annotated when so.
    """
    overall: dict[str, int] = defaultdict(int)
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    enriched_with_strategy = 0
    for rec in records:
        if not rec.request.ok:
            continue
        strategy = rec.decision_strategy
        if not strategy:
            continue
        enriched_with_strategy += 1
        overall[strategy] += 1
        by_cat[rec.request.my_category_tag][strategy] += 1

    result.strategy_distribution = dict(overall)
    result.category_strategy_counts = {cat: dict(s) for cat, s in by_cat.items()}
    if enriched_with_strategy == 0:
        result.note(
            "Per-request strategy_distribution empty: no routing_decision lines "
            "carried a strategy (run without --enrich-cwlogs, or the build does "
            "not emit per-request strategy). Run-level active_strategy still "
            "names the strategy."
        )


def multi_turn_analysis(records: list[EnrichedRecord], result: AnalysisResult) -> None:
    """Multi-turn views: conversation counts, routing-by-turn-position, and how
    often a conversation's chosen model switches between consecutive turns.

    Groups successful turns by ``conversation_id`` (ordered by ``turn_index``).
    ``turn_position_distribution`` answers "does the chosen model drift as the
    conversation grows?"; the switching summary answers "within one conversation,
    does routing stay pinned or move?". Body-only — no decision line needed.
    """
    convs: dict[str, list[EnrichedRecord]] = defaultdict(list)
    for rec in records:
        cid = rec.request.conversation_id
        if cid is None:
            cid = f"_anon_{id(rec)}"
        convs[cid].append(rec)

    result.total_conversations = len(convs)
    result.multi_turn_conversations = sum(
        1 for turns in convs.values() if any(t.request.num_turns > 1 for t in turns)
    )

    by_pos: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    conversations_considered = 0
    conversations_with_switch = 0
    for turns in convs.values():
        ok_turns = sorted(
            (t for t in turns if t.request.ok),
            key=lambda t: t.request.turn_index,
        )
        if not ok_turns:
            continue
        models_in_order: list[str] = []
        for t in ok_turns:
            model = _model_of(t)
            by_pos[t.request.turn_index][model] += 1
            models_in_order.append(model)
        if len(models_in_order) >= 2:
            conversations_considered += 1
            if len(set(models_in_order)) > 1:
                conversations_with_switch += 1

    result.turn_position_distribution = {
        pos: dict(m) for pos, m in sorted(by_pos.items())
    }
    result.conversation_model_switching = {
        "conversations": conversations_considered,
        "with_switch": conversations_with_switch,
        "switch_rate": (
            conversations_with_switch / conversations_considered
            if conversations_considered
            else None
        ),
    }


def analyze(
    records: list[EnrichedRecord],
    *,
    server_stats: RouteIQStats | None = None,
) -> AnalysisResult:
    """Run the generic distributions + the strategy-agnostic verdict.

    The active strategy is read from ``server_stats`` (surface C); when no stats
    were fetched, the run-level strategy is ``None`` and the verdict falls back to
    the generic distribution report. Never raises on missing data.
    """
    result = AnalysisResult()
    result.total_requests = len(records)
    result.successful_requests = sum(1 for r in records if r.request.ok)
    result.enriched_requests = sum(1 for r in records if r.has_decision)
    result.server_stats = server_stats
    if server_stats is not None:
        result.active_strategy = server_stats.active_strategy
        result.available_strategies = list(server_stats.available_strategies)

    model_distribution(records, result)
    strategy_distribution(records, result)
    multi_turn_analysis(records, result)

    # If we fetched a server-side strategy_distribution but enriched none from
    # decision lines, surface the server rollup so the report isn't empty.
    if (
        not result.strategy_distribution
        and server_stats
        and server_stats.strategy_distribution
    ):
        result.strategy_distribution = dict(server_stats.strategy_distribution)
        result.note(
            "Per-request strategy_distribution taken from server /stats/global "
            "rollup (no decision-line enrichment)."
        )

    if result.active_strategy is None:
        result.note(
            "Run-level active_strategy unknown (no control-plane stats fetched, "
            "or surface unreachable). Verdict falls back to the generic "
            "distribution report."
        )

    # the strategy-agnostic dispatch — ONE verdict for WHATEVER strategy runs.
    result.verdict = dispatch_verdict(
        result.active_strategy, records, server_stats, result
    )
    return result
