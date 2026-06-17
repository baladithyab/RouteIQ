"""Per-strategy verdict registry — the strategy-agnostic dispatch (RouteIQ-833c).

THE design requirement (RouteIQ-4f19 (c)): the harness must GENERALIZE over
WHATEVER routing strategy RouteIQ runs. It must NOT hardcode the Kumaraswamy-
Thompson bandit. So routing-quality verdicts are pluggable: a REGISTRY of
verdict functions is dispatched by the ACTIVE strategy name, and any
unregistered strategy falls back to a generic distribution verdict that NEVER
crashes — it states plainly "no strategy-specific verdict for <name>".

The bandit fan-out check is ONE plugin among many here, not the spine.

A verdict plugin is ``(active_strategy, records, stats, distribution) ->
StrategyVerdict``. It reads the already-computed generic distribution plus the
enriched records and asserts the routing PROPERTY that strategy family promises:

  * kumaraswamy-thompson / thompson / bandit  -> FAN-OUT
        healthy bandit SPREADS traffic across arms; an unhealthy one PINS one
        model. We flag pinned-to-one-model as unhealthy.
  * knn / svm / mlp / mf / elo / routerdc / hybrid / centroid / graph / automix
    / causallm  -> CATEGORY->MODEL CONSISTENCY
        a deterministic learned router should route the SAME category to a
        DOMINANT model; we score each category's top-model share.
  * cost-aware  -> CHEAP-ON-EASY
        easy categories should be served by cheaper (smaller) models; we flag
        premium/large models dominating easy traffic.
  * personalized / gmt  -> PER-USER DRIFT
        per-user preference learning should let different users diverge; we
        measure how many distinct models each user saw and whether users differ.
  * router-r1 / llmrouter-r1 / *multiround  -> LATENCY / COST
        an iterative reasoning router trades latency/cost for quality; we report
        observed per-request latency + token cost (no hard pass/fail, it's a
        tradeoff signal).

Matching is by SUBSTRING on the case-folded active-strategy name (RouteIQ names
are like ``llmrouter-knn``, ``llmrouter-cost-aware``,
``llmrouter-nadirclaw-centroid``, ``kumaraswamy-thompson``), so version suffixes
and the ``llmrouter-`` prefix don't break dispatch. The FIRST family whose token
set matches wins; an unmatched strategy hits the generic fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import (
    AnalysisResult,
    EnrichedRecord,
    RouteIQStats,
    StrategyVerdict,
)

# A verdict plugin signature. ``distribution`` is the already-computed
# model_distribution ({model: {count, share}}) so plugins don't recompute it.
VerdictFn = Callable[
    [str, list[EnrichedRecord], RouteIQStats | None, AnalysisResult],
    StrategyVerdict,
]

# Tokens (case-folded substrings) that route an active strategy to a family.
# Order matters: families are tried top-to-bottom and the first token hit wins,
# so more-specific families (cost-aware, personalized, router-r1) precede the
# broad learned-router consistency family.
_FAMILY_TOKENS: list[tuple[str, tuple[str, ...]]] = [
    ("fan-out", ("kumaraswamy", "thompson", "bandit")),
    # cost-cascade precedes cost-aware: a router that explicitly CASCADES cheap->
    # expensive (try-cheap-first) is a distinct family from a one-shot cost-aware
    # picker, so its more-specific tokens must win.
    ("cost-cascade", ("cost-cascade", "cost_cascade", "cascade", "cheap-first")),
    ("cost-aware", ("cost-aware", "cost_aware", "costaware")),
    # semantic-intent precedes the broad learned-router consistency family: an
    # intent-classifier router dispatches each SEMANTIC BUCKET to a model GROUP,
    # which is a stronger claim than generic per-category dominance.
    (
        "semantic-intent",
        ("semantic-intent", "semantic_intent", "semantic", "intent"),
    ),
    ("personalized", ("personalized", "personalised", "gmt", "per-user")),
    ("latency-cost", ("router-r1", "router_r1", "routerr1", "r1", "multiround")),
    (
        "consistency",
        (
            "knn",
            "svm",
            "mlp",
            "mf",
            "elo",
            "routerdc",
            "hybrid",
            "centroid",
            "graph",
            "automix",
            "causallm",
        ),
    ),
]

# Tokens that mark a model as "premium / large" for the cost-aware verdict.
_PREMIUM_TOKENS: tuple[str, ...] = (
    "opus",
    "gpt-4",
    "gpt4",
    "ultra",
    "-large",
    "405b",
    "70b",
    "o1",
    "premium",
    "max",
)

# Thresholds (documented, overridable). These are routing-HEALTH heuristics, not
# correctness oracles — the report states them plainly.
DEFAULT_FANOUT_MIN_MODELS = 2  # a healthy bandit must touch >= this many models
DEFAULT_CONSISTENCY_MIN_TOP_SHARE = 0.60  # dominant-model share for consistency
DEFAULT_COST_EASY_PREMIUM_THRESHOLD = 0.40  # flag premium share on easy > this


def family_for(active_strategy: str | None) -> str:
    """Return the verdict family name that handles ``active_strategy`` (or
    ``"generic"`` when none matches / the strategy is unknown)."""
    if not active_strategy:
        return "generic"
    lowered = active_strategy.lower()
    for family, tokens in _FAMILY_TOKENS:
        if any(token in lowered for token in tokens):
            return family
    return "generic"


# ---------------------------------------------------------------------------
# helpers shared by plugins
# ---------------------------------------------------------------------------


def _is_premium(model: str | None) -> bool:
    if not model:
        return False
    lowered = model.lower()
    return any(token in lowered for token in _PREMIUM_TOKENS)


def _distinct_real_models(distribution: dict[str, dict[str, float]]) -> list[str]:
    """Distinct concrete models from the model_distribution, dropping the
    placeholder buckets the analysis uses for a 2xx with no model field."""
    return [m for m in distribution if m not in ("<none>", "<unmapped>")]


def _top_share(distribution: dict[str, dict[str, float]]) -> tuple[str | None, float]:
    """The single most-used model and its share over the whole run."""
    real = {m: distribution[m]["share"] for m in _distinct_real_models(distribution)}
    if not real:
        return None, 0.0
    top = max(real, key=lambda m: real[m])
    return top, real[top]


# ---------------------------------------------------------------------------
# verdict plugins
# ---------------------------------------------------------------------------


def fan_out_verdict(
    active_strategy: str,
    records: list[EnrichedRecord],
    stats: RouteIQStats | None,
    result: AnalysisResult,
    *,
    min_models: int = DEFAULT_FANOUT_MIN_MODELS,
) -> StrategyVerdict:
    """Bandit FAN-OUT check (Kumaraswamy-Thompson and friends).

    A healthy exploring bandit SPREADS traffic across arms. We flag the unhealthy
    case: all successful traffic pinned to ONE model (or to fewer than
    ``min_models`` models). This is ONE plugin among many — the rest of the
    harness does not depend on it.
    """
    models = _distinct_real_models(result.model_distribution)
    n_models = len(models)
    top_model, top_share = _top_share(result.model_distribution)
    healthy = n_models >= min_models
    findings = {
        "distinct_models": n_models,
        "min_models": min_models,
        "top_model": top_model,
        "top_model_share": round(top_share, 4),
        "pinned": n_models <= 1,
    }
    if n_models <= 1:
        summary = (
            f"UNHEALTHY: bandit pinned all traffic to a single model "
            f"({top_model or 'n/a'}) — no exploration spread."
        )
    elif not healthy:
        summary = (
            f"UNHEALTHY: bandit fanned out to only {n_models} model(s) "
            f"(< {min_models}); top model {top_model} holds "
            f"{top_share:.1%}."
        )
    else:
        summary = (
            f"HEALTHY: bandit fanned out across {n_models} models; "
            f"top model {top_model} holds {top_share:.1%} (not pinned)."
        )
    return StrategyVerdict(
        strategy=active_strategy,
        family="fan-out",
        healthy=healthy,
        summary=summary,
        findings=findings,
        messages=[summary],
    )


def consistency_verdict(
    active_strategy: str,
    records: list[EnrichedRecord],
    stats: RouteIQStats | None,
    result: AnalysisResult,
    *,
    min_top_share: float = DEFAULT_CONSISTENCY_MIN_TOP_SHARE,
) -> StrategyVerdict:
    """CATEGORY->MODEL CONSISTENCY check (deterministic learned routers:
    knn/svm/mlp/mf/elo/centroid/...).

    A learned router should map the SAME category to a DOMINANT model. For each
    category we compute the top model's share of that category's traffic; the
    category is "consistent" when that share >= ``min_top_share``. The verdict is
    healthy when every category with traffic is consistent.
    """
    per_category: dict[str, Any] = {}
    inconsistent: list[str] = []
    for category, counts in sorted(result.category_model_counts.items()):
        total = sum(counts.values())
        if total == 0:
            continue
        top_model = max(counts, key=lambda m: counts[m])
        share = counts[top_model] / total
        consistent = share >= min_top_share
        per_category[category] = {
            "top_model": top_model,
            "top_share": round(share, 4),
            "n": total,
            "consistent": consistent,
        }
        if not consistent:
            inconsistent.append(category)
    if not per_category:
        return StrategyVerdict(
            strategy=active_strategy,
            family="consistency",
            healthy=None,
            summary="No successful per-category traffic to assess consistency.",
            findings={"per_category": {}, "min_top_share": min_top_share},
        )
    healthy = not inconsistent
    if healthy:
        summary = (
            f"HEALTHY: every category routes consistently to a dominant model "
            f"(top-model share >= {min_top_share:.0%})."
        )
    else:
        summary = (
            f"INCONSISTENT: {', '.join(inconsistent)} spread below the "
            f"{min_top_share:.0%} dominant-model threshold — learned router not "
            f"converging per category."
        )
    return StrategyVerdict(
        strategy=active_strategy,
        family="consistency",
        healthy=healthy,
        summary=summary,
        findings={
            "per_category": per_category,
            "min_top_share": min_top_share,
            "inconsistent_categories": inconsistent,
        },
        messages=[summary],
    )


def cost_aware_verdict(
    active_strategy: str,
    records: list[EnrichedRecord],
    stats: RouteIQStats | None,
    result: AnalysisResult,
    *,
    premium_threshold: float = DEFAULT_COST_EASY_PREMIUM_THRESHOLD,
) -> StrategyVerdict:
    """CHEAP-ON-EASY check (cost-aware strategy).

    Easy categories should be served by cheaper (smaller) models. We compute the
    share of premium/large models on easy-tagged traffic; a share above
    ``premium_threshold`` is flagged as wasteful (the cost smoking gun). Healthy
    when no easy category exceeds the threshold.
    """
    from .workload import easy_categories

    easy = easy_categories()
    per_category: dict[str, Any] = {}
    flagged: list[str] = []
    for category in sorted(easy):
        counts = result.category_model_counts.get(category, {})
        total = sum(counts.values())
        if total == 0:
            continue
        premium = sum(c for m, c in counts.items() if _is_premium(m))
        share = premium / total
        is_flagged = share > premium_threshold
        per_category[category] = {
            "n": total,
            "premium_count": premium,
            "premium_share": round(share, 4),
            "flagged": is_flagged,
        }
        if is_flagged:
            flagged.append(category)
    if not per_category:
        return StrategyVerdict(
            strategy=active_strategy,
            family="cost-aware",
            healthy=None,
            summary="No easy-category traffic to assess cheap-on-easy routing.",
            findings={"per_category": {}, "premium_threshold": premium_threshold},
        )
    healthy = not flagged
    if healthy:
        summary = (
            f"HEALTHY: easy categories served by cheaper models "
            f"(premium share <= {premium_threshold:.0%})."
        )
    else:
        summary = (
            f"WASTEFUL: premium models dominate easy traffic on "
            f"{', '.join(flagged)} (> {premium_threshold:.0%}) — cost-aware "
            f"router routing trivial prompts to expensive models."
        )
    return StrategyVerdict(
        strategy=active_strategy,
        family="cost-aware",
        healthy=healthy,
        summary=summary,
        findings={
            "per_category": per_category,
            "premium_threshold": premium_threshold,
            "flagged_categories": flagged,
        },
        messages=[summary],
    )


def cost_cascade_verdict(
    active_strategy: str,
    records: list[EnrichedRecord],
    stats: RouteIQStats | None,
    result: AnalysisResult,
    *,
    easy_premium_threshold: float = DEFAULT_COST_EASY_PREMIUM_THRESHOLD,
) -> StrategyVerdict:
    """CHEAP-FIRST CASCADE invariant (cost-cascade strategy, RouteIQ-f086).

    A cost cascade tries the cheapest arm FIRST and only escalates to a
    premium/large model when the cheap arm is insufficient. So across the WHOLE
    run, the cheap tier should carry the bulk of traffic, and the premium tier
    should concentrate on HARD work — never dominate EASY work.

    Two invariants are checked:

      1. EASY buckets must not be premium-dominated (premium share on easy must
         be <= ``easy_premium_threshold``) — the same cheap-on-easy guard the
         cost-aware family uses, but here it is the cascade's FLOOR.
      2. Cheap-first spread: the cheap (non-premium) tier must carry strictly
         MORE total successful traffic than the premium tier (cheap_share >
         premium_share over the whole run). A cascade that escalates everything
         to premium has lost its cheap-first invariant.

    Healthy when BOTH hold. Not assessable (healthy=None) when there is no
    successful traffic at all.
    """
    from .workload import easy_categories

    # (1) easy-bucket premium domination (cheap-on-easy floor).
    easy = easy_categories()
    easy_flagged: list[str] = []
    easy_detail: dict[str, Any] = {}
    for category in sorted(easy):
        counts = result.category_model_counts.get(category, {})
        total = sum(counts.values())
        if total == 0:
            continue
        premium = sum(c for m, c in counts.items() if _is_premium(m))
        share = premium / total
        flagged = share > easy_premium_threshold
        easy_detail[category] = {
            "n": total,
            "premium_share": round(share, 4),
            "flagged": flagged,
        }
        if flagged:
            easy_flagged.append(category)

    # (2) run-wide cheap-first spread.
    premium_total = 0
    cheap_total = 0
    for model in _distinct_real_models(result.model_distribution):
        count = int(result.model_distribution[model]["count"])
        if _is_premium(model):
            premium_total += count
        else:
            cheap_total += count
    graded_total = premium_total + cheap_total

    if graded_total == 0:
        return StrategyVerdict(
            strategy=active_strategy,
            family="cost-cascade",
            healthy=None,
            summary="No successful traffic to assess the cheap-first cascade.",
            findings={
                "cheap_total": 0,
                "premium_total": 0,
                "easy_per_category": easy_detail,
            },
        )

    cheap_share = cheap_total / graded_total
    premium_share = premium_total / graded_total
    cheap_first = cheap_total > premium_total
    healthy = cheap_first and not easy_flagged

    if healthy:
        summary = (
            f"HEALTHY: cheap-first cascade — cheap tier carries "
            f"{cheap_share:.1%} of traffic vs premium {premium_share:.1%}, and "
            f"no easy bucket is premium-dominated."
        )
    else:
        reasons: list[str] = []
        if not cheap_first:
            reasons.append(
                f"premium tier dominates ({premium_share:.1%} >= cheap "
                f"{cheap_share:.1%}) — cascade escalated past the cheap arm"
            )
        if easy_flagged:
            reasons.append(
                f"premium models dominate easy traffic on "
                f"{', '.join(easy_flagged)} (> {easy_premium_threshold:.0%})"
            )
        summary = "CASCADE BROKEN: " + "; ".join(reasons) + "."

    return StrategyVerdict(
        strategy=active_strategy,
        family="cost-cascade",
        healthy=healthy,
        summary=summary,
        findings={
            "cheap_total": cheap_total,
            "premium_total": premium_total,
            "cheap_share": round(cheap_share, 4),
            "premium_share": round(premium_share, 4),
            "cheap_first": cheap_first,
            "easy_premium_threshold": easy_premium_threshold,
            "easy_per_category": easy_detail,
            "easy_flagged_categories": easy_flagged,
        },
        messages=[summary],
    )


def semantic_intent_verdict(
    active_strategy: str,
    records: list[EnrichedRecord],
    stats: RouteIQStats | None,
    result: AnalysisResult,
    *,
    min_top_share: float = DEFAULT_CONSISTENCY_MIN_TOP_SHARE,
) -> StrategyVerdict:
    """BUCKET -> GROUP DISPATCH check (semantic-intent router, RouteIQ-f086).

    A semantic-intent router classifies each request into a semantic bucket and
    dispatches that bucket to a dedicated model (group). Two properties make the
    dispatch healthy:

      1. WITHIN-bucket dominance: each semantic bucket routes consistently to a
         dominant model (top-model share >= ``min_top_share``) — the bucket has a
         clear handler, not a smear across models.
      2. BETWEEN-bucket differentiation: distinct buckets dispatch to DISTINCT
         dominant models (the router actually discriminates by intent rather than
         collapsing every bucket onto one model). With >= 2 buckets carrying
         traffic we require at least two distinct dominant handlers.

    The harness's own ground-truth category tag is the proxy for the semantic
    bucket (the workload tags each request at send time). Healthy when both
    properties hold; not assessable (healthy=None) when fewer than one bucket
    carries traffic.
    """
    per_bucket: dict[str, Any] = {}
    smeared: list[str] = []
    dominant_by_bucket: dict[str, str] = {}
    for bucket, counts in sorted(result.category_model_counts.items()):
        total = sum(counts.values())
        if total == 0:
            continue
        top_model = max(counts, key=lambda m: counts[m])
        share = counts[top_model] / total
        dominant = share >= min_top_share
        per_bucket[bucket] = {
            "top_model": top_model,
            "top_share": round(share, 4),
            "n": total,
            "dominant": dominant,
        }
        dominant_by_bucket[bucket] = top_model
        if not dominant:
            smeared.append(bucket)

    if not per_bucket:
        return StrategyVerdict(
            strategy=active_strategy,
            family="semantic-intent",
            healthy=None,
            summary="No per-bucket traffic to assess intent dispatch.",
            findings={"per_bucket": {}, "min_top_share": min_top_share},
        )

    distinct_handlers = len(set(dominant_by_bucket.values()))
    n_buckets = len(per_bucket)
    # differentiation only meaningful with >= 2 buckets carrying traffic.
    differentiated = n_buckets < 2 or distinct_handlers > 1
    within_ok = not smeared
    healthy = within_ok and differentiated

    if healthy:
        summary = (
            f"HEALTHY: {n_buckets} semantic bucket(s) each dispatch to a dominant "
            f"handler (>= {min_top_share:.0%}) across {distinct_handlers} distinct "
            f"model(s) — intent dispatch differentiates by bucket."
        )
    else:
        reasons = []
        if smeared:
            reasons.append(
                f"buckets smeared below the {min_top_share:.0%} dominant-model "
                f"threshold: {', '.join(smeared)}"
            )
        if not differentiated:
            reasons.append(
                f"all {n_buckets} buckets collapse onto a single model "
                f"(no intent differentiation)"
            )
        summary = "INTENT DISPATCH WEAK: " + "; ".join(reasons) + "."

    return StrategyVerdict(
        strategy=active_strategy,
        family="semantic-intent",
        healthy=healthy,
        summary=summary,
        findings={
            "per_bucket": per_bucket,
            "min_top_share": min_top_share,
            "distinct_handlers": distinct_handlers,
            "smeared_buckets": smeared,
            "differentiated": differentiated,
        },
        messages=[summary],
    )


def personalized_verdict(
    active_strategy: str,
    records: list[EnrichedRecord],
    stats: RouteIQStats | None,
    result: AnalysisResult,
) -> StrategyVerdict:
    """PER-USER DRIFT check (personalized / GMT routers).

    Per-user preference learning should let different users diverge. We measure,
    per user, the distinct models seen and the dominant model; "drift" means at
    least two users have different dominant models.

    Source preference (RouteIQ-2bbe): when ``stats.per_user_recent_models`` was
    read from RouteIQ's caller-scoped ``/me/stats`` surface, that AUTHORITATIVE
    server-side per-user routing view is used (the server attributes each
    decision to the user's key). Otherwise we fall back to grouping the
    client-observed successful requests by ``user_id``. With no per-user data at
    all (multi-tenant sim off) the verdict is not assessable (healthy=None).
    """
    by_user: dict[str, dict[str, int]] = {}
    source = "client-observed records"
    server_per_user = stats.per_user_recent_models if stats is not None else None
    if server_per_user:
        # AUTHORITATIVE: /me/stats recent_models per user (RouteIQ-2bbe).
        source = "/me/stats recent_models (server-authoritative)"
        for uid, recent in server_per_user.items():
            counts = by_user.setdefault(uid, {})
            for model in recent:
                counts[model] = counts.get(model, 0) + 1
    else:
        for rec in records:
            if not rec.request.ok:
                continue
            uid = rec.request.user_id
            if not uid:
                continue
            model = rec.effective_model or "<none>"
            by_user.setdefault(uid, {}).setdefault(model, 0)
            by_user[uid][model] += 1
    if not by_user:
        return StrategyVerdict(
            strategy=active_strategy,
            family="personalized",
            healthy=None,
            summary=(
                "No per-user traffic (run without --num-users); per-user drift "
                "not assessable. Re-run with --num-users N to exercise "
                "personalized routing."
            ),
            findings={"users": 0},
        )
    dominant: dict[str, str] = {}
    per_user: dict[str, Any] = {}
    for uid, counts in by_user.items():
        top = max(counts, key=lambda m: counts[m])
        dominant[uid] = top
        per_user[uid] = {
            "distinct_models": len(counts),
            "dominant_model": top,
            "n": sum(counts.values()),
        }
    distinct_dominant = len(set(dominant.values()))
    drift = distinct_dominant > 1
    summary = (
        f"{'DRIFT OBSERVED' if drift else 'NO DRIFT'}: {len(by_user)} users, "
        f"{distinct_dominant} distinct dominant model(s) across users "
        f"(via {source}) — "
        f"{'personalized routing diverges per user' if drift else 'all users converge to one model'}."
    )
    return StrategyVerdict(
        strategy=active_strategy,
        family="personalized",
        # drift is the EXPECTED healthy signal for a personalized router; absence
        # is a soft warning, not a hard fail (one model may simply be best for all).
        healthy=drift if len(by_user) > 1 else None,
        summary=summary,
        findings={
            "users": len(by_user),
            "distinct_dominant_models": distinct_dominant,
            "per_user": per_user,
            "source": source,
        },
        messages=[summary],
    )


def latency_cost_verdict(
    active_strategy: str,
    records: list[EnrichedRecord],
    stats: RouteIQStats | None,
    result: AnalysisResult,
) -> StrategyVerdict:
    """LATENCY / COST tradeoff report (router-r1 / iterative multi-round).

    An iterative reasoning router trades latency + token cost for answer quality.
    There is no hard pass/fail here — we report the observed per-request client
    latency (p50/p95-ish via simple percentiles) and total tokens so an operator
    can judge the tradeoff. healthy is None (informational).
    """
    latencies = sorted(
        r.request.client_latency_ms
        for r in records
        if r.request.ok and r.request.client_latency_ms is not None
    )
    total_tokens = sum(r.request.total_tokens or 0 for r in records if r.request.ok)
    n = len(latencies)

    def _pct(p: float) -> float | None:
        if not latencies:
            return None
        idx = min(n - 1, int(round(p * (n - 1))))
        return round(latencies[idx], 2)

    findings = {
        "n_latency_samples": n,
        "latency_p50_ms": _pct(0.50),
        "latency_p95_ms": _pct(0.95),
        "total_tokens": total_tokens,
    }
    if n == 0:
        summary = "No successful requests to report latency/cost tradeoff."
    else:
        summary = (
            f"INFO (tradeoff): iterative router p50 latency "
            f"{findings['latency_p50_ms']}ms / p95 {findings['latency_p95_ms']}ms "
            f"over {n} requests; {total_tokens} total tokens consumed."
        )
    return StrategyVerdict(
        strategy=active_strategy,
        family="latency-cost",
        healthy=None,
        summary=summary,
        findings=findings,
        messages=[summary],
    )


def generic_verdict(
    active_strategy: str | None,
    records: list[EnrichedRecord],
    stats: RouteIQStats | None,
    result: AnalysisResult,
) -> StrategyVerdict:
    """The FALLBACK verdict for an UNKNOWN / unregistered strategy.

    NEVER crashes: it restates the always-available distribution (distinct models
    + top model + share) and clearly notes there is no strategy-specific verdict
    for this strategy. This is the safety net that makes the dispatch
    strategy-agnostic (RouteIQ-4f19 (c)).
    """
    name = active_strategy or "<unknown>"
    n_models = result.distinct_models
    top_model, top_share = _top_share(result.model_distribution)
    summary = (
        f"No strategy-specific verdict for '{name}'. Generic distribution: "
        f"{n_models} distinct model(s) over {result.successful_requests} "
        f"successful requests"
        + (
            f"; top model {top_model} ({top_share:.1%})."
            if top_model
            else " (no models observed)."
        )
    )
    return StrategyVerdict(
        strategy=name,
        family="generic",
        healthy=None,
        summary=summary,
        findings={
            "distinct_models": n_models,
            "top_model": top_model,
            "top_model_share": round(top_share, 4),
        },
        messages=[summary],
    )


# Registry: family name -> verdict plugin. Keyed by the family ``family_for``
# resolves the active strategy to. ``generic`` is intentionally absent — it is
# the dispatch's hard-coded fallback, not a registry entry, so it can never be
# accidentally unregistered.
_REGISTRY: dict[str, VerdictFn] = {
    "fan-out": fan_out_verdict,
    "consistency": consistency_verdict,
    "cost-aware": cost_aware_verdict,
    "cost-cascade": cost_cascade_verdict,
    "semantic-intent": semantic_intent_verdict,
    "personalized": personalized_verdict,
    "latency-cost": latency_cost_verdict,
}


def registered_families() -> list[str]:
    """The verdict families with a registered plugin (excludes the generic
    fallback). >= 3 families is the acceptance floor; we ship 5."""
    return sorted(_REGISTRY)


def dispatch_verdict(
    active_strategy: str | None,
    records: list[EnrichedRecord],
    stats: RouteIQStats | None,
    result: AnalysisResult,
) -> StrategyVerdict:
    """Dispatch the per-strategy verdict by the ACTIVE strategy name.

    Resolves the strategy to a family via ``family_for`` and invokes its
    registered plugin; an unmatched / unknown strategy hits ``generic_verdict``.
    NEVER raises — a plugin that itself errors degrades to the generic verdict
    with a note, so one buggy plugin can't crash the report.
    """
    family = family_for(active_strategy)
    plugin = _REGISTRY.get(family)
    if plugin is None or active_strategy is None:
        return generic_verdict(active_strategy, records, stats, result)
    try:
        return plugin(active_strategy, records, stats, result)
    except Exception as exc:  # noqa: BLE001 — one plugin must not crash the run
        verdict = generic_verdict(active_strategy, records, stats, result)
        verdict.messages.append(
            f"(verdict plugin '{family}' errored: {type(exc).__name__}: {exc}; "
            f"fell back to generic distribution.)"
        )
        return verdict
