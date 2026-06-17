"""Dataclasses for the RouteIQ router stress-test + validation harness.

These types are the spine of the harness: every stage (workload -> client ->
stats -> analysis -> report) reads or writes one of them.

RouteIQ's routing decision surface differs from an Envoy/x-sr-target gateway:

    A. response body ``model``      -> the CONCRETE backend model RouteIQ chose
                                       for a ``model: "auto"`` request.
    B. ``routing_decision`` log line -> the AUTHORITATIVE per-request decision
                                       (model + strategy), keyed on request id.
    C. RouteIQ control-plane stats   -> org-wide rollups (active strategy,
                                       model_distribution, strategy_distribution).

The join key between A and B is the OpenAI completion ``id`` (== request_id),
echoed in RouteIQ's ``routing_decision`` log line. ``RequestRecord`` holds what
the *client* observes (surface A); ``RoutingDecisionLine`` is the optional
surface-B enrichment; ``RouteIQStats`` snapshots surface C.

The decision line's ``strategy``/``model`` are a SOFT contract: a RouteIQ build
that does not emit the line, or emits it without a strategy, must degrade — never
``KeyError``. ``RoutingDecisionLine.present`` and ``RouteIQStats.active_strategy``
both default to the absent state so downstream analysis degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The five operator-facing workload buckets the harness tags at SEND time.
# These are the harness's OWN ground-truth labels — they are NOT a RouteIQ
# strategy, a RouteIQ model id, or a classifier domain. The analysis compares
# the harness tag against the model RouteIQ actually picked; it never assumes
# the bucket name equals any RouteIQ-internal label.
CATEGORIES: tuple[str, ...] = (
    "math",
    "code",
    "creative",
    "easy-chitchat",
    "hard-reasoning",
)

# Which buckets are "easy" — used by the cost-aware verdict plugin to check that
# cheap models serve easy traffic, and to flag wasteful premium-model routing on
# trivial prompts. Mirrors the source harness's EASY_CATEGORIES seam.
EASY_CATEGORIES: frozenset[str] = frozenset({"easy-chitchat"})


@dataclass(frozen=True)
class CategorySpec:
    """One workload bucket: its name, the prompt templates that generate
    traffic, and whether it is an "easy" bucket.

    ``is_easy`` drives the cost-aware verdict plugin's cheap-model-on-easy check.
    Unlike the source harness this carries no router-classifier coupling —
    RouteIQ's routing is strategy-driven, not domain-label-driven, so the bucket
    is purely the harness's own ground-truth tag.
    """

    name: str
    prompts: tuple[str, ...]
    is_easy: bool = False


@dataclass
class RequestRecord:
    """What the CLIENT captures per request (surface A).

    Populated entirely from the HTTP exchange. The ``request_id`` (== body
    ``id``) is the join key used to attach the optional surface-B routing
    decision line later.

    ONE record == ONE chat-completions call == ONE turn. A single-turn request
    is a conversation of length 1, so every analysis treats turns uniformly.
    Multi-turn conversations share one ``conversation_id`` with ``turn_index``
    (0-based) + ``num_turns`` so the analysis can slice routing by conversation
    position (does the chosen model drift as context grows?).
    """

    # --- ground truth tagged at send time (the harness's own bucket) ---
    my_category_tag: str
    prompt: str

    # --- conversation grouping (single-turn => a 1-turn conversation) ---
    conversation_id: str | None = None
    turn_index: int = 0
    num_turns: int = 1
    # optional per-conversation synthetic user id (personalized-routing drift)
    user_id: str | None = None

    # --- correlation join key (body ``id`` == request_id) ---
    request_id: str | None = None
    header_request_id: str | None = None

    # --- surface A: response body ---
    # the CONCRETE backend model RouteIQ routed this ``model:auto`` request to.
    body_model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # --- client-side observation ---
    http_status: int | None = None
    client_latency_ms: float | None = None
    sent_ts: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the request completed with a 2xx and carried an id."""
        return (
            self.error is None
            and self.http_status is not None
            and 200 <= self.http_status < 300
            and self.request_id is not None
        )


@dataclass
class RoutingDecisionLine:
    """One parsed RouteIQ ``routing_decision`` log line (optional surface B).

    Emitted by ``router_decision_callback`` per decision. The AUTHORITATIVE
    per-request (model, strategy) pair, keyed on ``request_id``. This is a SOFT
    contract: a build that does not emit the line, or emits it without a
    strategy, leaves ``present`` False / fields ``None``. Every reader checks
    ``present`` before taking its full (vs degraded) path — accessing the fields
    is always safe.
    """

    present: bool = False
    request_id: str | None = None
    model: str | None = None  # authoritative chosen backend model
    strategy: str | None = None  # authoritative strategy that chose it
    profile: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None


@dataclass
class EnrichedRecord:
    """A ``RequestRecord`` optionally joined with its ``routing_decision`` line.

    Reliability ranking of the chosen model: prefer the decision-line ``model``
    (surface B, RouteIQ-authored, authoritative) when present; else fall back to
    the response-body ``model`` (surface A). ``effective_model`` encodes that.
    The strategy comes ONLY from the decision line (the body carries no
    strategy) — when absent, the analysis names the run-level active strategy.
    """

    request: RequestRecord
    decision: RoutingDecisionLine | None = None

    @property
    def request_id(self) -> str | None:
        return self.request.request_id

    @property
    def has_decision(self) -> bool:
        return self.decision is not None and self.decision.present

    @property
    def effective_model(self) -> str | None:
        """Best available concrete model for this request: decision-line model
        wins (authoritative), else the response-body model."""
        if self.decision is not None and self.decision.present and self.decision.model:
            return self.decision.model
        return self.request.body_model

    @property
    def decision_strategy(self) -> str | None:
        """Per-request strategy from the decision line, or None (the analysis
        falls back to the run-level active strategy)."""
        if self.decision is not None and self.decision.present:
            return self.decision.strategy
        return None


@dataclass
class RouteIQStats:
    """Snapshot of RouteIQ's control-plane stats surfaces (surface C).

    Assembled by the stats client from ``/routing/config`` (active strategy +
    available set), ``/stats/global`` (model + strategy distributions), and
    ``/routing/stats`` (decision totals). Every field defaults to the absent /
    empty state so a partial or unreachable control plane never crashes the
    analysis — the report simply names what it could and could not read.
    """

    active_strategy: str | None = None
    available_strategies: list[str] = field(default_factory=list)
    model_distribution: dict[str, int] = field(default_factory=dict)
    strategy_distribution: dict[str, int] = field(default_factory=dict)
    total_decisions: int = 0
    # Per-user routing read from RouteIQ's caller-scoped ``/me/stats`` surface
    # (RouteIQ-2bbe): user_id -> ``recent_models`` the server attributes to that
    # user's key. Empty when the harness fired no synthetic users or could not
    # read /me/stats; the personalized verdict prefers this AUTHORITATIVE
    # server-side view over the client-observed body model when present.
    per_user_recent_models: dict[str, list[str]] = field(default_factory=dict)
    # raw payloads kept for the report's provenance / debugging section.
    raw: dict[str, Any] = field(default_factory=dict)
    # human-readable notes on which surfaces were/weren't reachable.
    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)


@dataclass
class StrategyVerdict:
    """One per-strategy verdict from a registered verdict plugin (or the generic
    fallback).

    ``strategy`` is the active strategy name the verdict was produced for.
    ``family`` is the plugin family that handled it (``fan-out``, ``consistency``,
    ``cost-aware``, ``cost-cascade``, ``semantic-intent``, ``personalized``,
    ``latency-cost``, or ``generic``).
    ``healthy`` is the plugin's pass/fail (None == not assessable, e.g. the
    generic fallback or insufficient data). ``findings`` carries the verdict's
    structured numbers; ``messages`` the human-readable lines for the report.
    """

    strategy: str
    family: str
    healthy: bool | None
    summary: str
    findings: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """The generic distribution analyses + the per-strategy verdict, assembled
    by ``analysis.analyze`` and consumed by ``report``.

    The distributions are ALWAYS emitted for any strategy (RouteIQ-4f19 (b)):
    model_distribution, strategy_distribution, and the per-category routing
    table. ``verdict`` is the strategy-agnostic dispatch result (RouteIQ-4f19
    (c) + RouteIQ-833c).
    """

    # --- run identity ---
    active_strategy: str | None = None
    available_strategies: list[str] = field(default_factory=list)

    # --- request totals ---
    total_requests: int = 0
    successful_requests: int = 0
    enriched_requests: int = 0  # how many had a routing_decision line attached
    total_conversations: int = 0
    multi_turn_conversations: int = 0

    # --- ALWAYS-available distributions (RouteIQ-4f19 (b)) ---
    # client-observed: how many requests went to each model (body/decision).
    model_distribution: dict[str, dict[str, float]] = field(default_factory=dict)
    distinct_models: int = 0
    # per-request strategy distribution from the decision lines (empty when no
    # lines were enriched — the active strategy still names the run).
    strategy_distribution: dict[str, int] = field(default_factory=dict)
    # category -> model -> count (the per-category routing table).
    category_model_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # category -> strategy -> count (which strategy handled which category;
    # only populated when decision lines carry per-request strategy).
    category_strategy_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    # --- multi-turn views ---
    # turn_index -> model -> count (does the chosen model drift as context grows)
    turn_position_distribution: dict[int, dict[str, int]] = field(default_factory=dict)
    # within-conversation model switching summary
    conversation_model_switching: dict[str, Any] = field(default_factory=dict)

    # --- control-plane snapshot (surface C), when fetched ---
    server_stats: RouteIQStats | None = None

    # --- the strategy-agnostic verdict (RouteIQ-4f19 (c)) ---
    verdict: StrategyVerdict | None = None

    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)
