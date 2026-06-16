"""Categorized workload generator for the RouteIQ stress harness (RouteIQ-b245).

Generates chat prompts in five operator-facing buckets — math, code, creative,
easy-chitchat, hard-reasoning — and tags each generated request with its
ground-truth category at SEND time.

This module is ROUTER-AGNOSTIC and ported faithfully from the source harness:
the five buckets are the *harness's own* taxonomy, deliberately decoupled from
any RouteIQ strategy, model id, or classifier label. The analysis later compares
the harness tag against the model RouteIQ actually chose; it never assumes the
two are equal. (The source harness additionally carried an MMLU-Pro
``expected_domains`` table for a classifier-accuracy finding — RouteIQ routes by
strategy, not by emitting domain labels, so that coupling is intentionally
dropped.)

Allocation uses largest-remainder (Hamilton) rounding so per-bucket counts sum
EXACTLY to the requested total — load-bearing for the ``--dry-run`` 10k plan.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .models import CategorySpec, RequestRecord

# ---------------------------------------------------------------------------
# Follow-up turn templates (multi-turn). Turn 0 uses the bucket's primary
# ``prompts``; turns 1..N-1 cycle these category-appropriate follow-ups. They
# keep the conversation IN its category while growing the context — the signal
# is whether the strategy's model choice stays stable or drifts as the history
# lengthens (turn_position_distribution + conversation_model_switching).
# ---------------------------------------------------------------------------
_FOLLOWUPS: dict[str, tuple[str, ...]] = {
    "math": (
        "Now redo it but show every intermediate step.",
        "What changes if one coefficient is off by one?",
        "Can you verify the answer a different way?",
        "Generalize the result to the n-dimensional case.",
    ),
    "code": (
        "Now add error handling and type hints.",
        "How would you unit-test that?",
        "Refactor it for readability.",
        "What's the time and space complexity now?",
    ),
    "creative": (
        "Make it darker in tone.",
        "Rewrite it from a different character's point of view.",
        "Now make it half as long.",
        "Add a twist ending.",
    ),
    "easy-chitchat": (
        "Nice — anything else you'd suggest?",
        "Cool, what about for a rainy day?",
        "Thanks! One more idea?",
        "Got it. How about something indoors?",
    ),
    "hard-reasoning": (
        "Now steelman the opposing view.",
        "What hidden assumption is doing the most work here?",
        "How would the answer change under uncertainty?",
        "Give a concrete real-world example.",
    ),
}

_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec(
        name="math",
        prompts=(
            "Compute the determinant of the 3x3 matrix [[2,1,0],[1,3,1],[0,1,2]].",
            "Find all real roots of x^3 - 6x^2 + 11x - 6 = 0.",
            "What is the integral of x*sin(x) dx? Show the steps.",
            "A fair die is rolled 4 times. What is the probability of at least one six?",
            "Solve the system: 2x + 3y = 12, 4x - y = 5.",
        ),
    ),
    CategorySpec(
        name="code",
        prompts=(
            "Write a Python function that returns the nth Fibonacci number iteratively.",
            "Explain what a Python decorator is and give a small example.",
            "Refactor this loop into a list comprehension: "
            "out = []\\nfor x in xs:\\n    if x > 0:\\n        out.append(x*x)",
            "What is the time complexity of binary search and why?",
            "Write a SQL query to find the second-highest salary from an employees table.",
        ),
    ),
    CategorySpec(
        name="creative",
        prompts=(
            "Write a four-line poem about a lighthouse in winter.",
            "Invent a short bedtime story about a robot who collects sounds.",
            "Describe a marketplace on a floating city using only the sense of smell.",
            "Write the opening paragraph of a mystery novel set on a night train.",
            "Compose a limerick about a cat who refuses to use the litter box.",
        ),
    ),
    CategorySpec(
        name="easy-chitchat",
        is_easy=True,
        prompts=(
            "Hey, how's it going today?",
            "What's a good way to say good morning to a coworker?",
            "Thanks for the help earlier!",
            "Can you recommend a fun weekend activity?",
            "Tell me a light joke to cheer me up.",
        ),
    ),
    CategorySpec(
        name="hard-reasoning",
        prompts=(
            "Three people check into a hotel for $30 total, the clerk refunds $5 "
            "but the bellhop keeps $2. Where is the missing dollar? Explain the flaw.",
            "If a tree falls in a forest and no one hears it, does it make a sound? "
            "Argue both the physical and philosophical positions.",
            "A train leaves city A at 60mph, another leaves city B 120 miles away at "
            "40mph toward it. Where and when do they meet, and what assumption matters?",
            "Should a self-driving car prioritize its passenger or pedestrians? "
            "Reason through the ethical and legal tradeoffs.",
            "Explain why correlation does not imply causation using a concrete example "
            "where a hidden confounder reverses the apparent effect.",
        ),
    ),
)

_SPEC_BY_NAME: dict[str, CategorySpec] = {spec.name: spec for spec in _SPECS}


def category_specs() -> tuple[CategorySpec, ...]:
    """Return all five bucket specs (stable order matches ``CATEGORIES``)."""
    return _SPECS


def easy_categories() -> frozenset[str]:
    """Return the set of bucket names flagged ``is_easy`` (cost-aware verdict)."""
    return frozenset(spec.name for spec in _SPECS if spec.is_easy)


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Turn a (possibly partial / unnormalized) weight map into a full
    distribution over the five buckets that sums to 1.0.

    Unspecified buckets default to weight 1.0 (uniform). Negative weights are
    clamped to 0. If every weight is 0 (or the map is empty after clamping),
    fall back to uniform so we never divide by zero.
    """
    raw: dict[str, float] = {}
    for spec in _SPECS:
        w = 1.0 if weights is None else float(weights.get(spec.name, 0.0))
        raw[spec.name] = max(0.0, w)
    total = sum(raw.values())
    if total <= 0.0:
        n = len(_SPECS)
        return {name: 1.0 / n for name in raw}
    return {name: w / total for name, w in raw.items()}


def _allocate_counts(num_requests: int, weights: dict[str, float]) -> dict[str, int]:
    """Distribute ``num_requests`` across buckets by ``weights`` using largest-
    remainder rounding so the per-bucket counts sum EXACTLY to ``num_requests``.

    Largest-remainder (Hamilton's method): hand out the floors first, then the
    leftover slots to the buckets with the biggest fractional parts.
    Deterministic given the fixed bucket order, keeping ``--dry-run`` plans
    reproducible.
    """
    if num_requests <= 0:
        return {name: 0 for name in weights}
    exact = {name: weights[name] * num_requests for name in weights}
    floors = {name: int(value) for name, value in exact.items()}
    remainder = num_requests - sum(floors.values())
    order = sorted(weights, key=lambda name: exact[name] - floors[name], reverse=True)
    for name in order[:remainder]:
        floors[name] += 1
    return floors


def generate(
    num_requests: int,
    category_weights: dict[str, float] | None = None,
) -> list[RequestRecord]:
    """Generate ``num_requests`` tagged ``RequestRecord``s across the five
    buckets per ``category_weights`` (None = uniform).

    Prompts cycle deterministically through each bucket's template list so a
    given (count, weights) pair always yields the same workload — reproducible
    ``--dry-run`` plans and stable tests. Returns records ordered
    bucket-by-bucket; the caller may shuffle if interleaving is desired.
    """
    weights = _normalize_weights(category_weights)
    counts = _allocate_counts(num_requests, weights)
    records: list[RequestRecord] = []
    for spec in _SPECS:
        count = counts[spec.name]
        for i in range(count):
            prompt = spec.prompts[i % len(spec.prompts)]
            records.append(RequestRecord(my_category_tag=spec.name, prompt=prompt))
    return records


def _followup_prompt(category: str, turn_index: int) -> str:
    """The prompt for turn ``turn_index`` (>=1) of a ``category`` conversation.

    Cycles the category's follow-up templates deterministically. Falls back to a
    generic deepening nudge if a category has no follow-ups registered.
    """
    pool = _FOLLOWUPS.get(category)
    if not pool:
        return "Please continue and go deeper on that."
    return pool[(turn_index - 1) % len(pool)]


def _turn_count(seq: int, lengths: tuple[int, ...]) -> int:
    """Pick a turn count for the ``seq``-th multi-turn conversation.

    Cycles ``lengths`` deterministically (no RNG / no clock) so a run with a
    fixed (count, lengths) is fully reproducible. Each length clamped to >=2.
    """
    if not lengths:
        lengths = (2, 3, 4, 5)
    return max(2, lengths[seq % len(lengths)])


def _conversation_id(category: str, seq: int) -> str:
    """Stable, collision-resistant conversation id (no RNG / no clock)."""
    digest = hashlib.sha1(f"{category}:{seq}".encode()).hexdigest()[:10]
    return f"conv-{category}-{seq:05d}-{digest}"


def _user_id(seq: int, num_users: int) -> str | None:
    """Assign a synthetic user id by round-robin over ``num_users`` (None when
    multi-tenant simulation is off). Drives the personalized-routing verdict's
    per-user drift seam without coupling the generator to RouteIQ governance.
    """
    if num_users <= 0:
        return None
    return f"user-{seq % num_users:03d}"


def generate_conversations(
    num_conversations: int,
    *,
    category_weights: dict[str, float] | None = None,
    turn_lengths: tuple[int, ...] = (2, 3, 4, 5),
    seq_offset: int = 0,
    num_users: int = 0,
) -> list[list[RequestRecord]]:
    """Generate ``num_conversations`` MULTI-TURN conversations.

    Returns a list of conversations, each a list of ``RequestRecord`` turns
    sharing one ``conversation_id`` (turn_index 0..N-1, num_turns set on every
    turn). Turn 0 uses the bucket's primary prompt; later turns use category
    follow-ups. Lengths cycle through ``turn_lengths`` (each >=2).

    ``seq_offset`` lets a caller interleave these with an independent single-turn
    batch without conversation-id collisions. ``num_users`` (>0) round-robins a
    synthetic user id onto every turn of a conversation for the personalized
    verdict seam.
    """
    if num_conversations <= 0:
        return []
    weights = _normalize_weights(category_weights)
    counts = _allocate_counts(num_conversations, weights)

    conversations: list[list[RequestRecord]] = []
    seq = seq_offset
    for spec in _SPECS:
        for c in range(counts[spec.name]):
            n_turns = _turn_count(seq, turn_lengths)
            conv_id = _conversation_id(spec.name, seq)
            uid = _user_id(seq, num_users)
            turns: list[RequestRecord] = []
            for t in range(n_turns):
                if t == 0:
                    prompt = spec.prompts[c % len(spec.prompts)]
                else:
                    prompt = _followup_prompt(spec.name, t)
                turns.append(
                    RequestRecord(
                        my_category_tag=spec.name,
                        prompt=prompt,
                        conversation_id=conv_id,
                        turn_index=t,
                        num_turns=n_turns,
                        user_id=uid,
                    )
                )
            conversations.append(turns)
            seq += 1
    return conversations


def generate_mixed(
    num_single_turn: int,
    num_conversations: int,
    *,
    category_weights: dict[str, float] | None = None,
    turn_lengths: tuple[int, ...] = (2, 3, 4, 5),
    num_users: int = 0,
) -> tuple[list[RequestRecord], list[list[RequestRecord]]]:
    """Generate a MIXED workload: ``num_single_turn`` independent 1-turn requests
    PLUS ``num_conversations`` multi-turn conversations.

    Returns ``(singles, conversations)``. Single-turn records are wrapped with
    their own per-record conversation ids (turn 0 of 1) so EVERY record carries
    consistent conversation metadata and the analysis treats both populations
    uniformly. The conversation batch starts its seq past the single-turn count
    so ids stay globally unique within the run.
    """
    singles = generate(num_single_turn, category_weights)
    for i, rec in enumerate(singles):
        rec.conversation_id = _conversation_id(f"single-{rec.my_category_tag}", i)
        rec.turn_index = 0
        rec.num_turns = 1
        rec.user_id = _user_id(i, num_users)
    conversations = generate_conversations(
        num_conversations,
        category_weights=category_weights,
        turn_lengths=turn_lengths,
        seq_offset=num_single_turn,
        num_users=num_users,
    )
    return singles, conversations


def conversation_plan_summary(
    num_conversations: int,
    *,
    category_weights: dict[str, float] | None = None,
    turn_lengths: tuple[int, ...] = (2, 3, 4, 5),
    seq_offset: int = 0,
) -> dict[str, Any]:
    """Plan a multi-turn batch WITHOUT generating records: per-category
    conversation counts plus the total turns they expand to (for ``--dry-run``).

    ``seq_offset`` MUST match the offset the real generation uses
    (``generate_mixed`` passes ``seq_offset=num_single_turn``). Because
    ``_turn_count`` cycles ``turn_lengths`` by ``seq % len(turn_lengths)``, the
    per-conversation length depends on the PHASE of ``seq`` — so the plan must
    start counting at the same offset or the reported total diverges from what
    gets fired whenever ``num_single_turn % len(turn_lengths) != 0``.
    """
    weights = _normalize_weights(category_weights)
    counts = _allocate_counts(num_conversations, weights)
    total_turns = 0
    seq = seq_offset
    per_cat_turns: dict[str, int] = {}
    for spec in _SPECS:
        cat_turns = 0
        for _ in range(counts[spec.name]):
            cat_turns += _turn_count(seq, turn_lengths)
            seq += 1
        per_cat_turns[spec.name] = cat_turns
        total_turns += cat_turns
    return {
        "conversations_per_category": counts,
        "turns_per_category": per_cat_turns,
        "total_conversations": num_conversations,
        "total_turns": total_turns,
    }


def plan_summary(
    num_requests: int,
    category_weights: dict[str, float] | None = None,
) -> dict[str, int]:
    """Return the per-bucket request counts WITHOUT generating records
    (for ``--dry-run``)."""
    weights = _normalize_weights(category_weights)
    return _allocate_counts(num_requests, weights)
