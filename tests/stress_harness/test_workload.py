"""Tests for the categorized workload generator (workload.py) — RouteIQ-b245.

Covers largest-remainder allocation (single + multi-turn), determinism, and the
10k headline allocation, all with no network.
"""

from __future__ import annotations

from stress_harness import workload
from stress_harness.models import CATEGORIES


def _counts(records):
    counts: dict[str, int] = {}
    for r in records:
        counts[r.my_category_tag] = counts.get(r.my_category_tag, 0) + 1
    return counts


def test_generate_count_matches_exactly():
    assert len(workload.generate(50)) == 50


def test_generate_uniform_distribution_is_balanced():
    counts = _counts(workload.generate(50))
    assert counts == {c: 10 for c in CATEGORIES}


def test_generate_respects_weights_and_sums_exactly():
    records = workload.generate(40, {"math": 3, "code": 1})
    assert len(records) == 40
    counts = _counts(records)
    assert counts.get("math", 0) == 30
    assert counts.get("code", 0) == 10
    assert "creative" not in counts


def test_largest_remainder_sums_exactly_for_awkward_total():
    # 37 over 5 uniform buckets can't divide evenly; largest-remainder must still
    # sum to exactly 37.
    plan = workload.plan_summary(37)
    assert sum(plan.values()) == 37
    assert len(workload.generate(37)) == 37


def test_plan_summary_matches_generate():
    plan = workload.plan_summary(37, {"math": 2, "easy-chitchat": 1})
    counts = _counts(workload.generate(37, {"math": 2, "easy-chitchat": 1}))
    assert plan == {**{c: 0 for c in CATEGORIES}, **counts}
    assert sum(plan.values()) == 37


def test_dry_run_10k_allocation_sums_to_10000():
    # the headline use: --num-requests 10000 allocates 10k across buckets.
    plan = workload.plan_summary(10000)
    assert sum(plan.values()) == 10000
    # uniform 10k over 5 buckets => 2000 each.
    assert plan == {c: 2000 for c in CATEGORIES}


def test_all_records_tagged_with_known_category():
    for r in workload.generate(25):
        assert r.my_category_tag in CATEGORIES
        assert r.prompt


def test_generate_is_deterministic():
    a = [(r.my_category_tag, r.prompt) for r in workload.generate(23, {"math": 2})]
    b = [(r.my_category_tag, r.prompt) for r in workload.generate(23, {"math": 2})]
    assert a == b


def test_conversations_share_id_and_have_turns():
    convs = workload.generate_conversations(5, turn_lengths=(2, 3))
    assert len(convs) == 5
    for conv in convs:
        assert len({t.conversation_id for t in conv}) == 1
        assert [t.turn_index for t in conv] == list(range(len(conv)))
        assert all(t.num_turns == len(conv) for t in conv)
        assert len(conv) >= 2  # multi-turn => >=2


def test_conversation_plan_total_turns_matches_generation():
    plan = workload.conversation_plan_summary(7, turn_lengths=(2, 4), seq_offset=11)
    convs = workload.generate_conversations(7, turn_lengths=(2, 4), seq_offset=11)
    actual_turns = sum(len(c) for c in convs)
    assert plan["total_turns"] == actual_turns
    assert plan["total_conversations"] == 7


def test_generate_mixed_user_ids_round_robin():
    singles, convs = workload.generate_mixed(6, 0, num_users=3)
    uids = {r.user_id for r in singles}
    assert uids == {"user-000", "user-001", "user-002"}
    # without --num-users every record has user_id None.
    singles2, _ = workload.generate_mixed(4, 0, num_users=0)
    assert all(r.user_id is None for r in singles2)


def test_easy_categories_seam():
    assert "easy-chitchat" in workload.easy_categories()
