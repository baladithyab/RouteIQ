"""Unit tests for the in-process routing stats accumulator (RouteIQ-aba9).

Covers record + read-back + reset + bounded eviction + thread-safety, plus
the singleton accessor contract.
"""

import threading

from litellm_llmrouter.router_decision_callback import (
    RoutingStatsAccumulator,
    get_stats_accumulator,
    reset_stats_accumulator,
)


class TestAccumulatorRecordAndReadBack:
    def test_record_increments_total(self):
        acc = RoutingStatsAccumulator()
        acc.record_decision(strategy="knn", model="gpt-4o")
        acc.record_decision(strategy="knn", model="gpt-4o")
        assert acc.global_snapshot()["total_decisions"] == 2

    def test_strategy_and_profile_distribution(self):
        acc = RoutingStatsAccumulator()
        acc.record_decision(strategy="knn", profile="auto")
        acc.record_decision(strategy="knn", profile="eco")
        acc.record_decision(strategy="mlp", profile="auto")
        snap = acc.global_snapshot()
        assert snap["strategy_distribution"] == {"knn": 2, "mlp": 1}
        assert snap["profile_distribution"] == {"auto": 2, "eco": 1}

    def test_centroid_decisions_counted(self):
        acc = RoutingStatsAccumulator()
        acc.record_decision(strategy="llmrouter-nadirclaw-centroid", model="m")
        acc.record_decision(strategy="knn", model="m")
        snap = acc.global_snapshot()
        assert snap["centroid_decisions"] == 1
        assert snap["total_decisions"] == 2

    def test_model_distribution(self):
        acc = RoutingStatsAccumulator()
        acc.record_decision(model="gpt-4o")
        acc.record_decision(model="gpt-4o")
        acc.record_decision(model="claude-haiku")
        assert acc.global_snapshot()["model_distribution"] == {
            "gpt-4o": 2,
            "claude-haiku": 1,
        }

    def test_average_latency(self):
        acc = RoutingStatsAccumulator()
        assert acc.average_latency_ms() == 0.0
        acc.record_latency(100.0)
        acc.record_latency(300.0)
        assert acc.average_latency_ms() == 200.0
        assert acc.global_snapshot()["average_latency_ms"] == 200.0

    def test_negative_latency_ignored(self):
        acc = RoutingStatsAccumulator()
        acc.record_latency(-5.0)
        assert acc.average_latency_ms() == 0.0

    def test_partial_telemetry_still_counts_total(self):
        """Middleware-style record with no model/strategy still bumps total."""
        acc = RoutingStatsAccumulator()
        acc.record_decision()
        snap = acc.global_snapshot()
        assert snap["total_decisions"] == 1
        assert snap["strategy_distribution"] == {}
        assert snap["model_distribution"] == {}


class TestPerKeyRollup:
    def test_key_snapshot_counts_and_recent_models(self):
        acc = RoutingStatsAccumulator()
        acc.record_decision(model="gpt-4o", key_id="k1")
        acc.record_decision(model="claude-haiku", key_id="k1")
        snap = acc.key_snapshot("k1")
        assert snap["decisions"] == 2
        # Most-recent-first.
        assert snap["recent_models"] == ["claude-haiku", "gpt-4o"]

    def test_key_snapshot_unknown_key_is_zeroed(self):
        acc = RoutingStatsAccumulator()
        snap = acc.key_snapshot("never-seen")
        assert snap == {"decisions": 0, "recent_models": []}

    def test_recent_models_bounded_and_dedup(self):
        acc = RoutingStatsAccumulator()
        # Record more than the per-key recent-models cap (10).
        for i in range(15):
            acc.record_decision(model=f"model-{i}", key_id="k1")
        snap = acc.key_snapshot("k1")
        assert snap["decisions"] == 15
        assert len(snap["recent_models"]) <= 10
        # Most recent model is surfaced first.
        assert snap["recent_models"][0] == "model-14"

    def test_repeated_model_moves_to_front_not_duplicated(self):
        acc = RoutingStatsAccumulator()
        acc.record_decision(model="a", key_id="k1")
        acc.record_decision(model="b", key_id="k1")
        acc.record_decision(model="a", key_id="k1")
        snap = acc.key_snapshot("k1")
        assert snap["decisions"] == 3
        assert snap["recent_models"] == ["a", "b"]

    def test_user_distribution_tracked(self):
        acc = RoutingStatsAccumulator()
        acc.record_decision(user_id="u1")
        acc.record_decision(user_id="u1")
        acc.record_decision(user_id="u2")
        # user counts are internal; global key_distribution unaffected.
        snap = acc.global_snapshot()
        assert snap["total_decisions"] == 3


class TestBoundedMemory:
    def test_key_rollups_capped(self):
        acc = RoutingStatsAccumulator(max_entries=5)
        for i in range(20):
            acc.record_decision(model="m", key_id=f"k{i}")
        snap = acc.global_snapshot()
        assert snap["tracked_keys"] == 5
        # Oldest evicted; newest retained.
        assert "k19" in snap["key_distribution"]
        assert "k0" not in snap["key_distribution"]

    def test_model_distribution_capped(self):
        acc = RoutingStatsAccumulator(max_entries=3)
        for i in range(10):
            acc.record_decision(model=f"model-{i}")
        assert len(acc.global_snapshot()["model_distribution"]) == 3


class TestSingletonAndReset:
    def test_singleton_identity(self):
        a = get_stats_accumulator()
        b = get_stats_accumulator()
        assert a is b

    def test_reset_clears_state(self):
        acc = get_stats_accumulator()
        acc.record_decision(strategy="knn", model="m", key_id="k1")
        assert acc.global_snapshot()["total_decisions"] == 1
        reset_stats_accumulator()
        fresh = get_stats_accumulator()
        assert fresh is not acc
        assert fresh.global_snapshot()["total_decisions"] == 0


class TestThreadSafety:
    def test_concurrent_records_are_consistent(self):
        acc = RoutingStatsAccumulator()
        n_threads = 8
        per_thread = 500

        def worker():
            for _ in range(per_thread):
                acc.record_decision(strategy="knn", model="m", key_id="shared")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = acc.global_snapshot()
        assert snap["total_decisions"] == n_threads * per_thread
        assert snap["strategy_distribution"]["knn"] == n_threads * per_thread
        assert acc.key_snapshot("shared")["decisions"] == n_threads * per_thread
