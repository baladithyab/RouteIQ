"""Unit tests for the cluster-wide shared stats store (RouteIQ-78fd).

The per-worker :class:`RoutingStatsAccumulator` only ever shows the worker that
served a scrape.  This store mirrors each worker's decision counters into a
shared Redis store and reads back the SUM across workers.  These tests prove:

  * two "workers" (two mirror calls) contribute to a single aggregate,
  * the global snapshot reflects BOTH workers' decisions (mock the shared store),
  * Redis-absent / disabled falls back to the per-worker snapshot (fail-open).

The shared store is mocked with an ``AsyncMock`` Redis backed by an in-memory
dict (no real Redis, no credentials), mirroring ``test_governance_spend``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from litellm_llmrouter.router_decision_callback import (
    cluster_global_snapshot,
    get_stats_accumulator,
    mirror_decision_to_cluster,
    reset_cluster_stats,
)

# The cluster-stats functions import get_async_redis_client locally from
# redis_pool on each call, so patching the redis_pool symbol is what matters.
_REDIS_POOL = "litellm_llmrouter.redis_pool.get_async_redis_client"


def _make_fake_redis():
    """An AsyncMock Redis backed by an in-memory hash store.

    Supports HINCRBY / HGETALL / EXPIRE / DELETE and a pipeline() that buffers
    the same ops and applies them on execute() — enough to exercise the
    write-through + read-back paths cred-free.
    """
    store: dict[str, dict[str, int]] = {}

    def _hincrby(name, field, amount):
        h = store.setdefault(name, {})
        h[field] = h.get(field, 0) + amount
        return h[field]

    def _hgetall(name):
        # Return a fresh copy with str keys (mirrors decode_responses=True).
        return dict(store.get(name, {}))

    def _delete(*names):
        n = 0
        for name in names:
            if name in store:
                del store[name]
                n += 1
        return n

    redis = AsyncMock()
    redis.hincrby = AsyncMock(side_effect=_hincrby)
    redis.hgetall = AsyncMock(side_effect=_hgetall)
    redis.expire = AsyncMock(return_value=True)
    redis.delete = AsyncMock(side_effect=_delete)

    class _Pipe:
        def __init__(self):
            self._ops = []

        def hincrby(self, name, field, amount):
            self._ops.append(("hincrby", (name, field, amount)))
            return self

        def expire(self, name, ttl):
            self._ops.append(("expire", (name, ttl)))
            return self

        async def execute(self):
            results = []
            for op, args in self._ops:
                if op == "hincrby":
                    results.append(_hincrby(*args))
                else:
                    results.append(True)
            self._ops.clear()
            return results

    redis.pipeline = lambda *a, **k: _Pipe()
    redis._store = store
    return redis


@pytest.mark.asyncio
async def test_two_workers_contribute_to_single_aggregate():
    """Two mirror calls (two workers) sum into one cluster aggregate."""
    redis = _make_fake_redis()
    with patch(_REDIS_POOL, AsyncMock(return_value=redis)):
        # Worker A records one centroid decision for key k1 on gpt-4o.
        await mirror_decision_to_cluster(
            strategy="llmrouter-nadirclaw-centroid",
            model="gpt-4o",
            profile="auto",
            key_id="k1",
            centroid=True,
        )
        # Worker B records one knn decision for key k2 on claude-haiku.
        await mirror_decision_to_cluster(
            strategy="knn",
            model="claude-haiku",
            profile="eco",
            key_id="k2",
            centroid=False,
        )

        snap = await cluster_global_snapshot()

    assert snap["cluster_wide"] is True
    # Both workers' decisions are reflected in the single aggregate.
    assert snap["total_decisions"] == 2
    assert snap["centroid_decisions"] == 1
    assert snap["strategy_distribution"] == {
        "llmrouter-nadirclaw-centroid": 1,
        "knn": 1,
    }
    assert snap["model_distribution"] == {"gpt-4o": 1, "claude-haiku": 1}
    assert snap["profile_distribution"] == {"auto": 1, "eco": 1}
    assert snap["key_distribution"] == {"k1": 1, "k2": 1}
    assert snap["tracked_keys"] == 2


@pytest.mark.asyncio
async def test_same_key_across_workers_sums():
    """Two workers serving the SAME key sum that key's count cluster-wide."""
    redis = _make_fake_redis()
    with patch(_REDIS_POOL, AsyncMock(return_value=redis)):
        await mirror_decision_to_cluster(strategy="knn", model="m", key_id="shared")
        await mirror_decision_to_cluster(strategy="knn", model="m", key_id="shared")
        snap = await cluster_global_snapshot()

    assert snap["cluster_wide"] is True
    assert snap["total_decisions"] == 2
    assert snap["key_distribution"] == {"shared": 2}
    assert snap["strategy_distribution"] == {"knn": 2}
    assert snap["tracked_keys"] == 1


@pytest.mark.asyncio
async def test_redis_unavailable_falls_back_to_local_snapshot():
    """When Redis is absent, the snapshot falls back to the per-worker view."""
    # Seed the LOCAL accumulator so the fallback has content to return.
    get_stats_accumulator().record_decision(strategy="knn", model="m", key_id="k1")

    with patch(_REDIS_POOL, AsyncMock(return_value=None)):
        mirrored = await mirror_decision_to_cluster(strategy="knn", model="m")
        snap = await cluster_global_snapshot()

    assert mirrored is False  # nothing written when Redis is absent
    assert snap["cluster_wide"] is False  # flagged as per-worker
    assert snap["total_decisions"] == 1  # local accumulator value
    assert snap["key_distribution"] == {"k1": 1}


@pytest.mark.asyncio
async def test_cluster_stats_disabled_skips_mirror(monkeypatch):
    """ROUTEIQ_CLUSTER_STATS_ENABLED=false disables write-through + read."""
    monkeypatch.setenv("ROUTEIQ_CLUSTER_STATS_ENABLED", "false")
    redis = _make_fake_redis()
    get_stats_accumulator().record_decision(strategy="knn", model="m", key_id="k1")

    with patch(_REDIS_POOL, AsyncMock(return_value=redis)):
        mirrored = await mirror_decision_to_cluster(strategy="knn", model="m")
        snap = await cluster_global_snapshot()

    assert mirrored is False
    assert redis.hincrby.await_count == 0  # never touched Redis
    assert snap["cluster_wide"] is False
    assert snap["total_decisions"] == 1  # local fallback


@pytest.mark.asyncio
async def test_fresh_cluster_returns_local_when_empty():
    """An empty shared store yields the (also-empty) local snapshot, not error."""
    redis = _make_fake_redis()
    with patch(_REDIS_POOL, AsyncMock(return_value=redis)):
        snap = await cluster_global_snapshot()

    # Nothing mirrored yet -> falls back to the local snapshot (cluster_wide False).
    assert snap["cluster_wide"] is False
    assert snap["total_decisions"] == 0


@pytest.mark.asyncio
async def test_reset_cluster_stats_clears_shared_keys():
    """reset_cluster_stats deletes the shared keys (operator/test reset)."""
    redis = _make_fake_redis()
    with patch(_REDIS_POOL, AsyncMock(return_value=redis)):
        await mirror_decision_to_cluster(strategy="knn", model="m", key_id="k1")
        assert redis._store  # something was written
        ok = await reset_cluster_stats()
        assert ok is True
        snap = await cluster_global_snapshot()

    # After reset the shared store is empty -> local fallback.
    assert snap["cluster_wide"] is False
    assert snap["total_decisions"] == 0


@pytest.mark.asyncio
async def test_mirror_failopen_on_redis_error():
    """A Redis exception during mirror is swallowed (fail-open, returns False)."""
    redis = _make_fake_redis()

    def _boom(*a, **k):
        raise RuntimeError("redis down")

    redis.pipeline = _boom
    with patch(_REDIS_POOL, AsyncMock(return_value=redis)):
        result = await mirror_decision_to_cluster(strategy="knn", model="m")

    assert result is False
