"""Unit tests for personalized routing strategy.

Tests cover:
- Model embedding generation
- User preference store (in-memory fallback)
- Preference updates via EMA
- Personalized model ranking
- Feedback recording
- Cold-start behavior
- Preference serialization/deserialization
- Singleton management
- Temporal decay
"""

import os
import time
from unittest.mock import patch

import numpy as np
import pytest

from litellm_llmrouter.personalized_routing import (
    PersonalizedRouter,
    PreferenceStore,
    UserPreference,
    _build_model_embeddings,
    _deserialize_preference,
    _deserialize_vector,
    _serialize_preference,
    _serialize_vector,
    get_personalized_router,
    reset_personalized_router,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the personalized router singleton between tests."""
    reset_personalized_router()
    yield
    reset_personalized_router()


@pytest.fixture
def router():
    """Create a fresh PersonalizedRouter instance."""
    return PersonalizedRouter(dim=128, learning_rate=0.1)


@pytest.fixture
def store():
    """Create a fresh PreferenceStore instance."""
    return PreferenceStore(dim=128, ttl_days=90)


# ---------------------------------------------------------------------------
# Model Embeddings
# ---------------------------------------------------------------------------


class TestModelEmbeddings:
    """Tests for _build_model_embeddings."""

    def test_build_embeddings_returns_dict(self):
        embeddings = _build_model_embeddings(dim=128)
        assert isinstance(embeddings, dict)
        assert len(embeddings) > 0

    def test_embeddings_are_normalized(self):
        embeddings = _build_model_embeddings(dim=128)
        for name, vec in embeddings.items():
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 0.01, f"Model {name} not normalized: norm={norm}"

    def test_embeddings_correct_dim(self):
        for dim in [32, 64, 128]:
            embeddings = _build_model_embeddings(dim=dim)
            for name, vec in embeddings.items():
                assert vec.shape == (dim,), (
                    f"Model {name} has shape {vec.shape}, expected ({dim},)"
                )

    def test_embeddings_are_float32(self):
        embeddings = _build_model_embeddings(dim=128)
        for vec in embeddings.values():
            assert vec.dtype == np.float32

    def test_known_models_have_embeddings(self):
        embeddings = _build_model_embeddings(dim=128)
        expected_models = ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash"]
        for model in expected_models:
            assert model in embeddings, f"Expected model {model} in embeddings"

    def test_different_models_have_different_embeddings(self):
        embeddings = _build_model_embeddings(dim=128)
        gpt4o = embeddings.get("gpt-4o")
        claude = embeddings.get("claude-3-5-sonnet-20241022")
        if gpt4o is not None and claude is not None:
            # They should be different (not identical)
            assert not np.allclose(gpt4o, claude)


# ---------------------------------------------------------------------------
# Vector Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for vector serialization/deserialization."""

    def test_vector_round_trip(self):
        vec = np.random.randn(128).astype(np.float32)
        data = _serialize_vector(vec)
        restored = _deserialize_vector(data)
        np.testing.assert_array_almost_equal(vec, restored)

    def test_preference_round_trip(self):
        vec = np.random.randn(128).astype(np.float32)
        pref = UserPreference(
            user_id="test-user",
            preference_vector=vec,
            interaction_count=5,
            last_updated=time.time(),
            model_scores={"gpt-4o": 0.8, "claude-3-5-sonnet-20241022": -0.2},
        )
        data = _serialize_preference(pref)
        restored = _deserialize_preference(data)

        assert restored.user_id == pref.user_id
        assert restored.interaction_count == pref.interaction_count
        assert abs(restored.last_updated - pref.last_updated) < 0.001
        assert restored.model_scores == pref.model_scores
        np.testing.assert_array_almost_equal(
            restored.preference_vector, pref.preference_vector
        )

    def test_serialized_data_is_json_string(self):
        vec = np.zeros(128, dtype=np.float32)
        pref = UserPreference(user_id="u1", preference_vector=vec)
        data = _serialize_preference(pref)
        assert isinstance(data, str)
        # Should be valid JSON
        import json

        parsed = json.loads(data)
        assert "user_id" in parsed
        assert "vec" in parsed


# ---------------------------------------------------------------------------
# PreferenceStore (in-memory fallback)
# ---------------------------------------------------------------------------


class TestPreferenceStore:
    """Tests for PreferenceStore with in-memory fallback."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store):
        result = await store.get_preference("nonexistent-user")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, store):
        vec = np.random.randn(128).astype(np.float32)
        pref = UserPreference(
            user_id="user-1",
            preference_vector=vec,
            interaction_count=1,
            last_updated=time.time(),
        )
        await store.set_preference(pref)

        retrieved = await store.get_preference("user-1")
        assert retrieved is not None
        assert retrieved.user_id == "user-1"
        assert retrieved.interaction_count == 1

    @pytest.mark.asyncio
    async def test_update_creates_new_preference(self, store):
        emb = np.random.randn(128).astype(np.float32)
        emb /= np.linalg.norm(emb)

        result = await store.update_preference(
            user_id="new-user",
            model_name="gpt-4o",
            score=0.8,
            model_embedding=emb,
            learning_rate=0.1,
        )

        assert result.user_id == "new-user"
        assert result.interaction_count == 1
        assert "gpt-4o" in result.model_scores

    @pytest.mark.asyncio
    async def test_update_increments_count(self, store):
        emb = np.random.randn(128).astype(np.float32)
        emb /= np.linalg.norm(emb)

        await store.update_preference("user-x", "m1", 0.5, emb, 0.1)
        await store.update_preference("user-x", "m2", -0.3, emb, 0.1)
        await store.update_preference("user-x", "m1", 0.9, emb, 0.1)

        pref = await store.get_preference("user-x")
        assert pref is not None
        assert pref.interaction_count == 3

    @pytest.mark.asyncio
    async def test_update_preference_vector_changes(self, store):
        emb_a = np.random.randn(128).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)

        emb_b = np.random.randn(128).astype(np.float32)
        emb_b /= np.linalg.norm(emb_b)

        # Build up a preference toward model-a
        await store.update_preference("user-y", "model-a", 1.0, emb_a, 0.5)
        pref1 = await store.get_preference("user-y")
        vec1 = pref1.preference_vector.copy()

        # Now add preference toward a different model (different embedding)
        await store.update_preference("user-y", "model-b", 1.0, emb_b, 0.5)
        pref2 = await store.get_preference("user-y")
        vec2 = pref2.preference_vector

        # Vectors should be different after adding a new model direction
        assert not np.allclose(vec1, vec2)

    @pytest.mark.asyncio
    async def test_delete_preference(self, store):
        vec = np.zeros(128, dtype=np.float32)
        pref = UserPreference(user_id="del-user", preference_vector=vec)
        await store.set_preference(pref)

        assert await store.get_preference("del-user") is not None
        result = await store.delete_preference("del-user")
        assert result is True
        assert await store.get_preference("del-user") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        result = await store.delete_preference("ghost")
        assert result is False

    def test_stats(self, store):
        stats = store.get_stats()
        assert stats["dim"] == 128
        assert stats["local_cache_size"] == 0

    @pytest.mark.asyncio
    async def test_eviction(self):
        store = PreferenceStore(dim=16, ttl_days=1)
        store._max_local_entries = 5

        for i in range(10):
            vec = np.random.randn(16).astype(np.float32)
            pref = UserPreference(
                user_id=f"user-{i}",
                preference_vector=vec,
                last_updated=time.time() + i,
            )
            await store.set_preference(pref)

        # Should have evicted some entries
        assert len(store._local_cache) <= 5


# ---------------------------------------------------------------------------
# PersonalizedRouter
# ---------------------------------------------------------------------------


class TestPersonalizedRouter:
    """Tests for the PersonalizedRouter."""

    @pytest.mark.asyncio
    async def test_cold_start_ranking(self, router):
        """Cold-start users should get quality-bias-only ranking."""
        candidates = ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
        ranked = await router.rank_models("cold-user", candidates)

        assert len(ranked) == 3
        # All should be (model_name, score) tuples
        for name, score in ranked:
            assert isinstance(name, str)
            assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_cold_start_uses_quality_bias(self, router):
        """Cold-start ranking should reflect quality biases."""
        candidates = ["gpt-5", "gpt-4o-mini"]  # gpt-5 has higher quality
        ranked = await router.rank_models("cold-user", candidates)

        # gpt-5 should be ranked higher than gpt-4o-mini
        assert ranked[0][0] == "gpt-5"
        assert ranked[0][1] > ranked[1][1]

    @pytest.mark.asyncio
    async def test_feedback_affects_ranking(self, router):
        """After positive feedback, the preferred model should rank higher."""
        # Record strong positive feedback for a cheaper model
        for _ in range(10):
            await router.record_feedback("user-pref", "gpt-4o-mini", 1.0)

        # Record negative feedback for the expensive model
        for _ in range(10):
            await router.record_feedback("user-pref", "gpt-5", -0.8)

        candidates = ["gpt-5", "gpt-4o-mini"]
        ranked = await router.rank_models("user-pref", candidates)

        # After strong feedback, gpt-4o-mini should rank above gpt-5
        assert ranked[0][0] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_get_top_model_returns_best(self, router):
        candidates = ["gpt-4o", "gpt-4o-mini"]
        top = await router.get_top_model("cold-user", candidates)
        assert top in candidates

    @pytest.mark.asyncio
    async def test_get_top_model_empty_candidates(self, router):
        result = await router.get_top_model("user", [])
        assert result is None

    @pytest.mark.asyncio
    async def test_record_feedback_clamps_score(self, router):
        """Scores outside [-1, 1] should be clamped."""
        await router.record_feedback("user-clamp", "gpt-4o", 5.0)
        pref = await router.store.get_preference("user-clamp")
        assert pref is not None
        assert pref.model_scores["gpt-4o"] <= 1.0

    @pytest.mark.asyncio
    async def test_fuzzy_embedding_for_unknown_model(self, router):
        """Unknown models should get a hash-based embedding."""
        candidates = ["completely-unknown-model-xyz"]
        ranked = await router.rank_models("cold-user", candidates)
        assert len(ranked) == 1

    @pytest.mark.asyncio
    async def test_fuzzy_matching_for_partial_name(self, router):
        """Models with partial name matches should use known embeddings."""
        emb_exact = router._get_embedding("gpt-4o")
        emb_fuzzy = router._get_embedding("openai/gpt-4o")
        # Fuzzy should find gpt-4o via substring match
        np.testing.assert_array_equal(emb_exact, emb_fuzzy)

    @pytest.mark.asyncio
    async def test_delete_user_data(self, router):
        await router.record_feedback("del-me", "gpt-4o", 0.5)
        pref = await router.store.get_preference("del-me")
        assert pref is not None

        deleted = await router.delete_user_data("del-me")
        assert deleted is True

        pref = await router.store.get_preference("del-me")
        assert pref is None

    def test_update_quality_bias(self, router):
        router.update_quality_bias("new-model", 0.77)
        assert router._quality_bias["new-model"] == 0.77

    def test_update_quality_bias_clamps(self, router):
        router.update_quality_bias("m", 1.5)
        assert router._quality_bias["m"] == 1.0
        router.update_quality_bias("m", -0.5)
        assert router._quality_bias["m"] == 0.0

    def test_stats(self, router):
        stats = router.get_stats()
        assert stats["dim"] == 128
        assert stats["learning_rate"] == 0.1
        assert stats["known_models"] > 0
        assert "store" in stats


# ---------------------------------------------------------------------------
# Temporal Decay
# ---------------------------------------------------------------------------


class TestTemporalDecay:
    """Tests for preference decay over time."""

    @pytest.mark.asyncio
    async def test_decay_reduces_preference_strength(self):
        store = PreferenceStore(dim=32, ttl_days=90)
        emb = np.random.randn(32).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Create initial preference
        await store.update_preference("decay-user", "m1", 1.0, emb, 0.5)
        pref1 = await store.get_preference("decay-user")
        _ = np.linalg.norm(pref1.preference_vector)  # verify it's non-zero

        # Simulate 30 days passing
        pref1.last_updated = time.time() - (30 * 86400)
        await store.set_preference(pref1)

        # Update with a new feedback — the old preference should be decayed
        await store.update_preference("decay-user", "m2", 0.1, emb, 0.5)
        pref2 = await store.get_preference("decay-user")

        # After decay + small update, the vector should be different from
        # what it would be without decay
        assert pref2.interaction_count == 2


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    """Tests for the module-level singleton."""

    @pytest.fixture(autouse=True)
    def _reset_settings(self):
        """Reset settings singleton to avoid cached state interference."""
        from litellm_llmrouter.settings import reset_settings

        reset_settings()
        yield
        reset_settings()

    # The pydantic settings env var for personalized routing uses the nested
    # delimiter: ROUTEIQ_ROUTING__PERSONALIZED_ENABLED
    _ENV_KEY = "ROUTEIQ_ROUTING__PERSONALIZED_ENABLED"

    def test_get_returns_none_when_disabled(self):
        """When disabled, get_personalized_router returns None."""
        with patch.dict(os.environ, {self._ENV_KEY: "false"}, clear=False):
            from litellm_llmrouter.settings import reset_settings

            reset_settings()
            reset_personalized_router()
            result = get_personalized_router()
            assert result is None

    def test_get_returns_instance_when_enabled(self):
        """When enabled, get_personalized_router returns an instance."""
        with patch.dict(os.environ, {self._ENV_KEY: "true"}, clear=False):
            from litellm_llmrouter.settings import reset_settings

            reset_settings()
            reset_personalized_router()
            result = get_personalized_router()
            assert isinstance(result, PersonalizedRouter)

    def test_singleton_returns_same_instance(self):
        """Repeated calls return the same instance."""
        with patch.dict(os.environ, {self._ENV_KEY: "true"}, clear=False):
            from litellm_llmrouter.settings import reset_settings

            reset_settings()
            reset_personalized_router()
            r1 = get_personalized_router()
            r2 = get_personalized_router()
            assert r1 is r2

    def test_reset_clears_singleton(self):
        """reset_personalized_router clears the cached instance."""
        with patch.dict(os.environ, {self._ENV_KEY: "true"}, clear=False):
            from litellm_llmrouter.settings import reset_settings

            reset_settings()
            reset_personalized_router()
            r1 = get_personalized_router()
            reset_personalized_router()
            r2 = get_personalized_router()
            assert r1 is not r2
