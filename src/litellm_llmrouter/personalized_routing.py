"""Personalized Routing Strategy for RouteIQ.

Learns per-user/per-team model preferences from feedback signals and adapts
routing decisions accordingly. Cold-start users fall back to centroid routing.

Architecture:
  - Per-user preference vectors stored in Redis (128-dim float32)
  - On each request: score = dot(user_pref, model_embedding) + quality_bias
  - User preferences updated online via exponential moving average from feedback
  - Feedback collected via POST /api/v1/routeiq/routing/feedback endpoint

Design inspired by GMTRouter's cross-attention concept but simplified:
  - No graph neural network (too complex for hot-path gateway)
  - No PyTorch-geometric dependency
  - Pure NumPy for scoring (~0.5ms per request)
  - Redis for preference storage (works across workers)

Configuration:
  ROUTEIQ_PERSONALIZED_ROUTING=true
  ROUTEIQ_PREFERENCE_DIM=128
  ROUTEIQ_PREFERENCE_LEARNING_RATE=0.1
  ROUTEIQ_PREFERENCE_DECAY=0.99 (per-day decay to forget stale preferences)
"""

import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("litellm_llmrouter.personalized_routing")

# ---------------------------------------------------------------------------
# Optional dependency handling
# ---------------------------------------------------------------------------

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PREFERENCE_DIM = int(os.getenv("ROUTEIQ_PREFERENCE_DIM", "128"))
_LEARNING_RATE = float(os.getenv("ROUTEIQ_PREFERENCE_LEARNING_RATE", "0.1"))
_DECAY_PER_DAY = float(os.getenv("ROUTEIQ_PREFERENCE_DECAY", "0.99"))
_REDIS_KEY_PREFIX = "routeiq:pref:"
_REDIS_TTL_DAYS = 90

# Quality biases — global average quality score per model (empirical).
# These provide a reasonable default ranking for cold-start users.
_DEFAULT_QUALITY_BIASES: Dict[str, float] = {
    # GPT-4o family
    "gpt-4o": 0.82,
    "gpt-4o-mini": 0.65,
    "gpt-4-turbo": 0.78,
    # GPT-4.1 family
    "gpt-4.1": 0.85,
    "gpt-4.1-mini": 0.70,
    "gpt-4.1-nano": 0.55,
    # GPT-5 family
    "gpt-5": 0.92,
    "gpt-5-mini": 0.75,
    "gpt-5.2": 0.90,
    "gpt-5.4": 0.93,
    # o-series
    "o1": 0.88,
    "o1-mini": 0.72,
    "o3-mini": 0.74,
    "o4-mini": 0.76,
    # Claude 3.x
    "claude-3-5-sonnet-20241022": 0.86,
    "claude-3-5-haiku-20241022": 0.68,
    "claude-3-opus-20240229": 0.88,
    "claude-sonnet-4-20250514": 0.87,
    # Claude 4.x
    "claude-opus-4-6-20250918": 0.91,
    "claude-sonnet-4-5-20250929": 0.88,
    "claude-haiku-4-5-20251001": 0.72,
    # Gemini
    "gemini-2.0-flash": 0.62,
    "gemini-2.5-pro-preview-05-06": 0.84,
    "gemini-1.5-pro": 0.80,
    "gemini-3-flash-preview": 0.70,
    # DeepSeek
    "deepseek-chat": 0.64,
    "deepseek-reasoner": 0.76,
}


# ── Model Embeddings (pre-computed, static) ──────────────────────────────


def _build_model_embeddings(dim: int = 128) -> Dict[str, "np.ndarray"]:
    """Build static model embedding vectors from model registry data.

    Each model's embedding encodes: cost tier, context window, capabilities,
    provider family, and random hash-based features for diversity.

    Args:
        dim: Dimensionality of the embedding vectors.

    Returns:
        Dict mapping model name to L2-normalized embedding vector.
    """
    if not _NUMPY_AVAILABLE:
        return {}

    from litellm_llmrouter.centroid_routing import (
        MODEL_CAPABILITIES,
        MODEL_CONTEXT_WINDOWS,
        MODEL_COSTS,
    )

    embeddings: Dict[str, np.ndarray] = {}
    all_models = set(list(MODEL_COSTS.keys()) + list(MODEL_CONTEXT_WINDOWS.keys()))

    for model_name in all_models:
        vec = np.zeros(dim, dtype=np.float32)

        # Cost features (dims 0-15)
        costs = MODEL_COSTS.get(model_name, {"input": 1.0, "output": 3.0})
        vec[0] = np.log1p(costs.get("input", 1.0))
        vec[1] = np.log1p(costs.get("output", 3.0))
        vec[2] = costs.get("input", 1.0) / 20.0  # normalized
        vec[3] = costs.get("output", 3.0) / 80.0
        # Cost ratio (output/input) — captures pricing asymmetry
        input_cost = costs.get("input", 1.0)
        if input_cost > 0:
            vec[4] = min(costs.get("output", 3.0) / input_cost, 10.0) / 10.0
        # Total cost bucket (dims 5-8, one-hot for cheap/mid/premium/luxury)
        total_cost = input_cost + costs.get("output", 3.0)
        if total_cost < 1.0:
            vec[5] = 1.0  # cheap
        elif total_cost < 5.0:
            vec[6] = 1.0  # mid
        elif total_cost < 20.0:
            vec[7] = 1.0  # premium
        else:
            vec[8] = 1.0  # luxury

        # Context window features (dims 16-31)
        if dim > 16:
            ctx = MODEL_CONTEXT_WINDOWS.get(model_name, 128_000)
            vec[16] = np.log1p(ctx) / 15.0  # normalized log
            if dim > 17:
                vec[17] = 1.0 if ctx >= 200_000 else 0.0  # large context flag
            if dim > 18:
                vec[18] = 1.0 if ctx >= 1_000_000 else 0.0  # million+ flag
            if dim > 19:
                vec[19] = min(ctx / 2_100_000, 1.0)  # normalized raw

        # Capability features (dims 32-63)
        if dim > 32:
            caps = MODEL_CAPABILITIES.get(model_name, {"text", "streaming"})
            cap_map = {
                "text": 32,
                "vision": 33,
                "function_calling": 34,
                "json_mode": 35,
                "streaming": 36,
                "reasoning": 37,
                "computer_use": 38,
            }
            for cap, idx in cap_map.items():
                if cap in caps and idx < dim:
                    vec[idx] = 1.0
            # Capability count — more capable models get a higher score
            if dim > 39:
                vec[39] = len(caps) / 8.0

        # Provider family features (dims 64-79) — hash-based
        if dim > 64:
            provider = model_name.split("-")[0] if "-" in model_name else model_name[:3]
            h = hashlib.md5(provider.encode()).digest()  # noqa: S324
            for i in range(min(16, dim - 64)):
                vec[64 + i] = (h[i % len(h)] / 255.0) - 0.5

        # Model identity features (dims 80-127) — hash-based for diversity
        if dim > 80:
            h2 = hashlib.md5(model_name.encode()).digest()  # noqa: S324
            for i in range(min(48, dim - 80)):
                vec[80 + i] = (h2[i % len(h2)] / 255.0) - 0.5

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        embeddings[model_name] = vec

    return embeddings


# ── User Preference Store ────────────────────────────────────────────────


@dataclass
class UserPreference:
    """Per-user model preference state."""

    user_id: str
    preference_vector: Any  # np.ndarray — typed as Any for non-numpy envs
    interaction_count: int = 0
    last_updated: float = 0.0
    model_scores: Dict[str, float] = field(default_factory=dict)


def _serialize_vector(vec: "np.ndarray") -> bytes:
    """Serialize a numpy float32 vector to compact bytes.

    Format: 4-byte little-endian int (dim) + dim * 4-byte float32 values.
    """
    dim = vec.shape[0]
    header = struct.pack("<I", dim)
    return header + vec.astype(np.float32).tobytes()


def _deserialize_vector(data: bytes) -> "np.ndarray":
    """Deserialize bytes back to a numpy float32 vector."""
    (dim,) = struct.unpack("<I", data[:4])
    return np.frombuffer(data[4 : 4 + dim * 4], dtype=np.float32).copy()


def _serialize_preference(pref: UserPreference) -> str:
    """Serialize a UserPreference to a JSON string with base64 vector.

    The preference vector is stored as hex-encoded bytes for Redis
    compatibility (Redis strings are binary-safe but JSON is not).
    """
    import base64

    vec_bytes = _serialize_vector(pref.preference_vector)
    return json.dumps(
        {
            "user_id": pref.user_id,
            "vec": base64.b64encode(vec_bytes).decode("ascii"),
            "interaction_count": pref.interaction_count,
            "last_updated": pref.last_updated,
            "model_scores": pref.model_scores,
        }
    )


def _deserialize_preference(data: str) -> UserPreference:
    """Deserialize a JSON string back to a UserPreference."""
    import base64

    obj = json.loads(data)
    vec_bytes = base64.b64decode(obj["vec"])
    return UserPreference(
        user_id=obj["user_id"],
        preference_vector=_deserialize_vector(vec_bytes),
        interaction_count=obj.get("interaction_count", 0),
        last_updated=obj.get("last_updated", 0.0),
        model_scores=obj.get("model_scores", {}),
    )


class PreferenceStore:
    """Redis-backed store for user preference vectors.

    Falls back to in-memory dict when Redis is unavailable.
    Thread-safe for the in-memory fallback via dict operations being
    atomic in CPython. Redis operations are inherently atomic.
    """

    def __init__(self, dim: int = 128, ttl_days: int = 90):
        self._dim = dim
        self._ttl_seconds = ttl_days * 86400
        self._local_cache: Dict[str, UserPreference] = {}
        self._max_local_entries = 10_000

    def _redis_key(self, user_id: str) -> str:
        """Build the Redis key for a user's preference."""
        return f"{_REDIS_KEY_PREFIX}{user_id}"

    async def _get_redis(self) -> Any:
        """Get the async Redis client, or None if unavailable."""
        try:
            from litellm_llmrouter.redis_pool import get_async_redis_client

            return await get_async_redis_client()
        except Exception:
            return None

    async def get_preference(self, user_id: str) -> Optional[UserPreference]:
        """Get user preference, checking Redis first, then local cache.

        Args:
            user_id: The user or team identifier.

        Returns:
            UserPreference if found, None for cold-start users.
        """
        # Check local cache first (hot path)
        cached = self._local_cache.get(user_id)
        if cached is not None:
            return cached

        # Try Redis
        redis = await self._get_redis()
        if redis is not None:
            try:
                data = await redis.get(self._redis_key(user_id))
                if data is not None:
                    pref = _deserialize_preference(data)
                    # Populate local cache
                    self._local_cache[user_id] = pref
                    self._evict_local_cache()
                    return pref
            except Exception as e:
                logger.debug("Redis GET failed for user %s: %s", user_id, e)

        return None

    async def set_preference(self, pref: UserPreference) -> None:
        """Store user preference in Redis + local cache.

        Args:
            pref: The user preference to store.
        """
        # Always update local cache
        self._local_cache[pref.user_id] = pref
        self._evict_local_cache()

        # Persist to Redis
        redis = await self._get_redis()
        if redis is not None:
            try:
                data = _serialize_preference(pref)
                await redis.set(
                    self._redis_key(pref.user_id),
                    data,
                    ex=self._ttl_seconds,
                )
            except Exception as e:
                logger.debug("Redis SET failed for user %s: %s", pref.user_id, e)

    async def update_preference(
        self,
        user_id: str,
        model_name: str,
        score: float,
        model_embedding: "np.ndarray",
        learning_rate: float = 0.1,
    ) -> UserPreference:
        """Update user preference based on feedback.

        Uses exponential moving average:
          if score > 0 (positive feedback):
            pref += lr * score * model_embedding
          if score < 0 (negative feedback):
            pref -= lr * abs(score) * model_embedding
          normalize(pref)

        Also applies temporal decay to forget stale preferences:
          days_since_update = (now - last_updated) / 86400
          pref *= decay_per_day ** days_since_update

        Args:
            user_id: The user or team identifier.
            model_name: The model that received feedback.
            score: Feedback score in [-1.0, 1.0].
            model_embedding: The model's embedding vector.
            learning_rate: Step size for the update.

        Returns:
            The updated UserPreference.
        """
        pref = await self.get_preference(user_id)
        now = time.time()

        if pref is None:
            # Cold start — initialize with a small nudge toward the feedback
            vec = np.zeros(self._dim, dtype=np.float32)
            vec += learning_rate * score * model_embedding
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            pref = UserPreference(
                user_id=user_id,
                preference_vector=vec,
                interaction_count=1,
                last_updated=now,
                model_scores={model_name: score},
            )
        else:
            # Apply temporal decay
            if pref.last_updated > 0:
                days_elapsed = (now - pref.last_updated) / 86400.0
                if days_elapsed > 0:
                    decay = _DECAY_PER_DAY**days_elapsed
                    pref.preference_vector = pref.preference_vector * decay

            # Apply EMA update
            pref.preference_vector = (
                pref.preference_vector + learning_rate * score * model_embedding
            )

            # Re-normalize
            norm = np.linalg.norm(pref.preference_vector)
            if norm > 0:
                pref.preference_vector = pref.preference_vector / norm

            pref.interaction_count += 1
            pref.last_updated = now

            # Track per-model cumulative scores (EMA)
            prev = pref.model_scores.get(model_name, 0.0)
            pref.model_scores[model_name] = 0.8 * prev + 0.2 * score

        await self.set_preference(pref)
        return pref

    async def delete_preference(self, user_id: str) -> bool:
        """Delete a user's preference data.

        Args:
            user_id: The user or team identifier.

        Returns:
            True if a preference was found and deleted.
        """
        found = user_id in self._local_cache
        self._local_cache.pop(user_id, None)

        redis = await self._get_redis()
        if redis is not None:
            try:
                result = await redis.delete(self._redis_key(user_id))
                found = found or (result > 0)
            except Exception as e:
                logger.debug("Redis DELETE failed for user %s: %s", user_id, e)

        return found

    def _evict_local_cache(self) -> None:
        """Evict oldest entries if local cache exceeds max size."""
        if len(self._local_cache) > self._max_local_entries:
            # Remove oldest 10% by last_updated
            entries = sorted(
                self._local_cache.items(),
                key=lambda x: x[1].last_updated,
            )
            to_remove = len(entries) - self._max_local_entries
            for key, _ in entries[:to_remove]:
                self._local_cache.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics for admin introspection."""
        return {
            "local_cache_size": len(self._local_cache),
            "max_local_entries": self._max_local_entries,
            "dim": self._dim,
            "ttl_days": self._ttl_seconds // 86400,
        }


# ── Personalized Router ──────────────────────────────────────────────────


class PersonalizedRouter:
    """Scores models based on user preferences.

    For each candidate model:
      score = preference_weight * dot(user_preference, model_embedding)
            + quality_weight * quality_bias

    Returns models ranked by preference score.

    Attributes:
        _dim: Dimensionality of preference/embedding vectors.
        _lr: Learning rate for preference updates.
        _pref_weight: Weight for personalization score component.
        _quality_weight: Weight for global quality bias component.
    """

    def __init__(
        self,
        dim: int = 128,
        learning_rate: float = 0.1,
        pref_weight: float = 0.7,
        quality_weight: float = 0.3,
    ):
        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "numpy is required for personalized routing. "
                "Install with: pip install numpy"
            )

        self._dim = dim
        self._lr = learning_rate
        self._pref_weight = pref_weight
        self._quality_weight = quality_weight
        self._model_embeddings = _build_model_embeddings(dim)
        self._store = PreferenceStore(dim=dim, ttl_days=_REDIS_TTL_DAYS)
        self._quality_bias: Dict[str, float] = dict(_DEFAULT_QUALITY_BIASES)
        self._initialized = True

        logger.info(
            "PersonalizedRouter initialized (dim=%d, lr=%.3f, models=%d)",
            dim,
            learning_rate,
            len(self._model_embeddings),
        )

    @property
    def store(self) -> PreferenceStore:
        """Access the preference store (for admin/testing)."""
        return self._store

    async def rank_models(
        self,
        user_id: str,
        candidates: List[str],
    ) -> List[Tuple[str, float]]:
        """Rank candidate models by user preference.

        Returns list of (model_name, score) sorted by score descending.
        Cold-start users (no preference) get equal scores based on quality
        bias only.

        Args:
            user_id: The user or team identifier.
            candidates: List of candidate model names to rank.

        Returns:
            Sorted list of (model_name, combined_score) tuples.
        """
        pref = await self._store.get_preference(user_id)

        scores: List[Tuple[str, float]] = []
        for model in candidates:
            emb = self._get_embedding(model)
            quality = self._quality_bias.get(model)
            if quality is None:
                quality = self._get_fuzzy_quality(model)

            if pref is None:
                # Cold start — quality bias only
                scores.append((model, quality))
            else:
                # Preference score = dot product (both vectors are L2-normalized)
                pref_score = float(np.dot(pref.preference_vector, emb))
                # Combined score (weighted)
                combined = (
                    self._pref_weight * pref_score + self._quality_weight * quality
                )
                scores.append((model, combined))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    async def get_top_model(
        self,
        user_id: str,
        candidates: List[str],
    ) -> Optional[str]:
        """Get the top-ranked model for a user.

        Convenience method that returns just the best model name.

        Args:
            user_id: The user or team identifier.
            candidates: List of candidate model names.

        Returns:
            The highest-scored model name, or None if candidates is empty.
        """
        if not candidates:
            return None
        ranked = await self.rank_models(user_id, candidates)
        return ranked[0][0] if ranked else None

    async def record_feedback(
        self,
        user_id: str,
        model_name: str,
        score: float,
    ) -> None:
        """Record routing feedback to update user preferences.

        Args:
            user_id: The user or team identifier.
            model_name: The model that was used.
            score: Feedback score in [-1.0, 1.0].
                   Positive = user liked the response.
                   Negative = user disliked the response.
        """
        # Clamp score to [-1.0, 1.0]
        score = max(-1.0, min(1.0, score))

        emb = self._get_embedding(model_name)
        await self._store.update_preference(user_id, model_name, score, emb, self._lr)

        logger.debug(
            "Recorded feedback: user=%s model=%s score=%.2f",
            user_id,
            model_name,
            score,
        )

    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all preference data for a user (GDPR/privacy).

        Args:
            user_id: The user or team identifier.

        Returns:
            True if data was found and deleted.
        """
        return await self._store.delete_preference(user_id)

    def update_quality_bias(self, model_name: str, bias: float) -> None:
        """Update the global quality bias for a model.

        Args:
            model_name: The model name.
            bias: Quality score in [0.0, 1.0].
        """
        self._quality_bias[model_name] = max(0.0, min(1.0, bias))

    def _get_embedding(self, model_name: str) -> "np.ndarray":
        """Get embedding for a model, with fuzzy matching fallback.

        Args:
            model_name: The model identifier.

        Returns:
            L2-normalized embedding vector.
        """
        # Exact match
        emb = self._model_embeddings.get(model_name)
        if emb is not None:
            return emb

        # Fuzzy match — check if model_name is a substring or prefix
        return self._get_fuzzy_embedding(model_name)

    def _get_fuzzy_embedding(self, model_name: str) -> "np.ndarray":
        """Get embedding with fuzzy matching, or generate a hash-based default.

        Tries substring matching against known models first. Falls back
        to a deterministic hash-based embedding so unknown models still
        get a stable (if arbitrary) embedding.

        Args:
            model_name: The model identifier.

        Returns:
            L2-normalized embedding vector.
        """
        # Try substring match
        for key, emb in self._model_embeddings.items():
            if key in model_name or model_name in key:
                return emb

        # Hash-based default — deterministic for the same model_name
        h = hashlib.md5(model_name.encode()).digest()  # noqa: S324
        vec = np.array(
            [h[i % len(h)] / 255.0 - 0.5 for i in range(self._dim)],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        # Cache for future lookups (bounded to prevent memory leak)
        if len(self._model_embeddings) < 1000:
            self._model_embeddings[model_name] = vec
        return vec

    def _get_fuzzy_quality(self, model_name: str) -> float:
        """Get quality bias with fuzzy matching.

        Args:
            model_name: The model identifier.

        Returns:
            Quality bias score.
        """
        for key, bias in self._quality_bias.items():
            if key in model_name or model_name in key:
                return bias
        return 0.5  # neutral default for unknown models

    def get_stats(self) -> Dict[str, Any]:
        """Return router statistics for admin introspection."""
        return {
            "dim": self._dim,
            "learning_rate": self._lr,
            "pref_weight": self._pref_weight,
            "quality_weight": self._quality_weight,
            "known_models": len(self._model_embeddings),
            "quality_biases": len(self._quality_bias),
            "store": self._store.get_stats(),
        }


# ── Singleton ────────────────────────────────────────────────────────────

_router: Optional[PersonalizedRouter] = None


def _is_personalized_routing_enabled() -> bool:
    """Check if personalized routing is enabled via settings or env var."""
    try:
        from litellm_llmrouter.settings import get_settings

        settings = get_settings()
        return getattr(
            getattr(settings, "routing", None),
            "personalized_enabled",
            False,
        )
    except Exception:
        return os.getenv("ROUTEIQ_PERSONALIZED_ROUTING", "false").lower() == "true"


def get_personalized_router() -> Optional[PersonalizedRouter]:
    """Get or create the personalized router singleton.

    Returns None if personalized routing is disabled or numpy is unavailable.

    Returns:
        PersonalizedRouter instance, or None if disabled.
    """
    global _router
    if _router is not None:
        return _router

    if not _is_personalized_routing_enabled():
        return None

    if not _NUMPY_AVAILABLE:
        logger.warning(
            "Personalized routing enabled but numpy not available — skipping"
        )
        return None

    try:
        _router = PersonalizedRouter(
            dim=_PREFERENCE_DIM,
            learning_rate=_LEARNING_RATE,
        )
        return _router
    except Exception as e:
        logger.warning("Failed to create PersonalizedRouter: %s", e)
        return None


def reset_personalized_router() -> None:
    """Reset the personalized router singleton.

    **Must** be called in test fixtures (``autouse=True``) to prevent
    cross-test contamination.
    """
    global _router
    _router = None
