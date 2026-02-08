"""
Semantic Cache: LLM Response Caching Infrastructure
=====================================================

Provides exact-match and semantic (embedding-based) caching for LLM responses.

Components:
- CacheKeyGenerator: Deterministic SHA-256 cache keys from request parameters
- CacheEntry: Serialized cache entry dataclass
- CacheStore protocol: Async get/set/get_similar interface
- InMemoryCache: L1 in-process LRU cache (OrderedDict-based)
- RedisCacheStore: L2 Redis cache with optional vector similarity

Cache keys include model, messages, temperature, top_p, max_tokens, and
other response-affecting parameters. Keys exclude stream, user, metadata,
and other non-deterministic fields.

Cacheability rules:
- Only cache when temperature <= threshold (default 0.1)
- Skip streaming requests
- Skip requests with no-cache header
- Skip empty or missing messages/model
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Parameters included in cache key (order doesn't matter, they are sorted)
CACHE_KEY_PARAMS: list[str] = [
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "max_tokens",
    "messages",
    "model",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
]

# Parameters explicitly excluded from cache key
CACHE_KEY_EXCLUDED: set[str] = {
    "api_key",
    "litellm_call_id",
    "litellm_params",
    "metadata",
    "proxy_server_request",
    "request_id",
    "stream",
    "timeout",
    "user",
}

# Default max cacheable temperature
DEFAULT_MAX_CACHEABLE_TEMPERATURE = 0.1


@dataclass
class CacheEntry:
    """Serialized cache entry for LLM responses."""

    response: dict[str, Any]
    """Serialized LLM response."""

    model: str
    """Model that generated the response."""

    created_at: float
    """Unix timestamp when entry was created."""

    token_count: int
    """Total tokens for cost tracking."""

    cache_key: str
    """Exact-match cache key."""

    embedding: list[float] | None = None
    """Embedding vector for semantic matching (None for exact-only)."""


def _normalize_value(value: Any) -> Any:
    """Normalize a parameter value for deterministic hashing."""
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    elif isinstance(value, float):
        return round(value, 10)
    elif isinstance(value, str):
        return value.strip()
    return value


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Normalize a chat message for cache key computation."""
    normalized: dict[str, Any] = {"role": msg.get("role", "")}

    content = msg.get("content")
    if content is not None:
        if isinstance(content, str):
            normalized["content"] = content.strip()
        elif isinstance(content, list):
            normalized["content"] = [_normalize_value(p) for p in content]
        else:
            normalized["content"] = content

    if msg.get("name"):
        normalized["name"] = str(msg["name"]).strip()

    if "tool_calls" in msg:
        normalized["tool_calls"] = sorted(
            msg["tool_calls"], key=lambda tc: tc.get("id", "")
        )

    if "tool_call_id" in msg:
        normalized["tool_call_id"] = msg["tool_call_id"]

    return normalized


class CacheKeyGenerator:
    """Generates cache keys from LLM request parameters."""

    @staticmethod
    def exact_key(
        model: str,
        messages: list[dict[str, Any]],
        user_id: str = "",
        team_id: str = "",
        **params: Any,
    ) -> str:
        """
        SHA-256 hash of canonicalized request parameters.

        Includes model, messages, temperature, top_p, max_tokens, and other
        response-affecting parameters. Excludes stream, user, metadata, etc.

        Cache keys are namespaced by user_id and team_id for PII isolation.

        Args:
            model: The LLM model name.
            messages: The chat messages list.
            user_id: User identifier for cache isolation.
            team_id: Team identifier for cache isolation.
            **params: Additional request parameters.

        Returns:
            Cache key in format "routeiq:cache:v1:{user}:{team}:<sha256_hex>".
        """
        canonical: dict[str, Any] = {}

        # Always include model
        if model:
            canonical["model"] = model.strip()

        # Normalize and include messages
        if messages:
            canonical["messages"] = [_normalize_message(m) for m in messages]

        # Include other cache-relevant parameters
        for param in CACHE_KEY_PARAMS:
            if param in ("model", "messages"):
                continue  # Already handled
            if param in params and params[param] is not None:
                canonical[param] = _normalize_value(params[param])

        canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        key_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        # Namespace by user/team for PII isolation
        uid = user_id or "_"
        tid = team_id or "_"
        return f"routeiq:cache:v1:{uid}:{tid}:{key_hash}"

    @staticmethod
    async def semantic_key(
        model: str,
        messages: list[dict[str, Any]],
        embedder: Any,
        user_id: str = "",
        team_id: str = "",
    ) -> tuple[str, list[float]]:
        """
        Generate embedding vector for semantic matching.

        Extracts user message text and computes embedding using the
        provided sentence-transformer model. The encode() call is
        offloaded to a thread pool to avoid blocking the async event loop.

        Args:
            model: The LLM model name (used as partition prefix).
            messages: The chat messages list.
            embedder: A sentence-transformers model with .encode() method.
            user_id: User identifier for cache isolation.
            team_id: Team identifier for cache isolation.

        Returns:
            Tuple of (model_prefix, embedding_vector).
        """
        text = extract_semantic_content(messages)
        if not text:
            text = ""

        # Offload CPU-bound encode() to thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: embedder.encode(text, convert_to_numpy=True)
        )
        embedding_list = embedding.tolist()
        uid = user_id or "_"
        tid = team_id or "_"
        prefix = f"routeiq:semcache:{uid}:{tid}:{model.strip()}"

        return (prefix, embedding_list)


def extract_semantic_content(messages: list[dict[str, Any]]) -> str:
    """
    Extract the semantically meaningful content from a message chain.

    Includes system message (truncated to 500 chars) and the last user message.

    Args:
        messages: The chat messages list.

    Returns:
        Concatenated semantic content string.
    """
    parts: list[str] = []

    # Include system message
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"[system] {content[:500]}")
            break

    # Include last user message
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"[user] {content}")
            elif isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                if text_parts:
                    parts.append(f"[user] {' '.join(text_parts)}")
            break

    return "\n".join(parts)


def is_cacheable_request(
    params: dict[str, Any],
    max_temperature: float = DEFAULT_MAX_CACHEABLE_TEMPERATURE,
    headers: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """
    Determine if a request is cacheable.

    Args:
        params: Request parameters (model, messages, temperature, etc.).
        max_temperature: Maximum temperature for cacheability.
        headers: HTTP request headers (lowercase keys).

    Returns:
        Tuple of (is_cacheable, reason).
    """
    # Check no-cache header
    if headers:
        cache_control = headers.get("x-routeiq-cache-control", "")
        if "no-cache" in cache_control:
            return False, "no-cache header"

    if not params.get("model"):
        return False, "missing model"

    if not params.get("messages"):
        return False, "missing messages"

    # Check temperature
    temp = params.get("temperature")
    if temp is not None:
        try:
            temp_val = float(temp)
        except (TypeError, ValueError):
            return False, "invalid temperature"
        if temp_val > max_temperature:
            return False, "temperature too high"

    # Don't cache streaming requests
    if params.get("stream"):
        return False, "streaming request"

    return True, "cacheable"


# =============================================================================
# Cache Store Protocol and Implementations
# =============================================================================


@runtime_checkable
class CacheStore(Protocol):
    """Protocol for cache store backends."""

    async def get(self, key: str) -> CacheEntry | None:
        """Get a cache entry by exact key."""
        ...

    async def set(self, key: str, entry: CacheEntry, ttl: int) -> None:
        """Set a cache entry with TTL in seconds."""
        ...

    async def get_similar(
        self,
        embedding: list[float],
        model: str,
        threshold: float,
    ) -> CacheEntry | None:
        """Find a similar cache entry by embedding similarity."""
        ...


class InMemoryCache:
    """
    L1 in-memory LRU cache for hot entries.

    Uses OrderedDict for O(1) lookup and insertion with LRU eviction.
    Entries are evicted when max_size is exceeded or TTL has expired.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._store: OrderedDict[str, tuple[CacheEntry, float]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> CacheEntry | None:
        """Get entry by key, respecting TTL and updating LRU order."""
        if key not in self._store:
            self._misses += 1
            return None

        entry, expiry = self._store[key]
        if time.time() > expiry:
            # Expired - remove and report miss
            del self._store[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._store.move_to_end(key)
        self._hits += 1
        return entry

    async def set(self, key: str, entry: CacheEntry, ttl: int) -> None:
        """Set entry with TTL, evicting LRU if at capacity."""
        expiry = time.time() + ttl

        # If key exists, remove it first (to update position)
        if key in self._store:
            del self._store[key]

        # Evict LRU entries if at capacity
        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)

        self._store[key] = (entry, expiry)

    async def get_similar(
        self,
        embedding: list[float],
        model: str,
        threshold: float,
    ) -> CacheEntry | None:
        """In-memory cache does not support semantic search."""
        return None

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._store)

    @property
    def hits(self) -> int:
        """Total cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses."""
        return self._misses

    def clear(self) -> None:
        """Clear all entries and reset counters."""
        self._store.clear()
        self._hits = 0
        self._misses = 0


class RedisCacheStore:
    """
    L2 Redis cache with optional vector similarity.

    Uses redis-py async client for exact match GET/SET with TTL.
    Semantic similarity uses brute-force cosine similarity scan
    over Redis hashes (no Redis VSS module dependency).
    """

    def __init__(
        self,
        redis_client: Any,
        key_prefix: str = "routeiq:cache:v1:",
        user_id: str = "",
        team_id: str = "",
    ) -> None:
        """
        Args:
            redis_client: An async redis client (redis.asyncio).
            key_prefix: Prefix for all cache keys in Redis.
            user_id: User identifier for PII namespace isolation.
            team_id: Team identifier for PII namespace isolation.
        """
        self._redis = redis_client
        uid = user_id or "_"
        tid = team_id or "_"
        self._prefix = f"{key_prefix}{uid}:{tid}:"

    def _prefixed_key(self, key: str) -> str:
        """Add prefix if not already present."""
        if key.startswith(self._prefix):
            return key
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> CacheEntry | None:
        """Get entry from Redis by exact key."""
        try:
            data = await self._redis.get(self._prefixed_key(key))
            if data is None:
                return None
            entry_dict = json.loads(data)
            return CacheEntry(**entry_dict)
        except Exception as e:
            logger.warning(f"Redis cache get failed: {e}")
            return None

    async def set(self, key: str, entry: CacheEntry, ttl: int) -> None:
        """Set entry in Redis with TTL."""
        try:
            entry_dict = {
                "response": entry.response,
                "model": entry.model,
                "created_at": entry.created_at,
                "token_count": entry.token_count,
                "cache_key": entry.cache_key,
                "embedding": entry.embedding,
            }
            data = json.dumps(entry_dict)
            await self._redis.set(
                self._prefixed_key(key),
                data,
                ex=ttl,
            )
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")

    async def get_similar(
        self,
        embedding: list[float],
        model: str,
        threshold: float,
    ) -> CacheEntry | None:
        """
        Brute-force cosine similarity search over cached embeddings.

        Scans Redis keys matching the model prefix and computes cosine
        similarity against stored embeddings. Returns the best match
        above the threshold.

        This is a simplified implementation without Redis VSS module.
        For production at scale, use Redis Stack with FT.SEARCH.
        """
        try:
            pattern = f"{self._prefix}sem:{model}:*"
            best_entry: CacheEntry | None = None
            best_score = 0.0

            async for key in self._redis.scan_iter(match=pattern, count=100):
                data = await self._redis.get(key)
                if data is None:
                    continue
                try:
                    entry_dict = json.loads(data)
                    stored_embedding = entry_dict.get("embedding")
                    if not stored_embedding:
                        continue
                    score = _cosine_similarity(embedding, stored_embedding)
                    if score >= threshold and score > best_score:
                        best_score = score
                        best_entry = CacheEntry(**entry_dict)
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

            return best_entry
        except Exception as e:
            logger.warning(f"Redis semantic search failed: {e}")
            return None


class CacheManager:
    """
    Facade for cache admin operations (stats, flush, list entries, SSE replay).

    Wraps L1 (InMemoryCache) and optional L2 (RedisCacheStore) stores.
    The SemanticCachePlugin creates and registers this during startup.
    """

    def __init__(
        self,
        l1: InMemoryCache | None = None,
        l2: RedisCacheStore | None = None,
        ttl: int = 3600,
        semantic_enabled: bool = False,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._ttl = ttl
        self._semantic_enabled = semantic_enabled
        self._total_hits = 0
        self._total_misses = 0

    def record_hit(self) -> None:
        """Record a cache hit for aggregate stats."""
        self._total_hits += 1

    def record_miss(self) -> None:
        """Record a cache miss for aggregate stats."""
        self._total_misses += 1

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics: hit/miss counts, entry count, configuration.

        Returns:
            Dictionary with cache statistics.
        """
        stats: dict[str, Any] = {
            "enabled": True,
            "semantic_enabled": self._semantic_enabled,
            "ttl_seconds": self._ttl,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
        }

        if self._l1 is not None:
            stats["l1"] = {
                "size": self._l1.size,
                "hits": self._l1.hits,
                "misses": self._l1.misses,
                "max_size": self._l1._max_size,
            }

        stats["l2_configured"] = self._l2 is not None
        return stats

    async def flush(self, prefix: str | None = None) -> int:
        """
        Flush all cache entries or entries matching a prefix.

        Args:
            prefix: Optional key prefix filter. If None, flush all.

        Returns:
            Number of entries removed.
        """
        count = 0

        if self._l1 is not None:
            if prefix is None:
                count += self._l1.size
                self._l1.clear()
            else:
                # Remove entries matching prefix from L1
                keys_to_remove = [k for k in self._l1._store if k.startswith(prefix)]
                for k in keys_to_remove:
                    del self._l1._store[k]
                count += len(keys_to_remove)

        # Reset aggregate counters on full flush
        if prefix is None:
            self._total_hits = 0
            self._total_misses = 0

        return count

    def list_entries(self, limit: int = 100) -> dict[str, Any]:
        """
        List cached keys with metadata.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            Dictionary with enabled status and entries list.
        """
        entries: list[dict[str, Any]] = []

        if self._l1 is not None:
            now = time.time()
            for key, (entry, expiry) in list(self._l1._store.items())[:limit]:
                entries.append(
                    {
                        "key": key,
                        "model": entry.model,
                        "created_at": entry.created_at,
                        "token_count": entry.token_count,
                        "ttl_remaining": max(0, int(expiry - now)),
                        "has_embedding": entry.embedding is not None,
                    }
                )

        return {"enabled": True, "count": len(entries), "entries": entries}

    async def replay_as_sse_stream(
        self, cached_response: dict[str, Any]
    ) -> "AsyncIterator[str]":
        """
        Convert a cached response to synthetic SSE chunks (OpenAI format).

        Yields SSE-formatted strings that mimic streaming output, ending with [DONE].
        This allows cached non-streaming responses to be served to clients that
        request streaming.

        Args:
            cached_response: The cached LLM response dict.

        Yields:
            SSE-formatted strings.
        """
        # Extract content from the cached response
        content = ""
        model = cached_response.get("model", "cached-model")
        choices = cached_response.get("choices", [])
        if choices:
            first_choice = choices[0]
            message = first_choice.get("message", {})
            content = message.get("content", "")

        # Build a synthetic streaming chunk
        chunk = {
            "id": f"chatcmpl-cache-{hash(json.dumps(cached_response, sort_keys=True, default=str)) & 0xFFFFFFFF:08x}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


# Module-level singleton for CacheManager
_cache_manager: CacheManager | None = None


def set_cache_manager(manager: CacheManager | None) -> None:
    """Register the global CacheManager singleton (called by SemanticCachePlugin)."""
    global _cache_manager
    _cache_manager = manager


def get_cache_manager() -> CacheManager | None:
    """Get the global CacheManager singleton, or None if cache is not enabled."""
    return _cache_manager


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using numpy when available."""
    if len(a) != len(b) or not a:
        return 0.0

    if NUMPY_AVAILABLE:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    # Pure-Python fallback
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot / (norm_a * norm_b))
