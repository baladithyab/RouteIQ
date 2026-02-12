"""
Conversation Affinity Tracker
=============================

Maps response_ids to provider deployments for provider affinity routing.

When a client sends a request with previous_response_id (Responses API),
the gateway must route to the same provider that generated that response.
This tracker stores the mapping and provides lookup.

Supports two backends:
- In-memory (default, single-node): TTL-based dict with background cleanup
- Redis (HA mode): Redis keys with TTL for multi-node deployments

Configuration via environment variables:
- CONVERSATION_AFFINITY_ENABLED: Enable/disable (default: false)
- CONVERSATION_AFFINITY_TTL: TTL in seconds for affinity records (default: 3600)
- CONVERSATION_AFFINITY_MAX_ENTRIES: Max in-memory entries before LRU eviction (default: 10000)

Usage:
    from litellm_llmrouter.conversation_affinity import (
        init_affinity_tracker,
        get_affinity_tracker,
        reset_affinity_tracker,
    )

    # On startup:
    tracker = init_affinity_tracker(redis_url="redis://localhost:6379")

    # Record a response:
    await tracker.record_response("resp_abc", "openai/gpt-4", "gpt-4")

    # Look up affinity:
    record = await tracker.get_affinity("resp_abc")
    if record:
        print(f"Route to: {record.provider_deployment}")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Feature flag and configuration defaults
CONVERSATION_AFFINITY_ENABLED = (
    os.getenv("CONVERSATION_AFFINITY_ENABLED", "false").lower() == "true"
)
DEFAULT_TTL_SECONDS = int(os.getenv("CONVERSATION_AFFINITY_TTL", "3600"))
DEFAULT_MAX_ENTRIES = int(os.getenv("CONVERSATION_AFFINITY_MAX_ENTRIES", "10000"))


@dataclass(frozen=True)
class AffinityRecord:
    """A record mapping a response_id to its provider deployment."""

    response_id: str
    provider_deployment: str  # e.g., "openai/gpt-4"
    model: str
    recorded_at: float
    expires_at: float

    def is_expired(self, now: float | None = None) -> bool:
        """Check if this record has expired."""
        if now is None:
            now = time.time()
        return now >= self.expires_at

    def to_json(self) -> str:
        """Serialize to JSON string for Redis storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> AffinityRecord:
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(**d)


class ConversationAffinityTracker:
    """
    Maps response_ids to provider deployments for provider affinity routing.

    When a client sends a request with previous_response_id (Responses API),
    the gateway must route to the same provider that generated that response.
    This tracker stores the mapping and provides lookup.

    Supports two backends:
    - In-memory (default, single-node): TTL-based dict with background cleanup
    - Redis (HA mode): Redis keys with TTL for multi-node deployments
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries

        # In-memory store: {response_id: AffinityRecord}
        # Also tracks insertion order for LRU eviction
        self._store: dict[str, AffinityRecord] = {}
        self._lock = asyncio.Lock()

        # Redis backend
        self._redis_url = redis_url
        self._redis: Any = None
        self._redis_available = False

        # Background cleanup task
        self._cleanup_task: asyncio.Task[None] | None = None

        logger.info(
            f"ConversationAffinityTracker initialized "
            f"(ttl={ttl_seconds}s, max_entries={max_entries}, "
            f"redis={'configured' if redis_url else 'disabled'})"
        )

    async def start(self) -> None:
        """Start the tracker, connecting to Redis if configured."""
        if self._redis_url:
            await self._connect_redis()

        # Start background cleanup for in-memory store
        self._cleanup_task = asyncio.create_task(self._background_cleanup())

    async def stop(self) -> None:
        """Stop the tracker and clean up resources."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None
            self._redis_available = False

    async def _connect_redis(self) -> None:
        """Attempt to connect to Redis.

        If an explicit ``redis_url`` was provided, it is used as-is via
        ``from_url``.  Otherwise, falls back to ``build_redis_url()``
        which constructs a URL from the shared REDIS_* env vars
        (including REDIS_PASSWORD).
        """
        try:
            import redis.asyncio as aioredis

            url = self._redis_url
            if url is None:
                from litellm_llmrouter.redis_pool import build_redis_url

                url = build_redis_url()

            self._redis = aioredis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=5.0,
            )
            # Test connection
            await self._redis.ping()
            self._redis_available = True
            logger.info(
                f"ConversationAffinityTracker: Redis connected ({self._redis_url})"
            )
        except Exception as e:
            logger.warning(
                f"ConversationAffinityTracker: Redis unavailable ({e}), "
                f"falling back to in-memory backend"
            )
            self._redis = None
            self._redis_available = False

    async def record_response(
        self, response_id: str, provider_deployment: str, model: str
    ) -> None:
        """Record which provider served a response."""
        now = time.time()
        record = AffinityRecord(
            response_id=response_id,
            provider_deployment=provider_deployment,
            model=model,
            recorded_at=now,
            expires_at=now + self._ttl_seconds,
        )

        # Try Redis first
        if self._redis_available:
            try:
                await self._redis_set(response_id, record)
                return
            except Exception as e:
                logger.warning(
                    f"ConversationAffinityTracker: Redis write failed ({e}), "
                    f"falling back to in-memory"
                )
                self._redis_available = False

        # In-memory fallback
        async with self._lock:
            # LRU eviction if at capacity
            if len(self._store) >= self._max_entries and response_id not in self._store:
                self._evict_oldest()

            self._store[response_id] = record

    async def get_affinity(self, response_id: str) -> AffinityRecord | None:
        """Look up provider affinity for a response_id."""
        # Try Redis first
        if self._redis_available:
            try:
                record = await self._redis_get(response_id)
                if record is not None:
                    return record
                # Not found in Redis - fall through (don't check memory)
                return None
            except Exception as e:
                logger.warning(
                    f"ConversationAffinityTracker: Redis read failed ({e}), "
                    f"falling back to in-memory"
                )
                self._redis_available = False

        # In-memory fallback
        async with self._lock:
            record = self._store.get(response_id)
            if record is None:
                return None
            if record.is_expired():
                del self._store[response_id]
                return None
            return record

    async def cleanup_expired(self) -> int:
        """Remove expired entries from in-memory store. Returns count removed."""
        now = time.time()
        async with self._lock:
            expired_keys = [k for k, v in self._store.items() if v.is_expired(now)]
            for k in expired_keys:
                del self._store[k]
            if expired_keys:
                logger.debug(
                    f"ConversationAffinityTracker: cleaned up {len(expired_keys)} "
                    f"expired entries"
                )
            return len(expired_keys)

    def _evict_oldest(self) -> None:
        """Evict the oldest entry (LRU) from in-memory store. Must hold lock."""
        if not self._store:
            return
        # dict preserves insertion order in Python 3.7+; first key is oldest
        oldest_key = next(iter(self._store))
        del self._store[oldest_key]
        logger.debug(
            f"ConversationAffinityTracker: evicted oldest entry '{oldest_key}'"
        )

    async def _redis_set(self, response_id: str, record: AffinityRecord) -> None:
        """Store a record in Redis with TTL."""
        key = f"affinity:{response_id}"
        await self._redis.set(key, record.to_json(), ex=self._ttl_seconds)

    async def _redis_get(self, response_id: str) -> AffinityRecord | None:
        """Retrieve a record from Redis."""
        key = f"affinity:{response_id}"
        data = await self._redis.get(key)
        if data is None:
            return None
        return AffinityRecord.from_json(data)

    async def _background_cleanup(self) -> None:
        """Periodically clean up expired in-memory entries."""
        while True:
            try:
                await asyncio.sleep(60)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ConversationAffinityTracker: cleanup error: {e}")

    @property
    def entry_count(self) -> int:
        """Current number of entries in the in-memory store."""
        return len(self._store)

    @property
    def ttl_seconds(self) -> int:
        """Configured TTL for affinity records."""
        return self._ttl_seconds

    @property
    def max_entries(self) -> int:
        """Configured max entries for in-memory store."""
        return self._max_entries

    @property
    def redis_available(self) -> bool:
        """Whether Redis backend is currently available."""
        return self._redis_available


# =============================================================================
# Module-level singleton
# =============================================================================

_tracker: ConversationAffinityTracker | None = None


def get_affinity_tracker() -> ConversationAffinityTracker | None:
    """
    Get the global affinity tracker singleton.

    Returns None if the tracker has not been initialized or if the feature
    is disabled.
    """
    return _tracker


def init_affinity_tracker(
    redis_url: str | None = None,
    ttl: int = DEFAULT_TTL_SECONDS,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> ConversationAffinityTracker:
    """
    Initialize the global affinity tracker singleton.

    Args:
        redis_url: Optional Redis URL for HA deployments.
        ttl: TTL in seconds for affinity records.
        max_entries: Max in-memory entries before LRU eviction.

    Returns:
        The initialized ConversationAffinityTracker instance.
    """
    global _tracker
    _tracker = ConversationAffinityTracker(
        redis_url=redis_url, ttl_seconds=ttl, max_entries=max_entries
    )
    logger.info("ConversationAffinityTracker: singleton initialized")
    return _tracker


def reset_affinity_tracker() -> None:
    """Reset the global affinity tracker singleton (for testing)."""
    global _tracker
    _tracker = None
