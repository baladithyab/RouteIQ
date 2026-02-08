"""
Semantic Cache Plugin: GatewayPlugin wiring for LLM response caching
======================================================================

Implements a GatewayPlugin that provides two-tier LLM response caching:
- L1: In-memory LRU cache (InMemoryCache)
- L2: Redis-backed shared cache (RedisCacheStore, optional)

Hooks:
- on_llm_pre_call: Check cache for exact and semantic matches.
  On hit, stores cached response in kwargs metadata.
- on_llm_success: Store successful responses in L1 + L2 cache.

Configuration (environment variables):
- CACHE_ENABLED=false            Feature flag (default off)
- CACHE_SEMANTIC_ENABLED=false   Semantic matching (default off)
- CACHE_TTL_SECONDS=3600         Default TTL in seconds
- CACHE_L1_MAX_SIZE=1000         Max entries in L1 cache
- CACHE_SIMILARITY_THRESHOLD=0.95 Cosine similarity threshold
- CACHE_REDIS_URL                Redis URL (optional, L2 disabled without it)
- CACHE_EMBEDDING_MODEL=all-MiniLM-L6-v2  Sentence-transformer model name
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from litellm_llmrouter.gateway.plugin_manager import (
    FailureMode,
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)
from litellm_llmrouter.semantic_cache import (
    CacheEntry,
    CacheKeyGenerator,
    CacheManager,
    InMemoryCache,
    RedisCacheStore,
    is_cacheable_request,
    set_cache_manager,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Lazy-loaded embedder singleton (mirrors strategies.py pattern)
_embedder_model: Any = None
_embedder_lock = threading.Lock()


def _get_embedder(model_name: str) -> Any:
    """
    Get or create a cached SentenceTransformer model.

    Uses lazy loading with thread-safe singleton pattern.
    """
    global _embedder_model

    with _embedder_lock:
        if _embedder_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading SentenceTransformer model: {model_name}")
                _embedder_model = SentenceTransformer(model_name, device="cpu")
                logger.info("SentenceTransformer model loaded for cache")
            except ImportError:
                raise ImportError(
                    "sentence-transformers package is required for semantic caching. "
                    "Install with: pip install sentence-transformers"
                )
        return _embedder_model


def _reset_embedder() -> None:
    """Reset the embedder singleton (for testing)."""
    global _embedder_model
    with _embedder_lock:
        _embedder_model = None


class SemanticCachePlugin(GatewayPlugin):
    """
    LLM Response Cache Plugin.

    Hooks into the LLM lifecycle via on_llm_pre_call (lookup) and
    on_llm_success (store). Uses L1 in-memory + optional L2 Redis caching.

    On cache hit during pre_call, the cached response is stored in
    kwargs metadata under ``_cache_hit_response`` with flag ``_cache_hit=True``.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._semantic_enabled = False
        self._ttl: int = 3600
        self._l1_max_size: int = 1000
        self._similarity_threshold: float = 0.95
        self._embedding_model: str = "all-MiniLM-L6-v2"
        self._redis_url: str | None = None
        self._max_cacheable_temperature: float = 0.1

        self._l1: InMemoryCache | None = None
        self._l2: RedisCacheStore | None = None
        self._redis_client: Any = None
        self._cache_manager: CacheManager | None = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="semantic-cache",
            version="1.0.0",
            capabilities={PluginCapability.MIDDLEWARE},
            priority=10,
            failure_mode=FailureMode.CONTINUE,
            description="Two-tier LLM response caching with exact and semantic matching",
        )

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Initialize cache backends from environment configuration."""
        self._enabled = os.getenv("CACHE_ENABLED", "false").lower() == "true"
        if not self._enabled:
            logger.info("Semantic cache plugin disabled (CACHE_ENABLED=false)")
            return

        self._semantic_enabled = (
            os.getenv("CACHE_SEMANTIC_ENABLED", "false").lower() == "true"
        )
        self._ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        self._l1_max_size = int(os.getenv("CACHE_L1_MAX_SIZE", "1000"))
        self._similarity_threshold = float(
            os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95")
        )
        self._embedding_model = os.getenv("CACHE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._redis_url = os.getenv("CACHE_REDIS_URL")
        temp_str = os.getenv("CACHE_MAX_TEMPERATURE", "0.1")
        self._max_cacheable_temperature = float(temp_str)

        # Initialize L1
        self._l1 = InMemoryCache(max_size=self._l1_max_size)
        logger.info(f"Cache L1 initialized (max_size={self._l1_max_size})")

        # Initialize L2 (Redis) if URL provided
        if self._redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis_client = aioredis.from_url(
                    self._redis_url, decode_responses=True
                )
                self._l2 = RedisCacheStore(redis_client=self._redis_client)
                logger.info(f"Cache L2 Redis initialized ({self._redis_url})")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self._l2 = None

        # Register the CacheManager singleton for admin endpoints
        self._cache_manager = CacheManager(
            l1=self._l1,
            l2=self._l2,
            ttl=self._ttl,
            semantic_enabled=self._semantic_enabled,
        )
        set_cache_manager(self._cache_manager)

        logger.info(
            f"Semantic cache plugin started "
            f"(semantic={self._semantic_enabled}, ttl={self._ttl}s)"
        )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Clean up cache resources."""
        # Unregister the CacheManager singleton
        set_cache_manager(None)
        self._cache_manager = None

        if self._l1:
            self._l1.clear()
            self._l1 = None

        if self._redis_client is not None:
            try:
                await self._redis_client.aclose()
            except Exception as e:
                logger.debug(f"Error closing Redis client: {e}")
            self._redis_client = None
            self._l2 = None

        logger.info("Semantic cache plugin shut down")

    async def on_llm_pre_call(
        self,
        model: str,
        messages: list[Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Check cache for exact and semantic matches before LLM call.

        On cache hit, stores response in kwargs metadata.

        Args:
            model: The model being called.
            messages: The messages list being sent.
            kwargs: Additional call parameters.

        Returns:
            Dict with ``_cache_hit`` flag and cached response metadata,
            or None on cache miss / when cache is disabled.
        """
        if not self._enabled or self._l1 is None:
            return None

        # Build params dict for cacheability check
        params = dict(kwargs)
        params["model"] = model
        params["messages"] = messages

        # Extract headers from kwargs metadata if present
        litellm_params = kwargs.get("litellm_params", {})
        proxy_request = litellm_params.get("proxy_server_request", {})
        headers = proxy_request.get("headers", {})

        cacheable, reason = is_cacheable_request(
            params,
            max_temperature=self._max_cacheable_temperature,
            headers=headers,
        )
        if not cacheable:
            logger.debug(f"Request not cacheable: {reason}")
            return None

        # Generate exact cache key
        exact_key = CacheKeyGenerator.exact_key(
            model=model,
            messages=messages,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ("model", "messages", "litellm_params")
            },
        )

        # L1 lookup
        entry = await self._l1.get(exact_key)
        if entry is not None:
            logger.debug(f"Cache L1 HIT: {exact_key[:20]}...")
            if self._cache_manager:
                self._cache_manager.record_hit()
            return self._build_hit_response(entry, "l1")

        # L2 lookup
        if self._l2 is not None:
            entry = await self._l2.get(exact_key)
            if entry is not None:
                # Populate L1 on L2 hit
                await self._l1.set(exact_key, entry, self._ttl)
                logger.debug(f"Cache L2 HIT: {exact_key[:20]}...")
                if self._cache_manager:
                    self._cache_manager.record_hit()
                return self._build_hit_response(entry, "l2")

        # Semantic lookup
        if self._semantic_enabled and self._l2 is not None:
            try:
                embedder = _get_embedder(self._embedding_model)
                prefix, embedding = CacheKeyGenerator.semantic_key(
                    model, messages, embedder
                )
                entry = await self._l2.get_similar(
                    embedding, model, self._similarity_threshold
                )
                if entry is not None:
                    # Populate L1 on semantic hit
                    await self._l1.set(exact_key, entry, self._ttl)
                    logger.debug(f"Cache semantic HIT for model={model}")
                    if self._cache_manager:
                        self._cache_manager.record_hit()
                    return self._build_hit_response(entry, "semantic")
            except Exception as e:
                logger.warning(f"Semantic cache lookup failed: {e}")

        logger.debug(f"Cache MISS: {exact_key[:20]}...")
        if self._cache_manager:
            self._cache_manager.record_miss()
        return None

    async def on_llm_success(
        self,
        model: str,
        response: Any,
        kwargs: dict[str, Any],
    ) -> None:
        """
        Store successful LLM response in cache.

        Skips storage if this was a cache hit (already cached) or if
        the request is not cacheable.

        Args:
            model: The model that was called.
            response: The LLM response object.
            kwargs: The call parameters that were used.
        """
        if not self._enabled or self._l1 is None:
            return

        # Don't re-cache hits
        metadata = kwargs.get("metadata", {}) or {}
        if metadata.get("_cache_hit"):
            return

        messages = kwargs.get("messages", [])
        params = dict(kwargs)
        params["model"] = model
        params["messages"] = messages

        # Extract headers
        litellm_params = kwargs.get("litellm_params", {})
        proxy_request = litellm_params.get("proxy_server_request", {})
        headers = proxy_request.get("headers", {})

        cacheable, reason = is_cacheable_request(
            params,
            max_temperature=self._max_cacheable_temperature,
            headers=headers,
        )
        if not cacheable:
            return

        # Serialize response
        try:
            if hasattr(response, "model_dump"):
                response_dict = response.model_dump()
            elif hasattr(response, "dict"):
                response_dict = response.dict()
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {"raw": str(response)}
        except Exception as e:
            logger.debug(f"Failed to serialize response for caching: {e}")
            return

        # Extract token count
        usage = response_dict.get("usage", {}) or {}
        token_count = usage.get("total_tokens", 0)

        exact_key = CacheKeyGenerator.exact_key(
            model=model,
            messages=messages,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ("model", "messages", "litellm_params", "metadata")
            },
        )

        entry = CacheEntry(
            response=response_dict,
            model=model,
            created_at=time.time(),
            token_count=token_count,
            cache_key=exact_key,
        )

        # Store embedding for semantic cache
        if self._semantic_enabled:
            try:
                embedder = _get_embedder(self._embedding_model)
                _, embedding = CacheKeyGenerator.semantic_key(model, messages, embedder)
                entry.embedding = embedding
            except Exception as e:
                logger.debug(f"Failed to compute embedding for cache: {e}")

        # Store in L1
        await self._l1.set(exact_key, entry, self._ttl)

        # Store in L2
        if self._l2 is not None:
            await self._l2.set(exact_key, entry, self._ttl)

            # Store semantic entry separately for similarity search
            if entry.embedding and self._semantic_enabled:
                import uuid

                sem_key = f"sem:{model}:{uuid.uuid4().hex[:12]}"
                await self._l2.set(sem_key, entry, self._ttl)

        logger.debug(f"Cached response: {exact_key[:20]}... (tokens={token_count})")

    async def health_check(self) -> dict[str, Any]:
        """Report cache health status."""
        if not self._enabled:
            return {"status": "ok", "cache_enabled": False}

        result: dict[str, Any] = {
            "status": "ok",
            "cache_enabled": True,
            "semantic_enabled": self._semantic_enabled,
            "ttl_seconds": self._ttl,
        }

        if self._l1 is not None:
            result["l1_size"] = self._l1.size
            result["l1_hits"] = self._l1.hits
            result["l1_misses"] = self._l1.misses

        if self._l2 is not None:
            try:
                await self._redis_client.ping()
                result["l2_status"] = "connected"
            except Exception:
                result["l2_status"] = "disconnected"
                result["status"] = "degraded"
        else:
            result["l2_status"] = "not configured"

        return result

    @staticmethod
    def _build_hit_response(entry: CacheEntry, tier: str) -> dict[str, Any]:
        """Build kwargs override dict for a cache hit."""
        cache_age = int(time.time() - entry.created_at)
        return {
            "metadata": {
                "_cache_hit": True,
                "_cache_hit_response": entry.response,
                "_cache_tier": tier,
                "_cache_key": entry.cache_key,
                "_cache_age_seconds": cache_age,
            },
            "_cache_headers": {
                "x-routeiq-cache": "HIT",
                "x-routeiq-cache-tier": tier,
                "x-routeiq-cache-age": str(cache_age),
            },
        }

    @staticmethod
    def build_miss_headers() -> dict[str, str]:
        """Build cache headers for a cache miss."""
        return {
            "x-routeiq-cache": "MISS",
        }
