"""
Tests for semantic cache infrastructure and plugin.

Covers:
- CacheKeyGenerator: exact key determinism, parameter canonicalization, semantic keys
- InMemoryCache: get/set, TTL expiry, LRU eviction
- RedisCacheStore: get/set with mock redis
- SemanticCachePlugin: cache miss → store, cache hit → return, feature flag
- Cacheability rules: temperature, streaming, no-cache header, edge cases
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.gateway.plugin_manager import (
    FailureMode,
    PluginCapability,
)
from litellm_llmrouter.semantic_cache import (
    CacheEntry,
    CacheKeyGenerator,
    CacheManager,
    InMemoryCache,
    RedisCacheStore,
    _cosine_similarity,
    extract_semantic_content,
    get_cache_manager,
    is_cacheable_request,
    set_cache_manager,
)
from litellm_llmrouter.gateway.plugins.cache_plugin import (
    SemanticCachePlugin,
    _reset_embedder,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_embedder_singleton():
    """Reset the embedder singleton before each test."""
    _reset_embedder()
    yield
    _reset_embedder()


@pytest.fixture(autouse=True)
def reset_cache_manager_singleton():
    """Reset the cache manager singleton before each test."""
    set_cache_manager(None)
    yield
    set_cache_manager(None)


@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]


@pytest.fixture
def sample_entry():
    return CacheEntry(
        response={"choices": [{"message": {"content": "Python is a language."}}]},
        model="gpt-4",
        created_at=time.time(),
        token_count=42,
        cache_key="routeiq:cache:v1:abc123",
    )


@pytest.fixture
def mock_redis():
    """Create a mock async redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.ping = AsyncMock()
    redis.scan_iter = MagicMock()
    redis.aclose = AsyncMock()
    return redis


# =============================================================================
# CacheKeyGenerator: exact_key
# =============================================================================


class TestCacheKeyGeneratorExact:
    def test_deterministic_same_input(self, sample_messages):
        """Same inputs produce the same cache key."""
        key1 = CacheKeyGenerator.exact_key("gpt-4", sample_messages, temperature=0.0)
        key2 = CacheKeyGenerator.exact_key("gpt-4", sample_messages, temperature=0.0)
        assert key1 == key2

    def test_deterministic_message_order(self):
        """Message order matters for cache key."""
        msgs1 = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        msgs2 = [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
        ]
        key1 = CacheKeyGenerator.exact_key("gpt-4", msgs1)
        key2 = CacheKeyGenerator.exact_key("gpt-4", msgs2)
        assert key1 != key2

    def test_different_model_different_key(self, sample_messages):
        """Different models produce different keys."""
        key1 = CacheKeyGenerator.exact_key("gpt-4", sample_messages)
        key2 = CacheKeyGenerator.exact_key("gpt-3.5-turbo", sample_messages)
        assert key1 != key2

    def test_different_temperature_different_key(self, sample_messages):
        """Different temperatures produce different keys."""
        key1 = CacheKeyGenerator.exact_key("gpt-4", sample_messages, temperature=0.0)
        key2 = CacheKeyGenerator.exact_key("gpt-4", sample_messages, temperature=0.5)
        assert key1 != key2

    def test_different_max_tokens_different_key(self, sample_messages):
        """Different max_tokens produce different keys."""
        key1 = CacheKeyGenerator.exact_key("gpt-4", sample_messages, max_tokens=100)
        key2 = CacheKeyGenerator.exact_key("gpt-4", sample_messages, max_tokens=200)
        assert key1 != key2

    def test_key_format(self, sample_messages):
        """Key has the expected prefix format."""
        key = CacheKeyGenerator.exact_key("gpt-4", sample_messages)
        assert key.startswith("routeiq:cache:v1:")
        # SHA-256 hex is 64 chars
        hash_part = key.split(":")[-1]
        assert len(hash_part) == 64

    def test_whitespace_normalization(self):
        """Trailing whitespace in messages is normalized."""
        msgs1 = [{"role": "user", "content": "hello  "}]
        msgs2 = [{"role": "user", "content": "hello"}]
        key1 = CacheKeyGenerator.exact_key("gpt-4", msgs1)
        key2 = CacheKeyGenerator.exact_key("gpt-4", msgs2)
        assert key1 == key2

    def test_excluded_params_ignored(self, sample_messages):
        """Excluded parameters (stream, user, etc.) don't affect the key."""
        key1 = CacheKeyGenerator.exact_key("gpt-4", sample_messages)
        key2 = CacheKeyGenerator.exact_key(
            "gpt-4", sample_messages, stream=True, user="user-123"
        )
        # stream and user are not in CACHE_KEY_PARAMS so they're ignored
        assert key1 == key2

    def test_none_params_excluded(self, sample_messages):
        """None-valued parameters are excluded from key."""
        key1 = CacheKeyGenerator.exact_key("gpt-4", sample_messages)
        key2 = CacheKeyGenerator.exact_key(
            "gpt-4", sample_messages, temperature=None, top_p=None
        )
        assert key1 == key2

    def test_dict_key_ordering(self):
        """Dict values are sorted by key for determinism."""
        msgs1 = [{"role": "user", "content": "test"}]
        key1 = CacheKeyGenerator.exact_key(
            "gpt-4", msgs1, response_format={"type": "json", "schema": "strict"}
        )
        key2 = CacheKeyGenerator.exact_key(
            "gpt-4", msgs1, response_format={"schema": "strict", "type": "json"}
        )
        assert key1 == key2

    def test_tool_calls_sorted_by_id(self):
        """Tool calls in messages are sorted by ID."""
        msgs1 = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_b", "function": {"name": "f2"}},
                    {"id": "call_a", "function": {"name": "f1"}},
                ],
            }
        ]
        msgs2 = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "function": {"name": "f1"}},
                    {"id": "call_b", "function": {"name": "f2"}},
                ],
            }
        ]
        key1 = CacheKeyGenerator.exact_key("gpt-4", msgs1)
        key2 = CacheKeyGenerator.exact_key("gpt-4", msgs2)
        assert key1 == key2

    def test_empty_messages(self):
        """Empty messages list produces a valid key."""
        key = CacheKeyGenerator.exact_key("gpt-4", [])
        assert key.startswith("routeiq:cache:v1:")

    def test_empty_model(self):
        """Empty model produces a valid key."""
        msgs = [{"role": "user", "content": "hello"}]
        key = CacheKeyGenerator.exact_key("", msgs)
        assert key.startswith("routeiq:cache:v1:")


# =============================================================================
# CacheKeyGenerator: semantic_key
# =============================================================================


class TestCacheKeyGeneratorSemantic:
    async def test_semantic_key_with_mock_embedder(self, sample_messages):
        """Semantic key generation works with a mock embedder."""
        mock_embedder = MagicMock()
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.1, 0.2, 0.3]
        mock_embedder.encode.return_value = mock_array

        prefix, embedding = await CacheKeyGenerator.semantic_key(
            "gpt-4", sample_messages, mock_embedder
        )

        assert prefix == "routeiq:semcache:_:_:gpt-4"
        assert embedding == [0.1, 0.2, 0.3]

    async def test_semantic_key_extracts_user_message(self, sample_messages):
        """Semantic key encodes the semantic content (system + last user msg)."""
        mock_embedder = MagicMock()
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.5]
        mock_embedder.encode.return_value = mock_array

        await CacheKeyGenerator.semantic_key("gpt-4", sample_messages, mock_embedder)

        # The encode is now run in an executor, so we check the extracted text
        # was passed to the embedder (may be via run_in_executor wrapping)
        assert mock_embedder.encode.called
        call_args = mock_embedder.encode.call_args
        text = call_args[0][0]
        assert "[system]" in text
        assert "[user]" in text
        assert "What is Python?" in text


# =============================================================================
# extract_semantic_content
# =============================================================================


class TestExtractSemanticContent:
    def test_system_and_user(self):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello world"},
        ]
        result = extract_semantic_content(messages)
        assert "[system] Be helpful" in result
        assert "[user] Hello world" in result

    def test_user_only(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = extract_semantic_content(messages)
        assert "[user] Hello" in result
        assert "[system]" not in result

    def test_system_truncated(self):
        long_system = "x" * 600
        messages = [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "test"},
        ]
        result = extract_semantic_content(messages)
        # System content is truncated to 500 chars
        system_part = result.split("\n")[0]
        assert len(system_part) <= len("[system] ") + 500

    def test_multipart_user_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "http://example.com"}},
                ],
            }
        ]
        result = extract_semantic_content(messages)
        assert "describe this" in result

    def test_empty_messages(self):
        result = extract_semantic_content([])
        assert result == ""

    def test_last_user_message_selected(self):
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second question"},
        ]
        result = extract_semantic_content(messages)
        assert "second question" in result
        assert "first question" not in result


# =============================================================================
# is_cacheable_request
# =============================================================================


class TestIsCacheableRequest:
    def test_cacheable_default_temperature(self):
        params = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        ok, reason = is_cacheable_request(params)
        assert ok is True
        assert reason == "cacheable"

    def test_cacheable_zero_temperature(self):
        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
        }
        ok, reason = is_cacheable_request(params)
        assert ok is True

    def test_not_cacheable_high_temperature(self):
        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
        }
        ok, reason = is_cacheable_request(params)
        assert ok is False
        assert "temperature" in reason

    def test_not_cacheable_streaming(self):
        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        ok, reason = is_cacheable_request(params)
        assert ok is False
        assert "streaming" in reason

    def test_not_cacheable_no_cache_header(self):
        params = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        headers = {"x-routeiq-cache-control": "no-cache"}
        ok, reason = is_cacheable_request(params, headers=headers)
        assert ok is False
        assert "no-cache" in reason

    def test_not_cacheable_missing_model(self):
        params = {"messages": [{"role": "user", "content": "hi"}]}
        ok, reason = is_cacheable_request(params)
        assert ok is False
        assert "model" in reason

    def test_not_cacheable_missing_messages(self):
        params = {"model": "gpt-4"}
        ok, reason = is_cacheable_request(params)
        assert ok is False
        assert "messages" in reason

    def test_not_cacheable_empty_messages(self):
        params = {"model": "gpt-4", "messages": []}
        ok, reason = is_cacheable_request(params)
        assert ok is False
        assert "messages" in reason

    def test_cacheable_temperature_at_threshold(self):
        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.1,
        }
        ok, reason = is_cacheable_request(params)
        assert ok is True

    def test_custom_max_temperature(self):
        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
        }
        ok, reason = is_cacheable_request(params, max_temperature=1.0)
        assert ok is True


# =============================================================================
# InMemoryCache
# =============================================================================


class TestInMemoryCache:
    async def test_get_miss(self):
        cache = InMemoryCache()
        result = await cache.get("nonexistent")
        assert result is None
        assert cache.misses == 1

    async def test_set_and_get(self, sample_entry):
        cache = InMemoryCache()
        await cache.set("key1", sample_entry, ttl=60)
        result = await cache.get("key1")
        assert result is not None
        assert result.model == "gpt-4"
        assert cache.hits == 1

    async def test_ttl_expiry(self, sample_entry):
        cache = InMemoryCache()
        # Set with TTL of 0 (already expired)
        await cache.set("key1", sample_entry, ttl=0)
        # Wait briefly to ensure time.time() has moved past expiry
        await asyncio.sleep(0.01)
        result = await cache.get("key1")
        assert result is None
        assert cache.misses == 1

    async def test_lru_eviction(self, sample_entry):
        cache = InMemoryCache(max_size=2)
        entry1 = CacheEntry(
            response={"v": 1},
            model="m1",
            created_at=time.time(),
            token_count=1,
            cache_key="k1",
        )
        entry2 = CacheEntry(
            response={"v": 2},
            model="m2",
            created_at=time.time(),
            token_count=2,
            cache_key="k2",
        )
        entry3 = CacheEntry(
            response={"v": 3},
            model="m3",
            created_at=time.time(),
            token_count=3,
            cache_key="k3",
        )

        await cache.set("k1", entry1, ttl=3600)
        await cache.set("k2", entry2, ttl=3600)
        assert cache.size == 2

        # Adding k3 should evict k1 (least recently used)
        await cache.set("k3", entry3, ttl=3600)
        assert cache.size == 2
        assert await cache.get("k1") is None  # Evicted
        assert await cache.get("k2") is not None
        assert await cache.get("k3") is not None

    async def test_lru_access_order(self):
        cache = InMemoryCache(max_size=2)
        entry1 = CacheEntry(
            response={"v": 1},
            model="m1",
            created_at=time.time(),
            token_count=1,
            cache_key="k1",
        )
        entry2 = CacheEntry(
            response={"v": 2},
            model="m2",
            created_at=time.time(),
            token_count=2,
            cache_key="k2",
        )
        entry3 = CacheEntry(
            response={"v": 3},
            model="m3",
            created_at=time.time(),
            token_count=3,
            cache_key="k3",
        )

        await cache.set("k1", entry1, ttl=3600)
        await cache.set("k2", entry2, ttl=3600)
        # Access k1 to make it recently used
        await cache.get("k1")
        # Adding k3 should evict k2 (now least recently used)
        await cache.set("k3", entry3, ttl=3600)
        assert await cache.get("k1") is not None
        assert await cache.get("k2") is None  # Evicted
        assert await cache.get("k3") is not None

    async def test_overwrite_existing_key(self, sample_entry):
        cache = InMemoryCache()
        await cache.set("key1", sample_entry, ttl=60)
        new_entry = CacheEntry(
            response={"new": True},
            model="gpt-4",
            created_at=time.time(),
            token_count=10,
            cache_key="key1",
        )
        await cache.set("key1", new_entry, ttl=60)
        result = await cache.get("key1")
        assert result is not None
        assert result.response == {"new": True}
        assert cache.size == 1

    async def test_clear(self, sample_entry):
        cache = InMemoryCache()
        await cache.set("key1", sample_entry, ttl=60)
        await cache.get("key1")
        await cache.get("nonexistent")
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0

    async def test_get_similar_returns_none(self):
        """In-memory cache does not support semantic search."""
        cache = InMemoryCache()
        result = await cache.get_similar([0.1, 0.2], "gpt-4", 0.95)
        assert result is None


# =============================================================================
# RedisCacheStore
# =============================================================================


class TestRedisCacheStore:
    async def test_get_miss(self, mock_redis):
        store = RedisCacheStore(redis_client=mock_redis)
        result = await store.get("missing-key")
        assert result is None

    async def test_get_hit(self, mock_redis, sample_entry):
        entry_dict = {
            "response": sample_entry.response,
            "model": sample_entry.model,
            "created_at": sample_entry.created_at,
            "token_count": sample_entry.token_count,
            "cache_key": sample_entry.cache_key,
            "embedding": None,
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(entry_dict))
        store = RedisCacheStore(redis_client=mock_redis)
        result = await store.get("routeiq:cache:v1:abc123")
        assert result is not None
        assert result.model == "gpt-4"
        assert result.token_count == 42

    async def test_set(self, mock_redis, sample_entry):
        store = RedisCacheStore(redis_client=mock_redis)
        await store.set("key1", sample_entry, ttl=3600)
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args.kwargs.get("ex") == 3600

    async def test_get_handles_redis_error(self, mock_redis):
        mock_redis.get = AsyncMock(side_effect=ConnectionError("connection lost"))
        store = RedisCacheStore(redis_client=mock_redis)
        result = await store.get("key1")
        assert result is None

    async def test_set_handles_redis_error(self, mock_redis, sample_entry):
        mock_redis.set = AsyncMock(side_effect=ConnectionError("connection lost"))
        store = RedisCacheStore(redis_client=mock_redis)
        # Should not raise
        await store.set("key1", sample_entry, ttl=3600)

    async def test_get_handles_invalid_json(self, mock_redis):
        mock_redis.get = AsyncMock(return_value="not-json{{{")
        store = RedisCacheStore(redis_client=mock_redis)
        result = await store.get("key1")
        assert result is None

    async def test_key_prefixing(self, mock_redis):
        store = RedisCacheStore(
            redis_client=mock_redis, key_prefix="test:", user_id="u1", team_id="t1"
        )
        await store.get("mykey")
        mock_redis.get.assert_called_with("test:u1:t1:mykey")

    async def test_key_already_prefixed(self, mock_redis):
        store = RedisCacheStore(
            redis_client=mock_redis, key_prefix="test:", user_id="u1", team_id="t1"
        )
        await store.get("test:u1:t1:mykey")
        mock_redis.get.assert_called_with("test:u1:t1:mykey")

    async def test_default_user_team_prefix(self, mock_redis):
        """Default user_id/team_id use '_' placeholder."""
        store = RedisCacheStore(redis_client=mock_redis, key_prefix="routeiq:")
        await store.get("mykey")
        mock_redis.get.assert_called_with("routeiq:_:_:mykey")


# =============================================================================
# _cosine_similarity
# =============================================================================


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# =============================================================================
# SemanticCachePlugin
# =============================================================================


class TestSemanticCachePlugin:
    async def test_disabled_by_default(self):
        """Plugin is a no-op when CACHE_ENABLED is not set."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        with patch.dict("os.environ", {}, clear=False):
            await plugin.startup(app)

        result = await plugin.on_llm_pre_call(
            "gpt-4", [{"role": "user", "content": "hi"}], {}
        )
        assert result is None

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_startup_initializes_l1(self):
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)
        assert plugin._l1 is not None
        assert plugin._enabled is True
        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true", "CACHE_L1_MAX_SIZE": "50"})
    async def test_startup_custom_l1_size(self):
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)
        assert plugin._l1 is not None
        assert plugin._l1._max_size == 50
        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_cache_miss_then_store(self):
        """On cache miss, on_llm_success stores the response."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "What is Python?"}]
        kwargs = {"messages": messages, "temperature": 0.0}

        # Pre-call should return None (miss)
        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is None

        # Simulate successful LLM response
        response = MagicMock()
        response.model_dump.return_value = {
            "choices": [{"message": {"content": "It's a language."}}],
            "usage": {"total_tokens": 20},
        }
        await plugin.on_llm_success("gpt-4", response, kwargs)

        # Now pre-call should return cached response (hit)
        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is not None
        assert result["metadata"]["_cache_hit"] is True
        assert result["metadata"]["_cache_tier"] == "l1"

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_cache_hit_not_re_cached(self):
        """Responses from cache hits are not re-stored."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test"}]

        # Store a response
        response = MagicMock()
        response.model_dump.return_value = {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"total_tokens": 10},
        }
        kwargs = {"messages": messages, "temperature": 0.0}
        await plugin.on_llm_success("gpt-4", response, kwargs)

        # Get the hit
        hit_result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert hit_result is not None

        # Simulate on_llm_success with cache hit metadata - should not re-cache
        initial_size = plugin._l1.size
        kwargs_with_hit = {
            "messages": messages,
            "temperature": 0.0,
            "metadata": {"_cache_hit": True},
        }
        await plugin.on_llm_success("gpt-4", response, kwargs_with_hit)
        assert plugin._l1.size == initial_size

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_streaming_not_cached(self):
        """Streaming requests are not cached."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test"}]
        kwargs = {"messages": messages, "temperature": 0.0, "stream": True}

        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is None

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_high_temperature_not_cached(self):
        """High temperature requests are not cached."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test"}]
        kwargs = {"messages": messages, "temperature": 0.9}

        # Pre-call miss
        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is None

        # on_llm_success should also skip (not cacheable)
        response = MagicMock()
        response.model_dump.return_value = {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"total_tokens": 10},
        }
        await plugin.on_llm_success("gpt-4", response, kwargs)
        assert plugin._l1.size == 0

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_no_cache_header(self):
        """Requests with no-cache header bypass cache."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test"}]
        kwargs = {
            "messages": messages,
            "temperature": 0.0,
            "litellm_params": {
                "proxy_server_request": {
                    "headers": {"x-routeiq-cache-control": "no-cache"}
                }
            },
        }

        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is None

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_dict_response_cached(self):
        """Dict responses are cached directly."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test"}]
        kwargs = {"messages": messages, "temperature": 0.0}
        response_dict = {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"total_tokens": 10},
        }

        await plugin.on_llm_success("gpt-4", response_dict, kwargs)
        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is not None
        assert result["metadata"]["_cache_hit"] is True

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_health_check_enabled(self):
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        health = await plugin.health_check()
        assert health["status"] == "ok"
        assert health["cache_enabled"] is True
        assert health["l1_size"] == 0

        await plugin.shutdown(app)

    async def test_health_check_disabled(self):
        plugin = SemanticCachePlugin()
        health = await plugin.health_check()
        assert health["status"] == "ok"
        assert health["cache_enabled"] is False

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_shutdown_clears_state(self):
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)
        assert plugin._l1 is not None
        await plugin.shutdown(app)
        assert plugin._l1 is None

    def test_metadata(self):
        plugin = SemanticCachePlugin()
        meta = plugin.metadata
        assert meta.name == "semantic-cache"
        assert meta.priority == 10
        assert PluginCapability.MIDDLEWARE in meta.capabilities
        assert meta.failure_mode == FailureMode.CONTINUE

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_missing_model_in_request(self):
        """Request with empty model is not cached."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "hi"}]
        kwargs = {"messages": messages}

        result = await plugin.on_llm_pre_call("", messages, kwargs)
        assert result is None

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_empty_messages_in_request(self):
        """Request with empty messages is not cached."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"messages": messages}

        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is None

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_cache_hit_response_structure(self):
        """Cache hit response contains expected metadata fields."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test"}]
        kwargs = {"messages": messages, "temperature": 0.0}
        response = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        await plugin.on_llm_success("gpt-4", response, kwargs)

        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is not None
        meta = result["metadata"]
        assert "_cache_hit" in meta
        assert "_cache_hit_response" in meta
        assert "_cache_tier" in meta
        assert "_cache_key" in meta
        assert "_cache_age_seconds" in meta

        await plugin.shutdown(app)

    @patch.dict(
        "os.environ",
        {"CACHE_ENABLED": "true", "CACHE_MAX_TEMPERATURE": "1.0"},
    )
    async def test_custom_max_temperature(self):
        """Custom max temperature via env var is respected."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test"}]
        kwargs = {"messages": messages, "temperature": 0.5}
        response = {"choices": [{"message": {"content": "ok"}}], "usage": {}}

        await plugin.on_llm_success("gpt-4", response, kwargs)
        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is not None
        assert result["metadata"]["_cache_hit"] is True

        await plugin.shutdown(app)

    @patch.dict(
        "os.environ",
        {"CACHE_ENABLED": "true", "CACHE_TTL_SECONDS": "120"},
    )
    async def test_custom_ttl(self):
        """Custom TTL from environment is applied."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)
        assert plugin._ttl == 120
        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_response_with_legacy_dict_method(self):
        """Response objects with .dict() (Pydantic v1) are handled."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test"}]
        kwargs = {"messages": messages, "temperature": 0.0}

        response = MagicMock(spec=[])
        del response.model_dump  # Remove model_dump to simulate v1
        response.dict = MagicMock(
            return_value={
                "choices": [{"message": {"content": "v1 answer"}}],
                "usage": {"total_tokens": 5},
            }
        )

        await plugin.on_llm_success("gpt-4", response, kwargs)
        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is not None

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_unserializable_response_skipped(self):
        """Responses that fail serialization are not cached."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test"}]
        kwargs = {"messages": messages, "temperature": 0.0}

        response = MagicMock()
        response.model_dump.side_effect = Exception("serialization failed")
        response.dict.side_effect = Exception("serialization failed")

        # Should not raise
        await plugin.on_llm_success("gpt-4", response, kwargs)
        assert plugin._l1.size == 0

        await plugin.shutdown(app)


# =============================================================================
# CacheManager
# =============================================================================


class TestCacheManager:
    def test_get_stats_basic(self):
        """get_stats returns expected structure."""
        l1 = InMemoryCache(max_size=100)
        manager = CacheManager(l1=l1, ttl=3600, semantic_enabled=False)
        stats = manager.get_stats()

        assert stats["enabled"] is True
        assert stats["semantic_enabled"] is False
        assert stats["ttl_seconds"] == 3600
        assert stats["total_hits"] == 0
        assert stats["total_misses"] == 0
        assert stats["l1"]["size"] == 0
        assert stats["l1"]["max_size"] == 100
        assert stats["l2_configured"] is False

    def test_record_hit_and_miss(self):
        """record_hit/miss increment aggregate counters."""
        l1 = InMemoryCache()
        manager = CacheManager(l1=l1)

        manager.record_hit()
        manager.record_hit()
        manager.record_miss()

        stats = manager.get_stats()
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 1

    async def test_flush_all(self, sample_entry):
        """flush() without prefix clears all L1 entries and resets counters."""
        l1 = InMemoryCache()
        await l1.set("k1", sample_entry, ttl=3600)
        await l1.set("k2", sample_entry, ttl=3600)

        manager = CacheManager(l1=l1)
        manager.record_hit()
        manager.record_miss()

        count = await manager.flush()
        assert count == 2
        assert l1.size == 0
        assert manager._total_hits == 0
        assert manager._total_misses == 0

    async def test_flush_with_prefix(self, sample_entry):
        """flush(prefix) removes only matching entries."""
        l1 = InMemoryCache()
        await l1.set("routeiq:cache:v1:a:123", sample_entry, ttl=3600)
        await l1.set("routeiq:cache:v1:a:456", sample_entry, ttl=3600)
        await l1.set("routeiq:cache:v1:b:789", sample_entry, ttl=3600)

        manager = CacheManager(l1=l1)
        count = await manager.flush(prefix="routeiq:cache:v1:a:")
        assert count == 2
        assert l1.size == 1

    async def test_flush_no_l1(self):
        """flush() with no L1 returns 0."""
        manager = CacheManager(l1=None)
        count = await manager.flush()
        assert count == 0

    def test_list_entries_empty(self):
        """list_entries returns empty when no entries."""
        l1 = InMemoryCache()
        manager = CacheManager(l1=l1)
        result = manager.list_entries()
        assert result["enabled"] is True
        assert result["count"] == 0
        assert result["entries"] == []

    async def test_list_entries_with_data(self, sample_entry):
        """list_entries returns entry metadata."""
        l1 = InMemoryCache()
        await l1.set("key1", sample_entry, ttl=3600)

        manager = CacheManager(l1=l1)
        result = manager.list_entries()
        assert result["count"] == 1
        assert len(result["entries"]) == 1
        entry = result["entries"][0]
        assert entry["key"] == "key1"
        assert entry["model"] == "gpt-4"
        assert entry["token_count"] == 42
        assert entry["ttl_remaining"] > 0
        assert entry["has_embedding"] is False

    async def test_list_entries_with_limit(self, sample_entry):
        """list_entries respects the limit parameter."""
        l1 = InMemoryCache()
        for i in range(5):
            await l1.set(f"key{i}", sample_entry, ttl=3600)

        manager = CacheManager(l1=l1)
        result = manager.list_entries(limit=3)
        assert result["count"] == 3

    def test_list_entries_no_l1(self):
        """list_entries with no L1 returns empty."""
        manager = CacheManager(l1=None)
        result = manager.list_entries()
        assert result["enabled"] is True
        assert result["count"] == 0


# =============================================================================
# CacheManager: SSE Streaming Replay
# =============================================================================


class TestSSEStreamingReplay:
    async def test_replay_basic_response(self):
        """replay_as_sse_stream produces valid SSE format."""
        l1 = InMemoryCache()
        manager = CacheManager(l1=l1)

        cached_response = {
            "model": "gpt-4",
            "choices": [{"message": {"content": "Hello world"}}],
        }

        chunks = []
        async for chunk in manager.replay_as_sse_stream(cached_response):
            chunks.append(chunk)

        assert len(chunks) == 2
        # First chunk should be SSE data with content
        assert chunks[0].startswith("data: ")
        assert chunks[0].endswith("\n\n")
        # Parse the JSON from the SSE data line
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["object"] == "chat.completion.chunk"
        assert data["model"] == "gpt-4"
        assert data["choices"][0]["delta"]["content"] == "Hello world"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["id"].startswith("chatcmpl-cache-")

        # Last chunk should be [DONE]
        assert chunks[1] == "data: [DONE]\n\n"

    async def test_replay_empty_response(self):
        """replay_as_sse_stream handles empty choices."""
        l1 = InMemoryCache()
        manager = CacheManager(l1=l1)

        cached_response = {"model": "gpt-3.5-turbo", "choices": []}

        chunks = []
        async for chunk in manager.replay_as_sse_stream(cached_response):
            chunks.append(chunk)

        assert len(chunks) == 2
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["choices"][0]["delta"]["content"] == ""
        assert data["model"] == "gpt-3.5-turbo"

    async def test_replay_no_model(self):
        """replay_as_sse_stream uses default model when not present."""
        l1 = InMemoryCache()
        manager = CacheManager(l1=l1)

        cached_response = {
            "choices": [{"message": {"content": "test"}}],
        }

        chunks = []
        async for chunk in manager.replay_as_sse_stream(cached_response):
            chunks.append(chunk)

        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["model"] == "cached-model"


# =============================================================================
# Cache Manager Singleton
# =============================================================================


class TestCacheManagerSingleton:
    def test_get_cache_manager_default_none(self):
        """get_cache_manager returns None when not set."""
        assert get_cache_manager() is None

    def test_set_and_get_cache_manager(self):
        """set/get cache manager works."""
        l1 = InMemoryCache()
        manager = CacheManager(l1=l1)
        set_cache_manager(manager)
        assert get_cache_manager() is manager

    def test_clear_cache_manager(self):
        """Setting to None clears the singleton."""
        l1 = InMemoryCache()
        manager = CacheManager(l1=l1)
        set_cache_manager(manager)
        set_cache_manager(None)
        assert get_cache_manager() is None


# =============================================================================
# Cache Response Headers
# =============================================================================


class TestCacheResponseHeaders:
    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_hit_response_includes_cache_headers(self):
        """Cache hit response includes _cache_headers with HIT, tier, and age."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "test headers"}]
        kwargs = {"messages": messages, "temperature": 0.0}
        response = {
            "choices": [{"message": {"content": "cached"}}],
            "usage": {"total_tokens": 5},
        }
        await plugin.on_llm_success("gpt-4", response, kwargs)

        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert result is not None

        # Check _cache_headers key
        headers = result["_cache_headers"]
        assert headers["x-routeiq-cache"] == "HIT"
        assert headers["x-routeiq-cache-tier"] == "l1"
        assert "x-routeiq-cache-age" in headers
        # Age should be a string-formatted integer
        age_val = int(headers["x-routeiq-cache-age"])
        assert age_val >= 0

        await plugin.shutdown(app)

    def test_miss_headers(self):
        """build_miss_headers returns MISS status."""
        headers = SemanticCachePlugin.build_miss_headers()
        assert headers["x-routeiq-cache"] == "MISS"

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_cache_hit_tier_l1(self):
        """L1 cache hit has tier=l1 in headers."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "tier test"}]
        kwargs = {"messages": messages, "temperature": 0.0}
        response = {
            "choices": [{"message": {"content": "l1 answer"}}],
            "usage": {"total_tokens": 5},
        }
        await plugin.on_llm_success("gpt-4", response, kwargs)
        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs)

        assert result["_cache_headers"]["x-routeiq-cache-tier"] == "l1"
        assert result["metadata"]["_cache_tier"] == "l1"

        await plugin.shutdown(app)


# =============================================================================
# Per-Request Cache Bypass
# =============================================================================


class TestPerRequestCacheBypass:
    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_no_cache_header_bypasses_cache(self):
        """x-routeiq-cache-control: no-cache skips cache lookup."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "bypass test"}]

        # Store a response first
        kwargs_store = {"messages": messages, "temperature": 0.0}
        response = {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"total_tokens": 5},
        }
        await plugin.on_llm_success("gpt-4", response, kwargs_store)

        # Verify it's cached without bypass
        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs_store)
        assert result is not None
        assert result["metadata"]["_cache_hit"] is True

        # Now use no-cache header to bypass
        kwargs_bypass = {
            "messages": messages,
            "temperature": 0.0,
            "litellm_params": {
                "proxy_server_request": {
                    "headers": {"x-routeiq-cache-control": "no-cache"}
                }
            },
        }
        result = await plugin.on_llm_pre_call("gpt-4", messages, kwargs_bypass)
        assert result is None  # Bypassed - cache miss

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_no_cache_header_also_prevents_storage(self):
        """x-routeiq-cache-control: no-cache also prevents storing response."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        messages = [{"role": "user", "content": "no store test"}]
        kwargs = {
            "messages": messages,
            "temperature": 0.0,
            "litellm_params": {
                "proxy_server_request": {
                    "headers": {"x-routeiq-cache-control": "no-cache"}
                }
            },
        }
        response = {
            "choices": [{"message": {"content": "should not be cached"}}],
            "usage": {"total_tokens": 5},
        }
        await plugin.on_llm_success("gpt-4", response, kwargs)
        assert plugin._l1.size == 0  # Not stored

        await plugin.shutdown(app)


# =============================================================================
# Cache Stats Tracking
# =============================================================================


class TestCacheStatsTracking:
    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_hit_increments_total_hits(self):
        """Cache hit increments CacheManager total_hits."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        manager = get_cache_manager()
        assert manager is not None
        assert manager._total_hits == 0

        messages = [{"role": "user", "content": "stats test"}]
        kwargs = {"messages": messages, "temperature": 0.0}
        response = {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"total_tokens": 5},
        }
        await plugin.on_llm_success("gpt-4", response, kwargs)

        # First lookup is a hit
        await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert manager._total_hits == 1
        assert manager._total_misses == 0

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_miss_increments_total_misses(self):
        """Cache miss increments CacheManager total_misses."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        manager = get_cache_manager()
        assert manager is not None

        messages = [{"role": "user", "content": "miss stats test"}]
        kwargs = {"messages": messages, "temperature": 0.0}

        # Lookup on empty cache is a miss
        await plugin.on_llm_pre_call("gpt-4", messages, kwargs)
        assert manager._total_misses == 1
        assert manager._total_hits == 0

        await plugin.shutdown(app)

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_cache_manager_registered_on_startup(self):
        """CacheManager singleton is registered during plugin startup."""
        assert get_cache_manager() is None

        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        manager = get_cache_manager()
        assert manager is not None
        assert manager._l1 is plugin._l1

        await plugin.shutdown(app)
        assert get_cache_manager() is None

    @patch.dict("os.environ", {"CACHE_ENABLED": "true"})
    async def test_stats_endpoint_integration(self):
        """get_stats reflects actual cache activity."""
        plugin = SemanticCachePlugin()
        app = MagicMock()
        await plugin.startup(app)

        manager = get_cache_manager()
        assert manager is not None

        messages = [{"role": "user", "content": "stats integration"}]
        kwargs = {"messages": messages, "temperature": 0.0}
        response = {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"total_tokens": 10},
        }

        # Store
        await plugin.on_llm_success("gpt-4", response, kwargs)

        # Miss on different query
        diff_msgs = [{"role": "user", "content": "different query"}]
        diff_kwargs = {"messages": diff_msgs, "temperature": 0.0}
        await plugin.on_llm_pre_call("gpt-4", diff_msgs, diff_kwargs)

        # Hit on original query
        await plugin.on_llm_pre_call("gpt-4", messages, kwargs)

        stats = manager.get_stats()
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 1
        assert stats["l1"]["size"] == 1

        await plugin.shutdown(app)


# =============================================================================
# Admin Cache Endpoints (unit tests via function call)
# =============================================================================


class TestAdminCacheEndpoints:
    """
    Test the admin cache endpoint logic via the CacheManager singleton.

    These tests verify the same logic that the admin endpoints call,
    testing through the get_cache_manager() singleton rather than
    through FastAPI dependency injection.
    """

    async def test_stats_when_cache_disabled(self):
        """get_cache_manager returns None when cache is not enabled."""
        assert get_cache_manager() is None

    async def test_flush_when_cache_disabled(self):
        """No cache manager means flush is not possible."""
        manager = get_cache_manager()
        assert manager is None

    async def test_entries_when_cache_disabled(self):
        """No cache manager means entries listing is not possible."""
        manager = get_cache_manager()
        assert manager is None

    async def test_stats_with_cache_enabled(self, sample_entry):
        """get_stats returns stats when cache is enabled."""
        l1 = InMemoryCache()
        await l1.set("k1", sample_entry, ttl=3600)
        manager = CacheManager(l1=l1, ttl=3600)
        manager.record_hit()
        set_cache_manager(manager)

        mgr = get_cache_manager()
        assert mgr is not None
        stats = mgr.get_stats()
        assert stats["enabled"] is True
        assert stats["total_hits"] == 1
        assert stats["l1"]["size"] == 1

    async def test_flush_with_cache_enabled(self, sample_entry):
        """flush returns count of removed entries."""
        l1 = InMemoryCache()
        await l1.set("k1", sample_entry, ttl=3600)
        await l1.set("k2", sample_entry, ttl=3600)
        manager = CacheManager(l1=l1, ttl=3600)
        set_cache_manager(manager)

        mgr = get_cache_manager()
        assert mgr is not None
        count = await mgr.flush()
        assert count == 2
        assert l1.size == 0

    async def test_entries_with_cache_enabled(self, sample_entry):
        """list_entries returns entry metadata."""
        l1 = InMemoryCache()
        await l1.set("k1", sample_entry, ttl=3600)
        manager = CacheManager(l1=l1, ttl=3600)
        set_cache_manager(manager)

        mgr = get_cache_manager()
        assert mgr is not None
        result = mgr.list_entries(limit=50)
        assert result["count"] == 1
        assert result["entries"][0]["key"] == "k1"
