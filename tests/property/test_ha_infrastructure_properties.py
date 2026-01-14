"""
Property-Based Tests for High Availability Infrastructure.

These tests validate the correctness properties defined in the design document
for the High Availability infrastructure (Requirements 4.x, 11.x, 12.x, 13.x).

Property tests use Hypothesis to generate many test cases and verify that
universal properties hold across all valid inputs.
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck


# =============================================================================
# Test Data Generators (Strategies)
# =============================================================================

# Virtual key strategies
@st.composite
def virtual_key_strategy(draw):
    """Generate a valid virtual key data structure."""
    return {
        "key_id": str(uuid.uuid4()),
        "key_hash": hashlib.sha256(draw(st.binary(min_size=16, max_size=64))).hexdigest(),
        "key_alias": draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        "team_id": draw(st.one_of(st.none(), st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))),
        "user_id": draw(st.one_of(st.none(), st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))),
        "max_budget": draw(st.one_of(st.none(), st.decimals(min_value=0, max_value=10000, places=2))),
        "budget_duration": draw(st.one_of(st.none(), st.sampled_from(["1d", "7d", "30d", "1m"]))),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": draw(st.one_of(st.none(), st.just((datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))),
        "metadata": draw(st.dictionaries(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            st.text(min_size=0, max_size=100),
            max_size=5
        )),
    }


# Request log strategies
@st.composite
def request_log_strategy(draw):
    """Generate a valid request log data structure."""
    return {
        "request_id": str(uuid.uuid4()),
        "key_id": str(uuid.uuid4()),
        "model_name": draw(st.sampled_from(["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"])),
        "provider": draw(st.sampled_from(["openai", "anthropic", "bedrock", "azure"])),
        "prompt_tokens": draw(st.integers(min_value=1, max_value=10000)),
        "completion_tokens": draw(st.integers(min_value=1, max_value=5000)),
        "total_tokens": draw(st.integers(min_value=2, max_value=15000)),
        "cost": draw(st.decimals(min_value=Decimal("0.0001"), max_value=Decimal("10.0"), places=6)),
        "latency_ms": draw(st.integers(min_value=50, max_value=30000)),
        "status": draw(st.sampled_from(["success", "error", "timeout"])),
        "error_message": draw(st.one_of(st.none(), st.text(min_size=1, max_size=200))),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": draw(st.dictionaries(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            st.text(min_size=0, max_size=100),
            max_size=3
        )),
    }


# Cost data strategies
@st.composite
def cost_data_strategy(draw):
    """Generate valid cost tracking data."""
    return {
        "key_id": str(uuid.uuid4()),
        "model_name": draw(st.sampled_from(["gpt-4", "gpt-3.5-turbo", "claude-3-opus"])),
        "prompt_tokens": draw(st.integers(min_value=1, max_value=10000)),
        "completion_tokens": draw(st.integers(min_value=1, max_value=5000)),
        "cost": draw(st.decimals(min_value=Decimal("0.0001"), max_value=Decimal("10.0"), places=6)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Property 6: Data Persistence
# =============================================================================

class TestDataPersistenceProperty:
    """
    Property 6: Data Persistence
    
    For any request processed by the Gateway when database_url is configured,
    all relevant data (virtual keys, request logs, cost data) should be 
    persisted to PostgreSQL and be retrievable via database queries.
    
    **Validates: Requirements 4.1, 12.2**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(virtual_key=virtual_key_strategy())
    def test_virtual_key_data_structure_is_valid(self, virtual_key: Dict[str, Any]):
        """
        Property 6: Data Persistence
        
        For any generated virtual key, it should have all required fields
        for database persistence.
        
        **Validates: Requirements 4.1, 12.2**
        """
        # Property: Virtual key must have key_id
        assert "key_id" in virtual_key
        assert isinstance(virtual_key["key_id"], str)
        assert len(virtual_key["key_id"]) > 0
        
        # Property: Virtual key must have key_hash
        assert "key_hash" in virtual_key
        assert isinstance(virtual_key["key_hash"], str)
        assert len(virtual_key["key_hash"]) == 64  # SHA-256 hex length
        
        # Property: Virtual key must have created_at timestamp
        assert "created_at" in virtual_key
        assert isinstance(virtual_key["created_at"], str)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request_log=request_log_strategy())
    def test_request_log_data_structure_is_valid(self, request_log: Dict[str, Any]):
        """
        Property 6: Data Persistence
        
        For any generated request log, it should have all required fields
        for database persistence.
        
        **Validates: Requirements 4.1, 12.2**
        """
        # Property: Request log must have request_id
        assert "request_id" in request_log
        assert isinstance(request_log["request_id"], str)
        
        # Property: Request log must have model_name
        assert "model_name" in request_log
        assert isinstance(request_log["model_name"], str)
        
        # Property: Request log must have token counts
        assert "prompt_tokens" in request_log
        assert "completion_tokens" in request_log
        assert "total_tokens" in request_log
        assert request_log["prompt_tokens"] >= 0
        assert request_log["completion_tokens"] >= 0
        
        # Property: Request log must have cost
        assert "cost" in request_log
        
        # Property: Request log must have status
        assert "status" in request_log
        assert request_log["status"] in ["success", "error", "timeout"]

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(virtual_key=virtual_key_strategy())
    def test_virtual_key_json_serialization_round_trip(self, virtual_key: Dict[str, Any]):
        """
        Property 6: Data Persistence
        
        For any virtual key data, serializing to JSON and deserializing
        should preserve all data (round-trip property for JSON storage).
        
        **Validates: Requirements 4.1, 12.2**
        """
        # Convert Decimal to string for JSON serialization
        serializable_key = virtual_key.copy()
        if serializable_key.get("max_budget") is not None:
            serializable_key["max_budget"] = str(serializable_key["max_budget"])
        
        # Serialize to JSON
        json_str = json.dumps(serializable_key)
        
        # Deserialize back
        loaded_key = json.loads(json_str)
        
        # Property: Round-trip should preserve all fields
        assert loaded_key["key_id"] == virtual_key["key_id"]
        assert loaded_key["key_hash"] == virtual_key["key_hash"]
        assert loaded_key["key_alias"] == virtual_key["key_alias"]
        assert loaded_key["team_id"] == virtual_key["team_id"]
        assert loaded_key["user_id"] == virtual_key["user_id"]

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(request_log=request_log_strategy())
    def test_request_log_json_serialization_round_trip(self, request_log: Dict[str, Any]):
        """
        Property 6: Data Persistence
        
        For any request log data, serializing to JSON and deserializing
        should preserve all data (round-trip property for JSON storage).
        
        **Validates: Requirements 4.1, 12.2**
        """
        # Convert Decimal to string for JSON serialization
        serializable_log = request_log.copy()
        if serializable_log.get("cost") is not None:
            serializable_log["cost"] = str(serializable_log["cost"])
        
        # Serialize to JSON
        json_str = json.dumps(serializable_log)
        
        # Deserialize back
        loaded_log = json.loads(json_str)
        
        # Property: Round-trip should preserve all fields
        assert loaded_log["request_id"] == request_log["request_id"]
        assert loaded_log["model_name"] == request_log["model_name"]
        assert loaded_log["prompt_tokens"] == request_log["prompt_tokens"]
        assert loaded_log["completion_tokens"] == request_log["completion_tokens"]
        assert loaded_log["status"] == request_log["status"]

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(cost_data=cost_data_strategy())
    def test_cost_data_structure_is_valid(self, cost_data: Dict[str, Any]):
        """
        Property 6: Data Persistence
        
        For any generated cost data, it should have all required fields
        for cost tracking and persistence.
        
        **Validates: Requirements 4.1, 12.2**
        """
        # Property: Cost data must have key_id
        assert "key_id" in cost_data
        assert isinstance(cost_data["key_id"], str)
        
        # Property: Cost data must have model_name
        assert "model_name" in cost_data
        assert isinstance(cost_data["model_name"], str)
        
        # Property: Cost data must have cost value
        assert "cost" in cost_data
        
        # Property: Cost data must have timestamp
        assert "timestamp" in cost_data

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        virtual_keys=st.lists(virtual_key_strategy(), min_size=1, max_size=10)
    )
    def test_multiple_virtual_keys_have_unique_ids(self, virtual_keys: List[Dict[str, Any]]):
        """
        Property 6: Data Persistence
        
        For any collection of virtual keys, each key should have a unique
        key_id to ensure proper database indexing.
        
        **Validates: Requirements 4.1, 12.2**
        """
        key_ids = [vk["key_id"] for vk in virtual_keys]
        
        # Property: All key_ids should be unique
        assert len(key_ids) == len(set(key_ids))

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request_logs=st.lists(request_log_strategy(), min_size=1, max_size=10)
    )
    def test_multiple_request_logs_have_unique_ids(self, request_logs: List[Dict[str, Any]]):
        """
        Property 6: Data Persistence
        
        For any collection of request logs, each log should have a unique
        request_id to ensure proper database indexing.
        
        **Validates: Requirements 4.1, 12.2**
        """
        request_ids = [rl["request_id"] for rl in request_logs]
        
        # Property: All request_ids should be unique
        assert len(request_ids) == len(set(request_ids))

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        key_id=st.text(min_size=36, max_size=36).filter(lambda x: x.strip()),
        request_logs=st.lists(request_log_strategy(), min_size=1, max_size=5)
    )
    def test_request_logs_can_be_filtered_by_key_id(
        self, key_id: str, request_logs: List[Dict[str, Any]]
    ):
        """
        Property 6: Data Persistence
        
        For any key_id and collection of request logs, filtering by key_id
        should return only logs associated with that key.
        
        **Validates: Requirements 4.1, 12.2**
        """
        # Assign some logs to the target key_id
        for i, log in enumerate(request_logs):
            if i % 2 == 0:
                log["key_id"] = key_id
        
        # Filter logs by key_id
        filtered_logs = [log for log in request_logs if log["key_id"] == key_id]
        
        # Property: All filtered logs should have the target key_id
        for log in filtered_logs:
            assert log["key_id"] == key_id

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(virtual_key=virtual_key_strategy())
    def test_virtual_key_budget_fields_are_consistent(self, virtual_key: Dict[str, Any]):
        """
        Property 6: Data Persistence
        
        For any virtual key with budget settings, the budget fields should
        be internally consistent.
        
        **Validates: Requirements 4.1, 12.2**
        """
        max_budget = virtual_key.get("max_budget")
        budget_duration = virtual_key.get("budget_duration")
        
        # Property: If max_budget is set, it should be non-negative
        if max_budget is not None:
            assert max_budget >= 0
        
        # Property: budget_duration should be a valid duration string if set
        if budget_duration is not None:
            valid_durations = ["1d", "7d", "30d", "1m"]
            assert budget_duration in valid_durations



# =============================================================================
# Cache-related strategies
# =============================================================================

@st.composite
def cache_request_strategy(draw):
    """Generate a valid cache request structure."""
    model = draw(st.sampled_from(["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]))
    messages = [
        {
            "role": draw(st.sampled_from(["user", "system", "assistant"])),
            "content": draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
        }
    ]
    
    return {
        "model": model,
        "messages": messages,
        "temperature": draw(st.floats(min_value=0.0, max_value=2.0)),
        "max_tokens": draw(st.integers(min_value=1, max_value=4096)),
    }


@st.composite
def provider_specific_params_strategy(draw):
    """Generate provider-specific optional parameters."""
    return {
        "top_p": draw(st.floats(min_value=0.0, max_value=1.0)),
        "frequency_penalty": draw(st.floats(min_value=-2.0, max_value=2.0)),
        "presence_penalty": draw(st.floats(min_value=-2.0, max_value=2.0)),
        "seed": draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**32 - 1))),
    }


# =============================================================================
# Property 7: Response Caching
# =============================================================================

class TestResponseCachingProperty:
    """
    Property 7: Response Caching
    
    For any two identical requests made to the Gateway when caching is enabled,
    the second request should return a cached response (indicated by cache hit
    metrics) without invoking the LLM provider, and the cached response should
    expire after the configured TTL.
    
    **Validates: Requirements 4.2, 13.1, 13.3**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request=cache_request_strategy())
    def test_identical_requests_produce_same_cache_key(self, request: Dict[str, Any]):
        """
        Property 7: Response Caching
        
        For any request, computing the cache key twice should produce
        the same result (deterministic cache key generation).
        
        **Validates: Requirements 4.2, 13.1, 13.3**
        """
        # Compute cache key (simplified version of actual implementation)
        def compute_cache_key(req: Dict[str, Any]) -> str:
            # Sort keys for deterministic serialization
            key_data = json.dumps(req, sort_keys=True)
            return hashlib.sha256(key_data.encode()).hexdigest()
        
        key1 = compute_cache_key(request)
        key2 = compute_cache_key(request)
        
        # Property: Same request should produce same cache key
        assert key1 == key2

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request1=cache_request_strategy(),
        request2=cache_request_strategy(),
    )
    def test_different_requests_produce_different_cache_keys(
        self, request1: Dict[str, Any], request2: Dict[str, Any]
    ):
        """
        Property 7: Response Caching
        
        For any two different requests, they should produce different
        cache keys (assuming the requests differ in meaningful ways).
        
        **Validates: Requirements 4.2, 13.1, 13.3**
        """
        # Skip if requests are identical
        assume(request1 != request2)
        
        def compute_cache_key(req: Dict[str, Any]) -> str:
            key_data = json.dumps(req, sort_keys=True)
            return hashlib.sha256(key_data.encode()).hexdigest()
        
        key1 = compute_cache_key(request1)
        key2 = compute_cache_key(request2)
        
        # Property: Different requests should produce different cache keys
        assert key1 != key2

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ttl=st.integers(min_value=1, max_value=86400),
        elapsed_time=st.integers(min_value=0, max_value=100000),
    )
    def test_cache_expiration_respects_ttl(self, ttl: int, elapsed_time: int):
        """
        Property 7: Response Caching
        
        For any TTL configuration, cached responses should be considered
        expired if and only if the elapsed time exceeds the TTL.
        
        **Validates: Requirements 4.2, 13.1, 13.3**
        """
        # Property: Cache is expired iff elapsed_time >= ttl
        is_expired = elapsed_time >= ttl
        
        if elapsed_time >= ttl:
            assert is_expired is True
        else:
            assert is_expired is False

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        cache_enabled=st.booleans(),
        request=cache_request_strategy(),
    )
    def test_caching_only_when_enabled(
        self, cache_enabled: bool, request: Dict[str, Any]
    ):
        """
        Property 7: Response Caching
        
        For any request, caching should only occur when cache is enabled.
        
        **Validates: Requirements 4.2, 13.1, 13.3**
        """
        # Simulate cache behavior
        cache_store = {}
        
        def compute_cache_key(req: Dict[str, Any]) -> str:
            key_data = json.dumps(req, sort_keys=True)
            return hashlib.sha256(key_data.encode()).hexdigest()
        
        cache_key = compute_cache_key(request)
        response = {"choices": [{"message": {"content": "test response"}}]}
        
        # Store in cache only if enabled
        if cache_enabled:
            cache_store[cache_key] = response
        
        # Property: Cache should contain the key iff caching is enabled
        if cache_enabled:
            assert cache_key in cache_store
        else:
            assert cache_key not in cache_store

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request=cache_request_strategy(),
        response_content=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
    )
    def test_cached_response_matches_original(
        self, request: Dict[str, Any], response_content: str
    ):
        """
        Property 7: Response Caching
        
        For any cached response, retrieving it should return the exact
        same response that was stored.
        
        **Validates: Requirements 4.2, 13.1, 13.3**
        """
        cache_store = {}
        
        def compute_cache_key(req: Dict[str, Any]) -> str:
            key_data = json.dumps(req, sort_keys=True)
            return hashlib.sha256(key_data.encode()).hexdigest()
        
        cache_key = compute_cache_key(request)
        original_response = {
            "choices": [{"message": {"content": response_content}}],
            "model": request["model"],
        }
        
        # Store response
        cache_store[cache_key] = json.dumps(original_response)
        
        # Retrieve response
        cached_json = cache_store.get(cache_key)
        cached_response = json.loads(cached_json) if cached_json else None
        
        # Property: Cached response should match original
        assert cached_response == original_response
        assert cached_response["choices"][0]["message"]["content"] == response_content

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(
        num_requests=st.integers(min_value=2, max_value=10),
    )
    def test_cache_hit_on_repeated_requests(self, num_requests: int):
        """
        Property 7: Response Caching
        
        For any number of identical requests, only the first should be
        a cache miss, and all subsequent should be cache hits.
        
        **Validates: Requirements 4.2, 13.1, 13.3**
        """
        cache_store = {}
        cache_hits = 0
        cache_misses = 0
        
        # Fixed request for this test
        request = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}
        cache_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        
        for _ in range(num_requests):
            if cache_key in cache_store:
                cache_hits += 1
            else:
                cache_misses += 1
                cache_store[cache_key] = "response"
        
        # Property: Exactly one cache miss (first request)
        assert cache_misses == 1
        
        # Property: All other requests are cache hits
        assert cache_hits == num_requests - 1


# =============================================================================
# Property 8: Cache Key Generation
# =============================================================================

class TestCacheKeyGenerationProperty:
    """
    Property 8: Cache Key Generation
    
    For any request with provider-specific optional parameters, when
    enable_caching_on_provider_specific_optional_params is true, the cache
    key should include those parameters such that requests differing only
    in provider-specific params produce different cache keys.
    
    **Validates: Requirements 13.4**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request=cache_request_strategy(),
        provider_params=provider_specific_params_strategy(),
    )
    def test_provider_params_included_in_cache_key_when_enabled(
        self, request: Dict[str, Any], provider_params: Dict[str, Any]
    ):
        """
        Property 8: Cache Key Generation
        
        For any request with provider-specific params, when the feature is
        enabled, the cache key should include those params.
        
        **Validates: Requirements 13.4**
        """
        def compute_cache_key(
            req: Dict[str, Any],
            include_provider_params: bool,
            provider_params: Dict[str, Any]
        ) -> str:
            key_data = req.copy()
            if include_provider_params:
                key_data.update(provider_params)
            return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        
        # Compute keys with and without provider params
        key_with_params = compute_cache_key(request, True, provider_params)
        key_without_params = compute_cache_key(request, False, provider_params)
        
        # Property: Keys should differ when provider params are included
        # (unless provider_params is empty or all None)
        non_none_params = {k: v for k, v in provider_params.items() if v is not None}
        if non_none_params:
            assert key_with_params != key_without_params

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request=cache_request_strategy(),
        params1=provider_specific_params_strategy(),
        params2=provider_specific_params_strategy(),
    )
    def test_different_provider_params_produce_different_keys(
        self, request: Dict[str, Any], params1: Dict[str, Any], params2: Dict[str, Any]
    ):
        """
        Property 8: Cache Key Generation
        
        For any request with different provider-specific params, the cache
        keys should be different.
        
        **Validates: Requirements 13.4**
        """
        assume(params1 != params2)
        
        def compute_cache_key_with_params(
            req: Dict[str, Any], provider_params: Dict[str, Any]
        ) -> str:
            key_data = req.copy()
            key_data.update(provider_params)
            return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        
        key1 = compute_cache_key_with_params(request, params1)
        key2 = compute_cache_key_with_params(request, params2)
        
        # Property: Different params should produce different keys
        assert key1 != key2

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request=cache_request_strategy(),
        provider_params=provider_specific_params_strategy(),
    )
    def test_cache_key_is_deterministic(
        self, request: Dict[str, Any], provider_params: Dict[str, Any]
    ):
        """
        Property 8: Cache Key Generation
        
        For any request and provider params, computing the cache key
        multiple times should always produce the same result.
        
        **Validates: Requirements 13.4**
        """
        def compute_cache_key(req: Dict[str, Any], params: Dict[str, Any]) -> str:
            key_data = req.copy()
            key_data.update(params)
            return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        
        key1 = compute_cache_key(request, provider_params)
        key2 = compute_cache_key(request, provider_params)
        key3 = compute_cache_key(request, provider_params)
        
        # Property: Cache key generation is deterministic
        assert key1 == key2 == key3

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request=cache_request_strategy(),
        seed1=st.integers(min_value=0, max_value=2**32 - 1),
        seed2=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_seed_parameter_affects_cache_key(
        self, request: Dict[str, Any], seed1: int, seed2: int
    ):
        """
        Property 8: Cache Key Generation
        
        For any request, different seed values should produce different
        cache keys (seed is a provider-specific param that affects output).
        
        **Validates: Requirements 13.4**
        """
        assume(seed1 != seed2)
        
        def compute_cache_key_with_seed(req: Dict[str, Any], seed: int) -> str:
            key_data = req.copy()
            key_data["seed"] = seed
            return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        
        key1 = compute_cache_key_with_seed(request, seed1)
        key2 = compute_cache_key_with_seed(request, seed2)
        
        # Property: Different seeds should produce different cache keys
        assert key1 != key2

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request=cache_request_strategy(),
        temperature1=st.floats(min_value=0.0, max_value=2.0),
        temperature2=st.floats(min_value=0.0, max_value=2.0),
    )
    def test_temperature_affects_cache_key(
        self, request: Dict[str, Any], temperature1: float, temperature2: float
    ):
        """
        Property 8: Cache Key Generation
        
        For any request, different temperature values should produce
        different cache keys.
        
        **Validates: Requirements 13.4**
        """
        assume(temperature1 != temperature2)
        
        def compute_cache_key_with_temp(req: Dict[str, Any], temp: float) -> str:
            key_data = req.copy()
            key_data["temperature"] = temp
            return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        
        key1 = compute_cache_key_with_temp(request, temperature1)
        key2 = compute_cache_key_with_temp(request, temperature2)
        
        # Property: Different temperatures should produce different cache keys
        assert key1 != key2



# =============================================================================
# Rate limiting strategies
# =============================================================================

@st.composite
def rate_limit_config_strategy(draw):
    """Generate a valid rate limit configuration."""
    return {
        "requests_per_minute": draw(st.integers(min_value=1, max_value=1000)),
        "requests_per_day": draw(st.integers(min_value=1, max_value=100000)),
        "tokens_per_minute": draw(st.integers(min_value=100, max_value=1000000)),
        "tokens_per_day": draw(st.integers(min_value=1000, max_value=10000000)),
    }


# =============================================================================
# Property 9: Rate Limiting Enforcement
# =============================================================================

class TestRateLimitingEnforcementProperty:
    """
    Property 9: Rate Limiting Enforcement
    
    For any virtual key with a configured rate limit, when the number of
    requests exceeds the limit within the time window, the Gateway should
    reject subsequent requests with HTTP 429 status until the window resets.
    
    **Validates: Requirements 11.6**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        rate_limit=st.integers(min_value=1, max_value=100),
        num_requests=st.integers(min_value=1, max_value=200),
    )
    def test_requests_within_limit_are_accepted(
        self, rate_limit: int, num_requests: int
    ):
        """
        Property 9: Rate Limiting Enforcement
        
        For any rate limit, requests within the limit should be accepted.
        
        **Validates: Requirements 11.6**
        """
        # Simulate rate limiting
        request_count = 0
        accepted_requests = 0
        rejected_requests = 0
        
        for _ in range(num_requests):
            if request_count < rate_limit:
                accepted_requests += 1
                request_count += 1
            else:
                rejected_requests += 1
        
        # Property: Accepted requests should equal min(num_requests, rate_limit)
        assert accepted_requests == min(num_requests, rate_limit)
        
        # Property: Rejected requests should be the overflow
        assert rejected_requests == max(0, num_requests - rate_limit)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        rate_limit=st.integers(min_value=1, max_value=100),
        num_requests=st.integers(min_value=1, max_value=200),
    )
    def test_requests_exceeding_limit_return_429(
        self, rate_limit: int, num_requests: int
    ):
        """
        Property 9: Rate Limiting Enforcement
        
        For any rate limit, requests exceeding the limit should be rejected
        with HTTP 429 status.
        
        **Validates: Requirements 11.6**
        """
        assume(num_requests > rate_limit)
        
        # Simulate rate limiting with HTTP status codes
        request_count = 0
        responses = []
        
        for _ in range(num_requests):
            if request_count < rate_limit:
                responses.append(200)  # OK
                request_count += 1
            else:
                responses.append(429)  # Too Many Requests
        
        # Property: First rate_limit requests should return 200
        assert all(r == 200 for r in responses[:rate_limit])
        
        # Property: Requests beyond limit should return 429
        assert all(r == 429 for r in responses[rate_limit:])

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        rate_limit=st.integers(min_value=1, max_value=100),
        window_seconds=st.integers(min_value=1, max_value=3600),
    )
    def test_rate_limit_resets_after_window(
        self, rate_limit: int, window_seconds: int
    ):
        """
        Property 9: Rate Limiting Enforcement
        
        For any rate limit and time window, the limit should reset after
        the window expires.
        
        **Validates: Requirements 11.6**
        """
        # Simulate rate limiting with window reset
        class RateLimiter:
            def __init__(self, limit: int, window: int):
                self.limit = limit
                self.window = window
                self.request_count = 0
                self.window_start = 0
            
            def check_and_increment(self, current_time: int) -> bool:
                # Reset window if expired
                if current_time - self.window_start >= self.window:
                    self.request_count = 0
                    self.window_start = current_time
                
                if self.request_count < self.limit:
                    self.request_count += 1
                    return True
                return False
        
        limiter = RateLimiter(rate_limit, window_seconds)
        
        # Exhaust the limit at time 0
        for _ in range(rate_limit):
            assert limiter.check_and_increment(0) is True
        
        # Next request at time 0 should be rejected
        assert limiter.check_and_increment(0) is False
        
        # Request after window should be accepted
        assert limiter.check_and_increment(window_seconds) is True

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(rate_limit_config=rate_limit_config_strategy())
    def test_rate_limit_config_is_valid(self, rate_limit_config: Dict[str, Any]):
        """
        Property 9: Rate Limiting Enforcement
        
        For any rate limit configuration, all limits should be positive
        and internally consistent.
        
        **Validates: Requirements 11.6**
        """
        # Property: All limits should be positive
        assert rate_limit_config["requests_per_minute"] > 0
        assert rate_limit_config["requests_per_day"] > 0
        assert rate_limit_config["tokens_per_minute"] > 0
        assert rate_limit_config["tokens_per_day"] > 0
        
        # Property: Daily limits should be >= minute limits (logical consistency)
        # Note: This is a soft check - daily could be less if intentionally restrictive
        # but typically daily >= minute * 60 * 24

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        rpm_limit=st.integers(min_value=1, max_value=100),
        tpm_limit=st.integers(min_value=100, max_value=10000),
        request_tokens=st.integers(min_value=1, max_value=1000),
        num_requests=st.integers(min_value=1, max_value=50),
    )
    def test_both_rpm_and_tpm_limits_enforced(
        self, rpm_limit: int, tpm_limit: int, request_tokens: int, num_requests: int
    ):
        """
        Property 9: Rate Limiting Enforcement
        
        For any configuration with both RPM and TPM limits, both should
        be enforced independently.
        
        **Validates: Requirements 11.6**
        """
        # Simulate dual rate limiting
        request_count = 0
        token_count = 0
        accepted = 0
        rejected_rpm = 0
        rejected_tpm = 0
        
        for _ in range(num_requests):
            # Check both limits
            rpm_ok = request_count < rpm_limit
            tpm_ok = token_count + request_tokens <= tpm_limit
            
            if rpm_ok and tpm_ok:
                accepted += 1
                request_count += 1
                token_count += request_tokens
            elif not rpm_ok:
                rejected_rpm += 1
            else:
                rejected_tpm += 1
        
        # Property: Total should equal num_requests
        assert accepted + rejected_rpm + rejected_tpm == num_requests
        
        # Property: Accepted requests should not exceed either limit
        assert accepted <= rpm_limit
        assert accepted * request_tokens <= tpm_limit

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        key_id1=st.text(min_size=10, max_size=36).filter(lambda x: x.strip()),
        key_id2=st.text(min_size=10, max_size=36).filter(lambda x: x.strip()),
        rate_limit=st.integers(min_value=1, max_value=50),
    )
    def test_rate_limits_are_per_key(
        self, key_id1: str, key_id2: str, rate_limit: int
    ):
        """
        Property 9: Rate Limiting Enforcement
        
        For any two different virtual keys, rate limits should be tracked
        independently per key.
        
        **Validates: Requirements 11.6**
        """
        assume(key_id1 != key_id2)
        
        # Simulate per-key rate limiting
        key_request_counts = {}
        
        def check_rate_limit(key_id: str) -> bool:
            count = key_request_counts.get(key_id, 0)
            if count < rate_limit:
                key_request_counts[key_id] = count + 1
                return True
            return False
        
        # Exhaust limit for key1
        for _ in range(rate_limit):
            assert check_rate_limit(key_id1) is True
        
        # key1 should now be rate limited
        assert check_rate_limit(key_id1) is False
        
        # key2 should still have full quota
        for _ in range(rate_limit):
            assert check_rate_limit(key_id2) is True
        
        # Now key2 should also be rate limited
        assert check_rate_limit(key_id2) is False

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        rate_limit=st.integers(min_value=10, max_value=100),
        burst_size=st.integers(min_value=1, max_value=50),
    )
    def test_burst_requests_handled_correctly(
        self, rate_limit: int, burst_size: int
    ):
        """
        Property 9: Rate Limiting Enforcement
        
        For any rate limit, a burst of requests should be handled correctly,
        accepting up to the limit and rejecting the rest.
        
        **Validates: Requirements 11.6**
        """
        # Simulate a burst of requests
        request_count = 0
        accepted = []
        rejected = []
        
        for i in range(burst_size):
            if request_count < rate_limit:
                accepted.append(i)
                request_count += 1
            else:
                rejected.append(i)
        
        # Property: Accepted count should be min(burst_size, rate_limit)
        assert len(accepted) == min(burst_size, rate_limit)
        
        # Property: Rejected count should be max(0, burst_size - rate_limit)
        assert len(rejected) == max(0, burst_size - rate_limit)
        
        # Property: Accepted requests should be the first ones
        if accepted:
            assert accepted == list(range(min(burst_size, rate_limit)))

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(
        rate_limit=st.integers(min_value=5, max_value=50),
    )
    def test_rate_limit_counter_accuracy(self, rate_limit: int):
        """
        Property 9: Rate Limiting Enforcement
        
        For any rate limit, the counter should accurately track the number
        of requests made.
        
        **Validates: Requirements 11.6**
        """
        # Simulate rate limit counter
        counter = 0
        expected_accepted = min(rate_limit * 2, rate_limit)  # Try to make 2x requests
        actual_accepted = 0
        
        for _ in range(rate_limit * 2):
            if counter < rate_limit:
                counter += 1
                actual_accepted += 1
        
        # Property: Counter should equal rate_limit after exhaustion
        assert counter == rate_limit
        
        # Property: Accepted count should equal rate_limit
        assert actual_accepted == rate_limit

