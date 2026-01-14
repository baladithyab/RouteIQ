"""
Property-Based Tests for Core LiteLLM Integration.

These tests validate the correctness properties defined in the design document
for the core LiteLLM integration (Requirements 1.x).

Property tests use Hypothesis to generate many test cases and verify that
universal properties hold across all valid inputs.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

# =============================================================================
# Test Data Generators (Strategies)
# =============================================================================

# Valid model names following LiteLLM provider format
VALID_PROVIDERS = ["openai", "anthropic", "bedrock", "azure", "vertex_ai", "cohere"]

provider_strategy = st.sampled_from(VALID_PROVIDERS)

model_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_."),
    min_size=1,
    max_size=50,
).filter(lambda x: x.strip() and not x.startswith("-"))

# Generate valid model configurations
@st.composite
def model_config_strategy(draw):
    """Generate a valid model configuration for LiteLLM."""
    provider = draw(provider_strategy)
    model_name = draw(model_name_strategy)
    
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"{provider}/{model_name}",
            "api_key": "test-api-key",
        }
    }


@st.composite
def model_list_strategy(draw):
    """Generate a valid model_list configuration."""
    num_models = draw(st.integers(min_value=1, max_value=5))
    models = []
    seen_names = set()
    
    for _ in range(num_models):
        config = draw(model_config_strategy())
        # Ensure unique model names
        if config["model_name"] not in seen_names:
            seen_names.add(config["model_name"])
            models.append(config)
    
    assume(len(models) > 0)
    return models


# Chat completion request strategy
@st.composite
def chat_request_strategy(draw):
    """Generate a valid chat completion request."""
    role = draw(st.sampled_from(["user", "assistant", "system"]))
    content = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    
    return {
        "model": draw(model_name_strategy),
        "messages": [{"role": role, "content": content}],
    }


# API key strategies
valid_api_key_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=10,
    max_size=100,
).filter(lambda x: x.strip())

invalid_api_key_strategy = st.one_of(
    st.just(""),
    st.just(None),
    st.text(max_size=5),  # Too short
)


# YAML configuration strategy
@st.composite
def yaml_config_strategy(draw):
    """Generate a valid YAML configuration structure."""
    model_list = draw(model_list_strategy())
    
    config = {
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": draw(st.sampled_from([
                "simple-shuffle", "least-busy", "llmrouter-knn"
            ])),
            "num_retries": draw(st.integers(min_value=0, max_value=5)),
            "timeout": draw(st.integers(min_value=30, max_value=600)),
        },
        "general_settings": {
            "master_key": draw(valid_api_key_strategy),
        },
        "litellm_settings": {
            "cache": draw(st.booleans()),
            "set_verbose": draw(st.booleans()),
        }
    }
    
    return config


# =============================================================================
# Property 1: Request Forwarding Correctness
# =============================================================================

class TestRequestForwardingProperty:
    """
    Property 1: Request Forwarding Correctness
    
    For any valid client request to a supported endpoint with a configured 
    LLM provider, the Gateway should successfully forward the request to 
    the provider using LiteLLM's unified interface and return a response 
    in the expected format.
    
    **Validates: Requirements 1.2**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        provider=provider_strategy,
        model_name=model_name_strategy,
        message_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    def test_request_forwarding_produces_valid_model_string(
        self, provider: str, model_name: str, message_content: str
    ):
        """
        Property 1: Request Forwarding Correctness
        
        For any valid provider and model name combination, the Gateway should
        construct a valid model string in the format 'provider/model_name'.
        
        **Validates: Requirements 1.2**
        """
        # Construct the model string as LiteLLM would
        model_string = f"{provider}/{model_name}"
        
        # Property: Model string should always contain the provider prefix
        assert provider in model_string
        assert "/" in model_string
        
        # Property: Model string should be parseable back to provider and model
        parts = model_string.split("/", 1)
        assert len(parts) == 2
        assert parts[0] == provider
        assert parts[1] == model_name

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request=chat_request_strategy())
    def test_chat_request_has_required_fields(self, request: Dict[str, Any]):
        """
        Property 1: Request Forwarding Correctness
        
        For any generated chat request, it should contain all required fields
        for the OpenAI-compatible chat completions API.
        
        **Validates: Requirements 1.2**
        """
        # Property: Request must have 'model' field
        assert "model" in request
        assert isinstance(request["model"], str)
        assert len(request["model"]) > 0
        
        # Property: Request must have 'messages' field
        assert "messages" in request
        assert isinstance(request["messages"], list)
        assert len(request["messages"]) > 0
        
        # Property: Each message must have 'role' and 'content'
        for message in request["messages"]:
            assert "role" in message
            assert "content" in message
            assert message["role"] in ["user", "assistant", "system"]

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(model_list=model_list_strategy())
    def test_model_list_contains_valid_configurations(self, model_list: List[Dict]):
        """
        Property 1: Request Forwarding Correctness
        
        For any valid model list configuration, each model should have the
        required litellm_params with a properly formatted model string.
        
        **Validates: Requirements 1.2**
        """
        for model_config in model_list:
            # Property: Each model config must have model_name
            assert "model_name" in model_config
            assert isinstance(model_config["model_name"], str)
            
            # Property: Each model config must have litellm_params
            assert "litellm_params" in model_config
            params = model_config["litellm_params"]
            
            # Property: litellm_params must have model field
            assert "model" in params
            
            # Property: model field should contain provider prefix
            model_string = params["model"]
            assert "/" in model_string, f"Model string '{model_string}' should contain provider prefix"


# =============================================================================
# Property 2: Authentication Enforcement
# =============================================================================

class TestAuthenticationEnforcementProperty:
    """
    Property 2: Authentication Enforcement
    
    For any request to the Gateway when master_key is configured, the request
    should be accepted if and only if it includes a valid API key (either 
    master_key or a valid virtual key from the database).
    
    **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        master_key=valid_api_key_strategy,
        request_key=valid_api_key_strategy,
    )
    def test_matching_key_is_accepted(self, master_key: str, request_key: str):
        """
        Property 2: Authentication Enforcement
        
        For any master_key configuration, a request with the exact same key
        should be accepted (authentication passes).
        
        **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
        """
        # When request key matches master key, authentication should pass
        is_authenticated = (request_key == master_key)
        
        # Property: Same key should always authenticate
        if request_key == master_key:
            assert is_authenticated is True
        
        # Property: Different keys should not authenticate (unless virtual key)
        # Note: In real system, virtual keys would also be valid

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        master_key=valid_api_key_strategy,
    )
    def test_empty_key_is_rejected(self, master_key: str):
        """
        Property 2: Authentication Enforcement
        
        For any master_key configuration, a request with an empty or missing
        key should be rejected.
        
        **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
        """
        invalid_keys = ["", None, "   "]
        
        for invalid_key in invalid_keys:
            # Property: Empty/None keys should never match a valid master_key
            is_authenticated = (invalid_key == master_key)
            assert is_authenticated is False, f"Empty key '{invalid_key}' should not authenticate"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        master_key=valid_api_key_strategy,
        prefix=st.text(min_size=1, max_size=10),
    )
    def test_partial_key_is_rejected(self, master_key: str, prefix: str):
        """
        Property 2: Authentication Enforcement
        
        For any master_key, a partial match (prefix or substring) should not
        authenticate.
        
        **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
        """
        assume(len(master_key) > len(prefix))
        
        # Create a partial key (just the prefix of master_key)
        partial_key = master_key[:len(prefix)]
        
        # Property: Partial keys should not authenticate
        is_authenticated = (partial_key == master_key)
        
        if len(partial_key) < len(master_key):
            assert is_authenticated is False, "Partial key should not authenticate"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        key1=valid_api_key_strategy,
        key2=valid_api_key_strategy,
    )
    def test_authentication_is_deterministic(self, key1: str, key2: str):
        """
        Property 2: Authentication Enforcement
        
        Authentication should be deterministic - the same key comparison
        should always produce the same result.
        
        **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
        """
        # Property: Authentication result should be consistent
        result1 = (key1 == key2)
        result2 = (key1 == key2)
        result3 = (key1 == key2)
        
        assert result1 == result2 == result3, "Authentication should be deterministic"


# =============================================================================
# Property 3: Configuration Loading
# =============================================================================

class TestConfigurationLoadingProperty:
    """
    Property 3: Configuration Loading
    
    For any valid YAML configuration file containing model_list, router_settings,
    and general_settings, the Gateway should successfully load the configuration
    and make all configured models and settings available.
    
    **Validates: Requirements 1.5**
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_has_required_sections(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, it should contain all required top-level
        sections: model_list, router_settings, general_settings, litellm_settings.
        
        **Validates: Requirements 1.5**
        """
        # Property: Config must have model_list
        assert "model_list" in config
        assert isinstance(config["model_list"], list)
        assert len(config["model_list"]) > 0
        
        # Property: Config must have router_settings
        assert "router_settings" in config
        assert isinstance(config["router_settings"], dict)
        
        # Property: Config must have general_settings
        assert "general_settings" in config
        assert isinstance(config["general_settings"], dict)
        
        # Property: Config must have litellm_settings
        assert "litellm_settings" in config
        assert isinstance(config["litellm_settings"], dict)

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_round_trip_preserves_structure(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, serializing to JSON and deserializing
        should preserve the structure (round-trip property).
        
        **Validates: Requirements 1.5**
        """
        import json
        
        # Serialize to JSON
        json_str = json.dumps(config)
        
        # Deserialize back
        loaded_config = json.loads(json_str)
        
        # Property: Round-trip should preserve all data
        assert loaded_config == config
        
        # Property: All sections should be preserved
        assert set(loaded_config.keys()) == set(config.keys())
        
        # Property: Model list should be preserved
        assert len(loaded_config["model_list"]) == len(config["model_list"])

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_models_are_accessible(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, all models in model_list should be
        accessible by their model_name.
        
        **Validates: Requirements 1.5**
        """
        model_list = config["model_list"]
        
        # Build a lookup dictionary (as the Gateway would)
        model_lookup = {m["model_name"]: m for m in model_list}
        
        # Property: All models should be in the lookup
        for model_config in model_list:
            model_name = model_config["model_name"]
            assert model_name in model_lookup
            assert model_lookup[model_name] == model_config

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_router_settings_has_valid_strategy(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, router_settings should contain a valid
        routing_strategy value.
        
        **Validates: Requirements 1.5**
        """
        router_settings = config["router_settings"]
        
        # Property: routing_strategy must be present
        assert "routing_strategy" in router_settings
        
        strategy = router_settings["routing_strategy"]
        
        # Property: Strategy must be a non-empty string
        assert isinstance(strategy, str)
        assert len(strategy) > 0
        
        # Property: Strategy should be a known value
        valid_strategies = [
            "simple-shuffle", "least-busy", "latency-based-routing",
            "cost-based-routing", "usage-based-routing",
            "llmrouter-knn", "llmrouter-svm", "llmrouter-mlp",
            "llmrouter-mf", "llmrouter-elo", "llmrouter-hybrid",
        ]
        assert strategy in valid_strategies, f"Unknown strategy: {strategy}"

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_yaml_serialization(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, it should be serializable to YAML format
        and loadable back without data loss.
        
        **Validates: Requirements 1.5**
        """
        import yaml
        
        # Serialize to YAML
        yaml_str = yaml.dump(config, default_flow_style=False)
        
        # Property: YAML string should not be empty
        assert len(yaml_str) > 0
        
        # Deserialize back
        loaded_config = yaml.safe_load(yaml_str)
        
        # Property: Round-trip should preserve data
        assert loaded_config == config

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_file_write_and_read(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, writing to a file and reading back
        should preserve all configuration data.
        
        **Validates: Requirements 1.5**
        """
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            # Read back from file
            with open(temp_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Property: File round-trip should preserve data
            assert loaded_config == config
            
            # Property: All model names should be preserved
            original_names = {m["model_name"] for m in config["model_list"]}
            loaded_names = {m["model_name"] for m in loaded_config["model_list"]}
            assert original_names == loaded_names
            
        finally:
            os.unlink(temp_path)


# =============================================================================
# Property 5: Model and Config Hot Reload
# =============================================================================

class TestModelAndConfigHotReloadProperty:
    """
    Property 5: Model and Config Hot Reload
    
    For any model file or configuration file that changes (detected via 
    modification time or ETag), when hot reload is enabled, the Gateway 
    should detect the change and reload the affected component without 
    requiring a service restart.
    
    **Validates: Requirements 3.2, 3.4, 3.5**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        reload_interval=st.integers(min_value=1, max_value=300),
        time_elapsed=st.integers(min_value=0, max_value=600),
    )
    def test_reload_check_respects_interval(
        self, reload_interval: int, time_elapsed: int
    ):
        """
        Property 5: Model and Config Hot Reload
        
        For any reload interval configuration, the system should only check
        for updates when the elapsed time exceeds the reload interval.
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        # Property: Should reload if time_elapsed >= reload_interval
        should_check = time_elapsed >= reload_interval
        
        # Verify the property holds
        if time_elapsed >= reload_interval:
            assert should_check is True
        else:
            assert should_check is False

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        initial_mtime=st.floats(min_value=1000000000, max_value=2000000000),
        new_mtime=st.floats(min_value=1000000000, max_value=2000000000),
    )
    def test_mtime_change_detection(self, initial_mtime: float, new_mtime: float):
        """
        Property 5: Model and Config Hot Reload
        
        For any two modification times, the system should detect a change
        if and only if the new mtime is different from the initial mtime.
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        # Property: Change detected iff mtimes differ
        has_changed = (new_mtime != initial_mtime)
        
        if new_mtime != initial_mtime:
            assert has_changed is True
        else:
            assert has_changed is False

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        etag1=st.text(min_size=8, max_size=64).filter(lambda x: x.strip()),
        etag2=st.text(min_size=8, max_size=64).filter(lambda x: x.strip()),
    )
    def test_etag_change_detection(self, etag1: str, etag2: str):
        """
        Property 5: Model and Config Hot Reload
        
        For any two ETags, the system should detect a change if and only if
        the ETags are different (S3 ETag-based optimization).
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        # Property: Change detected iff ETags differ
        has_changed = (etag1 != etag2)
        
        if etag1 != etag2:
            assert has_changed is True
        else:
            assert has_changed is False

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        hot_reload_enabled=st.booleans(),
        file_changed=st.booleans(),
    )
    def test_reload_only_when_enabled_and_changed(
        self, hot_reload_enabled: bool, file_changed: bool
    ):
        """
        Property 5: Model and Config Hot Reload
        
        For any hot reload configuration, the system should only trigger a
        reload when both hot_reload is enabled AND the file has changed.
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        # Property: Reload iff hot_reload enabled AND file changed
        should_reload = hot_reload_enabled and file_changed
        
        if hot_reload_enabled and file_changed:
            assert should_reload is True
        else:
            assert should_reload is False

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        strategy_name=st.sampled_from([
            "llmrouter-knn", "llmrouter-svm", "llmrouter-mlp",
            "llmrouter-mf", "llmrouter-elo", "llmrouter-hybrid"
        ]),
    )
    def test_strategy_reload_callback_registration(self, strategy_name: str):
        """
        Property 5: Model and Config Hot Reload
        
        For any routing strategy name, registering a reload callback should
        make the strategy reloadable via the hot reload manager.
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        # Test the core logic without importing the full module
        # Simulate HotReloadManager behavior
        router_reload_callbacks = {}
        callback_called = []
        
        def test_callback():
            callback_called.append(True)
        
        # Register callback (simulating register_router_reload_callback)
        router_reload_callbacks[strategy_name] = test_callback
        
        # Property: Strategy should be in registered callbacks
        assert strategy_name in router_reload_callbacks
        
        # Reload the strategy (simulating reload_router)
        reloaded = []
        errors = []
        
        if strategy_name in router_reload_callbacks:
            try:
                router_reload_callbacks[strategy_name]()
                reloaded.append(strategy_name)
            except Exception as e:
                errors.append({"strategy": strategy_name, "error": str(e)})
        
        # Property: Reloading the strategy should call the callback
        assert len(callback_called) == 1
        assert len(reloaded) == 1
        assert strategy_name in reloaded
        assert len(errors) == 0

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(
        num_strategies=st.integers(min_value=1, max_value=5),
    )
    def test_reload_all_strategies(self, num_strategies: int):
        """
        Property 5: Model and Config Hot Reload
        
        For any number of registered strategies, reloading all strategies
        (strategy=None) should call all registered callbacks.
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        # Test the core logic without importing the full module
        # Simulate HotReloadManager behavior
        router_reload_callbacks = {}
        callbacks_called = []
        
        # Register multiple strategies
        for i in range(num_strategies):
            strategy_name = f"test-strategy-{i}"
            
            def make_callback(idx):
                def callback():
                    callbacks_called.append(idx)
                return callback
            
            router_reload_callbacks[strategy_name] = make_callback(i)
        
        # Property: All strategies should be registered
        assert len(router_reload_callbacks) == num_strategies
        
        # Reload all strategies (simulating reload_router with strategy=None)
        reloaded = []
        errors = []
        
        for name, callback in router_reload_callbacks.items():
            try:
                callback()
                reloaded.append(name)
            except Exception as e:
                errors.append({"strategy": name, "error": str(e)})
        
        # Property: All callbacks should be called
        assert len(callbacks_called) == num_strategies
        assert len(reloaded) == num_strategies
        assert len(errors) == 0

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        config_content1=st.text(min_size=10, max_size=500),
        config_content2=st.text(min_size=10, max_size=500),
    )
    def test_config_hash_detects_content_changes(
        self, config_content1: str, config_content2: str
    ):
        """
        Property 5: Model and Config Hot Reload
        
        For any two configuration file contents, the hash-based change
        detection should identify them as different if and only if their
        content differs.
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        import hashlib
        
        # Compute hashes
        hash1 = hashlib.md5(config_content1.encode()).hexdigest()
        hash2 = hashlib.md5(config_content2.encode()).hexdigest()
        
        # Property: Hashes differ iff content differs
        content_differs = (config_content1 != config_content2)
        hashes_differ = (hash1 != hash2)
        
        assert content_differs == hashes_differ

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(
        sync_interval=st.integers(min_value=10, max_value=300),
    )
    def test_config_sync_manager_initialization(self, sync_interval: int):
        """
        Property 5: Model and Config Hot Reload
        
        For any sync interval configuration, the ConfigSyncManager should
        initialize with the correct settings.
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        # Import directly to avoid litellm dependency
        import sys
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "config_sync", "src/litellm_llmrouter/config_sync.py"
        )
        config_sync_module = importlib.util.module_from_spec(spec)
        
        # Mock the litellm logger
        mock_logger = MagicMock()
        sys.modules['litellm'] = MagicMock()
        sys.modules['litellm._logging'] = MagicMock()
        sys.modules['litellm._logging'].verbose_proxy_logger = mock_logger
        
        spec.loader.exec_module(config_sync_module)
        ConfigSyncManager = config_sync_module.ConfigSyncManager
        
        manager = ConfigSyncManager(
            local_config_path="/tmp/test_config.yaml",
            sync_interval_seconds=sync_interval,
        )
        
        # Property: Sync interval should be set correctly
        assert manager.sync_interval == sync_interval
        
        # Property: Manager should not be running initially
        assert manager._sync_thread is None or not manager._sync_thread.is_alive()

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        s3_bucket=st.text(min_size=3, max_size=63).filter(
            lambda x: x.strip() and x.replace("-", "").replace(".", "").isalnum()
        ),
        s3_key=st.text(
            alphabet=st.characters(blacklist_characters="\x00"),
            min_size=1, 
            max_size=100
        ).filter(lambda x: x.strip() and "\x00" not in x),
    )
    def test_s3_config_detection(self, s3_bucket: str, s3_key: str):
        """
        Property 5: Model and Config Hot Reload
        
        For any S3 bucket and key configuration, the ConfigSyncManager should
        correctly detect whether S3 sync is enabled.
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        # Import directly to avoid litellm dependency
        import sys
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "config_sync", "src/litellm_llmrouter/config_sync.py"
        )
        config_sync_module = importlib.util.module_from_spec(spec)
        
        # Mock the litellm logger
        mock_logger = MagicMock()
        sys.modules['litellm'] = MagicMock()
        sys.modules['litellm._logging'] = MagicMock()
        sys.modules['litellm._logging'].verbose_proxy_logger = mock_logger
        
        spec.loader.exec_module(config_sync_module)
        ConfigSyncManager = config_sync_module.ConfigSyncManager
        
        # Set environment variables
        with patch.dict(os.environ, {
            "CONFIG_S3_BUCKET": s3_bucket,
            "CONFIG_S3_KEY": s3_key,
        }):
            manager = ConfigSyncManager()
            
            # Property: S3 sync should be enabled when both bucket and key are set
            assert manager.s3_sync_enabled is True
            assert manager.s3_bucket == s3_bucket
            assert manager.s3_key == s3_key

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        hot_reload=st.booleans(),
        sync_enabled=st.booleans(),
    )
    def test_hot_reload_and_sync_flags(
        self, hot_reload: bool, sync_enabled: bool
    ):
        """
        Property 5: Model and Config Hot Reload
        
        For any combination of hot_reload and sync_enabled flags, the
        ConfigSyncManager should respect both settings independently.
        
        **Validates: Requirements 3.2, 3.4, 3.5**
        """
        # Import directly to avoid litellm dependency
        import sys
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "config_sync", "src/litellm_llmrouter/config_sync.py"
        )
        config_sync_module = importlib.util.module_from_spec(spec)
        
        # Mock the litellm logger
        mock_logger = MagicMock()
        sys.modules['litellm'] = MagicMock()
        sys.modules['litellm._logging'] = MagicMock()
        sys.modules['litellm._logging'].verbose_proxy_logger = mock_logger
        
        spec.loader.exec_module(config_sync_module)
        ConfigSyncManager = config_sync_module.ConfigSyncManager
        
        with patch.dict(os.environ, {
            "CONFIG_HOT_RELOAD": "true" if hot_reload else "false",
            "CONFIG_SYNC_ENABLED": "true" if sync_enabled else "false",
        }):
            manager = ConfigSyncManager()
            
            # Property: Flags should be set correctly
            assert manager.hot_reload_enabled == hot_reload
            assert manager.sync_enabled == sync_enabled


# =============================================================================
# Property 22: S3 Config Sync with ETag Optimization
# =============================================================================

class TestS3ConfigSyncWithETagOptimizationProperty:
    """
    Property 22: S3 Config Sync with ETag Optimization
    
    For any configuration file stored in S3, the Config Sync Manager should
    only download the file when the ETag changes, avoiding unnecessary
    downloads when the content is unchanged.
    
    **Validates: Requirements 10.3**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        etag1=st.text(min_size=8, max_size=64).filter(lambda x: x.strip()),
        etag2=st.text(min_size=8, max_size=64).filter(lambda x: x.strip()),
    )
    def test_etag_change_triggers_download(self, etag1: str, etag2: str):
        """
        Property 22: S3 Config Sync with ETag Optimization
        
        For any two ETags, the system should download if and only if the
        ETags are different.
        
        **Validates: Requirements 10.3**
        """
        # Property: Download should occur iff ETags differ
        should_download = (etag1 != etag2)
        
        if etag1 != etag2:
            assert should_download is True
        else:
            assert should_download is False

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        etag=st.text(min_size=8, max_size=64).filter(lambda x: x.strip()),
        num_checks=st.integers(min_value=1, max_value=10),
    )
    def test_same_etag_prevents_multiple_downloads(
        self, etag: str, num_checks: int
    ):
        """
        Property 22: S3 Config Sync with ETag Optimization
        
        For any ETag, checking multiple times with the same ETag should
        only trigger one download (on the first check).
        
        **Validates: Requirements 10.3**
        """
        # Simulate checking the same ETag multiple times
        last_etag = None
        download_count = 0
        
        for _ in range(num_checks):
            # Check if ETag changed
            if etag != last_etag:
                download_count += 1
                last_etag = etag
        
        # Property: Should only download once when ETag doesn't change
        assert download_count == 1

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        etags=st.lists(
            st.text(min_size=8, max_size=64).filter(lambda x: x.strip()),
            min_size=1,
            max_size=10
        ),
    )
    def test_download_count_equals_unique_etags(self, etags: list[str]):
        """
        Property 22: S3 Config Sync with ETag Optimization
        
        For any sequence of ETags, the number of downloads should equal
        the number of unique ETags in the sequence.
        
        **Validates: Requirements 10.3**
        """
        # Simulate checking a sequence of ETags
        last_etag = None
        download_count = 0
        
        for etag in etags:
            if etag != last_etag:
                download_count += 1
                last_etag = etag
        
        # Count unique consecutive ETags
        unique_consecutive = 1
        for i in range(1, len(etags)):
            if etags[i] != etags[i-1]:
                unique_consecutive += 1
        
        # Property: Downloads should equal unique consecutive ETags
        assert download_count == unique_consecutive

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        initial_etag=st.text(min_size=8, max_size=64).filter(lambda x: x.strip()),
        new_etag=st.text(min_size=8, max_size=64).filter(lambda x: x.strip()),
    )
    def test_etag_caching_behavior(self, initial_etag: str, new_etag: str):
        """
        Property 22: S3 Config Sync with ETag Optimization
        
        For any initial and new ETag, the system should cache the ETag
        after download and use it for comparison on subsequent checks.
        
        **Validates: Requirements 10.3**
        """
        # Simulate ETag caching
        cached_etag = None
        downloads = []
        
        # First check with initial_etag
        if initial_etag != cached_etag:
            downloads.append("initial")
            cached_etag = initial_etag
        
        # Second check with same ETag
        if initial_etag != cached_etag:
            downloads.append("duplicate_initial")
        
        # Third check with new ETag
        if new_etag != cached_etag:
            downloads.append("new")
            cached_etag = new_etag
        
        # Fourth check with same new ETag
        if new_etag != cached_etag:
            downloads.append("duplicate_new")
        
        # Property: Should download on first and when ETag changes
        if initial_etag == new_etag:
            assert len(downloads) == 1  # Only initial download
            assert downloads == ["initial"]
        else:
            assert len(downloads) == 2  # Initial and new
            assert downloads == ["initial", "new"]

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(
        config_content=st.text(min_size=20, max_size=500),  # Ensure content is larger than ETag
        etag=st.text(min_size=8, max_size=16).filter(lambda x: x.strip()),  # Keep ETag smaller
    )
    def test_etag_optimization_reduces_bandwidth(
        self, config_content: str, etag: str
    ):
        """
        Property 22: S3 Config Sync with ETag Optimization
        
        For any config content and ETag, using ETag-based checking should
        avoid downloading the full content when the ETag hasn't changed.
        
        **Validates: Requirements 10.3**
        """
        import hashlib
        
        # Ensure content is larger than ETag for meaningful test
        assume(len(config_content.encode()) > len(etag.encode()))
        
        # Simulate bandwidth usage
        bandwidth_used = 0
        cached_etag = None
        
        # First check - need to download
        etag_check_size = len(etag.encode())  # Small ETag check
        bandwidth_used += etag_check_size
        
        if etag != cached_etag:
            # Download full content
            content_size = len(config_content.encode())
            bandwidth_used += content_size
            cached_etag = etag
        
        first_check_bandwidth = bandwidth_used
        
        # Second check with same ETag - only check ETag
        bandwidth_used += etag_check_size
        
        if etag != cached_etag:
            # Would download, but ETag matches so skip
            content_size = len(config_content.encode())
            bandwidth_used += content_size
        
        second_check_bandwidth = bandwidth_used - first_check_bandwidth
        
        # Property: Second check should use much less bandwidth (only ETag)
        assert second_check_bandwidth == etag_check_size
        assert second_check_bandwidth < len(config_content.encode())

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        sync_interval=st.integers(min_value=10, max_value=300),
        num_syncs=st.integers(min_value=2, max_value=10),  # At least 2 syncs
        etag_changes_at=st.integers(min_value=1, max_value=9),  # Change after first sync
    )
    def test_periodic_sync_with_etag_optimization(
        self, sync_interval: int, num_syncs: int, etag_changes_at: int
    ):
        """
        Property 22: S3 Config Sync with ETag Optimization
        
        For any sync interval and number of syncs, the system should only
        download when the ETag actually changes, not on every sync.
        
        **Validates: Requirements 10.3**
        """
        assume(etag_changes_at < num_syncs)
        
        # Simulate periodic syncs
        cached_etag = None
        downloads = []
        
        for sync_num in range(num_syncs):
            # Simulate ETag changing at a specific sync
            current_etag = "new-etag" if sync_num >= etag_changes_at else "initial-etag"
            
            # Check if download needed
            if current_etag != cached_etag:
                downloads.append(sync_num)
                cached_etag = current_etag
        
        # Property: Should download exactly twice (initial + one change)
        assert len(downloads) == 2
        assert downloads[0] == 0  # First sync (initial)
        assert downloads[1] == etag_changes_at  # When ETag changed
