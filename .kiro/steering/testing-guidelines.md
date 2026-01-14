# Testing Guidelines

## Testing Framework

- **pytest** for unit and integration tests
- **Hypothesis** for property-based tests
- **pytest-asyncio** for async tests
- **pytest-docker** for integration tests with containers

## Test File Organization

```
tests/
├── unit/
│   ├── test_strategies.py
│   ├── test_config_sync.py
│   ├── test_hot_reload.py
│   ├── test_a2a_gateway.py
│   └── test_mcp_gateway.py
├── integration/
│   ├── test_litellm_integration.py
│   ├── test_redis_caching.py
│   ├── test_postgres_persistence.py
│   └── test_s3_sync.py
├── property/
│   ├── test_routing_properties.py
│   ├── test_caching_properties.py
│   ├── test_auth_properties.py
│   └── test_observability_properties.py
└── conftest.py
```

## Property-Based Testing

### Configuration
Run minimum 100 iterations per property test:

```python
from hypothesis import given, settings, strategies as st

@settings(max_examples=100)
@given(st.text(min_size=1))
def test_routing_returns_valid_model(query: str):
    ...
```

### Annotate with Design Properties
Always reference the design document property:

```python
def test_authentication_enforcement():
    """
    Property 2: Authentication Enforcement
    
    For any request to the Gateway when master_key is configured,
    the request should be accepted if and only if it includes a valid API key.
    
    Validates: Requirements 1.3, 7.5, 11.1, 11.4
    """
```

### Common Property Patterns

**Round-trip (serialization)**:
```python
@given(st.builds(Config, ...))
def test_config_round_trip(config: Config):
    serialized = config.to_yaml()
    deserialized = Config.from_yaml(serialized)
    assert config == deserialized
```

**Invariant preservation**:
```python
@given(st.lists(st.text()))
def test_cache_size_invariant(keys: list[str]):
    cache = Cache(max_size=10)
    for key in keys:
        cache.set(key, "value")
    assert len(cache) <= 10
```

**Idempotence**:
```python
@given(st.text())
def test_reload_idempotent(strategy_name: str):
    result1 = manager.reload_router(strategy_name)
    result2 = manager.reload_router(strategy_name)
    assert result1 == result2
```

## Unit Testing

### Test One Thing
```python
def test_strategy_registration_adds_to_callbacks():
    manager = HotReloadManager()
    callback = lambda: None
    manager.register_router_reload_callback("test-strategy", callback)
    assert "test-strategy" in manager._router_reload_callbacks
```

### Use Fixtures
```python
@pytest.fixture
def mock_s3_client():
    with mock_s3():
        client = boto3.client("s3")
        client.create_bucket(Bucket="test-bucket")
        yield client

def test_s3_etag_detection(mock_s3_client):
    ...
```

### Test Edge Cases
```python
def test_empty_config_raises_error():
    with pytest.raises(ValueError, match="Config cannot be empty"):
        load_config("")

def test_missing_model_path_uses_default():
    strategy = LLMRouterStrategyFamily("llmrouter-knn")
    assert strategy.model_path is None  # Falls back to env var
```

## Integration Testing

### Use Docker Compose
```python
@pytest.fixture(scope="session")
def docker_compose():
    compose = DockerCompose("docker-compose.test.yml")
    compose.start()
    yield compose
    compose.stop()
```

### Test Real Services
```python
@pytest.mark.integration
async def test_redis_caching(redis_client):
    # Make request
    response = await client.post("/v1/chat/completions", json={...})
    
    # Verify cached
    cached = await redis_client.get(cache_key)
    assert cached is not None
```

## Running Tests

**IMPORTANT**: Always use `uv run` to execute tests to ensure the correct virtual environment and dependencies are used.

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit/

# Property tests only
uv run pytest tests/property/

# Integration tests (requires Docker)
uv run pytest tests/integration/ --docker

# With coverage
uv run pytest --cov=src/litellm_llmrouter --cov-report=html

# Specific property
uv run pytest -k "test_authentication_enforcement"

# Verbose output
uv run pytest tests/property/ -v --tb=short
```

## Test Naming

Use descriptive names that explain what's being tested:

```python
# Good
def test_reload_router_with_invalid_strategy_returns_error():
def test_cache_expires_after_ttl():
def test_rate_limit_returns_429_when_exceeded():

# Bad
def test_reload():
def test_cache():
def test_rate_limit():
```
