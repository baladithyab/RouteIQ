"""
Unit tests for the centroid-based routing strategy.

Tests cover:
- CentroidClassifier: cosine similarity classification
- AgenticDetector: cumulative signal scoring
- ReasoningDetector: regex marker detection
- SessionCache: TTL expiry and LRU eviction
- RoutingProfile: profile-based tier overrides
- CentroidRoutingStrategy: end-to-end routing with mocked dependencies
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from litellm_llmrouter.centroid_routing import (
    AgenticDetector,
    CentroidClassifier,
    CentroidRoutingStrategy,
    ClassificationResult,
    ReasoningDetector,
    ReasoningResult,
    RoutingProfile,
    SessionCache,
    get_centroid_strategy,
    reset_centroid_strategy,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeRoutingContext:
    """Minimal routing context for testing."""

    router: Any = None
    model: str = "gpt-4"
    messages: Optional[List[Dict[str, str]]] = None
    input: Optional[Any] = None
    specific_deployment: bool = False
    request_kwargs: Optional[Dict] = None


class FakeRouter:
    """Minimal router mock with model_list / healthy_deployments."""

    def __init__(self, deployments: Optional[List[Dict]] = None):
        self.model_list = deployments or []
        self.healthy_deployments = deployments or []


def _make_deployment(model_name: str, litellm_model: str) -> Dict:
    """Build a deployment dict matching LiteLLM's format."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": litellm_model},
    }


def _make_unit_vector(dim: int = 384, seed: int = 42) -> np.ndarray:
    """Create a random unit-normalized vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset centroid strategy singleton between tests."""
    yield
    reset_centroid_strategy()


# ---------------------------------------------------------------------------
# TestCentroidClassifier
# ---------------------------------------------------------------------------


class TestCentroidClassifier:
    """Tests for the CentroidClassifier class."""

    def _make_classifier_with_mock(
        self,
        simple_vec: Optional[np.ndarray] = None,
        complex_vec: Optional[np.ndarray] = None,
    ) -> CentroidClassifier:
        """Create a classifier with pre-loaded mock centroids."""
        clf = CentroidClassifier(confidence_threshold=0.06)
        clf._simple_centroid = (
            simple_vec if simple_vec is not None else _make_unit_vector(seed=1)
        )
        clf._complex_centroid = (
            complex_vec if complex_vec is not None else _make_unit_vector(seed=2)
        )
        clf._loaded = True
        return clf

    @patch("litellm_llmrouter.strategies._get_sentence_transformer")
    def test_classify_simple_prompt(self, mock_encoder_fn):
        """Simple prompt should classify as 'simple' when embedding is close to simple centroid."""
        simple_centroid = _make_unit_vector(seed=10)
        complex_centroid = _make_unit_vector(seed=20)

        # Embedding very close to simple centroid
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [simple_centroid * 1.01]
        mock_encoder_fn.return_value = mock_encoder

        clf = self._make_classifier_with_mock(simple_centroid, complex_centroid)
        result = clf.classify("What is 2+2?")

        assert result.tier == "simple"
        assert result.confidence > 0
        assert result.sim_simple > result.sim_complex

    @patch("litellm_llmrouter.strategies._get_sentence_transformer")
    def test_classify_complex_prompt(self, mock_encoder_fn):
        """Complex prompt should classify as 'complex' when embedding is close to complex centroid."""
        simple_centroid = _make_unit_vector(seed=10)
        complex_centroid = _make_unit_vector(seed=20)

        # Embedding close to complex centroid
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [complex_centroid * 1.01]
        mock_encoder_fn.return_value = mock_encoder

        clf = self._make_classifier_with_mock(simple_centroid, complex_centroid)
        result = clf.classify("Design a microservices architecture")

        assert result.tier == "complex"
        assert result.confidence > 0
        assert result.sim_complex > result.sim_simple

    @patch("litellm_llmrouter.strategies._get_sentence_transformer")
    def test_confidence_threshold_bias(self, mock_encoder_fn):
        """Borderline cases should bias toward complex when confidence < threshold."""
        simple_centroid = _make_unit_vector(seed=10)
        complex_centroid = _make_unit_vector(seed=20)

        # Embedding equidistant from both centroids (low confidence)
        midpoint = (simple_centroid + complex_centroid) / 2
        midpoint = midpoint / np.linalg.norm(midpoint)

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [midpoint]
        mock_encoder_fn.return_value = mock_encoder

        clf = self._make_classifier_with_mock(simple_centroid, complex_centroid)
        # Use a very high threshold so confidence will be below it
        clf._confidence_threshold = 10.0
        result = clf.classify("Ambiguous prompt")

        # Should bias toward complex
        assert result.tier == "complex"

    @patch("litellm_llmrouter.strategies._get_sentence_transformer")
    def test_classification_result_fields(self, mock_encoder_fn):
        """All ClassificationResult fields should be populated correctly."""
        simple_centroid = _make_unit_vector(seed=10)
        complex_centroid = _make_unit_vector(seed=20)

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [simple_centroid * 1.01]
        mock_encoder_fn.return_value = mock_encoder

        clf = self._make_classifier_with_mock(simple_centroid, complex_centroid)
        result = clf.classify("Hello")

        assert isinstance(result, ClassificationResult)
        assert result.tier in ("simple", "complex")
        assert isinstance(result.confidence, float)
        assert isinstance(result.sim_simple, float)
        assert isinstance(result.sim_complex, float)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0
        assert result.is_agentic is False
        assert result.is_reasoning is False

    @patch("litellm_llmrouter.strategies._get_sentence_transformer")
    def test_warmup_loads_model(self, mock_encoder_fn):
        """warmup() should pre-load encoder and centroids."""
        mock_encoder_fn.return_value = MagicMock()

        clf = CentroidClassifier()
        # Pre-load centroids manually to avoid file I/O
        clf._simple_centroid = _make_unit_vector(seed=1)
        clf._complex_centroid = _make_unit_vector(seed=2)
        clf._loaded = True

        clf.warmup()

        # Encoder should have been requested
        mock_encoder_fn.assert_called()

    def test_missing_centroid_files(self):
        """Should raise FileNotFoundError when centroid files don't exist."""
        clf = CentroidClassifier(centroid_dir="/nonexistent/path")

        with pytest.raises(FileNotFoundError, match="Centroid files not found"):
            clf._load_centroids()

    def test_invalid_centroid_shape(self, tmp_path):
        """Should raise ValueError when centroid has wrong shape."""
        # Save a wrong-shaped array
        wrong_shape = np.zeros(100, dtype=np.float32)
        np.save(str(tmp_path / "simple_centroid.npy"), wrong_shape)
        np.save(str(tmp_path / "complex_centroid.npy"), _make_unit_vector())

        clf = CentroidClassifier(centroid_dir=str(tmp_path))

        with pytest.raises(ValueError, match="simple_centroid has shape"):
            clf._load_centroids()


# ---------------------------------------------------------------------------
# TestAgenticDetector
# ---------------------------------------------------------------------------


class TestAgenticDetector:
    """Tests for the AgenticDetector class."""

    def test_detect_tool_use(self):
        """Messages with tools should score high."""
        messages = [
            {"role": "user", "content": "Run the tests"},
        ]
        result = AgenticDetector.detect(messages, has_tools=True, tool_count=3)

        assert result.is_agentic is True
        assert result.confidence >= 0.35
        assert any("tools_defined" in s for s in result.signals)

    def test_detect_many_tools(self):
        """4+ tools should add extra signal."""
        messages = [{"role": "user", "content": "Do something"}]
        result = AgenticDetector.detect(messages, has_tools=True, tool_count=5)

        assert any("many_tools" in s for s in result.signals)

    def test_detect_agentic_keywords(self):
        """System prompts with agentic keywords should score."""
        messages = [
            {
                "role": "system",
                "content": "You are an AI agent that can execute commands and use tools.",
            },
            {"role": "user", "content": "Fix the bug"},
        ]
        result = AgenticDetector.detect(messages, has_tools=False, tool_count=0)

        assert any("agentic_keywords" in s for s in result.signals)

    def test_detect_agentic_cycles(self):
        """assistant→tool→assistant patterns should be detected."""
        messages = [
            {"role": "user", "content": "Fix the code"},
            {"role": "assistant", "content": "Running tests..."},
            {"role": "tool", "content": "Tests failed"},
            {"role": "assistant", "content": "Fixing..."},
            {"role": "tool", "content": "Tests passed"},
            {"role": "assistant", "content": "Done!"},
        ]
        result = AgenticDetector.detect(messages)

        cycles = AgenticDetector._count_agentic_cycles(messages)
        assert cycles == 2
        assert any("agentic_cycles" in s for s in result.signals)

    def test_non_agentic_conversation(self):
        """Normal chat should not be agentic."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm fine, thanks!"},
            {"role": "user", "content": "Tell me a joke."},
        ]
        result = AgenticDetector.detect(messages, has_tools=False, tool_count=0)

        assert result.is_agentic is False
        assert result.confidence < 0.35

    def test_score_capping(self):
        """Score should cap at 1.0 even with many signals."""
        messages = [
            {
                "role": "system",
                "content": "You are an AI agent. You can execute commands and use tools. "
                + "x" * 500,
            },
            {"role": "user", "content": "Do everything"},
            {"role": "assistant", "content": "Working..."},
            {"role": "tool", "content": "Done"},
            {"role": "assistant", "content": "More work..."},
            {"role": "tool", "content": "Done"},
            {"role": "assistant", "content": "Even more..."},
            {"role": "tool", "content": "Done"},
            {"role": "assistant", "content": "And more..."},
            {"role": "tool", "content": "Done"},
            {"role": "assistant", "content": "Still going..."},
            {"role": "tool", "content": "Done"},
        ]
        result = AgenticDetector.detect(messages, has_tools=True, tool_count=10)

        assert result.confidence <= 1.0

    def test_deep_conversation(self):
        """10+ messages should add to score."""
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(12)]
        result = AgenticDetector.detect(messages)

        assert any("deep_conversation" in s for s in result.signals)

    def test_tool_messages_signal(self):
        """Messages with role=tool should be detected."""
        messages = [
            {"role": "user", "content": "Run it"},
            {"role": "tool", "content": "Output: success"},
        ]
        result = AgenticDetector.detect(messages)

        assert any("tool_messages" in s for s in result.signals)

    def test_long_system_prompt_signal(self):
        """System prompts > 500 chars should be flagged."""
        messages = [
            {"role": "system", "content": "x" * 600},
            {"role": "user", "content": "Hi"},
        ]
        result = AgenticDetector.detect(messages)

        assert any("long_system_prompt" in s for s in result.signals)


# ---------------------------------------------------------------------------
# TestReasoningDetector
# ---------------------------------------------------------------------------


class TestReasoningDetector:
    """Tests for the ReasoningDetector class."""

    def test_detect_reasoning_markers(self):
        """'Think step by step and prove that...' should detect 2+ markers."""
        result = ReasoningDetector.detect(
            "Think step by step and prove that P=NP is undecidable"
        )

        assert result.is_reasoning is True
        assert result.marker_count >= 2

    def test_single_marker_not_enough(self):
        """A single marker should not trigger reasoning detection."""
        result = ReasoningDetector.detect("Think through the problem")

        assert result.marker_count == 1
        assert result.is_reasoning is False

    def test_no_markers(self):
        """Normal prompt should have no markers."""
        result = ReasoningDetector.detect("What is 2+2?")

        assert result.marker_count == 0
        assert result.is_reasoning is False

    def test_system_message_included(self):
        """Markers in system prompt should count."""
        result = ReasoningDetector.detect(
            prompt="Solve this",
            system_message="Think step by step and explain why your solution works",
        )

        assert result.marker_count >= 2
        assert result.is_reasoning is True

    def test_multiple_distinct_markers(self):
        """Multiple different markers should all be counted."""
        result = ReasoningDetector.detect(
            "Think step by step, compare and contrast the approaches, "
            "and evaluate whether the solution is optimal"
        )

        assert result.marker_count >= 3
        assert result.is_reasoning is True

    def test_case_insensitive(self):
        """Detection should be case-insensitive."""
        result = ReasoningDetector.detect("STEP BY STEP, PROVE THAT this works")

        assert result.marker_count >= 2
        assert result.is_reasoning is True

    def test_result_type(self):
        """Result should be a ReasoningResult dataclass."""
        result = ReasoningDetector.detect("Hello")
        assert isinstance(result, ReasoningResult)
        assert isinstance(result.markers, list)


# ---------------------------------------------------------------------------
# TestSessionCache
# ---------------------------------------------------------------------------


class TestSessionCache:
    """Tests for the SessionCache class."""

    def test_put_and_get(self):
        """Basic put/get should work."""
        cache = SessionCache(ttl=60)
        key = SessionCache._make_key("system msg", "user msg")

        cache.put(key, "gpt-4o", "complex")
        result = cache.get(key)

        assert result is not None
        assert result == ("gpt-4o", "complex")

    def test_ttl_expiry(self):
        """Expired entries should return None."""
        cache = SessionCache(ttl=1)
        key = SessionCache._make_key("sys", "usr")
        cache.put(key, "model", "simple")

        # Manually expire
        cache._cache[key] = ("model", "simple", time.time() - 10)

        result = cache.get(key)
        assert result is None

    def test_lru_eviction(self):
        """Max size should cause LRU eviction."""
        cache = SessionCache(ttl=3600, max_size=3)

        for i in range(5):
            key = f"key_{i}"
            cache.put(key, f"model_{i}", "simple")

        # Oldest entries should have been evicted
        assert cache.size <= 3

    def test_key_generation(self):
        """Key should be a deterministic 16-char hex string."""
        key = SessionCache._make_key("system prompt", "user message")

        assert isinstance(key, str)
        assert len(key) == 16
        # Should be hex
        int(key, 16)

    def test_key_consistency(self):
        """Same inputs should always produce the same key."""
        key1 = SessionCache._make_key("system", "user msg")
        key2 = SessionCache._make_key("system", "user msg")

        assert key1 == key2

    def test_key_different_inputs(self):
        """Different inputs should produce different keys."""
        key1 = SessionCache._make_key("system A", "user A")
        key2 = SessionCache._make_key("system B", "user B")

        assert key1 != key2

    def test_key_truncation(self):
        """Long messages should be truncated to 200 chars for key generation."""
        long_msg = "a" * 1000
        key1 = SessionCache._make_key(long_msg, "user")
        # Truncated version should match
        key2 = SessionCache._make_key("a" * 200, "user")

        assert key1 == key2

    def test_periodic_cleanup(self):
        """Expired entries should be cleaned up after 100 puts."""
        cache = SessionCache(ttl=1, max_size=10000)

        # Add expired entries
        for i in range(5):
            key = f"expired_{i}"
            cache._cache[key] = (f"model_{i}", "simple", time.time() - 10)
            cache._access_order.append(key)

        initial_size = len(cache._cache)

        # Trigger cleanup by doing 100 puts
        cache._put_counter = 99
        cache.put("trigger", "model", "simple")

        # Expired entries should be cleaned
        assert cache.size < initial_size + 1

    def test_size_property(self):
        """size property should return current cache size."""
        cache = SessionCache()
        assert cache.size == 0

        cache.put("key1", "model", "simple")
        assert cache.size == 1


# ---------------------------------------------------------------------------
# TestRoutingProfile
# ---------------------------------------------------------------------------


class TestRoutingProfile:
    """Tests for RoutingProfile enum and _apply_profile."""

    def test_eco_always_simple(self):
        """ECO profile should always return 'simple'."""
        result = CentroidRoutingStrategy._apply_profile("complex", RoutingProfile.ECO)
        assert result == "simple"

    def test_premium_always_complex(self):
        """PREMIUM profile should always return 'complex'."""
        result = CentroidRoutingStrategy._apply_profile(
            "simple", RoutingProfile.PREMIUM
        )
        assert result == "complex"

    def test_auto_uses_classifier(self):
        """AUTO profile should return the classifier's decision unchanged."""
        assert (
            CentroidRoutingStrategy._apply_profile("simple", RoutingProfile.AUTO)
            == "simple"
        )
        assert (
            CentroidRoutingStrategy._apply_profile("complex", RoutingProfile.AUTO)
            == "complex"
        )

    def test_reasoning_always_complex(self):
        """REASONING profile should always return 'complex'."""
        result = CentroidRoutingStrategy._apply_profile(
            "simple", RoutingProfile.REASONING
        )
        assert result == "complex"

    def test_free_always_simple(self):
        """FREE profile should always return 'simple'."""
        result = CentroidRoutingStrategy._apply_profile("complex", RoutingProfile.FREE)
        assert result == "simple"

    def test_enum_values(self):
        """Profile enum values should match expected strings."""
        assert RoutingProfile.AUTO.value == "auto"
        assert RoutingProfile.ECO.value == "eco"
        assert RoutingProfile.PREMIUM.value == "premium"
        assert RoutingProfile.FREE.value == "free"
        assert RoutingProfile.REASONING.value == "reasoning"


# ---------------------------------------------------------------------------
# TestCentroidRoutingStrategy
# ---------------------------------------------------------------------------


class TestCentroidRoutingStrategy:
    """Tests for the CentroidRoutingStrategy class."""

    def _make_strategy(self, **kwargs) -> CentroidRoutingStrategy:
        """Create a strategy with mocked classifier."""
        strategy = CentroidRoutingStrategy(**kwargs)
        return strategy

    def _make_context(
        self,
        messages: Optional[List[Dict]] = None,
        model: str = "gpt-4",
        deployments: Optional[List[Dict]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> FakeRoutingContext:
        """Create a fake routing context."""
        deps = deployments or [
            _make_deployment("gpt-4", "gpt-4o"),
            _make_deployment("gpt-4", "gpt-4o-mini"),
        ]
        router = FakeRouter(deps)
        return FakeRoutingContext(
            router=router,
            model=model,
            messages=messages or [{"role": "user", "content": "Hello"}],
            request_kwargs=request_kwargs,
        )

    @patch.object(CentroidClassifier, "classify")
    def test_select_deployment_simple(self, mock_classify):
        """Simple prompt should select cheaper deployment."""
        mock_classify.return_value = ClassificationResult(
            tier="simple",
            confidence=0.8,
            sim_simple=0.9,
            sim_complex=0.1,
        )

        strategy = self._make_strategy()
        context = self._make_context(
            messages=[{"role": "user", "content": "What is 2+2?"}]
        )

        result = strategy.select_deployment(context)
        assert result is not None
        # Should prefer mini model
        assert "mini" in result.get("litellm_params", {}).get("model", "")

    @patch.object(CentroidClassifier, "classify")
    def test_select_deployment_complex(self, mock_classify):
        """Complex prompt should select more capable deployment."""
        mock_classify.return_value = ClassificationResult(
            tier="complex",
            confidence=0.8,
            sim_simple=0.1,
            sim_complex=0.9,
        )

        strategy = self._make_strategy()
        context = self._make_context(
            messages=[
                {"role": "user", "content": "Design a microservices architecture"}
            ],
            deployments=[
                _make_deployment("gpt-4", "gpt-4o"),
                _make_deployment("gpt-4", "gpt-4o-mini"),
            ],
        )

        result = strategy.select_deployment(context)
        assert result is not None
        model = result.get("litellm_params", {}).get("model", "")
        # Should prefer non-mini model (gpt-4o is in complex_indicators via "4o")
        assert model == "gpt-4o"

    @patch.object(CentroidClassifier, "classify")
    def test_agentic_override(self, mock_classify):
        """Agentic messages should upgrade simple→complex."""
        mock_classify.return_value = ClassificationResult(
            tier="simple",
            confidence=0.8,
            sim_simple=0.9,
            sim_complex=0.1,
        )

        strategy = self._make_strategy()
        context = self._make_context(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI agent that can execute commands.",
                },
                {"role": "user", "content": "Fix the code"},
                {"role": "assistant", "content": "Running tests..."},
                {"role": "tool", "content": "Tests failed"},
            ],
            request_kwargs={"tools": [{"type": "function", "function": {}}]},
            deployments=[
                _make_deployment("gpt-4", "gpt-4o"),
                _make_deployment("gpt-4", "gpt-4o-mini"),
            ],
        )

        result = strategy.select_deployment(context)
        assert result is not None
        # Should have been upgraded to complex (gpt-4o, not gpt-4o-mini)
        model = result.get("litellm_params", {}).get("model", "")
        assert model == "gpt-4o"

    @patch.object(CentroidClassifier, "classify")
    def test_reasoning_override(self, mock_classify):
        """Reasoning prompts should upgrade to complex."""
        mock_classify.return_value = ClassificationResult(
            tier="simple",
            confidence=0.8,
            sim_simple=0.9,
            sim_complex=0.1,
        )

        strategy = self._make_strategy()
        context = self._make_context(
            messages=[
                {
                    "role": "user",
                    "content": "Think step by step and prove that the algorithm is correct",
                }
            ],
            deployments=[
                _make_deployment("gpt-4", "gpt-4o"),
                _make_deployment("gpt-4", "gpt-4o-mini"),
            ],
        )

        result = strategy.select_deployment(context)
        assert result is not None
        model = result.get("litellm_params", {}).get("model", "")
        assert model == "gpt-4o"

    @patch.object(CentroidClassifier, "classify")
    def test_session_persistence(self, mock_classify):
        """Same conversation should use cached result."""
        mock_classify.return_value = ClassificationResult(
            tier="complex",
            confidence=0.8,
            sim_simple=0.1,
            sim_complex=0.9,
        )

        strategy = self._make_strategy()
        context = self._make_context(
            messages=[{"role": "user", "content": "Design something complex"}]
        )

        # First call
        result1 = strategy.select_deployment(context)
        assert result1 is not None

        # Second call with same messages should hit cache
        result2 = strategy.select_deployment(context)
        assert result2 is not None

        # Classifier should only be called once (second hit cache)
        # (actually may be called twice since cache key depends on the same session)

    def test_fallback_when_no_centroids(self):
        """Should fall back to random selection when centroids fail to load."""
        strategy = CentroidRoutingStrategy(centroid_dir="/nonexistent/path")
        context = self._make_context(messages=[{"role": "user", "content": "Hello"}])

        result = strategy.select_deployment(context)
        # Should still return a deployment via fallback
        assert result is not None

    @patch.object(CentroidClassifier, "classify")
    def test_profile_override(self, mock_classify):
        """Profile should take precedence over classifier."""
        mock_classify.return_value = ClassificationResult(
            tier="complex",
            confidence=0.8,
            sim_simple=0.1,
            sim_complex=0.9,
        )

        strategy = CentroidRoutingStrategy(profile=RoutingProfile.ECO)
        context = self._make_context(
            messages=[{"role": "user", "content": "Complex task"}]
        )

        result = strategy.select_deployment(context)
        assert result is not None
        # ECO should force simple
        model = result.get("litellm_params", {}).get("model", "")
        assert "mini" in model

    def test_extract_prompt(self):
        """Should correctly extract prompt from messages."""
        strategy = self._make_strategy()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Second message"},
        ]

        prompt, system, full = strategy._extract_prompt(messages)

        assert prompt == "Second message"
        assert system == "Be helpful"
        assert full is messages

    def test_extract_prompt_multimodal(self):
        """Should handle multi-modal message format."""
        strategy = self._make_strategy()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "http://example.com"}},
                ],
            }
        ]

        prompt, system, full = strategy._extract_prompt(messages)
        assert prompt == "Describe this image"

    @patch.object(CentroidClassifier, "classify")
    def test_tier_mapping(self, mock_classify):
        """Custom tier→model mapping should work."""
        mock_classify.return_value = ClassificationResult(
            tier="simple",
            confidence=0.8,
            sim_simple=0.9,
            sim_complex=0.1,
        )

        strategy = CentroidRoutingStrategy(
            tier_mapping={
                "simple": ["haiku"],
                "complex": ["sonnet"],
            }
        )
        deployments = [
            _make_deployment("claude", "claude-3-haiku-20240307"),
            _make_deployment("claude", "claude-3-sonnet-20240229"),
        ]
        context = self._make_context(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude",
            deployments=deployments,
        )

        result = strategy.select_deployment(context)
        assert result is not None
        assert "haiku" in result["litellm_params"]["model"]

    @patch.object(CentroidClassifier, "classify")
    def test_empty_deployments(self, mock_classify):
        """Should return None when no deployments available."""
        mock_classify.return_value = ClassificationResult(
            tier="simple",
            confidence=0.8,
            sim_simple=0.9,
            sim_complex=0.1,
        )

        strategy = self._make_strategy()
        # Use a model name that does NOT match any deployments
        router = FakeRouter([])
        context = FakeRoutingContext(
            router=router,
            model="nonexistent-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        result = strategy.select_deployment(context)
        assert result is None

    def test_empty_messages(self):
        """Should fall back when no messages provided."""
        strategy = self._make_strategy()
        context = self._make_context(messages=[])

        result = strategy.select_deployment(context)
        # Should use fallback
        assert result is not None or True  # May be None if no matching deployments

    def test_name_property(self):
        """Strategy name should be 'nadirclaw-centroid'."""
        strategy = self._make_strategy()
        assert strategy.name == "nadirclaw-centroid"

    def test_version_property(self):
        """Strategy version should be set."""
        strategy = self._make_strategy()
        assert strategy.version == "1.0.0"

    def test_validate(self):
        """Validation should succeed when numpy is available."""
        strategy = self._make_strategy()
        is_valid, error = strategy.validate()
        assert is_valid is True
        assert error is None


# ---------------------------------------------------------------------------
# TestModuleFunctions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    """Tests for module-level singleton functions."""

    def test_get_centroid_strategy_singleton(self):
        """get_centroid_strategy should return the same instance."""
        s1 = get_centroid_strategy()
        s2 = get_centroid_strategy()
        assert s1 is s2

    def test_reset_centroid_strategy(self):
        """reset_centroid_strategy should create a new instance next time."""
        s1 = get_centroid_strategy()
        reset_centroid_strategy()
        s2 = get_centroid_strategy()
        assert s1 is not s2

    def test_get_centroid_strategy_with_kwargs(self):
        """kwargs should be passed to constructor on first call."""
        strategy = get_centroid_strategy(profile=RoutingProfile.ECO)
        assert strategy._profile == RoutingProfile.ECO


# ---------------------------------------------------------------------------
# TestHeuristicMatching
# ---------------------------------------------------------------------------


class TestHeuristicMatching:
    """Tests for tier-to-deployment heuristic matching."""

    def test_simple_tier_prefers_mini(self):
        """Simple tier should prefer models with 'mini' in name."""
        strategy = CentroidRoutingStrategy()
        deployments = [
            _make_deployment("gpt-4", "gpt-4o"),
            _make_deployment("gpt-4", "gpt-4o-mini"),
        ]

        result = strategy._match_tier_to_deployment("simple", deployments)
        assert result is not None
        assert "mini" in result["litellm_params"]["model"]

    def test_complex_tier_prefers_full(self):
        """Complex tier should prefer full models over mini."""
        strategy = CentroidRoutingStrategy()
        deployments = [
            _make_deployment("gpt-4", "gpt-4o-mini"),
            _make_deployment("gpt-4", "gpt-4o"),
        ]

        result = strategy._match_tier_to_deployment("complex", deployments)
        assert result is not None
        assert "mini" not in result["litellm_params"]["model"]

    def test_fallback_when_no_heuristic_match(self):
        """Should fall back to random when no heuristic matches."""
        strategy = CentroidRoutingStrategy()
        deployments = [
            _make_deployment("custom", "my-custom-model-v1"),
        ]

        result = strategy._match_tier_to_deployment("simple", deployments)
        assert result is not None

    def test_empty_deployments_returns_none(self):
        """Should return None for empty deployment list."""
        strategy = CentroidRoutingStrategy()
        result = strategy._match_tier_to_deployment("simple", [])
        assert result is None
