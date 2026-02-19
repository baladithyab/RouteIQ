"""
Centroid-Based Routing Strategy
================================

Zero-config intelligent routing strategy using pre-computed centroid vectors
for binary prompt complexity classification. Ported from NadirClaw's architecture.

Features:
- **CentroidClassifier**: ~2ms prompt classification using cosine similarity
  against pre-computed simple/complex centroid vectors (384-dim embeddings).
- **AgenticDetector**: Cumulative scoring for agentic patterns (tool use,
  multi-step execution, agentic keywords).
- **ReasoningDetector**: Regex-based reasoning marker detection (step-by-step,
  chain-of-thought, proofs, etc.).
- **SessionCache**: In-memory routing affinity cache with TTL and LRU eviction.
- **CentroidRoutingStrategy**: ``RoutingStrategy`` implementation that combines
  classification, agentic/reasoning detection, routing profiles, and session
  persistence for zero-config intelligent model routing.

Usage::

    from litellm_llmrouter.centroid_routing import (
        get_centroid_strategy,
        warmup_centroid_classifier,
    )

    # Pre-warm at startup
    warmup_centroid_classifier()

    # Register in the strategy registry
    from litellm_llmrouter.strategy_registry import get_routing_registry
    registry = get_routing_registry()
    registry.register("llmrouter-nadirclaw-centroid", get_centroid_strategy())

Attribution:
    Ported from NadirClaw's BinaryComplexityClassifier + routing intelligence.
    The centroid approach provides zero-config routing without requiring
    ML model training, unlike LLMRouter's KNN/SVM/MLP strategies.
"""

import hashlib
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
# Constants
# ---------------------------------------------------------------------------

# Default centroid directory relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CENTROID_DIR = str(_PROJECT_ROOT / "models" / "centroids")
_FALLBACK_CENTROID_DIR = str(_PROJECT_ROOT / "reference" / "NadirClaw" / "nadirclaw")

# Embedding dimension for all-MiniLM-L6-v2
_EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Result of centroid-based prompt classification."""

    tier: str
    """Classification tier: ``"simple"`` or ``"complex"``."""

    confidence: float
    """Confidence score: ``abs(sim_complex - sim_simple)``."""

    sim_simple: float
    """Cosine similarity to the simple centroid."""

    sim_complex: float
    """Cosine similarity to the complex centroid."""

    is_agentic: bool = False
    """Whether agentic patterns were detected."""

    is_reasoning: bool = False
    """Whether reasoning patterns were detected."""

    latency_ms: float = 0.0
    """Classification latency in milliseconds."""


@dataclass
class AgenticResult:
    """Result of agentic pattern detection."""

    is_agentic: bool
    """Whether the request is classified as agentic."""

    confidence: float
    """Agentic confidence score (0.0–1.0)."""

    signals: List[str] = field(default_factory=list)
    """List of detected agentic signals."""


@dataclass
class ReasoningResult:
    """Result of reasoning pattern detection."""

    is_reasoning: bool
    """Whether the request requires reasoning capabilities."""

    marker_count: int
    """Number of distinct reasoning markers found."""

    markers: List[str] = field(default_factory=list)
    """List of detected reasoning markers."""


# ---------------------------------------------------------------------------
# RoutingProfile enum
# ---------------------------------------------------------------------------


class RoutingProfile(str, Enum):
    """Routing profiles that control tier selection behavior."""

    AUTO = "auto"
    """Use classifier result (default)."""

    ECO = "eco"
    """Always route to simple (cheapest) tier."""

    PREMIUM = "premium"
    """Always route to complex (most capable) tier."""

    FREE = "free"
    """Always route to simple tier (cheapest option)."""

    REASONING = "reasoning"
    """Always route to complex tier (most capable for reasoning)."""


# ---------------------------------------------------------------------------
# CentroidClassifier
# ---------------------------------------------------------------------------


class CentroidClassifier:
    """Binary complexity classifier using centroid-based cosine similarity.

    Ported from NadirClaw's ``BinaryComplexityClassifier`` pattern.
    Uses pre-computed centroid vectors + ``all-MiniLM-L6-v2`` embeddings.
    Classification latency: ~2ms after model warmup.

    The classifier loads two 384-dimensional unit-normalized centroid
    vectors (``simple_centroid.npy`` and ``complex_centroid.npy``) and
    computes cosine similarity between the prompt embedding and each
    centroid. Borderline cases (confidence below threshold) are biased
    toward the complex tier — it is cheaper to over-serve a simple prompt
    than to under-serve a complex one.
    """

    CONFIDENCE_THRESHOLD = 0.06
    """Default confidence threshold from NadirClaw settings."""

    def __init__(
        self,
        centroid_dir: Optional[str] = None,
        confidence_threshold: float = 0.06,
    ):
        """Initialize the centroid classifier.

        Args:
            centroid_dir: Directory containing centroid ``.npy`` files.
                Defaults to ``models/centroids/`` with fallback to
                ``reference/NadirClaw/nadirclaw/``.
            confidence_threshold: Minimum confidence to trust the
                classifier's decision. Below this, bias toward complex.
        """
        self._centroid_dir = centroid_dir
        self._confidence_threshold = confidence_threshold
        self._simple_centroid: Optional[Any] = None
        self._complex_centroid: Optional[Any] = None
        self._loaded = False
        self._load_lock = threading.Lock()

    def _resolve_centroid_dir(self) -> str:
        """Resolve the centroid directory, falling back if needed."""
        if self._centroid_dir:
            return self._centroid_dir

        # Try default directory first
        if os.path.exists(os.path.join(_DEFAULT_CENTROID_DIR, "simple_centroid.npy")):
            return _DEFAULT_CENTROID_DIR

        # Fallback to NadirClaw reference
        if os.path.exists(os.path.join(_FALLBACK_CENTROID_DIR, "simple_centroid.npy")):
            logger.info(
                "Centroid files not found in models/centroids/, "
                "falling back to reference/NadirClaw/nadirclaw/"
            )
            return _FALLBACK_CENTROID_DIR

        return _DEFAULT_CENTROID_DIR

    def _load_centroids(self) -> None:
        """Load simple and complex centroid vectors from ``.npy`` files.

        Validates that centroids are 384-dimensional and approximately
        unit-normalized. Thread-safe via lock.

        Raises:
            FileNotFoundError: If centroid files don't exist.
            ImportError: If numpy is not installed.
            ValueError: If centroid shape is invalid.
        """
        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "numpy is required for centroid-based routing. "
                "Install with: pip install numpy"
            )

        with self._load_lock:
            if self._loaded:
                return

            centroid_dir = self._resolve_centroid_dir()
            simple_path = os.path.join(centroid_dir, "simple_centroid.npy")
            complex_path = os.path.join(centroid_dir, "complex_centroid.npy")

            if not os.path.exists(simple_path) or not os.path.exists(complex_path):
                raise FileNotFoundError(
                    f"Centroid files not found in {centroid_dir}. "
                    "Run 'python scripts/generate_centroids.py' or "
                    "'python scripts/copy_centroids.py' to create them."
                )

            self._simple_centroid = np.load(simple_path)
            self._complex_centroid = np.load(complex_path)

            # Validate shapes
            if self._simple_centroid.shape != (_EMBEDDING_DIM,):
                raise ValueError(
                    f"simple_centroid has shape {self._simple_centroid.shape}, "
                    f"expected ({_EMBEDDING_DIM},)"
                )
            if self._complex_centroid.shape != (_EMBEDDING_DIM,):
                raise ValueError(
                    f"complex_centroid has shape {self._complex_centroid.shape}, "
                    f"expected ({_EMBEDDING_DIM},)"
                )

            # Verify approximate unit normalization (tolerance for float32)
            simple_norm = float(np.linalg.norm(self._simple_centroid))
            complex_norm = float(np.linalg.norm(self._complex_centroid))
            if abs(simple_norm - 1.0) > 0.01:
                logger.warning(
                    "simple_centroid is not unit-normalized (norm=%.4f), normalizing",
                    simple_norm,
                )
                self._simple_centroid = self._simple_centroid / simple_norm
            if abs(complex_norm - 1.0) > 0.01:
                logger.warning(
                    "complex_centroid is not unit-normalized (norm=%.4f), normalizing",
                    complex_norm,
                )
                self._complex_centroid = self._complex_centroid / complex_norm

            self._loaded = True
            logger.info(
                "Centroid classifier loaded from %s "
                "(simple_norm=%.4f, complex_norm=%.4f)",
                centroid_dir,
                simple_norm,
                complex_norm,
            )

    def classify(self, prompt: str) -> ClassificationResult:
        """Classify a prompt as simple or complex.

        Encodes the prompt using ``all-MiniLM-L6-v2``, normalizes the
        embedding, and computes cosine similarity to both centroids.
        Borderline cases (confidence below threshold) are biased toward
        the complex tier.

        Args:
            prompt: The user prompt text to classify.

        Returns:
            :class:`ClassificationResult` with tier, confidence, and
            similarity scores.

        Raises:
            ImportError: If numpy or sentence-transformers is not installed.
            FileNotFoundError: If centroid files don't exist.
        """
        start = time.time()

        # Lazy load centroids
        if not self._loaded:
            self._load_centroids()

        # Get shared encoder from strategies module
        from litellm_llmrouter.strategies import (
            DEFAULT_EMBEDDING_MODEL,
            _get_sentence_transformer,
        )

        encoder = _get_sentence_transformer(DEFAULT_EMBEDDING_MODEL)

        # Encode and normalize
        emb = encoder.encode([prompt], show_progress_bar=False)[0]
        emb = emb / np.linalg.norm(emb)

        # Cosine similarity (dot product of unit vectors)
        sim_simple = float(np.dot(emb, self._simple_centroid))
        sim_complex = float(np.dot(emb, self._complex_centroid))

        # Confidence and classification
        confidence = abs(sim_complex - sim_simple)

        if confidence < self._confidence_threshold:
            # Borderline: bias toward complex (safer to over-serve)
            is_complex = True
        else:
            is_complex = sim_complex > sim_simple

        tier = "complex" if is_complex else "simple"
        latency_ms = (time.time() - start) * 1000

        return ClassificationResult(
            tier=tier,
            confidence=confidence,
            sim_simple=sim_simple,
            sim_complex=sim_complex,
            latency_ms=latency_ms,
        )

    def warmup(self) -> None:
        """Pre-load encoder and centroids for fast first-request latency."""
        from litellm_llmrouter.strategies import (
            DEFAULT_EMBEDDING_MODEL,
            _get_sentence_transformer,
        )

        _get_sentence_transformer(DEFAULT_EMBEDDING_MODEL)
        self._load_centroids()
        logger.info("Centroid classifier warmed up")


# ---------------------------------------------------------------------------
# AgenticDetector
# ---------------------------------------------------------------------------


class AgenticDetector:
    """Detects agentic patterns in messages (tool use, multi-step, etc.).

    Ported from NadirClaw's ``detect_agentic()`` function. Uses cumulative
    scoring across multiple signals with a threshold of 0.35.
    """

    _AGENTIC_SYSTEM_KEYWORDS = re.compile(
        r"\b(you are an? (?:ai |coding |software )?agent"
        r"|execute (?:commands?|tools?|code|tasks?)"
        r"|you (?:can|have access to|may) (?:use |call |run |execute )"
        r"(?:tools?|functions?|commands?)"
        r"|tool[ _]?(?:use|call|execution)"
        r"|multi[- ]?step"
        r"|(?:read|write|edit|create|delete) files?"
        r"|run (?:commands?|shell|bash|terminal)"
        r"|code execution"
        r"|file (?:system|access)"
        r"|web ?search"
        r"|browser"
        r"|autonomous)\b",
        re.IGNORECASE,
    )

    THRESHOLD = 0.35
    """Minimum score for agentic classification."""

    @classmethod
    def detect(
        cls,
        messages: List[Dict[str, Any]],
        has_tools: bool = False,
        tool_count: int = 0,
    ) -> AgenticResult:
        """Score agentic signals in a request.

        Cumulative scoring system:

        - ``tools_defined``: +0.35 if ``has_tools`` and ``tool_count >= 1``
        - ``many_tools``: +0.15 if ``tool_count >= 4``
        - ``tool_messages``: +0.30 if any message has ``role == "tool"``
        - ``agentic_cycles`` (>=2): +0.20
        - ``single_cycle``: +0.10 (exactly 1 cycle)
        - ``long_system_prompt``: +0.10 if system prompt > 500 chars
        - ``agentic_keywords``: +0.20 if regex matches system prompt
        - ``deep_conversation``: +0.10 if message count > 10

        Score is capped at 1.0. ``is_agentic = score >= 0.35``.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            has_tools: Whether tools are defined in the request.
            tool_count: Number of tools defined.

        Returns:
            :class:`AgenticResult` with score and detected signals.
        """
        score = 0.0
        signals: List[str] = []

        # Tool definitions present
        if has_tools and tool_count >= 1:
            score += 0.35
            signals.append(f"tools_defined({tool_count})")
        if tool_count >= 4:
            score += 0.15
            signals.append("many_tools")

        # Tool-role messages (active agentic loop)
        tool_msgs = sum(1 for m in messages if m.get("role") == "tool")
        if tool_msgs >= 1:
            score += 0.30
            signals.append(f"tool_messages({tool_msgs})")

        # Assistant→tool cycles
        cycles = cls._count_agentic_cycles(messages)
        if cycles >= 2:
            score += 0.20
            signals.append(f"agentic_cycles({cycles})")
        elif cycles == 1:
            score += 0.10
            signals.append("single_cycle")

        # Extract system prompt
        system_prompt = ""
        for m in messages:
            if m.get("role") in ("system", "developer"):
                content = m.get("content", "")
                if isinstance(content, str):
                    system_prompt = content
                break

        # Long system prompt
        if len(system_prompt) > 500:
            score += 0.10
            signals.append("long_system_prompt")

        # Agentic keywords in system prompt
        if system_prompt and cls._AGENTIC_SYSTEM_KEYWORDS.search(system_prompt):
            score += 0.20
            signals.append("agentic_keywords")

        # Deep conversation
        if len(messages) > 10:
            score += 0.10
            signals.append("deep_conversation")

        # Cap at 1.0
        confidence = min(score, 1.0)
        is_agentic = confidence >= cls.THRESHOLD

        return AgenticResult(
            is_agentic=is_agentic,
            confidence=confidence,
            signals=signals,
        )

    @staticmethod
    def _count_agentic_cycles(messages: List[Dict[str, Any]]) -> int:
        """Count assistant→tool role transitions in the message list.

        A cycle is defined as an assistant message immediately followed
        by a tool message.

        Args:
            messages: List of message dicts.

        Returns:
            Number of assistant→tool cycles.
        """
        cycles = 0
        roles = [m.get("role", "") for m in messages]
        i = 0
        while i < len(roles) - 1:
            if roles[i] == "assistant" and roles[i + 1] == "tool":
                cycles += 1
                i += 2
            else:
                i += 1
        return cycles


# ---------------------------------------------------------------------------
# ReasoningDetector
# ---------------------------------------------------------------------------


class ReasoningDetector:
    """Detects reasoning-heavy prompts (math, logic, analysis, proofs).

    Ported from NadirClaw's ``detect_reasoning()`` function. Uses regex
    pattern matching with a threshold of 2+ distinct markers.
    """

    _REASONING_MARKERS = re.compile(
        r"\b(step[- ]by[- ]step"
        r"|think (?:through|carefully|deeply|about)"
        r"|chain[- ]of[- ]thought"
        r"|let'?s? reason"
        r"|reason(?:ing)? (?:about|through)"
        r"|prove (?:that|this|the)"
        r"|formal (?:proof|verification)"
        r"|mathematical(?:ly)? (?:prove|show|derive)"
        r"|derive (?:the|a|an)"
        r"|analyze the (?:tradeoffs?|trade-offs?|implications?|consequences?)"
        r"|compare and contrast"
        r"|what are the (?:pros? and cons?|advantages? and disadvantages?)"
        r"|evaluate (?:the|whether|if)"
        r"|critically (?:analyze|assess|examine)"
        r"|explain (?:why|how|the reasoning)"
        r"|work through"
        r"|break (?:this|it) down"
        r"|logical(?:ly)? (?:deduce|infer|conclude))\b",
        re.IGNORECASE,
    )

    MARKER_THRESHOLD = 2
    """Need 2+ distinct markers to classify as reasoning."""

    @classmethod
    def detect(cls, prompt: str, system_message: str = "") -> ReasoningResult:
        """Detect if a prompt requires reasoning capabilities.

        Combines ``system_message`` and ``prompt``, finds all regex
        matches, and counts unique markers. ``is_reasoning`` is True
        when 2+ distinct markers are found.

        Args:
            prompt: User prompt text.
            system_message: System message text (optional).

        Returns:
            :class:`ReasoningResult` with marker count and detection status.
        """
        combined = f"{system_message} {prompt}"
        matches = cls._REASONING_MARKERS.findall(combined)
        # Unique markers (case-insensitive dedup)
        unique_markers = list({m.lower() for m in matches})
        marker_count = len(unique_markers)

        is_reasoning = marker_count >= cls.MARKER_THRESHOLD

        return ReasoningResult(
            is_reasoning=is_reasoning,
            marker_count=marker_count,
            markers=unique_markers,
        )


# ---------------------------------------------------------------------------
# SessionCache
# ---------------------------------------------------------------------------


class SessionCache:
    """In-memory session-based routing affinity cache with LRU eviction.

    Ported from NadirClaw's ``SessionCache``. Keyed by a hash of the
    system prompt + first user message. Expired entries are cleaned up
    periodically (every 100 puts).
    """

    def __init__(self, ttl: int = 1800, max_size: int = 10000):
        """Initialize session cache.

        Args:
            ttl: Time-to-live in seconds (default 30 minutes).
            max_size: Maximum cache entries before LRU eviction.
        """
        self._cache: Dict[str, Tuple[str, str, float]] = {}
        self._ttl = ttl
        self._max_size = max_size
        self._access_order: List[str] = []
        self._put_counter = 0
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(system_message: str, first_user_message: str) -> str:
        """Generate a deterministic session key.

        SHA-256 hash of ``sys:{first 200 chars}|usr:{first 200 chars}``,
        truncated to 16 hex characters.

        Args:
            system_message: System prompt text.
            first_user_message: First user message text.

        Returns:
            16-character hex session key.
        """
        raw = f"sys:{system_message[:200]}|usr:{first_user_message[:200]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Tuple[str, str]]:
        """Retrieve cached routing decision if not expired.

        Args:
            key: Session key from :meth:`_make_key`.

        Returns:
            Tuple of ``(model, tier)`` if cache hit, ``None`` otherwise.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            model, tier, ts = entry
            if time.time() - ts > self._ttl:
                # Expired
                del self._cache[key]
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                return None
            # Touch for LRU
            self._touch(key)
            return model, tier

    def put(self, key: str, model: str, tier: str) -> None:
        """Store a routing decision for a session.

        Periodically cleans up expired entries (every 100 puts) and
        evicts LRU entries when over capacity.

        Args:
            key: Session key from :meth:`_make_key`.
            model: Selected model name.
            tier: Selected tier (``"simple"`` or ``"complex"``).
        """
        with self._lock:
            self._put_counter += 1
            if self._put_counter >= 100:
                self._put_counter = 0
                self._clear_expired()

            self._cache[key] = (model, tier, time.time())
            self._touch(key)

            # Evict LRU if over capacity
            while len(self._cache) > self._max_size and self._access_order:
                oldest = self._access_order.pop(0)
                self._cache.pop(oldest, None)

    def _touch(self, key: str) -> None:
        """Move key to end of LRU access order."""
        try:
            self._access_order.remove(key)
        except ValueError:
            pass
        self._access_order.append(key)

    def _clear_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [k for k, (_, _, ts) in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]
            try:
                self._access_order.remove(k)
            except ValueError:
                pass
        return len(expired)

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# CentroidRoutingStrategy
# ---------------------------------------------------------------------------


class CentroidRoutingStrategy:
    """Zero-config intelligent routing strategy using centroid classification.

    Ported from NadirClaw's architecture. Provides intelligent model routing
    without requiring ML model training. Uses pre-computed centroid vectors
    for ~2ms prompt classification.

    Implements the :class:`~litellm_llmrouter.strategy_registry.RoutingStrategy`
    interface for integration with RouteIQ's strategy registry.

    Routing flow:

    1. Extract prompt from ``context.request_data["messages"]``
    2. Check session cache for existing affinity
    3. Classify prompt via :class:`CentroidClassifier`
    4. Detect agentic patterns via :class:`AgenticDetector`
    5. Detect reasoning patterns via :class:`ReasoningDetector`
    6. Apply routing modifiers (agentic: simple→complex, reasoning: →complex)
    7. Apply profile overrides (eco → simple, premium → complex, etc.)
    8. Match tier to available deployment from healthy deployments
    9. Cache result for session persistence
    10. Return selected deployment dict or ``None``
    """

    def __init__(
        self,
        centroid_dir: Optional[str] = None,
        confidence_threshold: float = 0.06,
        profile: RoutingProfile = RoutingProfile.AUTO,
        session_ttl: int = 1800,
        tier_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize centroid routing strategy.

        Args:
            centroid_dir: Directory containing centroid ``.npy`` files.
            confidence_threshold: Confidence threshold for classification.
            profile: Routing profile (auto, eco, premium, free, reasoning).
            session_ttl: Session cache TTL in seconds.
            tier_mapping: Maps tier names to model name patterns.
                Example: ``{"simple": ["gpt-4o-mini", "claude-haiku"],
                "complex": ["gpt-4o", "claude-sonnet"]}``
        """
        self._classifier = CentroidClassifier(centroid_dir, confidence_threshold)
        self._agentic_detector = AgenticDetector()
        self._reasoning_detector = ReasoningDetector()
        self._session_cache = SessionCache(ttl=session_ttl)
        self._profile = profile
        self._tier_mapping = tier_mapping or {}
        self._classifier_available = True

    @property
    def name(self) -> str:
        """Strategy name for telemetry and registry."""
        return "llmrouter-nadirclaw-centroid"

    @property
    def version(self) -> Optional[str]:
        """Strategy version."""
        return "1.0.0"

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate the strategy is ready to serve requests."""
        if not _NUMPY_AVAILABLE:
            return False, "numpy is required for centroid routing"
        return True, None

    def select_deployment(
        self,
        context: Any,
    ) -> Optional[Dict]:
        """Select a deployment for the given routing context.

        Implements the main routing flow: classify → detect modifiers →
        apply profile → match tier → cache → return.

        Args:
            context: :class:`~litellm_llmrouter.strategy_registry.RoutingContext`
                with request details.

        Returns:
            Selected deployment dict, or ``None`` if no selection possible.
        """
        # Extract messages from context
        messages = context.messages or []
        if not messages:
            # Try request_kwargs
            rk = context.request_kwargs or {}
            messages = rk.get("messages", [])

        if not messages:
            return self._fallback_deployment(context)

        # Extract prompt components
        prompt, system_message, full_messages = self._extract_prompt(messages)

        if not prompt:
            return self._fallback_deployment(context)

        # Check session cache
        session_key = SessionCache._make_key(system_message, prompt)
        cached = self._session_cache.get(session_key)
        if cached is not None:
            cached_model, cached_tier = cached
            logger.debug(
                "Session cache hit: key=%s, model=%s, tier=%s",
                session_key[:8],
                cached_model,
                cached_tier,
            )
            deployment = self._match_tier_to_deployment(
                cached_tier, self._get_healthy_deployments(context)
            )
            if deployment is not None:
                return deployment

        # Classify prompt
        try:
            result = self._classifier.classify(prompt)
            tier = result.tier
        except (ImportError, FileNotFoundError, ValueError) as e:
            logger.warning("Centroid classification failed: %s", e)
            self._classifier_available = False
            return self._fallback_deployment(context)

        # Detect agentic patterns
        has_tools = bool(
            (context.request_kwargs or {}).get("tools")
            or (context.request_kwargs or {}).get("functions")
        )
        tool_count = len(
            (context.request_kwargs or {}).get("tools", [])
            or (context.request_kwargs or {}).get("functions", [])
        )
        agentic_result = self._agentic_detector.detect(
            full_messages,
            has_tools=has_tools,
            tool_count=tool_count,
        )

        # Detect reasoning patterns
        reasoning_result = self._reasoning_detector.detect(prompt, system_message)

        # Apply modifiers
        # 1. Agentic override: simple → complex
        if agentic_result.is_agentic and tier == "simple":
            tier = "complex"
            logger.debug(
                "Agentic override: simple→complex (confidence=%.2f, signals=%s)",
                agentic_result.confidence,
                agentic_result.signals,
            )

        # 2. Reasoning override: → complex
        if reasoning_result.is_reasoning and tier != "complex":
            tier = "complex"
            logger.debug(
                "Reasoning override: →complex (markers=%d: %s)",
                reasoning_result.marker_count,
                reasoning_result.markers,
            )

        # 3. Apply profile overrides
        tier = self._apply_profile(tier, self._profile)

        # Match tier to deployment
        deployments = self._get_healthy_deployments(context)
        deployment = self._match_tier_to_deployment(tier, deployments)

        # Cache result
        if deployment is not None:
            model_name = deployment.get("litellm_params", {}).get("model", "")
            self._session_cache.put(session_key, model_name, tier)

        return deployment

    def _extract_prompt(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Extract prompt components from messages.

        Args:
            messages: List of message dicts.

        Returns:
            Tuple of ``(last_user_message, system_message, full_messages)``.
        """
        system_message = ""
        last_user_message = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Multi-modal: extract text parts
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                content = " ".join(parts)

            if not isinstance(content, str):
                content = str(content) if content else ""

            if role in ("system", "developer"):
                system_message = content
            elif role == "user":
                last_user_message = content

        return last_user_message, system_message, messages

    @staticmethod
    def _apply_profile(tier: str, profile: RoutingProfile) -> str:
        """Apply routing profile override to the tier.

        Args:
            tier: Current tier (``"simple"`` or ``"complex"``).
            profile: Routing profile.

        Returns:
            Final tier after profile override.
        """
        if profile == RoutingProfile.ECO:
            return "simple"
        elif profile == RoutingProfile.PREMIUM:
            return "complex"
        elif profile == RoutingProfile.FREE:
            return "simple"
        elif profile == RoutingProfile.REASONING:
            return "complex"
        # AUTO: use classifier result
        return tier

    def _match_tier_to_deployment(
        self, tier: str, deployments: List[Dict]
    ) -> Optional[Dict]:
        """Match a complexity tier to an available deployment.

        Uses ``tier_mapping`` if configured. Otherwise, uses a heuristic
        based on model name patterns (mini/nano/haiku → simple, otherwise
        → complex). Falls back to random selection if no match.

        Args:
            tier: Target tier (``"simple"`` or ``"complex"``).
            deployments: List of healthy deployment dicts.

        Returns:
            Selected deployment dict, or ``None`` if no deployments.
        """
        if not deployments:
            return None

        # If tier_mapping is configured, use it
        if self._tier_mapping and tier in self._tier_mapping:
            patterns = self._tier_mapping[tier]
            for deployment in deployments:
                model = deployment.get("litellm_params", {}).get("model", "")
                for pattern in patterns:
                    if pattern.lower() in model.lower():
                        return deployment

        # Heuristic matching based on model name patterns
        # Simple tier: prefer smaller/cheaper models
        # Complex tier: prefer larger/more capable models
        simple_indicators = [
            "mini",
            "nano",
            "haiku",
            "flash",
            "small",
            "light",
            "instant",
        ]
        complex_indicators = [
            "opus",
            "sonnet",
            "pro",
            "large",
            "turbo",
            "4o",
            "gpt-4",
            "claude-3-opus",
        ]

        simple_deployments = []
        complex_deployments = []
        other_deployments = []

        for deployment in deployments:
            model = deployment.get("litellm_params", {}).get("model", "").lower()

            is_simple = any(ind in model for ind in simple_indicators)
            is_complex = any(ind in model for ind in complex_indicators)

            if is_simple:
                # Simple indicators take priority (e.g., "gpt-4o-mini" is simple)
                simple_deployments.append(deployment)
            elif is_complex:
                complex_deployments.append(deployment)
            else:
                other_deployments.append(deployment)

        if tier == "simple":
            candidates = simple_deployments or other_deployments or complex_deployments
        else:
            candidates = complex_deployments or other_deployments or simple_deployments

        if candidates:
            return candidates[0]

        # Final fallback: random from all deployments
        return random.choice(deployments)

    @staticmethod
    def _get_healthy_deployments(context: Any) -> List[Dict]:
        """Get healthy deployments from the routing context.

        Args:
            context: Routing context with router instance.

        Returns:
            List of deployment dicts matching the requested model.
        """
        router = context.router
        if router is None:
            return []

        healthy = getattr(router, "healthy_deployments", None)
        if healthy is None:
            healthy = getattr(router, "model_list", [])

        return [dep for dep in healthy if dep.get("model_name") == context.model]

    @staticmethod
    def _fallback_deployment(context: Any) -> Optional[Dict]:
        """Select a random healthy deployment as fallback.

        Used when classification fails or no prompt is available.

        Args:
            context: Routing context.

        Returns:
            Random deployment dict for the requested model, or ``None``.
        """
        router = context.router
        if router is None:
            return None

        healthy = getattr(router, "healthy_deployments", None)
        if healthy is None:
            healthy = getattr(router, "model_list", [])

        candidates = [dep for dep in healthy if dep.get("model_name") == context.model]
        if candidates:
            return random.choice(candidates)
        return None


# ---------------------------------------------------------------------------
# Module-level singleton access
# ---------------------------------------------------------------------------

_centroid_strategy: Optional[CentroidRoutingStrategy] = None
_centroid_lock = threading.Lock()


def get_centroid_strategy(**kwargs: Any) -> CentroidRoutingStrategy:
    """Get or create the singleton :class:`CentroidRoutingStrategy`.

    Keyword arguments are passed to the constructor on first call.

    Returns:
        The singleton strategy instance.
    """
    global _centroid_strategy

    with _centroid_lock:
        if _centroid_strategy is None:
            _centroid_strategy = CentroidRoutingStrategy(**kwargs)
        return _centroid_strategy


def reset_centroid_strategy() -> None:
    """Reset the singleton strategy (for testing)."""
    global _centroid_strategy

    with _centroid_lock:
        _centroid_strategy = None


def warmup_centroid_classifier() -> None:
    """Pre-warm the centroid classifier (load encoder + centroids).

    Call at application startup to avoid cold-start latency on the
    first request.
    """
    strategy = get_centroid_strategy()
    try:
        strategy._classifier.warmup()
    except Exception as e:
        logger.warning("Centroid classifier warmup failed: %s", e)
