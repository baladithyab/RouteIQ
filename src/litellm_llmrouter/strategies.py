"""
LLMRouter Routing Strategies for LiteLLM
==========================================

This module implements the integration between LLMRouter's ML-based
routing strategies and LiteLLM's routing infrastructure.

Security Notes:
- Pickle loading is disabled by default due to RCE risk
- Set LLMROUTER_ALLOW_PICKLE_MODELS=true to enable (only in trusted environments)
- When pickle is enabled, use LLMROUTER_MODEL_MANIFEST_PATH for hash/signature verification
- Set LLMROUTER_ENFORCE_SIGNED_MODELS=true to require manifest verification
"""

import json
import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tempfile
import yaml
from litellm._logging import verbose_proxy_logger

# Import model artifact verification
from litellm_llmrouter.model_artifacts import (
    get_artifact_verifier,
    ModelVerificationError,
    ActiveModelVersion,
    PickleSignatureRequiredError,
    SignatureVerificationError,
    SignatureType,
)

# Import telemetry contracts for versioned event emission
from litellm_llmrouter.telemetry_contracts import (
    RouterDecisionEventBuilder,
    RoutingOutcome,
    ROUTER_DECISION_EVENT_NAME,
    ROUTER_DECISION_PAYLOAD_KEY,
)

# Import TG4.1 router decision span attributes helper
from litellm_llmrouter.observability import set_router_decision_attributes

# Import routing strategy base class for CostAwareRoutingStrategy
from litellm_llmrouter.strategy_registry import RoutingContext, RoutingStrategy

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Cost metric for cost-aware routing (lazy-initialized)
_cost_meter = None
_cost_histogram = None


def _get_cost_histogram():
    """Get or create the cost-per-1K-tokens histogram for cost-aware routing.

    Lazy-initializes the OpenTelemetry meter and histogram on first call.
    Returns None if OpenTelemetry metrics are not available.
    """
    global _cost_meter, _cost_histogram
    if _cost_histogram is None:
        try:
            from opentelemetry import metrics

            _cost_meter = metrics.get_meter("routeiq.routing")
            _cost_histogram = _cost_meter.create_histogram(
                "routeiq.routing.cost_per_1k_tokens",
                unit="USD",
                description="Cost per 1K tokens for the selected model",
            )
        except Exception:
            pass
    return _cost_histogram


# Lazy import for sentence-transformers to avoid startup cost if not needed
_sentence_transformer_model = None
_sentence_transformer_lock = threading.Lock()


def _get_sentence_transformer(model_name: str, device: str = "cpu"):
    """
    Get or create a cached SentenceTransformer model.

    Uses lazy loading with thread-safe singleton pattern to avoid
    loading the model multiple times across requests.

    Args:
        model_name: HuggingFace model name for sentence-transformers
        device: Device to load model on ('cpu', 'cuda', etc.)

    Returns:
        SentenceTransformer model instance
    """
    global _sentence_transformer_model

    with _sentence_transformer_lock:
        if _sentence_transformer_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                verbose_proxy_logger.info(
                    f"Loading SentenceTransformer model: {model_name} on {device}"
                )
                _sentence_transformer_model = SentenceTransformer(
                    model_name, device=device
                )
                verbose_proxy_logger.info(
                    "SentenceTransformer model loaded successfully"
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers package is required for KNN inference. "
                    "Install with: pip install sentence-transformers"
                )
        return _sentence_transformer_model


# Default embedding model matching the training pipeline
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Security: Pickle loading is disabled by default to prevent RCE
# Set LLMROUTER_ALLOW_PICKLE_MODELS=true to enable in trusted environments
ALLOW_PICKLE_MODELS = (
    os.getenv("LLMROUTER_ALLOW_PICKLE_MODELS", "false").lower() == "true"
)

# When pickle is allowed, require manifest verification by default
# Set LLMROUTER_ENFORCE_SIGNED_MODELS=false to bypass (not recommended)
ENFORCE_SIGNED_MODELS = os.getenv(
    "LLMROUTER_ENFORCE_SIGNED_MODELS", ""
).lower() == "true" or (
    ALLOW_PICKLE_MODELS
    and os.getenv("LLMROUTER_ENFORCE_SIGNED_MODELS", "").lower() != "false"
)

# Strict pickle mode: require signature verification for all pickle files
# When true, pickle files must have a valid signature in the manifest
# Set LLMROUTER_STRICT_PICKLE_MODE=true to enable (recommended for production)
STRICT_PICKLE_MODE = (
    os.getenv("LLMROUTER_STRICT_PICKLE_MODE", "false").lower() == "true"
)

# Pickle allowlist: SHA256 hashes of pickle files that bypass signature requirement
# Comma-separated list of hex-encoded SHA256 hashes
PICKLE_ALLOWLIST = set(
    h.strip()
    for h in os.getenv("LLMROUTER_PICKLE_ALLOWLIST", "").split(",")
    if h.strip()
)


class PickleSecurityError(Exception):
    """Raised when pickle loading is attempted but not explicitly allowed."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        super().__init__(
            f"Pickle loading is disabled for security (RCE risk). "
            f"To enable pickle model loading, set LLMROUTER_ALLOW_PICKLE_MODELS=true. "
            f"Model path: {model_path}"
        )


class ModelLoadError(Exception):
    """Raised when model loading fails during safe activation."""

    def __init__(
        self, model_path: str, reason: str, correlation_id: Optional[str] = None
    ):
        self.model_path = model_path
        self.reason = reason
        self.correlation_id = correlation_id
        super().__init__(f"Model loading failed for '{model_path}': {reason}")


class InferenceKNNRouter:
    """
    Lightweight inference-only KNN router that loads sklearn models directly.

    This class bypasses the UIUC LLMRouter's MetaRouter initialization which
    requires training data. Instead, it:
    - Loads a pre-trained sklearn KNeighborsClassifier from a .pkl file
    - Uses sentence-transformers for text embedding (same as training)
    - Predicts the best model label for a given query

    The trained .pkl file is produced by UIUC's KNNRouterTrainer which calls
    sklearn's KNeighborsClassifier.fit() and saves via pickle.

    Security:
    - Pickle loading requires LLMROUTER_ALLOW_PICKLE_MODELS=true
    - This protects against RCE via malicious pickle files
    - When enabled, artifacts are verified against manifest if configured

    Safe Activation:
    - Models are loaded into a temporary instance first
    - Only swapped to active if loading succeeds
    - On failure, the old model remains active

    Attributes:
        model_path: Path to the trained .pkl model file
        embedding_model: Name of the sentence-transformer model
        embedding_device: Device for embedding model ('cpu', 'cuda')
        knn_model: Loaded sklearn KNeighborsClassifier
        label_mapping: Optional mapping from predicted labels to LLM candidate keys
        model_version: Active model version metadata for observability
    """

    def __init__(
        self,
        model_path: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_device: str = "cpu",
        label_mapping: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize inference-only KNN router.

        Args:
            model_path: Path to the trained sklearn KNN model (.pkl file)
            embedding_model: HuggingFace model name for sentence embeddings
            embedding_device: Device for embedding model ('cpu', 'cuda', etc.)
            label_mapping: Optional dict mapping predicted labels to LLM keys
            correlation_id: Optional correlation ID for logging
        """
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.label_mapping = label_mapping or {}
        self.knn_model = None
        self.model_version: Optional[ActiveModelVersion] = None
        self._model_lock = threading.RLock()

        # Load the model with verification
        self._load_model(correlation_id=correlation_id)

    def _load_model(self, correlation_id: Optional[str] = None):
        """Load the sklearn KNN model from pickle file with verification.

        Security:
        - Requires LLMROUTER_ALLOW_PICKLE_MODELS=true environment variable.
        - Pickle deserialization can execute arbitrary code, so it's disabled by default.
        - When enabled, verifies artifact against manifest if LLMROUTER_MODEL_MANIFEST_PATH is set.
        - In strict mode (LLMROUTER_STRICT_PICKLE_MODE=true), requires signed manifest.

        Safe Activation:
        - Loads model into temporary variable first
        - Only swaps to active if successful
        - Records model version for observability
        """
        if not self.model_path:
            raise ValueError("model_path is required for InferenceKNNRouter")

        # Security check: pickle loading disabled by default
        if not ALLOW_PICKLE_MODELS:
            raise PickleSecurityError(self.model_path)

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"KNN model file not found: {self.model_path}")

        log_prefix = f"[{correlation_id}] " if correlation_id else ""

        # Verify artifact against manifest if enforcement is enabled
        verifier = get_artifact_verifier()
        require_manifest = ENFORCE_SIGNED_MODELS

        # Compute hash first for allowlist check and verification
        computed_hash = verifier.compute_sha256(self.model_path)

        # Check if hash is in allowlist (bypasses signature requirement)
        is_allowlisted = computed_hash in PICKLE_ALLOWLIST
        if is_allowlisted:
            verbose_proxy_logger.info(
                f"{log_prefix}Model in pickle allowlist: {self.model_path} "
                f"(sha256={computed_hash[:16]}...)"
            )

        # Strict pickle mode check
        if STRICT_PICKLE_MODE and not is_allowlisted:
            # Must have a signed manifest
            manifest = verifier._load_manifest()
            if manifest is None:
                raise PickleSignatureRequiredError(self.model_path, strict_mode=True)

            # Verify manifest has a signature
            if manifest.signature_type == SignatureType.NONE:
                raise PickleSignatureRequiredError(self.model_path, strict_mode=True)

            # Verify signature is valid
            try:
                verifier.verify_manifest_signature(manifest)
            except SignatureVerificationError as e:
                verbose_proxy_logger.error(
                    f"{log_prefix}STRICT_PICKLE_MODE: Signature verification failed: {e}"
                )
                raise PickleSignatureRequiredError(self.model_path, strict_mode=True)

            verbose_proxy_logger.info(
                f"{log_prefix}STRICT_PICKLE_MODE: Manifest signature verified for {self.model_path}"
            )

        try:
            verifier.verify_artifact(
                self.model_path,
                require_manifest=require_manifest,
                correlation_id=correlation_id,
            )
        except ModelVerificationError as e:
            verbose_proxy_logger.error(
                f"{log_prefix}Model verification failed: {e}. "
                f"Hash mismatch or manifest missing. Details: {e.details}"
            )
            raise

        verbose_proxy_logger.info(
            f"{log_prefix}Loading KNN model from: {self.model_path}"
        )

        # Safe activation: load into temp variable first
        try:
            with open(self.model_path, "rb") as f:
                new_model = pickle.load(f)
        except Exception as e:
            raise ModelLoadError(
                self.model_path,
                f"Pickle load failed: {e}",
                correlation_id,
            )

        # Verify it's a sklearn model with predict method
        if not hasattr(new_model, "predict"):
            raise ModelLoadError(
                self.model_path,
                f"Loaded model does not have 'predict' method. "
                f"Expected sklearn KNeighborsClassifier, got {type(new_model)}",
                correlation_id,
            )

        # Safe swap: only update if everything succeeded
        with self._model_lock:
            old_model = self.knn_model
            self.knn_model = new_model

            # Record active version for observability
            self.model_version = verifier.record_active_version(
                self.model_path,
                sha256=computed_hash,
                tags=["knn", "active"],
            )

        verbose_proxy_logger.info(
            f"{log_prefix}KNN model loaded successfully. Type: {type(self.knn_model).__name__}, "
            f"Version SHA256: {self.model_version.sha256[:16]}..."
        )

        # Clean up old model reference (let GC handle it)
        del old_model

    def reload_model(self, correlation_id: Optional[str] = None) -> bool:
        """
        Reload the model from disk with safe activation (for hot reload support).

        Safe Activation Pattern:
        1. Load new model into temporary instance
        2. Verify against manifest
        3. Only swap to active if successful
        4. Keep old model active on failure

        Args:
            correlation_id: Optional correlation ID for logging

        Returns:
            True if reload succeeded, False if failed (old model remains active)
        """
        log_prefix = f"[{correlation_id}] " if correlation_id else ""
        old_version = self.model_version

        try:
            self._load_model(correlation_id=correlation_id)
            verbose_proxy_logger.info(
                f"{log_prefix}Model reloaded successfully. "
                f"Old version: {old_version.sha256[:16] if old_version else 'none'}..., "
                f"New version: {self.model_version.sha256[:16] if self.model_version else 'none'}..."
            )
            return True
        except (ModelVerificationError, ModelLoadError, FileNotFoundError) as e:
            verbose_proxy_logger.error(
                f"{log_prefix}Model reload failed, keeping old model active. Error: {e}"
            )
            return False
        except Exception as e:
            verbose_proxy_logger.error(
                f"{log_prefix}Unexpected error during model reload, keeping old model active. Error: {e}"
            )
            return False

    def route(self, query: str) -> Optional[str]:
        """
        Route a query to the best model using KNN prediction.

        Args:
            query: User query text to route

        Returns:
            Predicted model label/key, or None if prediction fails
        """
        if self.knn_model is None:
            verbose_proxy_logger.warning("KNN model not loaded, cannot route")
            return None

        try:
            # Get embedding using the same model used in training
            embedder = _get_sentence_transformer(
                self.embedding_model, self.embedding_device
            )

            # Encode the query to get embedding vector
            # Shape: (embedding_dim,) -> need (1, embedding_dim) for predict
            embedding = embedder.encode([query], convert_to_numpy=True)

            # Predict using the KNN model
            predicted_label = self.knn_model.predict(embedding)[0]

            # Security: Log only query length and prediction, not query content (PII risk)
            verbose_proxy_logger.debug(
                f"KNN routing: query_length={len(query)} -> predicted={predicted_label}"
            )

            # Apply label mapping if configured
            if self.label_mapping and predicted_label in self.label_mapping:
                mapped_label = self.label_mapping[predicted_label]
                verbose_proxy_logger.debug(
                    f"KNN label mapping: {predicted_label} -> {mapped_label}"
                )
                return mapped_label

            return str(predicted_label)

        except Exception as e:
            verbose_proxy_logger.error(f"KNN routing error: {e}")
            return None


# Available LLMRouter strategies (matching llmrouter.models exports)
# See: https://github.com/ulab-uiuc/LLMRouter#-supported-routers
LLMROUTER_STRATEGIES = [
    # Single-round routers
    "llmrouter-knn",  # KNNRouter - K-Nearest Neighbors
    "llmrouter-svm",  # SVMRouter - Support Vector Machine
    "llmrouter-mlp",  # MLPRouter - Multi-Layer Perceptron
    "llmrouter-mf",  # MFRouter - Matrix Factorization
    "llmrouter-elo",  # EloRouter - Elo Rating based
    "llmrouter-routerdc",  # RouterDC - Dual Contrastive learning
    "llmrouter-hybrid",  # HybridLLMRouter - Probabilistic hybrid
    "llmrouter-causallm",  # CausalLMRouter - Transformer-based
    "llmrouter-graph",  # GraphRouter - Graph neural network
    "llmrouter-automix",  # AutomixRouter - Automatic model mixing
    # Multi-round routers
    "llmrouter-r1",  # RouterR1 - Pre-trained multi-turn router (requires vLLM)
    # Personalized routers
    "llmrouter-gmt",  # GMTRouter - Graph-based personalized router
    # Agentic routers
    "llmrouter-knn-multiround",  # KNNMultiRoundRouter - KNN agentic router
    "llmrouter-llm-multiround",  # LLMMultiRoundRouter - LLM agentic router
    # Baseline routers
    "llmrouter-smallest",  # SmallestLLM - Always picks smallest
    "llmrouter-largest",  # LargestLLM - Always picks largest
    # Custom routers
    "llmrouter-custom",  # User-defined custom router
    # Cost-aware routers
    "llmrouter-cost-aware",  # CostAwareRoutingStrategy - cheapest adequate model
]


# Default hyperparameters for each router type when no config is provided
# These match the defaults used in the UIUC LLMRouter library
DEFAULT_ROUTER_HPARAMS: Dict[str, Dict[str, Any]] = {
    "knn": {
        "n_neighbors": 5,
        "metric": "cosine",
        "weights": "distance",
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
    },
    "mlp": {
        "hidden_layer_sizes": [128, 64],
        "activation": "relu",
        "max_iter": 500,
    },
    "mf": {
        "n_factors": 64,
        "n_epochs": 20,
        "lr": 0.01,
    },
    "elo": {
        "k_factor": 32,
        "initial_rating": 1500,
    },
    "routerdc": {
        "temperature": 0.07,
        "hidden_size": 768,
    },
    "hybrid": {
        "threshold": 0.5,
    },
    "causallm": {
        "model_name": "gpt2",
        "max_length": 512,
    },
    "graph": {
        "hidden_dim": 128,
        "num_layers": 2,
    },
    "automix": {
        "alpha": 0.5,
    },
    "gmt": {
        "hidden_dim": 64,
    },
    "knn-multiround": {
        "n_neighbors": 5,
        "max_rounds": 3,
    },
    "llm-multiround": {
        "max_rounds": 3,
    },
    "smallest": {},
    "largest": {},
    "cost-aware": {
        "quality_threshold": 0.7,
        "cost_weight": 0.7,
        "inner_strategy": None,
        "inner_strategy_name": None,
        "max_cost_per_1k_tokens": None,
        "cost_refresh_interval": 3600,
        "enable_circuit_breaker_filtering": True,
    },
}


class LLMRouterStrategyFamily:
    """
    Wraps LLMRouter routing models to work with LiteLLM's routing infrastructure.

    This class provides:
    - Lazy loading of LLMRouter models
    - Hot-reloading support for model updates
    - Thread-safe model access
    - Mapping between LiteLLM deployments and LLMRouter model names
    """

    def __init__(
        self,
        strategy_name: str,
        model_path: Optional[str] = None,
        llm_data_path: Optional[str] = None,
        config_path: Optional[str] = None,
        hot_reload: bool = True,
        reload_interval: int = 300,
        model_s3_bucket: Optional[str] = None,
        model_s3_key: Optional[str] = None,
        # New inference-only KNN config keys
        embedding_model: Optional[str] = None,
        embedding_device: str = "cpu",
        label_mapping: Optional[Dict[str, str]] = None,
        use_inference_only: bool = True,  # Default to inference-only for KNN
        **kwargs,
    ):
        self.strategy_name = strategy_name
        self.model_path = model_path or os.environ.get("LLMROUTER_MODEL_PATH")
        self.llm_data_path = llm_data_path or os.environ.get("LLMROUTER_LLM_DATA_PATH")
        self.config_path = config_path
        self.hot_reload = hot_reload
        self.reload_interval = reload_interval
        self.model_s3_bucket = model_s3_bucket
        self.model_s3_key = model_s3_key
        # New inference-only config
        self.embedding_model = embedding_model or os.environ.get(
            "LLMROUTER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.embedding_device = embedding_device or os.environ.get(
            "LLMROUTER_EMBEDDING_DEVICE", "cpu"
        )
        self.label_mapping = label_mapping or {}
        self.use_inference_only = use_inference_only
        self.extra_kwargs = kwargs

        self._router = None
        self._router_lock = threading.RLock()
        self._last_load_time = 0
        self._model_mtime = 0

        # Resolve model_path if it's a directory (find .pkl file inside)
        self.model_path = self._resolve_model_path(self.model_path)

        # Load LLM candidates data
        self._llm_data = self._load_llm_data()

        verbose_proxy_logger.info(f"Initialized LLMRouter strategy: {strategy_name}")
        if self.use_inference_only and strategy_name == "llmrouter-knn":
            verbose_proxy_logger.info(
                f"  Using inference-only mode with embedding_model={self.embedding_model}"
            )

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        """
        Resolve model path to an actual file.

        If model_path is a directory, look for a .pkl file inside.
        This allows flexibility: users can specify either the directory
        or the exact .pkl file path.

        Args:
            model_path: Path to model file or directory

        Returns:
            Resolved path to the actual model file, or original path if resolution fails
        """
        if not model_path:
            return None

        path = Path(model_path)

        # If it's already a file, use it directly
        if path.is_file():
            return str(path)

        # If it's a directory, look for .pkl files
        if path.is_dir():
            pkl_files = list(path.glob("*.pkl"))
            if len(pkl_files) == 1:
                resolved = str(pkl_files[0])
                verbose_proxy_logger.info(
                    f"Resolved model directory to file: {resolved}"
                )
                return resolved
            elif len(pkl_files) > 1:
                # Multiple .pkl files - use the most recently modified
                pkl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                resolved = str(pkl_files[0])
                verbose_proxy_logger.warning(
                    f"Multiple .pkl files found, using most recent: {resolved}"
                )
                return resolved
            else:
                verbose_proxy_logger.warning(
                    f"Directory {model_path} contains no .pkl files"
                )

        # Return original path (may not exist yet, will be created by training)
        return model_path

    def _load_llm_data(self) -> Dict[str, Any]:
        """Load LLM candidates data from JSON file."""
        if not self.llm_data_path:
            return {}

        try:
            with open(self.llm_data_path, "r") as f:
                return json.load(f)
        except Exception as e:
            verbose_proxy_logger.warning(f"Failed to load LLM data: {e}")
            return {}

    def _should_reload(self) -> bool:
        """Check if model should be reloaded."""
        if not self.hot_reload or not self.model_path:
            return False

        # Check time-based reload
        if time.time() - self._last_load_time < self.reload_interval:
            return False

        # Check file modification time
        try:
            current_mtime = Path(self.model_path).stat().st_mtime
            if current_mtime > self._model_mtime:
                return True
        except OSError:
            pass

        return False

    def _get_or_create_config_path(self, strategy_type: str) -> str:
        """
        Get or create a temporary configuration file for the router.

        If config_path is provided, use it directly.
        Otherwise, create a temporary YAML file with default hyperparameters
        and placeholder paths for the given router type.

        The LLMRouter library expects a YAML config with specific keys:
        - hparam: Hyperparameters for the router algorithm
        - data_path: Paths to training data (placeholders for inference-only mode)
        - model_path: Paths for model loading/saving

        Args:
            strategy_type: The type of router (e.g., "knn", "svm", etc.)

        Returns:
            Path to the configuration file
        """
        if self.config_path:
            return self.config_path

        # Get default hyperparameters for the given router type
        hparams = DEFAULT_ROUTER_HPARAMS.get(strategy_type, {})

        # Build a minimal config structure that LLMRouter expects
        config = {
            "hparam": hparams,
            "data_path": {
                # Placeholder paths - not used during inference
                "routing_data_train": "/tmp/placeholder_train.jsonl",
                "routing_data_test": "/tmp/placeholder_test.jsonl",
                "query_embedding_data": "/tmp/placeholder_embeddings.pt",
                "llm_data": self.llm_data_path or "/tmp/placeholder_llm_data.json",
            },
            "model_path": {
                "load_model_path": self.model_path or "/tmp/placeholder_model.pkl",
                "save_model_path": self.model_path or "/tmp/placeholder_model.pkl",
            },
        }

        # Create a temporary YAML file
        fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="llmrouter_config_")
        try:
            with os.fdopen(fd, "w") as tmp:
                yaml.dump(config, tmp, default_flow_style=False)
            verbose_proxy_logger.info(
                f"Created temporary config for {strategy_type} router: {tmp_path}"
            )
            return tmp_path
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to create temporary config: {e}")
            os.close(fd)
            raise

    def _load_router(self):
        """Load the appropriate LLMRouter model based on strategy name."""
        strategy_type = self.strategy_name.replace("llmrouter-", "")

        # For KNN with inference-only mode, use our lightweight InferenceKNNRouter
        if strategy_type == "knn" and self.use_inference_only:
            return self._load_inference_knn_router()

        # Map strategy names to router classes
        router_map = {
            # Single-round routers
            "knn": ("KNNRouter", False),
            "svm": ("SVMRouter", False),
            "mlp": ("MLPRouter", False),
            "mf": ("MFRouter", False),
            "elo": ("EloRouter", False),
            "routerdc": ("RouterDC", False),
            "hybrid": ("HybridLLMRouter", False),
            "causallm": ("CausalLMRouter", True),  # optional
            "graph": ("GraphRouter", True),  # optional
            "automix": ("AutomixRouter", False),
            # Multi-round routers
            "r1": ("RouterR1", True),  # requires vLLM
            # Personalized routers
            "gmt": ("GMTRouter", False),
            # Agentic routers
            "knn-multiround": ("KNNMultiRoundRouter", False),
            "llm-multiround": ("LLMMultiRoundRouter", False),
            # Baseline routers
            "smallest": ("SmallestLLM", False),
            "largest": ("LargestLLM", False),
        }

        try:
            if strategy_type == "custom":
                return self._load_custom_router()

            if strategy_type not in router_map:
                verbose_proxy_logger.warning(
                    f"Unknown LLMRouter strategy: {strategy_type}, using MetaRouter"
                )
                from llmrouter.models import MetaRouter

                config_path = self._get_or_create_config_path(strategy_type)
                return MetaRouter(yaml_path=config_path)

            router_class_name, is_optional = router_map[strategy_type]

            # Import from llmrouter.models
            from llmrouter import models as llmrouter_models

            router_class = getattr(llmrouter_models, router_class_name, None)

            if router_class is None:
                if is_optional:
                    verbose_proxy_logger.warning(
                        f"Optional router {router_class_name} not available. "
                        "Install required dependencies."
                    )
                    return None
                else:
                    raise ImportError(f"Router class {router_class_name} not found")

            # Get or create config path with defaults if not provided
            config_path = self._get_or_create_config_path(strategy_type)
            router = router_class(yaml_path=config_path)

            # Load trained model if model_path is provided
            if self.model_path and hasattr(router, "load_router"):
                try:
                    router.load_router(self.model_path)
                    verbose_proxy_logger.info(
                        f"Loaded trained model from: {self.model_path}"
                    )
                except Exception as e:
                    verbose_proxy_logger.warning(
                        f"Could not load trained model: {e}. Using untrained router."
                    )

            return router

        except ImportError as e:
            verbose_proxy_logger.error(f"Failed to import LLMRouter: {e}")
            return None
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load router: {e}")
            return None

    def _load_inference_knn_router(self) -> Optional[InferenceKNNRouter]:
        """
        Load inference-only KNN router that bypasses UIUC MetaRouter.

        This avoids the 'hparam' / NoneType.loc errors that occur when
        UIUC's KNNRouter tries to load training data that doesn't exist
        in the gateway container.

        Returns:
            InferenceKNNRouter instance, or None if loading fails
        """
        if not self.model_path:
            verbose_proxy_logger.error(
                "model_path is required for inference-only KNN router. "
                "Set routing_strategy_args.model_path in config."
            )
            return None

        try:
            router = InferenceKNNRouter(
                model_path=self.model_path,
                embedding_model=self.embedding_model,
                embedding_device=self.embedding_device,
                label_mapping=self.label_mapping,
            )
            verbose_proxy_logger.info(
                f"Loaded inference-only KNN router from: {self.model_path}"
            )
            return router
        except FileNotFoundError as e:
            verbose_proxy_logger.warning(
                f"KNN model file not found: {e}. "
                "Ensure model is trained and deployed to model_path."
            )
            return None
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load inference-only KNN router: {e}")
            return None

    def _load_custom_router(self):
        """Load a custom router from the custom routers directory."""
        custom_path = os.environ.get(
            "LLMROUTER_CUSTOM_ROUTERS_PATH", "/app/custom_routers"
        )
        # Implementation for custom router loading
        verbose_proxy_logger.info(f"Loading custom router from: {custom_path}")
        return None

    @property
    def router(self):
        """Get the router instance, loading/reloading as needed."""
        with self._router_lock:
            if self._router is None or self._should_reload():
                # For inference-only KNN, check if we need to reload the model
                if (
                    self._router is not None
                    and isinstance(self._router, InferenceKNNRouter)
                    and self._should_reload()
                ):
                    verbose_proxy_logger.info(
                        "Hot reloading KNN model due to file change"
                    )
                    self._router.reload_model()
                else:
                    self._router = self._load_router()

                self._last_load_time = time.time()
                if self.model_path:
                    try:
                        self._model_mtime = Path(self.model_path).stat().st_mtime
                    except OSError:
                        pass
        return self._router

    def route_with_observability(
        self, query: str, model_list: list[str]
    ) -> Optional[str]:
        """
        Route a query with OpenTelemetry observability.

        Emits a versioned RouterDecisionEvent (routeiq.router_decision.v1)
        as a span event for downstream MLOps consumption.

        Args:
            query: User query to route
            model_list: List of available models

        Returns:
            Selected model name or None
        """
        start_time = time.time()
        selected_model = None
        error_info = None

        # Create span if OpenTelemetry is available
        if OTEL_AVAILABLE:
            try:
                from litellm_llmrouter.observability import get_observability_manager

                obs_manager = get_observability_manager()
                if obs_manager:
                    span = obs_manager.create_routing_span(
                        self.strategy_name, len(model_list)
                    )

                    with trace.use_span(span, end_on_exit=True):
                        # Perform routing
                        try:
                            if self.router and hasattr(self.router, "route"):
                                selected_model = self.router.route(query)
                        except Exception as e:
                            error_info = (type(e).__name__, str(e))
                            verbose_proxy_logger.error(f"Routing error: {e}")

                        # Calculate latency
                        latency_ms = (time.time() - start_time) * 1000

                        # Determine outcome and reason for TG4.1 span attributes
                        if error_info:
                            outcome = RoutingOutcome.ERROR.value
                            reason = f"strategy_error: {error_info[0]}"
                        elif selected_model:
                            outcome = RoutingOutcome.SUCCESS.value
                            reason = "strategy_prediction"
                        else:
                            outcome = (
                                RoutingOutcome.NO_CANDIDATES.value
                                if not model_list
                                else RoutingOutcome.FAILURE.value
                            )
                            reason = (
                                "no_candidates_available"
                                if not model_list
                                else "no_prediction"
                            )

                        # Get strategy version from router if available
                        strategy_version = None
                        if (
                            self.router
                            and hasattr(self.router, "model_version")
                            and self.router.model_version
                        ):
                            strategy_version = (
                                self.router.model_version.sha256[:16]
                                if hasattr(self.router.model_version, "sha256")
                                else str(self.router.model_version)
                            )

                        # Set TG4.1 router decision span attributes
                        set_router_decision_attributes(
                            span,
                            strategy=self.strategy_name,
                            model_selected=selected_model,
                            candidates_evaluated=len(model_list),
                            outcome=outcome,
                            reason=reason,
                            latency_ms=latency_ms,
                            error_type=error_info[0] if error_info else None,
                            error_message=error_info[1] if error_info else None,
                            strategy_version=strategy_version,
                            fallback_triggered=False,
                        )

                        # Build versioned telemetry event
                        event_builder = (
                            RouterDecisionEventBuilder()
                            .with_strategy(
                                name=self.strategy_name,
                                version=getattr(self, "model_version", None),
                            )
                            .with_input(
                                query_length=len(query),
                                # No PII: don't log query content
                            )
                            .with_candidates(
                                [
                                    {"model_name": m, "available": True}
                                    for m in model_list
                                ]
                            )
                            .with_selection(
                                selected=selected_model,
                                reason=(
                                    "strategy_prediction"
                                    if selected_model
                                    else "no_prediction"
                                ),
                            )
                            .with_timing(total_ms=latency_ms)
                        )

                        # Add trace context to event
                        span_context = span.get_span_context()
                        if span_context.is_valid:
                            event_builder.with_trace_context(
                                trace_id=format(span_context.trace_id, "032x"),
                                span_id=format(span_context.span_id, "016x"),
                            )

                        # Set outcome based on routing result
                        if error_info:
                            event_builder.with_outcome(
                                status=RoutingOutcome.ERROR,
                                error_type=error_info[0],
                                error_message=error_info[1],
                            )
                        elif selected_model:
                            event_builder.with_outcome(status=RoutingOutcome.SUCCESS)
                        else:
                            event_builder.with_outcome(
                                status=(
                                    RoutingOutcome.NO_CANDIDATES
                                    if not model_list
                                    else RoutingOutcome.FAILURE
                                )
                            )

                        # Build and emit the event
                        router_event = event_builder.build()

                        # Emit as span event with JSON payload
                        span.add_event(
                            name=ROUTER_DECISION_EVENT_NAME,
                            attributes={
                                ROUTER_DECISION_PAYLOAD_KEY: router_event.to_json(),
                            },
                        )

                        # Log routing decision
                        obs_manager.log_routing_decision(
                            strategy=self.strategy_name,
                            selected_model=selected_model or "none",
                            latency_ms=latency_ms,
                        )

                    return selected_model
            except Exception as e:
                verbose_proxy_logger.warning(f"Observability error: {e}")

        # Fallback: route without observability
        try:
            if self.router and hasattr(self.router, "route"):
                selected_model = self.router.route(query)
        except Exception as e:
            verbose_proxy_logger.error(f"Routing error (no observability): {e}")

        return selected_model


class CostAwareRoutingStrategy(RoutingStrategy):
    """
    Selects the cheapest model that meets a quality threshold.

    Enhanced with:
    - Inner strategy delegation for quality prediction (configurable)
    - Cost-quality Pareto frontier computation
    - Cached cost database with configurable refresh interval
    - Provider fallback when circuit breaker is open

    Algorithm:
    1. Filter out providers with open circuit breakers
    2. For each candidate deployment, look up model cost from cached cost DB
    3. Optionally predict quality score using an inner/delegate strategy
    4. Compute Pareto frontier of cost-quality trade-offs
    5. Filter candidates meeting the quality threshold
    6. Select cheapest from filtered set (using combined score)
    7. If no candidates meet threshold, fall back to best-quality selection

    Configuration:
        quality_threshold: Minimum acceptable quality score (0.0-1.0, default 0.7)
        cost_weight: How much to weight cost vs quality (0.0=quality only,
                     1.0=cost only, default 0.7)
        quality_weight: Inverse weight of cost_weight (auto-derived, default 0.3)
        inner_strategy: Inner strategy instance for quality prediction (optional)
        inner_strategy_name: Name of inner strategy for lazy resolution (optional)
        max_cost_per_1k_tokens: Hard cap on per-request cost (optional)
        cost_refresh_interval: Seconds between cost DB refreshes (default 3600)
        enable_circuit_breaker_filtering: Enable provider fallback on CB open
            (default True)
    """

    def __init__(
        self,
        quality_threshold: float = 0.7,
        cost_weight: float = 0.7,
        inner_strategy: Optional[RoutingStrategy] = None,
        max_cost_per_1k_tokens: Optional[float] = None,
        *,
        inner_strategy_name: Optional[str] = None,
        quality_weight: Optional[float] = None,
        cost_refresh_interval: int = 3600,
        enable_circuit_breaker_filtering: bool = True,
        **kwargs: Any,
    ):
        self._quality_threshold = max(0.0, min(1.0, quality_threshold))
        self._cost_weight = max(0.0, min(1.0, cost_weight))
        # quality_weight defaults to (1 - cost_weight) unless explicitly given
        if quality_weight is not None:
            self._quality_weight = max(0.0, min(1.0, quality_weight))
        else:
            self._quality_weight = 1.0 - self._cost_weight
        self._inner_strategy = inner_strategy
        self._inner_strategy_name = inner_strategy_name
        self._max_cost_per_1k_tokens = max_cost_per_1k_tokens

        # Cost database cache
        self._cost_db: Dict[str, float] = {}
        self._cost_refresh_interval = max(0, cost_refresh_interval)
        self._last_cost_refresh: float = 0.0

        # Provider fallback
        self._enable_circuit_breaker_filtering = enable_circuit_breaker_filtering

    @property
    def name(self) -> str:
        return "llmrouter-cost-aware"

    @property
    def version(self) -> Optional[str]:
        return "2.0.0"

    # ------------------------------------------------------------------
    # Inner strategy delegation
    # ------------------------------------------------------------------

    def _resolve_inner_strategy(self) -> Optional[RoutingStrategy]:
        """Lazily resolve the inner strategy by name from the registry.

        If an inner strategy instance is already set, returns it directly.
        Otherwise, looks up ``inner_strategy_name`` in the routing registry.

        Returns:
            The inner strategy instance, or None if not configured/found.
        """
        if self._inner_strategy is not None:
            return self._inner_strategy

        if self._inner_strategy_name:
            try:
                from litellm_llmrouter.strategy_registry import get_routing_registry

                registry = get_routing_registry()
                resolved = registry.get(self._inner_strategy_name)
                if resolved is not None:
                    self._inner_strategy = resolved
                    return resolved
            except Exception:
                pass

        return None

    # ------------------------------------------------------------------
    # Cost database
    # ------------------------------------------------------------------

    def _refresh_cost_db(self) -> None:
        """Refresh cost data from provider pricing.

        Uses ``litellm.model_cost`` as the source of truth.  The cost DB
        is cached for ``cost_refresh_interval`` seconds to avoid excessive
        lookups.
        """
        now = time.time()
        if now - self._last_cost_refresh < self._cost_refresh_interval:
            return
        try:
            import litellm

            if hasattr(litellm, "model_cost") and litellm.model_cost:
                for model, info in litellm.model_cost.items():
                    if isinstance(info, dict):
                        input_cost = info.get("input_cost_per_token", 0) or 0
                        output_cost = info.get("output_cost_per_token", 0) or 0
                        self._cost_db[model] = input_cost + output_cost
            self._last_cost_refresh = now
        except Exception:
            pass

    def _get_model_cost(self, model: str) -> float:
        """Get average cost per 1K tokens for a model.

        First checks the cached cost DB (refreshing if stale), then
        falls back to a direct ``litellm.model_cost`` lookup.  Returns
        ``inf`` for unknown models so they sort last.

        Args:
            model: Model identifier (e.g., 'gpt-4', 'claude-3-opus')

        Returns:
            Average cost per 1K tokens in USD
        """
        # Try cached cost DB first
        self._refresh_cost_db()
        if model in self._cost_db:
            # _cost_db stores raw per-token cost (input+output)
            raw = self._cost_db[model]
            per_1k = raw * 1000 / 2  # Average of input+output, per 1K
            return per_1k if per_1k > 0 else float("inf")

        # Fallback: direct litellm lookup
        try:
            import litellm

            cost_info = litellm.model_cost.get(model, {})
            input_cost = cost_info.get("input_cost_per_token", 0) * 1000
            output_cost = cost_info.get("output_cost_per_token", 0) * 1000
            avg_cost = (input_cost + output_cost) / 2
            return avg_cost if avg_cost > 0 else float("inf")
        except Exception:
            return float("inf")

    # ------------------------------------------------------------------
    # Quality prediction
    # ------------------------------------------------------------------

    def _predict_quality(
        self,
        context: RoutingContext,
        deployment: Dict,
    ) -> float:
        """Predict quality score for a deployment.

        If an inner strategy is configured (either directly or via name),
        delegates to it and returns 1.0 if it selects this deployment,
        0.5 otherwise.  Without an inner strategy, returns 1.0 for all
        candidates (effectively making this a pure cost optimizer).

        Args:
            context: Routing context with request details
            deployment: Candidate deployment dict

        Returns:
            Quality score between 0.0 and 1.0
        """
        inner = self._resolve_inner_strategy()
        if inner is None:
            return 1.0

        try:
            selected = inner.select_deployment(context)
            if selected is None:
                return 0.5
            selected_model = selected.get("litellm_params", {}).get("model", "")
            candidate_model = deployment.get("litellm_params", {}).get("model", "")
            return 1.0 if selected_model == candidate_model else 0.5
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------

    def _get_candidates(self, context: RoutingContext) -> List[Dict]:
        """Get candidate deployments from the router.

        Args:
            context: Routing context with router and model info

        Returns:
            List of deployment dicts matching the requested model
        """
        router = context.router
        healthy = getattr(router, "healthy_deployments", router.model_list)
        return [dep for dep in healthy if dep.get("model_name") == context.model]

    def _get_available_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Filter out candidates whose provider circuit breaker is open.

        When ``enable_circuit_breaker_filtering`` is True, providers with
        an open circuit breaker are excluded.  If all candidates are
        excluded, falls back to the full list (no worse than not filtering).

        Args:
            candidates: List of candidate deployments

        Returns:
            Filtered list of available candidates
        """
        if not self._enable_circuit_breaker_filtering:
            return candidates

        try:
            from litellm_llmrouter.resilience import get_circuit_breaker_manager

            cb_manager = get_circuit_breaker_manager()
            available: List[Dict] = []
            for candidate in candidates:
                provider = candidate.get("litellm_params", {}).get(
                    "custom_llm_provider", ""
                )
                if provider:
                    breaker = cb_manager.get_breaker(provider)
                    if not breaker.is_open:
                        available.append(candidate)
                else:
                    # No provider info -- include by default
                    available.append(candidate)
            # Fallback to all if every provider is down
            return available if available else candidates
        except Exception:
            return candidates

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    def _compute_pareto_optimal(
        self,
        candidates: List[Dict],
        quality_scores: Dict[str, float],
    ) -> List[Dict]:
        """Filter candidates to the cost-quality Pareto frontier.

        A candidate is *dominated* if another candidate is both cheaper
        **and** higher quality.  Candidates that are not dominated form
        the Pareto frontier.

        Args:
            candidates: List of candidate deployments
            quality_scores: Mapping of model name -> quality score

        Returns:
            List of non-dominated candidates (Pareto optimal set)
        """
        if len(candidates) <= 1:
            return candidates

        # Build (candidate, cost, quality) tuples
        scored: List[Tuple[Dict, float, float]] = []
        for dep in candidates:
            model = dep.get("litellm_params", {}).get("model", "")
            cost = self._get_model_cost(model)
            quality = quality_scores.get(model, 0.5)
            scored.append((dep, cost, quality))

        pareto: List[Dict] = []
        for i, (dep_i, cost_i, qual_i) in enumerate(scored):
            dominated = False
            for j, (dep_j, cost_j, qual_j) in enumerate(scored):
                if i == j:
                    continue
                # j dominates i if j is cheaper (or equal) AND higher quality
                # (or equal) with at least one strict inequality
                if cost_j <= cost_i and qual_j >= qual_i:
                    if cost_j < cost_i or qual_j > qual_i:
                        dominated = True
                        break
            if not dominated:
                pareto.append(dep_i)

        return pareto if pareto else candidates

    # ------------------------------------------------------------------
    # Combined scoring
    # ------------------------------------------------------------------

    def _compute_combined_score(
        self,
        quality: float,
        normalized_cost: float,
    ) -> float:
        """Compute combined quality-cost score.

        score = (1 - cost_weight) * quality + cost_weight * (1 - normalized_cost)

        Higher score is better. When cost_weight=1.0, only cost matters.
        When cost_weight=0.0, only quality matters.

        Args:
            quality: Quality score (0.0-1.0, higher is better)
            normalized_cost: Normalized cost (0.0-1.0, lower is cheaper)

        Returns:
            Combined score (higher is better)
        """
        return (1 - self._cost_weight) * quality + self._cost_weight * (
            1 - normalized_cost
        )

    # ------------------------------------------------------------------
    # Main routing logic
    # ------------------------------------------------------------------

    def select_deployment(
        self,
        context: RoutingContext,
    ) -> Optional[Dict]:
        """Select the cheapest deployment meeting the quality threshold.

        Enhanced algorithm:
        1. Get candidates matching the requested model
        2. Filter out providers with open circuit breakers
        3. Score each candidate for cost and quality
        4. Apply hard cost cap (max_cost_per_1k_tokens)
        5. Compute Pareto frontier
        6. Filter by quality threshold
        7. Select best by combined cost-quality score

        Args:
            context: Routing context with request details

        Returns:
            Selected deployment dict, or None if no candidates
        """
        candidates = self._get_candidates(context)
        if not candidates:
            return None

        # Filter by circuit breaker status
        candidates = self._get_available_candidates(candidates)

        if len(candidates) == 1:
            return candidates[0]

        # Score each candidate: (deployment, cost_per_1k, quality)
        scored: List[Tuple[Dict, float, float]] = []
        quality_scores: Dict[str, float] = {}
        for deployment in candidates:
            model = deployment.get("litellm_params", {}).get("model", "")
            cost_per_1k = self._get_model_cost(model)
            quality = self._predict_quality(context, deployment)
            quality_scores[model] = quality

            # Apply hard cost cap
            if (
                self._max_cost_per_1k_tokens is not None
                and cost_per_1k > self._max_cost_per_1k_tokens
            ):
                continue

            scored.append((deployment, cost_per_1k, quality))

        if not scored:
            # All candidates exceeded cost cap; fall back to best quality
            return self._select_best_quality(candidates, context)

        # Filter to candidates meeting quality threshold
        above_threshold = [
            (dep, cost, qual)
            for dep, cost, qual in scored
            if qual >= self._quality_threshold
        ]

        if not above_threshold:
            # No candidate meets quality threshold; fall back to best quality
            return self._select_best_quality(candidates, context)

        # Compute Pareto frontier among threshold-passing candidates
        threshold_deps = [dep for dep, _, _ in above_threshold]
        pareto_deps = self._compute_pareto_optimal(threshold_deps, quality_scores)

        # Rebuild scored list with only Pareto-optimal candidates
        pareto_ids = {id(dep) for dep in pareto_deps}
        pareto_scored = [
            (dep, cost, qual)
            for dep, cost, qual in above_threshold
            if id(dep) in pareto_ids
        ]
        # If Pareto filtering left nothing (shouldn't happen), use all
        if not pareto_scored:
            pareto_scored = above_threshold

        # Normalize costs for combined scoring
        costs = [cost for _, cost, _ in pareto_scored]
        min_cost = min(costs)
        max_cost = max(costs)
        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0

        best_deployment = None
        best_score = -1.0
        best_cost_per_1k: Optional[float] = None

        for dep, cost, qual in pareto_scored:
            normalized_cost = (cost - min_cost) / cost_range if cost_range > 0 else 0.0
            combined = self._compute_combined_score(qual, normalized_cost)
            if combined > best_score:
                best_score = combined
                best_deployment = dep
                best_cost_per_1k = cost

        # Emit cost-per-1K-tokens metric for observability
        if best_deployment is not None and best_cost_per_1k is not None:
            histogram = _get_cost_histogram()
            if histogram and best_cost_per_1k != float("inf"):
                selected_model = best_deployment.get("litellm_params", {}).get(
                    "model", "unknown"
                )
                histogram.record(
                    best_cost_per_1k,
                    {"model": selected_model, "strategy": "cost-aware"},
                )

        return best_deployment

    def _select_best_quality(
        self,
        candidates: List[Dict],
        context: RoutingContext,
    ) -> Optional[Dict]:
        """Fall back to selecting the best-quality candidate.

        If an inner strategy is configured, delegates to it.
        Otherwise, returns the first candidate.

        Args:
            candidates: List of candidate deployments
            context: Routing context

        Returns:
            Best quality deployment, or first candidate as fallback
        """
        inner = self._resolve_inner_strategy()
        if inner is not None:
            try:
                selected = inner.select_deployment(context)
                if selected is not None:
                    return selected
            except Exception:
                pass

        return candidates[0] if candidates else None

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate the strategy is ready to serve requests."""
        if self._quality_threshold < 0 or self._quality_threshold > 1:
            return False, "quality_threshold must be between 0.0 and 1.0"
        if self._cost_weight < 0 or self._cost_weight > 1:
            return False, "cost_weight must be between 0.0 and 1.0"
        return True, None


def register_llmrouter_strategies():
    """
    Register LLMRouter strategies with LiteLLM's routing infrastructure.

    This function should be called during startup to make LLMRouter
    strategies available for use in LiteLLM routing configurations.

    Returns:
        List of registered strategy names
    """
    verbose_proxy_logger.info(
        f"Registering {len(LLMROUTER_STRATEGIES)} LLMRouter strategies"
    )

    # Log available strategies
    for strategy in LLMROUTER_STRATEGIES:
        verbose_proxy_logger.debug(f"  - {strategy}")

    return LLMROUTER_STRATEGIES
