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
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import tempfile
import yaml
from litellm._logging import verbose_proxy_logger

try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Import model artifact verification
from litellm_llmrouter.model_artifacts import (
    get_artifact_verifier,
    ModelVerificationError,
    ActiveModelVersion,
)

# Import telemetry contracts for versioned event emission
from litellm_llmrouter.telemetry_contracts import (
    RouterDecisionEventBuilder,
    RoutingOutcome,
    ROUTER_DECISION_EVENT_NAME,
    ROUTER_DECISION_PAYLOAD_KEY,
)

try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

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
ENFORCE_SIGNED_MODELS = (
    os.getenv("LLMROUTER_ENFORCE_SIGNED_MODELS", "").lower() == "true"
    or (ALLOW_PICKLE_MODELS and os.getenv("LLMROUTER_ENFORCE_SIGNED_MODELS", "").lower() != "false")
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

    def __init__(self, model_path: str, reason: str, correlation_id: Optional[str] = None):
        self.model_path = model_path
        self.reason = reason
        self.correlation_id = correlation_id
        super().__init__(
            f"Model loading failed for '{model_path}': {reason}"
        )


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

        verbose_proxy_logger.info(f"{log_prefix}Loading KNN model from: {self.model_path}")

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

                        # Add result to span attributes
                        if selected_model:
                            span.set_attribute(
                                "llm.routing.selected_model", selected_model
                            )
                        span.set_attribute("llm.routing.latency_ms", latency_ms)

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
                                [{"model_name": m, "available": True} for m in model_list]
                            )
                            .with_selection(
                                selected=selected_model,
                                reason="strategy_prediction" if selected_model else "no_prediction",
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
                                status=RoutingOutcome.NO_CANDIDATES
                                if not model_list
                                else RoutingOutcome.FAILURE
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
