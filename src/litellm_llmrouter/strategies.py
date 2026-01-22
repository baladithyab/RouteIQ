"""
LLMRouter Routing Strategies for LiteLLM
==========================================

This module implements the integration between LLMRouter's ML-based
routing strategies and LiteLLM's routing infrastructure.
"""

import json
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import tempfile
import yaml
from litellm._logging import verbose_proxy_logger

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

    Attributes:
        model_path: Path to the trained .pkl model file
        embedding_model: Name of the sentence-transformer model
        embedding_device: Device for embedding model ('cpu', 'cuda')
        knn_model: Loaded sklearn KNeighborsClassifier
        label_mapping: Optional mapping from predicted labels to LLM candidate keys
    """

    def __init__(
        self,
        model_path: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_device: str = "cpu",
        label_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize inference-only KNN router.

        Args:
            model_path: Path to the trained sklearn KNN model (.pkl file)
            embedding_model: HuggingFace model name for sentence embeddings
            embedding_device: Device for embedding model ('cpu', 'cuda', etc.)
            label_mapping: Optional dict mapping predicted labels to LLM keys
        """
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.label_mapping = label_mapping or {}
        self.knn_model = None

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the sklearn KNN model from pickle file."""
        if not self.model_path:
            raise ValueError("model_path is required for InferenceKNNRouter")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"KNN model file not found: {self.model_path}")

        verbose_proxy_logger.info(f"Loading KNN model from: {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.knn_model = pickle.load(f)

        # Verify it's a sklearn model with predict method
        if not hasattr(self.knn_model, "predict"):
            raise TypeError(
                f"Loaded model does not have 'predict' method. "
                f"Expected sklearn KNeighborsClassifier, got {type(self.knn_model)}"
            )

        verbose_proxy_logger.info(
            f"KNN model loaded successfully. Type: {type(self.knn_model).__name__}"
        )

    def reload_model(self):
        """Reload the model from disk (for hot reload support)."""
        self._load_model()

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

            verbose_proxy_logger.debug(
                f"KNN routing: query='{query[:50]}...' -> predicted={predicted_label}"
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

        Args:
            query: User query to route
            model_list: List of available models

        Returns:
            Selected model name or None
        """
        start_time = time.time()
        selected_model = None

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
                        if self.router and hasattr(self.router, "route"):
                            selected_model = self.router.route(query)

                        # Add result to span
                        if selected_model:
                            span.set_attribute(
                                "llm.routing.selected_model", selected_model
                            )

                        latency_ms = (time.time() - start_time) * 1000
                        span.set_attribute("llm.routing.latency_ms", latency_ms)

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
        if self.router and hasattr(self.router, "route"):
            selected_model = self.router.route(query)

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
