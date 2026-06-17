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
import math
import os
import pickle
import threading
import time
from collections import deque
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


def _record_selection_metric(strategy: str, model: str) -> None:
    """Emit the routing.selection metric for an ML routing strategy.

    Best-effort: a missing meter (OTel disabled) is a no-op and any recording
    error is swallowed. Telemetry must never raise.
    """
    try:
        from litellm_llmrouter.metrics import get_gateway_metrics

        m = get_gateway_metrics()
        if m is not None:
            m.record_routing_selection(strategy, model)
    except Exception:  # pragma: no cover - telemetry must not break flow
        pass


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


class InferenceSVMRouter:
    """
    Lightweight inference-only SVM router that loads sklearn SVC models directly.

    This class bypasses the UIUC LLMRouter's MetaRouter initialization which
    requires training data. Instead, it:
    - Loads a pre-trained sklearn SVC/SVM from a .pkl file
    - Uses sentence-transformers for text embedding (same as training)
    - Predicts the best model label for a given query

    The trained .pkl file is produced by UIUC's SVMRouter trainer which calls
    sklearn's SVC.fit() and saves via pickle.

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
        svm_model: Loaded sklearn SVC model
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
        Initialize inference-only SVM router.

        Args:
            model_path: Path to the trained sklearn SVM model (.pkl file)
            embedding_model: HuggingFace model name for sentence embeddings
            embedding_device: Device for embedding model ('cpu', 'cuda', etc.)
            label_mapping: Optional dict mapping predicted labels to LLM keys
            correlation_id: Optional correlation ID for logging
        """
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.label_mapping = label_mapping or {}
        self.svm_model = None
        self.model_version: Optional[ActiveModelVersion] = None
        self._model_lock = threading.RLock()

        # Load the model with verification
        self._load_model(correlation_id=correlation_id)

    def _load_model(self, correlation_id: Optional[str] = None):
        """Load the sklearn SVM model from pickle file with verification.

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
            raise ValueError("model_path is required for InferenceSVMRouter")

        # Security check: pickle loading disabled by default
        if not ALLOW_PICKLE_MODELS:
            raise PickleSecurityError(self.model_path)

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"SVM model file not found: {self.model_path}")

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
            manifest = verifier._load_manifest()
            if manifest is None:
                raise PickleSignatureRequiredError(self.model_path, strict_mode=True)

            if manifest.signature_type == SignatureType.NONE:
                raise PickleSignatureRequiredError(self.model_path, strict_mode=True)

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
            f"{log_prefix}Loading SVM model from: {self.model_path}"
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
                f"Expected sklearn SVC, got {type(new_model)}",
                correlation_id,
            )

        # Safe swap: only update if everything succeeded
        with self._model_lock:
            old_model = self.svm_model
            self.svm_model = new_model

            # Record active version for observability
            self.model_version = verifier.record_active_version(
                self.model_path,
                sha256=computed_hash,
                tags=["svm", "active"],
            )

        verbose_proxy_logger.info(
            f"{log_prefix}SVM model loaded successfully. Type: {type(self.svm_model).__name__}, "
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
                f"{log_prefix}SVM model reloaded successfully. "
                f"Old version: {old_version.sha256[:16] if old_version else 'none'}..., "
                f"New version: {self.model_version.sha256[:16] if self.model_version else 'none'}..."
            )
            return True
        except (ModelVerificationError, ModelLoadError, FileNotFoundError) as e:
            verbose_proxy_logger.error(
                f"{log_prefix}SVM model reload failed, keeping old model active. Error: {e}"
            )
            return False
        except Exception as e:
            verbose_proxy_logger.error(
                f"{log_prefix}Unexpected error during SVM model reload, keeping old model active. Error: {e}"
            )
            return False

    def route(self, query: str) -> Optional[str]:
        """
        Route a query to the best model using SVM prediction.

        Args:
            query: User query text to route

        Returns:
            Predicted model label/key, or None if prediction fails
        """
        if self.svm_model is None:
            verbose_proxy_logger.warning("SVM model not loaded, cannot route")
            return None

        try:
            # Get embedding using the same model used in training
            embedder = _get_sentence_transformer(
                self.embedding_model, self.embedding_device
            )

            # Encode the query to get embedding vector
            # Shape: (embedding_dim,) -> need (1, embedding_dim) for predict
            embedding = embedder.encode([query], convert_to_numpy=True)

            # Predict using the SVM model
            predicted_label = self.svm_model.predict(embedding)[0]

            # Security: Log only query length and prediction, not query content (PII risk)
            verbose_proxy_logger.debug(
                f"SVM routing: query_length={len(query)} -> predicted={predicted_label}"
            )

            # Apply label mapping if configured
            if self.label_mapping and predicted_label in self.label_mapping:
                mapped_label = self.label_mapping[predicted_label]
                verbose_proxy_logger.debug(
                    f"SVM label mapping: {predicted_label} -> {mapped_label}"
                )
                return mapped_label

            return str(predicted_label)

        except Exception as e:
            verbose_proxy_logger.error(f"SVM routing error: {e}")
            return None


class InferenceMLPRouter:
    """
    Lightweight inference-only MLP router that loads PyTorch state_dict directly.

    This class bypasses the UIUC LLMRouter's MetaRouter initialization which
    requires training data. Instead, it:
    - Loads a pre-trained PyTorch MLP model from a state_dict file
    - Loads model metadata (class mappings, architecture params) from a companion JSON
    - Uses sentence-transformers for text embedding (same as training)
    - Predicts the best model label for a given query via forward pass + argmax

    The trained model is produced by UIUC's MLPRouter trainer using PyTorch's
    MLPClassifierNN (Linear layers + activation) and saved via torch.save(state_dict).

    Security:
    - Uses torch.load(weights_only=True) which is safer than pickle
    - Does NOT require LLMROUTER_ALLOW_PICKLE_MODELS since it avoids pickle
    - Artifact verification via manifest is still supported
    - The companion metadata JSON file must be present alongside the model

    Safe Activation:
    - Models are loaded into a temporary instance first
    - Only swapped to active if loading succeeds
    - On failure, the old model remains active

    Attributes:
        model_path: Path to the trained .pt/.pth model state_dict file
        metadata_path: Path to companion JSON with class mappings and architecture
        embedding_model: Name of the sentence-transformer model
        embedding_device: Device for embedding model ('cpu', 'cuda')
        mlp_model: Loaded PyTorch MLPClassifierNN model
        idx_to_model: Mapping from class index to model name
        model_to_idx: Mapping from model name to class index
        model_version: Active model version metadata for observability
    """

    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_device: str = "cpu",
        label_mapping: Optional[Dict[str, str]] = None,
        input_dim: int = 768,
        hidden_layer_sizes: Optional[List[int]] = None,
        num_classes: Optional[int] = None,
        activation: str = "relu",
        idx_to_model: Optional[Dict[int, str]] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize inference-only MLP router.

        The model architecture can be specified in two ways:
        1. Via a companion metadata JSON file (metadata_path) containing all params
        2. Via explicit constructor args (input_dim, hidden_layer_sizes, num_classes, etc.)

        The metadata JSON format:
        {
            "input_dim": 768,
            "hidden_layer_sizes": [128, 64],
            "num_classes": 5,
            "activation": "relu",
            "idx_to_model": {"0": "gpt-4", "1": "claude-3-opus", ...},
            "model_to_idx": {"gpt-4": 0, "claude-3-opus": 1, ...}
        }

        Args:
            model_path: Path to the trained PyTorch model state_dict (.pt/.pth)
            metadata_path: Optional path to JSON with architecture + class mappings.
                           If None, defaults to model_path with .json extension.
            embedding_model: HuggingFace model name for sentence embeddings
            embedding_device: Device for embedding model ('cpu', 'cuda', etc.)
            label_mapping: Optional dict mapping predicted labels to LLM keys
            input_dim: Input dimension (embedding size), default 768 for all-MiniLM-L6-v2
            hidden_layer_sizes: MLP hidden layer sizes, default [128, 64]
            num_classes: Number of output classes (candidate models)
            activation: Activation function ('relu', 'tanh', 'logistic', 'identity')
            idx_to_model: Mapping from class index to model name
            correlation_id: Optional correlation ID for logging
        """
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.label_mapping = label_mapping or {}
        self.mlp_model = None
        self.model_version: Optional[ActiveModelVersion] = None
        self._model_lock = threading.RLock()

        # Architecture parameters (may be overridden by metadata)
        self._input_dim = input_dim
        self._hidden_layer_sizes = hidden_layer_sizes or [128, 64]
        self._num_classes = num_classes
        self._activation = activation

        # Class mappings (may be overridden by metadata)
        self.idx_to_model: Dict[int, str] = idx_to_model or {}
        self.model_to_idx: Dict[str, int] = {}

        # Load metadata if available
        self.metadata_path = metadata_path or str(Path(model_path).with_suffix(".json"))
        self._load_metadata()

        # Load the model
        self._load_model(correlation_id=correlation_id)

    def _load_metadata(self):
        """Load model metadata (architecture params + class mappings) from JSON.

        The metadata file is a companion to the model state_dict and contains
        all parameters needed to reconstruct the model architecture and map
        class indices back to model names.

        Falls back to constructor-provided values if metadata file is not found.
        """
        metadata_path = Path(self.metadata_path)
        if not metadata_path.exists():
            verbose_proxy_logger.info(
                f"MLP metadata file not found at {self.metadata_path}, "
                "using constructor-provided architecture params"
            )
            # Build reverse mapping from idx_to_model
            if self.idx_to_model:
                self.model_to_idx = {v: k for k, v in self.idx_to_model.items()}
            return

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self._input_dim = metadata.get("input_dim", self._input_dim)
            self._hidden_layer_sizes = metadata.get(
                "hidden_layer_sizes", self._hidden_layer_sizes
            )
            self._num_classes = metadata.get("num_classes", self._num_classes)
            self._activation = metadata.get("activation", self._activation)

            # Load class mappings (JSON keys are strings, convert to int)
            raw_idx_to_model = metadata.get("idx_to_model", {})
            if raw_idx_to_model:
                self.idx_to_model = {int(k): v for k, v in raw_idx_to_model.items()}
                self.model_to_idx = {v: k for k, v in self.idx_to_model.items()}
            elif metadata.get("model_to_idx"):
                self.model_to_idx = metadata["model_to_idx"]
                self.idx_to_model = {v: k for k, v in self.model_to_idx.items()}

            # Infer num_classes from mappings if not explicitly set
            if self._num_classes is None and self.idx_to_model:
                self._num_classes = len(self.idx_to_model)

            verbose_proxy_logger.info(
                f"Loaded MLP metadata: input_dim={self._input_dim}, "
                f"hidden_layers={self._hidden_layer_sizes}, "
                f"num_classes={self._num_classes}, "
                f"activation={self._activation}"
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Failed to load MLP metadata from {self.metadata_path}: {e}. "
                "Using constructor-provided params."
            )

    def _load_model(self, correlation_id: Optional[str] = None):
        """Load the PyTorch MLP model from state_dict with verification.

        Security:
        - Uses torch.load(weights_only=True) to avoid arbitrary code execution.
        - This is SAFER than pickle loading since it only loads tensor data.
        - Artifact verification via manifest is still supported.

        Safe Activation:
        - Loads model into temporary variable first
        - Only swaps to active if successful
        - Records model version for observability
        """
        if not self.model_path:
            raise ValueError("model_path is required for InferenceMLPRouter")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"MLP model file not found: {self.model_path}")

        if self._num_classes is None:
            raise ValueError(
                "num_classes is required for InferenceMLPRouter. "
                "Provide via metadata JSON or constructor arg."
            )

        log_prefix = f"[{correlation_id}] " if correlation_id else ""

        # Verify artifact against manifest if enforcement is enabled
        verifier = get_artifact_verifier()
        computed_hash = verifier.compute_sha256(self.model_path)

        if ENFORCE_SIGNED_MODELS:
            try:
                verifier.verify_artifact(
                    self.model_path,
                    require_manifest=True,
                    correlation_id=correlation_id,
                )
            except ModelVerificationError as e:
                verbose_proxy_logger.error(
                    f"{log_prefix}MLP model verification failed: {e}. "
                    f"Hash mismatch or manifest missing. Details: {e.details}"
                )
                raise

        verbose_proxy_logger.info(
            f"{log_prefix}Loading MLP model from: {self.model_path}"
        )

        # Safe activation: load into temp variable first
        try:
            import torch

            # weights_only=True is safer than pickle — only loads tensor data
            state_dict = torch.load(
                self.model_path, map_location="cpu", weights_only=True
            )

            # Reconstruct the MLPClassifierNN architecture
            new_model = _build_mlp_classifier(
                input_dim=self._input_dim,
                hidden_layer_sizes=self._hidden_layer_sizes,
                num_classes=self._num_classes,
                activation=self._activation,
            )
            new_model.load_state_dict(state_dict)
            new_model.eval()

        except Exception as e:
            raise ModelLoadError(
                self.model_path,
                f"PyTorch model load failed: {e}",
                correlation_id,
            )

        # Safe swap: only update if everything succeeded
        with self._model_lock:
            old_model = self.mlp_model
            self.mlp_model = new_model

            self.model_version = verifier.record_active_version(
                self.model_path,
                sha256=computed_hash,
                tags=["mlp", "active"],
            )

        verbose_proxy_logger.info(
            f"{log_prefix}MLP model loaded successfully. "
            f"Architecture: {self._input_dim}->{self._hidden_layer_sizes}->{self._num_classes}, "
            f"Version SHA256: {self.model_version.sha256[:16]}..."
        )

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
            self._load_metadata()  # Reload metadata in case mappings changed
            self._load_model(correlation_id=correlation_id)
            verbose_proxy_logger.info(
                f"{log_prefix}MLP model reloaded successfully. "
                f"Old version: {old_version.sha256[:16] if old_version else 'none'}..., "
                f"New version: {self.model_version.sha256[:16] if self.model_version else 'none'}..."
            )
            return True
        except (ModelVerificationError, ModelLoadError, FileNotFoundError) as e:
            verbose_proxy_logger.error(
                f"{log_prefix}MLP model reload failed, keeping old model active. Error: {e}"
            )
            return False
        except Exception as e:
            verbose_proxy_logger.error(
                f"{log_prefix}Unexpected error during MLP model reload, keeping old model active. Error: {e}"
            )
            return False

    def route(self, query: str) -> Optional[str]:
        """
        Route a query to the best model using MLP forward pass + argmax.

        Args:
            query: User query text to route

        Returns:
            Predicted model label/key, or None if prediction fails
        """
        if self.mlp_model is None:
            verbose_proxy_logger.warning("MLP model not loaded, cannot route")
            return None

        try:
            import torch

            # Get embedding using the same model used in training
            embedder = _get_sentence_transformer(
                self.embedding_model, self.embedding_device
            )

            # Encode the query to get embedding vector
            embedding = embedder.encode([query], convert_to_numpy=True)

            # Convert to torch tensor and run forward pass
            with torch.no_grad():
                emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                logits = self.mlp_model(emb_tensor)
                predicted_idx = torch.argmax(logits, dim=-1).item()

            # Map index back to model name
            if self.idx_to_model:
                predicted_label = self.idx_to_model.get(
                    predicted_idx, str(predicted_idx)
                )
            else:
                predicted_label = str(predicted_idx)

            # Security: Log only query length and prediction, not query content (PII risk)
            verbose_proxy_logger.debug(
                f"MLP routing: query_length={len(query)} -> "
                f"predicted_idx={predicted_idx}, model={predicted_label}"
            )

            # Apply label mapping if configured
            if self.label_mapping and predicted_label in self.label_mapping:
                mapped_label = self.label_mapping[predicted_label]
                verbose_proxy_logger.debug(
                    f"MLP label mapping: {predicted_label} -> {mapped_label}"
                )
                return mapped_label

            return str(predicted_label)

        except Exception as e:
            verbose_proxy_logger.error(f"MLP routing error: {e}")
            return None


class InferenceMFRouter:
    """
    Lightweight inference-only Matrix Factorization router that loads PyTorch weights directly.

    This class bypasses the UIUC LLMRouter's MetaRouter initialization which
    requires training data. Instead, it:
    - Loads a pre-trained BilinearMF model from a state_dict file
    - Loads model metadata (class mappings, dimensions) from a companion JSON
    - Uses sentence-transformers for text embedding (same as training)
    - Scores all candidate models via bilinear interaction and picks the best

    The trained model is produced by UIUC's MFRouter trainer using the BilinearMF
    architecture: delta(M, q) = w2^T (v_m * (W1 * v_q)) and saved via torch.save(state_dict).

    Security:
    - Uses torch.load(weights_only=True) which is safer than pickle
    - Does NOT require LLMROUTER_ALLOW_PICKLE_MODELS since it avoids pickle
    - Artifact verification via manifest is still supported

    Safe Activation:
    - Models are loaded into a temporary instance first
    - Only swapped to active if loading succeeds
    - On failure, the old model remains active

    Attributes:
        model_path: Path to the trained .pt/.pth model state_dict file
        metadata_path: Path to companion JSON with class mappings and dimensions
        embedding_model: Name of the sentence-transformer model
        embedding_device: Device for embedding model ('cpu', 'cuda')
        mf_model: Loaded PyTorch BilinearMF model
        idx_to_model: Mapping from model index to model name
        model_version: Active model version metadata for observability
    """

    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_device: str = "cpu",
        label_mapping: Optional[Dict[str, str]] = None,
        latent_dim: int = 128,
        text_dim: int = 768,
        num_models: Optional[int] = None,
        idx_to_model: Optional[Dict[int, str]] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize inference-only MF router.

        The model architecture can be specified in two ways:
        1. Via a companion metadata JSON file (metadata_path) containing all params
        2. Via explicit constructor args (latent_dim, text_dim, num_models, etc.)

        The metadata JSON format:
        {
            "latent_dim": 128,
            "text_dim": 768,
            "num_models": 5,
            "idx_to_model": {"0": "gpt-4", "1": "claude-3-opus", ...},
            "model_to_idx": {"gpt-4": 0, "claude-3-opus": 1, ...}
        }

        Args:
            model_path: Path to the trained PyTorch model state_dict (.pt/.pth)
            metadata_path: Optional path to JSON with dimensions + class mappings.
                           If None, defaults to model_path with .json extension.
            embedding_model: HuggingFace model name for sentence embeddings
            embedding_device: Device for embedding model ('cpu', 'cuda', etc.)
            label_mapping: Optional dict mapping predicted labels to LLM keys
            latent_dim: Latent dimension for model embeddings (default 128)
            text_dim: Text embedding dimension (default 768 for all-MiniLM-L6-v2)
            num_models: Number of candidate models
            idx_to_model: Mapping from model index to model name
            correlation_id: Optional correlation ID for logging
        """
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.label_mapping = label_mapping or {}
        self.mf_model = None
        self.model_version: Optional[ActiveModelVersion] = None
        self._model_lock = threading.RLock()

        # Architecture parameters (may be overridden by metadata)
        self._latent_dim = latent_dim
        self._text_dim = text_dim
        self._num_models = num_models

        # Class mappings (may be overridden by metadata)
        self.idx_to_model: Dict[int, str] = idx_to_model or {}
        self.model_to_idx: Dict[str, int] = {}

        # Load metadata if available
        self.metadata_path = metadata_path or str(Path(model_path).with_suffix(".json"))
        self._load_metadata()

        # Load the model
        self._load_model(correlation_id=correlation_id)

    def _load_metadata(self):
        """Load model metadata (dimensions + class mappings) from JSON.

        The metadata file is a companion to the model state_dict and contains
        all parameters needed to reconstruct the BilinearMF architecture and map
        model indices back to model names.

        Falls back to constructor-provided values if metadata file is not found.
        """
        metadata_path = Path(self.metadata_path)
        if not metadata_path.exists():
            verbose_proxy_logger.info(
                f"MF metadata file not found at {self.metadata_path}, "
                "using constructor-provided architecture params"
            )
            if self.idx_to_model:
                self.model_to_idx = {v: k for k, v in self.idx_to_model.items()}
            return

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self._latent_dim = metadata.get("latent_dim", self._latent_dim)
            self._text_dim = metadata.get("text_dim", self._text_dim)
            self._num_models = metadata.get("num_models", self._num_models)

            # Load class mappings (JSON keys are strings, convert to int)
            raw_idx_to_model = metadata.get("idx_to_model", {})
            if raw_idx_to_model:
                self.idx_to_model = {int(k): v for k, v in raw_idx_to_model.items()}
                self.model_to_idx = {v: k for k, v in self.idx_to_model.items()}
            elif metadata.get("model_to_idx"):
                self.model_to_idx = metadata["model_to_idx"]
                self.idx_to_model = {v: k for k, v in self.model_to_idx.items()}

            # Infer num_models from mappings if not explicitly set
            if self._num_models is None and self.idx_to_model:
                self._num_models = len(self.idx_to_model)

            verbose_proxy_logger.info(
                f"Loaded MF metadata: latent_dim={self._latent_dim}, "
                f"text_dim={self._text_dim}, num_models={self._num_models}"
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Failed to load MF metadata from {self.metadata_path}: {e}. "
                "Using constructor-provided params."
            )

    def _load_model(self, correlation_id: Optional[str] = None):
        """Load the PyTorch BilinearMF model from state_dict with verification.

        Security:
        - Uses torch.load(weights_only=True) to avoid arbitrary code execution.
        - This is SAFER than pickle loading since it only loads tensor data.
        - Artifact verification via manifest is still supported.

        Safe Activation:
        - Loads model into temporary variable first
        - Only swaps to active if successful
        - Records model version for observability
        """
        if not self.model_path:
            raise ValueError("model_path is required for InferenceMFRouter")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"MF model file not found: {self.model_path}")

        if self._num_models is None:
            raise ValueError(
                "num_models is required for InferenceMFRouter. "
                "Provide via metadata JSON or constructor arg."
            )

        log_prefix = f"[{correlation_id}] " if correlation_id else ""

        # Verify artifact against manifest if enforcement is enabled
        verifier = get_artifact_verifier()
        computed_hash = verifier.compute_sha256(self.model_path)

        if ENFORCE_SIGNED_MODELS:
            try:
                verifier.verify_artifact(
                    self.model_path,
                    require_manifest=True,
                    correlation_id=correlation_id,
                )
            except ModelVerificationError as e:
                verbose_proxy_logger.error(
                    f"{log_prefix}MF model verification failed: {e}. "
                    f"Hash mismatch or manifest missing. Details: {e.details}"
                )
                raise

        verbose_proxy_logger.info(
            f"{log_prefix}Loading MF model from: {self.model_path}"
        )

        # Safe activation: load into temp variable first
        try:
            import torch

            # weights_only=True is safer than pickle — only loads tensor data
            state_dict = torch.load(
                self.model_path, map_location="cpu", weights_only=True
            )

            # Reconstruct the BilinearMF architecture
            new_model = _build_bilinear_mf(
                latent_dim=self._latent_dim,
                num_models=self._num_models,
                text_dim=self._text_dim,
            )
            new_model.load_state_dict(state_dict)
            new_model.eval()

        except Exception as e:
            raise ModelLoadError(
                self.model_path,
                f"PyTorch model load failed: {e}",
                correlation_id,
            )

        # Safe swap: only update if everything succeeded
        with self._model_lock:
            old_model = self.mf_model
            self.mf_model = new_model

            self.model_version = verifier.record_active_version(
                self.model_path,
                sha256=computed_hash,
                tags=["mf", "active"],
            )

        verbose_proxy_logger.info(
            f"{log_prefix}MF model loaded successfully. "
            f"Architecture: text_dim={self._text_dim}, latent_dim={self._latent_dim}, "
            f"num_models={self._num_models}, "
            f"Version SHA256: {self.model_version.sha256[:16]}..."
        )

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
            self._load_metadata()  # Reload metadata in case mappings changed
            self._load_model(correlation_id=correlation_id)
            verbose_proxy_logger.info(
                f"{log_prefix}MF model reloaded successfully. "
                f"Old version: {old_version.sha256[:16] if old_version else 'none'}..., "
                f"New version: {self.model_version.sha256[:16] if self.model_version else 'none'}..."
            )
            return True
        except (ModelVerificationError, ModelLoadError, FileNotFoundError) as e:
            verbose_proxy_logger.error(
                f"{log_prefix}MF model reload failed, keeping old model active. Error: {e}"
            )
            return False
        except Exception as e:
            verbose_proxy_logger.error(
                f"{log_prefix}Unexpected error during MF model reload, keeping old model active. Error: {e}"
            )
            return False

    def route(self, query: str) -> Optional[str]:
        """
        Route a query to the best model using bilinear MF scoring.

        The scoring function is: delta(M, q) = w2^T (v_m * (W1 * v_q))
        for all candidate models M, then argmax to pick the best.

        Args:
            query: User query text to route

        Returns:
            Predicted model label/key, or None if prediction fails
        """
        if self.mf_model is None:
            verbose_proxy_logger.warning("MF model not loaded, cannot route")
            return None

        try:
            import torch

            # Get embedding using the same model used in training
            embedder = _get_sentence_transformer(
                self.embedding_model, self.embedding_device
            )

            # Encode the query to get embedding vector
            embedding = embedder.encode([query], convert_to_numpy=True)

            # Convert to torch tensor, project to latent space, score all models
            with torch.no_grad():
                q_emb = torch.tensor(embedding[0], dtype=torch.float32)
                q_proj = self.mf_model.project_text(q_emb)
                scores = self.mf_model.score_all(q_proj)
                best_idx = torch.argmax(scores).item()

            # Map index back to model name
            if self.idx_to_model:
                predicted_label = self.idx_to_model.get(best_idx, str(best_idx))
            else:
                predicted_label = str(best_idx)

            # Security: Log only query length and prediction, not query content (PII risk)
            verbose_proxy_logger.debug(
                f"MF routing: query_length={len(query)} -> "
                f"best_idx={best_idx}, model={predicted_label}"
            )

            # Apply label mapping if configured
            if self.label_mapping and predicted_label in self.label_mapping:
                mapped_label = self.label_mapping[predicted_label]
                verbose_proxy_logger.debug(
                    f"MF label mapping: {predicted_label} -> {mapped_label}"
                )
                return mapped_label

            return str(predicted_label)

        except Exception as e:
            verbose_proxy_logger.error(f"MF routing error: {e}")
            return None


class InferenceELORouter:
    """
    Lightweight inference-only ELO router that loads pre-computed Elo ratings.

    This is the simplest inference adapter: Elo routing does not use query
    embeddings at all. It simply picks the model with the highest pre-computed
    Elo rating. The ratings are loaded from a JSON or pickle file containing
    a dictionary mapping model names to Elo scores.

    This class bypasses the UIUC LLMRouter's MetaRouter initialization which
    requires training data. Instead, it:
    - Loads pre-computed {model_name: elo_score} ratings from JSON or pickle
    - Returns the highest-rated model for every query

    The trained ratings are produced by UIUC's EloRouterTrainer which computes
    pairwise Elo ratings from model comparison data and saves them to disk.

    Security:
    - Prefers JSON format (no code execution risk)
    - For pickle format, requires LLMROUTER_ALLOW_PICKLE_MODELS=true
    - Artifact verification via manifest is supported for both formats

    Safe Activation:
    - Ratings are loaded into a temporary variable first
    - Only swapped to active if loading succeeds
    - On failure, the old ratings remain active

    Attributes:
        ratings_path: Path to the ratings file (.json or .pkl)
        elo_scores: Dictionary mapping model names to Elo scores
        model_version: Active model version metadata for observability
    """

    def __init__(
        self,
        ratings_path: str,
        label_mapping: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize inference-only ELO router.

        The ratings file can be in two formats:
        1. JSON: {"gpt-4": 1650, "claude-3-opus": 1580, "llama-3": 1420, ...}
        2. Pickle: Same dict structure, loaded via pickle.load()

        Args:
            ratings_path: Path to pre-computed Elo ratings (.json or .pkl)
            label_mapping: Optional dict mapping rating keys to LLM deployment keys
            correlation_id: Optional correlation ID for logging
        """
        self.ratings_path = ratings_path
        self.label_mapping = label_mapping or {}
        self.elo_scores: Optional[Dict[str, float]] = None
        self.model_version: Optional[ActiveModelVersion] = None
        self._model_lock = threading.RLock()

        # Load the ratings
        self._load_ratings(correlation_id=correlation_id)

    def _load_ratings(self, correlation_id: Optional[str] = None):
        """Load Elo ratings from JSON or pickle file with verification.

        Security:
        - JSON is preferred (no code execution risk).
        - Pickle requires LLMROUTER_ALLOW_PICKLE_MODELS=true.
        - Artifact verification via manifest is supported for both.

        Safe Activation:
        - Loads ratings into temporary variable first
        - Only swaps to active if successful
        - Records model version for observability
        """
        if not self.ratings_path:
            raise ValueError("ratings_path is required for InferenceELORouter")

        path = Path(self.ratings_path)
        if not path.exists():
            raise FileNotFoundError(f"ELO ratings file not found: {self.ratings_path}")

        log_prefix = f"[{correlation_id}] " if correlation_id else ""

        # Verify artifact against manifest if enforcement is enabled
        verifier = get_artifact_verifier()
        computed_hash = verifier.compute_sha256(self.ratings_path)

        if ENFORCE_SIGNED_MODELS:
            try:
                verifier.verify_artifact(
                    self.ratings_path,
                    require_manifest=True,
                    correlation_id=correlation_id,
                )
            except ModelVerificationError as e:
                verbose_proxy_logger.error(
                    f"{log_prefix}ELO ratings verification failed: {e}. "
                    f"Hash mismatch or manifest missing. Details: {e.details}"
                )
                raise

        verbose_proxy_logger.info(
            f"{log_prefix}Loading ELO ratings from: {self.ratings_path}"
        )

        # Safe activation: load into temp variable first
        is_pickle = path.suffix.lower() in (".pkl", ".pickle")
        try:
            if is_pickle:
                # Pickle format — requires explicit opt-in
                if not ALLOW_PICKLE_MODELS:
                    raise PickleSecurityError(self.ratings_path)
                with open(self.ratings_path, "rb") as f:
                    new_ratings = pickle.load(f)
            else:
                # JSON format — safe by default
                with open(self.ratings_path, "r") as f:
                    new_ratings = json.load(f)
        except (PickleSecurityError, json.JSONDecodeError):
            raise
        except Exception as e:
            raise ModelLoadError(
                self.ratings_path,
                f"Ratings load failed: {e}",
                correlation_id,
            )

        # Normalize: support pandas Series .to_dict() output and plain dicts
        if hasattr(new_ratings, "to_dict"):
            new_ratings = new_ratings.to_dict()

        if not isinstance(new_ratings, dict):
            raise ModelLoadError(
                self.ratings_path,
                f"Expected dict of {{model: score}}, got {type(new_ratings)}",
                correlation_id,
            )

        if not new_ratings:
            raise ModelLoadError(
                self.ratings_path,
                "Ratings file is empty",
                correlation_id,
            )

        # Safe swap: only update if everything succeeded
        with self._model_lock:
            old_ratings = self.elo_scores
            self.elo_scores = new_ratings

            self.model_version = verifier.record_active_version(
                self.ratings_path,
                sha256=computed_hash,
                tags=["elo", "active"],
            )

        verbose_proxy_logger.info(
            f"{log_prefix}ELO ratings loaded successfully. "
            f"{len(self.elo_scores)} models, "
            f"top model: {max(self.elo_scores.items(), key=lambda kv: kv[1])[0]}, "
            f"Version SHA256: {self.model_version.sha256[:16]}..."
        )

        del old_ratings

    def reload_model(self, correlation_id: Optional[str] = None) -> bool:
        """
        Reload ratings from disk with safe activation (for hot reload support).

        Safe Activation Pattern:
        1. Load new ratings into temporary variable
        2. Verify against manifest
        3. Only swap to active if successful
        4. Keep old ratings active on failure

        Args:
            correlation_id: Optional correlation ID for logging

        Returns:
            True if reload succeeded, False if failed (old ratings remain active)
        """
        log_prefix = f"[{correlation_id}] " if correlation_id else ""
        old_version = self.model_version

        try:
            self._load_ratings(correlation_id=correlation_id)
            verbose_proxy_logger.info(
                f"{log_prefix}ELO ratings reloaded successfully. "
                f"Old version: {old_version.sha256[:16] if old_version else 'none'}..., "
                f"New version: {self.model_version.sha256[:16] if self.model_version else 'none'}..."
            )
            return True
        except (ModelVerificationError, ModelLoadError, FileNotFoundError) as e:
            verbose_proxy_logger.error(
                f"{log_prefix}ELO ratings reload failed, keeping old ratings active. Error: {e}"
            )
            return False
        except Exception as e:
            verbose_proxy_logger.error(
                f"{log_prefix}Unexpected error during ELO ratings reload, keeping old ratings active. Error: {e}"
            )
            return False

    def route(self, query: str) -> Optional[str]:
        """
        Route a query to the highest-rated model.

        Note: ELO routing is query-independent — it always returns the model
        with the highest Elo rating. The query parameter is accepted for
        interface compatibility but is not used in the routing decision.

        Args:
            query: User query text (not used in ELO routing, but accepted
                   for interface compatibility with other routers)

        Returns:
            Name of the highest-rated model, or None if no ratings loaded
        """
        if self.elo_scores is None:
            verbose_proxy_logger.warning("ELO ratings not loaded, cannot route")
            return None

        try:
            # Pick the model with the highest Elo rating
            best_model = max(self.elo_scores.items(), key=lambda kv: kv[1])[0]

            verbose_proxy_logger.debug(
                f"ELO routing: selected={best_model} "
                f"(score={self.elo_scores[best_model]:.1f})"
            )

            # Apply label mapping if configured
            if self.label_mapping and best_model in self.label_mapping:
                mapped_label = self.label_mapping[best_model]
                verbose_proxy_logger.debug(
                    f"ELO label mapping: {best_model} -> {mapped_label}"
                )
                return mapped_label

            return str(best_model)

        except Exception as e:
            verbose_proxy_logger.error(f"ELO routing error: {e}")
            return None


# ---------------------------------------------------------------------------
# PyTorch model builders (standalone, no MetaRouter dependency)
# ---------------------------------------------------------------------------


def _build_mlp_classifier(
    input_dim: int,
    hidden_layer_sizes: List[int],
    num_classes: int,
    activation: str = "relu",
):
    """
    Build a standalone MLPClassifierNN matching UIUC's architecture.

    This reconstructs the exact same PyTorch model used by LLMRouter's
    MLPRouter trainer, so that a state_dict from training can be loaded
    directly.

    Architecture: input -> [Linear + activation] * N -> Linear -> output
    (No activation on final layer; softmax/argmax applied at inference.)

    Args:
        input_dim: Input feature dimension (embedding size)
        hidden_layer_sizes: List of hidden layer dimensions (e.g. [128, 64])
        num_classes: Number of output classes (candidate models)
        activation: Activation function name ('relu', 'tanh', 'logistic', 'identity')

    Returns:
        A PyTorch nn.Module with the same architecture as MLPClassifierNN
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _MLPClassifierNN(nn.Module):
        """Standalone MLP classifier matching UIUC MLPClassifierNN architecture."""

        def __init__(self, input_dim, hidden_layer_sizes, num_classes, activation):
            super().__init__()
            self.activation_name = activation
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_layer_sizes:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, num_classes))
            self.layers = nn.ModuleList(layers)

        def _get_activation(self):
            if self.activation_name == "relu":
                return F.relu
            elif self.activation_name == "tanh":
                return torch.tanh
            elif self.activation_name == "logistic":
                return torch.sigmoid
            elif self.activation_name == "identity":
                return lambda x: x
            else:
                return F.relu

        def forward(self, x):
            activation = self._get_activation()
            for layer in self.layers[:-1]:
                x = activation(layer(x))
            x = self.layers[-1](x)
            return x

        def predict(self, x):
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                return torch.argmax(logits, dim=-1)

    return _MLPClassifierNN(input_dim, hidden_layer_sizes, num_classes, activation)


def _build_bilinear_mf(
    latent_dim: int,
    num_models: int,
    text_dim: int,
):
    """
    Build a standalone BilinearMF matching UIUC's architecture.

    This reconstructs the exact same PyTorch model used by LLMRouter's
    MFRouter trainer, so that a state_dict from training can be loaded
    directly.

    Architecture:
    - P: nn.Embedding(num_models, latent_dim) — latent model embeddings
    - text_proj: nn.Linear(text_dim, latent_dim, bias=False) — text projection
    - classifier: nn.Linear(latent_dim, 1, bias=False) — final scoring

    Scoring: delta(M, q) = classifier(P_m * text_proj(q_emb))

    Args:
        latent_dim: Latent dimension for model/text embeddings
        num_models: Number of candidate models
        text_dim: Text embedding dimension (e.g. 768 for all-MiniLM-L6-v2)

    Returns:
        A PyTorch nn.Module with the same architecture as BilinearMF
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class _BilinearMF(nn.Module):
        """Standalone BilinearMF matching UIUC BilinearMF architecture."""

        def __init__(self, dim, num_models, text_dim):
            super().__init__()
            self.P = nn.Embedding(num_models, dim)
            self.text_proj = nn.Linear(text_dim, dim, bias=False)
            self.classifier = nn.Linear(dim, 1, bias=False)

        @property
        def device(self):
            return self.P.weight.device

        def project_text(self, q_emb):
            """Project raw text embedding into latent routing space."""
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)
            proj = self.text_proj(q_emb)
            return proj.squeeze(0)

        def forward(self, model_win, model_loss, q_emb):
            """Pairwise scoring: delta(win, q) - delta(loss, q)."""
            v_win = F.normalize(self.P(model_win), p=2, dim=-1)
            v_loss = F.normalize(self.P(model_loss), p=2, dim=-1)
            h = v_win - v_loss
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)
            interaction = h * q_emb
            logit = self.classifier(interaction).squeeze(-1)
            return logit

        def score_all(self, q_emb):
            """Return delta(M, q) for all models."""
            P_all = F.normalize(self.P.weight, p=2, dim=-1)
            interaction = P_all * q_emb
            logits = self.classifier(interaction).squeeze(-1)
            return logits

    return _BilinearMF(latent_dim, num_models, text_dim)


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
    "llmrouter-cost-cascade",  # CostCascadeRoutingStrategy - cheapest-first speculative escalation
    # Centroid-based routers
    "llmrouter-nadirclaw-centroid",  # CentroidRoutingStrategy - zero-config intelligent routing
    # Semantic / intent routers
    "llmrouter-semantic-intent",  # SemanticIntentRoutingStrategy - intent->model-group dispatch
    # Tag / regex / User-Agent routers (RouteIQ-6865)
    "llmrouter-tag-regex-ua",  # TagRegexUserAgentRoutingStrategy - tag/regex/UA -> group
    # Live-signal routers (RouteIQ-904b)
    "llmrouter-latency-sla",  # LatencySLARoutingStrategy - meet a p-latency target
    "llmrouter-usage-based",  # UsageBasedRoutingStrategy - spread by recent usage
    "llmrouter-least-busy",  # LeastBusyRoutingStrategy - fewest in-flight requests
    # Feature-vector contextual bandit (RouteIQ-6c67)
    "llmrouter-linucb",  # LinUCBRoutingStrategy - LinUCB feature-vector bandit
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
    "nadirclaw-centroid": {
        "confidence_threshold": 0.06,
        "session_ttl": 1800,
        "profile": "auto",
    },
    "cost-cascade": {
        "confidence_threshold": 0.7,
        "max_rungs": 4,
    },
    "semantic-intent": {
        "intent_model_groups": {},
        "similarity_threshold": 0.0,
    },
    "tag-regex-ua": {
        "tag_model_groups": {},
        "regex_model_groups": {},
        "user_agent_model_groups": {},
    },
    "latency-sla": {
        "p_latency_target_ms": 2000.0,
        "percentile": 0.95,
        "max_latency_samples": 256,
    },
    "usage-based": {},
    "least-busy": {},
    "linucb": {
        "alpha": 1.0,
        "tenant_buckets": 8,
        "seed": None,
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
        inference_strategies = {
            "llmrouter-knn",
            "llmrouter-svm",
            "llmrouter-mlp",
            "llmrouter-mf",
            "llmrouter-elo",
        }
        if self.use_inference_only and strategy_name in inference_strategies:
            verbose_proxy_logger.info(
                f"  Using inference-only mode for {strategy_name} "
                f"with embedding_model={self.embedding_model}"
            )

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        """
        Resolve model path to an actual file.

        If model_path is a directory, look for a model file inside.
        Supports .pkl (sklearn), .pt/.pth (PyTorch), and .json (ELO ratings).
        This allows flexibility: users can specify either the directory
        or the exact file path.

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

        # If it's a directory, look for model files
        if path.is_dir():
            # Look for model files in priority order: .pkl, .pt, .pth, .json
            model_extensions = ["*.pkl", "*.pt", "*.pth", "*.json"]
            model_files = []
            for ext in model_extensions:
                model_files.extend(path.glob(ext))

            if len(model_files) == 1:
                resolved = str(model_files[0])
                verbose_proxy_logger.info(
                    f"Resolved model directory to file: {resolved}"
                )
                return resolved
            elif len(model_files) > 1:
                # Multiple model files - use the most recently modified
                model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                resolved = str(model_files[0])
                verbose_proxy_logger.warning(
                    f"Multiple model files found, using most recent: {resolved}"
                )
                return resolved
            else:
                verbose_proxy_logger.warning(
                    f"Directory {model_path} contains no model files "
                    f"(.pkl, .pt, .pth, .json)"
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
        """Load the appropriate LLMRouter model based on strategy name.

        When use_inference_only is True (the default), lightweight inference
        adapters are used for KNN, SVM, MLP, MF, and ELO strategies. These
        adapters load pre-trained models directly without requiring the full
        MetaRouter training setup from UIUC LLMRouter.

        For other strategies, falls back to the full UIUC LLMRouter MetaRouter
        initialization which requires training data and config.
        """
        strategy_type = self.strategy_name.replace("llmrouter-", "")

        # Inference-only adapters for production deployment
        if self.use_inference_only:
            if strategy_type == "knn":
                return self._load_inference_knn_router()
            elif strategy_type == "svm":
                return self._load_inference_svm_router()
            elif strategy_type == "mlp":
                return self._load_inference_mlp_router()
            elif strategy_type == "mf":
                return self._load_inference_mf_router()
            elif strategy_type == "elo":
                return self._load_inference_elo_router()

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

    def _load_inference_svm_router(self) -> Optional[InferenceSVMRouter]:
        """
        Load inference-only SVM router that bypasses UIUC MetaRouter.

        Uses the same pickle-based loading pattern as KNN since upstream SVM
        uses sklearn's SVC which is serialized via pickle.

        Returns:
            InferenceSVMRouter instance, or None if loading fails
        """
        if not self.model_path:
            verbose_proxy_logger.error(
                "model_path is required for inference-only SVM router. "
                "Set routing_strategy_args.model_path in config."
            )
            return None

        try:
            router = InferenceSVMRouter(
                model_path=self.model_path,
                embedding_model=self.embedding_model,
                embedding_device=self.embedding_device,
                label_mapping=self.label_mapping,
            )
            verbose_proxy_logger.info(
                f"Loaded inference-only SVM router from: {self.model_path}"
            )
            return router
        except FileNotFoundError as e:
            verbose_proxy_logger.warning(
                f"SVM model file not found: {e}. "
                "Ensure model is trained and deployed to model_path."
            )
            return None
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load inference-only SVM router: {e}")
            return None

    def _load_inference_mlp_router(self) -> Optional[InferenceMLPRouter]:
        """
        Load inference-only MLP router that bypasses UIUC MetaRouter.

        Uses PyTorch state_dict loading (weights_only=True) which is safer
        than pickle. Requires a companion metadata JSON file with architecture
        parameters and class mappings.

        Returns:
            InferenceMLPRouter instance, or None if loading fails
        """
        if not self.model_path:
            verbose_proxy_logger.error(
                "model_path is required for inference-only MLP router. "
                "Set routing_strategy_args.model_path in config."
            )
            return None

        try:
            # Pass extra kwargs that may contain architecture params
            router = InferenceMLPRouter(
                model_path=self.model_path,
                metadata_path=self.extra_kwargs.get("metadata_path"),
                embedding_model=self.embedding_model,
                embedding_device=self.embedding_device,
                label_mapping=self.label_mapping,
                input_dim=self.extra_kwargs.get("input_dim", 768),
                hidden_layer_sizes=self.extra_kwargs.get("hidden_layer_sizes"),
                num_classes=self.extra_kwargs.get("num_classes"),
                activation=self.extra_kwargs.get("activation", "relu"),
                idx_to_model=self.extra_kwargs.get("idx_to_model"),
            )
            verbose_proxy_logger.info(
                f"Loaded inference-only MLP router from: {self.model_path}"
            )
            return router
        except FileNotFoundError as e:
            verbose_proxy_logger.warning(
                f"MLP model file not found: {e}. "
                "Ensure model is trained and deployed to model_path."
            )
            return None
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load inference-only MLP router: {e}")
            return None

    def _load_inference_mf_router(self) -> Optional[InferenceMFRouter]:
        """
        Load inference-only Matrix Factorization router that bypasses UIUC MetaRouter.

        Uses PyTorch state_dict loading (weights_only=True) which is safer
        than pickle. Requires a companion metadata JSON file with architecture
        dimensions and class mappings.

        Returns:
            InferenceMFRouter instance, or None if loading fails
        """
        if not self.model_path:
            verbose_proxy_logger.error(
                "model_path is required for inference-only MF router. "
                "Set routing_strategy_args.model_path in config."
            )
            return None

        try:
            router = InferenceMFRouter(
                model_path=self.model_path,
                metadata_path=self.extra_kwargs.get("metadata_path"),
                embedding_model=self.embedding_model,
                embedding_device=self.embedding_device,
                label_mapping=self.label_mapping,
                latent_dim=self.extra_kwargs.get("latent_dim", 128),
                text_dim=self.extra_kwargs.get("text_dim", 768),
                num_models=self.extra_kwargs.get("num_models"),
                idx_to_model=self.extra_kwargs.get("idx_to_model"),
            )
            verbose_proxy_logger.info(
                f"Loaded inference-only MF router from: {self.model_path}"
            )
            return router
        except FileNotFoundError as e:
            verbose_proxy_logger.warning(
                f"MF model file not found: {e}. "
                "Ensure model is trained and deployed to model_path."
            )
            return None
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load inference-only MF router: {e}")
            return None

    def _load_inference_elo_router(self) -> Optional[InferenceELORouter]:
        """
        Load inference-only ELO router that bypasses UIUC MetaRouter.

        ELO is the simplest adapter — it just loads pre-computed Elo ratings
        from a JSON or pickle file. JSON is preferred since it avoids the
        pickle code execution risk entirely.

        Returns:
            InferenceELORouter instance, or None if loading fails
        """
        if not self.model_path:
            verbose_proxy_logger.error(
                "model_path is required for inference-only ELO router. "
                "Set routing_strategy_args.model_path in config."
            )
            return None

        try:
            router = InferenceELORouter(
                ratings_path=self.model_path,
                label_mapping=self.label_mapping,
            )
            verbose_proxy_logger.info(
                f"Loaded inference-only ELO router from: {self.model_path}"
            )
            return router
        except FileNotFoundError as e:
            verbose_proxy_logger.warning(
                f"ELO ratings file not found: {e}. "
                "Ensure ratings are computed and deployed to model_path."
            )
            return None
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load inference-only ELO router: {e}")
            return None

    def _load_custom_router(self):
        """Load a custom router from the custom routers directory."""
        custom_path = os.environ.get(
            "LLMROUTER_CUSTOM_ROUTERS_PATH", "/app/custom_routers"
        )
        # Implementation for custom router loading
        verbose_proxy_logger.info(f"Loading custom router from: {custom_path}")
        return None

    # Inference adapter types that support hot reload via reload_model()
    _INFERENCE_ADAPTER_TYPES = (
        InferenceKNNRouter,
        InferenceSVMRouter,
        InferenceMLPRouter,
        InferenceMFRouter,
        InferenceELORouter,
    )

    @property
    def router(self):
        """Get the router instance, loading/reloading as needed."""
        with self._router_lock:
            if self._router is None or self._should_reload():
                # For inference-only adapters, use in-place reload_model()
                if (
                    self._router is not None
                    and isinstance(self._router, self._INFERENCE_ADAPTER_TYPES)
                    and self._should_reload()
                ):
                    adapter_name = type(self._router).__name__
                    verbose_proxy_logger.info(
                        f"Hot reloading {adapter_name} model due to file change"
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

                    if selected_model:
                        _record_selection_metric(self.strategy_name, selected_model)
                    return selected_model
            except Exception as e:
                verbose_proxy_logger.warning(f"Observability error: {e}")

        # Fallback: route without observability
        try:
            if self.router and hasattr(self.router, "route"):
                selected_model = self.router.route(query)
        except Exception as e:
            verbose_proxy_logger.error(f"Routing error (no observability): {e}")

        if selected_model:
            _record_selection_metric(self.strategy_name, selected_model)
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
            List of deployment dicts matching the requested model, with
            cooled-down / gov-banned arms removed (RouteIQ-a073). This strategy
            is registered live as ``"cost-aware"`` and bypasses LiteLLM's
            healthy-deployment pipeline like every RouteIQ strategy, so the
            pre-scoring filter must run here -- this seam was a gov-ban gap
            RouteIQ-a073's audit found beyond the centroid path.
        """
        router = context.router
        healthy = getattr(router, "healthy_deployments", router.model_list)
        group_matched = [
            dep for dep in healthy if dep.get("model_name") == context.model
        ]

        from litellm_llmrouter.candidate_filter import filter_routable_candidates

        return filter_routable_candidates(router, group_matched)

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
            selected_model = best_deployment.get("litellm_params", {}).get(
                "model", "unknown"
            )
            histogram = _get_cost_histogram()
            if histogram and best_cost_per_1k != float("inf"):
                histogram.record(
                    best_cost_per_1k,
                    {"model": selected_model, "strategy": "cost-aware"},
                )
            _record_selection_metric("cost-aware", selected_model)

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


class CostCascadeRoutingStrategy(RoutingStrategy):
    """Cost-aware speculative-cascade / escalation router (RouteIQ-90d0).

    DISTINCT from :class:`CostAwareRoutingStrategy` (heuristic Pareto, single
    shot, NOT a cascade). This strategy implements a *cascade*: route first to
    the CHEAPEST capable deployment, and escalate to a stronger/pricier
    deployment on a low-confidence / low-quality signal.

    Because one :meth:`select_deployment` call returns exactly one deployment,
    the cascade is realized in two complementary modes -- the strategy supports
    BOTH simultaneously and the active mode is decided per request:

    * **Mode (a) -- ladder-in-metadata (default).** The deployments are ordered
      cheapest -> strongest and the cheapest rung is returned. The full ordered
      escalation ladder (a list of ``{"model", "model_name", "cost_per_1k"}``
      rung descriptors) is exposed on the returned deployment under
      ``deployment["metadata"]["routeiq_cascade"]`` so the *caller* can retry up
      the ladder on a low-quality response.

      NOTE: the ladder is attached to a SHALLOW COPY of the selected deployment
      (``{**dep, "metadata": {...}}``) so the router's shared ``model_list``
      dict is never mutated -- the cascade metadata is per-decision.

    * **Mode (b) -- confidence-gated escalation.** When the caller supplies a
      prior-attempt signal in ``context.request_kwargs`` or ``context.metadata``
      -- key ``cascade_rung`` (0-based index of the rung already tried) and/or
      ``cascade_confidence`` (float in ``[0, 1]``) -- the strategy reads it and
      picks the NEXT rung up the ladder. Escalation happens when the prior rung
      is explicitly named, or when ``cascade_confidence`` is BELOW
      ``confidence_threshold``. The chosen rung is clamped to the ladder length
      (capped by ``max_rungs``).

    Cost ordering uses ``litellm.model_cost`` (input+output per-token, averaged).
    Arms with UNKNOWN cost sort LAST (``+inf``) so the cheap-first invariant
    degrades gracefully -- an all-unknown-cost group still routes (to its first
    member) instead of returning ``None``.

    Configuration (via ``ROUTEIQ_COST_CASCADE__*`` /
    :class:`~litellm_llmrouter.settings.CostCascadeSettings`):
        confidence_threshold: escalate below this prior-attempt confidence
            (default 0.7).
        max_rungs: cap the exposed/escalatable ladder length (default 4).

    Stress-harness validation: ``tools/stress_harness`` resolves the live active
    strategy by name and dispatches a per-strategy verdict; ``cost-cascade`` is
    not (yet) a registered verdict family, so it hits the harness's generic
    distribution verdict (``stress_harness.verdicts.generic_verdict``) which
    asserts the per-bucket routing distribution is non-degenerate. To exercise
    the cheap-first invariant explicitly, point the harness's easy-bucket at this
    strategy and confirm the cheapest rung dominates (the ladder is observable in
    the ``routeiq_cascade`` decision metadata via the routing-decision log).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_rungs: int = 4,
        **kwargs: Any,
    ):
        self._confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self._max_rungs = max(1, int(max_rungs))

    @property
    def name(self) -> str:
        return "llmrouter-cost-cascade"

    @property
    def version(self) -> Optional[str]:
        return "1.0.0"

    # ------------------------------------------------------------------
    # Cost lookup (litellm.model_cost; unknown -> +inf so it sorts last)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model_cost(model: str) -> float:
        """Average cost per 1K tokens for a model, or ``+inf`` if unknown.

        Reads ``litellm.model_cost`` (input+output per-token, averaged, scaled
        to 1K). Unknown / zero-cost models return ``+inf`` so they sort LAST in
        the cheapest-first ordering -- the graceful-degrade contract.
        """
        try:
            import litellm

            cost_info = litellm.model_cost.get(model, {}) or {}
            input_cost = (cost_info.get("input_cost_per_token", 0) or 0) * 1000
            output_cost = (cost_info.get("output_cost_per_token", 0) or 0) * 1000
            avg = (input_cost + output_cost) / 2
            return avg if avg > 0 else float("inf")
        except Exception:
            return float("inf")

    def _get_candidates(self, context: RoutingContext) -> List[Dict]:
        """Candidate deployments for the requested model group.

        Mirrors :meth:`CostAwareRoutingStrategy._get_candidates`: reads the
        static ``healthy_deployments`` and applies the RouteIQ pre-scoring
        cooldown / gov-ban filter (the RouteIQ path bypasses LiteLLM's pipeline).
        """
        router = context.router
        if router is None:
            return []
        healthy = getattr(router, "healthy_deployments", None)
        if healthy is None:
            healthy = getattr(router, "model_list", [])
        group_matched = [
            dep for dep in healthy if dep.get("model_name") == context.model
        ]

        from litellm_llmrouter.candidate_filter import filter_routable_candidates

        return filter_routable_candidates(router, group_matched)

    def _ordered_ladder(self, candidates: List[Dict]) -> List[Tuple[Dict, float]]:
        """Order candidates cheapest -> strongest (by ascending cost).

        Returns ``(deployment, cost_per_1k)`` tuples. Stable sort keeps the
        config order for equal-cost arms. Capped at ``max_rungs``.
        """
        scored = [
            (dep, self._get_model_cost(dep.get("litellm_params", {}).get("model", "")))
            for dep in candidates
        ]
        scored.sort(key=lambda pair: pair[1])
        return scored[: self._max_rungs]

    @staticmethod
    def _rung_descriptor(dep: Dict, cost: float) -> Dict[str, Any]:
        """Build a JSON-safe rung descriptor for the ladder metadata."""
        return {
            "model": dep.get("litellm_params", {}).get("model", ""),
            "model_name": dep.get("model_name", ""),
            "cost_per_1k": (None if cost == float("inf") else cost),
        }

    def _read_prior_signal(
        self, context: RoutingContext
    ) -> Tuple[Optional[int], Optional[float]]:
        """Read a prior-attempt cascade signal from the context (mode (b)).

        Looks in ``request_kwargs`` first, then ``metadata`` for ``cascade_rung``
        (int, 0-based) and ``cascade_confidence`` (float). Returns
        ``(rung, confidence)`` with ``None`` for absent / unparseable values.
        """
        sources: List[Dict] = []
        if context.request_kwargs:
            sources.append(context.request_kwargs)
        if context.metadata:
            sources.append(context.metadata)

        rung: Optional[int] = None
        confidence: Optional[float] = None
        for src in sources:
            if rung is None and "cascade_rung" in src:
                try:
                    rung = int(src["cascade_rung"])
                except (TypeError, ValueError):
                    rung = None
            if confidence is None and "cascade_confidence" in src:
                try:
                    confidence = float(src["cascade_confidence"])
                except (TypeError, ValueError):
                    confidence = None
        return rung, confidence

    def select_deployment(
        self,
        context: RoutingContext,
    ) -> Optional[Dict]:
        """Select the cheapest capable rung, or escalate on a prior signal.

        Returns a SHALLOW COPY of the selected deployment with the ordered
        escalation ladder attached under ``metadata["routeiq_cascade"]`` (never
        mutates the router's shared deployment dict). Returns ``None`` only when
        there are no candidates.
        """
        candidates = self._get_candidates(context)
        if not candidates:
            return None

        ladder = self._ordered_ladder(candidates)
        if not ladder:
            return None

        # Mode (b): confidence-gated escalation from a prior attempt.
        prior_rung, prior_confidence = self._read_prior_signal(context)
        target_index = 0
        escalated = False
        if prior_rung is not None:
            # The caller already tried rung ``prior_rung`` -> advance one rung.
            target_index = prior_rung + 1
            escalated = True
        elif prior_confidence is not None and (
            prior_confidence < self._confidence_threshold
        ):
            # Low-confidence prior result with no explicit rung: escalate once.
            target_index = 1
            escalated = True

        # Clamp to the ladder (cannot escalate past the strongest rung).
        if target_index >= len(ladder):
            target_index = len(ladder) - 1

        selected_dep, selected_cost = ladder[target_index]

        ladder_meta = [self._rung_descriptor(dep, cost) for dep, cost in ladder]
        existing_meta = dict(selected_dep.get("metadata", {}) or {})
        existing_meta["routeiq_cascade"] = {
            "ladder": ladder_meta,
            "selected_rung": target_index,
            "escalated": escalated,
            "confidence_threshold": self._confidence_threshold,
            "max_rungs": self._max_rungs,
        }
        result = {**selected_dep, "metadata": existing_meta}

        selected_model = selected_dep.get("litellm_params", {}).get("model", "unknown")
        histogram = _get_cost_histogram()
        if histogram and selected_cost != float("inf"):
            histogram.record(
                selected_cost,
                {"model": selected_model, "strategy": "cost-cascade"},
            )
        _record_selection_metric("cost-cascade", selected_model)

        return result

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate the strategy is ready to serve requests."""
        if not (0.0 <= self._confidence_threshold <= 1.0):
            return False, "confidence_threshold must be between 0.0 and 1.0"
        if self._max_rungs < 1:
            return False, "max_rungs must be >= 1"
        return True, None


class SemanticIntentRoutingStrategy(RoutingStrategy):
    """Semantic / embedding intent router (RouteIQ-7936).

    Routes by request INTENT -> model-group. The request text (last user
    message / system prompt, extracted from ``context.messages`` or
    ``context.input``) is classified into an intent label by EMBEDDING
    SIMILARITY to per-intent centroids, then the winning intent is mapped to a
    configured model group (a list of model-name patterns) and the request is
    routed to the cheapest-listed matching deployment.

    HOT-PATH DISCIPLINE (mirrors :mod:`centroid_routing`): the SentenceTransformer
    encoder is NEVER cold-loaded on the request path. The strategy only
    classifies when the shared encoder is ALREADY loaded -- it checks the module
    singleton ``strategies._sentence_transformer_model`` WITHOUT triggering a
    load, and falls through to a graceful fallback deployment if it is ``None``
    (classifier-not-loaded). The intent centroids are likewise built lazily and
    cached -- and only AFTER the encoder is confirmed loaded.

    The intent centroids are derived from the configured intent labels
    themselves (the label string + its model-group patterns are encoded as the
    centroid seed text). This is a zero-extra-artifact approach: no separate
    centroid ``.npy`` files are required -- the intent map IS the classifier
    definition. Operators pre-warm the encoder at startup (e.g. via the centroid
    warmup hook) to enable classification; until then the router degrades
    gracefully.

    Configuration (via ``ROUTEIQ_SEMANTIC_INTENT__*`` /
    :class:`~litellm_llmrouter.settings.SemanticIntentSettings`):
        intent_model_groups: ``{intent_label: [model_pattern, ...]}``.
        similarity_threshold: minimum top-1 similarity to accept a
            classification (below -> unmapped -> fallback).

    Dispatch outcomes:
        * intent classified + mapped -> route to a deployment in that group.
        * intent classified but UNMAPPED (no group / below threshold) -> fallback.
        * encoder NOT loaded -> graceful fallthrough (NO cold load) -> fallback.

    Stress-harness validation: ``tools/stress_harness`` resolves the active
    strategy by name; ``semantic-intent`` is not a registered verdict family, so
    it hits the generic distribution verdict
    (``stress_harness.verdicts.generic_verdict``), which confirms the routing
    distribution across the 5 workload buckets is non-degenerate. Because the
    harness fires categorized buckets (code / reasoning / chat / ...), a correct
    intent map produces a bucket-correlated model-group dispatch the generic
    verdict's per-bucket distribution surfaces.
    """

    def __init__(
        self,
        intent_model_groups: Optional[Dict[str, List[str]]] = None,
        similarity_threshold: float = 0.0,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        **kwargs: Any,
    ):
        self._intent_model_groups = intent_model_groups or {}
        self._similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self._embedding_model = embedding_model
        # Lazily-built, cached intent centroids: {intent_label: np.ndarray}.
        self._centroids: Optional[Dict[str, Any]] = None
        self._centroid_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "llmrouter-semantic-intent"

    @property
    def version(self) -> Optional[str]:
        return "1.0.0"

    # ------------------------------------------------------------------
    # Hot-path discipline: only classify if the encoder is ALREADY loaded
    # ------------------------------------------------------------------

    @staticmethod
    def _encoder_is_loaded() -> bool:
        """True iff the shared SentenceTransformer is already in memory.

        Reads the module-level singleton WITHOUT triggering a load. This is the
        hot-path guard: a cold encoder means we must NOT classify (no cold load
        on the request path) and instead fall through to the fallback.
        """
        return _sentence_transformer_model is not None

    def _build_centroids(self) -> Optional[Dict[str, Any]]:
        """Build (once) and cache per-intent centroid embeddings.

        Only called AFTER the encoder is confirmed loaded. The centroid seed
        text for each intent is the label plus its model-group patterns, encoded
        and unit-normalized. Returns ``None`` if numpy is unavailable or no
        intents are configured.
        """
        if self._centroids is not None:
            return self._centroids
        if not self._intent_model_groups:
            return None
        try:
            import numpy as np
        except ImportError:
            return None

        with self._centroid_lock:
            if self._centroids is not None:
                return self._centroids
            encoder = _get_sentence_transformer(self._embedding_model)
            centroids: Dict[str, Any] = {}
            for intent, patterns in self._intent_model_groups.items():
                seed = " ".join([intent, *(patterns or [])]).strip() or intent
                emb = encoder.encode([seed], show_progress_bar=False)[0]
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    emb = emb / norm
                centroids[intent] = emb
            self._centroids = centroids
            return self._centroids

    def _classify_intent(self, text: str) -> Tuple[Optional[str], float]:
        """Classify ``text`` into an intent by max cosine similarity.

        Returns ``(intent_label, similarity)``. Returns ``(None, 0.0)`` when the
        encoder is not loaded (graceful, NO cold load), numpy is missing, no
        centroids are configured, or the top-1 similarity is below the
        configured ``similarity_threshold`` (UNMAPPED).
        """
        if not text or not self._encoder_is_loaded():
            return None, 0.0
        try:
            import numpy as np
        except ImportError:
            return None, 0.0

        centroids = self._build_centroids()
        if not centroids:
            return None, 0.0

        encoder = _get_sentence_transformer(self._embedding_model)
        emb = encoder.encode([text], show_progress_bar=False)[0]
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm

        best_intent: Optional[str] = None
        best_sim = -1.0
        for intent, centroid in centroids.items():
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_intent = intent

        if best_intent is None or best_sim < self._similarity_threshold:
            return None, max(0.0, best_sim)
        return best_intent, best_sim

    @staticmethod
    def _extract_text(context: RoutingContext) -> str:
        """Extract classification text from messages or input.

        Prefers the last user message text; falls back to the system prompt,
        then to ``context.input`` (string or first list element).
        """
        messages = context.messages or []
        if not messages and context.request_kwargs:
            messages = context.request_kwargs.get("messages", []) or []

        system_text = ""
        last_user = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(parts)
            if not isinstance(content, str):
                content = str(content) if content else ""
            if role in ("system", "developer"):
                system_text = content
            elif role == "user":
                last_user = content

        if last_user:
            return last_user
        if system_text:
            return system_text

        inp = context.input
        if isinstance(inp, str):
            return inp
        if isinstance(inp, list) and inp:
            return str(inp[0])
        return ""

    def _get_candidates(self, context: RoutingContext) -> List[Dict]:
        """Candidate deployments for the requested model group (filtered)."""
        router = context.router
        if router is None:
            return []
        healthy = getattr(router, "healthy_deployments", None)
        if healthy is None:
            healthy = getattr(router, "model_list", [])
        group_matched = [
            dep for dep in healthy if dep.get("model_name") == context.model
        ]

        from litellm_llmrouter.candidate_filter import filter_routable_candidates

        return filter_routable_candidates(router, group_matched)

    def _match_group(self, intent: str, candidates: List[Dict]) -> Optional[Dict]:
        """Return the candidate whose model best matches the intent's group.

        The group is the ordered list of patterns mapped to ``intent``. Matching
        is two-pass to avoid substring collisions (e.g. pattern ``"gpt-4o"``
        is a substring of BOTH ``"gpt-4o"`` and ``"gpt-4o-mini"``):

        1. EXACT (case-insensitive) model-name equality, in pattern order, so a
           pattern that names a model exactly always wins over a broader match.
        2. SUBSTRING fallback, in pattern order, for patterns meant as prefixes
           / families.

        Returns ``None`` when the intent has no group or nothing matches.
        """
        patterns = self._intent_model_groups.get(intent)
        if not patterns:
            return None

        # Pass 1: exact model-name match (pattern order = priority).
        for pattern in patterns:
            p = pattern.lower()
            for dep in candidates:
                model = dep.get("litellm_params", {}).get("model", "").lower()
                if model == p:
                    return dep

        # Pass 2: substring match (pattern order = priority).
        for pattern in patterns:
            p = pattern.lower()
            for dep in candidates:
                model = dep.get("litellm_params", {}).get("model", "").lower()
                if p in model:
                    return dep
        return None

    @staticmethod
    def _fallback_deployment(candidates: List[Dict]) -> Optional[Dict]:
        """Graceful fallback: first available candidate, or ``None``."""
        return candidates[0] if candidates else None

    def select_deployment(
        self,
        context: RoutingContext,
    ) -> Optional[Dict]:
        """Route by classified intent -> model group, with graceful fallback.

        Flow: extract text -> (if encoder loaded) classify intent -> map intent
        to a model group -> route to a matching candidate. Every off-path
        condition (encoder not loaded, unmapped intent, no group match) falls
        back to the first available candidate. Returns ``None`` only when there
        are no candidates at all.
        """
        candidates = self._get_candidates(context)
        if not candidates:
            return None

        text = self._extract_text(context)
        intent, _sim = self._classify_intent(text)
        if intent is None:
            # encoder-not-loaded / unmapped / below-threshold -> graceful fallback
            return self._fallback_deployment(candidates)

        matched = self._match_group(intent, candidates)
        if matched is None:
            return self._fallback_deployment(candidates)

        selected_model = matched.get("litellm_params", {}).get("model", "unknown")
        _record_selection_metric("semantic-intent", selected_model)
        return matched

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate the strategy is ready to serve requests."""
        if not (0.0 <= self._similarity_threshold <= 1.0):
            return False, "similarity_threshold must be between 0.0 and 1.0"
        return True, None


class TagRegexUserAgentRoutingStrategy(RoutingStrategy):
    """Tag / regex / User-Agent routing strategy (RouteIQ-6865).

    Upstream LiteLLM ``tag_based_routing`` is BYPASSED by the RouteIQ custom
    routing strategy (``set_custom_routing_strategy``), so RouteIQ re-implements
    tag-style routing natively.  A request is matched against three signal
    sources, in strict priority order, and the FIRST matching rule maps the
    request to a model-group subset (an ordered list of model-name patterns):

    1. **Tag** -- an explicit tag in ``context.request_kwargs["tags"]`` /
       ``context.metadata["tags"]`` (a list or a scalar), matched EXACTLY against
       :attr:`tag_model_groups` keys.
    2. **Regex** -- a Python regex over the request text (last user message /
       ``context.input``), matched against :attr:`regex_model_groups` keys
       (``re.search``, first key in insertion order to match wins).
    3. **User-Agent** -- the ``User-Agent`` request header (read case-insensitively
       from ``context.request_kwargs["headers"]`` / ``context.metadata["headers"]``)
       matched as a case-insensitive SUBSTRING against
       :attr:`user_agent_model_groups` keys.

    Once a group is chosen, the request routes to the first candidate whose
    ``litellm_params.model`` matches one of the group's patterns (exact match
    first, then substring, both in pattern order).  Every off-path condition
    (no rule matched, matched group has no available candidate) falls back to the
    first available candidate -- it returns ``None`` only when there are no
    candidates at all.

    Stress-harness validation: ``tools/stress_harness`` resolves the active
    strategy by name; ``tag-regex-ua`` is not a registered verdict family so it
    hits the generic distribution verdict
    (``stress_harness.verdicts.generic_verdict``).  Drive a workload tagged with
    the configured tags (or matching regexes / User-Agents) and confirm the
    per-bucket model-group dispatch is the configured one.
    """

    def __init__(
        self,
        tag_model_groups: Optional[Dict[str, List[str]]] = None,
        regex_model_groups: Optional[Dict[str, List[str]]] = None,
        user_agent_model_groups: Optional[Dict[str, List[str]]] = None,
        **kwargs: Any,
    ):
        self._tag_model_groups = tag_model_groups or {}
        self._regex_model_groups = regex_model_groups or {}
        self._user_agent_model_groups = user_agent_model_groups or {}
        # Pre-compile regexes once; an invalid pattern is skipped (logged), never
        # raised, so one bad config entry cannot break routing.
        self._compiled_regex: List[Tuple[Any, List[str]]] = []
        import re

        for pattern, group in self._regex_model_groups.items():
            try:
                self._compiled_regex.append((re.compile(pattern), group or []))
            except re.error as exc:  # pragma: no cover - defensive
                verbose_proxy_logger.warning(
                    f"TagRouting: skipping invalid regex {pattern!r}: {exc}"
                )

    @property
    def name(self) -> str:
        return "llmrouter-tag-regex-ua"

    @property
    def version(self) -> Optional[str]:
        return "1.0.0"

    def _get_candidates(self, context: RoutingContext) -> List[Dict]:
        """Candidate deployments for the requested model group (filtered)."""
        router = context.router
        if router is None:
            return []
        healthy = getattr(router, "healthy_deployments", None)
        if healthy is None:
            healthy = getattr(router, "model_list", [])
        group_matched = [
            dep for dep in healthy if dep.get("model_name") == context.model
        ]

        from litellm_llmrouter.candidate_filter import filter_routable_candidates

        return filter_routable_candidates(router, group_matched)

    @staticmethod
    def _sources(context: RoutingContext) -> List[Dict]:
        """Ordered context dicts to read signals from (request_kwargs first)."""
        sources: List[Dict] = []
        if context.request_kwargs:
            sources.append(context.request_kwargs)
        if context.metadata:
            sources.append(context.metadata)
        return sources

    def _extract_tags(self, context: RoutingContext) -> List[str]:
        """Collect request tags from request_kwargs / metadata (list or scalar)."""
        tags: List[str] = []
        for src in self._sources(context):
            raw = src.get("tags")
            if raw is None:
                continue
            if isinstance(raw, (list, tuple, set)):
                tags.extend(str(t) for t in raw)
            else:
                tags.append(str(raw))
        return tags

    @staticmethod
    def _extract_text(context: RoutingContext) -> str:
        """Best-effort request text: last user message, else input."""
        for msg in reversed(context.messages or []):
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                joined = " ".join(parts).strip()
                if joined:
                    return joined
        inp = context.input
        if isinstance(inp, str):
            return inp
        if isinstance(inp, list) and inp:
            return str(inp[0])
        return ""

    def _extract_user_agent(self, context: RoutingContext) -> str:
        """Read the User-Agent header (case-insensitive key) from the context."""
        for src in self._sources(context):
            headers = src.get("headers")
            if not isinstance(headers, dict):
                continue
            for key, value in headers.items():
                if str(key).lower() == "user-agent" and value:
                    return str(value)
        return ""

    def _select_group(self, context: RoutingContext) -> Optional[List[str]]:
        """Resolve the model-group patterns for the request, in priority order.

        Tag (exact) -> regex (search) -> User-Agent (substring, ci). Returns the
        first matching group's pattern list, or ``None`` if nothing matched.
        """
        # 1. Tag (exact key match, request-order priority).
        for tag in self._extract_tags(context):
            group = self._tag_model_groups.get(tag)
            if group:
                return group

        # 2. Regex over the request text (insertion order = priority).
        text = self._extract_text(context)
        if text:
            for compiled, group in self._compiled_regex:
                if group and compiled.search(text):
                    return group

        # 3. User-Agent substring (case-insensitive, insertion order).
        ua = self._extract_user_agent(context).lower()
        if ua:
            for needle, group in self._user_agent_model_groups.items():
                if group and needle.lower() in ua:
                    return group
        return None

    @staticmethod
    def _match_in_group(patterns: List[str], candidates: List[Dict]) -> Optional[Dict]:
        """First candidate matching the group (exact model first, then substring)."""
        # Pass 1: exact (case-insensitive) model-name equality, pattern order.
        for pattern in patterns:
            p = pattern.lower()
            for dep in candidates:
                model = dep.get("litellm_params", {}).get("model", "").lower()
                if model == p:
                    return dep
        # Pass 2: substring, pattern order.
        for pattern in patterns:
            p = pattern.lower()
            for dep in candidates:
                model = dep.get("litellm_params", {}).get("model", "").lower()
                if p in model:
                    return dep
        return None

    def select_deployment(
        self,
        context: RoutingContext,
    ) -> Optional[Dict]:
        """Route by tag / regex / User-Agent -> model group, graceful fallback.

        Returns ``None`` only when there are no candidates; any non-match falls
        back to the first available candidate.
        """
        candidates = self._get_candidates(context)
        if not candidates:
            return None

        group = self._select_group(context)
        if group is None:
            return candidates[0]  # no rule matched -> fallback

        matched = self._match_in_group(group, candidates)
        if matched is None:
            return candidates[0]  # group matched but no candidate in it

        selected_model = matched.get("litellm_params", {}).get("model", "unknown")
        _record_selection_metric("tag-regex-ua", selected_model)
        return matched

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Ready iff at least one of the three maps is configured."""
        if not (
            self._tag_model_groups
            or self._regex_model_groups
            or self._user_agent_model_groups
        ):
            return False, "no tag / regex / user_agent model groups configured"
        return True, None


# ---------------------------------------------------------------------------
# Latency-SLA / usage-based / least-busy strategies (RouteIQ-904b)
# ---------------------------------------------------------------------------
#
# Upstream LiteLLM's latency-based / usage-based / least-busy routers are
# BYPASSED by RouteIQ's custom routing strategy (``set_custom_routing_strategy``).
# These three RouteIQ-native strategies re-implement that family, reading LIVE
# signals from the per-worker ``RoutingStatsAccumulator`` (decision counts +
# rolling latency) and a per-strategy in-flight counter, degrading gracefully
# (deterministic fallback) when no signal exists yet. They share a small mixin
# for candidate resolution.


class _SignalAwareRoutingMixin:
    """Shared candidate resolution for the live-signal routing strategies."""

    @staticmethod
    def _get_candidates(context: RoutingContext) -> List[Dict]:
        router = context.router
        if router is None:
            return []
        healthy = getattr(router, "healthy_deployments", None)
        if healthy is None:
            healthy = getattr(router, "model_list", [])
        group_matched = [
            dep for dep in healthy if dep.get("model_name") == context.model
        ]
        from litellm_llmrouter.candidate_filter import filter_routable_candidates

        return filter_routable_candidates(router, group_matched)

    @staticmethod
    def _arm_key(deployment: Dict) -> str:
        return str(deployment.get("litellm_params", {}).get("model", ""))


class LatencySLARoutingStrategy(_SignalAwareRoutingMixin, RoutingStrategy):
    """Latency-SLA router (RouteIQ-904b): pick a deployment meeting a p-latency
    target.

    Maintains a bounded per-model ring buffer of OBSERVED latency samples
    (milliseconds), fed via :meth:`record_latency` from the post-response path
    (or by tests).  On each decision it estimates each candidate's percentile
    latency and selects, in candidate (config) order, the FIRST deployment whose
    percentile latency is at or below ``p_latency_target_ms``.  If NONE meet the
    target (or none have samples yet -- cold start), it falls back to the
    lowest observed-percentile deployment, and finally to the first candidate
    (so it never returns ``None`` while candidates exist).

    Live-signal source: its OWN per-model sample buffer (process-local, bounded).
    A model with no samples is treated as meeting the SLA (optimistic cold-start)
    so a fresh deployment is given a chance instead of being starved.

    Stress-harness: not a registered verdict family -> generic distribution
    verdict.  Seed per-model latencies via :meth:`record_latency` and confirm
    the sub-target deployment dominates.
    """

    def __init__(
        self,
        p_latency_target_ms: float = 2000.0,
        percentile: float = 0.95,
        max_latency_samples: int = 256,
        **kwargs: Any,
    ):
        self._target_ms = max(0.0, float(p_latency_target_ms))
        self._percentile = max(0.0, min(1.0, percentile))
        self._max_samples = max(1, int(max_latency_samples))
        # model -> bounded list of latency samples (ms). Mutated under lock.
        self._samples: Dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "llmrouter-latency-sla"

    @property
    def version(self) -> Optional[str]:
        return "1.0.0"

    def record_latency(self, model: str, latency_ms: float) -> None:
        """Record one observed latency sample (ms) for ``model`` (bounded ring)."""
        if not model or latency_ms is None or latency_ms < 0:
            return
        with self._lock:
            buf = self._samples.get(model)
            if buf is None:
                buf = deque(maxlen=self._max_samples)
                self._samples[model] = buf
            buf.append(float(latency_ms))

    def _percentile_latency(self, model: str) -> Optional[float]:
        """Estimate the configured percentile latency for ``model``.

        Returns ``None`` when the model has no samples (cold start). Uses the
        nearest-rank percentile over a sorted copy of the bounded buffer.
        """
        with self._lock:
            buf = self._samples.get(model)
            samples = sorted(buf) if buf else []
        if not samples:
            return None
        if len(samples) == 1:
            return samples[0]
        # Nearest-rank: index = ceil(p * n) - 1, clamped.
        rank = max(
            0, min(len(samples) - 1, math.ceil(self._percentile * len(samples)) - 1)
        )
        return samples[rank]

    def select_deployment(
        self,
        context: RoutingContext,
    ) -> Optional[Dict]:
        candidates = self._get_candidates(context)
        if not candidates:
            return None

        best_below: Optional[Dict] = None
        lowest_dep: Optional[Dict] = None
        lowest_lat = float("inf")
        for dep in candidates:
            model = self._arm_key(dep)
            p_lat = self._percentile_latency(model)
            if p_lat is None:
                # Cold start: no samples -> treat as meeting the SLA (optimistic).
                if best_below is None:
                    best_below = dep
                continue
            if p_lat <= self._target_ms and best_below is None:
                best_below = dep
            if p_lat < lowest_lat:
                lowest_lat = p_lat
                lowest_dep = dep

        selected = best_below or lowest_dep or candidates[0]
        _record_selection_metric("latency-sla", self._arm_key(selected))
        return selected

    def validate(self) -> Tuple[bool, Optional[str]]:
        if not (0.0 <= self._percentile <= 1.0):
            return False, "percentile must be between 0.0 and 1.0"
        return True, None


class UsageBasedRoutingStrategy(_SignalAwareRoutingMixin, RoutingStrategy):
    """Usage-based router (RouteIQ-904b): spread load by RECENT decision usage.

    Reads the LIVE per-model decision counts from the process-local
    :class:`~litellm_llmrouter.router_decision_callback.RoutingStatsAccumulator`
    (``model_distribution``) and routes to the candidate with the FEWEST recorded
    decisions, so traffic spreads toward under-used deployments.  Ties (and the
    cold-start all-zero case) break on candidate (config) order, so the first
    decision is deterministic.

    Live-signal source: the shared stats accumulator (per-worker scope -- the
    same scope the admin stats panel reports).  Reads are best-effort and
    fail-open: if the accumulator is unavailable, every count is treated as 0 and
    the strategy degrades to first-candidate order.

    NOTE: the accumulator's model key is the ``litellm_params.model`` arm key
    recorded at decision time.  This strategy keys on the same value, so its view
    matches what the stats panel shows.

    Stress-harness: generic distribution verdict.  Over many decisions the
    selection distribution flattens across candidates (the spread property).
    """

    def __init__(self, **kwargs: Any):
        pass

    @property
    def name(self) -> str:
        return "llmrouter-usage-based"

    @property
    def version(self) -> Optional[str]:
        return "1.0.0"

    @staticmethod
    def _usage_counts() -> Dict[str, int]:
        """Live per-model decision counts from the stats accumulator (fail-open)."""
        try:
            from litellm_llmrouter.router_decision_callback import (
                get_stats_accumulator,
            )

            snap = get_stats_accumulator().global_snapshot()
            dist = snap.get("model_distribution", {}) or {}
            return {str(k): int(v) for k, v in dist.items()}
        except Exception:  # pragma: no cover - signal read must never break routing
            return {}

    def select_deployment(
        self,
        context: RoutingContext,
    ) -> Optional[Dict]:
        candidates = self._get_candidates(context)
        if not candidates:
            return None

        counts = self._usage_counts()
        best: Optional[Dict] = None
        best_count = float("inf")
        for dep in candidates:
            model = self._arm_key(dep)
            c = counts.get(model, 0)
            if c < best_count:
                best_count = c
                best = dep

        selected = best or candidates[0]
        _record_selection_metric("usage-based", self._arm_key(selected))
        return selected


class LeastBusyRoutingStrategy(_SignalAwareRoutingMixin, RoutingStrategy):
    """Least-busy router (RouteIQ-904b): pick the deployment with the fewest
    in-flight requests.

    Maintains a process-local per-model IN-FLIGHT counter.  The caller signals
    request lifecycle via :meth:`acquire` (on dispatch) and :meth:`release` (on
    completion); :meth:`select_deployment` picks the candidate with the lowest
    current in-flight count and increments it.  Ties (and the cold-start all-zero
    case) break on candidate order.

    Live-signal source: its OWN per-model in-flight gauge (the upstream
    ``least-busy`` strategy is bypassed, and the OTel ``backpressure.active_requests``
    gauge is write-only / process-aggregate, not per-model, so a strategy-owned
    counter is the available signal).  ``acquire`` is called inside
    :meth:`select_deployment`; ``release`` MUST be called by the response path
    (or a test) to decrement.  Releasing an unknown / already-zero model is a
    safe no-op (never goes negative).

    Stress-harness: generic distribution verdict.  Without paired release calls
    the counters only grow, so the strategy round-robins (each pick increments
    the chosen arm, making it momentarily busier) -- a useful no-feedback default.
    """

    def __init__(self, **kwargs: Any):
        self._inflight: Dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "llmrouter-least-busy"

    @property
    def version(self) -> Optional[str]:
        return "1.0.0"

    def acquire(self, model: str) -> None:
        """Increment the in-flight counter for ``model`` (request dispatched)."""
        if not model:
            return
        with self._lock:
            self._inflight[model] = self._inflight.get(model, 0) + 1

    def release(self, model: str) -> None:
        """Decrement the in-flight counter for ``model`` (request completed).

        Floors at 0 so an unmatched / double release never drives it negative.
        """
        if not model:
            return
        with self._lock:
            current = self._inflight.get(model, 0)
            if current > 0:
                self._inflight[model] = current - 1

    def in_flight(self, model: str) -> int:
        """Current in-flight count for ``model`` (0 if unseen)."""
        with self._lock:
            return self._inflight.get(model, 0)

    def select_deployment(
        self,
        context: RoutingContext,
    ) -> Optional[Dict]:
        candidates = self._get_candidates(context)
        if not candidates:
            return None

        with self._lock:
            best: Optional[Dict] = None
            best_count = float("inf")
            for dep in candidates:
                model = self._arm_key(dep)
                c = self._inflight.get(model, 0)
                if c < best_count:
                    best_count = c
                    best = dep
            selected = best or candidates[0]
            sel_model = self._arm_key(selected)
            # Acquire the slot under the same lock so concurrent selects spread.
            self._inflight[sel_model] = self._inflight.get(sel_model, 0) + 1

        _record_selection_metric("least-busy", sel_model)
        return selected


def register_cost_cascade_strategy() -> bool:
    """Register the cost-aware cascade strategy in the routing registry.

    Makes :class:`CostCascadeRoutingStrategy` available as
    ``"llmrouter-cost-cascade"`` for direct selection / A/B testing. Opt-in:
    only registers when ``settings.cost_cascade.enabled`` is True (RouteIQ-90d0).

    Returns:
        True if registration succeeded, False otherwise (disabled / error).
    """
    try:
        from litellm_llmrouter.settings import get_settings

        cfg = get_settings().cost_cascade
    except Exception:
        return False

    if not cfg.enabled:
        verbose_proxy_logger.debug(
            "Cost-cascade routing disabled (ROUTEIQ_COST_CASCADE__ENABLED=false)"
        )
        return False

    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        registry.register(
            "llmrouter-cost-cascade",
            CostCascadeRoutingStrategy(
                confidence_threshold=cfg.confidence_threshold,
                max_rungs=cfg.max_rungs,
            ),
        )
        verbose_proxy_logger.info(
            "Registered cost-cascade strategy as 'llmrouter-cost-cascade'"
        )
        return True
    except Exception as e:  # pragma: no cover - defensive
        verbose_proxy_logger.warning(f"Failed to register cost-cascade strategy: {e}")
        return False


def register_semantic_intent_strategy() -> bool:
    """Register the semantic intent router in the routing registry.

    Makes :class:`SemanticIntentRoutingStrategy` available as
    ``"llmrouter-semantic-intent"`` for direct selection / A/B testing. Opt-in:
    only registers when ``settings.semantic_intent.enabled`` is True
    (RouteIQ-7936).

    Returns:
        True if registration succeeded, False otherwise (disabled / error).
    """
    try:
        from litellm_llmrouter.settings import get_settings

        cfg = get_settings().semantic_intent
    except Exception:
        return False

    if not cfg.enabled:
        verbose_proxy_logger.debug(
            "Semantic-intent routing disabled (ROUTEIQ_SEMANTIC_INTENT__ENABLED=false)"
        )
        return False

    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        registry.register(
            "llmrouter-semantic-intent",
            SemanticIntentRoutingStrategy(
                intent_model_groups=cfg.intent_model_groups,
                similarity_threshold=cfg.similarity_threshold,
            ),
        )
        verbose_proxy_logger.info(
            "Registered semantic-intent strategy as 'llmrouter-semantic-intent'"
        )
        return True
    except Exception as e:  # pragma: no cover - defensive
        verbose_proxy_logger.warning(
            f"Failed to register semantic-intent strategy: {e}"
        )
        return False


def register_tag_routing_strategy() -> bool:
    """Register the tag/regex/User-Agent router in the routing registry.

    Makes :class:`TagRegexUserAgentRoutingStrategy` available as
    ``"llmrouter-tag-regex-ua"`` for direct selection / A/B testing. Opt-in:
    only registers when ``settings.tag_routing.enabled`` is True (RouteIQ-6865).

    Returns:
        True if registration succeeded, False otherwise (disabled / error).
    """
    try:
        from litellm_llmrouter.settings import get_settings

        cfg = get_settings().tag_routing
    except Exception:
        return False

    if not cfg.enabled:
        verbose_proxy_logger.debug(
            "Tag routing disabled (ROUTEIQ_TAG_ROUTING__ENABLED=false)"
        )
        return False

    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        registry.register(
            "llmrouter-tag-regex-ua",
            TagRegexUserAgentRoutingStrategy(
                tag_model_groups=cfg.tag_model_groups,
                regex_model_groups=cfg.regex_model_groups,
                user_agent_model_groups=cfg.user_agent_model_groups,
            ),
        )
        verbose_proxy_logger.info(
            "Registered tag routing strategy as 'llmrouter-tag-regex-ua'"
        )
        return True
    except Exception as e:  # pragma: no cover - defensive
        verbose_proxy_logger.warning(f"Failed to register tag routing strategy: {e}")
        return False


def register_latency_aware_strategies() -> bool:
    """Register the latency-SLA / usage-based / least-busy routers (RouteIQ-904b).

    Makes :class:`LatencySLARoutingStrategy` (``"llmrouter-latency-sla"``),
    :class:`UsageBasedRoutingStrategy` (``"llmrouter-usage-based"``), and
    :class:`LeastBusyRoutingStrategy` (``"llmrouter-least-busy"``) available for
    direct selection / A/B testing. Opt-in: only registers when
    ``settings.latency_aware.enabled`` is True. All three are registered
    together (one flag) but each under its own registry name.

    Returns:
        True if registration succeeded, False otherwise (disabled / error).
    """
    try:
        from litellm_llmrouter.settings import get_settings

        cfg = get_settings().latency_aware
    except Exception:
        return False

    if not cfg.enabled:
        verbose_proxy_logger.debug(
            "Latency-aware routing disabled (ROUTEIQ_LATENCY_AWARE__ENABLED=false)"
        )
        return False

    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        registry.register(
            "llmrouter-latency-sla",
            LatencySLARoutingStrategy(
                p_latency_target_ms=cfg.p_latency_target_ms,
                percentile=cfg.percentile,
                max_latency_samples=cfg.max_latency_samples,
            ),
        )
        registry.register("llmrouter-usage-based", UsageBasedRoutingStrategy())
        registry.register("llmrouter-least-busy", LeastBusyRoutingStrategy())
        verbose_proxy_logger.info(
            "Registered latency-sla / usage-based / least-busy strategies"
        )
        return True
    except Exception as e:  # pragma: no cover - defensive
        verbose_proxy_logger.warning(
            f"Failed to register latency-aware strategies: {e}"
        )
        return False


def register_linucb_strategy() -> bool:
    """Register the LinUCB contextual-bandit router (RouteIQ-6c67).

    Makes :class:`~litellm_llmrouter.kumaraswamy_thompson.LinUCBRoutingStrategy`
    available as ``"llmrouter-linucb"`` for direct selection / A/B testing.
    Opt-in: only registers when ``settings.linucb.enabled`` is True. ADDITIVE
    alongside the Kumaraswamy-Thompson bandit (independent flag / name).

    Returns:
        True if registration succeeded, False otherwise (disabled / error).
    """
    try:
        from litellm_llmrouter.kumaraswamy_thompson import (
            register_linucb_strategy as _reg,
        )

        return _reg()
    except Exception as e:  # pragma: no cover - defensive
        verbose_proxy_logger.warning(f"Failed to register LinUCB strategy: {e}")
        return False


def register_llmrouter_strategies():
    """
    Log available LLMRouter strategies and return their names.

    Despite its name, the ``llmrouter-*`` ML strategies are activated at request
    time through the ``RouteIQRoutingStrategy`` plugin (see
    ``custom_routing_strategy.py``); this function does NOT register them in the
    runtime registry.

    It DOES opt-in-register the RouteIQ-native, registry-backed strategies
    when enabled via settings:
    - ``llmrouter-cost-cascade`` (RouteIQ-90d0) via
      :func:`register_cost_cascade_strategy`
    - ``llmrouter-semantic-intent`` (RouteIQ-7936) via
      :func:`register_semantic_intent_strategy`
    - ``llmrouter-tag-regex-ua`` (RouteIQ-6865) via
      :func:`register_tag_routing_strategy`
    - ``llmrouter-latency-sla`` / ``llmrouter-usage-based`` /
      ``llmrouter-least-busy`` (RouteIQ-904b) via
      :func:`register_latency_aware_strategies`
    - ``llmrouter-linucb`` (RouteIQ-6c67) via :func:`register_linucb_strategy`

    This mirrors the centroid pattern (``register_centroid_strategy``): each is
    a no-op unless its respective ``enabled`` flag is set, so default startup
    behavior is unchanged.

    Returns:
        List of available strategy name strings (e.g. ["llmrouter-knn", ...]).
    """
    verbose_proxy_logger.info(
        f"\u2705 {len(LLMROUTER_STRATEGIES)} LLMRouter strategies available "
        "(activated via RouteIQRoutingStrategy plugin)"
    )

    # Log available strategies
    for strategy in LLMROUTER_STRATEGIES:
        verbose_proxy_logger.debug(f"  - {strategy}")

    # Opt-in registry-backed RouteIQ-native strategies (no-op unless enabled).
    register_cost_cascade_strategy()
    register_semantic_intent_strategy()
    register_tag_routing_strategy()
    register_latency_aware_strategies()
    register_linucb_strategy()

    return LLMROUTER_STRATEGIES
