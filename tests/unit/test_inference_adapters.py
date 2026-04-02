"""
Tests for inference-only routing adapters (SVM, MLP, MF, ELO).

These tests verify that:
1. InferenceSVMRouter loads sklearn SVC models directly without UIUC LLMRouter deps
2. InferenceMLPRouter loads PyTorch MLP state_dicts without MetaRouter
3. InferenceMFRouter loads PyTorch BilinearMF state_dicts without MetaRouter
4. InferenceELORouter loads pre-computed Elo ratings from JSON/pickle
5. All adapters use the same security controls as InferenceKNNRouter
6. Hot reload triggers properly based on file mtime changes
7. Label mapping works as expected
8. LLMRouterStrategyFamily dispatches to inference adapters when use_inference_only=True
"""

import json
import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Check for required packages
try:
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_pickle_loading(monkeypatch):
    """Enable pickle loading for tests and disable signed model enforcement."""
    monkeypatch.setenv("LLMROUTER_ALLOW_PICKLE_MODELS", "true")
    monkeypatch.setenv("LLMROUTER_ENFORCE_SIGNED_MODELS", "false")
    import litellm_llmrouter.strategies as strategies_module

    monkeypatch.setattr(strategies_module, "ALLOW_PICKLE_MODELS", True)
    monkeypatch.setattr(strategies_module, "ENFORCE_SIGNED_MODELS", False)
    yield


@pytest.fixture
def mock_embedder():
    """Create a mock sentence-transformer that returns fixed embeddings."""
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.5, 0.3, 0.8]])
    return embedder


# --- SVM fixtures ---


@pytest.fixture
def trained_svm_model():
    """Create a trained sklearn SVC model."""
    svm = SVC(kernel="rbf", C=1.0)
    X_train = np.array(
        [[0.1, 0.2, 0.3], [0.8, 0.9, 0.7], [0.4, 0.5, 0.6], [0.2, 0.1, 0.9]]
    )
    y_train = ["model-a", "model-b", "model-a", "model-b"]
    svm.fit(X_train, y_train)
    return svm


@pytest.fixture
def svm_pkl_file(trained_svm_model):
    """Save trained SVM to a temp .pkl file."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(trained_svm_model, f)
        path = f.name
    yield path
    os.unlink(path)


# --- MLP fixtures ---


@pytest.fixture
def mlp_model_and_metadata():
    """Create a trained MLP model (state_dict) and companion metadata JSON."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from litellm_llmrouter.strategies import _build_mlp_classifier

    input_dim = 3
    hidden_layer_sizes = [8, 4]
    num_classes = 2

    model = _build_mlp_classifier(input_dim, hidden_layer_sizes, num_classes, "relu")

    # Save state_dict
    model_fd, model_path = tempfile.mkstemp(suffix=".pt")
    os.close(model_fd)
    torch.save(model.state_dict(), model_path)

    # Save metadata JSON
    metadata_path = model_path.replace(".pt", ".json")
    metadata = {
        "input_dim": input_dim,
        "hidden_layer_sizes": hidden_layer_sizes,
        "num_classes": num_classes,
        "activation": "relu",
        "idx_to_model": {"0": "gpt-4", "1": "claude-3-opus"},
        "model_to_idx": {"gpt-4": 0, "claude-3-opus": 1},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    yield model_path, metadata_path

    os.unlink(model_path)
    if os.path.exists(metadata_path):
        os.unlink(metadata_path)


# --- MF fixtures ---


@pytest.fixture
def mf_model_and_metadata():
    """Create a trained BilinearMF model (state_dict) and companion metadata JSON."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from litellm_llmrouter.strategies import _build_bilinear_mf

    latent_dim = 8
    num_models = 3
    text_dim = 3  # Matching our mock embedder output dim

    model = _build_bilinear_mf(latent_dim, num_models, text_dim)

    # Save state_dict
    model_fd, model_path = tempfile.mkstemp(suffix=".pt")
    os.close(model_fd)
    torch.save(model.state_dict(), model_path)

    # Save metadata JSON
    metadata_path = model_path.replace(".pt", ".json")
    metadata = {
        "latent_dim": latent_dim,
        "text_dim": text_dim,
        "num_models": num_models,
        "idx_to_model": {"0": "gpt-4", "1": "claude-3-opus", "2": "llama-3"},
        "model_to_idx": {"gpt-4": 0, "claude-3-opus": 1, "llama-3": 2},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    yield model_path, metadata_path

    os.unlink(model_path)
    if os.path.exists(metadata_path):
        os.unlink(metadata_path)


# --- ELO fixtures ---


@pytest.fixture
def elo_json_file():
    """Create a JSON file with Elo ratings."""
    ratings = {
        "gpt-4": 1650.5,
        "claude-3-opus": 1580.2,
        "llama-3-70b": 1420.0,
        "gemini-pro": 1510.8,
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(ratings, f)
    yield path
    os.unlink(path)


@pytest.fixture
def elo_pkl_file():
    """Create a pickle file with Elo ratings."""
    ratings = {
        "gpt-4": 1650.5,
        "claude-3-opus": 1580.2,
        "llama-3-70b": 1420.0,
    }
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    with open(path, "wb") as f:
        pickle.dump(ratings, f)
    yield path
    os.unlink(path)


# ===========================================================================
# InferenceSVMRouter Tests
# ===========================================================================


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestInferenceSVMRouter:
    """Test the InferenceSVMRouter class."""

    def test_load_model_from_pkl(self, mock_embedder, svm_pkl_file):
        """Test that InferenceSVMRouter loads a sklearn SVC model from .pkl file."""
        from litellm_llmrouter.strategies import InferenceSVMRouter

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceSVMRouter(model_path=svm_pkl_file)

        assert router.svm_model is not None
        assert hasattr(router.svm_model, "predict")
        assert router.model_version is not None

    def test_route_query_returns_predicted_label(self, mock_embedder, svm_pkl_file):
        """Test routing returns a valid predicted label."""
        from litellm_llmrouter.strategies import InferenceSVMRouter

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceSVMRouter(model_path=svm_pkl_file)
            result = router.route("What is the weather?")

        assert result is not None
        assert result in ("model-a", "model-b")

    def test_route_with_label_mapping(self, mock_embedder, svm_pkl_file):
        """Test label mapping transforms predicted labels."""
        from litellm_llmrouter.strategies import InferenceSVMRouter

        mapping = {"model-a": "openai/gpt-4", "model-b": "anthropic/claude-3"}
        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceSVMRouter(
                model_path=svm_pkl_file,
                label_mapping=mapping,
            )
            result = router.route("Test query")

        assert result in ("openai/gpt-4", "anthropic/claude-3")

    def test_pickle_security_blocks_loading(self, monkeypatch, svm_pkl_file):
        """Test that pickle loading is blocked when env var is not set."""
        import litellm_llmrouter.strategies as strategies_module
        from litellm_llmrouter.strategies import InferenceSVMRouter, PickleSecurityError

        monkeypatch.setattr(strategies_module, "ALLOW_PICKLE_MODELS", False)

        with pytest.raises(PickleSecurityError):
            InferenceSVMRouter(model_path=svm_pkl_file)

    def test_model_file_not_found_raises_error(self):
        """Test that FileNotFoundError is raised for missing model file."""
        from litellm_llmrouter.strategies import InferenceSVMRouter

        with pytest.raises(FileNotFoundError):
            InferenceSVMRouter(model_path="/nonexistent/svm_model.pkl")

    def test_reload_model(self, mock_embedder, svm_pkl_file):
        """Test that reload_model works and returns True on success."""
        from litellm_llmrouter.strategies import InferenceSVMRouter

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceSVMRouter(model_path=svm_pkl_file)
            result = router.reload_model()

        assert result is True
        assert router.model_version is not None

    def test_invalid_model_raises_error(self):
        """Test that a non-sklearn model raises ModelLoadError."""
        from litellm_llmrouter.strategies import InferenceSVMRouter, ModelLoadError

        # Save a non-model object
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "a model"}, f)
            path = f.name

        try:
            with pytest.raises(ModelLoadError, match="predict"):
                InferenceSVMRouter(model_path=path)
        finally:
            os.unlink(path)


# ===========================================================================
# InferenceMLPRouter Tests
# ===========================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestInferenceMLPRouter:
    """Test the InferenceMLPRouter class."""

    def test_load_model_from_state_dict(self, mock_embedder, mlp_model_and_metadata):
        """Test that InferenceMLPRouter loads a PyTorch model from state_dict."""
        from litellm_llmrouter.strategies import InferenceMLPRouter

        model_path, metadata_path = mlp_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceMLPRouter(
                model_path=model_path,
                metadata_path=metadata_path,
            )

        assert router.mlp_model is not None
        assert router.model_version is not None
        assert router._num_classes == 2
        assert router.idx_to_model == {0: "gpt-4", 1: "claude-3-opus"}

    def test_route_query_returns_valid_model(
        self, mock_embedder, mlp_model_and_metadata
    ):
        """Test routing returns a valid model name from idx_to_model mapping."""
        from litellm_llmrouter.strategies import InferenceMLPRouter

        model_path, metadata_path = mlp_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceMLPRouter(
                model_path=model_path,
                metadata_path=metadata_path,
            )
            result = router.route("What is machine learning?")

        assert result is not None
        assert result in ("gpt-4", "claude-3-opus")

    def test_does_not_require_pickle_models_env(
        self, monkeypatch, mock_embedder, mlp_model_and_metadata
    ):
        """Test that MLP loading does NOT require LLMROUTER_ALLOW_PICKLE_MODELS."""
        import litellm_llmrouter.strategies as strategies_module
        from litellm_llmrouter.strategies import InferenceMLPRouter

        # Disable pickle loading
        monkeypatch.setattr(strategies_module, "ALLOW_PICKLE_MODELS", False)

        model_path, metadata_path = mlp_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            # Should NOT raise PickleSecurityError since MLP uses torch.load(weights_only=True)
            router = InferenceMLPRouter(
                model_path=model_path,
                metadata_path=metadata_path,
            )
        assert router.mlp_model is not None

    def test_model_file_not_found_raises_error(self):
        """Test that FileNotFoundError is raised for missing model file."""
        from litellm_llmrouter.strategies import InferenceMLPRouter

        with pytest.raises(FileNotFoundError):
            InferenceMLPRouter(
                model_path="/nonexistent/mlp_model.pt",
                num_classes=2,
            )

    def test_missing_num_classes_raises_error(self):
        """Test that ValueError is raised when num_classes is not provided."""
        from litellm_llmrouter.strategies import InferenceMLPRouter

        # Create a model file but no metadata with num_classes
        fd, model_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        if TORCH_AVAILABLE:
            torch.save({}, model_path)

        try:
            with pytest.raises(ValueError, match="num_classes"):
                InferenceMLPRouter(model_path=model_path)
        finally:
            os.unlink(model_path)

    def test_reload_model(self, mock_embedder, mlp_model_and_metadata):
        """Test that reload_model works for MLP."""
        from litellm_llmrouter.strategies import InferenceMLPRouter

        model_path, metadata_path = mlp_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceMLPRouter(
                model_path=model_path,
                metadata_path=metadata_path,
            )
            result = router.reload_model()

        assert result is True

    def test_route_with_label_mapping(self, mock_embedder, mlp_model_and_metadata):
        """Test label mapping transforms predicted labels for MLP."""
        from litellm_llmrouter.strategies import InferenceMLPRouter

        model_path, metadata_path = mlp_model_and_metadata
        mapping = {"gpt-4": "openai/gpt-4-turbo", "claude-3-opus": "anthropic/claude"}

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceMLPRouter(
                model_path=model_path,
                metadata_path=metadata_path,
                label_mapping=mapping,
            )
            result = router.route("Test")

        assert result in ("openai/gpt-4-turbo", "anthropic/claude")

    def test_constructor_provided_params(self, mock_embedder):
        """Test loading without metadata file, using constructor params."""
        from litellm_llmrouter.strategies import (
            InferenceMLPRouter,
            _build_mlp_classifier,
        )

        input_dim = 3
        hidden_layer_sizes = [4]
        num_classes = 2
        idx_to_model = {0: "model-a", 1: "model-b"}

        model = _build_mlp_classifier(
            input_dim, hidden_layer_sizes, num_classes, "relu"
        )
        fd, model_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(model.state_dict(), model_path)

        try:
            with patch(
                "litellm_llmrouter.strategies._get_sentence_transformer",
                return_value=mock_embedder,
            ):
                router = InferenceMLPRouter(
                    model_path=model_path,
                    input_dim=input_dim,
                    hidden_layer_sizes=hidden_layer_sizes,
                    num_classes=num_classes,
                    idx_to_model=idx_to_model,
                )
                result = router.route("Hello")

            assert result in ("model-a", "model-b")
        finally:
            os.unlink(model_path)


# ===========================================================================
# InferenceMFRouter Tests
# ===========================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestInferenceMFRouter:
    """Test the InferenceMFRouter class."""

    def test_load_model_from_state_dict(self, mock_embedder, mf_model_and_metadata):
        """Test that InferenceMFRouter loads a PyTorch model from state_dict."""
        from litellm_llmrouter.strategies import InferenceMFRouter

        model_path, metadata_path = mf_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceMFRouter(
                model_path=model_path,
                metadata_path=metadata_path,
            )

        assert router.mf_model is not None
        assert router.model_version is not None
        assert router._num_models == 3
        assert router.idx_to_model == {0: "gpt-4", 1: "claude-3-opus", 2: "llama-3"}

    def test_route_query_returns_valid_model(
        self, mock_embedder, mf_model_and_metadata
    ):
        """Test routing returns a valid model name."""
        from litellm_llmrouter.strategies import InferenceMFRouter

        model_path, metadata_path = mf_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceMFRouter(
                model_path=model_path,
                metadata_path=metadata_path,
            )
            result = router.route("Explain quantum computing")

        assert result is not None
        assert result in ("gpt-4", "claude-3-opus", "llama-3")

    def test_does_not_require_pickle_models_env(
        self, monkeypatch, mock_embedder, mf_model_and_metadata
    ):
        """Test that MF loading does NOT require LLMROUTER_ALLOW_PICKLE_MODELS."""
        import litellm_llmrouter.strategies as strategies_module
        from litellm_llmrouter.strategies import InferenceMFRouter

        monkeypatch.setattr(strategies_module, "ALLOW_PICKLE_MODELS", False)

        model_path, metadata_path = mf_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceMFRouter(
                model_path=model_path,
                metadata_path=metadata_path,
            )
        assert router.mf_model is not None

    def test_model_file_not_found_raises_error(self):
        """Test that FileNotFoundError is raised for missing model file."""
        from litellm_llmrouter.strategies import InferenceMFRouter

        with pytest.raises(FileNotFoundError):
            InferenceMFRouter(
                model_path="/nonexistent/mf_model.pt",
                num_models=3,
            )

    def test_reload_model(self, mock_embedder, mf_model_and_metadata):
        """Test that reload_model works for MF."""
        from litellm_llmrouter.strategies import InferenceMFRouter

        model_path, metadata_path = mf_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceMFRouter(
                model_path=model_path,
                metadata_path=metadata_path,
            )
            result = router.reload_model()

        assert result is True

    def test_route_with_label_mapping(self, mock_embedder, mf_model_and_metadata):
        """Test label mapping transforms predicted labels for MF."""
        from litellm_llmrouter.strategies import InferenceMFRouter

        model_path, metadata_path = mf_model_and_metadata
        mapping = {
            "gpt-4": "openai/gpt-4",
            "claude-3-opus": "anthropic/claude",
            "llama-3": "meta/llama",
        }

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceMFRouter(
                model_path=model_path,
                metadata_path=metadata_path,
                label_mapping=mapping,
            )
            result = router.route("Test")

        assert result in ("openai/gpt-4", "anthropic/claude", "meta/llama")


# ===========================================================================
# InferenceELORouter Tests
# ===========================================================================


class TestInferenceELORouter:
    """Test the InferenceELORouter class."""

    def test_load_ratings_from_json(self, elo_json_file):
        """Test that ELO ratings load from JSON file."""
        from litellm_llmrouter.strategies import InferenceELORouter

        router = InferenceELORouter(ratings_path=elo_json_file)

        assert router.elo_scores is not None
        assert len(router.elo_scores) == 4
        assert router.model_version is not None

    def test_load_ratings_from_pickle(self, elo_pkl_file):
        """Test that ELO ratings load from pickle file."""
        from litellm_llmrouter.strategies import InferenceELORouter

        router = InferenceELORouter(ratings_path=elo_pkl_file)

        assert router.elo_scores is not None
        assert len(router.elo_scores) == 3

    def test_route_returns_highest_rated_model(self, elo_json_file):
        """Test that routing returns the model with the highest Elo score."""
        from litellm_llmrouter.strategies import InferenceELORouter

        router = InferenceELORouter(ratings_path=elo_json_file)
        result = router.route("Any query — ELO is query-independent")

        # gpt-4 has the highest rating (1650.5)
        assert result == "gpt-4"

    def test_route_is_query_independent(self, elo_json_file):
        """Test that ELO returns the same model regardless of query content."""
        from litellm_llmrouter.strategies import InferenceELORouter

        router = InferenceELORouter(ratings_path=elo_json_file)

        result1 = router.route("What is AI?")
        result2 = router.route("Write a poem about cats")
        result3 = router.route("Debug my Python code")

        assert result1 == result2 == result3 == "gpt-4"

    def test_route_with_label_mapping(self, elo_json_file):
        """Test label mapping for ELO router."""
        from litellm_llmrouter.strategies import InferenceELORouter

        mapping = {"gpt-4": "openai/gpt-4-turbo"}
        router = InferenceELORouter(
            ratings_path=elo_json_file,
            label_mapping=mapping,
        )
        result = router.route("Test")

        assert result == "openai/gpt-4-turbo"

    def test_pickle_security_blocks_loading(self, monkeypatch, elo_pkl_file):
        """Test that pickle ELO loading is blocked when env var is not set."""
        import litellm_llmrouter.strategies as strategies_module
        from litellm_llmrouter.strategies import InferenceELORouter, PickleSecurityError

        monkeypatch.setattr(strategies_module, "ALLOW_PICKLE_MODELS", False)

        with pytest.raises(PickleSecurityError):
            InferenceELORouter(ratings_path=elo_pkl_file)

    def test_json_does_not_require_pickle_env(self, monkeypatch, elo_json_file):
        """Test that JSON ELO loading does NOT require pickle env var."""
        import litellm_llmrouter.strategies as strategies_module
        from litellm_llmrouter.strategies import InferenceELORouter

        monkeypatch.setattr(strategies_module, "ALLOW_PICKLE_MODELS", False)

        # Should NOT raise — JSON is safe
        router = InferenceELORouter(ratings_path=elo_json_file)
        assert router.elo_scores is not None

    def test_ratings_file_not_found_raises_error(self):
        """Test that FileNotFoundError is raised for missing ratings file."""
        from litellm_llmrouter.strategies import InferenceELORouter

        with pytest.raises(FileNotFoundError):
            InferenceELORouter(ratings_path="/nonexistent/elo_ratings.json")

    def test_empty_ratings_raises_error(self):
        """Test that empty ratings file raises ModelLoadError."""
        from litellm_llmrouter.strategies import InferenceELORouter, ModelLoadError

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump({}, f)

        try:
            with pytest.raises(ModelLoadError, match="empty"):
                InferenceELORouter(ratings_path=path)
        finally:
            os.unlink(path)

    def test_invalid_ratings_type_raises_error(self):
        """Test that non-dict ratings raise ModelLoadError."""
        from litellm_llmrouter.strategies import InferenceELORouter, ModelLoadError

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)

        try:
            with pytest.raises(ModelLoadError, match="dict"):
                InferenceELORouter(ratings_path=path)
        finally:
            os.unlink(path)

    def test_reload_ratings(self, elo_json_file):
        """Test that reload_model works for ELO."""
        from litellm_llmrouter.strategies import InferenceELORouter

        router = InferenceELORouter(ratings_path=elo_json_file)
        result = router.reload_model()

        assert result is True


# ===========================================================================
# LLMRouterStrategyFamily Integration Tests
# ===========================================================================


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestStrategyFamilySVMDispatch:
    """Test that LLMRouterStrategyFamily dispatches to InferenceSVMRouter."""

    def test_svm_uses_inference_adapter_by_default(self, mock_embedder, svm_pkl_file):
        """Test that llmrouter-svm uses InferenceSVMRouter by default."""
        from litellm_llmrouter.strategies import (
            InferenceSVMRouter,
            LLMRouterStrategyFamily,
        )

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            family = LLMRouterStrategyFamily(
                strategy_name="llmrouter-svm",
                model_path=svm_pkl_file,
            )
            router = family.router

        assert isinstance(router, InferenceSVMRouter)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestStrategyFamilyMLPDispatch:
    """Test that LLMRouterStrategyFamily dispatches to InferenceMLPRouter."""

    def test_mlp_uses_inference_adapter_by_default(
        self, mock_embedder, mlp_model_and_metadata
    ):
        """Test that llmrouter-mlp uses InferenceMLPRouter by default."""
        from litellm_llmrouter.strategies import (
            InferenceMLPRouter,
            LLMRouterStrategyFamily,
        )

        model_path, metadata_path = mlp_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            family = LLMRouterStrategyFamily(
                strategy_name="llmrouter-mlp",
                model_path=model_path,
                metadata_path=metadata_path,
            )
            router = family.router

        assert isinstance(router, InferenceMLPRouter)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestStrategyFamilyMFDispatch:
    """Test that LLMRouterStrategyFamily dispatches to InferenceMFRouter."""

    def test_mf_uses_inference_adapter_by_default(
        self, mock_embedder, mf_model_and_metadata
    ):
        """Test that llmrouter-mf uses InferenceMFRouter by default."""
        from litellm_llmrouter.strategies import (
            InferenceMFRouter,
            LLMRouterStrategyFamily,
        )

        model_path, metadata_path = mf_model_and_metadata

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            family = LLMRouterStrategyFamily(
                strategy_name="llmrouter-mf",
                model_path=model_path,
                metadata_path=metadata_path,
            )
            router = family.router

        assert isinstance(router, InferenceMFRouter)


class TestStrategyFamilyELODispatch:
    """Test that LLMRouterStrategyFamily dispatches to InferenceELORouter."""

    def test_elo_uses_inference_adapter_by_default(self, elo_json_file):
        """Test that llmrouter-elo uses InferenceELORouter by default."""
        from litellm_llmrouter.strategies import (
            InferenceELORouter,
            LLMRouterStrategyFamily,
        )

        family = LLMRouterStrategyFamily(
            strategy_name="llmrouter-elo",
            model_path=elo_json_file,
        )
        router = family.router

        assert isinstance(router, InferenceELORouter)

    def test_elo_model_path_required(self):
        """Test that model_path is required for ELO inference."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        family = LLMRouterStrategyFamily(
            strategy_name="llmrouter-elo",
            model_path=None,
        )
        # Should return None, not crash
        assert family.router is None


# ===========================================================================
# PyTorch Model Builder Tests
# ===========================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestModelBuilders:
    """Test the standalone PyTorch model builder functions."""

    def test_build_mlp_classifier(self):
        """Test _build_mlp_classifier creates valid model."""
        from litellm_llmrouter.strategies import _build_mlp_classifier

        model = _build_mlp_classifier(
            input_dim=768,
            hidden_layer_sizes=[128, 64],
            num_classes=5,
            activation="relu",
        )

        # Verify architecture
        assert hasattr(model, "layers")
        assert len(model.layers) == 3  # 2 hidden + 1 output

        # Verify forward pass works
        x = torch.randn(1, 768)
        output = model(x)
        assert output.shape == (1, 5)

        # Verify predict works
        pred = model.predict(x)
        assert pred.shape == (1,)
        assert 0 <= pred.item() < 5

    def test_build_bilinear_mf(self):
        """Test _build_bilinear_mf creates valid model."""
        from litellm_llmrouter.strategies import _build_bilinear_mf

        model = _build_bilinear_mf(
            latent_dim=64,
            num_models=10,
            text_dim=768,
        )

        # Verify architecture
        assert hasattr(model, "P")
        assert hasattr(model, "text_proj")
        assert hasattr(model, "classifier")

        # Verify project_text works
        q_emb = torch.randn(768)
        proj = model.project_text(q_emb)
        assert proj.shape == (64,)

        # Verify score_all works
        scores = model.score_all(proj)
        assert scores.shape == (10,)

    def test_mlp_state_dict_round_trip(self):
        """Test that saving and loading MLP state_dict preserves weights."""
        from litellm_llmrouter.strategies import _build_mlp_classifier

        model1 = _build_mlp_classifier(64, [32], 3, "relu")
        x = torch.randn(1, 64)
        output1 = model1(x)

        # Save and reload
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(model1.state_dict(), path)

        model2 = _build_mlp_classifier(64, [32], 3, "relu")
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        output2 = model2(x)

        os.unlink(path)

        assert torch.allclose(output1, output2)

    def test_bilinear_mf_state_dict_round_trip(self):
        """Test that saving and loading BilinearMF state_dict preserves weights."""
        from litellm_llmrouter.strategies import _build_bilinear_mf

        model1 = _build_bilinear_mf(32, 5, 64)
        q_emb = torch.randn(64)
        proj1 = model1.project_text(q_emb)
        scores1 = model1.score_all(proj1)

        # Save and reload
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(model1.state_dict(), path)

        model2 = _build_bilinear_mf(32, 5, 64)
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        proj2 = model2.project_text(q_emb)
        scores2 = model2.score_all(proj2)

        os.unlink(path)

        assert torch.allclose(scores1, scores2)
