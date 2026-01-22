"""
Tests for the inference-only KNN router.

These tests verify that:
1. InferenceKNNRouter loads sklearn models directly without UIUC LLMRouter deps
2. The router correctly embeds queries and predicts labels
3. Hot reload triggers properly based on file mtime changes
4. Label mapping works as expected
"""

import os
import pickle
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# Check if sklearn is available
try:
    from sklearn.neighbors import KNeighborsClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SKLEARN_AVAILABLE, reason="scikit-learn package not installed"
)


class TestInferenceKNNRouter:
    """Test the InferenceKNNRouter class."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock sentence transformer embedder."""
        mock = MagicMock()
        # Return a simple 3D embedding for testing
        mock.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        return mock

    @pytest.fixture
    def trained_knn_model(self):
        """Create a simple trained KNN model for testing."""
        # Create a simple KNN classifier with 3D embeddings
        knn = KNeighborsClassifier(n_neighbors=2)
        # Training data: 3D embeddings -> labels
        X_train = np.array(
            [
                [0.1, 0.2, 0.3],  # -> claude-sonnet
                [0.15, 0.25, 0.35],  # -> claude-sonnet
                [0.8, 0.9, 0.7],  # -> nova-pro
                [0.85, 0.95, 0.75],  # -> nova-pro
            ]
        )
        y_train = ["claude-sonnet", "claude-sonnet", "nova-pro", "nova-pro"]
        knn.fit(X_train, y_train)
        return knn

    @pytest.fixture
    def model_pkl_file(self, trained_knn_model):
        """Save trained model to a temporary .pkl file."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(trained_knn_model, f)
            pkl_path = f.name
        yield pkl_path
        # Cleanup
        if os.path.exists(pkl_path):
            os.unlink(pkl_path)

    def test_load_model_from_pkl(self, model_pkl_file, mock_embedder):
        """Test that InferenceKNNRouter loads a sklearn model from .pkl file."""
        from litellm_llmrouter.strategies import InferenceKNNRouter

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceKNNRouter(
                model_path=model_pkl_file,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                embedding_device="cpu",
            )

        # Verify model was loaded
        assert router.knn_model is not None
        assert hasattr(router.knn_model, "predict")
        assert isinstance(router.knn_model, KNeighborsClassifier)

    def test_route_query_returns_predicted_label(self, model_pkl_file, mock_embedder):
        """Test that routing a query returns the predicted label."""
        from litellm_llmrouter.strategies import InferenceKNNRouter

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceKNNRouter(
                model_path=model_pkl_file,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            )

            # Route a test query - mock embedder returns [0.1, 0.2, 0.3]
            # which should be closest to "claude-sonnet" training data
            result = router.route("What is the capital of France?")

        # Should return a label
        assert result is not None
        assert result == "claude-sonnet"

    def test_route_with_label_mapping(self, model_pkl_file, mock_embedder):
        """Test that label mapping transforms the predicted label."""
        from litellm_llmrouter.strategies import InferenceKNNRouter

        label_mapping = {
            "claude-sonnet": "bedrock/anthropic.claude-sonnet-v1",
            "nova-pro": "bedrock/amazon.nova-pro-v1",
        }

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            router = InferenceKNNRouter(
                model_path=model_pkl_file,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                label_mapping=label_mapping,
            )

            result = router.route("What is the capital of France?")

        # Should return the mapped label
        assert result == "bedrock/anthropic.claude-sonnet-v1"

    def test_model_file_not_found_raises_error(self):
        """Test that FileNotFoundError is raised for non-existent model file."""
        from litellm_llmrouter.strategies import InferenceKNNRouter

        with pytest.raises(FileNotFoundError, match="KNN model file not found"):
            InferenceKNNRouter(
                model_path="/nonexistent/path/model.pkl",
            )

    def test_invalid_model_raises_error(self):
        """Test that loading a non-sklearn model raises TypeError."""
        from litellm_llmrouter.strategies import InferenceKNNRouter

        # Create a .pkl file with an invalid object (no predict method)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "a model"}, f)
            pkl_path = f.name

        try:
            with pytest.raises(TypeError, match="does not have 'predict' method"):
                InferenceKNNRouter(model_path=pkl_path)
        finally:
            os.unlink(pkl_path)

    def test_reload_model(self, trained_knn_model, mock_embedder):
        """Test that reload_model updates the model from disk."""
        from litellm_llmrouter.strategies import InferenceKNNRouter

        # Create initial model file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(trained_knn_model, f)
            pkl_path = f.name

        try:
            with patch(
                "litellm_llmrouter.strategies._get_sentence_transformer",
                return_value=mock_embedder,
            ):
                router = InferenceKNNRouter(model_path=pkl_path)

                # Verify initial model
                initial_result = router.route("test query")
                assert initial_result == "claude-sonnet"

                # Create a new model with different training data
                new_knn = KNeighborsClassifier(n_neighbors=1)
                X_train = np.array(
                    [
                        [0.1, 0.2, 0.3],  # -> different-model
                        [0.8, 0.9, 0.7],  # -> other-model
                    ]
                )
                y_train = ["different-model", "other-model"]
                new_knn.fit(X_train, y_train)

                # Save new model to same path
                with open(pkl_path, "wb") as f:
                    pickle.dump(new_knn, f)

                # Reload
                router.reload_model()

                # Verify new model is loaded
                new_result = router.route("test query")
                assert new_result == "different-model"
        finally:
            os.unlink(pkl_path)


class TestLLMRouterStrategyFamilyKNN:
    """Test LLMRouterStrategyFamily with inference-only KNN."""

    @pytest.fixture
    def trained_knn_model(self):
        """Create a simple trained KNN model for testing."""
        knn = KNeighborsClassifier(n_neighbors=2)
        X_train = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.15, 0.25, 0.35],
                [0.8, 0.9, 0.7],
                [0.85, 0.95, 0.75],
            ]
        )
        y_train = ["model-a", "model-a", "model-b", "model-b"]
        knn.fit(X_train, y_train)
        return knn

    @pytest.fixture
    def model_pkl_file(self, trained_knn_model):
        """Save trained model to a temporary .pkl file."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(trained_knn_model, f)
            pkl_path = f.name
        yield pkl_path
        if os.path.exists(pkl_path):
            os.unlink(pkl_path)

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock sentence transformer embedder."""
        mock = MagicMock()
        mock.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        return mock

    def test_knn_uses_inference_only_router_by_default(
        self, model_pkl_file, mock_embedder
    ):
        """Test that llmrouter-knn uses InferenceKNNRouter by default."""
        from litellm_llmrouter.strategies import (
            LLMRouterStrategyFamily,
            InferenceKNNRouter,
        )

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            strategy = LLMRouterStrategyFamily(
                strategy_name="llmrouter-knn",
                model_path=model_pkl_file,
            )

            router = strategy.router

        # Should be InferenceKNNRouter, not UIUC KNNRouter
        assert isinstance(router, InferenceKNNRouter)

    def test_hot_reload_triggers_on_mtime_change(
        self, trained_knn_model, mock_embedder
    ):
        """Test that hot reload triggers when model file mtime changes."""
        from litellm_llmrouter.strategies import (
            LLMRouterStrategyFamily,
            InferenceKNNRouter,
        )

        # Create model file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(trained_knn_model, f)
            pkl_path = f.name

        try:
            with patch(
                "litellm_llmrouter.strategies._get_sentence_transformer",
                return_value=mock_embedder,
            ):
                strategy = LLMRouterStrategyFamily(
                    strategy_name="llmrouter-knn",
                    model_path=pkl_path,
                    hot_reload=True,
                    reload_interval=0,  # No time-based delay for test
                )

                # Initial load
                router1 = strategy.router
                assert isinstance(router1, InferenceKNNRouter)
                initial_mtime = strategy._model_mtime

                # Wait a bit and update the model file
                time.sleep(0.1)

                # Create new model
                new_knn = KNeighborsClassifier(n_neighbors=1)
                X_train = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 0.7]])
                y_train = ["new-model-a", "new-model-b"]
                new_knn.fit(X_train, y_train)

                with open(pkl_path, "wb") as f:
                    pickle.dump(new_knn, f)

                # Access router again - should trigger reload due to mtime change
                router2 = strategy.router

                # mtime should have been updated
                assert strategy._model_mtime > initial_mtime

                # Verify new model is used
                result = router2.route("test")
                assert result == "new-model-a"
        finally:
            os.unlink(pkl_path)

    def test_model_path_required_error(self):
        """Test that error is logged when model_path is not provided."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        strategy = LLMRouterStrategyFamily(
            strategy_name="llmrouter-knn",
            model_path=None,
        )

        # Router should be None since model_path is required
        router = strategy.router
        assert router is None

    def test_config_keys_passed_correctly(self, model_pkl_file, mock_embedder):
        """Test that embedding_model and embedding_device are passed correctly."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        with patch(
            "litellm_llmrouter.strategies._get_sentence_transformer",
            return_value=mock_embedder,
        ):
            strategy = LLMRouterStrategyFamily(
                strategy_name="llmrouter-knn",
                model_path=model_pkl_file,
                embedding_model="custom/embedding-model",
                embedding_device="cuda",
            )

            router = strategy.router

            # Check config was stored correctly
            assert strategy.embedding_model == "custom/embedding-model"
            assert strategy.embedding_device == "cuda"
            assert router.embedding_model == "custom/embedding-model"
            assert router.embedding_device == "cuda"


class TestSentenceTransformerCaching:
    """Test the sentence transformer caching logic."""

    def test_sentence_transformer_loaded_once(self):
        """Test that SentenceTransformer is only loaded once (singleton pattern)."""
        from litellm_llmrouter import strategies

        # Reset the global cache
        strategies._sentence_transformer_model = None

        mock_st_class = MagicMock()
        mock_st_instance = MagicMock()
        mock_st_class.return_value = mock_st_instance

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_class)},
        ):
            # First call should create the model
            result1 = strategies._get_sentence_transformer("test-model", "cpu")

            # Second call should return cached model
            result2 = strategies._get_sentence_transformer("test-model", "cpu")

        # Both should return the same instance
        assert result1 is result2


class TestDirectoryModelResolution:
    """Test model path resolution from directory."""

    def test_resolve_directory_with_single_pkl(self):
        """Test resolving a directory containing a single .pkl file."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a single .pkl file
            pkl_path = os.path.join(tmpdir, "model.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({"dummy": "model"}, f)

            strategy = LLMRouterStrategyFamily(
                strategy_name="llmrouter-knn",
                model_path=tmpdir,  # Pass directory, not file
                use_inference_only=False,  # Avoid actually loading
            )

            # Should have resolved to the .pkl file
            assert strategy.model_path == pkl_path

    def test_resolve_directory_with_multiple_pkl_uses_most_recent(self):
        """Test resolving a directory with multiple .pkl files uses most recent."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create older .pkl file
            old_pkl = os.path.join(tmpdir, "old_model.pkl")
            with open(old_pkl, "wb") as f:
                pickle.dump({"version": "old"}, f)

            time.sleep(0.1)

            # Create newer .pkl file
            new_pkl = os.path.join(tmpdir, "new_model.pkl")
            with open(new_pkl, "wb") as f:
                pickle.dump({"version": "new"}, f)

            strategy = LLMRouterStrategyFamily(
                strategy_name="llmrouter-knn",
                model_path=tmpdir,
                use_inference_only=False,
            )

            # Should have resolved to the newer .pkl file
            assert strategy.model_path == new_pkl

    def test_resolve_file_path_unchanged(self):
        """Test that file paths are used as-is."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"dummy": "model"}, f)
            pkl_path = f.name

        try:
            strategy = LLMRouterStrategyFamily(
                strategy_name="llmrouter-knn",
                model_path=pkl_path,
                use_inference_only=False,
            )

            # Should remain unchanged
            assert strategy.model_path == pkl_path
        finally:
            os.unlink(pkl_path)
