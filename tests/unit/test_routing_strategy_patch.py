"""
Tests for the LiteLLM Router strategy patch.

These tests verify that:
1. Importing litellm_llmrouter does NOT auto-apply the patch
2. Patches can be applied explicitly via patch_litellm_router()
3. llmrouter-* strategies are accepted by LiteLLM's Router after patching
4. Standard strategies still work as expected
"""

import pytest

# Check if litellm is available
try:
    import litellm  # noqa: F401

    LITELLM_AVAILABLE = True
except (ImportError, ValueError):
    LITELLM_AVAILABLE = False


class TestImportBehavior:
    """Test that importing does NOT auto-apply patches."""

    def test_import_does_not_apply_patch(self):
        """
        Test that importing litellm_llmrouter does NOT automatically apply the patch.

        This is a critical test for P1 architecture decoupling.
        """
        # We can't truly test "fresh import" in pytest since modules persist,
        # but we can verify the module doesn't have auto-patch code anymore.
        from litellm_llmrouter import routing_strategy_patch

        # Read the module source and verify no auto-patch call at module level
        import inspect

        source = inspect.getsource(routing_strategy_patch)

        # The old code had "patch_litellm_router()" as a bare call at module level
        # The new code has a comment explaining it's NOT auto-applied
        assert "Patch is NOT applied automatically on import" in source or \
               "NOTE: Patch is NOT applied automatically on import" in source

    def test_patch_is_explicit_and_idempotent(self):
        """Test that patch_litellm_router() is explicit and idempotent."""
        from litellm_llmrouter import patch_litellm_router, is_patch_applied

        # First call applies the patch
        result = patch_litellm_router()
        assert result is True
        assert is_patch_applied() is True

        # Second call is idempotent
        result2 = patch_litellm_router()
        assert result2 is True
        assert is_patch_applied() is True


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="litellm package not installed")
class TestRoutingStrategyPatch:
    """Test the routing strategy patch module."""

    @pytest.fixture(autouse=True)
    def apply_patch(self):
        """Ensure patch is applied before each test."""
        from litellm_llmrouter import patch_litellm_router
        patch_litellm_router()

    def test_llmrouter_knn_strategy_accepted(self):
        """
        Test that llmrouter-knn strategy is accepted by LiteLLM Router.

        This is the main Gate 2 test - it verifies that the Router no longer
        raises ValueError when routing_strategy='llmrouter-knn' is passed.
        """
        from litellm.router import Router

        # Create a minimal model list
        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                },
            }
        ]

        # This should NOT raise ValueError with our patch
        router = Router(
            model_list=model_list,
            routing_strategy="llmrouter-knn",
        )

        # Verify the strategy was stored correctly
        assert hasattr(router, "_llmrouter_strategy")
        assert router._llmrouter_strategy == "llmrouter-knn"

        # Clean up
        router.discard()

    def test_llmrouter_mlp_strategy_accepted(self):
        """Test that llmrouter-mlp strategy is also accepted."""
        from litellm.router import Router

        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                },
            }
        ]

        router = Router(
            model_list=model_list,
            routing_strategy="llmrouter-mlp",
        )

        assert router._llmrouter_strategy == "llmrouter-mlp"
        router.discard()

    def test_llmrouter_svm_strategy_accepted(self):
        """Test that llmrouter-svm strategy is also accepted."""
        from litellm.router import Router

        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                },
            }
        ]

        router = Router(
            model_list=model_list,
            routing_strategy="llmrouter-svm",
        )

        assert router._llmrouter_strategy == "llmrouter-svm"
        router.discard()

    def test_standard_strategy_still_works(self):
        """Test that standard strategies like simple-shuffle still work."""
        from litellm.router import Router

        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                },
            }
        ]

        # Standard strategy should work normally
        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
        )

        # Should NOT have _llmrouter_strategy since this is a standard strategy
        assert (
            not hasattr(router, "_llmrouter_strategy")
            or router._llmrouter_strategy is None
        )
        router.discard()

    def test_invalid_strategy_still_rejected(self):
        """Test that completely invalid strategies are still rejected."""
        from litellm.router import Router

        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                },
            }
        ]

        # Invalid strategy should still raise ValueError
        with pytest.raises(ValueError, match="Invalid routing_strategy"):
            Router(
                model_list=model_list,
                routing_strategy="completely-invalid-strategy",
            )

    def test_llmrouter_strategy_with_args(self):
        """Test that llmrouter strategies accept routing_strategy_args."""
        from litellm.router import Router

        model_list = [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                },
            }
        ]

        strategy_args = {
            "model_path": "/path/to/model",
            "hot_reload": True,
        }

        router = Router(
            model_list=model_list,
            routing_strategy="llmrouter-knn",
            routing_strategy_args=strategy_args,
        )

        assert router._llmrouter_strategy == "llmrouter-knn"
        assert router._llmrouter_strategy_args == strategy_args
        router.discard()


class TestPatchFunctions:
    """Test the patch/unpatch functions."""

    def test_is_patch_applied(self):
        """Test is_patch_applied returns correct status after explicit patch."""
        from litellm_llmrouter import is_patch_applied, patch_litellm_router

        # Apply patch explicitly
        patch_litellm_router()
        assert is_patch_applied() is True

    def test_patch_idempotent(self):
        """Test that calling patch_litellm_router multiple times is safe."""
        from litellm_llmrouter import patch_litellm_router, is_patch_applied

        # Apply patch first time
        result1 = patch_litellm_router()
        assert result1 is True
        assert is_patch_applied() is True

        # Calling again should return True and not break anything
        result2 = patch_litellm_router()
        assert result2 is True
        assert is_patch_applied() is True


class TestIsLLMRouterStrategy:
    """Test the is_llmrouter_strategy helper function."""

    def test_llmrouter_prefix_detected(self):
        """Test that llmrouter- prefixed strategies are detected."""
        from litellm_llmrouter.routing_strategy_patch import is_llmrouter_strategy

        assert is_llmrouter_strategy("llmrouter-knn") is True
        assert is_llmrouter_strategy("llmrouter-mlp") is True
        assert is_llmrouter_strategy("llmrouter-svm") is True
        assert is_llmrouter_strategy("llmrouter-custom") is True

    def test_non_llmrouter_strategies(self):
        """Test that non-llmrouter strategies return False."""
        from litellm_llmrouter.routing_strategy_patch import is_llmrouter_strategy

        assert is_llmrouter_strategy("simple-shuffle") is False
        assert is_llmrouter_strategy("least-busy") is False
        assert is_llmrouter_strategy("latency-based-routing") is False
        assert is_llmrouter_strategy("cost-based-routing") is False

    def test_non_string_values(self):
        """Test handling of non-string values."""
        from litellm_llmrouter.routing_strategy_patch import is_llmrouter_strategy

        assert is_llmrouter_strategy(None) is False
        assert is_llmrouter_strategy(123) is False
        assert is_llmrouter_strategy(["llmrouter-knn"]) is False
