"""
Tests for LLMRouter strategy integration.
"""

import os
import json
import tempfile


class TestLLMRouterStrategies:
    """Test LLMRouter strategy wrappers."""

    def test_strategy_list_defined(self):
        """Test that strategy list is defined."""
        from litellm_llmrouter.strategies import LLMROUTER_STRATEGIES

        assert len(LLMROUTER_STRATEGIES) > 0
        assert "llmrouter-knn" in LLMROUTER_STRATEGIES
        assert "llmrouter-mlp" in LLMROUTER_STRATEGIES
        assert "llmrouter-custom" in LLMROUTER_STRATEGIES

    def test_strategy_family_init(self):
        """Test LLMRouterStrategyFamily initialization."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        # Create a temporary LLM data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "gpt-4": {"provider": "openai"},
                    "claude-3": {"provider": "anthropic"},
                },
                f,
            )
            llm_data_path = f.name

        try:
            strategy = LLMRouterStrategyFamily(
                strategy_name="llmrouter-knn",
                llm_data_path=llm_data_path,
                hot_reload=False,
            )

            assert strategy.strategy_name == "llmrouter-knn"
            assert strategy.hot_reload is False
            assert len(strategy._llm_data) == 2
        finally:
            os.unlink(llm_data_path)

    def test_should_reload_disabled(self):
        """Test reload check when disabled."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        strategy = LLMRouterStrategyFamily(
            strategy_name="llmrouter-knn", hot_reload=False
        )

        assert strategy._should_reload() is False


class TestConfigLoader:
    """Test configuration loading utilities."""

    def test_imports(self):
        """Test that all exports are importable."""
        from litellm_llmrouter import (
            register_llmrouter_strategies,
            LLMROUTER_STRATEGIES,
            download_config_from_s3,
        )

        assert callable(register_llmrouter_strategies)
        assert callable(download_config_from_s3)
        assert isinstance(LLMROUTER_STRATEGIES, list)

    def test_version_defined(self):
        """Test that version is defined."""
        from litellm_llmrouter import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)


class TestLLMDataLoading:
    """Test LLM candidates data loading."""

    def test_load_llm_data_from_file(self):
        """Test loading LLM data from JSON file."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        test_data = {
            "model-a": {
                "provider": "provider-a",
                "capabilities": ["reasoning"],
                "quality_score": 0.9,
            },
            "model-b": {
                "provider": "provider-b",
                "capabilities": ["coding"],
                "quality_score": 0.85,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            llm_data_path = f.name

        try:
            strategy = LLMRouterStrategyFamily(
                strategy_name="llmrouter-knn", llm_data_path=llm_data_path
            )

            assert "model-a" in strategy._llm_data
            assert "model-b" in strategy._llm_data
            assert strategy._llm_data["model-a"]["quality_score"] == 0.9
        finally:
            os.unlink(llm_data_path)

    def test_load_llm_data_missing_file(self):
        """Test handling of missing LLM data file."""
        from litellm_llmrouter.strategies import LLMRouterStrategyFamily

        strategy = LLMRouterStrategyFamily(
            strategy_name="llmrouter-knn", llm_data_path="/nonexistent/path.json"
        )

        # Should return empty dict on error
        assert strategy._llm_data == {}
