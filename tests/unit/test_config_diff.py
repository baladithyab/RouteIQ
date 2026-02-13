"""Tests for incremental config diffing."""

from litellm_llmrouter.config_sync import diff_model_configs


class TestConfigDiff:
    def test_no_changes(self):
        old = [{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}]
        new = [{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}]
        result = diff_model_configs(old, new)
        assert result.added == []
        assert result.removed == []
        assert result.changed == []

    def test_added_model(self):
        old = [{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}]
        new = [
            {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}},
            {"model_name": "claude-3", "litellm_params": {"model": "claude-3-sonnet"}},
        ]
        result = diff_model_configs(old, new)
        assert len(result.added) == 1
        assert result.added[0]["model_name"] == "claude-3"

    def test_removed_model(self):
        old = [
            {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}},
            {"model_name": "claude-3", "litellm_params": {"model": "claude-3-sonnet"}},
        ]
        new = [{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}]
        result = diff_model_configs(old, new)
        assert len(result.removed) == 1
        assert result.removed[0]["model_name"] == "claude-3"

    def test_changed_model(self):
        old = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "old"},
            }
        ]
        new = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "new"},
            }
        ]
        result = diff_model_configs(old, new)
        assert result.added == []
        assert result.removed == []
        assert len(result.changed) == 1

    def test_empty_to_many(self):
        result = diff_model_configs([], [{"model_name": "m1"}, {"model_name": "m2"}])
        assert len(result.added) == 2
        assert result.removed == []

    def test_many_to_empty(self):
        result = diff_model_configs([{"model_name": "m1"}, {"model_name": "m2"}], [])
        assert result.added == []
        assert len(result.removed) == 2
