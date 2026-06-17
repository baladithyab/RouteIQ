"""The mlops_feedback_loop default decision (RouteIQ-3b4d).

DECISION: ``adapter_framework.mlops_feedback_loop`` defaults ON. The
intelligent-routing use case (Claude Code -> RouteIQ -> mixed-Bedrock auto-group)
is only worth it if the bandit LEARNS which arm is best automatically, so the
eval-judge -> bandit feedback loop closes by default. It remains a NO-OP unless a
continuous-learning strategy is registered AND the eval pipeline is enabled, so
the default costs nothing on its own. Opt out with
``ROUTEIQ_ADAPTER_FRAMEWORK__MLOPS_FEEDBACK_LOOP=false``.

These tests assert the new default and that, with the flag on (the default), the
loop wires AND the Kumaraswamy-Thompson bandit (a ``learns=True``,
``train_mode=continuous`` strategy) is registered as an eval feedback adapter.

Credential-free: no live AWS / LiteLLM; the eval pipeline runs with sample_rate=0.
"""

from __future__ import annotations

from litellm_llmrouter.adapters.mlops import (
    MLOpsCoordinator,
    wire_mlops_feedback_loop,
)
from litellm_llmrouter.strategy_registry import get_routing_registry


def test_default_is_on():
    """The flag now defaults ON (RouteIQ-3b4d)."""
    from litellm_llmrouter.settings import AdapterFrameworkSettings, get_settings

    assert AdapterFrameworkSettings().mlops_feedback_loop is True
    assert get_settings().adapter_framework.mlops_feedback_loop is True


def test_opt_out_via_env(monkeypatch):
    """The documented opt-out env var disables the default."""
    from litellm_llmrouter.settings import get_settings, reset_settings

    monkeypatch.setenv("ROUTEIQ_ADAPTER_FRAMEWORK__MLOPS_FEEDBACK_LOOP", "false")
    reset_settings()
    try:
        assert get_settings().adapter_framework.mlops_feedback_loop is False
    finally:
        reset_settings()


def test_loop_wires_under_default_flag_without_force(monkeypatch):
    """With the default (on) flag, the loop wires WITHOUT force=True.

    Proves the default actually closes the loop (not just that force works).
    """
    from litellm_llmrouter.eval_pipeline import EvalPipeline
    from litellm_llmrouter.settings import reset_settings

    # No env override => the new default (True) applies.
    reset_settings()
    coord = MLOpsCoordinator()
    pipeline = EvalPipeline(sample_rate=0.0)
    # force defaults to False -> the gate reads the (default-on) settings flag.
    assert wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=pipeline) is True


def test_bandit_registered_as_feedback_adapter_under_default():
    """The bandit (learns=True, continuous) is registered as a feedback adapter.

    Confirms the loop wires the bandit specifically: register the bandit in the
    routing registry, wire under the default flag, and assert it is discovered
    as a continuous learning adapter the eval aggregate fans out to.
    """
    from litellm_llmrouter.eval_pipeline import EvalPipeline
    from litellm_llmrouter.kumaraswamy_thompson import (
        STRATEGY_NAME,
        KumaraswamyThompsonStrategy,
    )

    registry = get_routing_registry()
    registry.register(STRATEGY_NAME, KumaraswamyThompsonStrategy(seed=7), version="v1")

    coord = MLOpsCoordinator()
    pipeline = EvalPipeline(sample_rate=0.0)
    # Default flag is on -> wires without force.
    assert wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=pipeline) is True

    # The bandit is registered as a learning adapter.
    assert STRATEGY_NAME in coord.list_adapters()

    # And it is CONTINUOUS, so the eval aggregate reaches it.
    manifest = KumaraswamyThompsonStrategy(seed=7).declare_capabilities()
    assert manifest.learns is True
    assert manifest.train_mode == "continuous"
