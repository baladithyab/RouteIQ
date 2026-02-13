"""Verify all registered strategies can be found in DEFAULT_ROUTER_HPARAMS."""

import pytest
from litellm_llmrouter.strategies import LLMROUTER_STRATEGIES, DEFAULT_ROUTER_HPARAMS


# Strategies that are RouteIQ-native (not from upstream LLMRouter)
ROUTEIQ_NATIVE_STRATEGIES = {"custom"}

# Strategies that exist but have no default hparams (they use no-arg constructors)
NO_HPARAMS_STRATEGIES = {"r1"}


@pytest.mark.parametrize("strategy", LLMROUTER_STRATEGIES)
def test_strategy_name_is_valid(strategy):
    """Each strategy name starts with llmrouter- prefix."""
    assert strategy.startswith("llmrouter-"), f"{strategy} missing llmrouter- prefix"


@pytest.mark.parametrize("strategy", LLMROUTER_STRATEGIES)
def test_strategy_has_hparams_or_is_known(strategy):
    """Each strategy has default hparams or is a known exception."""
    short_name = strategy.replace("llmrouter-", "")
    if short_name in ROUTEIQ_NATIVE_STRATEGIES:
        return  # RouteIQ-native, may not have LLMRouter hparams
    if short_name in NO_HPARAMS_STRATEGIES:
        return  # Known no-hparams strategies
    assert short_name in DEFAULT_ROUTER_HPARAMS, (
        f"Strategy '{short_name}' not found in DEFAULT_ROUTER_HPARAMS "
        f"and not in known exceptions"
    )


def test_all_hparams_have_strategy():
    """Every key in DEFAULT_ROUTER_HPARAMS maps to a registered strategy."""
    registered_short = {s.replace("llmrouter-", "") for s in LLMROUTER_STRATEGIES}
    for hparam_key in DEFAULT_ROUTER_HPARAMS:
        assert hparam_key in registered_short, (
            f"Hparams key '{hparam_key}' has no registered strategy"
        )
