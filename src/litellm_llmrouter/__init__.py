"""
LiteLLM + LLMRouter Integration
================================

This module provides the integration layer between LiteLLM's routing
infrastructure and LLMRouter's ML-based routing strategies.

Usage:
    from litellm_llmrouter import register_llmrouter_strategies
    
    # Register all LLMRouter strategies with LiteLLM
    register_llmrouter_strategies()
"""

from .strategies import (
    LLMRouterStrategyFamily,
    register_llmrouter_strategies,
    LLMROUTER_STRATEGIES,
)
from .config_loader import (
    download_config_from_s3,
    download_config_from_gcs,
    download_model_from_s3,
    download_custom_router_from_s3,
)

__version__ = "0.1.0"
__all__ = [
    "LLMRouterStrategyFamily",
    "register_llmrouter_strategies",
    "LLMROUTER_STRATEGIES",
    "download_config_from_s3",
    "download_config_from_gcs",
    "download_model_from_s3",
    "download_custom_router_from_s3",
]

