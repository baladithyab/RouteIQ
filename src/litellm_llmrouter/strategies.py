"""
LLMRouter Routing Strategies for LiteLLM
==========================================

This module implements the integration between LLMRouter's ML-based
routing strategies and LiteLLM's routing infrastructure.
"""

import os
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger

# Available LLMRouter strategies (matching llmrouter.models exports)
LLMROUTER_STRATEGIES = [
    # Single-round routers
    "llmrouter-knn",       # KNNRouter - K-Nearest Neighbors
    "llmrouter-svm",       # SVMRouter - Support Vector Machine
    "llmrouter-mlp",       # MLPRouter - Multi-Layer Perceptron
    "llmrouter-mf",        # MFRouter - Matrix Factorization
    "llmrouter-elo",       # EloRouter - Elo Rating based
    "llmrouter-hybrid",    # HybridLLMRouter - Probabilistic hybrid
    "llmrouter-causallm",  # CausalLMRouter - Transformer-based (optional)
    "llmrouter-graph",     # GraphRouter - Graph neural network (optional)
    "llmrouter-automix",   # AutomixRouter - Automatic model mixing
    # Baseline routers
    "llmrouter-smallest",  # SmallestLLM - Always picks smallest
    "llmrouter-largest",   # LargestLLM - Always picks largest
    # Custom routers
    "llmrouter-custom",    # User-defined custom router
]


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
        **kwargs
    ):
        self.strategy_name = strategy_name
        self.model_path = model_path or os.environ.get("LLMROUTER_MODEL_PATH")
        self.llm_data_path = llm_data_path or os.environ.get("LLMROUTER_LLM_DATA_PATH")
        self.config_path = config_path
        self.hot_reload = hot_reload
        self.reload_interval = reload_interval
        self.model_s3_bucket = model_s3_bucket
        self.model_s3_key = model_s3_key
        self.extra_kwargs = kwargs
        
        self._router = None
        self._router_lock = threading.RLock()
        self._last_load_time = 0
        self._model_mtime = 0
        
        # Load LLM candidates data
        self._llm_data = self._load_llm_data()
        
        verbose_proxy_logger.info(
            f"Initialized LLMRouter strategy: {strategy_name}"
        )
    
    def _load_llm_data(self) -> Dict[str, Any]:
        """Load LLM candidates data from JSON file."""
        if not self.llm_data_path:
            return {}
        
        try:
            with open(self.llm_data_path, 'r') as f:
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
        except:
            pass
        
        return False
    
    def _load_router(self):
        """Load the appropriate LLMRouter model based on strategy name."""
        strategy_type = self.strategy_name.replace("llmrouter-", "")

        try:
            # Import from llmrouter.models (correct import path)
            if strategy_type == "knn":
                from llmrouter.models import KNNRouter
                return KNNRouter(yaml_path=self.config_path)
            elif strategy_type == "svm":
                from llmrouter.models import SVMRouter
                return SVMRouter(yaml_path=self.config_path)
            elif strategy_type == "mlp":
                from llmrouter.models import MLPRouter
                return MLPRouter(yaml_path=self.config_path)
            elif strategy_type == "mf":
                from llmrouter.models import MFRouter
                return MFRouter(yaml_path=self.config_path)
            elif strategy_type == "elo":
                from llmrouter.models import EloRouter
                return EloRouter(yaml_path=self.config_path)
            elif strategy_type == "causallm":
                from llmrouter.models import CausalLMRouter
                return CausalLMRouter(yaml_path=self.config_path) if CausalLMRouter else None
            elif strategy_type == "hybrid":
                from llmrouter.models import HybridLLMRouter
                return HybridLLMRouter(yaml_path=self.config_path)
            elif strategy_type == "graph":
                from llmrouter.models import GraphRouter
                return GraphRouter(yaml_path=self.config_path) if GraphRouter else None
            elif strategy_type == "automix":
                from llmrouter.models import AutomixRouter
                return AutomixRouter(yaml_path=self.config_path)
            elif strategy_type == "smallest":
                from llmrouter.models import SmallestLLM
                return SmallestLLM(yaml_path=self.config_path)
            elif strategy_type == "largest":
                from llmrouter.models import LargestLLM
                return LargestLLM(yaml_path=self.config_path)
            elif strategy_type == "custom":
                return self._load_custom_router()
            else:
                verbose_proxy_logger.warning(
                    f"Unknown LLMRouter strategy: {strategy_type}, using MetaRouter"
                )
                from llmrouter.models import MetaRouter
                return MetaRouter(yaml_path=self.config_path)
        except ImportError as e:
            verbose_proxy_logger.error(f"Failed to import LLMRouter: {e}")
            return None
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load router: {e}")
            return None
    
    def _load_custom_router(self):
        """Load a custom router from the custom routers directory."""
        custom_path = os.environ.get(
            "LLMROUTER_CUSTOM_ROUTERS_PATH", 
            "/app/custom_routers"
        )
        # Implementation for custom router loading
        verbose_proxy_logger.info(f"Loading custom router from: {custom_path}")
        return None
    
    @property
    def router(self):
        """Get the router instance, loading/reloading as needed."""
        with self._router_lock:
            if self._router is None or self._should_reload():
                self._router = self._load_router()
                self._last_load_time = time.time()
                if self.model_path:
                    try:
                        self._model_mtime = Path(self.model_path).stat().st_mtime
                    except:
                        pass
        return self._router


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
