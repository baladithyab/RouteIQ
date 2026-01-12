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

# Available LLMRouter strategies
LLMROUTER_STRATEGIES = [
    "llmrouter-knn",
    "llmrouter-svm", 
    "llmrouter-mlp",
    "llmrouter-mf",
    "llmrouter-elo",
    "llmrouter-hybrid",
    "llmrouter-bert",
    "llmrouter-causallm",
    "llmrouter-graph",
    "llmrouter-automix",
    "llmrouter-custom",
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
            if strategy_type == "knn":
                from llmrouter.models.knn_router import KNNRouter
                return KNNRouter(yaml_path=self.config_path)
            elif strategy_type == "svm":
                from llmrouter.models.svm_router import SVMRouter
                return SVMRouter(yaml_path=self.config_path)
            elif strategy_type == "mlp":
                from llmrouter.models.mlp_router import MLPRouter
                return MLPRouter(yaml_path=self.config_path)
            elif strategy_type == "mf":
                from llmrouter.models.mf_router import MFRouter
                return MFRouter(yaml_path=self.config_path)
            elif strategy_type == "bert":
                from llmrouter.models.bert_router import BertRouter
                return BertRouter(yaml_path=self.config_path)
            elif strategy_type == "causallm":
                from llmrouter.models.causallm_router import CausalLMRouter
                return CausalLMRouter(yaml_path=self.config_path)
            elif strategy_type == "hybrid":
                from llmrouter.models.hybrid_router import HybridRouter
                return HybridRouter(yaml_path=self.config_path)
            elif strategy_type == "custom":
                return self._load_custom_router()
            else:
                verbose_proxy_logger.warning(
                    f"Unknown LLMRouter strategy: {strategy_type}, using random"
                )
                from llmrouter.models.meta_router import MetaRouter
                return None
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

