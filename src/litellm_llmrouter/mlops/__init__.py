"""MLOps closed-loop package (Cluster H + Cluster M).

Keeps routing strategies optimal on top of the EXISTING COLLECT / EVALUATE /
AGGREGATE / FEEDBACK eval loop:

- :mod:`litellm_llmrouter.mlops.drift` -- input-drift + routing-quality-regression
  detection (RouteIQ-6dce), emitting OTel drift signals.
- :mod:`litellm_llmrouter.mlops.upstream_router` -- delegation to an upstream
  LiteLLM adaptive/quality/auto router + durable update-queue flush
  (RouteIQ-8539).
- :mod:`litellm_llmrouter.mlops.sagemaker_registry` -- SageMaker Model Registry +
  Experiments adapter (RouteIQ-93e9), cred-free + operator-gated.
- :mod:`litellm_llmrouter.mlops.offline_eval` -- offline eval harness over a
  versioned golden dataset (RouteIQ-8d24), no live traffic.

Quality-gated champion/challenger promotion (RouteIQ-2a1c) and shadow/mirror
canary traffic (RouteIQ-4fd1) live in
:mod:`litellm_llmrouter.strategy_registry` (they extend the A/B + hot-swap
registry directly).

All MLOps features are settings-gated under ``settings.mlops`` and DEFAULT OFF.
"""

from litellm_llmrouter.mlops.drift import (
    DriftDetector,
    DriftReport,
    get_drift_detector,
    reset_drift_detector,
)
from litellm_llmrouter.mlops.offline_eval import (
    GoldenCase,
    GoldenDataset,
    OfflineEvalHarness,
    OfflineEvalReport,
    StrategyComparison,
    expected_quality_scorer,
    judge_scorer,
    load_golden_dataset,
)
from litellm_llmrouter.mlops.sagemaker_registry import (
    ExperimentRunResult,
    ModelPackageResult,
    SageMakerRegistryAdapter,
    get_sagemaker_registry_adapter,
    reset_sagemaker_registry_adapter,
)
from litellm_llmrouter.mlops.upstream_router import (
    UpstreamRouterDelegate,
    get_upstream_router_delegate,
    reset_upstream_router_delegate,
    wire_upstream_router_flush,
)

__all__ = [
    "DriftDetector",
    "DriftReport",
    "get_drift_detector",
    "reset_drift_detector",
    # Upstream router delegation (RouteIQ-8539)
    "UpstreamRouterDelegate",
    "get_upstream_router_delegate",
    "reset_upstream_router_delegate",
    "wire_upstream_router_flush",
    # SageMaker registry + experiments (RouteIQ-93e9)
    "SageMakerRegistryAdapter",
    "ModelPackageResult",
    "ExperimentRunResult",
    "get_sagemaker_registry_adapter",
    "reset_sagemaker_registry_adapter",
    # Offline eval harness (RouteIQ-8d24)
    "GoldenCase",
    "GoldenDataset",
    "OfflineEvalHarness",
    "OfflineEvalReport",
    "StrategyComparison",
    "expected_quality_scorer",
    "judge_scorer",
    "load_golden_dataset",
]
