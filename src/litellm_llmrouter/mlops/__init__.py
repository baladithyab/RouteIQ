"""MLOps closed-loop package (Cluster H).

Keeps routing strategies optimal on top of the EXISTING COLLECT / EVALUATE /
AGGREGATE / FEEDBACK eval loop:

- :mod:`litellm_llmrouter.mlops.drift` -- input-drift + routing-quality-regression
  detection (RouteIQ-6dce), emitting OTel drift signals.

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

__all__ = [
    "DriftDetector",
    "DriftReport",
    "get_drift_detector",
    "reset_drift_detector",
]
