"""Re-export personalized routing from the main RouteIQ package.

When installed standalone, this module provides the PersonalizedRouter
that learns per-user/per-team model preferences from feedback signals.
"""

try:
    from litellm_llmrouter.personalized_routing import (
        PersonalizedRouter,
        PreferenceStore,
        UserPreference,
        get_personalized_router,
        reset_personalized_router,
    )
except ImportError:
    raise ImportError(
        "routeiq-routing requires either the full routeiq gateway or "
        "a standalone copy of personalized_routing.py. "
        "Install the full gateway: pip install routeiq"
    )
