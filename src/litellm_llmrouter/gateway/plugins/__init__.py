"""
Gateway plugins package.

This package contains built-in plugins that can be enabled via LLMROUTER_PLUGINS.
"""

from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsDiscoveryPlugin

__all__ = ["SkillsDiscoveryPlugin"]
