"""Typed configuration for RouteIQ Gateway.

Replaces 124+ scattered ``os.environ.get()`` calls with a single validated
settings object.  All fields are documented, typed, and validated at startup.

The settings module is **purely additive** — existing code continues to
read ``os.environ`` as before.  Modules can migrate to ``get_settings()``
incrementally without a flag day.

Usage::

    from litellm_llmrouter.settings import get_settings

    settings = get_settings()
    if settings.redis.host:
        ...

Singleton & Reset Pattern:
    Follows the codebase convention of ``get_*()`` / ``reset_*()`` pairs.
    Tests **must** call ``reset_settings()`` in an ``autouse=True`` fixture.
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ============================================================================
# Enums
# ============================================================================


class RoutingProfile(str, Enum):
    """Routing profiles for centroid-based routing.

    Profiles control the cost/quality trade-off for model selection:
    - ``auto``      — Let the classifier decide based on prompt complexity.
    - ``eco``       — Prefer cheaper models when quality is sufficient.
    - ``premium``   — Prefer the highest-quality models regardless of cost.
    - ``free``      — Only use free-tier models.
    - ``reasoning`` — Prefer models with strong reasoning capabilities.
    """

    AUTO = "auto"
    ECO = "eco"
    PREMIUM = "premium"
    FREE = "free"
    REASONING = "reasoning"


class PolicyFailMode(str, Enum):
    """Policy engine behaviour when evaluation fails."""

    OPEN = "open"
    CLOSED = "closed"


class AuditFailMode(str, Enum):
    """Audit logging behaviour when the audit backend fails."""

    OPEN = "open"
    CLOSED = "closed"


class QuotaFailMode(str, Enum):
    """Quota enforcement behaviour when the quota backend fails."""

    OPEN = "open"
    CLOSED = "closed"


# ============================================================================
# Nested Settings Models
# ============================================================================


class RedisSettings(BaseModel):
    """Redis configuration.  Optional — gateway works without Redis.

    Used by: caching, HA leader election, quota enforcement, MCP HA sync,
    conversation affinity, and health checks.

    Env vars: ``REDIS_HOST``, ``REDIS_PORT``, ``REDIS_PASSWORD``,
    ``REDIS_SSL``, ``REDIS_DB``.
    """

    host: Optional[str] = Field(
        None,
        description="Redis host.  When unset, Redis features are disabled.",
    )
    port: int = Field(
        6379,
        ge=1,
        le=65535,
        description="Redis port.",
    )
    password: Optional[str] = Field(
        None,
        description="Redis password.  Omit for unauthenticated connections.",
    )
    ssl: bool = Field(
        False,
        description="Enable TLS when connecting to Redis.",
    )
    db: int = Field(
        0,
        ge=0,
        le=15,
        description="Redis database index.",
    )


class PostgresSettings(BaseModel):
    """PostgreSQL configuration.  Optional — gateway works without a DB.

    Env var: ``DATABASE_URL``.
    """

    url: Optional[str] = Field(
        None,
        description=(
            "PostgreSQL connection URL (e.g. ``postgresql://user:pass@host:5432/db``)."
        ),
    )
    pool_min: int = Field(
        2,
        ge=1,
        le=50,
        description="Minimum connection pool size.",
    )
    pool_max: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum connection pool size.",
    )
    ssl_mode: str = Field(
        "prefer",
        description=(
            "SSL mode: disable, allow, prefer, require, verify-ca, verify-full."
        ),
    )

    @field_validator("url")
    @classmethod
    def _validate_url_scheme(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        valid = ("postgresql://", "postgres://", "sqlite://")
        if not v.startswith(valid):
            warnings.warn(
                f"DATABASE_URL does not start with a recognised scheme "
                f"(expected one of {', '.join(valid)})",
                stacklevel=2,
            )
        return v


class OTelSettings(BaseModel):
    """OpenTelemetry observability configuration.

    Env vars: ``OTEL_ENABLED``, ``OTEL_EXPORTER_OTLP_ENDPOINT``,
    ``OTEL_SERVICE_NAME``, ``OTEL_SAMPLE_RATE``.
    """

    enabled: bool = Field(
        True,
        description="Enable OpenTelemetry traces, metrics, and logs.",
    )
    endpoint: Optional[str] = Field(
        None,
        description=(
            "OTLP collector endpoint "
            "(e.g. ``http://localhost:4317``).  Supports http, https, grpc."
        ),
    )
    service_name: str = Field(
        "litellm-gateway",
        description="OTel service.name resource attribute.",
    )
    sample_rate: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (0.0 = none, 1.0 = all).",
    )
    resource_service_name: str = Field(
        "routeiq",
        description="OTel resource attribute service.name for RouteIQ metrics.",
    )
    deployment_env: str = Field(
        "default",
        description="OTel resource attribute deployment.environment.",
    )
    metrics_namespace: str = Field(
        "RouteIQ",
        description="Namespace prefix for RouteIQ metrics.",
    )

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            valid = ("http://", "https://", "grpc://")
            if not v.startswith(valid):
                warnings.warn(
                    f"OTEL_EXPORTER_OTLP_ENDPOINT ('{v}') does not start "
                    f"with a recognised scheme (expected http, https, or grpc)",
                    stacklevel=2,
                )
        return v


class OIDCSettings(BaseModel):
    """OIDC / SSO integration configuration.

    Env vars: ``ROUTEIQ_OIDC_*``.  Requires the ``authlib`` package
    (install via ``pip install routeiq[oidc]``).
    """

    enabled: bool = Field(False, description="Enable OIDC authentication.")
    issuer_url: Optional[str] = Field(
        None,
        description="OIDC issuer URL (must support .well-known/openid-configuration).",
    )
    client_id: Optional[str] = Field(None, description="OAuth2 client ID.")
    client_secret: Optional[str] = Field(None, description="OAuth2 client secret.")
    user_id_claim: str = Field(
        "sub", description="JWT claim for unique user identifier."
    )
    email_claim: str = Field("email", description="JWT claim for email.")
    display_name_claim: str = Field("name", description="JWT claim for display name.")
    team_claim: Optional[str] = Field(
        None, description="JWT claim for team assignment."
    )
    org_claim: Optional[str] = Field(None, description="JWT claim for organization.")
    role_claim: Optional[str] = Field(None, description="JWT claim for roles.")
    role_mapping: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Maps IdP role names to lists of RouteIQ internal roles (JSON).",
    )
    default_role: str = Field(
        "internal_user",
        description="Default role when no mapping matches.",
    )
    allowed_email_domains: List[str] = Field(
        default_factory=list,
        description="Restrict logins to these email domains (empty = all allowed).",
    )
    auto_provision_users: bool = Field(
        True, description="Auto-create user records on first SSO login."
    )
    session_ttl: int = Field(
        1800,
        ge=60,
        le=86400,
        description="Validated identity cache TTL in seconds.",
    )
    token_exchange_enabled: bool = Field(
        True,
        description="Enable the /auth/token-exchange endpoint.",
    )
    max_key_ttl_days: int = Field(
        365,
        ge=1,
        le=3650,
        description="Maximum allowed TTL for exchanged API keys.",
    )
    default_key_ttl_days: int = Field(
        90,
        ge=1,
        le=3650,
        description="Default API key TTL when not specified.",
    )

    @field_validator("issuer_url")
    @classmethod
    def _validate_issuer_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.rstrip("/")
        if not v.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError(
                "OIDC issuer_url must use HTTPS (HTTP allowed only for localhost)"
            )
        return v


class RoutingSettings(BaseModel):
    """ML routing strategy configuration.

    Env vars: ``ROUTEIQ_USE_PLUGIN_STRATEGY``, ``ROUTEIQ_CENTROID_ROUTING``,
    ``ROUTEIQ_CENTROID_WARMUP``, ``ROUTEIQ_ROUTING_PROFILE``,
    ``ROUTEIQ_ROUTING_STRATEGY``, ``LLMROUTER_USE_PIPELINE``,
    ``LLMROUTER_ACTIVE_ROUTING_STRATEGY``, ``LLMROUTER_STRATEGY_WEIGHTS``,
    ``LLMROUTER_EXPERIMENT_ID``, ``LLMROUTER_EXPERIMENT_CONFIG``.
    """

    # use_plugin_strategy is always True. The legacy monkey-patch mode has
    # been removed. This property exists for backward compatibility with code
    # that reads ``settings.routing.use_plugin_strategy``.
    use_plugin_strategy: bool = Field(
        True,
        description="Always True. Legacy monkey-patch mode has been removed.",
    )
    centroid_enabled: bool = Field(
        True,
        description="Enable centroid-based zero-config routing fallback.",
    )
    centroid_warmup: bool = Field(
        False,
        description="Pre-warm centroid classifier at startup.",
    )
    default_profile: RoutingProfile = Field(
        RoutingProfile.AUTO,
        description="Default routing profile (auto/eco/premium/free/reasoning).",
    )
    strategy_override: Optional[str] = Field(
        None,
        description=(
            "Explicit routing strategy override "
            "(e.g. ``llmrouter-knn``).  Takes priority over config."
        ),
    )
    pipeline_enabled: bool = Field(
        True,
        description="Enable A/B pipeline routing via strategy registry.",
    )
    active_strategy: Optional[str] = Field(
        None,
        description="Active routing strategy name for A/B routing.",
    )
    strategy_weights: Optional[str] = Field(
        None,
        description="JSON dict of strategy name → weight for A/B routing.",
    )
    experiment_id: Optional[str] = Field(
        None,
        description="Experiment identifier for A/B routing telemetry.",
    )
    experiment_config: Optional[str] = Field(
        None,
        description="JSON config for A/B experiment (parsed at runtime).",
    )
    personalized_enabled: bool = Field(
        False,
        description=(
            "Enable personalized routing.  Learns per-user/per-team model "
            "preferences from feedback signals and adapts routing decisions."
        ),
    )
    personalized_dim: int = Field(
        128,
        ge=16,
        le=1024,
        description="Dimensionality of user preference vectors.",
    )
    personalized_learning_rate: float = Field(
        0.1,
        ge=0.001,
        le=1.0,
        description="Learning rate for preference updates.",
    )
    personalized_decay: float = Field(
        0.99,
        ge=0.5,
        le=1.0,
        description="Per-day decay factor for stale preferences (1.0 = no decay).",
    )


class SecuritySettings(BaseModel):
    """Security, auth, and access control configuration.

    Env vars: ``ADMIN_API_KEYS``, ``ADMIN_API_KEY``, ``ADMIN_AUTH_ENABLED``,
    ``ROUTEIQ_KEY_PREFIX``, ``ROUTEIQ_CORS_ORIGINS``,
    ``ROUTEIQ_CORS_CREDENTIALS``, ``LLMROUTER_ENFORCE_SIGNED_MODELS``,
    ``LLMROUTER_ALLOW_PICKLE_MODELS``, ``LLMROUTER_STRICT_PICKLE_MODE``,
    ``LLMROUTER_PICKLE_ALLOWLIST``.
    """

    admin_api_keys: Optional[str] = Field(
        None,
        description="Comma-separated admin API keys for control-plane auth.",
    )
    admin_api_key: Optional[str] = Field(
        None,
        description="Single admin API key (legacy, prefer ADMIN_API_KEYS).",
    )
    admin_auth_enabled: bool = Field(
        True,
        description=(
            "Enable admin auth.  Setting to false is NOT recommended for production."
        ),
    )
    key_prefix: str = Field(
        "sk-riq-",
        description="Prefix applied to master/admin keys.",
    )
    cors_origins: str = Field(
        "*",
        description="Comma-separated allowed CORS origins.",
    )
    cors_credentials: bool = Field(
        False,
        description="Allow credentials in CORS requests.",
    )
    enforce_signed_models: bool = Field(
        True,
        description="Require cryptographic verification of ML model artifacts.",
    )
    allow_pickle_models: bool = Field(
        False,
        description=(
            "Allow pickle model loading.  Security risk — only enable "
            "in trusted environments."
        ),
    )
    strict_pickle_mode: bool = Field(
        False,
        description="Strict pickle mode (hash-only allowlist).",
    )
    pickle_allowlist: Optional[str] = Field(
        None,
        description="Comma-separated SHA-256 hashes of allowed pickle files.",
    )

    @model_validator(mode="after")
    def _warn_pickle_without_signing(self) -> "SecuritySettings":
        if self.allow_pickle_models and not self.enforce_signed_models:
            warnings.warn(
                "LLMROUTER_ALLOW_PICKLE_MODELS is enabled but "
                "LLMROUTER_ENFORCE_SIGNED_MODELS is not.  This allows "
                "loading unverified pickle models (code-execution risk).",
                stacklevel=2,
            )
        return self


class PolicySettings(BaseModel):
    """OPA-style policy engine configuration.

    Env vars: ``POLICY_ENGINE_ENABLED``, ``POLICY_ENGINE_FAIL_MODE``,
    ``POLICY_CONFIG_PATH``.
    """

    enabled: bool = Field(False, description="Enable the policy engine.")
    fail_mode: PolicyFailMode = Field(
        PolicyFailMode.OPEN,
        description="Behaviour on policy evaluation errors (open/closed).",
    )
    config_path: Optional[str] = Field(
        None,
        description="Path to the policy YAML config file.",
    )


class AuditSettings(BaseModel):
    """Audit logging configuration.

    Env vars: ``AUDIT_LOG_ENABLED``, ``AUDIT_LOG_FAIL_MODE``.
    """

    enabled: bool = Field(True, description="Enable audit logging.")
    fail_mode: AuditFailMode = Field(
        AuditFailMode.OPEN,
        description="Behaviour when the audit backend fails.",
    )


class QuotaSettings(BaseModel):
    """Per-team / per-key quota enforcement.

    Env vars: ``ROUTEIQ_QUOTA_ENABLED``, ``ROUTEIQ_QUOTA_FAIL_MODE``,
    ``ROUTEIQ_QUOTA_LIMITS_JSON``.
    """

    enabled: bool = Field(False, description="Enable quota enforcement.")
    fail_mode: QuotaFailMode = Field(
        QuotaFailMode.OPEN,
        description="Behaviour when quota backend is unavailable.",
    )
    limits_json: Optional[str] = Field(
        None,
        description="JSON-encoded quota limits per team/key.",
    )
    default_spend_per_1k_tokens: float = Field(
        0.002,
        ge=0.0,
        description="Default USD cost per 1K tokens when LiteLLM cost is unavailable.",
    )
    cost_reconciliation_enabled: bool = Field(
        True,
        description="Enable post-response cost reconciliation.",
    )


class MCPSettings(BaseModel):
    """MCP (Model Context Protocol) gateway configuration.

    Env vars: ``MCP_GATEWAY_ENABLED``, ``LLMROUTER_ENABLE_MCP_TOOL_INVOCATION``,
    ``MCP_HA_SYNC_ENABLED``, ``MCP_SYNC_INTERVAL``, ``MCP_OAUTH_ENABLED``.
    """

    enabled: bool = Field(False, description="Enable the MCP gateway.")
    tool_invocation_enabled: bool = Field(
        False,
        description=(
            "Enable remote tool invocation.  Disabled by default for security."
        ),
    )
    ha_sync_enabled: bool = Field(
        True,
        description="Enable HA sync of MCP server registry via Redis.",
    )
    sync_interval: float = Field(
        5.0,
        ge=1.0,
        description="Interval (seconds) for HA registry sync.",
    )
    oauth_enabled: bool = Field(False, description="Enable OAuth for MCP.")


class A2ASettings(BaseModel):
    """A2A (Agent-to-Agent) gateway configuration.

    Env vars: ``A2A_GATEWAY_ENABLED``, ``A2A_BASE_URL``,
    ``A2A_RAW_STREAMING_ENABLED``, ``A2A_RAW_STREAMING_CHUNK_SIZE``,
    ``A2A_TASK_TTL_SECONDS``, ``A2A_TASK_STORE_MAX_TASKS``,
    ``A2A_TASK_RATE_LIMIT``.
    """

    enabled: bool = Field(
        False,
        description="Enable the A2A gateway convenience routes.",
    )
    base_url: str = Field(
        "",
        description="Base URL for A2A agent communications.",
    )
    raw_streaming_enabled: bool = Field(
        False,
        description="Use raw byte streaming (aiter_bytes) for true passthrough.",
    )
    raw_streaming_chunk_size: int = Field(
        8192,
        ge=1024,
        description="Chunk size in bytes for raw streaming mode.",
    )
    task_ttl_seconds: int = Field(
        3600,
        ge=60,
        description="TTL in seconds for A2A tasks in memory.",
    )
    task_store_max_tasks: int = Field(
        10000,
        ge=100,
        description="Maximum tasks in the A2A task store.",
    )
    task_rate_limit: int = Field(
        100,
        ge=1,
        description="Per-agent rate limit: max task creates per minute.",
    )


class ResilienceSettings(BaseModel):
    """Backpressure, drain, and circuit-breaker configuration.

    Env vars: ``ROUTEIQ_MAX_CONCURRENT_REQUESTS``,
    ``ROUTEIQ_DRAIN_TIMEOUT_SECONDS``, ``ROUTEIQ_BACKPRESSURE_EXCLUDED_PATHS``,
    ``ROUTEIQ_PROVIDER_CB_ENABLED``.
    """

    max_concurrent_requests: int = Field(
        0,
        ge=0,
        description="Max concurrent requests (0 = unlimited).",
    )
    drain_timeout_seconds: float = Field(
        30.0,
        ge=0.0,
        description="Graceful shutdown drain timeout in seconds.",
    )
    backpressure_excluded_paths: str = Field(
        "",
        description="Comma-separated paths excluded from backpressure.",
    )
    provider_circuit_breaker_enabled: bool = Field(
        False,
        description="Enable per-provider circuit breakers.",
    )


class ConfigSyncSettings(BaseModel):
    """Remote config sync from S3/GCS.

    Env vars: ``CONFIG_HOT_RELOAD``, ``CONFIG_SYNC_ENABLED``,
    ``CONFIG_S3_BUCKET``, ``CONFIG_S3_KEY``, ``CONFIG_GCS_BUCKET``,
    ``CONFIG_GCS_KEY``.
    """

    hot_reload: bool = Field(
        False,
        description="Enable filesystem-watching config hot-reload.",
    )
    sync_enabled: bool = Field(
        True,
        description="Enable background config sync.",
    )
    s3_bucket: Optional[str] = Field(
        None, description="S3 bucket for remote config sync."
    )
    s3_key: Optional[str] = Field(None, description="S3 key for the config file.")
    gcs_bucket: Optional[str] = Field(
        None, description="GCS bucket for remote config sync."
    )
    gcs_key: Optional[str] = Field(None, description="GCS key for the config file.")


class HTTPClientSettings(BaseModel):
    """Shared HTTP client pool configuration.

    Env vars: ``ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED``,
    ``HTTP_CLIENT_MAX_CONNECTIONS``, ``HTTP_CLIENT_MAX_KEEPALIVE``,
    ``HTTP_CLIENT_DEFAULT_TIMEOUT``.
    """

    pooling_enabled: bool = Field(
        True,
        description="Enable shared HTTP client connection pooling.",
    )
    max_connections: int = Field(
        100,
        ge=1,
        description="Maximum total connections in the pool.",
    )
    max_keepalive: int = Field(
        20,
        ge=1,
        description="Maximum keepalive connections.",
    )
    default_timeout: float = Field(
        60.0,
        ge=1.0,
        description="Default request timeout in seconds.",
    )


class SSRFSettings(BaseModel):
    """SSRF protection configuration.

    Env vars: ``LLMROUTER_OUTBOUND_URL_ALLOWLIST``,
    ``LLMROUTER_SSRF_USE_SYNC_DNS``, ``LLMROUTER_SSRF_DNS_TIMEOUT``,
    ``LLMROUTER_SSRF_DNS_CACHE_TTL``, ``LLMROUTER_SSRF_DNS_CACHE_SIZE``.
    """

    outbound_url_allowlist: str = Field(
        "",
        description="Comma-separated URLs to allow through SSRF checks.",
    )
    use_sync_dns: bool = Field(
        False,
        description="Use synchronous DNS resolution in SSRF checks.",
    )
    dns_timeout: float = Field(
        2.0,
        ge=0.1,
        description="DNS resolution timeout in seconds.",
    )
    dns_cache_ttl: int = Field(
        300,
        ge=0,
        description="DNS cache TTL in seconds.",
    )
    dns_cache_size: int = Field(
        256,
        ge=1,
        description="DNS cache maximum entries.",
    )


class ConversationAffinitySettings(BaseModel):
    """Conversation-based routing affinity.

    Env vars: ``CONVERSATION_AFFINITY_ENABLED``,
    ``CONVERSATION_AFFINITY_TTL``, ``CONVERSATION_AFFINITY_MAX_ENTRIES``.
    """

    enabled: bool = Field(
        False,
        description="Enable conversation-based routing affinity.",
    )
    ttl: int = Field(
        3600,
        ge=60,
        description="TTL for conversation affinity entries in seconds.",
    )
    max_entries: int = Field(
        10000,
        ge=100,
        description="Maximum number of cached affinity entries.",
    )


class PluginSettings(BaseModel):
    """Gateway plugin system configuration.

    Env vars: ``LLMROUTER_PLUGINS``, ``LLMROUTER_PLUGINS_ALLOWLIST``,
    ``LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES``,
    ``LLMROUTER_PLUGINS_FAILURE_MODE``, ``ROUTEIQ_PLUGIN_STARTUP_TIMEOUT``.
    """

    plugins: str = Field(
        "",
        description="Comma-separated list of plugin module paths to load.",
    )
    allowlist: Optional[str] = Field(
        None,
        description="Comma-separated plugin names allowed to load.",
    )
    allowed_capabilities: Optional[str] = Field(
        None,
        description="Comma-separated capabilities allowed for plugins.",
    )
    failure_mode: str = Field(
        "continue",
        description="Plugin failure mode: continue or abort.",
    )
    startup_timeout: float = Field(
        30.0,
        ge=1.0,
        description="Maximum seconds for plugin startup.",
    )


class ManagementSettings(BaseModel):
    """Management endpoint middleware configuration.

    Env vars: ``ROUTEIQ_MANAGEMENT_RBAC_ENABLED``,
    ``ROUTEIQ_MANAGEMENT_OTEL_ENABLED``.
    """

    rbac_enabled: bool = Field(
        False,
        description="Enable RBAC for LiteLLM management endpoints.",
    )
    otel_enabled: bool = Field(
        True,
        description="Emit OTel span attributes for management operations.",
    )


class CacheSettings(BaseModel):
    """Caching plugin configuration.

    Env vars: ``CACHE_ENABLED``, ``CACHE_SEMANTIC_ENABLED``,
    ``CACHE_TTL_SECONDS``, ``CACHE_L1_MAX_SIZE``,
    ``CACHE_SIMILARITY_THRESHOLD``, ``CACHE_EMBEDDING_MODEL``,
    ``CACHE_REDIS_URL``, ``CACHE_MAX_TEMPERATURE``.
    """

    enabled: bool = Field(False, description="Enable the cache plugin.")
    semantic_enabled: bool = Field(
        False, description="Enable semantic (embedding-based) caching."
    )
    ttl_seconds: int = Field(3600, ge=1, description="Cache entry TTL in seconds.")
    l1_max_size: int = Field(
        1000, ge=1, description="Max entries in L1 in-process cache."
    )
    similarity_threshold: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for semantic cache hits.",
    )
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="Sentence-transformer model for embeddings.",
    )
    redis_url: Optional[str] = Field(
        None,
        description="Redis URL for L2 cache (falls back to REDIS_HOST).",
    )
    max_temperature: float = Field(
        0.1,
        ge=0.0,
        le=2.0,
        description="Max temperature for cacheability.",
    )


class RouterR1Settings(BaseModel):
    """Router-R1 iterative reasoning router configuration.

    Implements the Router-R1 concept (NeurIPS 2025) natively using RouteIQ's
    own LLM proxy as both the reasoning engine and routing pool.

    Env vars: ``ROUTEIQ_ROUTER_R1_ENABLED``, ``ROUTEIQ_ROUTER_R1_MODEL``,
    ``ROUTEIQ_ROUTER_R1_MAX_ITERATIONS``, ``ROUTEIQ_ROUTER_R1_TIMEOUT``.
    """

    enabled: bool = Field(
        False,
        description="Enable the Router-R1 iterative reasoning router.",
    )
    model: str = Field(
        "gpt-4o-mini",
        description="LLM model used as the reasoning/routing agent.",
    )
    max_iterations: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum reasoning iterations per query.",
    )
    timeout: float = Field(
        30.0,
        ge=1.0,
        description="Timeout in seconds per iteration.",
    )


class ContextOptimizerSettings(BaseModel):
    """Context optimizer plugin configuration.

    Reduces LLM input tokens by 30-70% using deterministic, lossless transforms.

    Env vars: ``ROUTEIQ_CONTEXT_OPTIMIZE``,
    ``ROUTEIQ_CONTEXT_OPTIMIZE_MAX_TURNS``,
    ``ROUTEIQ_CONTEXT_OPTIMIZE_KEEP_LAST``.
    """

    mode: str = Field(
        "off",
        description="Optimization mode: off, safe, or aggressive.",
    )
    max_turns: int = Field(
        40,
        ge=1,
        description="Trim conversations with more turns than this.",
    )
    keep_last: int = Field(
        20,
        ge=1,
        description="Keep this many recent turns after trimming.",
    )


class EvaluatorSettings(BaseModel):
    """Evaluator plugin configuration.

    Env vars: ``ROUTEIQ_EVALUATOR_ENABLED``, ``ROUTEIQ_EVALUATOR_PLUGINS``.
    """

    enabled: bool = Field(
        False,
        description="Enable evaluator hooks for post-invocation scoring.",
    )
    plugins: str = Field(
        "",
        description="Comma-separated list of evaluator plugin paths.",
    )


class ContentFilterSettings(BaseModel):
    """Content/toxicity filter plugin configuration.

    Env vars: ``CONTENT_FILTER_ENABLED``, ``CONTENT_FILTER_THRESHOLD``,
    ``CONTENT_FILTER_ACTION``, ``CONTENT_FILTER_CATEGORIES``.
    """

    enabled: bool = Field(
        False,
        description="Enable content/toxicity filtering.",
    )
    threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Default score threshold for category violations.",
    )
    action: str = Field(
        "block",
        description="Default action: block, warn, or log.",
    )
    categories: str = Field(
        "violence,hate_speech,sexual,self_harm,illegal_activity",
        description="Comma-separated list of active filter categories.",
    )


class AgenticPipelineSettings(BaseModel):
    """Agentic multi-round routing pipeline configuration.

    Decomposes complex queries into sub-queries, routes independently,
    and aggregates responses.

    Env vars: ``ROUTEIQ_AGENTIC_PIPELINE``,
    ``ROUTEIQ_AGENTIC_ORCHESTRATOR_MODEL``,
    ``ROUTEIQ_AGENTIC_COMPLEXITY_THRESHOLD``,
    ``ROUTEIQ_AGENTIC_MAX_SUBQUERIES``,
    ``ROUTEIQ_AGENTIC_PARALLEL``,
    ``ROUTEIQ_AGENTIC_TIMEOUT``.
    """

    enabled: bool = Field(
        False,
        description="Enable the agentic multi-round routing pipeline.",
    )
    orchestrator_model: str = Field(
        "gpt-4o-mini",
        description="Model for decompose/aggregate orchestration.",
    )
    complexity_threshold: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Queries above this complexity score get decomposed.",
    )
    max_subqueries: int = Field(
        4,
        ge=1,
        le=10,
        description="Maximum decomposition fan-out.",
    )
    parallel: bool = Field(
        True,
        description="Execute sub-queries in parallel.",
    )
    timeout: float = Field(
        30.0,
        ge=1.0,
        description="Per sub-query timeout in seconds.",
    )


class PromptManagementSettings(BaseModel):
    """Prompt management & versioning configuration.

    Env var: ``ROUTEIQ_PROMPT_MANAGEMENT``.
    """

    enabled: bool = Field(
        False,
        description="Enable prompt management and versioning.",
    )


class CostTrackerSettings(BaseModel):
    """Cost tracker plugin configuration.

    Env var: ``COST_TRACKER_ENABLED``.
    """

    enabled: bool = Field(
        True,
        description="Enable per-request LLM cost tracking.",
    )


class SkillsDiscoverySettings(BaseModel):
    """Skills discovery plugin configuration.

    Env var: ``ROUTEIQ_SKILLS_DIR``.
    """

    skills_dir: Optional[str] = Field(
        None,
        description="Directory to load skills from (default: ./skills or ./docs/skills).",
    )


class EvalPipelineSettings(BaseModel):
    """Evaluation pipeline configuration.

    Closes the feedback loop between routing decisions and quality outcomes.
    When enabled, a fraction of requests are evaluated by an LLM-as-judge
    and the resulting quality scores feed back into routing strategies.

    Env vars: ``ROUTEIQ_EVAL_PIPELINE``, ``ROUTEIQ_EVAL_SAMPLE_RATE``,
    ``ROUTEIQ_EVAL_JUDGE_MODEL``, ``ROUTEIQ_EVAL_BATCH_SIZE``,
    ``ROUTEIQ_EVAL_FEEDBACK_INTERVAL``.
    """

    enabled: bool = Field(
        False,
        description="Enable the evaluation feedback loop pipeline.",
    )
    sample_rate: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of requests to evaluate (0.0-1.0).",
    )
    judge_model: str = Field(
        "gpt-4o-mini",
        description="LLM model used for LLM-as-judge evaluations.",
    )
    batch_size: int = Field(
        10,
        ge=1,
        description="Number of samples per evaluation batch.",
    )
    feedback_interval: int = Field(
        300,
        ge=10,
        description="Seconds between routing feedback updates.",
    )


class HASettings(BaseModel):
    """High-availability and leader election configuration.

    Env vars: ``LLMROUTER_HA_MODE``, ``ROUTEIQ_LEADER_MIGRATIONS``,
    ``ROUTEIQ_LEADER_ELECTION_BACKEND``.
    """

    mode: str = Field(
        "",
        description="HA mode (empty = disabled, set to 'leader-election').",
    )
    leader_migrations: bool = Field(
        False,
        description="Run DB migrations via leader election at startup.",
    )
    leader_election_backend: str = Field(
        "",
        description=(
            "Explicit leader election backend override: "
            "kubernetes, redis, postgres, none.  "
            "When empty, auto-detected from environment."
        ),
    )


# ============================================================================
# Root Settings
# ============================================================================


class GatewaySettings(BaseSettings):
    """Root configuration for RouteIQ Gateway.

    All settings can be overridden via environment variables.
    Nested models use underscore-separated prefixes
    (e.g. ``ROUTEIQ_REDIS_HOST``, ``ROUTEIQ_OTEL_ENABLED``).

    LiteLLM pass-through variables (``LITELLM_*``) use
    ``validation_alias`` so the existing env var names are preserved.
    """

    model_config = SettingsConfigDict(
        env_prefix="ROUTEIQ_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Core gateway settings
    # ------------------------------------------------------------------

    config_path: Optional[str] = Field(
        None,
        description="Path to YAML config file.",
        json_schema_extra={"env": "LITELLM_CONFIG_PATH"},
    )
    port: int = Field(
        4000,
        ge=1,
        le=65535,
        description="HTTP port to listen on.",
    )
    host: str = Field(
        "0.0.0.0",
        description="Host to bind to.",
    )
    workers: int = Field(
        1,
        ge=1,
        description=(
            "Uvicorn worker count.  Multi-worker requires "
            "plugin strategy mode (the default)."
        ),
    )
    env: str = Field(
        "production",
        description="Environment: production, staging, development.",
    )
    debug: bool = Field(
        False,
        description="Enable debug mode.",
    )
    admin_ui_enabled: bool = Field(
        False,
        description="Enable admin UI at /ui/.",
    )
    admin_ui_external_url: Optional[str] = Field(
        None,
        description=(
            "External URL where the UI is hosted (for CORS).  "
            "Leave empty when UI is embedded in the gateway container.  "
            "Set to the UI origin (e.g. 'https://ui.routeiq.example.com') "
            "for disaggregated deployments so the gateway auto-adds it "
            "to CORS allowed origins."
        ),
    )
    skip_env_validation: bool = Field(
        False,
        description="Skip advisory env validation at startup.",
    )
    own_app: bool = Field(
        True,
        description=(
            "Use RouteIQ-owned FastAPI app (ADR-0012).  "
            "Set to false to fall back to legacy mode where LiteLLM "
            "owns the FastAPI instance."
        ),
    )

    # ------------------------------------------------------------------
    # LiteLLM pass-through (preserve existing env var names)
    # ------------------------------------------------------------------

    litellm_master_key: Optional[str] = Field(
        None,
        description="Master API key for admin access (LITELLM_MASTER_KEY).",
    )
    litellm_config_path: Optional[str] = Field(
        None,
        description="Default config path (LITELLM_CONFIG_PATH).",
    )
    litellm_port: int = Field(
        4000,
        ge=1,
        le=65535,
        description="LiteLLM proxy port (LITELLM_PORT).",
    )
    litellm_host: str = Field(
        "0.0.0.0",
        description="LiteLLM proxy host (LITELLM_HOST).",
    )
    litellm_debug: bool = Field(
        False,
        description="LiteLLM debug mode (LITELLM_DEBUG).",
    )

    # ------------------------------------------------------------------
    # LLMRouter settings (LLMROUTER_* prefix)
    # ------------------------------------------------------------------

    llmrouter_router_callback_enabled: bool = Field(
        True,
        description="Emit router decision telemetry (TG4.1).",
    )

    # ------------------------------------------------------------------
    # Sub-configurations (nested models)
    # ------------------------------------------------------------------

    redis: RedisSettings = Field(
        default_factory=RedisSettings,
        description="Redis connection settings.",
    )
    postgres: PostgresSettings = Field(
        default_factory=PostgresSettings,
        description="PostgreSQL connection settings.",
    )
    otel: OTelSettings = Field(
        default_factory=OTelSettings,
        description="OpenTelemetry observability settings.",
    )
    oidc: OIDCSettings = Field(
        default_factory=OIDCSettings,
        description="OIDC / SSO integration settings.",
    )
    routing: RoutingSettings = Field(
        default_factory=RoutingSettings,
        description="ML routing strategy settings.",
    )
    security: SecuritySettings = Field(
        default_factory=SecuritySettings,
        description="Security and access control settings.",
    )
    policy: PolicySettings = Field(
        default_factory=PolicySettings,
        description="Policy engine settings.",
    )
    audit: AuditSettings = Field(
        default_factory=AuditSettings,
        description="Audit logging settings.",
    )
    quota: QuotaSettings = Field(
        default_factory=QuotaSettings,
        description="Quota enforcement settings.",
    )
    mcp: MCPSettings = Field(
        default_factory=MCPSettings,
        description="MCP gateway settings.",
    )
    a2a: A2ASettings = Field(
        default_factory=A2ASettings,
        description="A2A gateway settings.",
    )
    resilience: ResilienceSettings = Field(
        default_factory=ResilienceSettings,
        description="Backpressure and resilience settings.",
    )
    config_sync: ConfigSyncSettings = Field(
        default_factory=ConfigSyncSettings,
        description="Remote config sync settings.",
    )
    http_client: HTTPClientSettings = Field(
        default_factory=HTTPClientSettings,
        description="Shared HTTP client pool settings.",
    )
    ssrf: SSRFSettings = Field(
        default_factory=SSRFSettings,
        description="SSRF protection settings.",
    )
    conversation_affinity: ConversationAffinitySettings = Field(
        default_factory=ConversationAffinitySettings,
        description="Conversation routing affinity settings.",
    )
    plugins: PluginSettings = Field(
        default_factory=PluginSettings,
        description="Plugin system settings.",
    )
    management: ManagementSettings = Field(
        default_factory=ManagementSettings,
        description="Management middleware settings.",
    )
    cache: CacheSettings = Field(
        default_factory=CacheSettings,
        description="Caching plugin settings.",
    )
    ha: HASettings = Field(
        default_factory=HASettings,
        description="High-availability settings.",
    )
    eval_pipeline: EvalPipelineSettings = Field(
        default_factory=EvalPipelineSettings,
        description="Evaluation feedback loop pipeline settings.",
    )
    router_r1: RouterR1Settings = Field(
        default_factory=RouterR1Settings,
        description="Router-R1 iterative reasoning router settings.",
    )
    context_optimizer: ContextOptimizerSettings = Field(
        default_factory=ContextOptimizerSettings,
        description="Context optimizer plugin settings.",
    )
    evaluator: EvaluatorSettings = Field(
        default_factory=EvaluatorSettings,
        description="Evaluator plugin settings.",
    )
    content_filter: ContentFilterSettings = Field(
        default_factory=ContentFilterSettings,
        description="Content/toxicity filter settings.",
    )
    agentic_pipeline: AgenticPipelineSettings = Field(
        default_factory=AgenticPipelineSettings,
        description="Agentic multi-round routing pipeline settings.",
    )
    prompt_management: PromptManagementSettings = Field(
        default_factory=PromptManagementSettings,
        description="Prompt management and versioning settings.",
    )
    cost_tracker: CostTrackerSettings = Field(
        default_factory=CostTrackerSettings,
        description="Cost tracker plugin settings.",
    )
    skills_discovery: SkillsDiscoverySettings = Field(
        default_factory=SkillsDiscoverySettings,
        description="Skills discovery plugin settings.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("litellm_master_key")
    @classmethod
    def _validate_master_key(cls, v: Optional[str]) -> Optional[str]:
        """Warn if the master key looks like a placeholder."""
        if v and v.lower() in ("sk-1234", "changeme", "test", "your-key-here"):
            warnings.warn(
                "LITELLM_MASTER_KEY appears to be a placeholder value — "
                "admin endpoints may be unprotected",
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def _cross_validate(self) -> "GatewaySettings":
        """Cross-field validation after all fields are populated."""
        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins string into a list."""
        return [o.strip() for o in self.security.cors_origins.split(",")]

    @property
    def admin_api_keys_set(self) -> set[str]:
        """Parse admin API keys into a set (includes both fields)."""
        keys: set[str] = set()
        if self.security.admin_api_keys:
            for key in self.security.admin_api_keys.split(","):
                key = key.strip()
                if key:
                    keys.add(key)
        if self.security.admin_api_key:
            keys.add(self.security.admin_api_key.strip())
        return keys

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env.lower() == "production"

    @property
    def redis_configured(self) -> bool:
        """Check if Redis is configured (host is set)."""
        return self.redis.host is not None

    @property
    def postgres_configured(self) -> bool:
        """Check if PostgreSQL is configured (URL is set)."""
        return self.postgres.url is not None

    # ------------------------------------------------------------------
    # Legacy env-var compatibility helpers
    # ------------------------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        """Customise settings sources to include legacy env var mappings.

        Priority order (highest first):
        1. Init kwargs (programmatic overrides)
        2. Environment variables (with ROUTEIQ_ prefix)
        3. Dotenv files
        4. File secrets
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


# ============================================================================
# Singleton
# ============================================================================

_settings: Optional[GatewaySettings] = None


def get_settings(**overrides: Any) -> GatewaySettings:
    """Get or create the settings singleton.

    On first call, creates the ``GatewaySettings`` instance by reading
    environment variables.  Subsequent calls return the cached instance.

    Args:
        **overrides: Keyword arguments passed to ``GatewaySettings()`` on
            first creation only (useful for testing).

    Returns:
        The gateway settings singleton.
    """
    global _settings
    if _settings is None:
        _settings = GatewaySettings(**overrides)
    return _settings


def reset_settings() -> None:
    """Reset the settings singleton.

    **Must** be called in test fixtures (``autouse=True``) to prevent
    cross-test contamination from cached settings.
    """
    global _settings
    _settings = None
