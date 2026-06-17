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
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource

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


class GovernanceFailMode(str, Enum):
    """Governance budget/rate enforcement behaviour when the spend store fails.

    ``OPEN`` (default, back-compat) allows the request when the store
    (ElastiCache/Aurora) cannot confirm spend or RPM -- the historical
    fail-open behaviour.  ``CLOSED`` denies the request when a limit IS
    configured but the store is unavailable, so a store outage cannot leak
    spend/rate past the configured budget/RPM (RouteIQ-24fc).
    """

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
    username: Optional[str] = Field(
        None,
        description=(
            "Redis ACL/IAM username (REDIS_USERNAME). On the ADR-0029 IAM-auth path "
            "this is the CacheIamUserName (user_id == user_name). Read directly from "
            "the legacy REDIS_USERNAME env var by redis_pool._redis_settings()."
        ),
    )
    iam_auth: bool = Field(
        False,
        description=(
            "Mint an elasticache:Connect SigV4 token as the AUTH and present "
            "username as the IAM user (ADR-0029) instead of a static REDIS_PASSWORD. "
            "Env: ROUTEIQ_REDIS_IAM_AUTH (flat) -- mapped onto this nested field by "
            "GatewaySettings._map_flat_iam_aliases. Default OFF (static-cred path "
            "unchanged)."
        ),
    )
    iam_region: Optional[str] = Field(
        None,
        description=(
            "AWS region for the elasticache:Connect signer. Falls back to host "
            "parse / AWS_REGION. Serverless endpoints use shortened region tokens "
            "(use1), so set this or AWS_REGION explicitly on the serverless path."
        ),
    )
    cluster_mode: bool = Field(
        False,
        description=(
            "Build a ``redis.asyncio.RedisCluster`` / sync ``redis.RedisCluster`` "
            "client instead of a single-endpoint ``Redis`` (RouteIQ-5d8f). For "
            "ElastiCache cluster-mode-enabled / Redis Cluster deployments where the "
            "host is the cluster configuration endpoint. The static-cred + IAM-auth "
            "(SigV4 elasticache:Connect) paths both apply to the cluster client. "
            "Env: ROUTEIQ_REDIS__CLUSTER_MODE. Default OFF -> the single-endpoint "
            "path is byte-for-byte unchanged."
        ),
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
    iam_auth: bool = Field(
        False,
        description=(
            "Mint a short-lived (15-min) rds-db:connect IAM token per connection "
            "instead of using a static password (ADR-0028). Env: ROUTEIQ_DB_IAM_AUTH "
            "(flat) -- mapped onto this nested field by "
            "GatewaySettings._map_flat_iam_aliases. Default OFF (static-cred path "
            "unchanged)."
        ),
    )
    iam_region: Optional[str] = Field(
        None,
        description=(
            "AWS region for the rds-db:connect signer. Falls back to host parse "
            "(.<region>.rds.amazonaws.com) / AWS_REGION."
        ),
    )
    reader_host: Optional[str] = Field(
        None,
        description=(
            "Optional read-only endpoint host (Aurora reader endpoint / RDS Proxy "
            "read endpoint) for read replica routing (RouteIQ-65ed). When set, a "
            "SEPARATE read pool is built against this host (same port/db/user/IAM "
            "discipline as the writer) and used for read-only queries. Default "
            "empty -> reads go to the writer pool, so the single-pool path is "
            "byte-for-byte unchanged. Env: ROUTEIQ_POSTGRES__READER_HOST."
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
    routing_decision_log_enabled: bool = Field(
        True,
        description=(
            "Emit the structured ``routing_decision`` JSON log line per request "
            "(ADR-0027 P2 data-lake / CloudWatch metric-filter source). PII-safe: "
            "carries model names, token counts, latency, and a query_length only - "
            "never prompt or completion text. Disable to suppress the line "
            "entirely (env: ROUTEIQ_OTEL__ROUTING_DECISION_LOG_ENABLED)."
        ),
    )
    routing_decision_logger_name: str = Field(
        "routeiq.routing_decision",
        description=(
            "Dedicated logger name the structured routing_decision JSON line is "
            "emitted on. Routed to the dedicated CloudWatch routing log group by "
            "the Fluent Bit pipeline (the data-lake + dimensioned metric filter "
            "source). Kept off the root logger so it is independently routable."
        ),
    )
    error_log_enabled: bool = Field(
        True,
        description=(
            "Emit a structured error JSON line (top-level lowercased "
            '``"level": "error"`` + ``"event": "error"``) on the dedicated '
            "routing logger from the error path (ADR-0027 P2 hardening, "
            "RouteIQ-731c). This is the field the CloudWatch ``RouterErrorFilter`` "
            'selects on (``$.level = "error"``); without it the router-error-count '
            "alarm can never fire because no emitter produces a top-level ``level`` "
            "key. PII-safe: carries the error type + a scrubbed error message only. "
            "Disable to suppress the line "
            "(env: ROUTEIQ_OTEL__ERROR_LOG_ENABLED)."
        ),
    )
    xray_enabled: bool = Field(
        False,
        description=(
            "Emit AWS X-Ray-format trace IDs and propagate the "
            "``X-Amzn-Trace-Id`` header (RouteIQ-3c0a). When True, a NEW OTel "
            "TracerProvider is constructed with the AWS X-Ray ``id_generator`` "
            "(so spans carry X-Ray-format trace IDs) and the AWS X-Ray "
            "propagator is installed as the global text-map, giving CloudWatch "
            "Transaction Search / X-Ray a single edge-to-Bedrock trace. Default "
            "OFF so trace IDs stay W3C-format unless an operator opts in. "
            "Requires the optional ``opentelemetry-sdk-extension-aws`` + "
            "``opentelemetry-propagator-aws-xray`` packages (the ``otel`` "
            "extra); their absence degrades to W3C with a warning (never breaks "
            "boot). The X-Ray ID generator only applies when RouteIQ creates the "
            "provider - it cannot retrofit an existing reused SDK provider "
            "(env: ROUTEIQ_OTEL__XRAY_ENABLED)."
        ),
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


class CostCascadeSettings(BaseModel):
    """Cost-aware speculative-cascade routing strategy configuration (RouteIQ-90d0).

    Distinct from :class:`CostAwareRoutingStrategy` (heuristic Pareto, single-shot):
    the cascade orders deployments cheapest -> strongest and routes to the
    CHEAPEST capable rung first.  Each selection exposes the ordered escalation
    ladder in ``deployment["metadata"]["routeiq_cascade"]`` so the caller can
    retry UP the ladder on a low-quality/low-confidence signal (mode (a)).  When
    a prior-attempt confidence/rung signal is present in
    ``context.request_kwargs`` / ``context.metadata`` (keys ``cascade_rung`` or
    ``cascade_confidence``), the strategy advances to the NEXT rung itself
    (mode (b), confidence-gated).

    Costs are read from ``litellm.model_cost``; arms with unknown cost sort LAST
    (treated as most expensive) so the cheap-first invariant degrades gracefully.

    Env vars: ``ROUTEIQ_COST_CASCADE__ENABLED``,
    ``ROUTEIQ_COST_CASCADE__CONFIDENCE_THRESHOLD``,
    ``ROUTEIQ_COST_CASCADE__MAX_RUNGS`` (``__`` nested delimiter).
    """

    enabled: bool = Field(
        False,
        description="Register the cost-aware cascade strategy at startup.",
    )
    confidence_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum prior-attempt confidence below which the cascade escalates "
            "to the next (pricier/stronger) rung.  Only consulted in the "
            "confidence-gated mode (b) when a prior signal is present."
        ),
    )
    max_rungs: int = Field(
        4,
        ge=1,
        le=32,
        description=(
            "Maximum number of rungs to expose on the escalation ladder "
            "(cheapest -> strongest).  Bounds the ladder length in the returned "
            "metadata and caps how far mode (b) can escalate."
        ),
    )


class SemanticIntentSettings(BaseModel):
    """Semantic / embedding intent-router configuration (RouteIQ-7936).

    Routes by request INTENT -> model-group.  Classifies the request via
    embedding similarity to per-intent centroids (reusing the shared
    SentenceTransformer encoder), then maps the winning intent to a configured
    model-group / deployment subset.

    HOT-PATH DISCIPLINE: the embedding model is NEVER cold-loaded on the request
    path.  The strategy only classifies when the shared encoder is ALREADY
    loaded (mirrors the centroid_routing discipline); otherwise it falls through
    to a graceful fallback deployment.  Pre-warm at startup to enable
    classification.

    Env vars: ``ROUTEIQ_SEMANTIC_INTENT__ENABLED``,
    ``ROUTEIQ_SEMANTIC_INTENT__INTENT_MODEL_GROUPS`` (JSON map),
    ``ROUTEIQ_SEMANTIC_INTENT__SIMILARITY_THRESHOLD`` (``__`` nested delimiter).
    """

    enabled: bool = Field(
        False,
        description="Register the semantic intent router at startup.",
    )
    intent_model_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Map of intent label -> list of model name patterns that form the "
            "target model group for that intent (e.g. "
            '``{"code": ["gpt-4o", "claude-sonnet"], "chat": ["gpt-4o-mini"]}``). '
            "A request classified to an intent is routed to a deployment whose "
            "model matches one of the patterns; an unmapped intent falls back."
        ),
    )
    similarity_threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum top-1 cosine similarity to accept an intent classification. "
            "Below this the request is treated as UNMAPPED and falls back."
        ),
    )


class TagRoutingSettings(BaseModel):
    """Tag / regex / User-Agent routing strategy configuration (RouteIQ-6865).

    Upstream LiteLLM ``tag_based_routing`` is BYPASSED by the RouteIQ custom
    routing strategy (``set_custom_routing_strategy``), so RouteIQ re-implements
    tag routing natively.  A request is matched against three signal sources, in
    priority order, and the FIRST matching rule selects a model-group subset:

    1. An explicit request **tag** (``context.request_kwargs["tags"]`` /
       ``context.metadata["tags"]`` -- list or scalar) matched against
       ``tag_model_groups`` keys (exact).
    2. A **regex** over the request text (last user message / input) matched
       against ``regex_model_groups`` keys (each key is a Python regex).
    3. The **User-Agent** header (``context.request_kwargs["headers"]`` /
       ``context.metadata["headers"]``, case-insensitive) matched against
       ``user_agent_model_groups`` keys (substring, case-insensitive).

    Each value is an ordered list of model-name patterns; the request is routed
    to the first candidate whose ``litellm_params.model`` matches one pattern
    (exact first, then substring).  No match (or no candidate in the matched
    group) falls back to the first available candidate -- never ``None`` unless
    the group is empty.

    Env vars: ``ROUTEIQ_TAG_ROUTING__ENABLED``,
    ``ROUTEIQ_TAG_ROUTING__TAG_MODEL_GROUPS`` (JSON map),
    ``ROUTEIQ_TAG_ROUTING__REGEX_MODEL_GROUPS`` (JSON map),
    ``ROUTEIQ_TAG_ROUTING__USER_AGENT_MODEL_GROUPS`` (JSON map) (``__`` delimiter).
    """

    enabled: bool = Field(
        False,
        description="Register the tag/regex/User-Agent routing strategy at startup.",
    )
    tag_model_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Map of request tag -> ordered list of model-name patterns. A tag in "
            "``request_kwargs['tags']`` / ``metadata['tags']`` dispatches to the "
            'first matching candidate in its group (e.g. {"premium": ["gpt-4o"]}).'
        ),
    )
    regex_model_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Map of Python regex -> ordered list of model-name patterns. The "
            "request text is matched against each key; the first matching regex "
            "dispatches to its group. Consulted only when no tag matched."
        ),
    )
    user_agent_model_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Map of User-Agent substring (case-insensitive) -> ordered list of "
            "model-name patterns. Matched against the request User-Agent header. "
            "Consulted only when neither a tag nor a regex matched."
        ),
    )


class LatencyAwareSettings(BaseModel):
    """Latency-SLA / usage-based / least-busy routing strategy configuration
    (RouteIQ-904b).

    Upstream LiteLLM's ``latency-based-routing`` / ``usage-based-routing`` /
    ``least-busy`` strategies are BYPASSED by RouteIQ's custom routing strategy
    (``set_custom_routing_strategy``).  These three RouteIQ-native strategies
    re-implement that family, reading LIVE signals from the per-worker
    :class:`~litellm_llmrouter.router_decision_callback.RoutingStatsAccumulator`
    (decision counts + rolling latency) and a per-strategy in-flight counter,
    and degrade gracefully (deterministic fallback) when no signal exists yet.

    All three share this one settings block (gated by ``enabled``); each is
    independently registry-registered under its own name:
    ``llmrouter-latency-sla`` / ``llmrouter-usage-based`` /
    ``llmrouter-least-busy``.

    Env vars: ``ROUTEIQ_LATENCY_AWARE__ENABLED``,
    ``ROUTEIQ_LATENCY_AWARE__P_LATENCY_TARGET_MS``,
    ``ROUTEIQ_LATENCY_AWARE__PERCENTILE`` (``__`` nested delimiter).
    """

    enabled: bool = Field(
        False,
        description=(
            "Register the latency-SLA / usage-based / least-busy strategies "
            "at startup (all three, each under its own registry name)."
        ),
    )
    p_latency_target_ms: float = Field(
        2000.0,
        ge=0.0,
        description=(
            "Latency-SLA target (milliseconds) at the configured percentile. The "
            "latency-SLA strategy prefers the cheapest/first deployment whose "
            "observed percentile latency is at or below this target; if none "
            "meet it, it picks the lowest-latency deployment (best effort)."
        ),
    )
    percentile: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Latency percentile the SLA target is evaluated at (e.g. 0.95 = p95). "
            "Computed from the per-model latency samples held by the strategy."
        ),
    )
    max_latency_samples: int = Field(
        256,
        ge=1,
        le=4096,
        description=(
            "Per-model ring-buffer size for latency samples used to estimate the "
            "percentile. Bounded so memory stays constant on the hot path."
        ),
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
        "",
        description="Comma-separated allowed CORS origins. Empty = no CORS (must be explicitly configured).",
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


class GovernanceSettings(BaseModel):
    """Workspace/key governance budget + rate-limit enforcement behaviour.

    Env vars: nested ``ROUTEIQ_GOVERNANCE__FAIL_MODE`` (the ``__`` nested
    delimiter on :class:`GatewaySettings`).  The consumer in ``governance.py``
    ALSO honours the legacy flat ``ROUTEIQ_GOVERNANCE_FAIL_MODE`` via an
    ``os.getenv`` fallback (mirrors the ``QuotaConfig.from_env`` pattern), so
    operators can use either form.
    """

    fail_mode: GovernanceFailMode = Field(
        GovernanceFailMode.OPEN,
        description=(
            "Behaviour when the governance spend store (ElastiCache/Aurora) is "
            "unavailable.  OPEN (default) allows; CLOSED denies when a budget/RPM "
            "limit is configured but the store cannot confirm current usage."
        ),
    )
    banned_models: list[str] = Field(
        default_factory=list,
        description=(
            "Models removed from the routable candidate set BEFORE the routing "
            "strategy/bandit scores (data-retention / compliance ban, e.g. "
            "Fable 5). Matched against ``litellm_params.model`` (the bandit arm "
            "key) and ``model_name`` (the group name). Enforced RouteIQ-native "
            "(see ``candidate_filter.py``), NOT via LiteLLM's "
            "``async_filter_deployments`` hook -- that hook lives on the "
            "built-in selection path the custom strategy bypasses. Default empty "
            "(no ban) -> byte-stable no-op. "
            "Env: ``ROUTEIQ_GOVERNANCE__BANNED_MODELS``."
        ),
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

    # --- AWS AppConfig runtime retrieval (ADR-0026, RouteIQ-4333) -------------
    # Poll a deployed AppConfig configuration profile via the AppConfigData
    # data-plane API. Off by default so the chart/app stays cloud-agnostic. The
    # pod-role grant for this is RouteIQ-569f (appconfigdata:StartConfigurationSession
    # + appconfigdata:GetLatestConfiguration scoped to the profile ARN). Env vars
    # are nested under the ``config_sync`` prefix
    # (e.g. ROUTEIQ_CONFIG_SYNC__APPCONFIG_ENABLED).
    appconfig_enabled: bool = Field(
        False,
        description=(
            "Enable runtime AWS AppConfig retrieval (ADR-0026). Polls a deployed "
            "AppConfig configuration profile via the AppConfigData data-plane API "
            "(StartConfigurationSession + GetLatestConfiguration) and writes a "
            "changed body to the local config path. Default off."
        ),
    )
    appconfig_application: Optional[str] = Field(
        None,
        description=(
            "AppConfig application id or name to poll (CfnOutput "
            "AppConfigApplicationId). Required when appconfig_enabled is true."
        ),
    )
    appconfig_environment: Optional[str] = Field(
        None,
        description=(
            "AppConfig environment id or name to poll (e.g. the deploy env). "
            "Required when appconfig_enabled is true."
        ),
    )
    appconfig_profile: Optional[str] = Field(
        None,
        description=(
            "AppConfig configuration profile id or name to poll. Required when "
            "appconfig_enabled is true."
        ),
    )
    appconfig_poll_interval_seconds: int = Field(
        60,
        ge=15,
        description=(
            "Requested AppConfig poll interval in seconds (the data-plane API "
            "enforces a minimum of 15s; the loop honors the server's "
            "NextPollIntervalInSeconds when shorter)."
        ),
    )


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


class KumaraswamyThompsonSettings(BaseModel):
    """Kumaraswamy-Thompson online-bandit routing strategy configuration.

    The core bandit runs with the in-memory backend and NO external deps:
    ``InMemoryPosteriorBackend`` is the byte-stable DEFAULT backend.

    DURABLE (RouteIQ-95a8): set ``backend='file'`` to select the cred-free,
    pure-stdlib ``FilePosteriorBackend`` — it mirrors the in-memory semantics but
    persists posteriors to ``state_path`` via an atomic ``json.dump`` +
    ``os.replace`` (the governance-store pattern) so a worker restart RESUMES
    convergence instead of starting cold.  Persistence is debounced off the hot
    path; the prior posteriors are loaded at startup.  The default stays
    ``memory`` for byte-stable behavior.

    RESERVED / not yet wired (no consumer): the ``durable`` and
    ``cost_reward_alpha`` fields below have ZERO readers.
    ``register_kumaraswamy_thompson_strategy`` never reads them, and the durable
    Redis (hot) / Aurora (durable) posterior backends they would select are NOT
    built.  The fields are retained for env-var forward-compat only (RouteIQ-4654;
    the module docstring in ``kumaraswamy_thompson.py`` documents the same).

    Env vars: ``ROUTEIQ_KUMARASWAMY_THOMPSON__ENABLED``,
    ``ROUTEIQ_KUMARASWAMY_THOMPSON__BACKEND``,
    ``ROUTEIQ_KUMARASWAMY_THOMPSON__STATE_PATH``, etc. (``__`` nested delimiter).
    """

    enabled: bool = Field(
        False,
        description="Register the Kumaraswamy-Thompson strategy at startup.",
    )
    backend: str = Field(
        "memory",
        description=(
            "Posterior backend (memory | file).  ``memory`` (default) is the "
            "byte-stable per-worker ``InMemoryPosteriorBackend``.  ``file`` "
            "selects the cred-free DURABLE ``FilePosteriorBackend`` which "
            "persists posteriors to ``state_path`` (atomic json.dump + "
            "os.replace) and reloads them at startup so a restart resumes "
            "convergence (RouteIQ-95a8).  The durable Redis/Aurora backends "
            "remain NOT built (RouteIQ-4654)."
        ),
    )
    state_path: str = Field(
        "",
        description=(
            "Filesystem path for the DURABLE ``file`` backend's posterior "
            "snapshot (RouteIQ-95a8).  Only consulted when ``backend='file'``; "
            "empty disables persistence (the file backend then behaves like the "
            "in-memory one).  Env: "
            "``ROUTEIQ_KUMARASWAMY_THOMPSON__STATE_PATH``."
        ),
    )
    flush_interval_seconds: float = Field(
        5.0,
        ge=0.0,
        description=(
            "DURABLE ``file`` backend (RouteIQ-95a8): max seconds between debounced "
            "persists.  0 disables the time-based flush (count-based only)."
        ),
    )
    flush_dirty_threshold: int = Field(
        32,
        ge=1,
        description=(
            "DURABLE ``file`` backend (RouteIQ-95a8): persist after this many "
            "posterior mutations accumulate (off-hot-path debounce).  1 = "
            "write-through."
        ),
    )
    durable: str = Field(
        "none",
        description=(
            "RESERVED / not yet wired (no consumer): durable posterior store "
            "(none | aurora).  The durable Aurora posterior backend is NOT built "
            "and ``register_kumaraswamy_thompson_strategy`` never reads this "
            "field (RouteIQ-4654).  Retained for env-var forward-compat."
        ),
    )
    w_quality: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Reward weight on response quality.",
    )
    w_cost: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Reward weight on cost (cheaper => higher).",
    )
    w_latency: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Reward weight on latency (faster => higher).",
    )
    cost_reward_alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "RESERVED / not yet wired (no consumer): mixing rate of the cost "
            "term into the reward signal.  "
            "``register_kumaraswamy_thompson_strategy`` never reads this field "
            "(the strategy uses ``w_quality`` / ``w_cost`` / ``w_latency`` "
            "instead) — RouteIQ-4654.  Retained for env-var forward-compat."
        ),
    )
    decay_gamma: float = Field(
        0.99,
        ge=0.5,
        le=1.0,
        description="Per-day decay toward the prior (re-opens exploration).",
    )
    cold_start_kappa: float = Field(
        5.0,
        ge=0.0,
        le=50.0,
        description="Warm-start pseudo-count from the static quality table.",
    )
    seed: Optional[int] = Field(
        None,
        description="RNG seed for deterministic sampling in tests/replay.",
    )
    moment_fit: bool = Field(
        False,
        description=(
            "Use the doc-20 §3.1 option-2 moment-fit Beta(alpha,beta) -> "
            "Kumaraswamy(a,b) mapping (matches mean + variance via a 1-D "
            "root-find) instead of the option-1 ``a=alpha, b=beta`` shortcut. "
            "The shortcut distorts the posterior mean (Beta(51,51) mean 0.5 -> "
            "Kumaraswamy 0.9155) and can invert the exploit decision; the "
            "moment-fit restores the mean (~1e-8) and tracks the variance "
            "(~1.0x, vs the shortcut's distortion). Cached per-posterior on the "
            "exact (alpha,beta) so any evidence change refits (RouteIQ-f9e9). "
            "Default off for byte-stable backward-compat. "
            "Env: ``ROUTEIQ_KUMARASWAMY_THOMPSON__MOMENT_FIT``."
        ),
    )


class LinUCBSettings(BaseModel):
    """Feature-vector contextual-bandit (LinUCB) routing strategy configuration
    (RouteIQ-6c67).

    The Kumaraswamy-Thompson bandit is BUCKET-contextual (one posterior per
    coarse ``(task_bucket, arm)`` pair).  LinUCB is FEATURE-VECTOR contextual: it
    learns a linear reward model over a real-valued context vector (prompt-length
    feature, requested-profile one-hot, tenant hash bucket) and scores each arm
    by its UCB ``theta·x + alpha·sqrt(x^T A^-1 x)`` -- the second term drives
    cold-start exploration that shrinks as evidence accumulates.

    ADDITIVE alongside the KT bandit: it is a SEPARATE registry strategy
    (``llmrouter-linucb``), gated independently by ``enabled``.  Pure stdlib
    (no numpy): per-arm state is the incremental ``A^-1`` maintained via
    rank-1 Sherman-Morrison updates, matching the KT module's numpy-free
    discipline.  The RNG (used only for tie-breaking) is threaded as a
    ``random.Random`` object.

    Env vars: ``ROUTEIQ_LINUCB__ENABLED``, ``ROUTEIQ_LINUCB__ALPHA``,
    ``ROUTEIQ_LINUCB__TENANT_BUCKETS``, ``ROUTEIQ_LINUCB__SEED`` (``__`` delim).
    """

    enabled: bool = Field(
        False,
        description="Register the LinUCB contextual-bandit strategy at startup.",
    )
    alpha: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description=(
            "Exploration coefficient on the UCB confidence width. Higher = more "
            "exploration; 0 = pure greedy exploit. Default 1.0 (textbook LinUCB)."
        ),
    )
    tenant_buckets: int = Field(
        8,
        ge=1,
        le=256,
        description=(
            "Number of hashed tenant feature buckets in the context vector. "
            "Bounds the feature dimension so per-arm matrices stay small."
        ),
    )
    seed: Optional[int] = Field(
        None,
        description="RNG seed for deterministic tie-breaking in tests/replay.",
    )


class AdapterFrameworkSettings(BaseModel):
    """Strategy-agnostic routing-adapter framework configuration.

    Env vars: ``ROUTEIQ_ADAPTER_FRAMEWORK__ENTRYPOINT_DISCOVERY``, etc.
    """

    entrypoint_discovery: bool = Field(
        False,
        description="Discover out-of-tree adapters via entry-points at startup.",
    )
    capability_negotiation: bool = Field(
        False,
        description="Enable the pre-selection required-signals filter.",
    )
    mlops_feedback_loop: bool = Field(
        True,
        description=(
            "Wire the eval-pipeline FEEDBACK arm into the strategy-agnostic "
            "MLOps coordinator: discover continuous-train learning adapters "
            "from the registry and subscribe the coordinator to "
            "EvalPipeline.feedback_callbacks. "
            "DECISION (RouteIQ-3b4d): default ON. The intelligent-routing use "
            "case (Claude Code -> RouteIQ -> mixed-Bedrock auto-group) is only "
            "worth it if the bandit LEARNS which arm is best automatically, so "
            "the loop closes by default. It is a NO-OP unless (a) a continuous "
            "learning strategy is registered (e.g. the Kumaraswamy-Thompson "
            "bandit -- itself off by default) AND (b) the eval pipeline is "
            "enabled; with both off this flag costs nothing. OPT-OUT: set "
            "``ROUTEIQ_ADAPTER_FRAMEWORK__MLOPS_FEEDBACK_LOOP=false``. "
            "JUDGE-COST NOTE: when the eval pipeline's LLM-as-judge evaluator is "
            "enabled it makes one extra judge call per sampled decision -- size "
            "the judge model / sample rate accordingly (see "
            "docs/operations/claude-code-routing.md)."
        ),
    )


class MLOpsPromotionSettings(BaseModel):
    """Quality-gated champion/challenger promotion + rollback (RouteIQ-2a1c).

    Drives :class:`litellm_llmrouter.strategy_registry.ChampionChallengerPromoter`.
    When enabled, the promoter compares the aggregated routing quality of a
    designated *challenger* strategy/version against the current *champion* and
    PROMOTES the challenger (``registry.set_active``) when it beats the champion
    by ``margin`` over a window of at least ``min_samples`` evaluations.  If a
    freshly promoted challenger later REGRESSES below the prior champion by the
    same margin, it is ROLLED BACK to the champion.

    The quality signal is the eval loop's ``{strategy: quality}`` aggregate
    (``ModelQualityTracker`` scale, ``[0, 1]``).  Insufficient samples => no
    action (never promote on noise).  Default OFF.
    """

    enabled: bool = Field(
        False,
        description="Enable quality-gated champion/challenger promotion + rollback.",
    )
    margin: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum quality lead (challenger - champion, in [0,1]) required to "
            "promote; the same drop below champion triggers a rollback."
        ),
    )
    min_samples: int = Field(
        20,
        ge=1,
        description=(
            "Minimum aggregated quality samples for BOTH champion and challenger "
            "before any promotion/rollback decision is made."
        ),
    )


class MLOpsDriftSettings(BaseModel):
    """Input drift + routing-quality-regression detection (RouteIQ-6dce).

    Drives :class:`litellm_llmrouter.mlops.drift.DriftDetector`.  Detects (a)
    INPUT DRIFT -- a shift in the request/bucket distribution vs a captured
    baseline (population-stability-index over coarse buckets) -- and (b)
    ROUTING-QUALITY REGRESSION -- the aggregated quality dropping below a
    captured baseline by ``quality_regression_threshold``.  Drift signals are
    emitted as OTel gauges via ``metrics.py`` so CloudWatch / Prometheus can
    alarm.  Default OFF.
    """

    enabled: bool = Field(
        False,
        description="Enable input-drift + quality-regression detection.",
    )
    input_drift_threshold: float = Field(
        0.2,
        ge=0.0,
        description=(
            "Population-Stability-Index threshold above which the input "
            "(bucket) distribution counts as drifted (PSI>=0.2 is the textbook "
            "'moderate shift' line)."
        ),
    )
    quality_regression_threshold: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Absolute aggregated-quality drop (baseline - current, in [0,1]) "
            "above which a routing-quality regression signal fires."
        ),
    )
    min_samples: int = Field(
        30,
        ge=1,
        description=(
            "Minimum observations in BOTH the baseline and current windows "
            "before drift/regression is evaluated (avoids small-sample noise)."
        ),
    )
    window_size: int = Field(
        200,
        ge=1,
        description="Sliding-window size for the current bucket/quality samples.",
    )


class MLOpsShadowSettings(BaseModel):
    """Shadow / mirror (silent canary) traffic to candidate strategies
    (RouteIQ-4fd1).

    Drives :class:`litellm_llmrouter.strategy_registry.ShadowMirror`.  Mirrors a
    sampled fraction of routing decisions to a candidate strategy WITHOUT
    affecting the served response: it computes what the candidate WOULD have
    picked and records it for offline comparison.  No user impact.  Default OFF
    with ``sample_rate=0.0``.
    """

    enabled: bool = Field(
        False,
        description="Enable shadow/mirror evaluation of a candidate strategy.",
    )
    candidate_strategy: str = Field(
        "",
        description=(
            "Registry name of the candidate strategy to shadow-evaluate. Empty "
            "disables mirroring even when enabled."
        ),
    )
    sample_rate: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of served decisions to mirror to the candidate.",
    )
    max_records: int = Field(
        1000,
        ge=1,
        description="Bounded ring buffer of shadow comparison records to retain.",
    )


class MLOpsSettings(BaseModel):
    """MLOps closed-loop configuration (Cluster H).

    Bundles the three MLOps sub-loops that keep routing strategies optimal:
    quality-gated champion/challenger promotion (RouteIQ-2a1c), drift detection
    (RouteIQ-6dce), and shadow/mirror canary traffic (RouteIQ-4fd1).  All three
    are independently settings-gated and DEFAULT OFF -- importing or wiring the
    MLOps machinery changes nothing until an operator opts in.

    Builds on the EXISTING COLLECT/EVALUATE/AGGREGATE/FEEDBACK eval loop
    (``ModelQualityTracker`` / ``MLOpsCoordinator``); it does not duplicate it.

    Env vars (``__`` nested delimiter under the ``ROUTEIQ_`` prefix):
    ``ROUTEIQ_MLOPS__PROMOTION__ENABLED``,
    ``ROUTEIQ_MLOPS__DRIFT__ENABLED``,
    ``ROUTEIQ_MLOPS__SHADOW__ENABLED``, etc.
    """

    promotion: MLOpsPromotionSettings = Field(
        default_factory=MLOpsPromotionSettings,  # type: ignore[arg-type]
        description="Quality-gated champion/challenger promotion + rollback.",
    )
    drift: MLOpsDriftSettings = Field(
        default_factory=MLOpsDriftSettings,  # type: ignore[arg-type]
        description="Input-drift + routing-quality-regression detection.",
    )
    shadow: MLOpsShadowSettings = Field(
        default_factory=MLOpsShadowSettings,  # type: ignore[arg-type]
        description="Shadow/mirror (silent canary) traffic to candidate strategies.",
    )


class BedrockDiscoverySettings(BaseModel):
    """In-process AWS Bedrock model auto-discovery configuration (control plane).

    Drives :mod:`litellm_llmrouter.bedrock_discovery`, which enumerates every
    provider's serverless foundation models and cross-region inference profiles
    across ``source_regions`` and maps them into LiteLLM ``model_list`` entries.

    The scan is a CONTROL-PLANE concern -- it uses ``boto3.client("bedrock")``
    (NOT ``bedrock-runtime``) and only issues the three read-only control APIs
    ``ListFoundationModels`` / ``ListInferenceProfiles`` /
    ``GetFoundationModelAvailability``.

    Default DISABLED (opt-in): with ``enabled=False`` the discovery module is a
    byte-stable no-op, so importing it or calling ``discover_models()`` changes
    nothing until an operator turns it on.  Live multi-region scanning is
    therefore operator-gated.

    Env vars (``__`` nested delimiter under the ``ROUTEIQ_`` prefix):
    ``ROUTEIQ_BEDROCK_DISCOVERY__ENABLED``,
    ``ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS`` (accepts a plain
    comma-separated string ``us-east-1,eu-west-1`` OR a JSON list
    ``["us-east-1","eu-west-1"]`` -- the CSV form is normalised at the env-source
    decode point by :class:`_RouteIQEnvSettingsSource`, the JSON form mirrors the
    ``governance.banned_models`` nested-list convention; either form constructs
    ``GatewaySettings()`` cleanly),
    ``ROUTEIQ_BEDROCK_DISCOVERY__CHECK_AVAILABILITY``,
    ``ROUTEIQ_BEDROCK_DISCOVERY__RESIDENCY_CONSTRAINT``,
    ``ROUTEIQ_BEDROCK_DISCOVERY__REGISTER_COST``.
    """

    enabled: bool = Field(
        False,
        description=(
            "Enable in-process Bedrock model auto-discovery. Default OFF -- the "
            "discovery module no-ops and the model_list is untouched until an "
            "operator opts in (live multi-region scan is operator-gated)."
        ),
    )
    source_regions: list[str] = Field(
        default_factory=list,
        description=(
            "AWS regions to scan for serverless foundation models + inference "
            "profiles. Intersected with get_available_regions('bedrock'). Empty "
            "falls back to the boto3 session default region (AWS_REGION / profile "
            "/ IMDS). Env ``ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS`` accepts "
            "EITHER a comma-separated string (e.g. ``us-east-1,eu-west-1``) OR a "
            'JSON list (e.g. ``["us-east-1","eu-west-1"]``); the CSV form is '
            "normalised by the RouteIQ env source so neither crashes "
            "GatewaySettings()."
        ),
    )
    check_availability: bool = Field(
        False,
        description=(
            "Gate discovered models through GetFoundationModelAvailability "
            "(entitlement check). Default off -- the extra per-model control call "
            "is opt-in; when off all ACTIVE serverless models are kept."
        ),
    )
    residency_constraint: bool = Field(
        False,
        description=(
            "When True the selector SKIPS the global.* inference-profile tier and "
            "prefers an in-geo geographic profile (or the raw modelId) so traffic "
            "stays in-region (data-residency guard, spec #11-12). Default off "
            "(global-first)."
        ),
    )
    register_cost: bool = Field(
        True,
        description=(
            "When mapping to LiteLLM model_list, register a cost-key stub for "
            "global.* ids (often missing from LiteLLM's model_prices JSON, which "
            "breaks provider detection -- issue #17286). Default on."
        ),
    )
    auto_group: bool = Field(
        False,
        description=(
            "Synthesize a SINGLE mixed-Bedrock routing GROUP from discovery: "
            "every discovered serverless arm (Claude + Nova + gpt-oss + any "
            "future provider) shares one ``model_name`` (``auto_group_name``) so "
            "a routing strategy (e.g. the Kumaraswamy-Thompson bandit) cascades "
            "across the arms instead of the caller pinning one tier. This powers "
            "the Claude-Code cost-routing recipe (point ``model`` at the group "
            "name). Default OFF -- byte-stable: when off the per-model distinct "
            "``model_name`` mapping is unchanged. Env: "
            "``ROUTEIQ_BEDROCK_DISCOVERY__AUTO_GROUP``."
        ),
    )
    auto_group_name: str = Field(
        "claude-auto",
        description=(
            "The shared ``model_name`` for the synthesized auto-group (only used "
            "when ``auto_group=True``). Point an unmodified Anthropic client's "
            "``model`` at this value (or rewrite to it via the model-alias layer) "
            "to route through the mixed-Bedrock group. Env: "
            "``ROUTEIQ_BEDROCK_DISCOVERY__AUTO_GROUP_NAME``."
        ),
    )

    @field_validator("source_regions", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        """Accept a comma-separated string OR a list for ``source_regions``.

        This handles DIRECT/programmatic construction
        (``BedrockDiscoverySettings(source_regions="us-east-1,eu-west-1")``).
        The ENV form is normalised earlier, at the source level, by
        :class:`_RouteIQEnvSettingsSource` -- a field validator alone cannot fix
        the env path because pydantic-settings JSON-decodes the complex
        ``list[str]`` before any validator runs.
        """
        if isinstance(v, str):
            return [r.strip() for r in v.split(",") if r.strip()]
        return v


class BedrockRequestLeversSettings(BaseModel):
    """Per-request Bedrock Converse levers (data plane), default-OFF.

    Drives :class:`litellm_llmrouter.gateway.plugins.bedrock_request_levers
    .BedrockRequestLeversPlugin`, which mutates the outbound request on the
    correct litellm seam (``CustomLogger.async_pre_call_deployment_hook``) so
    the mutations actually reach the Bedrock ``Converse`` call args.

    Three independently-gated levers (each a byte-stable no-op when off):

    - ``request_metadata`` (RouteIQ-294a): forward tenant identity
      (workspace / key) as Bedrock ``requestMetadata`` tags for cost
      attribution. Lands as the top-level ``requestMetadata`` completion kwarg,
      which litellm's converse transform maps from ``non_default_params``.
    - ``team_callbacks`` (RouteIQ-9cd8): set per-team logging sinks as the
      TOP-LEVEL request kwargs ``callbacks`` / ``success_callback`` /
      ``failure_callback`` (where litellm sources dynamic per-request
      callbacks), NOT ``metadata[callbacks]`` (never read). The global
      registration is unaffected.
    - ``cache_point`` (RouteIQ-b9ee): drive Bedrock prompt-caching via
      litellm's ``cache_control_injection_points`` mechanism (system-prefix
      message injection + tool_config cachePoint), NOT a bare ``tools[]``
      ``cachePoint`` entry (dropped by the converse transform).

    Env vars (``__`` nested delimiter under ``ROUTEIQ_``):
    ``ROUTEIQ_BEDROCK_LEVERS__REQUEST_METADATA``,
    ``ROUTEIQ_BEDROCK_LEVERS__TEAM_CALLBACKS``,
    ``ROUTEIQ_BEDROCK_LEVERS__CACHE_POINT``,
    ``ROUTEIQ_BEDROCK_LEVERS__CACHE_SYSTEM``,
    ``ROUTEIQ_BEDROCK_LEVERS__CACHE_TOOLS``,
    ``ROUTEIQ_BEDROCK_LEVERS__METADATA_PREFIX``,
    ``ROUTEIQ_BEDROCK_LEVERS__TEAM_CALLBACK_MAP`` (JSON
    ``{"team-a": {"success_callback": ["s3"]}}``).
    """

    request_metadata: bool = Field(
        False,
        description=(
            "RouteIQ-294a: forward tenant identity (workspace_id / key_id) as "
            "Bedrock requestMetadata tags for cost attribution. Default OFF -- "
            "byte-stable: no requestMetadata kwarg is added until enabled."
        ),
    )
    metadata_prefix: str = Field(
        "routeiq_",
        description=(
            "Key prefix for the injected requestMetadata tags (Bedrock keys must "
            "match [a-zA-Z0-9 :_@$#=/+,.-]). Only used when request_metadata=True."
        ),
    )
    team_callbacks: bool = Field(
        False,
        description=(
            "RouteIQ-9cd8: set per-team logging sinks as TOP-LEVEL request kwargs "
            "callbacks/success_callback/failure_callback (the seam litellm reads "
            "dynamic per-request callbacks from). Default OFF; the global "
            "registration is untouched."
        ),
    )
    team_callback_map: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict,
        description=(
            "Map of team_id -> {success_callback|failure_callback|callbacks: "
            '[sink names]}. e.g. {"team-a": {"success_callback": ["s3"]}}. Only '
            "applied when team_callbacks=True and the request resolves to a "
            "mapped team."
        ),
    )
    cache_point: bool = Field(
        False,
        description=(
            "RouteIQ-b9ee: enable Bedrock prompt-caching cachePoint injection via "
            "litellm cache_control_injection_points. Default OFF -- byte-stable."
        ),
    )
    cache_system: bool = Field(
        True,
        description=(
            "When cache_point=True, inject a system-prefix cachePoint (message "
            "injection point, role=system). Default on."
        ),
    )
    cache_tools: bool = Field(
        True,
        description=(
            "When cache_point=True, inject a tool_config cachePoint so the tool "
            "schema prefix is cached. Default on. Only takes effect when the "
            "request carries tools."
        ),
    )


class ModelAliasSettings(BaseModel):
    """Pre-routing model-name alias/rewrite map (RouteIQ-0dcb).

    Claude Code (and any unmodified Anthropic SDK client) PINS a concrete model
    id in the request body, e.g. ``claude-sonnet-4-20250514``. LiteLLM matches
    ``model_list`` rows by EXACT ``model_name``, so a pinned id never lands on a
    synthesized routing group such as ``claude-auto`` -- the request would 400 or
    fall through to a single arm.

    This layer rewrites the request ``model`` field BEFORE the request reaches
    LiteLLM, at the RouteIQ app/route entry layer (a raw-ASGI middleware in
    ``model_alias.py``, NOT the broken ``on_llm_pre_call`` mutation seam). Two
    maps are applied in order:

    * ``exact`` -- an exact ``{requested_id: target_group}`` lookup (fast path).
    * ``regex`` -- ordered ``{pattern: target_group}`` rules; the FIRST full
      ``re.fullmatch`` wins. Patterns are compiled once; an invalid pattern is
      logged and skipped (fail-open -- a bad rule never blocks traffic).

    Default ``enabled=False`` with empty maps => IDENTITY (no rewrite), so the
    behavior is byte-stable until an operator opts in.

    Env vars (``__`` nested delimiter under ``ROUTEIQ_``):
    ``ROUTEIQ_MODEL_ALIAS__ENABLED``,
    ``ROUTEIQ_MODEL_ALIAS__EXACT`` (JSON object),
    ``ROUTEIQ_MODEL_ALIAS__REGEX`` (JSON object of ordered rules).
    """

    enabled: bool = Field(
        False,
        description=(
            "Enable the pre-routing model-alias rewrite layer. Default OFF "
            "(identity -- no rewrite). When on, the request ``model`` field is "
            "rewritten via ``exact`` then ``regex`` before LiteLLM sees it, so an "
            "unmodified Anthropic client pinning a concrete id is transparently "
            "routed through a target group (e.g. ``claude-auto``)."
        ),
    )
    exact: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Exact-match alias map ``{requested_model_id: target_group}``. "
            "Applied first; a hit short-circuits the regex pass. Env "
            "``ROUTEIQ_MODEL_ALIAS__EXACT`` takes a JSON object, e.g. "
            '``{"claude-sonnet-4-20250514": "claude-auto"}``.'
        ),
    )
    regex: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Ordered regex alias rules ``{pattern: target_group}``; the FIRST "
            "rule whose pattern ``re.fullmatch`` matches the requested id wins. "
            "Env ``ROUTEIQ_MODEL_ALIAS__REGEX`` takes a JSON object, e.g. "
            '``{"^claude-.*$": "claude-auto"}``. Dict insertion order is the '
            "evaluation order (Python 3.7+). An uncompilable pattern is logged "
            "and skipped (fail-open)."
        ),
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


class EngineMetricsSettings(BaseModel):
    """Self-hosted-engine ``/metrics`` scrape configuration (default OFF).

    Drives :mod:`litellm_llmrouter.engine_metrics`, the credless scraper that
    GETs a self-hosted inference engine's Prometheus ``/metrics`` (vLLM, vLLM
    Production Stack, AIBrix, llm-d -- all serve the ``vllm:*`` family) and
    parses the queue-depth / KV-cache-pressure gauges
    (``vllm:num_requests_waiting``, ``vllm:kv_cache_usage_perc`` /
    ``vllm:gpu_cache_usage_perc``, ...). Those signals feed a *future*
    KV/queue-aware Layer-1 router or an autoscaler-into-the-engine.

    LAYERING: this reads the engine *frontend's* aggregate ``/metrics`` so
    Layer-1 can decide whether to send more load to this engine arm; it never
    scrapes individual worker pods (that would collapse the two-layer model --
    see ``docs/architecture/aws-rearchitecture/51-multinode-large-model-serving.md``
    Part 3).

    Default DISABLED (opt-in): with ``enabled=False`` the scraper is a byte-stable
    no-op -- importing the module does no I/O, and ``scrape()`` returns an empty
    ``reachable=False`` snapshot without touching the network. Live scraping is
    therefore operator-gated.

    Env vars (``__`` nested delimiter under the ``ROUTEIQ_`` prefix):
    ``ROUTEIQ_ENGINE_METRICS__ENABLED``,
    ``ROUTEIQ_ENGINE_METRICS__SCRAPE_TIMEOUT``.
    """

    enabled: bool = Field(
        False,
        description=(
            "Enable the self-hosted-engine /metrics scrape. Default OFF -- the "
            "scraper no-ops and does zero network I/O until an operator opts in "
            "(live scrape is operator-gated)."
        ),
    )
    scrape_timeout: float = Field(
        2.0,
        ge=0.1,
        le=30.0,
        description=(
            "Per-scrape HTTP GET timeout in seconds. An engine that does not "
            "answer within this budget yields an empty (reachable=False) "
            "snapshot, never an exception."
        ),
    )


# ============================================================================
# Custom env source -- comma-separated coercion for documented list fields
# ============================================================================

#: Nested env fields (``list[str]``) whose DOCUMENTED env form is a plain
#: comma-separated string.  pydantic-settings JSON-decodes complex types
#: (``list[str]``) straight out of the env BEFORE any field/model validator
#: runs, so a CSV string would raise ``SettingsError`` and abort the WHOLE of
#: ``GatewaySettings()`` (not just the affected sub-model).  We normalise the
#: CSV form to a real list at the env-source decode point so both the
#: comma-separated and the JSON-list env forms construct cleanly.
_CSV_LIST_ENV_FIELDS = frozenset({"source_regions"})

#: Nested BaseModel fields whose DOCUMENTED flat env var is a bare bool that
#: should map onto the sub-model's ``enabled`` flag (RouteIQ-778e).  The flat
#: ``ROUTEIQ_EVAL_PIPELINE=true`` collides with the nested ``eval_pipeline``
#: BaseModel field: pydantic-settings JSON-decodes the bool ``true`` and then
#: tries to validate it as an ``EvalPipelineSettings`` model, raising
#: ``ValidationError`` and aborting ALL of ``GatewaySettings()``.  We expand the
#: bare bool to ``{"enabled": <bool>}`` at the decode point so the documented
#: enable var constructs cleanly (the ``__``-nested forms such as
#: ``ROUTEIQ_EVAL_PIPELINE__ENABLED=true`` are handled by pydantic natively and
#: are unaffected).
_FLAT_BOOL_MODEL_ENV_FIELDS = frozenset({"eval_pipeline"})

# Bare-bool string spellings accepted for a flat model-enable env var.
_TRUE_BOOL_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_BOOL_STRINGS = frozenset({"false", "0", "no", "off", ""})


class _RouteIQEnvSettingsSource(EnvSettingsSource):
    """``EnvSettingsSource`` that accepts comma-separated lists for a few fields.

    The base source JSON-decodes every complex (``list[str]`` etc.) nested env
    var.  For the fields in :data:`_CSV_LIST_ENV_FIELDS` we instead accept a
    plain comma-separated string (``a,b,c``) -- the documented env form -- and
    fall back to JSON for anything that parses as JSON (so a real JSON list
    ``["a","b"]`` still works).  This is what makes
    ``ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS=us-east-1,eu-west-1`` boot the
    gateway instead of crashing it.

    For the fields in :data:`_FLAT_BOOL_MODEL_ENV_FIELDS` we accept a DOCUMENTED
    flat bool (e.g. ``ROUTEIQ_EVAL_PIPELINE=true``) and expand it onto the
    sub-model's ``enabled`` flag, so the bare bool does not crash
    ``GatewaySettings()`` against the nested BaseModel (RouteIQ-778e).
    """

    def decode_complex_value(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> Any:
        if field_name in _CSV_LIST_ENV_FIELDS and isinstance(value, str):
            stripped = value.strip()
            # Defer to JSON when the operator gave a JSON list/scalar; only the
            # bare comma-separated form needs the shim.
            if not stripped.startswith(("[", '"')):
                return [r.strip() for r in stripped.split(",") if r.strip()]
        if field_name in _FLAT_BOOL_MODEL_ENV_FIELDS and isinstance(value, str):
            stripped = value.strip()
            # Defer to JSON when the operator gave a JSON object (the
            # ``{"enabled": true, ...}`` form, or a ``__``-nested expansion);
            # only the DOCUMENTED bare bool needs the shim.
            if not stripped.startswith("{"):
                lowered = stripped.lower()
                if lowered in _TRUE_BOOL_STRINGS:
                    return {"enabled": True}
                if lowered in _FALSE_BOOL_STRINGS:
                    return {"enabled": False}
        return super().decode_complex_value(field_name, field, value)


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
    llmrouter_governance_spend_tracking: bool = Field(
        True,
        validation_alias="LLMROUTER_GOVERNANCE_SPEND_TRACKING",
        description=(
            "Write post-response governance spend/RPM counters + usage-policy "
            "cost/token counters (P4 spend write path; env "
            "LLMROUTER_GOVERNANCE_SPEND_TRACKING).  Independent of OTEL/telemetry.  "
            "The ``validation_alias`` binds the bare env name (NOT the "
            "``ROUTEIQ_``-prefixed form), making this field track the historical "
            "env var so a future consumer can read it via ``get_settings()`` per "
            "ADR-0013.  NOTE: the live reader ``router_decision_callback."
            "_governance_spend_tracking_enabled()`` still reads ``os.getenv`` "
            "directly (RouteIQ-9f9f); the alias keeps the two in lock-step on the "
            "same env name + default until that consumer is migrated."
        ),
    )

    # ------------------------------------------------------------------
    # Sub-configurations (nested models)
    # ------------------------------------------------------------------

    redis: RedisSettings = Field(
        default_factory=RedisSettings,  # type: ignore[arg-type]
        description="Redis connection settings.",
    )
    postgres: PostgresSettings = Field(
        default_factory=PostgresSettings,  # type: ignore[arg-type]
        description="PostgreSQL connection settings.",
    )
    otel: OTelSettings = Field(
        default_factory=OTelSettings,  # type: ignore[arg-type]
        description="OpenTelemetry observability settings.",
    )
    oidc: OIDCSettings = Field(
        default_factory=OIDCSettings,  # type: ignore[arg-type]
        description="OIDC / SSO integration settings.",
    )
    routing: RoutingSettings = Field(
        default_factory=RoutingSettings,  # type: ignore[arg-type]
        description="ML routing strategy settings.",
    )
    security: SecuritySettings = Field(
        default_factory=SecuritySettings,  # type: ignore[arg-type]
        description="Security and access control settings.",
    )
    policy: PolicySettings = Field(
        default_factory=PolicySettings,  # type: ignore[arg-type]
        description="Policy engine settings.",
    )
    audit: AuditSettings = Field(
        default_factory=AuditSettings,  # type: ignore[arg-type]
        description="Audit logging settings.",
    )
    quota: QuotaSettings = Field(
        default_factory=QuotaSettings,  # type: ignore[arg-type]
        description="Quota enforcement settings.",
    )
    governance: GovernanceSettings = Field(
        default_factory=GovernanceSettings,  # type: ignore[arg-type]
        description="Workspace/key governance budget + rate-limit settings.",
    )
    mcp: MCPSettings = Field(
        default_factory=MCPSettings,  # type: ignore[arg-type]
        description="MCP gateway settings.",
    )
    a2a: A2ASettings = Field(
        default_factory=A2ASettings,  # type: ignore[arg-type]
        description="A2A gateway settings.",
    )
    resilience: ResilienceSettings = Field(
        default_factory=ResilienceSettings,  # type: ignore[arg-type]
        description="Backpressure and resilience settings.",
    )
    config_sync: ConfigSyncSettings = Field(
        default_factory=ConfigSyncSettings,  # type: ignore[arg-type]
        description="Remote config sync settings.",
    )
    http_client: HTTPClientSettings = Field(
        default_factory=HTTPClientSettings,  # type: ignore[arg-type]
        description="Shared HTTP client pool settings.",
    )
    ssrf: SSRFSettings = Field(
        default_factory=SSRFSettings,  # type: ignore[arg-type]
        description="SSRF protection settings.",
    )
    conversation_affinity: ConversationAffinitySettings = Field(
        default_factory=ConversationAffinitySettings,  # type: ignore[arg-type]
        description="Conversation routing affinity settings.",
    )
    plugins: PluginSettings = Field(
        default_factory=PluginSettings,  # type: ignore[arg-type]
        description="Plugin system settings.",
    )
    management: ManagementSettings = Field(
        default_factory=ManagementSettings,  # type: ignore[arg-type]
        description="Management middleware settings.",
    )
    cache: CacheSettings = Field(
        default_factory=CacheSettings,  # type: ignore[arg-type]
        description="Caching plugin settings.",
    )
    ha: HASettings = Field(
        default_factory=HASettings,  # type: ignore[arg-type]
        description="High-availability settings.",
    )
    eval_pipeline: EvalPipelineSettings = Field(
        default_factory=EvalPipelineSettings,  # type: ignore[arg-type]
        description="Evaluation feedback loop pipeline settings.",
    )
    router_r1: RouterR1Settings = Field(
        default_factory=RouterR1Settings,  # type: ignore[arg-type]
        description="Router-R1 iterative reasoning router settings.",
    )
    context_optimizer: ContextOptimizerSettings = Field(
        default_factory=ContextOptimizerSettings,  # type: ignore[arg-type]
        description="Context optimizer plugin settings.",
    )
    evaluator: EvaluatorSettings = Field(
        default_factory=EvaluatorSettings,  # type: ignore[arg-type]
        description="Evaluator plugin settings.",
    )
    content_filter: ContentFilterSettings = Field(
        default_factory=ContentFilterSettings,  # type: ignore[arg-type]
        description="Content/toxicity filter settings.",
    )
    agentic_pipeline: AgenticPipelineSettings = Field(
        default_factory=AgenticPipelineSettings,  # type: ignore[arg-type]
        description="Agentic multi-round routing pipeline settings.",
    )
    prompt_management: PromptManagementSettings = Field(
        default_factory=PromptManagementSettings,  # type: ignore[arg-type]
        description="Prompt management and versioning settings.",
    )
    cost_tracker: CostTrackerSettings = Field(
        default_factory=CostTrackerSettings,  # type: ignore[arg-type]
        description="Cost tracker plugin settings.",
    )
    skills_discovery: SkillsDiscoverySettings = Field(
        default_factory=SkillsDiscoverySettings,  # type: ignore[arg-type]
        description="Skills discovery plugin settings.",
    )
    kumaraswamy_thompson: KumaraswamyThompsonSettings = Field(
        default_factory=KumaraswamyThompsonSettings,  # type: ignore[arg-type]
        description="Kumaraswamy-Thompson bandit routing strategy settings.",
    )
    adapter_framework: AdapterFrameworkSettings = Field(
        default_factory=AdapterFrameworkSettings,  # type: ignore[arg-type]
        description="Strategy-agnostic routing-adapter framework settings.",
    )
    bedrock_discovery: BedrockDiscoverySettings = Field(
        default_factory=BedrockDiscoverySettings,  # type: ignore[arg-type]
        description="In-process Bedrock model auto-discovery settings (default off).",
    )
    bedrock_levers: BedrockRequestLeversSettings = Field(
        default_factory=BedrockRequestLeversSettings,  # type: ignore[arg-type]
        description=(
            "Per-request Bedrock Converse levers (requestMetadata / team "
            "callbacks / cachePoint) on the pre-call mutation seam (default off)."
        ),
    )
    model_alias: ModelAliasSettings = Field(
        default_factory=ModelAliasSettings,  # type: ignore[arg-type]
        description=(
            "Pre-routing model-name alias/rewrite map (default off / identity)."
        ),
    )
    cost_cascade: CostCascadeSettings = Field(
        default_factory=CostCascadeSettings,  # type: ignore[arg-type]
        description="Cost-aware speculative-cascade routing strategy settings.",
    )
    semantic_intent: SemanticIntentSettings = Field(
        default_factory=SemanticIntentSettings,  # type: ignore[arg-type]
        description="Semantic / embedding intent-router strategy settings.",
    )
    tag_routing: TagRoutingSettings = Field(
        default_factory=TagRoutingSettings,  # type: ignore[arg-type]
        description="Tag / regex / User-Agent routing strategy settings.",
    )
    latency_aware: LatencyAwareSettings = Field(
        default_factory=LatencyAwareSettings,  # type: ignore[arg-type]
        description=(
            "Latency-SLA / usage-based / least-busy routing strategy settings."
        ),
    )
    linucb: LinUCBSettings = Field(
        default_factory=LinUCBSettings,  # type: ignore[arg-type]
        description="LinUCB feature-vector contextual-bandit routing settings.",
    )
    mlops: MLOpsSettings = Field(
        default_factory=MLOpsSettings,  # type: ignore[arg-type]
        description=(
            "MLOps closed-loop settings: quality-gated promotion, drift "
            "detection, and shadow/mirror canary (all default off)."
        ),
    )
    engine_metrics: EngineMetricsSettings = Field(
        default_factory=EngineMetricsSettings,  # type: ignore[arg-type]
        description=(
            "Self-hosted-engine /metrics scrape settings (vLLM / Production "
            "Stack / AIBrix / llm-d queue-depth + KV-cache gauges; default off)."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _map_flat_iam_aliases(cls, data: Any) -> Any:
        """Map the flat IAM-auth env vars onto the nested settings fields.

        The chart/README emit the FLAT env names ``ROUTEIQ_DB_IAM_AUTH`` and
        ``ROUTEIQ_REDIS_IAM_AUTH`` (ADR-0028 / ADR-0029).  pydantic-settings
        does NOT resolve a ``validation_alias`` on a nested ``BaseModel`` field
        from a flat top-level env var (only the nested ``ROUTEIQ_POSTGRES__*`` /
        ``ROUTEIQ_REDIS__*`` form would work, which the chart does not emit), so
        we inject the flat values onto the nested ``postgres`` / ``redis`` dicts
        here, before validation.  The flat env var is the source of truth the
        chart already targets.  Both flags DEFAULT OFF (only set when the flat
        env var is present and truthy), so the static-cred path is unchanged.
        """
        import os as _os

        def _truthy(value: Any) -> bool:
            return str(value).strip().lower() in ("true", "1", "yes", "on")

        if not isinstance(data, dict):
            return data

        flat_map = (
            ("ROUTEIQ_DB_IAM_AUTH", "postgres"),
            ("ROUTEIQ_REDIS_IAM_AUTH", "redis"),
        )
        for env_name, section in flat_map:
            raw = _os.environ.get(env_name)
            if raw is None or not _truthy(raw):
                continue
            current = data.get(section)
            if current is None:
                data[section] = {"iam_auth": True}
            elif isinstance(current, dict):
                # Do not clobber an explicit nested override if one was given.
                current.setdefault("iam_auth", True)
            elif isinstance(current, BaseModel):
                # An already-constructed nested model (e.g. init kwargs); set
                # the flag only when it was left at its default.
                if getattr(current, "iam_auth", False) is False:
                    data[section] = current.model_copy(update={"iam_auth": True})

        # Map the OTel-STANDARD endpoint env var onto otel.endpoint (RouteIQ-cfe3).
        # The whole deploy surface (entrypoints, docker-compose, every config/*.yaml,
        # the docs, service_discovery.py) uses OTEL_EXPORTER_OTLP_ENDPOINT, and
        # ADR-0013 explicitly migrated it -- but the nested field only binds the
        # ROUTEIQ_OTEL__ENDPOINT form (nobody sets it), so post-migration the
        # ObservabilityManager silently exported to localhost while service
        # discovery probed the real collector (split brain). pydantic-settings
        # won't resolve a flat env var onto a nested BaseModel field, so inject it
        # here. The nested ROUTEIQ_OTEL__ENDPOINT form WINS if both are set.
        otlp_endpoint = _os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            current = data.get("otel")
            if current is None:
                data["otel"] = {"endpoint": otlp_endpoint}
            elif isinstance(current, dict):
                current.setdefault("endpoint", otlp_endpoint)
            elif isinstance(current, BaseModel):
                if getattr(current, "endpoint", None) is None:
                    data["otel"] = current.model_copy(
                        update={"endpoint": otlp_endpoint}
                    )
        return data

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

        The default ``env_settings`` source is swapped for
        :class:`_RouteIQEnvSettingsSource` so the DOCUMENTED comma-separated env
        form of CSV-list fields (e.g.
        ``ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS=us-east-1,eu-west-1``) is
        accepted instead of crashing ``GatewaySettings()`` on the JSON decode.
        """
        env_settings = _RouteIQEnvSettingsSource(settings_cls)
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
