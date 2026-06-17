// Gateway status response from GET /api/v1/routeiq/status
export interface GatewayStatus {
    version: string
    uptime_seconds: number
    uptime_formatted: string
    worker_count: number
    active_strategy: string | null
    routing_profile: string
    centroid_routing_enabled: boolean
    feature_flags: Record<string, boolean>
}

// Routing stats response from GET /api/v1/routeiq/routing/stats
export interface RoutingStats {
    total_decisions: number
    strategy_distribution: Record<string, number>
    profile_distribution: Record<string, number>
    centroid_decisions: number
    average_latency_ms: number
}

// Org-wide rollup from GET /api/v1/routeiq/stats/global (admin auth)
export interface GlobalStats {
    total_decisions: number
    strategy_distribution: Record<string, number>
    profile_distribution: Record<string, number>
    model_distribution: Record<string, number>
    key_distribution: Record<string, number>
    centroid_decisions: number
    average_latency_ms: number
    tracked_keys: number
    // True when the figures are the cross-replica aggregate (RouteIQ-78fd);
    // false when they fall back to the single serving worker.
    cluster_wide: boolean
}

// Caller-scoped usage from GET /api/v1/routeiq/me/stats (user auth, own-scope only)
export interface MyStats {
    key_id: string
    decision_count: number
    recent_models: string[]
    budget_remaining_usd: number | null
    budget_used_pct: number | null
    spend_usd: number | null
    max_budget_usd: number | null
}

// Self-service key (RouteIQ-3215). `key_id` is the NON-SECRET public id used
// for display / revoke; `masked` is a non-secret preview. `key` (the plaintext
// secret) is present ONLY on the create response and is never recoverable after.
export interface MyKey {
    key_id: string
    name: string | null
    owner_id: string
    created_at: string
    max_budget_usd: number | null
    allowed_models: string[]
    masked?: string | null
    key?: string | null
}

export interface MyKeyList {
    keys: MyKey[]
    count: number
}

export interface CreateMyKeyRequest {
    name?: string | null
    max_budget_usd?: number | null
    allowed_models?: string[]
}

// Model info from GET /api/v1/routeiq/models
export interface ModelInfo {
    model_name: string
    provider: string
    model_id: string
    status: 'active' | 'degraded' | 'unavailable'
}

// Request body for POST/PUT /api/v1/routeiq/models (admin auth, RouteIQ-eb2d)
export interface ModelUpsertRequest {
    model_name: string
    litellm_params: Record<string, unknown>
    model_info?: Record<string, unknown> | null
}

// Result of a model CRUD mutation
export interface ModelMutationResponse {
    model_name: string
    action: 'added' | 'updated' | 'removed'
    model_count: number
}

// Routing config from GET /api/v1/routeiq/routing/config
export interface RoutingConfig {
    active_strategy: string | null
    available_strategies: string[]
    routing_profile: string
    centroid_routing_enabled: boolean
    ab_testing: {
        enabled: boolean
        weights: Record<string, number>
        experiment_id: string | null
    }
}

// Update routing config request for POST /api/v1/routeiq/routing/config
export interface UpdateRoutingConfig {
    routing_profile?: string
    centroid_routing_enabled?: boolean
    active_strategy?: string
}

// --- Governance types ---

export interface Workspace {
    workspace_id: string
    name: string
    description: string
    allowed_models: string[]
    budget_cap: number | null
    rate_limit_rpm: number | null
    rate_limit_tpm: number | null
    guardrail_ids: string[]
    created_at: string
    updated_at: string
}

export interface CreateWorkspaceRequest {
    name: string
    description?: string
    allowed_models?: string[]
    budget_cap?: number | null
    rate_limit_rpm?: number | null
    rate_limit_tpm?: number | null
    guardrail_ids?: string[]
}

export interface UsagePolicy {
    policy_id: string
    name: string
    type: 'cost' | 'tokens' | 'requests'
    limit_value: number
    period: 'minute' | 'hour' | 'day' | 'month'
    conditions: Record<string, string>
    action: 'deny' | 'log' | 'alert'
    enabled: boolean
    current_usage: number
    created_at: string
}

export interface CreateUsagePolicyRequest {
    name: string
    type: 'cost' | 'tokens' | 'requests'
    limit_value: number
    period: 'minute' | 'hour' | 'day' | 'month'
    conditions?: Record<string, string>
    action: 'deny' | 'log' | 'alert'
    enabled?: boolean
}

export interface KeyGovernance {
    key_id: string
    workspace_id: string | null
    scopes: string[]
    budget_limit: number | null
    rate_limit_rpm: number | null
    allowed_models: string[]
    created_at: string
    updated_at: string
}

export interface UpdateKeyGovernanceRequest {
    scopes?: string[]
    budget_limit?: number | null
    rate_limit_rpm?: number | null
    allowed_models?: string[]
}

// --- Guardrail types ---

export interface GuardrailPolicy {
    guardrail_id: string
    name: string
    description: string
    check_type: 'regex_deny' | 'regex_require' | 'pii_detection' | 'toxicity' | 'prompt_injection' | 'custom'
    phase: 'input' | 'output' | 'both'
    action: 'deny' | 'redact' | 'log' | 'alert'
    enabled: boolean
    parameters: Record<string, unknown>
    created_at: string
    updated_at: string
}

export interface CreateGuardrailRequest {
    name: string
    description?: string
    check_type: GuardrailPolicy['check_type']
    phase: GuardrailPolicy['phase']
    action: GuardrailPolicy['action']
    enabled?: boolean
    parameters?: Record<string, unknown>
}

// --- Observability types ---

export interface ServiceStatus {
    name: string
    available: boolean
    latency_ms: number | null
    version: string | null
    error: string | null
}

export interface ServiceStatusResponse {
    services: ServiceStatus[]
}

export interface ModelQualityEntry {
    model: string
    score: number
    sample_count: number
}

export interface ModelQualityResponse {
    models: Record<string, number>
    ranking: [string, number][]
    details: ModelQualityEntry[]
}

export interface EvalStats {
    pending_samples: number
    evaluated_samples: number
    running: boolean
    last_run_at: string | null
    average_score: number | null
}

export interface EvalSample {
    sample_id: string
    model: string
    score: number | null
    status: 'pending' | 'evaluated' | 'failed'
    created_at: string
}

// --- Prompt types ---

export interface PromptVersion {
    version: number
    template: string
    created_at: string
    author: string | null
}

export interface PromptABTest {
    enabled: boolean
    weights: Record<number, number>
}

export interface PromptDefinition {
    name: string
    description: string
    active_version: number
    versions: PromptVersion[]
    tags: string[]
    ab_test: PromptABTest | null
    created_at: string
    updated_at: string
}

export interface CreatePromptRequest {
    name: string
    description?: string
    template: string
    tags?: string[]
}

export interface UpdatePromptABTestRequest {
    enabled: boolean
    weights: Record<number, number>
}
