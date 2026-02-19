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

// Model info from GET /api/v1/routeiq/models
export interface ModelInfo {
    model_name: string
    provider: string
    model_id: string
    status: 'active' | 'degraded' | 'unavailable'
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
