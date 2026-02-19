import { useQuery } from '@tanstack/react-query'
import apiClient from './client'
import type { GatewayStatus, RoutingStats, ModelInfo } from './types'

export function useGatewayStatus() {
    return useQuery<GatewayStatus>({
        queryKey: ['gateway-status'],
        queryFn: () => apiClient.get<GatewayStatus>('/api/v1/routeiq/status'),
        refetchInterval: 15_000,
    })
}

export function useRoutingStats() {
    return useQuery<RoutingStats>({
        queryKey: ['routing-stats'],
        queryFn: () => apiClient.get<RoutingStats>('/api/v1/routeiq/routing/stats'),
        refetchInterval: 10_000,
    })
}

export function useModels() {
    return useQuery<ModelInfo[]>({
        queryKey: ['models'],
        queryFn: () => apiClient.get<ModelInfo[]>('/api/v1/routeiq/models'),
        refetchInterval: 30_000,
    })
}
