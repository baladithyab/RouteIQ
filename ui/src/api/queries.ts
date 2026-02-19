import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import apiClient from './client'
import type { GatewayStatus, RoutingStats, ModelInfo, RoutingConfig, UpdateRoutingConfig } from './types'

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

export function useRoutingConfig() {
    return useQuery<RoutingConfig>({
        queryKey: ['routing-config'],
        queryFn: () => apiClient.get<RoutingConfig>('/api/v1/routeiq/routing/config'),
        refetchInterval: 15_000,
    })
}

export function useUpdateRoutingConfig() {
    const queryClient = useQueryClient()
    return useMutation<RoutingConfig, Error, UpdateRoutingConfig>({
        mutationFn: (config) => apiClient.post<RoutingConfig>('/api/v1/routeiq/routing/config', config),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['routing-config'] })
            queryClient.invalidateQueries({ queryKey: ['gateway-status'] })
        },
    })
}
