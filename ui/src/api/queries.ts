import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import apiClient from './client'
import type {
    GatewayStatus,
    RoutingStats,
    ModelInfo,
    RoutingConfig,
    UpdateRoutingConfig,
    Workspace,
    CreateWorkspaceRequest,
    UsagePolicy,
    CreateUsagePolicyRequest,
    KeyGovernance,
    UpdateKeyGovernanceRequest,
    GuardrailPolicy,
    CreateGuardrailRequest,
    ServiceStatusResponse,
    ModelQualityResponse,
    EvalStats,
    PromptDefinition,
    CreatePromptRequest,
    UpdatePromptABTestRequest,
} from './types'

// --- Existing hooks ---

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

// --- Governance hooks ---

export function useWorkspaces() {
    return useQuery<Workspace[]>({
        queryKey: ['workspaces'],
        queryFn: () => apiClient.get<Workspace[]>('/api/v1/routeiq/governance/workspaces'),
        refetchInterval: 30_000,
    })
}

export function useCreateWorkspace() {
    const queryClient = useQueryClient()
    return useMutation<Workspace, Error, CreateWorkspaceRequest>({
        mutationFn: (data) => apiClient.post<Workspace>('/api/v1/routeiq/governance/workspaces', data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['workspaces'] })
        },
    })
}

export function useDeleteWorkspace() {
    const queryClient = useQueryClient()
    return useMutation<void, Error, string>({
        mutationFn: (id) => apiClient.delete<void>(`/api/v1/routeiq/governance/workspaces/${id}`),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['workspaces'] })
        },
    })
}

export function useUsagePolicies() {
    return useQuery<UsagePolicy[]>({
        queryKey: ['usage-policies'],
        queryFn: () => apiClient.get<UsagePolicy[]>('/api/v1/routeiq/governance/policies'),
        refetchInterval: 30_000,
    })
}

export function useCreateUsagePolicy() {
    const queryClient = useQueryClient()
    return useMutation<UsagePolicy, Error, CreateUsagePolicyRequest>({
        mutationFn: (data) => apiClient.post<UsagePolicy>('/api/v1/routeiq/governance/policies', data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['usage-policies'] })
        },
    })
}

export function useToggleUsagePolicy() {
    const queryClient = useQueryClient()
    return useMutation<UsagePolicy, Error, { id: string; enabled: boolean }>({
        mutationFn: ({ id, enabled }) =>
            apiClient.put<UsagePolicy>(`/api/v1/routeiq/governance/policies/${id}`, { enabled }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['usage-policies'] })
        },
    })
}

export function useKeyGovernance(keyId: string | null) {
    return useQuery<KeyGovernance>({
        queryKey: ['key-governance', keyId],
        queryFn: () => apiClient.get<KeyGovernance>(`/api/v1/routeiq/governance/keys/${keyId}`),
        enabled: !!keyId,
    })
}

export function useUpdateKeyGovernance() {
    const queryClient = useQueryClient()
    return useMutation<KeyGovernance, Error, { keyId: string; data: UpdateKeyGovernanceRequest }>({
        mutationFn: ({ keyId, data }) =>
            apiClient.put<KeyGovernance>(`/api/v1/routeiq/governance/keys/${keyId}`, data),
        onSuccess: (_data, { keyId }) => {
            queryClient.invalidateQueries({ queryKey: ['key-governance', keyId] })
        },
    })
}

// --- Guardrail hooks ---

export function useGuardrails() {
    return useQuery<GuardrailPolicy[]>({
        queryKey: ['guardrails'],
        queryFn: () => apiClient.get<GuardrailPolicy[]>('/api/v1/routeiq/governance/guardrails'),
        refetchInterval: 30_000,
    })
}

export function useCreateGuardrail() {
    const queryClient = useQueryClient()
    return useMutation<GuardrailPolicy, Error, CreateGuardrailRequest>({
        mutationFn: (data) => apiClient.post<GuardrailPolicy>('/api/v1/routeiq/governance/guardrails', data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['guardrails'] })
        },
    })
}

export function useToggleGuardrail() {
    const queryClient = useQueryClient()
    return useMutation<GuardrailPolicy, Error, { id: string; enabled: boolean }>({
        mutationFn: ({ id, enabled }) =>
            apiClient.put<GuardrailPolicy>(`/api/v1/routeiq/governance/guardrails/${id}`, { enabled }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['guardrails'] })
        },
    })
}

export function useDeleteGuardrail() {
    const queryClient = useQueryClient()
    return useMutation<void, Error, string>({
        mutationFn: (id) => apiClient.delete<void>(`/api/v1/routeiq/governance/guardrails/${id}`),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['guardrails'] })
        },
    })
}

// --- Observability hooks ---

export function useServiceStatus() {
    return useQuery<ServiceStatusResponse>({
        queryKey: ['service-status'],
        queryFn: () => apiClient.get<ServiceStatusResponse>('/config/services'),
        refetchInterval: 15_000,
    })
}

export function useModelQuality() {
    return useQuery<ModelQualityResponse>({
        queryKey: ['model-quality'],
        queryFn: () => apiClient.get<ModelQualityResponse>('/api/v1/routeiq/eval/model-quality'),
        refetchInterval: 60_000,
    })
}

export function useEvalStats() {
    return useQuery<EvalStats>({
        queryKey: ['eval-stats'],
        queryFn: () => apiClient.get<EvalStats>('/api/v1/routeiq/eval/stats'),
        refetchInterval: 15_000,
    })
}

export function useRunEvalBatch() {
    const queryClient = useQueryClient()
    return useMutation<{ message: string }, Error, void>({
        mutationFn: () => apiClient.post<{ message: string }>('/api/v1/routeiq/eval/run-batch'),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['eval-stats'] })
        },
    })
}

// --- Prompt hooks ---

export function usePrompts() {
    return useQuery<PromptDefinition[]>({
        queryKey: ['prompts'],
        queryFn: () => apiClient.get<PromptDefinition[]>('/api/v1/routeiq/prompts'),
        refetchInterval: 30_000,
    })
}

export function useCreatePrompt() {
    const queryClient = useQueryClient()
    return useMutation<PromptDefinition, Error, CreatePromptRequest>({
        mutationFn: (data) => apiClient.post<PromptDefinition>('/api/v1/routeiq/prompts', data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['prompts'] })
        },
    })
}

export function useRollbackPrompt() {
    const queryClient = useQueryClient()
    return useMutation<PromptDefinition, Error, { name: string; version: number }>({
        mutationFn: ({ name, version }) =>
            apiClient.post<PromptDefinition>(`/api/v1/routeiq/prompts/${name}/rollback`, { version }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['prompts'] })
        },
    })
}

export function useUpdatePromptABTest() {
    const queryClient = useQueryClient()
    return useMutation<PromptDefinition, Error, { name: string; data: UpdatePromptABTestRequest }>({
        mutationFn: ({ name, data }) =>
            apiClient.put<PromptDefinition>(`/api/v1/routeiq/prompts/${name}/ab-test`, data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['prompts'] })
        },
    })
}
