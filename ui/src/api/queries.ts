import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import apiClient from './client'
import type {
    GatewayStatus,
    RoutingStats,
    GlobalStats,
    MyStats,
    MyKey,
    MyKeyList,
    CreateMyKeyRequest,
    ModelInfo,
    ModelUpsertRequest,
    ModelMutationResponse,
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
    McpServer,
    McpServerList,
    A2aAgent,
    A2aAgentList,
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

export function useGlobalStats() {
    return useQuery<GlobalStats>({
        queryKey: ['global-stats'],
        queryFn: () => apiClient.get<GlobalStats>('/api/v1/routeiq/stats/global'),
        refetchInterval: 10_000,
    })
}

// /me/stats is a USER-tier endpoint: it MUST be called with the end-user's
// token (RouteIQ-f98a), not the admin key, so the gateway resolves the caller's
// OWN identity. Gated on a held user token so it does not fire (and 401) before
// the user logs in.
export function useUserStats(enabled: boolean) {
    return useQuery<MyStats>({
        queryKey: ['user-stats'],
        queryFn: () => apiClient.get<MyStats>('/api/v1/routeiq/me/stats', 'user'),
        refetchInterval: 15_000,
        enabled,
    })
}

// --- Self-service key hooks (USER-tier, scope-isolated, RouteIQ-3215) ---
// These hit the user tier with the end-user token (not the admin key) so the
// gateway scopes every operation to the caller's OWN identity.

export function useMyKeys(enabled: boolean) {
    return useQuery<MyKeyList>({
        queryKey: ['my-keys'],
        queryFn: () => apiClient.get<MyKeyList>('/api/v1/routeiq/me/keys', 'user'),
        enabled,
    })
}

export function useCreateMyKey() {
    const queryClient = useQueryClient()
    return useMutation<MyKey, Error, CreateMyKeyRequest>({
        mutationFn: (data) =>
            apiClient.post<MyKey>('/api/v1/routeiq/me/keys', data, 'user'),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['my-keys'] })
        },
    })
}

export function useRevokeMyKey() {
    const queryClient = useQueryClient()
    return useMutation<{ deleted: boolean; key_id: string }, Error, string>({
        mutationFn: (keyId) =>
            apiClient.delete<{ deleted: boolean; key_id: string }>(
                `/api/v1/routeiq/me/keys/${encodeURIComponent(keyId)}`,
                'user',
            ),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['my-keys'] })
        },
    })
}

export function useModels() {
    return useQuery<ModelInfo[]>({
        queryKey: ['models'],
        queryFn: () => apiClient.get<ModelInfo[]>('/api/v1/routeiq/models'),
        refetchInterval: 30_000,
    })
}

// --- Model CRUD hooks (admin auth, RouteIQ-eb2d) ---

export function useAddModel() {
    const queryClient = useQueryClient()
    return useMutation<ModelMutationResponse, Error, ModelUpsertRequest>({
        mutationFn: (data) =>
            apiClient.post<ModelMutationResponse>('/api/v1/routeiq/models', data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['models'] })
        },
    })
}

export function useUpdateModel() {
    const queryClient = useQueryClient()
    return useMutation<ModelMutationResponse, Error, { modelName: string; data: ModelUpsertRequest }>({
        mutationFn: ({ modelName, data }) =>
            apiClient.put<ModelMutationResponse>(
                `/api/v1/routeiq/models/${encodeURIComponent(modelName)}`,
                data,
            ),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['models'] })
        },
    })
}

export function useDeleteModel() {
    const queryClient = useQueryClient()
    return useMutation<ModelMutationResponse, Error, string>({
        mutationFn: (modelName) =>
            apiClient.delete<ModelMutationResponse>(
                `/api/v1/routeiq/models/${encodeURIComponent(modelName)}`,
            ),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['models'] })
        },
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
        queryFn: () => apiClient.get('/api/v1/routeiq/governance/workspaces').then((r: any) => r.workspaces || r),
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
        queryFn: () => apiClient.get('/api/v1/routeiq/governance/policies').then((r: any) => r.policies || r),
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
        queryFn: () => apiClient.get('/api/v1/routeiq/governance/guardrails').then((r: any) => r.guardrails || r),
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
        queryFn: () => apiClient.get('/config/services').then((r: any) => r.services || r),
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
        queryFn: () => apiClient.get('/api/v1/routeiq/prompts').then((r: any) => r.prompts || r),
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
            apiClient.post<PromptDefinition>(`/api/v1/routeiq/prompts/${name}/ab-test`, data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['prompts'] })
        },
    })
}

// --- AI-Hub catalog hooks (RouteIQ-06cf) ---
// These GET endpoints 404 when the MCP / A2A gateways are disabled, so we do not
// retry (a 404 is a stable "feature off", not a transient failure).

export function useMcpServers() {
    return useQuery<McpServer[]>({
        queryKey: ['mcp-servers'],
        queryFn: () =>
            apiClient
                .get<McpServerList>('/llmrouter/mcp/servers')
                .then((r) => r.servers ?? []),
        refetchInterval: 30_000,
        retry: false,
    })
}

export function useA2aAgents() {
    return useQuery<A2aAgent[]>({
        queryKey: ['a2a-agents'],
        queryFn: () =>
            apiClient.get<A2aAgentList>('/a2a/agents').then((r) => r.agents ?? []),
        refetchInterval: 30_000,
        retry: false,
    })
}
