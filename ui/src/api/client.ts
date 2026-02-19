const API_BASE = import.meta.env.VITE_API_BASE || ''

interface ApiOptions {
    method?: string
    body?: unknown
    headers?: Record<string, string>
}

class ApiClient {
    private baseUrl: string
    private adminKey: string | null = null

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl
        this.adminKey = localStorage.getItem('routeiq_admin_key')
    }

    setAdminKey(key: string) {
        this.adminKey = key
        localStorage.setItem('routeiq_admin_key', key)
    }

    clearAdminKey() {
        this.adminKey = null
        localStorage.removeItem('routeiq_admin_key')
    }

    private getHeaders(): Record<string, string> {
        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
        }
        if (this.adminKey) {
            headers['X-Admin-API-Key'] = this.adminKey
        }
        return headers
    }

    async request<T>(path: string, options: ApiOptions = {}): Promise<T> {
        const { method = 'GET', body, headers = {} } = options
        const response = await fetch(`${this.baseUrl}${path}`, {
            method,
            headers: { ...this.getHeaders(), ...headers },
            body: body ? JSON.stringify(body) : undefined,
        })
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }))
            throw new Error(error.detail || `HTTP ${response.status}`)
        }
        return response.json()
    }

    get<T>(path: string) {
        return this.request<T>(path)
    }

    post<T>(path: string, body?: unknown) {
        return this.request<T>(path, { method: 'POST', body })
    }

    put<T>(path: string, body?: unknown) {
        return this.request<T>(path, { method: 'PUT', body })
    }

    delete<T>(path: string) {
        return this.request<T>(path, { method: 'DELETE' })
    }
}

export const apiClient = new ApiClient(API_BASE)
export default apiClient
