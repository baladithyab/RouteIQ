import config from '../config'

const API_BASE = config.API_BASE

const ADMIN_KEY_STORAGE = 'routeiq_admin_key'
const USER_TOKEN_STORAGE = 'routeiq_user_token'

interface ApiOptions {
    method?: string
    body?: unknown
    headers?: Record<string, string>
    /**
     * Which credential to attach:
     *  - 'admin' (default): the admin API key via X-Admin-API-Key (control plane)
     *  - 'user': the end-user's OIDC-derived token via Authorization: Bearer
     *    (data-plane, user-tier endpoints like /me/stats — RouteIQ-f98a)
     */
    auth?: 'admin' | 'user'
}

class ApiClient {
    private baseUrl: string
    private adminKey: string | null = null
    private userToken: string | null = null

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl
        this.adminKey = localStorage.getItem(ADMIN_KEY_STORAGE)
        this.userToken = localStorage.getItem(USER_TOKEN_STORAGE)
    }

    setAdminKey(key: string) {
        this.adminKey = key
        localStorage.setItem(ADMIN_KEY_STORAGE, key)
    }

    clearAdminKey() {
        this.adminKey = null
        localStorage.removeItem(ADMIN_KEY_STORAGE)
    }

    /**
     * Store the end-user's token (obtained via the OIDC login flow / token
     * exchange). Used as a Bearer credential for user-tier endpoints so the
     * UserStats view reflects the USER's identity, not the admin key.
     */
    setUserToken(token: string) {
        this.userToken = token
        localStorage.setItem(USER_TOKEN_STORAGE, token)
    }

    clearUserToken() {
        this.userToken = null
        localStorage.removeItem(USER_TOKEN_STORAGE)
    }

    getUserToken(): string | null {
        return this.userToken
    }

    isUserAuthenticated(): boolean {
        return !!this.userToken
    }

    private getHeaders(authMode: 'admin' | 'user'): Record<string, string> {
        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
        }
        if (authMode === 'user') {
            // Data-plane: present the end-user token as a Bearer credential so
            // the gateway's user_api_key_auth resolves the CALLER's identity.
            if (this.userToken) {
                headers['Authorization'] = `Bearer ${this.userToken}`
            }
        } else if (this.adminKey) {
            headers['X-Admin-API-Key'] = this.adminKey
        }
        return headers
    }

    async request<T>(path: string, options: ApiOptions = {}): Promise<T> {
        const { method = 'GET', body, headers = {}, auth = 'admin' } = options
        const response = await fetch(`${this.baseUrl}${path}`, {
            method,
            headers: { ...this.getHeaders(auth), ...headers },
            body: body ? JSON.stringify(body) : undefined,
        })
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }))
            throw new Error(error.detail || `HTTP ${response.status}`)
        }
        return response.json()
    }

    get<T>(path: string, auth: 'admin' | 'user' = 'admin') {
        return this.request<T>(path, { auth })
    }

    post<T>(path: string, body?: unknown, auth: 'admin' | 'user' = 'admin') {
        return this.request<T>(path, { method: 'POST', body, auth })
    }

    put<T>(path: string, body?: unknown, auth: 'admin' | 'user' = 'admin') {
        return this.request<T>(path, { method: 'PUT', body, auth })
    }

    delete<T>(path: string, auth: 'admin' | 'user' = 'admin') {
        return this.request<T>(path, { method: 'DELETE', auth })
    }
}

export const apiClient = new ApiClient(API_BASE)
export default apiClient
