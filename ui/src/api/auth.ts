import config from '../config'
import apiClient from './client'

/**
 * UI runtime config served by the gateway at /api/v1/routeiq/ui-config.
 * Advertises feature flags + the OIDC login URL so the SPA can drive a real
 * end-user login flow (RouteIQ-f98a) instead of reusing the admin key.
 */
export interface UiConfig {
    version: string
    features: {
        sso_login: boolean
        model_playground: boolean
        cost_analytics: boolean
    }
    oidc: {
        enabled: boolean
        login_url: string | null
    }
}

const UI_CONFIG_PATH = '/api/v1/routeiq/ui-config'

/**
 * Fetch the gateway-advertised UI config. Unauthenticated (the endpoint is
 * public so a disaggregated UI can discover gateway capabilities at runtime).
 */
export async function fetchUiConfig(): Promise<UiConfig> {
    const response = await fetch(`${config.API_BASE}${UI_CONFIG_PATH}`)
    if (!response.ok) {
        throw new Error(`Failed to load UI config: HTTP ${response.status}`)
    }
    return response.json()
}

/**
 * Build the OIDC login redirect URL. The gateway advertises the login path
 * (e.g. "/sso/login") via ui-config.oidc.login_url; we attach a redirect_uri
 * back to the SPA so the browser returns here after the IdP round-trip.
 */
export function buildLoginUrl(loginUrl: string): string {
    const base = config.API_BASE || ''
    const redirectUri = `${window.location.origin}${window.location.pathname}`
    const sep = loginUrl.includes('?') ? '&' : '?'
    return `${base}${loginUrl}${sep}redirect_uri=${encodeURIComponent(redirectUri)}`
}

/**
 * Begin the OIDC login flow by navigating the browser to the IdP.
 */
export function beginLogin(loginUrl: string): void {
    window.location.assign(buildLoginUrl(loginUrl))
}

/**
 * Capture an end-user token returned to the SPA after the OIDC round-trip.
 *
 * Two delivery channels are supported (both are mockable in tests):
 *  1. A "user_token" query/hash param appended by the gateway callback redirect.
 *  2. A token the caller supplies directly (e.g. test harness / token paste).
 *
 * Returns the captured token (and persists it on the client) or null when none
 * is present. Strips the token param from the URL so it is not left in history.
 */
export function captureTokenFromRedirect(search?: string, hash?: string): string | null {
    const fromQuery = new URLSearchParams(search ?? window.location.search).get('user_token')
    const rawHash = (hash ?? window.location.hash).replace(/^#/, '')
    const fromHash = new URLSearchParams(rawHash).get('user_token')
    const token = fromQuery || fromHash
    if (!token) {
        return null
    }
    apiClient.setUserToken(token)
    return token
}

/** Store a user token directly (token exchange / explicit entry). */
export function setUserToken(token: string): void {
    apiClient.setUserToken(token)
}

/** Whether an end-user token is currently held. */
export function isUserAuthenticated(): boolean {
    return apiClient.isUserAuthenticated()
}

/** Clear the held end-user token (logout). */
export function logoutUser(): void {
    apiClient.clearUserToken()
}
