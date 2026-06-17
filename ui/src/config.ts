/**
 * Runtime configuration loader.
 *
 * Priority:
 * 1. window.__ROUTEIQ_CONFIG__ (runtime, from public/config.js)
 * 2. VITE_* env vars (build-time)
 * 3. Defaults
 */
// Embedded dashboard config (CloudWatch / Managed Grafana) — RouteIQ-0c8e.
// `embed_url` (frameable) renders an inline iframe; `url` renders a deep-link
// button. Operator setup lives in docs/operations/dashboard-embedding.md.
export interface DashboardEmbedConfig {
  provider: 'cloudwatch' | 'grafana' | string
  url?: string
  embed_url?: string
}

interface RouteIQConfig {
  API_BASE: string
  VERSION: string
  FEATURES: {
    SSO_LOGIN: boolean
    MODEL_PLAYGROUND: boolean
    COST_ANALYTICS: boolean
  }
  // Optional embedded AWS dashboard (CloudWatch GenAI / Managed Grafana).
  DASHBOARD_EMBED?: DashboardEmbedConfig | null
  // X-Ray console base URL for per-request trace deep-links (RouteIQ-9d2d).
  XRAY_CONSOLE_BASE?: string | null
}

declare global {
  interface Window {
    __ROUTEIQ_CONFIG__?: Partial<RouteIQConfig>
  }
}

const runtimeConfig = window.__ROUTEIQ_CONFIG__ || {}

export const config: RouteIQConfig = {
  API_BASE: runtimeConfig.API_BASE || import.meta.env.VITE_API_BASE || '',
  VERSION: runtimeConfig.VERSION || import.meta.env.VITE_VERSION || '0.0.0-dev',
  FEATURES: {
    SSO_LOGIN: runtimeConfig.FEATURES?.SSO_LOGIN ?? false,
    MODEL_PLAYGROUND: runtimeConfig.FEATURES?.MODEL_PLAYGROUND ?? false,
    COST_ANALYTICS: runtimeConfig.FEATURES?.COST_ANALYTICS ?? false,
  },
  DASHBOARD_EMBED: runtimeConfig.DASHBOARD_EMBED ?? null,
  XRAY_CONSOLE_BASE: runtimeConfig.XRAY_CONSOLE_BASE ?? null,
}

export default config
