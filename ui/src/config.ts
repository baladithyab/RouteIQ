/**
 * Runtime configuration loader.
 *
 * Priority:
 * 1. window.__ROUTEIQ_CONFIG__ (runtime, from public/config.js)
 * 2. VITE_* env vars (build-time)
 * 3. Defaults
 */
interface RouteIQConfig {
  API_BASE: string
  VERSION: string
  FEATURES: {
    SSO_LOGIN: boolean
    MODEL_PLAYGROUND: boolean
    COST_ANALYTICS: boolean
  }
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
}

export default config
