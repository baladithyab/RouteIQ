// RouteIQ UI Runtime Configuration
// Override these values when deploying the UI separately from the gateway.
// This file is loaded before the app and can be replaced at deploy time
// (e.g., via ConfigMap, S3 upload, or Cloudflare Workers KV).
window.__ROUTEIQ_CONFIG__ = {
  // API base URL — empty string means same-origin (embedded mode)
  // For disaggregated mode, set to the gateway URL:
  // e.g., "https://gateway.example.com" or "https://api.routeiq.internal"
  API_BASE: "",

  // UI version (injected at build time)
  VERSION: "__VERSION__",

  // Feature flags
  FEATURES: {
    SSO_LOGIN: false,       // Show SSO login button
    MODEL_PLAYGROUND: false, // Show model playground tab
    COST_ANALYTICS: false,   // Show cost analytics
  },
};
