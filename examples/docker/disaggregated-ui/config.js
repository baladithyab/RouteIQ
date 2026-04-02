// Runtime configuration for disaggregated UI deployment.
// This file is volume-mounted into the nginx container.
window.__ROUTEIQ_CONFIG__ = {
  API_BASE: "http://localhost:4000",
  VERSION: "1.0.0",
  FEATURES: {
    SSO_LOGIN: false,
    MODEL_PLAYGROUND: false,
    COST_ANALYTICS: false,
  },
};
