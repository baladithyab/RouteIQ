# ADR-0018: Support Disaggregated UI Deployment

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: Coupled UI and Gateway

The RouteIQ Admin UI is currently built and embedded into the gateway Docker
image during the build process.  The Vite build happens in a `ui-builder`
stage, and the resulting static files are copied to `/app/ui/dist` where
the gateway serves them at `/ui/`.  This "embedded" mode works well for
single-container deployments, but creates friction for production scenarios:

1. **Independent scaling**: The UI is static HTML/JS/CSS.  Serving it from
   the gateway means every API-processing container also serves static files.
   In high-traffic deployments, the UI should be on a CDN (CloudFront, 
   Cloudflare, Vercel) while the gateway scales independently.

2. **Independent release cycles**: A UI-only change (cosmetic fix, copy
   update) currently requires rebuilding the entire gateway image, which
   includes installing Python dependencies, building ML model artifacts,
   and running multi-stage Docker builds (~10+ minutes).

3. **Multi-region / edge deployments**: A CDN-hosted UI can be served
   from edge locations worldwide, while the gateway runs in specific
   regions close to the LLM providers.

4. **Build-time config lock-in**: `VITE_API_BASE` is baked into the
   JavaScript bundle at build time.  The same build artifact cannot be
   pointed at different gateway URLs without rebuilding.

### Existing Support

The codebase already has partial support via `VITE_API_BASE || ''` in
`api/client.ts`, but this is a build-time variable.  There is no runtime
config mechanism, no standalone UI Dockerfile, and no CORS auto-configuration
for disaggregated deployments.

## Decision

Support two deployment modes for the Admin UI:

1. **Embedded mode** (existing, default): UI is built into the gateway
   container and served at `/ui/`.  Zero configuration required.

2. **Disaggregated mode** (new): UI is built and deployed independently
   (S3+CloudFront, Cloudflare Workers/Pages, Vercel, nginx, etc.) and
   configured to point at the gateway API URL.

### Key Design Choices

#### Runtime Configuration via `config.js`

A `config.js` file in the `public/` directory sets `window.__ROUTEIQ_CONFIG__`
before the app loads.  This file:
- Ships with sensible defaults (empty API_BASE = same-origin = embedded mode)
- Can be replaced at deploy time without rebuilding (volume mount, ConfigMap,
  S3 upload, Cloudflare KV)
- Is loaded via a `<script>` tag before the React app
- Is excluded from aggressive caching (Cache-Control: no-cache)

Priority chain: `window.__ROUTEIQ_CONFIG__` > `VITE_*` env vars > defaults.

#### Automatic CORS for External UI

`ROUTEIQ_ADMIN_UI_EXTERNAL_URL` is a new setting.  When set, the gateway
automatically adds that origin to the CORS allowed origins list.  This avoids
requiring operators to manually configure `ROUTEIQ_CORS_ORIGINS`.

#### UI Config API Endpoint

`GET /api/v1/routeiq/ui-config` returns feature flags and OIDC configuration.
The disaggregated UI can call this on startup to discover gateway capabilities
without hardcoding them in `config.js`.

#### Standalone UI Dockerfile

`ui/Dockerfile` builds the React SPA and serves it via nginx.  The nginx
config handles SPA routing (try_files fallback to index.html), aggressive
caching for hashed assets, and no-cache for `config.js`.

#### Configurable Vite Base Path

`VITE_BASE_PATH` env var allows overriding the base path at build time.
Defaults to `/ui/` for embedded mode.  Set to `/` for standalone deployments.

## Implementation

| Component | Change | File(s) |
|-----------|--------|---------|
| Runtime config | `public/config.js` + `src/config.ts` loader | `ui/public/config.js`, `ui/src/config.ts` |
| API client | Use runtime config instead of `import.meta.env` | `ui/src/api/client.ts` |
| Config script tag | Load `config.js` before app | `ui/index.html` |
| Vite config | Configurable base path, mode-aware sourcemaps | `ui/vite.config.ts` |
| CORS auto-config | Add external UI URL to allowed origins | `gateway/app.py` |
| Settings | `admin_ui_external_url` field | `settings.py` |
| UI config endpoint | `GET /api/v1/routeiq/ui-config` | `routes/admin_ui.py` |
| Standalone Dockerfile | nginx-based UI container | `ui/Dockerfile`, `ui/nginx.conf` |
| Example deployment | Docker Compose for disaggregated mode | `examples/docker/disaggregated-ui/` |

## Consequences

### Positive

- **Independent scaling**: UI can be served from a CDN.  Gateway containers
  handle only API traffic.  Reduces gateway resource usage.

- **Independent releases**: UI changes deploy in seconds (static file upload)
  without rebuilding the gateway image.

- **Same artifact, any target**: One UI build works against any gateway URL.
  Promotes dev/staging/production parity.

- **Zero breaking changes**: Embedded mode continues to work identically.
  The `config.js` file ships with defaults that preserve existing behavior.

- **Edge deployment**: CDN-hosted UI provides low-latency access worldwide.

### Negative

- **Additional deployment artifact**: Operators managing disaggregated mode
  must deploy two artifacts (gateway + UI) instead of one.  Mitigated by the
  example docker-compose and documentation.

- **CORS configuration**: Cross-origin deployments require proper CORS setup.
  Mitigated by `ROUTEIQ_ADMIN_UI_EXTERNAL_URL` auto-configuration.

- **Version skew**: UI and gateway can drift apart.  The `ui-config` endpoint
  provides version info so the UI can detect mismatches.

## Alternatives Considered

### Alternative A: Build-Time Only Configuration

Keep using `VITE_*` env vars and require a new build for each target.

- **Pros**: Simpler; no runtime config mechanism needed.
- **Cons**: Cannot reuse build artifacts; slow feedback loop for config
  changes; doesn't support CDN deployments cleanly.
- **Rejected**: Build-time config is the root cause of the coupling problem.

### Alternative B: Server-Side Rendering (SSR)

Switch to Next.js or Remix for server-side rendering.

- **Pros**: Can inject config at request time; SEO benefits (irrelevant for
  admin UI).
- **Cons**: Requires a Node.js runtime in addition to static hosting;
  significantly more complex; overkill for an admin dashboard.
- **Rejected**: Excessive complexity for the problem being solved.

### Alternative C: Environment Variable Injection at Serve Time

Use a shell script in the nginx entrypoint to `sed` env vars into the
built JavaScript bundle.

- **Pros**: No runtime config file; env vars are the single config source.
- **Cons**: Mutates the build artifact at startup; fragile; doesn't work
  with CDN deployments (no entrypoint script); cache-busting issues.
- **Rejected**: The `config.js` approach is simpler, CDN-compatible, and
  doesn't mutate build artifacts.

## References

- [ADR-0012: RouteIQ Owns Its FastAPI Application](0012-own-fastapi-app.md)
- [ADR-0009: Multi-Tier Docker Images](0009-multi-tier-docker-images.md)
- [Vite Environment Variables](https://vitejs.dev/guide/env-and-mode)
