# =============================================================================
# RouteIQ Admin UI — Standalone Dockerfile
# =============================================================================
# Builds the React SPA and serves it via nginx.
# Uses Bun for fast installs (~500ms vs ~15s with npm).
# config.js can be replaced at runtime via volume mount or ConfigMap.
#
# Build:
#   docker build -t routeiq-ui .
#
# Run:
#   docker run -p 8080:80 routeiq-ui
#
# Override runtime config:
#   docker run -p 8080:80 -v ./config.js:/usr/share/nginx/html/config.js:ro routeiq-ui
# =============================================================================

# Stage 1: Build the UI
FROM oven/bun:1-alpine AS builder
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile
COPY . .
RUN bun run build

# Stage 2: Serve via nginx
FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Config.js can be replaced at runtime via ConfigMap or volume mount
VOLUME /usr/share/nginx/html/config.js

EXPOSE 80
