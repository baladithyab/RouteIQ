# syntax=docker/dockerfile:1.4
# =============================================================================
# RouteIQ Gateway Production Dockerfile
# =============================================================================
# Multi-stage build for minimal, secure production images
# Uses uv for fast, reproducible Python package management with lockfile
# Supports: linux/amd64, linux/arm64
#
# Build optimization:
# - Lockfile-driven installs via `uv sync --frozen --no-editable` (RouteIQ-ba11)
#   — the resolved env (including the torch-CPU pin, RouteIQ-07ef) comes
#   straight from uv.lock; `--no-editable` makes the .venv self-contained and
#   copyable across stages. The old `uv pip install -e` bypassed uv.lock and
#   re-resolved against indexes, so the torch-CPU pin was non-deterministic.
# - Native compilation stays in the BUILDER; the runtime copies only /app/.venv
#   and carries no compilers (RouteIQ-30e9 / authority move #5).
# - BuildKit cache mounts for uv to avoid baking caches into layers
# - All cleanup in same RUN statements to minimize layer size
# - Dependencies installed before app source for better layer reuse
#
# Security features:
# - Pinned base images via sha256 digest for supply-chain integrity
# - Lockfile verification (--frozen flag fails on mismatch)
# - Non-root user execution
# - Chainguard Wolfi (glibc, minimal-CVE) runtime base (RouteIQ-30e9)
# - Read-only filesystem compatible
# - Health checks for orchestrators
#
# Build args:
#   PYTHON_VERSION    - Python version (default: 3.14)
#   UV_VERSION        - uv package manager version (default: 0.9)
#   LITELLM_VERSION   - LiteLLM version (default: 1.89.0)
#   LLMROUTER_COMMIT  - LLMRouter git commit hash
#   ROUTEIQ_EXTRAS    - Comma-separated pip extras (default: "prod", torch-free).
#                       Reshaped into repeated `uv sync --extra` flags at build
#                       time, so both single ("prod") and comma ("otel,cloud")
#                       forms work. Use "prod-ml" for the KNN/ML (torch-cpu) tier.
#   BUILD_UI          - Build and include admin UI (default: "true")
#   INSTALL_LLMROUTER - Build and install LLMRouter wheel (default: "true")
#
# Build variants:
#   Full / default (torch-free prod — centroid routing, ADR-0010):
#     docker build -f docker/Dockerfile -t routeiq:prod .
#   ML tier (adds KNN/ML strategies — torch-cpu, sentence-transformers, sklearn):
#     docker build -f docker/Dockerfile \
#       --build-arg ROUTEIQ_EXTRAS="prod-ml" -t routeiq:prod-ml .
#   Slim (no KNN/PyTorch/Prisma/UI):
#     docker build -f docker/Dockerfile \
#       --build-arg ROUTEIQ_EXTRAS="otel,cloud,a2a,hotreload,oidc" \
#       --build-arg BUILD_UI=false \
#       --build-arg INSTALL_LLMROUTER=false .
#
# Reproducible Builds:
#   This Dockerfile uses uv.lock for deterministic dependency resolution via
#   `uv sync --frozen` (fails on lock drift). Run `uv lock --check` locally to
#   verify the lockfile is up-to-date before building.
# =============================================================================

ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.9
ARG LITELLM_VERSION=1.89.0
# LLMRouter pinned to specific commit for reproducibility
ARG LLMROUTER_COMMIT=7890cd9d36951a3c73fec83619321f3704a7aaa8

# ==============================================================================
# Configurable build variants
# ==============================================================================
# ROUTEIQ_EXTRAS: pip extras to install. "prod" is torch-free (centroid routing).
#   ML tier:    "prod-ml" (adds knn -> torch-cpu + sentence-transformers + sklearn)
#   Slim:       "otel,cloud,a2a,hotreload,oidc" (no knn, no db/Prisma)
ARG ROUTEIQ_EXTRAS="prod"
# BUILD_UI: set to "false" to skip the admin UI React build stage
ARG BUILD_UI="true"
# INSTALL_LLMROUTER: set to "false" to skip LLMRouter wheel (not needed without KNN)
ARG INSTALL_LLMROUTER="true"

# ==============================================================================
# Base image digests for reproducibility
# To update the builder/ui digests:
#   docker pull <image> && docker inspect --format='{{index .RepoDigests 0}}' <image>
# To update the runtime (Wolfi) digest:
#   docker buildx imagetools inspect cgr.dev/chainguard/wolfi-base:latest
# ==============================================================================
ARG BUILDER_DIGEST=sha256:d414e96d3cc71a71164ad83402f91c4531a4e451505e2d481579b69dfe0a3986
# Chainguard Wolfi (glibc) multi-arch index digest (RouteIQ-30e9).
ARG RUNTIME_DIGEST=sha256:34977aa13765da89f60fee8fe5230e2bb1c55192df08e383c58221ee0d1277fb

# ==============================================================================
# Stage 1: Builder - Clone LLMRouter, build its wheel, and sync the .venv
# ==============================================================================
# All native compilation (uvloop, manylinux wheels) happens HERE on the Astral
# uv/Debian toolchain image; the runtime copies only the resulting /app/.venv.
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm@${BUILDER_DIGEST} AS builder

ARG LLMROUTER_COMMIT
ARG LITELLM_VERSION
ARG INSTALL_LLMROUTER
ARG ROUTEIQ_EXTRAS

# Install git for cloning LLMRouter (apt cleanup in same layer)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# --- 1a. Build the LLMRouter wheel (not on PyPI / not in uv.lock) -------------
# Clone at the pinned commit and build a wheel into /wheels. All git operations
# and cleanup in a single RUN to keep .git out of the layer. When
# INSTALL_LLMROUTER=false, create an empty /wheels dir and skip the build.
WORKDIR /build
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /wheels && \
    if [ "${INSTALL_LLMROUTER}" = "true" ]; then \
        git clone --depth 1 https://github.com/ulab-uiuc/LLMRouter.git /tmp/LLMRouter \
        && cd /tmp/LLMRouter \
        && git fetch --depth 1 origin ${LLMROUTER_COMMIT} \
        && git checkout ${LLMROUTER_COMMIT} \
        && uv build --wheel --out-dir /wheels \
        && rm -rf /tmp/LLMRouter; \
    else \
        echo "Skipping LLMRouter build (INSTALL_LLMROUTER=false)"; \
    fi

# --- 1b. Sync the project .venv from the lockfile (frozen, non-editable) ------
# Copy the lockfile + project metadata + source first (deps before nothing-else,
# this is the whole install). `uv sync --frozen` installs EXACTLY what uv.lock
# pins (fails on drift) so the torch-CPU pin (RouteIQ-07ef) is deterministic in
# the image; `--no-editable` installs the project into site-packages (not a .pth
# back to /app/src) so the .venv is self-contained and copyable to the runtime.
# ROUTEIQ_EXTRAS (comma-separated) is reshaped into repeated `--extra` flags.
WORKDIR /app
COPY pyproject.toml uv.lock /app/
COPY src/ /app/src/
COPY README.md /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    set -e && \
    EXTRA_FLAGS="" && \
    OLD_IFS="$IFS"; IFS=','; \
    for e in ${ROUTEIQ_EXTRAS}; do \
        [ -n "$e" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --extra ${e}"; \
    done; \
    IFS="$OLD_IFS"; \
    echo "uv sync extras: ${EXTRA_FLAGS}" && \
    # UV_EXTRA_INDEX_URL is a defense-in-depth fallback; the deterministic
    # torch-CPU enforcement is the [tool.uv.sources] pin baked into uv.lock.
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    uv sync --frozen --no-editable --no-dev ${EXTRA_FLAGS} && \
    # Install the LLMRouter wheel into the SAME .venv (it is not in uv.lock, so
    # it cannot ride the frozen sync). It carries no torch dependency itself.
    if [ "${INSTALL_LLMROUTER}" = "true" ] && ls /wheels/*.whl 1>/dev/null 2>&1; then \
        UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
        uv pip install --python /app/.venv/bin/python /wheels/*.whl; \
    fi && \
    rm -rf /wheels

# ==============================================================================
# Stage 2: UI Builder (optional) - Build admin UI React SPA
# ==============================================================================
# When BUILD_UI=false, this stage still runs but produces an empty /ui/dist.
# Docker BuildKit may skip unused stages, but we keep this safe either way.
# Uses Bun for faster installs (~500ms vs ~15s with npm).
FROM oven/bun:1-alpine AS ui-builder
ARG BUILD_UI
WORKDIR /ui
# Copy package files + lockfile first for layer caching
COPY ui/package.json ui/bun.lock ./
RUN if [ "${BUILD_UI}" = "true" ]; then bun install --frozen-lockfile; fi
# Copy source and build
COPY ui/ ./
RUN if [ "${BUILD_UI}" = "true" ]; then \
        bun run build; \
    else \
        mkdir -p /ui/dist && echo '<!DOCTYPE html><html><body>UI not included in slim build</body></html>' > /ui/dist/index.html; \
    fi

# ==============================================================================
# Stage 3: Runtime - Chainguard Wolfi (glibc) production image (RouteIQ-30e9)
# ==============================================================================
# Wolfi over Debian: minimal-CVE, glibc (so torch/scipy/sklearn manylinux wheels
# install as binaries — musl/Alpine would force source builds). Wolfi over
# distroless: the entrypoint is #!/bin/bash with shell-only constructs (seq,
# bit-shift arithmetic, [ ] tests, background &, command -v, heredocs) plus the
# prisma CLI, the curl healthcheck, and tini — all of which need a shell.
# Shell-less distroless is a HARD build-breaker; Wolfi ships a shell + apk.
FROM cgr.dev/chainguard/wolfi-base:latest@${RUNTIME_DIGEST} AS runtime

ARG LITELLM_VERSION
ARG LLMROUTER_COMMIT
ARG BUILD_DATE
ARG VCS_REF
ARG ROUTEIQ_EXTRAS
ARG BUILD_UI
ARG INSTALL_LLMROUTER

# OCI Image Labels (https://github.com/opencontainers/image-spec)
LABEL org.opencontainers.image.source="https://github.com/baladithyab/litellm-llm-router"
LABEL org.opencontainers.image.description="RouteIQ: Intelligent LLM Gateway with ML-Powered Routing"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="1.0.0-rc1"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="RouteIQ"
LABEL org.opencontainers.image.title="RouteIQ Gateway"
LABEL org.opencontainers.image.base.name="cgr.dev/chainguard/wolfi-base:latest"
LABEL litellm.version="${LITELLM_VERSION}"
LABEL llmrouter.commit="${LLMROUTER_COMMIT}"
LABEL build.reproducible="true"
LABEL build.lockfile-driven="uv.lock"

# Install minimal runtime dependencies via apk (Wolfi), all in a single layer.
# NO build-essential / compilers — native compilation lives in the builder and
# only the copied .venv is used at runtime (authority move #5).
#   bash            - entrypoint shebang (#!/bin/bash) + shell-only constructs
#   curl            - HEALTHCHECK
#   ca-certificates - TLS to providers / S3 / OIDC
#   tini            - init for signal handling (lands at /usr/bin/tini)
#   libgcc/libstdc++- glibc C++ runtime for native wheels (torch/scipy on ML tier)
#   python-3.14     - base interpreter the copied .venv resolves against
#   shadow          - groupadd/useradd (used below)
# nodejs-22 is added for prisma's query-engine generation on db/prod tiers only.
# NOTE: Wolfi/Chainguard ships VERSIONED node packages (nodejs-16 .. nodejs-26);
# a bare `nodejs` is NOT a valid apk package and fails the build. Pin the LTS
# (nodejs-22). Confirmed via a live `docker build` on this worktree.
RUN apk add --no-cache \
        bash \
        curl \
        ca-certificates \
        tini \
        libgcc \
        libstdc++ \
        python-3.14 \
        shadow \
    && if echo "${ROUTEIQ_EXTRAS}" | grep -qE '(prod|db)'; then \
        apk add --no-cache nodejs-22; \
    fi \
    # The builder's .venv interpreter symlinks (python / python3 / python3.14)
    # and pyvenv.cfg `home` point into /usr/local/bin (Astral image layout).
    # Wolfi installs python at /usr/bin/python3.14, so provide compat symlinks
    # for ALL three interpreter names at /usr/local/bin -> the Wolfi interpreter
    # (same 3.14 minor => ABI compatible) to make the copied .venv relocatable.
    && mkdir -p /usr/local/bin \
    && ln -sf /usr/bin/python3.14 /usr/local/bin/python3.14 \
    && ln -sf /usr/bin/python3.14 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.14 /usr/local/bin/python \
    && rm -rf /var/cache/apk/*

# Security: Create non-root user with specific UID/GID for consistency.
# Wolfi provides groupadd/useradd via the `shadow` apk package, so the original
# Debian invocation is kept verbatim (lower diff).
RUN groupadd -r -g 1000 litellm && \
    useradd -r -u 1000 -g litellm -d /app -s /sbin/nologin litellm

WORKDIR /app

# Make the copied .venv the default interpreter resolution path.
ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# Copy the fully-resolved, self-contained .venv from the builder. No compilers,
# no uv, no re-resolution happen in the runtime — the env is exactly uv.lock.
COPY --from=builder /app/.venv /app/.venv

# Pre-generate Prisma client and fix permissions on LiteLLM directories.
# Done as root to avoid runtime permission errors. Repathed to the copied .venv
# site-packages (self-contained, python3.14-minor-aligned with Wolfi 3.14.6).
# Skipped when db extra is not installed (slim builds without Prisma).
RUN if echo "${ROUTEIQ_EXTRAS}" | grep -qE '(prod|db)'; then \
        SITE_PACKAGES=$(/app/.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
        SCHEMA_PATH=$(/app/.venv/bin/python -c "import litellm, os; print(os.path.join(os.path.dirname(litellm.__file__), 'proxy', 'schema.prisma'))") && \
        if [ -f "$SCHEMA_PATH" ]; then \
            echo "Generating Prisma client from $SCHEMA_PATH..." && \
            /app/.venv/bin/python -m prisma generate --schema="$SCHEMA_PATH" || echo "Warning: prisma generate failed"; \
        fi && \
        # Fix permissions on LiteLLM directories that need write access at runtime.
        # Least-privilege: litellm user owns directories with 755 (rwxr-xr-x).
        chown -R litellm:litellm "$SITE_PACKAGES/prisma/" 2>/dev/null || true && \
        chmod -R 755 "$SITE_PACKAGES/prisma/" 2>/dev/null || true && \
        chown -R litellm:litellm "$SITE_PACKAGES/litellm/proxy/_experimental/out/" 2>/dev/null || true && \
        chmod -R 755 "$SITE_PACKAGES/litellm/proxy/_experimental/out/" 2>/dev/null || true; \
    else \
        echo "Skipping Prisma generation (db extra not installed)"; \
    fi

# Copy application files (after dependencies for better layer caching).
# NOTE: the project is installed non-editable into the .venv; this source copy
# is what PYTHONPATH=/app resolves for in-tree runtime assets / overrides.
COPY --chown=litellm:litellm src/litellm_llmrouter /app/litellm_llmrouter
COPY --chown=litellm:litellm docker/entrypoint.sh /app/entrypoint.sh
COPY --chown=litellm:litellm config/ /app/config/

# Copy admin UI dist (built in ui-builder stage)
# In slim builds (BUILD_UI=false), this copies a placeholder index.html
COPY --from=ui-builder --chown=litellm:litellm /ui/dist /app/ui/dist

# Create directories with proper permissions and cleanup
RUN mkdir -p /app/models /app/data /app/custom_routers && \
    chown -R litellm:litellm /app && \
    chmod +x /app/entrypoint.sh && \
    # Security: Remove unnecessary files
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -delete

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONFAULTHANDLER=1 \
    # Application config
    LITELLM_CONFIG_PATH=/app/config/config.yaml \
    LLMROUTER_MODELS_PATH=/app/models \
    LLMROUTER_CUSTOM_ROUTERS_PATH=/app/custom_routers \
    # OTEL defaults (override at runtime)
    OTEL_SERVICE_NAME=litellm-gateway \
    OTEL_TRACES_EXPORTER=none \
    OTEL_METRICS_EXPORTER=none \
    OTEL_LOGS_EXPORTER=none \
    # Gateway features
    A2A_GATEWAY_ENABLED=false \
    MCP_GATEWAY_ENABLED=false \
    # Admin UI
    ROUTEIQ_ADMIN_UI_ENABLED=false \
    # Config sync and hot reload
    CONFIG_HOT_RELOAD=false \
    CONFIG_SYNC_ENABLED=true \
    CONFIG_SYNC_INTERVAL=60 \
    # Store models/MCPs in database for persistence
    STORE_MODEL_IN_DB=false

# Health check for orchestrators (ECS, K8s, etc.)
# Uses the internal health endpoint which is unauthenticated
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:4000/_health/live || exit 1

EXPOSE 4000

# Security: Ensure proper signal handling for graceful shutdown
STOPSIGNAL SIGTERM

# Security: Run as non-root user
USER litellm

# =============================================================================
# ReadOnlyRootFilesystem Support
# =============================================================================
# This image supports Kubernetes readOnlyRootFilesystem: true with the following
# writable volume mounts required:
#   - /app/data      - Runtime data (if using local file storage)
#   - /app/models    - Model files (if hot-reloading models)
#   - /tmp           - Temporary files (Python, Prisma)
#   - /app/.cache    - Application cache (optional)
#
# Example K8s securityContext:
#   securityContext:
#     readOnlyRootFilesystem: true
#     runAsNonRoot: true
#     runAsUser: 1000
#   volumeMounts:
#     - name: tmp-vol
#       mountPath: /tmp
#     - name: data-vol
#       mountPath: /app/data
# =============================================================================

# Use tini as init system for proper signal handling (lands at /usr/bin/tini on Wolfi)
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["--config", "/app/config/config.yaml", "--port", "4000"]
