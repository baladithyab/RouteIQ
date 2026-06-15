{{/*
Expand the name of the chart.
*/}}
{{- define "routeiq-gateway.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "routeiq-gateway.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "routeiq-gateway.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "routeiq-gateway.labels" -}}
helm.sh/chart: {{ include "routeiq-gateway.chart" . }}
{{ include "routeiq-gateway.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "routeiq-gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "routeiq-gateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "routeiq-gateway.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "routeiq-gateway.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
Supports digest (immutable) or tag
*/}}
{{- define "routeiq-gateway.image" -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- $digest := .Values.image.digest -}}
{{- if $digest }}
{{- printf "%s@%s" $repository $digest }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Return the secret name
*/}}
{{- define "routeiq-gateway.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- include "routeiq-gateway.fullname" . }}-secrets
{{- end }}
{{- end }}

{{/*
Return the configmap name
*/}}
{{- define "routeiq-gateway.configMapName" -}}
{{- include "routeiq-gateway.fullname" . }}-config
{{- end }}

{{/*
Create pod anti-affinity rules
*/}}
{{- define "routeiq-gateway.podAntiAffinity" -}}
{{- if and .Values.podAntiAffinity.enabled (not .Values.affinity) }}
{{- if eq .Values.podAntiAffinity.type "hard" }}
requiredDuringSchedulingIgnoredDuringExecution:
  - labelSelector:
      matchLabels:
        {{- include "routeiq-gateway.selectorLabels" . | nindent 8 }}
    topologyKey: {{ .Values.podAntiAffinity.topologyKey }}
{{- else }}
preferredDuringSchedulingIgnoredDuringExecution:
  - weight: 100
    podAffinityTerm:
      labelSelector:
        matchLabels:
          {{- include "routeiq-gateway.selectorLabels" . | nindent 10 }}
      topologyKey: {{ .Values.podAntiAffinity.topologyKey }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "routeiq-gateway.imagePullSecrets" -}}
{{- $pullSecrets := list }}
{{- if .Values.global.imagePullSecrets }}
{{- range .Values.global.imagePullSecrets }}
{{- $pullSecrets = append $pullSecrets . }}
{{- end }}
{{- end }}
{{- if $pullSecrets }}
imagePullSecrets:
{{- toYaml $pullSecrets | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Database-only environment variables (for init containers like db-migrate)

RouteIQ-on-AWS P1 (ADR-0028) BOOT-RENDER NOTE: on the Aurora IAM-auth path
(externalPostgresql.existingSecret empty) this renders a COMPLETE, password-less
DATABASE_URL (postgresql://routeiq@<host>:5432/litellm?sslmode=require) and the
app (database.py) mints the 15-min rds-db:connect token in-process and splices it
in IN PYTHON -- the $(POSTGRES_PASSWORD) substring below only appears in the
static-password interim (Shape B). Do NOT trust K8s $(VAR) interpolation to
assemble the URL: K8s only expands a $(VAR) defined EARLIER in the env list and
does no second pass, so DATABASE_URL emitted before POSTGRES_PASSWORD leaves the
$(POSTGRES_PASSWORD) substring LITERAL. For Shape B the referent MUST be ordered
first. See research/p1/discover-chart-state.md section 3 (the highest-risk seam).
*/}}
{{- define "routeiq-gateway.databaseEnv" -}}
{{- if .Values.externalPostgresql.host }}
{{- if .Values.externalPostgresql.existingSecret }}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.externalPostgresql.existingSecret }}
      key: {{ .Values.externalPostgresql.existingSecretKey | default "password" }}
- name: DATABASE_URL
  value: {{ printf "postgresql://%s:$(POSTGRES_PASSWORD)@%s:%d/%s?sslmode=%s" .Values.externalPostgresql.username .Values.externalPostgresql.host (int .Values.externalPostgresql.port) .Values.externalPostgresql.database .Values.externalPostgresql.sslMode | quote }}
{{- else }}
# IAM-auth (ADR-0028): password-less complete URL; app mints rds-db:connect token.
# The flag is DERIVED from an empty existingSecret (single source of truth).
- name: DATABASE_URL
  value: {{ printf "postgresql://%s@%s:%d/%s?sslmode=%s" .Values.externalPostgresql.username .Values.externalPostgresql.host (int .Values.externalPostgresql.port) .Values.externalPostgresql.database .Values.externalPostgresql.sslMode | quote }}
- name: ROUTEIQ_DB_IAM_AUTH
  value: "true"
{{- end }}
{{- else if .Values.secrets.values.DATABASE_URL }}
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: {{ include "routeiq-gateway.secretName" . }}
      key: DATABASE_URL
{{- end }}
{{- end }}

{{/*
Environment variables for gateway configuration
*/}}
{{- define "routeiq-gateway.envVars" -}}
# Core configuration
- name: LITELLM_CONFIG_PATH
  value: {{ .Values.gateway.configPath | quote }}
- name: PORT
  value: {{ .Values.gateway.port | quote }}

# readOnlyRootFilesystem cache redirect (C6)
# The container runs as HOME=/app (Dockerfile useradd -d /app) under
# readOnlyRootFilesystem: true, so any library writing to $HOME/.cache
# (HuggingFace transformers/tokenizers, matplotlib, fontconfig) would EROFS
# because /app is read-only and /app/.cache is NOT a mounted volume. Redirect
# every cache into the existing writable /app/data emptyDir mount (fsGroup 1000)
# so the transformers/mmBERT routing strategies can load their models/tokenizers.
# Emitted unconditionally -- harmless when the root FS is writable.
- name: HF_HOME
  value: /app/data/.cache/huggingface
- name: HF_HUB_CACHE
  value: /app/data/.cache/huggingface/hub
- name: TRANSFORMERS_CACHE
  value: /app/data/.cache/huggingface
- name: XDG_CACHE_HOME
  value: /app/data/.cache
- name: MPLCONFIGDIR
  value: /app/data/.cache/matplotlib

# Feature flags
- name: MCP_GATEWAY_ENABLED
  value: {{ .Values.gateway.features.mcpGatewayEnabled | quote }}
- name: A2A_GATEWAY_ENABLED
  value: {{ .Values.gateway.features.a2aGatewayEnabled | quote }}
- name: MCP_HA_SYNC_ENABLED
  value: {{ .Values.gateway.features.mcpHaSyncEnabled | quote }}
- name: LLMROUTER_ENABLE_MCP_TOOL_INVOCATION
  value: {{ .Values.gateway.features.mcpToolInvocationEnabled | quote }}

# Hot reload settings
- name: LLMROUTER_HOT_RELOAD
  value: {{ .Values.gateway.hotReload.enabled | quote }}
- name: LLMROUTER_RELOAD_INTERVAL
  value: {{ .Values.gateway.hotReload.interval | quote }}
- name: CONFIG_HOT_RELOAD
  value: {{ .Values.gateway.hotReload.enabled | quote }}

# Config sync settings
- name: CONFIG_SYNC_ENABLED
  value: {{ .Values.gateway.configSync.enabled | quote }}
- name: CONFIG_SYNC_INTERVAL
  value: {{ .Values.gateway.configSync.interval | quote }}
{{- if .Values.gateway.configSync.s3.bucket }}
- name: CONFIG_S3_BUCKET
  value: {{ .Values.gateway.configSync.s3.bucket | quote }}
- name: CONFIG_S3_KEY
  value: {{ .Values.gateway.configSync.s3.key | quote }}
{{- end }}
{{- if .Values.gateway.configSync.gcs.bucket }}
- name: CONFIG_GCS_BUCKET
  value: {{ .Values.gateway.configSync.gcs.bucket | quote }}
- name: CONFIG_GCS_KEY
  value: {{ .Values.gateway.configSync.gcs.key | quote }}
{{- end }}

# Database settings
- name: STORE_MODEL_IN_DB
  value: {{ .Values.gateway.database.storeModelInDb | quote }}
- name: LITELLM_RUN_DB_MIGRATIONS
  value: {{ .Values.gateway.database.runMigrations | quote }}

# OTEL settings
- name: OTEL_ENABLED
  value: {{ .Values.gateway.otel.enabled | quote }}
- name: OTEL_SERVICE_NAME
  value: {{ .Values.gateway.otel.serviceName | quote }}
- name: OTEL_TRACES_EXPORTER
  value: {{ .Values.gateway.otel.tracesExporter | quote }}
- name: OTEL_METRICS_EXPORTER
  value: {{ .Values.gateway.otel.metricsExporter | quote }}
- name: OTEL_LOGS_EXPORTER
  value: {{ .Values.gateway.otel.logsExporter | quote }}
{{- if .Values.gateway.otel.endpoint }}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ .Values.gateway.otel.endpoint | quote }}
{{- end }}

# SSRF protection
- name: LLMROUTER_ALLOW_PRIVATE_IPS
  value: {{ .Values.gateway.ssrf.allowPrivateIps | quote }}
{{- if .Values.gateway.ssrf.allowlistHosts }}
- name: LLMROUTER_SSRF_ALLOWLIST_HOSTS
  value: {{ .Values.gateway.ssrf.allowlistHosts | quote }}
{{- end }}
{{- if .Values.gateway.ssrf.allowlistCidrs }}
- name: LLMROUTER_SSRF_ALLOWLIST_CIDRS
  value: {{ .Values.gateway.ssrf.allowlistCidrs | quote }}
{{- end }}

# RouteIQ-specific settings
- name: ROUTEIQ_USE_PLUGIN_STRATEGY
  value: {{ .Values.routeiq.pluginStrategy.enabled | quote }}
- name: ROUTEIQ_WORKERS
  value: {{ .Values.routeiq.workers | quote }}
- name: ROUTEIQ_CENTROID_ROUTING
  value: {{ .Values.routeiq.centroidRouting.enabled | quote }}
- name: ROUTEIQ_CENTROID_WARMUP
  value: {{ .Values.routeiq.centroidRouting.warmup | quote }}
- name: ROUTEIQ_ROUTING_PROFILE
  value: {{ .Values.routeiq.routingProfile | quote }}
- name: ROUTEIQ_ADMIN_UI_ENABLED
  value: {{ .Values.routeiq.adminUI.enabled | quote }}

# Leader election
{{- if .Values.routeiq.leaderElection.enabled }}
- name: LLMROUTER_HA_MODE
  value: "leader_election"
{{- if .Values.routeiq.leaderElection.backend }}
- name: ROUTEIQ_LEADER_ELECTION_BACKEND
  value: {{ .Values.routeiq.leaderElection.backend | quote }}
{{- end }}
{{- end }}

# Pod identity for leader election (Downward API)
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace

# External PostgreSQL
# P1 (ADR-0028) BOOT-RENDER: IAM-auth (existingSecret empty) renders a complete
# password-less URL (app mints the rds-db:connect token in Python); Shape-B
# interim orders POSTGRES_PASSWORD BEFORE DATABASE_URL so the K8s $(VAR) expansion
# actually fires (it does not for a $(VAR) defined later). See databaseEnv above.
{{- if .Values.externalPostgresql.host }}
{{- if .Values.externalPostgresql.existingSecret }}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.externalPostgresql.existingSecret }}
      key: {{ .Values.externalPostgresql.existingSecretKey | default "password" }}
- name: DATABASE_URL
  value: {{ printf "postgresql://%s:$(POSTGRES_PASSWORD)@%s:%d/%s?sslmode=%s" .Values.externalPostgresql.username .Values.externalPostgresql.host (int .Values.externalPostgresql.port) .Values.externalPostgresql.database .Values.externalPostgresql.sslMode | quote }}
{{- else }}
# IAM-auth (ADR-0028): password-less complete URL; app mints rds-db:connect token.
# The flag is DERIVED from an empty existingSecret (single source of truth).
- name: DATABASE_URL
  value: {{ printf "postgresql://%s@%s:%d/%s?sslmode=%s" .Values.externalPostgresql.username .Values.externalPostgresql.host (int .Values.externalPostgresql.port) .Values.externalPostgresql.database .Values.externalPostgresql.sslMode | quote }}
- name: ROUTEIQ_DB_IAM_AUTH
  value: "true"
{{- end }}
{{- end }}

# External Redis
# P1 (ADR-0029): serverless Valkey is TLS-mandatory (set externalRedis.ssl=true ->
# REDIS_SSL=true) + IAM-auth (existingSecret empty; app mints elasticache:Connect
# token and presents CacheIamUserName as the cache user).
{{- if .Values.externalRedis.host }}
- name: REDIS_HOST
  value: {{ .Values.externalRedis.host | quote }}
- name: REDIS_PORT
  value: {{ .Values.externalRedis.port | quote }}
- name: REDIS_DB
  value: {{ .Values.externalRedis.db | quote }}
- name: REDIS_SSL
  value: {{ .Values.externalRedis.ssl | quote }}
{{- if .Values.externalRedis.iamUserName }}
# IAM user the pod presents on connect (= RouteIqStateStack CfnOutput
# CacheIamUserName; user_id == user_name). Read directly by redis_pool.
- name: REDIS_USERNAME
  value: {{ .Values.externalRedis.iamUserName | quote }}
{{- end }}
{{- if .Values.externalRedis.iamAuth }}
# ADR-0029 IAM-auth: redis_pool mints an elasticache:Connect SigV4 token as the
# AUTH (15-min, refreshed per client build) instead of a static REDIS_PASSWORD.
- name: ROUTEIQ_REDIS_IAM_AUTH
  value: "true"
{{- end }}
{{- if .Values.externalRedis.existingSecret }}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.externalRedis.existingSecret }}
      key: {{ .Values.externalRedis.existingSecretKey | default "password" }}
{{- end }}
{{- end }}

# External OTel Collector
{{- if .Values.externalOtel.endpoint }}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ .Values.externalOtel.endpoint | quote }}
- name: OTEL_EXPORTER_OTLP_PROTOCOL
  value: {{ .Values.externalOtel.protocol | default "grpc" | quote }}
{{- end }}

# AWS substrate seam (P2: AMP / AppConfig CfnOutput consume + AWS_REGION)
# RouteIQ-bf9f. Every block is gated on a non-empty value so the default render
# stays byte-stable / cloud-agnostic.
{{- if .Values.aws.region }}
# LOAD-BEARING on AWS: every boto3 client (Bedrock, AppConfig poll, AMP, S3
# config sync, IAM-token minting) needs a region. Emitted as BOTH names because
# boto3 reads AWS_REGION first, AWS_DEFAULT_REGION as a fallback.
- name: AWS_REGION
  value: {{ .Values.aws.region | quote }}
- name: AWS_DEFAULT_REGION
  value: {{ .Values.aws.region | quote }}
{{- end }}
{{- if .Values.aws.appConfig.enabled }}
# AppConfig runtime config retrieval (ADR-0026 / RouteIQ-4333). Wires the
# config_sync AppConfig poll adapter via its ROUTEIQ_CONFIG_SYNC__APPCONFIG_*
# settings (ADR-0013 nested env prefix).
- name: ROUTEIQ_CONFIG_SYNC__APPCONFIG_ENABLED
  value: "true"
{{- if .Values.aws.appConfig.application }}
- name: ROUTEIQ_CONFIG_SYNC__APPCONFIG_APPLICATION
  value: {{ .Values.aws.appConfig.application | quote }}
{{- end }}
{{- if .Values.aws.appConfig.environment }}
- name: ROUTEIQ_CONFIG_SYNC__APPCONFIG_ENVIRONMENT
  value: {{ .Values.aws.appConfig.environment | quote }}
{{- end }}
{{- if .Values.aws.appConfig.profile }}
- name: ROUTEIQ_CONFIG_SYNC__APPCONFIG_PROFILE
  value: {{ .Values.aws.appConfig.profile | quote }}
{{- end }}
- name: ROUTEIQ_CONFIG_SYNC__APPCONFIG_POLL_INTERVAL_SECONDS
  value: {{ .Values.aws.appConfig.pollIntervalSeconds | quote }}
{{- end }}
{{- if .Values.aws.amp.remoteWriteUrl }}
# Amazon Managed Prometheus remote-write target (P2). Consumed by an ADOT
# collector sidecar (follow-up); this is the env seam.
- name: AMP_REMOTE_WRITE_URL
  value: {{ .Values.aws.amp.remoteWriteUrl | quote }}
{{- end }}

# OIDC / SSO
{{- if .Values.oidc.enabled }}
- name: ROUTEIQ_OIDC_ENABLED
  value: "true"
- name: ROUTEIQ_OIDC_ISSUER_URL
  value: {{ .Values.oidc.issuerUrl | quote }}
- name: ROUTEIQ_OIDC_CLIENT_ID
  value: {{ .Values.oidc.clientId | quote }}
{{- if .Values.oidc.existingSecret }}
- name: ROUTEIQ_OIDC_CLIENT_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ .Values.oidc.existingSecret }}
      key: {{ .Values.oidc.existingSecretKey | default "oidc-client-secret" }}
{{- end }}
{{- end }}
{{- end }}
