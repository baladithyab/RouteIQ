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
Environment variables for gateway configuration
*/}}
{{- define "routeiq-gateway.envVars" -}}
# Core configuration
- name: LITELLM_CONFIG_PATH
  value: {{ .Values.gateway.configPath | quote }}
- name: PORT
  value: {{ .Values.gateway.port | quote }}

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
{{- end }}
