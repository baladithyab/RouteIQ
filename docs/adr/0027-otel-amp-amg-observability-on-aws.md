# ADR-0027: AWS-Native Observability — AMP + AMG + CloudWatch over OTel GenAI

**Status**: Proposed
**Date**: 2026-06-14
**Decision Makers**: RouteIQ Core Team

## Context

RouteIQ already emits standards-aligned telemetry. [ADR-0019](0019-otel-genai-conventions.md)
adopted the OpenTelemetry GenAI semantic conventions, and `telemetry_contracts.py`
defines the `gen_ai.*` span-attribute contract (`telemetry_contracts.py:44,71-89`,
attribute constants `:664-680`) plus routing-decision events emitted from
`router_decision_callback.py`. The Helm chart ships a `ServiceMonitor`,
`PrometheusRule`, and a `grafana-dashboard` ConfigMap
(`deploy/charts/routeiq-gateway/templates/`).

What RouteIQ does **not** have is the AWS-native *destination* for any of this:
no managed Prometheus, no managed Grafana, no CloudWatch dashboards, no metric
filters that turn routing-decision log lines into queryable metrics, and no
alarms wired to a notification topic. The `ServiceMonitor` and `PrometheusRule`
presuppose a Prometheus + Alertmanager that nobody provisions; the
`grafana-dashboard` ConfigMap presupposes a Grafana that nobody runs. On AWS this
is all manual (`docs/deployment/aws.md`).

vllm-sr-on-aws built the full AWS-native observability stack in
`cdk/lib/observability_construct.py` (and a per-model variant in
`eks_cluster_construct.py`). RouteIQ should adopt the same destination layer and
map its existing `gen_ai.*` contract onto it.

## Decision

Provision an AWS-native observability stack in IaC and map RouteIQ's OTel GenAI
telemetry onto it: **Amazon Managed Prometheus** (metrics sink), **Amazon Managed
Grafana** (SRE surface), **CloudWatch metric filters + alarms + dashboards** (the
routing-decision analytics), and a **single SNS oncall topic**.

### Amazon Managed Prometheus (`observability_construct.py:321-333`)

An `aps.CfnWorkspace` (alias `routeiq-{env}`). RouteIQ's pods already expose a
Prometheus endpoint (the chart's `ServiceMonitor`); metrics reach AMP via an
**ADOT collector sidecar `remote_write`** to the hand-built remote-write URL
`https://aps-workspaces.{region}.amazonaws.com/workspaces/{wsid}/api/v1/remote_write`
(`:329-332`). The workload's IRSA role (ADR-0030) needs `aps:RemoteWrite` on the
workspace ARN.

### Amazon Managed Grafana (`observability_construct.py:335-403`)

A `grafana.CfnWorkspace`, **flag-gated** (default off, `:191`) because it
**requires IAM Identity Center enabled** (`:336-341`). When on:
`authentication_providers=["AWS_SSO"]`, `permission_type="SERVICE_MANAGED"`,
`data_sources=["PROMETHEUS","XRAY","CLOUDWATCH"]` (`:390-400`), with a
data-source IAM role granting `aps:QueryMetrics`, `cloudwatch:GetMetricData`, and
`xray:GetTraceSummaries` (`:343-389`). This replaces the chart's
`grafana-dashboard` ConfigMap with a managed Grafana that points at AMP — and it
slots directly into RouteIQ's existing OIDC/SSO posture ([ADR-0008](0008-oidc-identity-integration.md)).

### CloudWatch metric filters — the routing-decision bridge

The highest-value mapping. RouteIQ already emits a structured routing-decision
event per request (the `gen_ai.*` + routing fields from
`router_decision_callback.py`). Two patterns from vllm-sr apply:

- **Per-model dimensioned filter** (`eks_cluster_construct.py:757-767`): a
  `MetricFilter` with `dimensions={"model": "$.selected_model"}` over the routing
  log group produces a per-model metric (`routing_latency_ms_by_model`). For
  RouteIQ the JSON key is `$.gen_ai.response.model` (or
  `$.["gen_ai.response.model"]`). Note: cardinality must be bounded (closed model
  set) and **AWS forbids `default_value` on a dimensioned filter**.
- **Aggregate filters** for ingest/error counts
  (`observability_construct.py:818-827,792-801`).

### CloudWatch dashboard with Metrics-Insights (`observability_construct.py:1228-1401`)

The routing-share dashboard uses Metrics-Insights queries over the EMF-mirrored
metric family, e.g.
`SELECT SUM(llm_routing_decisions_total) FROM SCHEMA("vllm-sr", chosen_model) GROUP BY chosen_model`
(`:1249-1257`) and per-category choice (`:1261-1269`). RouteIQ's analogue:
`SELECT SUM(routeiq_routing_decisions_total) FROM SCHEMA("routeiq", gen_ai_response_model) GROUP BY gen_ai_response_model`
— turning the `gen_ai.response.model` attribute into a live per-model traffic
share widget. Where a dimension is runtime-learned, use `SEARCH()` MathExpressions
(`eks_cluster_construct.py:866-920`).

### Alarms + SNS (`observability_construct.py:405-412,578-1116`)

A single `routeiq-{env}-oncall` SNS topic (`:405-412`, TLS-enforced resource
policy) is the action for every alarm: p99 latency (`:578-603`), error/throttle
surge, semantic-cache hit-ratio collapse (`:714-754`), pod memory >90% OOMKill
indicator (`eks_cluster_construct.py:616-638`), and an AnomalyDetectionAlarm for
latency (`:827-841`). These map onto the RouteIQ metrics RouteIQ already exports
via `metrics.py`. **The log group the filters read must be CDK-created** — a CFN
MetricFilter requires the group to pre-exist, and Fluent Bit's `auto_create_group`
only makes it at runtime (`eks_cluster_construct.py:712-721`).

### Field-name reconciliation (important)

vllm-sr's router emits `bandit.*` / `llm.routing.reward` dotted keys, **not**
`gen_ai.*`. RouteIQ's advantage is that it already standardizes on `gen_ai.*`
(`telemetry_contracts.py:664-680`), so the metric-filter JSON paths and the
Metrics-Insights `SCHEMA(...)` dimension names should use the `gen_ai.*` names
directly. This keeps one contract from span → metric filter → dashboard.

## Consequences

### Positive

- **The chart's stubs get a home.** `ServiceMonitor`/`PrometheusRule`/
  `grafana-dashboard` finally point at provisioned AMP + AMG.
- **Routing decisions become queryable.** The per-model dimensioned filter +
  Metrics-Insights turn RouteIQ's routing-decision events into live traffic-share
  and latency widgets without an external analytics pipeline.
- **One contract end-to-end.** `gen_ai.*` flows from span → log → metric filter
  → dashboard unchanged (RouteIQ already owns the convention; ADR-0019).
- **SSO-native Grafana.** AMG reuses RouteIQ's IAM Identity Center / OIDC posture.

### Negative

- **ADOT sidecar required** for AMP remote-write (one more container in the pod).
- **AMG prerequisite** is IAM Identity Center; it stays flag-gated off until that
  exists (matches vllm-sr `:191`).
- **Dimensioned-filter cardinality.** Per-model filters require a bounded model
  set and forbid `default_value`; high-cardinality dimensions are unsafe.
- **Log group ordering.** The routing log group must be CDK-created before the
  metric filter, or first deploy rolls back.

## Alternatives Considered

### Alternative A: CloudWatch only (no AMP/AMG)

- **Pros**: Fewer services; metric filters + dashboards alone cover a lot.
- **Cons**: Loses PromQL and the chart's existing `ServiceMonitor`/Grafana
  artifacts; SRE muscle memory is Grafana/PromQL.
- **Rejected as the *whole* answer**: CloudWatch is the routing-analytics layer;
  AMP+AMG is the metrics/PromQL layer. Both, with AMG flag-gated.

### Alternative B: Self-managed Prometheus + Grafana on the cluster

- **Pros**: No managed-service cost; full control.
- **Cons**: Operational burden (HA, storage, upgrades, Alertmanager) — exactly
  what RouteIQ is trying to *stop* doing manually.
- **Rejected**: Managed services remove the ops burden.

### Alternative C: Third-party APM (Datadog/Arize/Honeycomb)

- **Pros**: Turnkey GenAI views; ADR-0019 already makes traces compatible.
- **Cons**: Egress + per-host/per-span cost; a non-AWS data plane for a deploy
  that is otherwise AWS-native.
- **Rejected for the default**: ADR-0019 keeps this option open (OTLP export),
  but the AWS-native stack is the provisioned default.

## References

- `cdk/lib/observability_construct.py` (vllm-sr-on-aws) — AMP `CfnWorkspace`
  (`:321-333`), AMG `grafana.CfnWorkspace` (`:335-403`), Metrics-Insights
  dashboard (`:1228-1401`), metric filters (`:818-827`), alarms + SNS
  (`:405-412,578-1116`)
- `cdk/lib/eks_cluster_construct.py` (vllm-sr-on-aws) — per-model dimensioned
  filter `routing_latency_ms_by_model` (`:757-767`), `SEARCH()` widgets
  (`:866-920`), CDK-created log-group ordering lesson (`:712-721`)
- `../architecture/aws-rearchitecture/vllmsr-patterns.md` — "observability_construct.py: AMP (CfnWorkspace) +
  AMG (grafana CfnWorkspace, gated) + CW dashboard + 9 alarms + 3 metric filters"
- `src/litellm_llmrouter/telemetry_contracts.py` — `gen_ai.*` contract (`:664-680`)
- `src/litellm_llmrouter/router_decision_callback.py` / `metrics.py` — emitters
- [ADR-0019: OpenTelemetry GenAI Semantic Conventions](0019-otel-genai-conventions.md)
- [ADR-0008: OIDC/SSO Identity Integration](0008-oidc-identity-integration.md)
- [ADR-0030: EKS Auto Mode + IRSA Deployment Substrate](0030-eks-auto-mode-irsa-substrate.md)
