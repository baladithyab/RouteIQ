# Dashboard Embedding (CloudWatch & Managed Grafana)

RouteIQ's Observability page can surface an embedded AWS dashboard alongside the
in-app panels, so operators get the full **CloudWatch GenAI Observability**
time-series (or an Amazon Managed Grafana board) without leaving the admin UI.

The in-app cards (Service Health, Model Quality, Eval Pipeline, cost/usage
trend charts) read RouteIQ's own stats endpoints; the embed is the
**cluster-wide, metrics-backend source of truth** (see
[Observability — routing-stats are per-worker](observability.md#routing-stats-endpoints-are-per-worker)).

## What gets embedded

RouteIQ ships a ready-to-import dashboard definition at
[`config/cloudwatch-dashboard.json`](https://github.com/baladithyab/RouteIQ/blob/main/config/cloudwatch-dashboard.json),
built on the OpenTelemetry GenAI semantic-convention metrics RouteIQ emits:

| Widget | Metric |
|--------|--------|
| GenAI Request Duration (p50/p95/p99) | `gen_ai.client.operation.duration` |
| Token Usage (Input vs Output) | `gen_ai.client.token.usage` |
| Requests / errors by model | `gen_ai.client.operation.duration` count + error rate |

These are the same `gen_ai.*` instruments scraped at `/metrics` and pushed via
the OTLP collector to Amazon Managed Prometheus (AMP) / CloudWatch.

## Option A — CloudWatch dashboard

1. **Import the dashboard.** Create a CloudWatch dashboard from the shipped
   definition:

    ```bash
    aws cloudwatch put-dashboard \
      --dashboard-name RouteIQ-GenAI \
      --dashboard-body file://config/cloudwatch-dashboard.json
    ```

2. **Get a shareable URL.** The Observability page accepts either a deep link
   (opens CloudWatch in a new tab) or an iframe `src`. For the iframe path you
   need a CloudWatch **dashboard sharing** URL (Actions → Share dashboard),
   which produces a public/SSO-gated `https://cloudwatch.amazonaws.com/dashboard.html?dashboard=...`
   link. Treat shared dashboards as sensitive — prefer SSO-gated sharing.

3. **Point the UI at it.** Set the embed URL in the UI runtime config
   (`ui/public/config.js`, replaceable at deploy time via ConfigMap / S3):

    ```js
    window.__ROUTEIQ_CONFIG__ = {
      // ... existing keys ...
      DASHBOARD_EMBED: {
        provider: "cloudwatch",
        // Deep-link (opens in a new tab) — always safe:
        url: "https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=RouteIQ-GenAI",
        // Optional iframe src for inline embedding (requires a shared dashboard URL):
        embed_url: "",
      },
    };
    ```

## Option B — Amazon Managed Grafana

1. Add CloudWatch (or AMP) as a data source in your Managed Grafana workspace.
2. Build or import a board using the same `gen_ai.*` metrics.
3. Enable embedding on the workspace (Grafana **Settings → Security → Allow
   embedding**) and create a snapshot or signed embed URL.
4. Point the UI at the Grafana board:

    ```js
    window.__ROUTEIQ_CONFIG__ = {
      DASHBOARD_EMBED: {
        provider: "grafana",
        url: "https://g-xxxx.grafana-workspace.us-east-1.amazonaws.com/d/routeiq-genai",
        embed_url: "https://g-xxxx.grafana-workspace.us-east-1.amazonaws.com/d-solo/routeiq-genai?panelId=1",
      },
    };
    ```

## How the UI renders it

- If `DASHBOARD_EMBED.embed_url` is set, the Observability page renders it in a
  sandboxed `<iframe>` (best for at-a-glance monitoring on a wall display).
- If only `url` is set, the page renders a **"Open dashboard"** deep-link button
  (the safe default — no third-party framing, no CSP relaxation needed).
- If neither is set, the embed card is hidden and only the in-app panels show.

!!! warning "Content-Security-Policy / framing"
    Embedding AWS consoles in an iframe requires the browser to allow framing of
    the AWS origin. CloudWatch's main console blocks framing (`X-Frame-Options`);
    only **shared dashboard** URLs and Grafana **embed/snapshot** URLs are
    frameable. When in doubt, use the deep-link (`url`) form — it never needs a
    CSP change.

## Per-request traces (X-Ray)

The Observability page's **Live Routing Decisions** view can deep-link each
recent decision to its trace. Set the X-Ray console base so the link resolves:

```js
window.__ROUTEIQ_CONFIG__ = {
  XRAY_CONSOLE_BASE: "https://console.aws.amazon.com/xray/home?region=us-east-1#/traces",
};
```

RouteIQ emits `router.*` + `gen_ai.*` span attributes for every decision (see
[Observability — Routing Decision Telemetry](observability.md#routing-decision-telemetry)),
so a trace ID resolves to the full per-request waterfall in X-Ray.
