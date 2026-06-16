"""RouteIQ router stress-test + routing-validation harness.

A standalone, deploy-independent harness that fires a categorized workload at
RouteIQ's OpenAI-compatible ``POST /v1/chat/completions`` endpoint (with
``model: "auto"`` so RouteIQ's ML router picks the backend), captures the chosen
model from the response body, and validates routing behaviour against RouteIQ's
own surfaces:

  * ``GET /api/v1/routeiq/routing/config``  -> the ACTIVE strategy + the
    available-strategy set (the surface that NAMES the strategy).
  * ``GET /api/v1/routeiq/stats/global``    -> org-wide ``model_distribution`` +
    ``strategy_distribution`` rollups.
  * ``GET /api/v1/routeiq/routing/stats``   -> routing-decision totals +
    per-strategy distribution.
  * (optionally) the ``routing_decision`` CloudWatch Logs line -> the
    authoritative per-request (model, strategy) decision, joined by request id.

THE design goal (RouteIQ-4f19 + RouteIQ-833c): the harness GENERALIZES over
*whatever* routing strategy RouteIQ is running. It reads the active strategy by
name and always emits a generic distribution report; a REGISTRY of per-strategy
verdict plugins is dispatched by the active strategy name, with a safe generic
fallback for any unregistered strategy. The Kumaraswamy-Thompson bandit fan-out
check is ONE plugin among many, never the spine.

Built against the wire CONTRACT, not a live endpoint: unit-tested with a mocked
HTTP transport (``httpx.MockTransport``) and fixture stats payloads. ``--dry-run``
short-circuits before any network call. CW Logs enrichment is opt-in and lazily
imports boto3 so the default path needs neither boto3 nor AWS credentials.
"""

from __future__ import annotations

__version__ = "0.1.0"
