"""CloudWatch Logs ``routing_decision`` enrichment (optional surface B, RouteIQ-b0df).

RouteIQ's ``router_decision_callback`` emits a ``routing_decision`` log line per
routing decision carrying the AUTHORITATIVE per-request (model, strategy) pair,
keyed on the request id. This module is OPTIONAL: the default harness path runs
body-only and needs neither boto3 nor AWS credentials.

  1. ``parse_decision_line`` — pure JSON-line -> ``RoutingDecisionLine`` parser.
     THE robustness surface: it tolerates the strategy field being absent (older
     builds, or a build that logs only the model), returning a line with
     ``strategy=None`` rather than raising. A line with no request id can't be
     joined, so it returns None.

  2. ``CloudWatchLogsEnricher`` — runs a Logs Insights query for a batch of
     request ids and attaches decision lines. boto3 is imported LAZILY inside the
     method so the package (and its tests) import cleanly without boto3 / creds.

The parser tolerates several emission forms for the same logical data because
RouteIQ may log the decision nested under ``routing_decision`` / ``decision``, or
flattened as top-level / dotted keys, and Logs Insights flattens rows to
``{field, value}`` cells:

  * nested    ``{"routing_decision": {"request_id":..,"model":..,"strategy":..}}``
  * flat      ``{"request_id":.., "model":.., "strategy":.., "message":"routing_decision"}``
  * dotted    ``{"routing_decision.model":.., "routing_decision.strategy":..}``
"""

from __future__ import annotations

import json
import time
from typing import Any

from .models import RoutingDecisionLine

# log group template; <env> filled by the CLI (e.g. prod, staging).
LOG_GROUP_TEMPLATE = "/aws/ecs/routeiq/{env}/gateway"

# the message marker RouteIQ uses for the routing-decision record.
_DECISION_MESSAGE_MARKER = "routing_decision"

# the request-id field name RouteIQ logs (request_id is canonical; some builds
# echo the OpenAI completion id as ``id``).
_REQUEST_ID_KEYS: tuple[str, ...] = ("request_id", "id", "completion_id")


def _flatten_decision(line: dict[str, Any]) -> dict[str, Any]:
    """Collapse a raw log line into one flat field map regardless of emission
    form (nested under routing_decision/decision, dotted, or already flat)."""
    flat: dict[str, Any] = {}
    # already-flat top-level keys (lowest priority).
    flat.update({k: v for k, v in line.items() if isinstance(k, str)})
    # dotted keys: routing_decision.<field> / decision.<field>.
    for prefix in (_DECISION_MESSAGE_MARKER, "decision"):
        dotted_prefix = f"{prefix}."
        for key, value in line.items():
            if isinstance(key, str) and key.startswith(dotted_prefix):
                flat[key[len(dotted_prefix) :]] = value
    # nested objects (highest priority — the explicit decision block wins).
    for prefix in (_DECISION_MESSAGE_MARKER, "decision"):
        nested = line.get(prefix)
        if isinstance(nested, dict):
            flat.update(nested)
    return flat


def parse_decision_line(raw_line: str | dict[str, Any]) -> RoutingDecisionLine | None:
    """Parse one CW Logs line (JSON string or pre-decoded dict) into a
    ``RoutingDecisionLine``.

    Returns None for a non-JSON / non-object line or one with no request id (can't
    be joined). The ``strategy`` field is optional — a line without it parses with
    ``strategy=None`` (never raises).
    """
    if isinstance(raw_line, str):
        try:
            data = json.loads(raw_line)
        except (ValueError, UnicodeDecodeError):
            return None
    else:
        data = raw_line
    if not isinstance(data, dict):
        return None

    flat = _flatten_decision(data)

    request_id = None
    for key in _REQUEST_ID_KEYS:
        candidate = _opt_str(flat.get(key))
        if candidate:
            request_id = candidate
            break
    if not request_id:
        return None

    return RoutingDecisionLine(
        present=True,
        request_id=request_id,
        model=_opt_str(flat.get("model")) or _opt_str(flat.get("selected_model")),
        strategy=_opt_str(flat.get("strategy"))
        or _opt_str(flat.get("active_strategy")),
        profile=_opt_str(flat.get("profile")) or _opt_str(flat.get("routing_profile")),
        latency_ms=_opt_float(flat.get("latency_ms")),
        cost_usd=_opt_float(flat.get("cost_usd")) or _opt_float(flat.get("cost")),
    )


def parse_lines(
    raw_lines: list[str | dict[str, Any]],
) -> dict[str, RoutingDecisionLine]:
    """Parse a batch of raw log lines into request_id -> RoutingDecisionLine.

    Later lines for the same request id win (the most recent decision record).
    """
    out: dict[str, RoutingDecisionLine] = {}
    for line in raw_lines:
        parsed = parse_decision_line(line)
        if parsed and parsed.request_id:
            out[parsed.request_id] = parsed
    return out


class CloudWatchLogsEnricher:
    """Fetches + parses ``routing_decision`` lines for a batch of request ids via
    Logs Insights. boto3 is imported lazily so the package never hard-depends on
    it.

    Disabled by default at the CLI (``--enrich-cwlogs`` opt-in). When boto3 or
    credentials are absent the enricher raises a clear error at QUERY time, not
    import time — body-only runs never touch this class.
    """

    def __init__(
        self,
        env: str,
        *,
        region: str | None = None,
        log_group: str | None = None,
        poll_delay_s: float = 60.0,
        query_window_pad_s: float = 300.0,
        poll_timeout_s: float = 60.0,
    ):
        self.log_group = log_group or LOG_GROUP_TEMPLATE.format(env=env)
        self.region = region
        self.poll_delay_s = poll_delay_s
        self.query_window_pad_s = query_window_pad_s
        self.poll_timeout_s = poll_timeout_s

    def _client(self) -> Any:
        """Lazily construct a boto3 ``logs`` client. Raises a clear RuntimeError
        when boto3 isn't installed so the failure is actionable."""
        try:
            import boto3  # noqa: PLC0415 — lazy by design
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for --enrich-cwlogs but is not installed; "
                "`uv sync --extra cloud` or run without CW Logs enrichment."
            ) from exc
        kwargs = {"region_name": self.region} if self.region else {}
        return boto3.client("logs", **kwargs)

    def build_query(self, request_ids: list[str]) -> str:
        """Build the Logs Insights query string for a set of request ids.

        Selects the request id plus the RouteIQ decision fields (model, strategy,
        profile, latency, cost). Insights returns absent fields as empty — no
        error if a build doesn't emit strategy. Exposed separately so it is
        unit-testable without any AWS call.
        """
        quoted = ", ".join(json.dumps(rid) for rid in request_ids)
        return (
            "fields @timestamp, request_id, id, model, selected_model, strategy, "
            "active_strategy, profile, routing_profile, latency_ms, cost_usd, "
            "message\n"
            f"| filter (request_id in [{quoted}]) or (id in [{quoted}])\n"
            "| sort @timestamp asc\n"
            "| limit 10000"
        )

    def enrich(
        self,
        request_ids: list[str],
        start_ts: float,
        end_ts: float,
        *,
        sleep: Any = time.sleep,
    ) -> dict[str, RoutingDecisionLine]:
        """Query CW Logs for ``request_ids`` in ``[start_ts, end_ts]`` (epoch
        seconds), poll to completion, return request_id -> RoutingDecisionLine.

        Waits ``poll_delay_s`` before querying (logs land ~1 min late). ``sleep``
        is injectable so tests run instantly. Returns {} for no request ids.
        Never raises on a missing strategy field — handled by the parser.
        """
        if not request_ids:
            return {}
        client = self._client()
        if self.poll_delay_s > 0:
            sleep(self.poll_delay_s)

        start = int(start_ts - self.query_window_pad_s)
        end = int(end_ts + self.query_window_pad_s)
        start_resp = client.start_query(
            logGroupName=self.log_group,
            startTime=start,
            endTime=end,
            queryString=self.build_query(request_ids),
        )
        query_id = start_resp["queryId"]

        deadline = self.poll_timeout_s
        waited = 0.0
        # Insights returns each row as a list of {field, value} cell dicts.
        results: list[list[dict[str, str]]] = []
        while True:
            resp = client.get_query_results(queryId=query_id)
            status = resp.get("status")
            if status == "Complete":
                results = resp.get("results", [])
                break
            if status in ("Failed", "Cancelled", "Timeout"):
                raise RuntimeError(f"CW Logs Insights query {status}: {query_id}")
            if waited >= deadline:
                raise RuntimeError(
                    f"CW Logs Insights query did not complete within {deadline}s "
                    f"(status={status})"
                )
            sleep(1.0)
            waited += 1.0

        return parse_lines([_row_to_dict(row) for row in results])


def _row_to_dict(row: list[dict[str, str]]) -> dict[str, Any]:
    """Convert a Logs Insights result row (list of {field,value} pairs) into a
    plain dict the parser understands. JSON-decodes values that look like JSON."""
    out: dict[str, Any] = {}
    for cell in row:
        field = cell.get("field")
        if not field or field.startswith("@"):
            continue  # skip Insights metadata columns (@ptr, @timestamp dup)
        out[field] = _maybe_json(cell.get("value"))
    return out


def _maybe_json(value: str | None) -> Any:
    """Best-effort decode an Insights string cell. Leaves plain strings alone."""
    if value is None:
        return None
    stripped = value.strip()
    if (stripped and stripped[0] in "[{") or stripped in ("true", "false", "null"):
        try:
            return json.loads(stripped)
        except ValueError:
            return value
    return value


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
