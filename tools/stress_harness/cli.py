"""argparse entrypoint wiring the RouteIQ stress harness end to end (RouteIQ-efa4).

Pipeline: workload.generate_mixed -> RouterClient.run (async, model:auto) ->
optional RouteIQStatsClient.fetch (active strategy + distributions) -> optional
cwlogs.enrich (per-request decision lines) -> analysis.analyze (generic
distributions + strategy-agnostic verdict) -> report.write_report.

``--dry-run`` short-circuits before any network call: it prints the generated
five-category plan (incl. the headline ``--num-requests 10000`` allocation) and
exits 0. CW Logs enrichment is opt-in (``--enrich-cwlogs``) and lazily imports
boto3, so the default path needs neither boto3 nor AWS credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from . import __version__
from .models import EnrichedRecord, RequestRecord, RouteIQStats


def _parse_category_weights(spec: str | None) -> dict[str, float] | None:
    """Parse ``--category-weights`` ``name=weight,name=weight`` into a dict.

    Returns None for an empty/absent spec (caller defaults to uniform). Raises
    ``argparse.ArgumentTypeError`` on a malformed pair.
    """
    if not spec:
        return None
    weights: dict[str, float] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"category-weights entry '{pair}' must be name=weight"
            )
        name, _, raw = pair.partition("=")
        try:
            weights[name.strip()] = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"category-weights value for '{name}' is not a number: {raw}"
            ) from exc
    return weights or None


def _parse_turn_lengths(spec: str | None) -> tuple[int, ...]:
    """Parse ``--turn-lengths`` ``2,3,5,8`` into a tuple of ints (each >=2).

    Returns the default ``(2,3,4,5)`` for an empty/absent spec.
    """
    if not spec:
        return (2, 3, 4, 5)
    if isinstance(spec, tuple):  # argparse may re-hand the parsed default
        return spec
    out: list[int] = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(max(2, int(raw)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"turn-lengths entry '{raw}' is not an integer"
            ) from exc
    return tuple(out) or (2, 3, 4, 5)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="routeiq-stress",
        description=(
            "Stress-test RouteIQ's /v1/chat/completions endpoint with model:auto "
            "and validate which backend model the ACTIVE routing strategy picked "
            "per category. Strategy-agnostic: reads the active strategy from "
            "RouteIQ's surfaces and dispatches a per-strategy verdict."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # --- target ---
    # NOT required at the argparse level: ``--dry-run`` short-circuits before any
    # network call, so it needs no target. ``main()`` enforces --base-url for a
    # real (non-dry-run) run instead, so an operator can preview the 10k plan with
    # just ``routeiq-stress --dry-run`` (RouteIQ-3b18).
    p.add_argument(
        "--base-url",
        default=None,
        help="RouteIQ gateway base URL, e.g. http://localhost:4000. "
        "Required for a real run; OPTIONAL under --dry-run (no network call).",
    )
    p.add_argument(
        "--token",
        default=None,
        help="RouteIQ data-plane key (Authorization: Bearer ...).",
    )
    p.add_argument(
        "--model",
        default="auto",
        help="OpenAI 'model' field. MUST stay 'auto' (default) to exercise the "
        "ML router; a pinned name bypasses routing.",
    )
    # --- control-plane stats (strategy + distributions) ---
    p.add_argument(
        "--stats-url",
        default=None,
        help="Base URL for RouteIQ control-plane stats (defaults to --base-url). "
        "Reads /routing/config + /stats/global + /routing/stats to NAME the "
        "active strategy and the server-side distributions.",
    )
    p.add_argument(
        "--admin-key",
        default=None,
        help="RouteIQ admin key (X-Admin-API-Key) for the control-plane stats.",
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip the control-plane stats fetch (verdict falls back to generic).",
    )

    # --- workload ---
    p.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Number of SINGLE-TURN requests (default 50; headline run: 10000).",
    )
    p.add_argument(
        "--num-conversations",
        type=int,
        default=0,
        help="Number of MULTI-TURN conversations to ALSO fire (each expands to "
        "several turns per --turn-lengths).",
    )
    p.add_argument(
        "--turn-lengths",
        type=_parse_turn_lengths,
        default=(2, 3, 4, 5),
        help="Comma list of conversation lengths to cycle, e.g. '2,3,5,8'. Each "
        "clamped to >=2. Default '2,3,4,5'.",
    )
    p.add_argument(
        "--num-users",
        type=int,
        default=0,
        help="Round-robin N synthetic user ids over the workload (drives the "
        "personalized-routing per-user-drift verdict). 0 = off.",
    )
    p.add_argument(
        "--concurrency", type=int, default=4, help="In-flight requests/conversations."
    )
    p.add_argument(
        "--category-weights",
        type=_parse_category_weights,
        default=None,
        help="Per-category weighting, e.g. 'math=2,code=2,easy-chitchat=1'. "
        "Unset => uniform.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request client timeout (s).",
    )

    # --- CW Logs enrichment (opt-in, lazy boto3) ---
    p.add_argument(
        "--enrich-cwlogs",
        action="store_true",
        help="Fetch the authoritative routing_decision line per request from CW "
        "Logs (needs boto3 + AWS creds). Off by default; harness runs body-only.",
    )
    p.add_argument("--env", default="prod", help="Env segment of the log group.")
    p.add_argument(
        "--log-group",
        default=None,
        help="Override log group (default /aws/ecs/routeiq/<env>/gateway).",
    )
    p.add_argument("--region", default=None, help="AWS region for CW Logs.")
    p.add_argument(
        "--cwlogs-delay",
        type=float,
        default=60.0,
        help="Seconds to wait before querying CW Logs (logs land ~1 min late).",
    )

    # --- output ---
    p.add_argument(
        "--out-dir",
        default="stress-out",
        help="Directory for report.md / report.json.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the workload + print the plan WITHOUT any network call.",
    )
    return p


def _print_dry_run(args: argparse.Namespace) -> None:
    """Print the workload plan for ``--dry-run`` (no network)."""
    from . import workload

    plan = workload.plan_summary(args.num_requests, args.category_weights)
    conv_plan = workload.conversation_plan_summary(
        args.num_conversations,
        category_weights=args.category_weights,
        turn_lengths=args.turn_lengths,
        # match generate_mixed, which offsets the conversation seq past the
        # single-turn count — otherwise the turn-length cycle phase differs and
        # the reported total diverges from what run() actually fires.
        seq_offset=args.num_requests,
    )
    total_requests = sum(plan.values()) + conv_plan["total_turns"]
    print("=== routeiq-stress DRY RUN (no requests fired) ===")
    print(f"base-url          : {args.base_url or '<unset> (dry-run needs none)'}")
    print(f"model field       : {args.model}")
    print(f"single-turn reqs  : {args.num_requests}")
    print(
        f"conversations     : {args.num_conversations}  "
        f"(turn-lengths {list(args.turn_lengths)})"
    )
    print(f"  -> conv turns   : {conv_plan['total_turns']}")
    print(f"synthetic users   : {args.num_users}")
    print(f"concurrency       : {args.concurrency}")
    print(f"enrich-cwlogs     : {args.enrich_cwlogs}")
    print(f"out-dir           : {args.out_dir}")
    print("")
    print("single-turn category plan (send-tag -> count):")
    for category in sorted(plan):
        print(f"  {category:<16} {plan[category]:>7}")
    if args.num_conversations > 0:
        print("")
        print("conversation plan (send-tag -> conversations / turns):")
        cpc = conv_plan["conversations_per_category"]
        tpc = conv_plan["turns_per_category"]
        for category in sorted(cpc):
            print(
                f"  {category:<16} {cpc[category]:>5} convs / {tpc[category]:>7} turns"
            )
    print("")
    print(
        f"TOTAL requests    : {total_requests}  "
        f"({sum(plan.values())} single-turn + {conv_plan['total_turns']} "
        f"conversation turns)"
    )


def _fetch_stats(args: argparse.Namespace) -> RouteIQStats | None:
    """Fetch the control-plane stats snapshot (active strategy + distributions),
    unless ``--no-stats``. Never aborts the run on failure."""
    if args.no_stats:
        return None
    from .stats_client import RouteIQStatsClient

    stats_url = args.stats_url or args.base_url
    client = RouteIQStatsClient(
        stats_url, admin_key=args.admin_key, timeout=args.timeout
    )
    try:
        return asyncio.run(client.fetch())
    except Exception as exc:  # noqa: BLE001 — degrade, never abort
        print(
            f"Control-plane stats fetch failed ({exc}); proceeding without "
            "active-strategy naming.",
            file=sys.stderr,
        )
        return None


def _enrich_records(
    records: list[RequestRecord], args: argparse.Namespace
) -> list[EnrichedRecord]:
    """Wrap records as ``EnrichedRecord``s and, under ``--enrich-cwlogs``, attach
    the authoritative routing_decision lines. Never aborts the run."""
    enriched = [EnrichedRecord(request=r) for r in records]
    if not args.enrich_cwlogs:
        return enriched

    from .cwlogs import CloudWatchLogsEnricher

    request_ids = [r.request_id for r in enriched if r.request_id]
    sent = [r.sent_ts for r in records if r.sent_ts]
    if not request_ids or not sent:
        print("CW Logs enrichment: no request_ids to query; skipping.", file=sys.stderr)
        return enriched

    enricher = CloudWatchLogsEnricher(
        env=args.env,
        region=args.region,
        log_group=args.log_group,
        poll_delay_s=args.cwlogs_delay,
    )
    try:
        decisions = enricher.enrich(request_ids, min(sent), max(sent))
    except Exception as exc:  # noqa: BLE001 — degrade, never abort the run
        print(
            f"CW Logs enrichment failed ({exc}); proceeding body-only.",
            file=sys.stderr,
        )
        return enriched

    for rec in enriched:
        if rec.request_id and rec.request_id in decisions:
            rec.decision = decisions[rec.request_id]
    return enriched


def run(args: argparse.Namespace) -> int:
    """Execute the full pipeline for parsed ``args``. Returns a process exit code."""
    from . import analysis, report, workload
    from .client import RouterClient

    singles, conversations = workload.generate_mixed(
        args.num_requests,
        args.num_conversations,
        category_weights=args.category_weights,
        turn_lengths=args.turn_lengths,
        num_users=args.num_users,
    )
    if not singles and not conversations:
        print(
            "No requests generated (num-requests and num-conversations both <= 0).",
            file=sys.stderr,
        )
        return 1

    client = RouterClient(
        args.base_url,
        token=args.token,
        model=args.model,
        timeout=args.timeout,
    )

    async def _fire_all() -> list[RequestRecord]:
        out: list[RequestRecord] = []
        if singles:
            out.extend(await client.run(singles, concurrency=args.concurrency))
        if conversations:
            done = await client.run_conversations(
                conversations, concurrency=args.concurrency
            )
            for conv in done:
                out.extend(conv)
        return out

    records = asyncio.run(_fire_all())

    server_stats = _fetch_stats(args)
    enriched = _enrich_records(records, args)

    result = analysis.analyze(enriched, server_stats=server_stats)
    paths = report.write_report(result, args.out_dir)

    verdict = result.verdict
    print(
        f"Fired {result.total_requests} requests "
        f"({result.successful_requests} ok, {result.enriched_requests} enriched) "
        f"across {result.total_conversations} conversations "
        f"({result.multi_turn_conversations} multi-turn); "
        f"{result.distinct_models} distinct models."
    )
    print(f"Active strategy : {result.active_strategy or '<unknown>'}")
    if verdict is not None:
        print(f"Verdict ({verdict.family}): {verdict.summary}")
    print(f"Report : {paths['markdown']}")
    print(f"JSON   : {paths['json']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        _print_dry_run(args)
        return 0
    # A real run MUST have a target. --base-url is optional at the argparse level
    # (so --dry-run needs no target) but is enforced here for the network path
    # (RouteIQ-3b18).
    if not args.base_url:
        parser.error("--base-url is required (unless --dry-run).")
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
