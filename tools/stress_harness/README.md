# RouteIQ router stress-test + routing-validation harness

A standalone, deploy-independent harness that fires a categorized workload at
RouteIQ's OpenAI-compatible `POST /v1/chat/completions` endpoint with
`model: "auto"` (so RouteIQ's ML router picks the backend), then validates the
routing behaviour against RouteIQ's own control-plane surfaces.

It is **strategy-agnostic**: it reads the *active* routing strategy by name and
dispatches a per-strategy verdict, with a safe generic fallback for any strategy
it has never heard of. The Kumaraswamy-Thompson bandit fan-out check is ONE
verdict plugin among many — never the spine.

Seeds: RouteIQ-b245 (workload + client), RouteIQ-4f19 (strategy-agnostic
analysis/report), RouteIQ-833c (strategy generalization + cross-strategy
compare), RouteIQ-efa4 (CLI + 10k + docs), RouteIQ-b0df (optional CW Logs
enrichment).

## Layout

```
tools/stress_harness/
  workload.py      # 5-bucket categorized generator (single + multi-turn, weighted)
  client.py        # async POST /v1/chat/completions, model:auto, Bearer auth
  stats_client.py  # reads RouteIQ /routing/config + /stats/global + /routing/stats
  cwlogs.py        # OPTIONAL routing_decision CW-Logs enrichment (lazy boto3)
  models.py        # RequestRecord / EnrichedRecord / RouteIQStats / AnalysisResult
  verdicts.py      # the per-strategy verdict REGISTRY + dispatch (the core seam)
  analysis.py      # always-available distributions + verdict dispatch
  compare.py       # cross-strategy diff of two runs on the same workload
  report.py        # markdown + JSON report
  cli.py           # argparse entrypoint
tests/stress_harness/   # mocked-transport unit tests (cred-free)
```

## Running

The harness is a standalone tool, not part of the installed `routeiq` package,
so put `tools/` on the path:

```bash
PYTHONPATH=tools uv run python -m stress_harness --base-url http://localhost:4000 --num-requests 50
```

### `--dry-run` (no network)

`--dry-run` short-circuits before any network call and prints the allocation.
Because it fires nothing, it needs **no `--base-url`** (RouteIQ-3b18):

```bash
PYTHONPATH=tools uv run python -m stress_harness --num-requests 10000 --dry-run
```

allocates exactly 10,000 requests across the five buckets (2000 each at uniform
weight) with no socket opened. Largest-remainder (Hamilton) rounding guarantees
the per-bucket counts sum *exactly* to the requested total for any weighting.
`--base-url` is required only for a *real* run (the CLI errors if it is missing
without `--dry-run`).

### The headline 10k run

```bash
PYTHONPATH=tools uv run python -m stress_harness \
    --base-url   http://localhost:4000 \
    --token      "$ROUTEIQ_KEY" \
    --admin-key  "$ROUTEIQ_ADMIN_KEY" \
    --num-requests 10000 \
    --concurrency  32 \
    --out-dir      stress-out-10k
```

Fires 10k `model:auto` requests, reads the active strategy + distributions from
RouteIQ's control plane, and writes `stress-out-10k/report.{md,json}`. Add
`--num-conversations 200 --turn-lengths 2,3,5,8` to mix in multi-turn traffic,
and `--num-users 50` to exercise per-user personalized routing.

### CLI flags

| flag | default | purpose |
|------|---------|---------|
| `--base-url` | (required for a real run; **optional under `--dry-run`**) | RouteIQ gateway base URL |
| `--token` | none | data-plane key (`Authorization: Bearer`) |
| `--model` | `auto` | OpenAI `model` field — keep `auto` to exercise routing |
| `--num-requests` | `50` | single-turn requests (headline: `10000`) |
| `--num-conversations` | `0` | multi-turn conversations to also fire |
| `--turn-lengths` | `2,3,4,5` | conversation lengths to cycle (each ≥2) |
| `--num-users` | `0` | round-robin N synthetic user ids (personalized verdict) |
| `--concurrency` | `4` | in-flight requests / conversations |
| `--category-weights` | uniform | e.g. `math=2,code=2,easy-chitchat=1` |
| `--stats-url` | `--base-url` | control-plane stats base URL |
| `--admin-key` | none | admin key (`X-Admin-API-Key`) for the stats surfaces |
| `--no-stats` | off | skip the control-plane fetch (verdict → generic) |
| `--enrich-cwlogs` | off | join authoritative `routing_decision` log lines (lazy boto3) |
| `--dry-run` | off | print the allocation, fire nothing |

## Strategy-generalization design (RouteIQ-4f19 + RouteIQ-833c)

This is the load-bearing design decision. RouteIQ has 18+ routing strategies
(KNN, SVM, MLP, MF, ELO, RouterDC, hybrid, centroid, cost-aware, GMT/personalized,
Router-R1, the Kumaraswamy-Thompson bandit, ...) and the operator can hot-swap
them at runtime via the registry. The harness must validate **whatever** strategy
is active without hardcoding any single one.

### 1. Read & name the active strategy

`stats_client.py` reads RouteIQ's surfaces:

- `GET /api/v1/routeiq/routing/config` → `active_strategy` + `available_strategies`
  (the surface that *names* the running strategy).
- `GET /api/v1/routeiq/stats/global` → org-wide `model_distribution` +
  `strategy_distribution` + `total_decisions`.
- `GET /api/v1/routeiq/routing/stats` → routing-decision totals + per-strategy
  distribution.

The reader is defensive: each endpoint may be absent, unauthorized, or shaped
slightly differently across versions. It merges what it can and annotates what it
cannot, never raising. The active strategy is named at the top of the report.

### 2. Always emit the generic distribution report

For ANY strategy, `analysis.py` computes (no strategy assumptions, no enrichment
required):

- **model_distribution** — requests per model (the primary routing-spread signal),
- **strategy_distribution** — per-request strategies from the decision lines, or
  the server-side `/stats/global` rollup,
- **per-category routing table** — which category routed to which model (+ which
  strategy, when decision lines carry it),
- **multi-turn views** — routing by turn position + within-conversation switching.

### 3. Per-strategy verdict registry (the dispatch)

`verdicts.py` holds a REGISTRY of verdict plugins, dispatched by the active
strategy name. Matching is by case-folded substring so the `llmrouter-` prefix
and version suffixes don't break dispatch. The first matching family wins:

| family | strategies | property checked |
|--------|------------|------------------|
| `fan-out` | kumaraswamy-thompson, thompson, bandit | healthy bandit **spreads** across arms; unhealthy = pinned to one model |
| `consistency` | knn, svm, mlp, mf, elo, routerdc, hybrid, centroid, graph, automix, causallm | same category routes to a **dominant** model |
| `cost-aware` | cost-aware | **cheap** models serve easy categories (flag premium-on-easy) |
| `cost-cascade` | cost-cascade, cascade, cheap-first | **cheap-first invariant**: cheap tier carries more traffic than premium AND no easy bucket is premium-dominated |
| `semantic-intent` | semantic-intent, semantic, intent | **bucket→group dispatch**: each semantic bucket routes dominantly AND distinct buckets dispatch to distinct models |
| `personalized` | personalized, gmt | per-user **drift** — different users diverge (reads `/me/stats` per-user routing when available, see below) |
| `latency-cost` | router-r1, *multiround | latency / token-cost **tradeoff** report (informational) |
| **generic** | *anything unregistered* | restates the distribution + "no strategy-specific verdict for `<name>`" — **never crashes** |

7 registered families + the generic fallback (the acceptance floor is 3). An
unknown / future strategy always gets the generic verdict; a verdict plugin that
itself errors degrades to generic with a note, so one buggy plugin can never crash
the report.

The `cost-cascade` family precedes `cost-aware`, and `semantic-intent` precedes
the broad `consistency` family, in the token-match order — so a router that
explicitly cascades cheap→expensive (RouteIQ-f086) or dispatches by intent gets
its more-specific verdict rather than the broad one.

### Per-user routing from `/me/stats` (RouteIQ-2bbe)

The `personalized` verdict prefers RouteIQ's **caller-scoped `/me/stats`**
surface when it can read it. `/me/stats` returns ONLY the authenticated caller's
own `recent_models`, so the harness reads the authoritative per-user routing view
by GETting it once per synthetic user with that user's own data-plane token
(`RouteIQStatsClient.fetch_per_user_recent_models({user_id: token})`). When that
server-side view is present it drives the drift verdict; otherwise the verdict
falls back to grouping the client-observed responses by `user_id`. The verdict's
`findings["source"]` records which view was used.

### 4. Cross-strategy compare (RouteIQ-833c)

`compare.py` diffs two runs' model + per-category distributions head-to-head. The
operator hot-swaps the active strategy via the registry between two identical
workloads, then:

```python
from stress_harness import analysis, compare
cmp = compare.compare_runs(run_knn, run_bandit)
print(compare.comparison_markdown(cmp))
```

emits a table comparing share-by-model and per-category top model under each
strategy (and which spread wider) — judging two strategies on the *same* traffic.

## Optional CW Logs enrichment (RouteIQ-b0df)

`--enrich-cwlogs` joins RouteIQ's authoritative `routing_decision` log lines by
request id (per-request model + strategy + latency + cost), giving the per-request
`strategy_distribution` and the per-category-strategy counts. boto3 is imported
**lazily**, so the default body-only path needs neither boto3 nor AWS credentials.
The parser tolerates several emission forms (nested `routing_decision` block,
dotted keys, flattened Insights rows) and a missing `strategy` field.

## Testing

All cred-free, no live endpoint — every HTTP path flows through an
`httpx.MockTransport` and the CW-Logs enricher runs against an injected fake
client:

```bash
uv run pytest tests/stress_harness/ -q
```

Covers: workload allocation (incl. the 10k plan), the client request/response
capture, the per-strategy verdict dispatch (fan-out + consistency + cost-aware +
cost-cascade + semantic-intent + personalized + latency-cost) and the
unknown-strategy generic fallback, the stats client reading/merging/degrading
(incl. the per-user `/me/stats` reader), the cross-strategy compare, the CW-Logs
parser **and** its CLI wiring, and the full CLI pipeline (`--dry-run` with and
without `--base-url`, CW-Logs enrichment, plus a mocked end-to-end run).

## Companion tool: semantic-intent centroid calibration (RouteIQ-44a1)

`tools/centroid_calibration.py` is a **standalone** operator utility (it does not
import or modify the routing strategy class) that reports **inter-centroid cosine
separation** so an operator can catch overlapping intent centroids at config-load
time instead of in production. Two input modes:

```bash
# 1. report separation for the shipped *_centroid.npy vectors
uv run python tools/centroid_calibration.py --centroid-dir models/centroids

# 2. A/B a proposed intent taxonomy from operator exemplar phrases
uv run python tools/centroid_calibration.py --exemplars intents.json --json
```

`cosine_separation = 1 - cosine_similarity` (0 = identical direction =
un-discriminable, 1 = orthogonal, 2 = opposite). A pair below `--min-separation`
(default `0.10`) is flagged `WEAK` and the CLI exits non-zero so it can gate a
config-load check. The core math is pure-numpy and credential-free; the exemplar
mode lazily uses `sentence-transformers` only on that path.
