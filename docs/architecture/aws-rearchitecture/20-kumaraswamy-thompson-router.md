# Kumaraswamy-Thompson Model Router — Algorithm Design

> Status: **DESIGN / aspirational.** This algorithm does not exist in RouteIQ's code today
> (confirmed by grep — no `thompson`, `beta_posterior`, or `kumaraswamy` symbols in
> `src/litellm_llmrouter/`). This document specifies a new `RoutingStrategy` subclass,
> grounds it in the real install seam, and is honest about where the math is exact vs.
> approximate. A reviewer can and should check every CDF/quantile statement below.

## 0. Scope and the real seam

RouteIQ already has a routing plug-in architecture; this design adds **one new strategy**, it
does not rewrite anything. The seam, top to bottom:

- **Install point:** `RouteIQRoutingStrategy(CustomRoutingStrategyBase)` is the object LiteLLM
  delegates to via `router.set_custom_routing_strategy(...)`
  (`src/litellm_llmrouter/custom_routing_strategy.py:140`, installed at
  `:886` in `install_routeiq_strategy`). Its `_route_via_pipeline`
  (`:429`) builds a `RoutingContext` and calls `pipeline.route(context)`.
- **Strategy contract:** every routing algorithm is a `RoutingStrategy(ABC)` implementing
  `select_deployment(self, context: RoutingContext) -> Optional[Dict]`
  (`src/litellm_llmrouter/strategy_registry.py:313`, abstract method at `:321`).
- **Registry + A/B + staged rollout:** `RoutingStrategyRegistry`
  (`strategy_registry.py:493`) holds named strategies, supports `register()` (`:564`),
  `set_active()` (`:645`), weighted A/B via `set_weights()` (`:676`) with deterministic
  `sha256(hash_key) % total_weight` bucketing (`_select_weighted`, `:946`), and
  `stage_strategy()` / `promote_staged()` (`:777` / `:830`) for canary promotion.
- **Closest existing analog (online feedback):** `personalized_routing.py` —
  `PreferenceStore.update_preference` (`:346`) does an **EMA** update of a per-user vector
  from a `score ∈ [-1, 1]` fed by `POST /api/v1/routeiq/routing/feedback`
  (`routes/config.py:903`, contract `{user_id, model, score, request_id?}`, validated to
  `[-1.0, 1.0]` at `:973`, dispatched to `record_feedback` at `:982`).
- **Closest existing analog (cost-aware native strategy):** `CostAwareRoutingStrategy`
  (`strategies.py:2669`), whose combined score is
  `(1 - cost_weight)·quality + cost_weight·(1 - normalized_cost)`
  (`_compute_combined_score`, `:2995`), plus a Pareto-frontier filter (`:2960`) and
  open-circuit-breaker exclusion (`_get_available_candidates`, `:2898`).

The new strategy is `KumaraswamyThompsonStrategy`, registered as
`"routeiq-kumaraswamy-thompson"`, taking its place beside the existing nine registry entries
(see §7). It **reuses the feedback endpoint shape verbatim** and **borrows the cost-blend
formula** from `CostAwareRoutingStrategy`.

---

## 1. Problem: model selection as a multi-armed bandit

Each routing decision is a contextual multi-armed bandit:

- **Arms** = the candidate deployments for the requested model group. In RouteIQ these are the
  rows of `healthy_deployments` whose `model_name` matches the request
  (`_get_model_list`, `custom_routing_strategy.py:738`); the arm key is the
  `litellm_params.model` string (e.g. `bedrock/global.anthropic.claude-haiku-4-5`,
  `bedrock/deepseek.v3.2-techu`). The arm set is **dynamic** — providers come and go, and
  circuit breakers remove arms (`strategies.py:2898`).
- **Context** = a coarse **task bucket** `b` (see §5) so the bandit is contextual without a
  full neural policy. We learn one posterior **per `(bucket, arm)`** pair, exactly the
  granularity LiteLLM's `adaptive_router` uses and the granularity VSR's `rl_driven` keys on
  `decision_name`.
- **Reward** = a single normalized scalar `r ∈ [0, 1]`, higher is better, defined in §4 as a
  blend of response **quality**, **cost**, and **latency**. The `[-1, 1]` feedback `score`
  from the existing endpoint maps onto this with `r_quality = (score + 1) / 2`.

The objective is to minimize cumulative regret: route to the arm with the highest expected
reward most of the time, while exploring enough to discover when a cheaper/newer arm has
become competitive. Pure argmax over a static quality table (what `_DEFAULT_QUALITY_BIASES`,
`personalized_routing.py:58`, gives at cold start) never explores; pure EMA
(`update_preference`) tracks a point estimate but carries **no uncertainty**, so it cannot
distinguish "this arm is reliably mediocre" from "we have only tried this arm twice." Thompson
sampling solves both by sampling from the **posterior** and picking the argmax of the samples —
uncertainty automatically drives exploration.

---

## 2. Why Beta posteriors (the standard Thompson baseline)

Model the reward of arm `i` as Bernoulli with unknown success probability `θ_i`
(a continuous reward in `[0,1]` is handled by Bernoulli-with-fractional-reward; see §4). The
**Beta distribution is the conjugate prior** for the Bernoulli/Binomial likelihood, which is
the entire reason it is the textbook Thompson choice:

- Prior: `θ_i ~ Beta(α_i, β_i)`, start `α_i = β_i = 1` (uniform on `[0,1]`).
- Likelihood of one reward `r` treated as a Bernoulli trial with success-mass `r`.
- **Posterior update (the rule):** on observing reward `r ∈ [0,1]` for arm `i`,
  ```
  α_i ← α_i + r
  β_i ← β_i + (1 − r)
  ```
  This is exact for `r ∈ {0,1}` and the standard "fractional reward" extension for
  `r ∈ [0,1]` (it keeps the posterior mean `α/(α+β)` an unbiased running estimate of mean
  reward and shrinks variance as evidence accrues). Conjugacy means the posterior stays Beta
  with no integral to evaluate.

**Standard Thompson step:** for each candidate arm `i`, draw `θ̃_i ~ Beta(α_i, β_i)`, then
select `argmax_i θ̃_i`.

The cost of the baseline is the **sampling step**. Drawing from a Beta requires either
`numpy.random.beta` / `scipy.stats.beta.rvs` (which internally use the Cheng / Johnk
rejection algorithms — variable-iteration, branch-heavy) or two Gamma draws with
`Beta = G(α) / (G(α) + G(β))`. There is **no closed-form inverse-CDF for the Beta** (the
regularized incomplete beta function `I_x(α,β)` must be inverted numerically). For a router on
the request hot path with N arms per decision, that is N rejection-sampled draws per request.
That latency tax is what motivates the Kumaraswamy substitution.

---

## 3. The Kumaraswamy approximation

The **Kumaraswamy distribution** `Kumaraswamy(a, b)` on `(0,1)` with shape parameters
`a, b > 0` has:

- **PDF:** `f(x; a, b) = a·b·x^(a−1)·(1 − x^a)^(b−1)`, for `x ∈ (0,1)`.
- **CDF (closed form):** `F(x; a, b) = 1 − (1 − x^a)^b`.
- **Quantile / inverse-CDF (closed form):** invert `F`. Set `u = 1 − (1 − x^a)^b` and solve:
  ```
  Q(u) = ( 1 − (1 − u)^(1/b) )^(1/a)
  ```

These three are exact and elementary (verify: differentiate `F` to recover `f`; substitute
`x = Q(u)` into `F` to recover `u`). This is the **entire point**: because `Q` is a closed-form
expression in `u`, sampling is **inverse-transform sampling**:

```
u ~ Uniform(0,1)
x = ( 1 − (1 − u)^(1/b) )^(1/a)
```

One uniform draw, two `pow` calls, no special functions, no rejection loop, constant time and
branch-free. That is the cheap-sampling property the Beta lacks.

### 3.1 When `Kumaraswamy(a,b) ≈ Beta(α,β)` — and where it diverges

The two families are genuinely similar in shape: both are unimodal/J-shaped/U-shaped on `(0,1)`
depending on parameters, and `Kumaraswamy(1,1) = Beta(1,1) = Uniform(0,1)` **exactly**. They
are **not** equal in general — Kumaraswamy is an *approximation* of Beta, and any claim
otherwise is false. The honest statement of the relationship:

**Exact special cases.**
- `Kumaraswamy(1, b) = Beta(1, b)` exactly (both have CDF `1−(1−x)^b`).
- `Kumaraswamy(a, 1) = Beta(a, 1)` exactly (both have CDF `x^a`).
- `Kumaraswamy(1, 1) = Beta(1, 1) = Uniform`.

For general `(α, β)` the two differ, **most visibly in the tails** — the Beta's tail behavior
near 0 and 1 is governed by the exponents `α−1` and `β−1` on `x` and `(1−x)`, while
Kumaraswamy's right-tail is governed by `(1−x^a)^(b−1)`, which decays differently. The
divergence is largest for large `α, β` (sharply peaked posteriors) and in the extreme
quantiles. For Thompson sampling this matters **less than it looks**: we only need the *ordering*
of sampled draws across arms to be approximately right, and the bulk/mode of the distribution —
not the 0.001 tail — dominates which arm wins the argmax. Tail error mostly perturbs the
exploration rate slightly, not the exploit decision.

**Moment-matching map (the practical fit).** There is **no clean closed-form `(α,β) → (a,b)`**;
the Kumaraswamy moments are
`E[X^n] = b·B(1 + n/a, b)` (with `B` the beta function), so matching mean and variance to a
Beta requires solving a small 2-equation nonlinear system in `(a, b)`. Be honest: that system
has no elementary solution. Two practical options, in order of preference:

1. **Match the mode and a spread proxy (recommended, cheap).** A Beta posterior with
   `α, β > 1` has mode `m = (α−1)/(α+β−2)`. The Kumaraswamy mode is
   `m = ((a−1)/(ab−1))^(1/a)`. Rather than solve this, use the **direct shape heuristic that
   works well in practice for the J/peaked posteriors a bandit produces:** set `a = α` and
   `b = β`. This is *not* moment-matched, but `Kumaraswamy(α, β)` is a known, well-behaved
   stand-in that (i) is exact on the three special cases above, (ii) has the same monotonic
   response to evidence (more successes pushes mass right, more failures pushes mass left),
   and (iii) preserves the `α=β=1` uniform prior. Because Thompson only needs
   *uncertainty-aware ordering*, `a=α, b=β` is the pragmatic default and is what this design
   ships.
2. **One-time numerical fit per arm (optional, exact-ish).** If a future eval shows the
   `a=α,b=β` shortcut materially mis-explores, precompute `(a,b)` from `(α,β)` by a 5-iteration
   Newton solve matching mean `α/(α+β)` and variance `αβ/((α+β)²(α+β+1))` to the Kumaraswamy
   moments, cached in the arm state and refreshed only when `α+β` crosses log-spaced
   thresholds. This keeps the hot path at one uniform draw + the quantile while moving the fit
   cost off the request path.

**Verdict on the math:** the CDF and quantile in §3 are exact and the cheap-sampling claim is
real. The Beta-equivalence claim is **only exact on the three special cases**; everywhere else
Kumaraswamy is a shape-similar approximation whose error lives in the tails. We adopt the
`a=α, b=β` shortcut as the default and gate the optional Newton fit behind an eval result, so we
never ship a fabricated moment-match formula.

---

## 4. Why it matters HERE (latency + reward shaping)

**Latency.** RouteIQ routing runs synchronously on the request path; `RoutingPipeline.route`
already records `latency_ms` per decision (`strategy_registry.py:1213`) and centroid routing
advertises a ~2ms budget (`custom_routing_strategy.py:158`). With N candidate arms per
decision, Beta-Thompson costs N rejection-sampled / Gamma-ratio draws; Kumaraswamy-Thompson
costs N × (one `random()` + two `pow`). No `scipy.stats.beta.rvs`, no `numpy.random.beta`
rejection loop — just arithmetic. For the 13-arm decisions VSR runs (see the `modelRefs` list,
`k8s/eaig/20-router-config.yaml:217`), this is the difference between a vectorizable arithmetic
kernel and N calls into a branchy sampler. The sampling math needs **no numpy at all**, which
matters because `personalized_routing.py` already degrades gracefully when numpy is absent
(`:44`); this strategy can run on the pure-Python path.

**Reward shaping (cost-aware, borrowed from VSR + `CostAwareRoutingStrategy`).** The Bernoulli
reward `r ∈ [0,1]` per arm is a blend, mirroring VSR's `cost_awareness` /
`cost_reward_alpha` and RouteIQ's own `_compute_combined_score`:

```
r = w_q · quality        # quality in [0,1]; from feedback score or LLM-judge (§8)
  + w_c · (1 − norm_cost) # cheaper = higher reward; norm_cost in [0,1] across the arm set
  + w_l · (1 − norm_lat)  # faster = higher reward; optional, w_l may be 0
        # with w_q + w_c + w_l = 1
```

- `norm_cost` reuses `CostAwareRoutingStrategy._get_model_cost` and the min-max normalization
  at `strategies.py:3083`. `(1 − norm_cost)` is exactly VSR's "cheaper-is-better" term.
- The blend weights map onto VSR config: `w_c ≈ cost_weight` (VSR uses `0.4`,
  `20-router-config.yaml:236`). VSR's `cost_reward_alpha: 0.5` (`:237`) is the *mixing rate of
  the cost term into the reward signal*; we expose it as a separate scalar `cost_reward_alpha`
  that scales how strongly cost enters `r` before the Beta update, so the same lever exists.
- Crucially the reward is computed **only when feedback arrives** (offline-update, §5), so the
  hot path pays nothing for reward shaping — it only samples.

---

## 5. Online update + cold start

**Update path (reuses the existing feedback endpoint verbatim).** The endpoint
`POST /api/v1/routeiq/routing/feedback` (`routes/config.py:903`) already accepts
`{user_id, model, score ∈ [-1,1], request_id?}` and calls `record_feedback`. We extend the
dispatch so that, when the Kumaraswamy-Thompson strategy is active, the same call also updates
the bandit:

```
on feedback(model, score, request_id):
    bucket = recover_bucket(request_id)         # the task bucket logged at decision time
    r_quality = (score + 1) / 2                 # map [-1,1] -> [0,1]
    r = shape_reward(r_quality, model, bucket)  # blend with cost/latency (§4)
    α[bucket, model] += cost_reward_alpha_weighted(r)        # = r in the simple case
    β[bucket, model] += (1 − r)
    # a=α, b=β by default (§3.1), or refresh the cached Newton fit
```

This is the *same* online-feedback shape as `PreferenceStore.update_preference`
(`personalized_routing.py:346`) — one scalar score in, an in-place state update out — except the
state is `(α, β)` counts instead of a 128-d EMA vector, and it carries uncertainty. Temporal
non-stationarity is handled the way `update_preference` already does it: a **decay** that
multiplies both `α` and `β` by `γ^(days_elapsed)` toward the prior (mirroring
`ROUTEIQ_PREFERENCE_DECAY=0.99`, `personalized_routing.py:400`), so a model that was good last
quarter doesn't dominate forever. Decay-on-Beta keeps the mean but re-inflates the variance,
which correctly *re-opens exploration* for stale arms.

**Per-(bucket, model) keying.** State is keyed `(task_bucket, model)`, matching LiteLLM
`adaptive_router`'s bucketed Beta-Bernoulli and VSR's per-`decision_name` posteriors. The
`request_id → bucket` mapping is logged at decision time into the router-decision telemetry
event (`strategy_registry.py:1368`) so feedback can recover the bucket without re-classifying.

**Cold start (the centroid classifier does the tiering).** A brand-new `(bucket, model)` pair
starts at `Beta(1,1)` — maximally uncertain, so Thompson explores it aggressively, which is the
desired behavior but can be wasteful. We warm-start two ways, both already present in the
codebase:

1. **Prior from the static quality table.** Seed `α = 1 + κ·q₀`, `β = 1 + κ·(1−q₀)` where
   `q₀` is the model's `_DEFAULT_QUALITY_BIASES` value (`personalized_routing.py:58`) and `κ`
   is a small pseudo-count (e.g. 5). This biases a cold arm toward its known prior quality
   without preventing exploration.
2. **Centroid classifier for the bucket itself.** The task bucket `b` is produced by the
   existing `CentroidRoutingStrategy` (`custom_routing_strategy.py:534`, the ~2ms zero-config
   classifier). The centroid label (its `RoutingProfile`/complexity tier) *is* the bucket key,
   so cold-start tiering is free — the same classifier that already routes when no ML model is
   loaded supplies the bandit's context dimension. New arms inherit the bucket's existing
   exploration history rather than starting blind.

---

## 6. Integration: the exact RouteIQ seam

A new file `src/litellm_llmrouter/kumaraswamy_thompson.py` defining:

```python
class KumaraswamyThompsonStrategy(RoutingStrategy):       # strategy_registry.py:313 ABC
    name = "routeiq-kumaraswamy-thompson"
    version = "v1"

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        cands = self._candidates(context)                 # reuse _get_model_list pattern
        cands = self._drop_open_breakers(cands)            # reuse strategies.py:2898
        bucket = self._bucket(context)                     # centroid tier (§5)
        best, best_draw = None, -1.0
        for dep in cands:
            a, b = self._state.get(bucket, model_of(dep))  # α,β -> a,b (§3.1 default a=α,b=β)
            u = random.random()
            x = (1 - (1 - u) ** (1.0 / b)) ** (1.0 / a)    # closed-form quantile (§3)
            if x > best_draw:
                best_draw, best = x, dep
        self._log_bucket(context.request_id, bucket)        # for feedback recovery (§5)
        return best

    def validate(self) -> tuple[bool, str | None]:          # strategy_registry.py:347
        return (self._state is not None), None
```

**Wiring (no new install seam):**
- Register at startup beside the others:
  `get_routing_registry().register("routeiq-kumaraswamy-thompson", KumaraswamyThompsonStrategy())`
  (same call site shape as `register_centroid_strategy`, `custom_routing_strategy.py:941`).
- It is selected through the **existing** `RouteIQRoutingStrategy._route_via_pipeline`
  (`custom_routing_strategy.py:429`) → `RoutingPipeline.route` → `registry.select_strategy`.
  **Zero changes** to `CustomRoutingStrategyBase` wiring — it is just another registry entry,
  exactly like `CostAwareRoutingStrategy`.
- Activate via `set_active("routeiq-kumaraswamy-thompson")` or A/B it via
  `set_weights({"llmrouter-nadirclaw-centroid": 90, "routeiq-kumaraswamy-thompson": 10})`
  (`strategy_registry.py:676`) — sticky per-user/request hashing comes for free (`:261`).
- Canary it with `stage_strategy(..., auto_promote=False)` then `promote_staged()` after the
  eval gate (§8) passes (`:777` / `:830`).

**State persistence (per the new ADRs — Aurora/Redis).** The `(α, β)` table is small
(`#buckets × #arms × 2 floats`) but must survive worker restarts (the in-memory-only failure
mode is exactly the `# no storage_path -> posteriors are in-memory (reset on restart)` caveat
VSR notes at `20-router-config.yaml:239`). Two tiers, mirroring `PreferenceStore`'s
"Redis + local cache, graceful fallback" design (`personalized_routing.py:266`):

- **Hot tier — Redis** (`redis_pool.get_async_redis_client`, already used at
  `personalized_routing.py:287`): key `routeiq:kts:{bucket}:{model}` → packed
  `(α, β, last_update)`. Atomic `HINCRBYFLOAT` on update; read-through local cache for the
  draw. This is the per-request store.
- **Durable tier — Aurora** (the new replay/feedback store the ADRs introduce): a
  `bandit_posteriors(bucket, model, alpha, beta, updated_at)` table, written async on the
  feedback path (not the route path). On cold boot, hydrate Redis from Aurora. This is the
  analog of VSR persisting posteriors to its EBS PVC, but cloud-native and shared across
  workers — solving the multi-worker drift the in-memory version would have.

---

## 7. Comparison table

It slots in as the **10th** registry strategy beside the existing nine
(`baseline`/`DefaultStrategy`, `llmrouter-knn`, `-svm`, `-mlp`, `-mf`, `-elo`,
`llmrouter-nadirclaw-centroid`, `llmrouter-cost-aware`, and the personalized re-ranker).

| Dimension | **Kumaraswamy-Thompson (this design)** | **VSR `rl_driven` Thompson** (`20-router-config.yaml:233`) | **LiteLLM `adaptive_router`** (Beta-Bernoulli, bucketed) |
|---|---|---|---|
| Posterior family | **Kumaraswamy(a,b)** as Beta stand-in | True **Beta(α,β)** | True **Beta(α,β)** |
| Sampling cost | **1 uniform + 2 `pow`, closed-form quantile, no numpy** | Beta rejection/Gamma-ratio draw | Beta draw |
| Reward | quality + cost + latency blend, `[0,1]` (§4) | `cost_awareness`, `cost_weight=0.4`, `cost_reward_alpha=0.5`, `exploration_rate=0.3` | quality (success/fail), bucketed |
| Context / keying | per-`(centroid-bucket, model)` | per-`decision_name` (config-defined node) | per-task-bucket |
| Cold start | static quality-bias prior + centroid tier | uniform prior, `exploration_rate` floor | uniform prior |
| State persistence | **Redis (hot) + Aurora (durable), shared across workers** | EBS PVC `storage_path` (in-memory if unset) | in-process |
| Install | new `RoutingStrategy` in the registry, A/B + staged | EAIG ext_proc node config (YAML) | LiteLLM router config |
| **Novel here** | closed-form-quantile sampling on the hot path; cloud-native shared posterior store; uncertainty-aware where RouteIQ's EMA personalizer is not | — | — |
| **Borrowed** | cost blend from `CostAwareRoutingStrategy:2995` + VSR `cost_weight`/`cost_reward_alpha`; feedback endpoint + decay from `personalized_routing.py`; Beta-Bernoulli + bucketing from `adaptive_router`; conjugate-Thompson from VSR | conjugate Thompson, cost-aware reward | conjugate Thompson, bucketing |

What is genuinely **novel**: (a) substituting the closed-form-quantile Kumaraswamy for the
Beta to kill rejection-sampling latency on a per-request, N-arm hot path; (b) a shared
Redis+Aurora posterior store that fixes the multi-worker / restart-amnesia failure both VSR
(in-memory unless `storage_path`) and the in-process LiteLLM variant have; (c) adding
*uncertainty-aware* exploration to RouteIQ, which today only has argmax (centroid/cost-aware)
and point-estimate EMA (personalized). Everything else is deliberately borrowed.

---

## 8. Validation plan

Validate **without a code rewrite** using machinery that already exists.

**A/B harness (registry).** Register the new strategy and run it as the treatment arm against
the current production strategy (centroid or cost-aware) via `set_weights`
(`strategy_registry.py:676`), e.g. `{"control": 90, "treatment": 10}`. Assignment is sticky
per-user (`get_ab_hash_key`, `:261`) and every decision already emits the
`routeiq.router_decision` telemetry event with strategy name, variant, candidates, selected
model, and latency (`strategy_registry.py:1368`) — so the experiment is observable from day one
with no new instrumentation.

**Reward labeling (LLM-judge eval loop).** Quality labels feed the bandit through the existing
eval pipeline (`eval_pipeline.py:359` `push_feedback`, surfaced at
`/api/v1/routeiq/eval/push-feedback`, `routes/config.py:1949`): an LLM judge scores sampled
responses, the score is pushed to the same `[-1,1]` feedback contract, and that drives both the
control's EMA and the treatment's `(α,β)` updates. **Honest caveat:** an LLM judge can reward
confident fabrication over honest uncertainty; the eval set must include adversarial/uncertain
prompts and a small human-graded holdout so the reward signal the bandit optimizes is the one
we actually want.

**Metrics that prove it beats argmax / EMA.** Headline metric: **cumulative reward per
1k requests** (the cost-shaped `r`, §4) — the treatment must beat control with the
`cost_weight` held identical so we are comparing *selection*, not reward definition. Secondary:
(1) **regret vs. an oracle** that always picks the empirically-best arm per bucket — Thompson
should converge to near-zero regret while a static-argmax control plateaus at fixed regret;
(2) **exploration coverage** — fraction of `(bucket, model)` pairs with `≥ k` observations
(Thompson should cover more than EMA, which never deliberately explores); (3) **recovery time**
— after deliberately degrading one arm's reward, time-to-reroute (the decay + variance
re-inflation should re-explore faster than EMA's slow drift); (4) **p50/p99 routing latency** —
must confirm Kumaraswamy's closed-form draw is at or below the Beta-Thompson and centroid
budgets. **Pre-spend gate:** before burning eval budget, verify the A/B flag actually moves the
selected-model distribution (control vs. treatment must differ on the *measured* metric) —
otherwise the experiment is uninterpretable.

**Promotion.** Only after the treatment wins cumulative-reward at fixed cost-weight, holds the
latency budget, and shows lower oracle-regret does it go from `stage_strategy` → `promote_staged`
(`:830`) to active.

---

## 9. Decision: defer the `moment_fit` default-flip to a canary (RouteIQ-c299, 2026-06-15)

§3.1 ships the option-1 `a=alpha, b=beta` shortcut as the default and gates the option-2
moment-fit behind an eval/observation result. RouteIQ-f9e9 implemented that option-2 fit in
`kumaraswamy_thompson.py` (`fit_kumaraswamy_moments`, `Posterior.shape(moment_fit=...)`) behind
the setting `ROUTEIQ_KUMARASWAMY_THOMPSON__MOMENT_FIT` (`settings.py`
`KumaraswamyThompsonSettings.moment_fit`), **default `False`**. This note records the decision
on whether to flip that default ON now.

**The math is verified correct and safe.** The moment-fit's `fit_kumaraswamy_moments`:

- preserves the three exact special cases (`Kuma(1,b)=Beta(1,b)`, `Kuma(a,1)=Beta(a,1)`,
  `Kuma(1,1)=Uniform`) as byte-stable corners (`kumaraswamy_thompson.py:301`);
- restores the posterior **mean** to within ~1e-8 of the Beta mean (vs the shortcut's gross
  distortion — e.g. `Beta(51,51)` mean `0.5` → shortcut Kumaraswamy mean `0.9155`, which can
  *invert the exploit decision*);
- tracks the **variance** at ~1.0x the Beta variance across the feasible region (verified at
  `Beta(2,2)`, `(8,8)`, `(51,51)`, `(100,200)`, `(1000,3)`: ratio `1.000`); for an extremely
  sharply-peaked symmetric posterior past the holdable-`a` floor the infeasible branch returns
  the cap and the variance runs slightly high (`Beta(500,500)` → ~1.67x), where high variance =
  slightly more exploration with **the mean held exact** (`d ~ 3.7e-7`) —
  `kumaraswamy_thompson.py:337-340`. The mean-fidelity (which drives the exploit decision) is
  preserved in every case; only the exploration rate is perturbed at the floor;
- is **off the hot path**: the fit is cached on the exact `(alpha, beta)` (RouteIQ-f9e9 defect-1
  fix; a log-spaced bucket key served stale fits within a bucket), recomputed only on an actual
  evidence change — ~30 `lgamma`/bisection calls per miss, and arms update at most once per
  request (`kumaraswamy_thompson.py:385-397`).

So the change is **mathematically a strict improvement** to the sampled-mean fidelity, and the
default-off posture exists purely for byte-stable backward-compat, not because the fit is wrong.

**Why the flip is DEFERRED, not done here.** Flipping the default ON is a **live bandit
behavior change**: it shifts the sampled `(a,b)` for every non-corner posterior, which shifts
the Thompson draw distribution and therefore the **arm-selection distribution** in production.
That is exactly the class of change that must be observed on real traffic before it becomes the
default, and that observation **cannot be run cred-free** (no live cluster / traffic here). This
is an explicit, justified deferral — not a skip.

**Acceptance gate for the flip (run on a canary with a live cluster).** Stage the moment-fit as
the treatment arm (it is a per-instance flag, so A/B it via the existing registry weights / a
canary deploy with `MOMENT_FIT=true` on the treatment replicas) and confirm, over a real
traffic window:

1. **Arm-selection distribution** under moment-fit vs the option-1 shortcut differs in the
   expected direction (the corrected mean should pull selection toward the empirically-better
   arm), and **cumulative regret** is **<=** the shortcut's at fixed cost-weight (§8's oracle-
   regret + cumulative-reward metrics) — i.e. it does not *worsen* selection.
2. **No latency regression**: the ~2.3ms cache-miss fit at production arm counts stays within
   the p50/p99 routing-latency budget (§8 secondary metric 4). Since the fit is cached per
   `(alpha,beta)` and runs at most once per feedback update, the steady-state hot path should be
   a cache hit; verify the cold-fit cost does not breach the budget at peak arm counts.

**On acceptance**, flip the code default (`KumaraswamyThompsonSettings.moment_fit` →
`True`), and update the "Default off for byte-stable backward-compat" note in `settings.py` +
`Posterior.shape`'s docstring to record that the default is now the moment-fit and the option-1
shortcut is the opt-out (the byte-stable-corner guarantee is unchanged either way). Until then
the default stays `False`.
