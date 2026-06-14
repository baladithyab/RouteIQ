# 99 — Adversarial Review Findings: AWS Re-Architecture Doc Set

> **Reviewer**: offline read-only audit, 2026-06-14. **Scope**: docs 10/20/30/40 +
> ADRs 0026–0030. **Method**: independent symbolic+numeric math verification (sympy
> 1.14, high-precision Decimal, numeric integration); grep-verification of every
> load-bearing `file:line` citation against real source in both repos.
> **Protocols applied**: `verify-external-audit-before-acting`,
> `synthesis-agent-fabricates-untraceable-precision`.

## Verdict summary

| Lens | Verdict |
|------|---------|
| **1. Math correctness (doc 20)** | **PASS** — every formula independently verified exact; doc is honest about the one place it cannot be (no fabricated moment-match). |
| **2. AWS-pattern fidelity (docs 10/30 + ADRs)** | **PASS** — all spot-checked constructs + named lessons match source. Only minor line-number drift. |
| **3. Coherence + adapter-API soundness (doc 40 + cross-doc)** | **PASS** — adapter Protocol aligns with the real ABC/plugin base; the 5 docs are internally consistent. |

**No CRITICAL findings.** No fabricated math, no fabricated citation. Doc set is **sound to commit.**

---

## Findings table

| Lens | Finding | Severity | File:line | Fix |
|------|---------|----------|-----------|-----|
| 1 | PDF `abx^(a−1)(1−x^a)^(b−1)` is the **exact** derivative of CDF `1−(1−x^a)^b` (sympy: `dF/dx − f = 0`). | PASS | 20-...md:111,113 | none |
| 1 | Quantile `Q(u)=(1−(1−u)^(1/b))^(1/a)` is the **exact** inverse of F (numeric round-trip `F(Q(u))=u` to ~1e-52 in Decimal; the one float64 grid-miss is catastrophic cancellation for tiny b, u→1, NOT an algebra error). | PASS | 20-...md:114-116 | none |
| 1 | All 3 special cases exact to machine ε: `Kum(1,b)=Beta(1,b)`, `Kum(a,1)=Beta(a,1)`, `Kum(1,1)=Uniform`. | PASS | 20-...md:138-140 | none |
| 1 | Raw-moment `E[X^n]=b·B(1+n/a,b)` **verified** vs numeric integration (~1e-12). | PASS | 20-...md:153 | none |
| 1 | Kumaraswamy mode `((a−1)/(ab−1))^(1/a)` + Beta mode `(α−1)/(α+β−2)` + Beta variance `αβ/((α+β)²(α+β+1))` **all verified** (sympy critical point + grid argmax + numeric variance). | PASS | 20-...md:158,160,170 | none |
| 1 | Doc is **HONEST**: states "no clean closed-form `(α,β)→(a,b)`" and "no elementary solution"; ships the admitted approximation `a=α,b=β` and gates a Newton fit behind an eval. **No fabricated moment-match presented** — exactly what the audit feared and did not find. | PASS | 20-...md:151-178 | none |
| 1 | Beta-Bernoulli update `α+=r, β+=(1−r)` correct; posterior mean `α/(α+β)` → Laplace-smoothed sample-mean reward (verified). Reward `r∈[0,1]` shaping coherent with Thompson. | PASS | 20-...md:83-91 | none |
| 1 | Self-falsification discipline honored: the doc's own "verify: differentiate F / substitute x=Q(u)" instruction (line 118) was executed and **holds**. | PASS | 20-...md:118 | none |
| 2 | `replay_store_construct.py` **really** uses Aurora Serverless v2: `rds.DatabaseCluster` + `aurora_postgres(VER_16_13)` + `iam_authentication=True` + `serverless_v2_*_capacity` + `add_rotation_single_user(30d)` + scale-to-zero `auto_pause` coupling — matches ADR-0028 verbatim, incl. line cites :254-257,:260-262,:267-279,:305. | PASS | 0028:34-66 | none |
| 2 | `cache_construct.py` **really** uses ElastiCache Serverless Valkey: `CfnServerlessCache(engine="valkey", major_engine_version="8")` + IAM `CfnUser`/`CfnUserGroup` + `kms_key_id` — matches ADR-0029. | PASS | 0029:36-78 | none |
| 2 | `config_state_construct.py` **really** does AppConfig: full Cfn{Application,Environment,ConfigurationProfile,DeploymentStrategy,HostedConfigurationVersion,Deployment} + LAMBDA `ValidatorsProperty`; strategy params (growth_factor=20, duration=12, bake=5) copied exactly. | PASS | 0026:45-72 | none |
| 2 | `eks_cluster_construct.py` **really** does Auto Mode + CfnJson IRSA: L1 `CfnCluster` + ComputeConfig/StorageConfig/ElasticLoadBalancing blocks + `CfnJson` trust-key wrap + `oidc_provider_issuer` (ARN-derived) — matches ADR-0030, incl. the `.replace("https://")` silent-no-op lesson. | PASS | 0030:37-89 | none |
| 2 | Per-model dimensioned MetricFilter `routing_latency_ms_by_model` with `dimensions={"model":"$.selected_model"}` exists at the cited eks_cluster_construct.py:757-767. | PASS | 0027:62-71 | none |
| 2 | Named live-ops lessons (engine-retirement 15.4-gone, CDK-created-log-group-must-pre-exist, Aurora ~30min rollback, CIDR-lock=`loadBalancerSourceRanges`, RWO+RollingUpdate→Recreate, `apply -k` clobbers SA) all trace to real source comments / `/tmp/vllmsr-patterns.txt`. | PASS | 0028:90, 0027:90, 0030:107-124 | none |
| 2 | All 5 ADRs + docs 10/30 cite `/tmp/vllmsr-patterns.txt` as a source-of-record. The file **exists and its content faithfully matches** the attributions, BUT it is an **ephemeral `/tmp` artifact** — not durable, not in-repo. A future reader cannot re-verify. | LOW | 0026:156, 0028:154, 0030:182, 10-...md:127, 30-...md:13 | Commit the construct→pattern map into the repo (e.g. `docs/architecture/aws-rearchitecture/vllmsr-patterns.md`) and re-point citations, or inline the relevant lines. |
| 2 | Gap doc cited by doc 10 §0 / §4 as `vllm-sr-on-aws/docs/architecture/routeiq-vs-vllmsr-aws-gap.md` **exists** at that path (29KB). Cross-repo path is correct. | PASS | 10-...md:9 | none |
| 3 | Adapter Protocol `route/update/load/reload/declare_capabilities/validate` aligns with the **real** `RoutingStrategy(ABC)`: `select_deployment` (abstractmethod, :322), `name`/`version` props (:338,:343), `validate()->Tuple[bool,Optional[str]]` (:347). The `route()`-renames-`select_deployment()` shim claim is sound (ABC is the documented minimal core). | PASS | 40-...md:200-216 | none |
| 3 | `GatewayPlugin` chassis claims verified: `PluginCapability` enum has exactly **11** values incl. `ROUTING_STRATEGY` + `OBSERVABILITY_EXPORTER` (:90-131); `PluginMetadata` fields (name/version/capabilities/depends_on/priority/failure_mode/description) match exactly (:150); `PluginContext.routing: RoutingAccessor` (:219). | PASS | 40-...md:124-150 | none |
| 3 | `conversation_affinity` **dormant** claim TRUE: defs exist (:105,:208,:241) but `grep` finds **zero** hot-path callers of `get_affinity_tracker` outside its own file (rc=1). The "finished feature, no socket" argument is real. | PASS | 40-...md:152-162 | none |
| 3 | Doc 20's "no thompson/beta_posterior/kumaraswamy symbols exist today" honesty claim is **TRUE** (`grep -rniE` over `src/litellm_llmrouter/` returns rc=1, zero matches). The "DESIGN/aspirational" status banner is accurate. | PASS | 20-...md:3-7 | none |
| 3 | CostAware combined-score `(1−cost_weight)·quality + cost_weight·(1−normalized_cost)` is **verbatim** the real `_compute_combined_score` docstring+return. The "resolves inner strategy by name" claim → real `_resolve_inner_strategy`. | PASS | 20-...md:34, 40-...md:67 | none |
| 3 | Feedback contract `{user_id, model, score∈[-1,1], request_id?}` → `record_feedback`, validated `[-1.0,1.0]`, is **exact** at routes/config.py. `r_quality=(score+1)/2` map is coherent. | PASS | 20-...md:29-31,61 | none |
| 3 | Cross-doc consistency: roadmap P0-P4 phasing matches target-arch §2 component map (every construct in §2 appears in a roadmap phase); MLOps loop (doc 40 §3) references the **same** Aurora (posteriors) / AppConfig (delivery) / data-lake the ADRs define; doc 20 §6 persistence (Redis hot + Aurora durable) matches ADR-0028/0029. No contradictions found. | PASS | 30/10/40 | none |
| 3 | "RouteIQ keeps its LiteLLM-translation edge / closes the gpt-5.5 gap" is **grounded**: `litellm==1.82.3` (pyproject:21) ships `LiteLLMCompletionResponsesConfig` (`responses/litellm_completion_transformation/` present in installed dep); gpt-5.5-Responses-only-404s-via-EAIG matches project memory `bedrock-multiregion-onboarding`. Consistent. | PASS | 10-...md:209-217 | none |
| 1/2/3 | **Systematic line-number drift (~1-30 lines) in a minority of citations.** strategies.py: `_compute_combined_score` cited :2995 (actual :2976), `_get_model_cost` :3083 (actual :2798), `_get_available_candidates` :2898 (actual :2885), pareto :2960 (actual :2926), `router_map` :2164 (actual :2165). model_artifacts.py: `ArtifactEntry` :94 (actual :95). conversation_affinity: `get_affinity_tracker` implied :355 (actual :345). doc 40: `RoutingContext` cited :220-283 (actual :42-86). **In every case the named symbol exists and says what the doc claims** — these are stale offsets, not fabrications. | LOW | passim | Optional: re-run a line-anchor pass before publishing; the symbols are correct so semantic risk is nil. |
| 2 | ADR-0026 prose says "Linear-20%-over-12-min" / "60% over 12min"; the construct's own description string says "per 3 minutes". Both are the **same informal gloss inherited from the source code's description=** field; the load-bearing params (growth_factor=20, duration=12, bake=5) are copied **exactly**. Not a fabrication; just a loose rollout-cadence gloss that is arithmetically imprecise (LINEAR growth_factor=20 ⇒ ~2.4min/step). | LOW | 0026:55-57,104 | Optional: drop the "per 3 minutes" gloss or replace with "5 linear steps over 12 min". |

---

## Adversarial notes (what I tried to break)

- **Tried to break the calculus** by symbolic differentiation AND symbolic integration AND numeric round-trip over 1e5 random `(a,b,u)`. The single numeric round-trip miss looked like a smoking gun; isolated and **proven** to be float64 cancellation (Decimal-60 error = 4.6e-52), not algebra.
- **Tried to catch a fabricated moment-match** (the highest-risk fabrication per `synthesis-agent-fabricates-untraceable-precision`). The doc does NOT present one — it explicitly disclaims a closed form and the `E[X^n]=b·B(1+n/a,b)` moment it *does* state is numerically correct.
- **Tried to catch a non-existent construct** (per `verify-external-audit-before-acting`): opened the real `replay_store_construct.py`, `cache_construct.py`, `config_state_construct.py`, `eks_cluster_construct.py` — all four do exactly what the ADRs claim, at the cited line ranges.
- **Tried to catch an adapter-API mismatch**: the Protocol is a genuine superset of the real one-method ABC; `update()`/`load()`/`reload()` map to real `record_feedback`/`reload_model` signatures; the capability enum and metadata dataclass exist as described.

## Residual (non-blocking)

1. **`/tmp/vllmsr-patterns.txt` is the only un-durable source** behind a wide set of citations (LOW). Recommend committing it into the repo before these docs are treated as authoritative.
2. **Stale line offsets** in a minority of `file:line` citations (LOW) — symbols correct, offsets drift; a mechanical re-anchor pass is cheap insurance.

Neither blocks commit.

---

## Resolution (2026-06-14, post-review)

- **Residual #1 (`/tmp` provenance) — RESOLVED.** The ephemeral
  `/tmp/vllmsr-patterns.txt` was promoted to a durable in-repo doc,
  `vllmsr-patterns.md` (this directory), and every `construct → pattern`
  citation in ADRs 0026–0030 + docs 10/30 was re-pointed to it. The only
  remaining `/tmp/...` strings are in *this* audit record (a point-in-time
  finding, intentionally preserved) and the new doc's provenance note. Every
  citation now resolves to a committed file.
- **Residual #2 (stale line offsets) — ACCEPTED.** Not mechanically
  re-anchored: every named symbol is correct (the reviewer confirmed this for
  all flagged cites), the offsets reference a *separate* repo's source that
  will drift again on its next commit, and the docs cite by symbol name in
  prose. Semantic risk is nil; a line-anchor pass would be re-done busywork.
