# Handoff — Pivot to RouteIQ-on-AWS (drop vllm-sr, build on LiteLLM)

> **Created:** 2026-06-14 12:11 PDT
> **From:** the `vllm-sr-on-aws` work stream (now wound down)
> **To:** active development in **this repo (`RouteIQ`)**
> **Repo HEAD at handoff:** `f09c937` — *docs: AWS-native re-architecture — deployment-pattern ADRs + Kumaraswamy-Thompson router + pluggable MLOps adapters*

---

## ▶ STARTER PROMPT (paste this to kick off the next session)

```
We're now working in the RouteIQ repo and dropping the vllm-sr-on-aws effort.
RouteIQ is already built on LiteLLM + LLMRouter — that IS our intelligently-routing
AI gateway. The vllm-sr AWS infra was torn down (greenfield) and its deployment
patterns are already documented here as ADRs 0026-0030 + docs/architecture/
aws-rearchitecture/. Our goal: make RouteIQ a production, HA, cloud-native
intelligently-routing AI gateway on AWS, and use LiteLLM's provider breadth +
translation to plug in extra features (the Responses-API bridge, MLOps, governance,
multi-account capacity).

Read docs/handoffs/2026-06-14-1211-pivot-to-routeiq-on-aws.md first, then
docs/architecture/aws-rearchitecture/30-migration-roadmap.md. Start Phase P0:
stand up the CDK foundation for RouteIQ on AWS (port the vllm-sr substrate
constructs cited in ADRs 0026-0030 into this repo's deploy/ — EKS Auto Mode + IRSA,
Aurora Serverless v2, ElastiCache Serverless, AppConfig, AMP/AMG observability).
Propose the P0 plan before writing CDK. Commit-only, never push. Co-Author trailer:
Claude Fable 5.
```

---

## Why the pivot makes sense (the one-paragraph thesis)

`vllm-sr-on-aws` was the effort to take **vllm-project/semantic-router** (a Go routing
app) and run it on an AWS-native substrate. Along the way we proved the *substrate* is
excellent and the *routing intelligence* is real — but RouteIQ **already is** the
"LiteLLM-in-front-of-a-router" synthesis (LiteLLM for provider breadth + format
translation, LLMRouter/UIUC for ML routing, plus governance + a React UI + ~9 native
strategies). The cleanest path to the same goal — *an intelligently-routing, HA,
cloud-native AI gateway* — is to **keep RouteIQ's layer and give it the AWS substrate
vllm-sr proved out**, rather than keep wrestling a Go app onto EKS. LiteLLM is the base
because it's what lets us "connect extra features" (it translates chat/messages↔Responses,
which even the Envoy AI Gateway architecturally cannot — that gap is documented).

**This is NOT abandoning the work — it's promoting the better layer.** The three projects
are different layers: LiteLLM = translation/breadth, vllm-sr/VSR = routing substrate
intelligence, RouteIQ = the synthesis. We're standing on the substrate lessons, building
on the LiteLLM-based gateway.

---

## What already landed in THIS repo (commit `f09c937`)

The vllm-sr deployment patterns are already ported here as design artifacts — **start here, don't re-derive:**

| File | What it gives you |
|---|---|
| `docs/adr/0026-appconfig-gitops-config-delivery.md` | AppConfig as the GitOps config plane (validator-gated rollout) over RouteIQ's S3+ETag hot-reload |
| `docs/adr/0027-otel-amp-amg-observability-on-aws.md` | AMP + AMG + CloudWatch mapped onto RouteIQ's existing `gen_ai.*` OTel contract |
| `docs/adr/0028-aurora-postgres-serverless-v2-state.md` | Aurora Serverless v2 for RouteIQ's `asyncpg`/`DATABASE_URL` state (scale-to-zero, IAM auth, KMS, 30d rotation) |
| `docs/adr/0029-elasticache-serverless-valkey-cache.md` | ElastiCache Serverless (Valkey) for cache + rate-limit + EMA state |
| `docs/adr/0030-eks-auto-mode-irsa-substrate.md` | EKS Auto Mode + IRSA — the substrate RouteIQ's Helm chart already expects to deploy onto |
| `docs/architecture/aws-rearchitecture/10-aws-native-target-architecture.md` | The full target: RouteIQ stateless control plane on EKS → Aurora/ElastiCache/AppConfig/AMP-AMG/data-lake |
| `docs/architecture/aws-rearchitecture/20-kumaraswamy-thompson-router.md` | Custom Thompson-sampling router w/ Kumaraswamy closed-form quantile sampling (math sympy-verified) — plugs into RouteIQ's `RoutingStrategy` ABC |
| `docs/architecture/aws-rearchitecture/30-migration-roadmap.md` | **The P0–P4 phased plan (~15–20 eng-weeks) — this is the work queue.** |
| `docs/architecture/aws-rearchitecture/40-pluggable-routing-and-mlops.md` | The pluggable routing-adapter framework + telemetry→train→sign→hot-reload MLOps loop |
| `docs/architecture/aws-rearchitecture/99-review-findings.md` | The adversarial review (3 lenses PASS, 0 CRITICAL) that verified the above |
| `docs/architecture/aws-rearchitecture/vllmsr-patterns.md` | The durable construct→pattern source map (which vllm-sr `cdk/lib/*.py` taught each pattern) |

Also relevant for orientation:
- `docs/architecture/four-way-comparison.md` + `vsr-vs-routeiq-decision-v3.md` — the LiteLLM vs LLMRouter vs RouteIQ vs VSR positioning.
- `../vllm-sr-on-aws/docs/architecture/routeiq-vs-vllmsr-aws-gap.md` — the cross-repo gap analysis that motivated this pivot.

---

## Preserved assets you can reuse (the vllm-sr infra is gone, the DATA is not)

The vllm-sr AWS infra was torn down 2026-06-14 (greenfield — 5 CFN stacks across 5
accounts, VPC clean). **The routing/MLOps corpus was preserved 3 ways** and is the seed
for RouteIQ's MLOps loop (doc 40):

- **RDS snapshot:** `vllmsr-replaystore-final-pre-routeiq-greenfield` (account `386931836011`, us-west-2, `available`) — full Aurora restore point.
- **Logical export (portable):** `pg_dump` SQL + an SFT-ready CSV of `router_replay_records` — **241 rows, 237 complete `prompt`+`completion`+`selected_model` tuples, 31 distinct models routed.**
  - Local: `../vllm-sr-on-aws/mlops-corpus-export/{router_replay_records.sql,sft_corpus.csv}`
  - Durable S3 (protected `-v2` bucket): `s3://vllm-sr-dev-386931836011-us-west-2-v2/mlops-corpus-export/pre-routeiq-greenfield-2026-06-14/`
- The `vllm-sr-on-aws` **repo/CDK is intact** — the 24 constructs in its `cdk/lib/` are the reference implementation to port from (cited per-pattern in the ADRs above).

---

## The work queue (from `30-migration-roadmap.md`)

| Phase | Scope | Est. | Type |
|---|---|---|---|
| **P0** | CDK foundation — port VPC/EKS Auto Mode/ECR/IRSA into this repo's `deploy/` | ~3–4w | port |
| **P1** | Externalized state — Aurora Serverless v2 + ElastiCache Serverless | ~2–3w | port |
| **P2** | Config + observability + data lake — AppConfig + AMP/AMG + Firehose | ~3w | port |
| **P3** | Routing + MLOps — Kumaraswamy-Thompson router + adapter framework + closed loop | ~4–6w | **build-new (RouteIQ's differentiation)** |
| **P4** | Governance / tenancy / UI hardening | ~3–4w | harden |

**Biggest risk:** P3 is the only net-new code (the rest is a tested CDK port). De-risk by building the router + MLOps loop against the preserved corpus before wiring it live.

---

## Environment & conventions (carry these into RouteIQ work)

**RouteIQ dev loop** (from `CLAUDE.md`):
```bash
uv sync --extra dev
uv run python -m litellm_llmrouter.startup --config config/config.yaml --port 4000
uv run pytest tests/unit/ -x                  # fast, no external deps
uv run ruff format src/ tests/ && uv run ruff check --fix src/ tests/
uv run mypy src/litellm_llmrouter/ --ignore-missing-imports
lefthook run pre-commit                       # ruff + yamllint + secret-scan (runs on commit)
```
- Source package: `src/litellm_llmrouter/`. Routing seam: the `RoutingStrategy` ABC + `strategy_registry.py`.
- **Always call it "RouteIQ." Do NOT rename `LITELLM_*` env vars. Do NOT claim RouteIQ implements features only inherited from upstream LiteLLM** (CLAUDE.md rules).
- This repo commits **directly to `main`** for docs (no feature branch for docs); pre-commit lefthook gate is green-required.

**Standing operational constraints** (from the vllm-sr work, still apply on AWS):
- AWS profile `baladita+Bedrock-Admin` (account `386931836011`, **us-west-2** — pass `--region` explicitly; the env default is us-east-1).
- **Commit only, never push.** Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Never log token/bearer values.** Never delete the `-v2` buckets.
- Deploy/cutover is **operator-gated** — produce committed, flag-gated IaC + runbooks; the operator runs the actual `cdk deploy`/`kubectl apply`.
- Fable 5 is **gov-banned** as a routable arm (do not add it to live routing config).

**Useful skills for this work** (auto-surface, but worth knowing):
- `preserve-state-before-greenfield-teardown` — the safe teardown/preserve sequence (used for the corpus above).
- `cfn-stack-teardown-rollback-recovery` — recovery if a future teardown wedges.
- `strategic-replatform-deliberation` — produced the gap analysis that grounds this pivot.

---

## First three concrete actions for the next session

1. **Read** this handoff → `30-migration-roadmap.md` → skim ADRs 0026–0030.
2. **Decide the CDK home**: this repo has `deploy/` (Helm chart today). Confirm whether the AWS CDK app lives in `deploy/cdk/` here, or whether we vendor `../vllm-sr-on-aws/cdk/lib/` constructs in. (Recommendation: new `deploy/cdk/` in this repo, porting the cited constructs — keeps RouteIQ self-contained.)
3. **Propose the P0 plan** (CDK foundation) before writing code — VPC + EKS Auto Mode + ECR + IRSA, parameterized for RouteIQ's stateless pods. Then build.

---

*Handoff authored at the close of the vllm-sr-on-aws stream. The substrate is proven, the
patterns are documented here, the corpus is preserved. RouteIQ is the gateway; LiteLLM is
the base that connects the extra features. Build forward.*
