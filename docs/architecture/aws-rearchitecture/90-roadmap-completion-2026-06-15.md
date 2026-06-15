# 90 — RouteIQ-on-AWS Roadmap Completion + Honest Deferred-Set

> **Status**: Roadmap complete. **Date**: 2026-06-15.
> **Outcome**: All 5 roadmap epics (P0–P4) merged + closed; zero blocking (P0/P1) work
> remains. The residual backlog is catalogued below with explicit justification per the
> "never defer without justification" standard.

## What shipped (the P0–P4 roadmap, all merged to `main`, never pushed)

The migration roadmap (`30-migration-roadmap.md`) is fully executed via the tiered
deep-work loop (research → plan → build → gate → adversarial review → reconcile), each
wave in an isolated git worktree with a rebase + squash-merge flow, every gate verified
by a re-run on `main` (disk-is-truth, not the workflow's word).

| Epic | Commit(s) | Deliverable | Gate (verified) |
|---|---|---|---|
| **P0** | `3a90096` | `deploy/cdk/ RouteIqStack` — VPC/SGs/endpoints, EKS Auto Mode L1, **EKS Pod Identity (not IRSA)**, ECR + GHCR pull-through, cdk-nag | CDK 36 |
| **P2** | `496c242` + `51d42ee` | `RouteIqObservabilityStack` — AppConfig (LAMBDA validator), AMP + flag-gated AMG + CW filters/alarms/SNS, Firehose→Parquet→Glue→Athena data-lake, structured `routing_decision` log line; P2-hardening (13 seeds) | CDK 209 |
| **P1** | `7c57eba` + `7fda292` | `RouteIqStateStack` — Aurora PG Serverless v2 + ElastiCache Serverless (Valkey), IAM auth, KMS, 30d rotation, schema-bootstrap; bootstrap-SG + IAM-token runtime | CDK 181, gw 325 |
| **P3** | `506fb89` | **Kumaraswamy-Thompson bandit** (one strategy) + **strategy-agnostic adapter/MLOps framework** (`train_mode ∈ {one_time, continuous}`), offline-de-risked vs the 241-row corpus | unit 647, prop 8 |
| **P4** | `08341e7` | governance/usage_policies → Aurora (additive, DB-optional), OIDC/UI edge (ALB/ACM/ESO chart wiring), NetworkPolicy egress hardening | CDK 228, gw 648 |
| **convergence** | `80ef50e` | wired the 2 P1 "built-but-not-wired" bugs (error-emitter live caller; spend write==read scope so workspace/key budgets enforce) + fail-closed governance mode | gw 566 |
| **cleanup** | `1ffeb51` | registry cold-process deadlock (RLock), OIDC test authlib-gate, Redis IAM token refresh, spend scope-helper dedup | oidc 94, full 3620 |

**Final gate on `main`**: `uv run pytest deploy/cdk/tests/` = 228 passed (cred-free synth +
cdk-nag, no AWS creds); `uv run pytest tests/unit/` = 3620 passed / 31 skipped / 0 failed;
`helm lint` = 0 failures. 28 commits ahead of `origin/main`, **never pushed** (operator-gated).

## Convergence outcome

**0 open P0 · 0 open P1** — no blocking-class work remains. The original roadmap-backlog is
complete. The seed tracker grew to ~70 issues because each wave's *adversarial review team*
(auditing the squash-merged snapshot, not live worktrees) found real follow-ups — that is
the system working. Severity drained monotonically wave-over-wave; what remains is
P2-or-lower.

## The honest deferred-set (residual open seeds — WHY each is not done this run)

Per the standing constraints (operator-gated deploy, cred-free, commit-only, no live AWS),
and the goal's "seeds outside the roadmap goal are logged out-of-scope, not auto-executed",
the residual seeds fall into three categories. **None is blocking; none is a roadmap epic.**

### A. Deploy-gated — cannot be done without a live AWS edge / cluster (operator-gated)
- `RouteIQ-4f59` **WAF construct** — attaches to a live public ALB-by-ARN that does not render
  at P0 (service is `ClusterIP`, ingress disabled, `alb_sg` PREP-ONLY). Un-wireable until a
  LoadBalancer edge exists.
- `RouteIQ-e9ab` **EKS Auto Mode IngressClassParams edge runbook** — documents the live
  Auto-Mode ALB path; needs a deployed cluster to validate.
- `RouteIQ-7069` **Fluent-Bit promotion live verification** — the config ships + helm-renders,
  but proving stdout-JSON promotion requires sampling a live routing log group after traffic
  (skill `eks-container-insights-fluentbit-wraps-stdout`). Mechanism done; verification is
  deploy-time-only.

### B. Net-new feature enhancements — beyond the roadmap's "build the substrate" goal
- `RouteIQ-f9e9` **Kumaraswamy moment-fit (doc-40 option-2)** — the bandit ships doc-20's
  `a=α,b=β` default; the review produced the eval that *triggers* the documented Newton
  moment-fit upgrade. A genuine enhancement (corrects exploit-decision distortion), not a
  blocker — the shipped default is the documented choice.
- `RouteIQ-1669` **AppConfig GitOps pipeline + deployer + audit** — ADR-0026's full
  CodePipeline delivery path (the construct delivers config; the *pipeline* around it is the
  enhancement).
- `RouteIQ-0716` **OIDC exchanged-key shared store** — multi-pod key consistency (currently
  in-process bounded dicts; `replicaCount=2` makes it sticky-session-dependent). Real
  multi-pod hardening, discrete data-plane work.
- `RouteIQ-c0be`/`9f14` native Bedrock Guardrail stage + activation; `RouteIQ-c2af` guardrail
  CRUD durability — all optional, flag-gated, beyond the P4 roadmap row.

### C. Low polish / docs / tech-debt (non-blocking, batched-or-deferred)
- `RouteIQ-1f1b`/`bd6e` ADR-0026–0030 IRSA→PodID doc rewrite (the amendment header suffices).
- `RouteIQ-d3fb`/`9738`-class constant/helper dedups; `RouteIQ-df6e` P1 forward-notes
  (largely satisfied by the fail-closed mode shipped in convergence).
- ~20 P3 follow-ups (backend wiring docstrings, adapter loader registration, snapshot
  coverage extensions) — incremental hardening, none blocking.

## Termination rationale

This is the convergence-stop-signal (`iterative-review-loop-convergence-stop-signal`), and
stopping here is a **success of the control system**: the roadmap is complete, all blocking
work is zero, and every residual is either (A) forbidden by the operator-gated/cred-free
constraints, (B) a net-new enhancement beyond the roadmap goal, or (C) low polish. Running
more waves would force out-of-scope work or gold-plate beyond the ask. The deferred-set is
catalogued (seed tracker, label-filtered) so the operator can pick up any item by priority.

**Next operator actions** (gated, not auto-run): `cdk bootstrap` + `cdk deploy` the stacks,
`helm upgrade` with the CfnOutput `--set` seams, then the deploy-gated verifications (A) and
the enhancement seeds (B) become live-actionable.
