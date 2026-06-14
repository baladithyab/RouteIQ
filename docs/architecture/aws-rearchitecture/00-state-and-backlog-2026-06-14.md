# 00 — Commit State & Backlog Audit (RouteIQ-on-AWS)

> **Status**: Living checkpoint. **Date**: 2026-06-14.
> **Purpose**: PHASE 1 (commit-state documentation) + PHASE 2 (backlog audit) of the
> deep-work-loop driving the RouteIQ-on-AWS pivot. This is the durable record of *where
> the repo is* and *what work remains* at the moment the AWS substrate build began.
> Companion to the canonical handoff (`docs/handoffs/2026-06-14-1211-pivot-to-routeiq-on-aws.md`)
> and the roadmap (`30-migration-roadmap.md`).

---

## 1. Commit state (PHASE 1)

### 1.1 HEAD and history

At checkpoint, `main` is **3 commits ahead of `origin/main`** (all unpushed — and they
stay that way; **commit-only, never push** is a standing constraint):

| SHA | Date (PDT) | Subject |
|---|---|---|
| `249201d` | 2026-06-14 12:44 | docs: expand the RouteIQ-on-AWS handoff into a self-contained working doc |
| `c48b39f` | 2026-06-14 12:13 | docs: handoff — pivot to RouteIQ-on-AWS (drop vllm-sr, build on LiteLLM) |
| `f09c937` | 2026-06-14 10:04 | docs: AWS-native re-architecture — deployment-pattern ADRs + Kumaraswamy-Thompson router + pluggable MLOps adapters |

These three commits landed the **design substrate** for the pivot: ADRs 0026–0030, the
`docs/architecture/aws-rearchitecture/` doc set (10/20/30/40/99 + `vllmsr-patterns.md`),
and the self-contained handoff. **No source or IaC has changed yet** — the pivot is, to
date, entirely a documentation/design effort. The build starts here.

### 1.2 Working-tree state at checkpoint

```
 M reference/LLMRouter      (submodule pointer move — DO NOT stage)
 M reference/litellm        (submodule pointer move — DO NOT stage)
?? .coverage               (local pytest-cov SQLite DB — now gitignored)
?? ha-gate-report.json     (failed local HA-gate finch-compose run — now gitignored)
```

- **`reference/LLMRouter` / `reference/litellm`** are READ-ONLY git submodules
  (CLAUDE.md hard rule). The `M` is a *pointer* difference (upstream advanced:
  LLMRouter → `bench/xroutebench`, litellm → `v1.81.3-stable.opus-4-6`), **not** a dirty
  working tree inside them. We never modify or stage `reference/`. The pinned app
  dependency remains `litellm==1.82.3` (`pyproject.toml`), independent of the submodule
  pointer.
- **`.coverage`** — SQLite coverage DB from a local `pytest --cov` run. Ephemeral,
  regenerated each run → added to `.gitignore`.
- **`ha-gate-report.json`** — output of a local HA failover gate that failed at the
  container-build step (`finch compose ... up -d --build` exited non-zero; not a code
  regression — a local tooling/build-env issue). Ephemeral run artifact → added to
  `.gitignore`.

**Net**: the only *intentional* tree changes this session are documentation + tracker
state + `.gitignore` hygiene. The two untracked files are now ignored rather than
committed (neither is a source artifact).

### 1.3 Rationale / context for the checkpoint

The `vllm-sr-on-aws` (VSR) effort was **wound down** — its AWS infra torn down to
greenfield, its repo + CDK kept intact as the **port reference**
(`../vllm-sr-on-aws/cdk/`), and its MLOps corpus preserved (RDS snapshot +
`pg_dump`/CSV; 241 rows / 237 SFT tuples / 31 models, in the protected `-v2` S3 bucket —
**never delete**). The pivot thesis (handoff §2): **RouteIQ already *is* the
"LiteLLM-in-front-of-an-ML-router" synthesis** VSR was converging toward, so the cleanest
path to an HA, cloud-native, intelligently-routing AI gateway is to **give RouteIQ the
AWS substrate VSR proved out**, not keep wrestling a Go app onto EKS. VSR's deployment
patterns are already ported here as ADRs 0026–0030.

---

## 2. Backlog audit (PHASE 2)

### 2.1 The trackers were stale — reconciled this checkpoint

Three os-eco trackers exist in-repo. At checkpoint they recorded only the **completed
v0.2.0 implementation phase**, not the forward AWS work:

| Tracker | State found | Action taken |
|---|---|---|
| **beads** (`BEADS.md`, no live `bd` CLI / empty backing store) | 7 task-groups `TG-IMPL-A`…`G`, **all Done** (Feb 2026): P0 critical fixes, plugin-arch migration, NadirClaw centroid routing, Admin UI MVP, cloud-native hardening, doc cleanup, codebase reduction. | **Migrated → seeds** as 7 *closed* issues (history preserved with close-reasons + the dependency notes). |
| **seeds** (`.seeds/`, live `sd` CLI) | 5 issues, **all closed** (Mar 2026 analysis tasks). | Kept; **forward P0–P4 backlog filed fresh** (below). |
| **mulch** (`.mulch/`, live `ml` CLI) | No domains configured. | Will record learnings on completion (os-eco RECORD step). |
| **overstory** (`.overstory/`) | Present (agent-defs: builder/coordinator/lead/merger/monitor/reviewer/scout/orchestrator), not running. | Available for multi-agent execution if needed. |

**Key finding**: the *real* backlog — the AWS re-architecture — had **zero live tracker
entries**. The roadmap (`30-migration-roadmap.md`) was the only record. This checkpoint
files it into seeds so os-eco has actionable work to `sd ready`.

### 2.2 The forward backlog — P0→P4 (now filed as seed epics)

Critical path: **P0 → {P1 ∥ P2} → P3 → P4** (~15–20 eng-weeks). Filed as `epic` seeds
labelled `aws-rearch`, dependency-wired so `sd ready` surfaces only unblocked work
(currently **P0 only**).

| Phase | Seed | Scope (one line) | Type | Risk | Est. |
|---|---|---|---|---|---|
| **P0** | `RouteIQ-5a85` | CDK foundation: VPC/SGs/endpoints + EKS Auto Mode + pod-auth factory + ECR/GHCR pull-through + cdk-nag → `deploy/cdk/ RouteIqStack`. | epic | Low (PORT) | ~3–4w |
| **P1** | `RouteIQ-6078` | Externalized state: Aurora PG Serverless v2 + ElastiCache Serverless (Valkey); wire chart `externalPostgresql`/`externalRedis` + ESO. | epic | Low–Med (PORT) | ~2–3w |
| **P2** | `RouteIQ-5a72` | Config + Obs + Data Lake: AppConfig + AMP/AMG/CW + Firehose; wire AppConfig poll + OTel emit + `routing_decision` log line. | epic | Low–Med (PORT) | ~3w |
| **P3** | `RouteIQ-3282` | Routing + MLOps: Kumaraswamy-Thompson bandit + adapter framework + closed MLOps loop on the preserved corpus. | epic | **Med–High (NET-NEW)** | ~4–6w |
| **P4** | `RouteIQ-0354` | Governance / multi-tenant / UI hardening on the substrate: govern+policies→Aurora, OIDC+UI behind ALB/ACM, NetworkPolicy + WAF/Guardrail. | epic | Med (HARDEN) | ~3–4w |

Dependency edges filed: `P1→P0`, `P2→P0`, `P3→P1`, `P3→P2`, `P4→P3`.

### 2.3 Categorization

**By priority**: P0 (Critical, blocks everything) → P1/P2 (High, parallelizable after P0)
→ P3/P4 (Medium; P3 is the differentiation, P4 the productionization).

**By type**:
- **PORT (low risk, tested constructs)**: P0, P1, P2 — VSR's constructs carry 446 tests,
  live across two substrates. The work is *adapt + parameterize for RouteIQ's single
  stateless pod*, plus small app-side wiring.
- **BUILD-NEW (the risk)**: P3 — Kumaraswamy-Thompson is a net-new algorithm; the closed
  MLOps loop is RouteIQ-built. De-risked by building against the **preserved corpus
  offline** before wiring live.
- **HARDEN (framework-on-substrate)**: P4 — the code exists (`governance.py`, `oidc.py`,
  React UI); the work is wiring to Aurora/ESO/ALB + multi-tenant ops.

**By complexity / dependency**: P0 is the gate. P1 and P2 are independent of each other
(can overlap) but both gate P3. P4 is last.

### 2.4 Granular P0 child seeds — deferred to research

P0's *child* tasks (one per VSR construct port + stack wiring + tests) are intentionally
**not hand-authored here**. Exactly which constructs, with which acceptance criteria, and
the load-bearing **IRSA-via-CfnJson vs EKS Pod Identity** decision, are the questions the
research+plan phases of the deep-work-loop answer authoritatively. They are filed as a
batch from `research/p0/child-seeds.json` once the P0 proposal
(`31-p0-cdk-foundation-proposal.md`) is reviewed and accepted. This honors the handoff's
non-negotiable: **propose the P0 plan before writing any CDK.**

---

## 3. Method note — how this backlog gets worked

Per the user directive and the `deep-work-loop-tiered` skill, the AWS work runs as a
**tiered deep-work loop** (Frame → Discover → Research → Plan → Act → Review, with
bounded backflow), with model tiers chosen by blast radius (Opus for comprehension/
implementation, the hyperresearch pipeline for external research, Fable solo for the
scale-setting Plan/Verdict), a **concurrent multi-lens critique panel** in the Review
phase, and **os-eco discipline** (`sd ready` → in_progress → close; `ml record` on
learnings). Deploy/cutover stays **operator-gated**: this work produces committed,
flag-gated IaC + CDK tests + runbooks + proposals; the operator runs
`cdk bootstrap`/`cdk deploy`/`kubectl`/`helm`.

The cred-free verification gate that makes operator-gating tractable: CDK constructs are
validated entirely in-process via `aws_cdk.assertions.Template.from_stack()` + `cdk-nag`
Aspects under `pytest` — **no AWS account, no CLI, no creds** — so a PR can prove template
structure + policy-as-code without ever touching AWS.
