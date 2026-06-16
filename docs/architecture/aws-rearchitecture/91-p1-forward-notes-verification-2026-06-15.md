# 91 — P1 Forward-Notes Verification (RouteIQ-df6e)

> **Status**: Verification complete. **Date**: 2026-06-15.
> **Scope**: The three carry-forward notes raised in the P0 review for the P1 state
> wave, each verified against the SHIPPED state code (`7c57eba` + `7fda292` +
> convergence `80ef50e`) — not against the planning text.
> **Outcome**: 2 of 3 fully addressed; note (1) addressed-as-stated (the named
> mechanism needs no STS endpoint) but surfaced a narrower, genuinely-separate
> residual logged for a follow-up seed.

This is a verification + documentation pass. No data-plane source was edited; the
P1/state work shipped earlier already resolved the substance of all three notes.
Cited line anchors were read symbol-by-symbol against the worktree at verification
time.

## Note (1): "STS interface endpoint RETURNS at P1 for rds-db IAM-auth token issuance"

**Status: ADDRESSED (the premise is mechanically moot) + a narrower residual logged.**

The P0 `NetworkConstruct` deliberately DROPPED the STS interface endpoint on the Pod
Identity path (`deploy/cdk/lib/network_construct.py:16-18,:186` — STS was load-bearing
only for IRSA's `AssumeRoleWithWebIdentity`; Pod Identity creds come from the
pod-identity agent, not an STS endpoint call). The carry-forward asked whether STS must
RETURN at P1 because the new Aurora path uses rds-db IAM auth.

Verified against the shipped runtime: it does **not**. The rds-db:connect token is minted
by `generate_db_auth_token`, which is a **local SigV4 sign with no network call** —
`src/litellm_llmrouter/database.py:120-132` (`_mint_db_token`, docstring: "SigV4, local,
no network call"). The ElastiCache `elasticache:Connect` token is likewise a local SigV4
presign (`src/litellm_llmrouter/redis_pool.py`, `_mint_elasticache_token`). Neither IAM-auth
token path issues an STS API call, so neither requires an STS interface endpoint. The
shipped P1 state stack correctly adds NO STS endpoint and is correct to omit it. The note's
stated rationale rested on a misunderstanding of the token-issuance mechanics.

The shipped `RouteIqStateStack` provisions the actual P1 IAM additions the note's spirit
cared about — `rds-db:connect` on the runtime `routeiq` dbuser ARN and
`elasticache:Connect` on the cache + IAM-user ARNs — as ARN-scoped statements on the P0
pod role (`deploy/cdk/lib/routeiq_state_stack.py:281-313`). These grants, not an STS
endpoint, are what the rds-db / cache IAM-auth path needs, and they are present.

**Residual (narrower, genuinely separate — for a follow-up seed):** there IS one real
runtime `sts:AssumeRole` on the shipped path — the **cross-account Bedrock capacity**
assume (`_add_capacity_assume_grant`, `deploy/cdk/lib/routeiq_stack.py:324-363`), which
LiteLLM's `BaseAWSLLM` performs at runtime to borrow another account's Bedrock quota. From
the private-app (`PRIVATE_WITH_EGRESS`) subnet this call currently egresses via the NAT
gateway (it works), NOT via a private STS interface endpoint. This is flag-gated and
default-OFF (empty `routeiq:capacity_account_ids` => no grant, byte-stable), so it is not a
P1 blocker. If/when the cross-account capacity feature is enabled AND an operator wants the
STS assume to stay on the private substrate (no NAT egress), an `STS` interface endpoint
should be added to `NetworkConstruct._make_interface_endpoints` and gated on a
capacity-enabled context flag. That is the precise, narrowed form of what note (1) was
gesturing at — distinct from the (moot) rds-db token-issuance claim.

## Note (2): governance/quota fail-open on Redis loss → ElastiCache scale-to-zero opens a spend/rate window unless `*_FAIL_MODE=closed`

**Status: FULLY ADDRESSED.**

The fail-mode knob exists, is documented, and is honoured by the consumers:

- `GovernanceFailMode` + `QuotaFailMode` enums (both default `OPEN` for back-compat) —
  `src/litellm_llmrouter/settings.py:72-90`. `GovernanceSettings.fail_mode`
  (`settings.py:595-612`) documents that `CLOSED` denies when a budget/RPM limit IS
  configured but the spend store (ElastiCache/Aurora) cannot confirm usage.
- Both the canonical nested env `ROUTEIQ_GOVERNANCE__FAIL_MODE` AND the legacy flat
  `ROUTEIQ_GOVERNANCE_FAIL_MODE` select fail-closed — `governance.py:248-268`
  (`_governance_fail_mode_closed`).
- The budget check and rate-limit check both fail-closed ONLY when a limit is configured
  (a no-limit request is always allowed, so fail-closed never denies without a limit to
  enforce) — `governance.py:515-562` (budget) and `:568-600` (rate). Convergence commit
  `80ef50e` wired this fail-closed governance mode.
- The ElastiCache `elasticache:Connect` grant (so the pod can IAM-auth to the cache that
  backs the spend/rate counters) is present and ARN-scoped —
  `routeiq_state_stack.py:305-310` + `cache_construct.py:265` (`grant_iam_connect`).

So the scale-to-zero spend/rate window the note warned about is closeable by an operator
setting `ROUTEIQ_GOVERNANCE__FAIL_MODE=closed` (or the flat alias), and the default
ElastiCache provisioning is a **0.5-ACU warm floor, NOT scale-to-zero**
(`replay_store_construct.py:23-25,:235-242` for Aurora; the cache is serverless always-on),
with `min_acu=0` scale-to-zero being an explicit opt-in behind a context flag. The
fail-open default plus the warm floor is a deliberate, documented availability choice; the
fail-closed knob is the documented mitigation.

## Note (3): the migrate initContainer is P1-gated on `externalPostgresql.host`

**Status: FULLY ADDRESSED (wired).**

The `db-migrate` initContainer renders only when `migrations.enabled` AND
(`externalPostgresql.host` OR `secrets.values.DATABASE_URL`) is set —
`deploy/charts/routeiq-gateway/templates/deployment.yaml:42-64`. Although
`migrations.enabled` defaults to `true` (`values.yaml:939-940`), the effective gate is the
**host**: `externalPostgresql.host` defaults to `""` (`values.yaml:419-422`), so with no DB
wired the initContainer does NOT render (byte-stable, no-DB => no migrate). When P1 wires
the Aurora endpoint — exactly the `--set externalPostgresql.host=<DbClusterEndpoint>`
mapping the state stack's `DbClusterEndpoint` CfnOutput documents
(`routeiq_state_stack.py:373-399`) — the initContainer renders and runs
`run_migrations()`. The gating is on the P1 host as the note required.

## Verification summary

| Note | Premise | Shipped reality | Status |
|---|---|---|---|
| (1) STS endpoint for rds-db IAM | STS must return at P1 | rds-db / cache tokens are LOCAL SigV4 (no STS call); no STS endpoint needed | Addressed (moot) + narrow residual: cross-account capacity `sts:AssumeRole` egresses via NAT, not a private STS endpoint (flag-gated, default-OFF) |
| (2) gov/quota fail-open window | scale-to-zero opens a window unless `*_FAIL_MODE=closed` | `GovernanceFailMode`/`QuotaFailMode` exist + documented + consumer-honoured; cache default is warm-floor not scale-to-zero | Addressed |
| (3) migrate initContainer P1-gated | gated on `externalPostgresql.host` | gated on `migrations.enabled` AND (`externalPostgresql.host` OR `DATABASE_URL`); host default `""` => byte-stable | Addressed |

**Follow-up seed to file (residual from note 1):** "Optionally add an `STS` interface
endpoint to `NetworkConstruct`, gated on a capacity-enabled context flag, so the
cross-account Bedrock-capacity `sts:AssumeRole` stays on the private substrate instead of
egressing via NAT. P2-or-lower: flag-gated + default-OFF today, so not a P1 blocker."
