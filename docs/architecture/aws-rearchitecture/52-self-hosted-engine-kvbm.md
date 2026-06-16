# 52 — Self-Hosted Engine (C3b) + KVBM Remote KV-Cache Tiers: the cred-free RouteIQ wiring

> **Status**: Implementation note for the cred-free half of the C3b + KVBM seeds.
> **Date**: 2026-06-15.
> **Seeds**: `RouteIQ-91fe` (C3b: first self-hosted engine as ONE `api_base`),
> `RouteIQ-28be` (Dynamo KVBM remote KV-cache tiers — single-node + S3 G4, EFA-free).
> **Authorities** (do not re-adjudicate here):
> [`50-litellm-universal-surface.md`](./50-litellm-universal-surface.md) §4 (the C3b
> engine verdict + the "one `api_base` per model" rule) and
> [`51-multinode-large-model-serving.md`](./51-multinode-large-model-serving.md) Part 2
> (KVBM tiers + the write-through sizing trap) and Part 3 (how it composes with RouteIQ).

This note covers what RouteIQ ships **cred-free** (config + a render test that proves the
chart seam) and what is **operator-gated** (the live GPU deploy). It deliberately does
**not** build the live engine — that needs real GPU hardware, a real cluster, and (for
KVBM G4) a real S3 bucket + IAM, all of which are the operator half.

---

## 1. The split — cred-free vs operator-gated

| Half | What | In this repo? |
|---|---|---|
| **Cred-free (RouteIQ side)** | The `model_list` row that points RouteIQ at ONE engine `api_base`; a chart render test proving the values→ConfigMap→`config.yaml` seam carries that row unchanged; this doc. | **Yes** — `config/config.self-hosted-engine.yaml` + `deploy/charts/routeiq-gateway/tests/test_render_self_hosted_engine.py`. |
| **Operator-gated (live deploy)** | The GPU NodePool (C3a), the engine itself (vLLM Production Stack / AIBrix / Dynamo) on GPU nodes, the Dynamo operator + `DynamoGraphDeployment` CRD + etcd + NATS for KVBM, the sized DRAM/NVMe tiers, the real S3 G4 bucket + IAM. | **No** — documented below as steps the operator runs against a live cluster. |

This mirrors the pattern already shipped this session for the native Bedrock Guardrail
construct (authoring-only, zero live consumer) and the GPU NodePool: build the cred-free,
default-off, byte-stable-when-off half with a cred-free test; leave the live deploy to the
operator half.

---

## 2. Cred-free deliverable — register the engine as ONE `api_base`

`config/config.self-hosted-engine.yaml` shows the one load-bearing `model_list` row:

```yaml
model_list:
  - model_name: oss-70b
    litellm_params:
      model: hosted_vllm/meta-llama/Llama-3.1-70B-Instruct
      api_base: http://aibrix-gateway.aibrix-system.svc.cluster.local:8000/v1
      api_key: fake-api-key   # hosted_vllm sentinel; the in-cluster engine is unauthenticated
```

The `provider hosted_vllm` (or `openai_like`) consumes the engine's OpenAI-compatible
`/v1` endpoint directly via `api_base` with no translation shim (`50-...` §4.2). The
`api_base` is the engine **frontend/gateway Service** — the KV-aware router (Production
Stack router, AIBrix Router, Dynamo Smart Router + FlashIndexer) lives **below** it.

**The one config error to avoid** (`51-...` §3.2, `50-...` §4.2): never register
individual workers / replicas / prefill+decode pods as separate `model_list` rows. That
collapses Layer-1 model-selection into Layer-2 replica-scheduling and makes RouteIQ fight
the engine's own cache-aware router. RouteIQ targets exactly ONE `api_base` per model.

To deploy this config via the chart, the operator pastes the `model_list` block into
`config.gateway` (the chart lands it verbatim in the gateway ConfigMap — see
`templates/configmap.yaml` → `.Values.config.gateway | nindent`). The render test
`test_render_self_hosted_engine.py` proves that seam: a self-hosted row passes through
values → ConfigMap `config.yaml` unchanged, and the default chart render still ships
**zero** self-hosted arms (byte-stable when off).

---

## 3. Cred-free deliverable — KVBM env knobs documented at the engine

KVBM (`51-...` Part 2) is a **Layer-2-and-below** concern that lives entirely inside the
ONE `api_base` above. RouteIQ config is **unchanged** whether or not KVBM is enabled — the
`DYN_KVBM_*` knobs are env on the **Dynamo engine** deployment, never on RouteIQ. The two
EFA-free shapes that land at C3b on the single-node engine:

1. **Single-node aggregated `G1 → G2 → G3`** (no EFA): `DynamoConnector`,
   `kv_role: kv_both`, set `DYN_KVBM_CPU_CACHE_GB` (G2) and `DYN_KVBM_DISK_CACHE_GB` (G3,
   NVMe node).
2. **Remote G4 object tier** (no EFA, standard TCP to S3/MinIO): `DYN_KVBM_OBJECT_ENABLED`,
   `DYN_KVBM_OBJECT_BUCKET`, `DYN_KVBM_OBJECT_REGION` (+ `DYN_KVBM_OBJECT_ENDPOINT` for
   MinIO). `ObjectLockManager` (conditional PUT `If-None-Match: *`) makes the cold tier
   multi-replica-safe.

The disaggregated prefill/decode shape (cross-node, EFA/RDMA) is **out of scope** — it is
the C3-deep EFA seed (`51-...` Part 1).

**Write-through sizing trap** (the single most important KVBM rule, `51-...` §2.2): each
tier must be **≥** the tier above it, or KVBM churns (offloads after every forward pass
for no benefit and can degrade performance):

```
DYN_KVBM_CPU_CACHE_GB  >= GPU KV-cache size       (G2 >= G1)
DYN_KVBM_DISK_CACHE_GB >= DYN_KVBM_CPU_CACHE_GB    (G3 >= G2)
```

The illustrative `DYN_KVBM_*` env block is in `config/config.self-hosted-engine.yaml` (in
the KVBM comment section). The actual tier sizes are **operator-sized to the real GPU KV
footprint**, which needs the live GPU — hence partial, not full, for the live sizing.

---

## 4. Operator-gated — the live deploy steps (OUT of scope here)

These run against a live cluster with GPU hardware and AWS credentials. None of it is in
this repo; it is the operator half.

1. **C3a GPU NodePool** — the custom Auto Mode GPU `NodePool` + `NodeClass` (already a
   separate seed) so a `nvidia.com/gpu` pod can actually schedule.
2. **Deploy ONE engine** on the GPU nodes — vLLM Production Stack *or* AIBrix per the
   `50-...` §4.3 verdict (both EKS-Auto-Mode-clean; AIBrix is most production-proven,
   Production Stack is the smallest step up with KV-cache-aware routing). Note the
   frontend Service DNS — that becomes the `api_base`.
3. **(KVBM) Dynamo path** — install the **Dynamo operator** + the `DynamoGraphDeployment`
   CRD (`apiVersion: nvidia.com/v1alpha1`) + **etcd + NATS** (required for leader/worker
   registration + the block-lifecycle event plane). Apply `agg_kvbm.yaml` for the
   single-node aggregated shape; add the `DYN_KVBM_OBJECT_*` env + an S3 bucket + IAM
   (Pod Identity with `s3:GetObject`/`PutObject` on the G4 bucket) for the remote tier.
   Size the tiers per the write-through rule in §3 against the **measured** GPU KV
   footprint. G3 wants a p5/p6 node with local NVMe + `--use-nixl-gds`.
4. **Wire RouteIQ** — paste the `model_list` row from §2 into the chart's `config.gateway`
   (or `config/config.self-hosted-engine.yaml`) with the real frontend Service `api_base`,
   redeploy the gateway. RouteIQ now routes Layer-1 to the engine; KVBM tiers are opaque
   below the one `api_base`.

**Acceptance carried by the live deploy** (`51-...` §4.3): one `api_base` per model (the
cred-free render test already guards this seam); write-through tier sizing
(`CPU >= GPU_KV`, `DISK >= CPU`); and — only for the cross-node EFA shape, which is the
separate C3-deep seed — the NIXL **LIBFABRIC backend** asserted (the silent ~98s-vs-~1s
TTFT trap).
