# 51 — Multi-Node Serving of Very Large Models: The C3-Deep Tier Below the Universal Surface

> **Status**: Architecture / forward roadmap. **Date**: 2026-06-15.
> **Thesis**: Serving a model **too big to fit one node** — and the remote KV-cache
> tiering that makes long context affordable — is a **Layer-2-and-below** concern. It
> lives *entirely inside one OpenAI `api_base`*: a Dynamo / vLLM **multi-node serving
> group**. RouteIQ still does **Layer-1 model selection** above it, unchanged. This is the
> **scale-gated C3-deep tier**, reached only *after* the C3a/C3b common path
> (single-node GPU NodePool + first engine) in
> [`50-litellm-universal-surface.md`](./50-litellm-universal-surface.md).
>
> **Honesty contract.** Claims are tagged **[VERIFIED]** (grounded in the research
> reports under `research/efa-multinode/`, which read NVIDIA `ai-dynamo/dynamo` repo docs
> directly via `gh` + DeepWiki + AWS EKS docs, with file paths cited) or **[SPECULATIVE]**
> (reasoned synthesis / a design we have not yet built). The **engine + EFA + KVBM**
> facts are owned by the research reports; this doc does not re-adjudicate them.
>
> **Sources.**
> - `research/efa-multinode/research-dynamo-kvbm.md` — the combined EFA + KVBM research
>   report (NVIDIA Dynamo repo docs, DeepWiki, AWS EKS EFA docs; all findings cited to
>   file paths). This is the EFA *and* KVBM authority (the synth-digest's `efa` key points
>   at this same file).
> - `research/efa-multinode/synth-digest.json` — the `efaVerdict` (ADOPT) digest.
> - Sibling doc: [`50-litellm-universal-surface.md`](./50-litellm-universal-surface.md) —
>   the C0–C3b roadmap, the two-layer routing model, and the "one `api_base` per model"
>   hard rule. This doc is the **C3-deep** extension of that roadmap; §4.2 / §4.3 there
>   are the layering authority and are not re-derived here.
> - Seeds context: `RouteIQ-acdc` (C3a, close the GPU NodePool gap) and `RouteIQ-91fe`
>   (C3b, first engine as ONE `api_base`) both already note the Dynamo+EFA disaggregation
>   tier as "C3 later" — this doc *is* that tier.

---

## 0. Where this sits — the C3 ladder

`50-litellm-universal-surface.md` §5 defines an additive **C0–C3** roadmap. C3
("self-hosted EKS inference") splits, by scale, into three rungs:

| Rung | Scope | Node strategy | EFA/RDMA? | Seed |
|---|---|---|---|---|
| **C3a** | Close the GPU NodePool gap | custom Auto Mode **Karpenter GPU NodePool** | No | `RouteIQ-acdc` |
| **C3b** | First engine, **single-node** model (Production Stack / AIBrix) as ONE `api_base` | C3a NodePool | No | `RouteIQ-91fe` |
| **C3-deep** *(this doc)* | **Very large models that don't fit one node** — multi-node tensor/pipeline parallel + Dynamo disaggregated prefill/decode + KVBM remote KV-cache tiers | EFA device plugin on **either** the Karpenter GPU NodePool (Shape 1) **or** a managed/self-managed node group + GPU Operator (Shape 2, topology-optimal) — see §1.1/§1.3 | **Yes** (cross-node) | NEW (this doc files two) |

The defining constraint of C3-deep is **the model does not fit on one node**, so a single
replica spans GPUs across multiple nodes (tensor/pipeline parallel), and the
high-bandwidth path between those nodes is **AWS EFA** — the only RDMA fabric on AWS
(no InfiniBand, no RoCE). Everything in this doc is about that cross-node fabric and what
rides on it. **The single-node KVBM offload win (§Part 2) is the one exception that does
*not* need EFA — and that is precisely why it can land at C3b without waiting for
C3-deep.**

The two-layer model from `50-...` §4.2 still holds verbatim: **RouteIQ = Layer 1 (model
selection); the engine = Layer 2 (replica scheduling), opaque below one `api_base`.** A
multi-node serving group is still **one logical model behind one `/v1` endpoint.** Part 3
confirms this against the Dynamo source.

```
caller: model="big-reasoning-405b"
        │
   ┌────▼──────────────────────────────────────┐
   │ RouteIQ  Layer 1: model SELECTION          │   (unchanged — §50 §4.2)
   │ K-Thompson bandit over healthy_deployments │
   └────┬──────────────────────────────────────┘
        │ picks ONE deployment: api_base = http://dynamo-frontend/v1
   ┌────▼──────────────────────────────────────────────────────────┐
   │ Dynamo serving group  (ONE api_base — everything below opaque)  │
   │   Frontend (OpenAI HttpService)                                  │
   │   ├─ Smart Router (KvRouter + FlashIndexer)   Layer 2: SCHEDULING│
   │   └─ Workers spanning MULTIPLE NODES:                            │
   │        prefill worker(s) ──KV via NIXL/EFA──► decode worker(s)   │
   │        each worker: tensor/pipeline-parallel ACROSS nodes (EFA)  │
   │        KVBM: G1 HBM → G2 DRAM → G3 NVMe → G4 S3 (remote tiers)   │
   └─────────────────────────────────────────────────────────────────┘
```

---

# PART 1 — Multi-node node strategy: the EFA + Karpenter verdict

## 1.1 The EFA + Karpenter verdict (what is supported where)

**[VERIFIED]** The hard line, from the universal-surface engine report quoted in `50-...`
§4.3 (Dynamo row) and reinforced by both C3 seeds (`RouteIQ-acdc`, `RouteIQ-91fe`):

> **⚠️ CORRECTION (factcheck-verified against the AWS "Manage EFA devices on Amazon EKS"
> doc).** There are TWO EFA mechanisms, and only one is Karpenter-unsupported:
> - The **AWS EFA Kubernetes device plugin** (exposes `vpc.amazonaws.com/efa`, which is
>   what NIXL-LIBFABRIC actually binds to) — **IS supported on Karpenter AND EKS Auto
>   Mode** (AWS doc: *"Use the EFA device plugin with Karpenter and EKS Auto Mode"*).
> - The newer **topology-aware EFA-DRA driver** (GPU↔EFA affinity placement + EFA device
>   *sharing* between pods, via Dynamic Resource Allocation) — **is NOT supported with
>   Karpenter or EKS Auto Mode**; it is managed/self-managed-node-group-only.
>
> So **functional multi-node EFA RDMA — and therefore Dynamo disaggregation over libfabric
> — IS provisionable on Karpenter/Auto Mode** via the device plugin. What you LOSE by
> staying on Karpenter is only DRA-native topology-aware affinity (a placement/perf
> optimization) and cross-pod EFA device sharing — NOT EFA itself. The node-group path is
> the *recommended* path for topology-optimal large-scale disaggregation, not the *only*
> path that works.

This is the architectural fork (refined). Restated as a support matrix:

| Capability | Auto Mode / Karpenter NodePool | Managed / self-managed node group |
|---|---|---|
| GPU node provisioning (g6/g6e/p-family) | **Yes** — custom NodePool + EC2NodeClass | Yes |
| NVIDIA device plugin + drivers | **Auto** — Bottlerocket Accelerated AMIs supply them; install NEITHER the device plugin NOR the GPU Operator | **Manual** — GPU-optimized AMI + self-install NVIDIA device plugin (or GPU Operator) |
| Single-node serving (Production Stack / AIBrix / `vllm serve`) | **Yes** — this is the common path (C3a/C3b) | Yes (but overkill) |
| **AWS EFA Kubernetes device plugin** (`vpc.amazonaws.com/efa`) — functional EFA RDMA / NIXL-LIBFABRIC / multi-node disaggregation | **Yes** — install via Helm `eks/aws-efa-k8s-device-plugin`, scoped to the GPU NodePool | **Yes** |
| **Topology-aware EFA-DRA driver** (GPU↔EFA affinity placement + EFA device sharing) | **NO** — Karpenter/Auto Mode unsupported | **Yes** |

**[VERIFIED]** What the EFA path requires on the node group, per the Dynamo EKS EFA guide
(`docs/kubernetes/cloud-providers/eks/efa.md`) and AWS EKS "Run ML training with EFA"
(`docs.aws.amazon.com/eks/latest/userguide/node-efa.html`):

- **EFA-enabled node group** — `eksctl` `efaEnabled: true`, with a **self-referencing EFA
  security group** (the SG must allow all traffic to/from itself so the EFA NICs on
  different nodes can reach each other).
- **AWS EFA Kubernetes device plugin** — Helm chart `eks/aws-efa-k8s-device-plugin`,
  advertising `vpc.amazonaws.com/efa` as an extended resource pods request.
- **GPU-Direct RDMA on the host** — kernel ≥ 5.12 gives the DMA-BUF path on modern
  AL2023 / Bottlerocket AMIs; otherwise the `efa_nv_peermem` module. This is what lets a
  GPU's memory be the source/sink of an EFA transfer without a CPU bounce.
- **EFA-built container** — Dynamo's `render.py --make-efa` + `docker build --target aws`,
  which adds **libfabric + aws-ofi-nccl** to the image. The stock image will *not* drive
  EFA.
- **NIXL must select the LIBFABRIC backend** — for KVBM (Rust)
  `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true`; for vLLM the `backends:["LIBFABRIC"]`
  kv-transfer-config. **This is the #1 silent-failure trap**: if NIXL lands on UCX it
  silently falls back to ~TCP speed — **no error, just ~100× slower**. The headline
  number from the EFA guide: **~98s TTFT (TCP fallback) vs ~1s with EFA** on Llama-3.1-8B
  at ISL 8000.
- **Pod spec requirements** — request `vpc.amazonaws.com/efa`; `hugepages-2Mi: 5120Mi`;
  **`securityContext.privileged: true`** (REQUIRED for NIXL to register CUDA VRAM via
  `fi_mr_reg` — `IPC_LOCK` alone is **insufficient**); `hostIPC: true`; a large
  `/dev/shm`.
- **Known kernel bug** — on GB200 / arm64 64K-page kernels, the bundled libfabric fails
  `fi_mr_reg` on CUDA VRAM (ofiwg/libfabric#12019); patch libfabric or pin v2.5.1.

**[VERIFIED]** Recommended GPU EFA instances (from the EFA guide instance table):
`p5.48xlarge` / `p5e` / `p5en.48xlarge`, `p6-b200.48xlarge` (3.2 Tbps), and the
**P6e-GB200 UltraServer** (arm64 Grace, up to 28.8 Tbps). These are the families that
both expose EFA NICs *and* have the local NVMe that the KVBM G3 disk tier (Part 2) wants.

## 1.2 Tensor / pipeline parallel across nodes — Grove + KAI (gang) + EFA + capacity

**[VERIFIED — researched 2026-06-16, cited below]** A model too large for one node is served
by **one logical replica that spans several nodes**. The K8s primitive for "a group of pods
that must be co-scheduled and addressed as one unit" is a **gang**. There are two real
options for Dynamo multi-node, and **NVIDIA's recommended/default is Grove + KAI** (the
Dynamo operator selects Grove when present and hard-errors on a multinode deploy if neither
Grove nor LWS is available):

- **Option 1 — RECOMMENDED: NVIDIA Grove (PodCliqueSet) enforced by the KAI Scheduler.**
  Grove (`github.com/ai-dynamo/grove`, `grove.io/v1alpha1`) is a *hierarchical* gang API:
  `PodCliqueSet` (the whole serving system; each replica is one gang) → `PodCliqueScalingGroup`
  (a set of cliques that scale together in a fixed ratio — e.g. 1 leader + N workers as one
  multi-node instance) → `PodClique` (one role: prefill / decode / router, each with its own
  `minAvailable`) → `PodGang` (the auto-generated scheduler-interface CR). This models
  **disaggregated prefill/decode at *different* scale ratios** with a *service-level*
  `minAvailable` so the scheduler only admits a functionally-complete end-to-end pipeline —
  the exact problem flat gang scheduling can't express. **KAI Scheduler**
  (`github.com/NVIDIA/KAI-Scheduler`, NVIDIA's open-sourced ex-Run:ai scheduler, CNCF Sandbox)
  is the scheduler that *enforces* Grove's PodGang: it has a `GroveGrouper` plugin that turns
  the PodGang into its `PodGroup` (MinMember + MinSubGroup) and applies the topology
  constraints. KAI is a **secondary scheduler, opt-in per pod** (`schedulerName: kai-scheduler`
  + `kai.scheduler/queue`) that **coexists** with the default kube-scheduler. The Dynamo DGD
  exposes Grove via `topologyConstraint` (`packDomain: rack|block`, NOT `host`, for multinode)
  + `cliqueStartupType`/`startsAfter` for role startup ordering, so you never touch Grove
  internals. Pin **KAI ≥ v0.13.0** (topology-aware scheduling) + **Grove ≥ v0.1.0-alpha.6**
  per the Dynamo 1.0.x compatibility matrix.
- **Option 2 — FALLBACK: LeaderWorkerSet (LWS ≥ 0.8) + Volcano.** Set
  `nvidia.com/enable-grove: "false"` on the DGD. LWS is the sig-apps multi-node primitive
  (leader + workers, `LeaderFirst`/`Parallel` startup); Volcano is the CNCF batch scheduler
  that provides the PodGroup gang underneath (LWS sets
  `gangSchedulingManagement.schedulerProvider=volcano`). Mature + widely adopted, but it
  models leader+workers, not multi-role disaggregated ratios as cleanly as Grove. Keep this
  as the escape hatch since Grove (alpha) + KAI (sandbox) are both early-2025-stage.

> **CORRECTION (prior drafts said "Grove-Volcano" / "LWS/JobSet"):** there is **no NVIDIA
> "Grove-Volcano" pairing** — Volcano pairs with **LWS**, and KAI pairs with **Grove**.
> JobSet is a more general multi-job primitive, not the Dynamo multinode path. Skip **Kueue**
> here (it's an admission/queueing layer, not a placement scheduler — KAI already has
> hierarchical queues).

The recommended cross-node setup, assembled from the verified pieces above:

1. **Gang scheduler** — Grove (PodCliqueSet) + KAI so the multi-node replica comes up
   all-or-nothing at the *service* level (a half-provisioned tensor-parallel group, or a
   prefill-only / decode-only pipeline, is useless). LWS + Volcano is the documented fallback.
2. **EFA fabric** — the §1.1 node-group + device-plugin + LIBFABRIC stack.
3. **Instances** — p5 / p6 family (EFA NICs + local NVMe); the **P6e-GB200 UltraServer**
   for the very top end (28.8 Tbps EFA, NVLink-domain across the rack).
4. **Capacity** — these instances are scarce. **[SPECULATIVE, AWS product]** **EC2
   Capacity Blocks for ML** (reserve a block of p5/p6 for a fixed window) and the
   **UltraServer** SKU are the realistic ways to *get* enough co-located GPUs for a
   multi-node replica; on-demand p5/p6 capacity is often unavailable in size. The research
   report names p5/p6/UltraServer as the target families but does not adjudicate the
   reservation mechanism — treat capacity acquisition as a known prerequisite, not a
   solved step.

**[VERIFIED — the silent failure that matters most.]** A tensor-parallel group that comes
up but lands NIXL on UCX instead of LIBFABRIC will *work* — it will just be ~100× slower
with **no error in the logs**. Any C3-deep acceptance test MUST assert the LIBFABRIC
backend is selected (and ideally measure TTFT against the ~1s-vs-~98s envelope), because
"it runs" is not evidence the fabric is doing its job.

## 1.3 The cleanest add to our Auto Mode cluster

**[VERIFIED constraint]** Our cluster (`deploy/cdk/lib/eks_cluster_construct.py`) is **EKS
Auto Mode**, CPU-only by default (`_AUTO_MODE_NODE_POOLS = ["general-purpose", "system"]`,
`50-...` §4.4). C3a adds a **custom Karpenter GPU NodePool** for the single-node common
path — and Auto Mode supplies the NVIDIA device plugin automatically, so that path is
clean.

C3-deep has **two viable shapes** (the §1.1 correction): functional multi-node EFA RDMA
works on BOTH Karpenter/Auto Mode (via the EFA device plugin) and a node group — the node
group only adds DRA-native topology-aware placement + EFA device sharing.

- **Shape 1 (Karpenter-native, lower disruption):** extend the C3a custom GPU NodePool /
  EC2NodeClass to EFA-capable instances (p5/p6) and install the **AWS EFA Kubernetes device
  plugin** (`eks/aws-efa-k8s-device-plugin`, scoped to the GPU NodePool labels) so pods can
  request `vpc.amazonaws.com/efa`. Auto Mode already supplies the NVIDIA device plugin. This
  keeps the whole cluster on one (managed-Karpenter) node source. You forgo DRA topology
  affinity — acceptable for many disaggregation setups; validate TTFT.
- **Shape 2 (additive node group, topology-optimal):** add a **managed/self-managed
  EFA-enabled GPU node group** alongside Auto Mode (not replacing it). CPU control-plane +
  C3a/C3b single-node serving stay on Auto Mode; only the multi-node-EFA serving group lands
  on the node group, where you install the **NVIDIA GPU Operator** + **EFA device plugin**
  (scoped via labels/taints so they never touch Auto Mode nodes) and gain DRA topology-aware
  EFA↔GPU affinity. Choose this when placement affinity / EFA device sharing measurably
  matters at scale.
- Either way, **taint the GPU/EFA nodes** so only the multi-node serving pods land there;
  the gang scheduler (Grove+KAI, or the LWS+Volcano fallback) tolerates the taint, and KAI
  is given **dedicated node pools** so it doesn't contend with the default kube-scheduler.

**[SPECULATIVE]** This "two node sources in one cluster" shape (Auto Mode for everything +
a managed EFA node group for the disaggregation tier) is the lowest-disruption way to add
C3-deep, but we have not built the CDK construct for it. It is the natural extension of the
C3a NodeRoleName-reuse pattern, just targeting a managed node group instead of a Karpenter
NodePool.

### 1.3.1 Gang-scheduler ↔ substrate interaction (the real C3-deep decision)

**[VERIFIED — researched 2026-06-16]** Grove+KAI *run* on EKS (KAI is a per-pod secondary
scheduler; "EKS Auto Mode: usable after installing the GPU Operator with the device plugin
disabled via the `nvidia.com/gpu.deploy.device-plugin: "false"` node label"). But two
substrate facts push the **best EFA-topology story toward Karpenter/AL2023 (Shape 2-ish), not
Auto Mode**:

1. **EFA-DRA + DRA + per-device EFA config are NOT supported on EKS Auto Mode or Karpenter**
   (AWS EKS docs). On Auto Mode you get the **EFA device plugin** (`vpc.amazonaws.com/efa`
   extended resource) only — and **automatic GPU↔EFA topology alignment requires AL2023
   accelerated AMIs, which Auto Mode does NOT use (it uses Bottlerocket)**. So Auto Mode gives
   you *functional* multi-node EFA RDMA but **not** automatic topology-aligned placement; you
   lean on Grove `topologyConstraint: packDomain: rack|block` + node labels + (Auto-Mode
   per-device EFA config is "coming soon").
2. **Karpenter is not gang-aware** (open `kubernetes-sigs/karpenter#2030`): it simulates
   kube-scheduler to provision and doesn't understand KAI's pod-group filtering, so it can
   scale the wrong mix or split a gang across pools. Mitigate with **dedicated KAI node pools**
   + per-workload inter-pod affinity; KAI's `NodeScaleAdjuster` treats Karpenter's nominated
   node as a *preference*. KAI's **spread** can also clash with Karpenter **bin-pack/
   consolidation** — separate preemptible vs non-preemptible workloads.

**Decision recorded:** for the C3-deep EFA RDMA tier, prefer **Karpenter + EKS-optimized
AL2023 accelerated AMIs (self-managed or managed node groups) with dedicated KAI node pools**
over Auto Mode/Bottlerocket, because Auto Mode cannot do EFA-DRA / automatic GPU↔EFA topology
alignment today. Revisit when Auto Mode ships per-device EFA config + EFA-DRA support. The
single-node C3a/C3b path stays on Auto Mode; only the multi-node-EFA serving group needs the
AL2023/Karpenter node source.

## 1.4 Decision tree — how many nodes does this model need?

```
            START: a new self-hosted model to serve
                          │
            Does ONE replica fit on ONE node?
                  ┌───────┴────────┐
                YES                NO  (too big for one node)
                  │                 │
         ┌────────▼────────┐        │
         │ C3a: Karpenter   │        │
         │ GPU NodePool     │        │
         │ (Auto Mode,      │        │
         │ device plugin    │        │
         │ AUTO, NO EFA)    │        │
         │ → C3b: one engine│        │
         │ as ONE api_base  │        │
         └─────────────────┘        │
                                     │
                  Does the cross-node path need RDMA bandwidth?
                  (tensor/pipeline parallel, or disaggregated
                   prefill/decode KV transfer)
                       ┌─────────────┴──────────────┐
                      NO                            YES
              (multi-node, TCP is OK —              │
               rare; small models, low QPS)         │
                       │                  ┌─────────▼──────────────────┐
              ┌────────▼─────────┐        │ C3-DEEP: multi-node + EFA   │
              │ multi-node,      │        │ managed/self-managed node   │
              │ NO EFA           │        │ group + GPU Operator + EFA  │
              │ (Karpenter MAY   │        │ device plugin + Grove+KAI   │
              │ still work; NIXL │        │ + p5/p6/UltraServer + NIXL  │
              │ on TCP — accept  │        │ LIBFABRIC backend           │
              │ the ~100x hit    │        │ + KVBM disagg (Part 2)      │
              │ knowingly)       │        │ EFA-DRA ⇒ NOT Auto Mode     │
              └──────────────────┘        └─────────────────────────────┘
```

The branch that matters: **single-node (C3a/C3b, Karpenter, common path) → multi-node-no-
EFA (rare, accept the TCP penalty knowingly) → multi-node-EFA (C3-deep, node groups
required).** The middle branch exists only to make the point explicit: multi-node without
EFA is *technically* possible but lands NIXL on TCP (~98s vs ~1s TTFT). It is almost never
the right answer at scale — if you are spanning nodes, you want EFA.

---

# PART 2 — Dynamo KVBM + remote KV-cache tiers

## 2.1 What KVBM is

**[VERIFIED]** KVBM (KV Block Manager) is, verbatim from `docs/components/kvbm/README.md`:
*"a scalable runtime component designed to handle memory allocation, management, and remote
sharing of Key-Value (KV) blocks for inference tasks across heterogeneous and distributed
environments. It acts as a unified memory layer and write-through cache for frameworks like
vLLM and TensorRT-LLM."*

The KV cache is the per-token attention state every transformer keeps for the whole
context. It normally lives entirely in GPU HBM, which caps both **how long a context can
be** and **how many concurrent sessions** a GPU holds. KVBM's job is to **offload KV-cache
blocks down a memory hierarchy** so HBM is no longer the hard ceiling, and to **avoid
recomputing** cache already produced. It is **write-through**: blocks computed on a worker
flow GPU→CPU→disk automatically, each deduplicated by **sequence hash** in a global
registry — so any worker that can reach the storage tier can reuse them.

## 2.2 The tier hierarchy (HBM → DRAM → NVMe → remote)

**[VERIFIED]** KVBM defines a **four-tier hierarchy** (`docs/design-docs/kvbm-design.md`):

| Tier | Medium | Role | Latency / capacity |
|---|---|---|---|
| **G1** | GPU HBM (VRAM) | active KV used by attention kernels | fastest / smallest |
| **G2** | CPU **pinned** DRAM | staging + promotion back to GPU | µs / medium |
| **G3** | local **NVMe / SSD** | persistent warm-block storage | ms / large |
| **G4** | **remote** (S3 / MinIO; or NFS/Lustre/GPFS) | cold/archival, cross-instance shared | highest / unlimited |

**[VERIFIED] How blocks move — NIXL + GPUDirect Storage (GDS).**
- **NIXL (NVIDIA Inference Xfer Library)** is the bottom transfer layer for *all* data/
  storage transactions: P2P GPU transfers, RDMA + NVLink remote memory sharing, dynamic
  block registration, and a **plugin interface for storage backends** (HBM, host DRAM,
  remote DRAM, SSD, filesystems, object stores).
- **Device→Host (offload):** CUDA D2H into a pinned host block. **Host→Disk:** NIXL Write
  via POSIX, **GDS when available** for the local G3 tier. **Disk→Device (onboard):** NIXL
  Read, **possibly via GDS** — direct NVMe→GPU DMA bypassing the CPU bounce buffer. The
  launch flag `--use-nixl-gds` (+ `use_gds` in `DiskCacheConfig`) enables it.
- **Cross-node KV transfer** rides NIXL over **UCX or libfabric**, over an RDMA transport.
  On AWS that transport is **EFA** — which is exactly the §1.1 fabric. **This is the
  dependency that ties Part 2's cross-node KVBM to Part 1's multi-node-EFA setup.**

**[VERIFIED] The G4 remote tier — what backends.** Object storage (**S3 / MinIO**) is the
native G4 backend (the `kvbm-engine` `object` module: `ObjectBlockOps` trait, an `s3`
submodule with `S3ObjectBlockClient`, and an `ObjectLockManager` that uses conditional PUT
`If-None-Match: *` so **multiple replicas don't duplicate-upload the same block** — i.e.
cross-replica shared cold cache is a first-class design point). Config env vars:
`DYN_KVBM_OBJECT_ENABLED`, `DYN_KVBM_OBJECT_BUCKET`, `DYN_KVBM_OBJECT_ENDPOINT`,
`DYN_KVBM_OBJECT_REGION`, `DYN_KVBM_OBJECT_ACCESS_KEY`, `DYN_KVBM_OBJECT_SECRET_KEY`.
(LMCache / FlexKV / SGLang HiCache + Mooncake are *indirect* remote tiers via the engine
backends, not native KVBM G4.)

**[VERIFIED] The write-through sizing trap.** Because KVBM is write-through, **each tier
must be ≥ the tier above it**. If `DYN_KVBM_CPU_CACHE_GB` < the GPU KV-cache size, KVBM
gives *no benefit and can degrade performance* (it churns, offloading after every forward
pass). Rule: `DYN_KVBM_CPU_CACHE_GB >= GPU_KV_GB` and
`DYN_KVBM_DISK_CACHE_GB >= DYN_KVBM_CPU_CACHE_GB`. Also: disk offload only fires for blocks
with frequency ≥ 2 by default (the SSD-lifespan filter,
`DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true` to disable); MLA models (DeepSeek) want
`DYN_KVBM_NCCL_MLA_MODE=true`.

## 2.3 Why it helps very large / long-context models

**[VERIFIED]** The README gives four value scenarios verbatim; the mechanism behind all of
them:

1. **Longer context + higher concurrency than HBM alone** — moving idle/cold blocks to
   G2/G3/G4 means the GPU no longer holds every active sequence's full cache, so you serve
   prompts longer than HBM allows and pack more concurrent sessions. ("Idle or partial
   conversations can be moved out of GPU memory, allowing active requests to proceed.")
2. **Share/reuse across requests AND replicas** — blocks deduped by sequence hash in a
   global registry; the **`FlashIndexer`** gives the KV-aware **Smart Router** *global
   visibility* of which blocks live where, so a block computed by one worker is reused by
   another (RDMA read) instead of recomputed. The G4 object tier with `ObjectLockManager`
   makes this **cross-replica** — system prompts / tool definitions computed once are
   reused across subagents, turns, and replicas. (Especially valuable for agentic
   workloads.)
3. **Recompute-vs-reload tradeoff** — the alternative to offload is recomputing evicted KV
   (raises TTFT). KVBM bets reload latency < recompute cost when reuse is high; the disk
   offload-frequency filter and write-through sizing rules exist to keep that bet positive.

**[VERIFIED] Benchmark evidence:** Qwen3-8B on H100, avg 20K ISL / 100 OSL — KVBM **host
offloading shows lower TTFT than pure GPU prefix caching, with the gap widening as QPS
rises** (`docs/assets/img/kvbm-agg-performance.png`; exact numbers live in the image, not
text). Note this is the *single-node host-offload* win — it does not need EFA.

## 2.4 The EKS infra KVBM needs — and the three deployment shapes

**[VERIFIED]** KVBM deploys via the **Dynamo operator** using the **`DynamoGraphDeployment`
CRD** (`apiVersion: nvidia.com/v1alpha1`, Helm `deploy/helm/charts/platform`). **etcd +
NATS are required** for leader/worker registration, discovery, and the block-lifecycle
event plane. Three distinct shapes, by ambition:

| Shape | Infra | EFA/RDMA? | Example manifest |
|---|---|---|---|
| **Single-node aggregated** (G1→G2→G3 local) | one GPU box, big DRAM, local NVMe | **No** | `agg_kvbm.yaml` — one `VllmDecodeWorker`, `DYN_KVBM_CPU_CACHE_GB=100`, `DynamoConnector`, `kv_role: kv_both` |
| **Remote G4 object tier** (S3/MinIO cold cache) | S3 bucket / MinIO + `DYN_KVBM_OBJECT_*` | **No** (standard TCP to S3) | (add the object env vars to either shape) |
| **Disaggregated prefill/decode across nodes** | multi-node GPU + EFA/RDMA fabric | **YES** | `disagg_kvbm.yaml` — separate prefill/decode workers, `PdConnector` wrapping `DynamoConnector` (offload) + `NixlConnector` (cross-worker transfer) |

- **G3 (local NVMe):** the p5/p6 families have local NVMe; size via
  `DYN_KVBM_DISK_CACHE_GB`. GDS (`--use-nixl-gds`) gives direct NVMe→GPU DMA. (Dynamo docs
  don't prescribe the K8s storage-provisioning mechanism beyond "G3 needs NVMe/SSD.")
- **G4 (S3/MinIO):** **no EFA** — standard TCP to the object store; `ObjectLockManager`
  makes it multi-replica-safe.
- **Cross-node KV transfer:** **EFA/RDMA — this is the §Part 1 multi-node-EFA setup**, the
  exact same `vpc.amazonaws.com/efa` + NIXL-LIBFABRIC + privileged-pod plumbing.

## 2.5 The dependency on Part 1

**[VERIFIED] — stated as the central dependency.** Two of the three KVBM shapes need **no
EFA**: single-node aggregated offload (G1→G2→G3) and the G4 S3/MinIO cold tier. These are
the **cheap, high-value, EFA-free wins** — they extend effective KV-cache capacity (longer
context, higher concurrency, cross-replica reuse) with **zero RouteIQ change and zero
networking change**, so they can land at **C3b** (single-node) without waiting for the
C3-deep node-group build.

Only the **disaggregated prefill/decode** shape requires the multi-node-EFA setup from
Part 1 — and there it shares the *exact same* EFA + NIXL-LIBFABRIC + privileged-pod
plumbing. **Treat Part 1's node-group/EFA investment and Part 2's disaggregation as one
infra investment, not two.** This is why the `efaVerdict` digest says: *single-node offload
+ S3 cold tier are the EFA-free wins; cross-node disaggregation is the only path that
REQUIRES the multi-node-EFA setup.*

---

# PART 3 — How it composes with RouteIQ

## 3.1 The two-layer model holds — confirmed against Dynamo source

**[VERIFIED]** Dynamo's frontend (`HttpService` in Rust,
`lib/llm/src/http/service/openai.rs`) exposes an **OpenAI-compatible HTTP API**:
`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`. DeepWiki on
`ai-dynamo/dynamo`, verbatim:

> *"An external client, such as a gateway or router performing model selection, interacts
> with Dynamo through a single OpenAI `api_base` URL. The internal complexities of
> Dynamo's architecture are abstracted away by this unified API endpoint."*

So the **entire** request flow — KV-aware Smart Router → prefill worker → KVBM offload/
transfer over NIXL/EFA → decode worker → stream back — happens **behind one endpoint**.
KVBM, the FlashIndexer, the KV router, disaggregated prefill/decode, and remote KV-cache
tiering are all **internal to the Dynamo serving group**. The two-layer model from
`50-...` §4.2 holds *with no change*:

- **Layer 1 — RouteIQ (model SELECTION).** A Dynamo serving group is **one `model_list`
  entry**: `api_base = http://<dynamo-frontend>/v1`, `custom_llm_provider: openai` (or
  `hosted_vllm`). RouteIQ's Kumaraswamy-Thompson bandit + 18 ML strategies pick *which
  model / api_base* exactly as for any OpenAI-compatible upstream. RouteIQ neither sees nor
  configures anything below the frontend.
- **Layer 2 and below — Dynamo (replica SCHEDULING + cache tiering), opaque to RouteIQ.**
  The **Dynamo Smart Router (KvRouter + FlashIndexer)** picks the worker with best KV-cache
  overlap + lowest load; **KVBM** tiers the cache G1→G4; **NIXL/EFA** moves blocks across
  nodes. None of this is visible to or configured by the gateway.

```yaml
# RouteIQ registers a whole multi-node Dynamo group as ONE arm:
model_list:
  - model_name: big-reasoning-405b
    litellm_params:
      model: hosted_vllm/big-reasoning-405b
      api_base: http://dynamo-frontend.dynamo-system.svc.cluster.local:8000/v1
      api_key: fake-api-key
# Behind that ONE api_base: multi-node tensor-parallel workers, disaggregated
# prefill/decode, KVBM G1->G4 tiering, NIXL-over-EFA. RouteIQ sees none of it.
```

## 3.2 The one config error to avoid

**[VERIFIED] — the single duplication risk, from `50-...` §4.2 and the KVBM report §5.**
Because LiteLLM is *itself* a load-balancer, a team can naturally register each Dynamo
**worker / replica** as its own LiteLLM deployment. **Do not.** That makes RouteIQ take
**cache-blind, topology-blind replica decisions that fight** Dynamo's KV-aware Smart Router
and the two layers collapse — the bandit would route around the very KV-cache-overlap and
disaggregation logic that justifies the whole C3-deep tier.

**The hard rule (acceptance criterion): RouteIQ targets exactly ONE `api_base` per model —
the Dynamo *frontend* Service — and delegates ALL per-replica / per-worker / cache-tier
scheduling downward.** Never register individual workers, prefill/decode pools, or replicas
as separate LiteLLM deployments. This is the same discipline C3b already carries; C3-deep
inherits it unchanged. (It is *fine* to have multiple `model_list` rows for one alias when
they are genuinely separate **capacity sources** — a Dynamo group *and* a Bedrock arm, per
`50-...` §1 — but never multiple rows pointing *inside* one Dynamo group.)

---

# PART 4 — Phased plan + where it slots vs the C0–C3b seeds

## 4.1 Slotting against the universal-surface roadmap

`50-...` §5 ships C0–C3 in effort/value order. C3 is the self-hosted tier; this doc splits
its tail into the scale-gated **C3-deep** rung. The full ladder:

| Phase | Scope | Node strategy | EFA? | Status |
|---|---|---|---|---|
| C0 | API-key + cross-region capacity pools | n/a (config) | No | seed `RouteIQ-e677` |
| C1 | Cross-account Bedrock arms | n/a (CDK port) | No | seed `RouteIQ-6150` |
| C2 | Gov-ban / non-routable arm | n/a (app) | No | seed `RouteIQ-badb` |
| **C3a** | Close GPU NodePool gap | Karpenter NodePool (Auto Mode) | No | seed `RouteIQ-acdc` |
| **C3b** | First engine, **single-node**, ONE `api_base` (+ **single-node KVBM offload + S3 G4** — EFA-free) | C3a NodePool | No | seed `RouteIQ-91fe` |
| **C3-deep** *(this doc)* | **Multi-node very-large-model serving + Dynamo disaggregation + cross-node KVBM** | **AL2023/Karpenter (or managed/self-managed node group) + GPU Operator + EFA device plugin + Grove+KAI (LWS+Volcano fallback)** | **Yes** | **NEW — two seeds filed below** |

## 4.2 The C3-deep phased plan (ordered, cheapest first)

Mirroring the KVBM report's recommendation order — **the EFA-free wins land first, the EFA
build is last and only-if-needed**:

1. **(at C3b, EFA-free) Single-node aggregated KVBM offload.** Set `DYN_KVBM_CPU_CACHE_GB`
   (≥ GPU KV size, to avoid the write-through churn trap) and optionally
   `DYN_KVBM_DISK_CACHE_GB` on an NVMe node. Bigger effective KV cache → longer context +
   more concurrency per GPU. **No EFA, no node group, no RouteIQ change.** This is the
   lowest-friction win and belongs *with C3b*, not in C3-deep.
2. **(at C3b, EFA-free) Add the G4 S3/MinIO cold tier** when there is cross-replica reuse
   (shared system prompts, agentic workloads). `DYN_KVBM_OBJECT_*` + a bucket;
   `ObjectLockManager` makes it multi-replica-safe. Still no EFA.
3. **(C3-deep, the gate) Stand up the multi-node-EFA node source** (Part 1, §1.3.1): prefer
   **Karpenter + AL2023 accelerated AMIs** (or a managed/self-managed EFA-enabled GPU node
   group) + GPU Operator + AWS EFA device plugin + **Grove+KAI gang scheduling** (LWS+Volcano
   fallback) + p5/p6/UltraServer + the **LIBFABRIC backend** (verify it — the silent ~100×
   trap). NOT Auto Mode for this tier (no EFA-DRA / no auto GPU↔EFA topology on Bottlerocket).
   This is the highest-effort, scale-gated step.
4. **(C3-deep) Deploy the very-large model as a Dynamo `DynamoGraphDeployment`** with
   disaggregated prefill/decode (`disagg_kvbm.yaml` shape: `PdConnector` =
   `DynamoConnector` + `NixlConnector`) **on top of** step 3's fabric, and register the
   frontend as **ONE `api_base`** in RouteIQ (Part 3 hard rule).

**Ordering rationale.** Steps 1–2 are config-only EFA-free wins that ride the C3b
single-node engine — do them as part of C3b. Step 3 is the genuine C3-deep gate (a new
node group + GPU Operator + EFA, the part Auto Mode/Karpenter cannot do). Step 4 only pays
off for models that **don't fit one node** or have prefill/decode-imbalanced,
long-context/reasoning/agentic workloads — per `50-...` §4.3, disaggregation is
scale-gated at ~8+ GPU nodes. **Do not build the EFA node group speculatively; build it
when a model actually requires multi-node serving.**

## 4.3 Acceptance criteria carried into C3-deep

- **One `api_base` per model** (Part 3.2) — never per-worker/per-replica. A cred-free
  config test asserting exactly one `model_list` row points into any given Dynamo
  frontend.
- **LIBFABRIC backend asserted** (Part 1.2) — a test/probe confirming NIXL selected
  LIBFABRIC, not the silent UCX/TCP fallback. "It runs" ≠ "the fabric works."
- **Write-through tier sizing** (Part 2.2) — `DYN_KVBM_CPU_CACHE_GB >= GPU_KV_GB` and
  `DYN_KVBM_DISK_CACHE_GB >= DYN_KVBM_CPU_CACHE_GB`, else KVBM degrades performance.
- **Node-group isolation** (Part 1.3) — the EFA node group is taint-scoped so the GPU
  Operator + EFA device plugin never touch Auto Mode nodes; the CPU control plane and
  C3a/C3b single-node serving stay on Auto Mode.

---

## 5. One-paragraph synthesis

Serving a model too big for one node is a **Layer-2-and-below** problem that lives entirely
inside **one OpenAI `api_base`** — a Dynamo multi-node serving group whose Frontend exposes
`/v1/chat/completions` while its Smart Router, KVBM cache tiering (G1 HBM → G2 DRAM → G3
NVMe → G4 S3 over NIXL/GDS), and disaggregated prefill/decode are opaque to the gateway, so
RouteIQ's Kumaraswamy-Thompson bandit keeps doing **Layer-1 model selection** unchanged and
registers the whole group as **ONE arm** (the one config error to avoid: never register
individual workers/replicas). The cross-node fabric is **AWS EFA** — the only RDMA fabric
on AWS — and the verdict is sharp: the **EFA-DRA driver is not supported on
Karpenter/Auto Mode** (and Auto Mode's Bottlerocket gives no auto GPU↔EFA topology), so the
multi-node-EFA tier prefers **Karpenter + AL2023 accelerated AMIs (or managed/self-managed
node groups) + GPU Operator + AWS EFA device plugin + Grove+KAI gang scheduling (LWS+Volcano
fallback) + p5/p6/UltraServer**, with NIXL forced onto the **LIBFABRIC backend** (the silent
~98s-vs-~1s TTFT trap). KVBM's single-node
offload and S3 G4 cold tier are **EFA-free** wins that ride C3b; only **disaggregated
prefill/decode** requires the Part-1 EFA setup — same plumbing, one investment. This is the
**C3-deep** rung beyond the C3a/C3b common path in
[`50-litellm-universal-surface.md`](./50-litellm-universal-surface.md): build the EFA node
group only when a model genuinely doesn't fit one node.
