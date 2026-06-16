# Universal-Surface & Deferred Residuals — Operator Runbook

> **Audience**: cluster/platform operators with live AWS + EKS credentials.
> **Scope**: the **LIVE / operator-gated halves** of four residuals whose
> *cred-free* halves (CDK constructs, chart wiring, RouteIQ code, tests) are
> already shipped. This document is the **explicit, actionable disposition** of
> the remaining work — not a silent gap. Each section names the exact steps,
> preconditions, the acceptance check, and the flag / `CfnOutput` / value to flip.

This runbook closes the operator side of:

| Seed | Title | Cred-free half (shipped) | This runbook covers |
|---|---|---|---|
| `RouteIQ-91fe` | C3b: first self-hosted engine as ONE `api_base` | `GpuNodePoolManifest` CfnOutput (`enable_gpu_nodepool`) | deploy GPU NodePool, install engine, register ONE frontend `api_base` |
| `RouteIQ-28be` | C3-deep (KVBM): remote KV-cache tiers | architecture (doc 51); EFA-free shapes land at C3b | single-node KVBM offload + S3 G4, tier-sizing check |
| `RouteIQ-2f97` | C3-deep (multi-node EFA) | architecture (doc 51, Part 1) | EFA NodePool + device plugin, NIXL=LIBFABRIC probe, libfabric pin |
| `RouteIQ-c299` | Flip Kumaraswamy moment-fit default ON | code + decision (doc 20 §9) shipped; default `False` | the canary procedure and the final code-default flip |
| `RouteIQ-4f59` | P4 WAF construct | `WafConstruct` (`enable_waf` + `waf_alb_arn`), 13 tests | live ALB attach |

**Layering invariant (applies to §1–§2, the universal-surface sections).** RouteIQ
is **Layer 1 — model selection**. A self-hosted engine (vLLM Production Stack,
AIBrix, Dynamo) is **Layer 2 — replica scheduling**, opaque below **one
OpenAI `api_base`**. The two layers *compose* only if RouteIQ targets exactly
**ONE `api_base` per model — the engine frontend Service**, never individual
replicas/workers. Registering replicas as separate `model_list` rows collapses
Layer-1 model-selection into Layer-2 replica-scheduling — **the one config error
to avoid**. See
[`docs/architecture/aws-rearchitecture/50-litellm-universal-surface.md`](../architecture/aws-rearchitecture/50-litellm-universal-surface.md)
§4.2 and
[`51-multinode-large-model-serving.md`](../architecture/aws-rearchitecture/51-multinode-large-model-serving.md)
Part 3.

---

## 1. Self-hosted inference engine as ONE `api_base` (`RouteIQ-91fe` + `RouteIQ-28be` single-node KVBM)

> **Seeds**: `RouteIQ-91fe` (C3b), `RouteIQ-28be` (KVBM, EFA-free shapes).
> **ADR / architecture**: ADR-0030 (EKS Auto Mode substrate),
> [doc 50](../architecture/aws-rearchitecture/50-litellm-universal-surface.md) §5
> (C0–C3 ladder), [doc 51](../architecture/aws-rearchitecture/51-multinode-large-model-serving.md)
> Part 2 + Part 4.
> **Cred-free half DONE**: the GPU NodePool manifest is emitted as the
> `GpuNodePoolManifest` `CfnOutput`, flag-gated by `routeiq:enable_gpu_nodepool`
> (default OFF; `deploy/cdk/lib/eks_cluster_construct.py::enable_gpu_node_pool`),
> with a `Template.from_stack` present-when-on / absent-when-off test.
> **This is the operator half** — provisioning the live GPU capacity, installing
> the engine, and wiring the chart.

### 1.0 Preconditions

- The P0 foundation stack (`RouteIqStack-<env>`) is deployed (EKS Auto Mode
  cluster, pod IAM role, ECR pull-through cache).
- `kubectl` context points at the cluster; `helm` is available.
- GPU instance quota in the region for the `g`/`p` families at generation > 4
  (g5/g6/g6e, p5/p6). On-demand p5/p6 is often unavailable in size — for KVBM
  G3 (local NVMe) you want a p5/p6 with local NVMe; plan EC2 capacity ahead.

### 1.1 Deploy the GPU NodePool (emit + apply the manifest)

The two AWS-managed Auto Mode node pools are **CPU-only**, so a pod requesting
`nvidia.com/gpu` sits `Pending` forever. Flip the flag, synth, and apply the
emitted manifest out-of-band.

1. **Enable the flag and deploy the foundation:**

   ```bash
   cd deploy/cdk
   cdk deploy RouteIqStack-<env> -c routeiq:enable_gpu_nodepool=true
   ```

   This adds exactly one `CfnOutput` named **`GpuNodePoolManifest`** whose value
   is the GPU Karpenter `NodePool` + EC2 `NodeClass` YAML. (It is a `CfnOutput`,
   not a `KubernetesManifest`: the app/CRD layer is applied out-of-band over the
   L1 `CfnCluster`.) The NodeClass **reuses** the existing Auto Mode node role
   (the `NodeRoleName` output), so no new access entry is needed.

2. **Read the emitted manifest and apply it:**

   ```bash
   aws cloudformation describe-stacks --stack-name RouteIqStack-<env> \
     --query "Stacks[0].Outputs[?OutputKey=='GpuNodePoolManifest'].OutputValue" \
     --output text > /tmp/gpu-nodepool.yaml
   kubectl apply -f /tmp/gpu-nodepool.yaml
   ```

   The manifest carries the `nvidia.com/gpu` `NoSchedule` taint (only
   GPU-tolerating pods land on accelerated nodes) and selects `g`+`p` categories
   at generation > 4, amd64, spot+on-demand. **Auto Mode supplies the NVIDIA
   device plugin + drivers + the accelerated Bottlerocket AMI** — do **NOT**
   install the device plugin or the GPU Operator on this path.

   **Acceptance check (NodePool live):**

   ```bash
   kubectl get nodepool routeiq-gpu -o jsonpath='{.status.conditions}'
   # schedule a probe GPU pod that tolerates nvidia.com/gpu and requests
   # resources.limits."nvidia.com/gpu": 1 -> it must reach Running (not Pending).
   ```

### 1.2 Install the engine (vLLM Production Stack OR AIBrix OR Dynamo operator)

Pick **one** engine. Per `RouteIQ-91fe`, the smallest step up from `vllm serve`
is the **vLLM Production Stack** (KV-cache-aware routing); **AIBrix** is the
most production-proven (LoRA-aware autoscaling + distributed KV cache); both are
EKS-Auto-Mode-clean. For KVBM (`RouteIQ-28be`) the engine is the **Dynamo
operator** (`DynamoGraphDeployment` CRD).

1. **vLLM Production Stack / AIBrix** — install per upstream Helm chart; pin the
   pods to the GPU NodePool (toleration for `nvidia.com/gpu` + a nodeSelector on
   `routeiq.ai/nodepool: gpu`, the label the manifest stamps). Expose a single
   in-cluster gateway/router Service (this becomes the `api_base`).

2. **Dynamo operator (for KVBM)** — install the Dynamo operator + the
   `DynamoGraphDeployment` CRD (`apiVersion: nvidia.com/v1alpha1`); **etcd +
   NATS are required** (leader/worker registration, discovery, block-lifecycle
   events). For the EFA-free single-node KVBM shapes use the `agg_kvbm.yaml`
   shape (one `VllmDecodeWorker`, `DynamoConnector`, `kv_role: kv_both`).

### 1.3 Single-node KVBM offload + S3 G4 (EFA-FREE) — `RouteIQ-28be`

Two of the three KVBM shapes need **no EFA** and ride the single-node C3b
engine. Land them here; **cross-node disaggregation is §2 (EFA-gated)**.

- **Shape (1) single-node aggregated G1→G2→G3** (no EFA): set the CPU/disk tiers
  on the worker. **Write-through sizing trap** — because KVBM is write-through,
  **each tier must be ≥ the tier above it**, or KVBM churns every forward pass
  and *degrades* performance:

  ```text
  DYN_KVBM_CPU_CACHE_GB  >= GPU_KV_GB           # CPU pinned DRAM >= GPU KV size
  DYN_KVBM_DISK_CACHE_GB >= DYN_KVBM_CPU_CACHE_GB   # local NVMe >= CPU
  ```

  G3 wants a p5/p6 with local NVMe + `--use-nixl-gds` (direct NVMe→GPU DMA).
  MLA models (DeepSeek) want `DYN_KVBM_NCCL_MLA_MODE=true`. Disk offload only
  fires for blocks with frequency ≥ 2 by default
  (`DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true` disables the filter).

- **Shape (2) remote G4 object tier** (no EFA, standard TCP to S3): add the
  object env vars + a bucket. `ObjectLockManager` (conditional PUT
  `If-None-Match: *`) makes it **multi-replica-safe** (cross-replica cold cache,
  e.g. shared system prompts for agentic workloads):

  ```text
  DYN_KVBM_OBJECT_ENABLED=true
  DYN_KVBM_OBJECT_BUCKET=<bucket>
  DYN_KVBM_OBJECT_REGION=<region>
  DYN_KVBM_OBJECT_ENDPOINT=<s3-or-minio-endpoint>
  # creds via Pod Identity (preferred) or DYN_KVBM_OBJECT_ACCESS_KEY / _SECRET_KEY
  ```

  **Acceptance check (tier sizing asserted):** confirm
  `DYN_KVBM_CPU_CACHE_GB >= GPU_KV_GB` and
  `DYN_KVBM_DISK_CACHE_GB >= DYN_KVBM_CPU_CACHE_GB` in the deployed worker env
  before declaring KVBM healthy. Optionally validate cross-replica reuse: a block
  computed by one replica is read (not recomputed) by another.

### 1.4 Register the engine frontend as ONE `api_base` (the layering invariant)

In the chart `values.yaml` `config.gateway` block (or an operator override),
add **one** `model_list` row per model pointing at the **engine frontend
Service**:

```yaml
config:
  gateway: |
    model_list:
      - model_name: big-reasoning-405b
        litellm_params:
          model: hosted_vllm/big-reasoning-405b   # or openai/<name> for openai_like
          api_base: http://dynamo-frontend.dynamo-system.svc.cluster.local:8000/v1
          api_key: fake-api-key
```

- **`api_base` is the in-cluster frontend Service** (the engine's gateway/router),
  **NOT** an individual replica/worker. Everything below the frontend
  (KV-aware Smart Router, KVBM G1→G4 tiering, NIXL transfers) is opaque to
  RouteIQ.
- Multiple `model_list` rows for one alias are fine **only** when they are
  genuinely separate *capacity sources* (e.g. a Dynamo group *and* a Bedrock
  arm) — never multiple rows pointing *inside* one engine group.

**Acceptance check (the invariant):** exactly one `model_list` row points into
any given engine frontend. Send a request for the model through RouteIQ and
confirm the routing decision targets the frontend `api_base` and the engine
serves it (a 200 with a completion). The cred-free config test for "one
`api_base` per model" already guards regressions in config; this check confirms
it end-to-end on the live engine.

---

## 2. Multi-node EFA serving (`RouteIQ-2f97`) + cross-node KVBM disaggregation (`RouteIQ-28be` shape 3)

> **Seeds**: `RouteIQ-2f97` (EFA device plugin), `RouteIQ-28be` (disaggregated
> prefill/decode shape).
> **ADR / architecture**: ADR-0030,
> [doc 51](../architecture/aws-rearchitecture/51-multinode-large-model-serving.md)
> Part 1 (EFA + Karpenter verdict) + Part 2.5 (the Part-1 dependency).
> **Cred-free half**: the EFA framing, support matrix, and acceptance criteria
> are the architecture doc (doc 51); there is **no `enable_efa` CDK flag yet** —
> the EFA NodePool / device-plugin is forward-roadmap and built **only when a
> model genuinely does not fit one node**. Gate this **after** §1 (C3b).
> **This is the operator half**: provision EFA, install the device plugin,
> probe the fabric, deploy the disaggregated graph.

### 2.0 Preconditions

- §1 (C3b single-node engine) is live and proven.
- A model that **genuinely does not fit one node** (a single replica must span
  GPUs across multiple nodes). If the model fits one node, **stop** — this tier
  is unnecessary and adds the EFA silent-failure surface for no benefit.
- EFA-capable GPU capacity reserved: `p5.48xlarge` / `p5e` / `p5en.48xlarge`,
  `p6-b200.48xlarge`, or the **P6e-GB200 UltraServer** (arm64 Grace). These both
  expose EFA NICs *and* carry the local NVMe the KVBM G3 tier wants. On-demand
  p5/p6 in size is scarce — plan **EC2 Capacity Blocks for ML** / the UltraServer
  SKU. (Capacity acquisition is a known prerequisite, not a solved step.)

### 2.1 Choose a shape and enable EFA

There are **two viable shapes** (doc 51 §1.1 correction — the AWS EFA Kubernetes
device plugin `vpc.amazonaws.com/efa` *is* supported on Karpenter/Auto Mode;
only the topology-aware **EFA-DRA driver** is node-group-only):

- **Shape 1 (Karpenter-native, lower disruption):** extend the §1 GPU NodePool /
  EC2NodeClass to EFA instances (p5/p6) and install the **AWS EFA Kubernetes
  device plugin** scoped to the NodePool labels. You forgo DRA topology affinity
  + EFA device sharing (a placement/perf optimization), **not EFA itself**.
- **Shape 2 (additive managed/self-managed node group, topology-optimal):** add
  an EFA-enabled GPU node group **alongside** Auto Mode; install the **NVIDIA GPU
  Operator** + EFA device plugin (scoped via labels/taints so they never touch
  Auto Mode nodes); gain DRA topology-aware EFA↔GPU affinity. Choose this when
  placement affinity / device sharing measurably matters at scale.

The forward-roadmap intent is a flag-gated `routeiq:enable_efa` context key that
emits an `EfaNodePoolManifest` `CfnOutput` (Shape 1) mirroring the GPU NodePool
pattern. **Until that construct is built, the operator hand-applies the EFA
NodePool + device-plugin manifests** described below.

1. **(Shape 1) Apply the EFA NodePool + EC2NodeClass** (an EFA-instance variant
   of the §1 GPU NodePool — `g`/`p` restricted to EFA SKUs, same node-role reuse).
   Ensure the EC2NodeClass attaches a **self-referencing EFA security group** (the
   SG must allow all traffic to/from itself so EFA NICs on different nodes reach
   each other).

2. **Install the AWS EFA Kubernetes device plugin**, scoped to the EFA NodePool
   labels:

   ```bash
   helm repo add eks https://aws.github.io/eks-charts
   helm install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin \
     --namespace kube-system \
     --set nodeSelector."routeiq\.ai/nodepool"=gpu-efa
   ```

   It advertises `vpc.amazonaws.com/efa` as an extended resource pods request.

3. **Taint the EFA nodes** so only the multi-node serving pods land there; the
   gang scheduler tolerates the taint, and **KAI gets dedicated node pools** so it
   doesn't contend with the default kube-scheduler.

### 2.1.1 Gang scheduler — Grove + KAI (recommended) vs LWS + Volcano (fallback)

NVIDIA's **default/recommended** Dynamo-multinode gang stack is **Grove + KAI** — the
Dynamo operator selects Grove when present and hard-errors on a multinode deploy if
**neither** Grove nor LWS is installed:

- **Option 1 (recommended):** **NVIDIA Grove** (`PodCliqueSet` → `PodCliqueScalingGroup`
  → `PodClique` → auto-generated `PodGang`) describes the whole disaggregated system and
  enforces gang completeness at the *service* level (≥1 complete prefill instance AND ≥1
  complete decode instance) while letting prefill/decode scale at **different ratios**.
  **KAI Scheduler** enforces the PodGang (its `GroveGrouper` plugin → KAI `PodGroup`
  MinMember/MinSubGroup). KAI is a **secondary scheduler, opt-in per pod**
  (`schedulerName: kai-scheduler` + `kai.scheduler/queue`) that coexists with
  kube-scheduler. On the DGD, set topology via `topologyConstraint`
  (`packDomain: rack|block`, not `host`) + startup order via
  `cliqueStartupType`/`startsAfter`. **Pin KAI ≥ v0.13.0** (topology-aware scheduling) +
  **Grove ≥ v0.1.0-alpha.6** per the Dynamo 1.0.x compatibility matrix. Install on EKS
  Auto Mode after installing the GPU Operator with the device plugin disabled via the
  `nvidia.com/gpu.deploy.device-plugin: "false"` node label.
- **Option 2 (fallback):** set `nvidia.com/enable-grove: "false"` on the DGD and use
  **LeaderWorkerSet (LWS ≥ 0.8) + Volcano** (LWS sets
  `gangSchedulingManagement.schedulerProvider=volcano`). Mature/widely-adopted; models
  leader+workers rather than multi-role disaggregated ratios.

> Both Grove (alpha) and KAI (CNCF Sandbox) are early-stage; **Karpenter is not gang-aware**
> (`kubernetes-sigs/karpenter#2030`) so it can split a gang across pools — mitigate with the
> dedicated KAI node pools above + per-workload inter-pod affinity. Do **not** add Kueue here
> (it's admission/queueing, not placement; KAI has hierarchical queues natively).

> **EFA-topology substrate note:** EFA-DRA + per-device EFA config are **not supported on
> EKS Auto Mode** (Bottlerocket), and automatic GPU↔EFA topology alignment needs **AL2023
> accelerated AMIs**. For the multi-node-EFA tier prefer **Karpenter + AL2023 (or managed
> node groups)** over Auto Mode; the single-node C3a/C3b path stays on Auto Mode.

### 2.2 Pod spec + container requirements (the non-negotiables)

The multi-node replica is one **gang** (Grove `PodCliqueSet`, or the LWS fallback)
co-scheduled all-or-nothing. Each pod **must**:

- request `vpc.amazonaws.com/efa`;
- set `hugepages-2Mi: 5120Mi`;
- set **`securityContext.privileged: true`** — **REQUIRED** for NIXL to register
  CUDA VRAM via `fi_mr_reg`; `IPC_LOCK` alone is **insufficient**;
- set `hostIPC: true` and a large `/dev/shm`.

The container must be an **EFA-built image** (Dynamo `render.py --make-efa` +
`docker build --target aws`, which adds **libfabric + aws-ofi-nccl**). The stock
image will **not** drive EFA. GPU-Direct RDMA needs kernel ≥ 5.12 (DMA-BUF path
on modern AL2023/Bottlerocket) else the `efa_nv_peermem` module.

### 2.3 The NIXL = LIBFABRIC probe (the documented silent trap)

**This is the #1 silent-failure trap.** NIXL must select the **LIBFABRIC**
backend. If it lands on **UCX**, it silently falls back to ~TCP speed — **no
error, ~100× slower**: **~98s TTFT (TCP fallback) vs ~1s with EFA** on
Llama-3.1-8B at ISL 8000. "It runs" is **not** evidence the fabric works.

- Force the backend:
  - **KVBM (Rust):** `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true`
  - **vLLM:** `backends: ["LIBFABRIC"]` in the kv-transfer-config.

- **Acceptance check (probe):** assert NIXL selected **LIBFABRIC** (not UCX/TCP)
  — inspect the NIXL backend in the worker logs/telemetry, and ideally **measure
  TTFT against the ~1s-vs-~98s envelope**. A TTFT near ~98s on a long-ISL prompt
  is the signature of the UCX/TCP fallback even when the pods are Running.

### 2.4 libfabric v2.5.1 pin (GB200 / arm64 64K-page bug)

On **GB200 / arm64 64K-page kernels**, the bundled libfabric fails `fi_mr_reg`
on CUDA VRAM (ofiwg/libfabric#12019). **Pin libfabric v2.5.1** (or patch) in the
EFA-built image when targeting the P6e-GB200 UltraServer / any arm64 64K-page
node. Without the pin, memory registration fails and the fabric never comes up.

### 2.5 Deploy the disaggregated graph + register ONE `api_base`

Deploy the very-large model as a Dynamo `DynamoGraphDeployment` with
disaggregated prefill/decode (`disagg_kvbm.yaml` shape: `PdConnector` =
`DynamoConnector` for offload + `NixlConnector` for cross-worker transfer) on
top of the EFA fabric. Then register the **frontend** as **ONE `api_base`** in
RouteIQ (the §1.4 invariant, unchanged). KVBM cross-node disaggregation rides
the **exact same** EFA + NIXL-LIBFABRIC + privileged-pod plumbing — **one infra
investment, not two** (doc 51 §2.5).

**Acceptance checks (this tier):**

1. A multi-node gang pod **schedules** (all-or-nothing) on the EFA nodes.
2. The NIXL=LIBFABRIC probe (§2.3) passes — fabric, not TCP.
3. Tier sizing (§1.3) holds for the cross-node KVBM workers.
4. Node-group isolation: the GPU Operator + EFA device plugin are taint/label
   scoped so they never touch Auto Mode nodes (CPU control plane + C3a/C3b stay
   on Auto Mode).
5. RouteIQ targets exactly **one** `api_base` (the frontend) for the model.

---

## 3. Kumaraswamy moment-fit default flip — the canary (`RouteIQ-c299`)

> **Seed**: `RouteIQ-c299`.
> **ADR / architecture**:
> [doc 20](../architecture/aws-rearchitecture/20-kumaraswamy-thompson-router.md)
> §9 (the recorded decision + acceptance gate). Code: `kumaraswamy_thompson.py`
> (`fit_kumaraswamy_moments`, `Posterior.shape(moment_fit=...)`), setting
> `settings.py::KumaraswamyThompsonSettings.moment_fit` (default **`False`**).
> **The RouteIQ-code side is already DONE** — the math is verified correct
> (mean restored to ~1e-8; variance ~1.0x feasible, ~1.67x only at the
> `Beta(500,500)` floor) and the decision is documented. **The only remaining
> work is the live flip — and the runbook below IS the closing deliverable for
> the RouteIQ side; the canary is the operator action.**

### 3.0 Why it is canary-gated (not flipped in code now)

Flipping the default ON is a **live bandit behavior change**: it shifts the
sampled `(a,b)` for every non-corner posterior, which shifts the Thompson draw
and therefore the **arm-selection distribution** in production. That class of
change must be **observed on real traffic** before it becomes the default, and
that observation **cannot be run cred-free**. The default-off posture exists
purely for byte-stable backward-compat, not because the fit is wrong.

### 3.1 Canary procedure

1. **Stage the treatment.** `moment_fit` is a per-instance flag, so A/B it via a
   canary deploy with the treatment replicas carrying:

   ```text
   ROUTEIQ_KUMARASWAMY_THOMPSON__MOMENT_FIT=true
   ```

   (or via the existing strategy-registry weights). Control replicas keep the
   option-1 `a=alpha, b=beta` shortcut (`MOMENT_FIT=false`, the current default).

2. **Observe over a real traffic window** the two acceptance metrics:
   - **Arm-selection distribution** under moment-fit vs the shortcut **differs in
     the expected direction** (the corrected mean pulls selection toward the
     empirically-better arm), AND **cumulative regret ≤** the shortcut's at fixed
     cost-weight (doc 20 §8 oracle-regret + cumulative-reward metrics). It must
     **not worsen** selection.
   - **No latency regression**: the ~2.3ms cache-miss fit at production arm counts
     stays within the p50/p99 routing-latency budget (doc 20 §8 secondary metric
     4). The fit is cached per exact `(alpha,beta)` and runs at most once per
     feedback update, so steady-state should be a cache hit; verify the cold-fit
     cost does not breach the budget at peak arm counts.

### 3.2 On acceptance — flip the code default

If both metrics pass, flip the code default and update the notes (this is the
RouteIQ-side close-out):

1. `settings.py::KumaraswamyThompsonSettings.moment_fit` → default **`True`**.
2. Update the **"Default off for byte-stable backward-compat"** note in
   `settings.py` and the `Posterior.shape` docstring in `kumaraswamy_thompson.py`
   to record that the default is now the moment-fit and the option-1 shortcut is
   the **opt-out** (the byte-stable-corner guarantee is unchanged either way).
3. Update doc 20 §9 to record the canary result and the flip.

**Acceptance check (flip):** on a canary with live traffic, arm-selection
differs in the expected direction with regret ≤ shortcut at fixed cost-weight,
and no p50/p99 latency regression. Until then the default stays `False`.

---

## 4. WAF live ALB attach (`RouteIQ-4f59`)

> **Seed**: `RouteIQ-4f59`.
> **ADR / architecture**: ADR-0006 (security-hardening defaults), ADR-0030
> (Auto Mode substrate); see also
> [`eks-auto-mode-ingress.md`](./eks-auto-mode-ingress.md) for the ALB edge.
> **The RouteIQ-construct side is already DONE** — `waf_construct.py` ships a
> REGIONAL WebACL + managed rule groups (IpReputation + KnownBadInputs BLOCK
> day-1; CommonRuleSet COUNT then `routeiq:waf_crs_block`) + rate limit + redacted
> logging + ALB association, flag-gated (`routeiq:enable_waf` + `routeiq:waf_alb_arn`),
> byte-stable OFF, cdk-nag clean, 13 tests. **The only remaining work is the live
> attach — and this runbook IS the closing deliverable for the RouteIQ side; the
> flip is the operator action.**

### 4.0 Why it is deploy-gated

At P0 the chart default is `service.type=ClusterIP` / `ingress.enabled=false`,
so **EKS Auto Mode renders NO ALB** — there is no in-stack ALB ARN to associate
to. The construct is instantiated **only** when `routeiq:enable_waf` is true
**AND** an operator supplies a non-empty `routeiq:waf_alb_arn` (the
`and self._waf_alb_arn` guard in `routeiq_stack.py` is load-bearing). With the
flag OFF the stack emits **zero** `AWS::WAFv2::*` resources and the snapshot stays
byte-stable.

### 4.1 Live attach procedure

1. **Render an ALB.** Flip the chart `ingress.enabled=true` so EKS Auto Mode
   renders an ALB from the Ingress. Under Auto Mode, ALB config (cert ARNs,
   `targetType: ip`, scheme) lives in an **`IngressClassParams`** CR — follow
   [`eks-auto-mode-ingress.md`](./eks-auto-mode-ingress.md) for that wiring (the
   `alb.ingress.kubernetes.io/*` annotation path is for the **self-managed** LBC,
   not Auto Mode).

   ```bash
   helm upgrade <release> deploy/charts/routeiq-gateway \
     --set ingress.enabled=true --set ingress.className=routeiq-alb
   ```

2. **Capture the ALB ARN** once Auto Mode provisions it:

   ```bash
   kubectl describe ingress <release>-routeiq-gateway   # address -> ALB
   aws elbv2 describe-load-balancers \
     --query "LoadBalancers[?DNSName=='<alb-dns>'].LoadBalancerArn" --output text
   ```

3. **Re-deploy the foundation with WAF on**, passing the live ALB ARN:

   ```bash
   cd deploy/cdk
   cdk deploy RouteIqStack-<env> \
     -c routeiq:enable_waf=true \
     -c routeiq:waf_alb_arn=arn:aws:elasticloadbalancing:<region>:<acct>:loadbalancer/app/<alb> \
     -c routeiq:waf_crs_block=false   # keep CRS in COUNT for the initial bake; flip true after
   ```

   This builds the `WafConstruct` and emits the `AWS::WAFv2::WebACLAssociation`
   binding the WebACL to the supplied ALB ARN.

   > **Bake-window note**: leave `routeiq:waf_crs_block=false` (CommonRuleSet in
   > COUNT) initially to observe false positives, then flip
   > `routeiq:waf_crs_block=true` once clean. `IpReputation` + `KnownBadInputs`
   > are BLOCK day-1 (near-zero FP).

### 4.2 Acceptance check

```bash
# 1. The WebACL is associated to the live ALB:
aws wafv2 get-web-acl-for-resource --region <region> \
  --resource-arn arn:aws:elasticloadbalancing:<region>:<acct>:loadbalancer/app/<alb>
# returns the routeiq-<env> WebACL.

# 2. A known-bad request is blocked (KnownBadInputs/IpReputation -> 403):
curl -i https://routeiq.<domain>/v1/chat/completions -d '<known-bad-signature>'

# 3. Logging redaction holds: Authorization + Cookie are redacted in the
#    WAF log group (confirm no bearer tokens in the delivered logs).
```

**Acceptance (operator):** WAF WebACL attached to the live ALB in a deployed
env (the `get-web-acl-for-resource` call returns the RouteIQ WebACL). On
acceptance, `RouteIQ-4f59` closes fully (construct shipped + live attach done).

---

## Disposition summary

| Seed | RouteIQ-side (cred-free) | Operator-side (this runbook) | Closing flag / output |
|---|---|---|---|
| `RouteIQ-91fe` | `GpuNodePoolManifest` output + test | deploy NodePool, install engine, register ONE `api_base` | `routeiq:enable_gpu_nodepool` |
| `RouteIQ-28be` | architecture (doc 51) | KVBM tiers (CPU≥GPU_KV, DISK≥CPU); S3 G4; disagg only after EFA | KVBM env (`DYN_KVBM_*`) |
| `RouteIQ-2f97` | architecture (doc 51 Part 1) | EFA NodePool + device plugin, NIXL=LIBFABRIC probe, libfabric v2.5.1 pin | (forward) `routeiq:enable_efa` / hand-applied manifests |
| `RouteIQ-c299` | code + decision (doc 20 §9), default `False` | canary observe, then flip code default | `ROUTEIQ_KUMARASWAMY_THOMPSON__MOMENT_FIT` → code default `True` |
| `RouteIQ-4f59` | `WafConstruct` + 13 tests, byte-stable OFF | render ALB, capture ARN, re-deploy WAF on | `routeiq:enable_waf` + `routeiq:waf_alb_arn` |
