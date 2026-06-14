# Discover: `eks_cluster_construct.py` (vllm-sr-on-aws) — deep read for RouteIQ adaptation

> **PROVENANCE — READ FIRST.** The task asked for a VERBATIM extraction of
> `cdk/lib/eks_cluster_construct.py`. **That file does not exist on this machine.**
> It lives only in the **separate `vllm-sr-on-aws` repository** (`cdk/lib/`), which
> is not checked out here. A full-filesystem `find / -name eks_cluster_construct.py`
> returned nothing; no `vllm-sr-on-aws` checkout exists locally; the RouteIQ git
> submodules are only `reference/litellm`, `reference/LLMRouter`, `reference/NadirClaw`.
> The `undefined/` path tokens in the task were never bound to a real local path.
>
> Therefore the symbol names, block shapes, and code skeletons below are
> **reconstructed from the durable in-repo records** that the RouteIQ team distilled
> from that source during the 2026-06-14 design session, NOT copied from the source
> file:
> - `docs/adr/0030-eks-auto-mode-irsa-substrate.md` (the decision record, with line cites into the source)
> - `docs/architecture/aws-rearchitecture/vllmsr-patterns.md` (the committed construct→pattern map)
> - `docs/architecture/aws-rearchitecture/99-review-findings.md` (adversarial audit that grep-verified every cite against the real source)
> - `~/.claude/skills/cdk-gotchas/references/cdk-irsa-l1-cluster-cfnjson-token-key.md` (canonical CfnJson/IRSA pattern, authored from the same vllm-sr Phase-0 work, 2026-06-07)
> - `/tmp/vllmsr-patterns.txt` (the ephemeral scratch the patterns doc superseded)
>
> Per `verify-external-audit-before-acting` and `synthesis-agent-fabricates-untraceable-precision`:
> identifiers attested by ≥2 of those sources are marked **[attested]**; structural
> shapes that are the standard CDK API for the named feature but whose exact source
> keyword spelling I could not read are marked **[reconstructed]**. **Treat nothing
> here as byte-exact source; re-read the real file in `vllm-sr-on-aws` before
> copy-pasting.** Line cites (`:NNN`) are ADR-0030's cites into the source and were
> accurate at the 2026-06-14 snapshot; offsets drift, symbol names are stable
> (99-review-findings residual #2).

---

## 1. `class EksClusterConstruct` + `__init__` signature/params

**Status: [attested]** the class name `EksClusterConstruct` and the factory/method
names `irsa_role()` and `enable_container_insights()` are named verbatim in the
handoff doc (`docs/handoffs/2026-06-14-1211-...:180`) and patterns doc. The exact
`__init__` parameter list is **[reconstructed]** — I could not read the signature;
the params below are inferred from what the construct provably consumes (VPC, K8s
version 1.33, the node/cluster IAM roles it builds, the CfnOutputs it emits).

```python
class EksClusterConstruct(Construct):          # [attested name]
    """EKS Auto Mode cluster (L1 CfnCluster) + OIDC provider + IRSA factory.
    CDK owns cluster + IAM + OIDC; Helm/kubectl deploys the app (no
    KubernetesManifest / HelmChart in this construct)."""

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        vpc: ec2.IVpc,                          # [reconstructed] cluster lands in this VPC
        # K8s 1.33 — attested at ADR :62,155
        # cluster_role / node_role built internally (CfnAccessEntry + node_role_arn)
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)
        ...
```

The class body, per ADR-0030 cites, contains in order: the L1 `CfnCluster` with the
3 Auto-Mode blocks (`:151-210`), the OIDC provider + `oidc_provider_issuer`
derivation (`:219-240`), the `irsa_role()` factory with the CfnJson trust map
(`:261-309`), the access entries (`:311-344`), the CfnOutputs (`:242-259`), and
`enable_container_insights` (`:346-967`).

---

## 2. The three Auto Mode blocks passed to `eks.CfnCluster` (`:174-187`)

**Status: [attested]** all three blocks exist and are toggled together
(ADR-0030 :41-48; 99-review-findings lens-2 row confirms
"ComputeConfig/StorageConfig/ElasticLoadBalancing blocks" against real source).
The property-class spelling is **[reconstructed]** to the standard CDK L1 API.

```python
cfn_cluster = eks.CfnCluster(
    self, "EksCluster",
    name=cluster_name,
    version="1.33",                                    # [attested] :62,155
    role_arn=cluster_role.role_arn,                    # [reconstructed]
    resources_vpc_config=eks.CfnCluster.ResourcesVpcConfigProperty(
        subnet_ids=vpc.select_subnets(...).subnet_ids, # [reconstructed]
    ),

    # ── Auto Mode block 1: ComputeConfig (nodePools + nodeRoleArn) :175-179 [attested]
    compute_config=eks.CfnCluster.ComputeConfigProperty(
        enabled=True,
        node_pools=["general-purpose", "system"],      # [attested] the two AWS-managed pools
        node_role_arn=node_role.role_arn,              # [attested] node IAM role
    ),

    # ── Auto Mode block 2: StorageConfig.blockStorage :180-182 [attested]
    storage_config=eks.CfnCluster.StorageConfigProperty(
        block_storage=eks.CfnCluster.BlockStorageProperty(
            enabled=True,
        ),
    ),

    # ── Auto Mode block 3: KubernetesNetworkConfig.elasticLoadBalancing :183-187 [attested]
    kubernetes_network_config=eks.CfnCluster.KubernetesNetworkConfigProperty(
        elastic_load_balancing=eks.CfnCluster.ElasticLoadBalancingProperty(
            enabled=True,
        ),
    ),

    # Auto Mode supplies CoreDNS / kube-proxy / VPC-CNI → don't bootstrap self-managed addons
    bootstrap_self_managed_addons=False,               # [attested] :198

    # API auth mode; creator-admin only covers the CFN exec role, NOT a human kubectl identity
    access_config=eks.CfnCluster.AccessConfigProperty(  # [attested behavior] :192-195
        authentication_mode="API",
        bootstrap_cluster_creator_admin_permissions=True,
    ),
)
```

Key facts (all [attested] in ADR-0030):
- Auto Mode is on the **L1 `CfnCluster` only** — not the stable L2; the alpha L2 is
  breaking-change-prone (`:16-22`). This is *why* the manual CfnJson IRSA path is forced.
- `node_pools=["general-purpose","system"]` collapses the Karpenter templates RouteIQ
  would otherwise hand-author; Auto Mode manages capacity.
- `bootstrap_self_managed_addons=False` because Auto Mode supplies the core addons.

---

## 3. `irsa_role(namespace, service_account, ...)` factory + CfnJson trust-key (`:261-309`)

**Status: [attested]** factory name `irsa_role`, signature head
`irsa_role(namespace, service_account, ...)`, the `CfnJson` wrap, the
`<oidc>:sub`/`<oidc>:aud` map keys, and the `WebIdentityPrincipal` are all attested
across ADR-0030 (:56-89), 99-review-findings (lens-2), and the
`cdk-irsa-l1-cluster-cfnjson-token-key` reference (which was authored from the same
2026-06-07 vllm-sr Phase-0 IRSA wiring). Variable names (`string_equals`, `cid`,
`oidc`) match the ADR's inlined snippet and the cdk-gotchas reference verbatim.

```python
def irsa_role(self, namespace: str, service_account: str, *,
              cid: str, managed_policy_statements=None) -> iam.Role:
    oidc = self.oidc_provider_issuer            # [attested] ARN-derived, scheme-stripped (NOT .replace)
    ns, sa = namespace, service_account

    # The load-bearing gotcha (:275-300): CFN map KEYS must be literal strings at
    # synth. The OIDC issuer is an Fn::GetAtt token → cannot be a key. Wrap the
    # token-keyed inner map in CfnJson so keys resolve at DEPLOY time.
    string_equals = CfnJson(self, f"{cid}TrustKeys", value={   # [attested]
        f"{oidc}:aud": "sts.amazonaws.com",                    # [attested]
        f"{oidc}:sub": f"system:serviceaccount:{ns}:{sa}",     # [attested]
    })

    role = iam.Role(self, cid, assumed_by=iam.WebIdentityPrincipal(  # [attested]
        self.oidc_provider_arn,                                      # provider ARN, not issuer
        conditions={"StringEquals": string_equals},                 # CfnJson IS the value [attested]
    ))
    # role then grants per-workload AWS perms (see §7); SA annotated out-of-band
    return role
```

Note `StringEquals` is a **literal operator key** (stays a normal dict key); only
its **value** (the token-keyed map) is the `CfnJson`. Wrapping the whole `conditions`
dict would be wrong — wrap the smallest token-keyed sub-map.

---

## 4. OIDC provider creation (`:219-224`)

**Status: [attested]** `iam.OpenIdConnectProvider`, `client_ids=["sts.amazonaws.com"]`,
and "issuer attribute" sourcing are all attested (ADR :58-60).

```python
oidc_provider = iam.OpenIdConnectProvider(          # [attested]
    self, "OidcProvider",
    url=<cluster issuer attribute>,                 # [attested] from cfn_cluster issuer attr
    client_ids=["sts.amazonaws.com"],               # [attested]
)
self.oidc_provider_arn = oidc_provider.open_id_connect_provider_arn  # [reconstructed name]
```

---

## 5. `oidc_provider_issuer` — the ARN-derived attribute (`:226-240`)

**Status: [attested]** the attribute name `oidc_provider_issuer` and the phrase
"scheme-stripped, ARN-derived issuer" are attested (ADR :60). The exact derivation
expression is **[reconstructed]**: the lesson is that the correct value is derived
from the **OIDC provider ARN** (which is a concrete string post-creation /
`Fn::Select`-sliceable), NOT from `issuer_url.replace("https://","")` on the raw
cluster issuer attribute (that `.replace` is a no-op on a token — see §8a).

```python
# CORRECT: derive the scheme-less issuer from the provider ARN (deploy-resolvable),
# expose it as the construct's public attribute used by irsa_role():
self.oidc_provider_issuer = <ARN-derived scheme-less issuer>   # [attested name] :226-240
```

`irsa_role()` (§3) reads exactly this attribute for the trust-map keys.

---

## 6. Access entries — `CfnAccessEntry` (`:311-344`)

**Status: [attested]** `CfnAccessEntry`, and the rationale that the creator-admin
flag covers only the CFN exec role (not a human/CI kubectl identity), are attested
(ADR :52-54, :311-344).

```python
# One explicit access entry per operator/CI identity — bootstrap_cluster_creator_admin_permissions
# does NOT grant kubectl access to a human or the CI role.
eks.CfnAccessEntry(self, "OperatorAccessEntry",     # [attested type]
    cluster_name=cfn_cluster.name,
    principal_arn=<operator/CI role arn>,           # [reconstructed]
    # + access policy association granting cluster-admin / namespace scope [reconstructed]
)
```

---

## 7. `enable_container_insights` (`:346-967`)

**Status: [attested]** method name `enable_container_insights`, the
`amazon-cloudwatch-observability` addon, a CDK-created routing log group, the
per-model metric filter `routing_latency_ms_by_model` with
`dimensions={"model":"$.selected_model"}` (`:757-767`), 7 alarms, a dashboard, an
SNS topic. The metric-filter dimension is **double-attested** (ADR-0030 :99-105 +
ADR-0027 :62-71 + 99-review-findings lens-2 row).

```python
def enable_container_insights(self) -> None:        # [attested name]
    eks.CfnAddon(self, "CwObservability",
        cluster_name=..., addon_name="amazon-cloudwatch-observability")  # [attested addon]
    log_group = logs.LogGroup(self, "RoutingLogGroup", ...)  # [attested] CDK-created, MUST pre-exist filter
    logs.MetricFilter(self, "RoutingLatencyByModel",         # [attested] :757-767
        log_group=log_group,
        metric_name="routing_latency_ms_by_model",
        dimensions={"model": "$.selected_model"},            # [attested] per-model dimension
        ...)
    # + 7 alarms, 1 dashboard, 1 SNS topic   [attested counts]
```

**Live-ops trap (vllmsr-patterns #8):** the CW MetricFilter needs a **CDK-created log
group that pre-exists**. A runtime-created (Fluent Bit) log group makes the
`CFN MetricFilter` fail on first deploy. Create the log group in CDK, not at runtime.

---

## 8. The two silent-failure traps (FLAGGED)

### Trap 8a — `.replace("https://")` is a silent no-op on an unresolved CFN token (`:231-237`)

**Status: [attested], root-caused live 2026-06-08.** `issuer_url.replace("https://","")`
on the raw cluster issuer attribute does **nothing** because the attribute is an
unresolved CloudFormation token (an `Fn::Join`/`Fn::GetAtt`), not a Python string —
`.replace` silently returns the token unchanged. The trust-map keys then become
`https://oidc...:sub` / `https://oidc...:aud`, and `AssumeRoleWithWebIdentity` fails
with **AccessDenied** at runtime (synth is green; failure is at pod start). **Fix:
always use the ARN-derived `oidc_provider_issuer` (§5), never `.replace` on the URL.**
This is the same family as the generic token-`.replace`/`.split` silent-noop in the
`cdk-gotchas` skill.

### Trap 8b — L2 `cluster.addServiceAccount()` would hide all of this, but L1 forbids it (`:281-286`)

**Status: [attested].** On the stable L2 `eks.Cluster`, `cluster.addServiceAccount()`
builds the `OpenIdConnectProvider` + the `CfnJson` trust map for you, hiding the
entire CfnJson dance. But Auto Mode forces the **L1 `CfnCluster`**, where
`addServiceAccount` does not exist — so the construct **must** hand-wire OIDC + the
CfnJson trust map (§3) manually. The hidden L2 path is a trap because it tempts you
to "just use addServiceAccount" and abandon Auto Mode. Keep L1 + manual CfnJson.

---

## 9. What RouteIQ MUST ADAPT for a SINGLE stateless pod (one IRSA role, not three)

vllm-sr-on-aws mints **three** IRSA roles (router→bedrock-mantle, EAIG gateway→native
Bedrock, bearer-minter) for its multi-workload, cross-account, bearer-token topology
(per `vllm_sr_eks_stack.py` in vllmsr-patterns.md:22). RouteIQ is **one stateless
gateway pod**. Adaptations:

1. **One IRSA role, not three.** Call `irsa_role(namespace, service_account)` **once**
   for RouteIQ's single ServiceAccount (the chart's `serviceaccount.yaml`). Drop the
   router/EAIG/bearer-minter split entirely. No bearer-minter role, no ≤12h bearer
   rotation CronJob (vllmsr lesson #5 does not apply).

2. **Collapse the cross-account capacity loop.** vllm-sr's per-account
   `CfnOutput`/web-identity-or-ArnPrincipal trust (`bedrock_capacity_member_stack.py`,
   separate-app, not StackSet) is multi-account capacity machinery RouteIQ does not
   need. Single account, single trust path (WebIdentityPrincipal → the one OIDC
   provider).

3. **The one role's grants** = exactly what RouteIQ's peer ADRs require, on a single
   role: `aps:RemoteWrite` (ADR-0027 AMP), `rds-db:connect` (ADR-0028 Aurora),
   `elasticache:Connect` (ADR-0029 Valkey), AppConfig read (ADR-0026), `bedrock:InvokeModel*`.
   No bedrock-mantle vs native-Bedrock distinction.

4. **Keep stateless → sidestep the RWO-EBS deadlock.** vllmsr lesson #3
   (RWO-EBS + RollingUpdate → Multi-Attach deadlock → `strategy: Recreate`) is moot
   *if RouteIQ stays stateless* (state lives in Aurora/ElastiCache per ADR-0028/0029).
   This is the safer default: keep the pod stateless and use plain RollingUpdate. Only
   if any Deployment ever mounts an RWO EBS PVC must it switch to `strategy.type: Recreate`.

5. **Keep the cluster machinery 1:1.** The L1 `CfnCluster` + 3 Auto-Mode blocks, OIDC
   provider, `oidc_provider_issuer`, CfnJson trust-key pattern, access entries, and
   `enable_container_insights` port **unchanged** — they are per-cluster, not
   per-workload. Only the IRSA **count** drops 3→1.

6. **CDK/Helm boundary stays.** No `KubernetesManifest`/`HelmChart` in the construct.
   Emit CfnOutputs (`ClusterName`, `ClusterEndpoint`, `OidcProviderArn`,
   `OidcProviderIssuerUrl`, the single IRSA role ARN); RouteIQ's existing Helm chart
   consumes them. Annotate the chart's `serviceaccount.yaml` with the role ARN
   (`eks.amazonaws.com/role-arn`) via `helm upgrade`/`envsubst` — **never `kubectl
   apply -k`**, which clobbers the SA annotation (vllmsr lesson #6).

7. **Chart-layer foot-guns to encode (app layer, not CDK):** NLB CIDR-lock is
   `spec.loadBalancerSourceRanges` in `service.yaml`, NOT the legacy
   `service.beta.kubernetes.io/...` annotation (ignored under Auto Mode).

---

## 10. Bottom line

- The requested file is **not present locally** — it is cross-repo (`vllm-sr-on-aws`).
  This doc is a faithful **reconstruction from in-repo distillations**, with every
  identifier tagged [attested]/[reconstructed]. **Re-read the real source before
  copying any code.**
- Symbols you can trust to be real (≥2 independent in-repo attestations):
  `EksClusterConstruct`, `irsa_role(namespace, service_account, ...)`,
  `enable_container_insights`, `oidc_provider_issuer`, the `CfnJson` trust-key with
  `<oidc>:sub`/`<oidc>:aud`, `WebIdentityPrincipal`, `CfnAccessEntry`, the 3 Auto-Mode
  blocks, `node_pools=["general-purpose","system"]`,
  `bootstrap_self_managed_addons=False`, `routing_latency_ms_by_model` /
  `dimensions={"model":"$.selected_model"}`.
- Both silent-failure traps confirmed: (8a) `.replace("https://")` no-op on a token →
  runtime `AssumeRoleWithWebIdentity` AccessDenied; (8b) L2 `addServiceAccount` hidden
  path that L1/Auto-Mode forbids.
- RouteIQ adaptation: **one IRSA role, not three**; drop cross-account/bearer-minter;
  keep the cluster construct 1:1; stay stateless to skip RWO+Recreate; Helm/envsubst
  the SA annotation, never `apply -k`.
