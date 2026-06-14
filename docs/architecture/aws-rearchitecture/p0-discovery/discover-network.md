# discover-network — Deep read of `network_construct.py`

> **Source of truth.** The file `lib/network_construct.py` does **not yet exist in the
> RouteIQ repo.** It is a P0 **port** from vllm-sr-on-aws, cited by ADR-0030 / doc 10
> (`10-aws-native-target-architecture.md:147`) and the P0 work queue
> (`30-migration-roadmap.md:38`, handoff `2026-06-14-1211-...md:182`).
> The canonical file read for this discovery is:
> `/Users/baladita/Documents/DevBox/vllm-sr-on-aws/cdk/lib/network_construct.py` (215 LOC).
> Everything below is extracted verbatim from that file unless marked "RouteIQ mapping".

---

## 1. `NetworkConstruct.__init__` signature

```python
class NetworkConstruct(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,                       # keyword-only, required
        vpc_cidr: str = "10.20.0.0/16",      # keyword-only, default /16
        nat_gateways: int = 1,               # keyword-only, default 1
        **kwargs,
    ) -> None:
```

- `scope`, `construct_id` — standard CDK positional `Construct` args.
- All tunables are **keyword-only** (`*` separator): `env_name`, `vpc_cidr`, `nat_gateways`.
- `self.env_name = env_name` is stored; otherwise `env_name` is not used inside the
  construct (no resource names embed it — they rely on CDK logical IDs).

### Public attributes (frozen interface contract per the module docstring §4)

| Attribute | Type | Meaning |
|---|---|---|
| `self.vpc` | `ec2.Vpc` | the VPC |
| `self.public_subnets` | `list[ISubnet]` | public tier |
| `self.private_app_subnets` | `list[ISubnet]` | `private-app` (PRIVATE_WITH_EGRESS) |
| `self.private_data_subnets` | `list[ISubnet]` | `private-data` (PRIVATE_ISOLATED) |
| `self.alb_sg` / `sr_task_sg` / `vpce_sg` / `efs_sg` / `redis_sg` / `milvus_sg` / `backend_vllm_sg` | `ec2.SecurityGroup` | the 7 SGs |
| `self.interface_endpoints` | `dict[str, InterfaceVpcEndpoint]` | keyed by construct id (11 entries) |
| `self.s3_gateway_endpoint` | gateway endpoint | S3 gateway |
| `self.private_dns_namespace` | `servicediscovery.PrivateDnsNamespace` | Cloud Map `sr.internal` |

---

## 2. VPC config

| Knob | Value |
|---|---|
| CIDR | `vpc_cidr` default **`10.20.0.0/16`** (CDK `ec2.IpAddresses.cidr(vpc_cidr)`) |
| AZs | **`max_azs=2`** (multi-AZ, 2 zones) |
| NAT gateways | **`nat_gateways=1`** (default; one shared NAT — cost/HA tradeoff knob) |
| DNS | `enable_dns_hostnames=True`, `enable_dns_support=True` (required for `private_dns_enabled` endpoints) |

### Subnet tiers (`subnet_configuration`, all `cidr_mask=24`)

| Name | Type | Purpose |
|---|---|---|
| `public` | `PUBLIC` | ALB / NAT |
| `private-app` | `PRIVATE_WITH_EGRESS` | Fargate/EKS task ENIs + **all 11 interface endpoints land here** |
| `private-data` | `PRIVATE_ISOLATED` | data tier (no egress) |

> **RouteIQ mapping.** Doc 10 §2 notes the EKS stack owns its own VPC pattern at
> **`10.30/16`** (`vllm_sr_eks_stack.py`) — i.e. on the RouteIQ port the CIDR/AZ/NAT
> values are the parameters to re-confirm. For a single stateless pod, `nat_gateways=1`
> + `max_azs=2` is the floor; bump NAT to 2+ for prod AZ-resilient egress.

---

## 3. Security groups (all `allow_all_outbound=True`)

7 SGs are instantiated **before** any ingress rule is wired (deliberate, to avoid the
CFN SG-to-SG circular-dependency trap — comment cites mulch
`cdk-nested-stack-sg-circular-dependency`). Descriptions are constrained to the EC2
charset allowlist (mulch `ec2-sg-description-charset` — no arrows/em-dashes/backticks).

| SG | `allow_all_outbound` | Ingress rule(s) wired in `_wire_sg_ingress()` |
|---|---|---|
| **`alb_sg`** | yes | TCP **443 from `Peer.any_ipv4()`** (HTTPS from internet; narrowable to org allowlist via future context flag) |
| **`sr_task_sg`** | yes | TCP **8080 from `alb_sg`** (Envoy HTTP); TCP **8700 from `alb_sg`** (dashboard HTTP) |
| **`vpce_sg`** | yes | TCP **443 from `sr_task_sg`** (this is the SG shared by all 11 interface endpoints) |
| **`efs_sg`** | yes | TCP **2049 (NFS) from `sr_task_sg`** |
| **`redis_sg`** (placeholder) | yes | TCP **6379 from `sr_task_sg`** (no resource at v1) |
| **`milvus_sg`** (placeholder) | yes | TCP **19530 from `sr_task_sg`** (no resource at v1) |
| **`backend_vllm_sg`** (placeholder) | yes | **`Port.all_tcp()` from `sr_task_sg`** (port-agnostic placeholder for self-hosted backends) |

> **RouteIQ mapping.** `sr_task_sg` → the RouteIQ pod ENI SG. RouteIQ's container listens
> on a single uvicorn port (default 4000), not Envoy 8080/8700 — re-point the `alb_sg → task`
> rule to the pod port. `efs_sg`/`milvus_sg`/`backend_vllm_sg` are NOT needed (RouteIQ is
> stateless, `emptyDir` only, no self-hosted backends). `vpce_sg` (443 from task) is the
> load-bearing one — it gates the interface endpoints below. `redis_sg` becomes the
> ElastiCache SG (P1, ADR-0029) but on Valkey-serverless the endpoint port is 6379 too.

---

## 4. VPC endpoints (full list)

### 4.1 Interface endpoints — **eleven**, in `_make_interface_endpoints()`

All share `vpce_sg`, all land in `PRIVATE_WITH_EGRESS` subnets, all
`private_dns_enabled=True`. Keyed in `self.interface_endpoints` by the construct id.

| # | Construct id | CDK service enum | RouteIQ-relevant? |
|---|---|---|---|
| 1 | `EcrApi` | `ECR` | **YES — load-bearing** (pull control plane) |
| 2 | `EcrDocker` | `ECR_DOCKER` | **YES — load-bearing** (image layer pull) |
| 3 | `CloudWatchLogs` | `CLOUDWATCH_LOGS` | **YES — load-bearing** (pod logs) |
| 4 | `SecretsManager` | `SECRETS_MANAGER` | **YES — load-bearing** (master/provider keys) |
| 5 | `Sts` | `STS` | **YES — load-bearing** (IRSA AssumeRoleWithWebIdentity) |
| 6 | `AppConfig` | `APPCONFIG` | YES (P2) — config control plane |
| 7 | `AppConfigData` | `APPCONFIGDATA` | YES (P2) — config **data/poll** plane |
| 8 | `Amp` | `PROMETHEUS_WORKSPACES` | YES (P2) — `aps-workspaces` remote-write |
| 9 | `XRay` | `XRAY` | P2 — only if X-Ray tracing on |
| 10 | **`BedrockRuntime`** | **`BEDROCK_RUNTIME`** | **YES — load-bearing CONFIRMED PRESENT** (primary inference backend) |
| 11 | `SagemakerRuntime` | `SAGEMAKER_RUNTIME` | NO — VSR-specific; RouteIQ does not invoke SageMaker endpoints |

> **`BEDROCK_RUNTIME` confirmation.** Present at line 200:
> `("BedrockRuntime", ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME)`. The module
> docstring (§lines 8-11) explicitly justifies it: provisioned per "doc 02 §4.4 / 12 §F-15
> … when Bedrock is the primary inference backend (vllm-sr-on-aws-4263)." Enum uses
> snake_case per mulch `mx-b66ef9`; requires `aws-cdk-lib >= 2.150.0`.

### 4.2 Gateway endpoint — **one**: S3 (`self.s3_gateway_endpoint`)

```python
self.s3_gateway_endpoint = self.vpc.add_gateway_endpoint(
    "S3Gateway",
    service=ec2.GatewayVpcEndpointAwsService.S3,
    subnets=[PRIVATE_WITH_EGRESS, PRIVATE_ISOLATED],   # route-table assoc on both private tiers
)
```
Gateway (route-table) endpoint, not an ENI — no SG, free. **Load-bearing for RouteIQ**
(ECR image layers pull from S3; config download; data-lake writes).

### 4.3 Cloud Map — `self.private_dns_namespace`

`servicediscovery.PrivateDnsNamespace` named **`sr.internal`**, attached to the VPC,
described "Cloud Map namespace for self-hosted vLLM backends." **VSR-specific; not needed
by RouteIQ** (no self-hosted backends — RouteIQ calls Bedrock/providers, no internal
service-discovery mesh). Rename/drop on the port.

---

## 5. MINIMAL subset for RouteIQ's single stateless pod

RouteIQ runs **one stateless pod** (`emptyDir` only, no PVC) on EKS Auto Mode in a
**private** subnet. With no public egress (or NAT-restricted), every AWS API the pod
touches must reach a VPC endpoint or it hangs. The minimal load-bearing set, each mapped
to its P-tier and ADR:

| Endpoint (this file's name) | Why the stateless pod needs it | Endpoint type | P-tier | ADR |
|---|---|---|---|---|
| **ECR api** (`EcrApi` / `ECR`) | image auth/metadata to start the pod | interface | **P0** | ADR-0030 (EKS substrate; node pulls image) |
| **ECR dkr** (`EcrDocker` / `ECR_DOCKER`) | pull container layers | interface | **P0** | ADR-0030 |
| **S3** (gateway) | ECR layer blobs (+ config download, data-lake writes) | **gateway** | **P0** | ADR-0030 (image pull); ADR-0026/0027 (config/lake reuse) |
| **STS** (`Sts` / `STS`) | IRSA `AssumeRoleWithWebIdentity` — the pod gets its creds here; nothing else works without it | interface | **P0** | ADR-0030 (IRSA factory) |
| **CloudWatch Logs** (`CLOUDWATCH_LOGS`) | pod/container log shipping; routing_decision log lines feed MetricFilters | interface | **P0** | ADR-0030 (substrate); ADR-0027 (metric filters consume them) |
| **Bedrock runtime** (`BedrockRuntime` / `BEDROCK_RUNTIME`) | **the data plane** — RouteIQ's primary inference backend; the gpt-5.5/Responses edge requires it | interface | **P0** (data-plane critical) | doc 10 §2 / handoff §7 (P0 IRSA pod role = "Bedrock invoke"); inference target |
| **Secrets Manager** (`SECRETS_MANAGER`) | master key + provider keys + (P1) Aurora/cache creds via External Secrets Operator | interface | **P0** for app secrets; **P1** for DB/cache creds | ADR-0030 (P0 pod role: Secrets); ADR-0028/0029 (P1 rotated creds) |
| **AppConfig** (`APPCONFIG`) | config control-plane (create/deploy) | interface | **P2** | ADR-0026 |
| **AppConfig Data** (`APPCONFIGDATA`) | config **poll** plane — the running pod reads strategy/A-B config | interface | **P2** | ADR-0026 |
| **aps-workspaces** (`Amp` / `PROMETHEUS_WORKSPACES`) | ADOT sidecar `remote_write` to AMP; IRSA needs `aps:RemoteWrite` | interface | **P2** | ADR-0027 |
| **ElastiCache** (Valkey serverless) | session affinity / circuit-breaker / rate-limit / L2 cache | **NO endpoint — in-VPC resource** reached via `redis_sg` on 6379 | **P1** | ADR-0029 |
| **RDS / Aurora PG** | governance keys/spend/workspaces, budgets, (P3) bandit posteriors | **NO endpoint — in-VPC resource** reached via DB SG on 5432 | **P1** | ADR-0028 |

### Key distinctions for the port

- **ElastiCache and RDS do NOT get interface endpoints.** They are VPC-resident resources
  (ENIs in the private subnets), reached directly over the SG matrix on 6379 / 5432 — not
  via a `com.amazonaws.*` interface endpoint. They are P1 (ADR-0028 / ADR-0029), not part
  of `network_construct.py`'s endpoint list. RouteIQ adds their SGs (`redis_sg` is the
  placeholder seed for ElastiCache; add an `aurora_sg` for the DB).
- **rds-db IAM auth** uses STS-issued tokens (`rds-db:connect`) — covered by the STS
  endpoint already; no separate "rds" interface endpoint required.
- **Drop two from the VSR list on the RouteIQ port:** `SagemakerRuntime` (RouteIQ never
  calls SageMaker) and the **`sr.internal` Cloud Map namespace** (no self-hosted backends).
  `XRay` is optional (keep only if X-Ray tracing is enabled; AMP/AMG is the primary
  observability path per ADR-0027).
- **P0-minimum (just-boots-and-serves-Bedrock) endpoint set = 6:** ECR api, ECR dkr,
  S3 gateway, STS, CloudWatch Logs, Bedrock runtime, plus Secrets Manager for app keys.
  AppConfig×2 + AMP arrive with P2; Aurora/ElastiCache are in-VPC P1 resources, not
  endpoints.

### IRSA pod-role policy implied (one role, per handoff §10 / roadmap P0 §44-46)

`ecr:*pull*` + `s3:Get*` (layers/config) + `sts:AssumeRoleWithWebIdentity` (the trust
itself) + `logs:*` + `bedrock:InvokeModel*` + `secretsmanager:GetSecretValue` (P0);
then `appconfig:*`/`appconfigdata:*` + `aps:RemoteWrite` (P2) + `rds-db:connect` +
`elasticache:Connect` (P1).

---

## 6. Notable porting gotchas (carried from the source file + ADRs)

1. **SG charset** — every SG description must stay in the EC2 allowlist
   (`a-zA-Z0-9 . _ - : / ( ) # , @ [ ] + = & ; { } ! $ *`); no arrows/em-dashes/backticks
   /pipes/angle-brackets/question-marks (mulch `ec2-sg-description-charset`).
2. **SG circular dependency** — instantiate all SGs first, wire ingress after (the file's
   two-phase pattern); RouteIQ must keep this when adding the Aurora SG.
3. **`BEDROCK_RUNTIME` enum** needs `aws-cdk-lib >= 2.150.0` (handoff pins
   `>=2.150.0,<3`).
4. **`private_dns_enabled=True` requires** `enable_dns_hostnames` + `enable_dns_support`
   on the VPC (both set here) — otherwise the Bedrock SDK won't resolve to the endpoint.
5. **Subnet placement** — all interface endpoints + the Fargate/pod ENIs live in
   `PRIVATE_WITH_EGRESS`; the S3 gateway route-table assoc covers both private tiers.
