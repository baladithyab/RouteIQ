# AWS AgentCore **Registry** ↔ RouteIQ/LiteLLM — Deep-Dive Integration Analysis

> **As of 2026-06-15.** Companion to the sibling Gateway doc
> [`agentcore-integration-and-arch-2026-06-15.md`](./agentcore-integration-and-arch-2026-06-15.md),
> which covered the AgentCore **Gateway** (Direction A = consume / Direction C =
> model-backend / Direction B = federate) and noted that the **Registry** is a *distinct*
> service. This doc goes deep on the **Registry only** — the discovery/governance catalog,
> not the runtime tool-call plane.
>
> **Honesty contract.** Claims are tagged **[VERIFIED]** (present in an official AWS doc
> read in-session, or present on disk in this repo) or **[SPECULATIVE]** (reasoned
> synthesis / inferred from API semantics, not shown end-to-end in one source). The
> authority for AgentCore Registry facts is the set of AWS docs read on 2026-06-15:
> `create-registry-record` (CLI v2.35.4 + boto3), `API_Descriptors`,
> `registry-mcp-endpoint` devguide, `create-registry` (CLI),
> `API_SubmitRegistryRecordForApproval`, and the 2026-04-09 launch blog. The LiteLLM facts
> are grounded in `reference/litellm/` on disk (pinned `litellm==1.82.3`).

---

## 0. TL;DR — the one-paragraph answer

**Yes, the AgentCore Registry can integrate with RouteIQ/LiteLLM — but it integrates
differently than the Gateway, and the two directions are NOT symmetric in cost.** The
Registry is a **discovery/governance catalog**, not a runtime tool-call plane. It is
*itself* an MCP server, but a deliberately minimal one: its MCP endpoint exposes **exactly
one tool, `search_registry_records`** — a *search* surface, not a `tools/list` of the
actual tools the records describe. This single fact reshapes both directions:

- **R1 (RouteIQ PUBLISHES to the registry)** is the **higher-value, smaller, and more
  honestly-shippable** direction. RouteIQ owns a real MCP surface and a real model catalog;
  publishing one `descriptorType: MCP` record (synced `fromUrl` to RouteIQ's MCP endpoint)
  makes RouteIQ's tools org-discoverable with approval-workflow governance. **Ranked
  R1 > R2.**
- **R2 (RouteIQ CONSUMES the registry)** is real but **lower-value and blocked by an auth
  gap**: the registry MCP endpoint's only tool returns *metadata about records*, so a
  consumer must do **two hops** (search the registry → then separately register each
  discovered MCP server's `fromUrl` into LiteLLM). And the consume path hits a hard
  constraint: **LiteLLM 1.82.3's MCP client has no SigV4 auth_type** (enum is
  `none|api_key|bearer_token|basic|oauth2`), so an **IAM-authorizer** registry is not
  natively reachable — you need a **CUSTOM_JWT (OAuth)** registry or the
  `mcp-proxy-for-aws` stdio bridge.

**Smallest shippable: R1, manual `create-registry-record` of RouteIQ's MCP endpoint as a
`descriptorType: MCP` record with `synchronizationConfiguration.fromUrl` → RouteIQ's
`/v1/mcp` URL, in an `AWS_IAM`-authorizer registry, `autoApproval` off.** No RouteIQ code
change; a CLI call plus a verified MCP URL.

---

## 1. What EXACTLY is the registry record model? [VERIFIED]

The Registry (AWS Agent Registry, preview, announced **2026-04-09**, GA-pending) lives in
the **`bedrock-agentcore-control`** API surface (the control plane; API version
`2023-06-05`). Two resource tiers:

### 1.1 The Registry (`CreateRegistry`)

```bash
aws bedrock-agentcore-control create-registry \
  --name <unique_per_account, alphanumeric + underscore> \
  [--description ...] \
  [--authorizer-type CUSTOM_JWT | AWS_IAM] \
  [--authorizer-configuration ...]   # required iff CUSTOM_JWT \
  [--approval-configuration '{"autoApproval": false}']
```

- **Response: `{ "registryArn": "string" }` only** — there is no `registryId` field in the
  response; you parse the id out of the ARN (`.../registry/<id>`) or pass the ARN
  everywhere (every API accepts ARN-or-id). **[VERIFIED — boto3 `create_registry`]**
- **`authorizerType` governs ONLY the consumer-facing Search + Invoke APIs (the registry's
  MCP endpoint), NOT the admin CRUDL APIs.** Verbatim from the doc: *"This controls the
  authorization method for the Search and Invoke APIs used by consumers, and does not affect
  the standard CRUDL APIs for registry and registry record management used by
  administrators."* Admin CRUDL is always plain IAM/SigV4 against
  `bedrock-agentcore-control`. **[VERIFIED]**
  - `AWS_IAM` → consumers (incl. the MCP endpoint) authenticate with SigV4 +
    `bedrock-agentcore:InvokeRegistryMcp` / `:SearchRegistryRecords` IAM actions.
  - `CUSTOM_JWT` → consumers authenticate with an OAuth bearer token via a
    `customJWTAuthorizer` (discoveryUrl + allowedAudience/allowedClients/allowedScopes,
    optional VPC-Lattice private endpoint). This is the path that lets teams build discovery
    UIs / connect IDE MCP clients **without IAM creds**.
- **`approvalConfiguration.autoApproval`** (boolean, default **`false`**) — when false,
  every record must be explicitly submitted + approved before it is discoverable.

### 1.2 The Record (`CreateRegistryRecord`) — the core model [VERIFIED]

```bash
aws bedrock-agentcore-control create-registry-record \
  --registry-id <arn-or-id> \
  --name <[a-zA-Z0-9][a-zA-Z0-9_\-\.\/]*, <=255> \
  [--description <=4096] \
  --descriptor-type MCP | A2A | CUSTOM | AGENT_SKILLS \
  [--descriptors <type-specific structure>] \
  [--record-version <[a-zA-Z0-9.-]+>] \
  [--synchronization-type URL]          # NOTE: enum quirk, see §5 \
  [--synchronization-configuration <fromUrl ...>] \
  [--client-token <idempotency>]
```

- **Asynchronous**: returns **HTTP 202** with
  `{ "recordArn": "string", "status": "CREATING" }`. The full status machine is
  `DRAFT | PENDING_APPROVAL | APPROVED | REJECTED | DEPRECATED | CREATING | UPDATING |
  CREATE_FAILED | UPDATE_FAILED`. **[VERIFIED — boto3 `create_registry_record` response]**
- **`descriptorType`** picks exactly one populated branch of the `descriptors` structure
  (`API_Descriptors`: *"Only the field that matches the record's descriptorType is
  populated"*). **[VERIFIED]**

#### The `descriptors` payload by type [VERIFIED]

Every inline content blob is a JSON/markdown **string**, min 1 / **max 102400 chars**:

| `descriptorType` | `descriptors` branch | Sub-fields |
|---|---|---|
| **MCP** | `mcp` | `server={schemaVersion, inlineContent}` (the MCP server definition, MCP-spec JSON) + `tools={protocolVersion, inlineContent}` (the tools list, MCP-spec JSON). Both `schemaVersion`/`protocolVersion` auto-detected from content if omitted. |
| **A2A** | `a2a` | `agentCard={schemaVersion, inlineContent}` (A2A agent-card JSON). |
| **CUSTOM** | `custom` | `inlineContent` (any JSON document — "APIs, Lambda functions, or servers not conforming to a standard protocol"). |
| **AGENT_SKILLS** | `agentSkills` | `skillMd={inlineContent}` (human-readable markdown) + `skillDefinition={schemaVersion, inlineContent}` (structured JSON). |

#### `synchronizationConfiguration.fromUrl` — the auto-pull mechanism [VERIFIED]

This is the registry's killer convenience: instead of hand-writing `descriptors`, you
**point at a live MCP/A2A endpoint and the registry pulls the metadata itself**.

```jsonc
{
  "fromUrl": {
    "url": "https://...",                          // REQUIRED, pattern: https://.*  (HTTPS only, <=2048)
    "credentialProviderConfigurations": [          // 0 or 1 entry (max 1!)
      {
        "credentialProviderType": "OAUTH" | "IAM",
        "credentialProvider": {                    // tagged union — exactly one branch
          "oauthCredentialProvider": {
            "providerArn": "arn:aws...:bedrock-agentcore:...",   // an AgentCore OAuth credential-provider resource
            "grantType": "CLIENT_CREDENTIALS",     // only value supported
            "scopes": [...], "customParameters": {...}
          },
          "iamCredentialProvider": {
            "roleArn": "arn:aws:iam::<acct>:role/...",  // role the registry ASSUMES to sign
            "service": "bedrock-agentcore",             // SigV4 signing service (or execute-api, etc.)
            "region": "us-west-2"                       // else inferred from URL hostname
          }
        }
      }
    ]
  }
}
```

Semantics: the registry periodically connects to `url` (signing with the supplied creds if
the target requires auth), runs MCP discovery, and refreshes the record's `descriptors`.
This is how a record stays current without manual re-publish. **[VERIFIED]**

### 1.3 The registry's OWN MCP endpoint [VERIFIED — this is the load-bearing fact]

Each registry exposes a fixed-shape MCP endpoint (**MCP spec 2025-11-25**, the same protocol
version RouteIQ + AgentCore Gateway speak):

```
https://bedrock-agentcore.<region>.amazonaws.com/registry/<registryId>/mcp
```

**It exposes exactly ONE tool** — not the tools of the records it holds:

```
Tool name:   search_registry_records
Description: Searches for registry records using natural language queries.
             Returns METADATA for matching records.
Params:      searchQuery (required, string)
             maxResults  (1-20, default 10)
             filter      (JSON: $eq,$ne,$in + $and,$or on name|descriptorType|version)
                         e.g. {"descriptorType": {"$eq": "MCP"}}
```

**Critical distinction:** `tools/call search_registry_records` returns **records' metadata**
(descriptors, ownership, how-to-invoke), it does **NOT** proxy or list the underlying tools
the way an AgentCore *Gateway* does. The registry is a **catalog you query**, not a
**gateway you call through**. Search uses hybrid keyword + (for longer NL queries) semantic
matching. **[VERIFIED — registry-mcp-endpoint devguide + launch blog]**

Auth on the MCP endpoint mirrors the registry's `authorizerType`:

- **`AWS_IAM`** → SigV4; IAM actions `bedrock-agentcore:InvokeRegistryMcp` (init +
  `tools/list`) and additionally `:SearchRegistryRecords` (for `tools/call`). AWS's own
  recommended client for this is the **`mcp-proxy-for-aws`** stdio shim
  (`uvx mcp-proxy-for-aws@latest <url> --service bedrock-agentcore --region <r>`), because
  raw MCP clients can't SigV4-sign. **[VERIFIED]**
- **`CUSTOM_JWT`** → OAuth bearer. Discoverable via
  `.well-known/oauth-protected-resource/registry/<id>/mcp` or the `WWW-Authenticate` header.
  Three client patterns: static bearer token, pre-registered OAuth client (`allowedClients`),
  or dynamic client registration (`allowedAudience`). **[VERIFIED]**

---

## 2. DIRECTION R1 — RouteIQ/LiteLLM **PUBLISHES** to the registry (RANK 1)

**Goal:** make RouteIQ discoverable org-wide. Other teams' agents/IDEs find "the RouteIQ
gateway" (and its tools, or its model catalog) by searching the registry, with
approval-workflow governance gating what becomes visible.

### 2.1 Which RouteIQ surface to publish — and as what `descriptorType`

| Candidate RouteIQ surface | descriptorType | Verdict |
|---|---|---|
| **RouteIQ's MCP endpoint** (`/v1/mcp`, seam (b) = upstream LiteLLM `global_mcp_server_manager`, real MCP JSON-RPC over streamable-HTTP) | **`MCP`** | **BEST.** This is a real MCP server. A `descriptorType: MCP` record with `fromUrl → <RouteIQ>/v1/mcp` lets the registry auto-pull RouteIQ's server def + tools list. Records stay current as RouteIQ's tool set changes. **[VERIFIED that RouteIQ has this surface — `routes/mcp.py` + ADR-0017; SPECULATIVE that the exact external path is `/v1/mcp` until §3-style empirical path confirmation runs]** |
| **RouteIQ's `/v1` model catalog** (`GET /v1/models`) | **`CUSTOM`** | Possible but awkward. `/v1/models` is an OpenAI model-list, not an MCP/A2A endpoint, so `fromUrl` auto-sync does **not** apply (sync only pulls MCP/A2A). You'd hand-author a `custom.inlineContent` JSON describing "RouteIQ routes to N models" and re-publish on change. Use only if the org wants the *model catalog itself* in the registry as a governed asset. |
| **RouteIQ as an A2A agent** (if A2A gateway enabled) | **`A2A`** | Only if `A2A_GATEWAY_ENABLED=true` and RouteIQ exposes a real A2A agent-card. Out of scope for the default config. |
| **RouteIQ's control/eval/governance tool-shaped endpoints** | **`MCP`** | If/when RouteIQ exposes its eval/governance ops as MCP tools, those are the *right* thing to register (genuinely tool-shaped), vs. `/v1/chat/completions` which is a model call, not a tool. |

**Recommendation: publish RouteIQ's MCP endpoint as a `descriptorType: MCP` record with
`fromUrl` sync.** This is the only candidate that gets free auto-refresh and is genuinely
the shape the registry was built for.

### 2.2 The concrete `CreateRegistryRecord` call (R1, MCP, fromUrl-synced)

```bash
# Pre-req: a registry exists. Choose authorizerType by who will DISCOVER (consume) it:
#   AWS_IAM      -> internal AWS-cred'd consumers (simplest to publish into; admin CRUDL is IAM anyway)
#   CUSTOM_JWT   -> IDE/MCP-client consumers without IAM creds (Kiro/Claude Code)
aws bedrock-agentcore-control create-registry \
  --name routeiq_gateway_registry \
  --authorizer-type AWS_IAM \
  --approval-configuration '{"autoApproval": false}'      # governance ON
# -> { "registryArn": "arn:aws:bedrock-agentcore:us-west-2:<acct>:registry/<12-16 char id>" }

# Publish RouteIQ's MCP endpoint as an auto-synced MCP record:
aws bedrock-agentcore-control create-registry-record \
  --registry-id <registryArn-or-id> \
  --name routeiq-gateway-mcp \
  --description "RouteIQ Gateway — multi-provider LLM routing + governed tool surface (LiteLLM MCP)" \
  --descriptor-type MCP \
  --synchronization-type URL \
  --synchronization-configuration '{
    "fromUrl": {
      "url": "https://routeiq.<your-domain>/v1/mcp",
      "credentialProviderConfigurations": [
        {
          "credentialProviderType": "IAM",
          "credentialProvider": {
            "iamCredentialProvider": {
              "roleArn": "arn:aws:iam::<acct>:role/RegistrySyncRole",
              "service": "bedrock-agentcore",
              "region": "us-west-2"
            }
          }
        }
      ]
    }
  }' \
  --record-version 1.0.0
# -> 202 { "recordArn": ".../record/<12-char id>", "status": "CREATING" }

# Governance gate (skip if autoApproval=true):
aws bedrock-agentcore-control submit-registry-record-for-approval \
  --registry-id <id> --record-id <12-char recordId>
# -> 202 { recordArn, recordId, registryArn, status: PENDING_APPROVAL (or APPROVED if auto), updatedAt }
```

**If RouteIQ's MCP endpoint requires no auth (or bearer), drop the
`credentialProviderConfigurations` (it's 0-or-1, optional).** Use it only when the registry's
sync probe must authenticate to RouteIQ.

**If you DON'T want auto-sync** (e.g. RouteIQ MCP endpoint not yet externally reachable):
omit `--synchronization-*` and hand-author `--descriptors` with `mcp.server.inlineContent`
(the MCP server definition JSON) + `mcp.tools.inlineContent` (the tools-list JSON), each
≤102400 chars.

### 2.3 R1 verified-vs-speculative

- **[VERIFIED]** The `CreateRegistryRecord` shape, MCP descriptor, `fromUrl` sync with IAM
  cred-provider, the 202/`recordArn`/`CREATING` response, the approval workflow.
- **[VERIFIED]** RouteIQ owns a real MCP server (seam (b), upstream LiteLLM, ADR-0017).
- **[SPECULATIVE]** That the registry's sync probe successfully discovers RouteIQ's tools
  over its specific external path + auth — depends on §3's mount-path confirmation and on
  RouteIQ's MCP endpoint being externally reachable from the AWS-managed sync fleet (it is an
  outbound HTTPS pull from AWS → your endpoint, so the endpoint must be internet- or
  PrivateLink-reachable, not cluster-internal). Treat the first publish as the verification.

### 2.4 Why R1 ranks #1

It uses the registry for what it is uniquely good at (org-wide discovery + governance +
approval), it's the smaller change (a CLI call, no consume-side auth gap), and it makes
RouteIQ a *first-class catalogued asset* — directly serving the "agent sprawl / reuse"
problem the registry was built for. RouteIQ is a natural thing to publish: it's the one
endpoint that fronts ~112 providers + a governed tool surface.

---

## 3. DIRECTION R2 — RouteIQ/LiteLLM **CONSUMES** the registry (RANK 2)

**Goal:** RouteIQ queries the registry's MCP endpoint to *discover* available MCP
servers/agents across the org, then dynamically registers the discovered MCP servers into
LiteLLM's `mcp_servers` so RouteIQ-routed models can use those tools.

### 3.1 The two-hop reality (this is the structural catch) [VERIFIED]

The registry MCP endpoint is **NOT** a `tools/list` of usable tools. Its only tool,
`search_registry_records`, returns **record metadata**. So R2 is inherently two hops:

1. **Discover (hop 1):** `tools/call search_registry_records {searchQuery, filter:
   {"descriptorType": {"$eq": "MCP"}}}` → list of MCP records, each carrying its `mcp`
   descriptor (server def + tools) and, where present, the `fromUrl` of the real MCP server.
2. **Wire (hop 2):** for each discovered record, extract the underlying MCP server URL and
   **register it as a NEW `mcp_servers` entry in LiteLLM** (config or `POST /v1/mcp/server`).
   The agent then calls that server *directly* (or through RouteIQ's MCP gateway) — **not
   through the registry**. The registry is a phone book, not a switchboard.

So "is the registry's MCP endpoint a `tools/list` LiteLLM's MCP client can consume?"
**Answer: yes, LiteLLM's MCP client can connect and call `search_registry_records`
[VERIFIED protocol compatibility] — but that one tool gives you a *directory*, not the target
tools. You still need hop 2 to make discovered tools usable.** This is the honest shape of R2.

### 3.2 The auth gap that blocks the naive consume path [VERIFIED — load-bearing]

To register the registry MCP endpoint *itself* as a LiteLLM `mcp_servers` entry, LiteLLM's
MCP client must authenticate to it. **LiteLLM 1.82.3 on disk supports only these auth_types**
(`reference/litellm/litellm/types/mcp.py::MCPAuth`):

```
none | api_key | bearer_token | basic | oauth2
```

There is **NO `sigv4` / `aws_sig` / `bedrock-agentcore` auth_type, and the MCP
server-manager does no SigV4 signing** (grep of `proxy/_experimental/mcp_server/*.py` for
sigv4/botocore/SigV4Auth = **0 hits**). The only `bedrock-agentcore` code in LiteLLM is
`llms/bedrock/chat/agentcore/transformation.py` — that is a **model-invocation** path
(`InvokeAgentRuntime` as an LLM), **not** an MCP-client signer. Therefore:

- **An `AWS_IAM`-authorizer registry's MCP endpoint is NOT natively reachable by LiteLLM's
  MCP client.** You must either (a) front it with **`mcp-proxy-for-aws`** as a `stdio` MCP
  server (LiteLLM does support `transport: stdio`), or (b) write a small RouteIQ-side
  SigV4-signing MCP shim.
- **A `CUSTOM_JWT`-authorizer registry's MCP endpoint IS reachable** via LiteLLM's
  `auth_type: oauth2` (client-credentials) or `bearer_token` (static token) — **this is the
  recommended R2 path.**

> **Note on the sibling doc.** The Gateway doc
> ([`agentcore-integration-and-arch-2026-06-15.md`](./agentcore-integration-and-arch-2026-06-15.md))
> §2.1 states LiteLLM ships a *"named AWS SigV4 / `bedrock-agentcore` MCP mode"* citing the
> LiteLLM MCP Overview docs. That capability is **NOT present in the pinned `1.82.3` on disk**
> for the MCP *client* (server_manager). Either it's a newer LiteLLM release than 1.82.3, or
> it refers to a different surface. **For R2 against an IAM registry today, assume LiteLLM
> CANNOT SigV4-sign and plan for CUSTOM_JWT or a proxy.** [VERIFIED against on-disk 1.82.3]

### 3.3 The concrete R2 wiring (CUSTOM_JWT registry, recommended)

```yaml
# LiteLLM config (loaded by proxy_server.py::_init_non_llm_configs -> load_servers_from_config)
mcp_servers:
  agentcore-registry:
    url: "https://bedrock-agentcore.us-west-2.amazonaws.com/registry/<registryId>/mcp"
    transport: http          # PIN http — see gotcha; LiteLLM 1.82.3 default is already http
    auth_type: oauth2        # CUSTOM_JWT registry; or bearer_token with a static access token
    # oauth2 client-credentials fields per litellm/types/mcp.py (client id/secret, token url)
    # extra_headers: ["Authorization"]   # to forward a caller-supplied bearer if proxying tokens
```

Then a RouteIQ-side discovery loop (NET-NEW small code, or manual): call
`search_registry_records`, parse MCP records, and `POST /v1/mcp/server` each discovered
server's `fromUrl` (with that server's own auth). **The registry entry above only lets you
SEARCH; the discovered servers are separate `mcp_servers` entries.**

For an **IAM registry**, replace the entry with a `stdio` proxy:

```yaml
mcp_servers:
  agentcore-registry:
    transport: stdio
    command: "uvx"
    args: ["mcp-proxy-for-aws@latest",
           "https://bedrock-agentcore.us-west-2.amazonaws.com/registry/<id>/mcp",
           "--service", "bedrock-agentcore", "--region", "us-west-2"]
```

(`mcp-proxy-for-aws` resolves AWS creds from the boto3 chain → RouteIQ's EKS Pod Identity
role, which would need `bedrock-agentcore:InvokeRegistryMcp` + `:SearchRegistryRecords`.)
**[VERIFIED proxy approach from AWS devguide; SPECULATIVE that LiteLLM's stdio transport
cleanly launches `uvx` in the RouteIQ pod — verify in the pod's runtime image.]**

### 3.4 Why R2 ranks #2

Lower value (RouteIQ consuming a *directory* of tools is useful only when there's a real
multi-team tool ecosystem to discover), higher friction (two hops + the SigV4 auth gap +
NET-NEW discovery-loop code), and it overlaps with the simpler Gateway-consume path
(Direction A in the sibling doc) when you already know the tool endpoint. R2 earns its place
specifically when **dynamic, governed, org-wide tool discovery** is the requirement — not
when you just need to reach one known MCP server.

---

## 4. Does the registry DUPLICATE LiteLLM's model list / RouteIQ's MCP gateway, or COMPLEMENT?

**Complement — they sit at different layers and answer different questions.** [VERIFIED by
contrasting documented purposes]

| Concern | RouteIQ's MCP gateway / LiteLLM model list | AWS Agent Registry |
|---|---|---|
| **Scope** | **In-gateway**: the tools/models *this RouteIQ instance* can route to right now. | **Org-wide**: every agent/tool/skill/MCP server across AWS + other clouds + on-prem, regardless of where hosted. |
| **Function** | **Runtime invocation surface** — you *call through* it (`tools/call`, `/v1/chat/completions`). | **Discovery + governance catalog** — you *search* it; it returns metadata + how-to-invoke. You do NOT call through it. |
| **Question answered** | "What can I invoke from here, now?" | "What exists in the org, who owns it, is it approved, how do I reach it?" |
| **Governance** | RouteIQ's own RBAC/policy/key model (per-tenant). | Org approval workflow (DRAFT→PENDING_APPROVAL→APPROVED→DEPRECATED), versioning, IAM publish/consume policies, custom compliance metadata. |
| **Search** | Exact registry/tool lookup. | **Hybrid keyword + semantic** ("payment processing" surfaces "billing"/"invoicing"). |

**No duplication.** RouteIQ's MCP gateway is a *runtime tool plane scoped to one gateway*;
the registry is an *org-wide discovery+governance index that points back at runtime planes
(including RouteIQ's)*. The natural relationship is **R1**: RouteIQ's gateway is one of the
runtime planes the registry catalogues. The registry adds the three things RouteIQ's
in-gateway list structurally cannot: cross-org visibility, approval-workflow governance, and
semantic discovery. RouteIQ adds the thing the registry structurally cannot: multi-tenant
runtime invocation with routing intelligence.

---

## 5. The registry-record-API gotchas [VERIFIED from on-disk AWS doc patterns]

1. **`recordId` regex accepts BOTH a bare id AND an ARN.** The `recordId` URI param on
   `SubmitRegistryRecordForApproval` is
   `(arn:...:registry/[a-zA-Z0-9]{12,16}/record/)?[a-zA-Z0-9]{12}` — the ARN prefix is an
   *optional* group, then a **fixed 12-char** id. So passing either form is valid, and a
   "recordId pattern" `ValidationException` is **rarely actually about the id form** —
   suspect the *registryId* (12-16 chars) or a stale/wrong record. **Probe the live API
   rather than trusting a regex-shaped error message.**
2. **`CreateRegistryRecord` returns only `recordArn` + `status` — NO top-level `recordId`.**
   You must parse the 12-char id out of the ARN tail (`.../record/<id>`) for the subsequent
   `submit-for-approval` / update / delete calls. `CreateRegistry` likewise returns only
   `registryArn`. **[VERIFIED — boto3 response syntaxes]**
3. **Records are processed ASYNC (202 + `status: CREATING`).** Do not assume a record is
   queryable/approvable the instant `create` returns — poll for `DRAFT` (or terminal
   `CREATE_FAILED`) before submitting for approval. A naive create→submit pipeline races.
4. **`onUpdate` is not safely deployable with vanilla IaC custom resources.** (Carried from
   the `agentcore-registry-record-api-contract` mulch record cited in the sibling doc; not
   re-verified against a doc this session, tagged **[SPECULATIVE/inherited]**.) When wiring a
   record via CDK/CloudFormation `AwsCustomResource`, use **onCreate + onDelete only** (or a
   custom provider that returns the bare 12-char id) — the record's own `onUpdate` path is
   unreliable. Re-publish a new record-version instead of in-place update.
5. **`synchronization-type` enum quirk.** The CLI help lists `--synchronization-type` with
   prose "Possible values include `FROM_URL` and `NONE`" but the enumerated **Possible
   values: `URL`** (singular). Pass **`URL`** (matches the `synchronizationConfiguration.
   fromUrl` field). **[VERIFIED — the create-registry-record CLI doc literally shows this
   mismatch; trust the enumerated `URL`.]** Probe live if rejected.
6. **A2A records need the full agent-card schema with non-empty `skills`.** (Inherited
   gotcha; `descriptorType: A2A` → `a2a.agentCard.inlineContent` must be a valid A2A 0.3
   card.) Not relevant to the recommended MCP path.
7. **`autoApproval` defaults to FALSE.** A freshly-created record is **not discoverable**
   until submitted + approved. If a search returns nothing right after publish, check the
   record `status` — it's likely `DRAFT`/`PENDING_APPROVAL`, not missing.
8. **`fromUrl` is HTTPS-only** (`pattern: https://.*`) and **at most ONE** credential
   provider config (`credentialProviderConfigurations` max length 1). The IAM cred-provider
   `roleArn` is the role *the registry assumes to sign the sync probe* — it is a
   registry-side role, distinct from RouteIQ's own pod role.

---

## 6. Smallest-shippable & ranked recommendation

**SHIP R1, manually, first. Defer R2 until a real multi-team tool ecosystem exists.**

**R1 smallest-shippable (no RouteIQ code change):**

1. `create-registry --authorizer-type AWS_IAM --approval-configuration '{"autoApproval":false}'`.
2. Confirm RouteIQ's external MCP URL (depends on the open `/v1` mount-path verification —
   seed #3 in the sibling doc; this BLOCKS the `fromUrl` value).
3. `create-registry-record --descriptor-type MCP --synchronization-type URL
   --synchronization-configuration '{fromUrl:{url:<RouteIQ /v1/mcp>}}'` (+ IAM cred-provider
   only if RouteIQ's MCP endpoint requires auth to the sync probe).
4. Parse `recordId` from `recordArn`; `submit-registry-record-for-approval`; approve.
5. Verify: `tools/call search_registry_records {"searchQuery":"routeiq routing gateway"}`
   from any MCP client returns the record. *This verifies the SPECULATIVE sync-discovery
   link.*

**R2 smallest-shippable (when needed):** stand up a **CUSTOM_JWT** registry (sidesteps the
SigV4 gap), add ONE `mcp_servers` entry `auth_type: oauth2`, and call
`search_registry_records` manually before building any discovery-loop automation.

**Ranking:** **R1 (publish) > R2 (consume).** R1 is higher-value (serves the registry's core
discovery/reuse purpose, makes RouteIQ a catalogued asset), smaller (CLI-only), and unblocked
by the auth gap. R2 is real but two-hop, lower-value, and gated on either a CUSTOM_JWT
registry or a SigV4 proxy because LiteLLM 1.82.3's MCP client cannot SigV4-sign.

**Suggested seeds (additive to the sibling doc's 5):**

- `feat: R1 publish RouteIQ MCP endpoint as AgentCore Registry MCP record (fromUrl sync)`
  (P2, small) — depends on the `/v1` mount-path verification seed.
- `spike: R2 consume AgentCore Registry via CUSTOM_JWT registry + search_registry_records`
  (P3, medium) — carries the LiteLLM-MCP-client-has-no-SigV4 constraint + two-hop wiring.
- `chore: confirm LiteLLM MCP-client SigV4/bedrock-agentcore support across versions`
  (P3, small) — reconcile the sibling-doc "named SigV4 MCP mode" claim against pinned 1.82.3
  (which lacks it); decide proxy-vs-upgrade for any IAM-registry consume path.

---

## 7. Source map

- **AgentCore Registry (authority, read 2026-06-15):** `create-registry-record` CLI v2.35.4
  + boto3; `API_Descriptors`; `registry-mcp-endpoint` devguide (the `search_registry_records`
  tool + auth); `create-registry` CLI; `API_SubmitRegistryRecordForApproval` (status machine
  + recordId regex); launch blog "AWS Agent Registry now in preview" (2026-04-09, hybrid
  search / approval / MCP-server / OAuth, 5 preview regions).
- **LiteLLM (on disk, `litellm==1.82.3`):** `reference/litellm/litellm/types/mcp.py`
  (`MCPAuth` enum = none/api_key/bearer_token/basic/oauth2; `MCPTransport` = sse/http/stdio,
  default http); `proxy/_experimental/mcp_server/mcp_server_manager.py` (config keys:
  url/transport/auth_type/extra_headers; no SigV4);
  `llms/bedrock/chat/agentcore/transformation.py` (model-invocation path, NOT an MCP signer).
- **RouteIQ:** `src/litellm_llmrouter/gateway/plugins/bedrock_agentcore_mcp.py` (inbound stub,
  `TSTALIASID` legacy URL); `src/litellm_llmrouter/mcp_gateway.py` (seam (a) REST gateway,
  auth_type none/api_key/bearer_token/oauth2); `routes/mcp.py` + ADR-0017 (seam (b) real MCP).
- **Sibling doc:**
  [`docs/architecture/agentcore-integration-and-arch-2026-06-15.md`](./agentcore-integration-and-arch-2026-06-15.md)
  (Gateway Directions A/C/B; the registry-record-API-contract gotcha provenance).
