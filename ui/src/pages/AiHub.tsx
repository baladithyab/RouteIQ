// AI-Hub: model / agent / MCP-server catalog + SSO/SCIM/MCP setup guidance
// (RouteIQ-06cf). These surfaces were API-only; this page renders them.
import { useModels, useMcpServers, useA2aAgents } from '../api/queries'
import Card from '../components/Card'

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    active: 'bg-green-100 text-green-800',
    degraded: 'bg-yellow-100 text-yellow-800',
    unavailable: 'bg-red-100 text-red-800',
  }
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[status] || 'bg-gray-100 text-gray-800'}`}>
      {status}
    </span>
  )
}

function Pill({ children, tone = 'blue' }: { children: React.ReactNode; tone?: 'blue' | 'emerald' | 'gray' }) {
  const tones: Record<string, string> = {
    blue: 'bg-blue-50 text-blue-700 border-blue-100',
    emerald: 'bg-emerald-50 text-emerald-700 border-emerald-100',
    gray: 'bg-gray-50 text-gray-600 border-gray-200',
  }
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono border ${tones[tone]}`}>
      {children}
    </span>
  )
}

// --- Model catalog (read-only view; CRUD lives on the Dashboard) ---
function ModelCatalog() {
  const models = useModels()
  return (
    <Card
      title="Model Catalog"
      isLoading={models.isLoading}
      isError={models.isError}
      error={models.error as Error}
      onRetry={() => models.refetch()}
      badge={<span className="text-xs text-gray-400">{models.data?.length ?? 0} models</span>}
    >
      {models.data && models.data.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Model</th>
                <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Provider</th>
                <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Model ID</th>
                <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {models.data.map((m) => (
                <tr key={m.model_id} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="py-2.5 px-3 font-medium text-gray-900">{m.model_name}</td>
                  <td className="py-2.5 px-3 text-gray-600">{m.provider}</td>
                  <td className="py-2.5 px-3 text-gray-500 font-mono text-xs">{m.model_id}</td>
                  <td className="py-2.5 px-3"><StatusBadge status={m.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-gray-400 mt-3">Add / edit / remove models on the Dashboard.</p>
        </div>
      ) : (
        <p className="text-gray-500 text-sm">No models configured</p>
      )}
    </Card>
  )
}

// --- A2A agent catalog ---
function AgentCatalog() {
  const agents = useA2aAgents()
  const disabled = agents.isError // /a2a/agents 404s when A2A is off

  return (
    <Card
      title="Agent Catalog (A2A)"
      isLoading={agents.isLoading}
      badge={<span className="text-xs text-gray-400">{agents.data?.length ?? 0} agents</span>}
    >
      {disabled ? (
        <p className="text-gray-500 text-sm">
          A2A gateway is not enabled. Set <code className="font-mono">A2A_GATEWAY_ENABLED=true</code> to register agents.
        </p>
      ) : agents.data && agents.data.length > 0 ? (
        <div className="space-y-3">
          {agents.data.map((a) => (
            <div key={a.agent_id} className="p-3 border border-gray-100 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="font-medium text-gray-900">{a.agent_name || a.agent_id}</div>
                <Pill tone="gray">{a.agent_id}</Pill>
              </div>
              {a.description && <p className="text-sm text-gray-600 mt-1">{a.description}</p>}
              {a.url && <p className="text-xs text-gray-400 font-mono mt-1 break-all">{a.url}</p>}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-500 text-sm">No agents registered</p>
      )}
    </Card>
  )
}

// --- MCP server catalog ---
function McpCatalog() {
  const servers = useMcpServers()
  const disabled = servers.isError // /llmrouter/mcp/servers 404s when MCP is off

  return (
    <Card
      title="MCP Server Catalog"
      isLoading={servers.isLoading}
      badge={<span className="text-xs text-gray-400">{servers.data?.length ?? 0} servers</span>}
    >
      {disabled ? (
        <p className="text-gray-500 text-sm">
          MCP gateway is not enabled. Set <code className="font-mono">MCP_GATEWAY_ENABLED=true</code> to register servers.
        </p>
      ) : servers.data && servers.data.length > 0 ? (
        <div className="space-y-3">
          {servers.data.map((s) => (
            <div key={s.server_id} className="p-3 border border-gray-100 rounded-lg">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="font-medium text-gray-900">{s.name}</div>
                <div className="flex items-center gap-1.5">
                  <Pill tone="gray">{s.transport}</Pill>
                  <Pill>{s.tools.length} tools</Pill>
                  <Pill tone="emerald">{s.resources.length} resources</Pill>
                </div>
              </div>
              <p className="text-xs text-gray-400 font-mono mt-1 break-all">{s.url}</p>
              {s.tools.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {s.tools.slice(0, 8).map((t) => (
                    <Pill key={t}>{t}</Pill>
                  ))}
                  {s.tools.length > 8 && <span className="text-xs text-gray-400">+{s.tools.length - 8} more</span>}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-500 text-sm">No MCP servers registered</p>
      )}
    </Card>
  )
}

// --- Setup guidance: SSO / SCIM / MCP. These are config-driven (no live CRUD
// endpoint), so the page documents the env/config switches inline. ---
function SetupStep({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="p-4 border border-gray-100 rounded-lg">
      <h4 className="text-sm font-semibold text-gray-800 mb-2">{title}</h4>
      <div className="text-sm text-gray-600 space-y-2">{children}</div>
    </div>
  )
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <pre className="bg-gray-900 text-gray-100 rounded-lg p-3 text-xs font-mono overflow-x-auto">
      {children}
    </pre>
  )
}

function SetupGuides() {
  return (
    <Card title="Identity & Protocol Setup">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <SetupStep title="SSO (OIDC)">
          <p>
            Connect Keycloak, Auth0, Okta, or Azure AD so users sign in with
            their own identity (the User Stats page drives this login flow).
          </p>
          <Code>{`OIDC_ENABLED=true
OIDC_ISSUER=https://idp.example.com
OIDC_CLIENT_ID=routeiq
OIDC_CLIENT_SECRET=...`}</Code>
          <p className="text-xs text-gray-400">See Governance → Identity (OIDC/SSO) in the docs.</p>
        </SetupStep>

        <SetupStep title="SCIM Provisioning">
          <p>
            Auto-provision users/groups from your IdP. SCIM maps directory
            groups to RouteIQ workspaces and roles.
          </p>
          <Code>{`SCIM_ENABLED=true
SCIM_BEARER_TOKEN=...
# IdP SCIM endpoint:
#   {gateway}/scim/v2`}</Code>
          <p className="text-xs text-gray-400">Point your IdP's SCIM connector at the gateway base + /scim/v2.</p>
        </SetupStep>

        <SetupStep title="MCP Servers">
          <p>
            Register Model Context Protocol servers so their tools/resources are
            discoverable. Tool invocation is off by default.
          </p>
          <Code>{`MCP_GATEWAY_ENABLED=true
LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true

# Register via the admin API:
POST /llmrouter/mcp/servers`}</Code>
          <p className="text-xs text-gray-400">Registered servers appear in the catalog above.</p>
        </SetupStep>
      </div>
    </Card>
  )
}

export default function AiHub() {
  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">AI Hub</h2>
      <div className="space-y-6">
        <ModelCatalog />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AgentCatalog />
          <McpCatalog />
        </div>
        <SetupGuides />
      </div>
    </div>
  )
}
