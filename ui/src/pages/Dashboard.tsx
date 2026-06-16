import { useState } from 'react'
import {
  useGatewayStatus,
  useRoutingStats,
  useGlobalStats,
  useModels,
  useAddModel,
  useUpdateModel,
  useDeleteModel,
} from '../api/queries'
import type { ModelInfo } from '../api/types'

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

function FeatureBadge({ name, enabled }: { name: string; enabled: boolean }) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mr-1 mb-1 ${
      enabled ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'
    }`}>
      {name}: {enabled ? 'on' : 'off'}
    </span>
  )
}

function Card({ title, children, isLoading, isError, error, onRetry }: {
  title: string
  children: React.ReactNode
  isLoading?: boolean
  isError?: boolean
  error?: Error | null
  onRetry?: () => void
}) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-100">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      </div>
      <div className="p-6">
        {isLoading ? (
          <div className="space-y-3">
            <div className="h-4 bg-gray-200 rounded animate-pulse w-3/4" />
            <div className="h-4 bg-gray-200 rounded animate-pulse w-1/2" />
            <div className="h-4 bg-gray-200 rounded animate-pulse w-2/3" />
          </div>
        ) : isError ? (
          <div className="text-center py-4">
            <p className="text-red-600 text-sm mb-2">{error?.message || 'Failed to load data'}</p>
            {onRetry && (
              <button onClick={onRetry} className="text-sm text-blue-600 hover:text-blue-800 font-medium">
                Retry
              </button>
            )}
          </div>
        ) : children}
      </div>
    </div>
  )
}

function StrategyBar({ distribution }: { distribution: Record<string, number> }) {
  const total = Object.values(distribution).reduce((a, b) => a + b, 0)
  if (total === 0) return <p className="text-gray-500 text-sm">No routing decisions yet</p>
  
  const colors = ['bg-blue-500', 'bg-emerald-500', 'bg-amber-500', 'bg-purple-500', 'bg-rose-500']
  const entries = Object.entries(distribution).sort(([,a], [,b]) => b - a)
  
  return (
    <div>
      <div className="flex rounded-full overflow-hidden h-3 mb-3">
        {entries.map(([name, count], i) => (
          <div
            key={name}
            className={`${colors[i % colors.length]}`}
            style={{ width: `${(count / total) * 100}%` }}
            title={`${name}: ${count}`}
          />
        ))}
      </div>
      <div className="space-y-1">
        {entries.map(([name, count], i) => (
          <div key={name} className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${colors[i % colors.length]}`} />
              <span className="text-gray-700">{name}</span>
            </div>
            <span className="text-gray-500 font-mono">{count.toLocaleString()} ({((count / total) * 100).toFixed(1)}%)</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function DistTable({ label, distribution }: { label: string; distribution: Record<string, number> }) {
  const entries = Object.entries(distribution).sort(([, a], [, b]) => b - a)
  const total = entries.reduce((sum, [, c]) => sum + c, 0)
  return (
    <div>
      <h4 className="text-sm font-medium text-gray-700 mb-3">{label}</h4>
      {entries.length === 0 ? (
        <p className="text-gray-500 text-sm">No data yet</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Name</th>
                <th className="text-right py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Decisions</th>
                <th className="text-right py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Share</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([name, count]) => (
                <tr key={name} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="py-2.5 px-3 font-medium text-gray-900 font-mono text-xs">{name}</td>
                  <td className="py-2.5 px-3 text-right text-gray-700 font-mono">{count.toLocaleString()}</td>
                  <td className="py-2.5 px-3 text-right text-gray-500">
                    {total > 0 ? ((count / total) * 100).toFixed(1) : '0.0'}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// --- Model CRUD form: drives the admin add/edit endpoints (RouteIQ-eb2d) ---
function ModelForm({
  initial,
  onClose,
}: {
  initial: ModelInfo | null // null => add, non-null => edit
  onClose: () => void
}) {
  const isEdit = initial !== null
  const [modelName, setModelName] = useState(initial?.model_name ?? '')
  const [modelId, setModelId] = useState(initial?.model_id ?? '')
  const addModel = useAddModel()
  const updateModel = useUpdateModel()
  const pending = addModel.isPending || updateModel.isPending
  const error = (addModel.error || updateModel.error) as Error | null

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const body = {
      model_name: modelName.trim(),
      litellm_params: { model: modelId.trim() },
    }
    if (isEdit && initial) {
      updateModel.mutate(
        { modelName: initial.model_name, data: body },
        { onSuccess: onClose },
      )
    } else {
      addModel.mutate(body, { onSuccess: onClose })
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
      <h4 className="text-sm font-semibold text-gray-800">
        {isEdit ? `Edit ${initial?.model_name}` : 'Add Model'}
      </h4>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">
            Model Name
          </label>
          <input
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="claude-3-5-sonnet"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">
            litellm_params.model
          </label>
          <input
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="anthropic/claude-3-5-sonnet"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm font-mono"
          />
        </div>
      </div>
      {error && <p className="text-red-600 text-xs">{error.message}</p>}
      <div className="flex gap-2">
        <button
          type="submit"
          disabled={pending || !modelName.trim() || !modelId.trim()}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-40"
        >
          {pending ? 'Saving…' : isEdit ? 'Save' : 'Add'}
        </button>
        <button
          type="button"
          onClick={onClose}
          className="px-4 py-2 rounded-lg bg-white border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50"
        >
          Cancel
        </button>
      </div>
    </form>
  )
}

export default function Dashboard() {
  const status = useGatewayStatus()
  const stats = useRoutingStats()
  const global = useGlobalStats()
  const models = useModels()
  const deleteModel = useDeleteModel()
  // null = form closed; 'new' = add form; a ModelInfo = edit that model.
  const [editing, setEditing] = useState<ModelInfo | 'new' | null>(null)

  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Dashboard</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Gateway Status */}
        <Card
          title="Gateway Status"
          isLoading={status.isLoading}
          isError={status.isError}
          error={status.error as Error}
          onRetry={() => status.refetch()}
        >
          {status.data && (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-2.5 h-2.5 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-green-700">Gateway Active</span>
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
                <div>
                  <dt className="text-xs text-gray-500 uppercase tracking-wide">Version</dt>
                  <dd className="text-sm font-medium text-gray-900 mt-0.5">{status.data.version}</dd>
                </div>
                <div>
                  <dt className="text-xs text-gray-500 uppercase tracking-wide">Uptime</dt>
                  <dd className="text-sm font-medium text-gray-900 mt-0.5">{status.data.uptime_formatted}</dd>
                </div>
                <div>
                  <dt className="text-xs text-gray-500 uppercase tracking-wide">Workers</dt>
                  <dd className="text-sm font-medium text-gray-900 mt-0.5">{status.data.worker_count}</dd>
                </div>
                <div>
                  <dt className="text-xs text-gray-500 uppercase tracking-wide">Strategy</dt>
                  <dd className="text-sm font-medium text-gray-900 mt-0.5">{status.data.active_strategy || 'None'}</dd>
                </div>
                <div>
                  <dt className="text-xs text-gray-500 uppercase tracking-wide">Profile</dt>
                  <dd className="text-sm font-medium text-gray-900 mt-0.5 capitalize">{status.data.routing_profile}</dd>
                </div>
                <div>
                  <dt className="text-xs text-gray-500 uppercase tracking-wide">Centroid Routing</dt>
                  <dd className="text-sm font-medium text-gray-900 mt-0.5">
                    {status.data.centroid_routing_enabled ? '✅ Enabled' : '❌ Disabled'}
                  </dd>
                </div>
              </dl>
              {status.data.feature_flags && Object.keys(status.data.feature_flags).length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <dt className="text-xs text-gray-500 uppercase tracking-wide mb-2">Feature Flags</dt>
                  <div className="flex flex-wrap">
                    {Object.entries(status.data.feature_flags).map(([name, enabled]) => (
                      <FeatureBadge key={name} name={name} enabled={enabled} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </Card>

        {/* Routing Stats */}
        <Card
          title="Routing Stats"
          isLoading={stats.isLoading}
          isError={stats.isError}
          error={stats.error as Error}
          onRetry={() => stats.refetch()}
        >
          {stats.data && (
            <div>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-gray-900">{stats.data.total_decisions.toLocaleString()}</div>
                  <div className="text-xs text-gray-500 mt-1">Total Decisions</div>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-gray-900">{stats.data.centroid_decisions.toLocaleString()}</div>
                  <div className="text-xs text-gray-500 mt-1">Centroid Decisions</div>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-gray-900">{stats.data.average_latency_ms.toFixed(1)}</div>
                  <div className="text-xs text-gray-500 mt-1">Avg Latency (ms)</div>
                </div>
              </div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Strategy Distribution</h4>
              <StrategyBar distribution={stats.data.strategy_distribution} />
              <h4 className="text-sm font-medium text-gray-700 mt-5 mb-3">Profile Distribution</h4>
              <StrategyBar distribution={stats.data.profile_distribution} />
            </div>
          )}
        </Card>
      </div>

      {/* Global Routing Rollup - full width (admin /stats/global) */}
      <Card
        title="Global Routing Rollup"
        isLoading={global.isLoading}
        isError={global.isError}
        error={global.error as Error}
        onRetry={() => global.refetch()}
      >
        {global.data && (
          <div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{global.data.total_decisions.toLocaleString()}</div>
                <div className="text-xs text-gray-500 mt-1">Total Decisions</div>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{global.data.tracked_keys.toLocaleString()}</div>
                <div className="text-xs text-gray-500 mt-1">Tracked Keys</div>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{global.data.centroid_decisions.toLocaleString()}</div>
                <div className="text-xs text-gray-500 mt-1">Centroid Decisions</div>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{global.data.average_latency_ms.toFixed(1)}</div>
                <div className="text-xs text-gray-500 mt-1">Avg Latency (ms)</div>
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <DistTable label="Per-Strategy" distribution={global.data.strategy_distribution} />
              <DistTable label="Per-Model" distribution={global.data.model_distribution} />
              <DistTable label="Per-Key" distribution={global.data.key_distribution} />
            </div>
          </div>
        )}
      </Card>

      <div className="mb-6" />

      {/* Model Overview - full width (with admin CRUD, RouteIQ-eb2d) */}
      <Card
        title="Model Overview"
        isLoading={models.isLoading}
        isError={models.isError}
        error={models.error as Error}
        onRetry={() => models.refetch()}
      >
        <div className="flex justify-end mb-3">
          {editing === null && (
            <button
              type="button"
              onClick={() => setEditing('new')}
              className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700"
            >
              + Add Model
            </button>
          )}
        </div>

        {editing !== null && (
          <ModelForm
            initial={editing === 'new' ? null : editing}
            onClose={() => setEditing(null)}
          />
        )}

        {deleteModel.error && (
          <p className="text-red-600 text-xs mb-3">{(deleteModel.error as Error).message}</p>
        )}

        {models.data && models.data.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Model Name</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Provider</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Model ID</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Status</th>
                  <th className="text-right py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.data.map((model) => (
                  <tr key={model.model_id} className="border-b border-gray-50 hover:bg-gray-50">
                    <td className="py-2.5 px-3 font-medium text-gray-900">{model.model_name}</td>
                    <td className="py-2.5 px-3 text-gray-600">{model.provider}</td>
                    <td className="py-2.5 px-3 text-gray-500 font-mono text-xs">{model.model_id}</td>
                    <td className="py-2.5 px-3"><StatusBadge status={model.status} /></td>
                    <td className="py-2.5 px-3 text-right whitespace-nowrap">
                      <button
                        type="button"
                        onClick={() => setEditing(model)}
                        className="text-blue-600 hover:text-blue-800 text-xs font-medium mr-3"
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={() => deleteModel.mutate(model.model_name)}
                        disabled={deleteModel.isPending}
                        className="text-red-600 hover:text-red-800 text-xs font-medium disabled:opacity-40"
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-sm">No models configured</p>
        )}
      </Card>
    </div>
  )
}
