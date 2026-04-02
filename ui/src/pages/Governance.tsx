import { useState, useEffect } from 'react'
import type React from 'react'
import {
  useWorkspaces,
  useCreateWorkspace,
  useDeleteWorkspace,
  useUsagePolicies,
  useCreateUsagePolicy,
  useToggleUsagePolicy,
  useKeyGovernance,
  useUpdateKeyGovernance,
} from '../api/queries'
import type { CreateWorkspaceRequest, CreateUsagePolicyRequest, UpdateKeyGovernanceRequest } from '../api/types'

function Card({ title, children, isLoading, isError, error, onRetry, badge }: {
  title: string
  children: React.ReactNode
  isLoading?: boolean
  isError?: boolean
  error?: Error | null
  onRetry?: () => void
  badge?: React.ReactNode
}) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        {badge}
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

function Toast({ message, type, onClose }: { message: string; type: 'success' | 'error'; onClose: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000)
    return () => clearTimeout(timer)
  }, [onClose])

  return (
    <div className={`fixed bottom-4 right-4 px-4 py-3 rounded-lg shadow-lg text-sm font-medium z-50 ${
      type === 'success' ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
    }`}>
      {message}
    </div>
  )
}

function ToggleSwitch({ enabled, onToggle, disabled }: { enabled: boolean; onToggle: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 ${
        enabled ? 'bg-blue-600' : 'bg-gray-200'
      }`}
      role="switch"
      aria-checked={enabled}
    >
      <span
        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
          enabled ? 'translate-x-5' : 'translate-x-0'
        }`}
      />
    </button>
  )
}

// --- Workspace Form Modal ---
function WorkspaceForm({ onSubmit, onCancel, isPending }: {
  onSubmit: (data: CreateWorkspaceRequest) => void
  onCancel: () => void
  isPending: boolean
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [budgetCap, setBudgetCap] = useState('')
  const [rateLimitRpm, setRateLimitRpm] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({
      name,
      description: description || undefined,
      budget_cap: budgetCap ? Number(budgetCap) : null,
      rate_limit_rpm: rateLimitRpm ? Number(rateLimitRpm) : null,
    })
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4">
        <div className="px-6 py-4 border-b border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900">Create Workspace</h3>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="my-workspace"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="Optional description"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Budget Cap ($)</label>
              <input
                type="number"
                value={budgetCap}
                onChange={(e) => setBudgetCap(e.target.value)}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="No limit"
                min="0"
                step="0.01"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Rate Limit (RPM)</label>
              <input
                type="number"
                value={rateLimitRpm}
                onChange={(e) => setRateLimitRpm(e.target.value)}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="No limit"
                min="0"
              />
            </div>
          </div>
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isPending || !name}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {isPending ? 'Creating...' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// --- Policy Form Modal ---
function PolicyForm({ onSubmit, onCancel, isPending }: {
  onSubmit: (data: CreateUsagePolicyRequest) => void
  onCancel: () => void
  isPending: boolean
}) {
  const [name, setName] = useState('')
  const [type, setType] = useState<'cost' | 'tokens' | 'requests'>('requests')
  const [limitValue, setLimitValue] = useState('')
  const [period, setPeriod] = useState<'minute' | 'hour' | 'day' | 'month'>('day')
  const [action, setAction] = useState<'deny' | 'log' | 'alert'>('deny')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({ name, type, limit_value: Number(limitValue), period, action })
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4">
        <div className="px-6 py-4 border-b border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900">Create Usage Policy</h3>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="Policy name"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
              <select
                value={type}
                onChange={(e) => setType(e.target.value as 'cost' | 'tokens' | 'requests')}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              >
                <option value="requests">Requests</option>
                <option value="tokens">Tokens</option>
                <option value="cost">Cost</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Limit</label>
              <input
                type="number"
                value={limitValue}
                onChange={(e) => setLimitValue(e.target.value)}
                required
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="1000"
                min="1"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Period</label>
              <select
                value={period}
                onChange={(e) => setPeriod(e.target.value as 'minute' | 'hour' | 'day' | 'month')}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              >
                <option value="minute">Per Minute</option>
                <option value="hour">Per Hour</option>
                <option value="day">Per Day</option>
                <option value="month">Per Month</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Action</label>
              <select
                value={action}
                onChange={(e) => setAction(e.target.value as 'deny' | 'log' | 'alert')}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              >
                <option value="deny">Deny</option>
                <option value="log">Log</option>
                <option value="alert">Alert</option>
              </select>
            </div>
          </div>
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isPending || !name || !limitValue}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {isPending ? 'Creating...' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// --- Workspaces Tab ---
function WorkspacesTab() {
  const workspaces = useWorkspaces()
  const createWorkspace = useCreateWorkspace()
  const deleteWorkspace = useDeleteWorkspace()
  const [showForm, setShowForm] = useState(false)
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const handleCreate = async (data: CreateWorkspaceRequest) => {
    try {
      await createWorkspace.mutateAsync(data)
      setShowForm(false)
      setToast({ message: 'Workspace created', type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to create: ${(err as Error).message}`, type: 'error' })
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await deleteWorkspace.mutateAsync(id)
      setDeleteConfirm(null)
      setToast({ message: 'Workspace deleted', type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to delete: ${(err as Error).message}`, type: 'error' })
    }
  }

  return (
    <>
      <Card
        title="Workspaces"
        isLoading={workspaces.isLoading}
        isError={workspaces.isError}
        error={workspaces.error as Error}
        onRetry={() => workspaces.refetch()}
        badge={
          <button
            onClick={() => setShowForm(true)}
            className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
          >
            Create Workspace
          </button>
        }
      >
        {workspaces.data && workspaces.data.length > 0 ? (
          <div className="space-y-4">
            {workspaces.data.map((ws) => (
              <div key={ws.workspace_id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h4 className="text-sm font-semibold text-gray-900">{ws.name}</h4>
                    <p className="text-xs text-gray-500 font-mono mt-0.5">{ws.workspace_id}</p>
                    {ws.description && (
                      <p className="text-xs text-gray-600 mt-1">{ws.description}</p>
                    )}
                  </div>
                  {deleteConfirm === ws.workspace_id ? (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleDelete(ws.workspace_id)}
                        disabled={deleteWorkspace.isPending}
                        className="px-2 py-1 text-xs font-medium text-white bg-red-600 rounded hover:bg-red-700 disabled:opacity-50"
                      >
                        Confirm
                      </button>
                      <button
                        onClick={() => setDeleteConfirm(null)}
                        className="px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setDeleteConfirm(ws.workspace_id)}
                      className="px-2 py-1 text-xs font-medium text-red-600 bg-red-50 rounded hover:bg-red-100"
                    >
                      Delete
                    </button>
                  )}
                </div>
                <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2">
                  <div>
                    <dt className="text-xs text-gray-500">Models</dt>
                    <dd className="text-sm font-medium text-gray-900">{ws.allowed_models.length}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-500">Budget Cap</dt>
                    <dd className="text-sm font-medium text-gray-900">
                      {ws.budget_cap !== null ? `$${ws.budget_cap.toFixed(2)}` : 'None'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-500">Rate Limits</dt>
                    <dd className="text-sm font-medium text-gray-900">
                      {ws.rate_limit_rpm !== null ? `${ws.rate_limit_rpm} RPM` : 'None'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-500">Guardrails</dt>
                    <dd className="text-sm font-medium text-gray-900">{ws.guardrail_ids.length}</dd>
                  </div>
                </dl>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-sm">No workspaces configured</p>
        )}
      </Card>

      {showForm && (
        <WorkspaceForm
          onSubmit={handleCreate}
          onCancel={() => setShowForm(false)}
          isPending={createWorkspace.isPending}
        />
      )}
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </>
  )
}

// --- Usage Policies Tab ---
function UsagePoliciesTab() {
  const policies = useUsagePolicies()
  const createPolicy = useCreateUsagePolicy()
  const togglePolicy = useToggleUsagePolicy()
  const [showForm, setShowForm] = useState(false)
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null)

  const handleCreate = async (data: CreateUsagePolicyRequest) => {
    try {
      await createPolicy.mutateAsync(data)
      setShowForm(false)
      setToast({ message: 'Policy created', type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to create: ${(err as Error).message}`, type: 'error' })
    }
  }

  const handleToggle = async (id: string, enabled: boolean) => {
    try {
      await togglePolicy.mutateAsync({ id, enabled: !enabled })
    } catch (err) {
      setToast({ message: `Failed to toggle: ${(err as Error).message}`, type: 'error' })
    }
  }

  const actionColors: Record<string, string> = {
    deny: 'bg-red-100 text-red-800',
    log: 'bg-gray-100 text-gray-800',
    alert: 'bg-amber-100 text-amber-800',
  }

  const typeColors: Record<string, string> = {
    cost: 'bg-emerald-100 text-emerald-800',
    tokens: 'bg-blue-100 text-blue-800',
    requests: 'bg-purple-100 text-purple-800',
  }

  return (
    <>
      <Card
        title="Usage Policies"
        isLoading={policies.isLoading}
        isError={policies.isError}
        error={policies.error as Error}
        onRetry={() => policies.refetch()}
        badge={
          <button
            onClick={() => setShowForm(true)}
            className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
          >
            Create Policy
          </button>
        }
      >
        {policies.data && policies.data.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Name</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Type</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Limit</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Period</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Action</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Usage</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Enabled</th>
                </tr>
              </thead>
              <tbody>
                {policies.data.map((policy) => {
                  const usagePct = policy.limit_value > 0 ? (policy.current_usage / policy.limit_value) * 100 : 0
                  return (
                    <tr key={policy.policy_id} className="border-b border-gray-50 hover:bg-gray-50">
                      <td className="py-2.5 px-3 font-medium text-gray-900">{policy.name}</td>
                      <td className="py-2.5 px-3">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${typeColors[policy.type] || 'bg-gray-100 text-gray-800'}`}>
                          {policy.type}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-gray-700 font-mono">{policy.limit_value.toLocaleString()}</td>
                      <td className="py-2.5 px-3 text-gray-600 capitalize">{policy.period}</td>
                      <td className="py-2.5 px-3">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${actionColors[policy.action] || 'bg-gray-100 text-gray-800'}`}>
                          {policy.action}
                        </span>
                      </td>
                      <td className="py-2.5 px-3">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 bg-gray-100 rounded-full h-2 max-w-[80px]">
                            <div
                              className={`rounded-full h-2 ${usagePct > 90 ? 'bg-red-500' : usagePct > 70 ? 'bg-amber-500' : 'bg-blue-500'}`}
                              style={{ width: `${Math.min(usagePct, 100)}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-500 whitespace-nowrap">
                            {policy.current_usage.toLocaleString()} / {policy.limit_value.toLocaleString()}
                          </span>
                        </div>
                      </td>
                      <td className="py-2.5 px-3">
                        <ToggleSwitch
                          enabled={policy.enabled}
                          onToggle={() => handleToggle(policy.policy_id, policy.enabled)}
                          disabled={togglePolicy.isPending}
                        />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-sm">No usage policies configured</p>
        )}
      </Card>

      {showForm && (
        <PolicyForm
          onSubmit={handleCreate}
          onCancel={() => setShowForm(false)}
          isPending={createPolicy.isPending}
        />
      )}
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </>
  )
}

// --- Key Governance Tab ---
function KeyGovernanceTab() {
  const [keyId, setKeyId] = useState('')
  const [searchId, setSearchId] = useState<string | null>(null)
  const keyGov = useKeyGovernance(searchId)
  const updateKeyGov = useUpdateKeyGovernance()
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null)
  const [editing, setEditing] = useState(false)
  const [editData, setEditData] = useState<UpdateKeyGovernanceRequest>({})

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (keyId.trim()) {
      setSearchId(keyId.trim())
      setEditing(false)
    }
  }

  const handleStartEdit = () => {
    if (keyGov.data) {
      setEditData({
        scopes: keyGov.data.scopes,
        budget_limit: keyGov.data.budget_limit,
        rate_limit_rpm: keyGov.data.rate_limit_rpm,
        allowed_models: keyGov.data.allowed_models,
      })
      setEditing(true)
    }
  }

  const handleSave = async () => {
    if (!searchId) return
    try {
      await updateKeyGov.mutateAsync({ keyId: searchId, data: editData })
      setEditing(false)
      setToast({ message: 'Key governance updated', type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to update: ${(err as Error).message}`, type: 'error' })
    }
  }

  return (
    <>
      <Card title="Key Governance">
        <form onSubmit={handleSearch} className="flex gap-3 mb-6">
          <input
            type="text"
            value={keyId}
            onChange={(e) => setKeyId(e.target.value)}
            className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            placeholder="Enter key ID to search..."
          />
          <button
            type="submit"
            disabled={!keyId.trim()}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            Search
          </button>
        </form>

        {keyGov.isLoading && (
          <div className="space-y-3">
            <div className="h-4 bg-gray-200 rounded animate-pulse w-3/4" />
            <div className="h-4 bg-gray-200 rounded animate-pulse w-1/2" />
          </div>
        )}

        {keyGov.isError && (
          <div className="text-center py-4">
            <p className="text-red-600 text-sm">{(keyGov.error as Error)?.message || 'Key not found'}</p>
          </div>
        )}

        {keyGov.data && !editing && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-semibold text-gray-900">Key: <span className="font-mono">{keyGov.data.key_id}</span></h4>
              <button
                onClick={handleStartEdit}
                className="px-3 py-1.5 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100"
              >
                Edit
              </button>
            </div>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
              <div>
                <dt className="text-xs text-gray-500 uppercase tracking-wide">Workspace</dt>
                <dd className="text-sm font-medium text-gray-900 mt-0.5">{keyGov.data.workspace_id || 'None'}</dd>
              </div>
              <div>
                <dt className="text-xs text-gray-500 uppercase tracking-wide">Budget Limit</dt>
                <dd className="text-sm font-medium text-gray-900 mt-0.5">
                  {keyGov.data.budget_limit !== null ? `$${keyGov.data.budget_limit.toFixed(2)}` : 'None'}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-gray-500 uppercase tracking-wide">Rate Limit</dt>
                <dd className="text-sm font-medium text-gray-900 mt-0.5">
                  {keyGov.data.rate_limit_rpm !== null ? `${keyGov.data.rate_limit_rpm} RPM` : 'None'}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-gray-500 uppercase tracking-wide">Scopes</dt>
                <dd className="text-sm font-medium text-gray-900 mt-0.5">
                  {keyGov.data.scopes.length > 0 ? keyGov.data.scopes.join(', ') : 'All'}
                </dd>
              </div>
            </dl>
            {keyGov.data.allowed_models.length > 0 && (
              <div className="mt-4">
                <dt className="text-xs text-gray-500 uppercase tracking-wide mb-2">Allowed Models</dt>
                <div className="flex flex-wrap gap-1">
                  {keyGov.data.allowed_models.map((m) => (
                    <span key={m} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700">
                      {m}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {keyGov.data && editing && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Budget Limit ($)</label>
              <input
                type="number"
                value={editData.budget_limit ?? ''}
                onChange={(e) => setEditData({ ...editData, budget_limit: e.target.value ? Number(e.target.value) : null })}
                className="block w-full max-w-xs rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="No limit"
                min="0"
                step="0.01"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Rate Limit (RPM)</label>
              <input
                type="number"
                value={editData.rate_limit_rpm ?? ''}
                onChange={(e) => setEditData({ ...editData, rate_limit_rpm: e.target.value ? Number(e.target.value) : null })}
                className="block w-full max-w-xs rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="No limit"
                min="0"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Scopes (comma-separated)</label>
              <input
                type="text"
                value={(editData.scopes || []).join(', ')}
                onChange={(e) => setEditData({ ...editData, scopes: e.target.value.split(',').map(s => s.trim()).filter(Boolean) })}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="chat, completions"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Allowed Models (comma-separated)</label>
              <input
                type="text"
                value={(editData.allowed_models || []).join(', ')}
                onChange={(e) => setEditData({ ...editData, allowed_models: e.target.value.split(',').map(s => s.trim()).filter(Boolean) })}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="gpt-4, claude-3-opus"
              />
            </div>
            <div className="flex gap-3 pt-2">
              <button
                onClick={handleSave}
                disabled={updateKeyGov.isPending}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {updateKeyGov.isPending ? 'Saving...' : 'Save'}
              </button>
              <button
                onClick={() => setEditing(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {!searchId && !keyGov.isLoading && (
          <p className="text-gray-500 text-sm">Enter a key ID above to view its governance rules.</p>
        )}
      </Card>
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </>
  )
}

// --- Main Governance Page ---
export default function Governance() {
  const [activeTab, setActiveTab] = useState<'workspaces' | 'policies' | 'keys'>('workspaces')

  const tabs = [
    { id: 'workspaces' as const, label: 'Workspaces' },
    { id: 'policies' as const, label: 'Usage Policies' },
    { id: 'keys' as const, label: 'Key Governance' },
  ]

  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Governance</h2>

      <div className="border-b border-gray-200 mb-6">
        <nav className="flex gap-6">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {activeTab === 'workspaces' && <WorkspacesTab />}
      {activeTab === 'policies' && <UsagePoliciesTab />}
      {activeTab === 'keys' && <KeyGovernanceTab />}
    </div>
  )
}
