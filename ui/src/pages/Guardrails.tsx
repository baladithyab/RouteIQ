import { useState, useEffect } from 'react'
import type React from 'react'
import {
  useGuardrails,
  useCreateGuardrail,
  useToggleGuardrail,
  useDeleteGuardrail,
} from '../api/queries'
import type { CreateGuardrailRequest, GuardrailPolicy } from '../api/types'

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

const CHECK_TYPES: { value: GuardrailPolicy['check_type']; label: string; description: string }[] = [
  { value: 'regex_deny', label: 'Regex Deny', description: 'Block requests matching a regex pattern' },
  { value: 'regex_require', label: 'Regex Require', description: 'Require requests to match a regex pattern' },
  { value: 'pii_detection', label: 'PII Detection', description: 'Detect and handle personally identifiable information' },
  { value: 'toxicity', label: 'Toxicity', description: 'Detect toxic or harmful content' },
  { value: 'prompt_injection', label: 'Prompt Injection', description: 'Detect prompt injection attempts' },
  { value: 'custom', label: 'Custom', description: 'Custom guardrail with user-defined parameters' },
]

function GuardrailForm({ onSubmit, onCancel, isPending }: {
  onSubmit: (data: CreateGuardrailRequest) => void
  onCancel: () => void
  isPending: boolean
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [checkType, setCheckType] = useState<GuardrailPolicy['check_type']>('regex_deny')
  const [phase, setPhase] = useState<GuardrailPolicy['phase']>('input')
  const [action, setAction] = useState<GuardrailPolicy['action']>('deny')
  const [pattern, setPattern] = useState('')
  const [customParams, setCustomParams] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    let parameters: Record<string, unknown> = {}
    if (checkType === 'regex_deny' || checkType === 'regex_require') {
      parameters = { pattern }
    } else if (checkType === 'custom') {
      try {
        parameters = customParams ? JSON.parse(customParams) : {}
      } catch {
        parameters = {}
      }
    }
    onSubmit({ name, description: description || undefined, check_type: checkType, phase, action, parameters })
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto">
        <div className="px-6 py-4 border-b border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900">Create Guardrail</h3>
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
              placeholder="block-credit-cards"
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
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Check Type</label>
            <div className="space-y-2">
              {CHECK_TYPES.map((ct) => (
                <label
                  key={ct.value}
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                    checkType === ct.value
                      ? 'border-blue-300 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <input
                    type="radio"
                    name="check-type"
                    value={ct.value}
                    checked={checkType === ct.value}
                    onChange={() => setCheckType(ct.value)}
                    className="mt-0.5 h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                  />
                  <div>
                    <div className="text-sm font-medium text-gray-900">{ct.label}</div>
                    <div className="text-xs text-gray-500 mt-0.5">{ct.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Phase</label>
              <select
                value={phase}
                onChange={(e) => setPhase(e.target.value as GuardrailPolicy['phase'])}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              >
                <option value="input">Input</option>
                <option value="output">Output</option>
                <option value="both">Both</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Action</label>
              <select
                value={action}
                onChange={(e) => setAction(e.target.value as GuardrailPolicy['action'])}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              >
                <option value="deny">Deny</option>
                <option value="redact">Redact</option>
                <option value="log">Log</option>
                <option value="alert">Alert</option>
              </select>
            </div>
          </div>

          {(checkType === 'regex_deny' || checkType === 'regex_require') && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Regex Pattern</label>
              <input
                type="text"
                value={pattern}
                onChange={(e) => setPattern(e.target.value)}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder="\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b"
              />
            </div>
          )}

          {checkType === 'custom' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Parameters (JSON)</label>
              <textarea
                value={customParams}
                onChange={(e) => setCustomParams(e.target.value)}
                rows={4}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                placeholder='{"threshold": 0.9, "categories": ["hate"]}'
              />
            </div>
          )}

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

export default function Guardrails() {
  const guardrails = useGuardrails()
  const createGuardrail = useCreateGuardrail()
  const toggleGuardrail = useToggleGuardrail()
  const deleteGuardrail = useDeleteGuardrail()
  const [showForm, setShowForm] = useState(false)
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const handleCreate = async (data: CreateGuardrailRequest) => {
    try {
      await createGuardrail.mutateAsync(data)
      setShowForm(false)
      setToast({ message: 'Guardrail created', type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to create: ${(err as Error).message}`, type: 'error' })
    }
  }

  const handleToggle = async (id: string, enabled: boolean) => {
    try {
      await toggleGuardrail.mutateAsync({ id, enabled: !enabled })
    } catch (err) {
      setToast({ message: `Failed to toggle: ${(err as Error).message}`, type: 'error' })
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await deleteGuardrail.mutateAsync(id)
      setDeleteConfirm(null)
      setToast({ message: 'Guardrail deleted', type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to delete: ${(err as Error).message}`, type: 'error' })
    }
  }

  const checkTypeColors: Record<string, string> = {
    regex_deny: 'bg-red-100 text-red-800',
    regex_require: 'bg-blue-100 text-blue-800',
    pii_detection: 'bg-amber-100 text-amber-800',
    toxicity: 'bg-orange-100 text-orange-800',
    prompt_injection: 'bg-purple-100 text-purple-800',
    custom: 'bg-gray-100 text-gray-800',
  }

  const phaseColors: Record<string, string> = {
    input: 'bg-cyan-100 text-cyan-800',
    output: 'bg-indigo-100 text-indigo-800',
    both: 'bg-violet-100 text-violet-800',
  }

  const actionColors: Record<string, string> = {
    deny: 'bg-red-100 text-red-800',
    redact: 'bg-amber-100 text-amber-800',
    log: 'bg-gray-100 text-gray-800',
    alert: 'bg-yellow-100 text-yellow-800',
  }

  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Guardrails</h2>

      <Card
        title="Guardrail Policies"
        isLoading={guardrails.isLoading}
        isError={guardrails.isError}
        error={guardrails.error as Error}
        onRetry={() => guardrails.refetch()}
        badge={
          <button
            onClick={() => setShowForm(true)}
            className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
          >
            Create Guardrail
          </button>
        }
      >
        {guardrails.data && guardrails.data.length > 0 ? (
          <div className="space-y-4">
            {guardrails.data.map((g) => (
              <div key={g.guardrail_id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="text-sm font-semibold text-gray-900">{g.name}</h4>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${checkTypeColors[g.check_type] || 'bg-gray-100 text-gray-800'}`}>
                        {g.check_type.replace('_', ' ')}
                      </span>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${phaseColors[g.phase] || 'bg-gray-100 text-gray-800'}`}>
                        {g.phase}
                      </span>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${actionColors[g.action] || 'bg-gray-100 text-gray-800'}`}>
                        {g.action}
                      </span>
                    </div>
                    {g.description && (
                      <p className="text-xs text-gray-600">{g.description}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-3 ml-4">
                    <ToggleSwitch
                      enabled={g.enabled}
                      onToggle={() => handleToggle(g.guardrail_id, g.enabled)}
                      disabled={toggleGuardrail.isPending}
                    />
                    {deleteConfirm === g.guardrail_id ? (
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleDelete(g.guardrail_id)}
                          disabled={deleteGuardrail.isPending}
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
                        onClick={() => setDeleteConfirm(g.guardrail_id)}
                        className="px-2 py-1 text-xs font-medium text-red-600 bg-red-50 rounded hover:bg-red-100"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>
                {Object.keys(g.parameters).length > 0 && (
                  <div className="mt-2 p-2 bg-gray-50 rounded text-xs font-mono text-gray-600 overflow-x-auto">
                    {JSON.stringify(g.parameters, null, 2)}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-sm">No guardrails configured</p>
        )}
      </Card>

      {showForm && (
        <GuardrailForm
          onSubmit={handleCreate}
          onCancel={() => setShowForm(false)}
          isPending={createGuardrail.isPending}
        />
      )}
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </div>
  )
}
