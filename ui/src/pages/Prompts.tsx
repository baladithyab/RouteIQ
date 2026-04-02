import { useState, useEffect } from 'react'
import type React from 'react'
import {
  usePrompts,
  useCreatePrompt,
  useRollbackPrompt,
  useUpdatePromptABTest,
} from '../api/queries'
import type { CreatePromptRequest, PromptDefinition, UpdatePromptABTestRequest } from '../api/types'

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

// Highlight {{variable}} patterns in a template string
function TemplatePreview({ template }: { template: string }) {
  const parts = template.split(/(\{\{\w+\}\})/g)
  return (
    <pre className="text-xs font-mono text-gray-700 whitespace-pre-wrap bg-gray-50 rounded-lg p-3 border border-gray-200 max-h-40 overflow-y-auto">
      {parts.map((part, i) =>
        /^\{\{\w+\}\}$/.test(part) ? (
          <span key={i} className="text-blue-600 bg-blue-50 px-0.5 rounded font-semibold">{part}</span>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </pre>
  )
}

// --- Create Prompt Form ---
function PromptForm({ onSubmit, onCancel, isPending }: {
  onSubmit: (data: CreatePromptRequest) => void
  onCancel: () => void
  isPending: boolean
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [template, setTemplate] = useState('')
  const [tags, setTags] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({
      name,
      description: description || undefined,
      template,
      tags: tags ? tags.split(',').map(t => t.trim()).filter(Boolean) : undefined,
    })
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
        <div className="px-6 py-4 border-b border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900">Create Prompt</h3>
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
              placeholder="summarize-v1"
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
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Template <span className="text-xs text-gray-400">(use {'{{variable}}'} for placeholders)</span>
            </label>
            <textarea
              value={template}
              onChange={(e) => setTemplate(e.target.value)}
              required
              rows={8}
              className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="You are a helpful assistant. Summarize the following text:\n\n{{text}}"
            />
            {template && (
              <div className="mt-2">
                <span className="text-xs text-gray-500">Preview:</span>
                <TemplatePreview template={template} />
              </div>
            )}
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Tags (comma-separated)</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="summarization, production"
            />
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
              disabled={isPending || !name || !template}
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

// --- A/B Test Config Modal ---
function ABTestModal({ prompt, onSubmit, onCancel, isPending }: {
  prompt: PromptDefinition
  onSubmit: (data: UpdatePromptABTestRequest) => void
  onCancel: () => void
  isPending: boolean
}) {
  const [enabled, setEnabled] = useState(prompt.ab_test?.enabled ?? false)
  const [weights, setWeights] = useState<Record<number, number>>(
    prompt.ab_test?.weights ?? {}
  )

  const handleWeightChange = (version: number, value: string) => {
    setWeights({ ...weights, [version]: Number(value) || 0 })
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({ enabled, weights })
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4">
        <div className="px-6 py-4 border-b border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900">A/B Test: {prompt.name}</h3>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Enable A/B Testing</span>
            <button
              type="button"
              onClick={() => setEnabled(!enabled)}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
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
          </div>

          {enabled && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Version Weights</label>
              <div className="space-y-2">
                {prompt.versions.map((v) => (
                  <div key={v.version} className="flex items-center gap-3">
                    <span className="text-sm text-gray-600 w-16">v{v.version}</span>
                    <input
                      type="number"
                      value={weights[v.version] ?? 0}
                      onChange={(e) => handleWeightChange(v.version, e.target.value)}
                      min="0"
                      max="100"
                      className="w-24 rounded-lg border border-gray-300 px-3 py-1.5 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    />
                    <span className="text-xs text-gray-400">weight</span>
                  </div>
                ))}
              </div>
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
              disabled={isPending}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {isPending ? 'Saving...' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// --- Prompt Card (individual) ---
function PromptCard({ prompt, onRollback, onABTest }: {
  prompt: PromptDefinition
  onRollback: (name: string, version: number) => void
  onABTest: (prompt: PromptDefinition) => void
}) {
  const [showVersions, setShowVersions] = useState(false)

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-sm font-semibold text-gray-900">{prompt.name}</h4>
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700">
              v{prompt.active_version}
            </span>
            {prompt.ab_test?.enabled && (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-700">
                A/B Test
              </span>
            )}
          </div>
          {prompt.description && (
            <p className="text-xs text-gray-600">{prompt.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onABTest(prompt)}
            className="px-2 py-1 text-xs font-medium text-purple-600 bg-purple-50 rounded hover:bg-purple-100"
          >
            A/B Test
          </button>
          <button
            onClick={() => setShowVersions(!showVersions)}
            className="px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
          >
            {showVersions ? 'Hide Versions' : 'Versions'}
          </button>
        </div>
      </div>

      {prompt.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {prompt.tags.map((tag) => (
            <span key={tag} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600">
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Active version template preview */}
      {prompt.versions.length > 0 && (
        <TemplatePreview
          template={prompt.versions.find(v => v.version === prompt.active_version)?.template || ''}
        />
      )}

      {/* Version history */}
      {showVersions && prompt.versions.length > 0 && (
        <div className="mt-4 border-t border-gray-100 pt-4">
          <h5 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Version History</h5>
          <div className="space-y-3">
            {[...prompt.versions].reverse().map((v) => (
              <div key={v.version} className={`p-3 rounded-lg border ${
                v.version === prompt.active_version ? 'border-blue-200 bg-blue-50' : 'border-gray-100'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-gray-900">v{v.version}</span>
                    {v.version === prompt.active_version && (
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700">
                        active
                      </span>
                    )}
                    <span className="text-xs text-gray-400">
                      {new Date(v.created_at).toLocaleDateString()}
                    </span>
                    {v.author && (
                      <span className="text-xs text-gray-400">by {v.author}</span>
                    )}
                  </div>
                  {v.version !== prompt.active_version && (
                    <button
                      onClick={() => onRollback(prompt.name, v.version)}
                      className="px-2 py-1 text-xs font-medium text-amber-600 bg-amber-50 rounded hover:bg-amber-100"
                    >
                      Rollback
                    </button>
                  )}
                </div>
                <TemplatePreview template={v.template} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function Prompts() {
  const prompts = usePrompts()
  const createPrompt = useCreatePrompt()
  const rollbackPrompt = useRollbackPrompt()
  const updateABTest = useUpdatePromptABTest()
  const [showForm, setShowForm] = useState(false)
  const [abTestPrompt, setAbTestPrompt] = useState<PromptDefinition | null>(null)
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null)

  const handleCreate = async (data: CreatePromptRequest) => {
    try {
      await createPrompt.mutateAsync(data)
      setShowForm(false)
      setToast({ message: 'Prompt created', type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to create: ${(err as Error).message}`, type: 'error' })
    }
  }

  const handleRollback = async (name: string, version: number) => {
    try {
      await rollbackPrompt.mutateAsync({ name, version })
      setToast({ message: `Rolled back to v${version}`, type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to rollback: ${(err as Error).message}`, type: 'error' })
    }
  }

  const handleABTest = async (data: UpdatePromptABTestRequest) => {
    if (!abTestPrompt) return
    try {
      await updateABTest.mutateAsync({ name: abTestPrompt.name, data })
      setAbTestPrompt(null)
      setToast({ message: 'A/B test updated', type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to update A/B test: ${(err as Error).message}`, type: 'error' })
    }
  }

  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Prompts</h2>

      <Card
        title="Prompt Library"
        isLoading={prompts.isLoading}
        isError={prompts.isError}
        error={prompts.error as Error}
        onRetry={() => prompts.refetch()}
        badge={
          <button
            onClick={() => setShowForm(true)}
            className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
          >
            Create Prompt
          </button>
        }
      >
        {prompts.data && prompts.data.length > 0 ? (
          <div className="space-y-4">
            {prompts.data.map((p) => (
              <PromptCard
                key={p.name}
                prompt={p}
                onRollback={handleRollback}
                onABTest={setAbTestPrompt}
              />
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-sm">No prompts configured</p>
        )}
      </Card>

      {showForm && (
        <PromptForm
          onSubmit={handleCreate}
          onCancel={() => setShowForm(false)}
          isPending={createPrompt.isPending}
        />
      )}

      {abTestPrompt && (
        <ABTestModal
          prompt={abTestPrompt}
          onSubmit={handleABTest}
          onCancel={() => setAbTestPrompt(null)}
          isPending={updateABTest.isPending}
        />
      )}

      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </div>
  )
}
