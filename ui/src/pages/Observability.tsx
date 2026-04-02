import { useState, useEffect } from 'react'
import type React from 'react'
import { useServiceStatus, useModelQuality, useEvalStats, useRunEvalBatch } from '../api/queries'

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

function StatusDot({ available }: { available: boolean }) {
  return (
    <div className={`w-2.5 h-2.5 rounded-full ${
      available ? 'bg-green-500 animate-pulse' : 'bg-red-500'
    }`} />
  )
}

function QualityBar({ score, maxScore }: { score: number; maxScore: number }) {
  const pct = maxScore > 0 ? (score / maxScore) * 100 : 0
  const color = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-blue-500' : pct >= 40 ? 'bg-amber-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-100 rounded-full h-3">
        <div className={`${color} rounded-full h-3 transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-sm font-mono text-gray-700 w-12 text-right">{score.toFixed(1)}</span>
    </div>
  )
}

// --- Services Section ---
function ServicesSection() {
  const services = useServiceStatus()

  return (
    <Card
      title="Service Health"
      isLoading={services.isLoading}
      isError={services.isError}
      error={services.error as Error}
      onRetry={() => services.refetch()}
    >
      {services.data && services.data.services.length > 0 ? (
        <div className="space-y-3">
          {services.data.services.map((svc) => (
            <div key={svc.name} className="flex items-center justify-between p-3 border border-gray-100 rounded-lg">
              <div className="flex items-center gap-3">
                <StatusDot available={svc.available} />
                <div>
                  <div className="text-sm font-medium text-gray-900">{svc.name}</div>
                  {svc.version && (
                    <div className="text-xs text-gray-500">v{svc.version}</div>
                  )}
                </div>
              </div>
              <div className="text-right">
                {svc.available ? (
                  <div>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      healthy
                    </span>
                    {svc.latency_ms !== null && (
                      <div className="text-xs text-gray-500 mt-0.5">{svc.latency_ms.toFixed(0)}ms</div>
                    )}
                  </div>
                ) : (
                  <div>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                      unavailable
                    </span>
                    {svc.error && (
                      <div className="text-xs text-red-500 mt-0.5 max-w-[200px] truncate">{svc.error}</div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-500 text-sm">No service status available</p>
      )}
    </Card>
  )
}

// --- Model Quality Section ---
function ModelQualitySection() {
  const quality = useModelQuality()

  const maxScore = quality.data?.ranking
    ? Math.max(...quality.data.ranking.map(([, s]) => s), 1)
    : 1

  return (
    <Card
      title="Model Quality"
      isLoading={quality.isLoading}
      isError={quality.isError}
      error={quality.error as Error}
      onRetry={() => quality.refetch()}
    >
      {quality.data && quality.data.ranking.length > 0 ? (
        <div>
          {/* Bar chart */}
          <div className="space-y-3 mb-6">
            {quality.data.ranking.map(([model, score], i) => (
              <div key={model}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-gray-700">
                    <span className="text-xs text-gray-400 mr-2">#{i + 1}</span>
                    {model}
                  </span>
                </div>
                <QualityBar score={score} maxScore={maxScore} />
              </div>
            ))}
          </div>

          {/* Detail table */}
          {quality.data.details && quality.data.details.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Model</th>
                    <th className="text-right py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Score</th>
                    <th className="text-right py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {quality.data.details.map((d) => (
                    <tr key={d.model} className="border-b border-gray-50 hover:bg-gray-50">
                      <td className="py-2.5 px-3 font-medium text-gray-900">{d.model}</td>
                      <td className="py-2.5 px-3 text-right text-gray-700 font-mono">{d.score.toFixed(2)}</td>
                      <td className="py-2.5 px-3 text-right text-gray-500">{d.sample_count.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        <p className="text-gray-500 text-sm">No model quality data available</p>
      )}
    </Card>
  )
}

// --- Eval Pipeline Section ---
function EvalPipelineSection() {
  const stats = useEvalStats()
  const runBatch = useRunEvalBatch()
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null)

  const handleRunBatch = async () => {
    try {
      await runBatch.mutateAsync()
      setToast({ message: 'Evaluation batch started', type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to start batch: ${(err as Error).message}`, type: 'error' })
    }
  }

  return (
    <>
      <Card
        title="Evaluation Pipeline"
        isLoading={stats.isLoading}
        isError={stats.isError}
        error={stats.error as Error}
        onRetry={() => stats.refetch()}
        badge={
          <button
            onClick={handleRunBatch}
            disabled={runBatch.isPending || (stats.data?.running ?? false)}
            className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {runBatch.isPending ? 'Starting...' : stats.data?.running ? 'Running...' : 'Run Batch'}
          </button>
        }
      >
        {stats.data && (
          <div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{stats.data.pending_samples.toLocaleString()}</div>
                <div className="text-xs text-gray-500 mt-1">Pending</div>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{stats.data.evaluated_samples.toLocaleString()}</div>
                <div className="text-xs text-gray-500 mt-1">Evaluated</div>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center gap-2">
                  <div className={`w-2.5 h-2.5 rounded-full ${
                    stats.data.running ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                  }`} />
                  <span className="text-sm font-medium text-gray-900">
                    {stats.data.running ? 'Running' : 'Idle'}
                  </span>
                </div>
                <div className="text-xs text-gray-500 mt-1">Status</div>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">
                  {stats.data.average_score !== null ? stats.data.average_score.toFixed(2) : '-'}
                </div>
                <div className="text-xs text-gray-500 mt-1">Avg Score</div>
              </div>
            </div>
            {stats.data.last_run_at && (
              <p className="text-xs text-gray-500">
                Last run: {new Date(stats.data.last_run_at).toLocaleString()}
              </p>
            )}
          </div>
        )}
      </Card>
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </>
  )
}

export default function Observability() {
  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Observability</h2>

      <div className="space-y-6">
        <ServicesSection />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ModelQualitySection />
          <EvalPipelineSection />
        </div>
      </div>
    </div>
  )
}
