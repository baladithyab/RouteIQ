import type React from 'react'
import { useUserStats, useGlobalStats } from '../api/queries'

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

function BudgetBar({ usedPct }: { usedPct: number }) {
  const pct = Math.max(0, Math.min(100, usedPct))
  const color = pct >= 90 ? 'bg-red-500' : pct >= 70 ? 'bg-amber-500' : 'bg-green-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-100 rounded-full h-3">
        <div className={`${color} rounded-full h-3 transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-sm font-mono text-gray-700 w-12 text-right">{pct.toFixed(0)}%</span>
    </div>
  )
}

function fmtUsd(value: number | null): string {
  return value === null ? '-' : `$${value.toFixed(2)}`
}

// --- My Stats: caller's own scope (GET /me/stats, user auth, own-scope only) ---
function MyStatsSection() {
  const me = useUserStats()

  return (
    <Card
      title="My Usage"
      isLoading={me.isLoading}
      isError={me.isError}
      error={me.error as Error}
      onRetry={() => me.refetch()}
    >
      {me.data && (
        <div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-6">
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{me.data.decision_count.toLocaleString()}</div>
              <div className="text-xs text-gray-500 mt-1">Decisions</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{fmtUsd(me.data.spend_usd)}</div>
              <div className="text-xs text-gray-500 mt-1">Spend</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{fmtUsd(me.data.budget_remaining_usd)}</div>
              <div className="text-xs text-gray-500 mt-1">Budget Remaining</div>
            </div>
          </div>

          <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-3 mb-6">
            <div>
              <dt className="text-xs text-gray-500 uppercase tracking-wide">Key ID</dt>
              <dd className="text-sm font-medium text-gray-900 mt-0.5 font-mono break-all">{me.data.key_id}</dd>
            </div>
            <div>
              <dt className="text-xs text-gray-500 uppercase tracking-wide">Max Budget</dt>
              <dd className="text-sm font-medium text-gray-900 mt-0.5">{fmtUsd(me.data.max_budget_usd)}</dd>
            </div>
          </dl>

          {me.data.budget_used_pct !== null && (
            <div className="mb-6">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Budget Used</h4>
              <BudgetBar usedPct={me.data.budget_used_pct} />
            </div>
          )}

          <h4 className="text-sm font-medium text-gray-700 mb-3">Recent Models</h4>
          {me.data.recent_models.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {me.data.recent_models.map((model) => (
                <span
                  key={model}
                  className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-mono bg-blue-50 text-blue-700 border border-blue-100"
                >
                  {model}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No routing decisions yet</p>
          )}
        </div>
      )}
    </Card>
  )
}

// --- Per-Key breakdown: admin rollup (GET /stats/global) ---
function PerKeySection() {
  const global = useGlobalStats()

  const entries = global.data
    ? Object.entries(global.data.key_distribution).sort(([, a], [, b]) => b - a)
    : []
  const total = entries.reduce((sum, [, c]) => sum + c, 0)

  return (
    <Card
      title="Per-Key Usage (admin)"
      isLoading={global.isLoading}
      isError={global.isError}
      error={global.error as Error}
      onRetry={() => global.refetch()}
    >
      {global.data && (
        <div>
          <p className="text-xs text-gray-500 mb-4">
            Org-wide rollup across {global.data.tracked_keys.toLocaleString()} tracked
            {global.data.tracked_keys === 1 ? ' key' : ' keys'}.
          </p>
          {entries.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Key ID</th>
                    <th className="text-right py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Decisions</th>
                    <th className="text-right py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Share</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map(([keyId, count]) => (
                    <tr key={keyId} className="border-b border-gray-50 hover:bg-gray-50">
                      <td className="py-2.5 px-3 font-medium text-gray-900 font-mono text-xs break-all">{keyId}</td>
                      <td className="py-2.5 px-3 text-right text-gray-700 font-mono">{count.toLocaleString()}</td>
                      <td className="py-2.5 px-3 text-right text-gray-500">
                        {total > 0 ? ((count / total) * 100).toFixed(1) : '0.0'}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No per-key usage tracked yet</p>
          )}
        </div>
      )}
    </Card>
  )
}

export default function UserStats() {
  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">User Stats</h2>
      <div className="space-y-6">
        <MyStatsSection />
        <PerKeySection />
      </div>
    </div>
  )
}
