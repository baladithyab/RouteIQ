import type React from 'react'
import { useEffect, useState } from 'react'
import { useUserStats, useGlobalStats } from '../api/queries'
import {
    fetchUiConfig,
    beginLogin,
    captureTokenFromRedirect,
    isUserAuthenticated,
    logoutUser,
    setUserToken,
    type UiConfig,
} from '../api/auth'

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

// --- Login: obtain a USER token via the gateway-advertised OIDC flow ---
// (RouteIQ-f98a). Falls back to a token-entry field when SSO is not configured
// so the user-tier path is still exercisable. Both channels feed the SAME
// apiClient user-token store that /me/stats reads.
function UserLogin({ onAuthenticated }: { onAuthenticated: () => void }) {
  const [uiConfig, setUiConfig] = useState<UiConfig | null>(null)
  const [configError, setConfigError] = useState<string | null>(null)
  const [manualToken, setManualToken] = useState('')

  useEffect(() => {
    let cancelled = false
    fetchUiConfig()
      .then((cfg) => {
        if (!cancelled) setUiConfig(cfg)
      })
      .catch((err: Error) => {
        if (!cancelled) setConfigError(err.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const ssoEnabled = !!uiConfig?.oidc.enabled && !!uiConfig?.oidc.login_url

  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const token = manualToken.trim()
    if (!token) return
    setUserToken(token)
    onAuthenticated()
  }

  return (
    <Card title="Sign in to view your usage">
      <div className="space-y-4">
        <p className="text-sm text-gray-600">
          Your usage is scoped to your own identity. Sign in with your account
          to view it &mdash; the admin key is not used here.
        </p>

        {ssoEnabled && uiConfig?.oidc.login_url && (
          <button
            type="button"
            onClick={() => beginLogin(uiConfig.oidc.login_url as string)}
            className="inline-flex items-center px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700"
          >
            Sign in with SSO
          </button>
        )}

        {!ssoEnabled && (
          <p className="text-xs text-amber-600">
            {configError
              ? `Could not load login config: ${configError}`
              : 'SSO is not configured on this gateway. Enter a user token to continue.'}
          </p>
        )}

        <form onSubmit={handleManualSubmit} className="space-y-2">
          <label className="block text-xs text-gray-500 uppercase tracking-wide">
            User token
          </label>
          <div className="flex gap-2">
            <input
              type="password"
              value={manualToken}
              onChange={(e) => setManualToken(e.target.value)}
              placeholder="sk-oidc-..."
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm font-mono"
            />
            <button
              type="submit"
              disabled={!manualToken.trim()}
              className="px-4 py-2 rounded-lg bg-gray-900 text-white text-sm font-medium hover:bg-gray-700 disabled:opacity-40"
            >
              Continue
            </button>
          </div>
        </form>
      </div>
    </Card>
  )
}

// --- My Stats: caller's own scope (GET /me/stats, user auth, own-scope only) ---
function MyStatsSection({ onLogout }: { onLogout: () => void }) {
  const me = useUserStats(true)

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

          <div className="mt-6 pt-4 border-t border-gray-100">
            <button
              type="button"
              onClick={onLogout}
              className="text-sm text-gray-500 hover:text-gray-800 font-medium"
            >
              Sign out
            </button>
          </div>
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
          <p className="text-xs text-gray-500 mb-4 flex items-center gap-2 flex-wrap">
            <span>
              Org-wide rollup across {global.data.tracked_keys.toLocaleString()} tracked
              {global.data.tracked_keys === 1 ? ' key' : ' keys'}.
            </span>
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                global.data.cluster_wide
                  ? 'bg-green-100 text-green-700'
                  : 'bg-amber-100 text-amber-700'
              }`}
              title={
                global.data.cluster_wide
                  ? 'Aggregated across all replicas via the shared store'
                  : 'Shared store unavailable — showing the single serving worker'
              }
            >
              {global.data.cluster_wide ? 'cluster-wide' : 'per-worker'}
            </span>
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
  // On mount, capture any user_token the gateway callback appended to the URL
  // (OIDC round-trip return), then reflect the held-token state.
  const [authed, setAuthed] = useState<boolean>(() => {
    captureTokenFromRedirect()
    return isUserAuthenticated()
  })

  const handleAuthenticated = () => setAuthed(true)
  const handleLogout = () => {
    logoutUser()
    setAuthed(false)
  }

  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">User Stats</h2>
      <div className="space-y-6">
        {authed ? (
          <MyStatsSection onLogout={handleLogout} />
        ) : (
          <UserLogin onAuthenticated={handleAuthenticated} />
        )}
        <PerKeySection />
      </div>
    </div>
  )
}
