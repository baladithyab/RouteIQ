import { useState, useEffect } from 'react'
import type React from 'react'
import { useRoutingConfig, useUpdateRoutingConfig } from '../api/queries'

const PROFILES = [
  { value: 'auto', label: 'Auto', description: 'Balanced routing across all tiers' },
  { value: 'eco', label: 'Eco', description: 'Cost-optimized, prefers cheaper models' },
  { value: 'premium', label: 'Premium', description: 'Quality-optimized, prefers best models' },
  { value: 'free', label: 'Free', description: 'Only uses free-tier models' },
  { value: 'reasoning', label: 'Reasoning', description: 'Optimized for reasoning tasks' },
]

function Card({ title, children, badge }: { title: string; children: React.ReactNode; badge?: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        {badge}
      </div>
      <div className="p-6">{children}</div>
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

export default function RoutingConfig() {
  const config = useRoutingConfig()
  const updateConfig = useUpdateRoutingConfig()
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null)

  const handleProfileChange = async (profile: string) => {
    try {
      await updateConfig.mutateAsync({ routing_profile: profile })
      setToast({ message: `Routing profile updated to "${profile}"`, type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to update profile: ${(err as Error).message}`, type: 'error' })
    }
  }

  const handleStrategyChange = async (strategy: string) => {
    try {
      await updateConfig.mutateAsync({ active_strategy: strategy })
      setToast({ message: `Active strategy changed to "${strategy}"`, type: 'success' })
    } catch (err) {
      setToast({ message: `Failed to update strategy: ${(err as Error).message}`, type: 'error' })
    }
  }

  const handleCentroidToggle = async () => {
    if (!config.data) return
    try {
      await updateConfig.mutateAsync({ centroid_routing_enabled: !config.data.centroid_routing_enabled })
      setToast({
        message: `Centroid routing ${config.data.centroid_routing_enabled ? 'disabled' : 'enabled'}`,
        type: 'success',
      })
    } catch (err) {
      setToast({ message: `Failed to toggle centroid routing: ${(err as Error).message}`, type: 'error' })
    }
  }

  if (config.isLoading) {
    return (
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Routing Configuration</h2>
        <div className="space-y-6">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="h-6 bg-gray-200 rounded animate-pulse w-1/3 mb-4" />
              <div className="h-4 bg-gray-200 rounded animate-pulse w-2/3" />
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (config.isError) {
    return (
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Routing Configuration</h2>
        <div className="bg-white rounded-xl shadow-sm border border-red-200 p-6 text-center">
          <p className="text-red-600 mb-2">{(config.error as Error)?.message || 'Failed to load configuration'}</p>
          <button onClick={() => config.refetch()} className="text-sm text-blue-600 hover:text-blue-800 font-medium">
            Retry
          </button>
        </div>
      </div>
    )
  }

  const data = config.data!

  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Routing Configuration</h2>
      
      <div className="space-y-6">
        {/* Active Strategy */}
        <Card title="Active Strategy">
          <div className="mb-4">
            <div className="flex items-center gap-3 mb-4">
              <span className="text-sm text-gray-500">Current:</span>
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                {data.active_strategy || 'None (using default)'}
              </span>
            </div>
            
            {data.available_strategies.length > 0 && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Switch Strategy</label>
                <select
                  value={data.active_strategy || ''}
                  onChange={(e) => handleStrategyChange(e.target.value)}
                  disabled={updateConfig.isPending}
                  className="block w-full max-w-md rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
                >
                  <option value="">Select a strategy...</option>
                  {data.available_strategies.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
            )}
            
            {data.available_strategies.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Available Strategies</h4>
                <div className="flex flex-wrap gap-2">
                  {data.available_strategies.map((s) => (
                    <span
                      key={s}
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        s === data.active_strategy
                          ? 'bg-blue-100 text-blue-800 ring-1 ring-blue-300'
                          : 'bg-gray-100 text-gray-700'
                      }`}
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Routing Profile */}
        <Card title="Routing Profile">
          <div className="space-y-3">
            {PROFILES.map((profile) => (
              <label
                key={profile.value}
                className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                  data.routing_profile === profile.value
                    ? 'border-blue-300 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                } ${updateConfig.isPending ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <input
                  type="radio"
                  name="routing-profile"
                  value={profile.value}
                  checked={data.routing_profile === profile.value}
                  onChange={() => handleProfileChange(profile.value)}
                  disabled={updateConfig.isPending}
                  className="mt-0.5 h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                />
                <div>
                  <div className="text-sm font-medium text-gray-900">{profile.label}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{profile.description}</div>
                </div>
              </label>
            ))}
          </div>
        </Card>

        {/* Centroid Routing Toggle */}
        <Card title="Centroid Routing">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-sm text-gray-700">
                Centroid routing provides <span className="font-medium">~2ms zero-config routing</span> as a fallback
                when ML models are unavailable.
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Uses pre-computed centroid vectors for fast model selection without requiring trained models.
              </p>
            </div>
            <button
              onClick={handleCentroidToggle}
              disabled={updateConfig.isPending}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 ${
                data.centroid_routing_enabled ? 'bg-blue-600' : 'bg-gray-200'
              }`}
              role="switch"
              aria-checked={data.centroid_routing_enabled}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                  data.centroid_routing_enabled ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
        </Card>

        {/* A/B Test Configuration (Read-Only) */}
        <Card
          title="A/B Test Configuration"
          badge={
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600">
              Read-only in MVP
            </span>
          }
        >
          <div>
            <div className="flex items-center gap-3 mb-4">
              <span className="text-sm text-gray-500">Status:</span>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                data.ab_testing.enabled
                  ? 'bg-green-100 text-green-800'
                  : 'bg-gray-100 text-gray-600'
              }`}>
                {data.ab_testing.enabled ? 'Active' : 'Inactive'}
              </span>
            </div>

            {data.ab_testing.enabled ? (
              <div>
                {data.ab_testing.experiment_id && (
                  <div className="mb-3">
                    <span className="text-xs text-gray-500">Experiment ID:</span>
                    <span className="ml-2 text-sm font-mono text-gray-700">{data.ab_testing.experiment_id}</span>
                  </div>
                )}
                
                <h4 className="text-sm font-medium text-gray-700 mb-2">Strategy Weights</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Strategy</th>
                        <th className="text-right py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Weight</th>
                        <th className="text-left py-2 px-3 text-xs text-gray-500 uppercase tracking-wide font-medium">Distribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(data.ab_testing.weights).map(([strategy, weight]) => {
                        const total = Object.values(data.ab_testing.weights).reduce((a, b) => a + b, 0)
                        const pct = total > 0 ? (weight / total) * 100 : 0
                        return (
                          <tr key={strategy} className="border-b border-gray-50">
                            <td className="py-2 px-3 font-medium text-gray-900">{strategy}</td>
                            <td className="py-2 px-3 text-right text-gray-600 font-mono">{weight}</td>
                            <td className="py-2 px-3">
                              <div className="flex items-center gap-2">
                                <div className="flex-1 bg-gray-100 rounded-full h-2">
                                  <div
                                    className="bg-blue-500 rounded-full h-2"
                                    style={{ width: `${pct}%` }}
                                  />
                                </div>
                                <span className="text-xs text-gray-500 w-12 text-right">{pct.toFixed(0)}%</span>
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-500">No active A/B test. Configure experiments via the API.</p>
            )}
          </div>
        </Card>
      </div>

      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </div>
  )
}
