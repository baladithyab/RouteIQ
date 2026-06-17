import type React from 'react'

// Shared panel card matching the look used across the existing pages
// (Dashboard / Observability / UserStats define a local copy each). New pages
// reuse this one to avoid drift.
export default function Card({
  title,
  children,
  isLoading,
  isError,
  error,
  onRetry,
  badge,
}: {
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
        ) : (
          children
        )}
      </div>
    </div>
  )
}
