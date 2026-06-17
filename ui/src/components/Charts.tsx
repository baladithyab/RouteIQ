// Lightweight, dependency-free SVG chart primitives (RouteIQ-c119).
//
// We deliberately avoid pulling in a charting library: the SPA already ships a
// 350kB bundle and these primitives keep cost/usage/token/latency time-series
// rendering byte-cheap and tsc-clean. All charts are pure SVG + Tailwind, fully
// responsive via a viewBox, and accept a simple { t, v } point series.

import type React from 'react'

export interface TimePoint {
  /** Epoch millis (or any monotonically increasing x). */
  t: number
  /** Series value at t. */
  v: number
}

export interface Series {
  label: string
  color: string
  points: TimePoint[]
}

const VIEW_W = 600
const VIEW_H = 200
const PAD_L = 8
const PAD_R = 8
const PAD_T = 12
const PAD_B = 22

function bounds(seriesList: Series[]): { minT: number; maxT: number; maxV: number } {
  let minT = Infinity
  let maxT = -Infinity
  let maxV = 0
  for (const s of seriesList) {
    for (const p of s.points) {
      if (p.t < minT) minT = p.t
      if (p.t > maxT) maxT = p.t
      if (p.v > maxV) maxV = p.v
    }
  }
  if (!isFinite(minT)) minT = 0
  if (!isFinite(maxT)) maxT = 1
  if (maxT === minT) maxT = minT + 1
  if (maxV <= 0) maxV = 1
  return { minT, maxT, maxV }
}

function scaleX(t: number, minT: number, maxT: number): number {
  return PAD_L + ((t - minT) / (maxT - minT)) * (VIEW_W - PAD_L - PAD_R)
}

function scaleY(v: number, maxV: number): number {
  return VIEW_H - PAD_B - (v / maxV) * (VIEW_H - PAD_T - PAD_B)
}

function pathFor(s: Series, minT: number, maxT: number, maxV: number): string {
  if (s.points.length === 0) return ''
  return s.points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${scaleX(p.t, minT, maxT).toFixed(1)},${scaleY(p.v, maxV).toFixed(1)}`)
    .join(' ')
}

function fmtTick(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  if (v >= 1) return v.toFixed(0)
  return v.toFixed(2)
}

/** Multi-series line chart over a shared time axis. */
export function LineChart({
  series,
  height = 200,
  yFormat = fmtTick,
  emptyLabel = 'No data yet',
}: {
  series: Series[]
  height?: number
  yFormat?: (v: number) => string
  emptyLabel?: string
}) {
  const hasData = series.some((s) => s.points.length > 0)
  const { minT, maxT, maxV } = bounds(series)

  if (!hasData) {
    return <p className="text-gray-500 text-sm py-6 text-center">{emptyLabel}</p>
  }

  const gridLines = [0, 0.25, 0.5, 0.75, 1]

  return (
    <div>
      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="none"
        className="w-full"
        style={{ height }}
        role="img"
        aria-label={series.map((s) => s.label).join(', ')}
      >
        {/* Horizontal grid + y labels */}
        {gridLines.map((g) => {
          const y = scaleY(maxV * g, maxV)
          return (
            <g key={g}>
              <line
                x1={PAD_L}
                x2={VIEW_W - PAD_R}
                y1={y}
                y2={y}
                stroke="#f1f5f9"
                strokeWidth={1}
              />
              <text x={PAD_L} y={y - 2} fontSize={9} fill="#94a3b8">
                {yFormat(maxV * g)}
              </text>
            </g>
          )
        })}
        {/* Series */}
        {series.map((s) => (
          <g key={s.label}>
            <path
              d={pathFor(s, minT, maxT, maxV)}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
            {s.points.length === 1 && (
              <circle
                cx={scaleX(s.points[0].t, minT, maxT)}
                cy={scaleY(s.points[0].v, maxV)}
                r={2.5}
                fill={s.color}
              />
            )}
          </g>
        ))}
      </svg>
      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-2">
        {series.map((s) => (
          <div key={s.label} className="flex items-center gap-1.5 text-xs text-gray-600">
            <span className="inline-block w-3 h-0.5 rounded" style={{ backgroundColor: s.color }} />
            {s.label}
          </div>
        ))}
      </div>
    </div>
  )
}

/** Stacked-area sparkline-style bar chart for a single bucketed series. */
export function BarSeries({
  points,
  color = '#3b82f6',
  height = 120,
  emptyLabel = 'No data yet',
  valueFormat = fmtTick,
}: {
  points: TimePoint[]
  color?: string
  height?: number
  emptyLabel?: string
  valueFormat?: (v: number) => string
}) {
  if (points.length === 0) {
    return <p className="text-gray-500 text-sm py-4 text-center">{emptyLabel}</p>
  }
  const maxV = Math.max(...points.map((p) => p.v), 1)
  const n = points.length
  const slot = (VIEW_W - PAD_L - PAD_R) / n
  const barW = Math.max(1, slot * 0.7)

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      preserveAspectRatio="none"
      className="w-full"
      style={{ height }}
      role="img"
      aria-label="bar series"
    >
      {points.map((p, i) => {
        const x = PAD_L + i * slot + (slot - barW) / 2
        const y = scaleY(p.v, maxV)
        const h = VIEW_H - PAD_B - y
        return (
          <rect key={i} x={x} y={y} width={barW} height={Math.max(0, h)} rx={1.5} fill={color}>
            <title>{valueFormat(p.v)}</title>
          </rect>
        )
      })}
    </svg>
  )
}

/** Headline metric tile with an optional delta indicator. */
export function MetricTile({
  label,
  value,
  delta,
  accent,
}: {
  label: string
  value: React.ReactNode
  delta?: number | null
  accent?: string
}) {
  return (
    <div className="text-center p-3 bg-gray-50 rounded-lg">
      <div className="text-2xl font-bold text-gray-900" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      <div className="text-xs text-gray-500 mt-1">{label}</div>
      {delta !== undefined && delta !== null && (
        <div className={`text-xs mt-0.5 font-medium ${delta >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
          {delta >= 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(1)}
        </div>
      )}
    </div>
  )
}
