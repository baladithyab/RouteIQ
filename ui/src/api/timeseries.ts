// Client-side time-series accumulation from RouteIQ's existing point-in-time
// stats endpoints (RouteIQ-c119 / RouteIQ-9d2d).
//
// RouteIQ's /routing/stats, /stats/global, and /me/stats are SINGLE-VALUE
// snapshots (counters + averages), not time-series. Rather than add a new
// python time-series endpoint (which would collide with the concurrent python
// waves), we sample those existing endpoints on the SPA's polling interval and
// keep a bounded in-memory ring buffer of samples. That gives us real
// over-time charts and a live decision feed without any backend change.

import { useEffect, useRef, useState } from 'react'

export interface Sample<T> {
  /** Capture time (epoch millis). */
  t: number
  value: T
}

const DEFAULT_CAP = 60

/**
 * Append `value` to a bounded, time-stamped ring buffer whenever it changes
 * (by reference). Returns the full sample history (oldest-first).
 *
 * The buffer is reference-stable across renders; only a NEW `value` (a fresh
 * query result) pushes a sample, so React-Query's refetch cadence drives the
 * sampling rate.
 */
export function useTimeSeries<T>(value: T | undefined, cap: number = DEFAULT_CAP): Sample<T>[] {
  const [history, setHistory] = useState<Sample<T>[]>([])
  const lastRef = useRef<T | undefined>(undefined)

  useEffect(() => {
    if (value === undefined) return
    if (lastRef.current === value) return
    lastRef.current = value
    // Sampling an external system (React Query's polled result) into a bounded
    // buffer — this is the "subscribe for updates" case the lint rule allows;
    // the lastRef guard makes it fire at most once per new query result.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setHistory((prev) => {
      const next = [...prev, { t: Date.now(), value }]
      return next.length > cap ? next.slice(next.length - cap) : next
    })
  }, [value, cap])

  return history
}

/** A single derived routing-decision event (delta between two snapshots). */
export interface DecisionEvent {
  /** Event time (epoch millis). */
  t: number
  /** Number of new decisions in this interval. */
  count: number
  /** Strategies that gained decisions this interval (name -> delta). */
  strategyDeltas: Record<string, number>
  /** Models that gained decisions this interval (name -> delta), if available. */
  modelDeltas: Record<string, number>
  /** Average decision latency (ms) reported at this sample. */
  avgLatencyMs: number
}

function diffDist(
  prev: Record<string, number> | undefined,
  curr: Record<string, number>,
): Record<string, number> {
  const out: Record<string, number> = {}
  for (const [k, v] of Object.entries(curr)) {
    const d = v - (prev?.[k] ?? 0)
    if (d > 0) out[k] = d
  }
  return out
}

export interface DecisionSnapshot {
  total_decisions: number
  strategy_distribution: Record<string, number>
  average_latency_ms: number
  model_distribution?: Record<string, number>
}

/**
 * Derive a bounded feed of routing-decision EVENTS from successive snapshots of
 * a counter-based stats endpoint. Each event is the positive delta between two
 * consecutive samples — i.e. the decisions that happened in that interval.
 */
export function useDecisionFeed(
  snapshot: DecisionSnapshot | undefined,
  cap: number = DEFAULT_CAP,
): DecisionEvent[] {
  const [feed, setFeed] = useState<DecisionEvent[]>([])
  const prevRef = useRef<DecisionSnapshot | undefined>(undefined)
  const lastSeenRef = useRef<DecisionSnapshot | undefined>(undefined)

  useEffect(() => {
    if (snapshot === undefined) return
    if (lastSeenRef.current === snapshot) return
    lastSeenRef.current = snapshot

    const prev = prevRef.current
    prevRef.current = snapshot
    if (prev === undefined) return // first sample: no delta yet

    const count = snapshot.total_decisions - prev.total_decisions
    if (count <= 0) return // no progress (or a counter reset); skip

    const event: DecisionEvent = {
      t: Date.now(),
      count,
      strategyDeltas: diffDist(prev.strategy_distribution, snapshot.strategy_distribution),
      modelDeltas: diffDist(prev.model_distribution, snapshot.model_distribution ?? {}),
      avgLatencyMs: snapshot.average_latency_ms,
    }
    // Same external-system sampling case as useTimeSeries: the lastSeenRef
    // guard ensures one append per new polled snapshot, not a render cascade.
    setFeed((p) => {
      const next = [event, ...p] // newest-first
      return next.length > cap ? next.slice(0, cap) : next
    })
  }, [snapshot, cap])

  return feed
}
