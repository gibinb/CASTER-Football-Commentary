/**
 * CASTER Phase 4 -- EventFeed component.
 *
 * Neo-Brutalist Glassmorphism + Maximalist visual intensity:
 * - Ultra-frosted list cards
 * - High-voltage priority badges (GOAL, FOUR, SIX, WICKET)
 * - Hyper-contrast confidence telemetry meters
 */
import { useRef, useEffect } from 'react'
import useCasterStore from '../../store/useCasterStore.js'

const PRIORITY_EVENTS = new Set([
  'WICKET', 'BOWLED', 'CAUGHT_OUT', 'FOUR', 'SIX', 'GOAL',
])

function ConfBar({ confidence }) {
  const pct = Math.round(confidence * 100)
  const cls = confidence < 0.5 ? 'low' : confidence < 0.75 ? 'mid' : 'high'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 90 }}>
      <div className="conf-bar" style={{ flex: 1 }}>
        <div className={`conf-bar-fill ${cls}`} style={{ width: pct + '%' }} />
      </div>
      <span className="mono" style={{ fontSize: '0.68rem', fontWeight: 800, color: '#fff' }}>{pct}%</span>
    </div>
  )
}

export default function EventFeed() {
  const events   = useCasterStore((s) => s.events)
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [events])

  return (
    <div className="brut-card" style={{ height: '100%' }}>
      <div className="brut-card-header">
        <span>⚡ CONFIRMED EVENT STREAM</span>
        <span style={{
          background: 'rgba(255, 255, 255, 0.15)',
          padding: '2px 8px',
          borderRadius: 6,
          fontSize: '0.68rem',
          fontFamily: 'var(--font-mono)',
        }}>
          {events.length} EVENTS
        </span>
      </div>
      <div
        ref={scrollRef}
        style={{
          overflowY: 'auto',
          maxHeight: 180,
          padding: 8,
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
        }}
      >
        {events.length === 0 && (
          <p style={{ padding: '16px 12px', color: 'var(--text-dim)', fontStyle: 'italic', fontSize: '0.8rem', textAlign: 'center' }}>
            No detected match events yet. Feed activates on verified actions.
          </p>
        )}

        {events.map((ev, i) => {
          const isPriority = PRIORITY_EVENTS.has(ev.event)
          return (
            <div
              key={i}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 12px',
                borderRadius: 10,
                background: isPriority ? 'rgba(255, 19, 85, 0.22)' : 'rgba(255, 255, 255, 0.05)',
                border: isPriority ? '1.5px solid var(--crimson)' : '1px solid rgba(255, 255, 255, 0.08)',
                backdropFilter: 'blur(10px)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="mono" style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>
                  {ev.timestamp || `F#${ev.frame_idx || i}`}
                </span>
                <span style={{
                  fontFamily: 'var(--font-display)',
                  fontWeight: 900,
                  fontSize: '0.82rem',
                  letterSpacing: '0.04em',
                  color: isPriority ? 'var(--volt)' : '#ffffff',
                }}>
                  {ev.event}
                </span>
                {ev.gate_reason && (
                  <span style={{
                    fontSize: '0.62rem',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-dim)',
                    background: 'rgba(0, 0, 0, 0.3)',
                    padding: '2px 6px',
                    borderRadius: 4,
                  }}>
                    {ev.gate_reason}
                  </span>
                )}
              </div>
              <ConfBar confidence={ev.confidence || 0.8} />
            </div>
          )
        })}
      </div>
    </div>
  )
}
