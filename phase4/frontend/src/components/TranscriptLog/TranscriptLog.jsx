/**
 * CASTER Phase 4 -- TranscriptLog component.
 *
 * Neo-Brutalist Glassmorphism + Maximalist Transcript Archive:
 * - Dynamic pill tags for backends (Groq LLM / Template / Local GGUF)
 * - Copy button with Apple tactile response
 * - Hyper-readable typography with dark contrast backdrop
 */
import { useRef, useEffect } from 'react'
import useCasterStore from '../../store/useCasterStore.js'

const BACKEND_BADGE = {
  template:   { label: 'TPL',  bg: 'rgba(255, 255, 255, 0.15)', color: '#fff', border: '1px solid rgba(255, 255, 255, 0.3)' },
  groq:       { label: 'GROQ', bg: 'var(--volt)', color: '#000', border: '1px solid #000' },
  local_gguf: { label: 'LLM',  bg: '#00f2fe', color: '#000', border: '1px solid #000' },
}

export default function TranscriptLog() {
  const transcript = useCasterStore((s) => s.transcript)
  const scrollRef  = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [transcript])

  const copyAll = () => {
    const text = transcript
      .map((t) => `[${t.time || '?'}] ${t.event}: ${t.text}`)
      .join('\n')
    navigator.clipboard.writeText(text).catch(() => {})
  }

  return (
    <div className="brut-card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="brut-card-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>📜 COMMENTARY ARCHIVE</span>
          <span style={{
            background: 'rgba(255, 255, 255, 0.15)',
            padding: '2px 8px',
            borderRadius: 6,
            fontSize: '0.68rem',
            fontFamily: 'var(--font-mono)',
          }}>
            {transcript.length}
          </span>
        </div>
        {transcript.length > 0 && (
          <button
            onClick={copyAll}
            className="btn-volt"
            style={{ fontSize: '0.65rem', padding: '4px 10px', height: 'auto' }}
          >
            📋 COPY ALL
          </button>
        )}
      </div>

      <div
        ref={scrollRef}
        style={{
          overflowY: 'auto',
          flex: 1,
          minHeight: 200,
          maxHeight: 340,
          padding: 8,
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
        }}
      >
        {transcript.length === 0 && (
          <p style={{ padding: '24px 16px', color: 'var(--text-dim)', fontStyle: 'italic', fontSize: '0.82rem', textAlign: 'center' }}>
            Live transcript log builds here in real time as events occur.
          </p>
        )}

        {transcript.map((entry, i) => {
          const badge = BACKEND_BADGE[entry.backend_used] || {
            label: entry.backend_used?.toUpperCase()?.slice(0, 4) || 'AI',
            bg: 'rgba(255, 255, 255, 0.15)',
            color: '#fff',
            border: '1px solid rgba(255, 255, 255, 0.3)',
          }
          return (
            <div
              key={i}
              style={{
                display: 'flex',
                gap: 12,
                alignItems: 'flex-start',
                padding: '8px 12px',
                borderRadius: 10,
                background: 'rgba(0, 0, 0, 0.25)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                backdropFilter: 'blur(10px)',
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 65 }}>
                <span className="mono" style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>
                  {entry.time || `F#${entry.frame_idx || i}`}
                </span>
                <span
                  style={{
                    background: badge.bg,
                    color: badge.color,
                    border: badge.border,
                    fontSize: '0.58rem',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 900,
                    padding: '2px 4px',
                    borderRadius: 4,
                    textAlign: 'center',
                  }}
                >
                  {badge.label}
                </span>
              </div>

              <div style={{ flex: 1 }}>
                <div style={{
                  fontSize: '0.72rem',
                  fontFamily: 'var(--font-display)',
                  fontWeight: 900,
                  color: 'var(--volt)',
                  marginBottom: 2,
                  letterSpacing: '0.04em',
                }}>
                  {entry.event}
                </div>
                <div style={{
                  fontSize: '0.85rem',
                  lineHeight: 1.45,
                  color: '#ffffff',
                  fontWeight: 500,
                }}>
                  {entry.text}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
