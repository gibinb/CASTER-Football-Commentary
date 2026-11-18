/**
 * CASTER Phase 4 -- LiveCaption component.
 *
 * Neo-Brutalist Frosted Glass iOS Live Caption:
 * - Frosted dynamic island aesthetic
 * - Maximalist live pulse indicator
 * - Dynamic token highlights with ultra-crisp readable typography
 */
import { useEffect } from 'react'
import useCasterStore from '../../store/useCasterStore.js'

export default function LiveCaption() {
  const captionWords       = useCasterStore((s) => s.captionWords)
  const lastCommentaryText = useCasterStore((s) => s.lastCommentaryText)
  const captionFlash       = useCasterStore((s) => s.captionFlash)
  const clearFlash         = useCasterStore((s) => s.clearCaptionFlash)
  const wsStatus           = useCasterStore((s) => s.wsStatus)

  // Clear the flash after 600ms
  useEffect(() => {
    if (!captionFlash) return
    const id = setTimeout(clearFlash, 600)
    return () => clearTimeout(id)
  }, [captionFlash, clearFlash])

  const isStreaming = captionWords.length > 0
  const hasPrevious = Boolean(lastCommentaryText)

  return (
    <div className="brut-card" style={{ height: '100%', minHeight: 110, display: 'flex', flexDirection: 'column' }}>
      <div className="brut-card-header">
        <span>🎙️ LIVE COMMENTARY</span>
        {wsStatus === 'connected' && (
          <span className="badge badge-live" style={{ fontSize: '0.62rem', padding: '2px 8px' }}>
            ● ON AIR
          </span>
        )}
      </div>
      <div
        className="brut-card-body"
        style={{
          background: captionFlash ? 'rgba(212, 255, 0, 0.25)' : 'rgba(0, 0, 0, 0.25)',
          transition: 'background 0.25s ease',
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          padding: '12px 16px',
        }}
      >
        {!isStreaming && !hasPrevious && wsStatus === 'idle' && (
          <p style={{ color: 'var(--text-dim)', fontStyle: 'italic', fontSize: '0.82rem' }}>
            Upload video footage to initiate real-time AI live broadcast...
          </p>
        )}
        {!isStreaming && !hasPrevious && wsStatus === 'connected' && (
          <p style={{ color: 'var(--volt)', fontStyle: 'italic', fontSize: '0.82rem', fontWeight: 600 }}>
            Scanning pitch events & players...
          </p>
        )}

        {/* Word-by-word streaming caption */}
        {isStreaming && (
          <div style={{
            fontFamily: 'var(--font-body)',
            fontSize: '1.05rem',
            fontWeight: 700,
            lineHeight: 1.4,
            color: '#ffffff',
          }}>
            {captionWords.map((word, i) => (
              <span
                key={i}
                style={{
                  marginRight: 4,
                  display: 'inline-block',
                  color: i === captionWords.length - 1 ? 'var(--volt)' : '#ffffff',
                  fontWeight: i === captionWords.length - 1 ? 900 : 700,
                  textShadow: i === captionWords.length - 1 ? '0 0 12px rgba(212, 255, 0, 0.8)' : 'none',
                }}
              >
                {word}
              </span>
            ))}
          </div>
        )}

        {/* Retain last commentary sentence on screen */}
        {!isStreaming && hasPrevious && (
          <div style={{
            fontFamily: 'var(--font-body)',
            fontSize: '0.98rem',
            fontWeight: 700,
            lineHeight: 1.4,
            color: '#f8fafc',
          }}>
            {lastCommentaryText}
          </div>
        )}
      </div>
    </div>
  )
}
