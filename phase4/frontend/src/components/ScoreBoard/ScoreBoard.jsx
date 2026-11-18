/**
 * CASTER Phase 4 -- ScoreBoard component.
 *
 * Striking Apple Glassmorphism + Neo-Brutalism & Maximalism:
 * - Ultra-frosted glass backdrop
 * - Massive high-voltage numeric display (Maximalist typography)
 * - Brutalist pill tags with bold contrast
 */
import useCasterStore from '../../store/useCasterStore.js'

export default function ScoreBoard() {
  const score     = useCasterStore((s) => s.score)
  const matchTime = useCasterStore((s) => s.matchTime)
  const sport     = useCasterStore((s) => s.sport)

  const timeLabel = sport === 'cricket' ? 'OVER' : "MIN'"

  return (
    <div className="brut-card" style={{ height: '100%', minHeight: 110, display: 'flex', flexDirection: 'column' }}>
      <div className="brut-card-header">
        <span>⚡ SCOREBOARD</span>
        <span style={{
          background: 'var(--volt)',
          color: '#000',
          fontSize: '0.62rem',
          padding: '2px 8px',
          borderRadius: 6,
          fontWeight: 900,
          border: '1px solid #000',
        }}>
          MATCH STATS
        </span>
      </div>
      <div style={{
        padding: '12px 14px',
        textAlign: 'center',
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        background: 'rgba(0, 0, 0, 0.25)',
      }}>
        <div style={{
          fontFamily: 'var(--font-display)',
          fontSize: '2.5rem',
          fontWeight: 900,
          color: '#ffffff',
          letterSpacing: '-0.03em',
          lineHeight: 1,
          textShadow: '0 4px 20px rgba(0, 0, 0, 0.6), 0 0 30px rgba(255, 255, 255, 0.3)',
        }}>
          {score || '0 - 0'}
        </div>
        {matchTime && (
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.72rem',
            color: 'var(--volt)',
            marginTop: 6,
            fontWeight: 800,
            background: 'rgba(212, 255, 0, 0.12)',
            padding: '2px 10px',
            borderRadius: 999,
            border: '1px solid rgba(212, 255, 0, 0.3)',
          }}>
            {timeLabel}: {matchTime}
          </div>
        )}
      </div>
    </div>
  )
}
