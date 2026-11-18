/**
 * CASTER Phase 4 -- SportBadge component.
 *
 * Neo-Brutalist Glassmorphism iOS Sport Badge:
 * - High-voltage glass pill with glowing border and neon accent
 * - Matches the sleek aesthetic of the app
 */
import useCasterStore from '../../store/useCasterStore.js'

const SPORT_CONFIG = {
  cricket:  { emoji: '🏏', name: 'CRICKET' },
  football: { emoji: '⚽', name: 'FOOTBALL' },
}

export default function SportBadge() {
  const sport = useCasterStore((s) => s.sport)

  if (!sport) return null

  const config = SPORT_CONFIG[sport] || { emoji: '🎯', name: sport.toUpperCase() }

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 8,
        padding: '5px 14px',
        borderRadius: 999,
        background: 'rgba(255, 255, 255, 0.08)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        border: '1.5px solid rgba(255, 255, 255, 0.22)',
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.2)',
      }}
    >
      <span style={{ fontSize: '1.1rem', filter: 'drop-shadow(0 0 6px rgba(255, 255, 255, 0.4))' }}>
        {config.emoji}
      </span>
      <span style={{
        fontFamily: 'var(--font-display)',
        fontSize: '0.78rem',
        fontWeight: 900,
        letterSpacing: '0.06em',
        color: '#ffffff',
      }}>
        {config.name}
      </span>
      <span style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '0.62rem',
        fontWeight: 800,
        color: 'var(--volt)',
        background: 'rgba(212, 255, 0, 0.15)',
        border: '1px solid rgba(212, 255, 0, 0.35)',
        padding: '1px 6px',
        borderRadius: 4,
        letterSpacing: '0.04em',
      }}>
        DETECTED
      </span>
    </div>
  )
}
