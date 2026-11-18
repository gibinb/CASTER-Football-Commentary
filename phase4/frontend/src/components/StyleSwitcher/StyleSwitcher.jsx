/**
 * CASTER Phase 4 -- StyleSwitcher component.
 *
 * Neo-Brutalist Glassmorphism iOS Style Switcher:
 * - Three high-contrast frosted pills
 * - Tactile active state with high-voltage neon volt highlight
 * - Real-time LoRA prompt-conditioning adapter toggle
 */
import useCasterStore from '../../store/useCasterStore.js'

const STYLES = [
  { key: 'dramatic',    label: '🎭 DRAMATIC',    desc: 'High energy, exclamations, punchy cadence' },
  { key: 'casual',      label: '😎 CASUAL',      desc: 'Fan terrace vibes, relaxed flow' },
  { key: 'statistical', label: '📊 STATS',       desc: 'Tactical metrics, speed & zone data' },
]

export default function StyleSwitcher({ onStyleChange }) {
  const style    = useCasterStore((s) => s.style)
  const setStyle = useCasterStore((s) => s.setStyle)

  const handleClick = (key) => {
    setStyle(key)
    if (onStyleChange) onStyleChange(key)
  }

  return (
    <div className="brut-card" style={{ height: '100%' }}>
      <div className="brut-card-header">
        <span>🎙️ COMMENTARY PERSONA</span>
        <span style={{ fontSize: '0.62rem', background: 'rgba(255, 255, 255, 0.15)', padding: '2px 8px', borderRadius: 6 }}>
          STYLE
        </span>
      </div>
      <div className="brut-card-body" style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: 10 }}>
        {STYLES.map((s) => {
          const isActive = style === s.key
          return (
            <button
              key={s.key}
              className={isActive ? 'active' : ''}
              onClick={() => handleClick(s.key)}
              style={{
                width: '100%',
                textAlign: 'left',
                padding: '10px 14px',
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
                borderRadius: 12,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 900 }}>{s.label}</span>
                {isActive && (
                  <span style={{
                    fontSize: '0.58rem',
                    background: '#000',
                    color: 'var(--volt)',
                    padding: '2px 6px',
                    borderRadius: 4,
                    fontWeight: 900,
                  }}>
                    ACTIVE
                  </span>
                )}
              </div>
              <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', fontWeight: 500, opacity: isActive ? 0.9 : 0.65 }}>
                {s.desc}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
