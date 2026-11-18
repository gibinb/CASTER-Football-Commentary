/**
 * CASTER Phase 4 -- App.jsx
 *
 * Striking Apple iOS Glassmorphism × Neo-Brutalism × Maximalism
 *
 * Architecture & Aesthetics:
 * - Multi-layered frosted glass panels (backdrop-filter: blur(28px))
 * - High-voltage maximalist accents: Neon Volt (#d4ff00), Hot Crimson (#ff1355), Apple Cyan (#00f2fe)
 * - Neo-brutalist structural outlines and crisp pill badges
 * - Full telemetry layout:
 *     Left column: High-res video canvas + side-by-side Commentary Style & 12D Radar
 *     Right column: Top side-by-side Scoreboard & Live Commentary, Middle Event Feed, Bottom Transcript Log
 */
import { useState, useCallback, useRef } from 'react'
import VideoPlayer    from './components/VideoPlayer/VideoPlayer.jsx'
import LiveCaption    from './components/LiveCaption/LiveCaption.jsx'
import StyleSwitcher  from './components/StyleSwitcher/StyleSwitcher.jsx'
import SportBadge     from './components/SportBadge/SportBadge.jsx'
import ScoreBoard     from './components/ScoreBoard/ScoreBoard.jsx'
import EventFeed      from './components/EventFeed/EventFeed.jsx'
import TranscriptLog  from './components/TranscriptLog/TranscriptLog.jsx'
import { useWebSocket } from './hooks/useWebSocket.js'
import { useTTS }       from './hooks/useTTS.js'
import useCasterStore   from './store/useCasterStore.js'

// Dynamic API origin for seamless deployment
const API = window.location.port === '5173'
  ? 'http://localhost:8000'
  : `${window.location.protocol}//${window.location.host}`

export default function App() {
  const [videoFile, setVideoFile]   = useState(null)
  const [dragOver, setDragOver]     = useState(false)
  const [uploading, setUploading]   = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const fileInputRef = useRef(null)

  const wsStatus    = useCasterStore((s) => s.wsStatus)
  const wsError     = useCasterStore((s) => s.wsError)
  const sport       = useCasterStore((s) => s.sport)
  const clipName    = useCasterStore((s) => s.clipName)
  const totalEvents = useCasterStore((s) => s.totalEvents)
  const muted       = useCasterStore((s) => s.muted)
  const style       = useCasterStore((s) => s.style)
  const { setClipName, toggleMute, reset } = useCasterStore()

  const { speak, mute, warmUp } = useTTS()

  const onSpeak = useCallback((text, lang) => {
    if (!muted) speak(text, lang)
  }, [muted, speak])

  const { sendStyle, sendControl } = useWebSocket(clipName, style, onSpeak)

  const handleFile = useCallback(async (file) => {
    if (!file) return
    setVideoFile(file)
    setUploadError(null)
    setUploading(true)

    warmUp()

    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API}/api/upload`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`)
      const data = await res.json()
      reset()
      setClipName(data.clip_name)
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setUploading(false)
    }
  }, [reset, setClipName, warmUp])

  const onDropFile = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onFileInput = useCallback((e) => {
    const file = e.target.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const handleMuteToggle = useCallback(() => {
    if (!muted) mute()
    toggleMute()
  }, [muted, mute, toggleMute])

  const handleStyleChange = useCallback((newStyle) => {
    sendStyle(newStyle)
  }, [sendStyle])

  const handleReset = useCallback(() => {
    reset()
    setVideoFile(null)
    setUploadError(null)
    mute()
  }, [reset, mute])

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* ======== APPLE FROSTED GLASS HEADER ======== */}
      <header style={{
        position: 'sticky',
        top: 0,
        zIndex: 100,
        background: 'rgba(15, 18, 28, 0.65)',
        backdropFilter: 'blur(30px) saturate(190%)',
        WebkitBackdropFilter: 'blur(30px) saturate(190%)',
        borderBottom: '1.5px solid rgba(255, 255, 255, 0.14)',
        padding: '12px 28px',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 16,
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.35)',
      }}>
        {/* Brand Logo & Subtitle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 42,
            height: 42,
            borderRadius: 14,
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.05) 100%)',
            backdropFilter: 'blur(16px)',
            border: '2px solid rgba(255, 255, 255, 0.35)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 6px 20px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.4)',
            fontSize: '1.4rem',
          }}>
            ⚽
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h1 style={{
                fontFamily: 'var(--font-display)',
                fontSize: '1.7rem',
                fontWeight: 900,
                letterSpacing: '-0.02em',
                lineHeight: 1,
                color: '#ffffff',
                textShadow: '0 2px 10px rgba(0, 0, 0, 0.5)',
              }}>
                CASTER
              </h1>
            </div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.65rem',
              color: 'var(--text-dim)',
              marginTop: 3,
              letterSpacing: '0.04em',
              fontWeight: 700,
              textTransform: 'uppercase',
            }}>
              Computer Vision Assisted Sports Text And Event Reporter
            </div>
          </div>
        </div>

        {/* Central Badges & Status */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
          {sport && <SportBadge />}

          <span className={`badge ${wsStatus === 'connected' ? 'badge-live' : wsStatus === 'done' ? 'badge-done' : wsStatus === 'error' ? 'badge-error' : 'badge-warn'}`}>
            {wsStatus === 'connected' ? '● LIVE AI ON AIR' : wsStatus === 'done' ? `✓ SESSION COMPLETED (${totalEvents} events)` : wsStatus === 'error' ? '✗ CONNECTION ERROR' : wsStatus.toUpperCase()}
          </span>

          {clipName && (
            <span style={{
              background: 'rgba(255, 255, 255, 0.08)',
              color: '#ffffff',
              fontSize: '0.72rem',
              fontFamily: 'var(--font-mono)',
              fontWeight: 700,
              padding: '4px 12px',
              borderRadius: 999,
              border: '1.5px solid rgba(255, 255, 255, 0.2)',
              backdropFilter: 'blur(10px)',
            }}>
              MATCH: <strong style={{ color: 'var(--volt)' }}>{clipName}</strong>
            </span>
          )}
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button
            onClick={handleMuteToggle}
            style={{
              background: muted ? 'rgba(255, 19, 85, 0.2)' : 'rgba(212, 255, 0, 0.15)',
              borderColor: muted ? 'var(--crimson)' : 'var(--volt)',
              color: muted ? '#ff1355' : 'var(--volt)',
            }}
          >
            {muted ? '🔇 AUDIO OFF' : '🔊 LIVE SPEECH ON'}
          </button>

          {clipName && (
            <button
              onClick={handleReset}
              className="btn-volt"
              style={{
                background: '#ffffff',
                color: '#000000',
              }}
            >
              🔄 NEW FOOTAGE
            </button>
          )}
        </div>
      </header>

      {/* ======== MAIN VIEWPORT ======== */}
      <main style={{ flex: 1, padding: 20 }}>

        {/* ---- FROSTED HERO UPLOAD ZONE ---- */}
        {!clipName && (
          <div style={{ maxWidth: 940, margin: '50px auto 0' }}>
            <div
              className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDropFile}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={onFileInput}
              />
              <div className="glass-icon-box">
                ⚽
              </div>

              <h2 style={{
                fontFamily: 'var(--font-display)',
                fontSize: '2rem',
                fontWeight: 900,
                color: '#ffffff',
                letterSpacing: '-0.02em',
                marginBottom: 8,
              }}>
                DROP FOOTBALL FOOTAGE HERE
              </h2>

              <p style={{
                fontFamily: 'var(--font-body)',
                fontSize: '0.95rem',
                color: 'var(--text-dim)',
                maxWidth: 500,
                margin: '0 auto',
                fontWeight: 600,
              }}>
                Upload any match clip (.MP4, .MOV, .MKV) to launch instant player tracking, event classification & AI commentary.
              </p>

              {uploading && (
                <div style={{
                  marginTop: 24,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 10,
                  background: 'var(--volt)',
                  color: '#000',
                  fontWeight: 900,
                  padding: '8px 20px',
                  borderRadius: 999,
                  border: '2px solid #000',
                  boxShadow: '0 0 20px rgba(212, 255, 0, 0.5)',
                }}>
                  ⏳ INGESTING FOOTAGE & STARTING VISION ENGINES...
                </div>
              )}

              {uploadError && (
                <div style={{
                  marginTop: 24,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 10,
                  background: '#ff1355',
                  color: '#fff',
                  fontWeight: 900,
                  padding: '8px 20px',
                  borderRadius: 999,
                  border: '2px solid #fff',
                }}>
                  ✗ {uploadError}
                </div>
              )}
            </div>

            {/* Maximalist Telemetry Stack Badges */}
            <div style={{
              marginTop: 24,
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: 12,
            }}>
              <div className="brut-card" style={{ padding: '12px 16px' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)', fontWeight: 800 }}>PHASE 1 DETECTION</span>
                <span style={{ fontSize: '0.9rem', fontWeight: 900, color: 'var(--volt)', marginTop: 2 }}>YOLOv8 + ByteTrack</span>
              </div>
              <div className="brut-card" style={{ padding: '12px 16px' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)', fontWeight: 800 }}>PHASE 2 CLASSIFIER</span>
                <span style={{ fontSize: '0.9rem', fontWeight: 900, color: '#00f2fe', marginTop: 2 }}>12D XGBoost Engine</span>
              </div>
              <div className="brut-card" style={{ padding: '12px 16px' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)', fontWeight: 800 }}>PHASE 3 NARRATIVE</span>
                <span style={{ fontSize: '0.9rem', fontWeight: 900, color: '#ff1355', marginTop: 2 }}>Groq Ultra-Fast LLaMA 3</span>
              </div>
              <div className="brut-card" style={{ padding: '12px 16px' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)', fontWeight: 800 }}>PHASE 4 BROADCAST</span>
                <span style={{ fontSize: '0.9rem', fontWeight: 900, color: '#ffffff', marginTop: 2 }}>WebSockets + Native TTS</span>
              </div>
            </div>
          </div>
        )}

        {/* ---- ACTIVE PIPELINE WORKSPACE ---- */}
        {clipName && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(550px, 1.4fr) minmax(420px, 1fr)',
            gap: 18,
            maxWidth: 1680,
            margin: '0 auto',
          }}>
            {/* Left Column: Video Player + Controls */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Video Player Card */}
              <div className="brut-card" style={{ padding: 0, overflow: 'hidden', background: '#000000' }}>
                <VideoPlayer videoFile={videoFile} sendControl={sendControl} />
              </div>

              {/* Bottom Left: Commentary Persona & 12D Feature Radar */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <StyleSwitcher onStyleChange={handleStyleChange} />
                <FeatureVectorPanel />
              </div>
            </div>

            {/* Right Column: Scoreboard/Commentary -> Event Feed -> Transcript Archive */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Top Row: Scoreboard + Live Commentary side-by-side */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <ScoreBoard />
                <LiveCaption />
              </div>

              {/* Middle: Real-time Event Feed */}
              <div style={{ height: 210 }}>
                <EventFeed />
              </div>

              {/* Bottom: Commentary Transcript Archive */}
              <div style={{ flex: 1, minHeight: 280 }}>
                <TranscriptLog />
              </div>
            </div>
          </div>
        )}

        {/* Error Notification */}
        {wsError && (
          <div style={{
            marginTop: 16,
            background: 'var(--crimson)',
            color: '#fff',
            fontFamily: 'var(--font-mono)',
            padding: '12px 20px',
            borderRadius: 12,
            fontWeight: 800,
            boxShadow: '0 0 30px rgba(255, 19, 85, 0.7)',
            border: '2px solid #fff',
          }}>
            ⚠️ SYSTEM ERROR: {wsError}
          </div>
        )}
      </main>

      {/* ======== APPLE MINIMALIST DENSE FOOTER ======== */}
      <footer style={{
        background: 'rgba(12, 14, 20, 0.8)',
        backdropFilter: 'blur(20px)',
        borderTop: '1.5px solid rgba(255, 255, 255, 0.12)',
        padding: '10px 28px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.68rem',
        color: 'var(--text-dim)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: 'var(--volt)', fontWeight: 900 }}>⚡ CASTER AI</span>
          <span>— Autonomous Computer Vision Commentary System</span>
        </div>
        <div style={{ display: 'flex', gap: 16 }}>
          <span>Zero Hallucination Grounding</span>
          <span>•</span>
          <span style={{ color: '#fff', fontWeight: 700 }}>Real-time Edge Vision</span>
        </div>
      </footer>
    </div>
  )
}

// ---- Feature Vector Radar Panel ----
function FeatureVectorPanel() {
  const latestFrame = useCasterStore((s) => s.latestFrame)
  const fv = latestFrame?.feature_vector

  const LABELS = ['ball_x','ball_y','dx','dy','spd','dist','near','zone','offcl','hgt','ctct','bndy']

  return (
    <div className="brut-card" style={{ height: '100%' }}>
      <div className="brut-card-header">
        <span>📊 12D TELEMETRY VECTOR</span>
        <span style={{ fontSize: '0.62rem', background: 'rgba(255, 255, 255, 0.15)', padding: '2px 8px', borderRadius: 6 }}>
          ML FEATURES
        </span>
      </div>
      <div className="brut-card-body" style={{ padding: 10 }}>
        {!fv && (
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)', textAlign: 'center', padding: '16px 0' }}>
            Awaiting active frame telemetry...
          </p>
        )}
        {fv && (
          <div className="fv-grid">
            {fv.map((val, i) => (
              <div key={i} className={`fv-cell ${val !== 0 ? 'active' : ''}`} title={LABELS[i]}>
                <div style={{ fontSize: '0.58rem', opacity: 0.7, fontWeight: 700 }}>{LABELS[i]}</div>
                <div style={{ marginTop: 2, fontWeight: 800 }}>{typeof val === 'number' ? val.toFixed(2) : val}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}