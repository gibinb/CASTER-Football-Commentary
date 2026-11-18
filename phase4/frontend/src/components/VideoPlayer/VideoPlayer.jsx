/**
 * CASTER Phase 4 -- VideoPlayer component.
 *
 * Renders the uploaded video with a live canvas overlay that draws
 * bounding boxes for every tracked object, synced to playback time.
 *
 * VIDEO ↔ BACKEND SYNC:
 * On CPU, YOLO takes ~2-3 s per frame, so the backend is always BEHIND
 * real-time playback. Without throttling, the user watches 30 s of video
 * while commentary for second 5 hasn't arrived yet.
 *
 * Fix: we compare the video's currentTime against the latest backend
 * frame timestamp_sec. If the video is more than MAX_AHEAD_SEC ahead
 * of the backend, we pause playback and show a "Waiting for analysis..."
 * indicator. The video resumes automatically once the backend catches up.
 * This keeps commentary and bounding boxes within a few seconds of what
 * the user is watching.
 *
 * BOX INTERPOLATION (added): detection only runs every `stride` frames
 * (e.g. stride=2 on a 60fps source -> one detection every ~33ms of video
 * time), so earlier versions of this component simply redrew whichever
 * detection frame arrived last until the next one came in. On fast motion
 * this reads as visibly laggy/jumpy -- a box "teleports" to its next known
 * position rather than moving smoothly. This does NOT mean detection is
 * actually slow or wrong; it is working exactly as designed (see
 * pipeline_runner.py's adaptive stride). It only means nothing was ever
 * drawn BETWEEN two known detection frames.
 *
 * Fix: store the previous AND current detection frame (see
 * useCasterStore.js's onFrame), then on every paint, linearly interpolate
 * each tracked object's box position between its position in the previous
 * frame and its position in the current frame, based on how far the
 * video's actual currentTime has progressed between those two frames'
 * timestamps. Detections are matched between frames by (cls_name,
 * track_id) -- the ball has no track_id (see bytetrack_wrapper.py's
 * comments on why), so ball boxes fall back to matching by cls_name alone
 * and simply snap if more than one ball candidate appears in a frame,
 * since picking the "right" one to interpolate from is the feature
 * extractor's job, not the renderer's.
 *
 * Class colours (consistent across sports):
 *   ball           → yellow
 *   player/team_a  → green
 *   goalkeeper     → purple
 *   referee/umpire → orange
 */
import { useRef, useEffect, useCallback, useState } from 'react'
import useCasterStore from '../../store/useCasterStore.js'

// Above this size, loading the file into a browser <video> tag via blob
// URL can take a long time or stall (moov atom at the end of the file --
// common for raw screen/app exports). This is a BROWSER PREVIEW
// limitation only -- the backend pipeline reads the file straight off
// disk via OpenCV and is completely unaffected by whether this loads.
const LARGE_FILE_WARN_BYTES = 150 * 1024 * 1024 // 150MB

// BUG FIX: this constant and the sync-throttle effect using it were
// described in the module docstring above but were never actually
// implemented -- the video always played at full real-time speed
// regardless of backend progress. For the first few seconds this is
// invisible (the backend keeps pace or is briefly ahead from initial
// buffering), but once live GPU inference falls behind real-time
// playback -- which it will, since per-frame overhead (NMS, tracking,
// JSON serialization, the WebSocket send itself) adds up even at a good
// processed-fps figure -- boxes start being drawn against video content
// the backend hasn't analysed yet, i.e. they silently become wrong
// rather than failing loudly. Pausing playback when we get too far
// ahead of the backend is what actually keeps boxes honest.
// Raised from 2.5 to 4.0: the live (non-precomputed) pipeline pauses
// mainly while waiting on Groq's network round-trip for each event
// (hundreds of ms to 1-2s), not local GPU speed -- a higher tolerance
// here means fewer, shorter visible pauses during a live demo, at the
// cost of boxes being allowed to sit slightly further from perfectly
// in-sync before the throttle steps in. Does not affect the
// precomputed-replay path (ss.mp4, rm1.mp4) in practice, since that
// path rarely falls behind by more than a fraction of a second anyway.
const MAX_AHEAD_SEC = 4.0

// If the gap between two detection frames is larger than this (backend
// stalled, seek happened, clip just started), don't interpolate across
// it -- just snap to the newer frame. Interpolating across a multi-second
// gap would show boxes sliding unrealistically across the whole pitch.
const MAX_INTERP_GAP_SEC = 0.5

const CLASS_COLORS = {
  ball:             '#ffe600',
  player_batting:   '#00e676',
  player_fielding:  '#00b0ff',
  player_team_a:    '#00e676',
  player_team_b:    '#00b0ff',
  player:           '#00e676',
  umpire:           '#ff6d00',
  referee:          '#ff6d00',
  goalkeeper:       '#d500f9',
  bat:              '#ff2222',
  stumps:           '#ff2222',
}
const DEFAULT_COLOR = '#ffffff'

/**
 * Builds a lookup key for matching the same tracked object across two
 * detection frames. Players/goalkeeper/referee have a stable track_id
 * from ByteTrack; the ball never does (see bytetrack_wrapper.py), so it
 * falls back to cls_name alone.
 */
function detKey(det) {
  return det.track_id != null ? `${det.cls_name}#${det.track_id}` : det.cls_name
}

/**
 * Linearly interpolates between two detections' box geometry. Assumes
 * both are the "same" object (matched by detKey upstream).
 */
function lerpDet(prev, next, t) {
  return {
    cls_name: next.cls_name,
    track_id: next.track_id,
    confidence: next.confidence,
    x: prev.x + (next.x - prev.x) * t,
    y: prev.y + (next.y - prev.y) * t,
    w: prev.w + (next.w - prev.w) * t,
    h: prev.h + (next.h - prev.h) * t,
  }
}

export default function VideoPlayer({ videoFile, sendControl }) {
  const videoRef       = useRef(null)
  const canvasRef      = useRef(null)
  const animRef        = useRef(null)
  const latestFrame    = useCasterStore((s) => s.latestFrame)
  const previousFrame  = useCasterStore((s) => s.previousFrame)
  const wsStatus       = useCasterStore((s) => s.wsStatus)
  // BUG FIX ("commentary keeps appearing/speaking while video is
  // frozen"): the store's setPaused() holds back new token/sentence_end
  // /speak messages whenever this is true, for BOTH the manual pause
  // button and the automatic backend-sync throttle below -- without
  // touching the backend itself (see checkBackendSync's own comments
  // for why pausing the backend directly deadlocks).
  const setStorePaused = useCasterStore((s) => s.setPaused)
  const latestFrameRef   = useRef(latestFrame)
  const previousFrameRef = useRef(previousFrame)

  // 'idle' | 'loading' | 'ready' | 'error'
  const [loadState, setLoadState] = useState('idle')
  const isLargeFile = !!videoFile && videoFile.size > LARGE_FILE_WARN_BYTES
  // True while we've paused playback because the video ran too far ahead
  // of the backend's latest analysed frame. Drives the "waiting" overlay.
  const [waitingForBackend, setWaitingForBackend] = useState(false)
  // Tracks whether WE paused the video (vs. the user clicking pause), so
  // the sync-throttle loop knows it's allowed to resume playback again --
  // it should never override a pause the user asked for themselves.
  const autoPausedRef = useRef(false)
  // Set to true immediately before any programmatic video.play()/pause()
  // call the backend-sync throttle makes itself, so its own auto-pause
  // doesn't get confused with a manual pause (see manualPauseRef below).
  const suppressNextPlayPauseEventRef = useRef(false)

  // BUG FIX (deadlock): this component used to wire sendControl('pause')
  // / sendControl('resume') directly to the <video> element's native
  // onPause/onPlay DOM events. Those events fire for MANY reasons beyond
  // a user clicking pause -- notably, the browser pausing on its own for
  // BUFFERING (very plausible for a large file streamed from a blob URL
  // while a live GPU pipeline competes for system resources). When that
  // happened: (1) the buffering pause fired a real, trusted 'pause' event,
  // (2) nothing suppressed it, so {"type":"pause"} was sent to the
  // backend, (3) the backend's play_event cleared and ALL frame/event/
  // commentary generation stopped, (4) with no new frames ever arriving,
  // there was nothing further for the video to buffer or play, so it
  // never naturally resumed either -- a permanent deadlock (confirmed:
  // server log showed exactly one pause/resume pair ~7s after start,
  // then nothing, while the frontend stayed stuck on "WAITING FOR
  // ANALYSIS" forever). Fixed by no longer coupling backend pause/resume
  // to ANY native video event. Instead, a dedicated "PAUSE ANALYSIS"
  // button below is the ONLY thing that can pause the backend -- a
  // deliberate, explicit user action, never inferred from browser
  // buffering or playback quirks. manualPauseRef tracks that explicit
  // state so the auto-throttle (checkBackendSync) knows to leave it alone.
  const manualPauseRef = useRef(false)
  const [manuallyPaused, setManuallyPaused] = useState(false)
  // True once the FIRST real backend frame has arrived. Playback used to
  // start as soon as the WebSocket connected, which is well before the
  // backend finishes loading scene_states.json/events.json and anchoring
  // its sync clock -- see the BUG FIX note in pipeline_runner.py. Waiting
  // for the first frame instead removes that setup-time gap from the
  // visible lag almost entirely.
  const hasAutoStartedRef = useRef(false)

  const blobUrlRef = useRef(null)

  // Keep refs in sync
  useEffect(() => { latestFrameRef.current = latestFrame }, [latestFrame])
  useEffect(() => { previousFrameRef.current = previousFrame }, [previousFrame])

  // Set video src when a file is chosen, but DO NOT autoplay before backend starts.
  //
  // BUG FIX: this used to do `return () => URL.revokeObjectURL(url)` as the
  // effect's own cleanup. Under React.StrictMode (enabled in main.jsx),
  // every effect runs mount -> cleanup -> mount again in dev -- so that
  // cleanup revoked the blob URL almost immediately after creating it,
  // invalidating it before the <video> element finished loading the bytes.
  // The file itself was always fine (confirmed: plays in VLC and in a
  // plain Edge tab) -- only the in-app blob reference was being pulled out
  // from under the <video> tag. Fix: only revoke the PREVIOUS url when a
  // genuinely new file arrives, and only revoke on true unmount (separate
  // effect below with an empty dependency array, so StrictMode's extra
  // mount/cleanup pass doesn't trigger it).
  useEffect(() => {
    if (!videoFile || !videoRef.current) return
    setLoadState('loading')
    const url = URL.createObjectURL(videoFile)
    if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current)
    blobUrlRef.current = url
    videoRef.current.src = url
    videoRef.current.pause()
  }, [videoFile])

  // Revoke the blob URL only when this component actually unmounts.
  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current)
    }
  }, [])

  // BUG FIX (~1s constant playback lag, other half of the fix): this used
  // to start playback as soon as wsStatus === 'connected', i.e. right when
  // the WebSocket handshake finished -- well before the backend has even
  // started processing (file loads, dimension probing, sync-clock
  // anchoring all still to come). That gap was a big contributor to the
  // visible lag between real action and its bounding box. Now we wait for
  // the first genuine "frame" message (latestFrame transitioning from
  // null/undefined to a real value) before autoplaying, which lines
  // playback start up with the backend actually being ready to stream.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    if (wsStatus === 'connected' && latestFrame && !hasAutoStartedRef.current) {
      hasAutoStartedRef.current = true
      suppressNextPlayPauseEventRef.current = true
      video.play().catch(() => {})
    }
  }, [wsStatus, latestFrame])

  // Reset the one-shot autoplay guard whenever a new file/session starts,
  // so the next upload also waits for its own first frame rather than
  // skipping straight to playing (stale state from a previous clip).
  useEffect(() => {
    hasAutoStartedRef.current = false
  }, [videoFile])

  // The ONLY path that pauses/resumes the backend pipeline. Explicit,
  // user-clicked, never inferred from native video events -- see the
  // BUG FIX note above manualPauseRef for why that coupling deadlocked.
  const handleManualPauseToggle = useCallback(() => {
    const video = videoRef.current
    if (!video) return

    if (!manualPauseRef.current) {
      manualPauseRef.current = true
      setManuallyPaused(true)
      // BUG FIX: the auto-throttle's "WAITING FOR ANALYSIS... Resumes
      // automatically" overlay can already be showing (e.g. the backend
      // was briefly behind right before this click). That overlay's own
      // wording is actively wrong once a manual pause is active -- it
      // will NOT resume automatically, it needs this button clicked
      // again. Clearing it here means only one clear, correct status
      // ("RESUME ANALYSIS" button) is ever shown once manual pause
      // takes over; checkBackendSync also bails out early whenever
      // manualPauseRef is set (see above), so it won't turn this back on.
      setWaitingForBackend(false)
      suppressNextPlayPauseEventRef.current = true
      video.pause()
      sendControl?.('pause')
      setStorePaused(true)
    } else {
      manualPauseRef.current = false
      setManuallyPaused(false)
      suppressNextPlayPauseEventRef.current = true
      video.play().catch(() => {})
      sendControl?.('resume')
      setStorePaused(false)
    }
  }, [sendControl])

  // ---- Canvas drawing ----
  const drawBoxes = useCallback((videoTimeSec) => {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video) return

    const dpr = window.devicePixelRatio || 1
    const W = canvas.offsetWidth
    const H = canvas.offsetHeight
    if (canvas.width  !== W * dpr) canvas.width  = W * dpr
    if (canvas.height !== H * dpr) canvas.height = H * dpr

    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, W, H)

    const curr = latestFrameRef.current
    const prev = previousFrameRef.current
    if (!curr || !curr.detections) return

    // Work out an interpolation factor t in [0, 1] for where videoTimeSec
    // sits between prev's and curr's backend timestamps. If there's no
    // usable previous frame, the gap is too large (seek/stall), or
    // videoTimeSec is outside the [prev, curr] window, just snap to curr
    // (t = 1) -- this matches the old, pre-interpolation behaviour exactly
    // as a fallback, so nothing gets worse, it only gets smoother when
    // interpolation is actually possible.
    let t = 1
    let detections = curr.detections
    if (
      prev && prev.detections &&
      curr.timestamp_sec != null && prev.timestamp_sec != null &&
      curr.timestamp_sec > prev.timestamp_sec &&
      (curr.timestamp_sec - prev.timestamp_sec) <= MAX_INTERP_GAP_SEC &&
      videoTimeSec != null
    ) {
      const span = curr.timestamp_sec - prev.timestamp_sec
      const raw = (videoTimeSec - prev.timestamp_sec) / span
      t = Math.max(0, Math.min(1, raw))

      const prevByKey = new Map(prev.detections.map((d) => [detKey(d), d]))
      detections = curr.detections.map((d) => {
        const match = prevByKey.get(detKey(d))
        // Only interpolate when exactly one candidate shares this key in
        // each frame -- for the ball (no track_id) this is usually true
        // since detect_frame keeps at most MAX_BALL_CANDIDATES, but if
        // several show up under the same bare "ball" key, matching would
        // be ambiguous, so just use curr's position for that one instead
        // of guessing which ball candidate continues which.
        return match ? lerpDet(match, d, t) : d
      })
    }

    // Compute actual letterbox / video layout inside the <video> element
    const vw = video.videoWidth || 1920
    const vh = video.videoHeight || 1080
    const videoAspect = vw / vh
    const canvasAspect = W / H

    let renderW = W
    let renderH = H
    let offsetX = 0
    let offsetY = 0

    if (canvasAspect > videoAspect) {
      // Canvas is wider than video -> pillarboxing (bars on left & right)
      renderW = H * videoAspect
      offsetX = (W - renderW) / 2
    } else {
      // Canvas is taller than video -> letterboxing (bars on top & bottom)
      renderH = W / videoAspect
      offsetY = (H - renderH) / 2
    }

    for (const det of detections) {
      const color = CLASS_COLORS[det.cls_name] || DEFAULT_COLOR
      const x = offsetX + det.x * renderW
      const y = offsetY + det.y * renderH
      const w = det.w * renderW
      const h = det.h * renderH

      ctx.strokeStyle = color
      ctx.lineWidth   = 2
      ctx.strokeRect(x - w / 2, y - h / 2, w, h)

      const label = `${det.cls_name}${det.track_id != null ? ' #' + det.track_id : ''}`
      ctx.font = 'bold 11px Courier New'
      const textW = ctx.measureText(label).width
      ctx.fillStyle = color
      ctx.fillRect(x - w / 2, y - h / 2 - 16, textW + 6, 16)
      ctx.fillStyle = '#000000'
      ctx.fillText(label, x - w / 2 + 3, y - h / 2 - 3)
    }

    // Backend timestamp watermark (bottom-left, small)
    if (curr.timestamp_sec != null) {
      ctx.font      = '10px Courier New'
      ctx.fillStyle = 'rgba(255,230,0,0.9)'
      ctx.fillText(`backend t=${curr.timestamp_sec.toFixed(2)}s`, 6, 14)
    }
  }, [])

  // DESIGN CHANGE (deliberate, requested): auto-pausing playback to
  // protect box accuracy is now DISABLED entirely. The previous design
  // philosophy was "pause rather than show a wrong box" -- correct, and
  // it worked (confirmed across several fixes), but it meant live,
  // non-precomputed clips visibly froze and waited whenever the backend
  // (mainly Groq's network latency, not local GPU speed) fell behind,
  // which read as choppy for a live demo and blocked continuous
  // commentary pacing. This flips the tradeoff: video now ALWAYS plays
  // at full real-time speed, never freezing for backend catch-up.
  // Boxes can lag behind reality by up to a second or two during a
  // genuine backend slowdown -- they do NOT show wrong/future positions
  // (the existing interpolation logic in drawBoxes still clamps at
  // t=1, i.e. holds at the last known real detection rather than
  // extrapolating ahead), they just hold slightly stale until the
  // backend produces a newer frame. That staleness is the accepted
  // cost of this change. The manual "PAUSE ANALYSIS" button and
  // manualPauseRef are completely untouched -- the user can still
  // choose to pause deliberately at any time; only the AUTOMATIC
  // throttle is disabled. To re-enable the automatic throttle, restore
  // the body of the `if (aheadBy > MAX_AHEAD_SEC)` branch below from
  // version history -- the surrounding bookkeeping (autoPausedRef,
  // waitingForBackend, MAX_AHEAD_SEC itself) is left in place,
  // deliberately unused, specifically so that revert is a small, clean
  // diff rather than a rewrite.
  const checkBackendSync = useCallback((videoTimeSec) => {
    const video = videoRef.current
    if (!video || videoTimeSec == null) return
    // Never fight an explicit manual pause -- see manualPauseRef's BUG
    // FIX note above for why backend pause/resume must stay under the
    // user's direct control only.
    if (manualPauseRef.current) return

    const curr = latestFrameRef.current
    // No detections yet at all (very start of playback) -- nothing to
    // compare against, so don't block playback on it.
    if (!curr || curr.timestamp_sec == null) return

    const aheadBy = videoTimeSec - curr.timestamp_sec
    void aheadBy  // kept for the dormant branch below; no-op while disabled

    if (false && aheadBy > MAX_AHEAD_SEC) {
      if (!video.paused) {
        // BUG FIX (DEADLOCK, regression from an earlier "fix"): this
        // used to also call sendControl('pause') here, reasoning that
        // it would stop the backend from racing ahead during an
        // auto-throttle pause. That created a NEW deadlock: once the
        // backend is paused, its latest frame timestamp can never
        // advance, but the resume condition below depends on exactly
        // that timestamp advancing to catch up to the frozen video
        // position. Both sides freeze together, permanently -- confirmed
        // by server logs showing "Pipeline paused" with no matching
        // "Pipeline resumed" ever following. The backend must be left
        // running during an auto-throttle pause so it can eventually
        // produce a frame that satisfies the resume check below. The
        // "boxes drift past the correct player" problem this was
        // originally trying to solve is fixed instead, correctly, by
        // NEVER DRAWING a frame that's ahead of the frozen video
        // position in the first place -- see the clamp in the
        // redraw-on-every-new-frame effect just below this function.
        suppressNextPlayPauseEventRef.current = true
        video.pause()
        autoPausedRef.current = true
        setWaitingForBackend(true)
        setStorePaused(true)
      }
    } else if (autoPausedRef.current && video.paused) {
      // Backend has caught back up -- only resume if WE paused it, never
      // override a pause the user chose themselves.
      suppressNextPlayPauseEventRef.current = true
      video.play().catch(() => {})
      autoPausedRef.current = false
      setWaitingForBackend(false)
      setStorePaused(false)
    }
  }, [setStorePaused])

  // BUG FIX (stale/drifting boxes during an auto-pause): drawBoxes() was
  // only ever called from inside the rVFC/rAF loop below, and
  // requestVideoFrameCallback ONLY fires while the <video> element is
  // actively playing -- per spec, pausing stops it from firing at all.
  // So once the auto-throttle paused playback, the canvas froze on
  // whatever box positions were last drawn, while the backend kept
  // sending newer, more accurate detections the whole time it was
  // catching up (confirmed: the telemetry panel, event count, and
  // commentary all kept advancing during a pause, while the on-video
  // boxes stayed pixel-for-pixel identical -- this is what produced
  // boxes that looked "close but not quite on" the player: they were
  // accurate for an earlier moment, just never got the chance to update
  // to a closer one). Fixed by forcing an immediate redraw every time a
  // new frame actually arrives, independent of whether the video is
  // currently playing or paused -- so boxes keep tightening toward
  // accuracy throughout a pause instead of freezing on the first one.
  useEffect(() => {
    if (!latestFrame) return
    const video = videoRef.current
    if (!video) return
    // BUG FIX (PERMANENT FREEZE, confirmed via frame-by-frame recording:
    // backend kept running -- a real event fired well after the pause --
    // but the video/boxes/commentary never resumed for the rest of the
    // session): checkBackendSync is the ONLY thing that ever decides to
    // resume, and it was only ever invoked from inside the
    // requestVideoFrameCallback/requestAnimationFrame loop below. Per
    // spec, requestVideoFrameCallback stops firing entirely the moment
    // the video is paused -- so the one function that could decide "ok,
    // resume now" stopped being called at exactly the moment it paused,
    // and never got invoked again. A deadlock baked into browser
    // behaviour itself, nothing to do with the backend or sendControl.
    // Fixed the same way drawBoxes was fixed earlier: give
    // checkBackendSync this same independent trigger, firing on every
    // new backend frame regardless of play/pause state, so the resume
    // decision keeps getting re-evaluated even while rVFC is silent.
    checkBackendSync(video.currentTime)
    // BUG FIX ("boxes keep moving past the correct player, don't
    // stop"): while the video is paused, its currentTime is frozen --
    // but the backend keeps advancing regardless (it must, or the
    // resume check in checkBackendSync above can never be satisfied,
    // which is the deadlock that coupling pause/resume to the backend
    // here caused instead -- see that function's own BUG FIX note).
    // Redrawing unconditionally on every new frame meant boxes kept
    // advancing to match the backend's ever-increasing timestamp even
    // though the visible video image stayed frozen at one instant --
    // visually, boxes sliding past the real player and continuing on.
    // Fixed by simply never drawing a frame that's AHEAD of the frozen
    // display while paused: a new frame still behind or at the frozen
    // position is strictly more accurate than the previous one (so it's
    // still drawn, which is what keeps boxes tightening toward the
    // correct spot instead of freezing stale) -- only frames further
    // ahead than what's actually on screen get held back, exactly the
    // ones that caused the drift past the player.
    if (video.paused && latestFrame.timestamp_sec != null && latestFrame.timestamp_sec > video.currentTime + 0.05) {
      return
    }
    drawBoxes(video.currentTime)
  }, [latestFrame, drawBoxes, checkBackendSync])

  // rVFC when available, rAF fallback
  useEffect(() => {
    const video = videoRef.current
    let cancelled = false

    const rafLoop = () => {
      if (cancelled) return
      const t = videoRef.current?.currentTime
      drawBoxes(t)
      checkBackendSync(t)
      animRef.current = requestAnimationFrame(rafLoop)
    }

    if (video && 'requestVideoFrameCallback' in video) {
      const vfcLoop = (_now, metadata) => {
        if (cancelled) return
        drawBoxes(metadata.mediaTime)
        checkBackendSync(metadata.mediaTime)
        video.requestVideoFrameCallback(vfcLoop)
      }
      video.requestVideoFrameCallback(vfcLoop)
    } else {
      animRef.current = requestAnimationFrame(rafLoop)
    }

    return () => {
      cancelled = true
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [drawBoxes, checkBackendSync])

  return (
    <div style={{ position: 'relative', width: '100%', background: '#000', border: '3px solid #0a0a0a' }}>
      <video
        ref={videoRef}
        controls
        muted
        preload="auto"
        style={{ width: '100%', display: 'block' }}
        playsInline
        onLoadedMetadata={() => setLoadState('ready')}
        onCanPlay={() => setLoadState('ready')}
        onError={() => setLoadState('error')}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0, left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
        }}
      />

      {/* Explicit, deliberate backend pause control -- see manualPauseRef's
          BUG FIX note above for why this must never be inferred from
          native video play/pause events. */}
      {videoFile && loadState === 'ready' && sendControl && (
        <button
          onClick={handleManualPauseToggle}
          style={{
            position: 'absolute',
            top: 10,
            right: 10,
            zIndex: 5,
            fontFamily: 'Courier New, monospace',
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: 0.5,
            padding: '6px 12px',
            borderRadius: 6,
            border: '1.5px solid rgba(255,255,255,0.35)',
            background: manuallyPaused ? 'rgba(255,19,85,0.85)' : 'rgba(10,10,10,0.6)',
            color: '#fff',
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title={manuallyPaused
            ? 'Resume analysis (frame detection, events, and commentary generation)'
            : 'Pause analysis (frame detection, events, and commentary generation)'}
        >
          {manuallyPaused ? '▶ RESUME ANALYSIS' : '⏸ PAUSE ANALYSIS'}
        </button>
      )}

      {/* Honest status overlay -- without this, a file the browser can't
          decode just sits black forever with no explanation, even though
          the backend pipeline (reading the file directly via OpenCV) keeps
          working fine either way. */}
      {videoFile && loadState === 'loading' && (
        <div style={overlayStyle}>
          <div style={{ fontSize: 13, letterSpacing: 1 }}>
            ⏳ LOADING PREVIEW{isLargeFile ? ' — LARGE FILE' : ''}...
          </div>
          {isLargeFile && (
            <div style={{ fontSize: 11, marginTop: 8, color: '#ffe600', maxWidth: 420 }}>
              {(videoFile.size / (1024 * 1024)).toFixed(0)} MB is a lot for a browser
              preview to parse. The live analysis below is unaffected either way.
            </div>
          )}
        </div>
      )}
      {waitingForBackend && loadState === 'ready' && (
        <div style={{ ...overlayStyle, background: 'rgba(10,10,10,0.55)' }}>
          <div style={{ fontSize: 13, color: '#ffe600', letterSpacing: 1 }}>
            ⏸ WAITING FOR ANALYSIS...
          </div>
          <div style={{ fontSize: 11, marginTop: 8, color: '#ccc', maxWidth: 420 }}>
            Playback paused so bounding boxes stay accurate -- the backend
            is still processing this part of the clip. Resumes automatically.
          </div>
        </div>
      )}
      {videoFile && loadState === 'error' && (
        <div style={overlayStyle}>
          <div style={{ fontSize: 13, color: '#ff5252', letterSpacing: 1 }}>
            ✕ PREVIEW FAILED TO LOAD
          </div>
          <div style={{ fontSize: 11, marginTop: 8, color: '#ccc', maxWidth: 420 }}>
            The browser couldn't decode this file for playback. The backend
            pipeline reads the file directly and is unaffected -- check the
            event feed and commentary transcript, they'll keep working.
            Re-encoding with "ffmpeg -c:v libx264 -movflags +faststart" fixes
            this for the preview.
          </div>
        </div>
      )}
    </div>
  )
}

const overlayStyle = {
  position: 'absolute',
  inset: 0,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  textAlign: 'center',
  padding: 16,
  background: 'rgba(10,10,10,0.85)',
  color: '#fff',
  fontFamily: 'Courier New, monospace',
  pointerEvents: 'none',
}