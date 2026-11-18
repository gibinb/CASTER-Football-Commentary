/**
 * CASTER global state store (Zustand).
 *
 * One store, one truth. Every component reads from here; only the
 * useWebSocket hook writes to it via the actions below.
 *
 * Slice layout:
 *   connection  -- WebSocket lifecycle
 *   pipeline    -- current clip metadata
 *   detections  -- latest frame bounding boxes (for canvas overlay)
 *   events      -- list of all confirmed events this session
 *   caption     -- live streaming caption state
 *   transcript  -- full list of completed commentary sentences
 *   score       -- running score/time fields
 *   style       -- active commentary style
 *   tts         -- mute toggle
 *   playback    -- whether the video is currently paused (manual or
 *                  auto-throttle) and the held-commentary queue this
 *                  drives -- see onToken/onSentenceEnd/onSpeak below
 */
import { create } from 'zustand'

const useCasterStore = create((set, get) => ({
  // ---- Connection ----
  wsStatus: 'idle',       // 'idle' | 'connecting' | 'connected' | 'done' | 'error'
  wsError: null,

  // ---- Pipeline metadata ----
  clipName: null,
  sport: null,            // 'cricket' | 'football' | null
  totalEvents: 0,

  // ---- Latest frame detections (for canvas overlay) ----
  // Each detection: { track_id, cls_name, confidence, x, y, w, h }
  // All coords are normalised 0-1 -- multiply by canvas size before drawing
  latestFrame: null,      // { frame_id, timestamp_sec, detections, feature_vector }
  // BUG FIX: previousFrame was never actually being set here -- onFrame
  // only ever wrote latestFrame, so VideoPlayer.jsx's interpolation logic
  // (which reads BOTH latestFrame and previousFrame to smooth box motion
  // between detection frames) always had previousFrame === undefined and
  // silently fell back to its no-interpolation snap behaviour the whole
  // time, even though the interpolation code itself was fine. Fixed by
  // shifting the old latestFrame into previousFrame on every new frame.
  previousFrame: null,

  // ---- Confirmed events this session ----
  events: [],             // array of { event, confidence, sport, score, time, current_style }

  // ---- Live caption (streaming word-by-word & last spoken line) ----
  captionWords: [],       // words accumulated for the current sentence
  captionNewWord: null,   // the most recently added word (highlighted in red briefly)
  captionFlash: false,    // true briefly after sentence_end (triggers CSS flash)
  lastCommentaryText: '', // keeps the completed commentary text visible until the next one arrives

  // ---- Full transcript ----
  transcript: [],         // array of { event, sport, score, time, style, text, backend_used }

  // ---- Live score / time ----
  score: '',
  matchTime: '',

  // ---- Commentary style (user can switch mid-session) ----
  style: 'dramatic',

  // ---- TTS ----
  muted: false,
  // Set once by useWebSocket.js on connect so held/queued 'speak'
  // messages can still reach the TTS callback once released -- see
  // setPaused()'s flush logic below.
  speakCallback: null,

  // ---- Playback hold state ----
  // BUG FIX ("commentary keeps appearing/speaking while the video is
  // visibly frozen"): pausing the BACKEND itself whenever the video
  // pauses was tried and reverted -- it deadlocks the auto-throttle
  // (see VideoPlayer.jsx's checkBackendSync comments for why: the
  // auto-resume condition depends on the backend continuing to
  // advance, so stopping the backend removes the only thing that could
  // ever satisfy that condition again). The backend must keep running.
  // Instead, this holds back NEW commentary (token/sentence_end/speak)
  // on the FRONTEND the moment playback pauses for any reason (manual
  // or auto-throttle), and releases everything that arrived meanwhile,
  // in order, the moment playback resumes -- so what the user sees and
  // hears always matches what's visibly on screen, without ever
  // touching the backend's own pacing.
  paused: false,
  pendingCommentary: [],  // queued {type, ...} messages while paused

  // =================================================================
  // Actions called by useWebSocket.js
  // =================================================================

  setWsStatus: (status) => set({ wsStatus: status }),
  setWsError: (err) => set({ wsError: err }),
  setSpeakCallback: (fn) => set({ speakCallback: fn }),

  onSportDetected: (sport) => set({ sport }),

  onFrame: (frameMsg) => set((state) => ({
    previousFrame: state.latestFrame,
    latestFrame: frameMsg,
  })),

  onEvent: (eventMsg) => set((state) => ({
    events: [...state.events.slice(-49), eventMsg],   // keep last 50
    score: eventMsg.score || state.score,
    matchTime: eventMsg.time || state.matchTime,
  })),

  // These three used to apply unconditionally. Now each checks
  // `paused` first: if held, the raw message is queued instead of
  // applied, and setPaused(false) below replays the queue through
  // these exact same functions (so the caption still builds up
  // word-by-word, sentence_end still fires once per sentence, etc.)
  // once playback actually resumes.
  onToken: (token) => {
    if (get().paused) {
      set((state) => ({ pendingCommentary: [...state.pendingCommentary, { type: 'token', token }] }))
      return
    }
    set((state) => ({
      captionWords: [...state.captionWords, token.trim()],
      captionNewWord: token.trim(),
    }))
  },

  onSentenceEnd: (msg) => {
    if (get().paused) {
      set((state) => ({ pendingCommentary: [...state.pendingCommentary, { type: 'sentence_end', msg }] }))
      return
    }
    set((state) => {
      const fullText = msg.full_text || state.captionWords.join(' ')
      const entry = {
        event: msg.event,
        sport: msg.sport,
        confidence: msg.confidence,
        time: state.matchTime,
        score: state.score,
        style: msg.style,
        text: fullText,
        backend_used: msg.backend_used,
      }
      return {
        transcript: [...state.transcript, entry],
        captionWords: [],
        captionNewWord: null,
        captionFlash: true,
        lastCommentaryText: fullText,
      }
    })
  },

  // New: 'speak' (TTS) messages used to go straight from
  // useWebSocket.js to the onSpeak callback, bypassing the store
  // entirely -- which meant they couldn't be held during a pause like
  // token/sentence_end. Routed through the store now so all three stay
  // in sync.
  onSpeak: (payload) => {
    if (get().paused) {
      set((state) => ({ pendingCommentary: [...state.pendingCommentary, { type: 'speak', payload }] }))
      return
    }
    get().speakCallback?.(payload.text, payload.lang)
  },

  // Called by VideoPlayer.jsx whenever playback pauses/resumes, for
  // BOTH the manual "PAUSE ANALYSIS" button and the automatic
  // backend-sync throttle -- see that file's handleManualPauseToggle
  // and checkBackendSync.
  setPaused: (paused) => {
    if (!paused && get().paused) {
      // Releasing a hold: replay everything queued, in order, through
      // the exact same logic as if it had arrived live just now.
      const queue = get().pendingCommentary
      set({ paused: false, pendingCommentary: [] })
      for (const item of queue) {
        if (item.type === 'token') get().onToken(item.token)
        else if (item.type === 'sentence_end') get().onSentenceEnd(item.msg)
        else if (item.type === 'speak') get().onSpeak(item.payload)
      }
      return
    }
    set({ paused })
  },

  clearCaptionFlash: () => set({ captionFlash: false }),

  onDone: (msg) => set({ wsStatus: 'done', totalEvents: msg.total_events }),

  onError: (msg) => set({ wsError: msg.message }),
  onWarning: (msg) => console.warn('[CASTER]', msg.message),

  setStyle: (style) => set({ style }),
  setClipName: (name) => set({ clipName: name }),
  toggleMute: () => set((state) => ({ muted: !state.muted })),

  reset: () => set({
    wsStatus: 'idle', wsError: null, clipName: null, sport: null,
    totalEvents: 0, latestFrame: null, previousFrame: null, events: [], captionWords: [],
    captionNewWord: null, captionFlash: false, lastCommentaryText: '', transcript: [],
    score: '', matchTime: '', paused: false, pendingCommentary: [],
  }),
}))

export default useCasterStore
