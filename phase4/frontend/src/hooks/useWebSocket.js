/**
 * CASTER Phase 4 -- useWebSocket hook.
 *
 * Opens the WebSocket to /api/ws/stream?clip=<name>&style=<style>
 * and routes every incoming message type to the appropriate Zustand
 * store action. Also exposes a `sendStyle(style)` function so
 * StyleSwitcher can request mid-session style changes.
 *
 * Message routing:
 *   sport_detected  -> store.onSportDetected
 *   frame           -> store.onFrame
 *   event           -> store.onEvent
 *   token           -> store.onToken
 *   sentence_end    -> store.onSentenceEnd
 *   speak           -> useTTS (via callback)
 *   done            -> store.onDone
 *   error           -> store.onError
 *   warning         -> store.onWarning
 *   style_changed   -> store.setStyle
 *
 * BUG FIX: the backend sends token messages shaped as
 * { "type": "token", "token": "FOUR " } (see ws_handler.py). onToken()
 * in useCasterStore.js expects just the token STRING (it calls
 * token.trim() directly) -- passing it the whole `msg` object instead
 * of `msg.token` threw "token.trim is not a function" on every single
 * streamed word. Fixed below by passing msg.token, not msg.
 */
import { useEffect, useRef, useCallback } from 'react'
import useCasterStore from '../store/useCasterStore.js'

const WS_BASE = 'ws://localhost:8000'

export function useWebSocket(clipName, style, onSpeak) {
  const wsRef = useRef(null)
  const store = useCasterStore()

  // BUG FIX: destructuring the store's onSpeak action as `onSpeak`
  // collided with this hook's own `onSpeak` PARAMETER (the TTS
  // callback passed in from App.jsx) -- esbuild/Vite correctly refused
  // to build with two bindings of the same name in the same scope
  // ("The symbol onSpeak has already been declared"). Renamed the
  // store's version to storeOnSpeak; the parameter is untouched.
  const {
    setWsStatus, setWsError, onSportDetected, onFrame,
    onEvent, onToken, onSentenceEnd, onSpeak: storeOnSpeak, setSpeakCallback, onDone, onError, onWarning,
    setStyle,
  } = store

  const connect = useCallback(() => {
    if (!clipName) return
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return

    const url = `${WS_BASE}/api/ws/stream?clip=${encodeURIComponent(clipName)}&style=${style}`
    setWsStatus('connecting')

    // BUG FIX ("commentary keeps playing/speaking while video is
    // frozen"): 'speak' messages used to call `onSpeak` directly here,
    // bypassing the store entirely -- so there was no way to hold them
    // back during a pause the way token/sentence_end now can be (see
    // useCasterStore.js's setPaused). Registering the real onSpeak
    // callback with the store once, here, lets store.onSpeak() either
    // call it immediately or queue the message for later, exactly like
    // the other two message types.
    setSpeakCallback(onSpeak)

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setWsStatus('connected')
    }

    ws.onmessage = (ev) => {
      let msg
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }

      switch (msg.type) {
        case 'sport_detected': onSportDetected(msg.sport); break
        case 'frame':          onFrame(msg); break
        case 'event':          onEvent(msg); break
        case 'token':          onToken(msg.token); break
        case 'sentence_end':   onSentenceEnd(msg); break
        case 'speak':           storeOnSpeak({ text: msg.text, lang: msg.lang || 'en-GB' }); break
        case 'done':           onDone(msg); break
        case 'error':          onError(msg); break
        case 'warning':        onWarning(msg); break
        case 'style_changed':  setStyle(msg.style); break
        default: break
      }
    }

    ws.onerror = () => {
      setWsError('WebSocket connection error. Is the backend running on port 8000?')
      setWsStatus('error')
    }

    ws.onclose = () => {
      if (store.wsStatus !== 'done') {
        setWsStatus('idle')
      }
    }
  }, [clipName, style])   // eslint-disable-line react-hooks/exhaustive-deps

  // Send a style-change message to the backend mid-session
  const sendStyle = useCallback((newStyle) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'set_style', style: newStyle }))
    }
  }, [])

  // Generic control channel -- currently used for {"type":"pause"} /
  // {"type":"resume"} so pausing the <video> element also pauses the
  // backend's frame/event/commentary generation (see ws_handler.py and
  // pipeline_runner.py's play_event). Without this, pausing the video
  // was purely cosmetic in the browser.
  const sendControl = useCallback((type) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type }))
    }
  }, [])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  // Auto-connect when clipName changes; auto-disconnect on unmount
  useEffect(() => {
    if (clipName) connect()
    return () => disconnect()
  }, [clipName])   // eslint-disable-line react-hooks/exhaustive-deps

  return { connect, disconnect, sendStyle, sendControl }
}