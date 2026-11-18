/**
 * CASTER Phase 4 -- useTTS hook.
 *
 * Web Speech API wrapper with:
 *   - Speech queue (so rapid events don't overlap)
 *   - Browser user-gesture warm-up (Chrome blocks speech until
 *     a user click has occurred -- we call this in the upload handler)
 *   - Mute toggle (cancels queue and blocks further utterances)
 *   - Rate / pitch tuned for sports broadcast energy
 *
 * The `warmedUp` flag is the fix for the most common Phase 4 bug:
 * "TTS doesn't work at all." Chrome requires speechSynthesis.speak()
 * to be called inside a user gesture handler at least once. We call
 * it silently (empty utterance) when the user clicks "Start" or
 * the upload button -- that single call unlocks TTS for the whole
 * session. See CASTER_Phase4_DeepDive.docx Section 4.2.
 */
import { useRef, useCallback } from 'react'
import useCasterStore from '../store/useCasterStore.js'

export function useTTS() {
  const queue = useRef([])
  const isSpeaking = useRef(false)
  const warmedUp = useRef(false)
  const { muted } = useCasterStore()

  // Call this inside ANY click/change handler before the first speak().
  // A silent empty utterance is enough to unlock TTS in Chrome.
  const warmUp = useCallback(() => {
    if (warmedUp.current) return
    if (!window.speechSynthesis) return
    const silent = new SpeechSynthesisUtterance('')
    silent.volume = 0
    window.speechSynthesis.speak(silent)
    warmedUp.current = true
  }, [])

  const speakNext = useCallback(() => {
    if (isSpeaking.current) return
    if (queue.current.length === 0) return

    const { text, lang } = queue.current.shift()

    if (!window.speechSynthesis || muted) {
      // Drain queue silently when muted
      speakNext()
      return
    }

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 1.15      // slightly faster = more broadcast energy
    utterance.pitch = 1.05
    utterance.lang = lang || 'en-GB'

    utterance.onstart = () => { isSpeaking.current = true }
    utterance.onend   = () => {
      isSpeaking.current = false
      speakNext()              // chain to next item in queue
    }
    utterance.onerror = () => {
      isSpeaking.current = false
      speakNext()
    }

    window.speechSynthesis.speak(utterance)
  }, [muted])

  const speak = useCallback((text, lang = 'en-GB') => {
    if (!text) return
    queue.current.push({ text, lang })
    speakNext()
  }, [speakNext])

  const mute = useCallback(() => {
    queue.current = []
    isSpeaking.current = false
    window.speechSynthesis?.cancel()
  }, [])

  return { speak, mute, warmUp, warmedUp }
}
