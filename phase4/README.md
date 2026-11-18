# CASTER — Phase 4: Integration and Polish

**Status: backend confirmed live against real footage; frontend actively iterating.**

Turns the working Phase 1–3 pipeline into a website: upload a clip, watch
bounding boxes drawn on it, see live word-by-word captions, hear spoken
commentary, switch style mid-session.

## WebSocket message shapes (from backend)

```json
{ "type": "sport_detected", "sport": "football" }
{ "type": "frame",          "frame_id": 0, "timestamp_sec": 0.033,
                             "detections": [...], "feature_vector": [...] }
{ "type": "event",          "event": "GOAL", "confidence": 0.91,
                             "score": "1-0", "time": "67" }
{ "type": "token",          "token": "GOAL " }
{ "type": "sentence_end",   "event": "GOAL", "full_text": "...",
                             "style": "dramatic", "backend_used": "groq" }
{ "type": "speak",          "text": "...", "lang": "en-GB" }
{ "type": "done",           "clip_name": "myclip", "total_events": 7 }
```
The React frontend never needs Phase 1–3 internals — it only consumes
these shapes. No `sport` field is sent to select — football is the only
sport; the field exists for forward-compat/display (sport badge), not
branching logic.

## Quick start

```bash
# Backend — from project root (where config.py lives)
pip install "uvicorn[standard]" fastapi httpx   # [standard] is required — plain uvicorn lacks WebSocket support
python phase4/run_phase4.py --check
python phase4/run_phase4.py
# http://localhost:8000, docs at /docs
# auto-detects port-in-use and switches to the next free port

# Frontend
cd phase4/frontend
npm install
npm run dev
# http://localhost:5173
```

## Coordinate scaling — the single most common bug here

Phase 1 stores detections in **normalised 0–1 coordinates**. The frontend
must multiply back up by the canvas's **rendered pixel size**
(`canvas.offsetWidth`/`offsetHeight`), never the source video's native
resolution. Getting this backwards makes every box visually track roughly
but land in the wrong place — easy to miss because boxes still *move*
with the action, just not precisely on it.

A related real bug (found+fixed): the precomputed-replay fast path once
hardcoded `fw=1920.0/fh=1080.0` when normalising boxes instead of reading
the real source video's dimensions via `cv2.VideoCapture` — any clip not
exactly 1920×1080 silently misplaced every box. Fixed by reading real
dimensions once at session start, with a 1920×1080 fallback + warning only
if the file is unreadable.

## TTS warm-up

Chrome refuses to start speech synthesis before a user gesture. `useTTS.js`
calls `speechSynthesis.speak(new SpeechSynthesisUtterance(''))` silently
inside the upload click handler the first time it runs — without this, TTS
silently does nothing with no error.

## Speech queue

Rapid events (goal immediately followed by another event) would overlap
without serialisation. `useTTS.js` keeps a `queue` ref and chains
`speakNext()` inside each utterance's `onend`, so sentences play
back-to-back with no overlap.

## Video preview — StrictMode blob-URL bug (fixed)

`VideoPlayer.jsx`'s blob-URL effect originally revoked the object URL in
its own cleanup function. Under React 18 `StrictMode`, effects mount →
cleanup → mount again in dev — so the blob URL got revoked almost
immediately after creation, before the `<video>` element finished loading
(looked like a codec/HEVC problem; it wasn't). Fixed: blob URL is now
tracked in a ref, only revoked when a genuinely new file arrives or on true
unmount.

## Standalone desktop players (debug tools, not part of the product)

`desktop_app.py` (Tkinter+PIL+OpenCV+pyttsx3) and `play_desktop.py`
(customtkinter, wall-clock-synced replay) bypass the FastAPI/React stack
entirely for local bounding-box/tracking debugging. `gen_tv_commentary.py`
and `apply_tv_lines.py` post-process commentary with a "broadcast style"
LLM rewrite or a canned phrase bank, respectively — **neither is the
trained Phase 3 pipeline output**; if used in a demo recording, disclose
as a scripted polish pass, not live model output.

## Exit gate

1. Upload a football clip → sport badge reads **FOOTBALL DETECTED**
2. Bounding boxes track ball/players as it plays
3. Live captions stream word-by-word, commentary is heard aloud
4. Switching style mid-playback changes tone on the next event
5. Transcript log shows the full session, copyable

## Running tests

```bash
python -m pytest phase4/tests/test_phase4_backend.py -v
```
Covers all API endpoints, offline replay path, and the full Phase 4→3→2→1
import chain — without needing a running server, YOLO weights, or a
trained classifier.