# CASTER — Phase 4: Integration and Polish

**Phase 4 turns the working pipeline into a website.** Football only — the
cricket/dual-sport architecture described in earlier drafts of this README
was fully removed, not deprecated, across Phases 1–4.

- **Video upload** via a brutalist/maximalist drag-and-drop UI
- **Bounding box overlay** drawn on a canvas, synced to playback time via
  `requestVideoFrameCallback` (with a `requestAnimationFrame` fallback)
- **Live streaming captions** — words appear one by one as the LLM generates them
- **Web Speech API TTS** with a speech queue so events never overlap
- **Style switcher** (Dramatic / Casual / Statistical), changeable mid-session
- **Sport badge** — always reads "FOOTBALL DETECTED" (there is no sport
  auto-detection logic anywhere in the project anymore; the badge is kept
  because the frontend/WS contract still documents a `sport_detected`
  message, not because detection actually happens)
- **Two execution paths**, chosen automatically per clip (see below)
- **Two standalone desktop apps** that bypass the web stack entirely (see
  "Desktop apps" below)

---

## What the frontend receives from the backend (WebSocket contract)

```json
{ "type": "sport_detected", "sport": "football" }
{ "type": "frame",          "frame_id": 0, "timestamp_sec": 0.033,
                             "detections": [...], "feature_vector": [...],
                             "ball_detected": true, "ball_track_id": 3,
                             "ball_missing_frames": 0 }
{ "type": "event",          "event": "PASS", "confidence": 0.73,
                             "sport": "football", "score": "0 - 0",
                             "time": "0", "gate_reason": "precomputed_event" }
{ "type": "token",          "token": "What " }
{ "type": "sentence_end",   "event": "PASS", "sport": "football",
                             "confidence": 0.73, "style": "dramatic",
                             "backend_used": "groq", "full_text": "..." }
{ "type": "speak",          "text": "...", "lang": "en-GB" }
{ "type": "warning",        "message": "..." }
{ "type": "error",          "message": "..." }
{ "type": "done",           "clip_name": "myclip", "total_events": 7 }
```

`detections[].x/y/w/h` are always normalised 0–1, divided by the **real
source-video dimensions** (see "Fixed" history below — this was not always
true). The frontend never needs to know about Phase 1/2/3 internals; it only
ever consumes these JSON shapes.

---

## Quick start

### Backend

```bash
# From the project root (where config.py lives)
pip install fastapi "uvicorn[standard]" python-multipart groq xgboost ultralytics opencv-python-headless

# Check all imports resolve correctly
python phase4/run_phase4.py --check

# Start the server
python phase4/run_phase4.py
# Backend runs at http://localhost:8000
# API docs at  http://localhost:8000/docs
```

> **`uvicorn[standard]`, not plain `uvicorn`.** Plain `uvicorn` has no
> WebSocket library bundled — `/api/ws/stream` will 404 with "No supported
> WebSocket library detected" even though every other endpoint works fine.

`run_phase4.py` also auto-detects if port 8000 is already in use (common on
Windows as error 10048) and switches to the next free port automatically
instead of crashing.

Set your Groq API key for high-quality commentary (free tier, generous daily
request limit):
```bash
# Linux/macOS
export GROQ_API_KEY=<your_key>
# Windows PowerShell
$env:GROQ_API_KEY = "<your_key>"
```
Without it, commentary falls back to the rule-based template backend — still
grounded, just less fluent. `local_gguf` is a third backend slot that tries
first and gracefully no-ops ("no fine-tuned model yet") until a QLoRA
fine-tune actually exists at `CASTER_GGUF_MODEL_PATH`.

### Frontend

```bash
cd phase4/frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## The two execution paths

`pipeline_runner.py`'s `PipelineRunner.run()` picks one of these per upload,
automatically:

- **`_run_pipeline()`** — the live path. Runs YOLO (`load_detector()`) →
  `iter_tracked_frames()` (ByteTrack) → `FeatureExtractor()` → XGBoost →
  confidence gate → `CommentaryEngine`, frame by frame, from scratch. This
  is what runs for any video that's never been processed before. On
  CPU-only hardware this is slow (YOLOv8m + ByteTrack, no GPU) — the
  adaptive stride logic keeps it roughly real-time-ish at the cost of some
  temporal resolution, and a console line reports the actual processed fps
  so you know what to expect (e.g. *"Running on CPU at ~5.0 processed fps
  (stride=12). Commentary will appear within ~5-10s of each event."*).
- **`_run_precomputed_pipeline()`** — the fast-replay path. If
  `outputs/<clip>_scene_states.json` and `outputs/<clip>_scene_json_events.json`
  already exist for this exact clip filename (i.e. you've run Phase 1/2
  against it before), it replays those in sync with video playback time
  instead of re-running YOLO. This is why **renaming a re-encoded file
  changes which path runs** — `rm1.mp4` has precomputed JSON, `rm1_h264.mp4`
  does not, so the fast path silently falls back to the slow live path for
  the renamed file. Rename back to match the original stem if you want the
  fast path.
- **`_run_offline_phase2_phase3()`** — a third, simpler fallback: replays a
  saved `*_scene_json_events.json` through Phase 3 only (no frame/bounding-box
  messages at all), for when you want to re-run commentary generation
  without touching Phase 1/2 output.

---

## File layout

```
phase4/
  run_phase4.py                 ← CLI entry point; --check, auto port-switch
  desktop_app.py                ← Standalone Tkinter+PIL+OpenCV+pyttsx3 app,
                                   bypasses FastAPI/React entirely
  play_desktop.py                ← Standalone customtkinter app, wall-clock
                                   real-time sync, smoother box interpolation
  gen_tv_commentary.py          ← Optional post-processing: replaces real
                                   commentary with a separate "TV style" Groq
                                   call (see disclosure note below)
  apply_tv_lines.py             ← Optional post-processing: replaces real
                                   commentary with hardcoded canned phrases
                                   cycled by index, no LLM call at all
  backend/
    main.py                     ← FastAPI app: /api/upload, /api/health,
                                   /api/transcript/<clip>
    pipeline_runner.py          ← Phase 1+2+3 wired into one async generator
                                   (PipelineRunner class, 3 execution paths)
    ws_handler.py                ← WebSocket routes: /api/ws/stream, /api/ws/replay
  frontend/
    package.json
    vite.config.js
    index.html
    src/
      main.jsx                  ← React entry point (React.StrictMode is ON
                                   — see the VideoPlayer bug below)
      index.css
      App.jsx                   ← Top-level layout
      store/
        useCasterStore.js       ← Zustand global state
      hooks/
        useWebSocket.js         ← WS lifecycle + message routing to store
        useTTS.js                ← Web Speech API + queue + warm-up fix
      components/
        VideoPlayer/            ← <video> + canvas bounding-box overlay
        LiveCaption/
        StyleSwitcher/
        SportBadge/
        ScoreBoard/
        EventFeed/
        TranscriptLog/
  tests/
    test_phase4_backend.py      ← 8 tests, no server/YOLO weights/trained
                                   model needed
```

> **Disclosure note on `gen_tv_commentary.py` / `apply_tv_lines.py`:** these
> two scripts deliberately produce more polished, broadcast-flavoured output
> than the real grounded pipeline does, by either prompting Groq separately
> with no structural grounding, or by substituting a fixed, human-authored
> phrase bank. If either is used to produce a demo recording, that should be
> disclosed as a scripted polish pass, not presented as live model output —
> it is not what `pipeline_runner.py`'s real path generates.

---

## Fixed (real bugs found and verified against this actual codebase)

### Phase 1/2/3 import mismatch (blocker, fixed)
`pipeline_runner.py` at one point predated the football-only conversion and
imported things that no longer exist (`SPORT_CRICKET` from `config.py`, a
deleted `sport_detector.py`/`sport_detector_clip.py`, a `SportDetector` class
that was never the real API). This crashed the server at startup with
`ImportError: cannot import name 'SPORT_CRICKET'`. Rewritten to match the
real, current Phase 1 API: `load_detector()`, `FeatureExtractor()` (no
arguments), `build_scene_state()` (no `sport` kwarg),
`iter_tracked_frames(detector, source, stride=...)` (a synchronous generator,
driven via `run_in_executor` so it doesn't block the event loop),
`MatchState(sport=, commentary_style=)` + `.football_score_str()` /
`.match_minute` instead of cricket/football ternaries.

### Hardcoded 1920×1080 bounding-box normalisation (recurring, fixed)
`_run_precomputed_pipeline()` normalised every detection box using
hardcoded `fw=1920.0, fh=1080.0` — `scene_states.json` only ever stores raw
*pixel* coordinates, never the source frame dimensions (confirmed:
`schemas.SceneState` has no such field). For any clip that isn't exactly
1920×1080, every box silently landed in the wrong place. Fixed by reading
the real dimensions off the source video via `cv2.VideoCapture`, once, with
a graceful fallback + a `"warning"` message only if the video file is
missing/unreadable. This regressed twice across different upload rounds
(from branches built before the fix existed) — worth checking for if you're
merging in another branch of this file.

### VideoPlayer.jsx: StrictMode blob-URL race (fixed)
A file could play perfectly in VLC, in Edge opened directly via `file://`,
and still show "PREVIEW FAILED TO LOAD" only inside the app. Root cause:
`main.jsx` has `React.StrictMode` enabled, which deliberately runs every
effect mount → cleanup → mount again in development. The effect that
creates the blob URL (`URL.createObjectURL(videoFile)`) had its own
`return () => URL.revokeObjectURL(url)` as cleanup — StrictMode's extra pass
revoked the blob URL almost immediately after creating it, invalidating it
before the `<video>` element finished loading. Fixed: the blob URL is now
tracked in a ref, revoked only when a genuinely new file arrives or on true
component unmount (separate effect, empty dependency array) — immune to
StrictMode's extra invocation.

### App.jsx: forced-stretch blank layout void (fixed)
The two-column grid had `minHeight: calc(100vh - 120px)` with the default
`alignItems: stretch` — this forced both columns to match a full-viewport
row height, and the left column's short content (Style + Feature Vector
panels) then stretched to fill that height with nothing in it, leaving a
large blank box. Fixed by changing the grid to `alignItems: 'start'` so each
column sizes to its own content.

### Known, non-code browser/OS quirks worth recognising on sight
- **`net::ERR_UPLOAD_FILE_CHANGED`** in the browser console — not a code
  bug. Chromium refuses to read a selected file if it detects the file was
  modified on disk between selection and actual read (common right after an
  `ffmpeg` re-encode finishes, or if a preview pane/antivirus still has the
  file open). Fix: close anything else touching the file, wait a moment,
  re-select it fresh in the app rather than reusing an old selection.
- **HEVC/x265 was investigated and ruled out** as the cause of preview
  failures for the two real clips tested in this project
  (`ffprobe`/`ffmpeg -i` confirmed both were plain H.264) — don't re-chase
  this theory without checking the actual codec first.
- **MP4 "moov atom at the end"** (common with CapCut/ByteDance mobile app
  exports — look for `encoder: bytevehwavc` in `ffmpeg -i` output) is a
  real, confirmed cause of browser-only preview failures: desktop players
  (VLC, and this project's own OpenCV-based desktop apps) can seek freely on
  local disk and don't care where the metadata index sits, but a browser
  `<video>` tag streaming from a blob URL needs that index near the front.
  **Fix with `ffmpeg -c copy -movflags +faststart`** (lossless, fast remux)
  — do **not** attempt to fix this with a naive Python script that just
  moves the `moov` bytes without also rewriting the `stco`/`co64` chunk
  offset tables inside it; those tables store absolute byte positions into
  `mdat` and become wrong the instant `moov` moves, which can silently
  corrupt playback in a new, harder-to-diagnose way.
- **`torch` import taking 10+ seconds** on a fresh environment is normal,
  not a hang — don't Ctrl+C it. If it's consistently much slower than that,
  add a Windows Defender real-time-scanning exclusion for the project's
  `.venv` folder.

---

## Open question, not yet resolved: Phase 3's system prompt

`phase3/inference/prompt_builder.py`'s `SYSTEM_PROMPT` was at one point
rewritten from explicit anti-hallucination hard rules ("use ONLY facts in
the JSON, do NOT invent player/team names, scoreline, crowd...") to a
broadcaster-persona prompt ("sound like a famous commentator, poetic,
thrilling") with no hard grounding rules left. The field-stripping that
hides `confidence`/`pitch_zone`/`ball_speed`/`recent_events` from the LLM is
still intact, so it can't quote numbers it was never given — but nothing
stops it from inventing crowd reactions or flourish not licensed by the
facts it does see (confirmed in real output: a commentary line described
the ball as moving at "electric pace" despite `ball_speed` never reaching
the model at all). This is a real trade-off against the project's own
stated "no hallucination guarantee" architecture and its dissertation claim
of grounded commentary — decide deliberately whether to keep it, and
disclose it if kept.

---

## The TTS warm-up fix

Chrome/Edge refuse to start speech synthesis until a user gesture has
occurred. `useTTS.js` calls
`window.speechSynthesis.speak(new SpeechSynthesisUtterance(''))` silently
inside the upload/drag handler the first time it runs. Without this, TTS
silently does nothing and there is no error message.

## The speech queue

Rapid events would overlap without serialisation. `useTTS.js` maintains a
`queue` ref and chains `speakNext()` inside each utterance's `onend`
callback, so sentences play back-to-back with no overlap and no timer.

---

## Phase 4 exit gate

You should be able to:
1. Open `http://localhost:5173`, pick "NEW FOOTAGE", and upload a football
   clip → sport badge reads **FOOTBALL DETECTED**
2. Watch bounding boxes appear on the video as it plays (if the browser
   preview itself fails to decode the file, that's a separate, cosmetic
   issue — see the browser/OS quirks above — the pipeline keeps working
   regardless)
3. See live captions stream in word-by-word and hear commentary aloud
4. Switch style mid-playback and confirm tone changes on the next event
5. Check `outputs/<clip>_commentary.json` was written once the session ends

If all five steps work cleanly, Phase 4 is working end to end.

---

## Running tests

```bash
# From project root
python -m pytest phase4/tests/test_phase4_backend.py -v
```

8 tests cover the health/upload/transcript endpoints, the offline replay
path, and cross-phase import chains (Phase 4 → Phase 3 → Phase 2 → Phase 1)
without needing a running server, YOLO weights, or a trained XGBoost model.