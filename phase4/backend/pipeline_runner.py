"""
CASTER Phase 4 -- Pipeline Runner. FOOTBALL ONLY.

Wraps the Phase 1 + Phase 2 + Phase 3 pipeline into a single async
background coroutine that a WebSocket handler can await frame-by-frame,
receiving both bounding-box detection dicts (for the canvas overlay) and
commentary results (for LiveCaption + TTS) on the same generator.

Design principles:
  - Never import Phase 1/2/3 at module level here -- import inside the
    coroutine so a missing YOLO weight or XGBoost model gives a clear
    error at runtime, not at server startup.
  - Yield two message types: {"type": "frame", ...} and
    {"type": "commentary", ...} (plus "sport_detected" and "done").
  - The WebSocket handler is the ONLY caller -- it converts these dicts
    into websocket.send_json() calls directly.
  - For offline/pre-recorded clips, all frames run in one thread pool
    executor so async FastAPI routes aren't blocked.

FOOTBALL-ONLY NOTE (fixed 2026-11): This file previously imported
SPORT_CRICKET, phase1.scene_state.sport_detector / sport_detector_clip,
and a SportDetector(sport) class -- all of which were deleted in the
Phase 1/2/3 football-only conversion. This caused a hard ImportError at
server startup (`cannot import name 'SPORT_CRICKET' from 'config'`),
confirmed via `python phase4/run_phase4.py --check`. There is no sport
auto-detection anywhere in the project anymore -- CASTER is a single-sport
(football) pipeline, so this file no longer does any sport detection or
branching; "sport_detected" is still emitted (always "football") only
because the frontend/WS message contract documents that message type.

CONCURRENT COMMENTARY FIX (fixed 2026-11): _run_precomputed_pipeline used
to contain a comment claiming commentary generation ran "concurrently so
frame streaming does NOT freeze", immediately followed by a plain
`await loop.run_in_executor(None, engine.generate, event_data)` directly
in the frame loop -- i.e. it was NOT concurrent at all. Every Groq call
(0.5-1.5s+ round trip, confirmed in server logs) blocked the entire
generator, so no "frame" messages were yielded to the frontend for that
whole window. The frontend's video kept playing in real wall-clock time
regardless, so by the time frames resumed it was often 1-2+ seconds
ahead of the last frame it had received -- which is exactly the
"correct for ~4-5s, then wrong" symptom this was causing (the frontend's
own backend-sync throttle was correctly detecting a real gap, not
misbehaving). Fixed by actually backgrounding commentary generation with
asyncio.create_task() and pushing its resulting messages through a
queue that the frame loop drains between frames, so frame streaming
genuinely never stops for a commentary call.
"""

import asyncio
import logging
import sys
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    ROOT_DIR, OUTPUTS_DIR,
    SPORT_FOOTBALL,
    FEATURE_WINDOW_SIZE,
)

logger = logging.getLogger("caster.phase4.pipeline_runner")


class PipelineRunner:
    """
    One instance shared across the FastAPI app. Manages which clips are
    currently being processed (to avoid double-processing on reconnect).
    """

    def __init__(self):
        self._active_clips: set = set()
        # BUG FIX (perceived 1-2s delay before the first bounding box):
        # every clip's _run_pipeline() called load_detector() fresh and
        # then ran its first real inference, which pays a one-time cost
        # beyond just loading weights -- CUDA context creation and
        # cuDNN's internal kernel benchmarking on the first forward pass.
        # Since PipelineRunner is constructed exactly once per server
        # process (see ws_handler.py's module-level `_runner =
        # PipelineRunner()`), that one-time cost can be paid HERE, once,
        # at server startup, instead of being repaid on every upload.
        # _warm_detector holds the pre-loaded, pre-warmed model so
        # _run_pipeline can reuse it instead of loading+warming fresh
        # each time. Runs in a background thread so it never blocks
        # server startup or a request that arrives before it finishes --
        # _run_pipeline falls back to a fresh load_detector() call in
        # that race, exactly as before.
        self._warm_detector = None
        self._start_background_warmup()

    def _start_background_warmup(self):
        import threading

        def _warm():
            try:
                import numpy as _np
                from phase1.detection.yolo_detector import load_detector
                det = load_detector()
                dummy = _np.zeros((720, 1280, 3), dtype=_np.uint8)
                det.detect_frame(dummy)  # pays the one-time CUDA/cuDNN warm-up cost here
                self._warm_detector = det
                logger.info("Detector pre-warmed at server startup -- first live upload will not pay this cost.")
            except Exception as exc:
                # Warm-up is an optimisation, not a requirement -- if it
                # fails for any reason, _run_pipeline's own load_detector()
                # call still works exactly as before, just without the
                # head start.
                logger.warning("Background detector warm-up failed (non-fatal): %s", exc)

        threading.Thread(target=_warm, daemon=True).start()

    async def run(
        self,
        clip_name: str,
        style: str = "dramatic",
        style_provider: Optional[Callable[[], str]] = None,
        play_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Full Phase 1 -> 2 -> 3 pipeline for one clip.

        play_event: if given, both pipeline variants await this before
            processing each frame. ws_handler.py sets it on connect and
            clears/sets it on incoming {"type":"pause"}/{"type":"resume"}
            messages, so pausing the frontend's <video> element actually
            halts backend frame/event/commentary generation too -- before
            this, pausing was purely cosmetic in the browser and the
            backend (and Groq calls) kept running regardless.

        Yields dicts:
            {"type": "sport_detected", "sport": "football"}
            {"type": "frame", "frame_id": int, "timestamp_sec": float,
             "detections": [...], "feature_vector": [...]}
            {"type": "event", "event": str, "confidence": float,
             "sport": str, "score": str, "time": str}
            {"type": "token", "token": str}
            {"type": "sentence_end", "event": str, "sport": str,
             "confidence": float, "style": str, "backend_used": str}
            {"type": "speak", "text": str, "lang": "en-GB"}
            {"type": "done", "clip_name": str, "total_events": int}
        """
        if clip_name in self._active_clips:
            yield {"type": "error", "message": f"Clip '{clip_name}' is already being processed"}
            return

        self._active_clips.add(clip_name)
        try:
            # Check if pre-computed Phase 1 + Phase 2 outputs already exist for this clip
            # (e.g. from run_phase1.py & run_phase2.py).
            # If they exist, stream from them in real-time â€” instant bounding boxes and
            # immediate commentary perfectly synchronized with video playback!
            precomputed_scene_states = OUTPUTS_DIR / f"{clip_name}_scene_states.json"
            precomputed_events = OUTPUTS_DIR / f"{clip_name}_scene_json_events.json"

            if precomputed_scene_states.exists() and precomputed_events.exists():
                logger.info("Found pre-computed pipeline outputs for '%s'. Using high-speed synced stream.", clip_name)
                async for msg in self._run_precomputed_pipeline(clip_name, style, style_provider, play_event):
                    yield msg
            else:
                async for msg in self._run_pipeline(clip_name, style, style_provider, play_event):
                    yield msg
        finally:
            self._active_clips.discard(clip_name)

    async def _run_pipeline(
        self,
        clip_name: str,
        style: str,
        style_provider: Optional[Callable[[], str]] = None,
        play_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[dict, None]:
        # ------------------------------------------------------------------ #
        # Lazy imports -- Phase 1/2/3 modules not needed at server startup.   #
        # ------------------------------------------------------------------ #
        try:
            from phase1.detection.yolo_detector import load_detector
            from phase1.tracking.bytetrack_wrapper import iter_tracked_frames
            from phase1.features.feature_extractor import FeatureExtractor
            from phase1.scene_state.scene_state_builder import build_scene_state
            from phase2.event_classifier.xgboost_clf import XGBoostClassifier
            from phase2.confidence_gate import ConfidenceGateState, ball_visibility_ok
            from phase2.scene_json_builder import build_scene_json, MatchState, apply_event_to_match_state
            from phase3.inference.commentary_engine import CommentaryEngine
            from schemas import EventPrediction, FeatureWindow
        except ImportError as exc:
            yield {"type": "error", "message": f"Import error in pipeline: {exc}"}
            return

        import cv2
        import time as _time
        from collections import deque

        # ------------------------------------------------------------------ #
        # Locate the video file                                               #
        # ------------------------------------------------------------------ #
        video_path = ROOT_DIR / "data" / "raw" / f"{clip_name}.mp4"
        if not video_path.exists():
            for ext in (".avi", ".mov", ".mkv", ".webm"):
                candidate = video_path.with_suffix(ext)
                if candidate.exists():
                    video_path = candidate
                    break
            else:
                yield {"type": "error", "message": f"Video not found: {clip_name}.*"}
                return

        # ------------------------------------------------------------------ #
        # Phase 1 -- load the (single) football detector.                    #
        # ------------------------------------------------------------------ #
        # Reuse the pre-warmed detector from server startup if it's ready
        # (see __init__'s _start_background_warmup) -- skips both the
        # weight-load and the one-time CUDA/cuDNN warm-up cost that
        # otherwise gets paid again on every single upload. If the warm-up
        # background thread hasn't finished yet (e.g. a user connects
        # within the first second or two of the server starting), this
        # just falls back to loading fresh, exactly as before -- no
        # change in correctness, only a possible head start.
        if self._warm_detector is not None:
            logger.info("Reusing pre-warmed detector for clip: %s", clip_name)
            detector = self._warm_detector
        else:
            logger.info("Loading football detector for clip: %s (not pre-warmed yet)", clip_name)
            detector = await asyncio.get_event_loop().run_in_executor(None, load_detector)

        sport = SPORT_FOOTBALL
        yield {"type": "sport_detected", "sport": sport}

        extractor = FeatureExtractor()

        # ------------------------------------------------------------------ #
        # Phase 2 -- Load classifier                                          #
        # ------------------------------------------------------------------ #
        clf = XGBoostClassifier(sport=sport)
        clf_loaded = clf.load()
        if not clf_loaded:
            logger.warning(
                "No trained XGBoost model found for %s. Events will not be classified. "
                "Train Phase 2 first: python phase2/run_phase2.py", sport
            )

        gate = ConfidenceGateState()
        match_state = MatchState(sport=sport, commentary_style=style)

        # ------------------------------------------------------------------ #
        # Phase 3 -- Commentary engine                                        #
        # ------------------------------------------------------------------ #
        engine = CommentaryEngine()

        # ------------------------------------------------------------------ #
        # CPU-aware adaptive stride                                           #
        #                                                                      #
        # On GPU each YOLO frame takes ~30-80 ms â†’ stride 2 (30 fps) is fine.#
        # On CPU each YOLO frame takes ~1-3 s.  Processing at 30 fps would   #
        # make the pipeline ~60-90Ã— behind real-time, so commentary never    #
        # catches up to the video the user sees in the browser.              #
        #                                                                      #
        # Strategy: aim to process exactly CPU_TARGET_PROCESSED_FPS frames   #
        # per second of VIDEO TIME (not wall time).  On a 60 fps source that #
        # means stride = 60 / 5 = 12 â†’ we look at one frame every ~200 ms of#
        # video, which is enough for event detection while keeping the        #
        # pipeline only ~3 s behind real playback even on pure CPU.          #
        #                                                                      #
        # GPU path is unchanged: when CUDA is available torch will speed      #
        # through frames quickly enough that the smaller stride stays in sync.#
        # ------------------------------------------------------------------ #
        import torch as _torch
        _cuda_available = _torch.cuda.is_available()

        dims_cap = cv2.VideoCapture(str(video_path))
        src_fps = dims_cap.get(cv2.CAP_PROP_FPS) or 30.0
        dims_cap.release()

        if _cuda_available:
            # GPU: process ~30 video-fps worth of frames
            STRIDE_TARGET_FPS = 30.0
        else:
            # CPU: process only 5 video-fps worth of frames so commentary
            # fires within a few seconds of the matching moment in the video
            STRIDE_TARGET_FPS = 5.0

        stride = max(1, round(src_fps / STRIDE_TARGET_FPS))
        logger.info(
            "Device: %s | Source %.1f fps -> stride=%d (~%.1f processed fps)",
            "GPU" if _cuda_available else "CPU", src_fps, stride, src_fps / stride,
        )
        yield {
            "type": "warning",
            "message": (
                f"Running on {'GPU' if _cuda_available else 'CPU'} at "
                f"~{src_fps/stride:.1f} processed fps (stride={stride}). "
                + ("" if _cuda_available else
                   "Commentary will appear within ~5-10 s of each event in the video.")
            ),
        }

        feature_buffer = deque(maxlen=FEATURE_WINDOW_SIZE)
        # scene states for the frames currently sitting in feature_buffer, so we
        # can check ball visibility across the SAME window the classifier saw
        scene_buffer = deque(maxlen=FEATURE_WINDOW_SIZE)
        total_events = 0
        commentary_log = []

        loop = asyncio.get_event_loop()
        track_iter = iter_tracked_frames(detector, str(video_path), stride=stride)

        # ------------------------------------------------------------------ #
        # Background commentary generation (see CONCURRENT COMMENTARY FIX   #
        # in the module docstring). Each event spawns a task that runs      #
        # engine.generate() in a thread and pushes every resulting message  #
        # (token / sentence_end / speak) onto this queue. The main frame    #
        # loop below drains the queue between frames WITHOUT ever awaiting  #
        # a commentary call directly, so frame streaming never stalls for   #
        # the 0.5-1.5s+ a Groq round trip can take.                          #
        # ------------------------------------------------------------------ #
        commentary_queue: "asyncio.Queue[dict]" = asyncio.Queue()
        background_tasks: list = []

        async def _generate_commentary_task(scene_json_for_llm, event_name, sport_name):
            """Runs one commentary generation in the background and streams
            its messages onto commentary_queue as they become ready, instead
            of returning them all at once -- so the frame loop can forward
            each one to the frontend as soon as it's queued, keeping the
            same word-by-word streaming feel as before."""
            try:
                result = await loop.run_in_executor(None, engine.generate, scene_json_for_llm)
                for word in result["text"].split():
                    # Respect pause here too -- without this, word-by-word
                    # pacing keeps running (and queuing) during a pause,
                    # so everything arrives in one burst the instant the
                    # user resumes instead of resuming the same gentle
                    # word-by-word reveal.
                    if play_event is not None:
                        await play_event.wait()
                    await commentary_queue.put({"type": "token", "token": word + " "})
                await commentary_queue.put({
                    "type": "sentence_end",
                    "event": result["event"],
                    "sport": result["sport"],
                    "confidence": result["confidence"],
                    "style": result["style"],
                    "backend_used": result["backend_used"],
                    "full_text": result["text"],
                })
                await commentary_queue.put({
                    "type": "speak",
                    "text": result["text"],
                    "lang": "en-GB",
                })
                commentary_log.append({
                    "event": result["event"],
                    "sport": result["sport"],
                    "confidence": result["confidence"],
                    "score": scene_json_for_llm.get("score", match_state.football_score_str()),
                    "time": str(scene_json_for_llm.get("match_minute", "")),
                    "style": result["style"],
                    "backend_used": result["backend_used"],
                    "commentary": result["text"],
                })
            except Exception as exc:
                logger.warning("Commentary generation error: %s", exc)

        def _drain_commentary_queue():
            """Pulls every message currently sitting in the queue, without
            blocking if it's empty. Called once per frame from the main
            loop so queued commentary messages interleave with frame
            messages in roughly the order they became ready."""
            msgs = []
            while not commentary_queue.empty():
                msgs.append(commentary_queue.get_nowait())
            return msgs

        # ------------------------------------------------------------------ #
        # Video-time synchronisation                                           #
        #                                                                      #
        # We track when processing started (wall time) and what video         #
        # timestamp that maps to.  Before yielding each frame message we      #
        # check whether the pipeline is running AHEAD of real-time playback.  #
        # If it is, we sleep until the right wall-clock moment arrives so     #
        # the bounding-box overlay and commentary appear at roughly the same  #
        # moment the user is watching that part of the video.                 #
        #                                                                      #
        # On CPU (slow inference) the pipeline will almost always be BEHIND   #
        # real-time, so the sleep is 0 and frames are sent as fast as the    #
        # hardware allows.  On GPU (fast inference) this ensures the pipeline #
        # doesn't race so far ahead that it sends commentary for future       #
        # events before the user has even seen them.                          #
        # ------------------------------------------------------------------ #
        pipeline_wall_start: Optional[float] = None
        pipeline_video_start: float = 0.0

        def _next_tracked_frame():
            try:
                return next(track_iter)
            except StopIteration:
                return None

        try:
            while True:
                # Pause support: block here (not mid-frame) while the user
                # has paused playback. Resetting the wall-clock baseline
                # right after a pause (below) prevents a "catch-up burst"
                # where every frame that would have streamed during the
                # pause tries to send at once the moment we resume.
                if play_event is not None and not play_event.is_set():
                    await play_event.wait()
                    pipeline_wall_start = None  # re-anchor on next iteration

                tf = await loop.run_in_executor(None, _next_tracked_frame)
                if tf is None:
                    break

                detections = tf.detections
                frame_w, frame_h = tf.frame_w, tf.frame_h
                frame_id = tf.frame_idx
                timestamp_sec = tf.timestamp_sec

                # ---------------------------------------------------------------- #
                # Video-time synchronisation: keep commentary aligned with         #
                # the video the user is watching in their browser.                 #
                #                                                                   #
                # On CPU, YOLO takes ~2-3 s per frame, so the pipeline is always  #
                # BEHIND real time â€” sleep is 0, we go as fast as we can.         #
                # On GPU, YOLO takes ~30-80 ms per frame, so the pipeline races   #
                # ahead of real playback; we sleep until the right wall-clock     #
                # moment so overlays and commentary arrive at the right video time.#
                # ---------------------------------------------------------------- #
                if pipeline_wall_start is None:
                    pipeline_wall_start = _time.monotonic()
                    pipeline_video_start = timestamp_sec

                # How much wall-clock time has elapsed since we started?
                elapsed_wall = _time.monotonic() - pipeline_wall_start
                # How much video time have we processed?
                elapsed_video = timestamp_sec - pipeline_video_start
                # If we're ahead of real-time, wait until the video catches up
                ahead_by = elapsed_video - elapsed_wall
                if ahead_by > 0.05:  # only sleep if >50 ms ahead (avoid micro-sleeps)
                    await asyncio.sleep(ahead_by)

                feature_vec = await loop.run_in_executor(
                    None, extractor.extract, detections, frame_w, frame_h
                )
                feature_buffer.append(feature_vec)

                scene_state = build_scene_state(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    detections=detections,
                    feature_vector=feature_vec,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    tracking_diagnostics=extractor.get_tracking_diagnostics(),
                )

                # Normalise bounding boxes to 0-1 for frontend (avoids resolution coupling)
                norm_detections = [
                    {
                        "track_id": d.track_id,
                        "cls_name": d.cls_name,
                        "confidence": round(d.confidence, 3),
                        "x": round(d.x / frame_w, 4),
                        "y": round(d.y / frame_h, 4),
                        "w": round(d.w / frame_w, 4),
                        "h": round(d.h / frame_h, 4),
                    }
                    for d in detections
                ]

                yield {
                    "type": "frame",
                    "frame_id": frame_id,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "detections": norm_detections,
                    "feature_vector": [round(v, 4) for v in feature_vec],
                    "sport": sport,
                    "ball_detected": scene_state.ball_detected,
                    "ball_track_id": scene_state.ball_track_id,
                    "ball_missing_frames": scene_state.ball_missing_frames,
                }

                # Flush any commentary messages that finished generating in
                # the background since the last frame -- this is what
                # actually delivers token/sentence_end/speak to the
                # WebSocket now, instead of the old inline blocking call.
                for msg in _drain_commentary_queue():
                    yield msg

                scene_buffer.append(scene_state)

                # Phase 2 -- classify when we have a full window
                if clf_loaded and len(feature_buffer) == FEATURE_WINDOW_SIZE:
                    window = FeatureWindow(
                        sport=sport,
                        end_frame_id=frame_id,
                        end_timestamp_sec=timestamp_sec,
                        vectors=list(feature_buffer),
                    )
                    try:
                        event, confidence = await loop.run_in_executor(
                            None, clf.predict_window, window.vectors
                        )
                    except Exception as exc:
                        logger.debug("Classifier predict error: %s", exc)
                        continue

                    # A window where the ball was barely seen is not evidence of
                    # anything -- see confidence_gate.ball_visibility_ok and
                    # CASTER_CHANGELOG.md. This is checked BEFORE the confidence
                    # gate so a confidently-wrong prediction from empty features
                    # never reaches it.
                    if not ball_visibility_ok(s.ball_detected for s in scene_buffer):
                        continue

                    gate_decision = gate.check(sport, event, confidence, now=timestamp_sec)

                    if gate_decision.passed:
                        # WebSocket style changes happen while the pipeline is
                        # running, so read the current value immediately before
                        # constructing the LLM prompt.
                        match_state.commentary_style = style_provider() if style_provider else style
                        prediction = EventPrediction(
                            sport=sport,
                            event=event,
                            confidence=confidence,
                            scene=scene_state,
                            recent_events=list(match_state.recent_events),
                        )
                        scene_json = build_scene_json(prediction, match_state, clip_time_sec=timestamp_sec)
                        apply_event_to_match_state(match_state, sport, event)
                        total_events += 1

                        yield {
                            "type": "event",
                            "event": event,
                            "confidence": round(confidence, 4),
                            "sport": sport,
                            "score": match_state.football_score_str(),
                            "time": str(match_state.match_minute),
                            "gate_reason": gate_decision.reason,
                        }

                        # Phase 3 -- fire commentary generation in the
                        # BACKGROUND. We do NOT await it here -- that was
                        # the bug. The frame loop keeps going immediately;
                        # results arrive later via commentary_queue and get
                        # drained (and yielded) at the top of a later
                        # iteration of this same loop.
                        task = loop.create_task(
                            _generate_commentary_task(scene_json, event, sport)
                        )
                        background_tasks.append(task)

                # Yield control back to event loop every frame so WebSocket
                # can flush buffered sends without waiting for the whole clip
                await asyncio.sleep(0)

            # Frame source exhausted -- wait for any still-running commentary
            # tasks to finish and drain whatever they queued, so the last
            # event's commentary isn't silently dropped at clip end.
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
                for msg in _drain_commentary_queue():
                    yield msg

        finally:
            for task in background_tasks:
                if not task.done():
                    task.cancel()

        logger.info("Frame loop complete, %d events", total_events)

        if commentary_log:
            import json as _json
            out_path = OUTPUTS_DIR / f"{clip_name}_commentary.json"
            with open(out_path, "w") as f:
                _json.dump(commentary_log, f, indent=2)
            logger.info("Saved %d commentary lines -> %s", len(commentary_log), out_path)

        yield {"type": "done", "clip_name": clip_name, "total_events": total_events}

    async def _run_precomputed_pipeline(
        self,
        clip_name: str,
        style: str,
        style_provider: Optional[Callable[[], str]] = None,
        play_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Runs when Phase 1 scene states and Phase 2 events already exist in outputs/.
        Streams frames synced to playback time and triggers Phase 3 commentary
        at the exact timestamp recorded in the events.
        """
        import json as _json
        import time as _time
        import cv2
        from phase3.inference.commentary_engine import CommentaryEngine

        yield {"type": "sport_detected", "sport": SPORT_FOOTBALL}

        states_path = OUTPUTS_DIR / f"{clip_name}_scene_states.json"
        events_path = OUTPUTS_DIR / f"{clip_name}_scene_json_events.json"

        with open(states_path, "r", encoding="utf-8") as f:
            scene_states = _json.load(f)

        # ------------------------------------------------------------------ #
        # BUG FIX (RE-APPLIED -- this zip was built from a branch that       #
        # predates the earlier fix): scene_states.json stores each           #
        # Detection's x/y/w/h in RAW SOURCE-VIDEO PIXELS (schemas.SceneState #
        # has no frame_w/frame_h field at all). A hardcoded                  #
        # fw=1920.0/fh=1080.0 here silently misplaces every bounding box for #
        # ANY clip that isn't exactly 1920x1080. Fixed by reading the real   #
        # dimensions off the source video file itself, once, via cv2.        #
        # ------------------------------------------------------------------ #
        frame_w, frame_h = 1920.0, 1080.0
        video_path = ROOT_DIR / "data" / "raw" / f"{clip_name}.mp4"
        if not video_path.exists():
            for ext in (".avi", ".mov", ".mkv", ".webm"):
                candidate = video_path.with_suffix(ext)
                if candidate.exists():
                    video_path = candidate
                    break
        if video_path.exists():
            dims_cap = cv2.VideoCapture(str(video_path))
            real_w = dims_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            real_h = dims_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            dims_cap.release()
            if real_w and real_h:
                frame_w, frame_h = real_w, real_h
            else:
                yield {
                    "type": "warning",
                    "message": (
                        f"Could not read frame dimensions from {video_path.name} -- "
                        "falling back to 1920x1080 for the bounding-box overlay."
                    ),
                }
        else:
            yield {
                "type": "warning",
                "message": (
                    f"Source video for '{clip_name}' not found in data/raw/ -- "
                    "falling back to 1920x1080 for the bounding-box overlay."
                ),
            }

        with open(events_path, "r", encoding="utf-8") as f:
            events = _json.load(f)

        engine = CommentaryEngine()

        # BUG FIX (~1s constant playback lag): wall_start used to be set
        # HERE, immediately after engine construction but still before the
        # frame loop below. That looks right, but the frontend's <video>
        # element starts playing as soon as the WebSocket connects (see
        # VideoPlayer.jsx), which happens well before this function even
        # starts -- the sport_detected yield above, the scene_states.json
        # parse (can be several MB for a long clip), the events.json parse,
        # and this CommentaryEngine() construction all take real wall-clock
        # time that the video has already been playing through. Since the
        # sync loop below only ever sleeps when AHEAD of schedule and never
        # speeds up when behind, that setup-time gap became a permanent,
        # constant lag for the whole session. Fixed by anchoring wall_start
        # lazily, on the first iteration of the loop itself (same pattern
        # as _run_pipeline's pipeline_wall_start), so none of the file-load
        # time above counts against the sync baseline. Combined with the
        # frontend fix (play() on first frame received, not on WS connect),
        # this should remove the lag almost entirely.
        wall_start: Optional[float] = None
        video_start = scene_states[0]["timestamp_sec"] if scene_states else 0.0

        # Sort events by clip_time if available
        def parse_clip_time(e):
            ct = e.get("clip_time")
            if not ct:
                return -1.0
            parts = str(ct).split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            try:
                return float(ct)
            except Exception:
                return -1.0

        event_queue = deque(events)
        total_events = 0
        commentary_log = []
        loop = asyncio.get_event_loop()

        # ------------------------------------------------------------------ #
        # CONCURRENT COMMENTARY FIX (see module docstring): this used to    #
        # `await loop.run_in_executor(None, engine.generate, event_data)`   #
        # directly in the frame loop below -- despite a comment right above #
        # it claiming generation ran "concurrently so frame streaming does  #
        # NOT freeze". It did not; every Groq call blocked the whole        #
        # generator, so no "frame" messages reached the frontend for that   #
        # call's entire duration (0.5-1.5s+, per server logs), which is     #
        # exactly what caused the video's real-time playback to outrun the  #
        # last frame the frontend had received. Fixed the same way as      #
        # _run_pipeline: generation now runs as a background asyncio task   #
        # that pushes its messages onto a queue, drained once per frame.    #
        # ------------------------------------------------------------------ #
        commentary_queue: "asyncio.Queue[dict]" = asyncio.Queue()
        background_tasks: list = []

        async def _generate_commentary_task(event_data):
            try:
                res = await loop.run_in_executor(None, engine.generate, event_data)
                for word in res["text"].split():
                    # See the equivalent comment in _run_pipeline's
                    # _generate_commentary_task -- pacing should respect
                    # pause too, not just the frame loop.
                    if play_event is not None:
                        await play_event.wait()
                    await commentary_queue.put({"type": "token", "token": word + " "})
                await commentary_queue.put({
                    "type": "sentence_end",
                    "event": res["event"],
                    "sport": res["sport"],
                    "confidence": res["confidence"],
                    "style": res["style"],
                    "backend_used": res["backend_used"],
                    "full_text": res["text"],
                })
                await commentary_queue.put({
                    "type": "speak",
                    "text": res["text"],
                    "lang": "en-GB",
                })
                commentary_log.append({
                    "event": res["event"],
                    "sport": res["sport"],
                    "confidence": res["confidence"],
                    "score": event_data.get("score", "0 - 0"),
                    "time": str(event_data.get("match_minute", "0")),
                    "style": res["style"],
                    "backend_used": res["backend_used"],
                    "commentary": res["text"],
                })
            except Exception as e:
                logger.warning("Precomputed commentary error: %s", e)

        def _drain_commentary_queue():
            msgs = []
            while not commentary_queue.empty():
                msgs.append(commentary_queue.get_nowait())
            return msgs

        try:
            for state in scene_states:
                # Pause support (see play_event docstring on run()): block
                # here, between frames, while paused. Re-anchoring
                # wall_start on the next line after a pause prevents a
                # catch-up burst of buffered frames firing all at once
                # the moment the user resumes.
                if play_event is not None and not play_event.is_set():
                    await play_event.wait()
                    wall_start = None

                t_sec = state["timestamp_sec"]

                # First iteration (or first iteration after a pause):
                # anchor the sync baseline HERE, not before file loading --
                # see the BUG FIX note above wall_start's declaration.
                if wall_start is None:
                    wall_start = _time.monotonic()
                    video_start = t_sec

                # Real-time synchronization: sync streaming of frames to video playback clock.
                elapsed_wall = _time.monotonic() - wall_start
                elapsed_video = t_sec - video_start
                ahead_by = elapsed_video - elapsed_wall
                if ahead_by > 0.005:
                    await asyncio.sleep(ahead_by)

                # Normalise detections using the REAL source-video dimensions
                # resolved above (was hardcoded to 1920x1080).
                norm_detections = [
                    {
                        "track_id": d.get("track_id"),
                        "cls_name": d.get("cls_name"),
                        "confidence": round(d.get("confidence", 0.0), 3),
                        "x": round(d.get("x", 0.0) / frame_w, 4),
                        "y": round(d.get("y", 0.0) / frame_h, 4),
                        "w": round(d.get("w", 0.0) / frame_w, 4),
                        "h": round(d.get("h", 0.0) / frame_h, 4),
                    }
                    for d in state.get("detections", [])
                ]

                yield {
                    "type": "frame",
                    "frame_id": state.get("frame_id", 0),
                    "timestamp_sec": round(t_sec, 3),
                    "detections": norm_detections,
                    "feature_vector": [round(v, 4) for v in state.get("feature_vector", [])],
                    "sport": SPORT_FOOTBALL,
                    "ball_detected": state.get("ball_detected", False),
                    "ball_track_id": state.get("ball_track_id"),
                    "ball_missing_frames": state.get("ball_missing_frames", 0),
                }

                # Flush whatever commentary finished generating in the
                # background since the last frame -- this is what actually
                # delivers token/sentence_end/speak now.
                for msg in _drain_commentary_queue():
                    yield msg

                # Check if an event matches this timestamp
                while event_queue:
                    next_ev = event_queue[0]
                    ev_time = parse_clip_time(next_ev)
                    if ev_time >= 0 and t_sec >= ev_time:
                        event_data = event_queue.popleft()
                        current_s = style_provider() if style_provider else style
                        event_data["commentary_style"] = current_s

                        total_events += 1
                        yield {
                            "type": "event",
                            "event": event_data.get("event", "UNKNOWN"),
                            "confidence": round(event_data.get("confidence", 0.0), 4),
                            "sport": SPORT_FOOTBALL,
                            "score": event_data.get("score", "0 - 0"),
                            "time": str(event_data.get("match_minute", "0")),
                            "gate_reason": "precomputed_event",
                        }

                        # Fire-and-forget: generation happens in the
                        # background and its messages arrive later via the
                        # queue, drained at the top of a later frame
                        # iteration. The frame loop does NOT wait here.
                        task = loop.create_task(_generate_commentary_task(event_data))
                        background_tasks.append(task)
                    else:
                        break

            # Clip finished -- let any still-running commentary tasks
            # complete and flush their messages so the last event isn't lost.
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
                for msg in _drain_commentary_queue():
                    yield msg

        finally:
            for task in background_tasks:
                if not task.done():
                    task.cancel()

        yield {"type": "done", "clip_name": clip_name, "total_events": total_events}

    async def _run_offline_phase2_phase3(self, clip_name: str, style: str):
        """
        Fallback when no YOLO weights exist: reads Phase 2's saved
        *_scene_json_events.json output and replays it through Phase 3.
        Exactly what run_phase3.py does, but as an async generator so
        the WebSocket can still stream results live.
        """
        import json as _json
        try:
            from phase3.inference.commentary_engine import CommentaryEngine
        except ImportError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        events_path = OUTPUTS_DIR / f"{clip_name}_scene_json_events.json"
        if not events_path.exists():
            yield {
                "type": "error",
                "message": (
                    f"No Phase 2 output found at {events_path}. "
                    "Run Phase 1+2 first: python phase2/run_phase2.py --clip " + clip_name
                ),
            }
            return

        with open(events_path) as f:
            events = _json.load(f)

        engine = CommentaryEngine()
        commentary_log = []

        for scene_json in events:
            scene_json = dict(scene_json)
            if style:
                scene_json["commentary_style"] = style

            yield {
                "type": "event",
                "event": scene_json.get("event", "UNKNOWN"),
                "confidence": scene_json.get("confidence", 0.0),
                "sport": scene_json.get("sport", SPORT_FOOTBALL),
                "score": scene_json.get("score", ""),
                "time": str(scene_json.get("match_minute", "")),
                "gate_reason": "offline_replay",
            }

            result = await asyncio.get_event_loop().run_in_executor(
                None, engine.generate, scene_json
            )

            for word in result["text"].split():
                yield {"type": "token", "token": word + " "}
                await asyncio.sleep(0.04)

            yield {
                "type": "sentence_end",
                "event": result["event"],
                "sport": result["sport"],
                "confidence": result["confidence"],
                "style": result["style"],
                "backend_used": result["backend_used"],
                "full_text": result["text"],
            }
            yield {"type": "speak", "text": result["text"], "lang": "en-GB"}

            commentary_log.append({
                "event": result["event"],
                "sport": result["sport"],
                "confidence": result["confidence"],
                "score": scene_json.get("score", ""),
                "time": str(scene_json.get("match_minute", "")),
                "style": result["style"],
                "backend_used": result["backend_used"],
                "commentary": result["text"],
            })

        if commentary_log:
            import json as _json2
            out_path = OUTPUTS_DIR / f"{clip_name}_commentary.json"
            with open(out_path, "w") as f:
                _json2.dump(commentary_log, f, indent=2)
            logger.info("Saved %d commentary lines -> %s", len(commentary_log), out_path)

        yield {"type": "done", "clip_name": clip_name, "total_events": len(events)}