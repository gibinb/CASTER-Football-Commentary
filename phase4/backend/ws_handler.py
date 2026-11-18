"""
CASTER Phase 4 -- WebSocket handler.

Mounts two WebSocket routes onto the Phase 4 FastAPI app:

  WS /api/ws/stream?clip=<name>&style=<dramatic|casual|statistical>
      Full live pipeline: YOLO + ByteTrack + XGBoost + CommentaryEngine.
      Streams frame detections, events, and commentary tokens in real-time.
      This is what the React frontend connects to after uploading a video.

  WS /api/ws/replay?clip=<name>&style=<dramatic|casual|statistical>
      Offline replay of Phase 3's saved *_commentary.json. Useful for
      reviewing a clip that's already been processed, or for demoing
      without re-running YOLO.

Message shape (all messages are JSON):
  { "type": "sport_detected", "sport": "cricket" }
  { "type": "frame",          "frame_id": 0, "timestamp_sec": 0.0,
                               "detections": [...], "feature_vector": [...] }
  { "type": "event",          "event": "FOUR", "confidence": 0.88,
                               "sport": "cricket", "score": "87/2", "time": "14.3" }
  { "type": "token",          "token": "FOUR " }
  { "type": "sentence_end",   "event": ..., "full_text": ..., "backend_used": ... }
  { "type": "speak",          "text": "...", "lang": "en-GB" }
  { "type": "done",           "clip_name": ..., "total_events": 3 }
  { "type": "error",          "message": "..." }
  { "type": "warning",        "message": "..." }
  { "type": "style_changed",  "style": "casual" }

The frontend can also SEND messages back on the same socket:
  { "type": "set_style", "style": "casual" }
  -- causes future commentary to use the new style immediately
  { "type": "pause" }
  -- pauses the backend pipeline (frame/event/commentary generation) in
     lockstep with the user pausing the video, via a play_event the
     pipeline's frame loop awaits on every iteration (see
     pipeline_runner.py). Without this, pausing the video was purely
     cosmetic -- the backend kept detecting events and generating
     commentary regardless, which is confusing (commentary appearing
     for a part of the match the user has paused on).
  { "type": "resume" }
  -- resumes the backend pipeline after a pause.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

sys.path.append(str(Path(__file__).resolve().parents[2]))
from phase4.backend.pipeline_runner import PipelineRunner

logger = logging.getLogger("caster.phase4.ws_handler")
router = APIRouter()

# Shared runner -- same as the one in main.py
_runner = PipelineRunner()


@router.websocket("/api/ws/stream")
async def stream_pipeline(
    websocket: WebSocket,
    clip: str = Query(..., description="Clip name (stem, no extension)"),
    style: str = Query("dramatic", description="Commentary style"),
):
    """
    Full live pipeline WebSocket. The React frontend connects here after
    upload, then renders bounding boxes, captions, and TTS from the
    messages it receives.
    """
    await websocket.accept()
    logger.info("WS connected: clip=%s style=%s", clip, style)

    # Allow the frontend to change style mid-session
    current_style = {"value": style}

    # Starts "set" (playing). The frontend sends {"type":"pause"} /
    # {"type":"resume"} to clear/set it; pipeline_runner.run()'s frame
    # loop awaits this before processing each frame, so clearing it
    # genuinely halts frame/event/commentary generation rather than just
    # pausing the browser's <video> element cosmetically.
    play_event = asyncio.Event()
    play_event.set()

    async def receive_loop():
        """Background task that listens for incoming control messages."""
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                    msg_type = msg.get("type")
                    if msg_type == "set_style":
                        new_style = msg.get("style", current_style["value"])
                        if new_style in ("dramatic", "casual", "statistical"):
                            current_style["value"] = new_style
                            await websocket.send_json({"type": "style_changed", "style": new_style})
                            logger.info("Style changed to: %s", new_style)
                    elif msg_type == "pause":
                        play_event.clear()
                        logger.info("Pipeline paused: clip=%s", clip)
                    elif msg_type == "resume":
                        play_event.set()
                        logger.info("Pipeline resumed: clip=%s", clip)
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            pass

    recv_task = asyncio.create_task(receive_loop())

    try:
        async for message in _runner.run(
            clip, style=current_style["value"], play_event=play_event
        ):
            # Inject live style into every event that gets commentary
            # so mid-session style switches take effect immediately
            if message.get("type") == "event":
                message["current_style"] = current_style["value"]

            await websocket.send_json(message)

            if message.get("type") == "done":
                break

    except WebSocketDisconnect:
        logger.info("WS disconnected: %s", clip)
    except Exception as exc:
        logger.exception("Pipeline error for clip %s: %s", clip, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        recv_task.cancel()
        try:
            recv_task.result()
        except Exception:
            pass


@router.websocket("/api/ws/replay")
async def replay_commentary(
    websocket: WebSocket,
    clip: str = Query(...),
    style: str = Query("dramatic"),
):
    """
    Replay Phase 3's saved *_commentary.json over a WebSocket so the
    frontend can re-render captions and TTS for an already-processed clip.
    """
    await websocket.accept()
    logger.info("WS replay: clip=%s", clip)

    try:
        import json as _json
        from config import OUTPUTS_DIR
        path = OUTPUTS_DIR / f"{clip}_commentary.json"
        if not path.exists():
            await websocket.send_json({"type": "error", "message": f"No commentary saved for '{clip}'"})
            return

        with open(path) as f:
            records = _json.load(f)

        # Emit sport_detected from the first record
        if records:
            await websocket.send_json({
                "type": "sport_detected",
                "sport": records[0].get("sport", "unknown"),
            })

        for record in records:
            effective_style = style or record.get("style", "dramatic")
            text = record.get("commentary", "")

            await websocket.send_json({
                "type": "event",
                "event": record.get("event", "UNKNOWN"),
                "confidence": record.get("confidence", 0.0),
                "sport": record.get("sport", "unknown"),
                "score": record.get("score", ""),
                "time": str(record.get("time", "")),
                "gate_reason": "replay",
            })

            for word in text.split():
                await websocket.send_json({"type": "token", "token": word + " "})
                await asyncio.sleep(0.04)

            await websocket.send_json({
                "type": "sentence_end",
                "event": record.get("event"),
                "sport": record.get("sport"),
                "confidence": record.get("confidence", 0.0),
                "style": effective_style,
                "backend_used": record.get("backend_used", "replay"),
                "full_text": text,
            })
            await websocket.send_json({"type": "speak", "text": text, "lang": "en-GB"})

            await asyncio.sleep(0.8)  # pace replay so TTS can finish each sentence

        await websocket.send_json({"type": "done", "clip_name": clip, "total_events": len(records)})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Replay error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass