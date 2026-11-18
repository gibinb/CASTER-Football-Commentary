"""
CASTER Phase 3 -- WebSocket streaming endpoints, mounted onto backend/main.py.

Exposes:
  WS /ws/commentary          -- consumes scene_json events pushed onto Redis
                                 by the Phase 1/2 pipeline, streams generated
                                 commentary word-by-word to Phase 4's React
                                 frontend (see CASTER_Phase4_DeepDive.docx
                                 Section 3 for the exact message shapes
                                 expected on the client side).
  WS /ws/commentary/direct   -- dev/testing convenience: send a scene_json
                                 dict directly over the socket (no Redis
                                 needed), matching the Phase 3 exit-gate
                                 example in the deep-dive doc.
"""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from phase3.inference.commentary_engine import CommentaryEngine
from phase3.backend.redis_queue import SceneEventQueue

router = APIRouter()
_engine = CommentaryEngine()


@router.websocket("/ws/commentary")
async def commentary_stream(websocket: WebSocket):
    await websocket.accept()
    queue = SceneEventQueue()

    try:
        while True:
            scene_json = await asyncio.get_event_loop().run_in_executor(None, queue.pop, 5)
            if scene_json is None:
                continue  # timeout, loop again -- keeps us responsive to disconnects

            result = _engine.generate(scene_json)

            # Emulate token streaming word-by-word so Phase 4's LiveCaption
            # component can render it the same way it would consume true
            # token-level streaming once local_gguf is wired in with
            # llama-cpp-python's native streaming API instead of subprocess.
            for word in result["text"].split():
                await websocket.send_json({"type": "token", "token": word + " "})
                await asyncio.sleep(0.05)

            await websocket.send_json({
                "type": "sentence_end",
                "event": result["event"],
                "sport": result["sport"],
                "confidence": result["confidence"],
                "style": result["style"],
                "backend_used": result["backend_used"],
            })
            await websocket.send_json({
                "type": "speak",
                "text": result["text"],
                "lang": "en-GB",
            })
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/commentary/direct")
async def commentary_direct(websocket: WebSocket):
    """Send a scene_json dict directly (already gated by Phase 2), get back
    one commentary message. No Redis needed -- good for Postman-style
    manual testing."""
    await websocket.accept()
    try:
        while True:
            raw_text = await websocket.receive_text()
            scene_json = json.loads(raw_text)
            result = _engine.generate(scene_json)
            await websocket.send_json({
                "type": "commentary",
                "text": result["text"],
                "style": result["style"],
                "backend_used": result["backend_used"],
            })
    except WebSocketDisconnect:
        pass
