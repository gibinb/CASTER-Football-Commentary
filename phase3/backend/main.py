"""
CASTER Phase 3 -- FastAPI app entry point.

Run with:
    uvicorn phase3.backend.main:app --reload --port 8001

Note: this server does NOT run the CV pipeline itself (that's Phase 1/2,
served from Phase 4's main FastAPI app per the deep-dive's architecture).
It only consumes scene_json events pushed onto Redis by that pipeline and
turns them into commentary. In a single-process deployment, Phase 4's
main.py can import CommentaryEngine directly instead of going through
Redis -- Redis decoupling matters most once you're running real-time
video at frame rate.
"""
import sys
from pathlib import Path

from fastapi import FastAPI

sys.path.append(str(Path(__file__).resolve().parents[2]))
from phase3.backend.websocket_stream import router as commentary_router
from phase3.inference.commentary_engine import CommentaryEngine

app = FastAPI(title="CASTER Phase 3 -- Commentary Engine")
app.include_router(commentary_router)

_engine = CommentaryEngine()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backends_configured": _engine.backend_priority,
    }
