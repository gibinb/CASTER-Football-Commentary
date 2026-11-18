"""
CASTER Phase 4 -- Main FastAPI application.

This is the SINGLE entry-point server that wires together the entire
CASTER pipeline into one running process:

    POST /api/upload         -- receive a video file, kick off Phase 1+2
    WS   /api/ws/stream      -- stream Phase 1 bounding-box detections
                                + Phase 2 events + Phase 3 commentary to
                                the React frontend all on one socket
    GET  /api/health         -- system status and model availability check
    GET  /api/transcript/{clip}  -- fetch saved commentary JSON from Phase 3

Architecture vs Phase 3's standalone backend/main.py
------------------------------------------------------
Phase 3's server only consumes pre-built Redis events and streams
commentary.  Phase 4's server RUNS the whole pipeline from raw video:
   video file  ->  Phase 1 (YOLO+ByteTrack+features)
               ->  Phase 2 (XGBoost classifier + gate + scene JSON)
               ->  Phase 3 (CommentaryEngine)
               ->  WebSocket stream  ->  React frontend

The Redis bridge from Phase 3 is still used internally (the Phase 1/2
CV worker pushes onto it; the Phase 3 LLM worker pops), so neither
worker blocks the other and the WebSocket can forward both bounding-box
frames AND commentary tokens as soon as each is ready.

Run with:
    uvicorn phase4.backend.main:app --reload --port 8000

then open http://localhost:5173 (Vite dev server).
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import OUTPUTS_DIR, ROOT_DIR
from phase4.backend.ws_handler import router as ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("caster.phase4")

app = FastAPI(
    title="CASTER â€” Computer Vision Assisted Sports Text And Event Reporter",
    description="Full pipeline: video upload â†’ CV â†’ event classification â†’ commentary â†’ WebSocket",
    version="4.0.0",
)

# Allow Vite dev server (localhost:5173) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)

# BUG FIX (detector pre-warmed twice, once per instance, at startup):
# this used to construct its OWN `PipelineRunner()` here -- a second,
# completely separate instance from the one ws_handler.py also
# constructs at its own module level. Only ws_handler.py's instance is
# ever actually used (the WebSocket route calls .run() on it); this
# module's copy was created, warmed up, and then never touched again
# for the rest of the process's life. Harmless before the background
# detector warm-up was added (an idle, empty object costs nothing), but
# once warm-up was added to __init__, it meant a second, wasted GPU
# warm-up (and the confusing duplicate "Detector pre-warmed..." log
# line at identical timestamps). Fixed by importing the SAME instance
# ws_handler.py already created, instead of making a new one.
from phase4.backend.ws_handler import _runner


@app.get("/api/health")
def health():
    """
    Returns current backend status and which Phase 3 backends are reachable.
    The React frontend polls this on load to show a status banner.
    """
    from phase3.inference.commentary_engine import CommentaryEngine
    engine = CommentaryEngine()

    phase1_ready = (ROOT_DIR / "phase1" / "weights").exists()
    phase2_ready = (ROOT_DIR / "phase2" / "saved_models").exists()

    return {
        "status": "ok",
        "phase1_weights_dir_exists": phase1_ready,
        "phase2_models_dir_exists": phase2_ready,
        "commentary_backends": engine.backend_priority,
        "outputs_dir": str(OUTPUTS_DIR),
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Accept a video file, save it to data/raw/, and register it
    for processing. The actual CV+commentary work starts when the
    frontend opens /api/ws/stream?clip=<clip_name>.

    Returns: { clip_name, sport (if already detected), status }
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    upload_dir = ROOT_DIR / "data" / "raw"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    contents = await file.read()
    dest.write_bytes(contents)
    logger.info("Uploaded: %s (%d bytes)", file.filename, len(contents))

    # Derive clip_name the same way run_phase1.py does (stem, no extension)
    clip_name = Path(file.filename).stem

    return JSONResponse({
        "clip_name": clip_name,
        "filename": file.filename,
        "size_bytes": len(contents),
        "status": "uploaded",
        "message": "Open WebSocket /api/ws/stream?clip=" + clip_name + " to begin processing",
    })


@app.get("/api/transcript/{clip_name}")
def get_transcript(clip_name: str):
    """
    Returns the saved commentary JSON produced by Phase 3 for a clip.
    The React TranscriptLog component fetches this after a session
    to let the user copy or review the full commentary.
    """
    path = OUTPUTS_DIR / f"{clip_name}_commentary.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for '{clip_name}'. Run the pipeline first.",
        )
    with open(path) as f:
        return JSONResponse(json.load(f))


@app.get("/api/clips")
def list_clips():
    """Returns all clip names that have Phase 3 commentary output saved."""
    clips = [
        p.stem.replace("_commentary", "")
        for p in OUTPUTS_DIR.glob("*_commentary.json")
    ]
    return {"clips": sorted(clips)}