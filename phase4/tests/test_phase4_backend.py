"""
CASTER Phase 4 -- backend unit and integration tests. FOOTBALL ONLY.

Tests cover:
  - FastAPI health endpoint
  - Video upload endpoint (synthetic file)
  - Transcript fetch endpoint (after synthetic Phase 3 output)
  - PipelineRunner's offline replay path (no YOLO needed)
  - WebSocket replay endpoint

Run with:
    python phase4/tests/test_phase4_backend.py

Or via pytest:
    python -m pytest phase4/tests/test_phase4_backend.py -v

All tests use TestClient (synchronous HTTPX) or mock data so they
work without a running server, without YOLO weights, and without a
trained XGBoost model.

FOOTBALL-ONLY NOTE (fixed 2026-11): this file previously built scene JSON
with sport="cricket" and constructed MatchState(sport="cricket",
score="83/2", over="13.5", ...) -- MatchState has no `score`/`over`
fields (it has home_score/away_score/match_minute, see
phase2/scene_json_builder.py), so test_phase4_imports_phase2_scene_builder
would have raised TypeError the moment it actually ran. Rewritten
football-only throughout, matching the real MatchState shape.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


# ---- helpers ----

def make_scene_json(event="GOAL", confidence=0.91, style="dramatic"):
    """Produces the exact shape phase2/scene_json_builder.py outputs."""
    return {
        "sport": "football", "event": event, "confidence": confidence,
        "pitch_zone": "penalty_box", "recent_events": ["pass"],
        "commentary_style": style, "language": "en",
        "ball_speed": 0.35, "boundary_proximity": True, "contact_flag": True,
        "score": "1 - 0", "match_minute": 67,
    }


# ============================================================
# Tests
# ============================================================

def test_health_endpoint():
    """GET /api/health returns 200 with status ok."""
    from fastapi.testclient import TestClient
    from phase4.backend.main import app
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "commentary_backends" in data
    print("PASS: GET /api/health returns ok")


def test_upload_endpoint_creates_file(tmp_path):
    """POST /api/upload saves the file and returns clip_name."""
    from fastapi.testclient import TestClient
    import phase4.backend.main as m
    import io

    # Patch ROOT_DIR inside the already-imported main module so the
    # upload handler uses tmp_path instead of the real project root.
    import config
    original_root = config.ROOT_DIR
    (tmp_path / "data" / "raw").mkdir(parents=True)
    config.ROOT_DIR = tmp_path

    # Re-import to pick up patched value
    import importlib
    importlib.reload(m)

    try:
        client = TestClient(m.app)
        fake_video = io.BytesIO(b"fake video content")
        resp = client.post(
            "/api/upload",
            files={"file": ("test_clip.mp4", fake_video, "video/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["clip_name"] == "test_clip"
        assert data["status"] == "uploaded"
        saved = tmp_path / "data" / "raw" / "test_clip.mp4"
        assert saved.exists(), f"Expected file at {saved}"
        print("PASS: POST /api/upload saves file and returns clip_name")
    finally:
        config.ROOT_DIR = original_root
        importlib.reload(m)


def test_transcript_endpoint_404_when_missing():
    """GET /api/transcript/<name> returns 404 when no commentary file exists."""
    from fastapi.testclient import TestClient
    from phase4.backend.main import app
    client = TestClient(app)
    resp = client.get("/api/transcript/__nonexistent_clip__")
    assert resp.status_code == 404
    print("PASS: GET /api/transcript returns 404 for unknown clip")


def test_transcript_endpoint_returns_json(tmp_path):
    """GET /api/transcript/<name> returns saved commentary JSON."""
    from fastapi.testclient import TestClient
    import phase4.backend.main as m
    import importlib
    import config

    original_outputs = config.OUTPUTS_DIR
    config.OUTPUTS_DIR = tmp_path
    importlib.reload(m)

    fake_transcript = [{"event": "GOAL", "sport": "football", "commentary": "GOAL!"}]
    (tmp_path / "myclip_commentary.json").write_text(json.dumps(fake_transcript))

    try:
        client = TestClient(m.app)
        resp = client.get("/api/transcript/myclip")
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["event"] == "GOAL"
        print("PASS: GET /api/transcript returns saved commentary JSON")
    finally:
        config.OUTPUTS_DIR = original_outputs
        importlib.reload(m)


def test_pipeline_runner_offline_replay(tmp_path):
    """
    PipelineRunner._run_offline_phase2_phase3 replays a saved
    *_scene_json_events.json through Phase 3's CommentaryEngine and
    yields commentary messages.
    """
    import asyncio
    import config
    import importlib

    original_outputs = config.OUTPUTS_DIR
    config.OUTPUTS_DIR = tmp_path

    # Write synthetic Phase 2 output (exactly what run_phase2.py produces)
    events = [
        make_scene_json(event="GOAL"),
        make_scene_json(event="SHOT_ON_TARGET"),
    ]
    (tmp_path / "synth_clip_scene_json_events.json").write_text(json.dumps(events))

    try:
        import phase4.backend.pipeline_runner as pr_module
        importlib.reload(pr_module)
        PipelineRunner = pr_module.PipelineRunner

        async def collect():
            runner = PipelineRunner()
            msgs = []
            async for msg in runner._run_offline_phase2_phase3("synth_clip", "dramatic"):
                msgs.append(msg)
            return msgs

        msgs = asyncio.run(collect())
        types = [m["type"] for m in msgs]

        assert "event" in types,         "expected at least one 'event' message"
        assert "sentence_end" in types,  "expected at least one 'sentence_end' message"
        assert "speak" in types,         "expected at least one 'speak' message"
        assert "done" in types,          "expected 'done' message at the end"

        done_msg = next(m for m in msgs if m["type"] == "done")
        assert done_msg["total_events"] == 2

        print(f"PASS: PipelineRunner offline replay yields {len(msgs)} messages across 2 events")
    finally:
        config.OUTPUTS_DIR = original_outputs
        importlib.reload(pr_module)


def test_ws_replay_endpoint(tmp_path):
    """
    WS /api/ws/replay streams all commentary entries from a saved
    *_commentary.json file and finishes with a 'done' message.
    """
    from fastapi.testclient import TestClient
    from phase4.backend.main import app
    import config

    original_outputs = config.OUTPUTS_DIR
    config.OUTPUTS_DIR = tmp_path

    fake_commentary = [
        {"event": "GOAL", "sport": "football", "commentary": "GOAL!", "confidence": 0.9,
         "time": "67", "score": "1 - 0", "style": "dramatic", "backend_used": "template"},
        {"event": "FOUL", "sport": "football", "commentary": "That's a foul!", "confidence": 0.8,
         "time": "70", "score": "1 - 0", "style": "dramatic", "backend_used": "template"},
    ]
    (tmp_path / "myclip_commentary.json").write_text(json.dumps(fake_commentary))

    try:
        client = TestClient(app)
        with client.websocket_connect("/api/ws/replay?clip=myclip&style=dramatic") as ws:
            received = []
            for _ in range(50):  # collect up to 50 messages
                try:
                    msg = ws.receive_json()
                    received.append(msg)
                    if msg["type"] == "done":
                        break
                except Exception:
                    break

        types = [m["type"] for m in received]
        assert "sport_detected" in types, "should emit sport_detected first"
        assert "event" in types
        assert "token" in types
        assert "sentence_end" in types
        assert "speak" in types
        assert "done" in types

        done_msg = next(m for m in received if m["type"] == "done")
        assert done_msg["total_events"] == 2
        print(f"PASS: WS /api/ws/replay streamed {len(received)} messages correctly")
    finally:
        config.OUTPUTS_DIR = original_outputs


def test_phase4_imports_phase3_engine():
    """Phase 4 can import Phase 3's CommentaryEngine and call generate()."""
    from phase3.inference.commentary_engine import CommentaryEngine
    engine = CommentaryEngine()
    result = engine.generate(make_scene_json("GOAL", style="casual"))
    assert result["text"]
    assert result["event"] == "GOAL"
    print(f"PASS: Phase 4 -> Phase 3 -> commentary: '{result['text']}' (backend: {result['backend_used']})")


def test_phase4_imports_phase2_scene_builder():
    """Phase 4 can import Phase 2's scene_json_builder and build a scene JSON."""
    from phase2.scene_json_builder import build_scene_json, MatchState, apply_event_to_match_state
    from schemas import EventPrediction, SceneState

    scene = SceneState(
        sport="football", frame_id=100, timestamp_sec=3.33,
        ball_position=[0.5, 0.5], ball_velocity=[0.02, -0.01],
        ball_speed=0.022, pitch_zone="midfield",
        boundary_proximity=False, official_signal=False, contact_flag=True,
        nearest_player_id=3, nearest_player_dist=0.12, players_nearby=2,
        feature_vector=[0.5, 0.5, 0.02, -0.01, 0.022, 0.12, 2, 2, 0, 0.5, 1, 0],
        detections=[],
    )
    prediction = EventPrediction(sport="football", event="GOAL", confidence=0.88, scene=scene)
    match_state = MatchState(sport="football", home_score=0, away_score=0,
                              match_minute=66, commentary_style="dramatic")
    apply_event_to_match_state(match_state, "football", "GOAL")

    scene_json = build_scene_json(prediction, match_state, clip_time_sec=scene.timestamp_sec)
    assert scene_json["event"] == "GOAL"
    print(f"PASS: Phase 4 -> Phase 2 scene JSON builder: score={scene_json['score']}")


def run_all():
    import tempfile
    tests = [
        (test_health_endpoint, {}),
        (test_upload_endpoint_creates_file, {"tmp_path": Path(tempfile.mkdtemp())}),
        (test_transcript_endpoint_404_when_missing, {}),
        (test_transcript_endpoint_returns_json, {"tmp_path": Path(tempfile.mkdtemp())}),
        (test_pipeline_runner_offline_replay, {"tmp_path": Path(tempfile.mkdtemp())}),
        (test_ws_replay_endpoint, {"tmp_path": Path(tempfile.mkdtemp())}),
        (test_phase4_imports_phase3_engine, {}),
        (test_phase4_imports_phase2_scene_builder, {}),
    ]

    passed = 0
    failed = 0
    for fn, kwargs in tests:
        try:
            fn(**kwargs)
            passed += 1
        except Exception as e:
            print(f"FAIL: {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all()