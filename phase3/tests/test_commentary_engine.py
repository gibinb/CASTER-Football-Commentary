"""
CASTER Phase 3 -- unit tests for the commentary engine, using synthetic
scene_json dicts (matching phase2/scene_json_builder.py's real output
shape) instead of a real trained classifier or fine-tuned model. This
verifies the ENGINE LOGIC is correct (backend fallback chain, word-limit
enforcement) independently of model quality. FOOTBALL ONLY.

Run with:
    python -m pytest phase3/tests/test_commentary_engine.py -v
or just:
    python phase3/tests/test_commentary_engine.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import SPORT_FOOTBALL
from phase3.inference.commentary_engine import CommentaryEngine


def make_football_scene(event="GOAL", confidence=0.9, style="dramatic"):
    return {
        "sport": SPORT_FOOTBALL,
        "event": event,
        "confidence": confidence,
        "pitch_zone": "penalty_box",
        "recent_events": ["pass", "dribble"],
        "commentary_style": style,
        "language": "en",
        "score_tracked": True,
        "event_number_in_clip": 1,
        "match_minute": 67,
        "score": "1 - 0",
        "ball_speed": 0.4,
        "boundary_proximity": True,
        "contact_flag": True,
    }


def test_engine_never_returns_empty_text():
    engine = CommentaryEngine(backend_priority=["local_gguf", "groq", "template"])
    result = engine.generate(make_football_scene())
    assert result["text"], "engine returned empty text"
    print("PASS: CommentaryEngine.generate always returns non-empty text")


def test_engine_falls_back_to_template_with_no_backends_configured():
    engine = CommentaryEngine(backend_priority=["local_gguf", "groq", "template"])
    result = engine.generate(make_football_scene())
    assert result["backend_used"] == "template", (
        f"expected template fallback in this sandbox (no GGUF model, no GROQ_API_KEY), "
        f"got {result['backend_used']}"
    )
    print("PASS: engine falls back to 'template' backend when others unavailable")


def test_engine_preserves_scene_metadata():
    scene = make_football_scene(event="TACKLE", style="statistical")
    engine = CommentaryEngine()
    result = engine.generate(scene)
    assert result["sport"] == SPORT_FOOTBALL
    assert result["event"] == "TACKLE"
    assert result["style"] == "statistical"
    print("PASS: CommentaryEngine preserves sport/event/style from scene_json")


def test_engine_respects_max_word_count():
    engine = CommentaryEngine()
    result = engine.generate(make_football_scene())
    word_count = len(result["text"].split())
    assert word_count <= 30, f"commentary exceeded 30 words: {word_count}"
    print(f"PASS: commentary stays within max word limit (got {word_count} words)")


def test_engine_works_for_football():
    engine = CommentaryEngine()
    result = engine.generate(make_football_scene(event="GOAL"))
    assert result["text"]
    assert result["sport"] == SPORT_FOOTBALL
    print("PASS: engine produces commentary for football scenes")


def test_engine_handles_all_football_events():
    from config import FOOTBALL_EVENTS
    engine = CommentaryEngine()
    for event in FOOTBALL_EVENTS:
        result = engine.generate(make_football_scene(event=event))
        assert result["text"], f"empty commentary for football event {event}"
    print(f"PASS: engine handles all {len(FOOTBALL_EVENTS)} football event types from config.py "
          f"(including NO_EVENT, even though confidence_gate.py never actually lets it reach here)")


def test_engine_handles_unknown_event_gracefully():
    engine = CommentaryEngine()
    scene = make_football_scene(event="SOME_FUTURE_EVENT_NOT_YET_ADDED")
    result = engine.generate(scene)
    assert result["text"], "engine should never crash or return empty on an unknown event"
    print("PASS: engine degrades gracefully on an event name it doesn't recognise")


def test_engine_varies_template_output_across_repeated_events():
    """Regression test for the real repetition bug found on the 51-event
    'ss' run. Forces the template backend (deterministic, no network) and
    fires the SAME event 10 times -- with a 6-7 phrase pool per
    event/style and the engine's recent_lines memory, output should not
    collapse to one repeated phrase the way the bug did."""
    engine = CommentaryEngine(backend_priority=["template"])
    texts = [engine.generate(make_football_scene(event="DRIBBLE"))["text"] for _ in range(10)]
    distinct = len(set(texts))
    assert distinct >= 4, (
        f"only {distinct} distinct lines across 10 same-type events (pool-aware "
        f"template selection should avoid collapsing like the real Groq bug did): {texts}"
    )
    print(f"PASS: {distinct}/10 distinct lines across repeated same-type events (template backend)")


def test_engine_tracks_recent_lines_per_event_type_separately():
    """DRIBBLE's history should not affect PASS's candidate pool."""
    engine = CommentaryEngine(backend_priority=["template"])
    for _ in range(3):
        engine.generate(make_football_scene(event="DRIBBLE"))
    assert list(engine._recent_by_event["PASS"]) == []
    assert len(engine._recent_by_event["DRIBBLE"]) == 3
    print("PASS: recent-lines memory is tracked separately per event type")


def test_engine_rejects_non_football_sport():
    engine = CommentaryEngine()
    scene = make_football_scene()
    scene["sport"] = "cricket"
    try:
        engine.generate(scene)
        raised = False
    except ValueError:
        raised = True
    assert raised, "engine should reject any non-football sport (CASTER is football-only)"
    print("PASS: engine raises ValueError for a non-football sport")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
