"""
CASTER Phase 3 -- unit tests for prompt_builder.py, and an end-to-end
test that writes a synthetic *_scene_json_events.json file (matching
what run_phase2.py actually produces) and runs run_phase3.py against it,
verifying the full offline pipeline works without needing real video,
YOLO weights, or a trained classifier. FOOTBALL ONLY.

Run with:
    python -m pytest phase3/tests/test_prompt_and_pipeline.py -v
or just:
    python phase3/tests/test_prompt_and_pipeline.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import SPORT_FOOTBALL, OUTPUTS_DIR
from phase3.inference.prompt_builder import build_user_prompt, time_field, SYSTEM_PROMPT
from phase3.run_phase3 import run as run_phase3


def make_football_scene(event="GOAL"):
    return {
        "sport": SPORT_FOOTBALL, "event": event, "confidence": 0.88,
        "pitch_zone": "penalty_box", "recent_events": ["pass", "dribble"],
        "commentary_style": "statistical", "language": "en",
        "score_tracked": True, "event_number_in_clip": 1,
        "match_minute": 67, "score": "1 - 0",
        "ball_speed": 0.4, "boundary_proximity": True, "contact_flag": True,
    }


def test_time_field_uses_match_minute_for_football():
    assert time_field(make_football_scene()) == "67"
    print("PASS: time_field() returns 'match_minute' for football")


def test_prompt_embeds_event_and_style_instruction():
    scene = make_football_scene()
    prompt = build_user_prompt(scene)
    assert "GOAL" in prompt
    assert "match_minute" in prompt  # raw JSON field name should appear
    print("PASS: build_user_prompt embeds event data and style instructions")


def test_system_prompt_forbids_invention():
    assert "invent" in SYSTEM_PROMPT.lower()
    print("PASS: SYSTEM_PROMPT explicitly forbids inventing facts")


def test_recent_lines_shown_as_avoid_list_in_prompt():
    """Regression test for a real quality bug found against a live 51-event
    Groq run on clip 'ss': the same event type (e.g. DRIBBLE, 20+ times)
    produced near-identical commentary ("What a slick dribble!" repeated
    verbatim 5+ times), because the prompt gave the model no signal that
    prior lines existed at all."""
    scene = make_football_scene(event="DRIBBLE")
    prompt_no_history = build_user_prompt(scene)
    assert "already used" not in prompt_no_history

    recent = ["He beats one, two -- what skill on the ball!", "Brilliant footwork!"]
    prompt_with_history = build_user_prompt(scene, recent_lines=recent)
    assert "already used" in prompt_with_history
    for line in recent:
        assert line in prompt_with_history
    print("PASS: recent_lines are shown to the LLM as an explicit avoid-list")


def test_clip_time_hidden_from_non_statistical_styles():
    """Regression test for a real bug found against a live Groq run: once
    clip_time started being populated by Phase 2, 'dramatic' style output
    began narrating it verbatim ("A crisp pass at 0:11!"). clip_time must
    only reach the prompt for the 'statistical' style."""
    scene = make_football_scene(event="PASS")
    scene["clip_time"] = "0:11"

    scene["commentary_style"] = "dramatic"
    assert "0:11" not in build_user_prompt(scene), "clip_time leaked into a dramatic-style prompt"

    scene["commentary_style"] = "casual"
    assert "0:11" not in build_user_prompt(scene), "clip_time leaked into a casual-style prompt"

    scene["commentary_style"] = "statistical"
    assert "0:11" in build_user_prompt(scene), "clip_time missing from a statistical-style prompt"

    print("PASS: clip_time only reaches the LLM prompt for the 'statistical' style")


def test_end_to_end_pipeline_against_synthetic_phase2_output():
    """
    Writes a synthetic *_scene_json_events.json file in EXACTLY the shape
    run_phase2.py produces, then runs Phase 3's run_phase3.run() against
    it -- the same integration test as running:
        python phase3/run_phase3.py --clip _test_synthetic_clip
    """
    clip_name = "_test_synthetic_clip"
    events = [make_football_scene(event="GOAL"), make_football_scene(event="TACKLE")]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    events_path = OUTPUTS_DIR / f"{clip_name}_scene_json_events.json"
    with open(events_path, "w") as f:
        json.dump(events, f, default=str)

    results = run_phase3(clip_name)

    assert len(results) == 2, f"expected 2 commentary lines, got {len(results)}"
    assert all(r["commentary"] for r in results), "found an empty commentary line"
    assert results[0]["sport"] == SPORT_FOOTBALL
    assert results[1]["sport"] == SPORT_FOOTBALL

    commentary_path = OUTPUTS_DIR / f"{clip_name}_commentary.json"
    assert commentary_path.exists(), "run_phase3 did not write its output file"

    # cleanup
    events_path.unlink(missing_ok=True)
    commentary_path.unlink(missing_ok=True)

    print("PASS: end-to-end run_phase3() works against a synthetic Phase 2 output file")


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
