"""
Unit tests for the Phase 2 scene state JSON builder, using synthetic
SceneState/EventPrediction objects so this works without real video,
YOLO weights, or a trained classifier. FOOTBALL ONLY.

Run with:
    python -m pytest phase2/tests/test_scene_json_builder.py -v
or just:
    python phase2/tests/test_scene_json_builder.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from schemas import SceneState, EventPrediction
from phase2.scene_json_builder import MatchState, build_scene_json, apply_event_to_match_state
from config import SPORT_FOOTBALL


def make_scene(sport: str = SPORT_FOOTBALL) -> SceneState:
    return SceneState(
        sport=sport, frame_id=100, timestamp_sec=3.3,
        ball_position=[0.8, 0.5], ball_velocity=[0.1, 0.0], ball_speed=0.35,
        pitch_zone="penalty_box",
        boundary_proximity=True, official_signal=False, contact_flag=True,
        nearest_player_id=7, nearest_player_dist=0.05, players_nearby=1,
        feature_vector=[0.0] * 12, detections=[],
    )


def test_football_json_has_football_specific_fields():
    scene = make_scene(SPORT_FOOTBALL)
    prediction = EventPrediction(sport=SPORT_FOOTBALL, event="GOAL", confidence=0.94, scene=scene)
    match_state = MatchState(sport=SPORT_FOOTBALL, home_score=1, away_score=0, match_minute=67)

    j = build_scene_json(prediction, match_state)

    assert j["sport"] == "football"
    assert j["event"] == "GOAL"
    assert j["match_minute"] == 67
    assert j["score"] == "1 - 0"
    print("PASS: football scene JSON contains match_minute/score")


def test_json_never_invents_fields_not_in_match_state_or_scene():
    scene = make_scene(SPORT_FOOTBALL)
    prediction = EventPrediction(sport=SPORT_FOOTBALL, event="PASS", confidence=0.8, scene=scene)
    match_state = MatchState(sport=SPORT_FOOTBALL)  # untouched defaults

    j = build_scene_json(prediction, match_state)

    assert j["score"] == "0 - 0", "score should reflect actual (default) match_state, not a guess"
    assert j["match_minute"] == 0, "match_minute should be 0 when not explicitly set, never invented"
    print("PASS: scene JSON reflects only explicitly-known match state, never invents values")


def test_unsupported_sport_raises():
    scene = make_scene(SPORT_FOOTBALL)
    prediction = EventPrediction(sport="cricket", event="FOUR", confidence=0.9, scene=scene)
    match_state = MatchState()
    try:
        build_scene_json(prediction, match_state)
        raised = False
    except ValueError:
        raised = True
    assert raised, "build_scene_json must reject any sport other than football"
    print("PASS: build_scene_json rejects non-football sport")


def test_recent_events_buffer_caps_at_max_history():
    match_state = MatchState(sport=SPORT_FOOTBALL)
    for ev in ["PASS", "PASS", "DRIBBLE", "FOUL", "PASS", "SHOT_ON_TARGET", "GOAL"]:
        apply_event_to_match_state(match_state, SPORT_FOOTBALL, ev)
    assert len(match_state.recent_events) == 5, "recent_events should cap at 5 by default"
    assert match_state.recent_events[-1] == "goal"
    print("PASS: recent_events rolling buffer caps correctly and keeps most recent last")


def test_event_counts_increment_per_event():
    match_state = MatchState(sport=SPORT_FOOTBALL)
    apply_event_to_match_state(match_state, SPORT_FOOTBALL, "PASS")
    apply_event_to_match_state(match_state, SPORT_FOOTBALL, "PASS")
    apply_event_to_match_state(match_state, SPORT_FOOTBALL, "GOAL")
    assert match_state.event_counts["PASS"] == 2
    assert match_state.event_counts["GOAL"] == 1
    print("PASS: event_counts tracks confirmed events per clip, not invented")


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
