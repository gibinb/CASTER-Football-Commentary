"""
Integration test: verifies SceneState construction end-to-end using
synthetic detections, without needing real video or YOLO weights.
Confirms the JSON Phase 2 will receive is well-formed.

Run with:
    python phase1/tests/test_scene_state.py
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from schemas import Detection
from phase1.features.feature_extractor import FeatureExtractor
from phase1.scene_state.scene_state_builder import build_scene_state
from config import FOOTBALL_ZONES, SPORT_FOOTBALL


FRAME_W, FRAME_H = 1280, 720


def test_scene_state_round_trips_to_json():
    ext = FeatureExtractor()
    ball = Detection(1, "ball", 0.9, 1100, 360, 10, 10)
    gk = Detection(2, "goalkeeper", 0.85, 1150, 360, 40, 100)

    vec = ext.extract([ball, gk], FRAME_W, FRAME_H)
    scene = build_scene_state(frame_id=10, timestamp_sec=0.33,
                              detections=[ball, gk], feature_vector=vec)

    payload = json.loads(scene.to_json())
    assert payload["sport"] == SPORT_FOOTBALL
    assert payload["pitch_zone"] in FOOTBALL_ZONES
    assert len(payload["feature_vector"]) == 12
    print("PASS: scene state serialises to valid JSON with the football zone vocabulary")


def test_scene_state_never_invents_data_with_no_detections():
    ext = FeatureExtractor()
    vec = ext.extract([], FRAME_W, FRAME_H)  # no ball, no players detected
    scene = build_scene_state(frame_id=0, timestamp_sec=0.0,
                              detections=[], feature_vector=vec,
                              tracking_diagnostics=ext.get_tracking_diagnostics())
    assert scene.nearest_player_id is None
    assert scene.players_nearby == 0
    assert scene.contact_flag is False
    assert scene.ball_detected is False
    print("PASS: scene state stays empty/false rather than guessing when nothing is detected")


def test_nearest_player_id_is_actually_nearest_not_just_first():
    """
    Regression test: build_scene_state must pick the player closest to
    the ball, not simply the first player in the detections list.
    """
    ext = FeatureExtractor()
    ball = Detection(1, "ball", 0.9, 100, 100, 10, 10)
    far_player = Detection(2, "player", 0.85, 1200, 700, 40, 100)   # listed FIRST, but far
    near_player = Detection(3, "player", 0.85, 110, 105, 40, 100)   # listed SECOND, but near

    vec = ext.extract([ball, far_player, near_player], FRAME_W, FRAME_H)
    scene = build_scene_state(frame_id=0, timestamp_sec=0.0,
                              detections=[ball, far_player, near_player],
                              feature_vector=vec, frame_w=FRAME_W, frame_h=FRAME_H)

    assert scene.nearest_player_id == 3, (
        f"expected nearest_player_id=3 (the actually-closer player), got {scene.nearest_player_id}"
    )
    print("PASS: nearest_player_id correctly picks the closest player, not just the first one")


def test_referee_sets_official_signal_and_is_not_counted_as_a_player():
    ext = FeatureExtractor()
    ball = Detection(1, "ball", 0.9, 640, 360, 10, 10)
    ref = Detection(2, "referee", 0.9, 650, 365, 40, 100)       # right next to the ball
    vec = ext.extract([ball, ref], FRAME_W, FRAME_H)
    scene = build_scene_state(frame_id=0, timestamp_sec=0.0, detections=[ball, ref],
                              feature_vector=vec, frame_w=FRAME_W, frame_h=FRAME_H)
    assert scene.official_signal is True
    assert scene.players_nearby == 0, "a referee must not count as a player near the ball"
    assert scene.nearest_player_id is None
    print("PASS: referee raises official_signal and is not counted as a player")


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
