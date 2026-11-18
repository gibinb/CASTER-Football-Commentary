"""
Unit tests for the Phase 1 feature extractor, using synthetic detections
instead of real video. This lets you verify the FEATURE MATH is correct
(velocity, ball gating, contact flag, boundary proximity) before
you've fine-tuned any YOLO weights or labelled any real data.

Run with:
    python -m pytest phase1/tests/test_feature_extractor.py -v
or just:
    python phase1/tests/test_feature_extractor.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from schemas import Detection
from phase1.features.feature_extractor import FeatureExtractor


FRAME_W, FRAME_H = 1280, 720


def make_ball(x, y, conf=0.9):
    return Detection(track_id=1, cls_name="ball", confidence=conf, x=x, y=y, w=10, h=10)


def make_player(track_id, x, y):
    # Generic "player" class (no team separation) -- matches the real
    # 4-class Roboflow fine-tuning dataset. See feature_extractor.py.
    return Detection(track_id=track_id, cls_name="player", confidence=0.9, x=x, y=y, w=40, h=100)


def confirm(ext, x, y, conf=0.9, frames=None):
    """
    Feeds the same ball position for enough consecutive frames to get it
    CONFIRMED (BALL_CONFIRM_FRAMES by default), returning the final vector.
    Tests use this instead of a single .extract() call wherever they need
    a trusted, already-established track to test something else on top of.
    """
    n = frames if frames is not None else ext.BALL_CONFIRM_FRAMES
    vec = None
    for _ in range(n):
        vec = ext.extract([make_ball(x, y, conf)], FRAME_W, FRAME_H)
    return vec


def test_vector_is_always_12_dimensional():
    ext = FeatureExtractor()
    vec = confirm(ext, 640, 360)
    assert len(vec) == 12, f"expected 12 features, got {len(vec)}"
    print("PASS: feature vector is 12-dimensional")


def test_ball_position_is_normalised():
    ext = FeatureExtractor()
    vec = confirm(ext, 640, 360)
    assert abs(vec[0] - 0.5) < 1e-6, "ball_x should normalise to 0.5 at frame centre"
    assert abs(vec[1] - 0.5) < 1e-6, "ball_y should normalise to 0.5 at frame centre"
    print("PASS: ball position normalises correctly")


def test_single_sighting_is_not_yet_trusted():
    """
    A brand-new ball candidate, seen only once, must NOT be reported as
    detected -- this is the entire point of temporal validation. Without
    it, a one-off false positive (a boot, a line marking) would be
    accepted exactly as readily as the real ball.
    """
    ext = FeatureExtractor()
    ext.extract([make_ball(640, 360)], FRAME_W, FRAME_H)
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is False, "a single sighting was trusted without confirmation"
    print("PASS: a single, unconfirmed sighting is not reported as a detected ball")


def test_two_consistent_sightings_confirm_the_ball():
    ext = FeatureExtractor()
    ext.extract([make_ball(640, 360)], FRAME_W, FRAME_H)          # 1st sighting: pending
    vec = ext.extract([make_ball(645, 362)], FRAME_W, FRAME_H)    # 2nd, consistent: confirmed
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is True, "two consistent sightings should confirm the ball"
    assert abs(vec[0] - 645 / FRAME_W) < 1e-6
    print("PASS: two consecutive, consistent sightings confirm the ball")


def test_velocity_is_correct_on_the_confirmation_frame():
    """
    The two positions seen while a candidate was pending must be
    backfilled into history, so velocity on the confirming frame reflects
    real movement rather than reading an empty history as zero speed.
    """
    ext = FeatureExtractor()
    ext.extract([make_ball(100, 360)], FRAME_W, FRAME_H)
    vec = ext.extract([make_ball(130, 360)], FRAME_W, FRAME_H)   # confirms here
    # 30px in one frame out of 1280 width = ~0.023 normalised: a hard but plausible shot.
    assert vec[4] > 0.02, f"expected meaningful speed on the confirming frame, got {vec[4]}"
    print("PASS: velocity is correct on the very frame a ball gets confirmed")


def test_a_higher_confidence_false_positive_does_not_steal_a_one_off_track():
    """
    The exact real-world case this fix targets: on Final.mp4 a false "ball"
    on a boot scored 0.71 confidence in the same frame the real ball scored
    only 0.31. A single far, more-confident candidate must not be able to
    hijack a just-started track; it needs its OWN second consistent sighting.
    """
    ext = FeatureExtractor()
    ext.extract([make_ball(640, 360, conf=0.31)], FRAME_W, FRAME_H)   # real ball, 1st sighting
    fake = make_ball(1100, 100, conf=0.71)                            # false positive, far away, more confident
    vec = ext.extract([fake, make_ball(645, 362, conf=0.30)], FRAME_W, FRAME_H)
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is True
    assert abs(vec[0] - 645 / FRAME_W) < 1e-6, (
        f"the one-off, more-confident false positive was trusted over the consistent real ball, x={vec[0]}"
    )
    print("PASS: a single higher-confidence false positive cannot steal a consistently-seen track")


def test_a_lone_false_positive_never_gets_confirmed_even_if_very_confident():
    ext = FeatureExtractor()
    ext.extract([make_ball(640, 360, conf=0.95)], FRAME_W, FRAME_H)   # one-off, very confident
    # nothing near it next frame -- it was noise
    ext.extract([], FRAME_W, FRAME_H)
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is False, "a one-off false positive got confirmed despite no second sighting"
    print("PASS: a one-off false positive is never confirmed, however confident")


def test_low_confidence_lone_candidate_never_even_becomes_a_pending_hypothesis():
    """The 263s case: a 0.16-confidence 'ball' on a boot must not even start a track."""
    ext = FeatureExtractor()
    ext.extract([make_ball(190, 740, conf=0.16)], FRAME_W, FRAME_H)
    ext.extract([make_ball(195, 742, conf=0.16)], FRAME_W, FRAME_H)   # even if it "repeats"
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is False, "a below-floor-confidence candidate was confirmed"
    print("PASS: a candidate below the confidence floor never becomes a pending hypothesis")


def test_confirmed_track_survives_an_ordinary_confidence_dip():
    """Once trusted, a track continues through an ordinary confidence dip (motion blur, partial occlusion)."""
    ext = FeatureExtractor()
    confirm(ext, 640, 360, conf=0.9)
    vec = ext.extract([make_ball(644, 362, conf=0.22)], FRAME_W, FRAME_H)   # same spot, dipped but above the floor
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is True, "a confirmed track was lost on an ordinary confidence dip"
    assert abs(vec[0] - 644 / FRAME_W) < 1e-6
    print("PASS: a confirmed track survives an ordinary confidence dip (above the continuation floor)")


def test_confirmed_track_does_not_drift_onto_a_very_low_confidence_wrong_object():
    """
    The actual Final.mp4 ~259s case: continuation checked proximity only, so an
    already-confirmed track drifted onto a genuinely wrong 0.19-confidence
    detection near its last position, while the real ball (0.56 confidence)
    sat elsewhere and was never even considered. Continuation must also
    respect a confidence floor, not proximity alone.
    """
    ext = FeatureExtractor()
    confirm(ext, 546 / 1920 * FRAME_W, 260 / 1080 * FRAME_H, conf=0.9)
    wrong = Detection(None, "ball", 0.19, 548, 262, 10, 10)        # same spot, but barely confident -- the bad pick
    real = Detection(None, "ball", 0.56, 1501 / 1920 * FRAME_W, 589 / 1080 * FRAME_H, 10, 10)  # the actual ball, far away
    ext.extract([wrong, real], FRAME_W, FRAME_H)
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is False, (
        "a 0.19-confidence detection was trusted as a continuation purely on proximity -- "
        "this is the exact bug observed in Final.mp4 at ~259s"
    )
    print("PASS: continuation requires a confidence floor, not proximity alone")


def test_confirmed_track_ignores_a_far_false_positive_mid_track():
    ext = FeatureExtractor()
    confirm(ext, 640, 360, conf=0.9)
    far = make_ball(1100, 100, conf=0.99)
    vec = ext.extract([far], FRAME_W, FRAME_H)
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is False, "a far false positive was treated as a continuation of the real track"
    assert abs(vec[0] - 1100 / FRAME_W) > 0.1, "the stale/fallback position should not equal the far false positive"
    print("PASS: a confirmed track is not derailed by a single far-away candidate")


def test_reacquiring_after_a_long_gap_still_needs_two_consistent_sightings():
    """
    This is the actual Semifinal01.mp4 263s bug: previously, after enough
    missed frames, ANY single candidate -- however unreliable -- was
    accepted immediately with no validation. Reacquisition must now go
    through the same two-hit check as a brand-new sighting.
    """
    ext = FeatureExtractor()
    confirm(ext, 640, 360, conf=0.9)
    for _ in range(ext.BALL_CONTINUATION_MAX_MISSING + 2):
        ext.extract([], FRAME_W, FRAME_H)
    # a single confident candidate reappears after the long gap
    vec = ext.extract([make_ball(190, 740, conf=0.80)], FRAME_W, FRAME_H)
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is False, (
        "a single post-gap sighting was trusted immediately -- this is the exact bug that accepted "
        "a 0.16-confidence boot as the ball after a long gap in Semifinal01.mp4 at 263s"
    )
    # but a SECOND consistent sighting right after it should confirm normally
    vec = ext.extract([make_ball(193, 742, conf=0.80)], FRAME_W, FRAME_H)
    diag = ext.get_tracking_diagnostics()
    assert diag["ball_detected"] is True, "a second consistent post-gap sighting should confirm the ball"
    print("PASS: reacquiring after a long gap requires two consistent sightings, same as a fresh track")


def test_boundary_proximity_flag():
    vec_centre = confirm(FeatureExtractor(), 640, 360)
    vec_edge = confirm(FeatureExtractor(), 5, 360)
    assert vec_centre[11] == 0.0, "ball at centre should not be near boundary"
    assert vec_edge[11] == 1.0, "ball near frame edge should trigger boundary flag"
    print("PASS: boundary proximity flag fires correctly")


def test_players_nearby_counts_only_close_players():
    ext = FeatureExtractor()
    near_player = make_player(2, 660, 370)   # very close
    far_player = make_player(3, 50, 50)       # far away
    ext.extract([make_ball(640, 360), near_player, far_player], FRAME_W, FRAME_H)
    vec = ext.extract([make_ball(641, 361), near_player, far_player], FRAME_W, FRAME_H)
    assert vec[6] == 1.0, f"expected exactly 1 nearby player, got {vec[6]}"
    print("PASS: players_nearby only counts players within radius")


def test_football_contact_flag_approximated_from_player_box():
    ext = FeatureExtractor()
    player = make_player(5, 640, 380)
    ext.extract([make_ball(640, 400), player], FRAME_W, FRAME_H)
    vec = ext.extract([make_ball(640, 400), player], FRAME_W, FRAME_H)
    assert vec[10] in (0.0, 1.0), "contact_flag must be a clean binary value"
    print(f"PASS: football contact flag computed without crashing (value={vec[10]})")


def test_window_fills_to_30_frames():
    ext = FeatureExtractor()
    for i in range(35):
        ext.extract([make_ball(640 + i, 360)], FRAME_W, FRAME_H)
    window = ext.get_window()
    assert len(window) == 30, f"window should cap at 30 frames, got {len(window)}"
    assert ext.window_is_full()
    print("PASS: sliding window correctly caps at 30 frames")


def test_shaky_ball_detection_cannot_set_contact_flag():
    """A confirmed-but-low-confidence-this-frame ball beside a player must not count as a touch."""
    ext = FeatureExtractor()
    player = Detection(1, "player", 0.9, 640, 300, 40, 140)   # ball sits at the player's feet
    confirm(ext, 640, 360, conf=0.9)
    sure = ext.extract([make_ball(640, 360, conf=0.9), player], FRAME_W, FRAME_H)
    assert sure[10] == 1.0, "setup problem: confident ball at feet should register contact"
    shaky = ext.extract([make_ball(640, 360, conf=0.16), player], FRAME_W, FRAME_H)
    assert shaky[10] == 0.0, "a 0.16-confidence frame on an already-confirmed track still set the contact flag"
    print("PASS: a low-confidence frame cannot raise the contact flag, even on a confirmed track")


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