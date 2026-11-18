"""
Scene state builder for CASTER.

This is the final stage of Phase 1 and the most important handoff point
in the whole project: it assembles everything the CV pipeline has
confirmed into one structured SceneState object (see schemas.py).

Phase 2's event classifier reads a sequence of these (and the matching
FeatureWindow) -- it never sees raw video or raw bounding boxes. Phase 3's
language model later only ever sees a JSON derived from this -- never the
video either. This structured grounding is the entire reason CASTER
avoids the hallucination problem that naive LLM-sports-commentary demos
have.
"""

from typing import List, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import FOOTBALL_ZONES, SPORT_FOOTBALL
from schemas import SceneState, Detection


def build_scene_state(
    frame_id: int,
    timestamp_sec: float,
    detections: List[Detection],
    feature_vector: List[float],
    frame_w: int = None,
    frame_h: int = None,
    tracking_diagnostics: Optional[dict] = None,
) -> SceneState:
    """
    Combines raw detections + the 12-number feature vector for this
    frame into a SceneState. Every field here is read directly off the
    feature vector or detections -- nothing is invented.

    frame_w/frame_h are optional but recommended: without them,
    nearest_player_id falls back to "first detected player" rather than
    the actual nearest one, since pixel coordinates can't be compared
    against the normalised ball position without them.

    tracking_diagnostics is the dict returned by
    FeatureExtractor.get_tracking_diagnostics() for THIS frame (keys:
    ball_detected, ball_track_id, ball_confidence, ball_candidate_count,
    ball_missing_frames -- see schemas.SceneState). Passing None leaves
    those fields at their dataclass defaults ("no ball seen this frame"),
    which keeps this function backward-compatible with any caller that
    doesn't have an extractor handy.
    """
    zones = FOOTBALL_ZONES

    ball_x, ball_y = feature_vector[0], feature_vector[1]
    ball_dx, ball_dy = feature_vector[2], feature_vector[3]
    ball_speed = feature_vector[4]
    nearest_player_dist = feature_vector[5]
    players_nearby = int(feature_vector[6])
    zone_idx = int(feature_vector[7])
    official_signal = bool(feature_vector[8])
    contact_flag = bool(feature_vector[10])
    boundary_proximity = bool(feature_vector[11])

    zone_name = zones[zone_idx] if 0 <= zone_idx < len(zones) else "unknown"

    nearest_player_id = _find_nearest_player_id(detections, ball_x, ball_y, frame_w, frame_h)

    diag = tracking_diagnostics or {}

    return SceneState(
        sport=SPORT_FOOTBALL,
        frame_id=frame_id,
        timestamp_sec=timestamp_sec,
        ball_position=[round(ball_x, 4), round(ball_y, 4)],
        ball_velocity=[round(ball_dx, 4), round(ball_dy, 4)],
        ball_speed=round(ball_speed, 4),
        pitch_zone=zone_name,
        boundary_proximity=boundary_proximity,
        official_signal=official_signal,
        contact_flag=contact_flag,
        nearest_player_id=nearest_player_id,
        nearest_player_dist=round(nearest_player_dist, 4) if nearest_player_dist is not None else None,
        players_nearby=players_nearby,
        feature_vector=[round(v, 4) for v in feature_vector],
        detections=detections,
        ball_detected=diag.get("ball_detected", False),
        ball_track_id=diag.get("ball_track_id"),
        ball_confidence=diag.get("ball_confidence"),
        ball_candidate_count=diag.get("ball_candidate_count", 0),
        ball_missing_frames=diag.get("ball_missing_frames", 0),
    )


def _find_nearest_player_id(
    detections: List[Detection],
    ball_x: float,
    ball_y: float,
    frame_w: int = None,
    frame_h: int = None,
) -> Optional[int]:
    """
    Finds which tracked player ID is actually nearest the ball, for the
    human-readable JSON record. ball_x/ball_y are normalised (0-1);
    detections carry raw pixel coordinates, so frame_w/frame_h are
    needed to normalise them for a fair comparison.

    If frame_w/frame_h aren't supplied, falls back to returning the
    first detected player rather than guessing -- this fallback is
    intentionally conservative rather than silently wrong.
    """
    # The fine-tuning dataset uses a generic "player" class (no team
    # separation) plus "goalkeeper" -- see feature_extractor.py's _find_players().
    player_classes = ("player", "goalkeeper")

    players = [d for d in detections if d.cls_name in player_classes and d.track_id is not None]
    if not players:
        return None

    if frame_w is None or frame_h is None:
        # Can't normalise pixel coords without frame dimensions -- the
        # authoritative nearest-player DISTANCE still lives correctly in
        # feature_vector[5] (computed in feature_extractor.py); this is
        # only the human-readable ID label in the JSON.
        return players[0].track_id

    import math
    best_dist = float("inf")
    best_id = players[0].track_id
    for p in players:
        px, py = p.x / frame_w, p.y / frame_h
        dist = math.sqrt((ball_x - px) ** 2 + (ball_y - py) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_id = p.track_id
    return best_id