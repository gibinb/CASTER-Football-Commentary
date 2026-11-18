"""
The shared data contract for CASTER.

This is the single most important file for connecting Phase 1 -> Phase 2 ->
Phase 3 -> Phase 4. Every phase reads and writes these same shapes, so no
phase needs to know how an earlier phase's internals work.

    Phase 1 produces:  SceneState   (one per frame, JSON-serialisable)
                        FeatureWindow (one per 30-frame window, feeds Phase 2)

    Phase 2 will consume FeatureWindow + SceneState, and produce:
                        EventPrediction (adds event name + confidence)

    Phase 3 will consume EventPrediction (post confidence-gate), and produce:
                        commentary text (plain string, streamed token by token)

    Phase 4 wires EventPrediction + commentary text into the WebSocket/React
    layer. It never touches anything below Phase 2's output.

IMPORTANT: the feature vector is always 12 numbers (see phase1/README.md).
CASTER is football-only. The vector is built from the ball, the players and
the referee; index 7 (pitch_zone) and index 10 (contact_flag) are the two
entries that depend on football-specific geometry.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# ---------------------------------------------------------------------------
# Feature vector index map (12 dimensions, shared by both sports)
# ---------------------------------------------------------------------------
# 0  ball_x                normalised 0-1 ball x position
# 1  ball_y                normalised 0-1 ball y position
# 2  ball_dx               x velocity (this frame minus previous frame)
# 3  ball_dy               y velocity
# 4  ball_speed            sqrt(dx^2 + dy^2)
# 5  nearest_player_dist   euclidean distance, ball to closest player (normalised)
# 6  players_nearby        count of players within a small radius of the ball
# 7  pitch_zone_id         int, index into config.FOOTBALL_ZONES
# 8  official_signal       binary flag, a referee was detected in the frame
# 9  ball_height_proxy     proxy for aerial vs grounded ball
# 10 contact_flag          binary, foot-ball contact (approximated from the nearest player's box)
# 11 boundary_proximity    binary, ball near the touchline / goal line (frame edge)
FEATURE_NAMES = [
    "ball_x", "ball_y", "ball_dx", "ball_dy", "ball_speed",
    "nearest_player_dist", "players_nearby", "pitch_zone_id",
    "official_signal", "ball_height_proxy", "contact_flag", "boundary_proximity",
]


@dataclass
class Detection:
    """A single YOLO detection in one frame."""
    track_id: Optional[int]
    cls_name: str
    confidence: float
    x: float          # bbox centre x, pixels
    y: float           # bbox centre y, pixels
    w: float
    h: float


@dataclass
class SceneState:
    """
    One frame's worth of structured, verified facts.
    This is what gets serialised to JSON and handed to Phase 2.
    Nothing in here is guessed -- every field traces back to a detection
    or a direct computation on detections.
    """
    sport: str                          # always "football" (kept so downstream JSON stays self-describing)
    frame_id: int
    timestamp_sec: float
    ball_position: List[float]          # [x, y] normalised 0-1
    ball_velocity: List[float]          # [dx, dy]
    ball_speed: float
    pitch_zone: str                     # human-readable zone name
    boundary_proximity: bool
    official_signal: bool
    contact_flag: bool
    nearest_player_id: Optional[int]
    nearest_player_dist: Optional[float]
    players_nearby: int
    feature_vector: List[float]         # the 12-number vector, see FEATURE_NAMES
    detections: List[Detection] = field(default_factory=list)

    # --- Ball-tracking diagnostics (added alongside the improved ball
    # disambiguation logic in feature_extractor.py's _resolve_ball()) ---
    # These are observation facts about the CURRENT frame's ball
    # detection/tracking quality, not inferred or guessed -- they exist
    # so run_phase1.py's logs and Phase 4's frontend can honestly show
    # when the ball genuinely wasn't visible/trackable in a frame,
    # instead of silently reusing a stale last-known position with no
    # indication anything was uncertain. All default to their "no ball
    # seen this frame" values so existing callers that construct a
    # SceneState without passing these (e.g. older tests) still work.
    ball_detected: bool = False
    ball_track_id: Optional[int] = None
    ball_confidence: Optional[float] = None
    ball_candidate_count: int = 0
    ball_missing_frames: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureWindow:
    """
    A (30, 12) sliding window of feature vectors, the unit Phase 2's
    classifier actually trains and predicts on. One window ends at
    each new frame once at least FEATURE_WINDOW_SIZE frames exist.
    """
    sport: str
    end_frame_id: int
    end_timestamp_sec: float
    vectors: List[List[float]]          # shape (window_size, 12)

    def shape(self):
        return (len(self.vectors), len(self.vectors[0]) if self.vectors else 0)


@dataclass
class EventPrediction:
    """
    Phase 2's output shape. Defined here now so Phase 1's output is already
    compatible -- Phase 2 just needs to add `event` and `confidence` on top
    of a SceneState.
    """
    sport: str
    event: str
    confidence: float
    scene: SceneState
    recent_events: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class LabelledWindow:
    """
    Phase 2's training data unit: a FeatureWindow paired with a
    human-assigned event label. This is what the labelling tool
    produces and what xgboost_clf.py / bilstm_clf.py train on.

    Stored as flat JSON-lines (one per line) in data/labels/<sport>.jsonl
    so labelling sessions can be resumed/appended without re-parsing a
    giant JSON array each time.
    """
    sport: str
    event: str                  # must be one of config.EVENTS_BY_SPORT[sport]
    vectors: List[List[float]]  # shape (window_size, 12)
    source_clip: str = ""       # which video this came from, for traceability
    end_frame_id: int = -1

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict())