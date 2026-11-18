"""
Feature extraction for CASTER.

Turns a frame's tracked bounding boxes into a fixed 12-number vector.
Pure Python + numpy -- no model training happens in this module.

The vector is always 12 numbers (index map in schemas.py). Two indices
depend on football-specific geometry:

    index 7  (pitch_zone_id)  -- coarse zone from the ball's position
    index 10 (contact_flag)   -- foot-ball proximity, approximated from
                                  the nearest player's full bounding box

Ball selection uses temporal validation, not single-frame trust: see
_resolve_ball()'s docstring below for why, and the comment block above
its class constants for what changed and when.

FOOTBALL CLASS SCHEME: real fine-tuning data was sourced from
Roboflow Universe's "football-player-detection-bfswn" dataset, which
uses a generic 4-class scheme (ball, goalkeeper, player, referee) --
it does NOT separate player_team_a/player_team_b, since no public
dataset labels team affiliation (jersey colours change every match).
_find_players() below matches this real class name ("player"), not the
5-class team-separated scheme originally planned. Team separation via
K-means clustering on jersey colour histograms, applied AFTER generic
player detection, remains documented future work -- see
CASTER_Phase1_DeepDive.docx Section 2.4 for the original design intent.
"""

import math
from collections import deque
from typing import List, Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import FOOTBALL_ZONES, FEATURE_WINDOW_SIZE
from schemas import Detection, FEATURE_NAMES


class FeatureExtractor:
    """
    One instance per video. Call .extract(detections, frame_w, frame_h)
    once per frame; it maintains its own history internally so velocity
    and windowed features work correctly across calls.
    """

    def __init__(self):
        self.zones = FOOTBALL_ZONES

        self.ball_history = deque(maxlen=10)     # last 10 normalised (x, y), REAL observations only
        self.speed_history = deque(maxlen=2)      # for acceleration if needed later
        self.window = deque(maxlen=FEATURE_WINDOW_SIZE)

        # --- ball-tracking diagnostics (see get_tracking_diagnostics()) ---
        # These record observation FACTS only -- never inferred -- so
        # confidence_gate.ball_visibility_ok() (Phase 2) and any UI/logs
        # can tell a real ball observation apart from the positional
        # fallback used below when nothing was detected this frame.
        self.last_observed_ball_pos: Optional[Tuple[float, float]] = None
        self.last_ball_track_id = None
        self.last_ball_confidence: Optional[float] = None
        self.last_ball_candidate_count: int = 0
        self.ball_missing_frames: int = 0

        # Unconfirmed ball candidates awaiting a second, consistent sighting
        # before they're trusted -- see _resolve_ball(). Each entry:
        # {"positions": [(x,y), ...] (normalised, one per consecutive frame
        # it has been seen in), "conf": most recent confidence}.
        self.pending: List[dict] = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def extract(self, detections: List[Detection], frame_w: int, frame_h: int) -> List[float]:
        """
        Computes the 12-number feature vector for one frame and appends
        it to the rolling window. Returns just this frame's vector;
        call self.get_window() to get the full (N, 12) window so far.
        """
        ball = self._resolve_ball(detections, frame_w, frame_h)
        players = self._find_players(detections)
        official_detected = self._official_signal(detections)

        ball_visible = ball is not None
        ball_pos = self._normalise(ball, frame_w, frame_h) if ball_visible else self._last_known_ball_pos()
        dx, dy, speed = self._velocity(ball_pos) if ball_visible else (0.0, 0.0, 0.0)

        nearest_dist, nearest_id = self._nearest_player(ball_pos, players, frame_w, frame_h)
        players_nearby = self._count_nearby(ball_pos, players, frame_w, frame_h, radius_norm=0.05)

        zone_id = self._pitch_zone(ball_pos)
        height_proxy = ball_pos[1]  # simple proxy: smaller y = higher in frame = more aerial
        contact_flag = self._contact_flag(ball, detections, frame_w, frame_h)
        boundary_flag = self._boundary_proximity(ball_pos)

        vector = [
            ball_pos[0], ball_pos[1],
            dx, dy, speed,
            nearest_dist if nearest_dist is not None else 1.0,
            float(players_nearby),
            float(zone_id),
            float(official_detected),
            height_proxy,
            float(contact_flag),
            float(boundary_flag),
        ]

        assert len(vector) == 12, "Feature vector must stay 12-dimensional"

        if ball_visible:
            # Real observation: safe to use for the next frame's velocity,
            # and clears the "missing" streak.
            self.ball_history.append(ball_pos)
            self.last_observed_ball_pos = ball_pos
            self.last_ball_track_id = ball.track_id
            self.last_ball_confidence = ball.confidence
            self.ball_missing_frames = 0
        else:
            # Do NOT push the (0.5, 0.5) / last-known fallback into
            # ball_history -- doing so would make the NEXT frame's
            # velocity calc measure a fake "jump" to a stale position
            # once the ball reappears. The fallback above exists only to
            # keep this frame's vector shape/values sane; it is not a
            # real observation.
            self.ball_missing_frames += 1
        self.window.append(vector)
        return vector

    def get_tracking_diagnostics(self) -> dict:
        """Returns observation facts for logs/UI/Phase 2's ball-visibility gate; none are inferred."""
        return {
            "ball_detected": self.ball_missing_frames == 0 and self.last_observed_ball_pos is not None,
            "ball_track_id": self.last_ball_track_id,
            "ball_confidence": self.last_ball_confidence,
            "ball_candidate_count": self.last_ball_candidate_count,
            "ball_missing_frames": self.ball_missing_frames,
        }

    def get_window(self) -> List[List[float]]:
        """Returns the current sliding window, up to FEATURE_WINDOW_SIZE frames."""
        return list(self.window)

    def window_is_full(self) -> bool:
        return len(self.window) == FEATURE_WINDOW_SIZE

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    # --- Temporal ball validation -------------------------------------
    #
    # Earlier versions of this gate trusted a single frame's YOLO output
    # outright (after a confidence floor and a distance check). Testing on
    # real match frames (dump_ball_candidates.py, Final.mp4 and others)
    # found cases that single-frame logic cannot fix: a false "ball" on a
    # boot scored 0.71 confidence in the SAME frame where the real ball
    # scored only 0.31. No confidence threshold separates those two
    # numbers correctly, because the wrong object was simply more
    # confident than the right one.
    #
    # The fix here is temporal, not a better threshold: a candidate is
    # only trusted once it has been seen in BALL_CONFIRM_FRAMES
    # consecutive processed frames in roughly the same place. A one-off
    # false positive (a boot, a line, a graphic) essentially never
    # reappears in the same spot the very next frame; the real ball does.
    # This costs latency -- about BALL_CONFIRM_FRAMES-1 frames (≈33ms
    # each at 30 processed fps) before a brand-new or reacquired ball
    # sighting is trusted -- and it will likely lower the raw
    # "ball_detected" percentage you see in check_tracking.py /
    # run_phase1.py logs, especially right after the ball reappears from
    # a gap. That trade is intentional: a position reported one frame
    # late is far less harmful to Phase 2's training labels than a
    # position reported immediately but wrong.
    #
    # Already-confirmed tracks are NOT re-validated every frame -- once
    # trusted, a track continues on proximity alone (within
    # BALL_CONTINUATION_MAX_MISSING missed frames), the same as before,
    # so a confirmed ball dipping in per-frame confidence (motion blur,
    # partial occlusion) doesn't lose its track. Re-validation only
    # applies when starting fresh: the very first sighting in a clip, or
    # reacquiring after the track has gone unseen for too long to trust
    # blindly. There is no special "reacquire after N misses, skip
    # validation" bypass any more -- that bypass was exactly what let a
    # single 0.16-confidence boot detection get accepted as the ball
    # after a long gap in Semifinal01.mp4 at 263s. Every fresh sighting
    # now goes through the same two-hit check, reacquisition included.
    BALL_MAX_JUMP_NORM = 0.04          # max normalised movement trusted between two consecutive sightings
    BALL_JUMP_GROWTH = 0.02            # extra allowance per missed frame while CONTINUING a confirmed track
    BALL_CONTINUATION_MAX_MISSING = 5  # beyond this many consecutive misses, stop continuing blindly --
                                       # require the two-hit check again, same as a brand-new sighting
    BALL_ACQUIRE_MIN_CONF = 0.25       # floor to even become a pending (unconfirmed) candidate,
                                       # and to extend one that is already pending
    BALL_CONFIRM_FRAMES = 2            # consecutive consistent sightings required before a candidate is trusted
    #
    # CONTINUATION had no confidence floor at all until this was added -- on
    # Final.mp4 at ~259s that let an already-confirmed track drift onto a
    # genuinely wrong, 0.19-confidence detection, simply because it happened
    # to be near where the track was last confirmed (the real ball, at 0.56
    # confidence, sat elsewhere and was never considered). 0.20 sits just
    # below BALL_ACQUIRE_MIN_CONF -- low enough that a confirmed track still
    # survives an ordinary confidence dip (motion blur, partial occlusion),
    # high enough to reject this measured failure.
    BALL_CONTINUATION_MIN_CONF = 0.20

    def _resolve_ball(self, detections: List[Detection], frame_w: int = 1, frame_h: int = 1) -> Optional[Detection]:
        candidates = sorted((d for d in detections if d.cls_name == "ball"),
                            key=lambda d: d.confidence, reverse=True)
        self.last_ball_candidate_count = len(candidates)
        used = [False] * len(candidates)
        this_frame_ball: Optional[Detection] = None

        # 1. Try to continue an already-confirmed track: proximity only,
        #    no confidence check -- a trusted track survives a confidence dip.
        last = self.last_observed_ball_pos
        if last is not None and self.ball_missing_frames < self.BALL_CONTINUATION_MAX_MISSING:
            allowed = self.BALL_MAX_JUMP_NORM + self.BALL_JUMP_GROWTH * self.ball_missing_frames
            for i, d in enumerate(candidates):
                if d.confidence < self.BALL_CONTINUATION_MIN_CONF:
                    continue
                nx, ny = self._normalise(d, frame_w, frame_h)
                if math.hypot(nx - last[0], ny - last[1]) <= allowed:
                    used[i] = True
                    this_frame_ball = d
                    break
            if this_frame_ball is not None:
                # The real track just reconfirmed itself; a rival hypothesis
                # that got no support this frame is stale -- drop it rather
                # than let it match against an aging, un-updated position.
                self.pending = []

        # 2. No continuation: advance pending hypotheses with whatever
        #    candidates remain. Support must be in THIS frame -- a
        #    hypothesis unsupported even once is dropped, not carried
        #    forward, which is what makes a one-off false blip fail to
        #    ever get promoted.
        if this_frame_ball is None:
            still_pending = []
            for hyp in self.pending:
                match_i = None
                hx, hy = hyp["positions"][-1]
                for i, d in enumerate(candidates):
                    if used[i] or d.confidence < self.BALL_ACQUIRE_MIN_CONF:
                        continue
                    nx, ny = self._normalise(d, frame_w, frame_h)
                    if math.hypot(nx - hx, ny - hy) <= self.BALL_MAX_JUMP_NORM:
                        match_i = i
                        break
                if match_i is None:
                    continue  # unsupported this frame -- this hypothesis dies here
                used[match_i] = True
                d = candidates[match_i]
                nx, ny = self._normalise(d, frame_w, frame_h)
                hyp["positions"].append((nx, ny))
                hyp["conf"] = d.confidence
                if len(hyp["positions"]) >= self.BALL_CONFIRM_FRAMES:
                    # Confirmed. Backfill ball_history with every position
                    # this hypothesis saw except the current one -- extract()'s
                    # normal post-processing appends the current one -- so
                    # velocity on this very frame reflects real movement
                    # instead of reading an empty history as zero speed.
                    for p in hyp["positions"][:-1]:
                        self.ball_history.append(p)
                    self.ball_missing_frames = 0   # the gap between the backfilled points is exactly 1 frame
                    self.pending = []              # a confirmed sighting resolves every competing hypothesis
                    this_frame_ball = d
                    break
                still_pending.append(hyp)
            if this_frame_ball is None:
                self.pending = still_pending

        # 3. Any candidate not already used seeds a new pending hypothesis,
        #    unless we already have a trusted ball this frame (in which case
        #    leftover candidates are noise -- ignore them rather than let
        #    them compete with a track we already trust).
        if this_frame_ball is None:
            for i, d in enumerate(candidates):
                if used[i] or d.confidence < self.BALL_ACQUIRE_MIN_CONF:
                    continue
                nx, ny = self._normalise(d, frame_w, frame_h)
                self.pending.append({"positions": [(nx, ny)], "conf": d.confidence})

        return this_frame_ball

    def _find_players(self, detections: List[Detection]) -> List[Detection]:
        # The real fine-tuning dataset (Roboflow "football-player-detection-bfswn")
        # uses a single generic "player" class -- no team separation, since no
        # public dataset labels team affiliation (jersey colours change every
        # match). Team separation remains documented future work via K-means
        # clustering on jersey colour histograms.
        return [d for d in detections if d.cls_name in ("player", "goalkeeper")]

    def _official_signal(self, detections: List[Detection]) -> bool:
        return any(d.cls_name == "referee" for d in detections)

    def _normalise(self, det: Detection, frame_w: int, frame_h: int) -> Tuple[float, float]:
        return (det.x / frame_w, det.y / frame_h)

    def _last_known_ball_pos(self) -> Tuple[float, float]:
        return self.ball_history[-1] if self.ball_history else (0.5, 0.5)

    def _velocity(self, ball_pos: Tuple[float, float]) -> Tuple[float, float, float]:
        if not self.ball_history:
            return 0.0, 0.0, 0.0
        prev = self.ball_history[-1]
        # If the ball was missing for k frames, the displacement since the last
        # real observation spans k+1 frames: report PER-FRAME velocity, not the
        # whole gap as one frame's movement.
        gap = self.ball_missing_frames + 1
        dx = (ball_pos[0] - prev[0]) / gap
        dy = (ball_pos[1] - prev[1]) / gap
        speed = math.sqrt(dx ** 2 + dy ** 2)
        return dx, dy, speed

    def _nearest_player(self, ball_pos, players: List[Detection], frame_w, frame_h):
        if not players:
            return None, None
        best_dist = float("inf")
        best_id = None
        for p in players:
            px, py = p.x / frame_w, p.y / frame_h
            dist = math.sqrt((ball_pos[0] - px) ** 2 + (ball_pos[1] - py) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_id = p.track_id
        return best_dist, best_id

    def _count_nearby(self, ball_pos, players: List[Detection], frame_w, frame_h, radius_norm: float) -> int:
        count = 0
        for p in players:
            px, py = p.x / frame_w, p.y / frame_h
            dist = math.sqrt((ball_pos[0] - px) ** 2 + (ball_pos[1] - py) ** 2)
            if dist <= radius_norm:
                count += 1
        return count

    def _pitch_zone(self, ball_pos: Tuple[float, float]) -> int:
        """
        Zone reflects the 2D field (penalty_box -> midfield -> wing ->
        goal_line), approximated from the ball's x-position in the FRAME
        (a linear bucket) and nudged to "wing" near the top/bottom edge.

        This is a simple geometric placeholder: it describes where the ball
        is in the camera image, not on the real pitch. Once you have real
        pitch calibration (homography), replace it with actual zone polygons.
        """
        x, y = ball_pos
        n_zones = len(self.zones)
        if y < 0.2 or y > 0.8:
            return self.zones.index("wing")
        return min(int(x * n_zones), n_zones - 1)

    # A touch is a strong claim ("someone just played the ball"). Don't make it
    # from a shaky detection: on Final.mp4 a 0.16-confidence "ball" on a boot
    # produced contact=True. Below this confidence we report no contact.
    BALL_CONTACT_MIN_CONF = 0.25

    def _contact_flag(self, ball: Optional[Detection], detections: List[Detection],
                       frame_w: int, frame_h: int) -> bool:
        """
        NO dedicated foot-detection class exists, so contact is approximated
        by proximity between the ball and the lower portion of each player's
        full bounding box. This is a known limitation (see phase1/README.md)
        -- a player's box covers their whole body, so the signal is noisy.
        Phase 2's classifier should treat it as a hint, not a fact.
        """
        if ball is None or ball.confidence < self.BALL_CONTACT_MIN_CONF:
            return False

        for p in self._find_players(detections):
            # approximate "foot" as the bottom 25% of the player's bbox
            foot_y = p.y + (p.h * 0.375)  # bbox y is centre, so shift toward bottom
            foot_det = Detection(p.track_id, p.cls_name, p.confidence, p.x, foot_y, p.w, p.h * 0.25)
            if self._boxes_close(ball, foot_det, frame_w, frame_h, threshold_norm=0.025):
                return True
        return False

    def _boxes_close(self, a: Detection, b: Detection, frame_w: int, frame_h: int, threshold_norm: float) -> bool:
        """
        Proximity check between two detections' CENTRE POINTS (normalised
        distance), not true bounding-box overlap/IoU. This mirrors the
        simple proximity style used in the Phase 1 deep-dive doc's own
        pseudocode (Section 5, contact flag) -- intentionally simple
        for an MVP. If contact detection proves too noisy once you have
        real fine-tuned weights, upgrading this to true IoU-based overlap
        (using a.w/a.h and b.w/b.h) is the natural next step.
        """
        ax, ay = a.x / frame_w, a.y / frame_h
        bx, by = b.x / frame_w, b.y / frame_h
        dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        return dist <= threshold_norm

    def _boundary_proximity(self, ball_pos: Tuple[float, float], margin: float = 0.05) -> bool:
        x, y = ball_pos
        return x <= margin or x >= (1 - margin) or y <= margin or y >= (1 - margin)