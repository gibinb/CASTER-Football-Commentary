"""
YOLOv8 detection wrapper for CASTER (football only).

Loads the fine-tuned football model (phase1/weights/football.pt: ball,
goalkeeper, player, referee) and runs it frame by frame with per-class
confidence thresholds.

Optionally runs inference through SAHI (sliced/tiled prediction) instead of
a single full-frame pass -- see CASTER_USE_SAHI below. This was evaluated
with phase1/sahi_ball_eval.py across three clips (two FIFA26-rendered, one
real broadcast footage): every above-floor candidate SAHI found that the
normal single pass missed turned out to be the real ball (5/5 checked), at
roughly a 3x inference-time cost. It is opt-in, not the default, because
of that cost -- enable it for clips where recall matters more than speed.

If weights/football.pt is missing, this falls back to plain pretrained
yolov8n.pt (COCO classes). COCO has "person" and "sports ball" but nothing
football-specific (no goalkeeper, no referee), so detections will be coarse
-- this is expected and explained in phase1/README.md.

Device selection: Ultralytics does NOT reliably auto-move inference onto a
CUDA GPU just because torch.cuda.is_available() is True -- a loaded YOLO
model sits on CPU until a device is explicitly resolved at call time, and in
practice leaving device unset can silently stay on CPU even when CUDA is
available (confirmed directly on this project: torch.cuda.is_available()
was True but detect_frame() still ran and returned a CPU-resident model).
DEVICE below is therefore computed once at import time and passed explicitly
into every predict()/track() call rather than relying on Ultralytics'
default resolution.
"""

import os
from pathlib import Path
from typing import List
import torch
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import FOOTBALL_CLASSES, WEIGHTS_DIR
from schemas import Detection

# --------------------------------------------------------------------------
# Inference tunables (override with environment variables, no code edits).
#
#   CASTER_IMGSZ       inference resolution. Broadcast frames are wide and the
#                      football is only a few pixels across, so 640 (the
#                      training default) often loses the ball entirely.
#                      Default 1280: measured on Final.mp4, 1280 gave 44% usable
#                      ball visibility versus 26% at 960, but runs ~1.8x slower
#                      (about 85 s per 10 s of video on a laptop CPU). Set
#                      CASTER_IMGSZ=960 (or 640) in .env for faster, rougher runs.
#   CASTER_PLAYER_CONF minimum confidence for players/goalkeeper/referee.
#   CASTER_BALL_CONF   minimum confidence for the ball -- deliberately lower,
#                      because a small fast ball never scores as confidently
#                      as a player. Downstream code only ever keeps the best
#                      few candidates, and feature_extractor rejects jumps.
#   CASTER_REFEREE_CONF minimum confidence for the referee -- deliberately
#                      higher than PLAYER_CONF, since a low-confidence
#                      referee detection is often actually a regular field
#                      player. See the comment on REFEREE_CONF below.
#   CASTER_USE_SAHI    "1" to run detection as sliced/tiled inference (via the
#                      `sahi` package: pip install sahi) instead of one
#                      full-frame pass. SAHI works on the frame's native
#                      resolution tile-by-tile, so CASTER_IMGSZ does not apply
#                      when this is on -- use CASTER_SAHI_SLICE_SIZE instead.
#                      Roughly 3x slower per frame; see the module docstring
#                      above for the measured recall gain this buys.
#   CASTER_SAHI_SLICE_SIZE  tile width/height in pixels (default 640).
#   CASTER_SAHI_OVERLAP     fraction of overlap between adjacent tiles (default 0.2).
#   CASTER_DEVICE      "cuda", "cpu", or a specific index like "0". Defaults
#                      to "cuda" automatically if torch.cuda.is_available(),
#                      otherwise "cpu". Override only if you need to force
#                      CPU on a CUDA-capable machine (e.g. to compare timing).
# --------------------------------------------------------------------------
INFER_IMGSZ = int(os.environ.get("CASTER_IMGSZ", "1280"))
PLAYER_CONF = float(os.environ.get("CASTER_PLAYER_CONF", "0.30"))
BALL_CONF = float(os.environ.get("CASTER_BALL_CONF", "0.15"))
# Measured on Final.mp4 (inspect_detections.py --auto-sample referee, 9 real frames
# checked): confidence does NOT cleanly separate correct from wrong referee
# detections -- 0.41 was wrong in one frame but correct in another, and 0.65 was
# wrong while 0.58 (lower) was correct. No single floor perfectly separates them.
# 0.70 is the lowest value that rejected every confirmed-wrong case found (0.31,
# 0.41, 0.65) while keeping the clearest correct ones (0.72, 0.79, 0.96) -- it
# trades away some weaker correct detections (an already-rare class gets rarer)
# to stop wrong ones from setting official_signal=True. Referee is a secondary
# signal for CASTER, not load-bearing for ball/player-driven commentary, so
# this is a deliberately conservative (precision over recall) choice.
REFEREE_CONF = float(os.environ.get("CASTER_REFEREE_CONF", "0.70"))
USE_SAHI = os.environ.get("CASTER_USE_SAHI", "0") == "1"
SAHI_SLICE_SIZE = int(os.environ.get("CASTER_SAHI_SLICE_SIZE", "640"))
SAHI_OVERLAP = float(os.environ.get("CASTER_SAHI_OVERLAP", "0.2"))

# Resolved once at import time. Explicit, not left to Ultralytics' own
# default device resolution -- see the module docstring for why.
DEVICE = os.environ.get("CASTER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Custom ByteTrack settings (see the comments inside the yaml for why).
TRACKER_YAML = Path(__file__).resolve().parents[1] / "configs" / "caster_bytetrack.yaml"


# Maps COCO class names onto our class names while running un-fine-tuned
# yolov8n.pt.
#
# "person" is mapped directly to "player", the class name
# feature_extractor.py's _find_players() recognizes. (An earlier version used
# a cosmetic placeholder label that matched nothing, so nearest_player_dist,
# players_nearby and contact_flag were silently wrong on every frame.)
# What this fallback still cannot give you without real fine-tuned weights:
#   - the goalkeeper / referee classes (COCO has no equivalent, so
#     official_signal stays False)
#   - team separation (no public dataset labels it)
#   - a ball detector tuned for a few-pixel broadcast football
_COCO_FALLBACK_HINTS = {"person": "player", "sports ball": "ball"}


class FootballDetector:
    """
    Loads and runs the YOLOv8 football model. Create one instance per
    process; reuse it for every video.
    """

    def __init__(self, weights_path: str = None):
        self.classes = FOOTBALL_CLASSES

        if weights_path is None:
            weights_path = self._default_weights_path()

        self.model = YOLO(weights_path)
        self._weights_path = weights_path

        # SAHI gets its own model handle (a thin wrapper around the same
        # checkpoint) built lazily on first use, not here -- importing sahi
        # and constructing its wrapper has real cost, and most runs don't
        # use it. See _sahi_model() below.
        self._sahi_model_cache = None

        # Decide "fine-tuned" from what the checkpoint actually contains, NOT
        # from its filename. A stock COCO checkpoint renamed to football.pt
        # has the 80 COCO classes ('person', 'sports ball', ...); treating it
        # as fine-tuned would skip the COCO->player/ball mapping below and
        # leave the pipeline with zero players and zero balls.
        raw_names = set(self.model.names.values())
        looks_like_coco = {"person", "sports ball"} <= raw_names
        self.using_finetuned_weights = Path(weights_path).exists() and not looks_like_coco

    def _default_weights_path(self) -> str:
        """
        Looks for fine-tuned weights at weights/football.pt first.
        Falls back to plain pretrained yolov8n.pt if that file is missing.
        """
        finetuned = WEIGHTS_DIR / "football.pt"
        if finetuned.exists():
            return str(finetuned)
        return "yolov8n.pt"  # auto-downloaded by ultralytics on first use

    def detect(self, frame) -> List[Detection]:
        """
        LEGACY: single shared confidence threshold. Kept only for callers
        outside the live pipeline that haven't been updated. Prefer
        detect_frame(), which uses the per-class PLAYER_CONF/BALL_CONF
        thresholds -- the ball needs a lower bar than players/officials to
        be reliably detected in broadcast footage; see the module-level
        comment near those constants for why.
        """
        return self.detect_frame(frame)

    def detect_frame(self, frame, imgsz: int = None) -> List[Detection]:
        """
        One pass over one frame with PER-CLASS confidence thresholds.
        Players need PLAYER_CONF; the ball only needs the lower BALL_CONF.
        Returned Detections have track_id=None -- ID assignment for players
        happens in tracking/bytetrack_wrapper.py, and the ball is
        deliberately never tracked (see the comments there).

        Routes to SAHI sliced inference instead of a single full-frame pass
        when CASTER_USE_SAHI=1 -- see the module docstring and the
        CASTER_USE_SAHI comment above for what that trades off.

        imgsz overrides the module-level INFER_IMGSZ for this one call only
        -- used by imgsz_sweep.py to benchmark several resolutions in a
        single process without re-importing the module between them. Leave
        unset for normal pipeline use (reads CASTER_IMGSZ as always).
        Ignored when SAHI is in use (see CASTER_USE_SAHI above).
        """
        if USE_SAHI:
            return self._detect_frame_sahi(frame)
        return self._detect_frame_normal(frame, imgsz)

    @staticmethod
    def _min_conf_for(name: str) -> float:
        if name == "ball":
            return BALL_CONF
        if name == "referee":
            return REFEREE_CONF
        return PLAYER_CONF

    def _detect_frame_normal(self, frame, imgsz: int = None) -> List[Detection]:
        results = self.model.predict(
            frame, verbose=False, imgsz=imgsz if imgsz is not None else INFER_IMGSZ,
            conf=min(PLAYER_CONF, BALL_CONF),
            device=DEVICE,
        )
        res = results[0]
        detections = []
        for box in res.boxes:
            name = self._resolve_class_name(res.names[int(box.cls[0])])
            conf = float(box.conf[0])
            if conf < self._min_conf_for(name):
                continue
            x, y, w, h = box.xywh[0].tolist()
            detections.append(Detection(
                track_id=None, cls_name=name, confidence=conf,
                x=x, y=y, w=w, h=h,
            ))
        return detections

    def _get_sahi_model(self):
        """Lazily builds the SAHI wrapper around this detector's own checkpoint."""
        if self._sahi_model_cache is not None:
            return self._sahi_model_cache
        try:
            from sahi import AutoDetectionModel
        except ImportError:
            raise ImportError(
                "CASTER_USE_SAHI=1 but the `sahi` package isn't installed. Run: pip install sahi"
            )
        self._sahi_model_cache = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self._weights_path,
            confidence_threshold=min(PLAYER_CONF, BALL_CONF),
            device=DEVICE,
        )
        return self._sahi_model_cache

    def _detect_frame_sahi(self, frame) -> List[Detection]:
        from sahi.predict import get_sliced_prediction
        result = get_sliced_prediction(
            frame, self._get_sahi_model(),
            slice_height=SAHI_SLICE_SIZE, slice_width=SAHI_SLICE_SIZE,
            overlap_height_ratio=SAHI_OVERLAP, overlap_width_ratio=SAHI_OVERLAP,
            verbose=0,
        )
        detections = []
        for p in result.object_prediction_list:
            name = self._resolve_class_name(p.category.name)
            conf = float(p.score.value)
            if conf < self._min_conf_for(name):
                continue
            b = p.bbox
            w, h = b.maxx - b.minx, b.maxy - b.miny
            x, y = b.minx + w / 2, b.miny + h / 2
            detections.append(Detection(
                track_id=None, cls_name=name, confidence=conf,
                x=x, y=y, w=w, h=h,
            ))
        return detections

    def track(self, source: str, stream: bool = True):
        """
        LEGACY: Ultralytics' built-in YOLO+ByteTrack. Kept for backward
        compatibility only. It only ever returns *confirmed* tracks, so a
        small, flickering object like the football is almost always dropped.
        The live pipeline and run_phase1.py now use
        tracking/bytetrack_wrapper.iter_tracked_frames() instead.
        """
        return self.model.track(
            source=source,
            tracker=str(TRACKER_YAML),
            stream=stream,
            persist=False,      # fresh tracker per video, never reuse stale tracks
            conf=min(PLAYER_CONF, BALL_CONF),
            imgsz=INFER_IMGSZ,
            verbose=False,
            device=DEVICE,
        )

    # Class names as they exist in a trained checkpoint sometimes don't
    # exactly match what the rest of the pipeline expects (the shipped
    # football.pt says "players"; an older team-split scheme used
    # player_team_a/b). Aliased here, once, so every downstream consumer
    # (feature_extractor, tracking) sees the canonical name. Add to this dict
    # any time `python -c "from ultralytics import YOLO;
    # print(YOLO('phase1/weights/football.pt').names)"` shows a name that
    # doesn't match config.FOOTBALL_CLASSES.
    _CLASS_NAME_ALIASES = {
        "players": "player",
        "player_team_a": "player",
        "player_team_b": "player",
    }

    def _resolve_class_name(self, raw_name: str) -> str:
        """Maps COCO names to a readable placeholder if running un-fine-tuned
        weights, then normalises any known checkpoint-specific alias
        regardless of weight source."""
        name = raw_name
        if not self.using_finetuned_weights:
            name = _COCO_FALLBACK_HINTS.get(name, name)
        return self._CLASS_NAME_ALIASES.get(name, name)


def load_detector(weights_path: str = None) -> FootballDetector:
    """Convenience: load the football detector once at startup."""
    return FootballDetector(weights_path)