"""
ByteTrack integration for CASTER.

ByteTrack is built directly into Ultralytics -- there's no separate
library to install. This module just wraps the .track() output into our
own Detection dataclass so the rest of the pipeline (features, scene
state) never has to know it's talking to Ultralytics specifically.

ByteTrack only sees bounding boxes and IDs; it knows nothing about
football. The football-specific work happens upstream (which class names
come out of detection) and downstream (what those class names mean in
feature extraction).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from schemas import Detection

# Ball candidates handed to the feature extractor per frame. It picks the one
# closest to the previous position and rejects implausible jumps, so a couple
# of candidates is enough and keeps false positives (white socks, logos) out.
MAX_BALL_CANDIDATES = 3


class _TrackInputs:
    """
    Minimal stand-in for Ultralytics' own detection-results object, built
    just well enough for BYTETracker.update() to accept it.

    Newer Ultralytics versions (checked against 8.4.x) have BYTETracker
    call results[mask] internally (see _split_detections /
    trackers/utils/stracks.parse_bboxes in the installed package) to split
    high/low confidence detections -- i.e. it needs boolean ARRAY SLICING,
    not just attribute access. A plain types.SimpleNamespace supports
    `.conf`/`.xywh`/`.cls` attribute access but not `namespace[mask]`,
    which is why that approach raised "SimpleNamespace object is not
    subscriptable". This class adds just __len__ and __getitem__ so the
    same conf/xywh/cls arrays we already build can be sliced the way
    BYTETracker expects, while staying a plain, dependency-free container
    (no ultralytics.engine.results.Results construction, which needs a
    source image and model names we don't have at this point).
    """
    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh, conf, cls):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return _TrackInputs(self.xywh[mask], self.conf[mask], self.cls[mask])



@dataclass
class TrackedFrame:
    """One processed frame: everything downstream needs, incl. real dimensions."""
    frame_idx: int          # index in the ORIGINAL video (not the strided one)
    timestamp_sec: float    # frame_idx / fps -- what the browser video clock uses
    frame_w: int
    frame_h: int
    detections: List[Detection]


# ---------------------------------------------------------------------------
# Why this file no longer calls model.track()
#
# Ultralytics' built-in tracker only returns *confirmed* tracks. ByteTrack
# confirms a new track on its second consecutive matched frame, so a tiny
# ball that flickers in and out of detection is almost never returned -- the
# old code then saw "no ball" on nearly every frame, the feature extractor
# fell back to (0.5, 0.5), and the event classifier was fed junk.
#
# Instead: ONE detection pass per frame (detector.detect_frame), then
#   * players / goalkeeper / referee -> our own BYTETracker (persistent IDs)
#   * ball                           -> passed through un-tracked
# ---------------------------------------------------------------------------

def _load_tracker_args() -> SimpleNamespace:
    import yaml
    from phase1.detection.yolo_detector import TRACKER_YAML
    with open(TRACKER_YAML, "r", encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("fuse_score", True)
    return SimpleNamespace(**cfg)


def _make_tracker(frame_rate: float):
    """
    Constructs a BYTETracker in a way that's tolerant of Ultralytics
    version differences. Ultralytics has changed BYTETracker's __init__
    signature across releases -- some versions accept frame_rate as a
    keyword, some accept it positionally, and some don't take it at all
    (frame rate is set via a separate attribute or not used). Rather than
    pin an exact ultralytics version, this inspects the real signature at
    import time and calls it the way THIS installed version expects.
    """
    import inspect
    from ultralytics.trackers.byte_tracker import BYTETracker
    try:  # ids restart at 1 for every video instead of counting up forever
        from ultralytics.trackers.basetrack import BaseTrack
        BaseTrack.reset_id()
    except Exception:  # pragma: no cover -- version differences
        pass

    args = _load_tracker_args()
    fr = max(1, int(round(frame_rate)))

    try:
        params = inspect.signature(BYTETracker.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover -- exotic build
        params = {}

    if "frame_rate" in params:
        tracker = BYTETracker(args, frame_rate=fr)
    else:
        # Older/newer builds that don't take frame_rate as a constructor
        # arg. Try positional as a fallback, then no-frame_rate at all.
        try:
            tracker = BYTETracker(args, fr)
        except TypeError:
            tracker = BYTETracker(args)
            for attr in ("frame_rate", "fps"):
                if hasattr(tracker, attr):
                    setattr(tracker, attr, fr)

    return tracker


def update_player_tracks(tracker, player_dets: List[Detection], frame=None) -> List[Detection]:
    """
    Feeds player-type detections to a BYTETracker and returns them again with
    persistent track_ids. Split out from the video loop so it can be unit
    tested without a video or a YOLO model.
    """
    names: List[str] = []
    cls_ids = []
    for d in player_dets:
        if d.cls_name not in names:
            names.append(d.cls_name)
        cls_ids.append(names.index(d.cls_name))

    if player_dets:
        xywh = np.array([[d.x, d.y, d.w, d.h] for d in player_dets], dtype=np.float32)
        conf = np.array([d.confidence for d in player_dets], dtype=np.float32)
        cls = np.array(cls_ids, dtype=np.float32)
    else:
        xywh = np.zeros((0, 4), dtype=np.float32)
        conf = np.zeros((0,), dtype=np.float32)
        cls = np.zeros((0,), dtype=np.float32)

    det = _TrackInputs(xywh=xywh, conf=conf, cls=cls)
    tracks = tracker.update(det, frame)

    out: List[Detection] = []
    arr = np.asarray(tracks)
    if arr.size == 0:
        return out
    for row in arr.reshape(-1, arr.shape[-1]):
        x1, y1, x2, y2 = (float(v) for v in row[:4])
        track_id = int(row[4])
        conf_v = float(row[5])
        cls_idx = int(row[6])
        if not (0 <= cls_idx < len(names)):
            continue
        out.append(Detection(
            track_id=track_id, cls_name=names[cls_idx], confidence=conf_v,
            x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0, w=x2 - x1, h=y2 - y1,
        ))
    return out


def iter_tracked_frames(detector, source: str, stride: int = 1,
                        max_frames: int = None,
                        start_sec: float = 0.0) -> Iterator[TrackedFrame]:
    """
    Reads `source` (file path or RTSP URL), runs YOLO on every `stride`-th
    frame, and yields a TrackedFrame per processed frame.

    stride: process every Nth frame. Use it to keep CPU inference near
        real-time on 50/60fps footage (stride=2 -> ~30 processed fps). The
        tracker is told the effective frame rate, so IDs survive striding.
        NEVER stride so far that a player moves more than ~a box width
        between processed frames (i.e. keep effective fps >= ~15) or
        IoU matching cannot link them and every frame births new IDs.
    """
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 240:   # 0 / NaN / nonsense (RTSP)
        fps = 30.0
    stride = max(1, int(stride))
    tracker = _make_tracker(fps / stride)

    frame_idx = 0
    emitted = 0
    if start_sec and start_sec > 0:
        # Skip broadcast graphics / line-up cards at the start of a clip.
        # frame_idx stays in ORIGINAL-video units so timestamps remain correct.
        frame_idx = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    try:
        while True:
            if not cap.grab():          # grab() is cheap: skipped frames are not decoded fully
                break
            if frame_idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                frame_h, frame_w = frame.shape[:2]

                raw = detector.detect_frame(frame)
                balls = sorted((d for d in raw if d.cls_name == "ball"),
                               key=lambda d: d.confidence, reverse=True)[:MAX_BALL_CANDIDATES]
                others = [d for d in raw if d.cls_name != "ball"]
                tracked = update_player_tracks(tracker, others, frame)

                yield TrackedFrame(
                    frame_idx=frame_idx,
                    timestamp_sec=frame_idx / fps,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    detections=tracked + balls,
                )
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()


def track_video(detector, source: str, stride: int = 1) -> Iterator[List[Detection]]:
    """
    Backward-compatible wrapper: yields just the List[Detection] per frame,
    same as before. Prefer iter_tracked_frames() when you also need the
    frame dimensions / timestamp.
    """
    for tf in iter_tracked_frames(detector, source, stride=stride):
        yield tf.detections


def verify_tracking_stability(track_history: dict, min_consecutive_frames: int = 100) -> dict:
    """
    Sanity check described in the Phase 1 doc Section 4.3: confirm the
    ball (or any tracked object) keeps the SAME numeric ID for at least
    `min_consecutive_frames` consecutive frames. If IDs flicker (same
    visible object jumping between IDs), the underlying YOLO detections
    are too unreliable for ByteTrack to confidently link them.

    track_history: {track_id: [frame_id, frame_id, ...]} -- frames where
        that ID appeared, in order.

    Returns a report dict you can print/log to manually verify Phase 1's
    exit gate before moving on to Phase 2.
    """
    report = {}
    for track_id, frames in track_history.items():
        if not frames:
            continue
        longest_run = 1
        current_run = 1
        for i in range(1, len(frames)):
            if frames[i] == frames[i - 1] + 1:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
        report[track_id] = {
            "total_appearances": len(frames),
            "longest_consecutive_run": longest_run,
            "stable": longest_run >= min_consecutive_frames,
        }
    return report