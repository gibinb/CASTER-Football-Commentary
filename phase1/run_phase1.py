"""
CASTER Phase 1 -- end-to-end demo script (football only).

Runs the full Phase 1 pipeline on a football clip:
    YOLOv8 detection -> ByteTrack player tracking -> 12-number feature
    vectors -> scene state JSON, printed live and saved to outputs/.

Usage (from the project root):
    python phase1/run_phase1.py --input data/raw/your_clip.mp4
    python phase1/run_phase1.py --input data/raw/your_clip.mp4 --start 255 --max-frames 300
    python phase1/run_phase1.py --input data/raw/your_clip.mp4 --stride 2      # 60 fps footage
    python phase1/run_phase1.py --input data/raw/your_clip.mp4 --weights phase1/weights/football2.pt

IMPORTANT: broadcast clips begin with line-up cards, adverts and replays.
The detector will happily find "balls" and "players" on a graphic, and every
number it produces there is meaningless. Use --start to jump into live play;
`python phase1/find_live_play.py --input <clip>` suggests a good value.

This is also the Phase 1 "exit gate": if this runs cleanly on a stretch of
live play, with plausible feature numbers and no crashes, Phase 1 is done.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR
from schemas import FeatureWindow

from phase1.detection.yolo_detector import FootballDetector, INFER_IMGSZ
from phase1.features.feature_extractor import FeatureExtractor
from phase1.scene_state.scene_state_builder import build_scene_state
from phase1.tracking.bytetrack_wrapper import verify_tracking_stability, iter_tracked_frames


def run(video_path: str, max_frames: int = None, weights_override: str = None,
        stride: int = None, start_sec: float = 0.0):
    print(f"\nCASTER Phase 1 -- processing: {video_path}\n")

    # --- Stage 1: load the detector --------------------------------------
    print("Stage 1/4: loading the football detector...")
    detector = FootballDetector(weights_path=weights_override)
    if weights_override:
        print(f"Using overridden weights: {weights_override}")
    print(f"imgsz={INFER_IMGSZ}  weights={'fine-tuned' if detector.using_finetuned_weights else 'COCO fallback (yolov8n.pt)'}\n")

    if not detector.using_finetuned_weights:
        print(
            "NOTE: running on pretrained yolov8n.pt (COCO classes), not the fine-tuned "
            "football model. Every detected COCO 'person' counts as a player, but there "
            "is no goalkeeper or referee class, and ball detection will be much weaker. "
            "Put your fine-tuned model at phase1/weights/football.pt.\n"
        )

    # --- Stage 2: YOLOv8 detection + ByteTrack tracking ------------------
    print("Stage 2/4: running YOLOv8 + ByteTrack...")
    extractor = FeatureExtractor()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # Process ~30 frames/second whatever the source frame rate. The tracker
    # needs CONSECUTIVE frames: never stride so far that players jump more
    # than a box width between processed frames (keep effective fps >= ~15).
    if stride is None:
        stride = max(1, round(src_fps / 30.0))
    print(f"Source {src_fps:.2f} fps, stride={stride} -> ~{src_fps / stride:.1f} processed fps"
          + (f", starting at {start_sec:g}s" if start_sec else "") + "\n")

    track_history = {}  # track_id -> list of processed-frame indices, for the stability check
    scene_states = []
    feature_windows = []

    frame_id = 0
    for tf in iter_tracked_frames(detector, video_path, stride=stride, max_frames=max_frames,
                                  start_sec=start_sec):
        detections = tf.detections
        frame_w, frame_h = tf.frame_w, tf.frame_h
        timestamp_sec = tf.timestamp_sec
        for d in detections:
            if d.track_id is not None:
                track_history.setdefault(d.track_id, []).append(frame_id)

        feature_vector = extractor.extract(detections, frame_w, frame_h)

        diagnostics = extractor.get_tracking_diagnostics()
        scene = build_scene_state(frame_id, timestamp_sec, detections, feature_vector,
                                  frame_w=frame_w, frame_h=frame_h,
                                  tracking_diagnostics=diagnostics)
        scene_states.append(scene)

        if extractor.window_is_full():
            feature_windows.append(FeatureWindow(
                sport=scene.sport,
                end_frame_id=frame_id,
                end_timestamp_sec=timestamp_sec,
                vectors=extractor.get_window(),
            ))

        if frame_id % 30 == 0:
            print(
                f"[{timestamp_sec:6.1f}s] ball=({scene.ball_position[0]:.2f},{scene.ball_position[1]:.2f}) "
                f"speed={scene.ball_speed:.3f} zone={scene.pitch_zone} "
                f"boundary={scene.boundary_proximity} contact={scene.contact_flag} "
                f"ball_detected={scene.ball_detected} ball_id={scene.ball_track_id} "
                f"missing_frames={scene.ball_missing_frames} candidates={scene.ball_candidate_count}"
            )

        frame_id += 1

    print(f"\nProcessed {frame_id} frames.\n")
    if frame_id == 0:
        print("No frames were processed -- check the input path and --start value.")
        return

    # --- Stage 3: tracking stability check (Phase 1 exit-gate check) -----
    print("Stage 3/4: verifying tracking stability...")
    stability_report = verify_tracking_stability(track_history)
    stable_count = sum(1 for r in stability_report.values() if r["stable"])
    print(f"{stable_count}/{len(stability_report)} tracked objects held a stable ID for 100+ consecutive frames.\n")
    ball_pct = 100.0 * sum(1 for st in scene_states if st.ball_detected) / len(scene_states)
    print(f"Unique track IDs created: {len(track_history)}  |  ball visible in {ball_pct:.0f}% of processed frames")
    if ball_pct < 30:
        print("WARNING: the ball is missing in most frames. The event classifier cannot work "
              "from that -- see README 'Known limitations'.\n")

    # --- Stage 4: save outputs --------------------------------------------
    print("Stage 4/4: saving outputs...")
    out_path = OUTPUTS_DIR / f"{Path(video_path).stem}_scene_states.json"
    with open(out_path, "w") as f:
        json.dump([s.to_dict() for s in scene_states], f, indent=2, default=str)
    print(f"Saved {len(scene_states)} scene states to {out_path}")

    windows_path = OUTPUTS_DIR / f"{Path(video_path).stem}_feature_windows.json"
    with open(windows_path, "w") as f:
        json.dump(
            [{"sport": w.sport, "end_frame_id": w.end_frame_id,
              "end_timestamp_sec": w.end_timestamp_sec, "vectors": w.vectors}
             for w in feature_windows],
            f, indent=2,
        )
    print(f"Saved {len(feature_windows)} feature windows (shape 30x12 each) to {windows_path}")

    print("\nPhase 1 complete. These feature windows + scene states are exactly")
    print("what Phase 2's event classifier will consume next.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CASTER Phase 1 end-to-end on a football clip")
    parser.add_argument("--input", required=True, help="Path to a football video clip")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on processed frames")
    parser.add_argument("--weights", default=None,
                        help="Use this .pt file instead of phase1/weights/football.pt, e.g. "
                             "--weights phase1/weights/football2.pt to test an alternative "
                             "model without touching the active one.")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start processing at this many seconds into the video "
                             "(skips line-up cards / intros / adverts).")
    parser.add_argument("--stride", type=int, default=None,
                        help="Process every Nth frame. Default: auto (~30 processed fps, so 60fps "
                             "footage uses 2). Keep the effective rate >= ~15 fps or player IDs "
                             "cannot be tracked between frames.")
    args = parser.parse_args()

    run(args.input, args.max_frames, args.weights, args.stride, args.start)
