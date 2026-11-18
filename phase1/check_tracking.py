"""
CASTER Phase 1 -- tracking sanity check (run this FIRST when boxes/IDs look wrong).

    python phase1/check_tracking.py --input data/raw/Semifinal02.mp4 --start 144
    python phase1/check_tracking.py --input data/raw/Semifinal02.mp4 --start 144 --stride 1 --frames 400

Pick --start inside a stretch of live play (python phase1/find_live_play.py --input <clip>
prints suggestions); line-up cards, adverts and replays give meaningless numbers.

It answers three questions in ~1 minute, printing numbers instead of making
you eyeball an overlay:

  1. ID CHURN      -- how many distinct player IDs were created per 100
                      processed frames. A healthy shot has roughly as many
                      IDs as players on screen; hundreds means every frame
                      is birthing new IDs (tracker not linking frames).
  2. LONGEST RUN   -- the longest consecutive-frame run of any single ID.
  3. BALL VISIBLE  -- % of frames with a ball candidate. If this is tiny the
                      event classifier is being fed empty features no matter
                      how good the tracker is: fix the ball detector first
                      (more supplementary labels, higher CASTER_IMGSZ).
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from phase1.detection.yolo_detector import FootballDetector, INFER_IMGSZ, PLAYER_CONF, BALL_CONF
from phase1.tracking.bytetrack_wrapper import iter_tracked_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--start", type=float, default=0.0, help="start at this second")
    ap.add_argument("--frames", type=int, default=300, help="processed frames to examine")
    args = ap.parse_args()

    det = FootballDetector()
    print(f"weights: {'fine-tuned' if det.using_finetuned_weights else 'COCO fallback (yolov8n.pt)'}  "
          f"imgsz={INFER_IMGSZ} player_conf={PLAYER_CONF} ball_conf={BALL_CONF} stride={args.stride}")

    ids_seen, runs, cur = set(), {}, {}
    ball_frames = n = 0
    for tf in iter_tracked_frames(det, args.input, stride=args.stride, max_frames=args.frames,
                              start_sec=args.start):
        n += 1
        present = {d.track_id for d in tf.detections if d.track_id is not None}
        ids_seen |= present
        for i in present:
            cur[i] = cur.get(i, 0) + 1
            runs[i] = max(runs.get(i, 0), cur[i])
        for i in list(cur):
            if i not in present:
                cur[i] = 0
        if any(d.cls_name == "ball" for d in tf.detections):
            ball_frames += 1

    if n == 0:
        print("No frames processed -- check the input path.")
        return
    print(f"\nprocessed frames : {n}")
    print(f"unique player IDs: {len(ids_seen)}   ({100 * len(ids_seen) / n:.1f} new IDs per 100 frames)")
    print(f"longest ID run   : {max(runs.values()) if runs else 0} consecutive frames")
    print(f"ball visible     : {100 * ball_frames / n:.0f}% of frames")
    print()
    if len(ids_seen) > n * 0.5:
        print("-> ID CHURN is high: IDs are not being linked between frames.")
    else:
        print("-> ID churn looks reasonable (camera cuts will still reset IDs).")
    if ball_frames / n < 0.30:
        print("-> BALL is rarely detected: commentary events will be unreliable until this improves.")


if __name__ == "__main__":
    main()