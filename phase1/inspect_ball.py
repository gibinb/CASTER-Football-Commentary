"""
Phase 1 ball sanity check. Reads outputs/<clip>_scene_states.json and answers:
is the "ball" actually moving, or is it a fixed false positive?

Usage (from project root):
    python phase1/inspect_ball.py --clip Semifinal01
    python phase1/inspect_ball.py --clip Semifinal01 --video data/raw/Semifinal01.mp4 --snap 0 3 7
"""
import argparse, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="name without extension, e.g. Semifinal01")
    ap.add_argument("--video", help="original video, only needed for --snap")
    ap.add_argument("--snap", type=float, nargs="*", default=[],
                    help="timestamps (seconds) to save annotated frames for")
    args = ap.parse_args()

    scenes = json.loads((ROOT / "outputs" / f"{args.clip}_scene_states.json").read_text())
    det = [s for s in scenes if s.get("ball_detected")]
    print(f"frames: {len(scenes)}   ball_detected: {len(det)} ({100*len(det)/len(scenes):.0f}%)")

    pos = Counter((round(s["ball_position"][0], 2), round(s["ball_position"][1], 2)) for s in det)
    print("\nmost common ball positions (x, y) -> frames:")
    for p, c in pos.most_common(6):
        print(f"  {p} -> {c} frames ({100*c/max(len(det),1):.0f}% of detected)")

    still = sum(1 for s in det if s["ball_speed"] < 0.002)
    print(f"\ndetected frames with speed ~0 : {still} ({100*still/max(len(det),1):.0f}%)")
    print(f"distinct positions (2 dp)    : {len(pos)}")
    if det and pos.most_common(1)[0][1] / len(det) > 0.30:
        print("-> WARNING: one fixed spot holds >30% of ball detections. Likely a static false"
              " positive (spot, logo, spare ball) or a dead-ball moment. Check the snapshots.")

    t_min, t_max = scenes[0]["timestamp_sec"], scenes[-1]["timestamp_sec"]
    print(f"\nprocessed time range: {t_min:.1f}s - {t_max:.1f}s  (--snap times must fall inside this)")

    if args.snap:
        import cv2
        cap = cv2.VideoCapture(args.video)
        out = ROOT / "outputs" / "ball_snaps"
        out.mkdir(exist_ok=True)
        for t in args.snap:
            if t < t_min or t > t_max:
                print(f"skipped {t:g}s: outside the processed range {t_min:.1f}-{t_max:.1f}s "
                      f"(re-run run_phase1 with a larger --max-frames or different --start)")
                continue
            s = min(scenes, key=lambda s: abs(s["timestamp_sec"] - t))
            cap.set(cv2.CAP_PROP_POS_MSEC, s["timestamp_sec"] * 1000)
            ok, frame = cap.read()
            if not ok:
                print(f"could not read frame at {t}s"); continue
            h, w = frame.shape[:2]
            for d in s["detections"]:
                if d["cls_name"] == "ball":
                    cv2.circle(frame, (int(d["x"]), int(d["y"])), 18, (0, 0, 255), 3)
            bx, by = s["ball_position"]
            cv2.drawMarker(frame, (int(bx * w), int(by * h)), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
            path = out / f"{args.clip}_{t:g}s.jpg"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}  (red circle = YOLO ball box, yellow cross = ball_position used)")


if __name__ == "__main__":
    main()