"""
Prints every 'ball' candidate YOLO produced near given timestamps, with
confidence and box size, and marks which one the pipeline actually used.

    python phase1/dump_ball_candidates.py --clip Final --times 257 258 259 263 264
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--times", type=float, nargs="+", required=True)
    args = ap.parse_args()

    scenes = json.loads((ROOT / "outputs" / f"{args.clip}_scene_states.json").read_text())
    t0, t1 = scenes[0]["timestamp_sec"], scenes[-1]["timestamp_sec"]
    for t in args.times:
        if not (t0 <= t <= t1):
            print(f"\n{t:g}s: outside processed range {t0:.1f}-{t1:.1f}s"); continue
        s = min(scenes, key=lambda s: abs(s["timestamp_sec"] - t))
        balls = [d for d in s["detections"] if d["cls_name"] == "ball"]
        print(f"\n{s['timestamp_sec']:.2f}s  used ball_position={s['ball_position']}  "
              f"ball_detected={s['ball_detected']}  missing_frames={s['ball_missing_frames']}  "
              f"contact={s['contact_flag']}")
        if not balls:
            print("   (no ball candidates)"); continue
        for d in sorted(balls, key=lambda d: -d["confidence"]):
            print(f"   candidate: x={d['x']:.0f} y={d['y']:.0f}  size={d['w']:.0f}x{d['h']:.0f}px  conf={d['confidence']:.2f}")


if __name__ == "__main__":
    main()
