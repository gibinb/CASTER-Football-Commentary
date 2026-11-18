"""
Compares ball-detection quality at several CASTER_IMGSZ values on the SAME
clip/stretch, so you can see the actual accuracy-vs-speed tradeoff instead
of guessing. Shells out to check_tracking.py once per imgsz value (keeps
this script simple and guarantees it measures exactly what the real
pipeline does, not a reimplementation of it).

Usage (from project root):
    python phase1/imgsz_sweep.py --input data/raw/Final.mp4 --start 255 --frames 300
    python phase1/imgsz_sweep.py --input data/raw/Final.mp4 --start 255 --frames 300 --sizes 960 1280 1536
"""
import argparse
import re
import subprocess
import sys
import time


def run_one(input_path, start, frames, imgsz):
    env_assign = f"CASTER_IMGSZ={imgsz}"
    cmd = [sys.executable, "phase1/check_tracking.py",
           "--input", input_path, "--start", str(start), "--frames", str(frames)]
    import os
    env = os.environ.copy()
    env["CASTER_IMGSZ"] = str(imgsz)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    out = result.stdout + result.stderr
    ball_m = re.search(r"ball visible\s*:\s*(\d+)% of frames", out)
    ids_m = re.search(r"unique player IDs:\s*(\d+)", out)
    if ball_m is None:
        print(f"--- imgsz={imgsz} produced no parseable output, raw log below ---")
        print(out[-2000:])
        return None
    return {
        "imgsz": imgsz,
        "ball_pct": int(ball_m.group(1)),
        "ids": int(ids_m.group(1)) if ids_m else None,
        "seconds": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--sizes", type=int, nargs="+", default=[960, 1280, 1536])
    args = ap.parse_args()

    rows = []
    for sz in args.sizes:
        print(f"running imgsz={sz} ...")
        r = run_one(args.input, args.start, args.frames, sz)
        if r:
            rows.append(r)
            print(f"  -> ball visible {r['ball_pct']}%, {r['ids']} player IDs, {r['seconds']:.0f}s wall time")

    if not rows:
        print("no runs produced usable output")
        return

    print(f"\n{'imgsz':>6} {'ball%':>7} {'ids':>5} {'seconds':>8} {'s/10s-clip':>11}")
    for r in rows:
        per_10s = r["seconds"] / (args.frames / 30.0) * 10.0
        print(f"{r['imgsz']:6d} {r['ball_pct']:6d}% {r['ids']:5} {r['seconds']:8.0f} {per_10s:10.0f}s")

    baseline = rows[0]
    best = max(rows, key=lambda r: r["ball_pct"])
    if best["imgsz"] != baseline["imgsz"]:
        gain = best["ball_pct"] - baseline["ball_pct"]
        slowdown = best["seconds"] / baseline["seconds"]
        print(f"\nimgsz={best['imgsz']} found the ball {gain:+d} percentage points more often than "
              f"imgsz={baseline['imgsz']}, at {slowdown:.1f}x the processing time. "
              f"Decide whether that trade is worth it for your use case.")
    else:
        print(f"\nThe smallest size tested ({baseline['imgsz']}) already did best (or tied) on ball "
              f"visibility -- going larger isn't paying off further on this clip.")


if __name__ == "__main__":
    main()
