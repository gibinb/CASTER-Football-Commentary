"""
Experiment: does sliced (tiled) inference find the ball in frames where a
normal full-frame pass misses it?

The ball is often only a few pixels wide in a 1920px-wide broadcast frame.
SAHI (https://github.com/obss/sahi) runs detection on overlapping tiles of
the full-resolution frame instead of one shrunk-down pass, which can help
recall on small objects -- at a real cost in latency, since each tile is a
separate inference call. This script measures BOTH sides of that trade on
a handful of real sampled frames, using your EXISTING fine-tuned
weights (no retraining), before deciding whether it's worth wiring into
the main pipeline.

Install (once):
    pip install sahi

Usage (from project root):
    python phase1/sahi_ball_eval.py --input data/raw/Final.mp4 --start 255 --num-frames 20
    python phase1/sahi_ball_eval.py --input data/raw/Final.mp4 --start 255 --num-frames 20 \
        --slice-size 640 --overlap 0.2
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import WEIGHTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip to begin sampling")
    ap.add_argument("--num-frames", type=int, default=20, help="how many frames to sample and compare")
    ap.add_argument("--sample-every-sec", type=float, default=0.5)
    ap.add_argument("--slice-size", type=int, default=640, help="tile width/height in pixels")
    ap.add_argument("--overlap", type=float, default=0.2, help="fraction of overlap between tiles")
    ap.add_argument("--weights", default=None, help="defaults to weights/football.pt")
    ap.add_argument("--normal-imgsz", type=int, default=1280)
    ap.add_argument("--save-annotated", action="store_true",
                    help="save a frame to outputs/sahi_snaps/ for every sliced-only ball candidate above "
                         "the extractor's BALL_ACQUIRE_MIN_CONF (0.25) -- these are the only sliced-only hits "
                         "that would actually change the pipeline's output, so they're the ones worth eyeballing")
    args = ap.parse_args()

    out_dir = None
    if args.save_annotated:
        out_dir = Path(WEIGHTS_DIR).parent.parent / "outputs" / "sahi_snaps"
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed -- this needs the same environment as the rest of Phase 1.")
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        raise SystemExit("sahi not installed -- run: pip install sahi")

    weights_path = args.weights or str(WEIGHTS_DIR / "football.pt")
    if not Path(weights_path).exists():
        raise SystemExit(f"weights not found at {weights_path}")

    normal_model = YOLO(weights_path)
    sliced_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights_path,
        confidence_threshold=0.1,   # low floor here on purpose: we want to SEE every raw candidate,
        device="cpu",               # the real pipeline's own confidence gate decides what to trust
    )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    step = max(1, int(args.sample_every_sec * fps))

    rows = []
    frame_idx = 0
    sampled = 0
    while sampled < args.num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            t_sec = args.start + frame_idx / fps

            t0 = time.perf_counter()
            normal_res = normal_model.predict(frame, imgsz=args.normal_imgsz, conf=0.1, verbose=False)[0]
            normal_time = time.perf_counter() - t0
            normal_balls = []
            for box in normal_res.boxes:
                name = normal_res.names[int(box.cls)]
                if name == "ball":
                    conf = float(box.conf)
                    normal_balls.append(conf)

            t0 = time.perf_counter()
            sliced_res = get_sliced_prediction(
                frame, sliced_model,
                slice_height=args.slice_size, slice_width=args.slice_size,
                overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                verbose=0,
            )
            sliced_time = time.perf_counter() - t0
            sliced_balls = [p.score.value for p in sliced_res.object_prediction_list
                            if p.category.name == "ball"]

            sliced_detections = [p for p in sliced_res.object_prediction_list if p.category.name == "ball"]
            rows.append({
                "t": t_sec,
                "normal_n": len(normal_balls), "normal_best": max(normal_balls, default=0.0),
                "normal_time": normal_time,
                "sliced_n": len(sliced_balls), "sliced_best": max(sliced_balls, default=0.0),
                "sliced_time": sliced_time,
            })

            # A sliced-only candidate only matters if it's above the floor the real extractor
            # would require to even consider it a pending ball candidate.
            if args.save_annotated and len(normal_balls) == 0 and sliced_detections:
                best = max(sliced_detections, key=lambda p: p.score.value)
                if best.score.value >= 0.25:
                    annotated = frame.copy()
                    for p in sliced_detections:
                        b = p.bbox
                        cx, cy = int((b.minx + b.maxx) / 2), int((b.miny + b.maxy) / 2)
                        cv2.circle(annotated, (cx, cy), 18, (0, 0, 255), 3)
                        cv2.putText(annotated, f"{p.score.value:.2f}", (cx + 20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    path = out_dir / f"sahi_sliced_only_{t_sec:g}s.jpg"
                    cv2.imwrite(str(path), annotated)
                    print(f"  saved {path}  (red circles = every sliced ball candidate this frame, "
                          f"confidence labelled)")
            print(f"[{t_sec:6.1f}s] normal: {len(normal_balls)} ball cand (best={max(normal_balls, default=0.0):.2f}, "
                  f"{normal_time*1000:.0f}ms)   sliced: {len(sliced_balls)} ball cand "
                  f"(best={max(sliced_balls, default=0.0):.2f}, {sliced_time*1000:.0f}ms)")
            sampled += 1
        frame_idx += 1
    cap.release()

    if not rows:
        print("no frames sampled -- check --start is inside the video")
        return

    normal_found = sum(1 for r in rows if r["normal_n"] > 0)
    sliced_found = sum(1 for r in rows if r["sliced_n"] > 0)
    sliced_only = sum(1 for r in rows if r["normal_n"] == 0 and r["sliced_n"] > 0)
    normal_only = sum(1 for r in rows if r["normal_n"] > 0 and r["sliced_n"] == 0)
    avg_normal_t = sum(r["normal_time"] for r in rows) / len(rows)
    avg_sliced_t = sum(r["sliced_time"] for r in rows) / len(rows)

    print(f"\n{len(rows)} frames sampled.")
    print(f"normal pass found >=1 ball candidate in {normal_found}/{len(rows)} frames")
    print(f"sliced pass found >=1 ball candidate in {sliced_found}/{len(rows)} frames")
    print(f"frames where ONLY sliced found a candidate: {sliced_only}")
    print(f"frames where ONLY normal found a candidate: {normal_only}")
    print(f"avg time per frame -- normal: {avg_normal_t*1000:.0f}ms   sliced: {avg_sliced_t*1000:.0f}ms "
          f"({avg_sliced_t/avg_normal_t:.1f}x slower)")

    if sliced_found > normal_found:
        print(f"\nSliced inference found more candidates on this sample. Worth checking whether the EXTRA "
              f"candidates it finds are real balls (not just more false positives) before wiring this into "
              f"the main pipeline -- more raw candidates only helps if they're correct ones.")
    else:
        print(f"\nSliced inference did not find more candidates on this sample. Given the "
              f"{avg_sliced_t/avg_normal_t:.1f}x slowdown, it's probably not worth integrating as-is.")


if __name__ == "__main__":
    main()