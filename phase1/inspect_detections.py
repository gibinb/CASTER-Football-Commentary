"""
Visual check for player/goalkeeper/referee (and ball, for consistency)
detections -- the same kind of check inspect_ball.py gave the ball class,
extended to every class. Reads an already-saved outputs/<clip>_scene_states.json
(run run_phase1.py first) rather than re-running YOLO, since every frame's
full detection list -- not just the ball -- is already saved there.

Usage (from project root):
    python phase1/inspect_detections.py --clip Final --video data/raw/Final.mp4 --snap 258 259
    python phase1/inspect_detections.py --clip Final --video data/raw/Final.mp4 --snap 258 --classes player

For a rare class (goalkeeper, referee), manually guessed --snap times will
usually miss it entirely -- a class with 100 detections across 300 frames
still only shows up in ~1 of every 10 one-second-spaced samples. Use
--auto-sample instead: it finds every frame where that class was actually
detected and picks a spread across the FULL CONFIDENCE RANGE (lowest,
~25th percentile, median, ~75th percentile, highest) automatically, so you
see both the strongest and weakest real detections rather than whatever
you happened to guess:

    python phase1/inspect_detections.py --clip Final --video data/raw/Final.mp4 \
        --auto-sample referee --auto-sample-n 8
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COLORS = {   # BGR, kept visually distinct from inspect_ball.py's red/yellow ball markers
    "ball": (0, 0, 255),         # red
    "player": (0, 220, 0),        # green
    "goalkeeper": (255, 140, 0),  # orange
    "referee": (0, 255, 255),     # yellow
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="name without extension, e.g. Final")
    ap.add_argument("--video", help="original video, only needed for --snap")
    ap.add_argument("--snap", type=float, nargs="*", default=[],
                    help="timestamps (seconds) to save annotated frames for")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="only draw these classes (default: all -- ball, player, goalkeeper, referee)")
    ap.add_argument("--auto-sample", default=None,
                    help="class name to auto-sample (e.g. referee, goalkeeper) -- finds every frame where "
                         "that class was detected and snaps a spread across its confidence range, "
                         "instead of you having to guess timestamps that happen to hit a rare class")
    ap.add_argument("--auto-sample-n", type=int, default=6,
                    help="how many frames to sample across the confidence range (default 6)")
    args = ap.parse_args()

    scenes = json.loads((ROOT / "outputs" / f"{args.clip}_scene_states.json").read_text())
    t0, t1 = scenes[0]["timestamp_sec"], scenes[-1]["timestamp_sec"]

    counts = Counter()
    conf_sum = Counter()
    for s in scenes:
        for d in s["detections"]:
            counts[d["cls_name"]] += 1
            conf_sum[d["cls_name"]] += d["confidence"]

    print(f"frames: {len(scenes)}   processed time range: {t0:.1f}s - {t1:.1f}s\n")
    print(f"{'class':<12} {'detections':>10} {'per frame':>10} {'avg conf':>9}")
    for cls in ("ball", "player", "goalkeeper", "referee"):
        n = counts.get(cls, 0)
        avg_conf = conf_sum[cls] / n if n else 0.0
        print(f"{cls:<12} {n:>10} {n/len(scenes):>10.2f} {avg_conf:>9.2f}")
    others = set(counts) - {"ball", "player", "goalkeeper", "referee"}
    if others:
        print(f"\nWARNING: unexpected class names seen: {sorted(others)} -- check class aliasing in yolo_detector.py")

    snap_times = list(args.snap)
    if args.auto_sample:
        hits = []  # (timestamp_sec, confidence) for every frame with >=1 detection of this class
        for s in scenes:
            best_conf = max((d["confidence"] for d in s["detections"] if d["cls_name"] == args.auto_sample),
                            default=None)
            if best_conf is not None:
                hits.append((s["timestamp_sec"], best_conf))
        if not hits:
            print(f"\n'{args.auto_sample}' was never detected in this run -- nothing to sample.")
            return
        hits.sort(key=lambda h: h[1])   # sort by confidence, low to high
        n = min(args.auto_sample_n, len(hits))
        # evenly spaced INDICES across the sorted-by-confidence list, so we see the
        # weakest, strongest, and a spread in between -- not just whatever we'd get
        # from picking frames in time order.
        idxs = sorted(set(round(i * (len(hits) - 1) / max(1, n - 1)) for i in range(n)))
        picked = [hits[i] for i in idxs]
        print(f"\n'{args.auto_sample}': {len(hits)} frames had a detection "
              f"(confidence range {hits[0][1]:.2f}-{hits[-1][1]:.2f}). Sampling {len(picked)} across that range:")
        for t, c in picked:
            print(f"  {t:.2f}s  conf={c:.2f}")
        snap_times = [t for t, _ in picked]
        if args.classes is None:
            args.classes = [args.auto_sample]   # keep the auto-sampled frames readable -- just this class

    if not snap_times:
        return
    if not args.video:
        print("\n--video is required to save --snap/--auto-sample frames")
        return

    import cv2
    cap = cv2.VideoCapture(args.video)
    out_dir = ROOT / "outputs" / "detection_snaps"
    out_dir.mkdir(exist_ok=True)
    wanted = set(args.classes) if args.classes else None

    print()
    for t in snap_times:
        if t < t0 or t > t1:
            print(f"skipped {t:g}s: outside the processed range {t0:.1f}-{t1:.1f}s")
            continue
        s = min(scenes, key=lambda s: abs(s["timestamp_sec"] - t))
        cap.set(cv2.CAP_PROP_POS_MSEC, s["timestamp_sec"] * 1000)
        ok, frame = cap.read()
        if not ok:
            print(f"could not read frame at {t}s")
            continue

        drawn = 0
        for d in s["detections"]:
            if wanted and d["cls_name"] not in wanted:
                continue
            color = COLORS.get(d["cls_name"], (255, 255, 255))
            x, y, w, h = d["x"], d["y"], d["w"], d["h"]
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d['cls_name']} {d['confidence']:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            drawn += 1

        suffix = f"_{'-'.join(sorted(wanted))}" if wanted else ""
        path = out_dir / f"{args.clip}_{t:g}s{suffix}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"saved {path}  ({drawn} boxes drawn, actual frame time {s['timestamp_sec']:.2f}s)")
    cap.release()


if __name__ == "__main__":
    main()