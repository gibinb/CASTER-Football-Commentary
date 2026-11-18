"""
Finds stretches of live play in a broadcast clip so you can pick a good --start.
No YOLO needed, two checks per sampled frame:

  1. Pitch-green ratio. Line-up cards, adverts, close-ups and replays are
     mostly not green; wide live shots are typically 60-90% green (a
     line-up card measured ~10%).
  2. A "graphic/stoppage" filter, added after the green check alone was
     found to accept offside-review freeze-frames and split-screen replays:
     these are still mostly green, but carry either an OFFSIDE-style dark
     text banner (lower-middle of frame) or a highly saturated cyan/teal
     line overlay (the automated offside line), which plain turf never has.
     A sampled frame only counts as "live" if it is green AND clear of both.

Calibrated on 14 real frames from this project's own test clips (FIFA World
Cup broadcast graphics): the cyan/dark-band thresholds below cleanly
separated 5 known graphic frames from 9 known real-play frames. That is a
small, single-broadcast-style sample -- treat this filter as a heuristic,
not a guarantee, especially on a different broadcaster's graphics.

Usage (from project root):
    python phase1/find_live_play.py --input data/raw/Final.mp4
    python phase1/find_live_play.py --input data/raw/Final.mp4 --max-minutes 30 --min-green 0.4
    python phase1/find_live_play.py --input data/raw/Final.mp4 --no-graphic-filter   # old behaviour
"""
import argparse
import cv2
import numpy as np


def green_ratio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(cv2.inRange(hsv, (35, 40, 30), (85, 255, 255)).mean() / 255.0)


def graphic_signals(frame, cyan_lo, cyan_hi):
    """
    Returns (cyan_ratio, dark_band_frac) for one frame.

    cyan_ratio: fraction of pixels in a narrow cyan/teal hue band, distinctly
    cooler than normal turf green (which skews yellow-green even with
    lighter/darker mowing stripes). This is the automated offside-line overlay.

    dark_band_frac: fraction of rows, within a lower-middle crop (avoids the
    always-on scoreboard, which sits top-left during live play too), whose
    average brightness is very low -- the dark pill/bar behind OFFSIDE-style
    banner text.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(hsv, (cyan_lo, 120, 100), (cyan_hi, 255, 255))
    cyan_ratio = float(cyan.mean() / 255.0)

    y0, y1 = int(h * 0.55), int(h * 0.95)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    region = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    row_means = region.mean(axis=1)
    dark_band_frac = float((row_means < 75).sum() / max(1, region.shape[0]))

    return cyan_ratio, dark_band_frac


def hist(frame):
    h = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-minutes", type=float, default=15.0, help="how far into the video to scan")
    ap.add_argument("--samples-per-sec", type=float, default=2.0)
    ap.add_argument("--min-green", type=float, default=0.40, help="green ratio that counts as live play")
    ap.add_argument("--min-seconds", type=float, default=10.0, help="shortest stretch worth reporting")
    ap.add_argument("--cut-corr", type=float, default=0.5, help="histogram correlation below this = camera cut")
    ap.add_argument("--no-graphic-filter", action="store_true",
                    help="disable the offside-banner/overlay filter (old green-only behaviour)")
    ap.add_argument("--cyan-ratio-threshold", type=float, default=0.02,
                    help="cyan-overlay fraction above this -> flagged as a graphic frame")
    ap.add_argument("--dark-band-threshold", type=float, default=0.08,
                    help="dark-band fraction above this -> flagged as a graphic frame")
    ap.add_argument("--cyan-hue-range", type=int, nargs=2, default=[85, 105],
                    help="OpenCV hue range (0-179) treated as the offside-line cyan/teal")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    scan_frames = int(min(total, args.max_minutes * 60 * fps)) if total > 0 else int(args.max_minutes * 60 * fps)
    step = max(1, int(round(fps / args.samples_per_sec)))

    times, greens, graphic_flags, cuts = [], [], [], []
    prev_h = None
    for i in range(scan_frames):
        if i % step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
        h = hist(small)
        cuts.append(prev_h is not None and cv2.compareHist(prev_h, h, cv2.HISTCMP_CORREL) < args.cut_corr)
        prev_h = h
        times.append(i / fps)
        greens.append(green_ratio(small))
        if args.no_graphic_filter:
            graphic_flags.append(False)
        else:
            cyan_r, dark_f = graphic_signals(small, args.cyan_hue_range[0], args.cyan_hue_range[1])
            graphic_flags.append(cyan_r > args.cyan_ratio_threshold or dark_f > args.dark_band_threshold)
    cap.release()
    if not times:
        raise SystemExit("no frames read")

    green_enough = [g >= args.min_green for g in greens]
    live = [ge and not gr for ge, gr in zip(green_enough, graphic_flags)]
    graphic_only = [ge and gr for ge, gr in zip(green_enough, graphic_flags)]  # green but filtered out

    print(f"{args.input}: {fps:.2f} fps, scanned {times[-1]:.0f}s, "
          f"{100*sum(live)/len(live):.0f}% of samples look like live play (green >= {args.min_green}"
          + ("" if args.no_graphic_filter else ", graphic/offside-overlay excluded") + ")")
    if not args.no_graphic_filter and sum(graphic_only):
        print(f"({100*sum(graphic_only)/len(live):.0f}% more were green but flagged as a graphic/stoppage "
              f"frame -- offside banner or review overlay -- and excluded)")

    # one character per 5 seconds: '#' mostly live, '!' mostly graphic/stoppage, '.' mostly not live/not green
    bucket = 5.0
    line = ""
    for b in range(int(times[-1] // bucket) + 1):
        idx = [k for k, t in enumerate(times) if b * bucket <= t < (b + 1) * bucket]
        if not idx:
            line += "."
        elif sum(live[k] for k in idx) / len(idx) >= 0.5:
            line += "#"
        elif sum(graphic_only[k] for k in idx) / len(idx) >= 0.5:
            line += "!"
        else:
            line += "."
    label = "# live, ! graphic/stoppage (offside review etc.), . not live" if not args.no_graphic_filter else "# live, . not live"
    print(f"\ntimeline (1 char = 5 s, {label}):")
    for k in range(0, len(line), 60):
        print(f"  {k*5//60:>3d}:{(k*5)%60:02d}  {line[k:k+60]}")

    # contiguous live stretches (allow 1.5 s of non-live noise inside)
    gap_allow = int(1.5 * args.samples_per_sec)
    segs, start, last_live = [], None, None
    for k, is_live in enumerate(live):
        if is_live:
            if start is None:
                start = k
            last_live = k
        elif start is not None and k - last_live > gap_allow:
            segs.append((start, last_live)); start = None
    if start is not None:
        segs.append((start, last_live))

    rows = []
    for a, b in segs:
        dur = times[b] - times[a]
        if dur >= args.min_seconds:
            n_cuts = sum(cuts[a + 1:b + 1])
            rows.append((a, b, dur, float(np.mean(greens[a:b + 1])), n_cuts))
    rows.sort(key=lambda r: r[2], reverse=True)

    if not rows:
        print(f"\nNo live stretch >= {args.min_seconds:.0f}s found. Try --max-minutes larger, --min-green 0.3, "
              f"or --no-graphic-filter if you suspect the new filter is too strict for this broadcast.")
        return
    print(f"\nlongest live-play stretches (cuts = hard camera cuts inside the stretch):")
    print(f"  {'start':>7} {'end':>7} {'len':>6} {'green':>6} {'cuts':>5} {'cuts/10s':>8}")
    for a, b, dur, g, c in rows[:6]:
        print(f"  {times[a]:7.0f} {times[b]:7.0f} {dur:5.0f}s {g:6.2f} {c:5d} {10*c/dur:8.1f}")
    best = rows[0]
    print(f"\nsuggested: --start {int(times[best[0]])}   (stretch is ~{best[2]:.0f}s long; "
          f"stretches with few cuts give the cleanest tracking numbers)")


if __name__ == "__main__":
    main()