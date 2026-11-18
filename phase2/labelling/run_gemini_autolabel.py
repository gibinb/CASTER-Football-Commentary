"""
CASTER Phase 2 -- Gemini auto-labelling CLI.

gemini_auto_labeler.py's auto_label_windows() was written and rate-limit
hardened (request pacing, 429 backoff, RateLimitExhausted handling) but
never wired to anything runnable -- nothing in the project called it. This
script is that missing entry point. It reuses label_tool.py's
load_windows() and append_label() UNCHANGED, so rows land in
data/labels/<sport>.jsonl in the exact same shape manual labelling
produces -- xgboost_clf.py doesn't need to know which tool wrote them.

Requires GEMINI_API_KEY in the environment (or .env, if your config loader
reads it). Get a free key at https://aistudio.google.com/apikey.

Usage:
    python phase2/labelling/run_gemini_autolabel.py --windows outputs/Semifinal01_feature_windows.json --video data/raw/Semifinal01.mp4 --sport football

    # Tighter spacing between sampled windows (more labels, more API calls):
    python phase2/labelling/run_gemini_autolabel.py --windows outputs/Semifinal01_feature_windows.json --video data/raw/Semifinal01.mp4 --sport football --stride 30 --max-windows 400

    # Target a specific moment instead of blind whole-clip sampling -- e.g.
    # you scrubbed the video and know a goal happens around 1:23 (83s):
    # tight stride over just that range gets far more useful labels per API
    # call than uniform sampling across a whole clip where rare events
    # (GOAL, OFFSIDE, CORNER_KICK, HEADER) are rare in wall-clock time too.
    python phase2/labelling/run_gemini_autolabel.py --windows outputs/Semifinal01_feature_windows.json --video data/raw/Semifinal01.mp4 --sport football --start-sec 75 --end-sec 95 --stride 30

Rate limits: request_interval defaults to 5.0s (12 req/min), which leaves
headroom under Gemini 2.5 Flash's free-tier 15 RPM cap (confirm your
model's actual cap at https://ai.dev/rate-limit; override with
--request-interval if it differs). A run that hits the DAILY cap stops
early and prints how many labels already made it to disk -- those are
safe; just re-run tomorrow (or point --windows/--video at a different
clip) to keep going. Labels are flushed to disk after every single save
(see label_tool.append_label), so an interrupted run never loses progress.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SPORT_FOOTBALL

from phase2.labelling.label_tool import load_windows, append_label
from phase2.labelling.gemini_auto_labeler import auto_label_windows, AutoLabelError


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True, help="Path to a *_feature_windows.json file from Phase 1")
    ap.add_argument("--video", required=True, help="Path to the source clip (Gemini needs the actual frames, not just the feature vectors)")
    ap.add_argument("--sport", required=True, choices=[SPORT_FOOTBALL])
    ap.add_argument("--stride", type=int, default=30,
                     help="Sample every Nth window (min 30, so sampled windows don't overlap). Lower = more API calls, denser coverage of the clip.")
    ap.add_argument("--min-confidence", type=float, default=0.85,
                     help="Discard a Gemini-proposed label below this confidence (default 0.85 -- deliberately strict, since an auto-labelled mistake trains the classifier on a lie).")
    ap.add_argument("--max-windows", type=int, default=250,
                     help="Hard cap on API calls this run, independent of rate limiting -- keeps one run from silently burning your whole daily quota on one clip.")
    ap.add_argument("--request-interval", type=float, default=5.0,
                     help="Minimum seconds between Gemini API calls. See module docstring for how to tune this against your model's actual RPM cap.")
    ap.add_argument("--start-sec", type=float, default=None,
                     help="Only sample windows ending at or after this many seconds into the clip. "
                          "Combine with --end-sec to target a known moment (e.g. a goal you scrubbed to) "
                          "instead of blind whole-clip sampling.")
    ap.add_argument("--end-sec", type=float, default=None,
                     help="Only sample windows ending at or before this many seconds into the clip.")
    args = ap.parse_args()

    windows = load_windows(args.windows)
    print(f"Loaded {len(windows)} feature windows from {args.windows}")

    if args.start_sec is not None or args.end_sec is not None:
        before = len(windows)
        lo = args.start_sec if args.start_sec is not None else float("-inf")
        hi = args.end_sec if args.end_sec is not None else float("inf")
        windows = [w for w in windows if lo <= w["end_timestamp_sec"] <= hi]
        print(f"Restricted to [{args.start_sec}, {args.end_sec}] seconds: "
              f"{before} -> {len(windows)} windows")
        if not windows:
            print("ERROR: no windows in that time range -- check your --start-sec/--end-sec against "
                  "the clip's actual length.")
            sys.exit(1)

    print(f"Sport: {args.sport}  |  sampling every {args.stride} windows  |  "
          f"min_confidence={args.min_confidence}  |  max_windows={args.max_windows}\n")

    try:
        saved, skipped, stopped_early, already_covered = auto_label_windows(
            windows,
            sport=args.sport,
            video_path=args.video,
            append_label=lambda sport, event, window, source_clip: append_label(sport, event, window, source_clip),
            stride=args.stride,
            min_confidence=args.min_confidence,
            max_windows=args.max_windows,
            request_interval=args.request_interval,
        )
    except AutoLabelError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"\n{saved} label(s) saved, {skipped} window(s) skipped (low confidence / "
          f"no event / too close to a prior save of the same event), "
          f"{already_covered} window(s) already labelled in an earlier session (no API call made).")
    if stopped_early:
        print("Run stopped early due to a rate limit -- see the message above for when to retry.")
    print(f"\nWritten to data/labels/{args.sport}.jsonl (appended, same as manual labelling).")
    print("Spot-check a sample of these before training -- Gemini can still mislabel an "
          "ambiguous moment even above the confidence threshold.")


if __name__ == "__main__":
    main()