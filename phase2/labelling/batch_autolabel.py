"""
CASTER Phase 2 -- batch Gemini auto-labelling across every clip in data/raw/.

run_gemini_autolabel.py operates on ONE clip at a time (one --windows /
--video pair per invocation). This script is the "run all of them"
wrapper: it finds every .mp4 under config.RAW_VIDEO_DIR, makes sure each
one has Phase 1 output (running phase1/run_phase1.py if the feature
windows file doesn't exist yet), then runs the Gemini auto-labeller on
each in turn, appending to the SAME data/labels/<sport>.jsonl file
run_gemini_autolabel.py and label_tool.py already write to -- so this is
purely a convenience wrapper, not a different data format or pipeline.

Usage:
    # Simplest: label every clip in data/raw/, whole-clip uniform sampling
    python phase2/labelling/batch_autolabel.py --sport football

    # Skip clips that already have Phase 1 output computed, don't re-run it
    python phase2/labelling/batch_autolabel.py --sport football --skip-phase1-if-exists

    # Same stride/confidence/rate-limit controls as run_gemini_autolabel.py,
    # applied identically to every clip in the batch:
    python phase2/labelling/batch_autolabel.py --sport football --stride 30 --max-windows 150

    # Process ONE named clip only (matches by filename, with or without
    # the .mp4 extension) -- for running clips one at a time instead of
    # the whole data/raw/ folder in one long sitting. Combine with
    # --skip-phase1-if-exists once you've already run Phase 1 separately
    # (recommended for long clips -- see module docstring note below).
    python phase2/labelling/batch_autolabel.py --sport football --clip Semifinal01 --skip-phase1-if-exists

RECOMMENDED WORKFLOW FOR LONG CLIPS: Phase 1 (YOLO + ByteTrack over every
frame) is pure local CPU work and scales with video LENGTH -- a 15-minute
clip can take a long time regardless of anything Gemini-related. The
Gemini labelling step itself is bounded by --max-windows regardless of
video length (capped API calls, not full-video processing). So for long
clips, run Phase 1 separately first (phase1/run_phase1.py --input ...,
once per clip, at your own pace), THEN run this script with
--skip-phase1-if-exists and --clip <name> per clip -- that keeps the
(fast, resumable, API-paced) labelling step decoupled from the (slow,
CPU-bound) detection step, instead of one long command doing both.

Note on targeted sampling: --start-sec/--end-sec here apply the SAME
range to every clip in the batch, which is rarely what you want (a goal
in clip A is not at the same timestamp as a goal in clip B). For
per-clip targeted sampling (e.g. "clip A's goal is at 75-95s, clip B's
is at 40-55s"), run run_gemini_autolabel.py individually per clip
instead -- this batch script is for broad, whole-clip coverage across
many clips, not for hand-targeting rare events in each one.

Rate limiting is tracked PER CLIP by auto_label_windows() internally
(it resets its own pacing clock on every call). To avoid a request
burst right at the boundary between two clips, this script sleeps one
full request_interval before starting each clip after the first.

A failure on one clip (corrupt video, Phase 1 crash, API error that
isn't a rate limit) is logged and the batch moves on to the next clip,
rather than aborting the whole run and losing progress on clips already
processed -- labels already saved to disk before a failure stay safe,
consistent with how a single run_gemini_autolabel.py run already
behaves on interruption.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SPORT_FOOTBALL, RAW_VIDEO_DIR, OUTPUTS_DIR, ROOT_DIR

from phase2.labelling.label_tool import load_windows, append_label
from phase2.labelling.gemini_auto_labeler import auto_label_windows, AutoLabelError
# RateLimitExhausted is caught INSIDE auto_label_windows() itself (it sets
# stopped_early=True and returns normally rather than raising out) -- this
# script only needs to catch AutoLabelError (bad API key, missing video,
# stride too small) for the "skip this clip, continue the batch" path.


def feature_windows_path_for(clip_path: Path) -> Path:
    return OUTPUTS_DIR / f"{clip_path.stem}_feature_windows.json"


def ensure_phase1_output(clip_path: Path, skip_if_exists: bool) -> bool:
    """
    Returns True if feature windows are ready to use, False if Phase 1
    could not be run/failed and this clip should be skipped.
    """
    windows_path = feature_windows_path_for(clip_path)
    if windows_path.exists() and skip_if_exists:
        print(f"  Phase 1 output already exists -> {windows_path.name} (skipping Phase 1)")
        return True

    print(f"  Running Phase 1 on {clip_path.name} ...")
    result = subprocess.run(
        [sys.executable, str(ROOT_DIR / "phase1" / "run_phase1.py"), "--input", str(clip_path)],
        cwd=str(ROOT_DIR),
    )
    if result.returncode != 0:
        print(f"  ERROR: Phase 1 failed on {clip_path.name} (exit code {result.returncode}) -- skipping this clip.")
        return False
    if not windows_path.exists():
        print(f"  ERROR: Phase 1 reported success but {windows_path.name} still doesn't exist -- skipping this clip.")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", required=True, choices=[SPORT_FOOTBALL])
    ap.add_argument("--stride", type=int, default=30,
                     help="Sample every Nth window per clip (min 30). Same meaning as run_gemini_autolabel.py.")
    ap.add_argument("--min-confidence", type=float, default=0.85)
    ap.add_argument("--max-windows", type=int, default=250,
                     help="Hard cap on API calls PER CLIP, not for the whole batch.")
    ap.add_argument("--request-interval", type=float, default=5.0)
    ap.add_argument("--start-sec", type=float, default=None,
                     help="Applied identically to every clip -- see module docstring for why that's "
                          "rarely what you want for a multi-clip batch.")
    ap.add_argument("--end-sec", type=float, default=None)
    ap.add_argument("--skip-phase1-if-exists", action="store_true",
                     help="Don't re-run Phase 1 for a clip that already has a *_feature_windows.json file.")
    ap.add_argument("--pattern", default="*.mp4",
                     help="Glob pattern for finding clips under data/raw/ (default: *.mp4)")
    ap.add_argument("--clip", default=None,
                     help="Process only this one clip (match by filename, with or without "
                          ".mp4) instead of every clip in data/raw/. Use this to run clips "
                          "one at a time -- see module docstring for the recommended workflow "
                          "on long videos.")
    args = ap.parse_args()

    clips = sorted(RAW_VIDEO_DIR.glob(args.pattern))
    if not clips:
        print(f"No clips found matching {args.pattern!r} in {RAW_VIDEO_DIR}")
        sys.exit(1)

    if args.clip is not None:
        wanted = args.clip if args.clip.lower().endswith(".mp4") else args.clip + ".mp4"
        matches = [c for c in clips if c.name.lower() == wanted.lower()]
        if not matches:
            available = ", ".join(c.name for c in clips)
            print(f"ERROR: no clip named {args.clip!r} found in {RAW_VIDEO_DIR}.")
            print(f"Available clips: {available}")
            sys.exit(1)
        clips = matches

    print(f"Found {len(clips)} clip(s) in {RAW_VIDEO_DIR}:")
    for c in clips:
        print(f"  - {c.name}")
    print()

    totals = {"saved": 0, "skipped": 0, "clips_processed": 0, "clips_failed": 0}
    first_clip = True

    for clip_path in clips:
        print(f"{'=' * 70}\n{clip_path.name}\n{'=' * 70}")

        if not ensure_phase1_output(clip_path, args.skip_phase1_if_exists):
            totals["clips_failed"] += 1
            continue

        windows = load_windows(str(feature_windows_path_for(clip_path)))
        print(f"  Loaded {len(windows)} feature windows")

        if args.start_sec is not None or args.end_sec is not None:
            before = len(windows)
            lo = args.start_sec if args.start_sec is not None else float("-inf")
            hi = args.end_sec if args.end_sec is not None else float("inf")
            windows = [w for w in windows if lo <= w["end_timestamp_sec"] <= hi]
            print(f"  Restricted to [{args.start_sec}, {args.end_sec}] seconds: {before} -> {len(windows)} windows")
            if not windows:
                print("  No windows in that range for this clip -- skipping.")
                continue

        if not first_clip:
            # Don't reset auto_label_windows()'s own internal pacing clock
            # mid-batch without first guaranteeing a gap since the previous
            # clip's last request.
            time.sleep(args.request_interval)
        first_clip = False

        try:
            saved, skipped, stopped_early, already_covered = auto_label_windows(
                windows,
                sport=args.sport,
                video_path=str(clip_path),
                append_label=append_label,
                stride=args.stride,
                min_confidence=args.min_confidence,
                max_windows=args.max_windows,
                request_interval=args.request_interval,
            )
        except AutoLabelError as exc:
            print(f"  ERROR: {exc} -- skipping this clip.")
            totals["clips_failed"] += 1
            continue
        except KeyboardInterrupt:
            # Ctrl+C (or an IDE stop button sending the same signal) during
            # a request. Whatever was already saved for THIS clip before
            # the interrupt is safe on disk (each label is written
            # immediately when earned, not batched), but auto_label_windows()
            # can't report a final saved/skipped count for a clip it never
            # returned from -- so totals below only reflect fully-completed
            # clips, not this one. That's a reporting gap, not a data-loss
            # one: nothing you'd already seen printed as "saved t=..." is
            # lost.
            print(f"\n  Interrupted (Ctrl+C) partway through {clip_path.name}. Labels already "
                  f"printed as 'saved t=...' above are safe on disk -- this clip's own total just "
                  f"isn't counted in the summary below since the run didn't finish normally.")
            print(f"\n{'=' * 70}")
            print(f"Batch stopped by user. {totals['clips_processed']} clip(s) fully completed "
                  f"before this one, {totals['saved']} label(s) saved from those "
                  f"(plus whatever this interrupted clip saved, shown above).")
            print(f"{'=' * 70}")
            sys.exit(130)  # conventional exit code for SIGINT

        print(f"\n  {saved} label(s) saved, {skipped} window(s) skipped, {already_covered} "
              f"already labelled in an earlier session (no API call made) for {clip_path.name}.")
        totals["saved"] += saved
        totals["skipped"] += skipped
        totals["clips_processed"] += 1

        if stopped_early:
            print("\n  Stopping the ENTIRE batch here (not just this clip) -- either the Gemini "
                  "rate limit/quota was exhausted, or the connection to Gemini was lost. "
                  "See the message above for which one.")
            print(f"  {totals['clips_processed']} clip(s) fully processed, "
                  f"{len(clips) - totals['clips_processed'] - totals['clips_failed']} not yet attempted.")
            print("  Re-run this same command later (or tomorrow) -- clips already processed are safe; "
                  "already-labelled windows won't be re-sent, since the next run picks up from scratch "
                  "on remaining clips (this script doesn't track partial-clip progress, only whole-clip).")
            break

    print(f"\n{'=' * 70}")
    print(f"Batch complete: {totals['clips_processed']} clip(s) processed, "
          f"{totals['clips_failed']} clip(s) failed/skipped, "
          f"{totals['saved']} total label(s) saved, {totals['skipped']} total window(s) skipped.")
    print(f"{'=' * 70}")
    print("\nSpot-check a sample of these before training -- Gemini can still mislabel an "
          "ambiguous moment even above the confidence threshold.")


if __name__ == "__main__":
    main()