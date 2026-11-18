"""
CASTER Phase 2 -- flag moments likely to contain a notable event, using
Phase 1's own already-computed features, so you don't have to scrub an
entire match hoping to spot a corner kick or an offside call.

WHY THIS EXISTS
----------------
run_gemini_autolabel.py's --start-sec/--end-sec only helps once you
already know roughly when a rare event happens -- but for a 25-minute
match, finding that moment by watching the whole thing is exactly the
slow, manual work targeted sampling was meant to avoid. Phase 1's
scene states already contain cheap, free signals that correlate with
specific event types, computed from the video Phase 1 already
processed -- nothing new to run, no API calls, instant:

  - boundary_proximity=True  -> the ball is near the touchline/goal
    line. A fairly specific signal (true in ~3% of frames in a real
    clip checked while building this) for CORNER_KICK, throw-ins,
    goal-line clearances.
  - a high ball-height reading (feature_vector index 9) -> an aerial
    ball, worth checking for HEADER, long passes, clearances.
  - a high ball-speed reading -> a struck/fast-moving ball, worth
    checking for SHOT_ON_TARGET, GOAL, a hard pass.
  - official_signal=True -> a referee was detected in-frame. This one
    is the WEAKEST signal here -- in real data it was true in ~11% of
    ALL frames, meaning it mostly just means "the referee is visible",
    not "the referee is actively signalling a foul/offside". Treat it
    as a last-resort category, not a reliable one.

None of these confirm an event -- they narrow a long clip down to a
short, ranked list of timestamps actually worth a 5-second glance,
instead of either scrubbing blind or hoping uniform stride sampling
in run_gemini_autolabel.py gets lucky. "Fast" and "high" are computed
as percentiles WITHIN the clip being scanned (not fixed numbers),
since camera framing and zoom differ between clips.

Usage:
    python phase2/labelling/find_candidates.py --clip LFinal
    python phase2/labelling/find_candidates.py --clip LFinal --category boundary
    python phase2/labelling/find_candidates.py --clip LFinal --top 15
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import OUTPUTS_DIR


def format_timestamp(seconds) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def find_candidates(scene_states, category: str, merge_gap_sec: float = 1.0):
    """
    Thresholds are set high (p97, not p90) deliberately: p90 still means
    ~10% of all frames qualify, which over a multi-minute clip produces
    dozens of scattered hits that merge into a handful of unhelpfully
    wide spans rather than isolated, genuinely rare moments -- confirmed
    empirically while building this (a 33s test clip collapsed to 2
    candidates spanning 21+ seconds each under p90). p97 is still a
    judgment call, not a validated cutoff -- if you get too few/many
    candidates, that's real tuning feedback, not a sign something's
    broken.
    """
    speeds = [s["ball_speed"] for s in scene_states if s.get("ball_detected")]
    heights = [s["feature_vector"][9] for s in scene_states if s.get("ball_detected")]
    speed_cut = sorted(speeds)[int(len(speeds) * 0.97)] if speeds else 0.0
    height_cut = sorted(heights)[int(len(heights) * 0.97)] if heights else 0.0

    checks = {
        "boundary": lambda s: s.get("boundary_proximity") is True,
        "aerial": lambda s: s.get("ball_detected") and s["feature_vector"][9] >= height_cut,
        "fast_ball": lambda s: s.get("ball_detected") and s["ball_speed"] >= speed_cut,
        "official": lambda s: s.get("official_signal") is True,
    }
    wanted = checks if category == "all" else {category: checks[category]}

    # Merge only WITHIN a category -- a "fast_ball" hit and an "official"
    # hit landing near each other in time are not necessarily the same
    # real-world moment, and merging across categories was what produced
    # the misleadingly wide spans mentioned above.
    merged = []
    for cat, check in wanted.items():
        cat_hits = sorted(s["timestamp_sec"] for s in scene_states if check(s))
        for ts in cat_hits:
            if merged and merged[-1]["categories"] == {cat} and ts - merged[-1]["end"] <= merge_gap_sec:
                merged[-1]["end"] = ts
            else:
                merged.append({"start": ts, "end": ts, "categories": {cat}})

    merged.sort(key=lambda m: m["start"])
    return merged, {"speed_cut": speed_cut, "height_cut": height_cut}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="Clip name (matches outputs/<clip>_scene_states.json)")
    ap.add_argument("--category", default="all", choices=["all", "boundary", "aerial", "fast_ball", "official"])
    ap.add_argument("--top", type=int, default=25, help="Show at most this many candidates")
    ap.add_argument("--merge-gap-sec", type=float, default=1.0,
                     help="Nearby flagged moments within this many seconds are merged into one candidate")
    args = ap.parse_args()

    path = OUTPUTS_DIR / f"{args.clip}_scene_states.json"
    if not path.exists():
        print(f"ERROR: {path} not found -- run Phase 1 on this clip first.")
        sys.exit(1)

    scene_states = json.load(open(path))
    merged, thresholds = find_candidates(scene_states, args.category, args.merge_gap_sec)

    print(f"{args.clip}: {len(scene_states)} frames scanned, category={args.category}")
    print(f"(thresholds used this clip: ball_speed >= {thresholds['speed_cut']:.4f}, "
          f"ball_height >= {thresholds['height_cut']:.4f})\n")

    if not merged:
        print("No candidates found for this category in this clip.")
        return

    print(f"{len(merged)} candidate moment(s), showing up to {args.top}:\n")
    print(f"{'timestamp':<12}{'categories':<30}{'suggested --start-sec/--end-sec'}")
    for m in merged[:args.top]:
        cats = ",".join(sorted(m["categories"]))
        start_sec = max(0, int(m["start"]) - 3)
        end_sec = int(m["end"]) + 3
        ts = format_timestamp(m["start"])
        print(f"{ts:<12}{cats:<30}--start-sec {start_sec} --end-sec {end_sec}")

    print("\nThese are candidates, not confirmed events -- glance at the video around each "
          "timestamp (or the saved frame images if you've already run a labeller pass near "
          "there) before deciding it's worth a targeted run_gemini_autolabel.py call. "
          "'official' candidates especially need a visual check -- that signal is weak.")


if __name__ == "__main__":
    main()
