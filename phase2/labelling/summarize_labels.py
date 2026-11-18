"""
CASTER Phase 2 -- summarize what's actually in data/labels/<sport>.jsonl.

Answers two questions you can't easily eyeball from the raw .jsonl:
  1. Which clips have I already run the Gemini auto-labeller against,
     and how many rows did each one contribute?
  2. What's my real class coverage right now -- which event classes
     are well represented, thin, or still completely missing?

Every row written by label_tool.py or gemini_auto_labeler.py carries a
source_clip field (e.g. "gemini-auto:Semifinal01.mp4" or a plain
filename for manual label_tool.py tags), so this is a straight
groupby over existing data -- no guessing, no video re-scanning.

Usage:
    python phase2/labelling/summarize_labels.py --sport football
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EVENTS_BY_SPORT, SUPPORTED_SPORTS, NO_EVENT
from phase2.labelling.label_tool import label_path_for


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", required=True, choices=SUPPORTED_SPORTS)
    args = ap.parse_args()

    path = label_path_for(args.sport)
    if not path.exists() or path.stat().st_size == 0:
        print(f"No labels yet at {path}")
        sys.exit(0)

    rows = [json.loads(line) for line in open(path) if line.strip()]
    print(f"{path}  --  {len(rows)} total labelled windows\n")

    # --- By clip ---
    by_clip = defaultdict(Counter)
    for r in rows:
        by_clip[r.get("source_clip", "UNKNOWN")][r["event"]] += 1

    print("=" * 70)
    print("BY CLIP (what you've actually run Gemini/label_tool against)")
    print("=" * 70)
    for clip, counts in sorted(by_clip.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(counts.values())
        top = ", ".join(f"{ev}={n}" for ev, n in counts.most_common(4))
        more = "" if len(counts) <= 4 else f" (+{len(counts) - 4} more classes)"
        print(f"  {clip:<34} {total:>4} rows   [{top}{more}]")

    # --- By class, across everything ---
    overall = Counter(r["event"] for r in rows)
    all_events = EVENTS_BY_SPORT[args.sport]

    print(f"\n{'=' * 70}")
    print("BY CLASS (overall coverage, across all clips combined)")
    print("=" * 70)
    for ev in all_events:
        n = overall.get(ev, 0)
        flag = ""
        if n == 0:
            flag = "  <-- ZERO examples, classifier cannot predict this at all"
        elif n < 10:
            flag = "  <-- thin, treat predictions for this class with caution"
        marker = " (never reaches Phase 3)" if ev == NO_EVENT else ""
        print(f"  {ev:<16}{marker:<24} {n:>4}{flag}")

    missing = [ev for ev in all_events if ev != NO_EVENT and overall.get(ev, 0) == 0]
    if missing:
        print(f"\nStill completely unrepresented: {', '.join(missing)}")


if __name__ == "__main__":
    main()
