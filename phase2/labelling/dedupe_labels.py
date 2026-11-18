"""
CASTER Phase 2 -- find and remove exact duplicate training rows.

WHY THIS EXISTS
----------------
Running batch_autolabel.py / run_gemini_autolabel.py more than once on
the same clip (e.g. once before a Phase 1 re-run with new weights, once
after) resamples and re-labels the SAME windows again, appending a
second copy on top of the first -- because each run only tracks what
IT saved this run, never what's already in data/labels/<sport>.jsonl
from a previous session. Over several labelling sessions this produces
exact duplicate (source_clip, end_frame_id) rows.

This is not harmless: a duplicated row adds no information, it just
silently over-weights that one window in the training distribution.
Worse, if two labelling sessions disagreed about a window (one said
GOAL, a later one said SHOT_ON_TARGET for the exact same frames), the
dataset now contains a genuine contradiction the classifier is being
asked to learn from both ways at once.

WHAT THIS DOES
---------------
Finds every (source_clip, end_frame_id) that appears more than once,
and splits them into two groups:

  CONSISTENT duplicates -- every copy agrees on the event. Safe to
  collapse to a single copy automatically; --apply does this.

  CONFLICTING duplicates -- copies disagree on the event. NOT
  auto-resolved here -- that needs a human decision, since there's no
  principled way to guess which labelling session was right. Use:
      python phase2/labelling/review_labels.py --sport <sport> --conflicts-only
  to go through them one at a time with the same frame-viewing
  workflow used for the GOAL spot-check.

Usage:
    python phase2/labelling/dedupe_labels.py --sport football
    python phase2/labelling/dedupe_labels.py --sport football --apply
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SUPPORTED_SPORTS
from phase2.labelling.label_tool import label_path_for


def find_duplicates(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("source_clip", ""), row.get("end_frame_id", -1))
        groups[key].append(row)

    consistent, conflicting = {}, {}
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        events = set(r["event"] for r in group)
        if len(events) == 1:
            consistent[key] = group
        else:
            conflicting[key] = group
    return consistent, conflicting


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", required=True, choices=SUPPORTED_SPORTS)
    ap.add_argument("--apply", action="store_true",
                     help="Actually remove CONSISTENT duplicate rows (keeping one copy of each). "
                          "Conflicting duplicates are never touched by --apply -- resolve those "
                          "with review_labels.py --conflicts-only first.")
    args = ap.parse_args()

    path = label_path_for(args.sport)
    if not path.exists() or path.stat().st_size == 0:
        print(f"No labels at {path}")
        return

    rows = [json.loads(line) for line in open(path) if line.strip()]
    consistent, conflicting = find_duplicates(rows)

    extra_consistent_rows = sum(len(g) - 1 for g in consistent.values())
    extra_conflicting_rows = sum(len(g) - 1 for g in conflicting.values())

    print(f"{path}  --  {len(rows)} total rows")
    print(f"\nConsistent duplicates (safe to auto-collapse): {len(consistent)} windows, "
          f"{extra_consistent_rows} redundant extra rows")
    by_clip = Counter(k[0] for k in consistent)
    for clip, n in by_clip.most_common():
        print(f"  {clip}: {n} duplicate window(s)")

    print(f"\nConflicting duplicates (need human review, NOT touched by --apply): "
          f"{len(conflicting)} windows, {extra_conflicting_rows} extra rows")
    if conflicting:
        by_clip_conf = Counter(k[0] for k in conflicting)
        for clip, n in by_clip_conf.most_common():
            print(f"  {clip}: {n} conflicting window(s)")
        print("\nExamples (first 5):")
        for key, group in list(conflicting.items())[:5]:
            events = [r["event"] for r in group]
            print(f"  {key[0]} frame {key[1]}: labelled as {events}")
        print(f"\nRun: python phase2/labelling/review_labels.py --sport {args.sport} --conflicts-only")

    if not args.apply:
        print(f"\nDry run only -- {path} was NOT modified. Re-run with --apply to remove the "
              f"{extra_consistent_rows} consistent duplicate rows.")
        return

    if extra_consistent_rows == 0:
        print("\nNothing to apply -- no consistent duplicates found.")
        return

    backup_path = path.with_suffix(path.suffix + ".bak")
    backup_path.write_text(path.read_text())
    print(f"\nBacked up original -> {backup_path}")

    seen = set()
    out_lines = []
    removed = 0
    for row in rows:
        key = (row.get("source_clip", ""), row.get("end_frame_id", -1))
        if key in consistent:
            if key in seen:
                removed += 1
                continue
            seen.add(key)
        out_lines.append(json.dumps(row))

    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    print(f"Removed {removed} redundant duplicate rows. Wrote {len(out_lines)} rows -> {path}")
    print(f"Conflicting duplicates were left untouched ({extra_conflicting_rows} rows across "
          f"{len(conflicting)} windows) -- resolve those separately.")


if __name__ == "__main__":
    main()
