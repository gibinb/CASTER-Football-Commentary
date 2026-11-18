"""
Removes synthetic placeholder rows from data/labels/<sport>.jsonl,
keeping only real rows labelled from actual video footage via
label_tool.py.

Why this exists: generate_synthetic_data.py APPENDS rows tagged
"source_clip": "synthetic" to the same .jsonl file label_tool.py writes
real rows to. Every re-run of generate_synthetic_data.py adds MORE
synthetic rows on top of whatever's already there -- it does not
overwrite. If you've run it more than once (or ran it after doing real
labelling), your real rows are still in the file, just increasingly
diluted underneath a growing pile of synthetic ones.

This script splits the two apart safely: it never deletes anything
outright. It writes a backup of the original file, a file containing
only the real rows, and a file containing only the synthetic rows, so
you can inspect counts before deciding what to do next.

Usage:
    python phase2/labelling/remove_synthetic_data.py --sport football

    # then, once you've confirmed the real-only file looks right:
    python phase2/labelling/remove_synthetic_data.py --sport football --apply
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import FEATURE_WINDOWS_LABEL_DIR, SUPPORTED_SPORTS


def label_path_for(sport: str) -> Path:
    return FEATURE_WINDOWS_LABEL_DIR / f"{sport}.jsonl"


def split_real_and_synthetic(sport: str, apply: bool = False):
    path = label_path_for(sport)
    if not path.exists():
        print(f"No label file found for {sport} at {path} -- nothing to do.")
        return

    real_lines, synthetic_lines = [], []
    malformed = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if row.get("source_clip") == "synthetic":
                synthetic_lines.append(line)
            else:
                real_lines.append(line)

    total = len(real_lines) + len(synthetic_lines)
    print(f"\n{sport}: {total} total rows ({len(real_lines)} real, "
          f"{len(synthetic_lines)} synthetic"
          f"{f', {malformed} malformed skipped' if malformed else ''})")

    backup_path = path.with_suffix(".jsonl.bak")
    real_only_path = path.with_name(f"{sport}_real_only.jsonl")
    synthetic_only_path = path.with_name(f"{sport}_synthetic_only.jsonl")

    if not apply:
        # Dry run: write inspection copies, touch nothing else.
        with open(real_only_path, "w") as f:
            f.write("\n".join(real_lines) + ("\n" if real_lines else ""))
        with open(synthetic_only_path, "w") as f:
            f.write("\n".join(synthetic_lines) + ("\n" if synthetic_lines else ""))
        print(f"  Dry run only -- {path} was NOT modified.")
        print(f"  Wrote {len(real_lines)} real rows -> {real_only_path}")
        print(f"  Wrote {len(synthetic_lines)} synthetic rows -> {synthetic_only_path}")
        print(f"  Inspect {real_only_path.name}, then re-run with --apply "
              f"to actually replace {path.name} with real-only data.")
        return

    # Apply: back up original, then overwrite the real file with real rows only.
    with open(backup_path, "w") as f:
        with open(path) as orig:
            f.write(orig.read())
    with open(path, "w") as f:
        f.write("\n".join(real_lines) + ("\n" if real_lines else ""))

    print(f"  Backed up original -> {backup_path}")
    print(f"  {path} now contains ONLY the {len(real_lines)} real rows.")
    if len(real_lines) == 0:
        print(
            f"  WARNING: zero real rows found for {sport}. You have not "
            f"labelled any real data yet for this sport -- see "
            f"phase2/README.md 'Labelling real data' before training."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Split real vs synthetic rows out of data/labels/<sport>.jsonl"
    )
    parser.add_argument("--sport", required=True, choices=list(SUPPORTED_SPORTS) + ["both"])
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually overwrite <sport>.jsonl with real-only rows (backs up "
             "the original first). Without this flag, only inspection copies "
             "are written and the original file is left untouched.",
    )
    args = parser.parse_args()

    sports = list(SUPPORTED_SPORTS) if args.sport == "both" else [args.sport]
    for sport in sports:
        split_real_and_synthetic(sport, apply=args.apply)


if __name__ == "__main__":
    main()
