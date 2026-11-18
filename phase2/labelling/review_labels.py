"""
CASTER Phase 2 -- spot-check Gemini (or manually) labelled rows against
the actual video frames, and correct what's wrong.

WHY THIS EXISTS
----------------
gemini_auto_labeler.py's own output has always carried the warning
"Gemini can still mislabel an ambiguous moment even above the confidence
threshold" -- but nothing in the project actually let you check. The
k-fold evaluation surfaced a concrete reason to: GOAL has 51 real
examples yet only ~4% recall, clearly worse than SHOT_ON_TARGET's 36
examples at ~14% -- a pattern that looks like label noise, not just a
thin class. This tool exists to find out.

HOW IT WORKS
------------
Following label_tool.py's own established pattern (see its docstring):
this does NOT build a video-replay widget. Each row prints its clip
name and exact timestamp, and saves a contact-sheet image of the SAME
six frames Gemini was shown (via gemini_auto_labeler.read_window_frames,
so there is no risk of sampling different frames than Gemini actually
judged) to outputs/review_frames/ for a quick glance -- open your own
video player to that timestamp for full motion context if a still
frame isn't enough to judge.

For each row you then confirm, relabel, or delete it. Decisions are
written immediately to a side-car review-state file (crash-safe, same
pattern as label_tool.py's immediate-append), so a session can always
be resumed later without re-reviewing rows you already decided.
Nothing touches the real data/labels/<sport>.jsonl until you explicitly
run with --apply, which backs up the original first.

Usage:
    # Review every real GOAL example (the concrete case that motivated this)
    python phase2/labelling/review_labels.py --sport football --event GOAL

    # Review everything, 20 rows per sitting
    python phase2/labelling/review_labels.py --sport football --max-rows 20

    # Once you're happy with your decisions, apply them for real:
    python phase2/labelling/review_labels.py --sport football --apply
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EVENTS_BY_SPORT, SUPPORTED_SPORTS, RAW_VIDEO_DIR, FEATURE_WINDOWS_LABEL_DIR, NO_EVENT, OUTPUTS_DIR
from phase2.labelling.gemini_auto_labeler import read_window_frames
from phase2.labelling.label_tool import label_path_for
from phase2.labelling.dedupe_labels import find_duplicates


def review_state_path(sport: str) -> Path:
    return FEATURE_WINDOWS_LABEL_DIR / f"{sport}_review_state.jsonl"


def _row_key(row: dict) -> str:
    """A row's identity for review purposes: which clip, which window.
    (source_clip, end_frame_id) is unique enough in practice -- two
    different real events landing on the exact same frame of the exact
    same clip would itself be a labelling error worth catching."""
    return f"{row.get('source_clip', '')}::{row.get('end_frame_id', -1)}"


def load_review_state(sport: str) -> dict:
    """Returns {row_key: decision_dict} from every decision already made
    in a prior session (or earlier in this one)."""
    path = review_state_path(sport)
    if not path.exists():
        return {}
    decisions = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        decisions[d["row_key"]] = d
    return decisions


def append_decision(sport: str, decision: dict):
    path = review_state_path(sport)
    with open(path, "a") as f:
        f.write(json.dumps(decision) + "\n")


def _resolve_video_path(source_clip: str) -> Path | None:
    """
    source_clip is either "gemini-auto:<name>.mp4" (gemini_auto_labeler.py)
    or a feature-windows JSON path like "outputs/Clip_feature_windows.json"
    (label_tool.py, which stores the --windows argument it was given).
    Either way, we just need the original clip's stem to find the video
    under data/raw/.
    """
    if source_clip.startswith("gemini-auto:"):
        name = source_clip[len("gemini-auto:"):]
        stem = Path(name).stem
    else:
        stem = Path(source_clip).stem
        if stem.endswith("_feature_windows"):
            stem = stem[: -len("_feature_windows")]

    for ext in (".mp4", ".mov", ".mkv", ".avi"):
        candidate = RAW_VIDEO_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def save_contact_sheet(frames, out_path: Path):
    """Stitches the (up to 6) frames into one 3x2 grid image, so a whole
    window can be glanced at in one file instead of six."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    frames = [cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f for f in frames]
    while len(frames) < 6:
        frames.append(np.zeros((h, w, 3), dtype="uint8"))
    rows = [np.hstack(frames[i:i + 3]) for i in range(0, 6, 3)]
    sheet = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def format_timestamp(end_frame_id, fps) -> str:
    """
    LabelledWindow (the schema actually written to data/labels/<sport>.jsonl)
    has no end_timestamp_sec field -- only end_frame_id -- so the timestamp
    has to be computed from the video's own FPS, not read off the row.
    """
    if not fps or fps <= 0:
        return "?"
    seconds = int(end_frame_id / fps)
    return f"{seconds // 60}:{seconds % 60:02d}"


def run_review(sport: str, event_filter: str | None, max_rows: int | None):
    path = label_path_for(sport)
    if not path.exists() or path.stat().st_size == 0:
        print(f"No labels yet at {path}")
        return

    rows = [json.loads(line) for line in open(path) if line.strip()]
    if event_filter:
        rows = [r for r in rows if r["event"] == event_filter]
    if not rows:
        print("No rows found" + (f" for event={event_filter}" if event_filter else ""))
        return

    already = load_review_state(sport)
    pending = [r for r in rows if _row_key(r) not in already]
    print(f"{len(rows)} matching rows, {len(rows) - len(pending)} already reviewed, "
          f"{len(pending)} pending.")
    if max_rows:
        pending = pending[:max_rows]
        print(f"Reviewing up to {max_rows} this session.\n")

    events = [e for e in EVENTS_BY_SPORT[sport] if e != NO_EVENT]
    menu = "  ".join(f"[{i}]{e}" for i, e in enumerate(events))

    video_cache = {}
    for i, row in enumerate(pending, start=1):
        source_clip = row.get("source_clip", "")
        video_path = video_cache.get(source_clip)
        if video_path is None:
            video_path = _resolve_video_path(source_clip)
            video_cache[source_clip] = video_path

        clip_label = Path(source_clip).name

        if video_path is None:
            print(f"\n--- {i}/{len(pending)}  clip={clip_label}  t=?  "
                  f"current label={row['event']} ---")
            print(f"  (could not find the source video for {source_clip!r} under {RAW_VIDEO_DIR} "
                  f"-- judge from the clip name alone, or skip)")
        else:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ts = format_timestamp(int(row["end_frame_id"]), fps)
            print(f"\n--- {i}/{len(pending)}  clip={clip_label}  t={ts}  "
                  f"current label={row['event']} ---")
            frames = read_window_frames(cap, int(row["end_frame_id"]))
            cap.release()
            # clip_label can contain characters invalid in a Windows filename
            # (the gemini-auto: prefix has a colon) -- use the video's own
            # stem for the saved file, and keep clip_label only for display.
            safe_name = video_path.stem
            out_path = OUTPUTS_DIR / "review_frames" / sport / f"{safe_name}_{row['end_frame_id']}.jpg"
            if save_contact_sheet(frames, out_path):
                print(f"  frames saved -> {out_path}  (or scrub your own player to t={ts})")
            else:
                print("  could not extract frames -- judge from timestamp alone, or skip")

        print(f"  [Enter]=correct as labelled   {menu}   [d]=delete (unusable)   "
              f"[s]=skip for now   [q]=quit")
        choice = input("review> ").strip().lower()

        if choice == "q":
            print("Stopping. Resume later with the same command -- already-decided rows are skipped.")
            break
        if choice == "s" or choice is None:
            continue
        if choice == "":
            decision = {"row_key": _row_key(row), "action": "correct"}
        elif choice == "d":
            decision = {"row_key": _row_key(row), "action": "delete"}
        elif choice.isdigit() and int(choice) < len(events):
            new_event = events[int(choice)]
            decision = {"row_key": _row_key(row), "action": "relabel", "new_event": new_event}
        else:
            print("  (not understood, treating as skip)")
            continue

        append_decision(sport, decision)

    print(f"\nDecisions saved to {review_state_path(sport)}. Run with --apply when ready "
          f"to write them into the real label file.")


def run_conflict_review(sport: str, max_rows: int | None):
    """
    Reviews windows where separate labelling sessions disagreed on the
    event (see dedupe_labels.py). Unlike the normal per-row review
    flow, there is no single "current label" to confirm -- the whole
    point is picking which of several conflicting labels (or none of
    them) is actually correct, so there is no bare-Enter "correct as
    labelled" option here; you must pick an explicit event or delete.
    """
    path = label_path_for(sport)
    if not path.exists() or path.stat().st_size == 0:
        print(f"No labels yet at {path}")
        return

    rows = [json.loads(line) for line in open(path) if line.strip()]
    _, conflicting = find_duplicates(rows)
    if not conflicting:
        print("No conflicting duplicates found -- nothing to resolve.")
        return

    already = load_review_state(sport)
    pending = {k: g for k, g in conflicting.items() if _row_key(g[0]) not in already}
    print(f"{len(conflicting)} conflicting windows, {len(conflicting) - len(pending)} already "
          f"resolved, {len(pending)} pending.")
    items = list(pending.items())
    if max_rows:
        items = items[:max_rows]
        print(f"Reviewing up to {max_rows} this session.\n")

    events = [e for e in EVENTS_BY_SPORT[sport] if e != NO_EVENT]
    menu = "  ".join(f"[{i}]{e}" for i, e in enumerate(events))

    for i, (key, group) in enumerate(items, start=1):
        source_clip, end_frame_id = key
        representative = group[0]
        label_counts = Counter(r["event"] for r in group)
        conflict_desc = ", ".join(f"{ev}({n}x)" for ev, n in label_counts.most_common())

        video_path = _resolve_video_path(source_clip)
        clip_label = Path(source_clip).name
        if video_path is None:
            print(f"\n--- {i}/{len(items)}  clip={clip_label}  t=?  conflicting labels: {conflict_desc} ---")
            print(f"  (could not find the source video for {source_clip!r} -- judge from clip name alone, or skip)")
        else:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ts = format_timestamp(int(end_frame_id), fps)
            print(f"\n--- {i}/{len(items)}  clip={clip_label}  t={ts}  conflicting labels: {conflict_desc} ---")
            frames = read_window_frames(cap, int(end_frame_id))
            cap.release()
            safe_name = video_path.stem
            out_path = OUTPUTS_DIR / "review_frames" / sport / f"CONFLICT_{safe_name}_{end_frame_id}.jpg"
            if save_contact_sheet(frames, out_path):
                print(f"  frames saved -> {out_path}  (or scrub your own player to t={ts})")

        print("  Pick which label is correct, or delete all copies of this window:")
        print(f"  {menu}   [d]=delete all copies   [s]=skip for now   [q]=quit")
        choice = input("resolve> ").strip().lower()

        if choice == "q":
            print("Stopping. Resume later with the same command.")
            break
        if choice == "s" or choice == "":
            continue
        row_key = _row_key(representative)
        if choice == "d":
            decision = {"row_key": row_key, "action": "delete"}
        elif choice.isdigit() and int(choice) < len(events):
            decision = {"row_key": row_key, "action": "relabel", "new_event": events[int(choice)]}
        else:
            print("  (not understood, treating as skip)")
            continue
        append_decision(sport, decision)

    print(f"\nDecisions saved to {review_state_path(sport)}. Run --apply when ready.")


def apply_review(sport: str):
    path = label_path_for(sport)
    decisions = load_review_state(sport)
    if not decisions:
        print(f"No review decisions found at {review_state_path(sport)} -- nothing to apply.")
        return

    rows = [json.loads(line) for line in open(path) if line.strip()]
    backup_path = path.with_suffix(path.suffix + ".bak")
    backup_path.write_text(path.read_text())
    print(f"Backed up original -> {backup_path}")

    kept, relabeled, deleted, unreviewed, collapsed = 0, 0, 0, 0, 0
    out_lines = []
    decided_keys_written = set()
    for row in rows:
        key = _row_key(row)
        decision = decisions.get(key)
        if decision is None:
            unreviewed += 1
            out_lines.append(json.dumps(row))
            continue

        # A decided key can legitimately belong to more than one ORIGINAL
        # row -- either a plain duplicate reviewed via the normal flow
        # (e.g. the same GOAL window appearing twice), or a conflicting
        # group resolved via --conflicts-only (which is always MULTIPLE
        # original rows by definition). Either way, one decision should
        # produce exactly ONE output row, not one per original copy --
        # otherwise "resolving a conflict" would still leave the dataset
        # with duplicate (now matching) rows.
        if key in decided_keys_written:
            collapsed += 1
            continue
        decided_keys_written.add(key)

        if decision["action"] == "correct":
            kept += 1
            out_lines.append(json.dumps(row))
        elif decision["action"] == "relabel":
            row["event"] = decision["new_event"]
            relabeled += 1
            out_lines.append(json.dumps(row))
        elif decision["action"] == "delete":
            deleted += 1
            # dropped -- not written

    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    print(f"Applied: {kept} confirmed correct, {relabeled} relabeled, {deleted} deleted, "
          f"{collapsed} extra duplicate copies collapsed away, "
          f"{unreviewed} left untouched (never reviewed).")
    print(f"Wrote {len(out_lines)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", required=True, choices=SUPPORTED_SPORTS)
    ap.add_argument("--event", default=None, help="Only review rows currently labelled this event (e.g. GOAL).")
    ap.add_argument("--max-rows", type=int, default=None, help="Stop after reviewing this many rows this session.")
    ap.add_argument("--apply", action="store_true", help="Apply saved review decisions to the real label file.")
    ap.add_argument("--conflicts-only", action="store_true",
                     help="Review windows where separate labelling sessions disagreed on the event "
                          "(see dedupe_labels.py), instead of reviewing by --event.")
    args = ap.parse_args()

    if args.apply:
        apply_review(args.sport)
    elif args.conflicts_only:
        run_conflict_review(args.sport, args.max_rows)
    else:
        run_review(args.sport, args.event, args.max_rows)


if __name__ == "__main__":
    main()