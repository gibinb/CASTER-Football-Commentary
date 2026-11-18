"""
CASTER Phase 2 -- feature-window labelling tool.

This is the slowest, most manual part of Phase 2 (see CASTER_Phase2_DeepDive
.docx Section 3 -- "expect roughly 3-4 hours to label 500-1000 windows").
This tool exists to make that process as fast as possible: it loads the
feature windows Phase 1 already saved to outputs/<clip>_feature_windows.json,
shows you the window's timestamp and a few raw stats, and lets you tag
the current window with a single keypress instead of typing or clicking
through a menu every time.

Usage:
    python phase2/labelling/label_tool.py --windows outputs/myclip_feature_windows.json --sport football

    # Jump straight to a known moment instead of starting at window 0 --
    # e.g. you already know (from Gemini missing it, or from watching the
    # match yourself) that something notable happens around t=293s:
    python phase2/labelling/label_tool.py --windows outputs/myclip_feature_windows.json --sport football --start-sec 288 --end-sec 300

    # IMPORTANT: feature_windows.json has one entry PER VIDEO FRAME
    # (~30/sec), not pre-spaced like Gemini's own sampling -- a small
    # --stride like 5 still leaves many windows per second. Use --stride
    # 30 (roughly one window per second) to get a genuinely small,
    # workable list when combined with --start-sec/--end-sec:
    python phase2/labelling/label_tool.py --windows outputs/myclip_feature_windows.json --sport football --start-sec 288 --end-sec 300 --stride 30

    # Add --video to see a saved image of each window's frames (the same
    # six frames Gemini would have judged) instead of only raw numbers --
    # essential when confirming something as visually specific as an
    # offside call, where the feature stats alone aren't enough to judge:
    python phase2/labelling/label_tool.py --windows outputs/myclip_feature_windows.json --video data/raw/myclip.mp4 --sport football --start-sec 288 --end-sec 300 --stride 30

What it does:
    1. Loads the FeatureWindow list Phase 1 produced for one clip.
    2. Shows window N of M, its end-timestamp, and quick-glance stats
       (ball speed, zone, contact flag, boundary proximity) so you can
       decide what event (if any) just happened without re-watching
       the raw video frame by frame.
    3. Press a number key (or click a button) to tag the CURRENT window
       with that event class. Press 'n' / click "No event" if nothing
       notable happened in this window -- this SAVES the window as a
       NO_EVENT example. This matters: most of any match is ordinary
       background play, and the classifier needs real examples of that
       too, or it has no way to ever predict anything other than one of
       the named events, even on completely unremarkable footage. Press
       's' / click "Skip" only for windows you genuinely cannot judge
       (corrupted frame, ambiguous replay, camera cut) -- true skips are
       NOT saved at all, unlike 'n'.
    4. Appends a LabelledWindow JSON line to data/labels/<sport>.jsonl
       immediately on every tag (including 'n'/NO_EVENT tags), so a
       crash or closed window never loses already-labelled data.

This tool intentionally does NOT replay the original video. Re-watching
the source clip alongside this tool (any video player, scrubbed to the
printed timestamp) is the recommended workflow -- this window keeps you
fast at the *tagging* step once you already know what happened.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EVENTS_BY_SPORT, SUPPORTED_SPORTS, FEATURE_WINDOWS_LABEL_DIR, NO_EVENT, OUTPUTS_DIR
from schemas import LabelledWindow

try:
    import tkinter as tk
    from tkinter import messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


def load_windows(path: str):
    with open(path) as f:
        raw = json.load(f)
    return raw


def filter_by_time(windows, start_sec, end_sec):
    """Same semantics as run_gemini_autolabel.py's --start-sec/--end-sec --
    keeps only windows ending within [start_sec, end_sec], so a known
    moment can be jumped to directly instead of scrubbing through every
    window in the clip from the start."""
    if start_sec is None and end_sec is None:
        return windows
    lo = start_sec if start_sec is not None else float("-inf")
    hi = end_sec if end_sec is not None else float("inf")
    return [w for w in windows if lo <= w["end_timestamp_sec"] <= hi]


def format_timestamp(seconds) -> str:
    """Raw seconds -> 'm:ss', so you never have to do this math by hand
    while cross-checking a timestamp against your own video player."""
    if seconds is None:
        return "?"
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def save_window_frames(video_path: str, window: dict, sport: str):
    """
    Saves a contact-sheet image of this window's frames (the same six
    frames gemini_auto_labeler.py would show the API) so you have a
    visual reference while deciding, not just raw feature numbers --
    essential for something as specific as confirming an exact offside
    frame, which the stats alone can't tell you. Returns the saved path,
    or None if a frame couldn't be read (e.g. at the very start/end of
    the video) or OpenCV isn't installed.

    The import of read_window_frames is INSIDE this function (not at
    module level) on purpose: gemini_auto_labeler.py imports
    label_path_for FROM this module, so an unconditional top-level
    import the other way would create a circular import that fails at
    module load time. By the time this function actually runs, both
    modules have already finished loading, so the import resolves fine.
    """
    try:
        import cv2
        import numpy as np
        from phase2.labelling.gemini_auto_labeler import read_window_frames
    except ImportError:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    frames = read_window_frames(cap, int(window["end_frame_id"]))
    cap.release()
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    frames = [cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f for f in frames]
    while len(frames) < 6:
        frames.append(np.zeros((h, w, 3), dtype="uint8"))
    rows = [np.hstack(frames[i:i + 3]) for i in range(0, 6, 3)]
    sheet = np.vstack(rows)

    out_dir = OUTPUTS_DIR / "label_tool_frames" / sport
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(video_path).stem}_{window['end_frame_id']}.jpg"
    if cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        return out_path
    return None


def label_path_for(sport: str) -> Path:
    FEATURE_WINDOWS_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    return FEATURE_WINDOWS_LABEL_DIR / f"{sport}.jsonl"


def append_label(sport: str, event: str, window: dict, source_clip: str):
    lw = LabelledWindow(
        sport=sport,
        event=event,
        vectors=window["vectors"],
        source_clip=source_clip,
        end_frame_id=window["end_frame_id"],
    )
    out_path = label_path_for(sport)
    with open(out_path, "a") as f:
        f.write(lw.to_json_line() + "\n")
    return out_path


class LabelToolGUI:
    """
    Minimal Tkinter scrubber. Number keys 1-9 and 0 map to the named
    event classes for the chosen sport (in config.EVENTS_BY_SPORT
    order, excluding NO_EVENT -- that one gets its own dedicated key
    since it is the single most common real label and deserves to not
    compete with the ten named events for number keys).
    'n' tags the window NO_EVENT (SAVED as training data -- see module
    docstring for why this matters). 's' truly skips the window
    (NOT saved at all) -- use only when you genuinely cannot judge it.
    """

    def __init__(self, windows, sport: str, source_clip: str, video_path: Optional[str] = None):
        self.windows = windows
        self.sport = sport
        self.source_clip = source_clip
        self.video_path = video_path
        self.events = [e for e in EVENTS_BY_SPORT[sport] if e != NO_EVENT]
        self.idx = 0
        self.labelled_count = 0

        self.root = tk.Tk()
        self.root.title(f"CASTER label tool -- {sport} -- {Path(source_clip).name}")
        self.root.geometry("640x480")

        self.info_label = tk.Label(self.root, font=("Courier", 12), justify="left", anchor="w")
        self.info_label.pack(fill="x", padx=16, pady=(16, 8))

        self.progress_label = tk.Label(self.root, font=("Courier", 10), fg="gray")
        self.progress_label.pack(fill="x", padx=16)

        button_frame = tk.Frame(self.root)
        button_frame.pack(padx=16, pady=16, fill="both", expand=True)

        for i, ev in enumerate(self.events):
            key = str((i + 1) % 10)  # keys 1..9 then 0
            btn = tk.Button(
                button_frame,
                text=f"[{key}] {ev}",
                width=22,
                command=lambda e=ev: self.tag(e),
            )
            btn.grid(row=i // 2, column=i % 2, padx=4, pady=4, sticky="we")

        no_event_btn = tk.Button(
            self.root, text="[n] No event (SAVES as NO_EVENT)", width=34,
            command=lambda: self.tag(NO_EVENT),
        )
        no_event_btn.pack(pady=(8, 2))

        skip_btn = tk.Button(self.root, text="[s] Skip (cannot judge -- NOT saved)", width=34, command=self.skip)
        skip_btn.pack(pady=(0, 16))

        self.root.bind("<Key>", self.on_key)
        self.show_current()

    def current_window(self):
        return self.windows[self.idx]

    def show_current(self):
        if self.idx >= len(self.windows):
            messagebox.showinfo("Done", "No more windows in this file.")
            self.root.quit()
            return

        w = self.current_window()
        vectors = w["vectors"]
        last = vectors[-1] if vectors else [0.0] * 12

        ts = w["end_timestamp_sec"]
        info = (
            f"Window {self.idx + 1} / {len(self.windows)}\n"
            f"end_timestamp_sec: {ts:.2f}   (= {format_timestamp(ts)})\n"
            f"end_frame_id:      {w['end_frame_id']}\n"
            f"--- last frame in window ---\n"
            f"ball_pos:        ({last[0]:.3f}, {last[1]:.3f})\n"
            f"ball_speed:      {last[4]:.3f}\n"
            f"nearest_dist:    {last[5]:.3f}\n"
            f"players_nearby:  {last[6]:.0f}\n"
            f"zone_id:         {last[7]:.0f}\n"
            f"official_signal: {bool(last[8])}\n"
            f"contact_flag:    {bool(last[10])}\n"
            f"boundary_prox:   {bool(last[11])}\n"
        )
        if self.video_path:
            frame_path = save_window_frames(self.video_path, w, self.sport)
            if frame_path:
                info += f"\nframes saved -> {frame_path}\n"
            else:
                info += "\n(could not extract frames for this window)\n"
        self.info_label.config(text=info)
        self.progress_label.config(text=f"Labelled so far this session: {self.labelled_count}")

    def tag(self, event: str):
        w = self.current_window()
        append_label(self.sport, event, w, self.source_clip)
        self.labelled_count += 1
        self.idx += 1
        self.show_current()

    def skip(self):
        self.idx += 1
        self.show_current()

    def on_key(self, event_obj):
        key = event_obj.char
        if key == "s":
            self.skip()
            return
        if key == "n":
            self.tag(NO_EVENT)
            return
        if key.isdigit():
            mapped_idx = (int(key) - 1) % 10
            if mapped_idx < len(self.events):
                self.tag(self.events[mapped_idx])

    def run(self):
        self.root.mainloop()


def run_headless(windows, sport: str, source_clip: str, video_path: Optional[str] = None):
    """
    Fallback when Tkinter isn't available (e.g. a headless server or
    Colab). Same workflow, plain terminal input instead of buttons.
    """
    events = [e for e in EVENTS_BY_SPORT[sport] if e != NO_EVENT]
    print(f"\nCASTER label tool (headless) -- {sport} -- {Path(source_clip).name}")
    print(f"{len(windows)} windows loaded.\n")
    print("Event classes:")
    for i, ev in enumerate(events):
        print(f"  [{(i + 1) % 10}] {ev}")
    print("  [n] no event (SAVES as NO_EVENT -- most windows are this; label it, don't skip it)")
    print("  [s] skip (cannot judge -- NOT saved) / [q] quit\n")

    labelled = 0
    for idx, w in enumerate(windows):
        vectors = w["vectors"]
        last = vectors[-1] if vectors else [0.0] * 12
        ts = w["end_timestamp_sec"]
        print(f"--- window {idx + 1}/{len(windows)}  t={ts:.2f}s ({format_timestamp(ts)}) "
              f"speed={last[4]:.3f} zone={last[7]:.0f} contact={bool(last[10])} "
              f"boundary={bool(last[11])} ---")
        if video_path:
            frame_path = save_window_frames(video_path, w, sport)
            if frame_path:
                print(f"  frames saved -> {frame_path}")
            else:
                print("  (could not extract frames for this window)")
        choice = input("label> ").strip().lower()
        if choice == "q":
            break
        if choice == "s" or choice == "":
            continue
        if choice == "n":
            append_label(sport, NO_EVENT, w, source_clip)
            labelled += 1
            continue
        if choice.isdigit():
            mapped_idx = (int(choice) - 1) % 10
            if mapped_idx < len(events):
                append_label(sport, events[mapped_idx], w, source_clip)
                labelled += 1
            else:
                print("  (no event at that key, skipped)")
    print(f"\nLabelled {labelled} windows. Saved to {label_path_for(sport)}")


def main():
    parser = argparse.ArgumentParser(description="Label CASTER Phase 1 feature windows with event tags")
    parser.add_argument("--windows", required=True, help="Path to a *_feature_windows.json file from Phase 1")
    parser.add_argument("--sport", required=True, choices=SUPPORTED_SPORTS)
    parser.add_argument("--headless", action="store_true", help="Force terminal mode even if Tkinter is available")
    parser.add_argument("--start-sec", type=float, default=None,
                         help="Only show windows ending at or after this many seconds into the clip. "
                              "Combine with --end-sec to jump straight to a known moment instead of "
                              "starting at window 0.")
    parser.add_argument("--end-sec", type=float, default=None,
                         help="Only show windows ending at or before this many seconds into the clip.")
    parser.add_argument("--stride", type=int, default=1,
                         help="Show every Nth window instead of every one (default 1 = exhaustive). "
                              "feature_windows.json has one entry per VIDEO FRAME (~30/sec) -- use "
                              "something like 30 (roughly one window per second), not a small number, "
                              "to actually get a short list when combined with --start-sec/--end-sec.")
    parser.add_argument("--video", default=None,
                         help="Path to the source video (e.g. data/raw/myclip.mp4). When given, saves "
                              "an image of each window's frames so you have a visual reference instead "
                              "of only raw feature numbers -- essential for confirming something visually "
                              "specific like an offside call.")
    args = parser.parse_args()

    windows = load_windows(args.windows)
    print(f"Loaded {len(windows)} feature windows from {args.windows}")

    if args.start_sec is not None or args.end_sec is not None:
        before = len(windows)
        windows = filter_by_time(windows, args.start_sec, args.end_sec)
        print(f"Restricted to [{args.start_sec}, {args.end_sec}] seconds: {before} -> {len(windows)} windows")
        if not windows:
            print("ERROR: no windows in that time range -- check your --start-sec/--end-sec against "
                  "the clip's actual length.")
            sys.exit(1)

    if args.stride > 1:
        before = len(windows)
        windows = windows[::args.stride]
        print(f"Stride {args.stride}: {before} -> {len(windows)} windows shown")

    if args.headless or not TKINTER_AVAILABLE:
        if not TKINTER_AVAILABLE and not args.headless:
            print("(Tkinter not available in this environment -- falling back to headless mode)")
        run_headless(windows, args.sport, args.windows, video_path=args.video)
    else:
        gui = LabelToolGUI(windows, args.sport, args.windows, video_path=args.video)
        gui.run()


if __name__ == "__main__":
    main()