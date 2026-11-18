"""
CASTER Phase 4 -- Standalone Python Desktop Broadcast Application.

Features:
- Native desktop window (built with Tkinter + PIL + OpenCV)
- Ultra-smooth video rendering with bounding boxes fused directly onto frames
- Live AI commentary panel with scrolling transcript
- Scoreboard and confirmed event stream
- Multithreaded text-to-speech commentary (pyttsx3) that never freezes playback
- Real-time precomputed stream support (60 fps fluid) with optional live CV pipeline

Run with:
    python phase4/desktop_app.py
    python phase4/desktop_app.py --clip rm1
"""

import os
import sys
import time
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import OUTPUTS_DIR, DATA_DIR
RAW_DATA_DIR = DATA_DIR / "raw"
from phase3.inference.commentary_engine import CommentaryEngine

# Audio engine
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

CLASS_COLORS = {
    'ball':             (0, 240, 255),    # Vibrant Yellow/Cyan in BGR
    'player':           (0, 230, 118),    # Emerald Green
    'player_team_a':    (0, 230, 118),
    'player_team_b':    (255, 176, 0),    # Bright Blue
    'goalkeeper':       (249, 0, 213),    # Magenta
    'referee':          (0, 109, 255),    # High-vis Orange
    'umpire':           (0, 109, 255),
}
DEFAULT_COLOR = (255, 255, 255)


class DesktopBroadcastApp(tk.Tk):
    def __init__(self, clip_name="rm1"):
        super().__init__()
        self.title("CASTER — Computer Vision Assisted Sports Text And Event Reporter")
        self.geometry("1400x860")
        self.configure(bg="#08090d")

        self.clip_name = clip_name
        self.video_cap = None
        self.is_playing = False
        self.current_frame_idx = 0
        self.fps = 30.0
        self.total_frames = 0
        self.commentary_engine = None

        # Thread-safe event queues
        self.tts_queue = queue.Queue()
        self.commentary_queue = queue.Queue()

        # Preloaded scene data
        self.scene_states = []
        self.events = []
        self.load_clip_data(self.clip_name)

        # Build UI layout
        self._build_ui()

        # Start background TTS thread
        if HAS_TTS:
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

        # Periodic queue checker for UI updates
        self.after(50, self._process_queues)

        # Auto-start video
        self.start_playback()

    def load_clip_data(self, clip):
        states_path = OUTPUTS_DIR / f"{clip}_scene_states.json"
        events_path = OUTPUTS_DIR / f"{clip}_scene_json_events.json"

        self.scene_states = []
        self.events = []

        if states_path.exists():
            with open(states_path, "r", encoding="utf-8") as f:
                self.scene_states = json.load(f)

        if events_path.exists():
            with open(events_path, "r", encoding="utf-8") as f:
                self.events = json.load(f)

        # Video source
        vpath = RAW_DATA_DIR / f"{clip}.mp4"
        if not vpath.exists():
            for ext in (".avi", ".mov", ".mkv"):
                cand = vpath.with_suffix(ext)
                if cand.exists():
                    vpath = cand
                    break

        if vpath.exists():
            if self.video_cap:
                self.video_cap.release()
            self.video_cap = cv2.VideoCapture(str(vpath))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.current_frame_idx = 0

    def _build_ui(self):
        # Header bar
        header = tk.Frame(self, bg="#10121a", height=60, bd=0)
        header.pack(fill=tk.X, side=tk.TOP, padx=16, pady=10)

        logo_label = tk.Label(header, text="⚽ CASTER", font=("Segoe UI", 18, "bold"), fg="#d4ff00", bg="#10121a")
        logo_label.pack(side=tk.LEFT, padx=10)

        sub_label = tk.Label(
            header,
            text="COMPUTER VISION ASSISTED SPORTS TEXT AND EVENT REPORTER",
            font=("Segoe UI", 8, "bold"),
            fg="#94a3b8",
            bg="#10121a"
        )
        sub_label.pack(side=tk.LEFT, padx=10)

        self.badge_status = tk.Label(
            header,
            text="● ON AIR (DESKTOP ENGINE)",
            font=("Consolas", 9, "bold"),
            fg="#000",
            bg="#00f5a0",
            padx=10,
            pady=3
        )
        self.badge_status.pack(side=tk.RIGHT, padx=10)

        # Main Workspace Container
        main_frame = tk.Frame(self, bg="#08090d")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        # Left Column: Video display + Telemetry
        left_col = tk.Frame(main_frame, bg="#08090d")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        # Video Canvas
        self.canvas_video = tk.Canvas(left_col, bg="#000000", highlightthickness=2, highlightbackground="#222736")
        self.canvas_video.pack(fill=tk.BOTH, expand=True)

        # Video Controls
        ctl_bar = tk.Frame(left_col, bg="#121520", height=45)
        ctl_bar.pack(fill=tk.X, pady=(10, 0))

        self.btn_play = tk.Button(
            ctl_bar, text="⏸ PAUSE", font=("Segoe UI", 9, "bold"),
            bg="#222736", fg="#fff", activebackground="#d4ff00", activeforeground="#000",
            bd=0, padx=14, pady=6, command=self.toggle_play
        )
        self.btn_play.pack(side=tk.LEFT, padx=8, pady=6)

        self.btn_restart = tk.Button(
            ctl_bar, text="🔄 REPLAY", font=("Segoe UI", 9, "bold"),
            bg="#222736", fg="#fff", activebackground="#00f2fe", activeforeground="#000",
            bd=0, padx=14, pady=6, command=self.restart_playback
        )
        self.btn_restart.pack(side=tk.LEFT, padx=8, pady=6)

        self.lbl_time = tk.Label(ctl_bar, text="00:00 / 00:00", font=("Consolas", 10, "bold"), fg="#d4ff00", bg="#121520")
        self.lbl_time.pack(side=tk.RIGHT, padx=14)

        # Right Column: Scoreboard, Live Caption, Event Feed & Transcript
        right_col = tk.Frame(main_frame, bg="#08090d", width=460)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_col.pack_propagate(False)

        # Top Row: Scoreboard & Status
        top_stats = tk.Frame(right_col, bg="#121520", bd=1, relief=tk.SOLID)
        top_stats.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top_stats, text="⚡ LIVE MATCH SCORE", font=("Segoe UI", 8, "bold"), fg="#94a3b8", bg="#121520").pack(anchor="w", padx=10, pady=(6, 0))
        self.lbl_score = tk.Label(top_stats, text="0 - 0", font=("Segoe UI", 32, "bold"), fg="#ffffff", bg="#121520")
        self.lbl_score.pack(pady=4)

        # Live Commentary Banner
        caption_box = tk.Frame(right_col, bg="#161b26", bd=1, relief=tk.SOLID)
        caption_box.pack(fill=tk.X, pady=(0, 10))

        tk.Label(caption_box, text="🎙️ LIVE TELEPROMPTER COMMENTARY", font=("Segoe UI", 8, "bold"), fg="#d4ff00", bg="#161b26").pack(anchor="w", padx=10, pady=(6, 4))
        self.lbl_caption = tk.Label(
            caption_box,
            text="Initializing pitch tracking & AI commentator...",
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff",
            bg="#161b26",
            wraplength=420,
            justify=tk.LEFT,
            padx=10,
            pady=10
        )
        self.lbl_caption.pack(fill=tk.X)

        # Event Feed Box
        event_box = tk.Frame(right_col, bg="#121520", bd=1, relief=tk.SOLID)
        event_box.pack(fill=tk.X, pady=(0, 10))

        tk.Label(event_box, text="⚡ CONFIRMED MATCH EVENTS", font=("Segoe UI", 8, "bold"), fg="#94a3b8", bg="#121520").pack(anchor="w", padx=10, pady=(6, 4))
        self.list_events = tk.Listbox(event_box, bg="#0d0f17", fg="#00f2fe", font=("Consolas", 9), bd=0, height=4, selectbackground="#d4ff00")
        self.list_events.pack(fill=tk.X, padx=8, pady=(0, 8))

        # Full Transcript Archive Box
        transcript_box = tk.Frame(right_col, bg="#121520", bd=1, relief=tk.SOLID)
        transcript_box.pack(fill=tk.BOTH, expand=True)

        tk.Label(transcript_box, text="📜 FULL COMMENTARY ARCHIVE", font=("Segoe UI", 8, "bold"), fg="#94a3b8", bg="#121520").pack(anchor="w", padx=10, pady=(6, 4))
        self.txt_transcript = tk.Text(transcript_box, bg="#0d0f17", fg="#ffffff", font=("Segoe UI", 9), bd=0, wrap=tk.WORD)
        self.txt_transcript.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(text="▶ PLAY" if not self.is_playing else "⏸ PAUSE")
        if self.is_playing:
            self._render_loop()

    def restart_playback(self):
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
        self.is_playing = True
        self.btn_play.config(text="⏸ PAUSE")
        self._render_loop()

    def start_playback(self):
        self.is_playing = True
        self._render_loop()

    def _render_loop(self):
        if not self.is_playing or not self.video_cap or not self.video_cap.isOpened():
            return

        ret, frame = self.video_cap.read()
        if not ret:
            # Loop video smoothly
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            ret, frame = self.video_cap.read()
            if not ret:
                return

        self.current_frame_idx += 1
        t_sec = self.current_frame_idx / self.fps

        # Update Time Label
        cur_min = int(t_sec // 60)
        cur_s = int(t_sec % 60)
        tot_sec = self.total_frames / self.fps
        tot_min = int(tot_sec // 60)
        tot_s = int(tot_sec % 60)
        self.lbl_time.config(text=f"{cur_min:02d}:{cur_s:02d} / {tot_min:02d}:{tot_s:02d}")

        # Find closest precomputed detection state
        detections = self._get_detections_for_sec(t_sec)

        # Draw detections directly onto frame array
        h, w = frame.shape[:2]
        for det in detections:
            cls_name = det.get("cls_name", "player")
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            track_id = det.get("track_id")

            # Check if coords are normalized (0-1) or absolute pixels
            x = det.get("x", 0.0)
            y = det.get("y", 0.0)
            bw = det.get("w", 0.0)
            bh = det.get("h", 0.0)

            if x <= 1.0 and bw <= 1.0:
                px = int((x - bw / 2.0) * w)
                py = int((y - bh / 2.0) * h)
                pw = int(bw * w)
                ph = int(bh * h)
            else:
                px = int(x - bw / 2.0)
                py = int(y - bh / 2.0)
                pw = int(bw)
                ph = int(bh)

            cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)
            label = f"{cls_name} #{track_id}" if track_id is not None else cls_name
            cv2.putText(frame, label, (px, max(18, py - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check for event triggers at this timestamp
        self._check_events(t_sec)

        # Convert OpenCV BGR to Tkinter PhotoImage
        canvas_w = max(100, self.canvas_video.winfo_width())
        canvas_h = max(100, self.canvas_video.winfo_height())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame_rgb)
        im = im.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image=im)

        self.canvas_video.delete("all")
        self.canvas_video.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Next frame delay in ms based on video fps
        delay_ms = max(1, int(1000.0 / self.fps))
        self.after(delay_ms, self._render_loop)

    def _get_detections_for_sec(self, t_sec):
        if not self.scene_states:
            return []
        # Find closest state
        closest = min(self.scene_states, key=lambda s: abs(s.get("timestamp_sec", 0.0) - t_sec))
        if abs(closest.get("timestamp_sec", 0.0) - t_sec) <= 1.2:
            return closest.get("detections", [])
        return []

    def _check_events(self, t_sec):
        for ev in list(self.events):
            ev_time = self._parse_event_time(ev)
            if ev_time >= 0 and abs(t_sec - ev_time) < 0.3:
                # Trigger event commentary
                self.events.remove(ev)
                self._trigger_commentary(ev)
                break

    def _parse_event_time(self, ev):
        ct = ev.get("clip_time")
        if not ct:
            return -1.0
        parts = str(ct).split(":")
        if len(parts) == 2:
            try:
                return float(parts[0]) * 60 + float(parts[1])
            except ValueError:
                return -1.0
        try:
            return float(ct)
        except ValueError:
            return -1.0

    def _trigger_commentary(self, ev):
        event_name = ev.get("event", "ACTION")
        conf = ev.get("confidence", 0.85)
        score = ev.get("score", "0 - 0")

        self.lbl_score.config(text=score)
        self.list_events.insert(0, f"⚡ {event_name} ({int(conf * 100)}%) — {score}")

        # Generate in separate thread so video playback is 100% fluid!
        threading.Thread(target=self._generate_commentary_worker, args=(ev,), daemon=True).start()

    def _generate_commentary_worker(self, ev):
        if self.commentary_engine is None:
            self.commentary_engine = CommentaryEngine()

        try:
            res = self.commentary_engine.generate(ev)
            text = res.get("text", "")
            if text:
                self.commentary_queue.put(text)
                self.tts_queue.put(text)
        except Exception as e:
            fallback = f"Key moment on the pitch as {ev.get('event', 'the action')} unfolds!"
            self.commentary_queue.put(fallback)
            self.tts_queue.put(fallback)

    def _process_queues(self):
        try:
            while not self.commentary_queue.empty():
                text = self.commentary_queue.get_nowait()
                self.lbl_caption.config(text=text)
                self.txt_transcript.insert(tk.END, f"• {text}\n\n")
                self.txt_transcript.see(tk.END)
        except queue.Empty:
            pass

        self.after(50, self._process_queues)

    def _tts_worker(self):
        if not HAS_TTS:
            return
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        while True:
            text = self.tts_queue.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
            self.tts_queue.task_done()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="rm1", help="Clip name (stem in data/raw)")
    args = parser.parse_args()

    app = DesktopBroadcastApp(clip_name=args.clip)
    app.mainloop()
