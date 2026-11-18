"""
Shared configuration for the whole CASTER project -- FOOTBALL ONLY.

Every phase imports from here instead of hardcoding paths or thresholds,
so changing a value in one place updates the entire pipeline.

Scope note: CASTER used to cover cricket as well. It is now football
(soccer) only. The *_BY_SPORT dictionaries below deliberately still exist,
holding a single "football" entry, so that Phase 2-4 code written against
that shape keeps working while those phases are converted. Any remaining
reference to SPORT_CRICKET / CRICKET_* in a later phase is a leftover that
should be deleted, not restored.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root -- sets GROQ_API_KEY, CASTER_IMGSZ, etc.
               # so you never need `export`/`$env:` in any terminal again.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
LABELS_DIR = DATA_DIR / "labels"
OUTPUTS_DIR = ROOT_DIR / "outputs"

PHASE1_DIR = ROOT_DIR / "phase1"
WEIGHTS_DIR = PHASE1_DIR / "weights"

OUTPUTS_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# Sport identifier -- written into every JSON output so later phases and the
# frontend always know what vocabulary they are looking at.
# ---------------------------------------------------------------------------

SPORT_FOOTBALL = "football"
SUPPORTED_SPORTS = (SPORT_FOOTBALL,)

# ---------------------------------------------------------------------------
# Detection classes
# ---------------------------------------------------------------------------
# These are the 4 classes of the real fine-tuning dataset (Roboflow
# "football-player-detection-bfswn"). No public dataset labels team
# affiliation as a YOLO class, since jersey colours change every match, so
# team separation remains documented future work (K-means on jersey colour
# histograms, applied AFTER generic 'player' detection).
# NOTE: the shipped weights/football.pt names the player class "players";
# phase1/detection/yolo_detector.py aliases that to "player".

FOOTBALL_CLASSES = [
    "ball",
    "goalkeeper",
    "player",
    "referee",
]

CLASSES_BY_SPORT = {SPORT_FOOTBALL: FOOTBALL_CLASSES}

# ---------------------------------------------------------------------------
# Pipeline thresholds
# ---------------------------------------------------------------------------

YOLO_CONFIDENCE_THRESHOLD = 0.4     # legacy default; the live pipeline uses per-class thresholds
                                    # (CASTER_PLAYER_CONF / CASTER_BALL_CONF / CASTER_REFEREE_CONF,
                                    # plus the opt-in CASTER_USE_SAHI sliced-inference mode -- all
                                    # in yolo_detector.py)
TRACKER_CONFIG = "bytetrack.yaml"   # legacy name; the real config is phase1/configs/caster_bytetrack.yaml
FEATURE_WINDOW_SIZE = 30            # frames per sliding window (~1s at 30fps)
FEATURE_VECTOR_SIZE = 12            # see schemas.py for the index map
TARGET_MAP50 = 0.55                 # deliberately modest detection bar (Phase 1 doc, Sec 2.6)

# Pitch zone vocabulary (index 7 of the feature vector)
FOOTBALL_ZONES = ["penalty_box", "midfield", "wing", "goal_line"]

ZONES_BY_SPORT = {SPORT_FOOTBALL: FOOTBALL_ZONES}

# ---------------------------------------------------------------------------
# Phase 2 -- event classes
# ---------------------------------------------------------------------------
# Labels the event classifier is trained to output. Order matters: label
# encoding (XGBoost) and the classifier's output layer (BiLSTM) both index
# into this list by position, so do not reorder it once you've labelled data
# or trained a model against it -- only append.

# A dedicated "nothing commentary-worthy happened" class. This MUST stay
# the LAST entry (label ids are positional -- see the ordering note
# above -- appending preserves every id already labelled or trained
# against; inserting earlier would silently relabel every event after
# it). Without this class, every single classified window was forced
# into one of the ten "real" events below even when the footage showed
# completely ordinary play, which is the overwhelming majority of any
# match -- the classifier had no way to say "nothing happened" and was
# structurally forced to guess a named event instead. That is the
# direct, confirmed cause of real-trained models collapsing onto one
# dominant class (DRIBBLE) for ~84-96% of all real-clip predictions.
# confidence_gate.py ALWAYS discards a NO_EVENT prediction regardless of
# confidence -- it is a training-time class only; Phase 3 never sees it.
NO_EVENT = "NO_EVENT"

FOOTBALL_EVENTS = [
    "PASS",
    "DRIBBLE",
    "TACKLE",
    "SHOT_ON_TARGET",
    "GOAL",
    "FOUL",
    "CORNER_KICK",
    "FREE_KICK",
    "HEADER",
    "OFFSIDE",
    NO_EVENT,
]

EVENTS_BY_SPORT = {SPORT_FOOTBALL: FOOTBALL_EVENTS}

# Events that get a separately-confirmed, higher-confidence path
# (PRIORITY_MIN_CONFIDENCE below). They still respect the cooldown: the
# classifier runs on heavily-overlapping sliding windows, so one real GOAL
# produces many consecutive raw predictions, and without a cooldown even a
# genuinely confident classifier would flood the feed with duplicates.
FOOTBALL_PRIORITY_EVENTS = {"GOAL"}

PRIORITY_EVENTS_BY_SPORT = {SPORT_FOOTBALL: FOOTBALL_PRIORITY_EVENTS}

# ---------------------------------------------------------------------------
# Phase 2 -- confidence gate thresholds
# ---------------------------------------------------------------------------
# Football's continuous play (passes and dribbles blending into each other)
# gets a moderate bar so the system doesn't go silent during build-up play.
CONFIDENCE_THRESHOLD_BY_SPORT = {SPORT_FOOTBALL: 0.70}

# A classifier runs on heavily-overlapping windows (one prediction per
# video frame after the first 30 frames). Every accepted event, including a
# goal, must therefore be debounced; otherwise one physical action becomes
# dozens of identical predictions and commentary lines.
MIN_SECONDS_BETWEEN_EVENTS = 6.0

# Priority events (GOAL) need a SEPARATELY HIGH confidence bar: an incorrect
# GOAL announcement is more disruptive to trust than a missed low-confidence
# prediction.
PRIORITY_MIN_CONFIDENCE = 0.85

# Phase 2 directories
PHASE2_DIR = ROOT_DIR / "phase2"
SAVED_MODELS_DIR = PHASE2_DIR / "saved_models"
FEATURE_WINDOWS_LABEL_DIR = DATA_DIR / "labels"

SAVED_MODELS_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# Phase 4 -- Integration and Polish
# ---------------------------------------------------------------------------

PHASE4_DIR = ROOT_DIR / "phase4"

# FastAPI server defaults (override via CLI args in run_phase4.py)
PHASE4_BACKEND_HOST = "0.0.0.0"
PHASE4_BACKEND_PORT = 8000

# React / Vite dev server (used in CORS allow-list in main.py)
PHASE4_FRONTEND_URL = "http://localhost:5173"

# WebSocket message types emitted by pipeline_runner.py and ws_handler.py.
# Listed here so they are a single source of truth if the frontend
# ever needs to import them (via a generated constants file).
WS_MSG_SPORT_DETECTED = "sport_detected"
WS_MSG_FRAME          = "frame"
WS_MSG_EVENT          = "event"
WS_MSG_TOKEN          = "token"
WS_MSG_SENTENCE_END   = "sentence_end"
WS_MSG_SPEAK          = "speak"
WS_MSG_DONE           = "done"
WS_MSG_ERROR          = "error"
WS_MSG_WARNING        = "warning"
WS_MSG_STYLE_CHANGED  = "style_changed"
