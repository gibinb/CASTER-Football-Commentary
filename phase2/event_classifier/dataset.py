"""
CASTER Phase 2 -- shared dataset loading utility.

Both xgboost_clf.py and bilstm_clf.py train on the exact same labelled
data (data/labels/<sport>.jsonl), just reshaped differently:

    XGBoost wants flat feature vectors -- it averages/aggregates each
    30-frame window down to a single row of summary statistics, since
    it has no native concept of a sequence.

    BiLSTM wants the full (30, 12) sequence intact, since the whole
    point of using an LSTM is to let it learn temporal patterns the
    flattened XGBoost view throws away.

Centralising the loading + label-encoding logic here means both
classifiers stay perfectly consistent about which event maps to which
integer label (alphabetical position in config.EVENTS_BY_SPORT[sport]
-- never re-derive this from whatever happens to appear in the data
file, since that order can change as you label more examples).
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EVENTS_BY_SPORT, FEATURE_WINDOWS_LABEL_DIR, FEATURE_VECTOR_SIZE


def label_path_for(sport: str) -> Path:
    return FEATURE_WINDOWS_LABEL_DIR / f"{sport}.jsonl"


def event_to_label_id(sport: str, event: str) -> int:
    events = EVENTS_BY_SPORT[sport]
    if event not in events:
        raise ValueError(
            f"'{event}' is not a registered {sport} event. "
            f"Valid events: {events}. Check config.EVENTS_BY_SPORT for typos."
        )
    return events.index(event)


def label_id_to_event(sport: str, label_id: int) -> str:
    return EVENTS_BY_SPORT[sport][label_id]


def load_labelled_windows(sport: str) -> Tuple[List[List[List[float]]], List[int]]:
    """
    Reads data/labels/<sport>.jsonl and returns:
        X_windows: list of (30, 12) windows (as plain nested lists)
        y:         list of integer label ids, same order, same length
    """
    path = label_path_for(sport)
    if not path.exists():
        raise FileNotFoundError(
            f"No labelled data found at {path}. Run either:\n"
            f"  python phase2/labelling/label_tool.py --windows <phase1_output> --sport {sport}\n"
            f"  python phase2/labelling/generate_synthetic_data.py --sport {sport}\n"
            f"before training."
        )

    X_windows, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            vectors = row["vectors"]
            if not vectors or len(vectors[0]) != FEATURE_VECTOR_SIZE:
                continue  # skip malformed rows rather than crashing the whole load
            X_windows.append(vectors)
            y.append(event_to_label_id(sport, row["event"]))

    return X_windows, y


def windows_to_flat_features(X_windows: List[List[List[float]]]) -> np.ndarray:
    """
    Aggregates each (30, 12) window down to a flat feature row for
    XGBoost. Uses mean + std + max across the time axis for every one
    of the 12 feature channels -- 36 columns total. Mean captures the
    average state of play, std captures how much it changed during the
    window (a sudden spike vs. steady state), max captures peak values
    (e.g. the fastest the ball moved at any point in the window, which
    matters a lot for distinguishing a FOUR from a DOT_BALL even if the
    average speed across the window looks similar).
    """
    arr = np.array(X_windows)  # shape (n_samples, 30, 12)
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    mx = arr.max(axis=1)
    return np.concatenate([mean, std, mx], axis=1)  # shape (n_samples, 36)


def class_counts(sport: str, y: List[int]) -> dict:
    events = EVENTS_BY_SPORT[sport]
    counts = {ev: 0 for ev in events}
    for label_id in y:
        counts[events[label_id]] += 1
    return counts


def print_class_distribution(sport: str, y: List[int]):
    counts = class_counts(sport, y)
    print(f"\nClass distribution for {sport} ({len(y)} total windows):")
    for ev, n in counts.items():
        bar = "#" * min(n, 50)
        print(f"  {ev:<16} {n:>4}  {bar}")
    zero_classes = [ev for ev, n in counts.items() if n == 0]
    if zero_classes:
        print(f"\n  WARNING: no examples at all for: {zero_classes}")
        print("  The classifier cannot learn classes it has never seen.")
