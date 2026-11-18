"""
CASTER Phase 2 -- XGBoost event classifier (MVP).

This is the classifier you should get working FIRST. It trains in
minutes on a laptop CPU, tolerates the small/imbalanced labelled
dataset you'll realistically have after labelling 500-1000 windows,
and de-risks the rest of the pipeline (confidence gate, JSON builder,
Phase 3 hookup) before you invest time in the heavier BiLSTM.

FOOTBALL ONLY. train_and_save("football") trains and saves the
football event classifier.

Usage:
    python phase2/event_classifier/xgboost_clf.py --sport football
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EVENTS_BY_SPORT, SAVED_MODELS_DIR, SUPPORTED_SPORTS

from phase2.event_classifier.dataset import (
    load_labelled_windows, windows_to_flat_features, print_class_distribution,
    label_id_to_event,
)

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


def model_path_for(sport: str) -> Path:
    return SAVED_MODELS_DIR / f"xgboost_{sport}.json"


def train_and_save(sport: str, test_size: float = 0.2, verbose: bool = True) -> "xgb.Booster":
    """
    Trains via XGBoost's native Booster API (xgb.train), NOT the
    sklearn XGBClassifier wrapper.

    Why this matters: dataset.py deliberately encodes each event by its
    FIXED position in the full config.EVENTS_BY_SPORT[sport] list (e.g.
    GOAL is always label 4, OFFSIDE is always label 9), specifically so
    that mapping stays stable as you label more real data over time --
    it must NOT depend on which classes happen to appear in today's
    data file. But sklearn's XGBClassifier.fit() requires labels to be
    dense and contiguous (0..n_classes_present-1) and silently remaps
    them -- which breaks the moment your real labelled data is missing
    even one class (the normal case early in real labelling: you will
    not have a GOAL example in your first 50 rows). The native Booster
    API has no such requirement: num_class stays fixed at the FULL
    event-list length regardless of what's actually present, and any
    label < num_class is valid even if it never appears in training.
    """
    if not DEPS_AVAILABLE:
        raise ImportError(
            "xgboost and scikit-learn are required for this script. "
            "Install with: pip install xgboost scikit-learn"
        )

    X_windows, y = load_labelled_windows(sport)
    if verbose:
        print_class_distribution(sport, y)

    X = windows_to_flat_features(X_windows)
    y = np.array(y)
    num_classes_total = len(EVENTS_BY_SPORT[sport])

    n_classes_present = len(set(y))
    can_stratify = n_classes_present > 1 and min(np.bincount(y)) >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if can_stratify else None,
    )

    # Inverse-frequency sample weighting (same formula as sklearn's
    # class_weight="balanced"): weight = n_samples / (n_classes_present *
    # class_count). Without this, a class with 1 training example (e.g.
    # CORNER_KICK, HEADER in early real data) contributes essentially
    # nothing to the loss compared to a class with 11+ examples (DRIBBLE),
    # so the model is implicitly optimised to just predict the majority
    # class whenever uncertain -- exactly the collapse we saw on rm1
    # before NO_EVENT existed, and the same failure mode would recur
    # between the remaining classes even with it. Weights are computed
    # from the TRAINING split only and applied only to dtrain -- the test
    # split and its printed report stay unweighted, so the accuracy
    # numbers you see still reflect real-world class frequency, not an
    # artificially inflated view of rare-class performance.
    train_class_counts = np.bincount(y_train, minlength=num_classes_total)
    n_train_classes_present = int((train_class_counts > 0).sum())
    sample_weight = np.array([
        len(y_train) / (n_train_classes_present * train_class_counts[label])
        for label in y_train
    ])

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    params = {
        "max_depth": 6,
        "eta": 0.1,              # learning_rate, in native Booster param naming
        "objective": "multi:softprob",
        "num_class": num_classes_total,
        "eval_metric": "mlogloss",
    }
    bst = xgb.train(params, dtrain, num_boost_round=200)

    if verbose and len(X_test) > 0:
        dtest = xgb.DMatrix(X_test)
        probs = bst.predict(dtest)
        y_pred = probs.argmax(axis=1)
        target_names = EVENTS_BY_SPORT[sport]
        labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
        print(f"\n--- {sport} classification report ---")
        print(classification_report(
            y_test, y_pred,
            labels=labels_present,
            target_names=[target_names[i] for i in labels_present],
            zero_division=0,
        ))

    out_path = model_path_for(sport)
    bst.save_model(str(out_path))
    print(f"Saved trained XGBoost model -> {out_path}")
    return bst


def load_classifier(sport: str) -> "xgb.Booster":
    """
    Loads a previously trained model. This is the function Phase 2's
    inference pipeline (run_phase2.py) actually calls -- training is a
    one-time offline step, inference loads the saved .json file.
    """
    if not DEPS_AVAILABLE:
        raise ImportError("xgboost is required. Install with: pip install xgboost")

    path = model_path_for(sport)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run:\n"
            f"  python phase2/event_classifier/xgboost_clf.py --sport {sport}\n"
            f"first."
        )
    bst = xgb.Booster()
    bst.load_model(str(path))
    return bst


def predict_window(clf: "xgb.Booster", sport: str, window_vectors) -> tuple:
    """
    Runs inference on a single (30, 12) window. Returns (event_name, confidence).
    This is the function run_phase2.py calls per window, after Phase 1
    hands it a freshly-completed FeatureWindow.
    """
    X = windows_to_flat_features([window_vectors])
    dmat = xgb.DMatrix(X)
    probs = clf.predict(dmat)[0]
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    event = label_id_to_event(sport, best_idx)
    return event, confidence


class XGBoostClassifier:
    """
    Thin object-oriented wrapper around the function-based API above
    (train_and_save / load_classifier / predict_window), so callers that
    prefer to hold a single classifier instance -- e.g. Phase 4's
    pipeline_runner.py, which loads a classifier once per video and
    calls .predict_window() repeatedly per frame window -- have a
    convenient interface without duplicating any logic.

        clf = XGBoostClassifier(sport="football")
        if clf.load():
            event, confidence = clf.predict_window(window_vectors)

    Internally this delegates to the exact same load_classifier() and
    predict_window() functions used by run_phase2.py and the test
    suite, so behaviour is identical either way you call it.
    """

    def __init__(self, sport: str):
        self.sport = sport
        self.model = None

    def load(self) -> bool:
        """Returns True if a trained model was found and loaded, False otherwise."""
        try:
            self.model = load_classifier(self.sport)
            return True
        except FileNotFoundError:
            self.model = None
            return False

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_window(self, window_vectors) -> tuple:
        """Same (event, confidence) return shape as the module-level predict_window()."""
        if self.model is None:
            raise RuntimeError(
                f"No model loaded for sport='{self.sport}'. Call .load() first, "
                f"or train one with: python phase2/event_classifier/xgboost_clf.py --sport {self.sport}"
            )
        return predict_window(self.model, self.sport, window_vectors)

    def train(self, verbose: bool = True):
        """Convenience wrapper around the module-level train_and_save()."""
        self.model = train_and_save(self.sport, verbose=verbose)
        return self.model


def main():
    parser = argparse.ArgumentParser(description="Train CASTER's XGBoost event classifier")
    parser.add_argument("--sport", required=True, choices=list(SUPPORTED_SPORTS) + ["both"])
    args = parser.parse_args()

    sports = list(SUPPORTED_SPORTS) if args.sport == "both" else [args.sport]
    for sport in sports:
        print(f"\n{'=' * 60}\nTraining XGBoost classifier: {sport}\n{'=' * 60}")
        train_and_save(sport)


if __name__ == "__main__":
    main()