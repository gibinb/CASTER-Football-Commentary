"""
Unit tests for the XGBoost event classifier -- verifies training,
prediction, and persistence (save/load) using throwaway synthetic data
in an ISOLATED temp directory. FOOTBALL ONLY.

IMPORTANT: these tests never read or write the real
data/labels/football.jsonl or phase2/saved_models/xgboost_football.json.
Both of those are real, actively-growing production artifacts once
you're running the Gemini auto-labeller against real clips -- if these
tests touched them directly, every `pytest` run would silently retrain
on whatever partial real data happened to exist at that moment and
overwrite your real trained model with the result. isolated_test_env()
below redirects both paths to a throwaway temp directory for the
duration of each test, so running the test suite is always safe to do
alongside real labelling work, in any order, any time.

Run with:
    python phase2/tests/test_xgboost_clf.py
or:
    python -m pytest phase2/tests/test_xgboost_clf.py -v
"""

import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import SPORT_FOOTBALL, EVENTS_BY_SPORT, ZONES_BY_SPORT
from phase2.labelling.generate_synthetic_data import generate, football_window_for_event
from phase2.event_classifier.xgboost_clf import train_and_save, load_classifier, predict_window, model_path_for


@contextmanager
def isolated_test_env():
    """
    Redirects FEATURE_WINDOWS_LABEL_DIR (used by label_tool.py, which
    generate_synthetic_data.py writes through, and by dataset.py, which
    xgboost_clf.py reads through) and SAVED_MODELS_DIR (used by
    xgboost_clf.py) to a throwaway temp directory for the duration of
    the `with` block. Each module bound its own copy of these names at
    import time (`from config import X`), so all three copies must be
    patched together -- patching config.X alone would not affect code
    that already imported X by value.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_labels = Path(tmp) / "labels"
        tmp_models = Path(tmp) / "saved_models"
        tmp_labels.mkdir()
        tmp_models.mkdir()
        with patch("phase2.labelling.label_tool.FEATURE_WINDOWS_LABEL_DIR", tmp_labels), \
             patch("phase2.event_classifier.dataset.FEATURE_WINDOWS_LABEL_DIR", tmp_labels), \
             patch("phase2.event_classifier.xgboost_clf.SAVED_MODELS_DIR", tmp_models):
            yield


def _fresh_synthetic_model(sport: str):
    """Generates fresh synthetic data and trains a model, both inside
    the caller's isolated_test_env() context."""
    generate(sport, n_per_class=40, source_tag="test_synthetic")
    train_and_save(sport, verbose=False)
    return load_classifier(sport)


def test_training_produces_model_file():
    with isolated_test_env():
        generate(SPORT_FOOTBALL, n_per_class=40, source_tag="test_synthetic")
        path = model_path_for(SPORT_FOOTBALL)
        train_and_save(SPORT_FOOTBALL, verbose=False)
        assert path.exists(), f"Expected model file at {path}"
    print("PASS: training produces a saved model file")


def test_load_classifier_returns_usable_model():
    with isolated_test_env():
        clf = _fresh_synthetic_model(SPORT_FOOTBALL)
        assert clf is not None
    print("PASS: load_classifier returns a usable model")


def test_predict_window_returns_valid_event_and_confidence():
    with isolated_test_env():
        clf = _fresh_synthetic_model(SPORT_FOOTBALL)
        dummy_window = [[0.0] * 12 for _ in range(30)]
        event, confidence = predict_window(clf, SPORT_FOOTBALL, dummy_window)
        assert event in EVENTS_BY_SPORT[SPORT_FOOTBALL], f"Unexpected event: {event}"
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
    print(f"PASS: predict_window returns valid event '{event}' with confidence {confidence:.3f}")


def test_football_classifier_trains_and_predicts():
    with isolated_test_env():
        clf = _fresh_synthetic_model(SPORT_FOOTBALL)
        dummy_window = [[0.0] * 12 for _ in range(30)]
        event, confidence = predict_window(clf, SPORT_FOOTBALL, dummy_window)
        assert event in EVENTS_BY_SPORT[SPORT_FOOTBALL], f"Unexpected event: {event}"
        assert 0.0 <= confidence <= 1.0
    print(f"PASS: football classifier trains and predicts '{event}' with confidence {confidence:.3f}")


def test_predict_goal_from_synthetic_features():
    with isolated_test_env():
        clf = _fresh_synthetic_model(SPORT_FOOTBALL)
        n_zones = len(ZONES_BY_SPORT[SPORT_FOOTBALL])
        goal_window = football_window_for_event("GOAL", n_zones)
        event, confidence = predict_window(clf, SPORT_FOOTBALL, goal_window)
        assert event == "GOAL", f"Expected GOAL, got {event}"
        assert confidence > 0.7
    print(f"PASS: GOAL synthetic window correctly classified with confidence {confidence:.3f}")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
