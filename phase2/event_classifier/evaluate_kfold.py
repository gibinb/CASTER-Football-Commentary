"""
CASTER Phase 2 -- stratified k-fold evaluation of the XGBoost event classifier.

WHY THIS EXISTS
----------------
Every classifier judgment made on this project so far came from
eyeballing raw predictions on one clip, or a single noisy 80/20
train/test split that changes completely depending on which 20%
happened to land in the test set. Neither answers, honestly, "if I
handed this model a new real window tomorrow, how often would it be
right" -- and neither can tell apart a class that's weak because it's
RARE in training data from a class that's weak because it genuinely
looks like another class (TACKLE vs FOUL, DRIBBLE vs PASS) to the
features Phase 1 hands it.

Stratified k-fold answers that properly: split the real labelled data
into K folds, each preserving the overall class proportions as closely
as the data allows, train K separate models (each held out from one
fold in turn), evaluate each on its own held-out fold, then report the
MEAN AND SPREAD across folds -- not a single number. A high spread
across folds is itself a signal: it means the result you'd see on any
one clip is closer to luck than to a real measurement.

This also produces a POOLED confusion matrix across all folds -- the
real, data-grounded answer to "should two classes be merged" (the
Phase 2 deep-dive doc's own documented fallback for persistently
confused classes), instead of guessing from a handful of raw
predictions on one clip.

WHAT THIS DOES NOT DO
----------------------
This does not replace xgboost_clf.py's train_and_save(), which still
trains the one deployed model on ALL available real data (more data
beats held-out evaluation once the method itself is trusted). This
script is a diagnostic you run periodically to decide: is the
classifier good enough yet, which classes most need more real data,
and should any classes be merged -- run_phase2.py never calls this at
inference time.

Usage:
    python phase2/event_classifier/evaluate_kfold.py --sport football
    python phase2/event_classifier/evaluate_kfold.py --sport football --k 5
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EVENTS_BY_SPORT, SUPPORTED_SPORTS, SAVED_MODELS_DIR

try:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

from phase2.event_classifier.dataset import load_labelled_windows, windows_to_flat_features


def _sample_weights(y_train, num_classes_total):
    """Same inverse-frequency weighting as xgboost_clf.py's train_and_save
    uses -- so this evaluates the method actually shipped, not a
    stripped-down version of it."""
    counts = np.bincount(y_train, minlength=num_classes_total)
    n_present = int((counts > 0).sum())
    return np.array([len(y_train) / (n_present * counts[label]) for label in y_train])


def run_kfold(sport: str, k: int, random_state: int = 42):
    if not DEPS_AVAILABLE:
        raise ImportError("xgboost and scikit-learn are required. Install with: pip install xgboost scikit-learn")

    X_windows, y = load_labelled_windows(sport)
    X = windows_to_flat_features(X_windows)
    y = np.array(y)
    num_classes_total = len(EVENTS_BY_SPORT[sport])
    class_names = EVENTS_BY_SPORT[sport]

    present_classes = sorted(set(y.tolist()))
    counts = Counter(y.tolist())
    min_class_count = min(counts[c] for c in present_classes)

    # StratifiedKFold needs at least k examples of every class to put one
    # in each fold -- auto-reduce k rather than crash, and say so clearly,
    # since forcing a fixed k regardless of data would either error out or
    # (worse) silently produce folds that aren't actually stratified.
    effective_k = min(k, min_class_count)
    if effective_k < 2:
        thin = [class_names[c] for c in present_classes if counts[c] == 1]
        print(f"ERROR: at least one class has only 1 example -- stratified k-fold needs at "
              f"least 2 per class to run at all. Classes with only 1 example: {', '.join(thin)}")
        print("Label more real examples of these classes before running k-fold evaluation.")
        sys.exit(1)
    if effective_k < k:
        thin = [f"{class_names[c]} ({counts[c]})" for c in present_classes if counts[c] < k]
        print(f"NOTE: reduced k from {k} to {effective_k} because these classes have fewer than "
              f"{k} examples: {', '.join(thin)}. Their numbers below will be noisier than the "
              f"better-represented classes -- that's the data talking, not a bug in this script.\n")

    skf = StratifiedKFold(n_splits=effective_k, shuffle=True, random_state=random_state)

    per_class_precision = defaultdict(list)
    per_class_recall = defaultdict(list)
    per_class_f1 = defaultdict(list)
    fold_accuracies = []
    fold_macro_f1 = []
    confusion = np.zeros((num_classes_total, num_classes_total), dtype=int)  # rows=true, cols=pred

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        weights = _sample_weights(y_train, num_classes_total)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
        dtest = xgb.DMatrix(X_test)
        params = {
            "max_depth": 6, "eta": 0.1, "objective": "multi:softprob",
            "num_class": num_classes_total, "eval_metric": "mlogloss",
        }
        bst = xgb.train(params, dtrain, num_boost_round=200)
        probs = bst.predict(dtest)
        y_pred = probs.argmax(axis=1)

        fold_accuracies.append(float((y_pred == y_test).mean()))

        fold_f1s = []
        for c in present_classes:
            tp = int(((y_pred == c) & (y_test == c)).sum())
            fp = int(((y_pred == c) & (y_test != c)).sum())
            fn = int(((y_pred != c) & (y_test == c)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class_precision[c].append(precision)
            per_class_recall[c].append(recall)
            per_class_f1[c].append(f1)
            fold_f1s.append(f1)
        fold_macro_f1.append(float(np.mean(fold_f1s)))

        for true_c, pred_c in zip(y_test.tolist(), y_pred.tolist()):
            confusion[true_c][pred_c] += 1

    # ---- report ----
    print(f"{'=' * 78}")
    print(f"Stratified {effective_k}-fold evaluation -- {sport} -- "
          f"{len(y)} real labelled windows, {len(present_classes)} classes present")
    print(f"{'=' * 78}\n")

    print(f"Overall accuracy:  {np.mean(fold_accuracies) * 100:5.1f}%  "
          f"(+/- {np.std(fold_accuracies) * 100:.1f} across folds)")
    print(f"Macro F1:          {np.mean(fold_macro_f1) * 100:5.1f}%  "
          f"(+/- {np.std(fold_macro_f1) * 100:.1f} across folds)\n")

    print(f"{'class':<16} {'n':>5} {'precision':>14} {'recall':>14} {'f1':>14}")
    for c in sorted(present_classes, key=lambda c: -counts[c]):
        name = class_names[c]
        n = counts[c]
        p_mean, p_std = float(np.mean(per_class_precision[c])), float(np.std(per_class_precision[c]))
        r_mean, r_std = float(np.mean(per_class_recall[c])), float(np.std(per_class_recall[c]))
        f_mean, f_std = float(np.mean(per_class_f1[c])), float(np.std(per_class_f1[c]))
        flag = "  <-- thin, noisy estimate" if n < 10 else ""
        print(f"{name:<16} {n:>5} "
              f"{p_mean * 100:>6.1f} +/-{p_std * 100:<5.1f} "
              f"{r_mean * 100:>6.1f} +/-{r_std * 100:<5.1f} "
              f"{f_mean * 100:>6.1f} +/-{f_std * 100:<5.1f}{flag}")

    missing = [class_names[i] for i in range(num_classes_total) if i not in present_classes]
    if missing:
        print(f"\nNo real examples at all (not evaluated): {', '.join(missing)}")

    print(f"\n{'=' * 78}")
    print("Pooled confusion matrix across all folds (rows=true, cols=predicted)")
    print(f"{'=' * 78}")
    header = "true\\pred".ljust(14) + "".join(f"{class_names[c][:8]:>9}" for c in present_classes)
    print(header)
    for true_c in present_classes:
        row = f"{class_names[true_c][:13]:<14}" + "".join(
            f"{confusion[true_c][pred_c]:>9}" for pred_c in present_classes
        )
        print(row)

    print(f"\n{'=' * 78}")
    print("Most confused class pairs (candidates for merging if this persists with more data)")
    print(f"{'=' * 78}")
    pairs = []
    for i in present_classes:
        for j in present_classes:
            if i != j and confusion[i][j] > 0:
                pairs.append((int(confusion[i][j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    if pairs:
        for count, true_name, pred_name in pairs[:8]:
            print(f"  {true_name} mistaken for {pred_name}: {count} time(s)")
    else:
        print("  None observed in held-out folds.")

    out_path = SAVED_MODELS_DIR / f"kfold_eval_{sport}.json"
    summary = {
        "sport": sport, "k": effective_k, "requested_k": k, "n_samples": len(y),
        "overall_accuracy_mean": float(np.mean(fold_accuracies)),
        "overall_accuracy_std": float(np.std(fold_accuracies)),
        "macro_f1_mean": float(np.mean(fold_macro_f1)),
        "macro_f1_std": float(np.std(fold_macro_f1)),
        "per_class": {
            class_names[c]: {
                "n": counts[c],
                "precision_mean": float(np.mean(per_class_precision[c])),
                "recall_mean": float(np.mean(per_class_recall[c])),
                "f1_mean": float(np.mean(per_class_f1[c])),
            } for c in present_classes
        },
        "classes_with_no_data": missing,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved raw numbers -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", required=True, choices=SUPPORTED_SPORTS)
    ap.add_argument("--k", type=int, default=5,
                     help="Number of folds (auto-reduced if any class has fewer examples than this)")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()
    run_kfold(args.sport, args.k, args.random_state)


if __name__ == "__main__":
    main()
