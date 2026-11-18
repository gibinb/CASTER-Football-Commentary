# CASTER — Phase 2: Event Intelligence

**Status: code complete (12/12 tests pass); data volume is the live bottleneck for 3 classes**

Takes Phase 1's 12-feature windows and classifies which football event is
occurring, with a confidence gate so only high-certainty predictions reach
Phase 3.

## Event classes (11, football only)

`GOAL`, `SHOT_ON_TARGET`, `TACKLE`, `FOUL`, `CORNER_KICK`, `FREE_KICK`,
`HEADER`, `GOALKEEPER_SAVE`, `OFFSIDE`, `PENALTY`, and `NO_EVENT` (appended
as the 11th class, not inserted — keeps label-id stability). `NO_EVENT`
exists specifically so ordinary play gets saved as real negative data
instead of being silently discarded by the labelling tools; it is always
rejected by the confidence gate and never reaches commentary.

## ⚠️ Read before using `generate_synthetic_data.py`

```bash
# STEP A — wiring smoke test ONLY. The resulting model is a placeholder.
python labelling/generate_synthetic_data.py --n-per-class 60
python event_classifier/xgboost_clf.py
python tests/test_confidence_gate.py
python tests/test_scene_json_builder.py
python tests/test_xgboost_clf.py

# STEP B — remove synthetic rows before any real training. Do not skip.
python labelling/remove_synthetic_data.py --apply
# (dry-run by default; writes *_real_only.jsonl / *_synthetic_only.jsonl
#  for inspection unless --apply is passed; backs up the original to .bak first)

# STEP C — label real data from real Phase 1 output
python ../phase1/run_phase1.py --input ../data/raw/myclip.mp4
python labelling/label_tool.py --windows ../outputs/myclip_feature_windows.json
# number keys tag an event, 'n' tags NO_EVENT, 's' skips, 'q' quits.
# Labels append to data/labels/football.jsonl immediately on every tag.

# STEP D — train on REAL labelled data only
python event_classifier/xgboost_clf.py

# STEP E — run Phase 2 on real Phase 1 output
python run_phase2.py --clip myclip --classifier xgboost
```

> **Do not skip to Phase 3 using a model trained only on synthetic data.**
> `generate_synthetic_data.py` exists purely to verify wiring end-to-end
> before you've labelled anything real. Predictions from a synthetic-only
> model don't correspond to anything in your actual footage — feeding them
> to Phase 3 produces fluent commentary describing events that never
> happened, defeating the entire hallucination-prevention architecture
> this project is built around. `generate_synthetic_data.py` **appends**,
> it does not overwrite — re-running it repeatedly dilutes real data under
> a growing pile of fake rows if Step B isn't run first.

## Confidence gate

```python
def should_trigger_commentary(prediction, min_confidence=0.50,
                               min_seconds_since_last=3.5):
    event, confidence = prediction
    if confidence < min_confidence:
        return False
    if time_since_last_event() < min_seconds_since_last:
        return False
    return True
```
`GOAL` overrides the cooldown at `PRIORITY_MIN_CONFIDENCE=0.85` — never
missed regardless of timer. `NO_EVENT` is always rejected, whatever its
confidence.

## Current real data state (last checked)

Football: 1086 real labelled rows across 12 source clips. k-fold ≈ 42.9%
(±3.5) accuracy, 15.2% (±6.0) macro F1. `NO_EVENT`, `DRIBBLE`-equivalent and
most priority classes have real signal; `CORNER_KICK` is still a dead class
(0% recall); `TACKLE`/`HEADER` need more labelling. This is a data-volume
problem, not a code problem — see root README's status section.

## Tooling

`evaluate_kfold.py` (the standing way to judge any change — replaced noisy
single-split accuracy), `dedupe_labels.py`, `review_labels.py`,
`find_candidates.py`, `batch_autolabel.py`, `summarize_labels.py`.