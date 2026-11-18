"""
CASTER Phase 2 -- end-to-end orchestrator.

Reads Phase 1's saved output (outputs/<clip>_feature_windows.json and
outputs/<clip>_scene_states.json), runs each feature window through a
trained classifier, applies the confidence gate, and builds scene
state JSON for every event that survives -- exactly the JSON shape
Phase 3's language model will read next.

This mirrors run_phase1.py's structure deliberately, so the two
scripts feel like one continuous pipeline rather than separate tools:

    python phase1/run_phase1.py --input data/raw/clip.mp4
    python phase2/run_phase2.py --clip clip --sport football --classifier xgboost

This is also the Phase 2 "exit gate" check from the deep-dive doc: if
this runs cleanly on a football clip's saved Phase 1 output, printing
a plausible timestamped event log with at least one scene JSON built,
Phase 2 is done.

Usage:
    python phase2/run_phase2.py --clip myclip --sport football --classifier xgboost
    python phase2/run_phase2.py --clip myclip --sport football --classifier bilstm
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR, SUPPORTED_SPORTS
from schemas import SceneState, Detection, EventPrediction

from phase2.confidence_gate import ConfidenceGateState, ball_visibility_ok
from phase2.scene_json_builder import MatchState, build_scene_json, apply_event_to_match_state


def load_feature_windows(clip_name: str):
    path = OUTPUTS_DIR / f"{clip_name}_feature_windows.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No feature windows found at {path}. Run Phase 1 first:\n"
            f"  python phase1/run_phase1.py --input data/raw/{clip_name}.mp4"
        )
    with open(path) as f:
        return json.load(f)


def load_scene_states(clip_name: str):
    path = OUTPUTS_DIR / f"{clip_name}_scene_states.json"
    if not path.exists():
        raise FileNotFoundError(f"No scene states found at {path}. Run Phase 1 first.")
    with open(path) as f:
        raw = json.load(f)

    scenes_by_frame = {}
    for s in raw:
        detections = [Detection(**d) for d in s.get("detections", [])]
        s = dict(s)
        s["detections"] = detections
        scenes_by_frame[s["frame_id"]] = SceneState(**s)
    return scenes_by_frame


def get_classifier(classifier_name: str, sport: str):
    if classifier_name == "xgboost":
        from phase2.event_classifier.xgboost_clf import load_classifier, predict_window
        model = load_classifier(sport)
        return lambda window: predict_window(model, sport, window)
    elif classifier_name == "bilstm":
        from phase2.event_classifier.bilstm_clf import load_classifier, predict_window
        model = load_classifier(sport)
        return lambda window: predict_window(model, sport, window)
    else:
        raise ValueError(f"unknown classifier: {classifier_name}. Use 'xgboost' or 'bilstm'.")


def run(clip_name: str, sport: str, classifier_name: str, commentary_style: str = "dramatic"):
    print(f"\nCASTER Phase 2 -- processing: {clip_name} ({sport}, {classifier_name})\n")

    windows = load_feature_windows(clip_name)
    scenes_by_frame = load_scene_states(clip_name)
    print(f"Loaded {len(windows)} feature windows and {len(scenes_by_frame)} scene states.\n")

    if not windows:
        print("No feature windows to process (clip may be shorter than the 30-frame window).")
        return []

    predict = get_classifier(classifier_name, sport)
    gate = ConfidenceGateState()
    match_state = MatchState(sport=sport, commentary_style=commentary_style)

    accepted_events = []
    discarded_count = 0
    no_ball_count = 0
    ordered_frame_ids = sorted(scenes_by_frame)
    position_of = {fid: i for i, fid in enumerate(ordered_frame_ids)}

    for w in windows:
        event, confidence = predict(w["vectors"])
        end_frame_id = w["end_frame_id"]
        end_timestamp = w["end_timestamp_sec"]

        scene = scenes_by_frame.get(end_frame_id)
        if scene is None:
            continue  # no matching scene state for this window's end frame

        # Skip windows where the ball was mostly not observed (see confidence_gate.py)
        end_pos = position_of[end_frame_id]
        window_scenes = [scenes_by_frame[f] for f in ordered_frame_ids[max(0, end_pos - len(w["vectors"]) + 1): end_pos + 1]]
        if not ball_visibility_ok(s.ball_detected for s in window_scenes):
            no_ball_count += 1
            discarded_count += 1
            continue

        decision = gate.check(sport, event, confidence, now=end_timestamp)

        if not decision.passed:
            discarded_count += 1
            continue

        prediction = EventPrediction(
            sport=sport, event=event, confidence=confidence,
            scene=scene, recent_events=list(match_state.recent_events),
        )

        match_state.match_minute = int(end_timestamp // 60)

        apply_event_to_match_state(match_state, sport, event)
        # clip_time_sec was never passed before -- every scene JSON was
        # missing the "clip_time" field entirely, and match_minute alone
        # is useless on any clip under a minute (rounds every event to 0,
        # which is what Phase 3's CLI was printing as "t=0" for every
        # single event on rm1). clip_time ("m:ss" into THIS clip) is the
        # only timestamp that's actually meaningful pre-fine-tune.
        scene_json = build_scene_json(prediction, match_state, clip_time_sec=end_timestamp)

        accepted_events.append(scene_json)

        override_tag = "  (priority confirmed)" if decision.reason == "priority_confirmed" else ""
        print(f"EVENT: {event:<16} at {end_timestamp:6.1f}s  confidence={confidence:.2f}{override_tag}")

    print(f"\n{len(accepted_events)} events passed the confidence gate, "
          f"{discarded_count} discarded ({no_ball_count} because the ball was not visible).\n")

    out_path = OUTPUTS_DIR / f"{clip_name}_scene_json_events.json"
    with open(out_path, "w") as f:
        json.dump(accepted_events, f, indent=2, default=str)
    print(f"Saved {len(accepted_events)} scene JSON events -> {out_path}")
    print("These are exactly what Phase 3's language model will read next.\n")

    return accepted_events


def main():
    parser = argparse.ArgumentParser(description="Run CASTER Phase 2 on Phase 1's saved output")
    parser.add_argument("--clip", required=True,
                         help="Clip name (without extension) -- matches Phase 1's output filenames")
    parser.add_argument("--sport", required=True, choices=SUPPORTED_SPORTS)
    parser.add_argument("--classifier", default="xgboost", choices=["xgboost", "bilstm"])
    parser.add_argument("--style", default="dramatic",
                         choices=["dramatic", "casual", "statistical"])
    args = parser.parse_args()

    run(args.clip, args.sport, args.classifier, commentary_style=args.style)


if __name__ == "__main__":
    main()