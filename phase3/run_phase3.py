"""
CASTER Phase 3 -- end-to-end offline runner. FOOTBALL ONLY.

Reads Phase 2's saved output (outputs/<clip>_scene_json_events.json) --
events that ALREADY passed the confidence gate -- and generates
commentary text for each one. This mirrors run_phase1.py and
run_phase2.py's structure deliberately, so all three feel like one
continuous pipeline:

    python phase1/run_phase1.py --input data/raw/clip.mp4
    python phase2/run_phase2.py --clip clip --sport football --classifier xgboost
    python phase3/run_phase3.py --clip clip

This is also the Phase 3 "exit gate" check: if this runs cleanly on a
football clip's saved Phase 2 output, printing a plausible commentary
line per event with no crashes, Phase 3 is done.

Usage:
    python phase3/run_phase3.py --clip myclip
    python phase3/run_phase3.py --clip myclip --style casual   # override every event's style
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR

from phase3.inference.commentary_engine import CommentaryEngine
from phase3.inference.prompt_builder import time_field


def load_scene_json_events(clip_name: str) -> list:
    path = OUTPUTS_DIR / f"{clip_name}_scene_json_events.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No scene JSON events found at {path}. Run Phase 2 first:\n"
            f"  python phase2/run_phase2.py --clip {clip_name} --sport football "
            f"--classifier xgboost"
        )
    with open(path) as f:
        return json.load(f)


def run(clip_name: str, style_override: str = None):
    print(f"\nCASTER Phase 3 -- processing: {clip_name}\n")

    events = load_scene_json_events(clip_name)
    print(f"Loaded {len(events)} confirmed events (already passed Phase 2's confidence gate).\n")

    if not events:
        print("No events to narrate -- Phase 2 discarded everything, or the clip was too short.")
        return []

    engine = CommentaryEngine()
    results = []

    for scene_json in events:
        if style_override:
            scene_json = dict(scene_json)
            scene_json["commentary_style"] = style_override

        result = engine.generate(scene_json)
        # time_field() (match_minute) is what goes INTO the LLM prompt --
        # keep it there, it matches the training-data time format
        # ("minute: 67"). For the CLI/log display, prefer clip_time
        # ("m:ss" elapsed in this clip) when present: match_minute rounds
        # down to whole minutes, so every event on a clip under 60s
        # (e.g. rm1, 33s) printed "t=0" for everything -- clip_time is
        # the only timestamp actually informative on short clips.
        display_time = scene_json.get("clip_time") or time_field(scene_json)
        results.append({
            "sport": result["sport"],
            "event": result["event"],
            "confidence": result["confidence"],
            "time": display_time,
            "match_minute": time_field(scene_json),
            "score": scene_json.get("score", ""),
            "style": result["style"],
            "commentary": result["text"],
            "backend_used": result["backend_used"],
        })
        print(f"[{result['sport']}] t={display_time:<8} {result['event']:<16} "
              f"({result['backend_used']:<10}) -> {result['text']}")

    out_path = OUTPUTS_DIR / f"{clip_name}_commentary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{len(results)} commentary lines generated -> {out_path}")
    print("These are exactly what Phase 4's WebSocket layer will stream next.\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CASTER Phase 3 on Phase 2's saved output")
    parser.add_argument("--clip", required=True,
                         help="Clip name (without extension) -- matches Phase 1/2's output filenames")
    parser.add_argument("--style", default=None,
                         choices=["dramatic", "casual", "statistical"],
                         help="Override commentary_style for every event "
                              "(else uses each event's own field from Phase 2)")
    args = parser.parse_args()

    run(args.clip, style_override=args.style)


if __name__ == "__main__":
    main()
