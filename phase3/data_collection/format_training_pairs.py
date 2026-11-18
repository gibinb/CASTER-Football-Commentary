"""
CASTER Phase 3 -- filters raw scraped commentary and builds instruction-
following training pairs, in the exact scene JSON shape produced by
phase2/scene_json_builder.py::build_scene_json. FOOTBALL ONLY. This is
the "meta-conditioning" step: every training example pairs a structured
scene description with the real commentary line a human wrote in
response to it, so the fine-tuned model learns to treat the JSON as the
sole source of truth (see CASTER_Phase3_DeepDive.docx, Section 3.2).

Usage:
    python phase3/data_collection/format_training_pairs.py \
        --football data/commentary/raw_football.jsonl \
        --out data/commentary/training_pairs.jsonl \
        --top-k 50000
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SPORT_FOOTBALL, ROOT_DIR

MIN_WORDS = 5
MAX_WORDS = 60
MAX_EXCLAMATIONS = 4


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------
def basic_quality_ok(text: str) -> bool:
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    if "http" in text or "www." in text:
        return False
    if text.count("!") > MAX_EXCLAMATIONS:
        return False
    if re.search(r"\[deleted\]|\[removed\]", text):
        return False
    return True


def quality_score(text: str) -> float:
    """Cheap heuristic (no extra ML dependency needed): prefers medium-length,
    punctuation-normal text. Swap in a real perplexity scorer here later if
    you want tighter fluency filtering."""
    words = text.split()
    length_score = 1.0 - abs(len(words) - 18) / 60.0
    punctuation_penalty = 0.1 * text.count("!!!")
    return max(0.0, length_score - punctuation_penalty)


# ---------------------------------------------------------------------------
# Style inference (crude heuristic, used only to route examples to the
# right style-adapter training subset)
# ---------------------------------------------------------------------------
def infer_style(text: str) -> str:
    if re.search(r"\d+%|\brun rate\b|\bxg\b|\baverage\b", text, re.I):
        return "statistical"
    if "!" in text or text.isupper():
        return "dramatic"
    return "casual"


# ---------------------------------------------------------------------------
# Scene JSON construction -- matches build_scene_json's real output shape
# ---------------------------------------------------------------------------
def build_scene_json_from_scraped(entry: dict) -> dict:
    sport = entry["sport"]
    if sport != SPORT_FOOTBALL:
        raise ValueError(f"unsupported sport: {sport!r} -- CASTER is football-only")
    scene = {
        "sport": sport,
        "event": entry.get("event", "UNKNOWN"),
        "confidence": 0.95,  # scraped ground-truth commentary treated as fully confident
        "pitch_zone": entry.get("pitch_zone", "unknown"),
        "recent_events": entry.get("recent_events", []),
        "commentary_style": infer_style(entry["text"]),
        "language": "en",
        "match_minute": entry.get("match_minute"),
        "score": entry.get("score", ""),
        "ball_speed": entry.get("ball_speed", 0.0),
        "boundary_proximity": entry.get("event") == "GOAL",
        "contact_flag": True,
    }
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--football", required=True, help="raw football jsonl from scrape_bbc_sport.py")
    parser.add_argument("--out", default=str(ROOT_DIR / "data" / "commentary" / "training_pairs.jsonl"))
    parser.add_argument("--top-k", type=int, default=50000)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    raw_entries = []
    for path in [args.football]:
        if not path:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    raw_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Quality filter + score
    scored = []
    for entry in raw_entries:
        text = entry.get("text", "")
        if not basic_quality_ok(text):
            continue
        entry["_quality"] = quality_score(text)
        scored.append(entry)

    scored.sort(key=lambda e: e["_quality"], reverse=True)
    kept = scored[: args.top_k]

    # Build training pairs
    pairs = []
    for entry in kept:
        scene = build_scene_json_from_scraped(entry)
        pairs.append({
            "instruction": "Generate sports commentary for this event: " + json.dumps(scene, default=str),
            "input_json": scene,
            "output": entry["text"],
        })

    random.Random(args.shuffle_seed).shuffle(pairs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, default=str) + "\n")

    print(f"filtered {len(raw_entries)} -> kept {len(kept)} -> built {len(pairs)} training pairs")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
