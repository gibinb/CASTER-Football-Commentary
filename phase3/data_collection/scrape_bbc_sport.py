"""
CASTER Phase 3 -- scrapes minute-by-minute live text commentary from BBC
Sport football match report pages for fine-tuning training data.

Maps raw event types to config.FOOTBALL_EVENTS exactly. BBC's page layout
is NOT the same as ESPNcricinfo's, so this is a separate scraper, not a
copy-paste -- it embeds match state as a JSON blob in a <script> tag; the
exact tag id can change over time, so this tries known ids and falls back
to a regex scan.

Usage:
    python phase3/data_collection/scrape_bbc_sport.py \
        --urls URL1 URL2 --out data/commentary/raw_football.jsonl
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SPORT_FOOTBALL, FOOTBALL_EVENTS, ROOT_DIR

USER_AGENT = "Mozilla/5.0 (compatible; CASTER-research-scraper/1.0)"
RAW_CACHE_DIR = ROOT_DIR / "data" / "raw" / "bbc_sport"
CANDIDATE_SCRIPT_IDS = ["__NEXT_DATA__", "tvip-script-app-store"]


def classify_football_event(event: dict) -> str:
    """Maps BBC's raw event type to config.FOOTBALL_EVENTS exactly."""
    etype = (event.get("type") or "").lower()
    mapping = {
        "goal": "GOAL",
        "shot-on-target": "SHOT_ON_TARGET",
        "shot": "SHOT_ON_TARGET",
        "tackle": "TACKLE",
        "foul": "FOUL",
        "corner": "CORNER_KICK",
        "free-kick": "FREE_KICK",
        "header": "HEADER",
        "offside": "OFFSIDE",
    }
    for key, value in mapping.items():
        if key in etype:
            return value
    return "PASS"  # low-signal filler, filtered out later by quality scoring


assert set(["GOAL", "SHOT_ON_TARGET", "TACKLE", "FOUL", "CORNER_KICK",
            "FREE_KICK", "HEADER", "OFFSIDE", "PASS"]).issubset(set(FOOTBALL_EVENTS)), \
    "scraper event labels have drifted from config.FOOTBALL_EVENTS -- fix classify_football_event()"


def extract_embedded_json(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    for script_id in CANDIDATE_SCRIPT_IDS:
        tag = soup.find("script", {"id": script_id})
        if tag and tag.string:
            try:
                return json.loads(tag.string)
            except json.JSONDecodeError:
                continue
    match = re.search(r"window\.__INITIAL_DATA__\s*=\s*(\{.*?\});", html, re.S)
    if match:
        return json.loads(match.group(1))
    raise ValueError("could not locate embedded match JSON on page")


def walk_for_events(node, out, depth=0):
    if depth > 8 or len(out) > 5000:
        return
    if isinstance(node, dict):
        text = node.get("text") or node.get("commentaryText")
        minute = node.get("minute") or node.get("time")
        if text and minute is not None:
            out.append({"minute": minute, "text": text, "raw": node})
        for v in node.values():
            walk_for_events(v, out, depth + 1)
    elif isinstance(node, list):
        for item in node:
            walk_for_events(item, out, depth + 1)


def extract_entries(match_json: dict, url: str) -> list:
    raw_events = []
    walk_for_events(match_json, raw_events)

    entries = []
    recent = []
    for ev in sorted(raw_events, key=lambda e: str(e["minute"])):
        event = classify_football_event(ev["raw"])
        entries.append({
            "match_url": url,
            "sport": SPORT_FOOTBALL,
            "match_minute": ev["minute"],
            "event": event,
            "score": ev["raw"].get("score", ""),
            "recent_events": [e.lower() for e in recent[-5:]],
            "text": ev["text"].strip(),
        })
        recent.append(event)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--out", default=str(ROOT_DIR / "data" / "commentary" / "raw_football.jsonl"))
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "a", encoding="utf-8") as out_f:
        for i, url in enumerate(args.urls):
            try:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
                resp.raise_for_status()
                with open(RAW_CACHE_DIR / f"match_{i}.html", "w") as cache_f:
                    cache_f.write(resp.text)

                match_json = extract_embedded_json(resp.text)
                entries = extract_entries(match_json, url)
                for e in entries:
                    out_f.write(json.dumps(e) + "\n")
                print(f"[ok] {url}: {len(entries)} entries")
            except Exception as exc:
                print(f"[skip] {url} failed: {exc}")
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
