"""
CASTER Phase 2 -- synthetic training data generator.

Real labelling (phase2/labelling/label_tool.py) requires real video and
roughly 3-4 hours of manual work per the deep-dive doc. This script
generates plausible, *clearly-structured* synthetic feature windows so
you can:

    1. Run the entire Phase 2 pipeline (train -> gate -> JSON builder)
       end-to-end on day one, before you've labelled a single real frame.
    2. Sanity-check that your training code, confidence gate, and JSON
       builder are wired correctly, using data where you already KNOW
       what the "correct" classifier behaviour should look like.
    3. Get a feel for realistic class imbalance (lots of dot balls /
       passes, few wickets / goals) before you go label real clips.

This data is NOT a substitute for real training data -- a classifier
trained only on this will learn the synthetic generation rules, not
real football. Replace it with phase2/labelling output before your
final dissertation submission.

Usage:
    python phase2/labelling/generate_synthetic_data.py --sport football --n-per-class 60
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    SPORT_FOOTBALL, SUPPORTED_SPORTS,
    EVENTS_BY_SPORT, FEATURE_WINDOW_SIZE, ZONES_BY_SPORT,
)
from schemas import LabelledWindow
from phase2.labelling.label_tool import label_path_for


def _rand_vector(ball_speed, zone_id, contact_flag, boundary_flag,
                  official_signal=0.0, players_nearby=1, n_zones=5):
    """One synthetic 12-dim feature vector with controllable signal fields."""
    return [
        random.uniform(0.1, 0.9),                  # ball_x
        random.uniform(0.1, 0.9),                  # ball_y
        random.uniform(-0.05, 0.05),               # ball_dx
        random.uniform(-0.05, 0.05),               # ball_dy
        ball_speed + random.uniform(-0.01, 0.01),  # ball_speed
        random.uniform(0.02, 0.3),                 # nearest_player_dist
        float(players_nearby),                     # players_nearby
        float(min(zone_id, n_zones - 1)),           # pitch_zone_id
        official_signal,                            # official_signal
        random.uniform(0.0, 1.0),                  # ball_height_proxy
        contact_flag,                                # contact_flag
        boundary_flag,                               # boundary_proximity
    ]


def _make_window(vector_fn, **kwargs):
    return [vector_fn(**kwargs) for _ in range(FEATURE_WINDOW_SIZE)]


# ---------------------------------------------------------------------------
# Per-event synthetic generators -- each encodes the "key detection signal"
# from the Phase 2 deep-dive doc's event tables, so a classifier trained on
# this data at least learns the RIGHT correlations, even with fake numbers.
# ---------------------------------------------------------------------------

def football_window_for_event(event: str, n_zones: int):
    if event == "PASS":
        return _make_window(_rand_vector, ball_speed=0.2, zone_id=random.randint(1, 2),
                             contact_flag=0.0, boundary_flag=0.0, players_nearby=1, n_zones=n_zones)
    if event == "DRIBBLE":
        return _make_window(_rand_vector, ball_speed=0.1, zone_id=random.randint(1, 2),
                             contact_flag=1.0, boundary_flag=0.0, players_nearby=1, n_zones=n_zones)
    if event == "TACKLE":
        return _make_window(_rand_vector, ball_speed=0.05, zone_id=random.randint(1, 2),
                             contact_flag=1.0, boundary_flag=0.0, players_nearby=2, n_zones=n_zones)
    if event == "SHOT_ON_TARGET":
        return _make_window(_rand_vector, ball_speed=0.4, zone_id=0,
                             contact_flag=1.0, boundary_flag=0.0, players_nearby=2, n_zones=n_zones)
    if event == "GOAL":
        return _make_window(_rand_vector, ball_speed=0.45, zone_id=n_zones - 1,
                             contact_flag=1.0, boundary_flag=1.0, players_nearby=4, n_zones=n_zones)
    if event == "FOUL":
        return _make_window(_rand_vector, ball_speed=0.05, zone_id=random.randint(0, n_zones - 1),
                             contact_flag=1.0, boundary_flag=0.0, official_signal=1.0, players_nearby=2, n_zones=n_zones)
    if event == "CORNER_KICK":
        return _make_window(_rand_vector, ball_speed=0.02, zone_id=n_zones - 1,
                             contact_flag=0.0, boundary_flag=1.0, players_nearby=3, n_zones=n_zones)
    if event == "FREE_KICK":
        return _make_window(_rand_vector, ball_speed=0.02, zone_id=random.randint(0, n_zones - 1),
                             contact_flag=0.0, boundary_flag=0.0, official_signal=1.0, players_nearby=3, n_zones=n_zones)
    if event == "HEADER":
        return _make_window(_rand_vector, ball_speed=0.3, zone_id=random.randint(0, 1),
                             contact_flag=1.0, boundary_flag=0.0, players_nearby=2, n_zones=n_zones)
    if event == "OFFSIDE":
        return _make_window(_rand_vector, ball_speed=0.25, zone_id=0,
                             contact_flag=0.0, boundary_flag=0.0, official_signal=1.0, players_nearby=1, n_zones=n_zones)
    if event == "NO_EVENT":
        # Deliberately the LEAST distinctive window: moderate, unremarkable
        # ball movement, no official signal, no boundary, random zone, light
        # player presence -- ordinary open play with nothing commentary-
        # worthy happening. This is intentionally close to DRIBBLE/PASS in
        # feature space (real ordinary play genuinely looks similar to a
        # real dribble or pass at a glance), which is exactly why real
        # labelled NO_EVENT examples matter more here than almost anywhere
        # else -- synthetic data alone cannot teach this boundary well.
        return _make_window(_rand_vector, ball_speed=random.uniform(0.08, 0.22),
                             zone_id=random.randint(0, n_zones - 1),
                             contact_flag=random.choice([0.0, 1.0]), boundary_flag=0.0,
                             players_nearby=random.choice([0, 1]), n_zones=n_zones)
    raise ValueError(f"unknown football event: {event}")


GENERATOR_BY_SPORT = {
    SPORT_FOOTBALL: football_window_for_event,
}


def generate(sport: str, n_per_class: int, source_tag: str = "synthetic", seed: int = 42):
    """
    seed defaults to a fixed value (not None) deliberately: this data is
    unseeded `random.uniform()` noise, so without a fixed seed, re-running
    this generator -- including implicitly, e.g. test_xgboost_clf.py's
    _ensure_synthetic_data() regenerating from scratch on a clean checkout
    -- produces a genuinely different dataset every time, which makes
    classifier behaviour (and test pass/fail) non-reproducible between
    runs. Pass seed=None explicitly if you want fresh random variety on
    purpose (e.g. to sanity-check robustness across different synthetic
    draws, not for anything you need to reproduce or debug later).
    """
    if seed is not None:
        random.seed(seed)

    events = EVENTS_BY_SPORT[sport]
    generator = GENERATOR_BY_SPORT[sport]
    n_zones = len(ZONES_BY_SPORT[sport])

    out_path = label_path_for(sport)
    written = 0
    with open(out_path, "a") as f:
        for event in events:
            for _ in range(n_per_class):
                vectors = generator(event, n_zones)
                lw = LabelledWindow(
                    sport=sport, event=event, vectors=vectors,
                    source_clip=source_tag, end_frame_id=-1,
                )
                f.write(lw.to_json_line() + "\n")
                written += 1
    print(f"Wrote {written} synthetic windows ({n_per_class} per class x {len(events)} classes) "
          f"for {sport} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic labelled windows for CASTER Phase 2")
    parser.add_argument("--sport", required=True, choices=list(SUPPORTED_SPORTS) + ["both"])
    parser.add_argument("--n-per-class", type=int, default=60,
                         help="How many synthetic windows to generate per event class")
    args = parser.parse_args()

    sports = list(SUPPORTED_SPORTS) if args.sport == "both" else [args.sport]
    for sport in sports:
        generate(sport, args.n_per_class)


if __name__ == "__main__":
    main()
