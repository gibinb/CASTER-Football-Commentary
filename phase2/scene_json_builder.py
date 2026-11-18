"""
CASTER Phase 2 -- scene state JSON builder. FOOTBALL ONLY.

The single most important design decision in the entire CASTER project.
Instead of sending raw video, bounding boxes, or feature vectors to the
language model in Phase 3, this module packages everything the LLM
needs into one small, structured JSON object. The LLM only ever sees
this JSON -- never the video, never raw detections -- which is exactly
why CASTER does not hallucinate the way naive LLM-sports-commentary
demos do. It is structurally incapable of inventing a player name or
score it was not explicitly given.

This module wraps an EventPrediction (schemas.py) -- which already
carries the full SceneState from Phase 1 plus the event/confidence
from Phase 2's classifier -- into the flat JSON shape Phase 3 expects.

Match-state fields that Phase 1's SceneState has no way of knowing
(running score, match minute) are NOT invented here either -- they
come from a MatchState object that the orchestrator (run_phase2.py)
updates as events are confirmed. This keeps the "never invent a fact"
guarantee intact all the way through Phase 2.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SPORT_FOOTBALL
from schemas import EventPrediction


@dataclass
class MatchState:
    """
    Running match-state memory the scene JSON builder reads from.
    Updated by the pipeline orchestrator whenever a score-changing
    event is confirmed -- NOT inferred or guessed inside this module.
    """
    sport: str = SPORT_FOOTBALL
    home_score: int = 0
    away_score: int = 0
    match_minute: int = 0
    commentary_style: str = "dramatic"
    language: str = "en"
    recent_events: List[str] = field(default_factory=list)
    # How many times each event has been confirmed so far in THIS clip.
    # A real, tracked fact (unlike score/minute), so the LLM may narrate it.
    event_counts: dict = field(default_factory=dict)
    # Score / minute are NOT read from the video (no scoreboard OCR),
    # so they stay at their placeholders unless something explicitly sets
    # this flag. Phase 3 will not narrate them while it is False.
    score_tracked: bool = False

    def push_event(self, event: str, max_history: int = 5):
        self.recent_events.append(event.lower())
        if len(self.recent_events) > max_history:
            self.recent_events = self.recent_events[-max_history:]

    def football_score_str(self) -> str:
        return f"{self.home_score} - {self.away_score}"


def format_clip_time(seconds) -> str:
    """Seconds into the uploaded clip -> 'm:ss'. This is CLIP time, not match time."""
    seconds = max(0, int(seconds or 0))
    return f"{seconds // 60}:{seconds % 60:02d}"


def build_scene_json(prediction: EventPrediction, match_state: MatchState,
                     clip_time_sec: Optional[float] = None) -> dict:
    """
    Produces the exact dict shape Phase 3's prompt template /
    fine-tuning data format expects. Every field is sourced either
    from `prediction` (already verified by Phase 1 + Phase 2's gate)
    or from `match_state` (maintained explicitly by the orchestrator,
    never guessed here).
    """
    scene = prediction.scene
    sport = prediction.sport

    if sport != SPORT_FOOTBALL:
        raise ValueError(f"unsupported sport: {sport!r} -- CASTER is football-only")

    base = {
        "sport": sport,
        "event": prediction.event,
        "confidence": round(prediction.confidence, 4),
        "pitch_zone": scene.pitch_zone,
        "recent_events": list(match_state.recent_events),
        "commentary_style": match_state.commentary_style,
        "language": match_state.language,
        "score_tracked": match_state.score_tracked,
        # 1 = first time this event was confirmed in the clip, 2 = second, ...
        "event_number_in_clip": match_state.event_counts.get(prediction.event, 0) + 1,
        "match_minute": match_state.match_minute,
        "score": match_state.football_score_str(),
        "ball_speed": scene.ball_speed,
        "boundary_proximity": scene.boundary_proximity,
        "contact_flag": scene.contact_flag,
    }
    if clip_time_sec is not None:
        base["clip_time"] = format_clip_time(clip_time_sec)

    return base


def apply_event_to_match_state(match_state: MatchState, sport: str, event: str):
    """
    Minimal, honest score-update logic -- deliberately simple. This
    updates ONLY the fields that can be confidently derived from the
    event name itself (e.g. a GOAL increments the scoring team). It
    does NOT guess things it cannot know, like which specific team
    scored in a contact-flag-only detection -- that requires
    possession/team tracking you may add later. Extend this function
    as your match-state tracking matures; do not have Phase 3 paper
    over gaps here with invented numbers.
    """
    if sport != SPORT_FOOTBALL:
        raise ValueError(f"unsupported sport: {sport!r} -- CASTER is football-only")

    # NOTE: without team-possession tracking, we cannot honestly say
    # WHICH team scored on a bare "GOAL" event -- incrementing a side
    # here would be an invented fact, exactly what this module exists
    # to prevent. Leave score incrementing as explicit future work
    # (requires possession/team-side tracking) rather than guessing.

    match_state.event_counts[event] = match_state.event_counts.get(event, 0) + 1
    match_state.push_event(event)
