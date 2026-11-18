"""
CASTER Phase 3 -- prompt construction. FOOTBALL ONLY.

Builds the system + user prompt sent to whichever LLM backend is active.
This is the ONE place that knows the exact shape of the scene JSON dict
produced by phase2/scene_json_builder.py::build_scene_json. If that
shape ever changes, update this file, not scattered dict lookups
throughout Phase 3.

Real shape produced by Phase 2 (verified against scene_json_builder.py,
football-only build_scene_json()):

    sport, event, confidence, pitch_zone, recent_events,
    commentary_style, language, score_tracked, event_number_in_clip,
    match_minute, score ("home - away"), ball_speed,
    boundary_proximity, contact_flag, clip_time
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SPORT_FOOTBALL

SYSTEM_PROMPT = """You are CASTER, a live sports commentator working from a computer-vision feed.
You receive a JSON object of CONFIRMED facts about ONE moment in a video clip.
Write exactly ONE sentence of commentary about that moment.

Hard rules:
- Use ONLY facts that are in the JSON. If a fact is not there, you do not know it.
- Do NOT invent or imply any of: player or team names, who has the ball, the
  scoreline, the match minute, kickoff / half-time / "opening minutes", the crowd
  or stadium atmosphere, replays, injuries, what happened just before, or how the
  play ended. Never write things like "the crowd erupts", "rockets toward goal",
  "after the corner" or "right from kickoff".
- Say what the event name says and nothing more. SHOT_ON_TARGET means a shot
  that was on target -- not a save, not a goal. CORNER_KICK means a corner was
  taken -- not where it landed.
- "clip_time" is seconds into an uploaded clip, NOT match time.
- Never mention confidence, detection or classifier scores, internal IDs, JSON,
  or the pipeline.
- Match the requested style, but keep it to at most 20 words, one sentence,
  no quotes, no preamble.
- A real commentator never says the same thing twice in a row. If you are
  shown recent lines you already used for this same event type, you MUST
  write something that sounds meaningfully different from every one of
  them -- different opening word, different sentence structure, different
  vocabulary -- not just a synonym swapped into the same template.""" 

STYLE_INSTRUCTIONS = {
    "dramatic": "Punchy and energetic, exclamation marks welcome -- but the "
                "excitement must come from the event itself, not invented detail.",
    "casual": "Sound like a fan chatting with friends. Conversational and light; "
              "a little slang is fine. Still no invented detail.",
    "statistical": "Lead with the numbers you actually have: clip_time, and "
                   "event_number_in_clip (e.g. 'second shot on target of the clip'). "
                   "Analytical and precise, minimal emotion. Do not quote any "
                   "number that is not in the JSON.",
}


def time_field(scene_json: dict) -> str:
    """Returns the match-time field -- 'match_minute' -- matching
    scene_json_builder.py's actual football-only output exactly."""
    return str(scene_json.get("match_minute", ""))


# Fields that exist in Phase 2's scene JSON for pipeline/internal purposes
# but must never be shown to the LLM -- stripped before the prompt is built
# (a system-prompt instruction alone is not reliable enough to stop a model
# from occasionally quoting a number it can see).
#
#   confidence, commentary_style, language, score_tracked -- meta / routing
#   pitch_zone   -- a crude image-space guess ("midfield" for anything near the
#                   frame centre), NOT a real pitch location. Leaving it in is
#                   how the model ended up saying "a corner kick in midfield".
#   ball_speed   -- normalised image units per frame; meaningless to say aloud.
#   recent_events-- earlier events; the model used them to invent a storyline
#                   ("after the corner..."). event_number_in_clip covers the
#                   legitimate use (repetition awareness).
INTERNAL_ONLY_FIELDS = {
    "confidence", "commentary_style", "language", "score_tracked",
    "pitch_zone", "ball_speed", "recent_events",
}

# Placeholders that nothing reads from the video. Hidden while
# score_tracked is explicitly False; if the key is absent (older JSON,
# hand-written test data) the fields are treated as real.
UNTRACKED_MATCH_FIELDS = {"score", "match_minute"}


def _narratable_fields(scene_json: dict, style: str) -> dict:
    """scene_json with internal/meta/untracked fields removed, safe to show the LLM.

    clip_time is only meaningful for the 'statistical' style (per
    STYLE_INSTRUCTIONS, which explicitly tells that style to lead with
    it) -- for 'dramatic'/'casual' it is hidden here, not just discouraged
    by prompt wording. Confirmed necessary against a real Groq run: once
    clip_time started actually being populated by Phase 2
    (run_phase2.py's clip_time_sec fix), dramatic-style output started
    narrating it verbatim ("A crisp pass at 0:11!"), which no real
    broadcaster says and isn't an invented fact so the system prompt's
    "only use given facts" rule didn't catch it.
    """
    hide = set(INTERNAL_ONLY_FIELDS)
    if scene_json.get("score_tracked") is False:
        hide |= UNTRACKED_MATCH_FIELDS
    if style != "statistical":
        hide.add("clip_time")
    out = {}
    for k, v in scene_json.items():
        if k in hide or v is None:
            continue
        if k in ("boundary_proximity", "contact_flag") and not v:
            continue   # only mention these when they are actually true
        out[k] = v
    return out


def build_user_prompt(scene_json: dict, recent_lines: list[str] | None = None) -> str:
    """
    recent_lines: previously generated commentary for this SAME event type
    earlier in this clip (most recent last), passed in by CommentaryEngine
    so the model has something concrete to differ from. This is what
    actually fixes repetition -- at low sampling temperature and with a
    near-identical, deliberately sparse JSON (most fields are stripped by
    _narratable_fields to prevent hallucination), the single biggest
    remaining driver of near-identical output across events of the same
    type is that the model has no signal telling it other lines already
    exist. An instruction alone ("vary your phrasing") without concrete
    examples to avoid is much weaker than showing the actual recent lines.
    """
    style = scene_json.get("commentary_style", "dramatic")
    if style not in STYLE_INSTRUCTIONS:
        style = "dramatic"

    parts = [
        f"Event data from computer vision pipeline:\n"
        f"{json.dumps(_narratable_fields(scene_json, style), indent=2, default=str)}\n",
        f"Style: {STYLE_INSTRUCTIONS[style]}\n",
    ]

    if recent_lines:
        formatted = "\n".join(f'- "{line}"' for line in recent_lines)
        parts.append(
            f"You already used these lines for this same event type earlier in "
            f"this clip -- do NOT repeat any of their wording, structure, or "
            f"stock phrases:\n{formatted}\n"
        )

    parts.append("Write the one-sentence commentary:")
    return "\n".join(parts)