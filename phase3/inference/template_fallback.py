"""
CASTER Phase 3 -- rule-based commentary generator. FOOTBALL ONLY.
Zero dependencies, zero cost, always works.

Each event/style has MULTIPLE phrases. The selection is seeded by
(event + timestamp) so the same event at different times produces
different commentary, eliminating repeated identical lines.
"""
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SPORT_FOOTBALL

FOOTBALL_TEMPLATES = {
    "PASS": {
        "dramatic": [
            "A slick pass finds their teammate in space!",
            "Inch-perfect delivery, splitting the defence wide open!",
            "Quick interchange, the ball moved with real purpose there!",
            "Clinical passing, building pressure down the flank!",
            "Great vision -- the ball slides through to the runner!",
            "They are moving the ball quickly now, defence under pressure!",
            "Crisp, accurate pass keeps the move alive!",
        ],
        "casual": [
            "Nice pass there, good vision.",
            "Ball played nicely into space.",
            "Smart pass, keeping it moving.",
            "Good delivery, found his man.",
        ],
        "statistical": [
            "Pass completed. Score remains {score}.",
            "Pass logged. Current score: {score}.",
        ],
    },
    "DRIBBLE": {
        "dramatic": [
            "Brilliant footwork, dancing past the defender!",
            "He beats one, two -- what skill on the ball!",
            "Electrifying run, leaving defenders in his wake!",
            "Superb close control as he weaves through the challenge!",
            "Explosive pace and trickery, the crowd is on its feet!",
            "Magical feet, the defender simply cannot live with him!",
            "Unstoppable burst of pace leaves the opponent stranded!",
        ],
        "casual": [
            "Nice bit of skill there.",
            "Good dribble, beat his man.",
            "Decent run with the ball.",
            "Neat footwork from the player.",
        ],
        "statistical": [
            "Dribble recorded. Score remains {score}.",
            "Dribble attempt: successful. Score: {score}.",
        ],
    },
    "TACKLE": {
        "dramatic": [
            "A crunching tackle wins back possession!",
            "Perfectly timed challenge -- ball won cleanly!",
            "Absolutely fantastic defending, what a tackle!",
            "He slides in and comes away with the ball!",
            "Brave defending, throwing himself into that challenge!",
            "Clinical tackle stops the attack dead in its tracks!",
        ],
        "casual": [
            "Good tackle there, ball won back.",
            "Nice challenge, gets the ball.",
            "Solid defending, tackled well.",
            "Great tackle, dispossesses the attacker.",
        ],
        "statistical": [
            "Tackle made. Score remains {score}.",
            "Tackle logged. Score: {score}.",
        ],
    },
    "SHOT_ON_TARGET": {
        "dramatic": [
            "A powerful effort forces a save from the keeper!",
            "Thunderbolt of a shot -- keeper tips it over!",
            "What a strike! The goalkeeper had to be at his best!",
            "Rasping drive straight at goal -- brilliant save!",
            "He has a real chance -- saved! Great goalkeeping!",
            "Brilliant strike, but the keeper is equal to it!",
        ],
        "casual": [
            "Good effort there, saved though.",
            "Shot on target, keeper holds.",
            "Decent attempt, well saved.",
            "Good save from the keeper there.",
        ],
        "statistical": [
            "Shot on target. Score remains {score}.",
            "On-target attempt logged. Score: {score}.",
        ],
    },
    "GOAL": {
        "dramatic": [
            "GOOOAL! An incredible finish and the crowd goes wild!",
            "IT IS IN! Absolute worldie, the net is bulging!",
            "GOAL! Clinical finish, the keeper had no chance whatsoever!",
            "IN OFF THE POST! The deflection beats the keeper -- GOAL!",
            "TOP BINS! What a finish, that is absolutely stunning!",
            "GOAL! Unstoppable strike, pure power and precision!",
            "He has done it! Composed finish, calmly slotted away -- GOAL!",
            "GOAL! From nowhere -- a moment of pure individual brilliance!",
        ],
        "casual": [
            "That is a goal! Great finish.",
            "Goal! Well taken, keeper beaten.",
            "In the back of the net! Lovely finish.",
            "He has scored! Quality goal that.",
            "Goal! Clean strike, no chance for the keeper.",
        ],
        "statistical": [
            "Goal scored. Score now {score}.",
            "Goal registered. Updated score: {score}.",
        ],
    },
    "FOUL": {
        "dramatic": [
            "The referee whistle blows -- a foul is given!",
            "Clumsy challenge -- the referee has no hesitation!",
            "Late tackle, and the referee reaches for his pocket!",
            "He has brought him down -- clear foul, whistle blown!",
            "Reckless challenge stops the attack -- foul given!",
            "The referee steps in -- that was a nasty challenge!",
        ],
        "casual": [
            "That is a foul there.",
            "Foul given, free kick coming.",
            "Referee blows up, foul called.",
            "Foul committed, play stopped.",
        ],
        "statistical": [
            "Foul committed. Score remains {score}.",
            "Foul logged. Score: {score}.",
        ],
    },
    "CORNER_KICK": {
        "dramatic": [
            "The ball goes out -- a corner kick to be taken!",
            "Forced out for a corner! Danger from the set piece!",
            "Corner awarded! This could be a real chance here!",
            "Ball over the line -- corner kick, great opportunity!",
            "Scrambled behind! Corner kick, the set-piece routine begins!",
        ],
        "casual": [
            "Corner given, let us see what comes of it.",
            "Ball out, corner kick.",
            "They have won a corner.",
        ],
        "statistical": [
            "Corner kick awarded. Score remains {score}.",
            "Corner logged. Score: {score}.",
        ],
    },
    "FREE_KICK": {
        "dramatic": [
            "Free kick awarded in a dangerous position!",
            "Excellent free kick position -- great opportunity to score!",
            "Dead ball situation in a threatening area -- free kick!",
            "The specialist steps up -- free kick, real danger here!",
            "Free kick just outside the box -- what a chance this is!",
            "Foul! And it is a free kick in a very promising position!",
        ],
        "casual": [
            "Free kick coming up.",
            "Free kick, good position.",
            "Free kick awarded here.",
            "Set piece for the attacking side.",
        ],
        "statistical": [
            "Free kick awarded. Score remains {score}.",
            "Free kick logged. Score: {score}.",
        ],
    },
    "HEADER": {
        "dramatic": [
            "Up he goes -- a powerful header!",
            "Rises above everyone -- bullet header!",
            "Towering header, real physical dominance!",
            "He attacks the ball magnificently -- huge header!",
            "Flick header at the near post -- great movement!",
            "Climbs highest in the box -- powerful header attempt!",
        ],
        "casual": [
            "Good header there.",
            "Nice header, won the aerial ball.",
            "Won the header, good jump.",
        ],
        "statistical": [
            "Header attempt recorded. Score remains {score}.",
            "Header logged. Score: {score}.",
        ],
    },
    "OFFSIDE": {
        "dramatic": [
            "Flag goes up -- offside called!",
            "The assistant referee raises the flag -- offside!",
            "Offside trap sprung to perfection -- brilliant defending!",
            "Exquisitely timed offside trap catches the striker!",
            "He is caught offside -- a marginal call but the flag is up!",
        ],
        "casual": [
            "That is offside.",
            "Flag up for offside.",
            "Caught offside there.",
            "Offside call, attack breaks down.",
        ],
        "statistical": [
            "Offside called. Score remains {score}.",
            "Offside logged. Score: {score}.",
        ],
    },
}

GENERIC_TEMPLATE = {
    "dramatic": [
        "Something is happening on the field -- stay tuned!",
        "Action in the middle of the park!",
        "Moment of excitement as play continues!",
    ],
    "casual": [
        "Bit of action there.",
        "Something happening on the field.",
        "Play continues.",
    ],
    "statistical": [
        "Event recorded: {event}. Score: {score}.",
        "{event} logged. Score: {score}.",
    ],
}


def generate_template_commentary(scene_json: dict, recent_lines: list[str] | None = None) -> str:
    """
    recent_lines: previously generated commentary for this same event type
    earlier in the clip (from CommentaryEngine's per-event memory). When
    given, this picks from whichever phrases in the table have NOT
    already been used, so the template backend -- not just the LLM
    backends -- avoids obvious looping once a clip has enough events of
    one type to exhaust the pool (e.g. the 51-event 'ss' run had 20+
    DRIBBLE events against a 7-phrase dramatic pool).
    """
    sport = scene_json.get("sport")
    if sport != SPORT_FOOTBALL:
        raise ValueError(f"unsupported sport: {sport!r} -- CASTER is football-only")
    event = scene_json["event"]
    # score/match_minute are placeholders unless something explicitly tracks
    # them (see scene_json_builder.py) -- never speak a fake scoreline.
    score = scene_json.get("score", "unknown") if scene_json.get("score_tracked") else "the score"
    style = scene_json.get("commentary_style", "dramatic")
    timestamp = scene_json.get("clip_time", scene_json.get("timestamp_sec", 0) or 0)
    if style not in ("dramatic", "casual", "statistical"):
        style = "dramatic"
    event_templates = FOOTBALL_TEMPLATES.get(event, {}).get(style)
    if not event_templates:
        event_templates = GENERIC_TEMPLATE[style]

    candidates = event_templates
    if recent_lines:
        # Compare against the UNFORMATTED templates' rendered form so a
        # template already used (even with a different {score}) is still
        # recognised as "used" -- recent_lines holds final rendered text.
        rendered_recent = {
            t.format(score=score, event=event) for t in event_templates
        } & set(recent_lines)
        unused = [t for t in event_templates
                  if t.format(score=score, event=event) not in rendered_recent]
        if unused:
            candidates = unused
        # if every phrase in the pool has been used, fall through to the
        # full pool -- repeating is better than crashing, and the seed
        # below still spreads repeats out rather than always picking #1

    # Seed with the raw string, not int(timestamp) -- real Phase 2 output's
    # clip_time is "m:ss" (see scene_json_builder.py's format_clip_time()),
    # not a number, so int() on it crashed here on every real pipeline run
    # (only missed by unit tests because none of them included clip_time).
    # random.Random() happily accepts a str/float/int seed, so just pass
    # whichever timestamp value we got through as-is.
    rng = random.Random(f"{event}:{timestamp}")
    template = rng.choice(candidates)
    return template.format(score=score, event=event)
