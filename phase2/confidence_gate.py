"""
CASTER Phase 2 -- confidence gate.

The classifier (XGBoost or BiLSTM) always returns a prediction, even
when barely more confident than a coin flip. Without a filter, every
window would trigger a commentary attempt, including windows where the
model is genuinely unsure -- producing contradictory or nonsensical
narration. This module is that filter.

Rules, applied in order:
    0. (callers) ball_visibility_ok() -- windows where the ball was not
       actually seen are dropped before classification is even trusted.
    0b. NO_EVENT is always discarded, at any confidence -- it means the
        classifier correctly recognised ordinary, non-notable play, not
        that commentary should be generated. See config.py's note on
        why this class exists.
    1. Confidence must clear the sport-specific threshold
       (config.CONFIDENCE_THRESHOLD_BY_SPORT) -- otherwise discard
       silently.
    2. At least MIN_SECONDS_BETWEEN_EVENTS must have passed since the
       last commentary line -- otherwise discard (prevents the TTS
       layer in Phase 4 from overlapping itself).
    3. Priority events (config.PRIORITY_EVENTS_BY_SPORT -- wickets,
       boundaries, goals) use a separately high confidence threshold.
       They do not bypass the cooldown: overlapping classification windows
       describe the same physical action and must produce one event.

Priority predictions need a stronger confirmation threshold because an
incorrect GOAL or WICKET is more disruptive than a missed low-confidence
prediction. The value is configured centrally in ``config.py``.

This module is intentionally stateless except for a small
ConfidenceGateState tracker you instantiate once per video and reuse
across the whole pipeline run -- it's the thing that remembers "when
was the last commentary line."
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    CONFIDENCE_THRESHOLD_BY_SPORT, MIN_SECONDS_BETWEEN_EVENTS,
    PRIORITY_EVENTS_BY_SPORT, PRIORITY_MIN_CONFIDENCE, NO_EVENT,
)


# A 30-frame window in which the ball was barely seen is not evidence of
# anything. When the ball is missing the feature extractor substitutes a
# fixed (0.5, 0.5) position; if the ball flickers in and out, that produces a
# fake "ball teleporting" signal that XGBoost happily classifies (usually as
# SHOT_ON_TARGET or CORNER_KICK) with high confidence. Require the ball to have
# actually been observed in a reasonable share of the window's frames.
MIN_BALL_VISIBLE_RATIO = 0.5


def ball_visibility_ok(ball_seen_flags, min_ratio: float = MIN_BALL_VISIBLE_RATIO) -> bool:
    """ball_seen_flags: iterable of bools, one per frame of the classifier window."""
    flags = list(ball_seen_flags)
    if not flags:
        return False
    return (sum(1 for f in flags if f) / len(flags)) >= min_ratio


@dataclass
class GateDecision:
    passed: bool
    reason: str               # "low_confidence" | "too_soon" | "priority_confirmed" | "priority_too_low" | "passed"
    event: str
    confidence: float


@dataclass
class ConfidenceGateState:
    """
    Tracks timing across calls. One instance per video being processed
    -- create it once in run_phase2.py, then call .check() on every
    new EventPrediction as it's produced.

    Uses wall-clock time.time() by default, which is correct for live
    streaming use (Phase 4's WebSocket pipeline). For OFFLINE batch
    processing of a pre-recorded clip (Phase 1's run_phase1.py mode),
    pass the video's own timestamp_sec instead via the `now` argument
    so gating is based on in-video time, not real processing wall time.
    """
    last_event_time: Optional[float] = field(default=None)

    def check(self, sport: str, event: str, confidence: float,
              now: Optional[float] = None) -> GateDecision:
        # NO_EVENT is a training-time-only class (see config.py) meaning
        # "the classifier correctly recognised this as ordinary,
        # non-notable play." It must NEVER reach Phase 3, regardless of
        # how confident the classifier is -- confidence here measures
        # "how sure am I nothing happened," not grounds for commentary.
        if event == NO_EVENT:
            return GateDecision(False, "no_event", event, confidence)

        now = now if now is not None else time.time()
        threshold = CONFIDENCE_THRESHOLD_BY_SPORT[sport]
        priority_events = PRIORITY_EVENTS_BY_SPORT[sport]

        is_priority = event in priority_events

        if is_priority:
            if confidence < PRIORITY_MIN_CONFIDENCE:
                # Even a priority event needs to clear a basic sanity
                # floor -- this is not the normal sport threshold, just
                # a guard against near-random noise flooding the feed.
                return GateDecision(False, "priority_too_low", event, confidence)
            if self.last_event_time is not None and (now - self.last_event_time) < MIN_SECONDS_BETWEEN_EVENTS:
                return GateDecision(False, "too_soon", event, confidence)
            self.last_event_time = now
            return GateDecision(True, "priority_confirmed", event, confidence)

        if confidence < threshold:
            return GateDecision(False, "low_confidence", event, confidence)

        if self.last_event_time is not None and (now - self.last_event_time) < MIN_SECONDS_BETWEEN_EVENTS:
            return GateDecision(False, "too_soon", event, confidence)

        self.last_event_time = now
        return GateDecision(True, "passed", event, confidence)

    def reset(self):
        self.last_event_time = None


def should_trigger_commentary(
    sport: str,
    event: str,
    confidence: float,
    last_event_time: Optional[float],
    now: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Stateless functional variant matching the exact pseudocode from
    the deep-dive docs, for callers that prefer to manage their own
    timing state rather than using ConfidenceGateState. Returns
    (should_trigger, reason).
    """
    if event == NO_EVENT:
        return False, "no_event"

    now = now if now is not None else time.time()
    threshold = CONFIDENCE_THRESHOLD_BY_SPORT[sport]
    priority_events = PRIORITY_EVENTS_BY_SPORT[sport]

    if event in priority_events:
        if confidence < PRIORITY_MIN_CONFIDENCE:
            return False, "priority_too_low"
        if last_event_time is not None and (now - last_event_time) < MIN_SECONDS_BETWEEN_EVENTS:
            return False, "too_soon"
        return True, "priority_confirmed"

    if confidence < threshold:
        return False, "low_confidence"

    if last_event_time is not None and (now - last_event_time) < MIN_SECONDS_BETWEEN_EVENTS:
        return False, "too_soon"

    return True, "passed"
