"""Regression tests for event debouncing in the live pipeline."""

import sys
from pathlib import Path

# Supports both `python phase2/tests/test_confidence_gate.py` from the
# repository root and pytest collection from any working directory.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import SPORT_FOOTBALL, MIN_SECONDS_BETWEEN_EVENTS, NO_EVENT
from phase2.confidence_gate import ConfidenceGateState, should_trigger_commentary


def test_no_event_always_rejected_regardless_of_confidence():
    # NO_EVENT means "the classifier correctly recognised ordinary,
    # non-notable play" -- it must NEVER reach Phase 3, no matter how
    # confident the prediction is. This is the fix for the real-data
    # run where a model with no NO_EVENT class was structurally forced
    # to guess a named event on ~84-96% of ordinary windows.
    decision = ConfidenceGateState().check(SPORT_FOOTBALL, NO_EVENT, 0.999, now=10.0)
    assert not decision.passed
    assert decision.reason == "no_event"

    # Same guarantee on the stateless functional variant.
    passed, reason = should_trigger_commentary(SPORT_FOOTBALL, NO_EVENT, 0.999, None, now=10.0)
    assert not passed
    assert reason == "no_event"


def test_rejects_low_confidence_goal():
    decision = ConfidenceGateState().check(SPORT_FOOTBALL, "GOAL", 0.82, now=10.0)
    assert not decision.passed
    assert decision.reason == "priority_too_low"


def test_debounces_overlapping_goal_windows():
    gate = ConfidenceGateState()
    first = gate.check(SPORT_FOOTBALL, "GOAL", 0.97, now=10.0)
    duplicate = gate.check(SPORT_FOOTBALL, "GOAL", 0.98, now=10.1)
    # Read the real cooldown from config instead of hardcoding a number
    # that silently goes stale the next time MIN_SECONDS_BETWEEN_EVENTS
    # is tuned -- this exact drift (test assumed 3.0s, config later
    # changed to 6.0s) is what broke this test originally.
    later = gate.check(SPORT_FOOTBALL, "GOAL", 0.98, now=10.0 + MIN_SECONDS_BETWEEN_EVENTS + 0.1)
    assert first.passed and first.reason == "priority_confirmed"
    assert not duplicate.passed and duplicate.reason == "too_soon"
    assert later.passed


if __name__ == "__main__":
    test_no_event_always_rejected_regardless_of_confidence()
    test_rejects_low_confidence_goal()
    test_debounces_overlapping_goal_windows()
    print("PASS: priority-event confidence, debounce, and NO_EVENT regression tests")