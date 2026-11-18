"""High-confidence, video-aware pseudo-labelling using Gemini Vision.

This is intentionally an assistive labeller, not ground truth: uncertain
responses are skipped and the saved JSON lines are marked in their source clip.
"""

import base64
import json
import os
import time
from pathlib import Path

import cv2
import requests

from config import EVENTS_BY_SPORT, NO_EVENT
from phase2.labelling.label_tool import label_path_for


class AutoLabelError(RuntimeError):
    pass


class RateLimitExhausted(RuntimeError):
    """Raised when Gemini keeps returning 429 even after retries -- signals
    the caller to stop the loop gracefully rather than crash the process."""
    pass


class ServerUnavailable(RuntimeError):
    """Raised when Gemini keeps returning a 5xx (500/502/503/504) even
    after retries -- these are the server explicitly telling us it's
    temporarily overloaded, not a problem with our request, so they get
    the same retry-then-stop-gracefully treatment as rate limiting and
    network errors rather than being treated as a permanent failure."""
    pass


class NetworkUnavailable(RuntimeError):
    """Raised when the API call itself can't be made (no connection, DNS
    failure, timeout) even after retries -- signals the caller to stop the
    loop gracefully, same as RateLimitExhausted, rather than let a raw
    requests.exceptions.ConnectionError crash the whole process with an
    unhandled traceback."""
    pass


def sample_window_frame_ids(end_frame_id: int, window_size: int = 30):
    """
    The six frame indices within one feature window that Gemini is shown
    -- shared here so any tool that wants to show a human the SAME frames
    Gemini judged (e.g. review_labels.py) samples identically, rather than
    reimplementing this math and risking a subtle mismatch.
    """
    start = max(0, end_frame_id - window_size + 1)
    return [start + round((window_size - 1) * i / 5) for i in range(6)]


def read_window_frames(cap, end_frame_id: int, window_size: int = 30, max_dim: int = 640):
    """
    Reads the six representative raw (BGR, numpy) frames for one window
    from an already-open cv2.VideoCapture, resized the same way Gemini
    receives them. Returns a list of frames (may be shorter than 6 if a
    seek fails near the start/end of the video).
    """
    frames = []
    for frame_id in sample_window_frame_ids(end_frame_id, window_size):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
        frames.append(frame)
    return frames


def _frame_parts(cap, end_frame_id: int, window_size: int = 30):
    """Return six representative JPEG frames (as Gemini API payload parts)
    from one feature window -- built on top of read_window_frames so the
    actual frame selection/resizing logic lives in exactly one place."""
    parts = []
    for frame in read_window_frames(cap, end_frame_id, window_size):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(encoded.tobytes()).decode("ascii")}})
    return parts


def _classify(parts, sport: str, api_key: str, model: str, *, max_retries: int = 4):
    # NO_EVENT is excluded from the prompt's allowed-events list on purpose:
    # SKIP already means exactly that to the model, and auto_label_windows()
    # below is what decides whether a SKIP becomes a saved NO_EVENT training
    # row (confident) or a true discard (uncertain) -- that distinction must
    # stay in code, not be left to the model to self-report.
    events = [e for e in EVENTS_BY_SPORT[sport] if e != NO_EVENT]
    prompt = (
        f"These six ordered frames cover about one second of a {sport} broadcast. "
        "Classify only an event that is visually clear. Do not guess from a scoreboard, replay graphic, "
        "or camera movement. Return JSON only: "
        '{"event":"ONE_ALLOWED_EVENT_OR_SKIP","confidence":0.0,"reason":"brief"}. '
        f"Allowed events: {', '.join(events)}. Use SKIP when the footage shows ordinary background play "
        "with nothing notable happening, OR when you are genuinely uncertain what is happening -- "
        "either way return SKIP; your confidence score is what tells us apart: be confident (high "
        "confidence) when you can clearly see it IS ordinary play, and low-confidence only when you "
        "truly cannot tell what's going on."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}, *parts]}],
               "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}

    backoff = 5.0
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=60)
        except requests.exceptions.RequestException as exc:
            # Covers no connection, DNS failure, connection reset, and
            # request timeout -- anything that means the call never got a
            # response at all, as opposed to the server responding with an
            # error status (handled below). Treated the same as a 429:
            # wait and retry a few times, since this is usually a brief
            # blip, not a permanent failure.
            if attempt == max_retries:
                raise NetworkUnavailable(
                    f"Could not reach Gemini after {max_retries} retries: {exc}"
                ) from exc
            print(f"  network error ({type(exc).__name__}) -- waiting {backoff:.0f}s before retry "
                  f"{attempt + 1}/{max_retries}...")
            time.sleep(backoff)
            backoff *= 2
            continue

        if response.ok:
            try:
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise AutoLabelError("Gemini returned an unreadable label response") from exc

        if response.status_code == 429:
            if attempt == max_retries:
                # Give the caller a chance to stop the loop cleanly and keep
                # whatever has already been saved, instead of crashing.
                raise RateLimitExhausted(
                    f"Gemini quota exceeded after {max_retries} retries: {response.text[:200]}"
                )
            wait = _retry_after_seconds(response) or backoff
            print(f"  rate limited (429) -- waiting {wait:.0f}s before retry "
                  f"{attempt + 1}/{max_retries}...")
            time.sleep(wait)
            backoff *= 2
            continue

        if response.status_code in (500, 502, 503, 504):
            # The server itself is saying this is temporary (e.g. 503's
            # own message is literally "please try again later") -- treat
            # it like 429, not like a permanent 4xx error. Without this,
            # a single transient blip from Google's side (which happens
            # more often during peak demand, per their own error text)
            # kills the entire remaining run instead of just waiting it
            # out, exactly what happened on the rm clip above.
            if attempt == max_retries:
                raise ServerUnavailable(
                    f"Gemini API {response.status_code} after {max_retries} retries: "
                    f"{response.text[:200]}"
                )
            print(f"  Gemini server error ({response.status_code}) -- waiting {backoff:.0f}s "
                  f"before retry {attempt + 1}/{max_retries}...")
            time.sleep(backoff)
            backoff *= 2
            continue

        # Any other status (400 bad request, 401/403 auth, 404 model not
        # found, etc.) is a real, non-transient problem -- retrying won't
        # help, so fail immediately rather than waste time/quota retrying.
        raise AutoLabelError(f"Gemini API {response.status_code}: {response.text[:300]}")

    # Unreachable in practice: every branch above either returns on success
    # or raises on the final retry attempt (attempt == max_retries), so the
    # loop can never exhaust via `continue` alone. This explicit raise
    # exists only so a static type checker (and anyone reading the code)
    # doesn't have to take that on faith -- without it, the function's
    # inferred return type includes None, which pyright/Pylance then flags
    # as a possible-None access everywhere the caller does result.get(...).
    raise AssertionError("_classify: retry loop exited without returning or raising -- this should be impossible")


def _retry_after_seconds(response):
    """Read Retry-After from headers, or from the RetryInfo block Gemini
    sometimes embeds in the error body. Returns None if not present."""
    header = response.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        details = response.json().get("error", {}).get("details", [])
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                delay = d.get("retryDelay", "")
                if delay.endswith("s"):
                    return float(delay[:-1])
    except (ValueError, AttributeError, json.JSONDecodeError):
        pass
    return None


def _already_labelled_keys(sport: str) -> set:
    """
    Every (source_clip, end_frame_id) pair already saved in
    data/labels/<sport>.jsonl, regardless of which tool wrote it
    (Gemini or label_tool.py). Re-running the labeller on a clip you've
    already labelled used to resample and re-send the SAME windows to
    the API every time, with nothing checking what was already saved --
    the direct cause of the 226-window duplicate problem found and
    cleaned up with dedupe_labels.py/review_labels.py. This stops that
    from recurring, and saves API quota on top of it.
    """
    path = label_path_for(sport)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    keys = set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        keys.add((row.get("source_clip", ""), row.get("end_frame_id", -1)))
    return keys


def auto_label_windows(windows, sport: str, video_path: str, append_label, *, stride: int = 30,
                       min_confidence: float = 0.85, max_windows: int = 250,
                       request_interval: float = 5.0):
    """Label spaced windows; returns (saved, skipped, stopped_early, already_covered).

    Requires GEMINI_API_KEY. request_interval enforces a minimum gap (in
    seconds) between API calls to stay under free-tier RPM limits. Google
    AI Studio's rate-limit dashboard (https://ai.dev/rate-limit) shows the
    free tier caps per model explicitly -- e.g. Gemini 3.5 Flash Lite is
    capped at 15 requests/minute. At 5.0s/request that's 12 req/min,
    leaving headroom under a 15 RPM cap. If you confirm a different RPM
    cap for your model on that dashboard, set request_interval to
    60/RPM (or slightly above it) rather than guessing.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise AutoLabelError("GEMINI_API_KEY is not set. Create a Gemini API key, set it in your environment, then retry.")
    path = Path(video_path)
    if not path.exists():
        raise AutoLabelError(f"Video not found: {path}")
    if stride < 30:
        raise AutoLabelError("--auto-stride must be at least 30 to avoid sending overlapping windows to the API")

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AutoLabelError(f"Could not open video: {path}")

    already_labelled = _already_labelled_keys(sport)
    clip_tag = f"gemini-auto:{path.name}"

    saved = skipped = attempted = already_covered = 0
    stopped_early = False
    last_saved_at = {}
    last_request_at = 0.0
    try:
        for index in range(0, len(windows), stride):
            if attempted >= max_windows:
                break
            window = windows[index]
            if (clip_tag, int(window["end_frame_id"])) in already_labelled:
                # Already labelled in a previous session (by this tool or
                # label_tool.py) -- skip without spending an API call or
                # reading frames. This is what stops re-running the
                # labeller on a clip from silently re-duplicating it.
                already_covered += 1
                continue
            parts = _frame_parts(cap, int(window["end_frame_id"]))
            if not parts:
                skipped += 1
                continue

            # Pace requests so we don't burst past the free-tier rate limit.
            elapsed = time.monotonic() - last_request_at
            if elapsed < request_interval:
                time.sleep(request_interval - elapsed)
            last_request_at = time.monotonic()

            attempted += 1
            try:
                result = _classify(parts, sport, api_key, model)
            except RateLimitExhausted as exc:
                print(f"  stopping early: {exc}")
                print(f"  {saved} label(s) already saved to disk are safe -- "
                      f"re-run later (e.g. tomorrow, or on a paid tier) to continue.")
                stopped_early = True
                break
            except NetworkUnavailable as exc:
                print(f"  stopping early: {exc}")
                print(f"  {saved} label(s) already saved to disk are safe. Check your connection "
                      f"and re-run to continue -- but note this script does NOT track which "
                      f"windows were already attempted across separate runs (only within one run), "
                      f"so a plain re-run will resample from the start of this clip and may save "
                      f"duplicate rows for timestamps already covered. To resume cleanly instead, "
                      f"re-run with --start-sec set to roughly the last 'saved t=...' timestamp "
                      f"printed above.")
                stopped_early = True
                break
            except ServerUnavailable as exc:
                print(f"  stopping early: {exc}")
                print(f"  {saved} label(s) already saved to disk are safe. This is Gemini being "
                      f"temporarily overloaded on their end, not a problem with your setup -- "
                      f"wait a few minutes and re-run. Same duplicate-row caveat as above applies: "
                      f"use --start-sec set to roughly the last 'saved t=...' timestamp printed "
                      f"above to resume cleanly instead of resampling from the start.")
                stopped_early = True
                break

            event = str(result.get("event", "SKIP")).upper()
            confidence = float(result.get("confidence", 0.0))
            timestamp = float(window.get("end_timestamp_sec", 0.0))

            if event == "SKIP":
                # A SKIP covers two different situations the prompt asks
                # Gemini to tell apart only via confidence: "I can clearly
                # see this is ordinary play" (high confidence -- a real,
                # valuable NO_EVENT training example, previously thrown
                # away entirely) vs "I genuinely don't know what's
                # happening" (low confidence -- not usable either way).
                # Only the confident case gets saved.
                if confidence < min_confidence:
                    skipped += 1
                    continue
                event = NO_EVENT
            elif event not in EVENTS_BY_SPORT[sport] or confidence < min_confidence:
                skipped += 1
                continue
            # Avoid repeatedly saving the same action from adjacent samples.
            if timestamp - last_saved_at.get(event, float("-inf")) < 3.0:
                skipped += 1
                continue
            append_label(sport, event, window, f"gemini-auto:{path.name}")
            last_saved_at[event] = timestamp
            saved += 1
            print(f"  saved t={timestamp:.1f}s {event} ({confidence:.2f})")
    finally:
        cap.release()
    return saved, skipped, stopped_early, already_covered