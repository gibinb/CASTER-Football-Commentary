"""
CASTER Phase 3 -- commentary engine. FOOTBALL ONLY.

This is the orchestrator every other Phase 3 module imports
(run_phase3.py, backend/main.py, backend/websocket_stream.py, both test
files) but which did not actually exist in the delivered codebase --
every one of those imports was crashing with
ModuleNotFoundError: No module named 'phase3.inference.commentary_engine'.
This file is the fix.

Responsibilities:
  - Try backends in order (default: local_gguf -> groq -> template),
    falling through to the next one on ANY failure (missing model,
    missing API key, network error, empty response, etc.) -- never lets
    a single backend's failure propagate out of generate().
  - 'template' (template_fallback.py) has zero external dependencies and
    always succeeds, so it is guaranteed to be in the chain -- generate()
    can never return empty text.
  - Hard-enforces the SYSTEM_PROMPT's word-limit instruction itself,
    rather than trusting an LLM backend to obey it -- Groq/local models
    occasionally run long.
  - Returns a single flat result dict with the exact keys run_phase3.py,
    websocket_stream.py and the test suite all expect: sport, event,
    confidence, style, text, backend_used.
"""
import logging
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SPORT_FOOTBALL

from phase3.inference import groq_client
from phase3.inference.template_fallback import generate_template_commentary

logger = logging.getLogger("caster.phase3.commentary_engine")

DEFAULT_BACKEND_PRIORITY = ["local_gguf", "groq", "template"]

# Matches SYSTEM_PROMPT's "at most 20 words" instruction (prompt_builder.py).
# Kept slightly separate here (not imported) so this cap can be tuned
# independently of prompt wording without touching prompt_builder.py.
MAX_WORDS = 20

# How many of the most recent lines PER EVENT TYPE to show the LLM as
# "don't repeat these" examples (see prompt_builder.build_user_prompt).
# Found necessary against a real 51-event run on 'ss': with the same
# event repeating every few events and the facts shown to the LLM
# deliberately sparse (see _narratable_fields), the model had no way to
# know prior lines existed at all. 4 is enough to break obvious looping
# without bloating every prompt.
RECENT_LINES_PER_EVENT = 4


class LocalGGUFUnavailable(Exception):
    """Raised when local llama.cpp/GGUF inference cannot be used -- no
    fine-tuned model configured yet, llama-cpp-python not installed, or
    the model failed to load. Mirrors groq_client.GroqUnavailable."""


_local_llm = None
_local_llm_load_failed = False


def _get_local_llm():
    """Lazily loads a local GGUF model (the QLoRA-fine-tuned Llama 3.2,
    once finetune/convert_to_gguf.sh has produced one) via
    llama-cpp-python, once per process.

    Controlled entirely by CASTER_GGUF_MODEL_PATH. If it's unset, this
    backend is simply skipped -- not an error -- which matches the
    project's actual current state (fine-tuning not done yet; Groq is
    the effective primary path, per /projects/.../overview.md).
    """
    global _local_llm, _local_llm_load_failed
    if _local_llm is not None:
        return _local_llm
    if _local_llm_load_failed:
        raise LocalGGUFUnavailable("local GGUF model previously failed to load this run")

    model_path = os.environ.get("CASTER_GGUF_MODEL_PATH", "")
    if not model_path:
        raise LocalGGUFUnavailable("CASTER_GGUF_MODEL_PATH not set -- no fine-tuned model yet")
    if not Path(model_path).exists():
        raise LocalGGUFUnavailable(f"CASTER_GGUF_MODEL_PATH points to a missing file: {model_path}")

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        _local_llm_load_failed = True
        raise LocalGGUFUnavailable(
            "llama-cpp-python not installed (pip install llama-cpp-python)"
        ) from exc

    try:
        _local_llm = Llama(model_path=model_path, n_ctx=512, verbose=False)
    except Exception as exc:  # noqa: BLE001 -- any load failure disables this backend
        _local_llm_load_failed = True
        raise LocalGGUFUnavailable(f"failed to load GGUF model: {exc}") from exc
    return _local_llm


def _local_gguf_generate(scene_json: dict, recent_lines: list[str] | None = None) -> str:
    from phase3.inference.prompt_builder import SYSTEM_PROMPT, build_user_prompt

    llm = _get_local_llm()
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(scene_json, recent_lines)}\n"
    # temperature raised to match groq_client.py's reasoning (see its comment) --
    # same repetition problem applies to any LLM backend, not just Groq.
    output = llm(prompt, max_tokens=80, temperature=0.9, stop=["\n"])
    text = output["choices"][0]["text"].strip()
    if not text:
        raise LocalGGUFUnavailable("empty response from local GGUF model")
    return text


def _enforce_word_limit(text: str, max_words: int = MAX_WORDS) -> str:
    """Hard safety net -- trims to max_words and re-terminates the
    sentence cleanly, regardless of which backend produced the text."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    truncated = re.sub(r"[,;:]$", "", truncated)
    if not truncated.endswith((".", "!", "?")):
        truncated += "."
    return truncated


_BACKEND_FUNCS = {
    "local_gguf": _local_gguf_generate,
    "groq": groq_client.generate,
    "template": generate_template_commentary,
}


class CommentaryEngine:
    """
    Usage:
        engine = CommentaryEngine()  # default chain: local_gguf -> groq -> template
        result = engine.generate(scene_json)
        # result == {"sport": ..., "event": ..., "confidence": ...,
        #            "style": ..., "text": ..., "backend_used": ...}
    """

    def __init__(self, backend_priority: list[str] | None = None):
        self.backend_priority = list(backend_priority) if backend_priority else list(
            DEFAULT_BACKEND_PRIORITY
        )
        if "template" not in self.backend_priority:
            # template is the zero-dependency, always-succeeds backend.
            # Never let it be configured out entirely -- that would let
            # generate() raise instead of degrading gracefully.
            self.backend_priority.append("template")

        # Per-event-type memory of recently generated lines, so repeated
        # events of the same type (the common case -- a clip usually has
        # many PASS/DRIBBLE events and few of everything else) don't all
        # get near-identical commentary. Lives for the lifetime of this
        # engine instance, i.e. one whole clip in run_phase3.py's usage.
        self._recent_by_event: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=RECENT_LINES_PER_EVENT)
        )

    def generate(self, scene_json: dict) -> dict:
        sport = scene_json.get("sport")
        if sport != SPORT_FOOTBALL:
            raise ValueError(f"unsupported sport: {sport!r} -- CASTER is football-only")

        event = scene_json.get("event", "UNKNOWN")
        recent_lines = list(self._recent_by_event[event])

        text = None
        backend_used = None

        for backend_name in self.backend_priority:
            fn = _BACKEND_FUNCS.get(backend_name)
            if fn is None:
                logger.warning("unknown backend %r in backend_priority, skipping", backend_name)
                continue
            try:
                candidate = fn(scene_json, recent_lines)
            except Exception as exc:  # noqa: BLE001 -- fall through to the next backend
                logger.info("backend %r unavailable (%s), trying next", backend_name, exc)
                continue
            if candidate:
                text = candidate
                backend_used = backend_name
                break

        if not text:
            # Should be unreachable (template always succeeds and is
            # guaranteed present in backend_priority), but never return
            # empty commentary under any circumstances.
            text = generate_template_commentary(scene_json, recent_lines)
            backend_used = "template"

        text = _enforce_word_limit(text)
        self._recent_by_event[event].append(text)

        return {
            "sport": sport,
            "event": event,
            "confidence": scene_json.get("confidence"),
            "style": scene_json.get("commentary_style", "dramatic"),
            "text": text,
            "backend_used": backend_used,
        }
