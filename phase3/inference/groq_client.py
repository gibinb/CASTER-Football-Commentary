"""
CASTER Phase 3 -- Groq free-tier API backend.

Fast (custom inference chips), free (14,400 requests/day), and a great
stand-in for your own fine-tuned model while it's still training, or for
demo recordings that need a snappier model than a laptop CPU gives locally.

Requires: pip install groq
Requires: GROQ_API_KEY environment variable (free signup at console.groq.com)

Model selection: Groq periodically decommissions models (this bit us once
already -- llama-3.1-70b-versatile, then its replacement llama-3.3-70b-versatile,
were both retired within the same project timeline). To avoid silently
falling back to the weaker template backend every time Groq retires a model,
GROQ_MODEL is checked first, then a short list of FALLBACK_MODELS is tried
in order -- only on the specific 'model_decommissioned' error, not on other
failures (auth, rate limit, etc. still raise immediately so they're visible).
Check https://console.groq.com/docs/deprecations periodically and update
GROQ_MODEL in .env when you see a deprecation notice, rather than relying
on the fallback list as a permanent fix.
"""
import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from phase3.inference.prompt_builder import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger("caster.phase3.groq_client")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Tried in order. First entries are the models actually confirmed available
# on this account (from the /v1/models listing) -- keep the older guesses
# after them in case this runs against a different Groq account later.
FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
]

# Models that do hidden chain-of-thought reasoning before their visible
# answer. At low max_tokens (or even 300) they can burn the whole budget
# on hidden reasoning and return an empty or truncated visible response --
# reasoning_effort="low" keeps more of the budget for the actual sentence.
REASONING_MODEL_MARKERS = ("gpt-oss", "qwen")

_verified_models_cache: list[str] | None = None


def _list_available_models(client) -> list[str]:
    """Query Groq's own /v1/models endpoint for the account's actually
    available models, instead of guessing hardcoded names that Groq can
    (and has, repeatedly) rename or retire without much warning. Cached
    for the process lifetime so we don't add a network call per event."""
    global _verified_models_cache
    if _verified_models_cache is not None:
        return _verified_models_cache
    try:
        response = client.models.list()
        _verified_models_cache = sorted(m.id for m in response.data)
        logger.info("Groq account has access to %d models: %s",
                    len(_verified_models_cache), ", ".join(_verified_models_cache))
    except Exception as exc:
        logger.info("could not list Groq models (%s) -- using hardcoded candidates as-is", exc)
        _verified_models_cache = []  # fall back to the hardcoded list untouched
    return _verified_models_cache


class GroqUnavailable(Exception):
    """Raised when Groq cannot be used (no key, import error, API error)."""


def _is_retryable_model_error(exc: Exception) -> bool:
    """True if trying the *next candidate model* is worth it -- either this
    specific model is unusable (decommissioned, unknown, no access) or it's
    rate-limited (different models have separate TPM/RPM buckets on Groq,
    so a 429 on one doesn't mean the account is out of quota entirely).
    Only genuinely systemic failures (bad key, network) should skip this
    and raise immediately."""
    message = str(exc).lower()
    model_error_markers = (
        "decommission",
        "model_not_found",
        "model_decommissioned",
        "does not exist",
        "do not have access",
        "invalid model",
        "unknown model",
        "empty response from groq",
        "rate_limit_exceeded",
        "rate limit reached",
    )
    if any(marker in message for marker in model_error_markers):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in (404, 429)


def generate(scene_json: dict, recent_lines: list[str] | None = None) -> str:
    if not GROQ_API_KEY:
        raise GroqUnavailable("GROQ_API_KEY not set")

    try:
        from groq import Groq
    except ImportError as exc:
        raise GroqUnavailable("groq package not installed (pip install groq)") from exc

    client = Groq(api_key=GROQ_API_KEY)
    models_to_try = [GROQ_MODEL] + [m for m in FALLBACK_MODELS if m != GROQ_MODEL]

    # Cross-check against what the account can actually see, when possible.
    # If the listing call itself fails (older SDK, transient error), just
    # fall through and try models_to_try as-is -- don't block on this.
    available = _list_available_models(client)
    if available:
        models_to_try = [m for m in models_to_try if m in available] or models_to_try

    last_error: Exception | None = None

    for model in models_to_try:
        try:
            logger.info("trying model '%s'", model)
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(scene_json, recent_lines)},
                ],
                # 0.4 (the old value) was too deterministic: with the
                # deliberately sparse, near-identical JSON two same-type
                # events produce (most fields are hidden to prevent
                # hallucination -- see _narratable_fields), low temperature
                # meant near-identical *wording* too, confirmed against a
                # real 51-event run where "What a slick dribble!" repeated
                # verbatim 5+ times. Raising this does NOT loosen fact
                # grounding -- that's enforced structurally by what's in
                # the prompt, not by sampling temperature -- it only
                # affects word/sentence-structure choice, which is exactly
                # what needed more variety. Paired with recent_lines above.
                temperature=0.9,
                max_tokens=400,  # generous headroom -- reasoning models (e.g. gpt-oss)
                                 # spend hidden tokens on chain-of-thought before the
                                 # visible answer; too low a cap starves the real output
            )
            if any(marker in model for marker in REASONING_MODEL_MARKERS):
                # Minimize hidden reasoning so more of max_tokens goes to the
                # actual sentence -- unsupported params are ignored by Groq's
                # API for models that don't recognise them, so this is safe
                # to pass unconditionally to anything matching the markers.
                kwargs["reasoning_effort"] = "low"
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason
            if not text:
                raise GroqUnavailable(
                    f"empty response from Groq (model={model}, finish_reason={finish_reason})"
                )
            if finish_reason == "length":
                logger.warning(
                    "model '%s' hit max_tokens and truncated mid-sentence -- "
                    "output kept anyway but may be cut off: %r", model, text
                )
            if model != GROQ_MODEL:
                logger.warning(
                    "GROQ_MODEL '%s' was skipped (unavailable or rate-limited) -- used "
                    "fallback '%s' instead.", GROQ_MODEL, model,
                )
            return text
        except Exception as exc:  # noqa: BLE001 -- evaluate below whether to try next model
            last_error = exc
            if _is_retryable_model_error(exc):
                continue  # try the next candidate model
            # Any other failure (auth, rate limit, network) -- don't mask it
            # by silently trying other models; surface it immediately.
            raise GroqUnavailable(str(exc)) from exc

    raise GroqUnavailable(
        f"all candidate models decommissioned or failed; last error: {last_error}"
    )