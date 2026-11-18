# CASTER — Phase 3: Commentary Engine

**Status: confirmed working end-to-end.**

Takes Phase 2's scene state JSON and turns it into a fluent spoken sentence,
in your chosen style, without ever looking at the video itself.

## Why fine-tune (planned) instead of just prompting

A base model prompted loosely will happily invent a scorer's name or a
crowd reaction never in the JSON. Fine-tuning on real commentary paired
with structured scene descriptions teaches the model to treat the JSON as
the only source of truth. **Not yet trained** — Groq is the effective
primary backend today; QLoRA fine-tuning of Llama 3.2 3B remains designed
but not executed.

## Backend fallback chain

1. Local fine-tuned GGUF (not yet trained — chain falls through today)
2. **Groq free tier** — `openai/gpt-oss-120b` (effective primary; set via
   `.env`'s `GROQ_MODEL`, code default is `gpt-oss-20b`)
3. Template/phrase-bank fallback — always available, always grounded, just
   less fluent

```powershell
# PowerShell
$env:GROQ_API_KEY = "your_key_here"   # free at console.groq.com
python run_phase3.py --clip your_clip
```
```bash
# macOS/Linux
export GROQ_API_KEY=your_key_here
python run_phase3.py --clip your_clip
```
Or add `GROQ_API_KEY=...` to `.env` (loaded via python-dotenv in
`config.py`).

> **Reasoning-model note:** Groq's `openai/gpt-oss-*` models are reasoning
> models — without `reasoning_effort="low"` and a capped `max_tokens`
> (≈400), hidden chain-of-thought silently consumes the token budget.
> Already handled in `groq_client.py`; relevant if you swap models.

## Grounding

`prompt_builder.py`'s `_narratable_fields()` strips internal-only fields
(`confidence`, `pitch_zone`, `ball_speed`, `recent_events`) before the LLM
ever sees the JSON — it can only narrate what's left. The system prompt
carries hard anti-hallucination rules ("use ONLY facts in the JSON, do not
invent player/team names, score, or crowd reaction").

## Anti-repetition

Same-event instances naturally produce near-identical stripped JSON, which
combined with a low temperature caused near-verbatim repeated lines across
consecutive events of the same type. Fixed via: a `recent_lines` param
threaded through every backend (Groq, local GGUF, template fallback)
rendering an explicit "don't repeat this wording" instruction, a rolling
per-event-type memory (`maxlen=4`, lives for the whole clip), and
temperature raised 0.4→0.9 (doesn't loosen fact-grounding — that's
structural, from what's *in* the JSON, not from temperature). Verified
against a real 51-event sequence with no phrase repeats within 5 events of
its own last use.

## Phase 3 trusts Phase 2 completely — by design, not a gap

Phase 3 never re-guesses whether an event actually happened. It has no way
to detect whether incoming scene JSON came from a classifier trained on
real labelled footage or on `generate_synthetic_data.py` placeholder rows
(see Phase 2 README). Confirm Phase 2 was trained on real data before
trusting Phase 3's output as meaningful.

## Exit gate