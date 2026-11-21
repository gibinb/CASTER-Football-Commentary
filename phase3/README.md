# CASTER — Phase 3: Commentary Engine

Turns Phase 2's confirmed, gated events into one sentence of spoken-style
football commentary, in a chosen style (dramatic / casual / statistical).
**Football only** — cricket support was fully removed, not deprecated,
across all four phases.

Phase 3 never re-guesses whether an event actually happened — its only
job is turning a verified fact into natural language. See "Phase 3
trusts Phase 2 completely" below for what that means in practice.

---

## Quickstart

```powershell
# PowerShell (per-session):
$env:GROQ_API_KEY = "your_key_here"   # free at console.groq.com
python phase3/run_phase3.py --clip your_clip

# OR (persists across sessions) -- add this line to your .env file
# instead, since config.py loads it via python-dotenv:
#   GROQ_API_KEY=your_key_here
#   GROQ_MODEL=openai/gpt-oss-120b
```

```bash
# macOS/Linux (bash/zsh):
export GROQ_API_KEY=your_key_here
python phase3/run_phase3.py --clip your_clip
```

`export` is bash-only syntax — in PowerShell it silently does nothing
(no error either), the key never gets set, and Phase 3 quietly falls
back to the zero-dependency `template` backend with no warning. It's
easy to assume Groq is running when it isn't — if every line in your
output shows `(template)` instead of `(groq)`, that's the tell.

Without `GROQ_API_KEY` set at all, Phase 3 still runs end-to-end on the
`template` backend alone — useful for testing the pipeline wiring, but
it's rule-based canned phrases, not an LLM.

`--clip your_clip` must match Phase 1/2's output filenames exactly —
Phase 3 reads `outputs/<clip>_scene_json_events.json`, which
`phase2/run_phase2.py --clip your_clip --sport football` produces. Run
Phase 1 -> 2 -> 3 in one command instead with `run_pipeline.py` at the
project root (see its own docstring for `--input`/`--skip-phase1`
usage).

### Exit gate -- what a clean run looks like

```
python phase3/run_phase3.py --clip rm1

CASTER Phase 3 -- processing: rm1

Loaded 4 confirmed events (already passed Phase 2's confidence gate).

[football] t=0:05     PASS             (groq      ) -> A crisp, contact pass!
[football] t=0:11     PASS             (groq      ) -> A sharp ball finds its target!
[football] t=0:18     PASS             (groq      ) -> Slick distribution splits the lines!
[football] t=0:24     PASS             (groq      ) -> A clean pass keeps possession ticking!

4 commentary lines generated -> outputs/rm1_commentary.json
These are exactly what Phase 4's WebSocket layer will stream next.
```

If every line reads clean, the backend used is `groq` (or `local_gguf`
once a fine-tuned model exists), and no two same-event lines repeat
near-verbatim, Phase 3 is working correctly.

---

## What Phase 3 receives from Phase 2

One JSON object per confirmed event, from
`outputs/<clip>_scene_json_events.json`:

```json
{
  "sport": "football",
  "event": "PASS",
  "confidence": 0.76,
  "pitch_zone": "midfield",
  "recent_events": ["dribble", "pass"],
  "commentary_style": "dramatic",
  "language": "en",
  "score_tracked": false,
  "event_number_in_clip": 3,
  "match_minute": 0,
  "score": "0 - 0",
  "ball_speed": 0.12,
  "boundary_proximity": false,
  "contact_flag": true,
  "clip_time": "0:24"
}
```

Every field here traces back to a real detection or a tracked match
state from Phase 1/2 -- nothing is invented. `prompt_builder.py` is the
single file that knows this exact shape; if Phase 2's output shape ever
changes, update it there, not scattered dict lookups through Phase 3.

### Phase 3 trusts Phase 2 completely -- by design, not as a gap

It has **no mechanism to detect** whether the incoming scene JSON came
from a classifier trained on real labelled footage, or one trained only
on `generate_synthetic_data.py` placeholder rows (see Phase 2's README,
"synthetic-data trap"). It will generate fluent, confident-sounding
commentary either way. Before treating Phase 3's output as meaningful,
confirm Phase 2's classifier was actually trained on real data.
Otherwise you get well-formatted commentary describing events that
never happened in your video -- the exact hallucination problem this
architecture exists to prevent, just moved one phase upstream.

---

## Backend fallback chain

`CommentaryEngine` (`inference/commentary_engine.py`) tries backends in
order, falling through to the next on **any** failure (missing model,
missing API key, network error, empty response) -- nothing here ever
propagates an exception out of `.generate()`:

1. **`local_gguf`** -- a locally fine-tuned Llama 3.2 via
   `llama-cpp-python`. Controlled entirely by the `CASTER_GGUF_MODEL_PATH`
   env var; if unset (the current default state -- QLoRA fine-tuning
   hasn't been run yet), this backend is skipped cleanly, not an error.
2. **`groq`** -- Groq's free-tier API (`inference/groq_client.py`).
   Effective primary backend right now. `GROQ_MODEL` defaults to
   `openai/gpt-oss-20b`; `openai/gpt-oss-120b` is the recommended
   override via `.env` (better quality, same free tier). If the
   configured model is deprecated or rate-limited, it automatically
   tries a short ordered fallback list (`FALLBACK_MODELS` in
   `groq_client.py`) before giving up. Reasoning-style models
   (`openai/gpt-oss-*`) are called with `reasoning_effort="low"` and
   `max_tokens=400` -- without this, hidden chain-of-thought can consume
   the entire token budget before any visible answer is written.
3. **`template`** -- zero-dependency, rule-based phrase bank
   (`inference/template_fallback.py`). Always succeeds. Guaranteed to
   be in the chain even if you construct `CommentaryEngine` with a
   custom `backend_priority` that omits it -- `generate()` can never
   return empty text.

Override the chain with `CommentaryEngine(backend_priority=[...])` if
you want to force a specific backend (e.g. `["template"]` for
deterministic, network-free testing).

---

## Grounding: what the LLM is and isn't shown

`_narratable_fields()` in `prompt_builder.py` strips fields the LLM
should never narrate before building the prompt:

- `confidence`, `commentary_style`, `language`, `score_tracked` -- always
  hidden (internal/meta, not facts about the match)
- `score`, `match_minute` -- hidden whenever `score_tracked` is `false`
  (never speak a fake scoreline)
- `boundary_proximity`, `contact_flag` -- only shown when `true` (no
  point narrating the absence of something)
- `clip_time` -- shown **only** when `commentary_style == "statistical"`.
  For `dramatic`/`casual` it's hidden even though it's a real fact,
  because a real commentator never says "at 0:11" mid-sentence -- this
  was a real bug found against a live run (Groq was narrating clip_time
  verbatim in dramatic style) and is now a regression test.

`SYSTEM_PROMPT` additionally hard-forbids inventing player/team names,
scorelines, crowd atmosphere, injuries, or narrative continuity ("after
the corner...") -- the model is structurally limited to what's actually
in the stripped JSON, which is the real hallucination-prevention
mechanism, not a polite request.

## Avoiding repetitive commentary

With the JSON deliberately this sparse, two instances of the same event
type (e.g. two `PASS` events) look nearly identical to the LLM -- at a
low sampling temperature, that produces near-identical wording. This
was a real bug found against a 51-event real run (the same phrase
repeating near-verbatim 5+ times). Fixed with two mechanisms working
together:

- **Temperature raised to 0.9** (was 0.4) -- does not loosen fact
  grounding, which is structural (see above); it only affects
  word/sentence-structure variety.
- **Per-event-type recent-line memory** -- `CommentaryEngine` keeps the
  last 4 lines generated for each event type (`RECENT_LINES_PER_EVENT`
  in `commentary_engine.py`), lives for the whole engine instance (i.e.
  one whole clip, since `run_phase3.py` creates one engine before the
  event loop). These are shown to the LLM as an explicit "you already
  used these, do not repeat their wording" list. The `template` backend
  uses the same memory to skip already-used phrases in its pool, not
  just the LLM backends.

This does not guarantee literary variety on a clip with 20+ events of
one type and only a handful of real distinguishing facts -- it
guarantees no near-back-to-back verbatim repeats, which is what the
real bug was.

---

## Word limit

`_enforce_word_limit()` hard-caps every generated line at 20 words
(`MAX_WORDS` in `commentary_engine.py`), re-terminating the sentence
cleanly if truncated -- enforced in code, not trusted to the LLM
obeying the prompt's own "at most 20 words" instruction.

---

## Data collection & fine-tuning (not yet run)

QLoRA fine-tuning of Llama 3.2 is designed but not yet executed -- Groq
remains the effective primary backend until it is.

- `data_collection/scrape_bbc_sport.py` -- scrapes BBC Sport live-text
  commentary (football only; the ESPNcricinfo cricket scraper was
  deleted entirely when cricket support was removed, not deprecated)
- `data_collection/format_training_pairs.py` -- formats scraped
  commentary into `(scene_json, commentary)` training pairs, in the
  exact shape `scene_json_builder.py` produces
- `finetune/qlora_finetune.py`, `train_style_adapters.py`,
  `merge_adapter.py`, `convert_to_gguf.sh` -- QLoRA fine-tune on a free
  Colab T4, train per-style LoRA adapters, merge and convert to GGUF
  for `local_gguf` backend use. Once a GGUF file exists, point
  `CASTER_GGUF_MODEL_PATH` at it and it becomes the primary backend
  automatically -- no code changes needed.

## FastAPI/WebSocket backend (used by Phase 4)

`backend/main.py` + `backend/websocket_stream.py` + `backend/redis_queue.py`
expose Phase 3 over a WebSocket for Phase 4's React frontend to consume
live. Needs `pip install fastapi "uvicorn[standard]" redis groq` (plain
`uvicorn` lacks WebSocket support -- use `uvicorn[standard]`) and a
running Redis instance for `redis_queue.py`. Not required for the
offline `run_phase3.py` CLI path above.

---

## Known limitations (honest, not blocking)

- Commentary specificity is capped by what Phase 1 can currently
  detect -- no team/player identity separation yet (Phase 1 future
  work), so Phase 3 can never say "his teammate" or a player's name.
- QLoRA fine-tuning not yet run -- all real commentary generated so far
  has come from Groq, not a model actually trained on this project's
  commentary style.
- On a clip with many consecutive events of one type, variety is bounded
  by the recent-line memory window (last 4) and the real facts
  available -- it won't repeat verbatim, but it will eventually echo
  similar sentence shapes on a long enough run of one event type.

---

## File map

```
phase3/
  inference/
    commentary_engine.py   -- backend-fallback orchestrator, anti-repetition memory
    groq_client.py          -- Groq backend (model fallback list, reasoning-model handling)
    template_fallback.py   -- zero-dependency rule-based backend, always succeeds
    prompt_builder.py      -- prompt construction, field-stripping, the one file that
                               knows Phase 2's scene JSON shape
  data_collection/
    scrape_bbc_sport.py
    format_training_pairs.py
  finetune/
    qlora_finetune.py, train_style_adapters.py, merge_adapter.py, convert_to_gguf.sh
  backend/
    main.py, websocket_stream.py, redis_queue.py
  run_phase3.py             -- offline CLI entry point (reads/writes outputs/*.json)
  tests/
    test_commentary_engine.py, test_prompt_and_pipeline.py
```