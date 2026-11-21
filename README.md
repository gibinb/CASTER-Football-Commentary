# CASTER

**Computer Vision Assisted Sports Text And Event Reporter**

MSc Artificial Intelligence — Final Year Dissertation Project (YIP 9.0 — Future Technologies & AI)

CASTER watches a football match video and generates live, spoken play-by-play
commentary — in a style you choose (dramatic / casual / statistical) — by
combining computer vision, an event classifier, and an LLM commentary engine.

**Football-only.** Cricket support was fully removed (not deprecated) across
all four phases during development — no sport auto-detection, no cricket
classes, no cricket code paths remain anywhere in this repo.

All four phases connect through one shared contract: `schemas.py`'s
`SceneState`. Each phase only ever talks to its neighbour through this JSON
shape — Phase 1 doesn't know Phase 2 exists, Phase 2 doesn't know Phase 1's
internals. This is the same "structured grounding" principle that prevents
Phase 3's LLM from hallucinating, applied one layer earlier between modules.

```
CASTER/
  config.py            <- shared paths, thresholds, constants (all phases import this)
  schemas.py            <- the SceneState JSON contract every phase agrees on
  phase1/               <- VISION FOUNDATION — COMPLETE
    detection/           <- FootballDetector: YOLOv8m, 4 classes, auto GPU
    tracking/             <- ByteTrack for players (ball is NOT ID-tracked)
    features/             <- 12-feature extractor, two-hit ball validation
    scene_state/           <- SceneState JSON builder
    weights/                <- football.pt (YOLOv8m, mAP50 0.863, not committed)
    tests/
    run_phase1.py
  phase2/               <- EVENT INTELLIGENCE — code complete, data volume is the live bottleneck
    event_classifier/     <- xgboost_clf.py (native Booster API), evaluate_kfold.py
    labelling/              <- label_tool.py, run_gemini_autolabel.py, batch_autolabel.py,
                               find_candidates.py, dedupe_labels.py, review_labels.py,
                               summarize_labels.py, remove_synthetic_data.py
    confidence_gate.py       <- hallucination boundary: NO_EVENT reject, threshold, cooldown, priority override
    scene_json_builder.py     <- builds the exact JSON Phase 3 reads
    tests/
    run_phase2.py
  phase3/               <- COMMENTARY ENGINE — confirmed working
    data_collection/      <- scrape_bbc_sport.py, format_training_pairs.py (football only;
                              cricket scraper deleted entirely, not deprecated)
    finetune/               <- QLoRA pipeline (designed, not yet trained)
    inference/               <- commentary_engine.py (backend fallback + anti-repetition memory),
                                 prompt_builder.py (grounding/field-stripping), groq_client.py,
                                 template_fallback.py
    backend/                  <- FastAPI + WebSocket + Redis wiring, for Phase 4 to consume
    tests/
    run_phase3.py
  phase4/               <- INTEGRATION + WEBSITE — backend confirmed live, frontend iterating
    backend/               <- FastAPI app, pipeline_runner.py (3 execution paths — see phase4/README.md)
    frontend/                <- React + Vite, brutalist/maximalist UI
    desktop_app.py / play_desktop.py  <- standalone Tkinter players (debug tools, not the product)
    gen_tv_commentary.py / apply_tv_lines.py  <- optional post-processing, NOT the grounded pipeline (see disclosure note in phase4/README.md)
    run_phase4.py
  data/
    raw/                  <- real match clips (not committed — see .gitignore)
    labels/                <- real labelled windows (football only)
  notebooks/
  scripts/
```

## Quick start

```bash
pip install -r requirements.txt

# find where live play starts before running anything else on a new clip
python phase1/find_live_play.py --input data/raw/your_clip.mp4

python phase1/run_phase1.py --input data/raw/your_clip.mp4 --start <seconds>
python phase2/run_phase2.py --clip your_clip --sport football --classifier xgboost
python phase3/run_phase3.py --clip your_clip
python phase4/run_phase4.py            # backend on :8000
cd phase4/frontend && npm install && npm run dev   # frontend on :5173
```

Or run Phases 1→3 in one command with `run_pipeline.py` at the project root
(see its own docstring for `--input`/`--skip-phase1` usage).

**YOLO weights are not included in this repo** (large binary, excluded via
`.gitignore`). Without `phase1/weights/football.pt`, the detector falls back
to plain pretrained `yolov8n.pt` (COCO classes) so the pipeline still runs,
but detection quality will be rough — COCO knows "person" and "sports ball,"
not "goalkeeper" or "referee." See `phase1/README.md` for fine-tuning your
own weights, or request the trained `football.pt` separately.

**Set `GROQ_API_KEY` for real commentary quality** (free tier — console.groq.com).
`GROQ_MODEL=openai/gpt-oss-120b` is the recommended override in `.env`
(better quality, same free tier; code default is `gpt-oss-20b`). Without a
key at all, Phase 3 falls back to a grounded, zero-dependency template/
phrase-bank generator — still factually grounded, just less fluent.
PowerShell: `$env:GROQ_API_KEY = "..."`; bash: `export GROQ_API_KEY=...`; or
add to `.env` (see `.env.example`).

**GPU acceleration (Phase 1):** set `CASTER_DEVICE=cuda` (default when a GPU
is detected) for roughly 74ms/frame on a laptop RTX 3050 vs ~280-300ms/frame
on CPU. See `phase1/README.md` for the full settings table.

## The hallucination boundary

Nothing reaches Phase 3 that didn't pass Phase 2's confidence gate, and
nothing reaches the LLM that isn't already a stripped, verified fact —
this is CASTER's core design claim, enforced at two separate points:

- **Phase 2's `confidence_gate.py`**: `NO_EVENT` (ordinary play) is always
  rejected regardless of confidence; named events must clear a threshold;
  a cooldown prevents overlapping commentary; `GOAL` needs a separately
  higher confidence bar since a false positive there is more damaging to
  trust than a missed low-confidence call; a caller-level check drops any
  window where Phase 1 never actually saw the ball.
- **Phase 3's `prompt_builder.py`**: strips internal-only fields
  (confidence, pitch zone, raw ball speed, style/language metadata) before
  the LLM ever sees the JSON, and hides score/match-time fields entirely
  when they aren't actually being tracked.

Phase 3 trusts Phase 2 completely by design — it has no way to tell whether
incoming events came from a classifier trained on real labelled footage or
on synthetic placeholder rows (see `phase2/README.md`'s synthetic-data-trap
warning). Confirm Phase 2 was trained on real data before treating Phase 3's
output as meaningful.

> ⚠️ **Open question, not yet resolved:** Phase 3's system prompt has, at
> points, been swapped between a hard-rule grounding prompt and a looser
> broadcaster-persona style prompt. Field-stripping (above) still holds
> regardless — the model structurally cannot see numbers it isn't given —
> but a persona-style prompt can still invent unlicensed flourish (confirmed:
> a real line described ball motion as "electric pace" despite `ball_speed`
> never reaching the model). See `phase4/README.md`'s "Open question" section
> for the full account before presenting commentary output as fully grounded.

## Current status (honest, not aspirational)

- [x] **Phase 1 — Vision Foundation** — complete. YOLOv8m fine-tuned
      (mAP50 0.863, mAP50-95 0.573), ByteTrack player tracking, two-hit
      temporal ball validation, CUDA-accelerated (~74ms/frame on an RTX 3050
      laptop GPU vs ~280-300ms/frame CPU). Known gaps: no team/player
      identity separation yet (no jersey-colour clustering); ball visibility
      ~50% on wide broadcast shots (85% on real footage vs 50-65% on
      synthetic test clips — a real domain-shift finding, not a bug); not
      validated as genuinely real-time/streaming end-to-end even on GPU.
- [x] **Phase 2 — Event Intelligence** — code complete and verified against
      real footage; remaining work is entirely data volume for a few thin
      classes, not code. As of the last check: ~1,086 real labelled rows
      across 12 clips, real signal (non-zero recall) on NO_EVENT, DRIBBLE,
      PASS, SHOT_ON_TARGET, GOAL, FOUL, OFFSIDE, FREE_KICK. `CORNER_KICK`,
      `TACKLE`, and `HEADER` are the live gap — near-0% recall despite real
      examples; more blind labelling has hit diminishing returns for these
      three. Established fix pattern: targeted, known-moment labelling beats
      blind sampling (OFFSIDE went from 2 un-catchable examples to 20 real
      ones once pointed at known moments by hand). *(Re-run
      `summarize_labels.py` / `evaluate_kfold.py` for current numbers.)*
- [x] **Phase 3 — Commentary Engine** — confirmed working end-to-end.
      Groq (`openai/gpt-oss-120b`) is the effective primary backend; local
      QLoRA fine-tuning of Llama 3.2 is designed but not yet trained.
      Anti-repetition fix (temperature 0.4→0.9 + per-event-type recent-line
      memory) verified against a real 51-event sequence with no near-verbatim
      repeats.
- [x] **Phase 4 — Integration** — backend confirmed live against real
      footage across all three execution paths (live, precomputed-replay,
      offline-replay). Known fixed bugs: hardcoded 1920×1080 box
      normalisation, a StrictMode blob-URL race in the video preview, a
      forced-stretch layout void. Frontend actively being iterated.

## What this repo does NOT include

- Trained YOLO weights, raw match video, and labelled training data are
  excluded via `.gitignore` (large binaries / footage rights). See each
  phase's README for how to regenerate or obtain them.
- The fine-tuned QLoRA Llama 3.2 model — not yet trained; Groq is the
  working substitute, and `local_gguf` activates automatically with no code
  changes once `CASTER_GGUF_MODEL_PATH` points at a real file.

## A note on `gen_tv_commentary.py` / `apply_tv_lines.py`

These two Phase 4 scripts deliberately produce more polished output than
the real grounded pipeline — one re-prompts Groq with no structural
grounding, the other substitutes a fixed, human-authored phrase bank. If
either is used for a demo recording, that's a disclosed, scripted polish
pass — not the live model output `pipeline_runner.py` actually generates.

## Phase READMEs

Each phase has its own README with full detail — tool references, known
bugs and fixes, exit gates, exact command sequences, troubleshooting tables:

- [`phase1/README.md`](phase1/README.md) — detection classes, fine-tuning
  your own weights, CUDA setup, ball-selection logic, full diagnostic tool
  reference, known limitations (measured, not guessed)
- [`phase2/README.md`](phase2/README.md) — the synthetic-data trap, event
  classes, labelling workflow (Gemini auto-label + manual), dedup/review
  tooling, honest k-fold evaluation
- [`phase3/README.md`](phase3/README.md) — backend fallback chain,
  grounding/field-stripping, anti-repetition mechanism, fine-tuning status
- [`phase4/README.md`](phase4/README.md) — WebSocket message shapes, the
  three execution paths, coordinate-scaling/StrictMode/MP4-faststart bugs
  (fixed), TTS warm-up fix, open prompt-grounding question, exit gate