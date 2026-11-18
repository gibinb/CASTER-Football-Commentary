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
    detection/           <- YOLOv8 wrapper (SportDetector)
    tracking/             <- ByteTrack integration (track_video())
    features/             <- 12-feature extractor
    scene_state/           <- scene state JSON builder
    weights/                <- football.pt (YOLOv8m, not committed — see below)
    tests/
    run_phase1.py
  phase2/               <- EVENT INTELLIGENCE — code complete, data volume is the live bottleneck
    event_classifier/     <- xgboost_clf.py (native Booster API)
    labelling/              <- label_tool.py, remove_synthetic_data.py, batch_autolabel.py
    confidence_gate.py       <- threshold + cooldown + priority-override rules
    scene_json_builder.py     <- builds the exact JSON Phase 3 reads
    tests/
    run_phase2.py
  phase3/               <- COMMENTARY ENGINE — confirmed working
    data_collection/      <- scrapers (BBC Sport; ESPNcricinfo scraper is cricket-era, stale)
    finetune/               <- QLoRA pipeline (not yet trained)
    inference/               <- commentary_engine.py, prompt_builder.py, groq_client.py
    backend/                  <- FastAPI + WebSocket wiring
    tests/
    run_phase3.py
  phase4/               <- INTEGRATION + WEBSITE — backend confirmed live, frontend iterating
    backend/               <- FastAPI app, pipeline_runner.py
    frontend/                <- React + Vite, brutalist/maximalist UI
    desktop_app.py / play_desktop.py  <- standalone Tkinter players (debug tools, not the product)
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

python phase1/run_phase1.py --input data/raw/your_clip.mp4
python phase2/run_phase2.py --clip your_clip --classifier xgboost
python phase3/run_phase3.py --clip your_clip
python phase4/run_phase4.py            # backend on :8000
cd phase4/frontend && npm install && npm run dev   # frontend on :5173
```

No `--sport` flag anywhere — the pipeline is football-only end to end.

**YOLO weights are not included in this repo** (large binary, excluded via
`.gitignore`). Without `phase1/weights/football.pt`, `SportDetector` falls
back to plain pretrained `yolov8n.pt` (COCO classes) so the pipeline still
runs, but detection quality will be rough — COCO knows "person" and "sports
ball," not "goalkeeper" or "referee." See `phase1/README.md` for fine-tuning
your own weights, or request the trained `football.pt` separately.

**Set `GROQ_API_KEY` for real commentary quality** (free tier, 14,400
req/day — https://console.groq.com). Without it, Phase 3 falls back to a
grounded template/phrase-bank generator — still factually grounded, just
less fluent. PowerShell: `$env:GROQ_API_KEY = "..."`; bash: `export
GROQ_API_KEY=...`; or add to `.env` (see `.env.example`).

## Current status (honest, not aspirational)

- [x] **Phase 1 — Vision Foundation** — complete. YOLOv8m fine-tuned
      (mAP50 0.863), ByteTrack integration, CUDA-accelerated (~74ms/frame on
      an RTX 3050 laptop GPU). Known gap: no team/player identity separation
      yet (no jersey-colour clustering) — documented future work, does not
      block downstream phases.
- [x] **Phase 2 — Event Intelligence** — code complete, 12/12 tests pass.
      Real bottleneck is labelled data volume, not code: as of the last
      session, 1086 real labelled rows give solid recall on most classes but
      `CORNER_KICK` remains a dead class (0% recall, 8 examples) and
      `TACKLE`/`HEADER` regressed to 0% despite real volume.
- [x] **Phase 3 — Commentary Engine** — confirmed working end-to-end.
      Groq (`openai/gpt-oss-120b`) is the effective primary backend; local
      QLoRA fine-tuning of Llama 3.2 is designed but not yet trained.
      Anti-repetition fix verified against a real 51-event sequence.
- [x] **Phase 4 — Integration** — backend confirmed live against real
      footage (sport detection, event feed, feature vectors, Groq commentary
      all working end-to-end). Frontend actively being iterated.

## What this repo does NOT include

- Trained YOLO weights, XGBoost models, raw match video, and labelled
  training data are excluded via `.gitignore` (large binaries / footage
  rights). See each phase's README for how to regenerate or obtain them.
- The fine-tuned QLoRA Llama 3.2 model — not yet trained; Groq is the
  working substitute.