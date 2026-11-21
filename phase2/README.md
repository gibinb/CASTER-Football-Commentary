# CASTER Phase 2 — Event Intelligence (FOOTBALL ONLY)

Phase 2 turns Phase 1's per-frame feature vectors into **named football
events** (PASS, GOAL, FOUL, ...), gates out anything the classifier
isn't confident about, and builds the structured scene JSON that Phase 3's
LLM narrates from. Nothing reaches Phase 3 that didn't pass the
confidence gate — this is CASTER's hallucination-prevention boundary.

**Status: code complete.** Every tool below has been built and verified
against real footage. The remaining work is entirely data volume for a
few thin classes — not code. See "Current data state" below.

---

## ⚠️ Before you do anything: the synthetic-data trap

`generate_synthetic_data.py` exists ONLY to prove the pipeline's wiring
works, using fake `random.uniform()` feature vectors that correspond to
nothing in any real footage. It is not a shortcut to real training data.

**Do not train on synthetic data and call it done.** A model trained only
on synthetic rows gives confident-looking predictions that describe
events that never happened — exactly the hallucination problem this
whole project's architecture exists to prevent, just moved one phase
upstream. Always remove synthetic rows (Step B below) before real
training (Step D).

---

## Quick start

```bash
# ============================================================
# STEP A — wiring smoke test only (NOT real training data)
# ============================================================
python phase2/labelling/generate_synthetic_data.py --sport football --n-per-class 60
python phase2/event_classifier/xgboost_clf.py --sport football
python phase2/tests/test_confidence_gate.py
python phase2/tests/test_scene_json_builder.py
python phase2/tests/test_xgboost_clf.py

# ============================================================
# STEP B — remove the synthetic placeholder data (required)
# ============================================================
python phase2/labelling/remove_synthetic_data.py --sport football --apply

# ============================================================
# STEP C — label real data from real Phase 1 output
# ============================================================
python phase1/run_phase1.py --input data/raw/myclip.mp4

# Fast path: let Gemini auto-label (see "Auto-labelling" below)
python phase2/labelling/run_gemini_autolabel.py \
    --windows outputs/myclip_feature_windows.json \
    --video data/raw/myclip.mp4 --sport football

# Manual path: tag windows yourself (see "Manual labelling" below)
python phase2/labelling/label_tool.py \
    --windows outputs/myclip_feature_windows.json --sport football

# ============================================================
# STEP D — check your data, then train on it
# ============================================================
python phase2/labelling/summarize_labels.py --sport football
python phase2/labelling/dedupe_labels.py --sport football --apply
python phase2/event_classifier/xgboost_clf.py --sport football
python phase2/event_classifier/evaluate_kfold.py --sport football

# ============================================================
# STEP E — run Phase 2 on real Phase 1 output
# ============================================================
python phase2/run_phase2.py --clip myclip --sport football --classifier xgboost
```

---

## The event classes

```
PASS, DRIBBLE, TACKLE, SHOT_ON_TARGET, GOAL, FOUL,
CORNER_KICK, FREE_KICK, HEADER, OFFSIDE, NO_EVENT
```

**`NO_EVENT` is the 11th class, appended last on purpose** — label IDs are
positional, so it must never be inserted earlier or every ID after it
silently shifts. Without this class, the classifier was structurally
forced to call *every* window a named event, even completely ordinary
play — which is why an early version collapsed onto one dominant class
(DRIBBLE, 84–96% of all predictions) regardless of what was actually
happening. `NO_EVENT` means "the classifier correctly recognised
ordinary play" — `confidence_gate.py` rejects it unconditionally, at any
confidence. It is training signal only; Phase 3 never sees it.

---

## The confidence gate

`confidence_gate.py` is CASTER's hallucination boundary. Rules, in order:

1. `NO_EVENT` → always rejected, any confidence.
2. Confidence must clear the sport-specific threshold.
3. At least `MIN_SECONDS_BETWEEN_EVENTS` must have passed since the last
   commentary line (prevents Phase 4's TTS from overlapping itself).
4. Priority events (`GOAL`) need a separately higher confidence bar — a
   false GOAL call is more disruptive to trust than a missed low-confidence one.
5. (Caller-level) `ball_visibility_ok()` drops windows where Phase 1
   never actually saw the ball — classification on a ball-less window
   isn't trustworthy regardless of what the classifier says.

---

## Auto-labelling with Gemini

`run_gemini_autolabel.py` (single clip) and `batch_autolabel.py` (loops
every clip in `data/raw/`) send sampled windows to Gemini Vision, saving
confident results straight into `data/labels/<sport>.jsonl`.

- **`--start-sec` / `--end-sec`** — target a known moment instead of
  blind whole-clip sampling. This matters a lot: blind stride-sampling
  reliably misses rare events (a 25-minute match might contain exactly
  one corner kick, and uniform sampling has to get lucky to land on it).
  If you know roughly when something happens, targeting it directly gets
  far more useful labels per API call.
- **Already-labelled windows are automatically skipped** — no API call,
  no duplicate row. Re-running the labeller on a clip you've already
  covered is safe and won't re-pollute your data (see "Duplicate
  labels" below for why this matters).
- **Network/server hiccups are handled gracefully** — a dropped
  connection or a Gemini 5xx retries with backoff (4 attempts,
  5→10→20→40s) before giving up cleanly; it does not hang forever, and
  it does not crash with an unhandled traceback.
- Gemini is assistive, not ground truth — it still makes real,
  predictable mistakes (see "Review your labels" below).
- **Gemini cannot judge OFFSIDE.** It's a defender-line-position
  judgment that isn't visible in 6 sampled frames without deep rule
  context. If you need real OFFSIDE examples, you have to know the
  moment yourself and force-label it manually (next section) — blind
  or targeted Gemini sampling will not find it.

### Finding rare events without scrubbing a whole match

`find_candidates.py` flags likely moments using Phase 1's own
already-computed features — zero new processing, instant:

| category | signal | good for |
|---|---|---|
| `boundary` | ball near touchline/goal line | CORNER_KICK |
| `aerial` | high ball-height percentile | HEADER |
| `fast_ball` | high ball-speed percentile | SHOT_ON_TARGET, GOAL |
| `official` | referee visible in frame | **weak — mostly just means visible, not signalling.** Confirmed noisy in practice; don't rely on it. |

```bash
python phase2/labelling/find_candidates.py --clip myclip --category boundary
```
Prints candidate timestamps with a ready-to-use `--start-sec`/`--end-sec`
suggestion for each — feed the good ones straight into
`run_gemini_autolabel.py`.

---

## Manual labelling

`label_tool.py` lets you tag windows yourself when you already know what
happened (an offside you spotted, a Gemini miss, anything Gemini
structurally can't judge). It does **not** replay the original video by
default — pair it with your own video player scrubbed to the printed
timestamp, or use `--video` to have it save an actual frame image per
window instead:

```bash
# Jump to a known moment instead of starting at window 0, and see
# frame images instead of only raw numbers:
python phase2/labelling/label_tool.py \
    --windows outputs/myclip_feature_windows.json \
    --video data/raw/myclip.mp4 --sport football \
    --start-sec 288 --end-sec 300 --stride 30
```

**`--stride` matters more than it sounds like it should**:
`feature_windows.json` has one entry *per video frame* (~30/sec), not
pre-spaced. `--stride 30` gives roughly one window per second — without
it, a 12-second range still shows ~360 individual prompts.

Keys: number keys tag a named event, `n` tags `NO_EVENT` (**saves** it —
most of any match is ordinary play, and the classifier needs real
negative examples or it can never predict anything else confidently),
`s` truly skips (does **not** save — use only when you genuinely can't
judge the window).

---

## Checking and cleaning your data

**Always check these before trusting a training run:**

```bash
python phase2/labelling/summarize_labels.py --sport football
```
Per-clip and per-class row counts — know what you actually have before
guessing.

```bash
python phase2/labelling/dedupe_labels.py --sport football
```
Dry-run report of duplicate windows. **This matters more than it sounds
like it should** — re-running labelling sessions on the same clip across
different sessions silently produced 226 duplicate rows (20.7% of the
dataset at the time this was caught), 23 of which had outright
*contradictory* labels (the same window called two different events).
Worse: duplicate rows were partly **leaking across k-fold train/test
splits** (the same window, as two different rows, landing on both sides
of a split) — a model being "tested" on an exact copy of something it
trained on isn't measuring real generalisation. This inflated an early
accuracy reading by ~3.8 percentage points in a controlled before/after
comparison. `--apply` removes the safe (consistent) duplicates
automatically; genuine conflicts need a human call —
`review_labels.py --conflicts-only` is built for exactly that.

```bash
python phase2/labelling/review_labels.py --sport football --event GOAL
```
Spot-check a class's real labels against saved frame images, relabel or
delete what Gemini got wrong, then `--apply` (auto-backs-up the original
file first). Worth doing whenever k-fold shows a class performing worse
than its row count would suggest — that pattern usually means label
quality, not volume, is the problem.

---

## Evaluating the classifier honestly

```bash
python phase2/event_classifier/evaluate_kfold.py --sport football
```

This is the standing way to judge whether any change — more data, a
code fix, a review pass — actually helped. A single train/test split is
too noisy to trust on an imbalanced, data-scarce classifier; this does
stratified k-fold (auto-reducing k if any class has too few examples to
split meaningfully), reports mean **and spread** per class, and prints a
pooled confusion matrix showing exactly which classes get mistaken for
which — the real, data-grounded way to decide if two classes should be
merged (a documented fallback if a confusion persists with more data,
not something to do preemptively).

**Read macro F1, not raw accuracy, as the headline number.** `NO_EVENT`
dominates raw accuracy numerically but never reaches Phase 3 — a model
can score well on raw accuracy by just getting good at recognising
ordinary play, while still being useless at the actual events that
matter. Macro F1 treats every class equally and is the more honest
signal for whether Phase 3 will actually get useful input.

---

## Current data state

*(Point-in-time snapshot — re-run `summarize_labels.py` and
`evaluate_kfold.py` yourself for the real current numbers; this will go
stale.)*

As of the last full check: ~1,086 real labelled rows across 12 source
clips. Real signal (non-zero recall) on NO_EVENT, DRIBBLE, PASS,
SHOT_ON_TARGET, GOAL, FOUL, OFFSIDE, FREE_KICK. **CORNER_KICK, TACKLE,
and HEADER are the live gap** — all three sit at or near 0% recall
despite having real examples. Direct evidence (HEADER specifically
gained 10 new real rows with zero recall improvement) that *more blind
labelling has hit diminishing returns* for these three — the fix is
either much more volume, a short compilation-style clip (far more
signal per minute than mining full matches for a rare event), or
accepting they stay weak for now. A weak class doesn't break anything —
the confidence gate just stays quiet about it, which is an acceptable,
documentable limitation, not a crash risk.

**Established, repeatedly-confirmed pattern**: targeted, known-location
labelling beats blind sampling. OFFSIDE went from un-catchable by Gemini
(2 examples) to 20 real examples once pointed at 4 known real moments by
hand via `label_tool.py --start-sec/--end-sec`.

---

## Known limitation outside Phase 2's scope

Phase 1 has no team/player identity separation yet (no jersey-colour
clustering, no OCR on numbers) — so Phase 2's scene JSON can never say
*which* team or player did something, only that an event of a given type
happened. This caps how specific Phase 3's commentary can ever get
("a pass was completed" vs. "he found his teammate") regardless of how
good Phase 2's classification becomes. This is a Phase 1 gap, not
something Phase 2 can fix.

---

## Tool reference (phase2/)

| Tool | Purpose |
|---|---|
| `confidence_gate.py` | The hallucination boundary — see above |
| `scene_json_builder.py` | Builds the structured JSON Phase 3 consumes |
| `run_phase2.py` | Runs the full classify→gate→JSON pipeline on real Phase 1 output |
| `event_classifier/xgboost_clf.py` | Trains/loads the classifier (native Booster API — handles sparse real data without crashing on missing classes) |
| `event_classifier/evaluate_kfold.py` | Honest stratified k-fold evaluation |
| `event_classifier/dataset.py` | Label encoding, feature loading |
| `labelling/generate_synthetic_data.py` | Wiring smoke-test data (see warning above) |
| `labelling/remove_synthetic_data.py` | Strips synthetic rows before real training |
| `labelling/label_tool.py` | Manual tagging, with `--start-sec`/`--end-sec`/`--stride`/`--video` |
| `labelling/run_gemini_autolabel.py` | Auto-label one clip via Gemini |
| `labelling/batch_autolabel.py` | Auto-label every clip in `data/raw/`, skips already-labelled windows |
| `labelling/find_candidates.py` | Flags likely rare-event moments from Phase 1's own features |
| `labelling/summarize_labels.py` | Per-clip/per-class coverage report |
| `labelling/dedupe_labels.py` | Finds and removes duplicate label rows |
| `labelling/review_labels.py` | Spot-check/correct real labels; `--conflicts-only` resolves duplicate disagreements |