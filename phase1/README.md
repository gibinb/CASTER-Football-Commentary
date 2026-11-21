# Phase 1 — Vision Foundation (football only)

CASTER (Computer Vision Assisted Sports Text And Event Reporter) is now a
**football-only** project. Phase 1 takes a football video clip and produces
structured, verified facts about every frame: where the players, goalkeepers,
referee and ball are, a persistent ID for each tracked person, and a
12-number feature vector per frame. Nothing here decides what *event*
happened (that is Phase 2) and nothing here writes commentary (Phase 3).

Everything downstream can only be as good as these numbers, so Phase 1 is
tested on its own, on real footage, before Phase 2 starts. Across one
debugging session this pipeline went from an actively-wrong ball detector
(a false positive could outscore the real ball) to a verified, evidence-based
system -- every fix below was checked against real frames before being kept.

```
video clip --> YOLOv8 (ball / player / goalkeeper / referee), optionally via SAHI
           --> ByteTrack (persistent player IDs; the ball is NOT tracked by ID)
           --> ball selection (two-hit temporal validation)
           --> 12-number feature vector per frame
           --> SceneState JSON per frame + 30-frame FeatureWindows  --> Phase 2
```

## What's in this folder

```
phase1/
  detection/yolo_detector.py       FootballDetector: loads weights/football.pt,
                                   per-class confidence thresholds (incl. SAHI path),
                                   runs on GPU automatically if available
  tracking/bytetrack_wrapper.py    own frame loop + ByteTrack for players;
                                   iter_tracked_frames(), verify_tracking_stability()
  features/feature_extractor.py    the 12-feature vector + two-hit ball validation
  scene_state/scene_state_builder.py   detections + features -> SceneState (schemas.py)
  configs/caster_bytetrack.yaml    tracker settings tuned for broadcast footage
  weights/football.pt              fine-tuned model (4 classes) -- you supply this
  tests/                           unit tests, no video or weights needed
  run_phase1.py                    MAIN ENTRY POINT: full Phase 1 on a clip
  check_tracking.py                quick numbers: ID churn, longest ID, raw ball visibility
  find_live_play.py                finds stretches of live play; filters offside-overlay graphics
  inspect_ball.py                  saves frames with the detected ball marked
  inspect_detections.py            saves frames with ALL classes marked; --auto-sample for rare classes
  dump_ball_candidates.py          lists every ball candidate YOLO produced, with confidence
  imgsz_sweep.py                   compares ball visibility/speed across CASTER_IMGSZ values
  sahi_ball_eval.py                compares normal vs. SAHI (tiled) inference on real frames
```

The project root must also contain `config.py` and `schemas.py` (shared by
all phases).

## Quick start

```powershell
# 1. unit tests (no video, no weights needed) -- all must pass
python phase1/tests/test_feature_extractor.py      # expect 17/17
python phase1/tests/test_scene_state.py             # expect 4/4

# 2. find where live play starts in your clip
python phase1/find_live_play.py --input data/raw/Final.mp4

# 3. tracking health on that stretch
python phase1/check_tracking.py --input data/raw/Final.mp4 --start 255 --frames 300

# 4. the full pipeline (saves outputs/<clip>_scene_states.json and _feature_windows.json)
python phase1/run_phase1.py --input data/raw/Final.mp4 --start 255 --max-frames 300
```

Always pass `--start` pointing at live play. `find_live_play.py` filters out
offside-overlay graphics automatically (see below), but it does NOT catch
celebration cutaways / crowd reaction shots after a goal -- spot-check a
couple of snapshots if a stretch looks suspicious.

On Windows PowerShell set an environment variable with
`$env:CASTER_IMGSZ = "1536"`, or put it in a `.env` file in the project root
(`config.py` loads it). Never use bash's `export`.

## The four building blocks

### 1. YOLOv8 detection (`detection/yolo_detector.py`)
One fine-tuned model, 4 classes: `ball`, `goalkeeper`, `player`, `referee`
(the Roboflow dataset "football-player-detection-bfswn"; the checkpoint calls
the player class `players`, which the detector renames to `player`). No public
dataset labels team affiliation, so there is no team separation (documented
future work: K-means on jersey colours).

**`weights/football.pt` is the finalized production model**: YOLOv8m, trained
100 full epochs at `imgsz=960`, mAP50 0.863 / mAP50-95 0.573. If you have
`weights/football2.pt` lying around, don't use it -- it's an interrupted
2-epoch checkpoint (YOLOv8s, mAP50 0.459) from an earlier training attempt,
and it also names the player class `players` *and* something else class-wise
that can silently mismatch the detector's aliasing -- it's a strictly worse,
superseded file, not an alternative.

Whether weights count as "fine-tuned" is decided from the class names inside
the checkpoint, not the file name. A stock COCO model (80 classes) is treated
as the fallback: `person` becomes `player`, `sports ball` becomes `ball`, and
there is no goalkeeper or referee.

**GPU support.** The detector now picks CUDA automatically when available
(`DEVICE = os.environ.get("CASTER_DEVICE", "cuda" if torch.cuda.is_available()
else "cpu")`, passed explicitly into every `.predict()`/`.track()` call and
into the SAHI `AutoDetectionModel`). Ultralytics does **not** reliably
auto-select CUDA just because `torch.cuda.is_available()` is true if you don't
pass `device=` yourself -- this used to silently stay on CPU even with a
working CUDA install. Measured on an RTX 3050 6GB Laptop GPU: **~74ms/frame**
at `imgsz=1280` on `football.pt`, a real, meaningful speedup over CPU (this
project's own CPU benchmark implied ~280-300ms+/frame at the same settings,
consistent with the earlier-documented ~85s to process 10s of video). Treat
the 74ms figure as laptop-GPU-specific, not a generic marketing number, if
you cite it anywhere.

To verify a model is genuinely running on GPU (not just torch-level):
checking `next(det.model.model.parameters()).device` right after load is
**not reliable** -- it can report `cpu` even when inference genuinely ran on
GPU, because Ultralytics' AutoBackend wraps a separate device-specific copy
at call time. Check GPU memory allocation jumping after a real
`detect_frame()` call, or `det.model.predictor.device` reporting `cuda:0`
directly, instead.

Settings (environment variables or `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `CASTER_DEVICE` | `cuda` if available, else `cpu` | inference device. Override to force `cpu` even with a working GPU (e.g. to reproduce the original CPU-only benchmarks), or to pin a specific CUDA device string. |
| `CASTER_IMGSZ` | `1280` | inference resolution. **Recommend overriding to `1536`**: measured on Final.mp4, 1536 gave 50% validated ball visibility vs 39% at 1280 (the `run_phase1.py` output, after the two-hit ball validator -- not the raw candidate rate `check_tracking.py` reports, which runs higher). Costs ~1.4x more time than 1280 on CPU; the GPU path makes this trade much cheaper to take. Use 960 for quick, rougher runs. |
| `CASTER_PLAYER_CONF` | `0.30` | minimum confidence for player / goalkeeper |
| `CASTER_BALL_CONF` | `0.15` | minimum confidence for the ball (deliberately lower: a few-pixel ball never scores like a player) |
| `CASTER_REFEREE_CONF` | `0.70` | minimum confidence for the referee -- deliberately **higher** than `PLAYER_CONF`. Measured on 9 real referee detections on Final.mp4 (confidence 0.31-0.96): confidence does NOT cleanly separate correct from wrong (0.41 was wrong in one frame, correct in another; 0.65 was wrong while 0.58 was correct). 0.70 is the lowest value that rejected every confirmed-wrong case while keeping the clearest correct ones -- a deliberately conservative, precision-over-recall choice, since `official_signal` is a secondary Phase 2 signal, not load-bearing for ball/player-driven commentary. |
| `CASTER_USE_SAHI` | `0` | `1` routes detection through SAHI sliced/tiled inference instead of one full-frame pass (`pip install sahi` first). Checked against 7 real sliced-only candidates across 3 clips (two FIFA26-rendered, one real broadcast footage): all 7 landed on the real ball/object. Costs roughly 3x the inference time. Opt-in, not default, because of that cost -- worth it for a clip where recall matters more than speed. |
| `CASTER_SAHI_SLICE_SIZE` | `640` | SAHI tile width/height in pixels. Ignored unless `CASTER_USE_SAHI=1`. |
| `CASTER_SAHI_OVERLAP` | `0.2` | fraction of overlap between adjacent SAHI tiles. |

### 2. ByteTrack tracking (`tracking/bytetrack_wrapper.py`)
Players, goalkeeper and referee get persistent IDs from ByteTrack
(`configs/caster_bytetrack.yaml`). The ball is deliberately **not** tracked by
ID: a tiny, flickering object is almost always dropped by a tracker, so it is
handled by the ball-selection logic below instead. The runner processes about
30 frames per second of video (a 60 fps clip uses stride 2); keep the effective
rate at or above ~15 fps or IDs cannot be linked between frames.

### 3. Ball selection (`features/feature_extractor.py`)
YOLO often returns several "ball" candidates per frame, some of them false
(boots, line markings, graphics, crowd fragments). Measured directly on real
frames: a false candidate scored 0.71 confidence while the real ball in the
SAME frame scored only 0.31 -- no confidence threshold alone can separate
that correctly, because the wrong object was simply more confident.

The fix is **temporal, not a threshold**: `_resolve_ball()` only trusts a
ball position once it has been seen in `BALL_CONFIRM_FRAMES` (2) consecutive
frames, roughly in the same place. A one-off false positive essentially never
repeats in the same spot the very next frame; the real ball does. Rules:

- a **new or reacquiring** track (first sighting, or after
  `BALL_CONTINUATION_MAX_MISSING` = 5 consecutive missed frames) needs
  confidence >= `BALL_ACQUIRE_MIN_CONF` (0.25) to even become a pending
  candidate, and needs a second, consistent sighting the very next frame to
  be confirmed -- there is NO bypass that trusts a single post-gap sighting
  (an earlier version had one; it is what let a 0.16-confidence boot get
  accepted as the ball after a gap in a real test clip);
- an **already-confirmed** track continues on proximity alone (within
  `BALL_MAX_JUMP_NORM` + `BALL_JUMP_GROWTH` per missed frame) -- but ALSO
  needs confidence >= `BALL_CONTINUATION_MIN_CONF` (0.20). This was added
  after a confirmed track was found to drift onto a genuinely wrong
  0.19-confidence detection purely because it was near the last position;
  0.20 is deliberately lower than the 0.25 acquisition floor, so a track
  still survives an ordinary confidence dip (motion blur, partial occlusion);
- velocity on the very frame a ball is confirmed is computed from the real
  backfilled trajectory, not read as zero from an empty history;
- `contact_flag` is never raised from a ball detection under 0.25 confidence,
  even on an already-confirmed track.

These thresholds came from real-frame evidence (`dump_ball_candidates.py`,
`inspect_ball.py`), checked and re-checked across multiple clips -- not
tuned-once-and-forgotten. They still are not perfect: when the real ball
simply never appears among YOLO's candidates for a frame (occluded,
contested), no selection logic can recover it -- see "Known limitations."

When the ball is not confirmed, the vector carries the last known position
(or 0.5, 0.5 before the first sighting) and `ball_detected=False` /
`ball_missing_frames` say so honestly in every SceneState. Consumers must
check those fields.

### 4. Feature vector and scene state
12 numbers per frame (index map in `schemas.py`): ball x/y, ball velocity
dx/dy/speed, nearest-player distance, players nearby, pitch zone, referee
visible, ball height proxy, contact flag, boundary proximity. A 30-frame
sliding window of these is one `FeatureWindow`, the unit Phase 2 trains on.
`pitch_zone` is a coarse bucket of the ball's x position **in the camera
image**, not a true pitch location.

## The diagnostic tools

| Tool | Use it when |
|---|---|
| `find_live_play.py` | before anything else on a new clip: prints a live/not-live timeline and a suggested `--start`. Checks pitch-green ratio AND an offside-overlay filter (a cyan/teal line-overlay color check plus an OFFSIDE-banner dark-band check, calibrated against real frames). Does NOT catch celebration/crowd-reaction cutaways -- spot-check if unsure. `--no-graphic-filter` reverts to green-only. |
| `check_tracking.py` | quick health check: unique IDs, longest ID run, % of frames with a raw ball candidate (before the two-hit validator, so it reads higher than `run_phase1`'s `ball visible` line). |
| `inspect_ball.py --clip NAME --video PATH --snap T1 T2` | after a `run_phase1` run: saves frames to `outputs/ball_snaps/` with a red circle on each YOLO ball box and a yellow cross where the pipeline thinks the ball is. Snap times must fall inside the processed range. |
| `inspect_detections.py --clip NAME --video PATH --snap T1 T2 [--classes player referee ...]` | same idea, extended to ALL classes (ball/player/goalkeeper/referee), color-coded boxes with confidence labels. Prints a per-class detection-rate summary even without `--video`/`--snap`. Use `--auto-sample CLASS -n N` to automatically find and sample real frames spanning a rare class's full confidence range, instead of guessing timestamps (a class with 100 hits in 300 frames is usually missed entirely by manually-guessed 1-second-spaced snaps). |
| `dump_ball_candidates.py --clip NAME --times T1 T2` | when the ball looks wrong: lists every candidate YOLO produced near those times with size and confidence, and shows which one was used. |
| `imgsz_sweep.py --input PATH --start N --frames N [--sizes 960 1280 1536]` | compares raw ball visibility and wall-clock time across several `CASTER_IMGSZ` values on the same clip/stretch, via repeated `check_tracking.py` runs -- shows the real speed/recall tradeoff instead of guessing. |
| `sahi_ball_eval.py --input PATH --start N --num-frames N [--save-annotated]` | compares normal vs. SAHI sliced inference on N real sampled frames: how often each finds a ball candidate, and the actual slowdown. `--save-annotated` saves images for the sliced-only candidates above the acquisition floor, so you can verify they're correct before trusting them. |

## Phase 1 exit gate

Phase 1 is done when, on a stretch of live play, with no crashes:

1. both test files pass (17/17, 4/4);
2. `run_phase1.py` prints plausible ball positions (ball speed mostly well
   under 0.05; a "ball" that never moves, or jumps across the frame, is a bug);
3. `inspect_ball.py` / `inspect_detections.py` snapshots show boxes on the
   real objects in most frames where detected;
4. outputs are written: `outputs/<clip>_scene_states.json` and
   `outputs/<clip>_feature_windows.json` (windows of shape 30x12).

This project cleared that bar on Final.mp4 (FIFA26-rendered), Semifinal01/02,
LFinal, sample1, and rm.mp4 (real broadcast footage). GiBiN confirmed Phase 1
done, then separately confirmed CUDA/GPU inference working end-to-end on the
real dev machine (Lenovo LOQ, RTX 3050 6GB Laptop GPU) -- see the GPU support
note above.

## Known limitations (measured, not guessed)

- **Ball visibility is about 50% on wide broadcast shots** (validated rate,
  Final.mp4, `CASTER_IMGSZ=1536`, two-hit validator applied). The ball is a
  few pixels wide and often hidden by players. Phase 2 must tolerate a
  missing ball in roughly half of frames.
- **Recall is capped by what YOLO offers as a candidate at all.** When the
  real ball is occluded or never detected in a frame, no selection logic can
  recover it -- this needs a better ball detector (more/harder training
  data) or a multi-hypothesis tracker, both deliberately out of scope so far.
  Documented future work.
- **The ball detector performs noticeably better on real broadcast footage
  than on the FIFA26-rendered test clips.** Measured: normal-pass baseline
  detection was 85% on one real-broadcast clip (rm.mp4) vs. 50-65% on the
  synthetic ones, at comparable settings. Most of this session's test clips
  were synthetic game-engine renders, not real camera footage -- treat the
  lower numbers above as a worst case, not the expected real-world rate.
- **The referee class has a real, now-mitigated accuracy problem.** Checked
  against 9 real detections: 3 of 9 were an ordinary field player mistaken
  for the referee, at confidences overlapping the correct detections'
  range. `CASTER_REFEREE_CONF=0.70` removes every confirmed-wrong case found
  so far, at the cost of some correct-but-weaker sightings (an already rare
  class gets rarer). `official_signal` is a secondary feature for Phase 2,
  not load-bearing for ball/player-driven commentary.
- **Player ID churn is high** on camera pans (roughly 30–200 unique IDs per
  10 s of live play across the test clips, against about 22 players on the
  pitch). ByteTrack has no camera-motion compensation. Nothing in Phase 1's
  feature vector uses IDs, so this does not affect Phase 2's inputs
  (confirmed by grepping Phase 2/3 code for `track_id`/`nearest_player_id`).
- **`contact_flag` is a rough hint**: the ball is compared with the bottom of
  each player's whole bounding box, not a real foot/limb detection.
- **No team/player identity separation yet** (documented future work: K-means
  on jersey colours). Scene JSON can never say "his teammate" until this is
  built -- Phase 1's gap, not Phase 2/3's.
- **Not real-time on CPU**, especially at `CASTER_IMGSZ=1536` or with
  `CASTER_USE_SAHI=1`; plan for upload-and-wait processing there. **On a
  CUDA-capable GPU** (set `CASTER_DEVICE=cuda`, the default when one is
  detected) per-frame inference drops to roughly 74ms on a laptop RTX 3050 at
  `imgsz=1280` -- much closer to real-time, though still not validated as a
  genuinely live/streaming pipeline end-to-end.
- **`find_live_play.py` does not filter celebration/crowd-reaction cutaway
  montages** (only offside-overlay graphics). A stretch right after a goal
  may still land on a non-play frame -- found on one test clip; not fixed.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| ball "visible" ~80% with very few distinct positions | you are on a graphic or a celebration cutaway; use `--start` inside live play (`find_live_play.py`), and spot-check with `inspect_ball.py` even if it looks "live" |
| `ball visible` < 30% warning | ball too small for the resolution (try `CASTER_IMGSZ=1536`) or hidden in a crowded shot; consider `CASTER_USE_SAHI=1` |
| `imgsz=` header shows a value you did not set | check `.env` and the PowerShell session variable |
| player boxes appear but `official_signal` never fires | referee class rarely clears `CASTER_REFEREE_CONF` (0.70 by design -- see Known limitations), or running on the COCO fallback |
| `ModuleNotFoundError: config` | run from the project root; `config.py` and `schemas.py` must sit beside the `phase1/` folder |
| `CASTER_USE_SAHI=1` raises an ImportError | run `pip install sahi` |
| inference still looks CPU-slow despite a working GPU | don't trust `next(det.model.model.parameters()).device` as a check -- it can say `cpu` even when inference genuinely ran on GPU. Check GPU memory usage during a real `detect_frame()` call, or confirm `CASTER_DEVICE`/`torch.cuda.is_available()` directly |
| you have both `football.pt` and `football2.pt` in `weights/` | use `football.pt` only -- `football2.pt` is a superseded, far weaker interrupted checkpoint, not an alternative |

## What changed from the cricket + football version

Cricket support was removed: `weights/cricket.pt` (it was actually an
un-fine-tuned COCO model), the YOLO sport-detector, the CLIP sport-detector,
all cricket classes, zones and events in `config.py`, and all `sport`
arguments. `FeatureExtractor()`, `FootballDetector()` and
`build_scene_state(frame_id, timestamp_sec, ...)` no longer take a sport.
The JSON still carries `"sport": "football"` so later phases stay
self-describing. Phase 2–4 code that references `SPORT_CRICKET` or
`CRICKET_*` will fail to import until it is converted; delete those references.

## What changed in the ball-detector debugging session

Starting state: a single-frame gate that could be, and was, fooled by a
higher-confidence false positive. In order: added a plausibility jump-gate
(fixed some cases, hit a hard ceiling on the 0.71-vs-0.31 case) -> replaced it
with two-hit temporal validation (fixed the ceiling case and a post-gap
reacquire bug) -> found and fixed a gap where continuation had no confidence
floor at all (a track could drift onto a nearby-but-wrong 0.19-confidence
object) -> evaluated and integrated SAHI as an opt-in high-recall mode
(verified 7/7 correct on real frames across 3 clips) -> found and fixed the
same class of problem on the referee class (0.70 floor, evidence-based) ->
discovered real broadcast footage detects substantially better than the
FIFA26-rendered test clips. Every step above was verified against real
frames before being kept, not just unit-tested.

## What changed in the hardware / GPU session

The project was CPU-only through every fix above. Switched to a CUDA-capable
dev machine (Lenovo LOQ, RTX 3050 6GB Laptop GPU): installed `torch==2.6.0+cu124`
(the `cu121` wheel index had no match for the current torch version; `cu124`
did) in the project's `venv`, confirmed `ultralytics`/`accelerate`/`peft`/
`bitsandbytes` stayed compatible with no reinstalls needed. Found the real
root cause of inference staying on CPU despite a working CUDA install
(Ultralytics does not auto-select CUDA without an explicit `device=` argument
on `.predict()`/`.track()`), fixed it, and verified the fix with GPU memory
allocation and `det.model.predictor.device` -- NOT with
`next(det.model.model.parameters()).device`, which is not a reliable check
here. Benchmarked ~74ms/frame at `imgsz=1280` on `football.pt`. Also deleted
two now-redundant artifacts found on the same machine: an unused root-level
`yolov8n.pt` (only ever an emergency fallback, never actually reached) and a
stale duplicate `.venv` folder left over from before the switch to `venv`.