# manual_reviewer — next phases plan

Self-contained handoff. Read this cold; the code referenced is enough to start.

## What's already done

| Phase | Status | Files |
|---|---|---|
| 1: aav4 contracts + dedup migration + pipeline_io | done | [pipeline_io/](../pipeline_io/), `Stage.HUMAN_REVIEW`, `HumanReviewResult` |
| 2: Label Studio ML backend | **done** | [ml_backend/](../ml_backend/), `SAM3ClickMaskRequest`/`Response`, `SAM3OneModel.click_mask` |
| 3: Cross-frame static-object propagation (image-mode) | done | [reconcile/](../reconcile/), [scripts/run_reconcile.py](../scripts/run_reconcile.py), `Stage.RECONCILE` |
| 4: Deduplicator carry-forward | done | `data_miner/modules/deduplicator.py` writes `dedup_status` / `dedup_cluster_id` |
| 5: SAM 3.1 model + server (originally Phase 5; merged into Phase 7) | done | [data_miner/auto_annotation_v4/models/sam3_1.py](../../data_miner/auto_annotation_v4/models/sam3_1.py), [model_servers/sam3_1.py](../../data_miner/auto_annotation_v4/model_servers/sam3_1.py) |
| 6: Rex-Omni model + server | **done** | [data_miner/auto_annotation_v4/models/rex_omni.py](../../data_miner/auto_annotation_v4/models/rex_omni.py), [model_servers/rex_omni.py](../../data_miner/auto_annotation_v4/model_servers/rex_omni.py), `DetectorName.REX_OMNI`, port 3015 |
| 7: Wire SAM 3.1 into manual_reviewer | done | [reconcile/sam3_client.py](../reconcile/sam3_client.py) `Sam3OneHttpClient`, `run_reconcile.py --backend sam3_1` default |
| 8: GD policy (manual_reviewer doesn't use GD; aav4 keeps it) | done | [docs/detectors.md](detectors.md) |

## What's left

Phase 2 and Phase 6 landed in code, but a live integration test against
real Label Studio + a real SAM 3.1 LitServe (run on 2026-04-28) surfaced
several wire bugs that the existing unit tests don't catch — they stub the
HTTP and SDK boundaries. These need to land in a focused PR before
Workflow A is usable end-to-end.

### Live-integration bugs found and resolved (2026-04-28)

| # | Where | Bug | Status |
|---|---|---|---|
| 1 | [models/sam3_1.py](../../data_miner/auto_annotation_v4/models/sam3_1.py) `refine()` / `click_mask()` | Sent bbox prompts as **pixel xyxy** but upstream wants **normalized [0,1] xywh** (`sam3_video_inference.py:891 assert (boxes_xywh <= 1).all()`). | **fixed** — added `_xyxy_norm_to_xywh_norm()` and convert at the seam. |
| 2 | [models/sam3_1.py](../../data_miner/auto_annotation_v4/models/sam3_1.py) `click_mask()` | `add_prompt(points=...)` requires `cached_frame_outputs` populated — only true mid-video. Fresh single-image session → `AssertionError: No cached outputs found`. | **fixed** — `SAM3OneModel.load()` now also builds `Sam3Image(enable_inst_interactivity=True)` from the same checkpoint. `click_mask` routes through `Sam3Processor.set_image` + `Sam3Image.predict_inst(point_coords=, point_labels=, multimask_output=True)` and picks the highest-IoU mask. True point-prompted segmentation. |
| 3 | [models/sam3_1.py](../../data_miner/auto_annotation_v4/models/sam3_1.py) `_first_object` / `_object_bbox_px` | Parser read `frame.get("objects")` but SAM 3.1 v3 emits parallel arrays `out_obj_ids` / `out_probs` / `out_boxes_xywh` / `out_binary_masks`. → 0 results from refine/click/text always. | **fixed** — added `_frame_objects()` helper that converts parallel arrays to the existing object-dict shape. `_object_bbox_px(obj, w, h)` now uses upstream `bbox_xywh_norm` directly when available, falls back to `_mask_to_bbox` otherwise. |
| 4 | [ml_backend/server.py](../ml_backend/server.py) `ManualReviewerMLBackend` | LS SDK's `_manager.predict` requires `cls._current_model` populated by `/setup` first; our smoke test POSTed straight to `/predict`. | **not actually a bug** — LS UI auto-calls `/setup` when the backend is attached in the project. Manual smoke tests need to POST `/setup` first. |

### Auto-label feature extensions (planning, 2026-04-28)

Discussion-first plan for the next round of auto-label features (smart_click
class fix, within-image visual prompting, cross-frame same-object
propagation, master toggles, UI layout) lives at
[auto_label_extensions.md](auto_label_extensions.md). Work through it with
the user one phase at a time before writing code.

### Quirks remaining (workflow-tunable, not blockers)

- SAM 3.1 text_detect is **prompt-sensitive**: on the datatang test frame
  `"vehicle"` → 3 boxes, `"car"` → 1, `"person"` → 1; but `"truck"` and
  `"bicycle"` → 0. Suggest reviewers try synonyms or use bbox-mode tools.
  (The same frame's aav4 `finalize` correctly classifies the truck via
  the VLM evaluate stage, so the seeded predictions are still correct.)
- smart_click returns the bbox labeled with the keypoint label
  (`positive`/`negative`) rather than a class. Reviewer changes the class
  via the rectangle dropdown after the box appears. Polish: make the
  ML backend pick the currently-selected `RectangleLabels` class from
  the LS context if present.
- 199 unit tests pass after the schema rewrite — but they still stub the
  HTTP boundary. Live-integration tests against a real SAM 3.1 LitServe
  remain the recommended next step (env-gated).

### Required follow-up work

Pick a focused PR that addresses #1-#4 with **live integration tests** (not
just stubbed unit tests). Suggested approach:

- **#1 bbox coord convention**: change `_denorm_xyxy` callsites in
  `refine()` / `click_mask()` to send normalized xywh
  `[x_norm, y_norm, w_norm, h_norm]`. Decode the response (which comes
  back in pixel coords from `_mask_to_bbox`) → normalize → return.
- **#2 click cache prereq**: the right fix is to route click→mask through
  `SAM3InteractiveImagePredictor` (single-image API at
  [scratchpad/DART/sam3/model/sam1_task_predictor.py](../../scratchpad/DART/sam3/model/sam1_task_predictor.py)),
  not the video predictor. Add a separate predictor instance to
  `SAM3OneModel.load()`, call `set_image()` + `predict(point_coords=...)`
  for click_mask. Leave the video predictor for refine/text/track.
- **#3 text_detect**: needs investigation — possibly the SAM 3 (not 3.1)
  checkpoint at HF `facebook/sam3` lacks the open-vocab text prompting
  the v3.1 video predictor expects, or our prompt-text → SAM 3.1 text-id
  bridging is misset in `add_prompt(text=...)`. Try via
  `Sam3MultiClassPredictorFast` directly (the same path `sam3_dart`
  uses, which works in production).
- **#4 LS SDK lifecycle**: align `ManualReviewerMLBackend.__init__` with
  the LS SDK contract — override `setup()`/`fit()` so `_manager` marks
  the model loaded. May need to delegate construction inside `setup()`
  rather than `__init__`. Add a live test that POSTs to a real backend
  process (`subprocess.Popen` of `python -m manual_reviewer.ml_backend.server`).

### Live-integration test gap

The 174 manual_reviewer tests all stub the SAM 3.1 HTTP client and the LS
ML SDK. They prove the protocol shapes parse, not that the wire works
end-to-end. The follow-up PR should add at least:

- `tests/integration/test_sam3_1_live.py` — only runs when
  `SAM3_1_URL` is reachable; smoke-tests refine + click_mask + text_detect
  against the real server with assertions on returned bbox ranges.
- `tests/integration/test_ml_backend_live.py` — boots the backend in a
  subprocess, POSTs a real LS-shaped payload, asserts `result[].value`
  shape.

These can be marked `@pytest.mark.integration` and excluded from the
default suite.

### Optional follow-up (already noted)

---

## Phase 2 — Label Studio ML backend

**Goal**: when a reviewer interacts with the LS canvas (clicks, draws a key-point, types a text query into a sidebar field), the LS frontend POSTs to a Python ML-backend service that returns predictions. The backend is a thin protocol adapter — *no model loading lives here*. It calls the existing SAM 3.1 server (port 3014) and reads cached aav4 proposals from `pipeline.db`.

### Files to create

```
manual_reviewer/ml_backend/
├── __init__.py
├── server.py              # LabelStudioMLBase subclass — entry point
├── routes.py              # Mode dispatch: smart_click, smart_text, batch
├── aav4_client.py         # SAM 3.1 HTTP client (reuse Sam3OneHttpClient) + sqlite reader
├── ls_payload.py          # Translate LS task/context dicts ↔ aav4 wire contracts
├── Dockerfile             # Packaging for docker-compose
└── README.md              # Operator notes
```

Tests:

```
manual_reviewer/tests/
├── test_ml_backend_routes.py    # Mode dispatch + LS context parsing
├── test_ml_backend_payload.py   # LS↔aav4 translation
└── test_ml_backend_server.py    # End-to-end with stubbed Sam3Client + sqlite fixture
```

Compose update: extend [docker-compose.review.yml](../docker-compose.review.yml) with an `ml_backend` service.

### Three modes (request shape determines which)

| LS Trigger | Mode | aav4 call | LS response |
|---|---|---|---|
| KeyPoint draw on canvas | `smart_click` | `Sam3OneHttpClient` — currently no click endpoint; **add `click_mask` mode** (point→mask) to [models/sam3_1.py](../../data_miner/auto_annotation_v4/models/sam3_1.py) and [model_servers/sam3_1.py](../../data_miner/auto_annotation_v4/model_servers/sam3_1.py). Wire contract: new `SAM3ClickMaskRequest`/`Response` in `wire.py`. | `RectangleLabels` region (mask's tight bbox; v1 lossy until LS supports masks natively) |
| TextArea submit (smart=true) | `smart_text` | `Sam3OneHttpClient.refine` semantics don't apply — use `text_detect` mode (already implemented). Wire: `DetectorRequest`/`DetectorResponse`. | List of `RectangleLabels` regions |
| Task open | `batch` | **No inference.** Read `proposals` table from pipeline.db keyed on `image_id`. Free perf — task open <500 ms. | List of `RectangleLabels` regions per cached candidate |

### Key design decisions (already made — do not relitigate)

- ML backend is a *protocol adapter*, not a model server. It must not import torch / sam3 / transformers.
- All inference goes through `Sam3OneHttpClient` (port 3014). If LS deploys without SAM 3.1, the smart modes fail; batch mode still works because it's pure DB.
- `pipeline.db` is the source of truth for which image is which. The ML backend gets `data.image_id` from the LS task and looks it up.
- Click→mask returns *just* the bbox in v1. Switching to true mask output requires LS XML config to use `BrushLabels` and is a v2 concern.

### Concrete signatures

```python
# manual_reviewer/ml_backend/server.py
from label_studio_ml.model import LabelStudioMLBase

class ManualReviewerMLBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sam3_client = Sam3OneHttpClient(url=...)
        self.db_path = Path(os.environ["AAV4_PIPELINE_DB"])

    def predict(self, tasks, context=None, **kwargs):
        # Dispatch on context.draft.region type.
        # Return list[dict] in LS prediction format.
```

```python
# manual_reviewer/ml_backend/routes.py
def smart_click(task, context, sam3_client) -> list[dict]: ...
def smart_text(task, context, sam3_client) -> list[dict]: ...
def batch_proposals(task, db_path) -> list[dict]: ...
```

### New aav4 contract needed

```python
# data_miner/auto_annotation_v4/configs/wire.py — append
class SAM3ClickMaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: str
    point: list[float]              # normalized [x, y]
    point_label: int = 1            # 1 = positive
    threshold: float = 0.5

class SAM3ClickMaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox: list[float] | None = None     # normalized [x1, y1, x2, y2]
    mask_rle: dict[str, Any] | None = None
    score: float = 0.0
```

`SAM3OneApi.decode_request` adds a fourth shape: presence of `point` (and absence of `bbox`/`seeds`/`prompts`) → click_mask mode. Underlying call is the same `Sam3VideoPredictor.add_prompt(points=...)`.

### LS configuration

Update [configs/labeling_config.xml](../configs/labeling_config.xml):

- `<KeyPoint name="click" smart="true">` to enable click prompts.
- `<TextArea name="text_query" smart="true">` for text-prompted detection.
- The ML backend URL goes into LS via env var or settings UI when the project is created.

### Verification

1. Bring up the stack: `docker compose -f manual_reviewer/docker-compose.review.yml up`.
2. Open the LS UI, open a task.
3. Click on the canvas — a `RectangleLabels` region appears within ~1s, anchored on the click.
4. Type "forklift" in the text field, submit — boxes appear.
5. Open a fresh task — `data.proposal_summary` is non-empty (batch mode succeeded with no model call).

Estimated: ~400 LoC implementation + ~200 LoC tests. Click→mask wire addition: ~50 LoC across `wire.py` / `models/sam3_1.py` / `model_servers/sam3_1.py`.

---

## Phase 6 — Rex-Omni detector (aav4 only, NOT manual_reviewer)

**Goal**: add Rex-Omni as a third aav4 detector for the auto-pipeline detect stage. Manual review uses SAM 3.1 only — Rex-Omni does not get wired into `manual_reviewer/`.

Rex-Omni is a 3B autoregressive MLLM detector. Latency ~1-3 s per image — too slow for click-driven flows, fine for batch detection.

### Files to create

```
data_miner/auto_annotation_v4/
├── models/rex_omni.py             # BaseDetectorModel impl
└── model_servers/rex_omni.py      # LitAPI wrapper
```

Plus:

- Add `DetectorName.REX_OMNI = "rex_omni"` to [enums.py](../../data_miner/auto_annotation_v4/configs/enums.py).
- Add `rex_omni` block to [servers.yaml](../../data_miner/auto_annotation_v4/configs/servers.yaml). Suggested: port 3015, `enabled: false`, `max_batch_size: 1` (autoregressive).
- Register in [model_servers/serve.py](../../data_miner/auto_annotation_v4/model_servers/serve.py)'s `_get_registry()`.
- Add a default-config entry for Rex-Omni in `default.yaml` if aav4 expects one (see how `sam3_dart` is configured).

Tests: `data_miner/auto_annotation_v4/tests/test_rex_omni.py` — protocol-level only, model load is GPU-bound and goes to user validation.

### Wire contract

Reuse existing `DetectorRequest`/`DetectorResponse`. Rex-Omni is text→detect.

### Implementation notes

- Rex-Omni HF model id: **CONFIRM AT TASK TIME** — search HF for the latest Rex-Omni checkpoint. Last known reference: 3B parameter MLLM.
- Likely uses `transformers.AutoModelForCausalLM` + a custom output decoder that parses MLLM-emitted bbox tokens back into pixel coords.
- Pattern reference: look at how [models/sam3_dart.py](../../data_miner/auto_annotation_v4/models/sam3_dart.py) handles model load / lazy import / threading lock.
- Servers do NOT batch (autoregressive decoding is sequence-length-bound, not batch-bound). `max_batch_size: 1`.

### Verification

1. `python -m data_miner.auto_annotation_v4.model_servers.rex_omni --port 3015 --gpu cuda:0`
2. `curl -X POST :3015/predict -d '{"image_path":"/tmp/x.jpg","prompts":["forklift"],"threshold":null}'`
3. Returns a `DetectorResponse` with boxes/scores/labels.
4. Wire into a small aav4 run with `runtime.detect_models: [rex_omni]` and confirm the `proposals` table fills with `model='rex_omni'` rows.

Estimated: ~250 LoC model + ~150 LoC server + ~150 LoC tests.

---

## Optional follow-up — Tracker-based reconcile upgrade

**Goal**: switch the reconciler from per-missing-frame image-mode `/refine` calls to a single SAM 3.1 video-tracker call per cluster. Win on long sequences (hundreds of frames per cluster); image-mode wins on short clusters.

### Files to modify

```
manual_reviewer/reconcile/propagate.py     # add `tracker` strategy
manual_reviewer/scripts/run_reconcile.py   # add --strategy {image_mode,tracker}
manual_reviewer/tests/test_reconcile.py    # tracker-strategy tests
```

### Approach

`Sam3OneHttpClient.track()` already exists (Phase 7e). For a cluster with K positive frames and N missing frames:

- Build seeds: one `SAM3VideoTrackSeed(obj_id=cluster_idx, frame_index=positive_idx, bbox=...)` per positive frame.
- Resource path: a directory of symlinks to all cluster images named `<frame_index>.jpg`. Need a temp-dir staging step.
- Call `client.track(resource_path=staging_dir, seeds=[...])`.
- Parse response: per-frame outputs that contain a tracked bbox become propagation candidates. Apply same `accept_score` / `seed_iou` thresholds.

### Catches

- Frame-index ordering: SAM 3.1 video predictor expects sequential frames. The reconciler clusters across non-contiguous source frames, so we re-index 0, 1, 2, ... in the staging dir and remember the inverse mapping.
- One-cluster-at-a-time vs all-clusters-in-one-track call: SAM 3.1 supports multi-`obj_id`, so a single track call can carry every cluster simultaneously. Saves session start/teardown overhead.

Estimated: ~200 LoC + ~100 LoC tests. Lower priority than Phase 2/6.

---

## Cross-cutting reminders

- **Auto mode active** — execute autonomously, minimize interruptions.
- **Existing test suite is green**: 124 manual_reviewer tests passing as of this handoff (`.venv/bin/python -m pytest manual_reviewer/tests/ -q`).
- **GPU validation deferred** — none of the model load / inference paths can be validated in CI. Tests stub HTTP clients and predictor classes.
- **No commits without explicit ask** — user manages commits.
- **Read [detectors.md](detectors.md) first** for the detector-policy contract: aav4 detectors stay untouched; manual_reviewer uses SAM 3.1 only.
- **Plan file from earlier context**: `/root/.claude/plans/explore-data-miner-auto-annotation-v4-aa-warm-spring.md` has the broader design rationale; this doc supersedes its phase ordering.

## Suggested execution order in next session

1. Re-run the test suite to confirm baseline green: `.venv/bin/python -m pytest manual_reviewer/tests/ -q`.
2. Phase 2 — ML backend. Includes adding SAM 3.1 click→mask wire/model/server entries (~50 LoC) plus the LS adapter (~400 LoC) plus tests (~200 LoC). Single biggest reviewer-value-per-LoC payoff.
3. Phase 6 — Rex-Omni. Pure aav4, parallelizable with Phase 2 if the user wants both. Web-search Rex-Omni HF model id at start.
4. Tracker reconcile upgrade — only if user prioritizes it; otherwise leave for a future round.
