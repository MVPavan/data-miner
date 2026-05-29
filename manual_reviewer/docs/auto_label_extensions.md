# Auto-labeling extensions plan

Implementation reference. Decisions made 2026-04-28; per-phase detail below.

| Phase | Decision | Status |
|---|---|---|
| A. smart_click class fix | Shared `<Labels>` block drives `<Rectangle>` + `<KeyPoint smart>`. Reviewer's hotkey class rides on `value.labels`. Negative clicks dropped v1. | decided |
| B. Within-image visual prompting | SAM 3.1 `add_geometric_prompt`, single-exemplar v1, hotkey trigger (`Shift+F`). Rex-Omni opt-in second pass. | decided |
| C. Cross-frame propagation | **v1 static only.** CosineGenerator → confirm/conflict/suggest reconciler against accepted annotations, written to LS `predictions[]`. Moving deferred. | decided |
| D. Master toggles + layout | Per-route env-var gates. Layout-2: drop right sidebar, image full-width, `frame_state` + notes in a footer row. | decided |

**Sequencing.** A → D-layout → B → C → D-toggles. A and D-layout are XML-only and unblock daily use immediately. Each phase ships with an env-gated live-integration test alongside the stubbed unit tests.

## Today's working set (live)

| # | Trigger | Route | Model |
|---|---|---|---|
| 1 | Task open | seeded `predictions[]` from `pipeline.db` | DB read |
| 2 | Task open via `/predict` (no context) | `batch_proposals` | DB read |
| 3 | KeyPoint draw | `smart_click` | SAM 3.1 `Sam3Image.predict_inst` |
| 4 | TextArea submit | `smart_search` | SAM 3.1 text grounding |
| 5 | smart-Rectangle (`from_name="smart_visual"`) | `smart_visual` | SAM 3.1 visual prompting |
| 6 | smart-Rectangle (`from_name="smart_track"`) | `smart_track` | SAM 3.1 video tracker |

---

## Phase A — smart_click class preservation

**Symptom.** Reviewer presses `8` (selects `bicycle`), places a positive keypoint click, bbox comes back labeled `other`.

**Root cause.** [routes.py:99-122](../ml_backend/routes.py#L99) reads `rectanglelabels` / `labels` from `context.result[*].value`, but the keypoint draft only carries `keypointlabels=["positive"]`. LS doesn't propagate the toolbar's active class into smart-tool drafts — only what was *drawn*.

**Fix.** Restructure [labeling_config.xml:22-55](../configs/labeling_config.xml#L22) so a single `<Labels>` block drives both rectangle and keypoint tools (LS supports this when both share `toName`):

```xml
<Labels name="label" toName="image">
  <Label value="forklift" background="#e74c3c"/>
  ... 24 classes, defined exactly once ...
</Labels>
<Rectangle name="rect" toName="image"/>
<KeyPoint name="kp" toName="image" smart="true" strokeWidth="3"/>
```

The reviewer's active class rides every draft as `value.labels`. `_picked_label_from_context` already accepts `labels` — only Python change is removing the now-defunct `keypointlabels` parsing.

**Negative clicks dropped v1.** Single positive click + manual bbox adjust covers the failure path. If revisited later: sibling `<Choices>` polarity toggle, or second smart keypoint tool.

**Verify.** Place positive keypoint with `bicycle` selected; the smart_click trace must show `label=bicycle` and the returned region's `value.labels` must be `["bicycle"]`.

**Open.** Does LS render the smart-tool draft with the active class color, or stay neutral until accepted? Observe live; tweak if cosmetic.

---

## Phase B — Within-image visual prompting

**Goal.** Reviewer draws one bbox, hotkey `Shift+F`, backend returns every matching instance in the same image as new RectangleLabels regions. Manual trigger only.

**Model: SAM 3.1.** `Sam3Processor.add_geometric_prompt(box, label=True, state)` ([sam3_image_processor.py:127-152](../../scratchpad/DART/sam3/model/sam3_image_processor.py#L127)) feeds the bbox into the same grounding head as text prompts; returns all boxes/masks/scores above `confidence_threshold`. ROI-aligns backbone features inside the prompt box via `SequenceGeometryEncoder` ([geometry_encoders.py:632-665](../../scratchpad/DART/sam3/model/geometry_encoders.py#L632)). Multi-exemplar + negative boxes natively supported. Free segmentation. HF API: `Sam3Processor(images=img, input_boxes=[[box]], input_boxes_labels=[[1]], ...)` → `Sam3Model(...)` → `post_process_instance_segmentation`.

**Rex-Omni: opt-in fallback.** `task="visual_prompting"` with `visual_prompt_boxes` + `categories`. Box-only (no masks), 1-3s autoregressive MLLM. Better recall on rare/fine-grained classes; SAM 3 wins on common categories. [models/rex_omni.py](../../data_miner/auto_annotation_v4/models/rex_omni.py) hardcodes `_DEFAULT_TASK = "detection"` — small extension to thread `visual_prompting` through.

### Wire shape

New contracts in [configs/wire.py](../../data_miner/auto_annotation_v4/configs/wire.py):

```python
class SAM3VisualPromptRequest(BaseModel):
    image_path: str
    exemplar_boxes_norm: list[list[float]]   # [[x1, y1, x2, y2], ...] in [0, 1]
    exemplar_labels: list[int] = []           # 1=positive, 0=negative; default all 1
    threshold: float = 0.4
    max_results: int = 50

class SAM3VisualPromptResponse(BaseModel):
    boxes_norm: list[list[float]] = []
    scores: list[float] = []
    mask_rles: list[dict] | None = None
```

Mirror response shape for Rex-Omni so the route can swap models behind a flag.

### LS UI

Reviewer selects a region on canvas, presses `Shift+F`, LS fires `/predict`. Backend sees a context with one rectangle region and a meta hint `{trigger: "visual_prompt"}` set via a hidden `<Choices>`. Investigate exact LS dispatch mechanism — likely `smart="true"` on the rectangle tool itself or a custom `<Action>` element.

*Rejected: B1 sidebar button (LS button UX awkward); B3 TextArea convention (brittle, undiscoverable).*

**Tuning knobs.** Threshold 0.4, `max_results` 50. Single-exemplar v1; multi-exemplar + negative exemplars deferred. Cap output at 50 boxes; surface a warning if hit.

---

## Phase C — Static-object propagation (v1; moving deferred)

**Scope.** v1 ships static only — camera-fixed / stationary instances (parked bicycle, fixed container). Reviewer seeds one bbox, the same identity is reconciled against every other survivor frame.

Moving deferred — sparse deduped frames break the tracker's motion model and force an appearance-ReID design with real lookalike risk. Not worth shipping until the static path is proven.

### Generator: per-frame propagated bbox

`Generator` interface returns `{frame_id: (bbox_norm, score)}` for every frame in scope. Two implementations:

- **CosineGenerator** (ship first). Crop seed bbox, encode (DINOv3), cache. For each scope frame: crop at the **same normalized coords** → encode → cosine vs seed. Keep frames where cosine ≥ 0.85. ~10ms/frame, ~500ms for 50 frames. No model server call.
- **TrackerGenerator** (v2). SAM 3.1 video tracker over the survivor folder. Better masks + bbox refinement + partial-occlusion handling. ~2-5s for 50 frames.

### Reconciler: confirm / conflict / suggest

Per scope frame, fetch existing **accepted** annotations via `/api/tasks/{id}/annotations`. Pick the existing bbox with highest IoU vs. the generator's propagated bbox. Branch:

| Generator | Existing E | Outcome | LS write |
|---|---|---|---|
| match | none | **suggest** | new prediction, `meta.source="propagated_static"` |
| match | IoU ≥ 0.7 | **confirm** | silent — annotate E with `meta.confirmed_by="propagation"` |
| match | 0.4 ≤ IoU < 0.7 | **weak-iou** | silent confirm with a soft flag |
| match | IoU < 0.4 | **conflict** | new prediction + `frame_state="needs_more_review"` |
| no match | any | **skip** | nothing |

Only **reviewer-accepted** annotations count as "existing". Cached `proposals` rows are too noisy to treat as ground truth — would generate spurious conflicts.

### Failure modes

| Scenario | Generator | Existing | Outcome | OK? |
|---|---|---|---|---|
| Object still there, untouched | match | none | suggest | ✓ |
| Object still there, prior agrees | match | IoU ≥ 0.7 | confirm | ✓ |
| Object still there, prior wrong place | match | IoU < 0.4 | conflict | ✓ |
| Object removed, untouched | low | none | skip | ✓ |
| Object removed, stale prior | low | exists | skip (we don't delete) | ✓ |
| Different instance at same spot | moderate | none | suggest, low score | ⚠ reviewer prunes |
| Camera drift ~10 px | match | drifted prior | confirm or weak-iou | ✓ |
| Brief person occlusion | borderline | none | skip or low-score suggest | ⚠ acceptable |

### Wire shape + scope

```python
def propagate_static(
    seed: Seed,                  # (image_id, bbox_norm, class_name)
    *,
    scope_image_ids: list[str],
    cosine_thresh: float = 0.85,
    iou_confirm: float = 0.7,
    iou_conflict: float = 0.4,
    ls_client: LSRestClient,
) -> Summary: ...
```

Scope: same `dedup_cluster_id` if it exists, else K=50 nearest survivors by DINOv3 frame embedding. Cap at 500 frames; reject larger calls with a clear error.

`Summary = {confirmed, suggested, conflicts, weak_iou, skipped}`. Written to the seed task's notes:
```
Propagated bicycle: 38 confirmed | 7 suggested | 3 conflicts | 1 weak-iou
```

### LS UI + execution

- Hotkey `Shift+P` on a selected region triggers propagation.
- Sidebar `<Choices>` `propagate_mode` (`auto` default; `static` / `disabled` overrides). v1 only honors `static` and `disabled`; `auto` acts as `static`. Hook for moving once that ships.
- Synchronous when scope ≤ 50 frames; otherwise return immediately, run on a `ThreadPoolExecutor` in `server.py`, write results as they land.

### Output destination

Predictions land on each target frame's LS task `predictions[]` only — no new pipeline.db rows. Reviewer's per-frame accept goes through the normal `export_to_aa_v4.py` flow as `Stage.HUMAN_REVIEW`. `stage="propagate"` persistence deferred to v2.

### Open

1. **Conflict UX.** Auto-set `frame_state="needs_more_review"` on conflicted tasks, or just emit the side-by-side bboxes? Lean both — watch in real use.
2. **Score visibility.** Confirm LS exposes `meta.cosine` in the per-region inspector; fall back to prepending the score to the class label string if not.
3. **Sharing with reconcile/.** [reconcile/iou_reconcile.py](../reconcile/iou_reconcile.py) does the same IoU + appearance pattern as a batch job. Factor out shared `_iou_match` / `_cosine_rank` helpers before they drift.

---

## Phase D — Master toggles + UI layout

### Env-var route gates

| Env var | Gates |
|---|---|
| `ENABLE_BATCH_PROPOSALS` | Task-open `batch_proposals` |
| `ENABLE_SMART_CLICK`     | KeyPoint → SAM 3.1 `click_mask` |
| `ENABLE_SMART_SEARCH`    | TextArea → SAM 3.1 `text_detect` |
| `ENABLE_SMART_VISUAL`    | smart-Rectangle (`from_name="smart_visual"`) → SAM 3.1 `visual_prompt` |
| `ENABLE_SMART_TRACK`     | smart-Rectangle (`from_name="smart_track"`) → SAM 3.1 `track` |

Default all `true`. [server.py:_predict_one](../ml_backend/server.py) gets a `_route_enabled(name)` helper checked before each dispatch; disabled routes log `route=X disabled by env` and return no regions. Restart-to-toggle is fine for v1.

*Rejected: D2 in-canvas checkboxes (sidebar gets crowded); D3 runtime config endpoint (premature for one reviewer).*

### Layout-2 (footer row, full-width image)

[labeling_config.xml:17-82](../configs/labeling_config.xml#L17) currently splits 70/30 with a mostly-empty right sidebar. Changes:

1. Drop the outer `<View style="display: flex;">` two-column wrapper.
2. Image, shared `<Labels>` palette (post-A), keypoint smart tool, text-prompt smart input all stack vertically.
3. `frame_state` (clean / needs_more_review / ambiguous_skip) → horizontal `<Choices>` strip below the image.
4. `<TextArea name="notes">` below the frame_state row.
5. `<Image maxHeight="78vh" ...>` so the canvas fills the viewport.

*Rejected: Layout-1 (just shrink sidebar — half measure); Layout-3 (CSS grid + collapsible — overkill).*

**Open.** Hide LS's `Auto-Accept Suggestions` bottom-bar toggle via Custom CSS? Lean yes — reviewers should accept consciously.

---

## Cross-cutting

### Latency targets

| Action | Today | Target |
|---|---|---|
| Task open (cached proposals) | ~10ms | <100ms |
| Smart click | ~250ms | <500ms |
| Smart text | ~700ms | <1.5s |
| Visual prompt (Phase B) | TBD | <1.5s |
| Static propagate (Phase C, 50-frame bag) | TBD | <2s sync; >50 async |

### Test strategy

- New wire contracts: unit tests with stubbed HTTP.
- New routes: unit tests with `_StubSam3Client` ([tests/test_ml_backend.py](../tests/test_ml_backend.py)).
- One env-gated live-integration test per phase (`pytest -m integration`) hitting a real LitServe — catches the wire-bug class that bit us in the original rollout (pixel xyxy vs normalized xywh).
