# aa_v4 Manual Review System

Human-in-the-loop layer that sits after `auto_annotation_v4` and corrects
the residue. Team-scale annotation with model assist, customizable
workflow, multiple input modes.

---

## 1. Stack

| Component | Choice |
|---|---|
| Annotation tool | Label Studio (Postgres backend) |
| Primary detection + segmentation + tracking | SAM 3.1 (`facebook/sam3.1`) |
| Alternative detection + visual exemplar | Rex-Omni (`IDEA-Research/Rex-Omni-AWQ`) — optional |
| Frame similarity | DINOv3 (already cached as `.npy`) |
| Model serving | LitServe (existing pattern) |
| ML protocol adapters | Label Studio ML Backend SDK |

SAM 3.1 covers every model role on its own: text → all-instance
bboxes/masks (open-vocab detection), point/box → mask (interactive),
exemplar box → all similar in same image, multi-object video tracking.

Rex-Omni is a parallel detector for projects where recall matters more
than latency. SAM 3.1 (DETR-based) and Rex-Omni (autoregressive MLLM)
have uncorrelated failure modes — running both and reconciling lifts
recall on dense or hard scenes. Selectable per project.

Not used: SAM 2 (subsumed by SAM 3.1), SigLIP2 (overlaps SAM 3.1's text
prompts), Nuclio, FiftyOne.

---

## 2. Pipeline order

Video projects:

```
Raw video / frames_raw/
        │
        ▼   scene cut (PySceneDetect)
   segments
        │
        ▼   SAM 3.1 video tracker stage (per segment)
   per-frame: {track_id, class, bbox, mask}
        │
        ▼   dedup (DINOv3 + FAISS) — carries annotations & track_ids forward
   dedup'd frames + labels + track_ids
        │
        ▼   build_tasks.py (sort + import)
   Label Studio project
```

Image / non-video projects:

```
aa_v4 outputs (labels/, traces/, review/)
        │
        ▼   group by DINOv3 similarity cluster
   scene clusters
        │
        ▼   cross-frame accumulation (§4)
   reconciled detections per cluster
        │
        ▼   build_tasks.py
   Label Studio project
```

Hard rule: video tracking only runs on raw frames. Never on dedup'd
frames — SAM 3.1's memory module needs dense temporal context.

---

## 3. SAM 3.1 video tracker stage

A naive forward pass from frame 0 misses any object that enters
mid-segment. Three layers, in this order:

**Layer 1 — initial seed via text prompts.** SAM 3.1 with the 24-class
list as text prompts on the segment's first frame. Returns all instances
with bboxes + class labels in one call. Optionally also run Rex-Omni
detection and merge by IoU for higher recall on the seed frame.

**Layer 2 — periodic redetection.** Every K frames (K=20 default),
re-run SAM 3.1 detection on that frame with the class list. Compare
against current tracker state by IoU. Anything new gets added as a new
track from the current frame.

**Layer 3 — reverse propagation on new tracks.** When Layer 2 adds a
track at frame K, run SAM 3.1's `propagate_in_video(reverse=True)`
backward from K with early termination on mask-quality collapse. Fills
missed detections; correctly stops at the actual entry frame.

Optional Layer 4 — re-entry track linking. Same physical object across
an occlusion gets two track IDs. ReID step (DINOv3 appearance embedding +
Hungarian matching). Skip until reviewer feedback proves it's needed.

```python
# scripts/review/propagate_segment.py — sketch
def propagate(state, frames, class_names, redetect_every=20):
    seed_dets = sam3_detect(frames[0], class_names)
    # Optional: merge with rex_omni_detect(frames[0], class_names)
    for det in seed_dets:
        predictor.add_new_points_or_box(state, frame_idx=0,
                                        obj_id=det.id, box=det.bbox,
                                        class_name=det.class_name)

    for fi, ids, masks in predictor.propagate_in_video(state):
        if fi % redetect_every == 0 and fi > 0:
            current = sam3_detect(frames[fi], class_names)
            new_objs = filter_by_iou(current, masks_to_boxes(masks), 0.3)
            for det in new_objs:
                new_id = mint_id()
                predictor.add_new_points_or_box(state, frame_idx=fi,
                                                obj_id=new_id, box=det.bbox)
                for pi, _, pm in predictor.propagate_in_video(
                        state, start_frame_idx=fi, reverse=True, max_frames=fi):
                    if mask_collapsed(pm) or iou_dropped(pm):
                        break
                    record(pi, pm, track_id=new_id, class_name=det.class_name)
```

Output: `tracks.json` — per-frame `{track_id, class, bbox, mask_rle}`.

---

## 4. Cross-frame detection accumulation (image-only projects)

A single model on a single frame misses instances. CCTV with 10 parked
bicycles → one inference might find 6, another frame might find 8 with
2 different ones. Reconcile across frames in the same scene to get a
robust union.

For video projects this is handled natively by §3's tracker stage.
For image-only projects (or dedup'd frame batches without tracking
history), explicit reconciliation is needed.

**Algorithm.** Group frames into scene clusters via DINOv3 similarity.
Within a cluster, run model(s) on each frame, spatially cluster the
union of detections by IoU, keep canonical box per cluster.

```python
# scripts/review/reconcile_scene.py — sketch
def reconcile(frames, models, iou_threshold=0.5, min_votes=1):
    """Run model(s) on every frame; reconcile overlapping detections.

    frames: list of frames in same scene cluster (DINOv3 similar)
    models: list of detector callables (SAM 3.1 alone, or +Rex-Omni)
    min_votes: minimum (frame, model) votes required to accept a detection
    """
    all_dets = []
    for frame in frames:
        for model in models:
            for det in model(frame, class_names):
                all_dets.append({**det, "frame": frame.id, "model": model.name})

    clusters = greedy_iou_cluster(all_dets, iou_threshold)
    canonical = []
    for cluster in clusters:
        if len(cluster) < min_votes:
            continue
        # Pick highest-confidence; or compute weighted-average box
        best = max(cluster, key=lambda d: d["confidence"])
        best["votes"] = len(cluster)
        best["seen_by"] = sorted({d["model"] for d in cluster})
        canonical.append(best)
    return canonical
```

**Two axes of robustness:**
- *Cross-frame*: multiple frames of the same static scene → catches what
  any single frame missed. Most useful for fixed cameras (CCTV) where
  pixel coordinates align directly.
- *Cross-model*: SAM 3.1 + Rex-Omni on each frame → catches what either
  architecture alone would miss. Different model families fail
  differently.

Both compose. Output: each frame in the cluster receives the same
reconciled detection set as pre-annotations (clipped to that frame's
visible region; for fixed cameras the regions are identical).

**Tuning parameters (per project):**
- `iou_threshold` — 0.5 default; 0.3 for sparse scenes, 0.7 for dense.
- `min_votes` — 1 = trust every detection (max recall, more false
  positives); 2-3 = robust to single-frame/single-model noise.
- Cluster size — too small = no reconciliation gain; too large = mixes
  unrelated scenes. DINOv3 cosine ≥ 0.85 is a reasonable default.

**Moving cameras.** For non-fixed cameras, pixel coordinates don't align
across frames. Either: (a) skip cross-frame accumulation for that
project, (b) compute a homography between adjacent frames using DINOv3
patch features and align before clustering. Option (b) is real
engineering; defer until needed.

---

## 5. Dedup with annotation carry-forward

Existing dedup picks one survivor per cluster. Patch:

- For each surviving frame, copy its YOLO label file (with track_ids as
  attributes) and its mask sidecar to the dedup output dir.
- Cluster-level merge: if a cluster contains the same track_id across
  multiple frames, the survivor inherits the union of the cluster's
  annotations for that track (rare with good dedup; safety net).

~30 LoC change in `Deduplicator.deduplicate()`.

YOLO line format extension: `class_id cx cy w h track_id` (track_id as a
6th column). Or a sidecar `{frame_stem}.tracks.json` if extending the
YOLO format breaks downstream tooling.

---

## 6. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Label Studio (Postgres, Django, Vue UI)                │
│  - Project: aa_v4_review                                │
│  - XML labeling interface                               │
│  - 1 ML backend connected via URL                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼ HTTP :9091
            ┌──────────────────────────┐
            │  ML Backend: detector    │
            │  (LabelStudioMLBase)     │
            │  - Routes to SAM 3.1 or  │
            │    Rex-Omni per config   │
            │  - 4 modes (§8)          │
            └──────────┬───────────────┘
                       │
        ┌──────────────┴──────────────┐
        │ HTTP :3003                  │ HTTP :3007
        ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  LitServe: serve_sam3.py │  │  LitServe:               │
│  (image + all modes)     │  │  serve_rex_omni.py       │
│                          │  │  (optional)              │
└──────────────────────────┘  └──────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  LitServe: serve_sam3_video.py  (video tracker stage)    │
│  Called by propagate_segment.py — not by Label Studio    │
└──────────────────────────────────────────────────────────┘
```

ML backend is a protocol adapter only. Models load and live in LitServe.
Backend restarts are millisecond-fast (no model reload). Same model
servers feed auto-pipeline, review, cross-frame reconciliation, and the
video tracker stage.

---

## 7. Label Studio labeling interface

```xml
<View>
  <Header value="aa_v4 Review"/>
  <View style="display: flex;">
    <View style="flex: 0 0 70%;">
      <Image name="image" value="$image" zoom="true" zoomControl="true"/>

      <RectangleLabels name="bbox" toName="image" smart="true">
        <Label value="forklift"   background="#e74c3c"/>
        <Label value="palletjack" background="#3498db"/>
        <Label value="person"     background="#2ecc71"/>
        <!-- … 24 classes … -->
      </RectangleLabels>

      <KeyPointLabels name="kp" toName="image" smart="true">
        <Label value="positive" background="#27ae60"/>
        <Label value="negative" background="#c0392b"/>
      </KeyPointLabels>

      <BrushLabels name="mask" toName="image">
        <!-- same 24 classes for mask output -->
      </BrushLabels>

      <TextArea name="track_id" toName="image" perRegion="true"
                placeholder="track_id" editable="true"/>
    </View>

    <View style="flex: 0 0 30%; padding-left: 10px;">
      <TextArea name="text_prompt" toName="image"
                placeholder="e.g. 'forklift'" rows="2"/>

      <Choices name="frame_state" toName="image" choice="single">
        <Choice value="clean"/>
        <Choice value="needs_more_review"/>
        <Choice value="ambiguous_skip"/>
      </Choices>

      <TextArea name="notes" toName="image" rows="3"/>
    </View>
  </View>
</View>
```

`smart="true"` triggers the ML backend on draw/click. `perRegion="true"`
makes `track_id` an attribute per detection; auto-filled at import,
editable when reviewer needs to merge tracks.

---

## 8. ML backend — detector with model choice

```python
# label_studio_ml/detector_backend.py
class DetectorBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = self.get_param("detector", "sam3")  # "sam3" | "rex_omni" | "both"

    def predict(self, tasks, context=None, **kwargs):
        if context and self._has_keypoints(context):
            return self._mode_click_to_mask(tasks, context)  # SAM 3.1 only
        if context and self._has_rectangle(context):
            return self._mode_exemplar(tasks, context)       # SAM 3.1 or Rex-Omni
        if context and self._has_text(context):
            return self._mode_text_detect(tasks, context)    # SAM 3.1 or Rex-Omni or both
        return self._mode_batch_detect(tasks)                # SAM 3.1 or Rex-Omni or both
```

Four entry points; mode choice respects the project-level `detector`
config (`sam3`, `rex_omni`, or `both` for ensemble).

- *Batch detection* on task open — text prompts with the 24-class list.
  If `both`, run each model and reconcile by IoU before returning.
- *Text detect* on text-field submit — single-class refinement (e.g.,
  reviewer types `"bicycle"` to find any missed bicycles).
- *Click → mask* on KeyPoint draw — SAM 3.1 only (Rex-Omni doesn't
  produce masks).
- *Exemplar* on Rectangle draw — drawn box used as a positive exemplar;
  returns all similar instances in the same image.

Backend code is ~350 LoC including the ensemble reconciliation path.

---

## 9. Sort & batch

`build_tasks.py` reads aa_v4 outputs + DINOv3 cache + traces, applies
one of three sort modes, optionally runs cross-frame reconciliation
(§4), pushes to Label Studio with pre-annotations:

| Mode | Sort key | Use |
|---|---|---|
| Similarity | DINOv3 cosine to seed | Replicate fixes within a visual cluster; enables §4 reconciliation |
| Uncertainty | min(VLM confidence) ASC, agreeing_models ASC | High-impact errors first |
| Video segment | (video_id, frame_number) | Reviewing a specific video's tracks together |

For tracked data: similarity sort keeps tracks visually grouped;
uncertainty sort surfaces the tracker's weakest frames first.

For image-only data with similarity sort: §4 reconciliation runs
automatically per cluster, pre-annotations are the reconciled union.

Bulk-edit pattern in Label Studio: filter by `track_id == X` (or by
cluster_id for image-only), multi-select detections, change class —
applies to all members of that group in one action.

---

## 10. Files

| File | Purpose | LoC |
|---|---|---|
| `scripts/review/propagate_segment.py` | SAM 3.1 video tracker stage with periodic redetection + reverse fill | ~400 |
| `scripts/review/scene_cut.py` | PySceneDetect wrapper, segment boundaries | ~80 |
| `scripts/review/reconcile_scene.py` | Cross-frame & cross-model detection reconciliation | ~200 |
| `scripts/review/build_tasks.py` | aa_v4 → Label Studio tasks JSON, sort + reconciliation + pre-annotation | ~300 |
| `scripts/review/export_to_aa_v4.py` | Label Studio export → aa_v4 dir, append `human_review` to traces | ~150 |
| `scripts/review/labeling_config.xml` | XML interface from §7 | ~60 |
| `label_studio_ml/detector_backend.py` | Detector protocol adapter (4 modes, model routing) | ~350 |
| `auto_annotation_v4/model_servers/serve_sam3.py` | Existing — update model loader to `facebook/sam3.1`; add detection mode | +100 |
| `auto_annotation_v4/model_servers/serve_sam3_video.py` | LitServe SAM 3.1 video predictor (stateful per session) | ~250 |
| `auto_annotation_v4/model_servers/serve_rex_omni.py` | LitServe Rex-Omni server (optional alternative detector) | ~150 |
| `auto_annotation_v4/models/rex_omni.py` | Rex-Omni adapter (prepare/infer/postprocess) | ~200 |
| `data_miner/modules/deduplicator.py` | Patch: carry annotations + track_ids forward | +30 |
| `docker-compose.review.yml` | Label Studio + Postgres + 1 ML backend (+ optional Rex-Omni) | ~60 |

Total: ~2100 LoC. Bulk is mechanical (LitServe wrappers, JSON shape
adapters). Rex-Omni files (~350 LoC) are skippable if you don't need the
ensemble path.

---

## 11. Phased rollout

1. **Walking skeleton (2-3 days).** Label Studio + Postgres up. XML
   config. `build_tasks.py` imports 100 frames with aa_v4
   pre-annotations. No ML backend, no reconciliation. Round-trip via
   `export_to_aa_v4.py`. Validate.

2. **SAM 3.1 ML backend, all four modes (4-5 days).** Update
   `serve_sam3.py` to load `facebook/sam3.1` and add detection mode.
   Verify wire schema. Write `detector_backend.py` (sam3 mode only)
   covering click → mask, text → bboxes, exemplar → similar, batch
   detection.

3. **Cross-frame reconciliation (2-3 days).** Build
   `reconcile_scene.py` with IoU clustering + min_votes. Wire into
   `build_tasks.py` similarity-sort path. Validate on a CCTV-style
   batch where ground truth is known.

4. **Rex-Omni alternative (3-4 days, optional).** Build
   `serve_rex_omni.py` and `models/rex_omni.py`. Extend
   `detector_backend.py` to route to Rex-Omni or run ensemble.
   Bake-off vs SAM 3.1 alone on a representative batch — if recall
   gain doesn't justify latency cost, keep Rex-Omni out of default
   path.

5. **Video tracker stage (1 week).** Build `serve_sam3_video.py` and
   `propagate_segment.py` with all three layers (seed via SAM 3.1
   detection, periodic redetect, reverse fill). Tune termination
   heuristics on 10 hand-labeled segments. Patch dedup to carry
   track_ids.

6. **Workflow customization (only if reviewer feedback says so).**
   Webhooks for state-flagged tasks. Active learning loop. Custom
   plugin for cross-frame replication if manual flow is too slow.

Phases 1-2 are usable on their own. Phase 3 dramatically improves
image-only project quality. Phase 4 is opt-in. Phase 5 only needed for
video projects.

---

## 12. Gotchas

**Reverse propagation is the silent killer.** Lenient termination =
phantom annotations on frames before the object existed. Strict = miss
real detections. Tune on a hand-labeled validation set before production
runs. Compare reverse-fill against ground truth on 10 segments minimum.

**Class consistency under reverse propagation.** SAM 3.1 tracks the
mask, not the class. The seeded label is propagated backward; if the
spatial location actually contained something else earlier, the label
is wrong. Mitigation: re-run SAM 3.1 detection on the earliest
reverse-fill frame as a class sanity check.

**Cross-frame reconciliation needs aligned coordinates.** Works
out-of-the-box for fixed cameras (CCTV); needs homography or feature
alignment for moving cameras. Don't enable similarity-cluster
reconciliation on moving-camera projects without an alignment step or
you'll get duplicate detections from drift.

**`min_votes > 1` reduces recall.** Setting `min_votes=2` filters
single-frame detections — including legitimate ones that only appear in
one frame due to brief occlusion or lighting. Use `min_votes=1` by
default; raise only when single-model false positives become a problem.

**Memory cost on long uncut segments.** SAM 3.1's memory grows per
frame. Cap segment length at 100-150 frames; let scene-cut splits handle
most of this naturally. For long uncut shots (static cameras), force-
split at the cap.

**Exemplar prompts in Label Studio are same-image only.** SAM 3.1's
public HF API takes exemplar boxes as coordinates within the target
image. Cross-image exemplars (corrected box from frame A applied to
frame B) require either a different API path or a custom workflow. For
24 known classes, text prompts make this moot.

**Re-entry tracks get separate IDs.** Same physical object across an
occlusion produces two track IDs. Reviewer manually merges via the
`track_id` attribute. Build automated ReID only if reviewer time data
shows it's needed.

**Tracker misses become reviewer adds.** When tracker drops a track,
reviewer's manually-added box gets a fresh track_id, not the original.
Bulk-edit on track_id in Label Studio is the manual fix; document in
SOP.

**Rex-Omni latency.** 3B autoregressive MLLM, generation-bound. Per-call
latency is 1-3s depending on hardware vs SAM 3.1's ~200ms. Acceptable
for batch reconciliation; painful for interactive click-driven flows.
Don't route click → mask through Rex-Omni.

**ML backend stays a protocol adapter.** No model loading inside it.
Models live in LitServe with their existing lifecycle. Backend restart =
milliseconds, not minutes.

**Trace round-trip is mandatory.** `export_to_aa_v4.py` reads original
trace, appends `human_review` stage with reviewer ID, timestamp, ML
backend modes consulted, reconciliation parameters. Don't drop the
audit trail.

**Postgres from day one.** SQLite default collapses past ~10k tasks.

---

## 13. Open questions

- SAM 3.1 wire-format compatibility with existing `serve_sam3.py`. The
  3.1 model card claims backward-compatible inputs but adds Object
  Multiplex outputs for multi-object tracking. Smoke-test before
  assuming.
- SAM 3.1 batch detection latency with all 24 classes per image.
  Measure on existing GPU; if > 2s/image at task-open, consider running
  it only on uncertainty-sorted top-N tasks.
- SAM 3.1 small-class recall (wallet, head, cellphone). Bake-off vs
  existing aa_v4 ensemble. If SAM 3.1 underperforms, keep ensemble for
  small-class detection in the auto-pipeline; SAM 3.1 still owns the
  review-path interactive role.
- Rex-Omni recall lift on hard scenes vs latency cost. Does running
  ensemble (SAM 3.1 + Rex-Omni) actually catch enough additional
  instances to justify the 3-5× inference time?
- Optimal `iou_threshold` and `min_votes` for §4 reconciliation across
  project types (CCTV vs handheld vs drone).
- Termination thresholds for reverse propagation in §3. Tune on
  validation set.
- YOLO format extension vs sidecar JSON for track_ids and reconciliation
  metadata.
- Cross-image exemplar API path in `facebookresearch/sam3` repo, in
  case reviewer workflow ends up wanting "use frame A's correction as
  exemplar for frame B." Architecturally supported; public API path
  uncertain.
