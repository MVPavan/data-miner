# Smart-tools plan — Nuclio functions wrapping SAM 3.1

The four smart tools we have today on Label Studio
([manual_reviewer/ml_backend/routes.py](../../manual_reviewer/ml_backend/routes.py))
get reimplemented as **CVAT Nuclio functions**. Our existing SAM 3.1
service at `http://<host>:3014/predict` is the model server; the Nuclio
functions are thin HTTP adapters between CVAT's AI Tools protocol and
SAM 3.1's HTTP API.

```
CVAT canvas (browser)
  └─ AI Tools panel  →  POST /api/lambda/functions/<name>/invoke
                         └─ CVAT lambda_manager  →  Nuclio dashboard
                                                     └─ Function pod  →  HTTP →  SAM 3.1 (:3014)
                                                                                     ↓
                                                                                 GPU model
```

Nuclio docs: [docs.cvat.ai/docs/administration/community/advanced/installation_automatic_annotation/](https://docs.cvat.ai/docs/administration/community/advanced/installation_automatic_annotation/),
[docs.cvat.ai/docs/manual/advanced/ai-tools/](https://docs.cvat.ai/docs/manual/advanced/ai-tools/),
[docs.cvat.ai/docs/manual/advanced/serverless-tutorial/](https://docs.cvat.ai/docs/manual/advanced/serverless-tutorial/).

---

## What lands in `manual_reviewer_cvat/serverless/`

```
serverless/
├── README.md                    deploy/test instructions
├── _shared/
│   ├── sam3_client.py           HTTP client for :3014 (lifted from manual_reviewer/reconcile/sam3_client.py)
│   └── ls_payload.py            response builders, lifted from manual_reviewer/ml_backend/ls_payload.py
│                                  rewritten for CVAT response shape (see below)
├── sam3_1_click/                smart_click — interactor with click prompt
│   ├── function.yaml
│   ├── main.py
│   └── Dockerfile
├── sam3_1_text/                 smart_search — detector with text prompt
│   ├── function.yaml
│   ├── main.py
│   └── Dockerfile
├── sam3_1_visual/               smart_visual — interactor with bbox prompt (NET NEW)
│   ├── function.yaml
│   ├── main.py
│   └── Dockerfile
└── sam3_1_track/                smart_track — tracker
    ├── function.yaml
    ├── main.py
    └── Dockerfile
```

`_shared/` content gets baked into each Dockerfile via
`COPY` (Nuclio doesn't have a native shared-code mechanism).

---

## Per-tool spec

### smart_click — Nuclio kind: `interactor`

| | |
|---|---|
| CVAT kind | `interactor` |
| Input | `{image, pos_points: [[x,y],...], neg_points: [[x,y],...], obj_bbox?}` |
| SAM 3.1 call | `Sam3OneHttpClient.click_mask(image, pos_points, neg_points)` |
| Output | `{points: [[x,y],...]}` (polygon — CVAT auto-converts to bbox/mask) |
| Reference | [manual_reviewer/ml_backend/routes.py](../../manual_reviewer/ml_backend/routes.py) `smart_click` handler. |

CVAT supports both polygon-points and RLE-mask returns; polygon is
simpler and round-trips cleanly through CVAT's bbox/polygon/mask
conversion in the editor.

Example reference SAM Nuclio function (CVAT bundles this):
[github.com/cvat-ai/cvat/tree/develop/serverless/pytorch/facebookresearch/sam](https://github.com/cvat-ai/cvat/tree/develop/serverless/pytorch/facebookresearch/sam).
Crib the function.yaml + main.py shape directly.

### smart_search — Nuclio kind: `detector`

| | |
|---|---|
| CVAT kind | `detector` |
| Input | `{image, threshold?, attributes: {prompt: "<class name>"}}` |
| SAM 3.1 call | `Sam3OneHttpClient.text_detect(image, prompt)` |
| Output | `[{label, points: [xtl,ytl,xbr,ybr], confidence, type: "rectangle"}, ...]` |
| Reference | [manual_reviewer/ml_backend/routes.py](../../manual_reviewer/ml_backend/routes.py) `smart_search` handler. |

CVAT's detector functions can declare attributes — the prompt text is
typed into a CVAT modal at run time. Example reference:
[github.com/cvat-ai/cvat/tree/develop/serverless/onnx/WongKinYiu](https://github.com/cvat-ai/cvat/tree/develop/serverless/onnx/WongKinYiu)
(YOLOv7 detector — same shape).

**Open question for Phase 3**: CVAT v2.32+ ships SAM3 with a
"label-as-text-prompt" mode. If that handles our text-detect needs out
of the box, skip `sam3_1_text` entirely and use the bundled function.
Decide after Phase 1 smoke test.

### smart_visual — Nuclio kind: `interactor` with bbox prompt (NET NEW)

| | |
|---|---|
| CVAT kind | `interactor` |
| Input | `{image, obj_bbox: [xtl,ytl,xbr,ybr]}` |
| SAM 3.1 call | `Sam3OneHttpClient.visual_prompt(image, exemplar_bbox)` |
| Output | `[{points: [xtl,ytl,xbr,ybr], confidence}, ...]` (multiple boxes — see below) |
| Reference | [manual_reviewer/ml_backend/routes.py](../../manual_reviewer/ml_backend/routes.py) `smart_visual` handler + dedup logic. |

**Caveat**: CVAT's `interactor` kind is designed to return **one** region
(a refined version of the prompted region). `smart_visual` returns
**many** regions (every instance similar to the exemplar). Two
options:

1. Implement as `detector` instead — but detectors don't take a bbox
   prompt natively; CVAT's detector UI doesn't have a "draw exemplar
   first" affordance.
2. Implement as `interactor` and have the function return a polygon
   collection that CVAT will spawn as multiple shapes — needs testing
   against CVAT's auto-conversion rules.

Likely outcome: extend the detector kind by passing the exemplar bbox
as a JSON-encoded attribute the user fills via "draw bbox first → run
detector". UX papercut — confirm with reviewers in Phase 3 dry-run.

This is the only **net new** smart tool. The other three have CVAT-
native templates to crib from.

### smart_track — Nuclio kind: `tracker`

| | |
|---|---|
| CVAT kind | `tracker` |
| Input | `{image, shape: [xtl,ytl,xbr,ybr], state?: <opaque blob>}` (called per frame) |
| SAM 3.1 call | `Sam3OneHttpClient.track(seed_frame, sibling_frames, seed_bbox)` |
| Output | `{shape: [xtl,ytl,xbr,ybr], state: <opaque blob>}` |
| Reference | [manual_reviewer/ml_backend/smart_track_lib.py](../../manual_reviewer/ml_backend/smart_track_lib.py) — static-object motion-threshold filter. |

CVAT calls trackers **frame-by-frame** with state passed through. Our
existing `smart_track_lib.py` does its own batched cross-frame
propagation in one shot. Two options:

1. Adapt to CVAT's frame-by-frame contract — call SAM 3.1 once per
   frame and stash the model state in CVAT's opaque-blob mechanism.
2. Compare against CVAT's bundled SAM2 Tracker on the same Datatang
   clips — if SAM2 is good enough for static objects, skip writing
   our own tracker function entirely.

Likely outcome: Phase 3 starts by trying SAM2 Tracker on real Datatang
data. Only write a custom function if SAM2 underperforms our SAM 3.1
on the static-camera footage we care about.

The motion-threshold "static-only filter" from `smart_track_lib.py`
(reject if center moves >0.05 normalized OR confidence <0.5) is a
post-hoc filter — could live in the Nuclio function or as a CVAT
"Run annotation action" cleanup pass.

---

## Order of work in Phase 3

1. `smart_click` first — already proven in Phase 2.
2. `smart_search` — try CVAT-native SAM3 text-prompt mode first; only
   write `sam3_1_text` if the native mode doesn't handle our prompt
   set. Skip if not needed.
3. `smart_track` — try CVAT-native SAM2 Tracker first; only write
   `sam3_1_track` if SAM2 underperforms.
4. `smart_visual` last — net-new logic, hardest CVAT integration
   surface, gets the most attention.

**Sequencing rationale**: do the cheap wins first (1-3 may be 0-1 new
functions), then concentrate on the genuinely hard one (`smart_visual`).

---

## Validation per function

For each Nuclio function, before declaring it done:

1. **Unit-test against the SAM 3.1 service directly**: same inputs to
   the Nuclio handler and to a `curl http://<host>:3014/predict` produce
   matching outputs (bbox coordinates within ~1px).
2. **End-to-end test in CVAT canvas**: a real reviewer draws/clicks
   in the CVAT UI, the function fires, the result lands as an editable
   shape on canvas. Latency under 1.5s for `smart_click` (per-click
   budget); under 5s for `smart_search` / `smart_visual`; per-frame
   under 0.5s for `smart_track`.
3. **Cross-check against the LS-side equivalent**: same image, same
   prompt, the LS smart-tool produces the same shape (within tolerance).
   This guards against accidental contract drift between the two
   adapters.
4. **Negative cases**: SAM 3.1 service down → Nuclio function returns
   500 with a useful error in the CVAT toast (not a silent hang).
   Empty prompt / out-of-range coordinates → friendly error, no
   crash.

---

## Things explicitly OUT of scope for these functions

- Authentication between Nuclio and SAM 3.1. Both run on the same host;
  network is loopback.
- Caching. SAM 3.1 already memoizes by image hash internally; the
  Nuclio function is a pass-through.
- Per-user rate limits. CVAT's job assignment provides natural
  serialization at one reviewer per job.
- Model versioning across CVAT releases. CVAT's lambda dashboard
  records the function name and version; bumping our SAM 3.1 weights
  is independent of CVAT.
