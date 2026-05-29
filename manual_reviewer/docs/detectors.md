# Detectors used by manual_reviewer

Manual review uses **SAM 3.1 only** for all model-driven tasks:

| Task | Endpoint | Wire contract | Used by |
|---|---|---|---|
| bbox refinement | `sam3_1` `/predict` (refine mode) | `SAM3RefineRequest` / `SAM3RefineResponse` | reconciler (`run_reconcile.py`) |
| text → detect | `sam3_1` `/predict` (text mode) | `DetectorRequest` / `DetectorResponse` | ML backend (Phase 2) |
| video tracker | `sam3_1` `/predict` (track mode) | `SAM3VideoTrackRequest` / `SAM3VideoTrackResponse` | tracker-based reconciler upgrade |
| click → mask | (Phase 2 — not yet wired) | (Phase 2 — wire TBD) | ML backend |

Server: `data_miner.auto_annotation_v4.model_servers.sam3_1.SAM3OneApi`,
default port `3014`. Mode dispatch is by request shape (presence of
`seeds`, `bbox`, or `prompts`); single LitServe endpoint, single GPU
worker (`max_batch_size=1`) because SAM 3.1 sessions are stateful.

## Detectors NOT used by manual_reviewer

The aav4 auto-pipeline keeps these detectors for historical reasons. None
of them are reachable from any `manual_reviewer/` code path:

- **`grounding_dino`** — aav4 detect stage; replaced by SAM 3.1 in manual
  review. Do not call it from manual_reviewer.
- **`sam3_dart`** — aav4 refine stage. Manual review used to call it for
  bbox refinement (port 3013); now superseded by SAM 3.1 (port 3014).
  Kept available behind `run_reconcile.py --backend sam3_dart` as a
  fallback for environments where SAM 3.1 isn't deployed yet.
- **`falcon`**, **`owlvit2`**, **`omdet_turbo`** — aav4-only detectors.

## Why the split

aav4 was tuned and validated against `grounding_dino` + `sam3_dart`. Manual
review is a separate workflow with different latency / interactivity
constraints, and standardising on SAM 3.1 there gives:

- one model that covers refine + text-detect + click-mask + tracker (no
  multi-model orchestration in the LS ML backend);
- native video-tracker support, which sam3_dart doesn't have;
- alignment with the sam3 upstream API (Meta's `Sam3VideoPredictor`).

Modifying aav4 to switch detectors would invalidate its existing recall /
latency baselines, so the two workflows deliberately stay independent.
