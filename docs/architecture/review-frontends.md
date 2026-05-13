# Review Frontends Architecture

Status: adopted 2026-05-13.

## Decision

Label Studio and CVAT are both maintained review frontends for the same
auto-annotation pipeline. CVAT is not a global replacement for Label Studio.
Different users or teams may annotate in either frontend, and review work must
be transferable in both directions.

The canonical pipeline writeback remains `auto_annotation_v4`:

```text
Label Studio task/completion  ----\
                               review exchange model -> HumanReviewResult -> Stage.HUMAN_REVIEW
CVAT task/job/annotation      ----/
```

## Boundaries

- `data_miner/auto_annotation_v4/` owns checkpoint DB contracts, pipeline
  stages, model-server integration, and prompt authority.
- `manual_reviewer/` owns the Label Studio frontend, LS task building, LS
  export parsing, LS backup/sync behavior, and LS smart-tool protocol adapter.
- `manual_reviewer_cvat/` owns the CVAT frontend, CVAT stack lifecycle, CVAT
  task/job management, CVAT export parsing, and CVAT smart-tool protocol
  adapter.
- Shared annotation exchange logic should live outside frontend-specific code
  once both directions are implemented.

## Exchange Requirements

The exchange layer should preserve enough metadata that a reviewer can move
between tools without losing auditability:

- `image_id` or equivalent stable frame key.
- Source media path or URL, clip identity, and frame order when available.
- Label names and the class-list version used when the task was created.
- Rectangle geometry in normalized coordinates plus original image size.
- Reviewer identity, assignment, status, timestamps, and source frontend.
- Whether a region came from model prediction, human edit, migrated LS work,
  migrated CVAT work, or smart-tool draft.

First implementation can be box-only. Masks, tracks, and richer smart-tool
metadata can be added once the box round trip is stable.

## Adapter Shape

Use a neutral exchange model between frontends:

```text
LS JSON -> neutral review task -> CVAT task/job
CVAT/DATUMARO -> neutral review task -> LS task/completion
LS JSON -> neutral review result -> HumanReviewResult
CVAT/DATUMARO -> neutral review result -> HumanReviewResult
```

This avoids making either frontend's JSON shape the source of truth for the
other. It also keeps `HumanReviewResult` focused on pipeline writeback rather
than frontend migration details.

## Safety Rules

- Do not stop or decommission either frontend as part of source cleanup.
- Do not move, archive, or delete gitignored runtime data without explicit
  user review.
- Keep compatibility shims if production Label Studio imports move into a
  shared package.
- Add fixture-level round-trip tests before using an exchange adapter for real
  reviewer work.

## Current Implementation Notes

- `data_miner/annotation_io/` contains the frontend-neutral box/task/result
  contracts and conversion helpers into v4 `HumanReviewResult`.
- `data_miner/annotation_io/writeback.py` contains shared human-review trace
  append and YOLO label rewrite helpers used by both review frontends.
- `manual_reviewer/pipeline_io/` contains the production Label Studio parser
  and v4 writeback behavior.
- `manual_reviewer_cvat/pipeline_io/datumaro_parser.py` contains the CVAT
  Datumaro bbox parser into the same neutral exchange result model.
- `manual_reviewer_cvat/migrations_from_LS/` already has LS -> CVAT migration
  scaffolding, but it should no longer be treated as one-way decommissioning
  infrastructure.
- `manual_reviewer_cvat/scripts/export_to_aa_v4.py` is the planned CVAT -> v4
  writeback command.
- The SAM 3.1 HTTP client in `manual_reviewer/reconcile/sam3_client.py` is a
  candidate for a shared import path if both frontends need identical smart
  tool client behavior.