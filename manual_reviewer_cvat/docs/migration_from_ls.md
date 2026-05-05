# LS → CVAT migration: how reviewers resume their work

Goal: when we stand up CVAT, the 5 reviewers should open it and find the
exact tasks they were working on in LS, with all of their submitted boxes
already on the canvas. Zero re-annotation.

This is the job of [scripts/migrate_from_ls.py](../scripts/migrate_from_ls.py).
This doc is the spec.

## Mapping table

| LS concept                  | CVAT equivalent                              | Notes |
|-----------------------------|----------------------------------------------|-------|
| Project (id 9)              | Project                                      | One target project per LS source. |
| Task (1 image)              | Image inside a Task                          | CVAT Tasks are clip-sized; LS Tasks are frame-sized. We re-aggregate. |
| Annotation                  | Annotation on a frame                        | RectangleLabels → CVAT `rectangle` shape. |
| Prediction (yellow)         | Annotation, `source` attribute = `proposal`  | CVAT has no separate predictions channel. |
| `image_id` in task data     | Frame `name` (stripped of dirs/extensions)   | Canonical join key. |
| `annotator.email`           | `User.email`                                 | Used to set Job.assignee. |
| `was_cancelled`             | Job.state = "in progress" + frame tag `skipped` | We do not auto-mark anything `completed`. |
| `from_name=smart_*` regions | DROPPED                                      | Not relevant to CVAT review pass. |

## Resume contract — what reviewers see after migration

For each LS task that has **any** completion in project 9:

1. The matching CVAT image will have the YOLO pre-seed boxes (from
   `seed_tasks_from_yolo.py`) **plus** the reviewer's submitted edits
   layered on top.
2. The CVAT Job containing that image is assigned to the same reviewer
   who took the LS task.
3. The Job state is `in progress` so the reviewer must re-Submit before
   it counts as done in CVAT. This is intentional: it gives us a clean
   CVAT timestamp + reviewer attribution for the audit trail and
   surfaces any migration mistakes immediately ("wait, this isn't what
   I labelled").

For LS tasks with **no** completions:
- Pre-seed only, no extra annotations migrated.
- Job assignee preserved if LS had set one.

## Conflict resolution rules

- **Multiple completions on one LS task by different reviewers**: keep the
  most recent by `updated_at`; log the others to the report. CVAT-side, the
  CVAT job is assigned to the most-recent annotator.
- **LS reviewer ≠ CVAT reviewer (renamed account, etc.)**: match on email,
  not username. If the email isn't in CVAT, fall back to the project owner
  and flag in the report.
- **Image present in LS but missing from CVAT**: log + skip. Means
  `seed_tasks_from_yolo.py` didn't pick that image up; investigate before
  cutting over.
- **Image present in CVAT but missing from LS**: leave CVAT pre-seed alone.
  Reviewer will see only the YOLO predictions for that frame.

## Region-ID sanitisation

LS task 348 broke because LSF used colons in image_id as React keys; we patched
the persisted region IDs to replace `:` → `_`. CVAT does NOT have this bug
(verified — CVAT stores filenames as Postgres strings, not DOM keys), but
for safety we strip colons from any LS-imported region IDs before POSTing
to CVAT. Cost is zero, defensive value is non-zero.

## Dry-run output

```
$ python -m manual_reviewer_cvat.scripts.migrate_from_ls ... --dry-run

image_id,ls_task_id,cvat_task_id,ls_annotator,cvat_assignee,regions,status
2018-10-09-06:33:57,348,42,pavan@jci.com,pavan,65,WOULD_MIGRATE
2018-10-09-06:34:00,349,42,pavan@jci.com,pavan,72,WOULD_MIGRATE
... 822 rows ...
SUMMARY: 822 to migrate / 41 skipped (no_completions) / 3 skipped (no_match)
```

Inspect the CSV. If skip counts look wrong, fix the underlying issue in
`seed_tasks_from_yolo.py` before re-running without `--dry-run`.

## Idempotency

Safe to re-run. CVAT's `POST /api/tasks/{id}/annotations` with `action=create`
appends; we use `action=update` (replace) to make the script idempotent so a
re-run with the same LS state produces the same CVAT state. The Job
state/assignee PATCH is also idempotent.

If a reviewer has already started working in CVAT after a migration, **do not
re-run the migration on that project** — it would clobber their CVAT-side
edits with the LS state. Track this with a `migrated.lock` file written by
the script in `manual_reviewer_cvat/.migration_state/<cvat-project-id>.lock`.

## What this script does NOT do

- Does not delete LS data. LS keeps running through cutover week.
- Does not move users (use `create_users.py` for that).
- Does not migrate the `pipeline.db` round-trip — that's `export_to_aa_v4.py`,
  which only reads from CVAT going forward.
- Does not migrate smart-tool drafts, reviewer notes, or task-level metadata
  beyond what's listed in the mapping table above. Anything beyond that has
  to be reviewed manually.
