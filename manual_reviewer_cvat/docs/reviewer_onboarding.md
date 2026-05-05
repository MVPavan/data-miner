# Reviewer onboarding (CVAT)

For pavan, sree, raj, sathish, deepak.

## Login

URL: `http://<host>:8081`
Username: your first name (lowercase)
Password: as shared by admin (change on first login via top-right avatar → Account)

## First-time setup (do this once, together with admin)

1. Top-right avatar → **Settings** → **Workspace** tab.
2. Toggle **Auto save** → **ON**. Default interval 15s is fine.
3. Toggle **Show all tabs** off (hides panels you don't use).
4. Save.

Without auto-save you can lose up to a session of work on browser crash.
**Do this before annotating anything.**

## Finding your work

1. Top nav → **Jobs**.
2. The list defaults to "all jobs" — click the **Assignee** filter on the
   right and pick yourself. You'll see only your assigned jobs.
3. Sort by **Last updated ▾** to resume where you left off.
4. Click a job → opens the annotation workspace.

## Bbox edit cheat sheet

| Action               | Hotkey      |
|----------------------|-------------|
| Draw rectangle       | N           |
| Save                 | Ctrl+S      |
| Undo                 | Ctrl+Z      |
| Redo                 | Ctrl+Shift+Z |
| Next frame in clip   | F           |
| Previous frame       | D           |
| Next object          | Tab         |
| Previous object      | Shift+Tab   |
| Delete selected      | Del         |
| Change label         | click box → label dropdown in right panel |
| Save & next job      | Ctrl+F      |

## Workflow per frame

1. Frame loads with the YOLO pre-seed boxes already drawn (yellow border).
2. For each box: keep / fix / delete.
3. Add anything missing with **N**.
4. **Ctrl+S** to save.
5. **F** to advance to next frame in the clip.
6. When all frames in the clip are done: top bar → **Submit annotations**.
   This marks the job `completed`. Pick the next job from the Jobs page.

## Class palette

24 classes total — see the right-side label list. Most common:
person, car, bicycle, motorcycle, head, mask. Full list in
[output/dataset_selection/datatang_diverse_1000/yolo/classes.txt](../../output/dataset_selection/datatang_diverse_1000/yolo/classes.txt).

If a class is missing or wrong, **don't make one up** — message admin and
keep going on what's there.

## What to do if…

- **Browser crashed mid-frame**: reopen the job; auto-save covers up to ~15s
  of unsaved work. The yellow boxes are still there.
- **Image won't load**: refresh page. If still broken after 30s, message
  admin with the job ID and frame number.
- **Box won't resize / wrong label dropdown**: hard-refresh (Ctrl+Shift+R).
  CVAT keeps state server-side so you won't lose anything.
- **You think a YOLO pre-seed box is just wrong (e.g. car labelled as bike)**:
  click it, change the label dropdown, save. Keep moving — the export
  pipeline records every change as a "relabeled" entry in the audit trail.

## Resuming work that was originally in Label Studio

If your name was on tasks in the old LS instance, those tasks have been
migrated to CVAT and should appear in your Jobs list **already pre-populated
with whatever you submitted in LS**. Verify on the first 2-3 frames that
your previous work is intact, then continue. If anything looks missing,
stop and message admin before submitting — the migration is reversible.
