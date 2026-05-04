# Reviewer onboarding — datatang_diverse_1000_collab

5-person collaborative review of 1000 images selected from DataTang_val
(402 distinct video clips, FPS-deduplicated). Whole clips are assigned
to one reviewer so the smart-track tool can propagate annotations
across sibling frames within your slice.

## 1. Log in

URL: `http://<HOST>:8080/`

| account | password (rotate on first login) |
|---|---|
| pavan@jci.com  | review2026 |
| sree@jci.com   | review2026 |
| raj@jci.com    | review2026 |
| sathish@jci.com | review2026 |
| deepak@jci.com  | review2026 |

Once in, top-right avatar → **Account & Settings** → **Reset token**
to mint your personal API token (only needed if you'll script
anything; not needed for normal review).

## 2. Find your tasks

1. Open project **datatang_diverse_1000_collab**.
2. Click **Tabs** (top of the data manager) → **+ Add Tab**.
3. **Filters** → **+ Add Filter** → field `Data → assigned_to`,
   condition `equals`, value `<your-name>` (e.g. `pavan`).
4. Save the tab as `mine`. Switch to it; you'll only see your slice.

Frames per reviewer (live counts):

| reviewer | frames | clips |
|---|---|---|
| pavan | 304 | 81 |
| sree | 189 | 81 |
| raj | 183 | 80 |
| sathish | 169 | 80 |
| deepak | 155 | 80 |

## 3. Smart tools — what to test

The XML defines four smart tools. Switch tools using the **Auto:**
selector at the top of the canvas. Hotkeys:

| tool | hotkey | how to use | what it does |
|---|---|---|---|
| smart_visual | `Shift+V` | Draw a box around an example object | SAM 3.1 finds visually similar objects in the same image |
| smart_click | `Shift+C` | Click a point on an object | SAM 3.1 returns a tight bbox around what you clicked |
| smart_search | `Shift+S` | Type a class word in the text field, submit | SAM 3.1 detects all instances of that class in the image |
| smart_track | `Shift+T` | Draw a box on a **static** object | SAM 3.1 propagates the box to every sibling frame in the same clip (within your assignment) |

Class hotkeys: `1-0`, `q-w-e-r-t-y-u-i-o-p`, `a-s-d` (skipping `v`, `f`).

Pre-loaded YOLO predictions appear as boxes when you open a task —
edit/delete/relabel as needed. Mark frames `clean`,
`needs_more_review`, or `ambiguous_skip` in the right panel.

## 4. Smart-track scope (important)

Smart-track propagation is **scoped to your assigned frames only**.
A track you start on a Moto_Bicycle frame won't write predictions
onto another reviewer's frames in the same clip — even if the clip
is split across reviewers (it usually isn't, but the safety net is
there).

If you draw a box on a moving object, smart_track will reject it
spatially (drift threshold) and give you back nothing. That's by
design — the tool is for static-object propagation only.

## 5. What we'd love feedback on

1. **smart_click** — does the bbox land tightly on what you clicked,
   or is it loose / wrong object?
2. **smart_visual** — when you draw an example, are the returned
   "similar" boxes useful or noisy?
3. **smart_search** — typing `person`, `truck`, `bicycle`: do the
   returned detections match? Any obvious classes that fail?
4. **smart_track** — propagation count, false propagations on moving
   objects (should be filtered), missed propagations on objects that
   are actually static.
5. **Overall flow** — what slows you down? What hotkey is missing?
   What should the sidebar show that it doesn't?

Capture feedback however convenient — a Slack thread, a Google Doc,
or a `feedback.md` we can iterate on. For specific frame issues,
quote the LS task ID (visible in the URL: `/tasks/<id>`).

## 6. Backups

Annotations and the LS sqlite are snapshotted every 5 minutes via
`manage_stack.sh backup` (cron). Worst-case loss is one cron interval.
Don't be afraid to experiment.

## 7. Trouble?

- Login fails → password reset wasn't applied; ping the admin.
- Smart tool spins forever → SAM 3.1 server (`http://<host>:3014`)
  may be down. Admin: `manage_stack.sh status` then `restart sam3_1`.
- Predictions don't appear → ML backend (`http://<host>:9090`) may be
  down. Same fix: `manage_stack.sh restart ml_backend`.
