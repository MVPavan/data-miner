# RESUME — manual_reviewer_cvat

This file is the handoff for resuming the CVAT migration work on a host
machine with Docker access (the original devcontainer can't run docker
compose against the host daemon).

Read top-to-bottom once. Then jump to **§ Resume prompts** and paste the
relevant one into a fresh Claude Code session on the host machine.

---

## Why this work exists

Two needs converged into one project:

| Track | Driver | Status |
|---|---|---|
| **A. Datatang-1000 review migration** | LS UX is slowing the 5-reviewer team on the immediate dataset. Switch them to CVAT without losing in-flight work. | Scaffolded — see [migrations_from_LS/README.md](migrations_from_LS/README.md), [migrations_from_LS/scripts/](migrations_from_LS/scripts/), [migrations_from_LS/docs/migration_from_ls.md](migrations_from_LS/docs/migration_from_ls.md). Stub bodies, not yet runnable end-to-end. |
| **B. Permanent multi-team annotation tool** | Long-term productivity tool. Need CVAT-grade UX **plus** our SAM 3.1 smart tools (click→mask, text→detect, visual prompt, video track). | Strategic decision made (vanilla CVAT + Nuclio, not fork). Plan documented in [docs/long_term_vision.md](docs/long_term_vision.md) and [docs/smart_tools_plan.md](docs/smart_tools_plan.md). Implementation pending. |

Track A is the urgent operational need. Track B is the architecture that
makes A worth doing for the next 3-5 years across multiple teams.

**The two tracks share the same CVAT stack.** Track A doesn't need smart
tools (YOLO predictions are pre-seeded). Track B layers Nuclio functions
onto the same CVAT install once Track A is stable.

---

## Strategic decision (do not re-litigate)

We considered three options for a permanent multi-team tool:

1. Modify Label Studio UI — rejected: LS bones (single-user-first, flat
   task model, weaker review workflow) are the actual UX problem; right-
   click menus don't fix that.
2. Fork CVAT — rejected for permanent tool: ~13 wks to 1.0, ~1 wk/quarter
   rebase tax against the hottest upstream files, dated frontend stack
   (Redux+thunk, Webpack, SVG.js canvas), ramps poorly for new hires.
3. **Vanilla CVAT + Nuclio + custom functions** — chosen: ~6 wks to 1.0,
   no recurring rebase tax, custom code lives in a separate repo (your
   IP), upstream improvements ride free, public docs onboard new hires.

Full rationale in [docs/long_term_vision.md](docs/long_term_vision.md).

---

## Where we left off

- `manual_reviewer_cvat/` exists as a sibling to [manual_reviewer/](../manual_reviewer/)
  with stub Python scripts (~75 LOC each), a self-contained
  [docker-compose.cvat.yml](docker-compose.cvat.yml) pinned to
  `cvat/server:v2.18.0` (Track A — Nuclio omitted on purpose), and a
  Datatang-specific [README.md](README.md).
- The strategic pivot to "permanent tool with smart tools" happened in
  conversation only. Files reflecting Track B (Nuclio plan, smart-tool
  integration plan, long-term vision) are added in this handoff.
- **Nothing has been started on the host machine yet.** No CVAT containers
  pulled, no Nuclio installed, no functions deployed.
- LS at :8080 is still the live review tool. Do not stop it during
  Track B experimentation.

---

## Phase plan (do these on the host machine, in order)

| Phase | Goal | Effort | Resume prompt |
|---|---|---|---|
| **1. CVAT smoke test** | Vanilla CVAT up on :8081 with bundled SAM3 working in the AI Tools panel. Baseline UX. | 1-2 days | [Prompt 1](#prompt-1--phase-1-vanilla-cvat-smoke-test) |
| **2. Nuclio + first SAM 3.1 function** | Replace bundled SAM3 with our SAM 3.1 service for `smart_click`. Live in CVAT canvas. | 1 wk | [Prompt 2](#prompt-2--phase-2-first-sam-31-nuclio-function) |
| **3. Remaining smart tools** | `smart_search` (text), `smart_visual` (exemplar), `smart_track` (video). | 3 wks | [Prompt 3](#prompt-3--phase-3-remaining-smart-tools) |
| **4. Pipeline integration** | Port `build_tasks.py` / `export_to_aa_v4.py` / `sync_ls_to_disk.py` to CVAT REST. | 2 wks | [Prompt 4](#prompt-4--phase-4-pipeline-integration) |
| **5. Track A cutover** | Migrate Datatang-1000 reviewers from LS to CVAT (existing scaffolded plan). | 1 wk | [Prompt 5](#prompt-5--phase-5-datatang-cutover) |

Phase gates: do not advance if the previous phase isn't validated by a
real reviewer using a real task. The whole point of this rebuild is UX —
measure UX, not just HTTP 200s.

---

## What lives where

- **CVAT stack**: [docker-compose.cvat.yml](docker-compose.cvat.yml). Pinned
  to `v2.18.0` for Track A. Track B may bump to current stable
  (check [hub.docker.com/r/cvat/server/tags](https://hub.docker.com/r/cvat/server/tags))
  to get native SAM3 click interactor + SAM2 video tracker.
- **Nuclio additions**: not yet present. Will land in `serverless/`
  alongside the upstream cvat-ai compose split. See
  [docs/smart_tools_plan.md](docs/smart_tools_plan.md).
- **Existing SAM 3.1 service**: stays at `http://<host>:3014/predict`. We
  do **not** rewrite it. Nuclio functions are thin HTTP adapters. Source:
  [scratchpad/DART/](../scratchpad/DART/). Lifecycle:
  [manual_reviewer/scripts/manage_stack.sh](../manual_reviewer/scripts/manage_stack.sh).
- **Reference smart-tool implementations**: keep
  [manual_reviewer/ml_backend/aav4_client.py](../manual_reviewer/ml_backend/aav4_client.py),
  [manual_reviewer/ml_backend/routes.py](../manual_reviewer/ml_backend/routes.py),
  [manual_reviewer/ml_backend/smart_track_lib.py](../manual_reviewer/ml_backend/smart_track_lib.py)
  as the spec. Their HTTP contracts (click_mask / text_detect /
  visual_prompt / track) are what the Nuclio functions wrap.

---

## Resume prompts

Paste one of these into a fresh Claude Code session on the host-Docker
machine. Each is self-contained — assumes no memory of this conversation.

### Prompt 1 — Phase 1: vanilla CVAT smoke test

```
Read /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/RESUME.md
and /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/docs/long_term_vision.md
for context.

Goal: stand up vanilla upstream CVAT with Nuclio components on this host
machine on port 8081. LS is on :8080 — do not touch it. Use the official
upstream cvat-ai/cvat docker-compose layout (compose.yml +
components/serverless/docker-compose.serverless.yml), not the pinned-to-
v2.18.0 file already in this repo (that one omits Nuclio for the
Track A migration; we want native AI Tools for the smoke test).

Steps:
1. git clone https://github.com/cvat-ai/cvat.git into /tmp/cvat-upstream
   (or wherever convenient outside this repo).
2. Bring up CVAT + Nuclio per the official docs:
   https://docs.cvat.ai/docs/administration/basics/installation/
   https://docs.cvat.ai/docs/administration/community/advanced/installation_automatic_annotation/
3. Override CVAT_HOST_PORT to 8081 to avoid clashing with LS.
4. Deploy the bundled SAM3 (or SAM2 if SAM3 isn't shipped yet) function
   via the serverless deploy script.
5. Create a smoke-test project, upload 3-5 frames from
   output/dataset_selection/datatang_diverse_1000/, draw one bbox to
   confirm the editor works, then test the AI Tools panel click→mask
   interactor.
6. Report: CVAT version pulled, Nuclio function names deployed, screenshot
   or text confirmation that smart_click works in the canvas, any
   gotchas hit during install.

Do not write to manual_reviewer_cvat/ yet — this is exploratory. Save
notes in /tmp/cvat-upstream/SMOKE_NOTES.md.

Stop and ask before proceeding to Phase 2 (replacing the bundled model
with our SAM 3.1 service). The decision gate is whether the CVAT UX
feels like a clear improvement over LS for our reviewers.
```

### Prompt 2 — Phase 2: first SAM 3.1 Nuclio function

```
Read /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/RESUME.md
and /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/docs/smart_tools_plan.md
for context. Phase 1 (vanilla CVAT smoke test) is complete.

Goal: write a custom Nuclio interactor function that wraps our existing
SAM 3.1 service's click_mask endpoint at http://<host>:3014/predict.
Replace the bundled SAM model in CVAT's AI Tools panel with this one.

Reference implementations:
- HTTP contract:
  /media/data_2/vlm/code/data_miner/manual_reviewer/ml_backend/aav4_client.py
  /media/data_2/vlm/code/data_miner/manual_reviewer/reconcile/sam3_client.py
- Existing route logic (LS-flavoured, port to Nuclio):
  /media/data_2/vlm/code/data_miner/manual_reviewer/ml_backend/routes.py (smart_click handler)

CVAT Nuclio function spec:
- Function kind: "interactor" (per
  https://docs.cvat.ai/docs/manual/advanced/ai-tools/)
- Input: {image (base64 or URL), pos_points, neg_points, obj_bbox, threshold}
- Output: {points: [[x, y], ...]} for polygon mask OR {mask: <RLE>} for
  raster — read the reference functions in cvat-ai/cvat/serverless/
  pytorch/facebookresearch/sam/nuclio/ for the exact return shape.

Land code in
/media/data_2/vlm/code/data_miner/manual_reviewer_cvat/serverless/sam3_1_click/
with: function.yaml (Nuclio manifest), main.py (handler), Dockerfile.
Use deploy_cpu.sh or deploy_gpu.sh from upstream cvat for the install.

Validate: open the same smoke-test project from Phase 1, switch the
AI Tools interactor model to "SAM3.1 Click", click on an object, confirm
the returned region matches what the same input produces against
http://<host>:3014/predict directly (compare via curl).

Stop and report once smart_click works end-to-end. Do not start
smart_search/smart_visual/smart_track yet.
```

### Prompt 3 — Phase 3: remaining smart tools

```
Read /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/RESUME.md
and /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/docs/smart_tools_plan.md.
Phases 1-2 complete (smart_click via SAM 3.1 Nuclio function works).

Goal: ship the remaining three smart tools as Nuclio functions wrapping
our SAM 3.1 service. Land them all in
/media/data_2/vlm/code/data_miner/manual_reviewer_cvat/serverless/.

Tools:

1. smart_search — text prompt → boxes. Function kind: "detector".
   Wraps SAM 3.1 text_detect (sam3_client.py text_detect method).
   Reference: manual_reviewer/ml_backend/routes.py smart_search handler.

2. smart_visual — exemplar bbox → boxes. Function kind: "interactor"
   with bbox prompt (no clicks). Wraps SAM 3.1 visual_prompt method.
   Reference: manual_reviewer/ml_backend/routes.py smart_visual handler.
   This is the only NET NEW logic — no upstream CVAT function does
   exemplar-based detection.

3. smart_track — bbox + frame range → propagated boxes per frame.
   Function kind: "tracker". Wraps SAM 3.1 video track. Reference:
   manual_reviewer/ml_backend/smart_track_lib.py for the static-object
   motion-threshold filter; consider whether to keep it or trust SAM 3.1's
   own confidence drop. Compare end-to-end behaviour against
   CVAT's bundled SAM2 tracker on the same clip — keep whichever is
   better.

For each: function.yaml + main.py + Dockerfile + a README with the
deploy command. Validate each on a real clip from the Datatang-1000
dataset with a real reviewer (one of the 5).

Stop and report once all four smart tools are live. Document any
divergence from the LS-side behaviour (e.g. CVAT's tracker UX vs
ours).
```

### Prompt 4 — Phase 4: pipeline integration

```
Read /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/RESUME.md.
Phases 1-3 complete (CVAT + Nuclio with all 4 smart tools work).

Goal: port the three pipeline-integration scripts from manual_reviewer/
to manual_reviewer_cvat/, using cvat-sdk (`pip install cvat-sdk`).

Source → target mapping:
- manual_reviewer/scripts/build_tasks.py
  → manual_reviewer_cvat/scripts/build_tasks.py
  (pipeline.db → CVAT tasks with seeded predictions). For seeded
  predictions, use CVAT's "import annotations" on a freshly-created
  task with stage='annotation' so reviewers see them as editable
  starting boxes. Test that re-running with --skip-existing is
  idempotent.

- manual_reviewer/scripts/export_to_aa_v4.py
  → manual_reviewer_cvat/scripts/export_to_aa_v4.py (already a stub)
  (CVAT export → pipeline.db Stage.HUMAN_REVIEW). Use Datumaro export
  format. Preserve the source classification (added/edited/relabeled/
  kept_dropped) by diffing exported boxes against the seeded predictions
  — see manual_reviewer/pipeline_io/ls_export_parser.py for the
  classification logic to mirror.

- manual_reviewer/scripts/sync_ls_to_disk.py
  → manual_reviewer_cvat/scripts/sync_cvat_to_disk.py
  (cron-poll annotation backup). Use CVAT EventsAPI export
  (https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/events-api/)
  on a 5-minute cron. Same idempotent-diff pattern as today; output
  shape can be CVAT-flavoured.

Tests: round-trip a small fixture pipeline.db through build → review
(simulate via SDK) → export. Confirm Stage.HUMAN_REVIEW rows match
what the LS path produces today.

Stop and report once all three scripts are runnable and tested.
```

### Prompt 5 — Phase 5: Datatang cutover

```
Read /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/RESUME.md
and /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/migrations_from_LS/README.md
and /media/data_2/vlm/code/data_miner/manual_reviewer_cvat/migrations_from_LS/docs/migration_from_ls.md.

Phases 1-4 complete (CVAT + Nuclio + smart tools + pipeline integration
all working). Now cut over the live Datatang-1000 review from LS to CVAT.

Existing scaffolded scripts (Track A — all stubs, fill bodies):
- migrations_from_LS/scripts/create_users.py
- migrations_from_LS/scripts/seed_tasks_from_yolo.py
- migrations_from_LS/scripts/migrate_from_ls.py  ← critical, preserves in-flight LS work
- scripts/manage_cvat.sh                         (partial — CVAT stack lifecycle)
- scripts/export_to_aa_v4.py                     (general round-trip, not migration-specific)

Goal: fill in the stub bodies, then run the cutover end-to-end on the
production Datatang-1000 dataset.

Sequence per migrations_from_LS/docs/migration_from_ls.md:
1. Bring up CVAT stack (already up from Phase 1+).
2. Create 5 reviewers via migrations_from_LS/scripts/create_users.py.
3. Create the Datatang-1000 project + label palette.
4. seed_tasks_from_yolo.py — one CVAT task per video clip, round-robin
   by frame count (largest clips → pavan first).
5. migrate_from_ls.py --dry-run first — show matched LS completions
   per CVAT task. Hand a sample to one reviewer to spot-check.
6. migrate_from_ls.py without --dry-run — commit imports.
7. Reviewers log in, finish their slices in CVAT.
8. Keep the LS instance + 5-min cron backup running for 7 days post-
   cutover as a fallback.

Stop and report after dry-run preview — do NOT auto-commit imports
without explicit human go-ahead.
```

---

## Things to NOT do

- Do not rebuild the SAM 3.1 service. It works at :3014. Nuclio functions
  are thin HTTP wrappers, not new model servers.
- Do not fork CVAT. The strategic decision was vanilla + Nuclio.
- Do not stop the LS stack until Phase 5 step 8.
- Do not commit anything to `manual_reviewer/` (the LS-based system) — it
  stays as the live production tool until cutover.
- Do not deploy this to production (multi-team users) until all 5 phases
  are green and one full Datatang cycle has run cleanly through CVAT.

---

## Open questions to revisit on the host machine

- CVAT server version: pinned `v2.18.0` in this repo's compose file (Track A).
  Track B should use the latest stable that ships native SAM3 + SAM2 video
  tracker. Reconcile on Day 1 of Phase 1.
- Right-click context menu on a bbox for "smart action" — not native in
  CVAT. Mitigation: smart_visual on a redrawn exemplar serves the same
  workflow. If reviewers complain after 2 weeks of real use, escalate to
  considering a small fork or upstream PR.
- Multi-tenancy across teams: one CVAT install with multiple Organizations,
  or separate installs per team? Decide before broad rollout.
