# migrations_from_LS

One-time tooling to cut the live 5-reviewer Datatang-1000 review over
from the Label-Studio-based [manual_reviewer/](../../manual_reviewer/)
to the CVAT stack at [..](../).

This whole subdirectory is **transient**. Once cutover is done and the
LS instance is decommissioned, archive it (don't delete — the audit trail
is reproducible from these scripts).

> Strategic context (why CVAT at all): see
> [../docs/long_term_vision.md](../docs/long_term_vision.md). Phase plan
> + resume prompts for picking this up on a host-Docker machine: see
> [../RESUME.md](../RESUME.md) Phase 5.

---

## Goals

1. Five reviewers (pavan, sree, raj, sathish, deepak) split ~1000 video
   clips, **whole clips per user** (so any tracker work stays inside one
   user's slice).
2. Pre-seed each task with the existing YOLO predictions from
   [../../output/dataset_selection/datatang_diverse_1000/yolo/labels/](../../output/dataset_selection/datatang_diverse_1000/yolo/labels/).
3. **Resume in CVAT**: any work already submitted in LS project 9 is
   re-imported as CVAT pre-annotations so reviewers don't redo it.
4. Round-trip back to `pipeline.db` as `Stage.HUMAN_REVIEW` (handled by
   the general [../scripts/export_to_aa_v4.py](../scripts/export_to_aa_v4.py),
   not by a migration-specific script — the round-trip is permanent
   infrastructure, not migration code).
5. Auto-save on for every reviewer, 5-min Postgres+media backup cron
   (handled by [../scripts/manage_cvat.sh](../scripts/manage_cvat.sh)).

---

## Layout

```
migrations_from_LS/
├── README.md                       you are here
├── docs/
│   ├── migration_from_ls.md        spec for migrate_from_ls.py
│   └── reviewer_onboarding.md      cheat sheet for the 5 reviewers (post-cutover)
├── pipeline_io/
│   └── (LS state reader + LS↔CVAT annotation mapper land here)
└── scripts/
    ├── create_users.py             bootstrap 5 reviewers + create the project
    ├── seed_tasks_from_yolo.py     YOLO + clips → CVAT tasks, RR by frame count
    └── migrate_from_ls.py          LS project 9 completions → CVAT pre-annotations
```

All Python scripts are currently **stubs** (~75 LOC each — docstring +
arg parser + `NotImplementedError`). Bodies get filled when Phase 5 of
[../RESUME.md](../RESUME.md) runs on the host-Docker machine.

---

## Quickstart — what to run (Phase 5 cutover)

All commands assume CWD = repo root.

### 0. Smoke-test prerequisites (no install yet)

```bash
docker --version          # need 20.10+
docker compose version    # need v2 plugin
nvidia-smi                # CVAT itself doesn't need GPU; only existing sam3_1 does
ss -tlnp | grep -E ':(8081|8123|8282|5433)'   # must be empty
```

### 1. Bring up the CVAT stack on :8081

```bash
cd manual_reviewer_cvat
cp configs/stack.env.example configs/stack.env   # edit if you want
./scripts/manage_cvat.sh start
./scripts/manage_cvat.sh status                  # confirm cvat_server healthy
```

UI: http://127.0.0.1:8081/

### 2. Create the superuser, then the 5 reviewers

```bash
./scripts/manage_cvat.sh create-superuser admin admin@example.com  # interactive
python -m manual_reviewer_cvat.migrations_from_LS.scripts.create_users \
  --cvat-url http://127.0.0.1:8081 \
  --admin-user admin --admin-pass <set-during-superuser> \
  --reviewers pavan,sree,raj,sathish,deepak \
  --default-password change-me-on-first-login
```

### 3. Create the project + label palette

```bash
python -m manual_reviewer_cvat.migrations_from_LS.scripts.create_users --create-project \
  --project-name "Datatang Diverse 1000" \
  --classes-file output/dataset_selection/datatang_diverse_1000/yolo/classes.txt
# → prints: PROJECT_ID=<n>
```

The label list is read from `classes.txt`. CVAT label IDs == positions
in that file; the `class_registry → name` mapping lives in
[scripts/seed_tasks_from_yolo.py](scripts/seed_tasks_from_yolo.py) (same
logic as [../../manual_reviewer/scripts/build_tasks_from_yolo.py](../../manual_reviewer/scripts/build_tasks_from_yolo.py)).

### 4. Seed tasks from YOLO

One Task per clip, descending round-robin by frame count → pavan gets
the biggest clips first.

```bash
python -m manual_reviewer_cvat.migrations_from_LS.scripts.seed_tasks_from_yolo \
  --cvat-url http://127.0.0.1:8081 \
  --admin-user admin --admin-pass ... \
  --project-id <PROJECT_ID> \
  --dataset output/dataset_selection/datatang_diverse_1000 \
  --aav4-config data_miner/auto_annotation_v4/configs/default.yaml \
  --reviewers pavan,sree,raj,sathish,deepak \
  --strategy frame-count-rr \
  --image-mount-mode shared-folder    # or 'upload'
```

### 5. Resume from LS — import in-flight LS work into CVAT

This is the crucial step so reviewers don't lose progress. It pulls
every completion from LS project 9 and writes them as CVAT annotations
on the matching CVAT task (matched by `image_id`):

```bash
python -m manual_reviewer_cvat.migrations_from_LS.scripts.migrate_from_ls \
  --ls-url http://127.0.0.1:8080 \
  --ls-token "$LS_TOKEN" \
  --ls-project 9 \
  --cvat-url http://127.0.0.1:8081 \
  --cvat-user admin --cvat-pass ... \
  --cvat-project <PROJECT_ID> \
  --dry-run   # preview matched tasks first; remove --dry-run to commit
```

Flow inside the migration:

1. Read every LS task in project 9 → `(image_id, completions[], reviewer_email)`.
2. Translate LS RectangleLabels → CVAT shape rows (label name lookup,
   normalised `xtl/ytl/xbr/ybr`).
3. For each CVAT task whose `image_id` matches, POST `annotations` with
   the imported regions tagged `source=lsf-migrated` so they're visible
   as pre-existing work, not just predictions.
4. Set the CVAT Job assignee to the reviewer who originally took it in LS.
5. Mark migrated jobs `state=in progress` (not `completed`) so the
   original reviewer reopens it and finishes.

See [docs/migration_from_ls.md](docs/migration_from_ls.md) for the full
spec and edge cases (frames split across users, conflicting drafts, etc.).

### 6. Hand off to reviewers

```
URL: http://<host>:8081
Username: <their first name>
Password: change-me-on-first-login (force change on first login)

Settings → Workspace → "Enable auto save" ✅  ← do this together on day 1
```

Hotkeys cheat sheet in [docs/reviewer_onboarding.md](docs/reviewer_onboarding.md).

### 7. Backup

```bash
./scripts/manage_cvat.sh install-cron        # 5-min Postgres + cvat_data dump
# OR for cron-less hosts:
./scripts/manage_cvat.sh watch-backup start  # in-process loop, pidfile-managed
```

### 8. Round-trip to pipeline.db

This is **not** migration code — it's the general round-trip used for
all CVAT review going forward. Lives at
[../scripts/export_to_aa_v4.py](../scripts/export_to_aa_v4.py).

```bash
python -m manual_reviewer_cvat.scripts.export_to_aa_v4 \
  --cvat-url http://127.0.0.1:8081 \
  --cvat-user admin --cvat-pass ... \
  --cvat-project <PROJECT_ID> \
  --pipeline-db /tmp/datatang_review/pipeline.db \
  --since 2026-05-04T00:00:00
```

---

## Cutover policy

- **Do not stop LS** until `export_to_aa_v4.py` has produced one full
  pass matching what LS would have produced.
- Keep the LS 5-min cron backup running through cutover week.
- Once 7 reviewing days have passed cleanly on CVAT, stop LS via
  `manual_reviewer/scripts/manage_stack.sh stop ls` (ml_backend +
  sam3_1 can stay up, untouched).

---

## After cutover

- Tag `migrations_from_LS/` as historical, link the run report (CSV
  output of `migrate_from_ls.py --report`) somewhere durable, and
  archive the LS instance.
- The 5-reviewer team continues working in CVAT; new datasets onboard
  via [../scripts/](../scripts/) only — no further touches to this
  directory.
