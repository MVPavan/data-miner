# Project Brief

Status: adopted 2026-05-13 (Bodha-flavored harness adapted to data-miner).

## What Is data-miner

A PostgreSQL-backed, supervisor-managed video processing pipeline that turns YouTube videos into large-scale computer vision datasets. The main pipeline runs the cascade:

```
download → extract → filter → cross-dedup → detect
```

with each stage owned by a long-running worker process holding a row-level Postgres lock on the row it is processing, refreshed via heartbeat.

**Users:** ML engineers building object-detection / segmentation training sets from open video sources.

## Stack

- **Language:** Python 3.12+ (pyproject pins `>=3.12, <3.13`).
- **Package manager:** `uv` (`uv sync`, `uv add`, `uv run`).
- **CLI:** `click` — entry point `data-miner` → [`data_miner/cli.py`](../../data_miner/cli.py).
- **ORM:** `sqlmodel` over PostgreSQL — [`data_miner/db/models.py`](../../data_miner/db/models.py) defines `Project`, `Video`, `ProjectVideo`.
- **Config:** OmegaConf YAML merging + Pydantic `BaseModel` validators in [`data_miner/config/`](../../data_miner/config/).
- **Worker orchestration:** `supervisord` — `data-miner workers setup` writes `/etc/supervisor/conf.d/data_miner.conf`; `supervisorctl` runs the group.
- **Concurrency:** Postgres row-level locks + heartbeat (`locked_by`, `locked_at`, `heartbeat_at` on `Video` and `ProjectVideo`).
- **ML model serving:** `litserve` HTTP servers under `data_miner/auto_annotation_v4/servers/` (gdino, falcon, sam3, sam3-DART, owlvit2). Lifecycle in `manual_reviewer/scripts/manage_stack.sh` for the manual_reviewer subproject.
- **Broker / cache:** Redis (`redis`, `hiredis`). Cache, never the source of truth.
- **Embedding / dedup:** SigLIP2 for filter, DINOv3 + FAISS for cross-video dedup, LanceDB for vector store.
- **Detection:** GroundingDINO, OWLv2, SAM 3.1 (vanilla + DART variant), Falcon-Perception.
- **Logging:** stdlib `logging` configured via `data_miner.logging`; Loki shipping available (`python-logging-loki`).
- **Secrets:** `.env` loaded by `python-dotenv`. `.env.example` is the committed template.

## Active subprojects (siblings of `data_miner/`)

| Subproject | What it is | Phase tracker |
|------------|------------|----------------|
| [`manual_reviewer/`](../../manual_reviewer/) | Label Studio + SAM 3.1 ML backend for human review. Smart-tool drafts come from the ML backend; users accept/correct. | [`manual_reviewer/docs/next_phases.md`](../../manual_reviewer/docs/next_phases.md) — Phases 1/3/4/5/7/8 done; Phase 2 (LS ML backend hardening) and Phase 6 (Rex-Omni) pending. |
| [`manual_reviewer_cvat/`](../../manual_reviewer_cvat/) | CVAT-based review (permanent multi-team tool). Vanilla CVAT + Nuclio smart-tools, **not** a fork. | [`manual_reviewer_cvat/RESUME.md`](../../manual_reviewer_cvat/RESUME.md) — pending host-Docker machine. Two tracks: (A) Datatang-1000 LS→CVAT migration, (B) permanent Nuclio + SAM 3.1 stack. |
| [`data_miner/auto_annotation_v4/`](../../data_miner/auto_annotation_v4/) | Current auto-annotation engine: SAM 3.1, Rex-Omni, VLM finalization. Replaces v3. | Phase tracker lives inside that package (Phases 1+2+3 complete per user memory). |
| [`data_miner/auto_annotation_v3/`](../../data_miner/auto_annotation_v3/) | Legacy auto-annotation (superseded; kept for parity/benchmark). | — |
| [`annotation-validator/`](../../annotation-validator/) | YOLO bbox sanity checker (filter_inner_bboxes, vllm-backed). | — |
| [`detection_metrics/`](../../detection_metrics/) | Detection evaluation + dataset format utilities. | — |
| [`export/`](../../export/) | Clean YOLO dataset export pipeline. | — |

## How work happens

- The main `data_miner` pipeline is **operational** — there is no central `docs/roadmap.md`. Improvements there land as targeted feature work.
- Subproject phase-tracked work uses each subproject's own roadmap file (see table above), not a repo-wide one. The Bodha-style `/phase-execution`, `/run-phases`, `/prepare-phases` commands are parked in [`.claude/_future-adoption/`](../_future-adoption/README.md) until/unless we adopt a repo-wide cascade.
- For long-running model servers (manual_reviewer's SAM 3.1 + LS + ml_backend), always go through `manual_reviewer/scripts/manage_stack.sh {start|stop|restart|status|logs}` — never launch them ad-hoc (see user memory `feedback_stack_management`).

## Non-negotiable constraints

- **PostgreSQL is the sole state authority.** Workers gate work on row state; Redis is a broker/cache only.
- **Locks are heartbeat-renewed**, not fire-and-forget. A worker that crashes without renewing its heartbeat loses the lock to the next claimant — this is intentional. See `data_miner/db/models.py`.
- **`manage_stack.sh` is the lifecycle owner** for manual_reviewer's sam3_1 / ls / ml_backend services. Ad-hoc `python serve_sam3.py` or `python -m label_studio` violations cause pidfile + env-var drift.
- **Labeling XML is rendered from `classes.txt`** via `manual_reviewer/configs/build_labeling_config.py`. Never hand-edit the rendered `labeling_config.xml` palette.
- **LS annotation backup is cron-pull, not webhook** (LS Community 1.23 webhook is unreliable). The pull cadence is 5 minutes via `sync_ls_to_disk.py`.
- **Smart-tool dispatch priority** in manual_reviewer ML backend: `smart_visual > smart_click > smart_search > smart_track`. `smart_track` only fires when no other smart draft is present.
- **Secrets via `.env`** — never commit, never log values. `.env.example` is the committed template.

## Notes for agents

- The data-miner pipeline already exists and runs in production; treat changes there as careful surgery, not greenfield design.
- For new work in subprojects, check the subproject's `docs/` or `RESUME.md` first — phase state is tracked per-subproject, not centrally.
- The `scratchpad/` folder is gitignored; use it for spikes, throwaway scripts, and experiment outputs.
- The Bodha-coded process machinery (architecture-trace, component-review, phase-execution, design-evolve, bodha-memory-eval) is parked in [`.claude/_future-adoption/`](../_future-adoption/README.md) for possible later adoption. It is intentionally out of skill/command auto-discovery and should not be invoked today.
