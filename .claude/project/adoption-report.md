# Adoption Report

Status: adopted on 2026-05-13.

The Bodha-flavored `.claude/` overlay was copied into the data-miner repo. This pass adapted it to the actual data-miner project. Generic harness was preserved with minimal edits; Bodha-specific design machinery was parked (not deleted) under `.claude/_future-adoption/` for potential later use.

## Inputs read

- Copied overlay:
  - [`.claude/`](.) (entire tree, pre-adoption)
  - [`AGENTS.md`](../../AGENTS.md), [`CLAUDE.md`](../../CLAUDE.md)
- Target repo evidence:
  - [`pyproject.toml`](../../pyproject.toml), [`README.md`](../../README.md)
  - [`data_miner/cli.py`](../../data_miner/cli.py)
  - [`data_miner/db/models.py`](../../data_miner/db/models.py)
  - [`data_miner/config/loader.py`](../../data_miner/config/loader.py)
  - [`data_miner/workers/`](../../data_miner/workers/), [`data_miner/auto_annotation_v4/`](../../data_miner/auto_annotation_v4/)
  - [`manual_reviewer/docs/next_phases.md`](../../manual_reviewer/docs/next_phases.md), [`manual_reviewer/scripts/manage_stack.sh`](../../manual_reviewer/scripts/manage_stack.sh), [`manual_reviewer/configs/build_labeling_config.py`](../../manual_reviewer/configs/build_labeling_config.py)
  - [`manual_reviewer_cvat/RESUME.md`](../../manual_reviewer_cvat/RESUME.md), `manual_reviewer_cvat/docs/{why_cvat,long_term_vision,smart_tools_plan}.md`
  - [`docs/architecture/overview.md`](../../docs/architecture/overview.md), full `docs/` tree listing
  - User auto-memory at `/root/.claude/projects/-media-data-2-vlm-code-data-miner/memory/MEMORY.md`
  - `git log --oneline -20`

## Files updated

### Project overlay rewritten for data-miner
- `.claude/project/brief.md` — data-miner stack (Python 3.12 / uv / SQLModel+Postgres / supervisord workers / OmegaConf+Pydantic config / LitServe model servers / Redis cache); subproject map; non-negotiable constraints derived from the codebase + user memory.
- `.claude/project/invariants.md` — 10 mechanically checkable invariants tied to real files (`data_miner/cli.py`, `data_miner/db/models.py`, `manual_reviewer/scripts/manage_stack.sh`, etc.) and to the parking decision (INV-10).
- `.claude/project/verification.md` — real Python verification commands (`uv sync`, `ruff check`, `pytest`, `data-miner --help`, manual_reviewer stack-status). Doc-only Bodha checks removed.
- `.claude/project/docs-index.md` — points at the actual docs/ tree (architecture, user-guide, k3s) and the subproject roadmaps.
- `.claude/project/learnings.md` — reset to empty template per user direction. The valuable data-miner learnings already live in the user auto-memory (e.g. `feedback_smart_click_dedup`, `feedback_ls_annotation_backup`, `feedback_stack_management`) and should not be duplicated.
- `.claude/project/adoption-report.md` — this file.

### Small edits to keep generic harness portable
- `.claude/rules/python/safety.md` — removed reference to Bodha invariant 34 and Infisical secrets; replaced with data-miner reality (`.env` + `python-dotenv`, env-vars only at process entry, `tenacity` for retries, Postgres-locking-as-concurrency-bound).
- `.claude/rules/python/coding-style.md` — removed Bodha-only file-path exception; relaxed "pydantic-settings required" to "OmegaConf in main pipeline, pydantic-settings OK in subprojects"; relaxed "no `os.environ`" to "env vars only at entry points"; relaxed "no `print()`" to "no `print()` in workers/libs; OK in CLI scripts"; updated layering example from Bodha (api→core→infrastructure) to data-miner (CLI/workers→modules→db/models).
- `.claude/docs/codex-usage-guide.md` — replaced "Using Codex with Bodha Skills" subsection with project-neutral examples; pointed to `_future-adoption/` for the parked Bodha examples.
- `AGENTS.md` — read-order step 3 now points to subproject roadmaps (manual_reviewer/docs/next_phases.md, manual_reviewer_cvat/RESUME.md) instead of `docs/roadmap.md` + `docs/status.md`; "Phase Execution" section now notes that the Bodha-style phase commands are parked.
- `CLAUDE.md` unchanged (`@AGENTS.md`).

### Kept verbatim (generic — no edits needed)
- Agents: `planner`, `implementer`, `code-reviewer`, `spec-reviewer`, `docs-researcher`.
- Skills: `ak-guide`, `brainstorming`, `planning`, `document-review`, `systematic-debugging`, `test-driven-development`, `verification-before-completion`, `subagent-driven-development`, `html-artifact`, `cost-estimate`.
- Commands: `adopt`, `check-invariants`, `use-codex`.
- Rules: `rules/core/01-delegation.md`, `rules/core/02-knowledge-discoverability.md`, `rules/python/testing.md`.
- Hook: `.claude/hooks/block-dangerous-git.sh`.
- `.claude/settings.json` (hook wiring) and `.claude/settings.local.json` (project-specific Bash allowlist — already data-miner-native).
- `.claude/docs/codex-usage-guide.md` minus the one Bodha subsection (above).

### Parked under `.claude/_future-adoption/`
- `skills/`: `architecture-trace/`, `bodha-memory-eval/`, `component-review/`, `design-evolve/`, `phase-execution/`.
- `commands/`: `run-phases.md`, `prepare-phases.md`.
- `archived-guides/`: `html-artifact-*.skill.zip` + `instructtions.md` (stale packaging archives from the Bodha overlay).
- `_future-adoption/README.md` — explains why each item is parked and the re-adoption checklist (decide on design-versioning, create `docs/roadmap.md` + `docs/status.md` + `docs/checklist.md` + `docs/progress.md`, then move parked items back).

## Assumptions

- The main `data_miner/` pipeline stays on OmegaConf for config; switching to pydantic-settings is a deliberate future decision, not implied by adopting the Bodha harness.
- Subprojects (`manual_reviewer`, `manual_reviewer_cvat`, `auto_annotation_v*`) keep their own phase trackers (`docs/next_phases.md`, `RESUME.md`) rather than rolling up to a repo-wide roadmap.
- The user auto-memory at `/root/.claude/projects/-media-data-2-vlm-code-data-miner/memory/` continues to hold per-user feedback (smart-tool dispatch, stack management, etc.); the project overlay does not re-state those.
- The data-miner main pipeline is operational, not greenfield; rules and verification commands assume code already runs.

## Conflicts / decisions

- The Bodha overlay's `python/coding-style.md` mandated `pydantic-settings`; data-miner's main pipeline uses OmegaConf. Resolution: kept OmegaConf as the documented convention for the main pipeline; allowed pydantic-settings in subprojects that already use it.
- The Bodha overlay's `safety.md` referenced Infisical for secrets. Resolution: replaced with `.env` + `python-dotenv` to match repo reality.
- The Bodha overlay's `coding-style.md` named a Bodha-specific file (`src/bodha/infrastructure/gateway/model_spec.py`) as an exception to `arbitrary_types_allowed=False`. Resolution: removed the specific-file callout; kept the broader principle scoped to "framework-owned types that don't cross a serialization boundary" so it applies to ML model handles / torch tensors.
- The Bodha overlay's `verification.md` was doc-grep-only because Bodha had no source tree yet. Resolution: replaced with real Python verification commands; data-miner has running code.

## Next review step

- Periodic re-check of invariants via `/check-invariants` after material changes to the schema or worker layer.
- If a repo-wide roadmap convention is later adopted, re-adopt the parked skills/commands per the checklist in [`_future-adoption/README.md`](../_future-adoption/README.md).
