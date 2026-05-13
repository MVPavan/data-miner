# Mechanically Checkable Invariants

Status: adopted for data-miner on 2026-05-13.

These are the current mechanically checkable project facts. Each invariant has an exact command. Promote new invariants only after the code, manifest, or schema they assert is real.

## [INV-01] CLI entry point exists

- Statement: `data-miner` CLI is implemented in `data_miner/cli.py` and registered as a project script in `pyproject.toml`.
- Check: `test -f data_miner/cli.py && rg -n '^data-miner = "data_miner\.cli:main"' pyproject.toml`
- Must return: matching line for the `[project.scripts]` entry.
- Why it matters: the documented quickstart (`data-miner init-db`, `data-miner populate`, `data-miner workers ...`) depends on this entry point.

## [INV-02] Python pin is 3.12

- Statement: `pyproject.toml` targets Python 3.12 only.
- Check: `rg -n 'requires-python = ">=3\.12, <3\.13"' pyproject.toml`
- Must return: one matching line.
- Why it matters: several deps (PyTorch CUDA builds, SAM 3.1, DART) are wheel-pinned for 3.12. Loosening this requires reverification.

## [INV-03] PostgreSQL is the state authority — SQLModel-backed

- Statement: the main pipeline state lives in SQLModel-backed tables `Project`, `Video`, `ProjectVideo` defined in `data_miner/db/models.py`.
- Check: `rg -n '^class (Project|Video|ProjectVideo)\(SQLModel' data_miner/db/models.py`
- Must return: three matching lines.
- Why it matters: workers gate work on Postgres row state. Adding a new pipeline stage means a new column (or a new table) here, not a Redis key.

## [INV-04] Workers hold heartbeat-renewed Postgres locks

- Statement: `Video` and `ProjectVideo` carry `locked_by`, `locked_at`, and `heartbeat_at` columns.
- Check: `rg -n 'locked_by|locked_at|heartbeat_at' data_miner/db/models.py | wc -l`
- Must return: at least 6 (two tables × three fields).
- Why it matters: stale locks from crashed workers are reclaimed by checking heartbeat age. A new pipeline stage that adds a worker must carry these fields too.

## [INV-05] Supervisord is the worker orchestrator

- Statement: worker setup writes a supervisord config and the worker code references supervisor.
- Check: `rg -l 'supervisor' data_miner/cli.py data_miner/config/loader.py data_miner/config/__init__.py | wc -l`
- Must return: at least 1.
- Why it matters: workers are not launched ad-hoc with `nohup python ...` in production. The `data-miner workers {setup|start|stop|restart|status}` group is the only supported lifecycle path.

## [INV-06] OmegaConf is the main pipeline's config loader

- Statement: `data_miner/config/loader.py` uses `OmegaConf.load`.
- Check: `rg -n 'from omegaconf import OmegaConf|OmegaConf\.load' data_miner/config/loader.py`
- Must return: at least 2 matching lines.
- Why it matters: the rule in [`.claude/rules/python/coding-style.md`](../rules/python/coding-style.md) keys off this fact. Replacing OmegaConf with pydantic-settings is a deliberate decision, not a drive-by change.

## [INV-07] manual_reviewer stack is managed by `manage_stack.sh`

- Statement: there is a single script that owns the lifecycle of sam3_1, ls, and ml_backend.
- Check: `test -x manual_reviewer/scripts/manage_stack.sh && rg -n '^(start|stop|restart|status|logs)\)' manual_reviewer/scripts/manage_stack.sh`
- Must return: file is executable and the case branches exist.
- Why it matters: per user memory `feedback_stack_management`, launching these ad-hoc breaks pidfile + env-var consistency.

## [INV-08] Labeling XML is generated from `classes.txt`

- Statement: `manual_reviewer/configs/build_labeling_config.py` exists and is the source of truth for the Label-Studio labeling config XML.
- Check: `test -f manual_reviewer/configs/build_labeling_config.py`
- Must return: file exists.
- Why it matters: per user memory `feedback_labeling_xml_classes_source`, the `<Label>` palette in `labeling_config.xml` must not be hand-edited.

## [INV-09] `.env` is gitignored

- Statement: `.env` is ignored by git; only `.env.example` is committed.
- Check: `git check-ignore .env && test -f .env.example && ! git ls-files --error-unmatch .env 2>/dev/null`
- Must return: `.env` is ignored (first cmd exits 0), `.env.example` is present, and `.env` is not tracked (the `!` inverts an expected non-zero from `ls-files`).
- Why it matters: secrets-via-env-var convention. Committing `.env` leaks credentials. The current `.gitignore` matches via `*.env` (line 20), not a literal `.env` line — `git check-ignore` is the authoritative way to confirm.

## [INV-10] No Bodha-coded skills are auto-discoverable

- Statement: `.claude/skills/` contains only the generic skills kept after the 2026-05-13 adoption pass; Bodha-coded skills are parked under `_future-adoption/`.
- Check: `! ls .claude/skills/ 2>/dev/null | rg -q '^(architecture-trace|bodha-memory-eval|component-review|design-evolve|phase-execution)$'`
- Must return: success (no matches).
- Why it matters: the parking is deliberate; an accidental `mv` back into `.claude/skills/` would re-expose Bodha-tuned skills that assume directory trees data-miner does not have.
