# Verification Commands

Status: adopted for data-miner on 2026-05-13.

The repo runs Python code, ships a CLI, and has tests colocated per subproject. Use the commands below to back any "DONE" claim with fresh evidence.

## Adopted commands

### Quick (every change)

- `uv sync` — environment matches `uv.lock`.
- `ruff check data_miner manual_reviewer manual_reviewer_cvat annotation-validator` — lint passes on the four main packages.
- `python -c "import data_miner; print(data_miner.__name__)"` — top-level import succeeds.
- `data-miner --help` — CLI registers and lists all top-level commands.

### Full (before declaring a feature done)

- `ruff format --check .` — formatting clean (do not auto-format unrelated files).
- `pytest data_miner/auto_annotation_v3/tests data_miner/auto_annotation_v4/tests manual_reviewer/tests manual_reviewer_cvat/tests -x -q` — all subproject test suites pass, fail-fast.
- `data-miner status --help` — `status` subcommand registers (proves `cli.py` parses end-to-end).
- Invariant suite: `rg -n '^## \[INV-' .claude/project/invariants.md` then run each invariant's check command.

### Subproject-scoped

- **manual_reviewer**: `bash manual_reviewer/scripts/manage_stack.sh status` after any change to the ML backend or smart-tool dispatch.
- **manual_reviewer (ML backend unit tests)**: `pytest manual_reviewer/tests -x -q`.
- **auto_annotation_v4 servers**: `python -m data_miner.auto_annotation_v4.servers.launch_all --health-check-only` (verifies all configured LitServe servers respond on `/health`).
- **manual_reviewer_cvat smoke**: see `manual_reviewer_cvat/RESUME.md` for the host-Docker-bound checks; do not invoke from this machine without the prerequisites listed there.

### Database

- Schema sanity: `python -c "from data_miner.db.models import Project, Video, ProjectVideo; print([m.__tablename__ for m in (Project, Video, ProjectVideo)])"`.
- Migration check (when SQLModel models change): manually exercise `data-miner init-db --force` against a throwaway database, then run the quick suite.

## Rules

- Run the command, read the output, report the actual result. Do not paraphrase pass/fail from intent.
- A green `ruff check` does not imply tests pass; a green test run on one subproject does not imply the others pass.
- For ML model servers, "healthy" means the `/health` endpoint returns 200 **and** a single inference succeeds — health probes alone are insufficient (some servers warm models lazily).
- When the change touches worker locking, also run a manual two-worker race: start two `python -m data_miner.workers.<stage>` against the same project and confirm only one claims a given row.

## What changed from the prior Bodha-flavored version

The previous `verification.md` checked design-doc consistency (`docs/design/v_2_7/core/*` invariants). Data-miner has runnable code, so the verification surface here is real Python checks, not doc grep. The earlier doc-only fallbacks are no longer applicable and have been removed.
