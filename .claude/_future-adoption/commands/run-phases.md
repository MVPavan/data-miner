---
description: Execute all remaining phases from a roadmap sequentially with auto-approval and context management.
---

# Run All Remaining Phases

Execute every incomplete phase from a roadmap in order, compacting context between phases to keep the workflow running continuously.

## Inputs

- **Optional roadmap path** — passed as argument (e.g., `/run-phases docs/roadmap-reviews.md`).
- If no argument provided, defaults to `docs/roadmap.md`.

## How It Works

1. Determine the roadmap file:
   - If user provided a path argument, use that roadmap.
   - Otherwise, use the default `docs/roadmap.md`.
2. Read `docs/status.md` to find all phases and their statuses.
3. Read the roadmap file to determine phase order and identify the first phase with status `NOT_STARTED` or `IN_PROGRESS`.
4. Execute that phase using `/phase-execution N` **with the roadmap path** (e.g., `/phase-execution R1.1 --roadmap docs/roadmap-reviews.md`).
5. Within the phase execution, auto-approve all user confirmation prompts (plan approval, etc.) — do not pause and wait for input.
6. After the phase completes and passes its exit criterion, commit the changes.
7. Run `/compact` to summarize context and free up space.
8. Re-read `docs/status.md` and the roadmap file to find the next incomplete phase (file state is the source of truth, not conversation memory).
9. Continue from step 4 with the next phase.
10. Stop when all phases in the roadmap are DONE or a phase fails its exit criterion.

## Phase Order

Read the execution order from the roadmap file itself. Examples:

**Default roadmap** (`docs/roadmap.md`):
```
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7a → 7b → 8
```

**Review roadmap** (`docs/roadmap-reviews.md`):
```
R1.1 → R1.2 → R1.3 → R1.4 → R1.5 → R1.6
```

The roadmap file is authoritative for phase order — do not hardcode phase sequences.

## Auto-Approval Rules

For this workflow, treat the following as auto-approved:

- Plan approval for deep phases (Step 2c in phase-execution) — approve immediately
- Commit prompts — commit after each phase
- Codex critique — run if available, follow capacity policy in AGENTS.md

## Context Management

**Use `/compact` between phases** so the workflow continues uninterrupted. `/clear` would terminate the session and require the user to manually re-invoke this command — defeating the purpose of automation.

After compaction, conversation details may be lossy. That's fine because all persistent state lives in files:

- `docs/status.md` — which phases are done (re-read after every compact)
- `docs/progress.md` — per-deliverable details
- `docs/checklist.md` — current phase tasks
- `docs/plans/` — approved plans for deep phases

**Critical rule:** After every `/compact`, re-read `docs/status.md` and `docs/checklist.md` before continuing. Never rely on conversation memory for phase status — always check the files.

**Within a phase:** use `/compact` between deliverables if context grows large (especially in deep phases with 8+ deliverables).

## Failure Handling

- If a phase fails its exit criterion: stop, report what failed, do NOT continue to the next phase.
- If Codex is at capacity: follow the Codex capacity policy in AGENTS.md.
- If a test fails: invoke systematic-debugging before retrying. If it fails twice, stop and report.
- Subagent failures (529, timeouts): handled by the subagent-driven-development skill's own failure recovery rules.

## Execution

Now execute this workflow:

1. Read `docs/status.md` and identify the next incomplete phase.
2. Begin executing it with `/phase-execution` and auto-approval.
3. After completion, commit, `/compact`, re-read status files, and continue with the next phase.
