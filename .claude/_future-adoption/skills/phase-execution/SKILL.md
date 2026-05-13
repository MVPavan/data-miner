---
name: phase-execution
description: Execute a development phase from the roadmap, following the full cycle (plan, Codex critique, implement, verify, review).
---

# Phase Execution

Thin orchestrator that drives one complete phase of the Bodha development roadmap. Delegates planning and implementation to specialized skills — does not reimplement them.

## Use It When

The user says "start phase N", "execute phase N", or "begin phase N".

## Inputs

- Phase number/name (e.g., `3`, `7a`, `R1.1`)
- **Optional roadmap path** — passed as argument (e.g., `/phase-execution R1.1 docs/roadmap-reviews.md`).
- If no argument provided, defaults to `docs/roadmap.md`.

## Workflow

### Step 1 — Load Context

1. Determine the roadmap file:
   - If `--roadmap` was provided, use that file.
   - Otherwise, use `docs/roadmap.md`.
   - If the phase identifier starts with `R` (review phase), auto-detect `docs/roadmap-reviews.md` as the roadmap if no explicit `--roadmap` was given.
2. Read the roadmap file — find the phase section, deliverables, spec references, exit criterion.
3. Read `docs/status.md` — confirm phase is NOT_STARTED or IN_PROGRESS.
4. Check blocked-by dependencies — all prerequisite phases must be DONE.
5. Classify phase risk:
   - For original phases: `standard` (0, 1, 4), `deep` (2, 3, 5, 6, 7)
   - For review phases: read the risk level from the roadmap file (each R-phase specifies its own risk)
6. Present the phase summary to the user: deliverables, risk level, blocked-by status.

### Step 2 — Plan (deep phases only)

Skip for `standard` phases.

1. Invoke the **planning skill** with the phase deliverables and spec references as input.
   - Planning skill produces `docs/plans/phase-N-<name>.md`.
   - Planning skill handles Codex plan critique internally (its step 7).
2. Invoke the **document-review skill** on the finalized plan doc.
   - Checks for: gaps, scope bloat, missing constraints, wrong spec references, risky assumptions.
   - Apply any findings to the plan before presenting to the user.
3. Present the plan to the user for approval before proceeding.
4. Do NOT proceed to step 3 without user approval.

### Step 3 — Create Phase Checklist

Replace the "Current Phase Tasks" section in `docs/checklist.md` with:
- Phase name and start date
- All deliverables as checkbox items (with test-first markers)
- Phase plan path (deep only)
- Exit criterion (from roadmap)

### Step 4 — Execute Deliverables

**For `deep` phases:**

Invoke the **subagent-driven-development skill** with the plan from step 2.
- Subagent-dev handles: task packets, implementer dispatch, spec review, code review, Codex code review.
- For deliverables marked **test-first**: invoke the **test-driven-development skill** within the subagent task packet.
- If a test fails unexpectedly during implementation: invoke the **systematic-debugging skill** before retrying.
- After each deliverable completes, this skill updates:
  - `docs/checklist.md` — check off the task
  - `docs/progress.md` — log commit SHA, test results, spec read, notes
- Confirm completion of each deliverable to the user before moving on.

**For `standard` phases:**

Execute deliverables directly (no subagent dispatch needed):
1. Read the Spec Reference from the roadmap file.
2. If marked **test-first**: invoke the **test-driven-development skill**.
3. Implement.
4. Run the Verify check from the deliverable row.
5. If verification fails: invoke the **systematic-debugging skill**.
6. Update `docs/checklist.md` and `docs/progress.md`.
7. Confirm to the user before moving on.

### Step 5 — Exit Criterion

1. Run the phase exit criterion from the roadmap file.
2. Invoke the **verification-before-completion skill** to confirm the phase is genuinely done.
3. If it fails: diagnose, fix, re-run. Do not skip.
4. Report the actual result to the user.

### Step 6 — Update Status

1. Update `docs/status.md`:
   - Phase row → DONE, exit criterion met.
   - Component rows delivered in this phase → status updated.
2. Run `git status` — report uncommitted changes.
3. Summarize: what was built, test results, open items.

## What This Skill Owns

- Phase lifecycle (load → plan → execute → exit → status update)
- Session tracking (`docs/checklist.md`, `docs/progress.md`)
- Exit criterion enforcement
- Status file updates
- Invoking TDD, debugging, and verification skills at the right moments

## What This Skill Delegates

- **Plan creation + Codex plan critique** → planning skill
- **Plan quality gate** → document-review skill
- **Task dispatch + spec review + code review + Codex code review** → subagent-driven-development skill
- **Test-first execution** → test-driven-development skill
- **Unexpected failures** → systematic-debugging skill
- **Completion proof** → verification-before-completion skill
- **Brainstorming** → brainstorming skill (if requirements are unclear, return to brainstorm before planning)

## Rules

- Never skip Codex steps — they are enforced by the delegated skills, not by this skill.
- Never mark a phase done without running the exit criterion.
- Never proceed past step 2 without user approval (deep phases).
- If a deliverable is blocked or fails verification, stop and report.
- Do not modify `infra/proxy-setup/` — it is independent.
