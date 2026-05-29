---
description: Turn input documents (designs, reviews, specs) into roadmap phases ready for /run-phases execution.
---

# Prepare Phases

Takes one or more input documents and produces all the artifacts that `/run-phases` needs: brainstorm, roadmap, status rows, and initial checklist/progress scaffolding.

## Inputs

- One or more document paths (design docs, review docs, spec updates) — passed as arguments.
- If no arguments, ask the user which documents to process.

## Workflow

### Step 1 — Understand the Input

1. Read each input document.
2. Read `AGENTS.md`, `.claude/project/brief.md`, `.claude/project/docs-index.md`.
3. Read `docs/status.md` to understand current project state.
4. Classify the input:
   - **New design docs**: new capabilities or components to build
   - **Review/audit docs**: findings to remediate against existing implementation
   - **Spec updates**: changes to existing design docs that require implementation updates

### Step 2 — Brainstorm

Invoke the **brainstorming skill** with the input documents as context:

1. Determine what needs to happen to integrate this input into the project.
2. Clarify scope, success criteria, and non-goals with the user.
3. Identify phases, deliverables, risk levels, dependencies, and execution order.
4. If real choices exist (e.g., new roadmap vs append, grouping strategy), present options to the user.
5. Get Codex criticism on the brainstorm (standard/deep work).
6. Produce `docs/brainstorms/YYYY-MM-DD-<topic>-requirements.md`.

### Step 3 — Roadmap Decision

Ask the user:

> Should these phases go into a **new roadmap file** (e.g., `docs/roadmap-<topic>.md`) or be **appended to an existing roadmap**?

- If new: create `docs/roadmap-<topic>.md` following the format of `docs/roadmap.md`.
- If append: add new phase sections to the specified existing roadmap file.

The roadmap must include for each phase:
- Goal
- Deliverables table (task, create/modify paths, verify)
- Spec references table
- Exit criterion
- Test focus
- Risk classification (standard/deep)
- For review phases: which original phases are modified

### Step 4 — Update Status

Add rows for the new phases to `docs/status.md`:
- Phase name, status (NOT_STARTED), exit criterion met (No)
- For review phases: include "Modifies" column
- Preserve existing status rows — only add new ones.

### Step 5 — Scaffold Tracking Docs

1. If `docs/checklist.md` exists, leave it alone (it gets replaced per-phase by `/phase-execution`).
2. If `docs/progress.md` exists, leave it alone (it gets appended per-deliverable).
3. If neither exists, create them with empty templates.

### Step 6 — Summary

Present to the user:
- Number of phases created
- Roadmap file path
- Execution order
- Risk classification per phase
- Which command to run next: `/run-phases` with the roadmap path if non-default

Example: "Created 6 phases in `docs/roadmap-reviews.md`. Run `/run-phases docs/roadmap-reviews.md` to begin execution."

## What This Command Produces

| Artifact | Path |
|----------|------|
| Brainstorm requirements | `docs/brainstorms/YYYY-MM-DD-<topic>-requirements.md` |
| Roadmap | `docs/roadmap-<topic>.md` or appended to existing |
| Status rows | `docs/status.md` (new rows added) |

## What This Command Does NOT Do

- Does not create plan docs (that's `/phase-execution` Step 2)
- Does not execute any implementation (that's `/run-phases`)
- Does not modify existing phase rows in status.md
- Does not write code

## Rules

- Always brainstorm before creating roadmap — do not skip to roadmap creation.
- Always ask the user about new vs append roadmap — do not assume.
- Use the same roadmap format as `docs/roadmap.md` for consistency.
- Use repo-relative paths only.
- Get Codex criticism on the brainstorm for standard/deep work.
- If the input documents are unclear or contradictory, stop and ask the user before proceeding.
