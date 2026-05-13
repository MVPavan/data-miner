---
name: subagent-driven-development
description: Use when executing an approved standard or deep plan through bounded tasks with implementer and review agents.
---

# Subagent-Driven Development

Use this when the work is large enough that one session should coordinate rather than hold all implementation context itself.

## Default Model

- coordinator: reads, plans, curates context, reviews, and synthesizes
- implementer: writes code inside a bounded task
- spec-reviewer: checks plan and requirement compliance
- code-reviewer: checks code quality and safety

## Workflow

1. Read the plan once and extract the tasks.
2. Build a task packet for each task:
   - goal
   - owned and forbidden files
   - origin doc or plan section
   - relevant invariants
   - required tests
   - verification commands
   - commit policy
   - **test-first flag**: if the task is marked test-first, include instruction to invoke the **test-driven-development skill** before writing implementation code
3. Dispatch a fresh implementer per task with only that packet.
4. Require one of: `DONE`, `DONE_WITH_CONCERNS`, `NEEDS_CONTEXT`, `BLOCKED`.
5. If the implementer returns `BLOCKED` or verification fails unexpectedly, invoke the **systematic-debugging skill** before re-dispatching.
6. Run spec review first.
7. Run code-quality review second.
8. Fix and re-review until both pass.
9. In Claude Code with Codex available, run Codex code review before final completion:
   - `standard` work: `/codex:review --wait` (standard defect spotting).
   - `deep` work: `/codex:adversarial-review --wait` (actively tries to break the change).

## Task Sizing

Keep each task packet **small and bounded**. A task that is too large risks subagent failure (API overload, context overflow, or timeout) and wastes all progress on failure.

- **One deliverable per subagent** — never bundle multiple deliverables into one task packet.
- **Cap the prompt** — include only the files and spec sections the implementer needs, not the full plan or full spec.
- **Cap the result** — instruct the implementer to return a summary (status, test counts, file list) not full file contents. Full output stays in the subagent's context, not the coordinator's.

## Failure Recovery

- **On 529 (API overload):** wait 5 seconds, retry the same task once. If it fails again, implement the deliverable inline (directly in the coordinator) instead of via subagent.
- **On subagent BLOCKED:** invoke systematic-debugging before re-dispatching. If blocked twice, split the task smaller or implement inline.
- **On Codex at capacity:** follow the Codex capacity policy in AGENTS.md.
- **Fallback to inline is always valid** — subagent delegation is an optimization, not a requirement. If subagents are consistently failing, switch to direct implementation for the rest of the phase.

## Rules

- Do not run multiple implementers on the same files in parallel.
- Do not pass raw session history to workers.
- Prefer a smaller worker model and a stronger planner or reviewer model.
- Re-dispatch the same worker only for follow-up fixes on the same bounded task.
- If a task blocks twice, improve context or split the task. Do not keep retrying blindly.
