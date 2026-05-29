---
name: brainstorming
description: Use for standard or deep feature work when requirements are ambiguous, exploratory, or not yet approved for implementation.
---

# Brainstorming

This skill defines what to build before planning how to build it.

## Use It When

- the request is exploratory or ambiguous
- multiple approaches are plausible
- user-facing behavior or scope still needs decisions
- the task is `standard` or `deep`

Skip it for clearly `small` work with explicit behavior.

## Workflow

1. Read `AGENTS.md`, `.claude/project/brief.md`, and `.claude/project/docs-index.md`.
2. Scan only enough repo context to answer:
   - does something similar already exist
   - what constraints are already real
   - which docs are authoritative
3. Ask one question at a time.
4. Clarify the problem, users, success criteria, and non-goals.
5. Pressure-test the request for scope, reuse, and carrying cost.
6. If real choices remain, present 2-3 approaches with tradeoffs.
7. Recommend one direction.
8. If durable decisions were made, write `docs/brainstorms/YYYY-MM-DD-<topic>-requirements.md`.
9. Review the new document before treating it as source of truth.
10. In Claude Code with Codex available, get Codex criticism for `standard` and `deep` brainstorms before finalizing.

## Rules

- Do not write code while using this skill.
- Keep implementation details out unless the decision itself is technical.
- Use repo-relative paths only.
