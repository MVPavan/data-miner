---
name: planning
description: Use when approved requirements or a clear multi-step request need a durable implementation plan before coding.
---

# Planning

This skill turns approved requirements into an execution-ready plan.

## Workflow

1. Start from approved requirements or explicit assumptions.
2. If product behavior is still unclear, return to brainstorming.
3. Research local code, tests, project docs, and prior learnings before fixing the plan shape.
4. Write a right-sized plan under `docs/plans/`.
5. Include:
   - goal and scope
   - origin doc or assumptions
   - repo-relative create/modify/test paths
   - risks, invariants, and verification
   - the smallest tracer-bullet step that proves the path
6. For risky behavior changes, call out test-first or characterization-first execution.
7. In Claude Code with Codex available, critique plans before finalizing:
   - Use `/codex:rescue --wait --fresh "Critique this plan for scope gaps, ordering issues, missing test coverage, design doc misalignment: <plan summary>"`.
   - Do NOT use `/codex:review` or `/codex:adversarial-review` — those operate on git diffs, not plan documents.
8. Hand the plan to subagent-driven development or direct execution, depending on scope.

## Rules

- Keep the plan portable and repo-relative.
- Do not pre-write large code blocks or shell choreography.
- Make tasks small enough to verify independently.
