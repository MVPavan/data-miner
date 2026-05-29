---
name: systematic-debugging
description: Use for bugs, failing tests, broken integrations, or confusing behavior before proposing fixes.
---

# Systematic Debugging

Do not guess. Find the root cause first.

## Workflow

1. Reproduce the issue and capture the exact symptom.
2. Read the error output, stack trace, and recent changes carefully.
3. Narrow the failing boundary by comparing working and broken paths.
4. Form one root-cause hypothesis at a time.
5. Test the smallest change or diagnostic that can falsify that hypothesis.
6. Only after the cause is credible, write or update the proving test and fix it.
7. Re-run verification and confirm the original symptom is gone.

## Rules

- No stacked guesses or bundled fixes.
- If one hypothesis fails, stop and form a new one from the evidence.
- If the issue crosses multiple systems or one hypothesis already failed, Codex can act as a falsifier in Claude Code.
