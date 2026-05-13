---
name: verification-before-completion
description: Use before claiming a change is complete, fixed, or ready. Requires fresh verification evidence.
---

# Verification Before Completion

Evidence comes before claims.

## Workflow

1. Identify the command that proves the claim.
2. Run the command fresh.
3. Read the output and exit status.
4. Report the actual result.
5. Check git status before presenting completion.

Use `.claude/project/verification.md` and `.claude/project/invariants.md` as the source of truth.

Do not rely on memory, confidence, partial checks, or agent reports.
