---
name: document-review
description: Use when a requirements or plan document exists and needs a focused review for gaps, scope bloat, missing constraints, or risky assumptions.
---

# Document Review

Review the document itself before using it as a source of truth.

## Workflow

1. Read the document and classify it as requirements or plan.
2. Check it against current repo context and any authoritative project docs.
3. Look for:
   - internal inconsistency
   - missing constraints or unverifiable claims
   - scope bloat
   - missing tests or verification
   - security, data, or performance risks when relevant
4. Fix obvious wording or structure issues inline when the correction is unambiguous.
5. Surface decision-level issues instead of silently rewriting intent.
6. In Claude Code with Codex available, get Codex criticism only when the document has not already had a Codex pass or the document is unusually risky.
