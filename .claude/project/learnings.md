# Durable Learnings

Add entries here only after a fix or pattern has been verified.
Each entry must be reusable across sessions; ephemeral debug notes do not belong here.

Do not duplicate guidance that already lives in:

- [`.claude/project/brief.md`](brief.md) — stack and non-negotiable constraints
- [`.claude/project/invariants.md`](invariants.md) — mechanically checkable facts
- [`.claude/rules/`](../rules/) — coding-style, safety, testing
- The auto-memory at `/root/.claude/projects/-media-data-2-vlm-code-data-miner/memory/` — per-user durable feedback and project state

A learning here is appropriate when it is repo-public (not just user-preference), survived a fix-and-verify cycle, and is likely to be relevant again.

## Entry format

```
### YYYY-MM-DD — Short title

- Scope:        which package, layer, or behavior the learning applies to
- Trigger:      the symptom or task that forced you to discover it
- Rule:         the durable rule to follow next time
- Evidence:     commit SHA, test file, or source path that proves it
- Related docs: pointers to design/architecture docs (when applicable)
```

---

<!-- New entries below. Newest first. -->
