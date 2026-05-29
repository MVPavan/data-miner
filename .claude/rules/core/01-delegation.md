# Delegation

Use delegation to reduce context pressure, not to add ceremony for its own sake.

- The coordinator owns scope classification, planning, context curation, review, and final synthesis.
- The worker owns bounded execution inside a clear file and behavior scope.
- Use a fresh worker per task.
- Do not pass raw session history to workers. Pass the exact task, files, invariants, and verification commands they need.
- Do not run multiple implementers in parallel against the same files.
- `standard` and `deep` work defaults to coordinator plus worker. `small` work can stay inline.
