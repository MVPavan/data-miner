---
name: test-driven-development
description: Use for risky behavior changes, bug fixes, or legacy edits where test-first or characterization-first execution is the safest path.
---

# Test-Driven Development

Write one test first. Watch it fail for the right reason. Then write the minimum code to pass.

## Use It When

- a bug fix needs proof
- behavior is changing in a risky area
- legacy behavior needs characterization before edits
- the plan or dispatch explicitly says `test-first` or `characterization-first`

## Workflow

1. Pick one behavior, not a whole feature slice.
2. Write a test through a public interface when possible.
3. Run it and confirm it fails for the expected reason.
4. Write the smallest code change that makes it pass.
5. Re-run the relevant tests.
6. Refactor only while green.
7. Repeat one behavior at a time.

## Rules

- Do not write a whole batch of tests first.
- Prefer behavior over implementation detail.
- For legacy code, a characterization test is acceptable before the fix.
- If the test strategy is disputed in Claude Code, Codex can critique the strategy before wider implementation.
