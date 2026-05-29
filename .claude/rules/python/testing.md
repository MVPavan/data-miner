# Python Testing

- Use `pytest` unless the repo already standardizes on a different framework.
- For behavior-risky work, write a failing test or characterization test before changing the code.
- Prefer unit tests for pure logic and integration tests for I/O boundaries.
- Test behavior through public interfaces, not private implementation details.
- Prefer parametrization and shared fixtures over repetitive test bodies.
- Do not mock internals by default; mock hard external boundaries when needed.
- Reuse the repo's existing test directories, fixtures, and markers where they exist.
- The exact commands to trust live in `.claude/project/verification.md`.
