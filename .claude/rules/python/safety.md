# Python Safety — Bodha

## Input & Query Safety

- Parameterize all SQL and query inputs — never f-strings or string concatenation.
- Validate user-controlled file paths and shell arguments.
- Do not use `eval`, `exec`, or unsafe deserialization on untrusted input.

## Configuration & Secrets

- No `os.environ` reads in business logic — inject config at construction time.
- Secrets via Infisical — never in YAML, git, or logs.

## I/O & Async Safety

- All external I/O must have explicit timeouts and bounded retries.
- No blocking I/O in async context — use `asyncio.to_thread()` for blocking calls.
- Bounded concurrency for parallel operations — never unbounded `gather()`.
- Connection pools for all store access — never create raw connections.
- Never swallow exceptions silently in async background tasks.
- Redis failure degrades quality, never blocks response (invariant 34).
