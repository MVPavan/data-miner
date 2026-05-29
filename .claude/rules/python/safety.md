# Python Safety — data-miner

## Input & Query Safety

- Parameterize all SQL and query inputs — never f-strings or string concatenation. SQLModel/SQLAlchemy already does this; do not bypass it.
- Validate user-controlled file paths and shell arguments. Many pipeline scripts shell out to `yt-dlp`, `ffmpeg`/`av`, supervisor — quote paths and reject `..` traversal in IDs that come from external sources.
- Do not use `eval`, `exec`, or unsafe deserialization (`pickle.load`, `yaml.load` without SafeLoader) on untrusted input.

## Configuration & Secrets

- Read `os.environ` only at process entry points (CLI, worker `__main__`, server boot). Inject the resulting config into modules as parameters; modules deep in the call graph should not reach for env vars.
- Secrets via `.env` (loaded by `python-dotenv`) — never commit `.env` to git, never log secret values. The repo already has `.env.example` as the template; new secrets go there.

## I/O & Async Safety

- All external I/O must have explicit timeouts and bounded retries. `tenacity` is the convention in this repo.
- No blocking I/O in async context — use `asyncio.to_thread()` for blocking calls.
- Bounded concurrency for parallel operations — never unbounded `gather()`. Workers use Postgres row-level locking + heartbeat to bound concurrency at the DB layer; in-process fan-out should use an explicit semaphore.
- Pool DB connections — never create raw `psycopg2.connect` per request. Use `data_miner.db.connection`.
- Never swallow exceptions silently in async background tasks. Workers must mark the row FAILED with context, not just exit.
- Cache failures (e.g. Redis) must degrade quality, never block the request path. Redis is a broker/cache, not the source of truth.
