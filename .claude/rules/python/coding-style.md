# Python Coding Style — data-miner

## Formatting & Linting

- **Formatter:** `ruff format`
- **Linter:** `ruff check`
- **Type checker:** `mypy` where types are present. Do not retrofit `--strict` across legacy modules in one pass — annotate at module boundaries first.
- Public functions and methods at module boundaries should have type annotations.
- Annotate return types on new code (no implicit `-> None`).

## Data Modeling

- Prefer Pydantic `BaseModel` for new boundary data (configs, API payloads, worker job records, ML I/O contracts). Existing `dataclass` / `SQLModel` code stays as it is — match the surrounding module's pattern.
- For new Pydantic models, prefer `model_config = ConfigDict(frozen=True)` when the model is immutable by intent.
- `arbitrary_types_allowed=True` is acceptable when wrapping framework-owned types that are not Pydantic models (torch tensors, ML model handles, opencv arrays). Keep these wrappers process-local; never cross a serialization boundary (Redis broker payload, Postgres column, on-disk job JSON) with them.
- No mutable default arguments — use `field(default_factory=list)`.
- Use enums for fixed sets of values (status codes, stages, sources). The codebase already does this for `VideoStatus`, `ProjectVideoStatus`, `ProjectStatus`, `SourceType` — match that pattern.
- Use `Protocol` for duck typing over `ABC`.

## Strings & Constants

- Avoid loose string literals for protocol values (DB status strings, Redis channel names, supervisor program names, config keys). Declare them as enum values or module-level constants.
- `UPPER_SNAKE_CASE` for module-level constants.
- Format-string field names inside logger calls are fine (`logger.info("frame_extracted", video_id=v.id)`).

## Configuration

- The main pipeline uses `OmegaConf` (YAML merging) + Pydantic `BaseModel` validators in [`data_miner/config/`](../../../data_miner/config/). Match that pattern for new modules inside `data_miner/`.
- `pydantic-settings` is acceptable for new self-contained subprojects (e.g. `manual_reviewer/`, `manual_reviewer_cvat/`) when the subproject's existing config code already uses it. Do not migrate the main `data_miner/` pipeline away from OmegaConf without explicit user approval.
- Inject config at construction time — see `safety.md` for env-var rules.

## File Organization

- 200–500 lines per file typical, 800 max.
- One class per file for core components.
- Maintain strict file separation among core components — do not mix functionalities.
- No circular imports — dependency flows downward: CLI/workers → modules → db/models. If a dependency feels backwards, extract the shared type into a `contracts.py` or `types.py` near the lower layer.

## DRY Principle

- Strictly follow DRY — do not repeat the same logic twice.
- If code is shared across components, extract it into a utility function in the appropriate shared module.
- Prefer a small, well-named utility over copy-pasting 3+ lines.

## Naming

- `snake_case` for files, functions, variables.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for constants.
- No single-letter variables except in comprehensions (`x for x in items`).

## Documentation

- Every function must have a concise docstring explaining what it does (not how).
- Do not write obvious comments. Every comment must add value.
- Prefer self-documenting names over comments where possible.

## Package Management

- Use `uv` exclusively — no `pip`, `pip install`, or `poetry`.
- Add dependencies: `uv add <package>` or `uv pip install <package>`.
- Run scripts: `uv run <script>`.
- Sync environment: `uv sync`.

## Imports

- Standard library → third-party → local (isort order, enforced by ruff).
- Prefer explicit imports over `from module import *`.
- Absolute imports rooted at the top-level package (e.g. `from data_miner.workers.base import ...`, `from manual_reviewer.ml_backend.smart_click import ...`) are the convention. Relative imports are acceptable for closely-coupled siblings inside the same subpackage.

## Error Handling

- No bare `except:` — always specify exception type.
- In long-running workers and library code, prefer a configured logger (`logging` or `structlog` if the module already uses it) over `print()`. `print()` is fine in CLI scripts whose output the user reads directly.
- Keep logging concise — every log line should carry diagnostic context (job id, video id, stage).
