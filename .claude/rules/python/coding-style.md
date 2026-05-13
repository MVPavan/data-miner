# Python Coding Style — Bodha

## Formatting & Linting

- **Formatter:** `ruff format`
- **Linter:** `ruff check` with strict rules
- **Type checker:** `mypy --strict`
- All public functions and methods MUST have type annotations.
- All return types MUST be annotated (no implicit `-> None`).

## Data Modeling

- Use Pydantic `BaseModel` for all data structures (invariant 21). Prefer over dataclasses.
- All Pydantic models: `model_config = ConfigDict(frozen=True)`.
- Default to `arbitrary_types_allowed=False`. **Documented exception (2026-04-10):** `src/bodha/infrastructure/gateway/model_spec.py` may set `arbitrary_types_allowed=True` to wrap `pydantic_ai.models.openai.OpenAIChatModel` and `pydantic_ai.settings.ModelSettings`, which are framework-owned types that are not Pydantic models. The exception is scoped to that one file and those two types only; no other file may widen this. `ModelSpec` is process-local and never crosses a serialization boundary (Temporal, Redis, Postgres), so the config-level type check being relaxed has no runtime impact beyond the intended one.
- No mutable default arguments — use `field(default_factory=list)`.
- Use enums for every fixed set of values (config options, status codes, types, categories).
- Use `Protocol` for duck typing over ABC.

## Strings & Constants

- No loose string literals scattered in code. All user-facing strings, log messages with domain meaning, config keys, and identifiers must be declared as constants or enum values in a central location per component.
- `UPPER_SNAKE_CASE` for constants.
- Exception: format strings in structlog calls may inline field names.

## Configuration

- All configuration via `pydantic-settings` loaded from YAML files.
- Config injection at construction time — see `safety.md` for boundary rules.

## File Organization

- 200–500 lines per file typical, 800 max.
- One class per file for core components.
- Maintain strict file separation among core components — do not mix functionalities.
- No circular imports — dependency flows downward: `api` → `core` → `infrastructure`.
- If a dependency feels backwards, extract the shared type into `contracts/`.

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
- Relative imports within `bodha/` package.

## Error Handling

- No bare `except:` — always specify exception type.
- No `print()` — use `structlog` with context (scope_id, session_id, component).
- Keep logging concise — every log line should have diagnostic value.
