---
name: architecture-trace
description: Trace through all Bodha code and design docs, cross-reference design vs implementation, and produce a hierarchical set of Mermaid architecture graphs from overview down to per-component internals.
---

# Architecture Trace

Deep codebase-to-design tracing skill. Launches parallel Explore agents to read every source file and design doc, then orchestrates with Opus to produce a hierarchy of architecture diagrams.

**Resilient by design**: every agent writes results to checkpoint files. Failures are isolated. Re-running resumes from the last successful checkpoint.

## Use It When

- The user asks to map or visualize the current architecture
- The user wants to compare design docs against implementation
- The user says "trace the architecture", "architecture graph", or `/architecture-trace`

## Inputs

- **Design version** (optional, default: latest under `docs/design/`). Example: `v_2_7`.
- **Output path** (optional, default: `docs/architecture-trace/at-claude/`).
- **`--resume`** (optional): skip steps whose checkpoint files already exist.
- **`--force`** (optional): ignore existing checkpoints and re-run everything.

Default behavior (no flag): auto-detect. If checkpoint files exist, treat as resume. If none exist, fresh run.

---

## Checkpoint & Resumability Protocol

All intermediate and final results are written to disk, never held only in agent return values.

### Directory layout

```
docs/architecture-trace/at-claude/<version>/
  _checkpoints/                  # intermediate agent outputs
    01-design-docs.json          # Agent 1 result
    02-core-components.json      # Agent 2 result
    03-contracts-infra.json      # Agent 3 result
    04-retrieval-resolution.json # Agent 4 result
    05-tests-config.json         # Agent 5 result
    06-cross-reference.json      # Step 2 result
    manifest.json                # tracks status of every step
  00-INDEX.md                    # final index
  00-system-overview.md          # Level 0 graph
  01-*.md                        # Level 1 per-component graphs
  02-*.md                        # Level 2 cross-cutting graphs
```

### `manifest.json` schema

```json
{
  "version": "v_2_7",
  "started_at": "2026-04-09T...",
  "steps": {
    "agent-1-design-docs":        { "status": "done|failed|pending", "checkpoint": "01-design-docs.json",          "finished_at": "..." },
    "agent-2-core-components":    { "status": "pending",             "checkpoint": "02-core-components.json",       "finished_at": null },
    "agent-3-contracts-infra":    { "status": "pending",             "checkpoint": "03-contracts-infra.json",       "finished_at": null },
    "agent-4-retrieval-resolution":{ "status": "pending",            "checkpoint": "04-retrieval-resolution.json",  "finished_at": null },
    "agent-5-tests-config":       { "status": "pending",             "checkpoint": "05-tests-config.json",          "finished_at": null },
    "cross-reference":            { "status": "pending",             "checkpoint": "06-cross-reference.json",       "finished_at": null },
    "graph-level-0":              { "status": "pending",             "checkpoint": null,                            "finished_at": null },
    "graph-level-1":              { "status": "pending",             "checkpoint": null,                            "finished_at": null },
    "graph-level-2":              { "status": "pending",             "checkpoint": null,                            "finished_at": null },
    "index":                      { "status": "pending",             "checkpoint": null,                            "finished_at": null }
  }
}
```

### Rules for checkpoints

1. **Write before returning.** Every Explore agent prompt MUST include an instruction to write its structured result to its checkpoint file using the Write tool. The agent's return message is a summary only — the real data is on disk.
2. **Read from disk, not from return values.** The orchestrator reads checkpoint files to feed into subsequent steps. This decouples agent execution from orchestrator context.
3. **Manifest updates are atomic.** After each step completes (success or failure), update `manifest.json` with the new status and timestamp.
4. **On resume:** read `manifest.json`, skip any step with `status: "done"` whose checkpoint file still exists on disk. Re-run steps with `status: "failed"` or `status: "pending"`.
5. **On agent failure:** catch the failure, mark the step as `"failed"` in the manifest, log the error, and continue with remaining independent steps. Do NOT abort the entire run.
6. **Partial cross-reference is allowed.** If some agents failed, Step 2 should still run with whatever checkpoints are available, noting gaps. The graphs will mark missing data clearly.

---

## Workflow

### Step 0 — Prepare

1. Read `AGENTS.md` and `.claude/project/brief.md` for project context.
2. Determine the design version to trace. If not specified, pick the latest `docs/design/v_*` directory.
3. Create directory structure:
   ```
   mkdir -p docs/architecture-trace/at-claude/<version>/_checkpoints
   ```
4. Check for existing `manifest.json`:
   - **Exists + not `--force`**: enter resume mode. Report which steps are done/failed/pending.
   - **Does not exist OR `--force`**: create fresh `manifest.json` with all steps `"pending"`.
5. Report the plan to the user: which steps will run, which will be skipped (if resuming).

### Step 1 — Parallel Exploration (Explore agents, sonnet/gpt 5.4 mini model)

For each agent below, **skip if its manifest status is `"done"` and checkpoint file exists** (resume mode).

Launch all non-skipped agents **in parallel**. Use `model: "sonnet/gpt 5.4 mini"`, `subagent_type: "Explore"`.

**Critical: every agent prompt must include these instructions:**
> Write your complete structured result to `docs/architecture-trace/at-claude/<version>/_checkpoints/<checkpoint-file>` using the Write tool before returning. Format: JSON with a top-level object. Your return message should be a short summary (under 300 words) confirming what you wrote and any issues encountered.

---

**Agent 1 — Design Docs Survey** -> `01-design-docs.json`
- Thoroughness: `"very thorough"`
- Read every file under `docs/design/<version>/core/`.
- For each design doc, extract:
  - Component name and purpose
  - Key abstractions (classes, protocols, models) — list names and one-line descriptions
  - External dependencies and integration points
  - Data flows (what goes in, what comes out)
  - Contracts and invariants mentioned
- Write result as JSON: `{ "components": [ { "name": "...", "doc_path": "...", "purpose": "...", "abstractions": [...], "dependencies": [...], "data_flows": [...], "contracts": [...] } ] }`

**Agent 2 — Source Code: Core Components** -> `02-core-components.json`
- Thoroughness: `"very thorough"`
- Read every `.py` file under `src/bodha/` (exclude `__pycache__`, `infrastructure/`, `tools/`, `contracts/`, `retrieval/`, `resolution/`).
- Covers: `manas/`, `buddhi/`, `dharana/`, `dhriti/`, `skills/`, and root `__init__.py` / top-level modules.
- For each package, then each file, extract:
  - All classes: name, bases, public methods with type signatures
  - All module-level functions with type signatures
  - All imports (split: stdlib, third-party, internal)
  - All Pydantic models with field names and types
  - All enums and constants
  - Protocol definitions
  - Key decorators and patterns (Temporal workflows/activities, etc.)
- Write result as JSON: `{ "packages": { "<name>": { "files": { "<filename>": { "classes": [...], "functions": [...], "imports": {...}, "models": [...], "enums": [...], "constants": [...], "protocols": [...] } } } } }`

**Agent 3 — Source Code: Contracts & Infrastructure** -> `03-contracts-infra.json`
- Thoroughness: `"very thorough"`
- Focus on `src/bodha/contracts/`, `src/bodha/infrastructure/`, and `src/bodha/tools/`.
- Extract:
  - All shared types, protocols, and base classes in contracts
  - Infrastructure adapters: what each adapter wraps, its interface
  - Gateway layer: API endpoints, request/response models
  - Temporal workflows and activities (names, inputs, outputs)
  - Projectors and their event handlers
  - Observability setup (metrics, tracing, logging patterns)
- Write result as JSON with sections: `{ "contracts": {...}, "infrastructure": {...}, "tools": {...} }`

**Agent 4 — Source Code: Retrieval & Resolution** -> `04-retrieval-resolution.json`
- Thoroughness: `"very thorough"`
- Focus on `src/bodha/retrieval/` and `src/bodha/resolution/`.
- Extract:
  - Retrieval pipeline stages (dhi, smriti, feedback): classes, methods, flow
  - Resolution strategies and their interfaces
  - How retrieval connects to the rest of the system (imports, calls)
- Write result as JSON: `{ "retrieval": {...}, "resolution": {...} }`

**Agent 5 — Test & Config Survey** -> `05-tests-config.json`
- Thoroughness: `"medium"`
- Read `tests/` directory structure and representative test files (not every line — focus on organization, fixtures, markers, what each test module covers).
- Read `pyproject.toml`, any config YAML files, Docker/compose files.
- Extract:
  - Test organization: which components have tests, test count per area
  - Key fixtures and conftest patterns
  - Dependencies and their versions (from pyproject.toml)
  - Configuration shape and defaults
  - Docker services and their roles
- Write result as JSON: `{ "tests": {...}, "config": {...}, "dependencies": {...}, "docker": {...} }`

---

**After all agents complete (or fail):**

1. For each agent, check if the checkpoint file was written:
   - **File exists**: mark step `"done"` in manifest.
   - **File missing + agent returned error**: mark step `"failed"`, record error message.
   - **File missing + agent returned success**: mark step `"failed"` with note "agent claimed success but no checkpoint found".
2. Update `manifest.json`.
3. Report to user: "X/5 explorations succeeded. [Failed: agent-N, agent-M]. Proceeding with available data."

### Step 2 — Cross-Reference (Opus orchestrator)

**Skip if manifest status is `"done"` and `06-cross-reference.json` exists.**

Read all available checkpoint files from `_checkpoints/` (only `"done"` ones). The orchestrator (Opus) performs:

1. **Design-to-Code Mapping**: For each component in `01-design-docs.json`, find the corresponding source package(s) in `02-core-components.json` / `03-contracts-infra.json` / `04-retrieval-resolution.json`. Note:
   - Components present in design but missing in code (mark `"design_only"`)
   - Components present in code but not in design (mark `"code_only"`)
   - Mismatches in abstractions, naming, or contracts (mark `"divergent"`)
   - Fully matching (mark `"aligned"`)

2. **Dependency Graph Extraction**: From imports data, build:
   - Package-level dependency edges: `[{ "from": "manas", "to": "contracts", "via": ["MemoryEvent", "..."] }]`
   - External dependency edges: `[{ "package": "temporal", "used_by": ["infrastructure"] }]`

3. **Data Flow Tracing**: From data flow fields and method signatures, trace:
   - Memory ingestion path
   - Memory retrieval path
   - Event/projection flows
   - Temporal workflow orchestration

4. **Write `06-cross-reference.json`** with sections: `{ "mapping": [...], "dependency_graph": {...}, "data_flows": {...}, "gaps": [...] }`

5. Mark step `"done"` in manifest. If it fails, mark `"failed"` — Level 0/1/2 graphs can still be partially generated from raw checkpoints.

### Step 3 — Graph Generation (Opus orchestrator)

Generate graphs in sub-steps. Each sub-step is independently resumable — check manifest before each.

**Important**: after writing each `.md` file, update manifest immediately. Do not batch.

#### Step 3a — Level 0: `00-system-overview.md`
**Skip if `graph-level-0` is `"done"` in manifest and file exists.**

- Read `06-cross-reference.json` (or fall back to raw checkpoints if cross-reference failed).
- Single high-level Mermaid graph showing all major components.
- Edges show primary data/control flows between components.
- Color-code nodes: `style` green = `"aligned"`, yellow = `"divergent"` or partial, red = `"design_only"` or `"code_only"`.
- Write file. Update manifest step `"graph-level-0"` to `"done"`.

#### Step 3b — Level 1: Per-Component Files
**Skip already-written files.** Track individually in manifest under `"graph-level-1"` as a sub-object:
```json
"graph-level-1": { "status": "in_progress", "components": { "manas": "done", "buddhi": "pending", ... } }
```

For each component (manas, buddhi, dharana, dhriti, chitta, retrieval, resolution, infrastructure, tools, skills, contracts), write `01-<component>.md` containing:

- **Design Summary**: What the design doc says this component does (from `01-design-docs.json`)
- **Implementation Summary**: What the code actually contains (from code checkpoints)
- **Delta**: Differences between design and implementation
- **Internal Architecture Graph**: Mermaid diagram — classes, relationships (inheritance `--|>`, composition `*--`, dependency `..>`), key methods
- **Data Flow Graph**: Mermaid diagram — how data moves through this component
- **Integration Points**: Mermaid diagram — connections to other components

After writing each file, update the component status in manifest.

**If a component's data is missing** (agent that covers it failed): still write the file with available data, clearly marking `[DATA UNAVAILABLE — agent-N failed]` sections.

#### Step 3c — Level 2: Cross-Cutting Concerns
**Skip already-written files.** Track individually in manifest under `"graph-level-2"`.

- `02-data-flow-full.md`: End-to-end data flow across all components (Mermaid sequence diagram + flowchart)
- `02-dependency-matrix.md`: Full import dependency matrix as a markdown table + Mermaid graph
- `02-temporal-workflows.md`: Temporal workflow and activity graph (Mermaid)
- `02-contracts-and-protocols.md`: All shared types and their consumers (Mermaid class diagram)
- `02-design-coverage.md`: Summary table — each design component, its status (aligned/divergent/design-only/code-only), implementation location, notes

After writing each file, update manifest.

### Step 4 — Index Generation

**Skip if `index` is `"done"` in manifest and `00-INDEX.md` exists.**

Create `00-INDEX.md` in the output directory:
- Table of contents linking to every generated file (mark any that failed to generate)
- Legend for color coding and diagram conventions
- Timestamp, design version, and run metadata
- Summary statistics: files traced, components found, coverage %, steps succeeded/failed
- If any steps failed: a "Known Gaps" section listing what's missing and why

Update manifest step `"index"` to `"done"`.

### Step 5 — Verification & Report

1. Read `manifest.json` — count done/failed/pending.
2. Confirm all expected output files exist (or are accounted for as failed).
3. Spot-check 2-3 Mermaid blocks for valid syntax (balanced brackets, proper arrow syntax `-->`, `-.->`, `--|>`).
4. Present to the user:
   - Link to `00-INDEX.md`
   - Summary: "X/Y steps completed. Z components graphed. N gaps."
   - If there were failures: list them with suggested re-run command (`/architecture-trace --resume`)

---

## Failure Handling Matrix

| Failure | Impact | Recovery |
|---------|--------|----------|
| Agent 1 (design docs) fails | No design summaries; cross-reference has code-only view | Graphs generated from code only, design sections marked `[UNAVAILABLE]`. Re-run: `/architecture-trace --resume` |
| Agent 2-4 (code) fails | Missing code coverage for affected packages | Design-only view for those components. Re-run: `/architecture-trace --resume` |
| Agent 5 (tests/config) fails | No test/config info in output | Graphs still generated. Config section omitted. Re-run: `/architecture-trace --resume` |
| Cross-reference (Step 2) fails | Raw data exists but no mapping | Level 1 graphs use raw checkpoints directly (less cross-referencing). Re-run: `/architecture-trace --resume` |
| Graph generation partially fails | Some `.md` files missing | Manifest tracks which components succeeded. Re-run picks up remaining. |
| Context limit hit mid-step | Current step may be incomplete | Manifest shows last completed sub-step. Re-run resumes from there. |

---

## Output Structure

```
docs/architecture-trace/at-claude/<version>/
  _checkpoints/
    manifest.json
    01-design-docs.json
    02-core-components.json
    03-contracts-infra.json
    04-retrieval-resolution.json
    05-tests-config.json
    06-cross-reference.json
  00-INDEX.md
  00-system-overview.md
  01-manas.md
  01-buddhi.md
  01-dharana.md
  01-dhriti.md
  01-chitta.md
  01-retrieval.md
  01-resolution.md
  01-infrastructure.md
  01-tools.md
  01-skills.md
  01-contracts.md
  02-data-flow-full.md
  02-dependency-matrix.md
  02-temporal-workflows.md
  02-contracts-and-protocols.md
  02-design-coverage.md
```

## Rules

- Use Explore agents with `model: "sonnet/gpt 5.4 mini"` for all reading/exploration.
- Use Opus (main agent) for cross-referencing, analysis, and graph generation.
- Launch Explore agents in parallel wherever possible.
- Do not skip any source file or design doc — thoroughness is non-negotiable.
- Every Mermaid diagram must be syntactically valid.
- Do not modify any source code or design docs. This is a read-only analysis.
- All output goes to `docs/architecture-trace/at-claude/<version>/`.
- If a component exists in design but not in code, still create its Level 1 file documenting the gap.
- If a component exists in code but not in design, still create its Level 1 file documenting the undocumented component.
- **Agents write to disk, orchestrator reads from disk.** Never rely on agent return values for data — only for status confirmation.
- **Update manifest after every atomic step.** Not at the end — after each file write, each agent completion, each sub-step.
- **Never abort on partial failure.** Degrade gracefully, mark gaps, continue.
