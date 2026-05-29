# Parked Bodha-Coded Harness Pieces

Status: parked 2026-05-13 during initial adoption of the Bodha-flavored Claude setup to data-miner.

Files in this tree are **out of Claude Code's auto-discovery path**:

- `.claude/skills/<name>/SKILL.md` is the only path the skill loader scans, so anything under `_future-adoption/skills/` does **not** appear in the available-skills list.
- `.claude/commands/<name>.md` is the only path slash-commands scan, so anything under `_future-adoption/commands/` is not callable as `/<name>`.

That keeps the active harness focused on what works for data-miner today, while preserving the Bodha guidance verbatim for a future, deeper adoption pass.

## What is parked here

### `skills/`

| Skill | Why parked |
|-------|------------|
| `architecture-trace/` | Assumes `docs/design/<version>/` + `src/bodha/<component>/` + `docs/architecture-trace/at-claude / at-codex` output trees. Data-miner doesn't use design-versioning today. |
| `bodha-memory-eval/` | Validates Bodha design versions against memory-system eval datasets (DS-20, DS-40, DS-80). Not applicable until data-miner has a comparable eval harness. |
| `component-review/` | Drives per-component review under `docs/reviews/<version>/`. Requires the architecture-trace outputs above. |
| `design-evolve/` | Merges discussion files into core docs to produce next version under `docs/design/<version+1>/core/`. Needs the version-folder convention. |
| `phase-execution/` | Orchestrates one phase plan→Codex→implement→verify→status. Requires `docs/roadmap.md`, `docs/status.md`, `docs/checklist.md`, `docs/progress.md` — none of which exist at repo root today. (`manual_reviewer/docs/next_phases.md` is a per-subproject lightweight equivalent.) |

### `commands/`

| Command | Why parked |
|---------|------------|
| `run-phases.md` | Drives `docs/roadmap.md` → `docs/status.md` cascade. Depends on parked `phase-execution` skill. |
| `prepare-phases.md` | Produces brainstorm + roadmap + status rows. Same dependency. |

### `archived-guides/`

`html-artifact-*.skill.zip` and `instructtions.md` — stale packaging archives that shipped with the Bodha overlay. Kept for reference but not active.

## How to re-adopt later

If/when you decide to switch data-miner to the full Bodha-style workflow, the prerequisites are:

1. **Decide on the design-versioning convention** — e.g. `docs/design/v_1/core/data-miner-design.md` as the authoritative top doc.
   - Enables: `design-evolve`, `architecture-trace`, `component-review`.
2. **Create the roadmap cascade** at repo root:
   - `docs/roadmap.md` — phases with goal, deliverables table, spec refs, exit criterion, test focus, risk tier.
   - `docs/status.md` — per-phase NOT_STARTED / IN_PROGRESS / DONE / BLOCKED table.
   - `docs/checklist.md` — per-active-phase task list (swapped each phase).
   - `docs/progress.md` — append-only per-deliverable log.
   - Enables: `phase-execution`, `run-phases`, `prepare-phases`.
3. **Update [`.claude/project/invariants.md`](../project/invariants.md)** with checkable invariants tied to the new doc structure.
4. **Move the relevant parked dir back**, e.g.:
   ```bash
   mv .claude/_future-adoption/skills/phase-execution .claude/skills/
   mv .claude/_future-adoption/commands/run-phases.md .claude/commands/
   ```
5. **Update [`AGENTS.md`](../../AGENTS.md)** to re-reference `/phase-execution`, `/run-phases`, `docs/roadmap.md`, `docs/status.md`.

## Subproject scope (alternative path)

Instead of one repo-wide roadmap, you can re-adopt these per subproject:

- `data_miner/auto_annotation_v4/docs/roadmap.md` — already runs phase-style work.
- `manual_reviewer/docs/next_phases.md` — already a lightweight phase tracker.
- `manual_reviewer_cvat/RESUME.md` — already tracks resumable migration tracks.

Parking lets you decide the scope and convention without rushing it.

## Patterns from this harness worth borrowing now (without full adoption)

Even before re-adopting the parked pieces, several Bodha-borne patterns are useful in data-miner today and are kept in the active harness:

1. **Frozen project overlay** at [`.claude/project/`](../project/) — `brief.md`, `invariants.md`, `verification.md`, `docs-index.md`, `learnings.md`. Every agent reads these first.
2. **Mechanically checkable invariants** — `invariants.md` ships `rg`/`test` commands, not aspirational prose.
3. **`use-codex.md` as authoritative repo policy** — small Markdown file owns invocation rules for the Codex plugin; subagents read it before calling.
4. **`spec-reviewer` + `code-reviewer` as two-stage gate** — separates "did we build the right thing" from "is the code OK."
5. **Block-dangerous-git hook** at [`.claude/hooks/block-dangerous-git.sh`](../hooks/block-dangerous-git.sh).
6. **`docs-researcher` haiku subagent** for Context7 lookups — cheap, fast, keeps main context clean.
7. **Single `AGENTS.md` read-order** — predictable boot sequence each session.

## Do not

- Do not invoke parked skills via the Skill tool. They are intentionally out of the discoverable path.
- Do not edit parked content as part of unrelated work. If you need to update a parked file, do it as part of a deliberate re-adoption pass.
