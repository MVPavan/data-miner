# Docs Index

Status: adopted for data-miner on 2026-05-13.

Use this to point at the right doc before guessing. Repo-relative paths only.

## Top-level repo docs

| Path | Purpose | Authority | Read when |
|------|---------|-----------|-----------|
| [`README.md`](../../README.md) | Project landing page + quickstart | canonical | First read for someone new to the repo |
| [`AGENTS.md`](../../AGENTS.md) | Agent operating guide (read order, working mode, Codex policy) | authoritative | Every new session |
| [`CLAUDE.md`](../../CLAUDE.md) | One-liner that imports `AGENTS.md` | — | — |
| [`pyproject.toml`](../../pyproject.toml) | Deps, Python pin, project scripts | canonical | Adding deps, debugging install |

## Architecture (`docs/architecture/`)

| Path | Component | Read when |
|------|-----------|-----------|
| [`docs/architecture/overview.md`](../../docs/architecture/overview.md) | Top-level system diagram, design decisions, component map | Always first for system-level decisions |
| [`docs/architecture/database-models.md`](../../docs/architecture/database-models.md) | `Project`, `Video`, `ProjectVideo` schema, status enums, locking columns | Changing the DB schema or worker locking behavior |
| [`docs/architecture/workers.md`](../../docs/architecture/workers.md) | Supervisord setup, worker lifecycle, heartbeat-locking flow | Changing worker boot, lifecycle, or claim semantics |
| [`docs/architecture/review-frontends.md`](../../docs/architecture/review-frontends.md) | Label Studio + CVAT dual-frontend architecture and exchange boundaries | Changing review tooling, LS/CVAT migration, or shared annotation I/O |

## User guide (`docs/user-guide/`)

| Path | Purpose | Read when |
|------|---------|-----------|
| [`docs/user-guide/installation.md`](../../docs/user-guide/installation.md) | uv-based environment setup, GPU optional-deps | Setting up a fresh machine |
| [`docs/user-guide/quickstart.md`](../../docs/user-guide/quickstart.md) | Smallest viable end-to-end run | Onboarding |
| [`docs/user-guide/configuration.md`](../../docs/user-guide/configuration.md) | Config YAML schema, OmegaConf overlay rules | Adding or tuning a stage's config |
| [`docs/user-guide/cli-reference.md`](../../docs/user-guide/cli-reference.md) | Every `data-miner ...` subcommand | Debugging or extending the CLI |
| [`docs/user-guide/fabric-deployment.md`](../../docs/user-guide/fabric-deployment.md) | Multi-host deployment with Fabric | Spinning up distributed workers |

## Development

| Path | Purpose | Read when |
|------|---------|-----------|
| [`docs/development/contributing.md`](../../docs/development/contributing.md) | Contribution conventions | Before a PR |
| [`docs/development/legacy-auto-annotation-cleanup.md`](../../docs/development/legacy-auto-annotation-cleanup.md) | Audit record for removed legacy auto-annotation packages | When checking why only v4 remains |
| [`docs/updates/monthly_update_2026-03-03_to_2026-05-03.md`](../../docs/updates/monthly_update_2026-03-03_to_2026-05-03.md) | Most recent activity digest | Catching up after time away |

## Infra / Kubernetes (`docs/k3s/`)

| Path | Purpose | Read when |
|------|---------|-----------|
| [`docs/k3s/01-concepts.md`](../../docs/k3s/01-concepts.md) | k3s/k8s core concepts | Touching cluster-side deployment |
| [`docs/k3s/01b-questions-answered.md`](../../docs/k3s/01b-questions-answered.md) | FAQ on k3s decisions | Same |
| [`docs/k3s/02-installation.md`](../../docs/k3s/02-installation.md) | k3s install steps | Same |
| [`docs/k3s/k8s-core-concepts.md`](../../docs/k3s/k8s-core-concepts.md), [`kubectl-commands.md`](../../docs/k3s/kubectl-commands.md) | k8s primer + kubectl cheatsheet | Reference while debugging cluster |

## Subproject docs

| Path | Purpose | Authority | Read when |
|------|---------|-----------|-----------|
| [`manual_reviewer/docs/next_phases.md`](../../manual_reviewer/docs/next_phases.md) | Phase plan for the Label Studio + SAM 3.1 review tool. Phases 1/3/4/5/7/8 done; 2 and 6 pending. | canonical for manual_reviewer | Picking up the next manual_reviewer phase |
| [`manual_reviewer_cvat/RESUME.md`](../../manual_reviewer_cvat/RESUME.md) | Resume plan for the CVAT stack. Existing docs still contain cutover language; pair with `review-frontends.md` for the maintained LS+CVAT direction. | canonical for manual_reviewer_cvat | Resuming CVAT stack work |
| [`manual_reviewer_cvat/docs/why_cvat.md`](../../manual_reviewer_cvat/docs/why_cvat.md) | Rationale for CVAT over Label Studio for the permanent tool | supporting | Justifying or revisiting the CVAT decision |
| [`manual_reviewer_cvat/docs/long_term_vision.md`](../../manual_reviewer_cvat/docs/long_term_vision.md) | End-state design for the permanent multi-team review tool | supporting | Scoping new work in manual_reviewer_cvat |
| [`manual_reviewer_cvat/docs/smart_tools_plan.md`](../../manual_reviewer_cvat/docs/smart_tools_plan.md) | Nuclio + SAM 3.1 smart-tool plan | supporting | Implementing or debugging smart tools |

## Auto-annotation engine

[`data_miner/auto_annotation_v4/`](../../data_miner/auto_annotation_v4/) is the
current and only retained auto-annotation engine (SAM 3.1, Rex-Omni, VLM
finalization). Older tracked `auto_annotation`, `auto_annotation_v2`, and
`auto_annotation_v3` source packages were removed after prompt preservation.

Prompt preservation for legacy cleanup lives under [`data_miner/auto_annotation_v4/prompts/archive/legacy/`](../../data_miner/auto_annotation_v4/prompts/archive/legacy/). Review it before deleting older auto-annotation packages.

## Diagrams

[`docs/diagrams/`](../../docs/diagrams/) holds drawio sources + rendered PNGs (`data_miner_System_Architecture.drawio`, `data_miner_dataflow.drawio`, `video_miner.drawio`, etc.). Update the drawio when changing system shape; export PNG after.

## Pointers off this index

- Engineering rules: [`.claude/rules/`](../rules/).
- Claude/Codex policy: [`AGENTS.md`](../../AGENTS.md) § Claude and Codex, [`.claude/commands/use-codex.md`](../commands/use-codex.md).
- Parked Bodha harness (future adoption): [`.claude/_future-adoption/README.md`](../_future-adoption/README.md).
