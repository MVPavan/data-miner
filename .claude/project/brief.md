# Project Brief

Status: active development - check docs/status.md for progress

Last updated: 2026-04-07

## What Is Bodha

A universal agent memory system — persistent, structured, governed memory for AI agents. Gives agents durable recall, controlled write paths, retrieval intelligence, grounded extraction, background consolidation, and document/tool/skill integration.

**Users:** agent-platform engineers building personal assistants, domain agents, and multi-agent systems.

## Stack

- **Language:** Python 3.12+, fully async
- **Package manager:** uv
- **Data models:** Pydantic (all models frozen)
- **Config:** pydantic-settings from YAML
- **Stores:** PostgreSQL (write authority), Neo4j, Qdrant, Oxigraph, Redis (cache only)
- **Orchestration:** Temporal (sole orchestrator)
- **LLM access:** LiteLLM Proxy (all model calls routed through Gateway)
- **Observability:** Langfuse (LLM tracing), Grafana + Alloy + Loki (operational logging), structlog
- **Infrastructure:** Docker Compose, Alembic (migrations), Infisical (secrets)

## How Work Happens

- Development follows `docs/roadmap.md` — 9 phases, contract-first thin vertical slice.
- Current state tracked in `docs/status.md`.
- Start a phase: `/phase-execution N` — it handles planning, execution, review, and tracking.
- Design authority: `docs/design/v_2_7/core/bodha-design-v2_7.md` is the top doc. See `.claude/project/docs-index.md` for the full map.
- `infra/proxy-setup/` runs cli-proxy-api only since 2026-04-10. LiteLLM moved into `infra/dev-stack/docker-compose.yml` and reuses `bodha-postgres` for its state (in a `litellm` database created by a one-shot `litellm-db-init` sidecar). The two stacks communicate over the `bodha-llm-gateway` external docker network — run `docker network create bodha-llm-gateway` once during setup. Daily proxy-setup operation is `docker compose -f docker-compose-cliproxy-api.yaml up -d` (there is no plain `docker-compose.yaml` there anymore; `docker-compose.base.yaml` is a frozen historical snapshot). Do not re-add a second LiteLLM or a second Postgres to `infra/proxy-setup/`.

## Non-Negotiable Constraints

- PostgreSQL is the sole write authority.
- Temporal is the sole orchestrator.
- Phase 1.5 extraction requires `source_quote` grounding + non-LLM verification before Phase 2.
- Tools and skills are registry infrastructure, not memories — they bypass Dhṛti.
- Raw document chunks never enter Citta.
- Redis failure degrades quality, never blocks response.
- Pydantic BaseModel everywhere (invariant 21).

## Notes for Agents

- Read `docs-index.md` to find which design doc to read for any component.
- Read `docs/roadmap.md` Spec References to find the exact section for any implementation task.
- Do not inherit pre-v2.7 assumptions when they conflict with the v2.7 design set.
- Use `scratchpad/` for temporary files — it is gitignored.
