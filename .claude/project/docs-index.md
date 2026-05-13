# Docs Index

Status: adopted for v2.7 on 2026-04-07

## Core Design Documents

| Path | Component | Responsibility | Risk | Authority | Read When |
|------|-----------|----------------|------|-----------|-----------|
| `docs/design/v_2_7/core/bodha-design-v2_7.md` | Common architecture | System-level architecture, contracts, principles, v2.7 extraction hardening, Claude-Mem adoption | high | authoritative | Always first for system-level decisions |
| `docs/design/v_2_7/core/bodha-buddhi-v2_7.md` | Buddhi | Turn orchestration, tool gating, model tiers, Gateway, extraction model routing | high | authoritative | Changing orchestration or model-access behavior |
| `docs/design/v_2_7/core/bodha-manas-v2_7.md` | Manas | Context packing, compression, cache breakpoints, repacking | high | authoritative | Changing prompt packing, context budgets, or repacking |
| `docs/design/v_2_7/core/bodha-retrieval-v2_7.md` | Retrieval | Dhī intent classification and Smṛti retrieval stages | high | authoritative | Changing recall behavior, caching, or retrieval plans |
| `docs/design/v_2_7/core/bodha-dhriti-v2_7.md` | Dhṛti | Extraction, Phase 1.5 grounding verification, evaluation, HOLD, provenance, write-gate | high | authoritative | Changing what gets stored or how memory writes are judged |
| `docs/design/v_2_7/core/bodha-chitta-v2_7.md` | Chitta | Store authority, schemas, adapter boundaries, projections | high | authoritative | Changing schemas, stores, or write/read boundaries |
| `docs/design/v_2_7/core/bodha-dharana-v2_7.md` | Dhāraṇā | Background integrity, consolidation, enrichment, revalidation jobs | high | authoritative | Changing async maintenance or recovery behavior |
| `docs/design/v_2_7/core/bodha-infrastructure-v2_7.md` | Infrastructure | ProjectionEngine, overlay, adapters, Temporal, observability, config, packaging | high | authoritative | Changing infrastructure or deployment surfaces |
| `docs/design/v_2_7/core/bodha-tools-skills-v2_7.md` | Tools and skills | Capability registries, logging, loading, graduation pipeline | high | authoritative | Changing capability discovery or skill lifecycle |
| `docs/design/v_2_7/core/bodha-rag-v2_7.md` | Document RAG | Bookshelf model, multi-RAG routing, document provenance, `sources` behavior | high | authoritative | Changing document retrieval or document-memory interaction |
| `docs/design/v_2_7/core/bodha-job-plan-v2_7.md` | Job plan | Job inventory, ownership, triggers, sequencing, Phase 1.5 verification flow | medium | authoritative | Mapping work to jobs, workers, or failure domains |
| `docs/design/v_2_7/core/memory-benchmarks-v2_7.md` | Benchmarks | Benchmark selection and evaluation coverage | medium | authoritative | Defining memory-quality evaluation strategy |

## Supporting Documents

| Path | Purpose | Authority | Read When |
|------|---------|-----------|-----------|
| `docs/design/v_2_6/core/new_discussions/dhriti-extraction-discussions.md` | Pre-v2.7 extraction tradeoff analysis that informed the hardened extraction pipeline | supporting | Understanding why Phase 1.5 and escalation were introduced |
| `docs/design/v_2_6/core/bodha_subscription_proxy_architecture.md` | Recommended local model-routing topology for subscription-backed access | supporting | Wiring local gateway and provider auth topology |
| `docs/design/v_2_6/core/new_discussions/claude-mem-adoption-log.md` | Claude-Mem adoption decisions partially absorbed into v2.7 common design | supporting | Evaluating sidecar, shadow-log, and progressive-disclosure proposals |
| `infra/dev-stack/litellm_config.yaml` | LiteLLM proxy configuration for local model routing (moved from `infra/proxy-setup/` on 2026-04-10) | supporting | Configuring or debugging model proxy setup |

## Development Planning Documents

| Path | Purpose | Authority | Read When |
|------|---------|-----------|-----------|
| `docs/brainstorms/2026-04-07-development-roadmap-requirements.md` | Approved brainstorm: build order, test strategy, scope, Codex-informed decisions | approved | Understanding why the roadmap is structured the way it is |
| `docs/design/v_2_7/core/new_discussions/bodha-pydanticai-temporal-litellm-integration.md` | Source brainstorm for the PydanticAI + TemporalAgent + LiteLLM integration (P0 §0 corrections, C1–C8 facts, §12 hot-path cost-benefit, §12.6 reversibility) | approved | Working on hot-path agents, capability middleware, provider factory, virtual-key flow, or the hot-path deviation |
| `docs/roadmap.md` | Execution plan: 9 phases, deliverables, spec references, exit criteria, test focus | canonical | Starting any implementation phase; finding which design doc section to read for a task |
| `docs/roadmap-pydanticai-integration.md` | Execution plan for the PydanticAI integration (P0–P6): phase gates, empirical C1–C8 corrections, Temporal/LiteLLM contracts, CI invariants | canonical | Touching anything in the agent middleware chain, background-agent wrapping, or the gateway factory |
| `docs/roadmap-eval-harness.md` | Execution plan for the eval harness (E-pre through E-C2): DS-20 memory substrate benchmark, 5 evaluators, Pydantic Evals + Langfuse. 2 Codex adversarial rounds applied. | canonical | Building or modifying the eval harness, fixture schema, evaluators, quiescence engine, or reporting |
| `docs/status.md` | Per-component status tracker with blocked-by and verified-by columns | canonical | Checking current state; finding what's ready to start next |
| `docs/checklist.md` | Active phase task checklist (replaced per phase) | session | During implementation; tracking what's done and what's next |
| `docs/progress.md` | Per-deliverable completion log with commit SHAs, test results, notes | durable | Resuming work across sessions; reviewing what was done |
| `docs/design/memory-design-tests/v_2/` | Golden evaluation datasets: DS-20, DS-40, DS-80 | authoritative | Validating extraction/retrieval/system quality at phase exits |

Use this file to point agents at the right docs before they guess.

- List only durable docs that materially improve decisions.
- Mark one doc as authoritative when multiple docs overlap.
- Use repo-relative paths only.

Current repo reality: no Bodha source tree, manifest, or test directory yet. This table maps authoritative design components. Infrastructure setup exists under `infra/`.
