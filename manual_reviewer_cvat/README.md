## manual_reviewer_cvat

CVAT-based review frontend maintained alongside the Label-Studio-based
[manual_reviewer/](../manual_reviewer/).

> **Resuming this work?** Read [RESUME.md](RESUME.md) first — handoff
> context for picking up on a host-Docker machine, including phase plan
> and copy-paste resume prompts.

LS keeps running on `:8080` while this stack stands up on `:8081`; they
do not conflict. The long-term direction is dual frontend operation with
LS <-> CVAT task exchange, not mandatory LS decommissioning. See
[../docs/architecture/review-frontends.md](../docs/architecture/review-frontends.md).

---

## Two tracks share this directory

### Track A — Datatang-1000 LS -> CVAT migration and exchange seed
Lives entirely in [migrations_from_LS/](migrations_from_LS/). Bootstrap
the 5 reviewers, seed tasks from YOLO, import in-flight LS work,
and hand off. The migration code should evolve toward reusable exchange
adapters where possible.

### Track B — Permanent multi-team review frontend
The CVAT stack itself + Nuclio smart-tools layer wrapping our SAM 3.1
service. Implementation pending, plan in
[docs/long_term_vision.md](docs/long_term_vision.md) and
[docs/smart_tools_plan.md](docs/smart_tools_plan.md). The Nuclio
functions land in `serverless/` (created during Phase 2-3 of
[RESUME.md](RESUME.md)).

---

## Layout

```
manual_reviewer_cvat/
├── README.md                       you are here
├── RESUME.md                       handoff doc — read first if resuming
├── docker-compose.cvat.yml         CVAT stack (Track A pin v2.18.0; bump for Track B)
├── configs/
│   ├── stack.env.example
│   └── aerospike.conf              Redis-on-disk config (CVAT requirement)
├── docs/
│   ├── why_cvat.md                 short rationale + sources
│   ├── long_term_vision.md         strategic decision: vanilla CVAT + Nuclio
│   └── smart_tools_plan.md         Nuclio function spec for the 4 smart tools
├── scripts/
│   ├── manage_cvat.sh              start / stop / status / logs / backup the stack
│   └── export_to_aa_v4.py          general CVAT → pipeline.db Stage.HUMAN_REVIEW round-trip
├── pipeline_io/                    Datumaro parser + future cvat_client.py
├── migrations_from_LS/             ── Track A (LS -> CVAT seed/exchange) ──────
│   ├── README.md                   Datatang-1000 LS -> CVAT seed quickstart
│   ├── docs/
│   │   ├── migration_from_ls.md    spec for migrate_from_ls.py
│   │   └── reviewer_onboarding.md  cheat sheet for the 5 reviewers
│   ├── pipeline_io/                LS state reader + LS↔CVAT annotation mapper
│   └── scripts/
│       ├── create_users.py         bootstrap 5 reviewers + create project
│       ├── seed_tasks_from_yolo.py YOLO + clips → CVAT tasks, RR by frame count
│       └── migrate_from_ls.py      LS project 9 completions → CVAT pre-annotations
└── tests/
```

The live CVAT API scripts are currently **stubs** (docstring + signature +
`NotImplementedError`). Pure parser code can still land here: `pipeline_io/`
already contains the Datumaro bbox export parser used as the CVAT side of the
shared LS/CVAT exchange model. New work should preserve the dual LS+CVAT
direction in `review-frontends.md`.

---

## Quickstart

- **Standing up CVAT** (Track A or B — same stack):
  `./scripts/manage_cvat.sh start` → http://127.0.0.1:8081
- **Datatang-1000 LS -> CVAT seed run** (Track A): see
  [migrations_from_LS/README.md](migrations_from_LS/README.md).
- **Smart-tools / Nuclio rollout** (Track B): see
  [RESUME.md](RESUME.md) Phases 2-4.

---

## Why CVAT (vs Label Studio / Labelbox / fork-CVAT)

Short answer: vanilla CVAT + Nuclio is the only option that gets us
CVAT-grade UX **and** keeps the smart-tools layer maintainable for
3-5 years across multiple teams without recurring rebase tax.
Long answer: [docs/long_term_vision.md](docs/long_term_vision.md).
