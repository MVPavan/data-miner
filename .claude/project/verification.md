# Verification Commands

Status: adopted for v2.7 on 2026-04-07

The current Bodha repo contains design docs, infrastructure setup, and harness files, not runnable application code.
Use the current repo-reality checks below until an implementation exists. Do not claim code verification that the repo cannot run yet.

## Adopted Commands

- Quick:
  - `test -f docs/design/v_2_7/core/bodha-design-v2_7.md`
  - `test -f docs/design/v_2_7/core/bodha-job-plan-v2_7.md`
  - `find docs/design/v_2_7/core -maxdepth 1 -name '*.md' | sort`
- Full:
  - `find docs/design/v_2_7/core -maxdepth 1 -name '*.md' | sort`
  - `find docs/design/v_2_6/core/new_discussions -maxdepth 1 -name '*.md' | sort`
  - `rg -n "Status:\\*\\* Authoritative|Status:\\s+Authoritative|Temporal as sole orchestrator|pydantic-settings replaces OmegaConf|Tools and skills are \\*\\*identity infrastructure\\*\\*|Raw chunks and verbatim passages never enter Citta|source_quote|Phase 1\\.5|GPT-5\\.4 Nano|Claude-Mem" docs/design/v_2_7/core`
  - `rg -n "^## \\[INV-" .claude/project/invariants.md`
- Roadmap consistency:
  - `test -f docs/roadmap.md`
  - `test -f docs/status.md`
  - `test -f docs/brainstorms/2026-04-07-development-roadmap-requirements.md`
  - `rg -c "NOT_STARTED|IN_PROGRESS|DONE|BLOCKED" docs/status.md`
- Extended:
  - Human review of `.claude/project/brief.md` and `.claude/project/docs-index.md` against the latest authoritative design docs

Run the repo's own scripts or CI-equivalent commands when they exist. Do not invent a weaker substitute.

## Future Implementation Verification

When Bodha source, manifests, and test config are added, replace the doc-only suite above with repo-native commands immediately.
If the repo is still Python and no stronger commands exist, start from:

- `ruff check .`
- `ruff format --check .`
- `mypy .`
- `pytest`
